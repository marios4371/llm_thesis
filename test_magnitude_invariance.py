"""Offline validation of magnitude_invariance.py against the real run data.

Nothing here touches a GPU or an LLM. The point is to prove the deterministic
half is correct BEFORE a GPU hour is spent on the half that needs a model --
the discipline that caught the sidecar join bugs before the SC run.

The load-bearing check is the last one. Comparing a structure with itself
passes by construction, and a verifier that only does that is a rubber stamp:
precisely the current failure of SIV, which reports "verified" on 20 of the 27
errors. So the structure is corrupted on one side and we measure how often the
disagreement is caught -- and, for every mutation that slips through, whether
it changes the computed value at the ORIGINAL magnitudes. A miss on a
value-preserving mutation is not a miss; a miss on one that does change the
value would be a blind spot created by the probes.
"""
from __future__ import annotations

import json
import os
import random
import re as _re
import sys

import pandas as pd

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import magnitude_invariance as M

RUN = 'results_September/results/MAS_SHT/results/mas_sht_math7b_20260912_094311.csv'


def load_problems():
    import grounding_probe as G
    texts = G.load_texts()
    d = pd.read_csv(RUN)
    rows = []
    for _, r in d.iterrows():
        t = texts.get(r.problem_id)
        if t is None:
            continue
        try:
            givens = json.loads(r.blueprint_givens)
            eqs = json.loads(r.blueprint_equations)
        except (json.JSONDecodeError, TypeError):
            givens, eqs = {}, []
        rows.append(dict(pid=r.problem_id, text=t, givens=givens, equations=eqs,
                         csv_bp_answer=r.siv_blueprint_answer, gold=r.gold))
    return rows


def push_to_probes(givens, shrink):
    """Real given values -> their probe stand-ins."""
    return {k: (shrink.original_to_probe[float(v)]
                if isinstance(v, (int, float)) and not isinstance(v, bool)
                and float(v) in shrink.original_to_probe else v)
            for k, v in givens.items()}


def mutate_equations(eqs, rng):
    """One realistic transcription-style corruption of the structure."""
    kinds = []
    joined = "\n".join(eqs)
    if _re.search(r'[+\-*/]', joined):
        kinds.append('operator')
    if len(_re.findall(r"givens\['[^']+'\]", joined)) >= 2:
        kinds.append('swap_operands')
    if not kinds:
        return None, None
    kind = rng.choice(kinds)
    out = list(eqs)

    if kind == 'operator':
        spots_by_eq = [(i, list(_re.finditer(
            r'(?<=[\w\)\]])(\s*)([+\-*/])(\s*)(?=[\w\(\[])', e)))
            for i, e in enumerate(out)]
        spots_by_eq = [(i, s) for i, s in spots_by_eq if s]
        if not spots_by_eq:
            return None, None
        i, spots = rng.choice(spots_by_eq)
        flip = {'+': '-', '-': '+', '*': '/', '/': '*'}
        m = rng.choice(spots)
        out[i] = out[i][:m.start(2)] + flip[m.group(2)] + out[i][m.end(2):]
        return out, f"operator {m.group(2)} -> {flip[m.group(2)]}"

    uniq = sorted(set(_re.findall(r"givens\['([^']+)'\]", joined)))
    if len(uniq) < 2:
        return None, None
    a, b = rng.sample(uniq, 2)
    out = [e.replace(f"givens['{a}']", "@@TMP@@")
            .replace(f"givens['{b}']", f"givens['{a}']")
            .replace("@@TMP@@", f"givens['{b}']") for e in out]
    return out, f"swapped givens {a} <-> {b}"


def fault_injection(rows, seed=0):
    """Returns (attempted, caught, value_preserving_misses, blind_spots, examples)."""
    rng = random.Random(seed)
    attempted = caught = preserving = blind = 0
    examples = []
    for r in rows:
        s = M.shrink_text(r['text'])
        if not s.changed or not r['equations'] or not r['givens']:
            continue
        bp_big = {'givens': r['givens'], 'equations': r['equations']}
        v_orig = M.evaluate(bp_big)
        if v_orig is None:
            continue
        mutated, what = mutate_equations(r['equations'], rng)
        if mutated is None or mutated == r['equations']:
            continue
        res = M.metamorphic_check(
            bp_big, {'givens': push_to_probes(r['givens'], s),
                     'equations': mutated}, s)
        if res.agree is None:
            continue
        attempted += 1
        if res.agree is False:
            caught += 1
            continue
        # slipped through -- is there anything there to catch?
        v_mut = M.evaluate({'givens': r['givens'], 'equations': mutated})
        if v_mut is None or abs(v_mut - v_orig) <= 1e-9 * max(abs(v_orig), 1.0):
            preserving += 1
        else:
            blind += 1
            if len(examples) < 5:
                examples.append((r['pid'], what, v_orig, v_mut))
    return attempted, caught, preserving, blind, examples


def main() -> int:
    rows = load_problems()
    print(f"problems loaded: {len(rows)}\n")
    fails = []

    # ---- 1. shrink is well-formed -------------------------------------
    n_changed = n_collision = n_missed = n_exhausted = 0
    for r in rows:
        s = M.shrink_text(r['text'])
        if s.changed:
            n_changed += 1
        orig_nums = M.text_number_set(r['text'])
        for p in s.probe_to_original:
            if p in orig_nums:
                n_collision += 1
                fails.append(f"{r['pid']}: probe {p} already in the problem text")
        for v in M.text_number_set(s.shrunk_text):
            if v >= M.DEFAULT_THRESHOLD:
                n_missed += 1
                fails.append(f"{r['pid']}: {v} survived the shrink")
        if s.skipped_no_probe:
            n_exhausted += 1
        if len(set(s.probe_to_original.values())) != len(s.probe_to_original):
            fails.append(f"{r['pid']}: probe mapping is not injective")

    print("1. SHRINK")
    print(f"   problems containing a large number : {n_changed}/{len(rows)}")
    print(f"   probe collided with existing number: {n_collision}")
    print(f"   large number survived the shrink   : {n_missed}")
    print(f"   ran out of probes                  : {n_exhausted}")

    # ---- 2. rebind round-trips ----------------------------------------
    n_rt = n_rt_ok = 0
    for r in rows:
        s = M.shrink_text(r['text'])
        if not s.changed or not r['givens']:
            continue
        back, _ = M.rebind_givens(push_to_probes(r['givens'], s), s)
        n_rt += 1
        if back == r['givens']:
            n_rt_ok += 1
        else:
            diff = {k: (r['givens'][k], back.get(k))
                    for k in r['givens'] if back.get(k) != r['givens'][k]}
            fails.append(f"{r['pid']}: rebind round-trip lost {diff}")
    print("\n2. REBIND ROUND-TRIP (real givens -> probes -> back)")
    print(f"   exact round-trip: {n_rt_ok}/{n_rt}")

    # ---- 3. evaluator agrees with the pipeline's own number -----------
    n_ev = n_match = n_none = 0
    for r in rows:
        if not r['equations'] or not r['givens']:
            continue
        got = M.evaluate({'givens': r['givens'], 'equations': r['equations']})
        try:
            want = float(r['csv_bp_answer'])
        except (TypeError, ValueError):
            continue
        n_ev += 1
        if got is None:
            n_none += 1
        elif abs(got - want) <= 1e-6 * max(abs(want), 1.0):
            n_match += 1
        else:
            fails.append(f"{r['pid']}: evaluator {got} != pipeline {want}")
    print("\n3. EVALUATOR vs THE PIPELINE'S OWN siv_blueprint_answer")
    print(f"   agrees: {n_match}/{n_ev}   (unevaluable here: {n_none})")

    # ---- 4. identical structures must always agree --------------------
    n_id = n_id_agree = 0
    for r in rows:
        s = M.shrink_text(r['text'])
        if not s.changed or not r['equations'] or not r['givens']:
            continue
        res = M.metamorphic_check(
            {'givens': r['givens'], 'equations': r['equations']},
            {'givens': push_to_probes(r['givens'], s), 'equations': r['equations']}, s)
        if res.agree is None:
            continue
        n_id += 1
        if res.agree:
            n_id_agree += 1
        else:
            fails.append(f"{r['pid']}: identical structures reported as diverging")
    print("\n4. SELF-CONSISTENCY (identical structure must always agree)")
    print(f"   agree: {n_id_agree}/{n_id}")

    # ---- 5. does it DISCRIMINATE? -------------------------------------
    print("\n5. FAULT INJECTION (corrupt one side; 5 seeds)")
    tot = tot_caught = tot_pres = tot_blind = 0
    for seed in range(5):
        a, c, p, b, ex = fault_injection(rows, seed=seed)
        tot += a; tot_caught += c; tot_pres += p; tot_blind += b
        print(f"   seed {seed}: {a:3d} mutations -> caught {c:3d}, "
              f"value-preserving {p:2d}, BLIND {b}")
        for pid, what, vo, vm in ex:
            fails.append(f"{pid}: blind spot -- {what} changed {vo} to {vm} "
                         f"but probes agreed")
    detect_all = 100 * tot_caught / tot if tot else 0
    real = tot - tot_pres
    detect_real = 100 * tot_caught / real if real else 0
    print(f"   detection over all mutations          : {tot_caught}/{tot} "
          f"= {detect_all:.1f}%")
    print(f"   detection over mutations that MATTER  : {tot_caught}/{real} "
          f"= {detect_real:.1f}%")
    print(f"   blind spots (probes hid a real change): {tot_blind}")

    print("\n" + "=" * 70)
    if fails:
        print(f"FAILURES: {len(fails)}")
        for f in fails[:12]:
            print("   " + f)
        if len(fails) > 12:
            print(f"   ... and {len(fails)-12} more")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == '__main__':
    sys.exit(main())
