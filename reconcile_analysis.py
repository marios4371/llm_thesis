"""
[v17.0] Read a v17 run, and reproduce the offline evidence behind it.

Two modes.

    python reconcile_analysis.py --replay
        Runs reconciliation over the four stored runs offline, using each
        row's independently derived answer as the stand-in second derivation.
        This is where the numbers quoted in the v17 changelog come from, and
        re-running it is how you check that a change to reconcile.py did not
        move them. Needs the cached HF datasets for the problem texts.

    python reconcile_analysis.py --run <mas_v17.csv> [--cmp b2_cot.csv b_pal.csv]
                                 [--sidecar reconcile_trace.jsonl]
        Reads a finished v17 run. Reports what shipped, what it cost, the
        risk-coverage curve of the certificate, and the PAIRED comparison
        against any external baselines given -- which, unlike every version up
        to v16, is now the only comparison that exists, because the system no
        longer contains a baseline to quote.

What to look at first, in order:

1. The paired deltas against b2_cot and b_pal. If MAS is not above both, the
   accuracy claim is not there, and the honest report says so.
2. The certificate's risk-coverage. Coverage at a given risk is the claim this
   version is actually built on: the system knows when it knows.
3. `stage`. `agree` is free evidence; `unresolved` is what the third derivation
   costs money on. If `unresolved` is small, most of the budget is being saved.
4. `certified` against correctness. A certificate that is not much more precise
   than the base rate is not a certificate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import reconcile

STORED_RUNS = {
    'v14.8': 'results_August/full_mas_20260819.csv',
    'v15.2': 'results_August/mas_full_20260824.csv',
    'v15.5': 'results_August/fullmas_20260830.csv',
    'v15.6': 'results_September/results/MAS_SHT/results/mas_sht_math7b_20260912_094311.csv',
}
PAL_RUN = 'results_September/b_pal_20260915.csv'


def num(x) -> Optional[float]:
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def ok(v: Optional[float], gold: Optional[float]) -> bool:
    """The grader, at relative 1e-4. Absolute 1e-3 is kept as a floor so small
    integer answers behave as they always did."""
    if v is None or gold is None:
        return False
    return abs(v - gold) <= max(1e-3, 1e-4 * abs(gold))


def mcnemar(wins: int, losses: int) -> float:
    """Exact two-sided binomial p for a paired comparison."""
    try:
        from scipy.stats import binomtest
        return float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
    except Exception:
        return float('nan')


# =====================================================================
# Mode 1 — offline replay over the stored runs
# =====================================================================

def replay() -> int:
    try:
        from grounding_probe import load_texts
        texts = load_texts()
    except Exception as exc:
        print(f"cannot load problem texts ({type(exc).__name__}: {exc}).")
        print("This mode needs the cached HF datasets; --run does not.")
        return 2

    pal = pd.read_csv(PAL_RUN).set_index('problem_id').predicted.map(num) \
        if os.path.exists(PAL_RUN) else pd.Series(dtype=float)

    rows = []
    for run, path in STORED_RUNS.items():
        if not os.path.exists(path):
            print(f"  (skipping {run}: {path} not present)")
            continue
        for _, r in pd.read_csv(path).iterrows():
            try:
                bp = {'givens': json.loads(r.blueprint_givens),
                      'equations': json.loads(r.blueprint_equations)}
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue
            if not bp['givens'] or not bp['equations']:
                continue
            text = texts.get(r.problem_id, '')
            if not text:
                continue
            gold = num(r.gold)
            refs = [('baseline', num(r.baseline_ans))]
            if run == 'v15.6':
                refs.append(('pal', pal.get(r.problem_id)))
            for refname, ref in refs:
                if ref is None:
                    continue
                bp_val = reconcile._evaluate_blueprint(bp, ref)
                rec = reconcile.reconcile(bp, bp_val, ref, text)
                rows.append(dict(
                    run=run, pid=r.problem_id, ref=refname, gold=gold,
                    ref_ok=ok(ref, gold), bp_ok=ok(bp_val, gold),
                    stage=rec.stage, certified=rec.certified,
                    shipped=rec.answer if rec.answer is not None else ref,
                    given=(rec.repair.given if rec.repair else ''),
                    declared=(rec.repair.declared if rec.repair else None),
                    wanted=(rec.repair.reconstructed if rec.repair else None)))

    R = pd.DataFrame(rows)
    if R.empty:
        print("no evaluable rows found")
        return 2
    R['ship_ok'] = [ok(num(s), g) for s, g in zip(R.shipped, R.gold)]

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)
    for refname in [x for x in ('baseline', 'pal') if (R.ref == x).any()]:
        S = R[R.ref == refname]
        print(f"\n{'=' * 72}\nsecond derivation = {refname}   n={len(S)}\n{'=' * 72}")
        print(S.groupby('stage').agg(n=('stage', 'size'),
                                     reference_correct=('ref_ok', 'mean'),
                                     blueprint_correct=('bp_ok', 'mean'),
                                     shipped_correct=('ship_ok', 'mean')).round(3))
        C = S[S.certified]
        U = S[~S.certified]
        print(f"\ncertificate: coverage {len(C) / len(S):.1%}  "
              f"precision {C.ship_ok.mean():.1%}  |  "
              f"uncertified n={len(U)} at {U.ref_ok.mean():.1%}")

        Q = S[S.stage == reconcile.REPAIRED_BLUEPRINT]
        if len(Q):
            print(f"\ninverse-grounded operand repairs: {len(Q)}, "
                  f"{Q.ship_ok.sum()}/{len(Q)} correct")
            print(Q[['run', 'pid', 'given', 'declared', 'wanted', 'ship_ok']]
                  .to_string(index=False))
            print("\nby run (the point: v15.0-15.2 forward snapping already removes")
            print("this defect class where it can, so the inverse pass is a safety")
            print("net, not a source of accuracy on the current pipeline):")
            print(Q.groupby('run').size().to_string())
    return 0


# =====================================================================
# Mode 2 — a finished v17 run
# =====================================================================

def analyse_run(path: str, cmp_paths: List[str], sidecar: Optional[str]) -> int:
    d = pd.read_csv(path)
    n = len(d)
    print(f"\n{'=' * 72}\n{path}   n={n}   solver_version="
          f"{d.solver_version.iloc[0] if 'solver_version' in d else '?'}\n{'=' * 72}")

    d['gold_n'] = d.gold.map(num)
    d['pred_n'] = d.predicted.map(num)
    d['ok'] = [ok(p, g) for p, g in zip(d.pred_n, d.gold_n)]

    stage = d.get('sht_triage', pd.Series(['?'] * n)).fillna('?')
    strat = d.get('sht_final_strategy', pd.Series([''] * n)).fillna('')
    calls = d.get('sht_api_calls', d.get('num_llm_calls', pd.Series([np.nan] * n)))

    print(f"\naccuracy {d.ok.mean():.2%}  ({d.ok.sum()}/{n})")
    print(f"LLM calls per problem: mean {calls.mean():.2f}  total {calls.sum():.0f}")

    if 'baseline_ans' in d and d.baseline_ans.notna().any():
        print("\nNOTE: baseline_ans is populated, so this run had an INTERNAL "
              "baseline.\nThat is the v16 configuration, not v17 -- check "
              "MAS_INTERNAL_BASELINE.")

    print("\nby reconciliation stage:")
    t = pd.DataFrame({'stage': stage, 'ok': d.ok, 'calls': calls})
    print(t.groupby('stage').agg(n=('ok', 'size'), accuracy=('ok', 'mean'),
                                 calls=('calls', 'mean')).round(3))
    print("\nby what decided the answer:")
    t2 = pd.DataFrame({'strategy': strat, 'ok': d.ok})
    print(t2.groupby('strategy').agg(n=('ok', 'size'), accuracy=('ok', 'mean')).round(3))

    # --- the certificate ---------------------------------------------------
    # A certified row is one where two derivations produced in different
    # languages agreed, with or without a single repaired operand.
    certified = stage.isin([reconcile.AGREE, reconcile.REPAIRED_BLUEPRINT,
                            reconcile.REPAIRED_PROGRAM]) | \
        (strat == 'third_confirms_blueprint')
    cov = certified.mean()
    risk_cov = 1 - d.ok[certified].mean() if certified.any() else float('nan')
    risk_all = 1 - d.ok.mean()
    print(f"\ncertificate: coverage {cov:.1%} at risk {risk_cov:.2%}  "
          f"(risk over all rows {risk_all:.2%}, "
          f"{risk_all / risk_cov:.1f}x reduction)"
          if certified.any() and risk_cov > 0 else
          f"\ncertificate: coverage {cov:.1%}, zero errors on covered rows "
          f"(risk over all rows {risk_all:.2%})")
    if (~certified).any():
        print(f"uncertified: n={int((~certified).sum())} at "
              f"{d.ok[~certified].mean():.1%} accuracy")

    # --- candidate marginals ----------------------------------------------
    for col, label in (('cand_program', 'independent program'),
                       ('cand_blueprint', 'architect equations (CAS)'),
                       ('cand_third', 'third derivation')):
        if col in d:
            v = d[col].map(num)
            m = v.notna()
            if m.any():
                acc = np.mean([ok(a, g) for a, g in zip(v[m], d.gold_n[m])])
                print(f"  {label:<28} present {m.sum():>4}  accuracy {acc:.2%}")
    if 'cand_program' in d and 'cand_blueprint' in d:
        p, b = d.cand_program.map(num), d.cand_blueprint.map(num)
        m = p.notna() & b.notna()
        oracle = np.mean([ok(x, g) or ok(y, g)
                          for x, y, g in zip(p[m], b[m], d.gold_n[m])])
        ident = np.mean([reconcile.agrees(x, y, 1e-9) for x, y in zip(p[m], b[m])])
        print(f"  oracle over the two derivations      {oracle:.2%}   "
              f"(they are identical on {ident:.1%} of rows)")
        print("  If that identity rate approaches 1.0 the derivations have "
              "re-coupled;\n  on the v16 pipeline it was 94% and the oracle "
              "equalled the program alone.")

    # --- paired comparisons ------------------------------------------------
    for cp in cmp_paths:
        if not os.path.exists(cp):
            print(f"\n(comparator {cp} not found)")
            continue
        c = pd.read_csv(cp)
        name = c.system.iloc[0] if 'system' in c else os.path.basename(cp)
        c = c[['problem_id', 'predicted']].rename(columns={'predicted': 'cmp'})
        j = d.merge(c, on='problem_id')
        if j.empty:
            print(f"\n(comparator {name}: no overlapping problems)")
            continue
        j['cmp_ok'] = [ok(num(v), g) for v, g in zip(j.cmp, j.gold_n)]
        w = int((j.ok & ~j.cmp_ok).sum())
        l = int((~j.ok & j.cmp_ok).sum())
        print(f"\nvs {name}  (paired, n={len(j)}): "
              f"MAS {j.ok.mean():.2%} vs {j.cmp_ok.mean():.2%}  "
              f"delta {100 * (j.ok.mean() - j.cmp_ok.mean()):+.2f}pp  "
              f"W={w} L={l}  p={mcnemar(w, l):.3f}")

    # --- sidecar -----------------------------------------------------------
    if sidecar and os.path.exists(sidecar):
        recs = [json.loads(x) for x in open(sidecar, encoding='utf-8') if x.strip()]
        print(f"\nsidecar: {len(recs)} records")
        reps = [r['reconcile']['repair'] for r in recs
                if r.get('reconcile', {}).get('repair')]
        if reps:
            print(f"operand repairs: {len(reps)}")
            for r in reps[:20]:
                print(f"  {r['side']:<9} {r['given']:<34} "
                      f"{r['declared']!r} -> {r['reconstructed']!r}")
        else:
            print("no operand repairs fired (expected on a post-v15.2 pipeline: "
                  "forward snapping\nalready removes this defect class where it can)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--replay', action='store_true',
                    help='reproduce the offline evidence over the four stored runs')
    ap.add_argument('--run', help='a finished v17 run CSV')
    ap.add_argument('--cmp', nargs='*', default=[],
                    help='external baseline CSVs to pair against (b2_cot, b_pal)')
    ap.add_argument('--sidecar', help='reconcile_trace.jsonl from the same run')
    a = ap.parse_args()
    if a.replay:
        return replay()
    if a.run:
        return analyse_run(a.run, a.cmp, a.sidecar)
    ap.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
