"""
[v30.0] Dev screen: Re-Read Verification (RRV). When v22's verifier ties
different answers, does the SAME verifier, made to read the problem again
after each solution, pick the right one more often than v22's argmax?

The rule and the mechanism are in reread_verification.py; the evidence and the
literature in V30_PLAN.md. This screen GENERATES NOTHING: no solutions, no
questions, no readings. It re-scores stored samples with the stored verifier.
Inputs (the 100 main + 20 guard rows of the v22 confirmation, seed 45):
  results_September/preset_V25_dev.json  C (5 plain) + Q (5 Reader-view)
                                          samples with v22's step rewards
  results_September/preset_V26_dev.json  the Prototype agent's familiar
                                          versions (78 rows; mechanism only)

Offline, on the same stored pools: v22's rule gives B5 78, RA 80, RA5 81,
B10 81 (reproduced by test_v30.py). B10 has a contested tie on 64 main rows;
the argmax is right on 48 of the 57 that contain the right answer (84%), and
a perfect tie-breaker would give B10 90.

PRE-REGISTERED, frozen 2026-10-04 before any RRV score exists
------------------------------------------------------------
  SCREEN (100 main rows)  RRV-B10 vs B10 (v22's rule on the same 10 samples),
                          format V2, rule = rrv_key in reread_verification.py:
                          net >= +3 rows with W >= 2L -> GO: build the
                          fresh-row confirmation;  net <= 0 -> STOP;  otherwise WEAK
  GUARD (20 GSM-Plus rows) RRV-B10 - B10 >= -1 row -> no harm
  HELD (secondary)        the 100 v21 seed-44 main rows (pretest_data/v21_traces.json +
                          v22_dev_prm.json), which no version from v24 to v29 read:
                          RRV-B5 - B5 >= 0 -> CONSISTENT, else INCONSISTENT. Only B5
                          exists there and v22 wins 30 of the 34 winnable ties, so it
                          is a transfer / no-harm check, not a second route to GO.
  DRIFT (gate, not a bar) V0 re-scored on 3 stored samples must match the
                          stored last-step reward within 0.02, or the run stops
Reported, not decisive: RRV on the B5, RA and RA5 pools; RRV-RA vs B5 (the v22
system); the dispute-level accuracy against v22's 84%; the V1 control
(Leviathan's front repetition, same rule); the EXPLORATORY full re-ranking by
the RRV score alone; the AUROC of V0 vs V2 on tied samples; the V2p mechanism
check (does the score move when the re-read copy is the familiar version?).
These rows motivated the idea (the error analysis read them), so the screen
decides whether a confirmation on FRESH rows is worth running; it is never a
thesis claim by itself.

MODES
-----
    python pretest_v30.py --max-hours 3.0     # Kaggle or Colab, one T4; ~1 h to the verdict
    python pretest_v30.py --summary-only      # re-read, no GPU
    python pretest_v30.py --stub              # offline plumbing, no GPU
Re-running the same command resumes (every row is saved as it is scored).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import pretest_v21 as V21
import pretest_v24 as V24
import reread_verification as RV
import score_prm_v22 as P

DEV_V25 = 'results_September/preset_V25_dev.json'
DEV_V26 = 'results_September/preset_V26_dev.json'
HELD_TRACES = 'pretest_data/v21_traces.json'      # seed-44 rows: 5 plain samples each
HELD_PRM = 'pretest_data/v22_dev_prm.json'        # v22's step rewards for them
OUT = 'pretest_v30.json'
OUT_STUB = 'pretest_v30_stub.json'
POOL_ORDER = ('B10', 'RA', 'RA5', 'B5')
ALL_REFS = RV.refs('B10')
DRIFT_ROWS = 3


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def load_rows(v25_path: str = DEV_V25, v26_path: str = DEV_V26) -> List[Dict]:
    protos = {}
    if os.path.exists(v26_path):
        for r in V24._read(v26_path)['rows']:
            p = r.get('prototype') or {}
            if p.get('ok') and not p.get('identical') and p.get('text'):
                protos[r['pid']] = p['text']
    rows = []
    for r in V24._read(v25_path)['rows']:
        rows.append({'key': f"v30:{r['pid']}", 'pid': r['pid'], 'set': r['set'], 'gold': r['gold'],
                     'template': r.get('template'), 'text': r['text'], 'C': r['C'], 'Q': r['Q'],
                     'prototype': protos.get(r['pid'])})
    return rows


def _num(x) -> Optional[float]:
    try:
        return None if x in (None, '', 'None') else float(x)
    except (TypeError, ValueError):
        return None


def load_held_rows(traces: str = HELD_TRACES, prm: str = HELD_PRM) -> List[Dict]:
    """The v21 seed-44 rows (100 main + 40 guard): 5 plain samples with v22's
    step rewards and no Reader view, so only the B5 pool exists. No version
    from v24 to v29 read them, so they are the held-out check of RRV-B5."""
    if not (os.path.exists(traces) and os.path.exists(prm)):
        return []
    texts: Dict[str, str] = {}
    for m in P.MANIFESTS:
        for r in V24._read(m)['rows']:
            texts[r['problem_id']] = r['text']
    rewards = {r['pid']: r for r in V24._read(prm)['rows']}
    rows = []
    for r in V24._read(traces)['rows']:
        if r['pid'] not in texts or r['pid'] not in rewards:
            continue
        cs = []
        for s, q in zip(r['samples'][:5], rewards[r['pid']]['samples'][:5]):
            cs.append({'raw': s['raw'], 'answer': _num(s.get('answer')), 'prm': q.get('prm') or None})
        rows.append({'key': f"v30h:{r['pid']}", 'pid': r['pid'],
                     'set': 'held' if r.get('set') == 'main' else 'held-guard',
                     'gold': _num(r['gold']), 'template': V24.template_of(r['pid']),
                     'text': texts[r['pid']], 'C': cs, 'Q': [], 'prototype': None})
    return rows


def is_held(row: Dict) -> bool:
    return row['set'].startswith('held')


def sample(row: Dict, ref: str) -> Dict:
    return row[ref[0]][int(ref[1:])]


# ---------------------------------------------------------------------------
# what each pass scores: (ref, format) pairs, highest priority first
# ---------------------------------------------------------------------------

def plan(row: Dict, stage: str) -> List[Tuple[str, str]]:
    if is_held(row):              # only the B5 pool exists there, and only pass 1b scores it
        return [(ref, 'V2') for ref in RV.needs(row, ['B5'])] if stage == STAGES[1] else []
    if stage == STAGES[1]:
        return []
    tie_b10 = RV.needs(row, ['B10'])
    tie_all = RV.needs(row, POOL_ORDER)
    if stage == 'pass 1 (B10 ties, V2)':
        return [(ref, 'V2') for ref in tie_b10]
    if stage == 'pass 2 (other pools, V1 control)':
        return [(ref, 'V2') for ref in tie_all] + [(ref, 'V1') for ref in tie_all]
    if stage == 'pass 3 (mechanism, V2p)':
        return [(ref, 'V2p') for ref in tie_b10] if row.get('prototype') else []
    if stage == 'pass 4 (every sample, V2)':
        return [(ref, 'V2') for ref in ALL_REFS]
    raise ValueError(stage)


STAGES = ('pass 1 (B10 ties, V2)', 'pass 1b (held-out v21 rows, B5 ties, V2)',
          'pass 2 (other pools, V1 control)', 'pass 3 (mechanism, V2p)', 'pass 4 (every sample, V2)')


def missing(row: Dict, rec: Dict, stage: str) -> List[Tuple[str, str]]:
    sc = rec.get('scores', {})
    return [(ref, f) for ref, f in plan(row, stage)
            if sample(row, ref).get('answer') is not None and (sc.get(ref) or {}).get(f) is None]


def primary_complete(row: Dict, rec: Dict) -> bool:
    return not missing(row, rec, STAGES[0])


# ---------------------------------------------------------------------------
# the verifier
# ---------------------------------------------------------------------------

class LogOddsPRM(P.PRMScorer):
    """v22's scorer, returning the step log-odds (logit_pos - logit_neg) in
    float32 instead of the probability."""

    def score_lo(self, problem: str, steps: List[str]) -> List[float]:
        torch = self.torch
        msgs = [{'role': 'system', 'content': P.SYSTEM},
                {'role': 'user', 'content': problem},
                {'role': 'assistant', 'content': P.SEP.join(steps) + P.SEP}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = self.tok.encode(text, return_tensors='pt')
        ids = ids.to(next(self.model.parameters()).device)
        with torch.no_grad():
            logits = self.model(input_ids=ids)[0].float()[0]          # [seq, 2]
        sel = logits[ids[0] == self.sep_id]
        return (sel[:, 1] - sel[:, 0]).cpu().tolist()


class _StubPRM:
    """Offline: the last step's log-odds is higher for a right answer, with
    noise, so the plumbing has something to find."""

    def __init__(self, rows: Sequence[Dict]):
        self.right = {}
        for r in rows:
            for s in r['C'] + r['Q']:
                self.right[P.SEP.join(P.split_steps(s['raw']))] = V21.correct(s.get('answer'), r['gold'])

    def score_lo(self, problem: str, steps: List[str]) -> List[float]:
        own = steps[:-2] if len(steps) > 2 and steps[-2].startswith(RV.REREAD) else steps
        ok = self.right.get(P.SEP.join(own), False)
        rng = random.Random(len(problem) + sum(len(s) for s in steps))
        return [5.0] * (len(steps) - 1) + [(9.0 if ok else 7.0) + rng.gauss(0, 1.5)]


def build_prm(args, rows):
    if args.stub:
        return _StubPRM(rows)
    import pretest_v25 as V25
    return LogOddsPRM(P.PRM, four_bit=True, device_map={'': V25.roomiest_gpu()})


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x > -60 else 0.0


def drift_check(scorer, rows: Sequence[Dict], recs: Dict, stub: bool) -> None:
    """Gate: V0 re-scored must reproduce the stored last-step reward, or the
    chat template / step split / quantisation differ from v22's."""
    gaps = []
    for r in [r for r in rows if r['set'] == 'main'][:DRIFT_ROWS]:
        s = sample(r, 'C0')
        if s.get('prm') is None:
            continue
        lo = scorer.score_lo(*RV.inputs('V0', r['text'], s['raw'], s['answer'] or 0.0))[-1]
        gaps.append(abs(sigmoid(lo) - s['prm'][-1]))
    recs.setdefault('_meta', {})['drift'] = gaps
    print(f"  drift check (V0 vs stored last-step reward): {[round(g, 5) for g in gaps]}", flush=True)
    if gaps and max(gaps) > RV.DRIFT_MAX and not stub:
        raise RuntimeError(f"V0 does not reproduce the stored rewards (max gap {max(gaps):.4f}): "
                           "the verifier input differs from v22's; do NOT read RRV scores")


def score_one(scorer, row: Dict, ref: str, fmt: str) -> float:
    s = sample(row, ref)
    user, steps = RV.inputs(fmt, row['text'], s['raw'], s['answer'], row.get('prototype'))
    return float(scorer.score_lo(user, steps)[-1])


# ---------------------------------------------------------------------------
# reading the results
# ---------------------------------------------------------------------------

def auroc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def tie_samples(rows: Sequence[Dict], recs: Dict, fmt: str) -> List[Tuple[Dict, Dict]]:
    out = []
    for r in rows:
        cands = RV.candidates(r, 'B10', recs.get(r['key'], {}).get('scores'), fmt)
        out += [(r, cands[i]) for i in RV.contested_tie(cands) if cands[i]['rrv'] is not None]
    return out


def dispute_report(rows: Sequence[Dict], recs: Dict, pool: str = RV.PRIMARY_POOL,
                   fmt: str = RV.PRIMARY_FMT) -> List[str]:
    n = right_in = v22 = rrv = 0
    for r in rows:
        sc = recs.get(r['key'], {}).get('scores')
        cands = RV.candidates(r, pool, sc, fmt)
        tie = RV.contested_tie(cands)
        if not tie:
            continue
        n += 1
        if not any(V21.correct(cands[i]['answer'], r['gold']) for i in tie):
            continue
        right_in += 1
        v22 += V21.correct(RV.choose(r, pool, sc, False), r['gold'])
        rrv += V21.correct(RV.choose(r, pool, sc, True, fmt), r['gold'])
    if not right_in:
        return [f"{pool} {fmt}: no contested tie with the right answer"]
    return [f"{pool} {fmt}: {n} contested ties, {right_in} contain the right answer; right after the tie: "
            f"v22 {v22}/{right_in} ({100 * v22 / right_in:.0f}%), RRV {rrv}/{right_in} "
            f"({100 * rrv / right_in:.0f}%)"]


def mechanism_report(rows: Sequence[Dict], recs: Dict) -> List[str]:
    lines = []
    # does the score separate right from wrong inside the ties?
    for fmt in ('V2', 'V1'):
        ts = tie_samples(rows, recs, fmt)
        pos = [c['rrv'] for r, c in ts if V21.correct(c['answer'], r['gold'])]
        neg = [c['rrv'] for r, c in ts if not V21.correct(c['answer'], r['gold'])]
        a = auroc(pos, neg)
        p0 = [c['real'] for r, c in ts if V21.correct(c['answer'], r['gold'])]
        n0 = [c['real'] for r, c in ts if not V21.correct(c['answer'], r['gold'])]
        a0 = auroc(p0, n0)
        if a is not None:
            lines.append(f"AUROC inside B10's contested ties ({len(pos)} right, {len(neg)} wrong samples): "
                         f"v22 last step {a0:.2f}  vs  {fmt} {a:.2f}")
    # does the score move when the re-read copy is the familiar version?
    xs, ys, d_right, d_wrong = [], [], [], []
    for r in rows:
        sc = recs.get(r['key'], {}).get('scores', {})
        for ref, v in sc.items():
            if v.get('V2') is None or v.get('V2p') is None:
                continue
            xs.append(v['V2'])
            ys.append(v['V2p'])
            ok = V21.correct(sample(r, ref).get('answer'), r['gold'])
            (d_right if ok else d_wrong).append(v['V2'] - v['V2p'])
    if len(xs) >= 3:
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        sy = math.sqrt(sum((y - my) ** 2 for y in ys))
        rr = sxy / (sx * sy) if sx and sy else float('nan')
        mean = lambda v: sum(v) / len(v) if v else float('nan')
        lines.append(f"V2 vs V2p (real copy vs familiar copy), {len(xs)} samples: r = {rr:.2f} "
                     f"(v27's V0 under the prototype: 0.93); drop real->familiar: right answers "
                     f"{mean(d_right):+.2f}, wrong answers {mean(d_wrong):+.2f} log-odds")
    return lines


def summarise(rows: List[Dict], recs: Dict, stub: bool = False) -> int:
    held = [r for r in rows if r['set'] == 'held']
    held_done = all(not missing(r, recs.get(r['key'], {}), STAGES[1]) for r in rows if is_held(r))
    rows = [r for r in rows if not is_held(r)]
    main = [r for r in rows if r['set'] == 'main']
    guard = [r for r in rows if r['set'] == 'guard']
    prim = [r for r in rows if primary_complete(r, recs.get(r['key'], {}))]
    print('\n' + '=' * 78)
    print(f"  v30 Re-Read Verification: {len(prim)}/{len(rows)} rows complete for the "
          f"pre-registered reading (B10, V2)")
    drift = recs.get('_meta', {}).get('drift')
    if drift is not None:
        print(f"  drift gate (V0 re-scored vs stored): max {max(drift) if drift else 0:.5f}")
    print(f"\n    {'pool':5s} {'v22':>4s} {'RRV':>4s} {'W':>3s} {'L':>3s}   {'V1':>4s}   {'full':>4s}"
          f"   (100 main rows; V1 = front repetition, full = EXPLORATORY re-rank by V2 alone)")
    for pool in POOL_ORDER:
        s = RV.paired(main, recs, pool)
        v1 = RV.paired(main, recs, pool, fmt='V1')
        full = [RV.full_pick(r, pool, recs.get(r['key'], {}).get('scores')) for r in main]
        nf = sum(f is not None for f in full)
        fr = sum(V21.correct(f, r['gold']) for f, r in zip(full, main) if f is not None)
        print(f"    {pool:5s} {s['base']:4d} {s['rrv']:4d} {s['w']:3d} {s['l']:3d}   {v1['rrv']:4d}   "
              + (f"{fr:4d}" if nf == len(main) else f"  -- ({nf}/{len(main)} rows scored)"))
    sysc = RV.paired(main, recs, 'RA', against='B5')
    print(f"    RRV-RA vs B5 (the v22 system, 4 vs 5 solver samples): {sysc['rrv']} vs {sysc['base']}, "
          f"W={sysc['w']} L={sysc['l']}")
    for line in dispute_report(main, recs) + dispute_report(main, recs, fmt='V1') + mechanism_report(main, recs):
        print('  ' + line)
    c = RV.paired(main, recs, RV.PRIMARY_POOL)
    if c['wins'] or c['losses']:
        print(f"  flips on B10: wrong->right {c['wins']}; right->wrong {c['losses']}")
    if held:
        h = RV.paired(held, recs, 'B5')
        print(f"\n  held-out v21 rows (seed 44, {len(held)} main, B5 only): RRV-B5 {h['rrv']} vs B5 {h['base']}, "
              f"W={h['w']} L={h['l']}" + ('' if held_done else '   (PARTIAL)'))
        for line in dispute_report(held, recs, pool='B5'):
            print('  ' + line)
    print('\n  pre-registered reading:')
    if len(prim) < len(rows):
        print(f"    none: PARTIAL run, {len(prim)}/{len(rows)} rows complete for B10. Re-run the same command.")
        return 0
    for v in RV.screen_verdict(main, guard, recs):
        print('    ' + v)
    if held and held_done:
        print('    ' + RV.held_verdict(held, recs))
    return 0


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _save(path: str, recs: Dict, meta: Dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(dict(meta, rows=recs), fh, ensure_ascii=False)
    os.replace(tmp, path)


def run(args) -> int:
    rows = load_rows(args.v25, args.v26) + load_held_rows(args.held_traces, args.held_prm)
    out_path = args.out or (OUT_STUB if args.stub else OUT)
    recs: Dict[str, Dict] = {}
    if os.path.exists(out_path) and args.resume:
        recs = V24._read(out_path).get('rows', {})
        print(f"resuming: {len(recs)} rows in {out_path}")
    for r in rows:
        recs.setdefault(r['key'], {'scores': {}})
    meta = {'verifier': P.PRM, 'eps': RV.EPS, 'pools': RV.POOLS, 'formats': list(RV.FORMATS),
            'reread': RV.REREAD, 'repeat': RV.REPEAT, 'source_v25': args.v25, 'source_v26': args.v26}
    print(f"rows: {len(rows)} ({sum(r['set'] == 'main' for r in rows)} main), "
          f"with a usable prototype: {sum(bool(r['prototype']) for r in rows)}", flush=True)
    scorer = None
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else 0.0
    t0, made = time.time(), 0
    stages = STAGES if args.all_passes else STAGES[:4]
    for stage in stages:
        todo = [r for r in rows if missing(r, recs[r['key']], stage)]
        if not todo:
            continue
        print(f"\n{stage}: {len(todo)} rows, {sum(len(missing(r, recs[r['key']], stage)) for r in todo)} scores",
              flush=True)
        if scorer is None:
            scorer = build_prm(args, rows)
            if '_meta' not in recs or 'drift' not in recs['_meta']:
                drift_check(scorer, rows, recs, args.stub)
                _save(out_path, recs, meta)
        for i, row in enumerate(todo, 1):
            rec = recs[row['key']]
            for ref, fmt in missing(row, rec, stage):
                if deadline and time.time() > deadline:
                    _save(out_path, recs, meta)
                    print("\n  --max-hours reached; re-run the identical command to resume.")
                    return summarise(rows, recs, args.stub)
                rec['scores'].setdefault(ref, {})[fmt] = score_one(scorer, row, ref, fmt)
                made += 1
            _save(out_path, recs, meta)
            if stage in STAGES[:2]:
                pool = 'B5' if is_held(row) else 'B10'
                b = V21.correct(RV.choose(row, pool, None, False), row['gold'])
                a = V21.correct(RV.choose(row, pool, rec['scores'], True), row['gold'])
                print(f"[{i:3d}/{len(todo)}] {row['set']:5s} {row['pid'][:24]:24s} "
                      f"tie={len(RV.needs(row, [pool]))}  {pool}: v22 {'R' if b else '-'} RRV {'R' if a else '-'}  "
                      f"{(time.time() - t0) / max(1, made):4.1f}s/score", flush=True)
    _save(out_path, recs, meta)
    print(f"\n  saved: {out_path}")
    return summarise(rows, recs, args.stub)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--v25', default=DEV_V25)
    ap.add_argument('--v26', default=DEV_V26)
    ap.add_argument('--held-traces', default=HELD_TRACES)
    ap.add_argument('--held-prm', default=HELD_PRM)
    ap.add_argument('--out', default='')
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--max-hours', type=float, default=0.0)
    ap.add_argument('--all-passes', action='store_true',
                    help='also score every sample in V2 (needed only for the exploratory full re-rank)')
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()
    if args.summary_only:
        path = args.out or (OUT_STUB if args.stub else OUT)
        rows = load_rows(args.v25, args.v26) + load_held_rows(args.held_traces, args.held_prm)
        return summarise(rows, V24._read(path).get('rows', {}), args.stub)
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
