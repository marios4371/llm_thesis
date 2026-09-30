"""
[v27.0] Dev screen: Null-Hypothesis Verification (NHV). Does scoring each
candidate against the familiar version of the problem break the verifier's
saturated ties in favour of the solution that applies the problem's twist?

The rule is in null_hypothesis.py; the evidence and the literature in
V27_PLAN.md. This screen GENERATES NO SOLUTIONS. It reuses the stored v25 dev
pools (the 100 main + 20 guard rows of the v22 confirmation, seed 45):
  C  5 plain samples per row, with the verifier's rewards under the real problem
  Q  5 samples under the Reader's reading (v24 ASQ), same rewards
and adds, per row:
  a prototype   the v26 Prototype agent's familiar version (reused from the v26
                pilot for its 78 rows, written here for the other 42 with the
                same prompt and parser)
  null rewards  the same verifier (Qwen2.5-Math-PRM-7B) on every C and Q sample,
                with the PROTOTYPE as the problem
A row whose prototype is identical to the problem (or failed) gets no null
rewards: NHV equals v22 there by construction.

PRE-REGISTERED, frozen 2026-09-30, before any null reward exists
-----------------------------------------------------------------
  SCREEN (100 main rows)  NH-RA vs RA (RA = verifier best of 2 plain + 2
                          reader-view samples, v25):
                          net >= +3 rows with W >= 2L -> GO: build the fresh-row
                          confirmation;  net <= 0 -> STOP;  otherwise WEAK
  GUARD (20 GSM-Plus rows) NH-RA - RA >= -1 row -> no harm
  Reported, not decisive: NH on the B5, RA5 and B10 pools; NH-RA vs B5 (the
  v22 system); which picks the null test changed and how they ended; the null
  rewards on the self-consistent recitation rows.
These rows motivated the idea (the ceiling analysis looked at them), so the
screen decides whether a confirmation on FRESH rows is worth running; it is
never a thesis claim by itself. Cost note: NHV adds one call (the prototype)
and doubles the verifier's passes; the confirmation must compare at equal
calls (B6).

MODES
-----
    python pretest_v27.py --max-hours 3.0      # Kaggle (one T4 is enough), ~1-1.5 h
    python pretest_v27.py --summary-only       # re-read, no GPU
    python pretest_v27.py --stub --limit 8     # offline plumbing, no GPU
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

import null_hypothesis as NH
import pretest_v21 as V21
import pretest_v24 as V24
import pretest_v25 as V25
import pretest_v26 as V26
import prototype_contrast as PC
import score_prm_v22 as P

DEV_V25 = 'results_September/preset_V25_dev.json'
DEV_V26 = 'results_September/preset_V26_dev.json'
OUT = 'pretest_v27.json'
OUT_STUB = 'pretest_v27_stub.json'
VIEWS = ('C', 'Q')


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def load_rows(v25_path: str = DEV_V25, v26_path: str = DEV_V26) -> List[Dict]:
    protos = {r['pid']: r['prototype'] for r in V24._read(v26_path)['rows'] if 'prototype' in r}
    rows = []
    for r in V24._read(v25_path)['rows']:
        rec = {'key': f"v27:{r['pid']}", 'pid': r['pid'], 'set': r['set'], 'gold': r['gold'],
               'template': r.get('template'), 'text': r['text']}
        for v in VIEWS:
            rec[v] = [{'raw': s['raw'], 'answer': s.get('answer'), 'prm': s.get('prm')}
                      for s in r[v][:5]]
        if r['pid'] in protos:
            rec['prototype'] = dict(protos[r['pid']], source='v26')
        rows.append(rec)
    return rows


def merge(row: Dict, prior: Optional[Dict]) -> Dict:
    if not prior:
        return row
    if 'prototype' in prior:
        row['prototype'] = prior['prototype']
    for v in VIEWS:
        for s, p in zip(row[v], prior.get(v, [])):
            if p.get('pprm') is not None and p.get('answer') == s.get('answer'):
                s['pprm'] = p['pprm']
    for k in ('seconds',):
        if k in prior:
            row[k] = prior[k]
    return row


def lean(rec: Dict) -> Dict:
    """What the result file keeps: no raw texts (they stay in the v25 file)."""
    out = {k: rec[k] for k in ('key', 'pid', 'set', 'gold', 'template') if k in rec}
    for k in ('prototype', 'seconds'):
        if k in rec:
            out[k] = rec[k]
    for v in VIEWS:
        out[v] = [{k: s.get(k) for k in ('answer', 'prm', 'pprm') if k in s} for s in rec[v]]
    return out


def needs_null(rec: Dict) -> bool:
    return NH.usable_null(rec) and any(s.get('pprm') is None for v in VIEWS for s in rec[v])


# ---------------------------------------------------------------------------
# agents
# ---------------------------------------------------------------------------

class _StubReader:
    def call_model(self, msgs, **kw):
        text = msgs[-1]['content'].split('Problem: ', 1)[-1]
        sents = PC.SR.sentences(text)
        return ' '.join(sents[:-2] + sents[-1:]) if len(sents) > 2 else text


def _key(text: str) -> str:
    import re
    return re.sub(r'\s+', ' ', str(text)).strip()


class _StubPRM:
    """Offline: a right answer scores 1.0 under the real problem; under a
    prototype, samples whose answer is right score lower, so the null test has
    something to find."""

    def __init__(self, rows: List[Dict]):
        self.by = {}
        for r in rows:
            for v in VIEWS:
                for s in r[v]:
                    self.by[_key(s['raw'])] = V21.correct(s.get('answer'), r['gold'])

    def score(self, problem, steps):
        ok = self.by.get(_key('\n\n'.join(steps)), False)
        rng = random.Random(len(''.join(steps)))
        return [0.99] * (len(steps) - 1) + [0.4 + 0.2 * rng.random() if ok else 0.95]


def build_reader(args):
    if args.stub:
        return _StubReader()
    _solver, reader = V24.build(args, [])       # the solver client is never called
    return reader


def build_prm(args, rows):
    if args.stub:
        return _StubPRM(rows)
    return P.PRMScorer(P.PRM, four_bit=True, device_map={'': V25.roomiest_gpu()})


def _free(client) -> None:
    if client is not None and hasattr(client, '_local_model'):
        client._local_model = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def block(rows: List[Dict], title: str) -> None:
    if not rows:
        return
    n = len(rows)
    print('\n' + '=' * 78)
    usable = sum(NH.usable_null(r) for r in rows)
    print(f"  {title}  n={n}  rows with a usable, non-identical prototype: {usable}")
    print(f"    {'pool':5s} {'v22':>5s} {'NH':>5s}   NH vs v22")
    for pool in NH.POOLS:
        a = sum(NH.right(r, pool, False) for r in rows)
        b = sum(NH.right(r, pool, True) for r in rows)
        c = NH.paired(rows, (pool, True), (pool, False))
        print(f"    {pool:5s} {a:5d} {b:5d}   W={c['w']} L={c['l']} net={c['net']:+d}  sign p={c['p']:.3f}")
    c = NH.paired(rows, ('RA', True), ('B5', False))
    print(f"    NH-RA vs B5 (the v22 system, 5 calls; NH-RA uses 6): W={c['w']} L={c['l']} "
          f"net={c['net']:+d}  sign p={c['p']:.3f}")
    ch = NH.paired(rows, ('RA', True), ('RA', False))
    print(f"      NH-RA wins  {', '.join(p.replace('gsm-symbolic_', '') for p in ch['wins']) or '-'}")
    print(f"      NH-RA losses {', '.join(p.replace('gsm-symbolic_', '') for p in ch['losses']) or '-'}")
    reasons = {}
    for r in rows:
        cands = NH.candidates(r, 'RA')
        _, why = NH.pick(cands, NH.usable_null(r))
        reasons[why] = reasons.get(why, 0) + 1
    print(f"    how the RA pick was made: {reasons}")


def recitation_rows(rows: List[Dict]) -> None:
    rec = []
    for r in rows:
        ans = [s.get('answer') for s in r['C']]
        cl = P.L.clusters(ans)
        if cl and len(cl[0]) >= 4 and not any(V21.correct(a, r['gold']) for a in ans):
            rec.append(r)
    if not rec:
        return
    print(f"\n  self-consistent wrong rows (>=4 of 5 plain samples agree, none right): {len(rec)}")
    for r in rec:
        cands = NH.candidates(r, 'B10')
        wrong = [c for c in cands if not V21.correct(c['answer'], r['gold'])]
        good = [c for c in cands if V21.correct(c['answer'], r['gold'])]
        f = lambda cs, k: (f"{max(c[k] for c in cs if c[k] is not None):.3f}"
                           if any(c[k] is not None for c in cs) else '-')
        print(f"    {r['pid'][-8:]:>8s} proto={'same' if not NH.usable_null(r) else 'edited'}  "
              f"wrong: real {f(wrong, 'real') if wrong else '-'} null {f(wrong, 'null') if wrong else '-'}  |  "
              f"right: real {f(good, 'real') if good else '-'} null {f(good, 'null') if good else '-'}")


def summarise(recs: List[Dict], stub: bool) -> int:
    rows = [r for r in recs if NH.complete(r)]
    pr = [r['prototype'] for r in recs if 'prototype' in r]
    if pr:
        print(f"  prototypes: {len(pr)} ({sum(p.get('source') == 'v26' for p in pr)} reused from v26), "
              f"usable {sum(bool(p.get('ok')) for p in pr)}, identical {sum(bool(p.get('identical')) for p in pr)}")
    main = [r for r in rows if r['set'] == 'main']
    guard = [r for r in rows if r['set'] == 'guard']
    block(main, 'DEV MAIN')
    recitation_rows(main)
    block(guard, 'DEV GUARD')
    print('\n  pre-registered reading:')
    if not rows:
        print('    none: no complete rows yet')
        return 0
    if len(rows) < len(recs) and not stub:
        print(f"    none: PARTIAL run, {len(rows)}/{len(recs)} rows complete. "
              f"Re-run the identical command to finish.")
        return 0
    for v in NH.screen_verdict(main, guard):
        print('    ' + v)
    return 0


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _save(path: str, recs: List[Dict], meta: Dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(dict(meta, rows=[lean(r) for r in recs]), fh, ensure_ascii=False)
    os.replace(tmp, path)


def run(args) -> int:
    V26._PROTOS.update(n=0, ok=0, shown=False)
    rows = load_rows(args.v25, args.v26)
    if args.limit:
        rows = [r for r in rows if r['set'] == 'main'][:args.limit] + \
               [r for r in rows if r['set'] == 'guard'][:max(1, args.limit // 4)]
    out_path = args.out or (OUT_STUB if args.stub else OUT)
    prior = {}
    if os.path.exists(out_path) and args.resume and not (args.stub and out_path == OUT):
        prior = {r['key']: r for r in V24._read(out_path).get('rows', [])}
        print(f"resuming: {len(prior)} rows in {out_path}")
    recs = [merge(r, prior.get(r['key'])) for r in rows]
    meta = {'verifier': P.PRM, 'eps': NH.EPS, 'pools': NH.POOLS, 'source_v25': args.v25,
            'source_v26': args.v26}
    t0 = time.time()
    deadline = t0 + args.max_hours * 3600 if args.max_hours else 0.0
    todo_p = [r for r in recs if 'prototype' not in r]
    print(f"rows: {len(recs)}  prototypes to write: {len(todo_p)}  "
          f"rows to score: {sum(1 for r in recs if needs_null(r) or 'prototype' not in r)}")
    if todo_p:
        reader = build_reader(args)
        for i, r in enumerate(todo_p, 1):
            if deadline and time.time() > deadline:
                print("\n  --max-hours reached; re-run the identical command to resume.")
                _save(out_path, recs, meta)
                return summarise(recs, args.stub)
            try:
                r['prototype'] = dict(V26.write_prototype(reader, r['text']), source='v27')
            except RuntimeError:
                _save(out_path, recs, meta)
                raise
            p = r['prototype']
            print(f"  prototype [{i:3d}/{len(todo_p)}] {r['pid'][:24]:24s} "
                  f"{'same' if p['identical'] else ('FAILED ' + p['reason'] if not p['ok'] else 'edited')}",
                  flush=True)
            _save(out_path, recs, meta)
        _free(reader)
    todo = [r for r in recs if needs_null(r)]
    if todo:
        scorer = build_prm(args, recs)
        for i, r in enumerate(todo, 1):
            if deadline and time.time() > deadline:
                print("\n  --max-hours reached; re-run the identical command to resume.")
                break
            ts = time.time()
            proto = r['prototype']['text']
            for v in VIEWS:
                for s in r[v]:
                    if s.get('pprm') is None:
                        s['pprm'] = scorer.score(proto, P.split_steps(s['raw']))
            r['seconds'] = round(r.get('seconds', 0) + time.time() - ts, 1)
            c = NH.candidates(r, 'RA')
            a, _ = NH.pick(c, False)
            b, why = NH.pick(c, True)
            tag = '' if a == b else f"  RA pick changed: {c[a]['answer']:g} -> {c[b]['answer']:g}"
            print(f"  null [{i:3d}/{len(todo)}] {r['set']:5s} {r['pid'][:24]:24s} "
                  f"{(time.time() - t0) / i:5.0f}s/row{tag}", flush=True)
            _save(out_path, recs, meta)
    _save(out_path, recs, meta)
    print(f"\n  saved: {out_path}")
    return summarise(recs, args.stub)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--v25', default=DEV_V25)
    ap.add_argument('--v26', default=DEV_V26)
    ap.add_argument('--out', default='')
    ap.add_argument('--preset', default='qwen_math7b_mixed')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--max-hours', type=float, default=0.0)
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()
    if args.summary_only:
        path = args.out or OUT
        return summarise(V24._read(path)['rows'], args.stub)
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
