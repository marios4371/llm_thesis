"""
[v35.0] Offline replay of Trace-level SIV (trace_siv.py) on every stored pool
that has 5 PRM-scored plain samples per row. No model, no GPU.

Pools (rows x 5 samples, Qwen2.5-Math-7B-Instruct, T 0.8, Qwen2.5-Math-PRM-7B):
  P2a   v22 confirmation, seed 45   results_September/preset_v22.json
  P2b   v25 confirmation, seed 48   results_Octomber/pretest_v25_2ndsession.json
  MATH  v34 MATH-500, seed 34       results_Octomber/pretest_v34.json
  V21   v21 rows, seed 44           results_September/preset_V21.json + prm_scores_v22.json
  guards: the GSM-Plus rows of P2a, P2b and V21.
P2a, P2b and MATH are the rows the audit was DEBUGGED on (its false positives
were read there); V21 was never looked at before its numbers were printed.

    python replay_v35.py            # the table, the W/L and the audit statistics
    python replay_v35.py --rows     # also every row where ADAPT-R and ADAPT differ
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from math import comb
from typing import Dict, List, Optional

import pretest_v21 as V21
import trace_siv as T

POOLS = {
    'P2a': 'results_September/preset_v22.json',
    'P2b': 'results_Octomber/pretest_v25_2ndsession.json',
    'MATH': 'results_Octomber/pretest_v34.json',
    'V21': 'results_September/preset_V21.json',
}
V21_PRM = 'results_September/prm_scores_v22.json'
MATH_MANIFEST = 'pretest_data/v34_math500.json'


def _read(path: str) -> Dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def load(pool: str) -> List[Dict]:
    """Rows {pool, set, pid, gold, C: [sample]}; a sample has raw, answer, prm."""
    path = POOLS[pool]
    if not os.path.exists(path):
        return []
    d = _read(path)
    out = []
    if pool == 'P2a':
        for r in d['rows']:
            out.append({'set': r['set'], 'pid': r['pid'], 'gold': r['gold'], 'C': r['samples'][:5]})
    elif pool == 'P2b':
        for r in d['rows']:
            out.append({'set': r['set'], 'pid': r['pid'], 'gold': r['gold'], 'C': (r.get('C') or [])[:5]})
    elif pool == 'MATH':
        man = {r['problem_id']: r for r in _read(MATH_MANIFEST)['rows']}
        for k, r in d['rows'].items():
            out.append({'set': 'main', 'pid': k, 'gold': man[k]['gold'], 'C': r['C'][:5]})
    elif pool == 'V21':
        if not os.path.exists(V21_PRM):
            return []
        prm = {r['pid']: r for r in _read(V21_PRM)['rows']}
        for r in d['rows']:
            q = prm.get(r['pid'])
            if not q:
                continue
            c = [{'raw': s.get('raw', ''), 'answer': s.get('answer'), 'prm': p.get('prm')}
                 for s, p in zip(r['samples'][:5], q['samples'][:5])]
            out.append({'set': r['set'], 'pid': r['pid'], 'gold': r['gold'], 'C': c})
    for r in out:
        r['pool'] = pool
        for s in r['C']:
            s['ok'] = V21.correct(s.get('answer'), r['gold'])
            if 'audit' not in s:
                s['audit'] = T.audit(s.get('raw') or '')
    return out


# ---------------------------------------------------------------------------
# arms (all read off the same five samples)
# ---------------------------------------------------------------------------

def _plurality(samples, idx) -> Optional[float]:
    groups: List[List[int]] = []
    for i in idx:
        a = samples[i].get('answer')
        if a is None:
            continue
        for g in groups:
            if V21.correct(a, samples[g[0]]['answer']):
                g.append(i)
                break
        else:
            groups.append([i])
    if not groups:
        return None
    groups.sort(key=lambda g: (-len(g), g[0]))
    return samples[groups[0][0]]['answer']


def _adapt(samples, refute: bool):
    if refute:
        return T.adapt_r(samples)
    b = T.best(samples, range(T.GATE_K))
    if b is not None and min(samples[b]['prm']) >= T.GATE_MIN:
        return samples[b]['answer'], T.GATE_K
    b = T.best(samples, range(T.FULL_K))
    return (None if b is None else samples[b]['answer']), T.FULL_K


def _pick(samples, idx):
    b = T.best(samples, idx)
    return None if b is None else samples[b]['answer']


ARMS = {
    'SC5': lambda c: (_plurality(c, range(5)), 5),
    'SC5-R': lambda c: (_plurality(c, T.survivors(c, range(5)) or range(5)), 5),
    'B5': lambda c: (_pick(c, range(5)), 5),
    'B5-R': lambda c: (_pick(c, T.survivors(c, range(5)) or range(5)), 5),
    'ADAPT': lambda c: _adapt(c, False),
    'ADAPT-R': lambda c: _adapt(c, True),
}


def sign_p(w: int, l: int) -> float:
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def right(r: Dict, arm: str) -> bool:
    return V21.correct(ARMS[arm](r['C'])[0], r['gold'])


def table(groups: Dict[str, List[Dict]]) -> None:
    print(f"\n  {'arm':8s}" + ''.join(f"{g:>12s}" for g in groups) + '   calls/row (main)')
    for arm in ARMS:
        main = [r for g in groups.values() for r in g if r['set'] == 'main']
        calls = sum(ARMS[arm](r['C'])[1] for r in main) / max(1, len(main))
        print(f"  {arm:8s}" + ''.join(f"{sum(right(r, arm) for r in rows):5d}/{len(rows):<6d}"
                                      for rows in groups.values()) + f"   {calls:.2f}")


def wl(rows: List[Dict], x: str, y: str) -> str:
    w = sum(right(r, x) and not right(r, y) for r in rows)
    l = sum(right(r, y) and not right(r, x) for r in rows)
    return f"{x} vs {y}: W{w} L{l} net {w - l:+d} (sign p = {sign_p(w, l):.4f})"


def audit_stats(rows: List[Dict], name: str) -> None:
    s = [x for r in rows for x in r['C'] if x.get('answer') is not None]
    ok, bad = [x for x in s if x['ok']], [x for x in s if not x['ok']]
    f = lambda x: any(v == T.FALSE for st in x['audit'] for _, v in st)
    rf = lambda x: T.refutes(x['audit'], x.get('prm'))
    steps = [(x['prm'][i] if x.get('prm') and i < len(x['prm']) else None)
             for x in s for i, st in enumerate(x['audit']) if any(v == T.FALSE for _, v in st)]
    steps = [p for p in steps if p is not None]
    print(f"  {name:6s} a FALSE claim in {sum(map(f, ok)):3d}/{len(ok)} right and {sum(map(f, bad)):3d}/{len(bad)} "
          f"wrong samples; refuted {sum(map(rf, ok))} right / {sum(map(rf, bad))} wrong; "
          f"PRM >= 0.9 on {sum(p >= 0.9 for p in steps)}/{len(steps)} false steps")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', action='store_true')
    args = ap.parse_args()
    pools = {p: load(p) for p in POOLS}
    missing = [p for p, rows in pools.items() if not rows]
    if missing:
        print(f"  missing pools (not in this checkout): {', '.join(missing)}")
    main_ = {p: [r for r in rows if r['set'] == 'main'] for p, rows in pools.items() if rows}
    guard = [r for rows in pools.values() for r in rows if r['set'] != 'main']
    groups = dict(main_)
    if 'P2a' in main_ and 'P2b' in main_:
        groups = {'P2a': main_['P2a'], 'P2b': main_['P2b'], 'P2(a+b)': main_['P2a'] + main_['P2b'],
                  **{k: v for k, v in main_.items() if k not in ('P2a', 'P2b')}}
    groups['guard'] = guard
    print('=' * 78 + '\n  v35 Trace-level SIV: offline replay (right / rows)')
    table(groups)
    print()
    for name, rows in groups.items():
        if name in ('P2a', 'P2b'):
            continue
        for x, y in (('ADAPT-R', 'SC5'), ('ADAPT-R', 'ADAPT'), ('ADAPT-R', 'B5')):
            print(f"  {name:8s} {wl(rows, x, y)}")
    print('\n  audit (sample level):')
    for name, rows in groups.items():
        if name not in ('P2a', 'P2b'):
            audit_stats(rows, name[:6])
    if args.rows:
        print('\n  rows where ADAPT-R and ADAPT differ:')
        for name, rows in groups.items():
            for r in rows:
                a, b = right(r, 'ADAPT-R'), right(r, 'ADAPT')
                if a != b and name not in ('P2(a+b)',):
                    print(f"    {name:6s} {r['pid']:26s} ADAPT-R {'right' if a else 'WRONG'}  gold {r['gold']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
