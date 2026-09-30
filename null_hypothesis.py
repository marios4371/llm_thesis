"""
[v27.0] Null-Hypothesis Verification (NHV): let the verifier test each
candidate against the familiar version of the problem it may be reciting.

WHY (measured on stored runs, no GPU)
-------------------------------------
- The v22 verifier (Qwen2.5-Math-PRM-7B, best of k by last-step reward) is the
  thesis's confirmed result: B5 78 vs SC@5 68 on fresh P2 rows. But on the 8
  rows where >= 4 of 5 samples agree on a WRONG answer, it gives that answer a
  last-step reward of 1.000 every time, and a min-step reward of 0.96-1.00 on
  7 of them. It shares the Solver's reading: "2 + 12 = 14 m per plant" when
  the text says the 12 m already include the width. It catches the one
  arithmetic slip (min 0.32) and none of the misreadings.
- A second reading does not help: the Qwen Reader writes the same "14 m per
  plant" in its own notes (v25 dev).
- What does see the twist is the v26 Prototype agent: asked for the familiar
  version, it deletes exactly "(including the plants width)". The model can
  NAME what is unusual; it does not APPLY it.
- v26 showed where that knowledge must NOT go: into generation. Truncating or
  contrasting the Solver's distribution kills the low-probability branches
  that carry the right reading on hard rows (-9.2 pp per sample).

THE RULE
--------
Generation is left alone. The verifier scores every candidate twice: under the
real problem (as in v22) and under the prototype, the "null hypothesis" that
the problem is its familiar version. A candidate that recites is a correct
solution of the null, so both rewards are high. A candidate that applies the
twist is correct for the real problem and WRONG for the null, so the null
reward drops. Hence:
    1. top = the best real last-step reward in the pool;
    2. T   = candidates within EPS = 1e-3 of top (the saturated tie: 42 of 100
             dev rows have two or more different answers in it);
    3. if the prototype is identical to the problem (or failed), or T holds a
       single answer, choose as v22 does;
    4. otherwise choose the candidate in T with the LOWEST null reward;
       equal null rewards fall back to v22's order (higher real reward,
       then pool position), so an uninformative null test changes nothing.
It only reorders candidates the verifier already rates as tied, so it can
never pick a candidate v22's verifier rates clearly worse. In effect it turns
the verifier's own recitation bias into a detector for it.

WHAT IS AND IS NOT NEW (checked 2026-09-30, V27_PLAN.md)
-------------------------------------------------------
Not new: PRM best-of-N (v22's base); counterfactual inputs as a test in
general (Isomorphic Perturbation Testing, arXiv 2604.15149, uses ISOMORPHIC
perturbations to detect reward hacking in RL on logic induction; the
counterfactual-contexts beam search, arXiv 2609.37041, contrasts fixed "good"
/ "poor reasoning" templates).
New here, as far as the searches found: scoring candidates against a
counterfactual CANONICAL version of the same problem (written by an agent),
at inference time, to separate recited from problem-specific solutions.

Everything here is model-free and tested offline (test_v27.py).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pretest_v21 as V21
import score_prm_v22 as P

EPS = 1e-3
# pool -> (plain samples, reader-view samples); drawn in order, plain first
POOLS = {'B5': (5, 0), 'RA': (2, 2), 'RA5': (3, 2), 'B10': (5, 5)}
PRIMARY_POOL = 'RA'

# pre-registered bars (frozen 2026-09-30, before any null reward exists)
GO_NET = 3            # NH-RA vs RA on the 100 dev main rows: net >= +3 with W >= 2L
GO_RATIO = 2
GUARD_MIN_NET = -1    # the 20 GSM-Plus guard rows: NH-RA - RA >= -1


def last(prm: Optional[Sequence[float]]) -> Optional[float]:
    return None if prm is None else P.agg(list(prm), 'last')


def candidates(rec: Dict, pool: str) -> List[Dict]:
    c, q = POOLS[pool]
    out = []
    for view, n in (('C', c), ('Q', q)):
        for s in rec[view][:n]:
            out.append({'view': view, 'answer': s.get('answer'), 'real': last(s.get('prm')),
                        'null': last(s.get('pprm'))})
    return out


def usable_null(rec: Dict) -> bool:
    p = rec.get('prototype') or {}
    return bool(p.get('ok')) and not p.get('identical')


def pick(cands: List[Dict], use_null: bool) -> Tuple[Optional[int], str]:
    """Index of the chosen candidate and why. v22's rule is use_null=False."""
    idx = [i for i, c in enumerate(cands) if c['answer'] is not None and c['real'] is not None]
    if not idx:
        return None, 'empty'
    top = max(cands[i]['real'] for i in idx)
    b = max(idx, key=lambda i: (cands[i]['real'], -i))
    if not use_null:
        return b, 'v22'
    tie = [i for i in idx if cands[i]['real'] >= top - EPS]
    if len(P.L.clusters([cands[i]['answer'] for i in tie])) <= 1:
        return b, 'no contested tie'
    if any(cands[i]['null'] is None for i in tie):
        return b, 'null reward missing'
    # lowest null reward first; equal null rewards keep v22's order (higher
    # real reward, then pool position), so an uninformative null changes nothing
    return min(tie, key=lambda i: (cands[i]['null'], -cands[i]['real'], i)), 'null test'


def choose(rec: Dict, pool: str, nh: bool) -> Optional[float]:
    cands = candidates(rec, pool)
    i, _ = pick(cands, nh and usable_null(rec))
    return None if i is None else cands[i]['answer']


def right(rec: Dict, pool: str, nh: bool) -> bool:
    return V21.correct(choose(rec, pool, nh), rec['gold'])


def complete(rec: Dict) -> bool:
    """Every sample any pool uses has its real reward, and, when the row has a
    usable prototype, its null reward."""
    if 'prototype' not in rec:
        return False
    need_null = usable_null(rec)
    for view, n in (('C', 5), ('Q', 5)):
        ss = rec[view][:n]
        if len(ss) < n or any(s.get('prm') is None for s in ss):
            return False
        if need_null and any(s.get('pprm') is None for s in ss):
            return False
    return True


def paired(rows: Sequence[Dict], x: Tuple[str, bool], y: Tuple[str, bool]) -> Dict:
    w = [r['pid'] for r in rows if right(r, *x) and not right(r, *y)]
    l = [r['pid'] for r in rows if right(r, *y) and not right(r, *x)]
    import reading_arbitration as RA
    return {'w': len(w), 'l': len(l), 'net': len(w) - len(l), 'p': RA.sign_p(len(w), len(l)),
            'wins': w, 'losses': l}


def screen_verdict(main: Sequence[Dict], guard: Sequence[Dict]) -> List[str]:
    c = paired(main, (PRIMARY_POOL, True), (PRIMARY_POOL, False))
    facts = f"NH-RA vs RA: W={c['w']} L={c['l']} net={c['net']:+d} (sign p={c['p']:.3f})"
    if c['net'] >= GO_NET and c['w'] >= GO_RATIO * c['l']:
        out = [f"SCREEN   GO: the null test resolves the verifier's ties ({facts})"]
    elif c['net'] <= 0:
        out = [f"SCREEN   STOP: the null test does not help ({facts})"]
    else:
        out = [f"SCREEN   WEAK: positive but under the bar ({facts})"]
    if guard:
        g = paired(guard, (PRIMARY_POOL, True), (PRIMARY_POOL, False))
        out.append(f"GUARD    NH-RA - RA = {g['net']:+d} rows -> "
                   + ("no harm" if g['net'] >= GUARD_MIN_NET else "HARM"))
    return out
