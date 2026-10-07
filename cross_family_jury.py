"""
[v31.0] Cross-Family Tie Jury (XJ): where v22's verifier is blind, ask a
model from ANOTHER family to vote.

WHY THIS VERSION EXISTS
-----------------------
Five pre-registered screens (v24, v27, v28, v29, v30) measured the same thing:
inside the Qwen family nothing checks fidelity to the problem better than the
generator. The Reader recites, the PRM is problem-insensitive (r = 0.93, and
0.87 even after re-reading), a same-family questioner cannot localise the
dispute, and training on 59 pairs does not transfer. Every one of those
agents shares the generator's reading.

The literature says where the missing information is:
  * Tan et al., EMNLP 2025 ("Too Consistent to Detect", arXiv 2505.17656):
    self-consistent errors are MODEL-SPECIFIC; cross-family overlap is
    5.4-20.4%. They only DETECT them (a trained probe on a second model's
    hidden states) and state that mitigation is open.
  * "LLMs as a Jury" (arXiv 2607.10139): independently trained models err
    differently, so cross-model agreement on the final answer can beat PRMs.
    Its stated ceiling is a "shared-error floor" where models share a
    misconception; it never looks at recitation of perturbed problems, where
    every family has memorised the same GSM8K originals.
  * Thesis finding (v14, v22 ADAPT): verification works as ROUTING, not as
    arbitration. v22's PRM is decisive on most rows; it is blind exactly on
    its contested ties (64/100 dev rows; argmax right on 48/57 = 84%).

XJ composes those three: the PRM decides wherever it decides; only inside its
contested top tie, a juror from another family (DeepSeek-Math-7B-RL: other
lab, other pre-training corpus, RL-trained) solves the problem k times and
the tied answer it reproduces most often wins. The juror never sees a
candidate, so it cannot adopt a reading (v24's failure mode), and it answers
the whole problem, so nobody has to localise the dispute (v29's failure).

The open question it measures is the Jury paper's ceiling on THIS error type:
do two families share P2's recitation errors (shared-error floor), or are
they model-specific (Tan et al.)?
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import null_hypothesis as NH
import pretest_v21 as V21
import reread_verification as RV
import score_prm_v22 as P

JUROR = 'deepseek-ai/deepseek-math-7b-rl'
JUROR_2 = 'allenai/OLMo-2-1124-7B-Instruct'      # exploratory second family
K_JUROR = 5
TEMPERATURE = 0.8
MAX_TOKENS = 1024
PRIMARY_POOL = 'B10'

# pre-registered bars (frozen 2026-10-06, before any juror sample exists)
GO_NET = 3                       # XJ-B10 vs B10 on the 100 dev main rows: net >= +3 with W >= 2L
GO_RATIO = 2
GUARD_MIN_NET = -1               # the 20 GSM-Plus guard rows
HELD_MIN_NET = 0                 # the 100 held-out v21 main rows: XJ-B5 - B5 >= 0 -> CONSISTENT


def agree(a: Optional[float], b: Optional[float]) -> bool:
    return a is not None and b is not None and (V21.correct(a, b) or V21.correct(b, a))


def xj_pick(cands: List[Dict], votes: Optional[Sequence[Optional[float]]]) -> Tuple[Optional[int], str]:
    """Index of the chosen candidate and why. With votes=None this is v22.

    Inside the contested tie, each distinct answer scores the number of juror
    answers that agree with it. The highest score wins; the winner's
    representative is its best candidate in v22's order (higher last-step
    reward, then lower pool index). Score ties between answers, or no juror
    agreement with any tied answer, keep v22's pick.
    """
    i0, why = NH.pick(cands, False)
    if i0 is None or votes is None:
        return i0, why
    tie = RV.contested_tie(cands)
    if not tie:
        return i0, 'no contested tie'
    groups = P.L.clusters([cands[i]['answer'] for i in tie])
    scored = []
    for g in groups:
        members = [tie[k] for k in g]
        rep = max(members, key=lambda i: (cands[i]['real'], -i))
        scored.append((sum(agree(v, cands[rep]['answer']) for v in votes), rep))
    best = max(s for s, _ in scored)
    if best == 0:
        return i0, 'no juror agreement'
    top = [rep for s, rep in scored if s == best]
    if len(top) > 1:
        return i0, 'juror tie'
    return top[0], 'jury'


def choose(row: Dict, pool: str, votes: Optional[Sequence[Optional[float]]]) -> Optional[float]:
    cands = RV.candidates(row, pool)
    i, _ = xj_pick(cands, votes)
    return None if i is None else cands[i]['answer']


def plain_vote(answers: Sequence[Optional[float]]) -> Optional[float]:
    """Plurality over answers; ties go to the first seen (EXPLORATORY Jury-10 control)."""
    groups = P.L.clusters(list(answers))
    return None if not groups else answers[groups[0][0]]


def paired(rows: Sequence[Dict], recs: Dict, pool: str, juror: str = JUROR,
           against: Optional[str] = None) -> Dict:
    """XJ on `pool` vs v22's rule on `against` (default: the same pool). A row
    without juror votes keeps v22's pick."""
    import reading_arbitration as RA
    wins, losses, a_right, b_right = [], [], 0, 0
    for r in rows:
        votes = votes_of(recs.get(r['key'], {}), juror)
        a = V21.correct(choose(r, pool, votes), r['gold'])
        b = V21.correct(choose(r, against or pool, None), r['gold'])
        a_right += a
        b_right += b
        if a and not b:
            wins.append(r['pid'])
        if b and not a:
            losses.append(r['pid'])
    return {'xj': a_right, 'base': b_right, 'w': len(wins), 'l': len(losses),
            'net': len(wins) - len(losses), 'p': RA.sign_p(len(wins), len(losses)),
            'wins': wins, 'losses': losses, 'n': len(rows)}


def votes_of(rec: Dict, juror: str = JUROR) -> Optional[List[Optional[float]]]:
    s = (rec.get('jurors') or {}).get(juror)
    return None if not s else [x.get('answer') for x in s]


def screen_verdict(main: Sequence[Dict], guard: Sequence[Dict], recs: Dict) -> List[str]:
    c = paired(main, recs, PRIMARY_POOL)
    facts = (f"XJ-B10 {c['xj']} vs B10 {c['base']}: W={c['w']} L={c['l']} "
             f"net {c['net']:+d} (sign p={c['p']:.3f})")
    if c['net'] >= GO_NET and c['w'] >= GO_RATIO * c['l']:
        out = [f"SCREEN   GO: {facts} -> build the fresh-row confirmation"]
    elif c['net'] <= 0:
        out = [f"SCREEN   STOP: {facts}"]
    else:
        out = [f"SCREEN   WEAK: {facts}"]
    if guard:
        g = paired(guard, recs, PRIMARY_POOL)
        out.append(f"GUARD    XJ-B10 - B10 = {g['net']:+d} rows (W={g['w']} L={g['l']}) -> "
                   + ('no harm' if g['net'] >= GUARD_MIN_NET else 'HARM'))
    return out


def held_verdict(held: Sequence[Dict], recs: Dict) -> str:
    h = paired(held, recs, 'B5')
    facts = f"XJ-B5 {h['xj']} vs B5 {h['base']} on {h['n']} held-out rows: W={h['w']} L={h['l']}"
    return (f"HELD     CONSISTENT: {facts}" if h['net'] >= HELD_MIN_NET
            else f"HELD     INCONSISTENT: {facts} -> do not run the confirmation before discussing")
