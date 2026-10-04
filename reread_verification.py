"""
[v30.0] Re-Read Verification (RRV): the verifier reads the problem again
AFTER the solution, then scores the final answer once more.

WHY THIS VERSION EXISTS
-----------------------
v22's verifier (Qwen2.5-Math-PRM-7B, best last-step reward) is the thesis's one
confirmed gain, and the measured reason it stops gaining is that it does not
read the problem:
  * On the 100 dev main rows, B10 (5 plain + 5 Reader-view samples) has a
    dispute (different answers inside the PRM's top tie, 1e-3) on 64 rows. The
    argmax is right on 48 of the 57 disputes that contain the right answer.
  * The 9 lost disputes hold EXPLICIT misreadings that score 1.000, e.g.
    p2_1268: the picked solution writes "Since there are two groups" while the
    text says "These 3 groups"; its step reward dips to 0.787 and the last
    step is back at 1.00000.
  * v27 measured it directly: rewards under the familiar version of the
    problem correlate r = 0.93 with rewards under the real one.
  * Xu et al. 2025 (arXiv 2502.14619) found the same for four other reward
    models: removing the question barely moves the reward. Every remedy they
    list retrains the reward model.
  * v24/v27/v28/v29: no same-family agent checks problem fidelity better than
    the generator. RRV adds no agent and no generation; it changes only the
    order in which the existing verifier sees the text.

THE MECHANISM: CAUSAL ATTENTION
-------------------------------
In the PRM's input the problem comes first, so every problem token is encoded
before the solution exists. A misreading can be noticed only by the solution
tokens looking back, and the last-step reward evidently reads that comparison
weakly. Leviathan et al. 2025 (arXiv 2512.14982) show that simply repeating
the prompt helps NON-reasoning LLMs (47 wins, 0 losses in 70 tests), most when
the material to judge precedes the question (options-first multiple choice),
because the second copy attends to the first. A PRM is the extreme
non-reasoning LLM: one forward pass, no chain of thought. Their future work
lists partial repetition and attention analysis; it never mentions verifiers.
RE2 (Xu et al., EMNLP 2024) re-reads for the generator, not the judge.

RRV puts a second copy of the problem AFTER the solution, so the problem is
re-encoded with the solution's claims in view, and then re-asserts the answer
as the final step. The reward of that step is the RRV score.

FORMATS (the PRM, its system prompt and its step separator are v22's)
---------------------------------------------------------------------
  V0  v22         user: problem              assistant: s1 .. sn
  V1  front copy  user: problem + "Let me repeat that: " + problem
                                             assistant: s1 .. sn
                  (Leviathan's verbose form: the literature control)
  V2  RRV         user: problem              assistant: s1 .. sn,
                                             "Let me read the problem again: " + problem,
                                             "So the answer is \\boxed{a}."
  V2p mechanism   as V2, but the re-read copy is the FAMILIAR version
                  (the v26 Prototype agent's text): a problem-aware verifier
                  must move when the copy changes; v27's V0 barely did.
Scores are log-odds of the last step (logit_pos - logit_neg) in float32, so
the saturation near probability 1 does not erase differences.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import null_hypothesis as NH
import pretest_v21 as V21
import score_prm_v22 as P

EPS = NH.EPS                     # 1e-3: the v25/v27/v29 definition of the PRM's top tie
POOLS = dict(NH.POOLS)           # B5 (5,0), RA (2,2), RA5 (3,2), B10 (5,5)
PRIMARY_POOL = 'B10'
PRIMARY_FMT = 'V2'
FORMATS = ('V2', 'V1', 'V2p')

REREAD = 'Let me read the problem again: '
REPEAT = 'Let me repeat that: '

# pre-registered bars (frozen 2026-10-04, before any RRV score exists)
GO_NET = 3                       # RRV-B10 vs B10 on the 100 dev main rows: net >= +3 with W >= 2L
GO_RATIO = 2
GUARD_MIN_NET = -1               # the 20 GSM-Plus guard rows: RRV-B10 - B10 >= -1
HELD_MIN_NET = 0                 # the 100 held-out v21 main rows: RRV-B5 - B5 >= 0 -> CONSISTENT
DRIFT_MAX = 0.02                # V0 re-scored vs the stored last-step reward (probability)


# ---------------------------------------------------------------------------
# the verifier's inputs
# ---------------------------------------------------------------------------

def fmt_answer(a: float) -> str:
    """132000 -> '132000', 11.11 -> '11.11', 0.5 -> '0.5'; never an exponent."""
    if a == int(a) and abs(a) < 1e15:
        return str(int(a))
    return f'{a:.6f}'.rstrip('0').rstrip('.')


def reread_steps(copy: str, steps: List[str], answer: float) -> List[str]:
    """V2 / V2p: the solution's steps, the problem read again, the answer again."""
    return list(steps) + [REREAD + str(copy).strip(), f'So the answer is $\\boxed{{{fmt_answer(answer)}}}$.']


def repeat_front(problem: str) -> str:
    """V1: Leviathan et al.'s verbose prompt repetition."""
    p = str(problem).strip()
    return f'{p}\n{REPEAT}{p}'


def inputs(fmt: str, problem: str, raw: str, answer: float,
           prototype: Optional[str] = None) -> Tuple[str, List[str]]:
    """(user text, assistant steps) the PRM scores for one sample in one format."""
    steps = P.split_steps(raw)
    if fmt == 'V0':
        return problem, steps
    if fmt == 'V1':
        return repeat_front(problem), steps
    if fmt == 'V2':
        return problem, reread_steps(problem, steps, answer)
    if fmt == 'V2p':
        if not prototype:
            raise ValueError('V2p needs a prototype')
        return problem, reread_steps(prototype, steps, answer)
    raise ValueError(fmt)


# ---------------------------------------------------------------------------
# candidates and the rule
# ---------------------------------------------------------------------------

def refs(pool: str) -> List[str]:
    c, q = POOLS[pool]
    return [f'C{j}' for j in range(c)] + [f'Q{j}' for j in range(q)]


def candidates(row: Dict, pool: str, scores: Optional[Dict] = None, fmt: str = PRIMARY_FMT) -> List[Dict]:
    """NH.candidates (v22's stored last-step reward as `real`) plus each
    sample's reference ('C0', 'Q3', ...) and its RRV log-odds in `fmt` (None
    when not scored yet)."""
    out = NH.candidates(row, pool)
    scores = scores or {}
    for cand, ref in zip(out, refs(pool)):
        cand['ref'] = ref
        cand['rrv'] = (scores.get(ref) or {}).get(fmt)
    return out


def contested_tie(cands: List[Dict]) -> List[int]:
    """Indices inside the PRM's top tie (real >= top - EPS) when that tie holds
    two or more different answers; [] otherwise."""
    idx = [i for i, c in enumerate(cands) if c['answer'] is not None and c['real'] is not None]
    if not idx:
        return []
    top = max(cands[i]['real'] for i in idx)
    tie = [i for i in idx if cands[i]['real'] >= top - EPS]
    return tie if len(P.L.clusters([cands[i]['answer'] for i in tie])) >= 2 else []


def rrv_key(cand: Dict, index: int) -> tuple:
    """Sort key inside a contested tie; the candidate with the LARGEST key wins.

    This is the pre-registered rule. It is called only on candidates that sit
    in the PRM's contested top tie, and only once every one of them has an RRV
    score, so `cand['rrv']` (log-odds of the re-asserted answer, float) and
    `cand['real']` (v22's last-step probability, float) are both present.
    `index` is the position in the pool (C samples first); v22 breaks exact
    ties by the LOWER index.
    """
    # the re-read score decides; an exact RRV tie keeps v22's order
    return (cand['rrv'], cand['real'], -index)


def pick(cands: List[Dict], use_rrv: bool) -> Tuple[Optional[int], str]:
    """Index of the chosen candidate and why. use_rrv=False is v22's rule."""
    i0, why = NH.pick(cands, False)
    if not use_rrv or i0 is None:
        return i0, why
    tie = contested_tie(cands)
    if not tie:
        return i0, 'no contested tie'
    if any(cands[i]['rrv'] is None for i in tie):
        return i0, 'rrv missing'
    return max(tie, key=lambda i: rrv_key(cands[i], i)), 'rrv'


def choose(row: Dict, pool: str, scores: Optional[Dict], use_rrv: bool,
           fmt: str = PRIMARY_FMT) -> Optional[float]:
    cands = candidates(row, pool, scores, fmt)
    i, _ = pick(cands, use_rrv)
    return None if i is None else cands[i]['answer']


def full_pick(row: Dict, pool: str, scores: Optional[Dict], fmt: str = PRIMARY_FMT) -> Optional[float]:
    """EXPLORATORY: argmax of the RRV score over the whole pool (v22's reward
    ignored). None when any sample lacks a score."""
    cands = candidates(row, pool, scores, fmt)
    idx = [i for i, c in enumerate(cands) if c['answer'] is not None]
    if not idx or any(cands[i]['rrv'] is None for i in idx):
        return None
    return cands[max(idx, key=lambda i: (cands[i]['rrv'], -i))]['answer']


def needs(row: Dict, pools: Sequence[str]) -> List[str]:
    """The sample references a tie-break on `pools` needs, in pool order."""
    out: List[str] = []
    for pool in pools:
        cands = candidates(row, pool)
        for i in contested_tie(cands):
            if cands[i]['ref'] not in out:
                out.append(cands[i]['ref'])
    return out


# ---------------------------------------------------------------------------
# reading the results
# ---------------------------------------------------------------------------

def paired(rows: Sequence[Dict], recs: Dict, pool: str, fmt: str = PRIMARY_FMT,
           against: Optional[str] = None) -> Dict:
    """RRV on `pool` vs v22's rule on `against` (default: the same pool)."""
    import reading_arbitration as RA
    wins, losses, a_right, b_right = [], [], 0, 0
    for r in rows:
        sc = recs.get(r['key'], {}).get('scores', {})
        a = V21.correct(choose(r, pool, sc, True, fmt), r['gold'])
        b = V21.correct(choose(r, against or pool, None, False), r['gold'])
        a_right += a
        b_right += b
        if a and not b:
            wins.append(r['pid'])
        if b and not a:
            losses.append(r['pid'])
    return {'rrv': a_right, 'base': b_right, 'w': len(wins), 'l': len(losses),
            'net': len(wins) - len(losses), 'p': RA.sign_p(len(wins), len(losses)),
            'wins': wins, 'losses': losses, 'n': len(rows)}


def screen_verdict(main: Sequence[Dict], guard: Sequence[Dict], recs: Dict) -> List[str]:
    c = paired(main, recs, PRIMARY_POOL)
    facts = (f"RRV-B10 {c['rrv']} vs B10 {c['base']}: W={c['w']} L={c['l']} "
             f"net {c['net']:+d} (sign p={c['p']:.3f})")
    if c['net'] >= GO_NET and c['w'] >= GO_RATIO * c['l']:
        out = [f"SCREEN   GO: {facts} -> build the fresh-row confirmation"]
    elif c['net'] <= 0:
        out = [f"SCREEN   STOP: {facts}"]
    else:
        out = [f"SCREEN   WEAK: {facts}"]
    if guard:
        g = paired(guard, recs, PRIMARY_POOL)
        out.append(f"GUARD    RRV-B10 - B10 = {g['net']:+d} rows (W={g['w']} L={g['l']}) -> "
                   + ('no harm' if g['net'] >= GUARD_MIN_NET else 'HARM'))
    return out


def held_verdict(held: Sequence[Dict], recs: Dict) -> str:
    """Secondary, pre-registered: the v21 seed-44 rows no error analysis read.
    Only B5 exists there and v22 already wins 30 of its 34 winnable ties, so
    this is a transfer/no-harm check, not a second chance at GO."""
    h = paired(held, recs, 'B5')
    facts = f"RRV-B5 {h['rrv']} vs B5 {h['base']} on {h['n']} held-out rows: W={h['w']} L={h['l']}"
    return (f"HELD     CONSISTENT: {facts}" if h['net'] >= HELD_MIN_NET
            else f"HELD     INCONSISTENT: {facts} -> do not run the confirmation before discussing")
