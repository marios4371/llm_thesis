"""
[v29.0] Dispute-to-Question (DQ): when the verifier cannot decide between
two readings of the problem, ask the problem itself.

Why (V29_PLAN.md has the numbers)
---------------------------------
The v22 system (Solver samples + the PRM verifier, best of n by last-step
reward) is the thesis's one confirmed gain: B5 78 vs SC@5 68 on fresh rows.
Its remaining errors on the 100 seed-45 rows split cleanly:
  B5 (5 plain samples)        78 right; 13 rows where the right answer is
                              never sampled but another reading finds it,
                              5 never reachable, 4 selection losses
  B10 (+ 5 Reader-view)       81 right; coverage fixed (13 -> 4), but 10
                              selection losses: the PRM scores two readings
                              both ~1.0 and keeps the wrong one
Each lost dispute hinges on one local fact about the problem ("These 3
groups" read as 2; the return trip taken as 42 minutes instead of 55). The
PRM checks that a solution agrees with itself, not with the problem (Xu et
al. 2025), so it cannot settle them. A narrow question about the text can.

The rule (only on rows where the PRM's top is a dispute)
--------------------------------------------------------
  1 the v22 pick is the incumbent; every other answer in the PRM's top tie
    (last-step reward within EPS of the max) is a challenger, represented by
    its best candidate
  2 the Questioner (the Reader model, greedy) reads the incumbent's and a
    challenger's solution, finds the first point where they read or compute
    differently, and writes ONE short question about the problem with a
    numeric answer, plus the value each solution uses for it
  3 the question is answered FACTORED: fresh calls that see only the problem
    and the question, never a solution (the Solver 3 times, the Reader twice,
    sampled). v24 measured that the Solver adopts any reading it is shown;
    here it is shown none. (Chain-of-Verification's factored variant.)
  4 the challenger replaces the incumbent only if at least OVERRIDE_VOTES of
    the 5 answers give the challenger's value. Challengers go one at a time,
    each against the current incumbent.

The published idea and its open window: LMAD (arXiv 2608.01463) localises
the earliest conflict between agents, but resolves it by DEBATE (each agent
sees the others' claims), on multi-hop QA only, with no verifier in the loop.
DQ localises the conflict between verifier-tied solutions of a math word
problem and resolves it with a factored question about the source text.

Pure: no torch, no model at import.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import null_hypothesis as NH
import premise_ledger as L
import pretest_v21 as V21

# ---------------------------------------------------------------------------
# pre-registered constants (frozen 2026-10-03, before any question exists)
# ---------------------------------------------------------------------------

EPS = NH.EPS                    # 1e-3: the v25/v27 definition of the PRM's top tie
POOLS = dict(NH.POOLS)          # B5 (5,0), RA (2,2), RA5 (3,2), B10 (5,5)
PRIMARY_POOL = 'B10'
MAX_CHALLENGERS = 2             # per row and pool, best first
K_SOLVER = 3                    # factored answers from the Solver (Qwen2.5-Math-7B-Instruct)
K_READER = 2                    # factored answers from the Reader (Qwen2.5-7B-Instruct)
OVERRIDE_VOTES = 4              # of the 5 answers, to replace the incumbent
ANSWER_TEMPERATURE = 0.8
Q_MAX_TOKENS = 320
A_MAX_TOKENS = 400
SOL_CHARS = 2500                # each solution as the Questioner sees it

GO_NET = 3                      # DQ-B10 vs B10 on the 100 main rows: net >= +3 with W >= 2L
GO_RATIO = 2
GUARD_MIN_NET = -1              # the 20 GSM-Plus guard rows: DQ-B10 - B10 >= -1


# ---------------------------------------------------------------------------
# candidates and disputes
# ---------------------------------------------------------------------------

def candidates(rec: Dict, pool: str) -> List[Dict]:
    """NH.candidates plus where each one came from ('C0', 'Q3', ...)."""
    out = NH.candidates(rec, pool)
    c, q = POOLS[pool]
    refs = [f'C{j}' for j in range(min(c, len(rec.get('C', []))))] + \
           [f'Q{j}' for j in range(min(q, len(rec.get('Q', []))))]
    for cand, ref in zip(out, refs):
        cand['ref'] = ref
    return out


def raw_of(rec: Dict, ref: str) -> str:
    return str(rec[ref[0]][int(ref[1:])].get('raw') or '')


def contenders(cands: List[Dict]) -> Tuple[Optional[int], List[int]]:
    """(incumbent, challengers): the v22 pick, then the best candidate of each
    other answer inside the PRM's top tie, best first, at most MAX_CHALLENGERS."""
    inc, _ = NH.pick(cands, False)
    if inc is None:
        return None, []
    idx = [i for i, c in enumerate(cands) if c['answer'] is not None and c['real'] is not None]
    top = max(cands[i]['real'] for i in idx)
    tie = [i for i in idx if cands[i]['real'] >= top - EPS]
    groups = L.clusters([cands[i]['answer'] for i in tie])
    reps = []
    for g in groups:
        members = [tie[k] for k in g]
        if any(V21.correct(cands[m]['answer'], cands[inc]['answer']) for m in members):
            continue
        reps.append(max(members, key=lambda m: (cands[m]['real'], -m)))
    reps.sort(key=lambda m: (-cands[m]['real'], m))
    return inc, reps[:MAX_CHALLENGERS]


def pair_key(ref_a: str, ref_b: str) -> str:
    """Order-free: one question per pair of solutions, whatever the pool."""
    return '|'.join(sorted((ref_a, ref_b)))


# ---------------------------------------------------------------------------
# the Questioner
# ---------------------------------------------------------------------------

Q_SYSTEM = (
    "You compare two solutions of the same math word problem. They reach different final "
    "answers, so at some point they read the problem differently or compute a quantity "
    "differently. Find the FIRST such point. Then write ONE short question about the problem "
    "that has a single numeric answer and settles which solution is right at that point. "
    "Rules: the question must be answerable from the problem text alone; do not mention the "
    "solutions; do not put either solution's value in the question. Then give the value each "
    "solution uses for that quantity, as a plain number.")

# Written for this purpose; neither is a GSM8K or GSM-Symbolic problem.
Q_DEMOS = [
    ("A baker bakes 5 trays of 12 cookies. She keeps 2 trays for the shop and gives the rest to "
     "4 schools, sharing them equally. How many cookies does each school get?",
     "Total cookies: 5 x 12 = 60.\nEach school gets 60 / 4 = 15.\nAnswer: 15",
     "Trays given away: 5 - 2 = 3.\nCookies given away: 3 x 12 = 36.\nEach school gets 36 / 4 = 9.\n"
     "Answer: 9",
     "QUESTION: How many cookies does the baker give to the schools in total?\nSOLUTION 1: 60\n"
     "SOLUTION 2: 36"),
    ("A cyclist has to cover 36 km. She rides the first 30 km at 15 km per hour, then a flat tyre "
     "makes her walk the remaining distance at 4 km per hour. How many hours does the whole trip "
     "take?",
     "Riding: 30 / 15 = 2 hours.\nWalking: 36 / 4 = 9 hours.\nTotal: 2 + 9 = 11 hours.\nAnswer: 11",
     "Riding: 30 / 15 = 2 hours.\nRemaining distance: 36 - 30 = 6 km.\nWalking: 6 / 4 = 1.5 hours.\n"
     "Total: 2 + 1.5 = 3.5 hours.\nAnswer: 3.5",
     "QUESTION: How many kilometres does the cyclist walk?\nSOLUTION 1: 36\nSOLUTION 2: 6"),
]


def _q_user(problem: str, s1: str, s2: str) -> str:
    return (f"Problem: {problem}\n\nSolution 1:\n{s1[:SOL_CHARS]}\n\nSolution 2:\n{s2[:SOL_CHARS]}\n\n"
            "Answer in exactly three lines:\nQUESTION: ...\nSOLUTION 1: <number>\nSOLUTION 2: <number>")


def question_messages(problem: str, sol1: str, sol2: str) -> List[Dict[str, str]]:
    msgs = [{'role': 'system', 'content': Q_SYSTEM}]
    for p, a, b, out in Q_DEMOS:
        msgs.append({'role': 'user', 'content': _q_user(p, a, b)})
        msgs.append({'role': 'assistant', 'content': out})
    msgs.append({'role': 'user', 'content': _q_user(problem, sol1, sol2)})
    return msgs


_NUM = r'[-+]?\$?\s*\d[\d,]*(?:\.\d+)?'


def _num(s: str) -> Optional[float]:
    m = re.search(_NUM, s or '')
    if not m:
        return None
    try:
        return float(re.sub(r'[\s$,]', '', m.group(0)))
    except ValueError:
        return None


def problem_values(problem: str) -> List[float]:
    return [p.value for p in L.extract_premises(problem)]


def parse_question(raw: str, problem: str) -> Dict:
    """{'ok', 'reason', 'question', 'v1', 'v2'} from the Questioner's output."""
    text = str(raw or '')
    q = re.search(r'QUESTION\s*:\s*(.+)', text, re.I)
    a = re.search(r'SOLUTION\s*1\s*:\s*(.+)', text, re.I)
    b = re.search(r'SOLUTION\s*2\s*:\s*(.+)', text, re.I)
    out = {'raw': text[:2000], 'question': q.group(1).strip() if q else '',
           'v1': _num(a.group(1)) if a else None, 'v2': _num(b.group(1)) if b else None}
    reason = ''
    if not out['question']:
        reason = 'no question'
    elif out['v1'] is None or out['v2'] is None:
        reason = 'no value for a solution'
    elif V21.correct(out['v1'], out['v2']):
        reason = 'both solutions use the same value'
    else:
        # leakage: a number in the question that the problem does not state and
        # that is one of the two disputed values would hand over the answer
        given = problem_values(problem)
        for m in re.finditer(_NUM, out['question']):
            v = _num(m.group(0))
            if v is None or any(V21.correct(v, g) for g in given):
                continue
            if V21.correct(v, out['v1']) or V21.correct(v, out['v2']):
                reason = "the question contains a disputed value"
                break
    out.update(ok=not reason, reason=reason)
    return out


# ---------------------------------------------------------------------------
# the factored answer
# ---------------------------------------------------------------------------

def answer_prompt(problem: str, question: str) -> str:
    """What the answerers see: the problem and the question. Never a solution."""
    return (f"Read the problem and answer the question about it.\n\nProblem: {problem}\n\n"
            f"Question: {question}\n\nThink briefly, then give the final number on a line "
            f"starting with 'Answer:'.")


def tally(votes: Sequence[Optional[float]], v_inc: float, v_ch: float) -> Tuple[int, int]:
    n_inc = sum(1 for v in votes if V21.correct(v, v_inc))
    n_ch = sum(1 for v in votes if V21.correct(v, v_ch))
    return n_inc, n_ch


def overrides(votes: Sequence[Optional[float]], v_inc: float, v_ch: float,
              rule: str = 'strict') -> bool:
    """strict (pre-registered): >= OVERRIDE_VOTES answers give the challenger's
    value. majority (exploratory): more answers for the challenger than for
    the incumbent."""
    n_inc, n_ch = tally(votes, v_inc, v_ch)
    if rule == 'strict':
        return n_ch >= OVERRIDE_VOTES
    if rule == 'majority':
        return n_ch > n_inc
    raise ValueError(rule)


def values_for(pair: Dict, ref_inc: str, ref_ch: str) -> Optional[Tuple[float, float]]:
    """The values the incumbent's and the challenger's solutions use, from a
    pair record whose Solution 1 / 2 are the pair's refs in sorted order."""
    if not pair or not pair.get('ok'):
        return None
    first = sorted((ref_inc, ref_ch))[0]
    v_first, v_second = pair['v1'], pair['v2']
    return (v_first, v_second) if ref_inc == first else (v_second, v_first)


# ---------------------------------------------------------------------------
# the rule over one pool
# ---------------------------------------------------------------------------

def resolve(rec: Dict, pool: str, get_pair: Callable[[str, str], Optional[Dict]],
            rule: str = 'strict') -> Tuple[Optional[float], Dict]:
    """The DQ pick for one row and pool. get_pair(ref_a, ref_b) returns the
    pair record (question, values, votes) or None when it is not available;
    a missing pair keeps the incumbent and marks the result incomplete."""
    cands = candidates(rec, pool)
    inc, chs = contenders(cands)
    info = {'dispute': bool(chs), 'pairs': [], 'complete': True, 'changed': False}
    if inc is None:
        return None, info
    cur = inc
    for ch in chs:
        ref_inc, ref_ch = cands[cur]['ref'], cands[ch]['ref']
        if V21.correct(cands[ch]['answer'], cands[cur]['answer']):
            continue
        pair = get_pair(ref_inc, ref_ch)
        info['pairs'].append(pair_key(ref_inc, ref_ch))
        if pair is None or 'votes' not in pair:
            info['complete'] = False
            continue
        vals = values_for(pair, ref_inc, ref_ch)
        if vals is None:
            continue
        if overrides(pair['votes'], vals[0], vals[1], rule):
            cur = ch
            info['changed'] = True
    return cands[cur]['answer'], info


def v22_pick(rec: Dict, pool: str) -> Optional[float]:
    cands = candidates(rec, pool)
    i, _ = NH.pick(cands, False)
    return None if i is None else cands[i]['answer']
