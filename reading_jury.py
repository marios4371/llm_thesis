"""
[v32.0] Reading Jury (RJ): readers from OTHER model families, one solver,
and a tie-break that counts families instead of samples.

WHY THIS VERSION EXISTS (measured on stored runs, no GPU)
---------------------------------------------------------
v22's verifier (Qwen2.5-Math-PRM-7B, best last-step reward) is the thesis's
one confirmed gain: B5 78 vs SC@5 68 on fresh P2 rows. What it cannot decide
are the reading disputes inside its saturated top tie (B10: 64/100 dev rows,
argmax right on 48/57 = 84%). Six screens tried to settle them from inside
the Qwen family and all failed (v24, v27, v28, v29, v30), and v31 tried a
second family as a JUROR that solves on its own. Three facts from those runs
point to the design below:
  1. ADOPTION (v24): the Solver takes over any reading written next to the
     sentences: +19.6 pp per sample on the rows it misread, -12.5 pp on the
     rows it read well. A reading moves answers; a pointer does not (V2P).
  2. FAMILY-SPECIFIC MISREADINGS. On the 13 dev rows where >= 3 of the 5 plain
     samples agree on a WRONG answer, the same-family Reader view (Qwen2.5-7B-
     Instruct notes -> Qwen-Math) repeats that answer on 5/13. The DeepSeek
     juror repeats it on 1/13. Both are right on 5/13, although the juror gets
     only 44.8% of its samples right overall. A second view from the same
     family is not a second vote: counting C and Q as independent ("prefer the
     tied answer present in both views") gives B10 74 vs 81 (W2 L9).
  3. THE JUROR LOST ON MATH, NOT ON READING (v31). Its losses come from single
     stray votes and arithmetic scatter (p2_1824: 521.25, 24, 1.5, 104, 130);
     where it was confident (3-4 of 5) it won p2_1972, 1268 and 890.

THE METHOD
----------
Split each agent's job by what it is good at, across families:
  Readers  two general instruct models of OTHER families (Phi-3.5-mini,
           Microsoft; Granite-3.1-8B, IBM) write v24's sentence-anchored
           situation notes, greedy, once per row. They never solve.
  Solver   Qwen2.5-Math-7B-Instruct, unchanged: 5 plain samples, plus K_READ
           samples under each foreign reading (v24's ASQ prompt).
  Verifier v22's PRM scores every sample against the ORIGINAL text.
  Rule     v22 everywhere. Only inside a contested top tie (two or more
           answers within EPS of the best last-step reward) does
           `jury_choice` decide, from which FAMILIES produced each answer.
The foreign model contributes its reading (where its errors are decorrelated
from Qwen's) and none of its arithmetic (where it is weaker than Qwen), and
the selection treats every Qwen sample, plain or Qwen-read, as one witness.

THE LIMITATIONS IT TAKES UP
---------------------------
  * Tan et al., EMNLP 2025 (arXiv 2505.17656): self-consistent errors are
    model-specific (cross-family overlap 5-20%); they only DETECT them and
    leave mitigation open.
  * Li et al., "Rethinking Mixture-of-Agents" (arXiv 2502.00674, TMLR 2026):
    mixing families trades quality for diversity, because the weaker models'
    answers are worse. Here the weaker family never answers.
  * Yan et al., RoR-Bench (arXiv 2504.00509): no fix for recitation "without
    over-reliance on user's clarifications". The foreign Reader supplies the
    clarification.
  * v31 (this thesis): a cross-family juror helps only if it is as accurate
    as the generator. Here its accuracy is the Solver's.
Closest prior work: Rephrase-and-Respond's two-step variant (Deng et al.
2023) uses one stronger rephraser to give a weaker responder "better
questions"; no ensemble of readers, no error decorrelation, no verifier.
Not new here: heterogeneous agents (X-MAS), PRM best-of-n, situation notes
(MathWorld, v24).

Everything in this file is pure and model-free (test_v32.py tests it
offline). pretest_v32.py draws the readings and samples.
"""
from __future__ import annotations

import collections
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import null_hypothesis as NH
import pretest_v21 as V21
import reading_arbitration as RA
import reread_verification as RV
import score_prm_v22 as P

# ---------------------------------------------------------------------------
# agents and pools (frozen with the pre-registration, V32_PLAN.md section 5)
# ---------------------------------------------------------------------------

OWN_FAMILY = 'qwen'
# (view, model, family): the order is the order in the pool
READERS: Tuple[Tuple[str, str, str], ...] = (
    ('F1', 'microsoft/Phi-3.5-mini-instruct', 'phi'),
    ('F2', 'ibm-granite/granite-3.1-8b-instruct', 'granite'),
)
READER_VIEWS = tuple(v for v, _, _ in READERS)
READER_MODEL = {v: m for v, m, _ in READERS}
READER_FAMILY = {v: f for v, _, f in READERS}
K_READ = 2                       # Solver samples per foreign reading
TEMPERATURE = 0.8                # v22's sampler: T 0.8, top-p 0.95, top-k 50, 1024 tokens
MAX_TOKENS = 1024
EPS = NH.EPS                     # 1e-3: the PRM's top tie, as in v25/v27/v29/v30/v31

# pool -> ((view, number of samples), ...), plain first
POOLS: Dict[str, Tuple[Tuple[str, int], ...]] = {
    'B5': (('C', 5),),
    'B9': (('C', 5), ('Q', 4)),                    # RJ's 9 solver samples, same-family reader
    'B10': (('C', 5), ('Q', 5)),                   # THE comparator: v22 + the Qwen Reader (stored)
    'RJ': (('C', 5), ('F1', K_READ), ('F2', K_READ)),   # PRIMARY: 9 solver samples, 2 foreign readings
    'RJ1': (('C', 5), ('F1', K_READ)),             # one foreign family (exploratory)
    'RJ2': (('C', 5), ('F2', K_READ)),
}
PRIMARY_POOL, COMPARATOR_POOL = 'RJ', 'B10'

# pre-registered bars (V32_PLAN.md section 5, frozen before any foreign sample exists)
GO_NET = 3            # RJ (jury rule) vs B10 (v22's rule), 100 dev main rows: net >= +3 with W >= 2L
GO_RATIO = 2
GUARD_MIN_NET = -1    # the 20 GSM-Plus guard rows: RJ - B10 >= -1
HELD_MIN_NET = 0      # 100 held-out v21 rows: RJ (jury rule) - RJ (v22's rule) >= 0 -> CONSISTENT


def family_of(view: str, sample: Dict) -> str:
    """The family whose READING produced the sample. Plain and Qwen-Reader
    samples are Qwen's. A foreign-view sample carries the family it was drawn
    under: the reader's when its reading was usable, Qwen's when the prompt
    fell back to the plain one (then it is just another Qwen sample)."""
    if view in ('C', 'Q'):
        return OWN_FAMILY
    return sample.get('family') or READER_FAMILY.get(view, OWN_FAMILY)


def agree(a: Optional[float], b: Optional[float]) -> bool:
    return a is not None and b is not None and (V21.correct(a, b) or V21.correct(b, a))


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------

def available(row: Dict, pool: str) -> bool:
    """Every sample the pool needs exists and has its step rewards."""
    for view, n in POOLS[pool]:
        ss = row.get(view) or []
        if len(ss) < n or any(s.get('prm') is None for s in ss[:n]):
            return False
    return True


def candidates(row: Dict, pool: str) -> List[Dict]:
    """The pool in draw order (plain first), each with v22's last-step reward
    as `real` (the field names NH.pick and RV.contested_tie read)."""
    out = []
    for view, n in POOLS[pool]:
        for j, s in enumerate((row.get(view) or [])[:n]):
            out.append({'ref': f'{view}{j}', 'view': view, 'family': family_of(view, s),
                        'answer': s.get('answer'), 'real': NH.last(s.get('prm'))})
    return out


def tie_groups(cands: List[Dict], tie: Sequence[int]) -> List[Dict]:
    """One entry per distinct answer inside the contested tie, in v22's order:
    groups[0] holds the answer v22 picks (best last-step reward, then the
    lower pool position). Support is counted over the WHOLE pool and over the
    tie separately, so the rule can use either."""
    groups = []
    for cl in P.L.clusters([cands[i]['answer'] for i in tie]):
        members = [tie[k] for k in cl]
        rep = max(members, key=lambda i: (cands[i]['real'], -i))
        ans = cands[rep]['answer']
        in_pool = [j for j, c in enumerate(cands) if agree(c['answer'], ans)]
        groups.append({'answer': ans, 'rep': rep, 'real': cands[rep]['real'],
                       'families': frozenset(cands[j]['family'] for j in in_pool),
                       'tie_families': frozenset(cands[i]['family'] for i in members),
                       'n_pool': len(in_pool), 'n_tie': len(members)})
    groups.sort(key=lambda g: (-g['real'], g['rep']))
    return groups


# ---------------------------------------------------------------------------
# THE RULE
# ---------------------------------------------------------------------------

def jury_choice(groups: List[Dict]) -> Optional[int]:
    """Which answer wins inside v22's contested tie. This is the method's one
    decision, and it is frozen before any foreign-reader sample exists.

    `groups` has one dict per distinct answer in the PRM's top tie, in v22's
    order: groups[0] is the answer v22 would pick. Each dict has
        'families'      frozenset of the families with >= 1 sample of this
                        answer ANYWHERE in the pool, e.g. {'qwen', 'phi'}
        'tie_families'  the same, counting only samples inside the tie
        'n_pool'        how many pool samples give this answer
        'n_tie'         how many of them are inside the tie
        'real'          the PRM's last-step reward of its best sample
    'qwen' is the Solver's own family: the 5 plain samples, and any foreign-
    view sample whose reading failed (it fell back to the plain prompt). The
    foreign readers are 'phi' and 'granite'.

    Return the index of the winning group, or None to keep v22's pick.

    What test_v32.py requires of any rule written here:
      1. No group has a family other than 'qwen' -> None or 0. With only the
         Solver's own family there is no new witness, so the rule must give
         v22 exactly (on the stored B10 pool it must reproduce 81).
      2. One answer is supported by BOTH foreign families while v22's answer
         is supported by 'qwen' alone -> choose the former. That is the
         premise of the method.
      3. Return None or an int in range(len(groups)).
    """
    # Frozen 2026-10-10 (V32_PLAN.md section 5), chosen by the author before any
    # foreign-reader sample existed: count families over the whole pool; move
    # off v22's pick only for a UNIQUE leader backed by at least two families.
    # One family against another (1 vs 1), or a tie in family count, keeps v22:
    # v31 lost most of its rows to such single-witness overrides.
    score = [len(g['families']) for g in groups]
    best = max(score)
    if best < 2 or score.count(best) > 1:
        return None          # no new witness, or a tie: keep v22
    return score.index(best)


def rj_pick(cands: List[Dict], rule: Optional[Callable[[List[Dict]], Optional[int]]] = None,
            use_rule: bool = True) -> Tuple[Optional[int], str]:
    """Index of the chosen candidate and why. use_rule=False is v22's rule."""
    i0, why = NH.pick(cands, False)
    if i0 is None or not use_rule:
        return i0, why
    tie = RV.contested_tie(cands)
    if not tie:
        return i0, 'no contested tie'
    groups = tie_groups(cands, tie)
    k = (rule or jury_choice)(groups)
    if k is None or k == 0:
        return i0, 'kept v22'
    if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k < len(groups):
        raise ValueError(f"jury_choice returned {k!r} for {len(groups)} groups")
    return groups[k]['rep'], 'jury'


def choose(row: Dict, pool: str, use_rule: bool = True, rule=None) -> Optional[float]:
    cands = candidates(row, pool)
    i, _ = rj_pick(cands, rule, use_rule)
    return None if i is None else cands[i]['answer']


def right(row: Dict, pool: str, use_rule: bool = True, rule=None) -> bool:
    return V21.correct(choose(row, pool, use_rule, rule), row['gold'])


def oracle(row: Dict, pool: str) -> bool:
    return any(V21.correct(c['answer'], row['gold']) for c in candidates(row, pool))


# ---------------------------------------------------------------------------
# paired comparisons and the pre-registered verdicts
# ---------------------------------------------------------------------------

def paired(rows: Sequence[Dict], x: Tuple[str, bool], y: Tuple[str, bool], rule=None) -> Dict:
    """x and y are (pool, use_rule). Rows where either pool is unavailable are
    skipped and counted in 'skipped'."""
    rows = list(rows)
    ok = [r for r in rows if available(r, x[0]) and available(r, y[0])]
    a = [right(r, x[0], x[1], rule) for r in ok]
    b = [right(r, y[0], y[1], rule) for r in ok]
    wins = [r for r, p, q in zip(ok, a, b) if p and not q]
    losses = [r for r, p, q in zip(ok, a, b) if q and not p]
    by = collections.Counter()
    for r in wins:
        by[r.get('template')] += 1
    for r in losses:
        by[r.get('template')] -= 1
    net = len(wins) - len(losses)
    return {'x': sum(a), 'y': sum(b), 'n': len(ok), 'skipped': len(rows) - len(ok),
            'w': len(wins), 'l': len(losses), 'net': net,
            'loto': net - max(max(by.values(), default=0), 0),
            'p': RA.sign_p(len(wins), len(losses)),
            'wins': [r['pid'] for r in wins], 'losses': [r['pid'] for r in losses]}


def screen_verdict(main: Sequence[Dict], guard: Sequence[Dict], rule=None) -> List[str]:
    c = paired(main, (PRIMARY_POOL, True), (COMPARATOR_POOL, False), rule)
    facts = (f"RJ {c['x']} vs B10 {c['y']} on {c['n']} rows: W={c['w']} L={c['l']} "
             f"net {c['net']:+d}, {c['loto']:+d} without the most-helped template "
             f"(sign p={c['p']:.3f})")
    if c['net'] >= GO_NET and c['w'] >= GO_RATIO * c['l']:
        out = [f"SCREEN   GO: {facts} -> build the fresh-row confirmation"]
    elif c['net'] <= 0:
        out = [f"SCREEN   STOP: {facts}"]
    else:
        out = [f"SCREEN   WEAK: {facts} (under the bar; the confirmation is the user's call)"]
    if guard:
        g = paired(guard, (PRIMARY_POOL, True), (COMPARATOR_POOL, False), rule)
        out.append(f"GUARD    RJ - B10 = {g['net']:+d} rows on {g['n']} (W={g['w']} L={g['l']}) -> "
                   + ('no harm' if g['net'] >= GUARD_MIN_NET else 'HARM'))
    return out


def held_verdict(held: Sequence[Dict], rule=None) -> str:
    h = paired(held, (PRIMARY_POOL, True), (PRIMARY_POOL, False), rule)
    facts = (f"RJ with the jury rule {h['x']} vs the same pool with v22's rule {h['y']} on "
             f"{h['n']} held-out rows: W={h['w']} L={h['l']}")
    return (f"HELD     CONSISTENT: {facts}" if h['net'] >= HELD_MIN_NET
            else f"HELD     INCONSISTENT: {facts} -> discuss before any confirmation")


# ---------------------------------------------------------------------------
# mechanism telemetry (reported, never used by the rule)
# ---------------------------------------------------------------------------

def view_accuracy(rows: Sequence[Dict], view: str) -> Tuple[int, int]:
    right_n = n = 0
    for r in rows:
        for s in r.get(view) or []:
            n += 1
            right_n += V21.correct(s.get('answer'), r['gold'])
    return right_n, n


def self_consistent_wrong(row: Dict) -> Optional[float]:
    """The answer >= 3 of the 5 plain samples agree on, if it is WRONG."""
    a = [s.get('answer') for s in (row.get('C') or [])[:5]]
    groups = [g for g in P.L.clusters(a) if len(g) >= 3]
    if not groups or V21.correct(a[groups[0][0]], row['gold']):
        return None
    return a[groups[0][0]]


def shared_error_floor(rows: Sequence[Dict], view: str) -> Dict:
    """On rows with a self-consistent wrong plain answer: per sample of
    `view`, how often it repeats that answer and how often it is right. Per
    sample, so views with 5 and with 2 samples compare."""
    rep = rt = n = n_rows = 0
    for r in rows:
        wrong = self_consistent_wrong(r)
        ss = [s for s in (r.get(view) or []) if view in ('C', 'Q') or family_of(view, s) != OWN_FAMILY]
        if wrong is None or not ss:
            continue
        n_rows += 1
        for s in ss:
            n += 1
            rep += agree(s.get('answer'), wrong)
            rt += V21.correct(s.get('answer'), r['gold'])
    return {'rows': n_rows, 'samples': n, 'repeat': rep, 'right': rt}
