"""
[v25.0] Reading Arbitration: a process verifier chooses between two agents'
readings of the problem.

THE PROBLEM IT IS BUILT FOR (measured on stored runs, no GPU)
------------------------------------------------------------
After v22 the system is Solver + Verifier: k sampled CoT solutions,
Qwen2.5-Math-PRM-7B scores every step, and the sample with the best last-step
reward wins (B5 = 78/100 on fresh P2 rows vs SC@5 68). What is left of B5's
errors is no longer selection:
  - B5 is wrong on 22 of the 100 rows, and on 18 of them NONE of the 5 plain
    samples is right. No verifier, vote or restart policy that picks among
    those samples can reach them.
  - The wrong samples agree. These are the "self-consistent errors" of Tan et
    al. (EMNLP 2025): every consistency check is blind to them, they do not
    shrink with scale, and that paper leaves their CAUSE open.
v24 measured the cause on P2, by intervention. Give the Solver a different
reading of the problem (the Reader agent's sentence-anchored situation QAs)
and it ADOPTS that reading, in both directions: +19.6 pp per sample on rows
it had misread, -12.5 pp on rows it had read well. Always-on, that nets out
negative (v24 REFUTED). But the two readings fail on DIFFERENT rows:
  - oracle over C2 + Q2 (2 plain samples + 2 samples under the Reader's
    reading) is 88/100, against 82/100 for C5, with one sample fewer;
  - on 11 of B5's 22 errors a reader-view sample is right (9 only there).
And a vote cannot collect it: SC over all 10 samples is 68, exactly S5. The
diversity-sampling theory says the same ("under majority voting, diversity
may vanish", Wang et al., UAI 2026).

THE METHOD
----------
    problem --> Solver, plain prompt ................ 2 samples  (reading 1)
            --> Reader (Qwen2.5-7B-Instruct, greedy) . situation QAs
            --> Solver, reading interleaved (v24 ASQ)  2 samples  (reading 2)
            --> Verifier (PRM) scores all 4 against the ORIGINAL problem text
            --> the sample with the best last-step reward is the answer
Five LLM calls, the same as B5 (4 solver samples + 1 Reader call ~ 5 solver
samples of wall clock). The verifier never sees the Reader's notes, so a
wrong note has to survive a check against the text it claims to describe;
the plain samples stay in the pool, so a wrong reading cannot overwrite a
right one unless the verifier prefers it. That is v24's fidelity cost
turned into a choice made per row by a third agent.

Nothing here is trained, and nothing looks at gold.

WHAT IS AND IS NOT NEW (checked 2026-09-28, V25_PLAN.md has the sources)
------------------------------------------------------------------------
Not new: PRM best-of-N (Lightman 2023; Qwen PRM, 2025); a second model that
rephrases before a solver (RaR two-step; DivSampling's dual-model
rephrasing); heterogeneous agents (ReConcile; the MAD literature).
New here, as far as the searches found:
  1. The diversity is in the READING, written by a different agent as
     sentence-anchored situation QAs, and aimed at self-consistent errors.
     DivSampling perturbs the prompt at random and selects with the GROUND
     TRUTH (oracle Best-of-N); here a learned verifier selects.
  2. The verifier arbitrates BETWEEN readings, at equal calls, with the plain
     reading kept in the pool, which is what contains the Reader's fidelity
     cost that sank v24.
  3. Tan et al. only DETECT self-consistent errors (a cross-model probe over
     hidden states); this recovers the right answer on some of them.

Everything in this file is deterministic and model-free, so it is tested
offline (test_v25.py). pretest_v25.py runs it.
"""
from __future__ import annotations

import collections
import math
from typing import Dict, List, Optional, Sequence, Tuple

import pretest_v21 as V21
import score_prm_v22 as P

PLAIN, READ = 'C', 'Q'

# Every arm is (plain samples, reader-view samples, selector). Samples are
# taken in draw order, plain first, and best-of-N breaks ties by position, so
# a tie goes to the plain reading.
ARMS: Dict[str, Tuple[int, int, str]] = {
    'S5': (5, 0, 'vote'),    # self-consistency, 5 calls
    'B4': (4, 0, 'best'),    # the verifier on 4 plain samples: RA's solver samples
    'B5': (5, 0, 'best'),    # the v22 system, 5 calls: THE comparator
    'RA': (2, 2, 'best'),    # PRIMARY, 5 calls: 2 plain + Reader + 2 read
    'RA5': (3, 2, 'best'),   # RA with B5's 5 solver samples (6 calls)
    'RA7': (5, 2, 'best'),   # everything the confirmation draws (8 calls)
    'VRA': (2, 2, 'vote'),   # RA's pool with a vote instead of the verifier
    'BQ2': (0, 2, 'best'),   # the reader view alone
}
DEV_ARMS: Dict[str, Tuple[int, int, str]] = {
    'BQ5': (0, 5, 'best'),   # all five samples under the reading (v24's ASQ + verifier)
    'SQ5': (0, 5, 'vote'),   # v24's ASQ SC@5, recomputed
}
PRIMARY, COMPARATOR = 'RA', 'B5'


def calls(arm: str) -> int:
    c, q, _ = {**ARMS, **DEV_ARMS}[arm]
    return c + q + (1 if q else 0)


# the pre-registered bars (frozen 2026-09-28, before any reader-view sample
# has a verifier score; see pretest_v25.py)
PRIMARY_MIN_NET = 4          # RA - B5 in rows, on 100 fresh main rows
PRIMARY_RATIO = 2            # and wins >= 2 x losses
PRIMARY_LOTO = 2             # and still >= +2 without the most-helped template
GUARD_MIN_NET = -1           # guard rows: RA - B5 >= -1
DEV_GO_NET = 3               # dev screen: GO if RA - B5 >= +3 with W >= 2L; STOP if <= 0


# ---------------------------------------------------------------------------
# one row
# ---------------------------------------------------------------------------

def last(sample: Dict) -> Optional[float]:
    """The verifier's last-step reward (the v22-confirmed aggregation), or
    None when the sample has not been scored yet."""
    prm = sample.get('prm')
    if prm is None:
        return None
    return P.agg(prm, 'last')


def pool(rec: Dict, c: int, q: int) -> Tuple[List[Optional[float]], List[Optional[float]], List[str]]:
    """The first c plain samples then the first q reader-view samples:
    answers, last-step rewards, and which view each came from."""
    ss = rec.get(PLAIN, [])[:c] + rec.get(READ, [])[:q]
    views = [PLAIN] * len(rec.get(PLAIN, [])[:c]) + [READ] * len(rec.get(READ, [])[:q])
    return [s.get('answer') for s in ss], [last(s) for s in ss], views


def available(rec: Dict, arm: str) -> bool:
    c, q, how = {**ARMS, **DEV_ARMS}[arm]
    if len(rec.get(PLAIN, [])) < c or len(rec.get(READ, [])) < q:
        return False
    if how == 'best':
        return all(w is not None for w in pool(rec, c, q)[1])
    return True


def choose(rec: Dict, arm: str) -> Optional[float]:
    c, q, how = {**ARMS, **DEV_ARMS}[arm]
    ans, w, _ = pool(rec, c, q)
    if how == 'vote':
        return P.plain_vote(ans)
    return P.best_of_n(ans, [x if x is not None else -1.0 for x in w])


def right(rec: Dict, arm: str) -> bool:
    return V21.correct(choose(rec, arm), rec['gold'])


def oracle(rec: Dict, c: int, q: int) -> bool:
    return any(V21.correct(a, rec['gold']) for a in pool(rec, c, q)[0])


def chosen_view(rec: Dict, arm: str) -> Optional[str]:
    """Which reading the verifier's pick came from (reported, never used)."""
    c, q, how = {**ARMS, **DEV_ARMS}[arm]
    ans, w, views = pool(rec, c, q)
    idx = [i for i, a in enumerate(ans) if a is not None]
    if how != 'best' or not idx:
        return None
    return views[max(idx, key=lambda i: ((w[i] if w[i] is not None else -1.0), -i))]


def note_steps(reading: Dict, n_sentences: int) -> List[str]:
    """The Reader's notes as the 'steps' of a would-be solution, in sentence
    order, for the exploratory reading-vetting score. The method never uses
    it."""
    out = []
    notes = (reading or {}).get('notes') or {}
    for i in range(1, n_sentences + 1):
        for qa in notes.get(str(i), []):
            q, a = (list(qa) + ['', ''])[:2]
            out.append(f"{q} {a}".strip())
    return out


# ---------------------------------------------------------------------------
# paired readings over rows
# ---------------------------------------------------------------------------

def template(rec: Dict):
    t = rec.get('template')
    return t if t is not None else rec['key']


def paired(rows: Sequence[Dict], x: str, y: str) -> Dict:
    wins = [r for r in rows if right(r, x) and not right(r, y)]
    losses = [r for r in rows if right(r, y) and not right(r, x)]
    by = collections.Counter()
    for r in wins:
        by[template(r)] += 1
    for r in losses:
        by[template(r)] -= 1
    net = len(wins) - len(losses)
    best = max(by.values(), default=0)
    return {'w': len(wins), 'l': len(losses), 'net': net,
            'loto': net - max(best, 0), 'p': sign_p(len(wins), len(losses)),
            'wins': [r['pid'] for r in wins], 'losses': [r['pid'] for r in losses]}


def sign_p(w: int, l: int) -> float:
    """Exact two-sided sign test on the discordant rows."""
    n = w + l
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def gain_source(rows: Sequence[Dict], x: str = PRIMARY, y: str = COMPARATOR) -> Dict[str, int]:
    """Where x's wins over y come from: rows whose right answer was already
    among the 5 plain samples (the verifier chose better) versus rows where
    only the reader view had it (the reading added coverage)."""
    out = collections.Counter()
    for r in rows:
        if right(r, x) and not right(r, y):
            out['selection' if oracle(r, 5, 0) else 'reading'] += 1
    return dict(out)


def auroc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    """P(score of a positive > score of a negative), ties count half."""
    if not pos or not neg:
        return None
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return s / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
# the pre-registered verdicts
# ---------------------------------------------------------------------------

def primary_verdict(main: Sequence[Dict]) -> str:
    c = paired(main, PRIMARY, COMPARATOR)
    facts = (f"RA {sum(right(r, PRIMARY) for r in main)} vs B5 "
             f"{sum(right(r, COMPARATOR) for r in main)} of {len(main)}: W={c['w']} L={c['l']} "
             f"net={c['net']:+d}, {c['loto']:+d} without the most-helped template, "
             f"sign p={c['p']:.3f}")
    if c['net'] >= PRIMARY_MIN_NET and c['w'] >= PRIMARY_RATIO * c['l'] and c['loto'] >= PRIMARY_LOTO:
        return f"PRIMARY  SUPPORTED: reading arbitration beats the verifier on plain samples at equal calls ({facts})"
    if c['net'] <= 0:
        return f"PRIMARY  REFUTED: RA <= B5 ({facts})"
    return f"PRIMARY  INCONCLUSIVE: {facts}"


def guard_verdict(guard: Sequence[Dict]) -> str:
    c = paired(guard, PRIMARY, COMPARATOR)
    return (f"GUARD    RA - B5 = {c['net']:+d} rows (W={c['w']} L={c['l']}) -> "
            + ("no harm" if c['net'] >= GUARD_MIN_NET else "HARM"))


def dev_verdict(main: Sequence[Dict]) -> str:
    c = paired(main, PRIMARY, COMPARATOR)
    facts = f"W={c['w']} L={c['l']} net={c['net']:+d}"
    if c['net'] >= DEV_GO_NET and c['w'] >= PRIMARY_RATIO * c['l']:
        return f"SCREEN   GO: run the fresh-row confirmation ({facts})"
    if c['net'] <= 0:
        return f"SCREEN   STOP: the verifier does not arbitrate readings on the dev rows ({facts})"
    return f"SCREEN   WEAK: positive but under the bar; the confirmation is the user's call ({facts})"
