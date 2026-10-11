"""
[v33.0] Verifier-Gated Reading Arbitration (VGRA): the verifier routes the
compute. The Solver draws three samples; only when the verifier doubts the
best of them is a Reader agent called, and the verifier then arbitrates
between the Solver's reading and the Reader's.

WHY THIS VERSION EXISTS
-----------------------
Seven versions (v26-v32) chased the same residual: the ~9 reading disputes
inside the PRM's saturated top tie, a slice with a ceiling of +9 rows and
+-4 rows of noise at n = 100. None of them could come out positive. In the
meantime three components that DID work were never put together:
  * v22 Verifier (Qwen2.5-Math-PRM-7B, best last-step reward): B5 78 vs
    SC@5 68 on fresh rows, W11 L1. Confirmed.
  * v24 Reader (Qwen2.5-7B-Instruct, sentence-anchored situation notes): the
    Solver adopts the reading, +19.6 pp per sample on the rows it misread,
    but -12.5 pp on the rows it already read well. Useful only where needed.
  * v22 ADAPT gate (min step reward of the best of 3 plain samples < 0.85):
    finds the rows where the verifier doubts. Defined on the v21 rows,
    before the rows below existed.
Composed on the stored dev rows (v22 seed 45, 100 P2 rows; offline):
                                  right   calls/row
    SC@5 (plurality)                68      5
    B5 (v22)                        78      5
    VGRA                            80      3.48
  On the 16 gated rows: B3 4, B5 7, VGRA 11. VGRA vs SC@5 W13 L1 (p = 0.002);
  vs B5 W6 L4. Guard (20 GSM-Plus rows): 17 vs B5 18.
These rows are NOT fresh for the composition, so the claim is made only on
the v25 confirmation rows (seed 48), which no run has ever touched.

THE RULE (frozen, V33_PLAN.md section 5)
----------------------------------------
  1. Draw C1..C3 (the plain CoT prompt, v22's sampler); score with the PRM.
  2. b = the best of C1..C3 by last-step reward (ties: the lower position).
  3. If min(step rewards of b) >= GATE_MIN (0.85): answer = b's answer.
     3 calls.
  4. Otherwise call the Reader once (v24's notes), draw Q1..Q2 under its
     reading, score them with the PRM against the ORIGINAL text, and answer
     with v22's rule over {C1, C2, C3, Q1, Q2} (best last-step reward, ties to
     the plain sample). 6 calls.

WHAT IS AND IS NOT NEW (checked 2026-10-11)
-------------------------------------------
Not new: PRM best-of-n; uncertainty-gated test-time compute (Entropy-Gated
Branching, uncertainty-aware budget allocation, calibrated-PRM instance-
adaptive scaling, BETAPRM: all spend MORE SAMPLES or BRANCHES where the
model or verifier is unsure); rephrasing the problem (RaR, always on, no
verifier).
New: on verifier doubt, the compute goes to a DIFFERENT AGENT that re-reads
the problem, and the verifier arbitrates between the two readings. That
turns the v24 Reader, harmful as an always-on agent, into a routed one.

Everything here is pure and model-free (test_v33.py). The samples come from
pretest_v25.py's confirmation mode (unchanged), driven by pretest_v33.py.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import null_hypothesis as NH
import pretest_v21 as V21
import reading_arbitration as RA
import score_prm_v22 as P

GATE_K = 3            # plain samples drawn first
GATE_MIN = 0.85       # v22's ADAPT threshold on the min step reward (defined on v21 rows)
READ_K = 2            # samples under the Reader's reading, only when gated
CALLS_UNGATED = GATE_K                    # 3
CALLS_GATED = GATE_K + 1 + READ_K         # 6: + one Reader call + two read samples

# pre-registered bars (V33_PLAN.md section 5; frozen 2026-10-11 before any
# confirmation sample exists)
SUP_MIN_NET = 5       # VGRA vs SC@5 (main): net >= +5, W >= 2L and sign p < 0.05 -> SUPPORTED
SUP_RATIO = 2
SUP_ALPHA = 0.05
NI_MIN_NET = 0        # VGRA vs B5 (main): net >= 0 at <= NI_MAX_CALLS calls/row -> NON-INFERIOR
NI_MAX_CALLS = 4.0
BEAT_MIN_NET = 3      # ... net >= +3 with W >= 2L -> BEATS B5
GUARD_MIN_NET = -1    # guard rows: VGRA - B5 >= -1


# ---------------------------------------------------------------------------
# samples and candidates
# ---------------------------------------------------------------------------

def _last(s: Dict) -> Optional[float]:
    return NH.last(s.get('prm'))


def cands(rec: Dict, c: int, q: int) -> List[Dict]:
    """C[:c] then Q[:q], plain first, in the shape NH.pick reads."""
    out = [{'answer': s.get('answer'), 'real': _last(s), 'view': 'C'} for s in (rec.get('C') or [])[:c]]
    out += [{'answer': s.get('answer'), 'real': _last(s), 'view': 'Q'} for s in (rec.get('Q') or [])[:q]]
    return out


def v22_pick(cs: List[Dict]) -> Optional[float]:
    i, _ = NH.pick(cs, False)
    return None if i is None else cs[i]['answer']


def complete(rec: Dict) -> bool:
    """Every sample any arm reads exists and is scored: C1..C5 and Q1..Q2."""
    c, q = rec.get('C') or [], rec.get('Q') or []
    return (len(c) >= 5 and len(q) >= READ_K and all(s.get('prm') is not None for s in c[:5])
            and all(s.get('prm') is not None for s in q[:READ_K]))


# ---------------------------------------------------------------------------
# THE RULE
# ---------------------------------------------------------------------------

def gated(rec: Dict) -> bool:
    """The verifier doubts: the min step reward of the best of the first
    GATE_K plain samples is below GATE_MIN. Reads C1..C3 only."""
    cs = [s for s in (rec.get('C') or [])[:GATE_K]]
    scored = [(i, s) for i, s in enumerate(cs) if s.get('prm')]
    if not scored:
        return True
    i, b = max(scored, key=lambda t: (_last(t[1]), -t[0]))
    return min(b['prm']) < GATE_MIN


def vgra(rec: Dict) -> Tuple[Optional[float], int]:
    """(answer, LLM calls) of the v33 system on one row."""
    if gated(rec):
        return v22_pick(cands(rec, GATE_K, READ_K)), CALLS_GATED
    return v22_pick(cands(rec, GATE_K, 0)), CALLS_UNGATED


# ---------------------------------------------------------------------------
# comparators (all read off the same samples)
# ---------------------------------------------------------------------------

def plurality(answers: Sequence[Optional[float]]) -> Optional[float]:
    g = P.L.clusters(list(answers))
    return None if not g else answers[g[0][0]]


ARMS: Dict[str, Callable[[Dict], Tuple[Optional[float], int]]] = {
    'VGRA': vgra,
    'B5': lambda r: (v22_pick(cands(r, 5, 0)), 5),
    'S5': lambda r: (plurality([s.get('answer') for s in (r.get('C') or [])[:5]]), 5),
    'B3': lambda r: (v22_pick(cands(r, 3, 0)), 3),
    'S3': lambda r: (plurality([s.get('answer') for s in (r.get('C') or [])[:3]]), 3),
    'RA': lambda r: (v22_pick(cands(r, 2, 2)), 5),                       # v25, always-on reading
    'ADAPT5': lambda r: ((v22_pick(cands(r, 5, 0)), 5) if gated(r)       # same gate, plain samples
                         else (v22_pick(cands(r, 3, 0)), 3)),
}


def right(rec: Dict, arm: str) -> bool:
    return V21.correct(ARMS[arm](rec)[0], rec['gold'])


def calls(rows: Sequence[Dict], arm: str) -> float:
    return sum(ARMS[arm](r)[1] for r in rows) / max(1, len(rows))


def paired(rows: Sequence[Dict], x: str, y: str) -> Dict:
    wins = [r['pid'] for r in rows if right(r, x) and not right(r, y)]
    losses = [r['pid'] for r in rows if right(r, y) and not right(r, x)]
    return {'x': sum(right(r, x) for r in rows), 'y': sum(right(r, y) for r in rows),
            'w': len(wins), 'l': len(losses), 'net': len(wins) - len(losses),
            'p': RA.sign_p(len(wins), len(losses)), 'wins': wins, 'losses': losses, 'n': len(rows)}


# ---------------------------------------------------------------------------
# the pre-registered verdicts
# ---------------------------------------------------------------------------

def verdicts(main: Sequence[Dict], guard: Sequence[Dict]) -> List[str]:
    out = []
    s = paired(main, 'VGRA', 'S5')
    facts = (f"VGRA {s['x']} vs SC@5 {s['y']} of {s['n']}: W={s['w']} L={s['l']} net={s['net']:+d}, "
             f"sign p={s['p']:.4f}; calls/row {calls(main, 'VGRA'):.2f} vs 5")
    if s['net'] >= SUP_MIN_NET and s['w'] >= SUP_RATIO * s['l'] and s['p'] < SUP_ALPHA:
        out.append(f"PRIMARY 1  SUPPORTED: gated reading beats self-consistency with less compute ({facts})")
    elif s['net'] <= 0:
        out.append(f"PRIMARY 1  REFUTED: {facts}")
    else:
        out.append(f"PRIMARY 1  INCONCLUSIVE: {facts}")
    b = paired(main, 'VGRA', 'B5')
    c = calls(main, 'VGRA')
    facts = (f"VGRA {b['x']} vs B5 {b['y']}: W={b['w']} L={b['l']} net={b['net']:+d}, sign p={b['p']:.3f}; "
             f"calls/row {c:.2f} vs 5 ({100 * c / 5:.0f}% of B5's compute)")
    if b['net'] >= BEAT_MIN_NET and b['w'] >= SUP_RATIO * b['l'] and c <= NI_MAX_CALLS:
        out.append(f"PRIMARY 2  BEATS the PRM best-of-5 with less compute ({facts})")
    elif b['net'] >= NI_MIN_NET and c <= NI_MAX_CALLS:
        out.append(f"PRIMARY 2  NON-INFERIOR to the PRM best-of-5 with less compute ({facts})")
    else:
        out.append(f"PRIMARY 2  NOT MET: {facts}")
    if guard:
        g = paired(guard, 'VGRA', 'B5')
        out.append(f"GUARD      VGRA - B5 = {g['net']:+d} rows on {g['n']} (W={g['w']} L={g['l']}) -> "
                   + ('no harm' if g['net'] >= GUARD_MIN_NET else 'HARM'))
    return out


def gated_rows_line(rows: Sequence[Dict]) -> str:
    g = [r for r in rows if gated(r)]
    if not g:
        return "gated rows: none"
    k = lambda arm: sum(right(r, arm) for r in g)
    return (f"gated rows {len(g)}/{len(rows)}: right with B3 {k('B3')}, B5 {k('B5')}, "
            f"ADAPT5 (plain samples) {k('ADAPT5')}, VGRA (reader) {k('VGRA')}")
