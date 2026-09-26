"""
[v23.0] Verifier-guided Tabu Restarts (VTR): the Verifier's rejection of a
step reaches the Solver as NEGATIVE evidence only, in a fresh context.

THE PROBLEM IT IS BUILT FOR (measured, see v23_diagnosis.py)
------------------------------------------------------------
On GSM-Symbolic P2 the Solver+Verifier system (B5, 78% on the v22 fresh
rows) has almost no selection headroom left: oracle@5 is 82 (86 vs 90 on the
v21 rows). Of its 22 errors, 18 are rows where NONE of the five samples is
right, so no vote and no verifier can fix them; they need a sample that
reads the problem differently. On the 32 rows where the Verifier doubts a
wrong leader, a fresh sample is right only 16% of the time and repeats the
leader's ANSWER only 22% of the time, but re-derives the value of the step
the Verifier doubts 43% of the time: what the failures share is a step, not
an answer. Once a right restart exists it outscores the wrong leader 58% of
the time, and a wrong one outscores a right leader only 7% of the time, so
the bottleneck is producing the right restart, not choosing it.

Three repairs have been measured against that step, and none beat blind
restarts:
  * V2P (v22): quote the implicated problem sentence  -> re-read the same way
  * RW  (v22): continue from before the doubted step    -> anchors on the prefix
  * RST (v22): fresh samples                            -> best of the three,
                                                           but blind: it re-draws
                                                           the step it escapes
V2P gave the Solver POSITIVE attention (look here) and the Solver looked and
saw the same thing. RW gave it its own failed prefix. RST gave it nothing.

WHAT VTR DOES
-------------
Tabu search (Glover, 1986) escapes a local optimum by keeping a short list of
recently rejected moves and forbidding them. VTR transplants that to the
space of READINGS of a word problem, with three agents:

  Solver    k=3 sampled CoT drafts.
  Verifier  a step-level reward for every step (Qwen2.5-Math-PRM-7B).
  Tabu      deterministic, no model. Takes the draft the Verifier would
  keeper    output (the leader: highest last-step reward). If the Verifier
            doubts it (lowest step reward < TAU), the keeper puts that
            step on the tabu list.
  Solver    re-solves from scratch, twice, in a FRESH context that contains
            the problem and the tabu list -- "this step was checked and is
            WRONG" -- and nothing else from the failed attempt.
  Verifier  scores the restarts against the ORIGINAL problem (no note) and
            the answer is the best of {3 drafts + 2 restarts} by last-step
            reward. The drafts stay in the pool, which is tabu search's
            aspiration criterion: a tabu reading can still win if nothing
            the restarts produce scores higher.

The information-flow rule is the design: the Solver never sees a full
failed solution (the anchoring that killed RW and self-correction), it
never gets a positive cue to re-read (what killed V2P), and it is not blind
either (what limits RST). It gets one verifier-localised negative fact.

WHAT IS AND IS NOT NEW (checked 2026-09-26)
-------------------------------------------
Not new, used as components or baselines: best-of-N with a PRM (Lightman
2023; Qwen PRM); rewind-and-continue from a doubted step (StepCo, ACL 2025);
self-refinement with the failed solution in context (Self-Refine, Reflexion);
hints built from earlier ANSWERS (Progressive-Hint Prompting, 2023); invalid
demonstrations from OTHER problems (Contrastive CoT, 2023); independently
answered verification questions (CoVe, SelfCheck).
New here: (1) the negative evidence is a single STEP of the SAME problem,
chosen by a learned step verifier; (2) it is delivered to a fresh context,
never with the failed solution; (3) the restart is framed and evaluated as
tabu search over readings, with the verifier as the objective and the
original drafts as the aspiration pool; (4) the ANS control (the same
frame, with the leader's final answer as the only negative fact) isolates
whether the verifier's LOCALISATION is what matters, as opposed to excluding
the modal answer.

Everything in this file is deterministic and model-free, so it is unit
tested offline (test_v23.py). pretest_v23.py runs it.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import pretest_v21 as V21
import score_prm_v22 as P

TAU = 0.95            # frozen on the v21 dev rows only; see v23_diagnosis.py
N_RESTARTS = 2        # the same 2 extra samples RST/V2P/RW spent in v22
K_DRAFTS = 3
MAX_STEP_CHARS = 700

TABU_MARKER = 'was checked, and this step of it was found to be WRONG'
TABU_MARKER_MULTI = 'were checked, and these steps of them were found to be WRONG'
ANS_MARKER = 'was checked, and its final answer'
DO_NOT = "Do not repeat that mistake; solve the problem again from the beginning."

_NUM = re.compile(r"(?<![A-Za-z_])-?\d+(?:,\d{3})*(?:\.\d+)?")
_CALC = re.compile(
    r"\d\s*(?:=|\+|-|\*|/|×|÷|\\times|\\div|\\cdot)\s*[\d(\\{]"   # 3 + 4, 3 \times 4
    r"|\\[dt]?frac\s*\{"                                          # \frac{..}{..}
    r"|=\s*-?\$?\\?\(?\s*\d")                                    # = 12


def last(prm: Sequence[float]) -> float:
    return float(prm[-1]) if prm else 0.0


def low(prm: Sequence[float]) -> float:
    return float(min(prm)) if prm else 0.0


def leader(samples: Sequence[Dict], k: int = K_DRAFTS) -> Optional[int]:
    """The draft the Verifier would output: highest last-step reward among
    the first k drafts that parsed to an answer, ties to the earliest. The
    same choice as score_prm_v22.best_of_n, so the leader's answer IS B3."""
    ss = list(samples)[:k]
    idx = [i for i, s in enumerate(ss) if s.get('answer') is not None]
    if not idx:
        return None
    return max(idx, key=lambda i: (last(ss[i].get('prm') or []), -i))


# The PRM's own step convention (blank-line separated), so step i here is
# reward i there. Shared, not copied, so the two can never drift apart.
split_steps = P.split_steps


def substantive(step: str) -> bool:
    """A step that computes something. A plan ("1. Find the swimmers. 2. ...")
    or an introduction can be the lowest-scored step, but forbidding it tells
    the Solver nothing."""
    return bool(step) and bool(_CALC.search(step))


def doubted_step(steps: Sequence[str], prm: Sequence[float]) -> Tuple[Optional[int], str]:
    """The lowest-rewarded SUBSTANTIVE step. Steps and rewards are aligned by
    position; a trace truncated after scoring only loses its tail."""
    n = min(len(steps), len(prm))
    for i in sorted(range(n), key=lambda i: (prm[i], i)):
        if substantive(steps[i]):
            return i, steps[i]
    return None, ''


def compact(step: str, limit: int = MAX_STEP_CHARS) -> str:
    s = re.sub(r'[ \t]+\n', '\n', str(step).strip())
    s = re.sub(r'\n{2,}', '\n', s)
    if len(s) <= limit:
        return s
    half = limit // 2
    return s[:half].rstrip() + '\n[...]\n' + s[-(half - 7):].lstrip()


def fmt(x: Optional[float]) -> str:
    if x is None:
        return '?'
    x = float(x)
    return str(int(x)) if x == int(x) else f'{x:g}'


def tabu_note(steps: Sequence[str]) -> str:
    steps = [compact(s) for s in steps if s and str(s).strip()]
    if not steps:
        return ''
    if len(steps) == 1:
        return ("Warning: an earlier solution of this problem was checked, and this "
                "step of it was found to be WRONG:\n\"\"\"\n" + steps[0] + "\n\"\"\"\n" + DO_NOT)
    body = '\n'.join(f'{i}. """\n{s}\n"""' for i, s in enumerate(steps, 1))
    return ("Warning: earlier solutions of this problem were checked, and these steps "
            "of them were found to be WRONG:\n" + body + "\n" + DO_NOT)


def answer_note(answer: Optional[float]) -> str:
    """The attribution control: the same frame, but the only negative fact is
    the leader's final answer. No localisation."""
    if answer is None:
        return ''
    return ("Warning: an earlier solution of this problem was checked, and its final "
            f"answer, {fmt(answer)}, was found to be WRONG.\n" + DO_NOT)


def restart_prompt(problem: str, note: str = '') -> str:
    """Exactly the CoT prompt every earlier arm used, with the note appended to
    the problem where V2P put its note. An empty note is a plain restart."""
    return V21.COT_PROMPT.format(problem=problem + ('\n\n' + note if note else ''))


def plan(samples: Sequence[Dict], raws: Optional[Sequence[str]] = None,
         tau: float = TAU, k: int = K_DRAFTS) -> Dict:
    """The tabu keeper's whole decision for one row, from the drafts alone.

    `samples` carry 'answer' and 'prm'; `raws` are the texts the rewards were
    computed on (defaults to samples[i]['raw'])."""
    L = leader(samples, k)
    out: Dict = {'leader': L, 'leader_answer': None, 'leader_min': None,
                 'triggered': False, 'cut': None, 'doubted': '', 'fallback': None}
    if L is None:
        # nothing parsed: every repair arm is a plain restart
        out.update(triggered=True, fallback='no-leader')
        return out
    s = samples[L]
    raw = raws[L] if raws is not None else s.get('raw', '')
    prm = list(s.get('prm') or [])
    out['leader_answer'] = s.get('answer')
    out['leader_min'] = low(prm)
    if not prm or low(prm) >= tau:
        return out
    out['triggered'] = True
    cut, step = doubted_step(split_steps(raw), prm)
    out['cut'], out['doubted'] = cut, step
    if cut is None:
        out['fallback'] = 'answer'   # no computing step: TABU degrades to ANS
    return out


def notes_for(p: Dict) -> Dict[str, str]:
    """The note each arm sends, from a plan(). RST/RSTN send none."""
    tabu = tabu_note([p['doubted']]) if p.get('doubted') else answer_note(p.get('leader_answer'))
    return {'TABU': tabu, 'ANS': answer_note(p.get('leader_answer'))}


def numbers(text: str) -> List[float]:
    out = []
    for t in _NUM.findall(str(text or '')):
        try:
            out.append(float(t.replace(',', '')))
        except ValueError:
            pass
    return out


def doubted_value(step: str, problem: str) -> Optional[float]:
    """The value the doubted step concludes with, unless the problem states it
    (then it is a given being used, not a conclusion being drawn)."""
    v = numbers(step)
    if not v:
        return None
    given = numbers(problem)
    return None if any(abs(v[-1] - g) < 1e-9 for g in given) else v[-1]


def contains(raw: str, value: Optional[float]) -> Optional[bool]:
    if value is None:
        return None
    return any(abs(value - x) < 1e-6 for x in numbers(raw))


# Best-of-N by last-step reward, ties to the earliest: the v22 B3/B5 rule.
best_of = P.best_of_n
