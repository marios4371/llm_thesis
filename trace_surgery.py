"""
[v38.0] Symbolic Surgery: trace-level SIV turned from a refuter into a repairer.

WHY THIS VERSION EXISTS
-----------------------
Trace-level SIV (trace_siv.py, v35) audits the arithmetic claims a Qwen-Math
chain of thought writes out and DROPS every sample with a provably false one.
That fixes selection, never coverage: a row whose five samples all hold the
same slip has nothing left to select. On 2026-10-20 the repeated slips turned
out to be carry errors in multi-operand sums (84% of the slips repeated in
>= 2 samples are +-1 at one digit position, against 24% of isolated slips,
Fisher p = 4e-9). Resampling repeats them and the PRM scores them >= 0.9, so
only an intervention can fix them. The CAS already knows the right value.

THE OPERATION (per refuted sample; nothing here calls a model)
--------------------------------------------------------------
  locate     the FIRST false claim of the trace, in reading order, under the
             audit's own rules (run-on chains and steps the trace itself calls
             false are not refuted, exactly as trace_siv.check_step)
  propagate  replace the claimed value by the CAS value and carry the
             correction forward by numeric matching: later left-hand sides that
             reuse a corrected value are re-evaluated, the claims that become
             false are corrected in turn, and the final answer is mapped when it
             equals a corrected value. Free and offline. Values the trace
             carries in PROSE are not re-derived (the documented blind spot).
  resume     cut the trace right after that claim's relation sign, write the CAS
             value, and let the solver continue from the corrected prefix (one
             partial call, same sampling settings; pretest_v38.resume_call).
The repaired trace is re-audited and re-scored by the PRM before it returns to
the pool; a repair that is itself refuted is dropped like any refuted sample.
Which of propagate / resume / plain refutation a sample gets is the policy
`surgery_policy`, frozen on 2026-10-20 before any v38 sample (its docstring).

THE RULE (ADAPT-RS = ADAPT-R with surgery; v22's gate and pick, unchanged)
--------------------------------------------------------------------------
  1. Draw C1..C3; audit; operate on each refuted sample per the policy.
  2. b = the best survivor by last-step PRM reward (ties: lower position).
  3. If b exists and min(step rewards of b) >= 0.85: answer b.
  4. Otherwise draw C4, C5, operate on them, and answer with the best survivor
     of C1..C5 (if none survives, v22's pick over the five originals).
Calls = 3 or 5, plus one partial call per resumed sample.

WHAT IS AND IS NOT NEW (checked 2026-10-20)
------------------------------------------
Not new: a post-hoc calculator that re-executes the equations of ONE chain of
thought and propagates results by string matching (Wei et al. 2022, arXiv
2201.11903, appendix); probing for arithmetic errors and re-prompting the
model (arXiv 2507.12379); a fine-tuned "Fix & Continue" (LEMMA).
New here: splicing the exact CAS value into a SAMPLED trace and resuming it
inside verifier-routed sampling, with the repaired trace re-audited and
re-scored before it competes; the repair targets correlated slips, the one
error class neither voting nor a same-policy PRM can see.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import score_prm_v22 as P
import trace_siv as T

PROPAGATE, RESUME, REFUTE = 'propagate', 'resume', 'refute'
ACTIONS = (PROPAGATE, RESUME, REFUTE)

NUM =re.compile(r'(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d])|(?<![\w.,])\d+(?:\.\d+)?')
_SPLIT = re.compile(r'\\\\|\n|\\q?quad|\\text\{\s*(?:or|and|so|since|if|then|thus|hence)\b[^{}]*\}')
_REL = re.compile(r'(=|\\approx|≈)')
_EXPONENT = re.compile(r'[\^_]\s*\{?\s*$')                 # `x^2`, `a_{1}`: an exponent or index is not a carried value
_UNITS = re.compile(r'\\(?:text|mathrm|textbf|mbox|textit|mathit)\s*\{[^{}]*\}')
Fix = Tuple[float, float]                                   # (claimed wrong value, CAS value)


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------

def close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def fmt(v: float) -> str:
    """How a CAS value is written into a trace: an integer plainly, else at most
    ten decimals with the trailing zeros dropped (never an exponent)."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f'{v:.10f}'.rstrip('0').rstrip('.')


def subst(expr: str, fix: Sequence[Fix]) -> str:
    """Every number literal of expr equal to a corrected wrong value becomes the
    CAS value (numeric matching, Wei et al. 2022 style)."""
    if not fix:
        return expr

    def rep(m):
        s = m.group(0)
        if _EXPONENT.search(expr[:m.start()]):
            return s
        try:
            v = float(s.replace(',', ''))
        except ValueError:
            return s
        for wrong, right in fix:
            if close(v, wrong):
                return fmt(right)
        return s
    return NUM.sub(rep, expr)


def _shown(b: str, va: float, vb: float) -> float:
    """The rounding the right side shows (`33.33` tolerates 0.005), as trace_siv._tol."""
    tol = 1e-9 * max(1.0, abs(va), abs(vb))
    n = T.normalize(_UNITS.sub(' ', b))
    d = T._decimals(n[0]) if n else None
    return max(tol, 0.5 * 10 ** (-d) + 1e-9) if d is not None else tol


def unitless_value(side: str) -> Optional[float]:
    """trace_siv.value of one side with \\text{} unit words removed, so that
    `5573040 \\text{ bales} - 18 \\text{ bales}` has a value (the audit itself
    already refutes such claims through its unit symbols)."""
    v = T.value(side)
    return v if v is not None else T.value(_UNITS.sub(' ', side))


# ---------------------------------------------------------------------------
# where the claims sit in the raw text
# ---------------------------------------------------------------------------

def _steps(raw: str) -> List[Tuple[int, str]]:
    """(offset in raw, text) of every PRM step (score_prm_v22.split_steps)."""
    out, cur = [], 0
    for st in P.split_steps(raw or ''):
        k = raw.find(st, cur)
        if k < 0:                                            # '(empty)' or a reflowed step
            out.append((-1, st))
            continue
        out.append((k, st))
        cur = k + len(st)
    return out


def pieces(step: str) -> List[Tuple[int, str]]:
    """(offset in step, text) of the math pieces trace_siv.segments checks, in
    READING order (segments() lists LaTeX before plain text). Offsets index the
    original text; '&' is left in place (callers drop it before checking)."""
    segs: List[Tuple[int, str]] = []
    for m in T._MATH_SEG.finditer(step):
        g = next(i for i in range(1, 5) if m.group(i) is not None)
        segs.append((m.start(g), m.group(g)))
    masked = T._MATH_SEG.sub(lambda m: ' ' * len(m.group(0)), step)
    segs += [(m.start(1), m.group(1)) for m in T._PLAIN.finditer(masked)]
    out = []
    for off, seg in sorted(segs):
        cur = 0
        for m in list(_SPLIT.finditer(seg)) + [None]:
            end = m.start() if m else len(seg)
            piece = seg[cur:end]
            if piece and piece.strip() and len(piece.replace('&', '')) <= T.MAX_SEG:
                out.append((off + cur, piece))
            if m:
                cur = m.end()
    return out


# ---------------------------------------------------------------------------
# the walk: locate, correct, carry forward
# ---------------------------------------------------------------------------

def walk(raw: str) -> Dict:
    """One pass over the trace in reading order.

    Returns {'site': the first false claim or None, 'fixes': [(wrong, right)],
    'edits': [(start, end, text)] on raw}. A claim is corrected when the audit
    would refute it once earlier corrections are substituted into its left side
    (so a claim that reused a wrong value is corrected too: that is the
    propagation). Only unit-free numeric sides are corrected; congruences and
    the ANCHOR check refute without a single claimed value and are left alone.
    """
    fix: List[Fix] = []
    edits: List[Tuple[int, int, str]] = []
    site = None
    for si, (soff, step) in enumerate(_steps(raw or '')):
        disowned = bool(T._DISOWNED.search(T._MATH_SEG.sub(' ', step)))
        for poff, piece in pieces(step):
            if '\\equiv' in piece:
                continue
            toks = _REL.split(piece)
            parts_o, rels = toks[0::2], toks[1::2]
            parts = [p.replace('&', '') for p in parts_o]
            pos = [poff + sum(len(t) for t in toks[:2 * j]) for j in range(len(parts))]
            carry = None
            new = list(parts_o)
            fixed = set()
            for i in range(len(parts) - 1):
                a, b = parts[i], parts[i + 1]
                if i not in fixed:                           # as a left side it reads the corrected values
                    new[i] = subst(parts_o[i], fix)
                if not a.strip() or not b.strip():
                    carry = None
                    continue
                a_sub = subst(a, fix)
                va = carry if carry is not None else unitless_value(a_sub)
                vb = unitless_value(b)
                if va is None or vb is None:
                    carry = None
                    continue
                if carry is not None:                        # `= wrong = more`: the chain carries the CAS value
                    verdict = T.FALSE if abs(va - vb) > _shown(b, va, vb) else T.TRUE
                else:
                    verdict = T.check_pair(a_sub, b, approx=rels[i] != '=')
                    if verdict == T.FALSE and T._runon(parts, i):
                        verdict = None
                carry = None
                if verdict != T.FALSE or disowned or close(va, vb):
                    continue
                fix.append((vb, va))
                carry = va
                fixed.add(i + 1)
                lits = [m for m in NUM.finditer(parts_o[i + 1]) if close(float(m.group(0).replace(',', '')), vb)]
                if len(lits) == 1:
                    new[i + 1] = parts_o[i + 1][:lits[0].start()] + fmt(va) + parts_o[i + 1][lits[0].end():]
                else:                                        # an expression on the right: write the value instead
                    lead = parts_o[i + 1][:len(parts_o[i + 1]) - len(parts_o[i + 1].lstrip())]
                    tail = parts_o[i + 1][len(parts_o[i + 1].rstrip()):]
                    new[i + 1] = lead + fmt(va) + tail
                if site is None and soff >= 0:
                    cut = soff + pos[i] + len(parts_o[i]) + len(rels[i])
                    site = {'step': si, 'claim': f"{a.strip()} {rels[i]} {b.strip()}"[:160],
                            'lhs': a.strip()[:200], 'claimed': vb, 'value': va, 'cut': cut,
                            'rhs_literal': len(lits) == 1}
            if soff >= 0:
                for j, (o, n) in enumerate(zip(parts_o, new)):
                    if o != n:
                        edits.append((soff + pos[j], soff + pos[j] + len(o), n))
    return {'site': site, 'fixes': fix, 'edits': edits}


def first_false(raw: str) -> Optional[Dict]:
    """The first correctable false claim: step index, claim text, claimed value,
    CAS value and `cut`, the offset in raw just after its relation sign."""
    return walk(raw)['site']


def map_answer(answer: Optional[float], fix: Sequence[Fix]) -> Optional[float]:
    if answer is None:
        return None
    for wrong, right in fix:
        if close(answer, wrong):
            return right
    return answer


_FINAL = re.compile(r'\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}|answer\s*(?:is)?\s*[:=]\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)',
                    re.I)


def propagate(raw: str, answer: Optional[float]) -> Optional[Dict]:
    """Repair by propagation. None when nothing in the trace is correctable;
    else {'raw': the repaired text, 'answer': the mapped final answer,
    'fixes', 'site', 'reached': whether the correction reached the answer}.
    The repaired text has every corrected claim rewritten and the final answer
    (the last \\boxed{} or 'Answer:' number) mapped; prose is left as it is."""
    w = walk(raw or '')
    if not w['fixes']:
        return None
    new_answer = map_answer(answer, w['fixes'])
    edits = list(w['edits'])
    if answer is not None and new_answer is not None and not close(new_answer, answer):
        hits = list(_FINAL.finditer(raw))
        for m in hits[-2:]:
            g = 1 if m.group(1) is not None else 2
            try:
                if close(float(m.group(g).replace(',', '')), answer):
                    edits.append((m.start(g), m.end(g), fmt(new_answer)))
            except ValueError:
                pass
    text = raw
    last = len(raw) + 1
    for s, e, n in sorted(set(edits), reverse=True):
        if e <= last:                                        # overlapping edits: keep the later one
            text = text[:s] + n + text[e:]
            last = s
    return {'raw': text, 'answer': new_answer, 'fixes': w['fixes'], 'site': w['site'],
            'reached': answer is not None and new_answer is not None and not close(new_answer, answer)}


def resume_prefix(raw: str, site: Optional[Dict] = None) -> Optional[str]:
    """The corrected prefix the solver resumes from: the trace up to the first
    false claim's relation sign, then the CAS value. None if nothing to cut."""
    site = site if site is not None else first_false(raw)
    if not site or site.get('cut') is None:
        return None
    return raw[:site['cut']].rstrip() + ' ' + fmt(site['value'])


def signature(fixes: Sequence[Fix]) -> List[Tuple[float, float]]:
    """A slip's identity across samples: (claimed, CAS) rounded, so that the
    same carry error written with other spacing or separators is one slip."""
    return [(round(w, 6), round(r, 6)) for w, r in fixes]


def n_sharing(samples: Sequence[Dict], i: int, idx: Sequence[int]) -> int:
    """How many samples among idx (sample i included) hold sample i's FIRST
    false claim among their own corrected slips: >= 2 is a correlated slip."""
    me = (samples[i].get('surgery') or {}).get('site')
    if not me:
        return 0
    key = (round(me['claimed'], 6), round(me['value'], 6))
    return sum(key in set(map(tuple, (samples[j].get('surgery') or {}).get('slips') or [])) for j in idx)


def operate(raw: str, answer: Optional[float]) -> Optional[Dict]:
    """Everything the run needs for one refuted sample, model-free: the site, the
    propagated repair and the resume prefix. The runner adds the PRM scores and
    the resumed continuation; the audits of both are recomputed offline."""
    p = propagate(raw, answer)
    if p is None:
        return None
    return {'site': p['site'], 'slips': signature(p['fixes']),
            'prop': {'raw': p['raw'], 'answer': p['answer'], 'reached': p['reached']},
            'prefix': resume_prefix(raw, p['site'])}


# ---------------------------------------------------------------------------
# the policy (the thesis author's decision) and the rule
# ---------------------------------------------------------------------------

def surgery_policy(sample: Dict, n_samples_sharing_the_false_claim: int) -> str:
    """Which operation a REFUTED sample gets: PROPAGATE, RESUME or REFUTE.

    `sample` carries 'answer', 'prm' (step rewards), 'audit' (trace_siv
    verdicts) and 'surgery' = operate(...): its 'site' (the first false claim:
    'claim', 'claimed', 'value', 'step', 'rhs_literal'), 'slips' (every
    corrected (claimed, CAS) pair) and 'prop' with 'reached' (True when the
    propagated correction changed the final answer). n_samples_sharing... is
    n_sharing() over the samples drawn so far (this one included): 2 or more
    means the same slip recurs, i.e. a correlated slip.

    The trade-offs (measured on the 680 stored rows, V38_PLAN.md section 2):
      * PROPAGATE costs nothing, but it cannot re-derive values the trace
        carries in prose; when 'reached' is False the final answer did not move.
      * RESUME costs one partial call, and the model may re-derive the same
        wrong value after the splice (the slip is a carry habit, not a typo).
      * REFUTE (v35) costs nothing and never adds a wrong candidate, but a
        correlated slip cannot be fixed by resampling: when every sample holds
        it, refutation leaves nothing right to pick.
    Must return one of ACTIONS.

    FROZEN 2026-10-20, before any v38 sample exists (V38_PLAN.md section 3):
      1. the correction reached the final answer       -> PROPAGATE
         (free; rescues 75/108 such wrong samples on the stored rows)
      2. it did not, and the slip recurs (n >= 2)        -> RESUME
         (propagation rescues 0/80 when it does not reach the answer, and a
         correlated slip is the one error resampling repeats: pay the call)
      3. it did not, and the slip is isolated (n = 1)    -> REFUTE
         (the other samples are independent draws without this slip; these
         slips are often symptoms of a derivation already off course, such
         as `30 - 35 = 3`, where a resume would buy a call for little)
    """
    if sample['surgery']['prop']['reached']:
        return PROPAGATE
    if n_samples_sharing_the_false_claim >= 2:
        return RESUME
    return REFUTE


def repaired(sample: Dict, action: str) -> Optional[Dict]:
    """The sample that re-enters the pool after `action`, or None (dropped).
    It needs 'answer', 'prm' and 'audit'; a repair the audit still refutes is
    dropped (the re-audit of the repaired trace)."""
    s = sample.get('surgery') or {}
    rep = s.get('prop') if action == PROPAGATE else s.get('resume') if action == RESUME else None
    if not rep or rep.get('answer') is None or not rep.get('prm'):
        return None
    if T.refutes(rep.get('audit') or [], rep.get('prm')):
        return None
    return rep


def pool(samples: Sequence[Dict], idx: Sequence[int], policy: Callable[[Dict, int], str] = None,
         resumed: Optional[set] = None) -> List[Dict]:
    """The candidates of the drawn samples idx: unrefuted samples as they are,
    refuted ones through the policy. The indices that were resumed are added to
    `resumed` (a resume is a real call: re-deciding a sample later reuses it)."""
    policy = policy or surgery_policy
    out = []
    for i in idx:
        x = samples[i]
        if x.get('answer') is None or not x.get('prm'):
            continue
        if not T.refutes(x.get('audit') or [], x.get('prm')):
            out.append(x)
            continue
        if not x.get('surgery'):
            continue
        action = policy(x, n_sharing(samples, i, idx))
        if action not in ACTIONS:
            raise ValueError(f'surgery_policy returned {action!r}, not one of {ACTIONS}')
        if action == RESUME and resumed is not None:
            resumed.add(i)
        r = repaired(x, action)
        if r is not None:
            out.append(r)
    return out


def adapt_rs(samples: Sequence[Dict], policy: Callable[[Dict, int], str] = None) -> Tuple[Optional[float], int]:
    """(answer, solver calls) of ADAPT-RS; each resumed sample costs one call."""
    resumed: set = set()
    first = list(range(min(T.GATE_K, len(samples))))
    cand = pool(samples, first, policy, resumed)
    b = T.best(cand, range(len(cand)))
    if b is not None and min(cand[b]['prm']) >= T.GATE_MIN:
        return cand[b]['answer'], T.GATE_K + len(resumed)
    full = list(range(min(T.FULL_K, len(samples))))
    cand = pool(samples, full, policy, resumed)
    b = T.best(cand, range(len(cand)))
    if b is None:
        o = T.best(samples, full)                            # nothing survives: v22's pick
        return (None if o is None else samples[o]['answer']), T.FULL_K + len(resumed)
    return cand[b]['answer'], T.FULL_K + len(resumed)
