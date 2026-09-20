"""
[v18.0] Scaled-twin prompting: read the structure where the model can read it.

WHAT THE AUDIT CHANGED
----------------------
Until 2026-09-20 this project believed two things about large-number problems:
that the CoT baseline collapses because of ARITHMETIC (last-digit slips), and
that program-based derivation is the strong path there (PAL "+13pp on >=100k").
Both were artifacts of gsm-hard's gold. Restricting to gold-consistent rows
(no text/code number mismatch, no negative gold, no |gold|<0.01 -- 17.2% of the
dataset fails one of these) gives, pooled over four runs:

    clean gsm-hard, largest number < 100k     CoT 98.8%   (n=83)
    clean gsm-hard, largest number >= 100k    CoT 73.8%   (n=160)

    of CoT's 42 errors on the second row:
        arithmetic slip (<0.1% from gold)        8
        off by a factor ~2..1/2                 17     <- wrong operation
        far from gold                           12     <- wrong structure
        near (<1%)                               5

So the collapse is real, but it is STRUCTURAL: the same model reads the same
templates correctly with small numbers and misreads them with 7-digit numbers.
And CoT is the best reader available at 7B -- on clean big rows CoT 80.0%, PAL
80.0%, program 65%, equations 70% (seed 44, n=40) -- so the fix has to keep
the derivation in CoT rather than route it through code.

THE MECHANISM
-------------
Give the model a TWIN of the problem in which every large number has been
divided by the same power of ten (so order, ratios and the story survive), ask
it to solve the twin first, then the original "with exactly the same steps".
The structure is read where the model reads well; the arithmetic on the real
numbers happens with that structure already fixed in context. One call, no
extraction, no rebinding, no second model.

Why not the v16.2 shrink: that replaced each large number with an unrelated
small prime, independently, which turned "drink 5,956,168 of 25,956,168" into
"drink 13 of 11" and the model, sensibly, changed the structure to cope. It
was also measured on rows that were 24% gold-defective, through the Architect
(the weaker reader), and had to rebind a blueprint. Uniform scaling preserves
feasibility by construction, and the twin is only a warm-up: nothing is
rebound, the model itself carries the structure across.

WHAT THE PRETEST MEASURES (pretest_scaled_twin.py)
--------------------------------------------------
On the 40 gold-consistent seed-44 rows with a number >= 100k, where CoT alone
is 32/40:

    scaled     twin = uniformly scaled copy      deployable
    oracle     twin = the ORIGINAL GSM8K problem (gsm-hard keeps it in the gold
               program's docstring, 1316/1319 rows)   the ceiling, not deployable

If the oracle arm does not clearly beat 32/40, magnitude is not the cause and
this direction closes. If it does and the scaled arm tracks it, the mechanism
is real and cheap. If the oracle wins and the scaled arm does not, the problem
is constructing a coherent twin, which is a smaller, well-defined problem.

Prior art to cite: analogical prompting (Yasunaga et al. 2023) recalls related
problems rather than constructing a scaled instance; Gaur & Saunshi (2023)
pair symbolic and numeric versions of the same problem; GSM-Symbolic
(Mirzadeh et al. 2024) shows number changes alone move accuracy, which is the
phenomenon this exploits rather than measures.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Unsigned, comma-aware, decimal-aware -- the same shape the rest of the repo
# uses to read problem text.
NUMBER_RE = re.compile(r'\d[\d,]*(?:\.\d+)?')

DEFAULT_THRESHOLD = 100_000.0   # the measured collapse boundary
MIN_DIGITS = 2                  # the smallest scaled number keeps this many digits


@dataclass
class ScaleResult:
    original_text: str
    twin_text: str
    divisor: float = 1.0
    mapping: Dict[float, float] = field(default_factory=dict)   # original -> scaled
    n_scaled: int = 0
    refused: str = ""            # why no twin was produced, if so

    @property
    def ok(self) -> bool:
        return self.n_scaled > 0 and not self.refused


def _parse(tok: str) -> Optional[float]:
    try:
        return float(tok.replace(",", ""))
    except ValueError:
        return None


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def scale_text(text: str, threshold: float = DEFAULT_THRESHOLD,
               min_digits: int = MIN_DIGITS) -> ScaleResult:
    """Divide every number >= threshold by ONE common power of ten.

    The divisor is chosen so the smallest large number keeps `min_digits`
    digits; larger ones keep proportionally more. Equal values map to equal
    values (coreference survives) and substitution runs per regex match, so a
    number that is a prefix of another (gsm-hard_620 has both 5072217 and
    5072217640) is never corrupted. If two distinct large numbers would round
    to the same twin value the divisor is reduced until they separate; if that
    is impossible the twin is refused rather than produced wrong.
    """
    res = ScaleResult(original_text=text, twin_text=text)
    if not text:
        res.refused = "empty text"
        return res

    values = []
    for m in NUMBER_RE.finditer(text):
        v = _parse(m.group(0))
        if v is not None and v >= threshold:
            values.append(v)
    big = sorted(set(values))
    if not big:
        res.refused = "no number reaches the threshold"
        return res

    smallest = big[0]
    k = int(math.floor(math.log10(smallest))) - (min_digits - 1)
    while k > 0:
        divisor = 10.0 ** k
        scaled = {v: float(round(v / divisor)) for v in big}
        if len(set(scaled.values())) == len(big) and min(scaled.values()) >= 1:
            break
        k -= 1                      # keep one more digit and try again
    else:
        res.refused = "large numbers cannot be separated at any scale"
        return res

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        v = _parse(tok)
        if v is None or v < threshold:
            return tok
        return _fmt(scaled[v])

    res.twin_text = NUMBER_RE.sub(repl, text)
    res.divisor = divisor
    res.mapping = dict(scaled)
    res.n_scaled = len(scaled)
    return res


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

TWIN_PROMPT = (
    "Below are two versions of the same word problem. Problem A uses smaller "
    "numbers; Problem B is the real question. The story and the sequence of "
    "steps are identical in both.\n\n"
    "Problem A:\n{twin}\n\n"
    "Problem B:\n{original}\n\n"
    "First, solve Problem A step by step and state its result on a line "
    "starting with 'Answer A:'.\n"
    "Then solve Problem B using exactly the same steps, with Problem B's "
    "numbers, showing the arithmetic. State the final numeric answer to "
    "Problem B on the LAST line, starting with 'Answer B:'."
)


def twin_prompt(twin_text: str, original_text: str) -> str:
    return TWIN_PROMPT.format(twin=twin_text.strip(), original=original_text.strip())


# ---------------------------------------------------------------------------
# Reading the two answers back
# ---------------------------------------------------------------------------

_NUM_IN_LINE = re.compile(r'-?\d[\d,]*(?:\.\d+)?')
_BOXED = re.compile(r'\\boxed\{(-?\d[\d,]*(?:\.\d+)?)\}')


def _last_number(s: str) -> Optional[float]:
    nums = _NUM_IN_LINE.findall(s)
    return _parse(nums[-1]) if nums else None


def extract_answers(raw: str) -> Tuple[Optional[float], Optional[float]]:
    """(answer_A, answer_B). B is what ships; A is a diagnostic.

    B: the last 'Answer B:' line; failing that the last \\boxed{}; failing
    that the last number after the last 'Answer B' mention. Never falls back
    to the last number of the whole text, because that would silently return
    Problem A's answer when the model stopped early -- the one confusion this
    format must not allow.
    """
    s = str(raw or "")
    a = b = None
    a_lines = re.findall(r'(?im)^\s*\**\s*answer\s*a\s*\**\s*[:=]\s*(.+)$', s)
    if a_lines:
        a = _last_number(a_lines[-1])
    b_lines = re.findall(r'(?im)^\s*\**\s*answer\s*b\s*\**\s*[:=]\s*(.+)$', s)
    if b_lines:
        b = _last_number(b_lines[-1])
    if b is None:
        boxed = _BOXED.findall(s)
        if boxed:
            b = _parse(boxed[-1])
    if b is None:
        idx = s.lower().rfind("answer b")
        if idx >= 0:
            b = _last_number(s[idx:])
    return a, b
