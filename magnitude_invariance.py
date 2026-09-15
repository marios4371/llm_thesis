"""
[v16.0] Magnitude-invariant metamorphic verification — the deterministic half.

Measured motivation. Inside gsm-hard, split only by the largest number in the
problem text, the system scores 98.4% when that number is under 100 and 0.0%
when it is over 10M; the baseline traces the same curve, so this is the model,
not the pipeline. gsm-hard IS GSM8K with large numbers substituted, and 39 of
the 106 large-number errors land within 0.1% of gold (9810512 vs 9810510,
9202335 vs 9202337). The reasoning is right and the arithmetic slips in the
last digits.

The mechanism. A solution's STRUCTURE does not depend on the magnitude of its
inputs, so: solve a magnitude-shrunk copy of the problem, where the model is
measured at 98.4%; keep the structure; rebind the original values; evaluate
exactly with the CAS. The model never handles a large number -- it reasons on
small ones, and deterministic machinery supplies the real values.

This module is everything in that loop that needs no LLM and no GPU: the
shrink, the rebind, and the metamorphic agreement check. Evaluation is
delegated to the shipped SIV rather than reimplemented, so the numbers here
are the same numbers the pipeline produces.

Probe values are distinct PRIMES absent from the problem text. Distinctness
and absence keep the rebind unambiguous; primality is what makes agreement
between two structures evidence rather than coincidence -- on small composite
probes like 2 and 4 many different formulas collide on the same value, which
would manufacture exactly the false corroboration that Run 0 showed is fatal.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Numbers with optional thousands separators and an optional decimal tail.
NUMBER_RE = re.compile(r'\d[\d,]*(?:\.\d+)?')

# Small primes, used in order. 2/3/5 are deliberately excluded: they are the
# commonest accidental factors in word problems ("twice", "half", "5%"), and a
# probe that collides with the problem's own arithmetic defeats the point.
PROBE_PRIMES: Tuple[int, ...] = (
    7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
    79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
)

DEFAULT_THRESHOLD = 10_000


@dataclass
class ShrinkResult:
    """The shrunk problem plus everything needed to undo the substitution."""
    original_text: str
    shrunk_text: str
    probe_to_original: Dict[float, float] = field(default_factory=dict)
    original_to_probe: Dict[float, float] = field(default_factory=dict)
    n_shrunk: int = 0
    skipped_no_probe: List[float] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.n_shrunk > 0


def parse_number(tok: str) -> Optional[float]:
    try:
        return float(tok.replace(',', ''))
    except ValueError:
        return None


def text_number_set(text: str) -> set:
    """Every numeric literal in the text, as floats -- used to guarantee a
    probe cannot collide with a number the problem already mentions."""
    out = set()
    for m in NUMBER_RE.finditer(text or ''):
        v = parse_number(m.group(0))
        if v is not None:
            out.add(v)
    return out


def shrink_text(text: str, threshold: int = DEFAULT_THRESHOLD) -> ShrinkResult:
    """Replace every number >= threshold with a distinct small prime.

    Substitution runs per regex match, so a number that is a prefix of another
    (gsm-hard_620 carries both 5072217 and 5072217640) cannot be corrupted by
    substring replacement. Equal values map to the same probe, keeping any
    coreference in the problem intact.
    """
    res = ShrinkResult(original_text=text, shrunk_text=text)
    if not text:
        return res

    present = text_number_set(text)
    available = [p for p in PROBE_PRIMES if float(p) not in present]
    assigned: Dict[float, float] = {}
    used = iter(available)

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        val = parse_number(tok)
        if val is None or val < threshold:
            return tok
        if val in assigned:
            return _fmt(assigned[val])
        try:
            probe = float(next(used))
        except StopIteration:
            res.skipped_no_probe.append(val)
            return tok
        assigned[val] = probe
        return _fmt(probe)

    res.shrunk_text = NUMBER_RE.sub(repl, text)
    res.original_to_probe = dict(assigned)
    res.probe_to_original = {p: o for o, p in assigned.items()}
    res.n_shrunk = len(assigned)
    return res


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


def rebind_givens(givens: Dict[str, object],
                  shrink: ShrinkResult,
                  tol: float = 1e-9) -> Tuple[Dict[str, object], List[str]]:
    """Swap probe values in a blueprint's givens back to the originals.

    Only EXACT probe matches are rebound. Givens are inputs read off the
    problem; anything computed from them lives in the equations, so a value
    that is merely probe-derived (probe * 12) must NOT be rescaled here -- the
    equations will recompute it once the inputs are real. Returns the rebound
    givens and the names that were changed.
    """
    out: Dict[str, object] = {}
    touched: List[str] = []
    for k, v in givens.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            hit = next((p for p in shrink.probe_to_original
                        if abs(float(v) - p) <= tol), None)
            if hit is not None:
                out[k] = shrink.probe_to_original[hit]
                touched.append(k)
                continue
        out[k] = v
    return out, touched


def evaluate(blueprint: Dict[str, object]) -> Optional[float]:
    """Blueprint's own value, via the shipped verifier -- not a second
    implementation of the CAS, so these numbers match the pipeline's."""
    from siv_module import SymbolicInverseVerifier as SIV
    if not blueprint.get('equations') or not blueprint.get('givens'):
        return None
    try:
        r = SIV.verify(blueprint, 0.0)
    except Exception:
        return None
    if getattr(r, 'has_structural_defect', False):
        return None
    return r.blueprint_answer


@dataclass
class MetamorphicResult:
    agree: Optional[bool]          # None when either side could not be evaluated
    value_big_structure: Optional[float]
    value_small_structure: Optional[float]
    rel_error: Optional[float]
    final_answer: Optional[float]  # small structure rebound to real values
    reason: str = ''


def metamorphic_check(bp_big: Dict[str, object],
                      bp_small: Dict[str, object],
                      shrink: ShrinkResult,
                      tol: float = 1e-6) -> MetamorphicResult:
    """Do the two structures agree in the regime where the model is reliable?

    Both blueprints are evaluated AT THE SHRUNK VALUES. The one written for the
    original problem is pushed down to the probes; the one written for the
    shrunk problem is already there. Agreement means a structure derived where
    the model measures 98.4% corroborates the structure derived where it
    measures 50%, on an input the two were produced from independently.

    The returned final_answer is the SMALL structure rebound to the real
    values and evaluated exactly -- the whole point being that the model never
    has to do arithmetic on a large number.
    """
    small_givens = bp_small.get('givens') or {}
    big_givens = bp_big.get('givens') or {}

    # push the big-problem blueprint down onto the probes
    pushed = {}
    for k, v in big_givens.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            hit = next((o for o in shrink.original_to_probe
                        if abs(float(v) - o) <= 1e-9), None)
            pushed[k] = shrink.original_to_probe[hit] if hit is not None else v
        else:
            pushed[k] = v

    v_big = evaluate({**bp_big, 'givens': pushed})
    v_small = evaluate(bp_small)
    if v_big is None or v_small is None:
        return MetamorphicResult(None, v_big, v_small, None, None,
                                 'one side did not evaluate')

    rel = abs(v_big - v_small) / max(abs(v_small), 1e-9)
    agree = rel <= tol

    rebound, _ = rebind_givens(small_givens, shrink)
    final = evaluate({**bp_small, 'givens': rebound})
    return MetamorphicResult(agree, v_big, v_small, rel, final,
                             'agree' if agree else 'structures diverge')
