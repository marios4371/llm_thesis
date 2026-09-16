"""
[v16.3] Near-agreement as a structural fingerprint.

The idea. Verification by agreement always asks whether two derivations produce
the SAME answer. On these runs that question carries no information: where the
baseline and the blueprint agree exactly (257 of 533 rows) both are right 97.7%
of the time, so there is nothing to decide. The informative event is the one
exact matching throws away -- agreement that is NEAR but not exact.

Why that should mean something. 39 of the 106 large-number errors land within
0.1% of gold: 9810512 against 9810510, 9202335 against 9202337. The model sets
up the right computation and slips in the last digits of an 8-digit product. So
two derivations landing within 0.001% of each other almost certainly ran the
SAME computation and differ only by that slip. That corroborates the structure
-- and once the structure is corroborated the exact value should come from the
CAS, which does not slip, rather than from either noisy answer.

Measured over the four pooled runs (n=533 rows carrying both numbers):

    baseline vs blueprint    n     baseline   blueprint
    exactly equal          257       97.7%       97.7%    nothing to gain
    near, not equal         20       40.0%       80.0%    <-- the signal
    within 1%               14       42.9%       21.4%
    far apart              240       60.0%       11.7%

The band is sharply peaked, not a trend: just outside it the blueprint is far
WORSE than the baseline, which is what makes this a fingerprint rather than a
general "trust the blueprint more" rule -- the latter was tested and pooled to
-2.0pp.

Evidence. Pooled in-sample at tau=1e-4: +1.33pp, W=10 L=2, p=0.039. Leave-one-
run-out, picking tau on three runs and scoring the fourth: +1.00pp, W=9 L=3,
p=0.146 -- positive in 3 of 4 folds and the ONLY held-out-positive result among
everything tested (selection rules reached -1.50pp, magnitude routing -0.83pp).
Not significant held-out; a fresh-seed run is what would settle it.

Two properties worth noting. The effect survives tau from 1e-6 to 1e-3 and
breaks cleanly at 1e-2, so it is a plateau with an edge rather than a fitted
threshold. And 21 of its 23 activations fall on problems whose largest number
reaches 100k, against 1 activation in 249 small-number rows -- exactly where
the theory puts it, since an arithmetic slip is only a SMALL RELATIVE error
when the numbers are large. On plain GSM8K, where answers are 18 or 72, a slip
produces a completely different number and this band does not exist at all.

Cost: zero extra LLM calls. Both numbers are already computed on every row.
"""
from __future__ import annotations

from typing import Optional

# Plateau midpoint. Anything from 1e-6 to 1e-3 behaves the same; 1e-2 does not.
DEFAULT_TAU = 1e-4


def relative_gap(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """|a - b| / |b|, or None when either side is missing."""
    if a is None or b is None:
        return None
    return abs(a - b) / max(abs(b), 1e-9)


def structurally_corroborated(cot_answer: Optional[float],
                              symbolic_answer: Optional[float],
                              tau: float = DEFAULT_TAU) -> bool:
    """True when the two derivations agree up to arithmetic noise.

    Strictly greater than zero is load-bearing: exact equality means the two
    ran the same computation AND got the same digits, which the table above
    shows is uninformative. It is the residual slip that identifies a shared
    structure reached by an independent route.
    """
    gap = relative_gap(cot_answer, symbolic_answer)
    return gap is not None and 0.0 < gap < tau


def resolve(cot_answer: Optional[float],
            symbolic_answer: Optional[float],
            tau: float = DEFAULT_TAU) -> Optional[float]:
    """The answer to return: the exact symbolic value when the structure is
    corroborated, otherwise the model's own answer untouched.

    Deliberately conservative -- it changes nothing outside the band, so it
    cannot regress the 97.7% of rows where the two already agree.
    """
    if structurally_corroborated(cot_answer, symbolic_answer, tau):
        return symbolic_answer
    return cot_answer
