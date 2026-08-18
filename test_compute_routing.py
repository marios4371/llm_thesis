"""
Compute-routing tests ([v14.6])
===============================

`_route_compute` decides, using only signals that are already free, whether
generating Critic alternatives is worth one Critic call plus one Programmer
run per alternative.

    A  the free candidates already corroborate  -> skip, return the anchor
    B  no corroboration AND the audit FAILED    -> spend, this is the stratum
                                                   where SHT earns its keep
    C  no corroboration AND the audit PASSED    -> skip, the certificate marks
       (or never ran)                              an easy problem, not a
                                                   correct derivation

Corroboration is judged exactly as `_honest_votes` counts, so the
{primary, blueprint_eval} pair alone is NOT corroboration — only a pair that
includes the independent zero-shot baseline is.

Run:
    python test_compute_routing.py
"""

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, '.')

from Mas_solver import QualityEnhancedMultiAgentSolver, HypothesisResult
from siv_module import SIVResult

_FAILURES = []


class _Router(QualityEnhancedMultiAgentSolver):
    """Bare instance: _route_compute touches no state beyond _answers_match."""
    def __init__(self):
        pass


_r = _Router()


def _c(hid, answer):
    return HypothesisResult(
        hypothesis_id=hid, strategy_name=hid, blueprint={}, code=None,
        code_success=True, execution_output="", answer=str(answer),
        parsed_answer=answer, confidence=0.5, agent_response=None,
    )


def _siv(audit_passed):
    return SIVResult(execution_audit_passed=audit_passed, blueprint_answer=1.0,
                     verified=audit_passed, confidence=0.97 if audit_passed else 0.0)


def check(label, cands, siv, expected_tier):
    tier, reason = _r._route_compute(cands, siv)
    ok = tier == expected_tier
    print(f"  {'✓' if ok else '✗'} {label}: -> {tier} ({reason}), expected {expected_tier}")
    if not ok:
        _FAILURES.append(label)


print("=" * 70)
print("COMPUTE ROUTING")
print("=" * 70)

print("\n[A] Free corroboration -> skip, the answer is already settled")
check("primary agrees with baseline",
      [_c("primary", 42.0), _c("baseline", 42.0), _c("blueprint_eval", 7.0)],
      _siv(False), "A")
check("blueprint_eval agrees with baseline",
      [_c("primary", 7.0), _c("baseline", 42.0), _c("blueprint_eval", 42.0)],
      _siv(False), "A")
check("corroboration wins even when the audit failed",
      [_c("primary", 42.0), _c("baseline", 42.0)], _siv(False), "A")

print("\n[B] No corroboration + audit FAILED -> spend")
check("all three disagree, audit failed",
      [_c("primary", 1.0), _c("baseline", 2.0), _c("blueprint_eval", 3.0)],
      _siv(False), "B")
check("no baseline anchor at all -> nothing to fall back on",
      [_c("primary", 1.0), _c("blueprint_eval", 3.0)], _siv(True), "B")

print("\n[C] No corroboration + audit PASSED -> skip, certificate = easy problem")
check("all three disagree, audit passed",
      [_c("primary", 1.0), _c("baseline", 2.0), _c("blueprint_eval", 3.0)],
      _siv(True), "C")
check("no SIV verdict at all",
      [_c("primary", 1.0), _c("baseline", 2.0)], None, "C")

print("\n[!] The pseudo-vote must NOT count as corroboration")
check("primary == blueprint_eval, baseline dissents, audit passed",
      [_c("primary", 99.0), _c("baseline", 5.0), _c("blueprint_eval", 99.0)],
      _siv(True), "C")
check("primary == blueprint_eval, baseline dissents, audit failed",
      [_c("primary", 99.0), _c("baseline", 5.0), _c("blueprint_eval", 99.0)],
      _siv(False), "B")

print("\n[~] Unparsed candidates are ignored, not treated as agreement")
check("baseline unparsed -> no anchor -> B",
      [_c("primary", 1.0),
       HypothesisResult("baseline", "baseline", {}, None, True, "", "unknown",
                        None, 0.5, None)],
      _siv(True), "B")

print("\n" + "=" * 70)
if _FAILURES:
    print(f"FAILED ({len(_FAILURES)}): {_FAILURES}")
    sys.exit(1)
print("ALL PASSED")
