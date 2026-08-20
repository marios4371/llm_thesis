"""
Vote-independence tests for triage / evidence ranking
=====================================================

`_honest_votes` is the single place that decides how many INDEPENDENT
derivations back an answer. Both `_triage_candidates` (majority) and
`_evidence_score` (rank component 1) call it, so a pseudo-vote leaking
through here reaches every selection path.

Two families must collapse:

    {primary, blueprint_eval}  — [v13.0] the audit-passed CAS value IS the
                                 primary's own equations
    {alt_1, alt_2, ...}        — [v14.5] one Critic call, one parent
                                 blueprint, one line of reasoning

Regression guarded here: on mas_full_20260817 (v14.4) alt_1 == alt_2 on all
5 rows where a correction_of_* strategy won triage; those rows scored 0/5
against the discarded baseline's 4/5.

Run:
    python test_honest_votes.py            (script style — like test_siv.py)
"""

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, '.')

from Mas_solver import QualityEnhancedMultiAgentSolver, HypothesisResult

_votes = QualityEnhancedMultiAgentSolver._honest_votes

_FAILURES = []


def _cand(hid, answer=1.0):
    return HypothesisResult(
        hypothesis_id=hid,
        strategy_name=hid,
        blueprint={},
        code="",
        code_success=True,
        execution_output="",
        answer=str(answer),
        parsed_answer=answer,
        confidence=0.9,
        agent_response=None,
    )


def check(label, members, expected):
    got = _votes([_cand(m) for m in members])
    ok = got == expected
    print(f"  {'✓' if ok else '✗'} {label}: {members} -> {got} (expected {expected})")
    if not ok:
        _FAILURES.append(label)


print("=" * 70)
print("HONEST VOTE COUNTING")
print("=" * 70)

print("\n[1] Genuinely independent derivations each count")
check("baseline alone", ["baseline"], 1)
check("primary + baseline", ["primary", "baseline"], 2)
check("baseline + blueprint_eval (no primary)", ["baseline", "blueprint_eval"], 2)
check("primary + baseline + alt_1", ["primary", "baseline", "alt_1"], 3)

print("\n[2] [v13.0] {primary, blueprint_eval} is one derivation")
check("primary + blueprint_eval", ["primary", "blueprint_eval"], 1)
check("primary + blueprint_eval + baseline",
      ["primary", "blueprint_eval", "baseline"], 2)

print("\n[3] [v14.5] the alt_* family is one derivation")
check("alt_1 alone", ["alt_1"], 1)
check("alt_1 + alt_2", ["alt_1", "alt_2"], 1)
check("alt_1 + alt_2 + alt_3", ["alt_1", "alt_2", "alt_3"], 1)
check("alt_1 + alt_2 + baseline", ["alt_1", "alt_2", "baseline"], 2)

print("\n[3b] [v15.0] primary + ONE alt is genuine 2-vote corroboration")
check("primary + alt_1 (no baseline)", ["primary", "alt_1"], 2)
check("primary + alt_1 + alt_2 (alt pair still collapses)",
      ["primary", "alt_1", "alt_2"], 2)

print("\n[4] Both rules compose")
check("primary + blueprint_eval + alt_1 + alt_2",
      ["primary", "blueprint_eval", "alt_1", "alt_2"], 2)
check("full house",
      ["primary", "blueprint_eval", "alt_1", "alt_2", "baseline"], 3)

print("\n[5] The field regression: two agreeing alts must not out-vote the baseline")
alts = [_cand("alt_1", 8.0), _cand("alt_2", 8.0)]
base = [_cand("baseline", 14.0)]
av, bv = _votes(alts), _votes(base)
ok = not (av >= 2 and av > bv)
print(f"  {'✓' if ok else '✗'} svamp_test_217 shape: alts={av} vote(s) vs baseline={bv} "
      f"-> majority {'correctly BLOCKED' if ok else 'WRONGLY FIRES'}")
if not ok:
    _FAILURES.append("alt majority blocked")

print("\n" + "=" * 70)
if _FAILURES:
    print(f"FAILED ({len(_FAILURES)}): {_FAILURES}")
    sys.exit(1)
print("ALL PASSED")
