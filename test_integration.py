"""
Integration tests for SIV + downstream pipeline contract
=========================================================

Tests SIV's integration points with the rest of MAS-SHT WITHOUT making
real LLM calls. These verify the contract between siv_module.py and the
parts of Mas_solver.py that consume SIVResult:

    Scenario 1 — Verified case:    SIV passes → gate would skip SHT
    Scenario 2 — Inconsistent:     SIV detects wrong answer → localization
    Scenario 3 — Non-invertible:   SIV neutral → gate falls through
    Scenario 4 — CSV columns:      SIV output produces the expected fields

Run:
    python test_integration.py            (script style — like test_siv.py)
"""

import sys

# Make siv_module importable from the project root.
sys.path.insert(0, '.')

from siv_module import SymbolicInverseVerifier, SIVResult


def _print_siv(label, result):
    print(f"  {label}: verified={result.verified} "
          f"conf={result.confidence:.2f} "
          f"matched={result.givens_matched}/{result.givens_total} "
          f"invertible={result.invertible}")
    if result.failed_givens:
        print(f"  Failed givens: {result.failed_givens}")


# =====================================================================
# Scenario 1 — Correct answer; SIV verified; gate would skip SHT
# =====================================================================
def scenario_1_verified_skip_sht():
    print("=" * 60)
    print("Scenario 1: Correct answer, SIV verified → would SKIP SHT")
    print("-" * 60)

    blueprint = {
        "givens": {"initial": 10, "eaten": 3, "bought": 5},
        "equations": [
            "remaining = givens['initial'] - givens['eaten']",
            "answer = remaining + givens['bought']",
        ],
    }
    correct_answer = 12.0

    result = SymbolicInverseVerifier.verify(blueprint, correct_answer)
    _print_siv("SIV", result)

    assert result.verified, "SIV should verify a correct answer"
    assert result.confidence >= 0.90, f"Confidence too low: {result.confidence}"
    assert not result.failed_givens, "No givens should fail"

    # The Mas_solver _confidence_gate consumes (verified, confidence) and would
    # short-circuit to "siv_execution_consistent" — we mirror its predicate here:
    would_skip_sht = result.verified and result.confidence >= 0.90
    assert would_skip_sht, "Gate predicate should allow skipping SHT"
    print("  ✓ PASSED — Gate would skip SHT (saves ~4 LLM calls)\n")


# =====================================================================
# Scenario 2 — Wrong answer; SIV inconsistent; localization report works
# =====================================================================
def scenario_2_inconsistency_triggers_targeted_sht():
    print("=" * 60)
    print("Scenario 2: Wrong answer → SIV inconsistency → targeted SHT")
    print("-" * 60)

    blueprint = {
        "givens": {"base_salary": 5000, "bonus_rate": 0.15, "months": 12},
        "equations": [
            "monthly_bonus = givens['base_salary'] * givens['bonus_rate']",
            "annual_bonus = monthly_bonus * givens['months']",
            "answer = givens['base_salary'] * givens['months'] + annual_bonus",
        ],
    }
    correct_answer = 5000 * 12 + 5000 * 0.15 * 12  # 69000
    wrong_answer = 80000.0  # Off by 11000

    result = SymbolicInverseVerifier.verify(blueprint, wrong_answer)
    _print_siv("SIV", result)

    # The gate consumes (invertible AND not verified) → "siv_inconsistency"
    assert result.invertible, "Chain should be invertible"
    assert not result.verified, "Wrong answer must not be verified"
    assert len(result.failed_givens) > 0, "Should localize at least one failing given"

    # The Critic in Mas_solver receives this for targeted repair:
    report = SymbolicInverseVerifier.get_error_localization_report(result)
    assert isinstance(report, str) and len(report) > 0, "Localization report must be non-empty"
    print(f"  Localization report (truncated): {report[:200]}")
    print("  ✓ PASSED — Critic would receive targeted error info\n")


# =====================================================================
# Scenario 3 — Non-invertible chain; SIV neutral; gate falls through
# =====================================================================
def scenario_3_non_invertible_falls_through():
    print("=" * 60)
    print("Scenario 3: Non-invertible chain → SIV neutral → gate fall-through")
    print("-" * 60)

    # No equations means SIV cannot verify anything.
    blueprint = {
        "givens": {"x": 5},
        "equations": [],  # Empty chain — nothing to invert
    }
    result = SymbolicInverseVerifier.verify(blueprint, 42.0)
    _print_siv("SIV", result)

    # The gate predicate: SIV provides no positive signal, so it should NOT
    # be the one short-circuiting the decision. Other criteria
    # (baseline disagreement, programmer failure, etc.) take over.
    would_skip_sht = result.verified and result.confidence >= 0.90
    assert not would_skip_sht, "SIV must not allow skip when chain isn't invertible"
    print("  ✓ PASSED — Gate falls through to standard criteria\n")


# =====================================================================
# Scenario 4 — CSV output schema
# =====================================================================
def scenario_4_csv_output_schema():
    print("=" * 60)
    print("Scenario 4: SIV produces all fields the CSV exporter expects")
    print("-" * 60)

    blueprint = {
        "givens": {"a": 7, "b": 3},
        "equations": ["answer = givens['a'] + givens['b']"],
    }
    result = SymbolicInverseVerifier.verify(blueprint, 10.0)

    # Mirror what Mas_solver.QualityAwarePipeline.run() reads off the SIV result
    # for its CSV columns (the canonical fields, not the v10.0+ extensions
    # that are gated by attribute presence).
    required_attrs = [
        "verified", "confidence", "givens_matched", "givens_total",
        "invertible", "failed_givens", "trace",
    ]
    missing = [a for a in required_attrs if not hasattr(result, a)]
    assert not missing, f"SIVResult is missing required attrs: {missing}"

    # Spot-check that the values are usable downstream
    assert isinstance(result.verified, bool)
    assert isinstance(result.confidence, float)
    assert isinstance(result.givens_matched, int)
    assert isinstance(result.givens_total, int)
    assert isinstance(result.invertible, bool)
    assert isinstance(result.failed_givens, list)
    assert isinstance(result.trace, str)

    print(f"  Schema fields: {required_attrs}")
    print(f"  Sample row: verified={result.verified} "
          f"conf={result.confidence:.2f} "
          f"matched={result.givens_matched}/{result.givens_total}")
    print("  ✓ PASSED — CSV exporter can serialize this row\n")


# =====================================================================
# Entrypoint
# =====================================================================
if __name__ == "__main__":
    scenarios = [
        scenario_1_verified_skip_sht,
        scenario_2_inconsistency_triggers_targeted_sht,
        scenario_3_non_invertible_falls_through,
        scenario_4_csv_output_schema,
    ]
    for fn in scenarios:
        try:
            fn()
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}\n")
            raise

    print("=" * 60)
    print("  ALL INTEGRATION TESTS PASSED ✓")
    print("=" * 60)
    print("\nSIV integration points verified:")
    print("  1. Confidence gate: SIV verified → skip SHT (save API calls)")
    print("  2. Error localization: SIV failed → targeted Critic report")
    print("  3. Neutral fallback: non-invertible → standard gate criteria")
    print("  4. CSV output: SIVResult exposes all schema fields")
