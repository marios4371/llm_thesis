"""
Test Symbolic Inverse Verification (SIV) Module
Verifies the core novel contribution works correctly.
"""

import sys
sys.path.insert(0, '/home/claude')
from siv_module import SymbolicInverseVerifier, SIVResult

def test_1_simple_correct():
    """Jane has 10 apples, eats 3, buys 5. Answer = 12."""
    print("=" * 60)
    print("TEST 1: Simple correct solution (should VERIFY)")
    
    blueprint = {
        "givens": {"initial_apples": 10, "eaten_apples": 3, "bought_apples": 5},
        "equations": [
            "remaining = givens['initial_apples'] - givens['eaten_apples']",
            "answer = remaining + givens['bought_apples']"
        ]
    }
    
    result = SymbolicInverseVerifier.verify(blueprint, 12.0)
    
    print(f"  Verified: {result.verified}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Matched: {result.givens_matched}/{result.givens_total}")
    print(f"  Failed: {result.failed_givens}")
    print(f"  Trace:\n{result.trace}")
    assert result.verified, "Should be verified!"
    assert result.confidence >= 0.95, f"Confidence should be >= 0.95, got {result.confidence}"
    print("  ✓ PASSED\n")


def test_2_wrong_answer():
    """Same problem but wrong answer (15 instead of 12). Should FAIL."""
    print("=" * 60)
    print("TEST 2: Wrong answer (should FAIL)")
    
    blueprint = {
        "givens": {"initial_apples": 10, "eaten_apples": 3, "bought_apples": 5},
        "equations": [
            "remaining = givens['initial_apples'] - givens['eaten_apples']",
            "answer = remaining + givens['bought_apples']"
        ]
    }
    
    result = SymbolicInverseVerifier.verify(blueprint, 15.0)
    
    print(f"  Verified: {result.verified}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Matched: {result.givens_matched}/{result.givens_total}")
    print(f"  Failed: {result.failed_givens}")
    print(f"  Trace:\n{result.trace}")
    assert not result.verified, "Should NOT be verified with wrong answer!"
    assert len(result.failed_givens) > 0, "Should identify failed givens"
    print("  ✓ PASSED\n")


def test_3_multiplication():
    """Rate problem: 8 workers × 5 hours × $12/hour = $480."""
    print("=" * 60)
    print("TEST 3: Multiplication chain (correct)")
    
    blueprint = {
        "givens": {"workers": 8, "hours": 5, "hourly_rate": 12},
        "equations": [
            "total_hours = givens['workers'] * givens['hours']",
            "answer = total_hours * givens['hourly_rate']"
        ]
    }
    
    result = SymbolicInverseVerifier.verify(blueprint, 480.0)
    
    print(f"  Verified: {result.verified}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Matched: {result.givens_matched}/{result.givens_total}")
    print(f"  Trace:\n{result.trace}")
    assert result.verified, "Should be verified!"
    print("  ✓ PASSED\n")


def test_4_wrong_given():
    """Blueprint has wrong given (says 8 workers but problem says 10). 
    Answer computed with wrong given. SIV should detect mismatch."""
    print("=" * 60)
    print("TEST 4: Wrong answer due to incorrect given extraction")
    
    # Blueprint incorrectly extracted 8 workers (actual is 10)
    blueprint = {
        "givens": {"workers": 8, "hours": 5, "hourly_rate": 12},
        "equations": [
            "total_hours = givens['workers'] * givens['hours']",
            "answer = total_hours * givens['hourly_rate']"
        ]
    }
    
    # If the correct answer were 600 (10*5*12), but we computed 480 (8*5*12)
    # SIV should verify 480 as consistent with the blueprint (it IS consistent)
    result_consistent = SymbolicInverseVerifier.verify(blueprint, 480.0)
    print(f"  With answer=480 (consistent with blueprint): verified={result_consistent.verified}")
    assert result_consistent.verified, "480 is consistent with blueprint givens"
    
    # If someone claims answer=600 with this blueprint, SIV should catch it
    result_inconsistent = SymbolicInverseVerifier.verify(blueprint, 600.0)
    print(f"  With answer=600 (INconsistent with blueprint): verified={result_inconsistent.verified}")
    print(f"  Failed givens: {result_inconsistent.failed_givens}")
    assert not result_inconsistent.verified, "600 is NOT consistent with these givens"
    print("  ✓ PASSED\n")


def test_5_division():
    """Division problem: 100 items / 4 boxes = 25 per box."""
    print("=" * 60)
    print("TEST 5: Division (correct)")
    
    blueprint = {
        "givens": {"total_items": 100, "num_boxes": 4},
        "equations": [
            "answer = givens['total_items'] / givens['num_boxes']"
        ]
    }
    
    result = SymbolicInverseVerifier.verify(blueprint, 25.0)
    
    print(f"  Verified: {result.verified}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Trace:\n{result.trace}")
    assert result.verified, "Should be verified!"
    print("  ✓ PASSED\n")


def test_6_complex_chain():
    """Multi-step: profit = (revenue - cost) * quantity - tax."""
    print("=" * 60)
    print("TEST 6: Complex 4-step chain (correct)")
    
    blueprint = {
        "givens": {"price": 50, "cost": 30, "quantity": 100, "tax_rate": 0.1},
        "equations": [
            "profit_per_unit = givens['price'] - givens['cost']",
            "gross_profit = profit_per_unit * givens['quantity']",
            "tax = gross_profit * givens['tax_rate']",
            "answer = gross_profit - tax"
        ]
    }
    
    # (50-30)*100 = 2000, tax = 200, answer = 1800
    result = SymbolicInverseVerifier.verify(blueprint, 1800.0)
    
    print(f"  Verified: {result.verified}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Matched: {result.givens_matched}/{result.givens_total}")
    print(f"  Trace:\n{result.trace}")
    assert result.verified, "Should be verified!"
    print("  ✓ PASSED\n")


def test_7_error_localization():
    """Test that SIV localizes the specific wrong given."""
    print("=" * 60)
    print("TEST 7: Error localization — which given is wrong?")
    
    blueprint = {
        "givens": {"price": 50, "cost": 30, "quantity": 100},
        "equations": [
            "profit_per_unit = givens['price'] - givens['cost']",
            "answer = profit_per_unit * givens['quantity']"
        ]
    }
    
    # Correct answer is 2000. Give wrong answer 2500.
    result = SymbolicInverseVerifier.verify(blueprint, 2500.0)
    
    print(f"  Verified: {result.verified}")
    print(f"  Failed givens: {result.failed_givens}")
    
    report = SymbolicInverseVerifier.get_error_localization_report(result)
    print(f"  Error report:\n{report}")
    
    assert not result.verified
    assert len(result.failed_givens) > 0, "Should identify at least one inconsistent given"
    print("  ✓ PASSED\n")


def test_8_no_equations():
    """Edge case: empty blueprint."""
    print("=" * 60)
    print("TEST 8: Empty blueprint (edge case)")
    
    blueprint = {"givens": {}, "equations": []}
    result = SymbolicInverseVerifier.verify(blueprint, 42.0)
    
    print(f"  Verified: {result.verified}")
    print(f"  Confidence: {result.confidence}")
    assert not result.verified
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SYMBOLIC INVERSE VERIFICATION (SIV) — Unit Tests")
    print("=" * 60 + "\n")
    
    test_1_simple_correct()
    test_2_wrong_answer()
    test_3_multiplication()
    test_4_wrong_given()
    test_5_division()
    test_6_complex_chain()
    test_7_error_localization()
    test_8_no_equations()
    
    print("=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
