"""
Offline tests for the deterministic blueprint repairer (v14.2).

Every REAL case below is a verbatim blueprint from the v14.0 run
(mas_full_20260803.csv) — the first run that recorded blueprint contents, and
therefore the first evidence of what actually breaks. Of its 25 structurally
broken blueprints this repairer fixes 19 without an LLM.

The refusal cases matter as much as the fixes: a wrong fuzzy match turns a
loudly-broken blueprint into a silently-wrong one, which is worse than leaving
it broken. Those tests pin the repairer's willingness to give up.

Run:  python test_blueprint_repair.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blueprint_repair import repair_blueprint
from siv_module import SymbolicInverseVerifier as SIV

failures = []


def check(label, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(label)


def structurally_ok(bp):
    return not SIV.verify(bp, 0.0).has_structural_defect


def case(label, bp, want_fixed, expect_value=None, computed_answer=None):
    """Repair `bp`; assert whether the structural defect cleared."""
    fixed_bp, fixes = repair_blueprint(bp)
    ok = structurally_ok(fixed_bp)
    check(f"{label}: {'repaired' if want_fixed else 'correctly refused'}",
          ok == want_fixed,
          f"fixes={fixes[:2]}" if fixes else "no fixes applied")
    if want_fixed and expect_value is not None:
        r = SIV.verify(fixed_bp, computed_answer if computed_answer is not None else expect_value)
        check(f"{label}: evaluates to {expect_value}",
              r.blueprint_answer is not None and abs(r.blueprint_answer - expect_value) < 1e-6,
              f"got {r.blueprint_answer}")
    return fixed_bp, fixes


def part1_real_defects():
    print("\n" + "=" * 70)
    print("PART 1 — verbatim broken blueprints from the v14.0 run")
    print("=" * 70)

    # gsm-hard_334 — unbalanced subscripts, three of them
    case("unbalanced givens[ (gsm-hard_334)",
         {"givens": {"initial_fairies": 100.0, "fairies_flew_away": 20.0},
          "equations": ["fairies_from_east = givens['initial_fairies' / 2",
                        "total_fairies = givens['initial_fairies' + fairies_from_east",
                        "answer = total_fairies - givens['fairies_flew_away'"]},
         want_fixed=True, expect_value=130.0)

    # gsm-hard_740 — near-miss key: peach_bought vs declared peaches_bought
    case("near-miss key (gsm-hard_740)",
         {"givens": {"peach_cost": 2.0, "peaches_bought": 3.0},
          "equations": ["answer = givens['peach_cost'] * givens['peach_bought']"]},
         want_fixed=True, expect_value=6.0)

    # gsm-hard_476 — the DECLARED name is the typo'd one; repair must follow it
    case("near-miss where the declaration is misspelled (gsm-hard_476)",
         {"givens": {"plane_cost": 10.0, "honthly_hanger_rental": 5.0},
          "equations": ["monthly_fuel_cost = 2 * givens['monthly_hanger_rental']",
                        "answer = givens['plane_cost'] + monthly_fuel_cost"]},
         want_fixed=True, expect_value=20.0)

    # gsm8k_test_1035 — a computed variable read back through givens[...]
    case("computed var via givens[] (gsm8k_test_1035)",
         {"givens": {"driveway_width": 10.0, "distance_between_bottles": 2.0,
                     "time_to_move_between_bottles": 3.0},
          "equations": ["number_of_intervals = (givens['driveway_width'] / givens['distance_between_bottles']) - 1",
                        "answer = givens['time_to_move_between_bottles'] * givens['number_of_intervals']"]},
         want_fixed=True, expect_value=12.0)

    # gsm8k_test_714 — bare undefined name, near-miss of an earlier result
    case("bare typo of a computed var (gsm8k_test_714)",
         {"givens": {"cost_to_go_picking": 5.0, "cost_per_pound": 2.0, "pounds_picked": 3.0,
                     "store_price_per_pound": 6.0},
          "equations": ["total_cost_picking = givens['cost_to_go_picking'] + (givens['pounds_picked']*givens['cost_per_pound'])",
                        "total_cost_store = givens['pounds_picked']*givens['store_price_per_pound']",
                        "answer = total_cost_store - total_cost_pick"]},
         want_fixed=True, expect_value=7.0)

    # gsm8k_test_271 — numeric value stored as a string (the coercible case)
    case("numeric-string given",
         {"givens": {"houses": "3", "windows": 4},
          "equations": ["answer = givens['houses'] * givens['windows']"]},
         want_fixed=True, expect_value=12.0)


def part2_refusals():
    print("\n" + "=" * 70)
    print("PART 2 — defects the repairer must NOT guess at")
    print("=" * 70)

    # gsm-hard_1206 — 'snails_aquariumt' sits between aquarium1 and aquarium2.
    # Either choice is a coin flip dressed up as a repair.
    bp, fixes = case("ambiguous between two equally-close givens (gsm-hard_1206)",
                     {"givens": {"snails_aquarium1": 10.0, "snails_aquarium2": 20.0},
                      "equations": ["answer = givens['snails_aquariumt'] - givens['snails_aquarium1']"]},
                     want_fixed=False)
    check("ambiguous case rewrote nothing", not fixes, f"fixes={fixes}")

    # gsm8k_test_541 — the given is genuinely absent, not misspelled.
    case("genuinely missing given (gsm8k_test_541)",
         {"givens": {"monthly_target": 100.0},
          "equations": ["savings = 15 * givens['daily_savings_first_half']",
                        "answer = givens['monthly_target'] - savings"]},
         want_fixed=False)

    # gsm8k_test_936 — '5 pm' needs semantics, not string surgery.
    case("non-numeric value needing semantics (gsm8k_test_936)",
         {"givens": {"cost_per_hour": 77.0, "stay_start_time": "5 pm",
                     "stay_end_time": "10 am next morning"},
          "equations": ["hours = givens['stay_end_time'] - givens['stay_start_time']",
                        "answer = hours * givens['cost_per_hour']"]},
         want_fixed=False)

    # gsm8k_test_271's real value was 't' — garbage, not a number.
    case("garbage value is not coercible",
         {"givens": {"number_of_houses": "t", "bedrooms": 3.0},
          "equations": ["answer = givens['number_of_houses'] * givens['bedrooms']"]},
         want_fixed=False)


def part3_no_false_repairs():
    print("\n" + "=" * 70)
    print("PART 3 — healthy blueprints must pass through untouched")
    print("=" * 70)

    healthy = [
        ("simple chain", {"a": 10.0, "b": 3.0},
         ["t = givens['a'] - givens['b']", "answer = t + 5"]),
        ("bare given names", {"total_bumps": 180.0, "total_heads": 100.0},
         ["answer = total_bumps - total_heads"]),
        ("builtins", {"a": -5.0, "b": 9.0},
         ["m = max(abs(givens['a']), givens['b'])", "answer = m"]),
        ("unused distractor given", {"a": 5.0, "distractor": 99.0},
         ["answer = givens['a'] * 2"]),
        ("int division", {"n": 7.0, "p": 2.0},
         ["t = int(givens['n'] / givens['p'])", "answer = t * givens['p']"]),
        ("similar-but-distinct given names", {"cost_a": 3.0, "cost_b": 4.0},
         ["answer = givens['cost_a'] + givens['cost_b']"]),
    ]
    for name, g, eq in healthy:
        bp = {"givens": g, "equations": eq}
        out, fixes = repair_blueprint(bp)
        check(f"untouched: {name}", not fixes, f"unexpected fixes={fixes}")
        check(f"still evaluable: {name}", structurally_ok(out))


def part4_symbol_identity_bug():
    print("\n" + "=" * 70)
    print("PART 4 — [v14.2] bare given names are auditable at all")
    print("=" * 70)
    # Regression guard for a bug that predates v14: SIV built givens as
    # Symbol(name, real=True) while parse_expr produced Symbol(name). SymPy
    # treats those as different symbols, so .subs() matched nothing, the
    # expression kept free symbols, float() raised, and the row was recorded
    # blueprint_answer=None / invertible=False — indistinguishable from a
    # genuinely broken blueprint. Every bare-name blueprint was silently
    # unauditable.
    r = SIV.verify({"givens": {"total_bumps": 180.0, "total_heads": 100.0},
                    "equations": ["answer = total_bumps - total_heads"]}, 80.0)
    check("bare-name blueprint evaluates", r.blueprint_answer == 80.0,
          f"blueprint_answer={r.blueprint_answer}")
    check("bare-name blueprint passes audit", r.execution_audit_passed is True)
    check("bare-name blueprint reports no phantom defect", not r.has_structural_defect,
          f"undefined={r.undefined_symbols}")
    check("bare-name blueprint is invertible", r.invertible is True)

    # Mixed styles in one chain must also work.
    r = SIV.verify({"givens": {"a": 10.0, "b": 4.0},
                    "equations": ["t = a * givens['b']", "answer = t - a"]}, 30.0)
    check("mixed bare + subscript styles", r.execution_audit_passed is True,
          f"blueprint_answer={r.blueprint_answer}")


if __name__ == "__main__":
    print("=" * 70)
    print("v14.2 — DETERMINISTIC BLUEPRINT REPAIR (offline, no models)")
    print("=" * 70)
    part1_real_defects()
    part2_refusals()
    part3_no_false_repairs()
    part4_symbol_identity_bug()
    print("\n" + "=" * 70)
    if failures:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)
    print("  ALL CHECKS PASSED")
    print("=" * 70)
