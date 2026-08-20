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


def part5_text_snapping():
    """[v15.0] Operands must be numbers the problem actually contains."""
    print("\n" + "=" * 70)
    print("PART 5 - [v15.0] snapping givens to numbers present in the text")
    print("=" * 70)

    # gsm8k_test_457, verbatim: the text says 50 and 30, the model wrote 55/31.
    text457 = ("While playing with her friends in their school playground, "
               "Katelyn saw 50 fairies flying above the nearby forest. After "
               "about twenty minutes, one of her friends saw half as many "
               "fairies as Katelyn saw come from the east and join the fairies "
               "that were there. Ten minutes later, 30 fairies flew away. How "
               "many fairies are remaining?")
    bp, fixes = repair_blueprint(
        {"givens": {"initial_fairies": 55, "fairies_flew_away": 31},
         "equations": ["joined = givens['initial_fairies'] / 2",
                       "total = givens['initial_fairies'] + joined",
                       "answer = total - givens['fairies_flew_away']"]},
        problem_text=text457)
    check("misread operands snapped back to the text",
          bp["givens"]["initial_fairies"] == 50.0
          and bp["givens"]["fairies_flew_away"] == 30.0,
          f"got {bp['givens']}")
    r = SIV.verify(bp, 45.0)
    check("and the chain now evaluates to the gold answer",
          r.blueprint_answer is not None and abs(r.blueprint_answer - 45.0) < 1e-6,
          f"got {r.blueprint_answer}")

    # svamp_test_23: a dropped LEADING digit, caught by the suffix rule.
    bp, fixes = repair_blueprint(
        {"givens": {"girls": 69, "boys": 36},
         "equations": ["answer = givens['girls'] - givens['boys']"]},
        problem_text="In a school there are 569 girls and 236 boys. "
                     "How many more girls than boys does the school have?")
    check("dropped leading digit restored",
          bp["givens"]["girls"] == 569.0 and bp["givens"]["boys"] == 236.0,
          f"got {bp['givens']}")

    # Values genuinely in the text must never move.
    bp, fixes = repair_blueprint(
        {"givens": {"a": 40, "b": 41},
         "equations": ["answer = givens['a'] + givens['b']"]},
        problem_text="He had 40 apples and 41 pears.")
    check("grounded values are left alone", bp["givens"] == {"a": 40, "b": 41},
          f"got {bp['givens']}")

    # Ambiguity must abstain: 45 is one digit from BOTH 40 and 41.
    bp, fixes = repair_blueprint(
        {"givens": {"x": 45},
         "equations": ["answer = givens['x'] * 2"]},
        problem_text="He had 40 apples, 41 pears and 46 plums.")
    check("ambiguous slip is refused, not guessed",
          bp["givens"]["x"] == 45, f"got {bp['givens']}")

    # Structural constants (percent divisor, halves) are not text operands.
    bp, fixes = repair_blueprint(
        {"givens": {"pct": 20, "half": 2},
         "equations": ["answer = givens['pct'] / 100 * givens['half']"]},
        problem_text="Enrolment rose by 20 percent.")
    check("structural constants are not snapped",
          bp["givens"]["half"] == 2, f"got {bp['givens']}")

    # No text supplied -> the pass is inert (backwards compatible).
    bp, fixes = repair_blueprint(
        {"givens": {"x": 55}, "equations": ["answer = givens['x']"]})
    check("no problem text -> snapping is a no-op",
          bp["givens"]["x"] == 55 and not any("problem text" in f for f in fixes),
          f"got {bp['givens']}, fixes={fixes}")


def part2_refusals():
    print("\n" + "=" * 70)
    print("PART 2 — defects the repairer must NOT guess at")
    print("=" * 70)

    # gsm-hard_1206 — 'snails_aquariumt' is equidistant from aquarium1 and
    # aquarium2, so string distance alone IS a coin flip and v14.2 refused it.
    # [v14.8] Context settles it without guessing: this equation already reads
    # aquarium1, so resolving to aquarium1 would produce 'x - x', a degenerate
    # reading no blueprint intends. aquarium2 is the only sibling that leaves
    # the expression meaningful. The refusal boundary therefore moved from
    # "string distance is ambiguous" to "string distance is ambiguous AND
    # context does not disambiguate" — see the next case, which still refuses.
    #
    # NOTE: this recovers EVALUABILITY, not correctness. On the real
    # gsm-hard_1206 the repaired chain evaluates to 13.5 against a gold of 7.0.
    # A candidate SIV can then reject beats no candidate at all.
    bp, fixes = case("one-char corruption, context disambiguates (gsm-hard_1206)",
                     {"givens": {"snails_aquarium1": 10.0, "snails_aquarium2": 20.0},
                      "equations": ["answer = givens['snails_aquariumt'] - givens['snails_aquarium1']"]},
                     want_fixed=True, expect_value=10.0)
    check("sibling exclusion was the rule that fired",
          any("unused sibling" in f for f in fixes), f"fixes={fixes}")

    # The same corruption with NO sibling already referenced: exclusion cannot
    # fire, both readings stay live, and the repairer must still keep its hands
    # off. This is what v14.2's refusal was really protecting.
    bp2, fixes2 = case("one-char corruption, nothing disambiguates it",
                       {"givens": {"snails_aquarium1": 10.0, "snails_aquarium2": 20.0},
                        "equations": ["answer = givens['snails_aquariumt'] * 2"]},
                       want_fixed=False)
    check("undisambiguated case rewrote nothing",
          not any("unused sibling" in f for f in fixes2), f"fixes={fixes2}")

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


def part6_v152_classes():
    """[v15.2] Expression givens, long-operand snapping, misspelt accessor."""
    print("\n" + "=" * 70)
    print("PART 6 - [v15.2] repair classes the grounding pass could not reach")
    print("=" * 70)

    # gsm8k_test_457 (v3_inventory route, verbatim shape): a given declared as
    # a FORMULA over another given, whose operand is itself misread. Snapping
    # must fire first (55 -> 50), the expression second (50 / 2 = 25).
    text457 = ("Katelyn saw 50 fairies flying above the nearby forest. One of "
               "her friends saw half as many fairies as Katelyn saw come from "
               "the east. Ten minutes later, 30 fairies flew away. How many "
               "fairies are remaining?")
    bp, fixes = repair_blueprint(
        {"givens": {"initial_fairies": 55,
                    "fairies_from_east": "initial_fairies / 2",
                    "fairies_flew_ay": 31},
         "equations": ["total_fairies = givens['initial_fairies'] + givens['fairies_from_east']",
                       "answer = total_fairies - givens['fairies_flew_ay']"]},
        problem_text=text457)
    check("expression given evaluated AFTER its operand was snapped",
          bp["givens"].get("fairies_from_east") == 25.0, f"got {bp['givens']}")
    r = SIV.verify(bp, 45.0)
    check("snap + expression compose to the gold answer",
          r.blueprint_answer is not None and abs(r.blueprint_answer - 45.0) < 1e-6,
          f"got {r.blueprint_answer}")

    # gsm8k_test_501, verbatim shape: fraction strings as given values.
    bp, fixes = repair_blueprint(
        {"givens": {"skein_yards": 364, "mariah_used": "1/4", "grandma_used": "1/2"},
         "equations": ["mariah_yards = givens['skein_yards'] * givens['mariah_used']",
                       "grandma_yards = givens['skein_yards'] * givens['grandma_used']",
                       "answer = mariah_yards + grandma_yards"]},
        problem_text="Mariah used 1/4 of a skein. Her grandma used 1/2 of a "
                     "skein. There are 364 yards in a skein.")
    r = SIV.verify(bp, 273.0)
    check("fraction-string givens evaluate ('1/4' -> 0.25)",
          r.blueprint_answer is not None and abs(r.blueprint_answer - 273.0) < 1e-6,
          f"got {r.blueprint_answer}, givens={bp['givens']}")

    # A chain: one expression given reading another expression given.
    bp, fixes = repair_blueprint(
        {"givens": {"base": 10, "half": "base / 2", "quarter": "half / 2"},
         "equations": ["answer = givens['base'] + givens['half'] + givens['quarter']"]})
    r = SIV.verify(bp, 17.5)
    check("chained expression givens resolve iteratively",
          r.blueprint_answer is not None and abs(r.blueprint_answer - 17.5) < 1e-6,
          f"got {r.blueprint_answer}, givens={bp['givens']}")

    # gsm-hard_865 (v1 route, verbatim shape): a 7-digit operand with TWO
    # mangled digits. The one-slip rule refuses; the long-operand rule snaps
    # because the text contains exactly one number of that magnitude.
    bp, fixes = repair_blueprint(
        {"givens": {"total_shells": 9371084, "pct_alphas": 0.4, "pct_finders": 0.6},
         "equations": ["by_alphas = givens['pct_alphas'] * givens['total_shells']",
                       "remaining = givens['total_shells'] - by_alphas",
                       "by_finders = givens['pct_finders'] * remaining",
                       "answer = remaining - by_finders"]},
        problem_text="Twenty tourists discovered 9370284 shells. Team Alphas "
                     "found 40% of the shells, and team The finders found 60% "
                     "of the remaining shells.")
    check("two-digit mangling of a long operand snapped",
          bp["givens"]["total_shells"] == 9370284.0, f"got {bp['givens']}")
    r = SIV.verify(bp, 2248868.16)
    check("and the chain reaches the gold answer",
          r.blueprint_answer is not None and abs(r.blueprint_answer - 2248868.16) < 1e-2,
          f"got {r.blueprint_answer}")

    # TWO long candidates -> abstain, exactly like the short-number rule.
    bp, fixes = repair_blueprint(
        {"givens": {"x": 9371084},
         "equations": ["answer = givens['x'] * 2"]},
        problem_text="There were 9370284 shells on one beach and 9372184 on another.")
    check("two long candidates within reach: refused",
          bp["givens"]["x"] == 9371084, f"got {bp['givens']}")

    # Short numbers must NOT gain the two-edit privilege: 47 is two edits from
    # 12 in a text with only 12 -- snapping that would be pure invention.
    bp, fixes = repair_blueprint(
        {"givens": {"x": 47}, "equations": ["answer = givens['x'] * 2"]},
        problem_text="She bought 12 eggs.")
    check("short numbers keep the strict one-slip rule",
          bp["givens"]["x"] == 47, f"got {bp['givens']}")

    # gsm-hard_13 (v1 route, verbatim): the ACCESSOR is misspelt, not the key.
    bp, fixes = repair_blueprint(
        {"givens": {"spilled_orange_liters": 3.0, "initial_orange_liters": 10.0},
         "equations": ["answer = givens['initial_orange_liters'] - givvens['spilled_orange_liters']"]})
    r = SIV.verify(bp, 7.0)
    check("misspelt accessor givvens[...] respelt",
          r.blueprint_answer is not None and abs(r.blueprint_answer - 7.0) < 1e-6,
          f"got {r.blueprint_answer}, fixes={fixes}")

    # Refusals: values that need SEMANTICS still flow to the LLM repair path.
    bp, fixes = repair_blueprint(
        {"givens": {"cost_per_hour": 77.0, "start": "5 pm", "end": "10 am next morning"},
         "equations": ["hours = givens['end'] - givens['start']",
                       "answer = hours * givens['cost_per_hour']"]})
    check("'5 pm' is not an expression: refused",
          bp["givens"].get("start") == "5 pm", f"got {bp['givens']}")
    bp, fixes = repair_blueprint(
        {"givens": {"total": 100.0, "share": "total / n_people"},
         "equations": ["answer = givens['share']"]})
    check("expression with an unresolvable name: refused",
          bp["givens"].get("share") == "total / n_people", f"got {bp['givens']}")


if __name__ == "__main__":
    print("=" * 70)
    print("v14.2 — DETERMINISTIC BLUEPRINT REPAIR (offline, no models)")
    print("=" * 70)
    part1_real_defects()
    part2_refusals()
    part3_no_false_repairs()
    part4_symbol_identity_bug()
    part5_text_snapping()
    part6_v152_classes()
    print("\n" + "=" * 70)
    if failures:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)
    print("  ALL CHECKS PASSED")
    print("=" * 70)
