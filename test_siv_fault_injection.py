"""
Controlled fault-injection evaluation of SIV Layer 2 (Fault Localization).

[v12.2] Motivation: real-world MAS-SHT runs almost never exercise Layer 2 on
a genuine multi-variable blueprint (1/150 in the n=150 Kaggle pilot), so its
per-variable localization claim cannot be validated from field data alone.
This suite validates the MECHANISM directly and exhaustively: construct a
correct blueprint, corrupt exactly ONE given at a time (holding the rest at
their true values), and measure whether SIV's per-given reconstruction
isolates that one variable.

Finding: it does not, in the general case. When a blueprint's equations
collapse to a single scalar expression `answer = f(g1, ..., gn)` (which is
what `_build_symbolic_chain` always produces, since the schema requires
every equation chain to terminate in one `answer` assignment), holding all
OTHER givens fixed at their declared values and inverting for one given at a
time will show a mismatch for EVERY used given whenever the forward audit
fails -- not just the one that was actually corrupted. This is a structural
property of one-at-a-time algebraic inversion against a single collapsed
constraint, not a bug: perturbing any one input generically requires a
compensating change in every other input's "reconstructed" value to explain
the same observed discrepancy.

We therefore measure two SEPARATE things Layer 2 can honestly be evaluated
on:
  - EXCLUSION (does it correctly separate declared-but-unused/distractor
    givens from the givens that actually feed the equations)? This is
    Layer 2's genuinely working, validated capability.
  - ISOLATION (does failed_givens uniquely name the ONE corrupted variable,
    among the USED ones)? We measure recall (is the true fault always in the
    reported set?) and precision (what fraction of the reported set is the
    true fault?) across blueprints with 1..6 used givens, to quantify exactly
    how isolation degrades as the number of chain-connected variables grows.
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
sys.path.insert(0, '.')
from siv_module import SymbolicInverseVerifier


# Each case: (name, givens (name -> true value), equations, distractor_names)
# `equations` reference every given except those listed in distractor_names,
# which are deliberately declared but never used (to test exclusion).
CASES = [
    ("2-given additive", {"a": 10, "b": 20}, ["answer = givens['a'] + givens['b']"], []),
    ("2-given multiplicative", {"a": 4, "b": 5}, ["answer = givens['a'] * givens['b']"], []),
    ("3-given additive", {"a": 10, "b": 20, "c": 30}, ["answer = givens['a'] + givens['b'] + givens['c']"], []),
    ("3-given mixed (+/-)", {"a": 10, "b": 20, "c": 30}, ["answer = givens['a'] + givens['b'] - givens['c']"], []),
    ("3-given w/ 1 distractor", {"a": 10, "b": 20, "unused": 999}, ["answer = givens['a'] + givens['b']"], ["unused"]),
    ("multi-step subtract", {"a": 10, "b": 20, "c": 5}, [
        "subtotal = givens['a'] + givens['b']", "answer = subtotal - givens['c']"], []),
    ("multi-step w/ 2 distractors", {"a": 10, "b": 20, "c": 5, "d1": 7, "d2": 3}, [
        "subtotal = givens['a'] + givens['b']", "answer = subtotal - givens['c']"], ["d1", "d2"]),
    ("rate problem (workers*hours*rate)", {"workers": 8, "hours": 5, "rate": 12}, [
        "total_hours = givens['workers'] * givens['hours']", "answer = total_hours * givens['rate']"], []),
    ("profit chain (4 givens)", {"price": 50, "cost": 30, "quantity": 100, "tax_rate": 0.1}, [
        "profit_per_unit = givens['price'] - givens['cost']",
        "gross_profit = profit_per_unit * givens['quantity']",
        "tax = gross_profit * givens['tax_rate']",
        "answer = gross_profit - tax"], []),
    ("division", {"total": 100, "boxes": 4}, ["answer = givens['total'] / givens['boxes']"], []),
    ("5-given chain", {"a": 10, "b": 5, "c": 3, "d": 20, "e": 2}, [
        "step1 = givens['a'] * givens['b']", "step2 = step1 - givens['c']",
        "answer = step2 + givens['d'] - givens['e']"], []),
    ("6-given chain w/ 1 distractor", {"a": 10, "b": 5, "c": 3, "d": 20, "e": 2, "junk": 42}, [
        "step1 = givens['a'] * givens['b']", "step2 = step1 - givens['c']",
        "answer = step2 + givens['d'] - givens['e']"], ["junk"]),
]


def _true_answer(equations, givens):
    """Evaluate the equation chain in plain Python to get the ground-truth answer."""
    env = {"givens": dict(givens)}
    for eq in equations:
        lhs, rhs = eq.split("=", 1)
        env[lhs.strip()] = eval(rhs, {"__builtins__": {}}, env)  # noqa: S307 -- test-only, trusted input
    return env["answer"]


def run_fault_injection():
    print("=" * 70)
    print("SIV LAYER 2 — CONTROLLED FAULT-INJECTION EVALUATION")
    print("=" * 70)

    exclusion_total = 0
    exclusion_correct = 0
    recall_hits = 0
    recall_total = 0
    precisions = []
    isolation_hits = 0
    isolation_total = 0
    by_n_used = {}  # n_used -> list of (precision, isolated bool)

    for name, true_givens, equations, distractors in CASES:
        used = [g for g in true_givens if g not in distractors]
        true_answer = _true_answer(equations, true_givens)

        # --- Exclusion check: correct blueprint, no corruption ---
        bp_correct = {"givens": dict(true_givens), "equations": equations}
        r0 = SymbolicInverseVerifier.verify(bp_correct, true_answer)
        exclusion_total += 1
        if set(r0.unused_givens) == set(distractors):
            exclusion_correct += 1
        else:
            print(f"  [EXCLUSION MISS] {name}: expected unused={distractors}, got={r0.unused_givens}")

        # --- Isolation check: corrupt exactly one USED given at a time ---
        for corrupt_key in used:
            corrupted_givens = dict(true_givens)
            # Corrupt by a deterministic, nonzero relative offset.
            corrupted_givens[corrupt_key] = true_givens[corrupt_key] + max(1.0, abs(true_givens[corrupt_key]) * 0.3)
            bp = {"givens": corrupted_givens, "equations": equations}
            result = SymbolicInverseVerifier.verify(bp, true_answer)

            failed = set(result.failed_givens)
            recall_total += 1
            if corrupt_key in failed:
                recall_hits += 1
            if failed:
                precision = (1.0 if corrupt_key in failed else 0.0) / len(failed)
                precisions.append(precision)
            isolation_total += 1
            isolated = failed == {corrupt_key}
            if isolated:
                isolation_hits += 1
            by_n_used.setdefault(len(used), []).append((
                (1.0 if corrupt_key in failed else 0.0) / len(failed) if failed else 0.0,
                isolated,
            ))

    print(f"\nExclusion (declared-but-unused correctly identified): "
          f"{exclusion_correct}/{exclusion_total} = {exclusion_correct/exclusion_total:.1%}")
    print(f"Recall (true fault always present in failed_givens):    "
          f"{recall_hits}/{recall_total} = {recall_hits/recall_total:.1%}")
    print(f"Mean precision (1 / |failed_givens| when fault present): "
          f"{sum(precisions)/len(precisions):.1%}")
    print(f"Exact isolation (failed_givens == {{the one corrupted var}}): "
          f"{isolation_hits}/{isolation_total} = {isolation_hits/isolation_total:.1%}")

    print("\nBy number of used (chain-connected) givens:")
    for n_used in sorted(by_n_used):
        entries = by_n_used[n_used]
        mean_prec = sum(p for p, _ in entries) / len(entries)
        iso_rate = sum(1 for _, i in entries if i) / len(entries)
        print(f"  n_used={n_used}: n_cases={len(entries):2d}  mean_precision={mean_prec:.1%}  isolation_rate={iso_rate:.1%}")

    assert exclusion_correct == exclusion_total, "Exclusion (distractor detection) should be perfect"
    assert recall_hits == recall_total, "The true fault should always appear in failed_givens (recall)"
    # Isolation is expected to be perfect ONLY when exactly one given is used;
    # this assertion documents (not merely observes) the structural limitation.
    single_given_cases = by_n_used.get(1, [])
    multi_given_cases = [e for n, es in by_n_used.items() if n >= 2 for e in es]
    if multi_given_cases:
        multi_isolation_rate = sum(1 for _, i in multi_given_cases if i) / len(multi_given_cases)
        assert multi_isolation_rate == 0.0, (
            "Expected exact isolation to fail for every >=2-used-given case "
            f"(structural limitation), got {multi_isolation_rate:.1%}"
        )
    print("\n  ASSERTIONS PASSED (exclusion=100%, recall=100%, "
          "multi-variable isolation=0% as predicted)")


if __name__ == "__main__":
    run_fault_injection()
