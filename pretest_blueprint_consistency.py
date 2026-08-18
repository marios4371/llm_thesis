"""
Pre-test: are the Architect's blueprint errors DIVERSE enough for agreement
between independent blueprints to work as a correctness filter?

WHY THIS EXISTS
---------------
Measured on mas_full_20260817 (v14.4, n=150), the technique in isolation --
no zero-shot baseline anywhere in the candidate pool -- looks like this:

    primary (Programmer)   50.0%      blueprint_eval  20.7%
    alt_1 (Critic)         12.0%      alt_2 (Critic)   9.3%
    ORACLE over its own candidates    55.33%

A perfect selector over the current candidates is therefore worth 55.33%, i.e.
every possible improvement to SHT/arbitration is capped at +5.33pp. The binding
constraint is upstream, and it is sharp:

    blueprint correct -> the Programmer is right 96.8% of the time
    blueprint wrong   -> the Programmer is right 43.3% of the time
    blueprints correct: 31/150 (20.7%)
    transcription destroyed a correct blueprint: 1 problem

The Programmer is not the problem. Fitting those two conditionals gives

    primary_accuracy ~= 0.433 + 0.535 * blueprint_accuracy

so blueprint accuracy has to reach roughly 64% before the technique clears the
zero-shot CoT baseline on its own. It is at 20.7%.

No blueprint-only signal predicts correctness (measured on the same run:
unused_givens +2.7pp, needed-a-deterministic-fix -5.1pp, extracted_from_cot
+2.7pp, invertibility useless at n=1). The only signal that works is AGREEMENT
between independent derivations: blueprint == primary implies 71.4% correct,
blueprint != primary implies 2.3%.

THE HYPOTHESIS THIS SCRIPT TESTS
--------------------------------
Generate k blueprints per problem instead of 1. Evaluating each with the CAS is
free (zero LLM calls), so k blueprints yield k candidate answers at no
verification cost. The correct answer is ONE value while wrong answers should
be scattered, so "two independent blueprints landed on the same number" ought
to be a high-precision correctness filter rather than a mere majority vote.

That whole argument rests on ERROR DIVERSITY -- exactly the assumption that
just failed elsewhere in this system. The Critic's alt_1 and alt_2 agreed with
each other on all 5 rows where they won triage, and were wrong on all 5,
because they came from one call against one parent blueprint. If the
Mathematician's errors are correlated the same way, k=5 buys nothing and costs
4 extra calls per problem. This script decides that in ~30 minutes instead of
10 GPU-hours.

WHERE THE DIVERSITY COMES FROM
------------------------------
Not temperature. The local_hf backend decodes greedily on purpose (do_sample is
False -- v10.3 fixed an fp16 probability overflow that hard-crashed sampling),
and math_temp is 0.0, so re-calling with an identical prompt returns a
byte-identical blueprint. Diversity therefore has to come from the prompt, via
run_mathematician_analysis(strategy_hint=...). That is arguably the better
experiment anyway: these variants change the reasoning ROUTE rather than adding
token noise, and decorrelation is the whole point.

Variant 0 is the unmodified production prompt, so its column is directly
comparable to the blueprint accuracy of any full run.

WHAT IT MEASURES
----------------
  * per-variant blueprint accuracy (variant 0 = today's number)
  * whether the variants actually produced different blueprints at all -- if
    they did not, prompt variation failed and the rest of the report is void
  * error decorrelation: when two variants are both wrong, do they agree on the
    same wrong value? Low is good.
  * the ">=2 agree" rule: how often it fires (coverage) and how often it is
    right when it fires (precision) -- the two numbers that decide v15
  * projected blueprint accuracy for k = 1..K

USAGE
    python pretest_blueprint_consistency.py --n 25
    python pretest_blueprint_consistency.py --n 25 --preset qwen_math7b_mixed
"""

import argparse
import itertools
import json
import sys
import time
from collections import Counter

from Mas_solver import (
    QualityEnhancedMultiAgentSolver, UnifiedLLMClient, AgentRole,
    HETEROGENEOUS_PRESETS, EnhancedProblemManager,
    BLUEPRINT_STRATEGY_HINTS, _extract_last_number, SOLVER_VERSION,
)
from siv_module import SymbolicInverseVerifier as SIV


# Imported from Mas_solver so this pre-test and the production ensemble can
# never drift apart. Index 0 is the untouched production prompt.
VARIANTS = BLUEPRINT_STRATEGY_HINTS


def blueprint_value(bp, gold):
    """Score one blueprint with no LLM involvement. -> (status, value)."""
    if not bp or not bp.get("equations") or not bp.get("givens"):
        return "empty", None
    r = SIV.verify(bp, 0.0)
    if getattr(r, "has_structural_defect", False):
        return "structural", None
    if r.blueprint_answer is None:
        return "unevaluable", None
    v = float(r.blueprint_answer)
    if gold is None:
        return "wrong", v
    ok = abs(v - gold) < 1e-6 or abs(v - gold) <= 1e-6 * max(abs(gold), 1.0)
    return ("correct" if ok else "wrong"), v


def same(a, b):
    if a is None or b is None:
        return False
    return abs(a - b) <= max(1e-6, 1e-6 * max(abs(a), abs(b)))


def agreement_pick(values):
    """The '>=2 independent blueprints agree' rule.

    Returns (picked_value, support), or (None, 0) when nothing reaches two
    votes. Ordering is by support then first appearance, so it is deterministic.
    """
    groups = []
    for v in [x for x in values if x is not None]:
        for g in groups:
            if same(v, g[0]):
                g[1] += 1
                break
        else:
            groups.append([v, 1])
    if not groups:
        return None, 0
    groups.sort(key=lambda g: -g[1])
    return (groups[0][0], groups[0][1]) if groups[0][1] >= 2 else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--preset", default="qwen_math7b_mixed")
    ap.add_argument("--datasets", default="gsm8k_test,gsm-hard,svamp")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=len(VARIANTS),
                    help="variants to use, 2..%d" % len(VARIANTS))
    ap.add_argument("--out", default="pretest_blueprint_consistency.json")
    args = ap.parse_args()

    if args.preset not in HETEROGENEOUS_PRESETS:
        print("Unknown preset %r. Available: %s"
              % (args.preset, list(HETEROGENEOUS_PRESETS)))
        return 1
    variants = VARIANTS[:max(2, min(args.k, len(VARIANTS)))]
    names = [v[0] for v in variants]

    mc = HETEROGENEOUS_PRESETS[args.preset][AgentRole.MATHEMATICIAN]
    print("solver_version=%s  preset=%s  k=%d" % (SOLVER_VERSION, args.preset, len(variants)))
    print("Mathematician: %s/%s" % (mc.provider, mc.model_name))
    print("NOTE: greedy decoding - diversity comes from the prompt, not sampling.\n")

    client = UnifiedLLMClient(provider=mc.provider, use_cache=False,
                              model_override=mc.model_name,
                              load_4bit=getattr(mc, "load_4bit", False))
    solver = QualityEnhancedMultiAgentSolver(clients={r: client for r in AgentRole})

    pm = EnhancedProblemManager(random_seed=args.seed)
    ds = [d.strip() for d in args.datasets.split(",") if d.strip()]
    problems = pm.load_random_problems(ds, args.n)
    print("%d problems from %s\n" % (len(problems), ds))

    rows = []
    t0 = time.time()
    for i, p in enumerate(problems):
        q = p.get("puzzle") or p.get("question") or ""
        gold = _extract_last_number(str(p.get("answer", "")))
        rec = {"id": p.get("id", i), "gold": gold, "variants": {}}
        for name, hint, rfirst in variants:
            prev = solver.math_reasoning_first
            solver.math_reasoning_first = rfirst
            try:
                bp = solver.run_mathematician_analysis(q, strategy_hint=hint)
            except Exception as e:
                print("  [%d] %s: EXCEPTION %s: %s" % (i + 1, name, type(e).__name__, e))
                bp = {}
            finally:
                solver.math_reasoning_first = prev
            status, val = blueprint_value(bp, gold)
            rec["variants"][name] = {
                "status": status, "value": val,
                "equations": bp.get("equations", []) or [],
                "givens": bp.get("givens", {}) or {},
            }
        pick, support = agreement_pick([rec["variants"][n]["value"] for n in names])
        rec["pick"], rec["support"] = pick, support
        rec["pick_correct"] = bool(pick is not None and gold is not None and same(pick, gold))
        rows.append(rec)
        tag = "  ".join(
            "%s=%s" % (n.split("_")[0],
                       "OK" if rec["variants"][n]["status"] == "correct"
                       else rec["variants"][n]["status"][:4])
            for n in names)
        print("  [%d/%d] gold=%s  %s  pick=%s (support %d) %s"
              % (i + 1, len(problems), gold, tag, pick, support,
                 "RIGHT" if rec["pick_correct"] else ""))

    n = max(1, len(rows))

    print("\n" + "=" * 74)
    print("PER-VARIANT BLUEPRINT ACCURACY")
    print("=" * 74)
    for nm in names:
        c = Counter(r["variants"][nm]["status"] for r in rows)
        print("  %-20s correct %3d/%d = %5.1f%%   (wrong %d, structural %d, "
              "unevaluable %d, empty %d)"
              % (nm, c["correct"], n, 100.0 * c["correct"] / n,
                 c["wrong"], c["structural"], c["unevaluable"], c["empty"]))

    print("\n" + "=" * 74)
    print("SANITY: DID PROMPT VARIATION PRODUCE DIFFERENT BLUEPRINTS?")
    print("=" * 74)
    ident = sum(1 for r in rows
                if len({json.dumps(r["variants"][nm]["equations"]) for nm in names}) == 1)
    print("  problems where ALL %d variants gave identical equations: %d/%d = %.1f%%"
          % (len(names), ident, n, 100.0 * ident / n))
    if ident > 0.8 * n:
        print("  >>> VARIATION FAILED. The rest of this report is meaningless.")

    print("\n" + "=" * 74)
    print("ERROR DECORRELATION (the number that decides v15)")
    print("=" * 74)
    both_wrong = same_wrong = 0
    for r in rows:
        for a, b in itertools.combinations(names, 2):
            va, vb = r["variants"][a], r["variants"][b]
            if va["status"] == "wrong" and vb["status"] == "wrong":
                both_wrong += 1
                if same(va["value"], vb["value"]):
                    same_wrong += 1
    if both_wrong:
        print("  variant pairs where BOTH are wrong: %d" % both_wrong)
        print("    ...and they agree on the SAME wrong value: %d = %.1f%%"
              % (same_wrong, 100.0 * same_wrong / both_wrong))
        print("    low  (<20%) -> errors are diverse, agreement is a real filter")
        print("    high (>50%) -> correlated errors, k>1 buys nothing")
    else:
        print("  no pair was wrong twice - nothing to measure at this sample size")

    print("\n" + "=" * 74)
    print("THE '>=2 AGREE' RULE")
    print("=" * 74)
    fired = [r for r in rows if r["pick"] is not None]
    right = [r for r in fired if r["pick_correct"]]
    base = sum(1 for r in rows if r["variants"][names[0]]["status"] == "correct")
    combo = sum(1 for r in rows
                if (r["pick_correct"] if r["pick"] is not None
                    else r["variants"][names[0]]["status"] == "correct"))
    print("  coverage  : fires on %d/%d = %.1f%% of problems"
          % (len(fired), n, 100.0 * len(fired) / n))
    if fired:
        print("  precision : correct on %d/%d = %.1f%% when it fires"
              % (len(right), len(fired), 100.0 * len(right) / len(fired)))
    print("  vs today  : variant 0 alone is correct on %d/%d = %.1f%%"
          % (base, n, 100.0 * base / n))
    print("  rule accuracy (pick when it fires, else variant 0): %d/%d = %.1f%%  (%+d problems)"
          % (combo, n, 100.0 * combo / n, combo - base))

    print("\n" + "=" * 74)
    print("PROJECTED BLUEPRINT ACCURACY BY k")
    print("=" * 74)
    print("  %-4s%12s%12s" % ("k", ">=2 agree", "rule acc"))
    for k in range(1, len(names) + 1):
        sub = names[:k]
        acc = cov = 0
        for r in rows:
            pk, _ = agreement_pick([r["variants"][nm]["value"] for nm in sub])
            if pk is not None:
                cov += 1
                acc += 1 if (r["gold"] is not None and same(pk, r["gold"])) else 0
            else:
                acc += 1 if r["variants"][names[0]]["status"] == "correct" else 0
        print("  %-4d%11.1f%%%11.1f%%" % (k, 100.0 * cov / n, 100.0 * acc / n))

    elapsed = time.time() - t0
    print("\n  %.1f min total, %.1fs/problem (%d Mathematician calls each)"
          % (elapsed / 60.0, elapsed / n, len(names)))

    print("\n" + "=" * 74)
    print("DECISION")
    print("=" * 74)
    print("  BUILD v15 if: decorrelation is low (<20% same-wrong-value) AND the")
    print("  rule beats variant 0 by more than a couple of problems.")
    print("  DO NOT BUILD if the variants mostly agree while being wrong - that")
    print("  reproduces the alt_1/alt_2 failure, and is itself a thesis result:")
    print("  the Architect's errors are systematic, not sampling noise.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"preset": args.preset, "solver_version": SOLVER_VERSION,
                   "datasets": ds, "n": len(rows), "variants": names,
                   "rows": rows}, f, indent=2, default=str)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
