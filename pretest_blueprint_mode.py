"""
Pre-test: does reasoning-first blueprint generation actually produce better
blueprints than the v13.0 json-only prompt?

WHY THIS EXISTS
---------------
On the v13.0 run (n=150) the Mathematician's blueprints were the binding
constraint on the whole system, not SIV and not the selection layer:

  * blueprint answer == gold, on rows where the chain was evaluable: 35.4%
    (61% on the unrepaired primary-JSON subset)
  * the same model's free chain-of-thought on the same rows:          88.6%
  * on gsm8k_test specifically: baseline 100%, blueprints 41.7%

The blueprint path was uniquely right on 2/150 problems while the baseline was
uniquely right on 90, so oracle{baseline, blueprint} = 76.7% vs the baseline's
75.3%: nothing downstream of the blueprint can add more than ~1.4pp until the
blueprints themselves improve. v14.0's hypothesis is that the old prompt simply
left the model no scratchpad ("Return ONLY valid JSON, no preamble" while also
asking it to "mentally trace through your equations", with constrained decoding
disabled). Reasoning-first asks for DERIVATION: then BLUEPRINT: in one call.

That hypothesis is UNVALIDATED. The only field evidence is n=5. This script
measures it on a handful of problems in ~15 minutes instead of discovering the
answer 8 hours into a full run.

WHAT IT MEASURES
----------------
Only the Mathematician. No Programmer, no SHT, no baselines, no judge. For each
problem, in each mode:
  - was a blueprint with equations produced at all?
  - does its equation chain evaluate (SIV Layer 0/1)?
  - does that value equal the gold answer?          <- the metric that matters
  - did it hit the structural-defect path (undefined names)?

The last column is the direct measure of what v14.0's Layer 0 repair targets.

USAGE (Kaggle, before committing a full run)
--------------------------------------------
    !python pretest_blueprint_mode.py --n 20
    !python pretest_blueprint_mode.py --n 20 --preset qwen_math7b_mixed

Then set MATH_REASONING_FIRST in Cell 3 to whichever mode won. If the two are
within noise at n=20, keep the v13.0 behaviour (json-only): an unvalidated
prompt change has no business in a thesis run.
"""

import argparse
import json
import sys
import time

from Mas_solver import (
    QualityEnhancedMultiAgentSolver, UnifiedLLMClient, AgentRole,
    HETEROGENEOUS_PRESETS, EnhancedProblemManager,
    _extract_last_number, SOLVER_VERSION,
)
from siv_module import SymbolicInverseVerifier as SIV


def evaluate_blueprint(bp: dict, gold: float):
    """
    Score one blueprint without any LLM involvement.

    Returns (status, blueprint_answer) where status is one of:
      empty      — no equations/givens at all
      structural — chain references undefined names (v14.0 Layer 0)
      unevaluable— chain built but forward evaluation failed
      wrong      — evaluates, but not to gold
      correct    — evaluates to gold
    """
    if not bp.get("equations") or not bp.get("givens"):
        return "empty", None

    # computed_answer is irrelevant here: we want the blueprint's OWN value, so
    # pass a sentinel and read blueprint_answer off the forward audit.
    r = SIV.verify(bp, 0.0)
    if r.has_structural_defect:
        return "structural", None
    if r.blueprint_answer is None:
        return "unevaluable", None
    if gold is None:
        return "wrong", r.blueprint_answer
    diff = abs(r.blueprint_answer - gold)
    ok = diff < 1e-6 or diff <= 1e-6 * max(abs(gold), 1.0)
    return ("correct" if ok else "wrong"), r.blueprint_answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="problems per mode")
    ap.add_argument("--preset", default="qwen_math7b_mixed",
                    help="HETEROGENEOUS_PRESETS key")
    ap.add_argument("--datasets", default="gsm8k_test,gsm-hard,svamp",
                    help="EnhancedProblemManager dataset keys (note: 'svamp', "
                         "not 'svamp_test' — the latter is the id prefix)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="pretest_blueprint_mode.json")
    args = ap.parse_args()

    if args.preset not in HETEROGENEOUS_PRESETS:
        print(f"Unknown preset {args.preset!r}. Available: {list(HETEROGENEOUS_PRESETS)}")
        return 1

    # Only the Mathematician's client is built — this is deliberately not the
    # full pipeline, so the pre-test cannot be confounded by the Programmer, the
    # judge, or the selection layer, and it loads at most one model.
    mc = HETEROGENEOUS_PRESETS[args.preset][AgentRole.MATHEMATICIAN]
    print(f"solver_version={SOLVER_VERSION}  preset={args.preset}")
    print(f"Mathematician: {mc.provider}/{mc.model_name} "
          f"(4bit={getattr(mc, 'load_4bit', False)})")

    client = UnifiedLLMClient(provider=mc.provider, use_cache=False,
                             model_override=mc.model_name,
                             load_4bit=getattr(mc, 'load_4bit', False))
    clients = {role: client for role in AgentRole}
    solver = QualityEnhancedMultiAgentSolver(clients=clients)

    pm = EnhancedProblemManager(random_seed=args.seed)
    ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    problems = pm.load_random_problems(ds_list, args.n)
    print(f"{len(problems)} problems from {ds_list}\n")

    STATUSES = ["correct", "wrong", "structural", "unevaluable", "empty"]
    results = {}

    for mode_name, flag in [("json_only (v13.0)", False),
                            ("reasoning_first (v14.0)", True)]:
        solver.math_reasoning_first = flag
        counts = {s: 0 for s in STATUSES}
        rows = []
        t0 = time.time()
        print("=" * 72)
        print(f"MODE: {mode_name}")
        print("=" * 72)

        for i, p in enumerate(problems):
            # load_random_problems returns {'puzzle', 'answer', 'dataset', 'id'}
            question = p.get("puzzle") or p.get("question") or ""
            gold = _extract_last_number(str(p.get("answer", "")))
            try:
                bp = solver.run_mathematician_analysis(question)
            except Exception as e:
                print(f"  [{i+1}/{len(problems)}] EXCEPTION: {type(e).__name__}: {e}")
                counts["empty"] += 1
                continue
            status, bp_ans = evaluate_blueprint(bp, gold)
            counts[status] += 1
            rows.append({
                "id": p.get("id", i),
                "status": status, "blueprint_answer": bp_ans, "gold": gold,
                "n_givens": len(bp.get("givens", {}) or {}),
                "n_equations": len(bp.get("equations", []) or []),
                "provenance": ("tautological" if bp.get("_local_hf_fallback")
                               else "extracted_from_cot" if bp.get("_extracted_from_cot")
                               else "primary_json"),
            })
            print(f"  [{i+1}/{len(problems)}] {status:<12} "
                  f"bp={bp_ans} gold={gold} "
                  f"({len(bp.get('givens', {}) or {})}g/"
                  f"{len(bp.get('equations', []) or [])}eq)")

        elapsed = time.time() - t0
        n = max(1, len(problems))
        results[mode_name] = {"counts": counts, "rows": rows,
                              "seconds": elapsed, "n": len(problems)}
        print(f"\n  {mode_name}: "
              + "  ".join(f"{s}={counts[s]}" for s in STATUSES)
              + f"\n  blueprint accuracy = {counts['correct']}/{n} "
                f"= {100*counts['correct']/n:.1f}%   ({elapsed/n:.1f}s/problem)\n")

    print("=" * 72)
    print("COMPARISON")
    print("=" * 72)
    hdr = f"{'mode':<26}" + "".join(f"{s:>13}" for s in STATUSES) + f"{'acc':>8}"
    print(hdr)
    for mode_name, r in results.items():
        c, n = r["counts"], max(1, r["n"])
        print(f"{mode_name:<26}" + "".join(f"{c[s]:>13}" for s in STATUSES)
              + f"{100*c['correct']/n:>7.1f}%")

    a = results["json_only (v13.0)"]["counts"]["correct"]
    b = results["reasoning_first (v14.0)"]["counts"]["correct"]
    n = max(1, len(problems))
    delta = 100 * (b - a) / n
    print(f"\ndelta (reasoning_first - json_only) = {b - a} problems = {delta:+.1f}pp")
    print("\nAt this sample size treat anything under ~3 problems as noise.")
    print("  clear win  -> set MATH_REASONING_FIRST = True  in Cell 3")
    print("  no win     -> set MATH_REASONING_FIRST = False (keep v13.0 behaviour)")
    print("Also compare the 'structural' column: it counts the blueprints v14.0's")
    print("Layer 0 repair path now catches instead of silently discarding.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"preset": args.preset, "solver_version": SOLVER_VERSION,
                   "datasets": ds_list, "n": len(problems), "results": results},
                  f, indent=2, default=str)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
