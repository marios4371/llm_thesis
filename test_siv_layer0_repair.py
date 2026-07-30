"""
Offline validation for the v14.0 changes: SIV Layer 0 (structural defect
detection) and the repair-eligibility gate.

Why this suite exists
---------------------
On the v13.0 run (mas_full_20260712_02.csv, n=150) 30 of the 109 audited
problems came back with blueprint_answer=None and invertible=False — SIV ran,
produced no usable verdict, and the repair loop then SKIPPED exactly those rows
because its guard was `not siv_result.invertible -> return`. This suite pins
down both halves of the v14.0 fix:

  1. SIV names the structural defect instead of failing silently.
  2. The repair gate fires on those rows (and still does not fire when SIV had
     nothing to audit, which is a different thing that also leaves
     execution_audit_passed=False).

No models, no network, no GPU: every assertion runs against the real
SymbolicInverseVerifier and the real _attempt_blueprint_repair, with only the
two LLM-calling methods stubbed.

Run:  python test_siv_layer0_repair.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from siv_module import SymbolicInverseVerifier as SIV

failures = []


def check(label, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(label)


# =====================================================================
# Part 1 — Layer 0 detection
# =====================================================================

def part1_layer0():
    print("\n" + "=" * 70)
    print("PART 1 — Layer 0: structural defect detection")
    print("=" * 70)

    # --- 1a. Undefined symbol on an equation's RHS -------------------------
    # This is the exact shape of the 30 silent-failure rows: 'rate' is neither
    # a given nor assigned by an earlier equation.
    r = SIV.verify({"givens": {"a": 10}, "equations": [
        "t = givens['a'] * rate", "answer = t"]}, 20)
    check("1a undefined symbol detected", r.has_structural_defect)
    check("1a offending name reported", "rate" in r.undefined_symbols,
          f"undefined_symbols={r.undefined_symbols}")
    check("1a audit did not pass", r.execution_audit_passed is False)
    check("1a not marked as skipped", r.skipped is False)
    check("1a chain reported non-invertible", r.invertible is False)
    report = SIV.get_error_localization_report(r)
    check("1a report leads with Layer 0", "[Layer 0" in report.split("\n")[1])
    check("1a report names the symbol", "'rate'" in report)
    check("1a report does NOT dump implicated givens",
          "Re-examine equations involving" not in report)

    # --- 1b. Undeclared givens key ----------------------------------------
    # SymPy cannot even parse this ("'Symbol' object is not subscriptable"),
    # so the deterministic pre-parse scan is the only thing that can report it.
    r = SIV.verify({"givens": {"a": 2}, "equations": ["answer = givens['aa'] * 3"]}, 6)
    check("1b undeclared key detected", r.has_structural_defect)
    check("1b key reported", any("aa" in k for k in r.undeclared_given_keys),
          f"undeclared_given_keys={r.undeclared_given_keys}")
    check("1b not marked as skipped", r.skipped is False)
    check("1b report names the key", "givens['aa']" in SIV.get_error_localization_report(r))

    # --- 1c. Non-numeric given referenced ---------------------------------
    r = SIV.verify({"givens": {"a": 2, "label": "five"},
                    "equations": ["answer = givens['a'] * givens['label']"]}, 10)
    check("1c non-numeric given flagged", r.has_structural_defect)
    check("1c flagged as declared-but-not-numeric",
          any("not numeric" in k for k in r.undeclared_given_keys),
          f"undeclared_given_keys={r.undeclared_given_keys}")

    # --- 1d. NO false positives on healthy blueprints ---------------------
    # The whole value of Layer 0 rests on it never firing on a good blueprint.
    healthy = [
        ("simple chain", {"a": 10, "b": 3},
         ["t = givens['a'] - givens['b']", "answer = t + 5"], 12),
        ("multi-step reuse", {"a": 100, "r": 0.2},
         ["disc = givens['a'] * givens['r']", "final = givens['a'] - disc",
          "answer = final * 2"], 160),
        ("floor division", {"n": 7, "p": 2},
         ["t = int(givens['n'] / givens['p'])", "answer = t * givens['p']"], 6),
        ("unused given (distractor)", {"a": 5, "distractor": 99},
         ["answer = givens['a'] * 2"], 10),
        ("comment line", {"a": 2}, ["# compute", "answer = givens['a'] + 1"], 3),
        ("no explicit answer var", {"a": 2}, ["total = givens['a'] * 3"], 6),
        ("builtin funcs", {"a": -5, "b": 9},
         ["m = max(abs(givens['a']), givens['b'])", "answer = m"], 9),
    ]
    for name, g, eq, ans in healthy:
        r = SIV.verify({"givens": g, "equations": eq}, ans)
        check(f"1d no false positive: {name}", not r.has_structural_defect,
              f"undefined={r.undefined_symbols} keys={r.undeclared_given_keys}")

    # --- 1e. Arithmetic inconsistency is NOT a structural defect ----------
    # Layer 1's own signal must stay distinguishable from Layer 0's.
    r = SIV.verify({"givens": {"price": 50, "qty": 100},
                    "equations": ["answer = givens['price'] * givens['qty']"]}, 2500)
    check("1e arithmetic mismatch has no structural defect",
          not r.has_structural_defect)
    check("1e audit failed", r.execution_audit_passed is False)
    check("1e blueprint answer still computed", r.blueprint_answer == 5000.0,
          f"blueprint_answer={r.blueprint_answer}")
    rep = SIV.get_error_localization_report(r)
    check("1e report leads with Layer 1", "[Layer 1" in rep.split("\n")[1])
    check("1e report carries the concrete discrepancy",
          "5000" in rep and "2500" in rep)

    # --- 1f. skipped vs failed ------------------------------------------
    r = SIV.verify({"givens": {"a": 1}, "equations": []}, 5)
    check("1f no equations -> skipped", r.skipped is True)
    r = SIV.verify({"givens": {"a": "x"}, "equations": ["answer = 1"]}, 1)
    check("1f no numeric givens -> skipped", r.skipped is True)
    r = SIV.verify({"givens": {"a": 10}, "equations": ["answer = givens['a']"]}, 10)
    check("1f healthy audit -> not skipped", r.skipped is False)
    check("1f healthy audit -> passed", r.execution_audit_passed is True)


# =====================================================================
# Part 2 — Repair-eligibility gate
# =====================================================================

def part2_repair_gate():
    print("\n" + "=" * 70)
    print("PART 2 — repair gate (_attempt_blueprint_repair)")
    print("=" * 70)

    from Mas_solver import QualityEnhancedMultiAgentSolver, AgentResponse

    # Bypass __init__: no clients, no models, no network. Only the two
    # LLM-calling methods are stubbed; the gate and the report wiring are real.
    solver = object.__new__(QualityEnhancedMultiAgentSolver)

    calls = {"mathematician": 0, "programmer": 0, "last_repair_context": "",
             "last_prior_blueprint": None}

    REPAIRED_BP = {"givens": {"a": 10, "b": 2},
                   "equations": ["t = givens['a'] * givens['b']", "answer = t"]}

    def fake_mathematician(problem, repair_context="", prior_blueprint=None):
        calls["mathematician"] += 1
        calls["last_repair_context"] = repair_context
        calls["last_prior_blueprint"] = prior_blueprint
        return dict(REPAIRED_BP)

    def fake_programmer(problem, blueprint, max_attempts=3):
        calls["programmer"] += 1
        return AgentResponse(agent="Programmer (optimized)", answer="20",
                             parsed="20", confidence=1.0, reasoning_trace="code")

    solver.run_mathematician_analysis = fake_mathematician
    solver.run_programmer_solver = fake_programmer

    def reset():
        calls.update({"mathematician": 0, "programmer": 0,
                      "last_repair_context": "", "last_prior_blueprint": None})

    orig_resp = AgentResponse(agent="Programmer (optimized)", answer="99",
                              parsed="99", confidence=1.0, reasoning_trace="code")

    # --- 2a. THE REGRESSION THIS RELEASE FIXES ---------------------------
    # Structural defect => non-invertible => v13.0 returned early here.
    broken_bp = {"givens": {"a": 10}, "equations": ["t = givens['a'] * rate", "answer = t"]}
    siv_broken = SIV.verify(broken_bp, 99)
    check("2a precondition: the v13.0 guard would have skipped this",
          siv_broken.invertible is False and siv_broken.execution_audit_passed is False)
    reset()
    bp, resp, new_siv, repaired = solver._attempt_blueprint_repair(
        "problem text", broken_bp, orig_resp, siv_broken)
    check("2a repair FIRES on a structurally broken blueprint", repaired is True)
    check("2a Mathematician was re-invoked", calls["mathematician"] == 1)
    check("2a repaired blueprint replaced the broken one",
          bp.get("equations") == REPAIRED_BP["equations"])
    check("2a repaired flag set on blueprint", bp.get("_blueprint_repaired") is True)
    check("2a post-repair audit passes", new_siv is not None
          and new_siv.execution_audit_passed is True)
    check("2a repair context named the undefined symbol",
          "rate" in calls["last_repair_context"])
    check("2a prior blueprint was shown to the Mathematician",
          calls["last_prior_blueprint"] is not None
          and calls["last_prior_blueprint"].get("equations") == broken_bp["equations"])

    # --- 2b. Arithmetic inconsistency still repairs (v13.0 behaviour kept) -
    arith_bp = {"givens": {"price": 50, "qty": 100},
                "equations": ["answer = givens['price'] * givens['qty']"]}
    siv_arith = SIV.verify(arith_bp, 2500)
    reset()
    _, _, _, repaired = solver._attempt_blueprint_repair(
        "problem text", arith_bp, orig_resp, siv_arith)
    check("2b repair still fires on arithmetic inconsistency", repaired is True)
    check("2b repair context carries the Layer 1 discrepancy",
          "5000" in calls["last_repair_context"])

    # --- 2c. Must NOT fire when the audit passed --------------------------
    good_bp = {"givens": {"a": 10, "b": 2},
               "equations": ["t = givens['a'] * givens['b']", "answer = t"]}
    siv_good = SIV.verify(good_bp, 20)
    reset()
    bp, resp, new_siv, repaired = solver._attempt_blueprint_repair(
        "problem text", good_bp, orig_resp, siv_good)
    check("2c no repair when audit passed", repaired is False)
    check("2c no LLM calls spent", calls["mathematician"] == 0 and calls["programmer"] == 0)
    check("2c original triple returned unchanged", bp is good_bp and resp is orig_resp)

    # --- 2d. Must NOT fire when SIV had nothing to audit ------------------
    # Both of these leave execution_audit_passed=False, which is why the
    # `skipped` flag had to exist: without it the new gate would burn two LLM
    # calls re-deriving blueprints SIV never actually looked at.
    for label, bp_in, ans in [
        ("no equations", {"givens": {"a": 1}, "equations": []}, 5),
        ("no numeric givens", {"givens": {"a": "x"}, "equations": ["answer = 1"]}, 1),
    ]:
        siv_skipped = SIV.verify(bp_in, ans)
        reset()
        _, _, _, repaired = solver._attempt_blueprint_repair(
            "problem text", bp_in, orig_resp, siv_skipped)
        check(f"2d no repair when SIV skipped ({label})", repaired is False)
        check(f"2d no LLM calls spent ({label})", calls["mathematician"] == 0)

    # --- 2e. siv_result=None is still handled -----------------------------
    reset()
    _, _, _, repaired = solver._attempt_blueprint_repair(
        "problem text", good_bp, orig_resp, None)
    check("2e no repair when siv_result is None", repaired is False)

    # --- 2f. Tautological flag on a SymPy-derived post-repair answer ------
    # SymbolicSolver.solve_from_blueprint EXECS the blueprint's own equations,
    # so Layer 1 passes by construction and certifies nothing. On the v13.0 run
    # 4/4 such rows were recorded verified=True, 2 of them plainly wrong.
    def fake_programmer_sympy(problem, blueprint, max_attempts=3):
        calls["programmer"] += 1
        return AgentResponse(agent="SymPy (symbolic fallback)", answer="20",
                             parsed="20", confidence=0.8, reasoning_trace="trace")

    solver.run_programmer_solver = fake_programmer_sympy
    reset()
    _, resp, new_siv, repaired = solver._attempt_blueprint_repair(
        "problem text", broken_bp, orig_resp, siv_broken)
    check("2f repair still fires", repaired is True)
    check("2f SymPy-derived post-repair audit marked tautological",
          new_siv is not None and new_siv.tautological is True)
    solver.run_programmer_solver = fake_programmer
    reset()
    _, _, new_siv, _ = solver._attempt_blueprint_repair(
        "problem text", broken_bp, orig_resp, siv_broken)
    check("2f executed-code post-repair audit NOT tautological",
          new_siv is not None and new_siv.tautological is False)


# =====================================================================
# Part 3 — reasoning-first blueprint parsing
# =====================================================================

def part3_reasoning_first():
    print("\n" + "=" * 70)
    print("PART 3 — reasoning-first blueprint mode")
    print("=" * 70)

    from Mas_solver import (
        QualityEnhancedMultiAgentSolver, _after_blueprint_marker,
        _extract_blueprint_json,
    )

    GOOD_JSON = ('{"unknown": "total", "givens": {"a": 10, "b": 3}, '
                 '"equations": ["t = givens[\'a\'] - givens[\'b\']", "answer = t + 5"], '
                 '"expected_answer": "12", "solution_steps": ["s1"], '
                 '"distractor_check": "None"}')

    # --- 3a. marker slicing ------------------------------------------------
    reply = f"DERIVATION:\nJane starts with 10, eats 3 -> 7, buys 5 -> 12.\nBLUEPRINT:\n{GOOD_JSON}"
    bp = _extract_blueprint_json(_after_blueprint_marker(reply))
    check("3a blueprint parsed from two-part reply", bp.get("equations") and bp.get("givens"),
          f"givens={bp.get('givens')}")
    check("3a givens correct", bp.get("givens") == {"a": 10, "b": 3})

    # A derivation containing braces is exactly what would break the naive
    # first-'{'-to-last-'}' scan, which is why the marker slice exists.
    reply = ("DERIVATION:\nThe candidate set is {1, 2, 3} and f(x) = {x if x>0}.\n"
             f"So the answer is 12.\nBLUEPRINT:\n{GOOD_JSON}")
    bp = _extract_blueprint_json(_after_blueprint_marker(reply))
    check("3b braces in the derivation do not break parsing",
          bp.get("givens") == {"a": 10, "b": 3}, f"givens={bp.get('givens')}")

    # --- 3c. graceful degradation -----------------------------------------
    plain = f"Here is the blueprint:\n{GOOD_JSON}"
    check("3c no marker -> text passed through unchanged",
          _after_blueprint_marker(plain) == plain)
    truncated = "DERIVATION:\nStep 1: 10 - 3 = 7. Step 2: 7 + 5 = 12.\nBLUEPRINT:"
    check("3c marker with no JSON after it -> full text kept (CoT fallback can use it)",
          _after_blueprint_marker(truncated) == truncated)

    # --- 3d. end-to-end through run_mathematician_analysis -----------------
    class FakeClient:
        provider = "local_hf"

        def __init__(self):
            self.last_msgs = None

        def call_model(self, msgs, **kw):
            self.last_msgs = msgs
            self.last_kw = kw
            return (f"DERIVATION:\nJane has 10, eats 3, buys 5. 10-3=7, 7+5=12.\n"
                    f"BLUEPRINT:\n{GOOD_JSON}")

    solver = object.__new__(QualityEnhancedMultiAgentSolver)
    solver.math_temp = 0.0
    solver.math_reasoning_first = True
    fake = FakeClient()
    solver._get_client = lambda role: fake

    bp = solver.run_mathematician_analysis("Jane has 10 apples...")
    sys_txt = fake.last_msgs[0]["content"]
    check("3d prompt asks for DERIVATION then BLUEPRINT",
          "DERIVATION:" in sys_txt and "BLUEPRINT:" in sys_txt)
    check("3d contradictory 'ONLY valid JSON' rule removed",
          "Return ONLY valid JSON, no preamble" not in sys_txt)
    check("3d token budget raised for the derivation",
          fake.last_kw.get("max_tokens", 0) > 1536,
          f"max_tokens={fake.last_kw.get('max_tokens')}")
    check("3d blueprint extracted end-to-end", bp.get("givens") == {"a": 10, "b": 3})

    # json-only mode must still behave exactly as v13.0 did
    solver.math_reasoning_first = False
    fake2 = FakeClient()
    fake2.call_model = lambda msgs, **kw: (
        setattr(fake2, "last_msgs", msgs), setattr(fake2, "last_kw", kw), GOOD_JSON)[-1]
    solver._get_client = lambda role: fake2
    bp = solver.run_mathematician_analysis("Jane has 10 apples...")
    check("3e json-only mode keeps the v13.0 prompt",
          "Return ONLY valid JSON, no preamble" in fake2.last_msgs[0]["content"])
    check("3e json-only mode keeps the v13.0 token budget",
          fake2.last_kw.get("max_tokens") == 1536,
          f"max_tokens={fake2.last_kw.get('max_tokens')}")
    check("3e json-only mode still parses", bp.get("givens") == {"a": 10, "b": 3})

    # a repair call must never be switched into two-part mode
    solver.math_reasoning_first = True
    fake3 = FakeClient()
    fake3.call_model = lambda msgs, **kw: (
        setattr(fake3, "last_msgs", msgs), setattr(fake3, "last_kw", kw), GOOD_JSON)[-1]
    solver._get_client = lambda role: fake3
    solver.run_mathematician_analysis(
        "Jane has 10 apples...",
        repair_context="[Layer 0 — Structure] 'rate' is never defined",
        prior_blueprint={"givens": {"a": 1}, "equations": ["answer = givens['a'] * rate"]})
    user_txt = fake3.last_msgs[1]["content"]
    check("3f repair prompt stays single-part", "DERIVATION:" not in fake3.last_msgs[0]["content"])
    check("3f repair prompt echoes the prior blueprint", "rate" in user_txt and "givens" in user_txt)
    check("3f repair prompt carries the SIV report", "Layer 0" in user_txt)


if __name__ == "__main__":
    print("=" * 70)
    print("v14.0 — SIV LAYER 0 + REPAIR GATE (offline, no models)")
    print("=" * 70)
    part1_layer0()
    part2_repair_gate()
    part3_reasoning_first()
    print("\n" + "=" * 70)
    if failures:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"    - {f}")
        print("=" * 70)
        sys.exit(1)
    print("  ALL CHECKS PASSED")
    print("=" * 70)
