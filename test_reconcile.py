"""[v17.0] Guards for the reconciliation of two independent derivations.

Fully offline: no models, no GPU, no network, no HuggingFace cache. Every
blueprint, every problem statement and every number in PART 2 is copied
verbatim out of a finished run, so these cases are field data rather than
constructions that happen to suit the code.

Run as `python test_reconcile.py` (this project has no pytest).
"""
from __future__ import annotations

import sys

import reconcile
from reconcile import OperandRepair

FAILS = []
CHECKS = [0]


def check(cond, label):
    CHECKS[0] += 1
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILS.append(label)


def head(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# =====================================================================
# PART 1 — the agreement band
# =====================================================================

def part1():
    head("PART 1 — agreement, and what it does NOT cover")

    check(reconcile.agrees(100.0, 100.0), "exact equality agrees")
    check(reconcile.agrees(9810512.0, 9810512.4),
          "a last-digit slip at 7 digits is inside the band")
    check(not reconcile.agrees(74.0, 76.0), "74 vs 76 is not agreement")
    check(not reconcile.agrees(None, 76.0), "a missing side never agrees")
    check(not reconcile.agrees(100.0, 101.0), "1% apart is outside the band")

    # The band is the one v16.3 measured and the one SIV uses. If any of the
    # three drifts, this catches it.
    import near_agreement
    from siv_module import SymbolicInverseVerifier
    check(reconcile.DEFAULT_TAU == near_agreement.DEFAULT_TAU
          == SymbolicInverseVerifier.TOLERANCE_REL == 1e-4,
          "one tolerance, three modules, still 1e-4")

    bp = {"givens": {"a": 2.0, "b": 3.0}, "equations": ["answer = givens['a'] + givens['b']"]}
    r = reconcile.reconcile(bp, 5.0, 5.0, "You have 2 and 3.")
    check(r.stage == reconcile.AGREE and r.certified and r.answer == 5.0,
          "two derivations that agree are certified")

    r = reconcile.reconcile(bp, None, 5.0, "You have 2 and 3.")
    check(r.stage == reconcile.PROGRAM_ONLY and not r.certified,
          "equations that do not evaluate leave the program uncertified")
    r = reconcile.reconcile(bp, 5.0, None, "You have 2 and 3.")
    check(r.stage == reconcile.BLUEPRINT_ONLY and not r.certified and r.answer == 5.0,
          "a failed program leaves the blueprint uncertified")
    r = reconcile.reconcile(bp, None, None, "x")
    check(r.stage == reconcile.NOTHING and r.answer is None,
          "no derivation, no answer")

    # On agreement the SYMBOLIC value ships, because SymPy is exact where a
    # Python float is not. Inside the band they can only differ in the tail.
    r = reconcile.reconcile(bp, 9810512.0, 9810512.4, "2 and 3")
    check(r.answer == 9810512.0, "on near agreement the exact value ships")


# =====================================================================
# PART 2 — real rows: the Architect miscopied exactly one operand
# =====================================================================
# Copied verbatim from full_mas_20260819.csv (v14.8) and mas_full_20260824.csv
# (v15.2); the statements come from the datasets those runs drew from. In each
# one the reference answer is the independently derived number, the blueprint
# disagrees with it, and the disagreement is explained by a single given the
# problem text does not contain.

REAL_SLIPS = [
    dict(
        pid="gsm8k_test_619", gold=76.0, reference=76.0,
        text=("Each pole on a road intersection has 4 street lights. If the number "
              "of poles at each intersection is 6, and the road has 4 "
              "intersections, calculate the total number of functioning street "
              "lights if 20 streetlights from the total number are not working."),
        givens={"lights_per_pole": 4, "poles_per_intersection": 6,
                "intersections": 4, "non_functioning_lights": 22},
        equations=[
            "total_lights = givens['lights_per_pole'] * givens['poles_per_intersection'] * givens['intersections']",
            "functioning_lights = total_lights - givens['non_functioning_lights']"],
        given="non_functioning_lights", declared=22.0, wanted=20.0,
    ),
    dict(
        pid="gsm-hard_946", gold=594.0, reference=594.0,
        text=("The caretaker of the docks needs to buy some new line. He wants 3 "
              "feet of line for every foot of dock. Right now, there is 200 feet "
              "of dock, and he has 6 feet of new line. How many feet of line does "
              "he need to buy in total?"),
        givens={"dock_length": 22, "current_line": 6},
        equations=["required_line = 3 * givens['dock_length']",
                   "answer = required_line - givens['current_line']"],
        given="dock_length", declared=22.0, wanted=200.0,
    ),
    dict(
        pid="svamp_test_294", gold=1.0, reference=1.0,
        text=("Mary is baking a cake. The recipe calls for 5 cups of sugar and 13 "
              "cups of flour. She already put in some cups of flour. If she still "
              "needs 12 more cups of flour How many cups of flour did she put in?"),
        givens={"required_flour": 13, "remaining_flour": 11},
        equations=["answer = givens['required_flour'] - givens['remaining_flour']"],
        given="remaining_flour", declared=11.0, wanted=12.0,
    ),
    dict(
        pid="gsm8k_test_16", gold=230.0, reference=230.0,
        text=("Two trains leave San Rafael at the same time. They begin traveling "
              "westward, both traveling for 80 miles. The next day, they travel "
              "northwards, covering 150 miles. What's the distance covered by each "
              "train in the two days?"),
        givens={"miles_westward": 80, "miles_northward": 155},
        equations=["first_day_distance = givens['miles_westward']",
                   "second_day_distance = givens['miles_northward']",
                   "answer = first_day_distance + second_day_distance"],
        given="miles_northward", declared=155.0, wanted=150.0,
    ),
]


def part2():
    head("PART 2 — real single-operand slips, attributed and repaired")

    for case in REAL_SLIPS:
        bp = {"givens": dict(case["givens"]), "equations": list(case["equations"])}
        before = reconcile._evaluate_blueprint(bp, case["reference"])
        check(not reconcile.agrees(before, case["reference"]),
              f"{case['pid']}: the blueprint really did disagree ({before})")

        accepted, _ = reconcile.attribute_blueprint_slip(
            bp, case["reference"], case["text"])
        check(len(accepted) == 1,
              f"{case['pid']}: exactly one operand is implicated (got {len(accepted)})")
        if len(accepted) == 1:
            got = accepted[0]
            check(got.given == case["given"],
                  f"{case['pid']}: names {case['given']}")
            check(abs(got.declared - case["declared"]) < 1e-9
                  and abs(got.reconstructed - case["wanted"]) < 1e-6,
                  f"{case['pid']}: {case['declared']} -> {case['wanted']}")

        rec = reconcile.reconcile(bp, before, case["reference"], case["text"])
        check(rec.stage == reconcile.REPAIRED_BLUEPRINT,
              f"{case['pid']}: reconciles by repair")
        check(rec.certified, f"{case['pid']}: the restored agreement is certified")
        check(abs(rec.answer - case["gold"]) < 1e-6,
              f"{case['pid']}: ships {case['gold']}")
        # the repaired value has to be a number the problem text states
        check(reconcile._in_text(rec.repair.reconstructed,
                                 reconcile._text_numbers(case["text"])),
              f"{case['pid']}: the substituted value appears in the problem text")
        # and the original blueprint must not have been mutated
        check(bp["givens"][case["given"]] == case["givens"][case["given"]],
              f"{case['pid']}: the input blueprint is left untouched")


# =====================================================================
# PART 3 — refusals. Every one of these is a case where shipping would be
# worse than leaving the disagreement standing.
# =====================================================================

def part3():
    head("PART 3 — refusals")

    # (a) A fudge that would force agreement without matching the text.
    # gsm-hard_1266, verbatim. Two givens are wrong (the bundle discount is 22
    # rather than 20 AND the cable cost is 912 rather than 9202361), so the
    # inversion asks for 9202358.4 -- a number nobody wrote. At a loose
    # tolerance this passed and shipped; it must not.
    bp = {"givens": {"netflix_cost": 10, "hulu_cost": 10, "disney_plus_cost": 10,
                     "bundle_discount": 22, "cable_package_cost": 912},
          "equations": [
              "total_service_cost = givens['netflix_cost'] + givens['hulu_cost'] + givens['disney_plus_cost']",
              "discount_amount = total_service_cost * (givens['bundle_discount']/100)",
              "total_bundled_cost = total_service_cost - discount_amount",
              "answer = givens['cable_package_cost'] - total_bundled_cost"]}
    text = ("Tim decides to cancel his cable subscription and get streaming "
            "services.  He gets Netflix for $10 a month.  Hulu and Disney Plus "
            "normally cost $10 a month each but he saves 20% for bundling.  How "
            "much money does he save by cancelling his $9202361 cable package?")
    rec = reconcile.reconcile(bp, reconcile._evaluate_blueprint(bp, 9202335.0),
                              9202335.0, text)
    check(rec.stage == reconcile.UNRESOLVED,
          "a two-operand defect is not explained away as one slip")
    check(not rec.certified, "and nothing is certified there")

    # (b) Every operand IS in the text, so a disagreement is structural: the
    # equations are wrong, and no substitution may paper over that.
    bp = {"givens": {"a": 40.0, "b": 8.0},
          "equations": ["answer = givens['a'] * givens['b']"]}
    rec = reconcile.reconcile(bp, 320.0, 48.0, "Add 40 and 8.")
    check(rec.stage == reconcile.UNRESOLVED and rec.repair is None,
          "grounded operands + disagreement = structural, no repair")

    # (c) Ambiguity: two ungrounded givens both reconstruct onto text numbers.
    bp = {"givens": {"x": 99.0, "y": 98.0},
          "equations": ["answer = givens['x'] + givens['y']"]}
    accepted, _ = reconcile.attribute_blueprint_slip(bp, 12.0, "Take 5 and 7 please.")
    check(len(accepted) != 1, "two candidate culprits means no unique attribution")
    rec = reconcile.reconcile(bp, 197.0, 12.0, "Take 5 and 7 please.")
    check(rec.stage == reconcile.UNRESOLVED, "ambiguity refuses rather than guesses")

    # (d) A given literally named 'answer' is circular and must be skipped.
    bp = {"givens": {"answer": 14.33}, "equations": ["answer = givens['answer']"]}
    accepted, _ = reconcile.attribute_blueprint_slip(bp, 15.0, "It costs 15 dollars.")
    check(not accepted, "the 'answer' given is never 'reconstructed'")

    # (e) Structural constants are not transcription slips.
    bp = {"givens": {"half": 0.5, "total": 10.0},
          "equations": ["answer = givens['total'] * givens['half']"]}
    accepted, _ = reconcile.attribute_blueprint_slip(bp, 7.0, "Ten items, half of them.")
    check(not any(a.given == "half" for a in accepted),
          "0.5 is a structural constant, not a miscopied operand")

    # (f) Repair disabled is honoured.
    bp = {"givens": {"dock_length": 22, "current_line": 6},
          "equations": ["required_line = 3 * givens['dock_length']",
                        "answer = required_line - givens['current_line']"]}
    rec = reconcile.reconcile(bp, 60.0, 594.0, "3 feet per foot, 200 feet of dock, 6 feet of line.",
                              enable_repair=False)
    check(rec.stage == reconcile.UNRESOLVED and rec.repair is None,
          "enable_repair=False leaves the disagreement alone")


# =====================================================================
# PART 4 — the program side
# =====================================================================

def part4():
    head("PART 4 — the symmetric repair, on the program's own givens")

    code = ("givens = {'lights_per_pole': 4, 'broken': 22}\n"
            "answer = givens['lights_per_pole'] * 10 - givens['broken']\n"
            "print(answer)\n")
    text = "Each pole has 4 lights over 10 poles and 20 are broken."
    new_code, repairs = reconcile.snap_program_givens(code, text)
    check(new_code is not None and len(repairs) == 1
          and repairs[0].given == "broken" and repairs[0].reconstructed == 20.0,
          "an ungrounded program operand snaps to the text")
    check(new_code is not None and "givens['lights_per_pole']" in new_code
          and new_code.count("print(answer)") == 1,
          "the rest of the program survives the splice untouched")
    check(new_code is not None and reconcile._literal_number(
        __import__("ast").parse(new_code).body[0].value.values[1]) == 20.0,
        "the rewritten source still parses and carries the new value")

    ok_code, _ = reconcile.snap_program_givens(
        "givens = {'a': 4, 'b': 20}\nanswer = givens['a']\n",
        "Four and twenty: 4 and 20.")
    check(ok_code is None, "a fully grounded program is left alone")

    check(reconcile.snap_program_givens(
        "givens = {'a': compute(1)}\nanswer = 1\n", "1 and 2")[0] is None,
        "a computed value refuses the whole rewrite")
    check(reconcile.snap_program_givens("answer = 1 + 1\n", "1 and 2")[0] is None,
          "no givens dict, nothing to do")
    check(reconcile.snap_program_givens("def f(:\n", "1")[0] is None,
          "unparseable source is handled, not raised")

    # Negative literals. blueprint_repair._TEXT_NUMBER is unsigned by design
    # (v15.0), so a negative operand can never be matched to a text literal and
    # must therefore never be rewritten -- but it must also not be misread as
    # its absolute value, which would silently corrupt the program. Both halves
    # are checked here because four measured runs depend on that behaviour and
    # this module must not quietly change it.
    nc, reps = reconcile.snap_program_givens(
        "givens = {'debt': -22, 'other': 5}\nanswer = givens['debt']\n",
        "A debt of -20 and 5 more.")
    check(nc is None and not reps,
          "a negative operand is refused, not snapped to an unsigned text number")

    nc, reps = reconcile.snap_program_givens(
        "givens = {'debt': -22, 'broken': 22}\nanswer = givens['broken']\n",
        "20 are broken, and the balance is -20.")
    check(nc is not None and len(reps) == 1 and reps[0].given == "broken"
          and "-22" in nc and "'broken': 20.0" in nc,
          "a neighbouring negative operand survives someone else's rewrite intact")

    # End to end: the program is only repaired when the repair CREATES
    # agreement with the other derivation.
    bp = {"givens": {"a": 4.0, "b": 20.0},
          "equations": ["answer = givens['a'] * 10 - givens['b']"]}
    calls = []

    def rerun(src):
        calls.append(src)
        return 20.0                      # what the snapped program would print

    rec = reconcile.reconcile(bp, 20.0, 18.0, text,
                              program_code=code, rerun_program=rerun)
    check(rec.stage == reconcile.REPAIRED_PROGRAM and rec.certified
          and rec.answer == 20.0 and len(calls) == 1,
          "a program repair that restores agreement is certified")

    rec = reconcile.reconcile(bp, 20.0, 18.0, text, program_code=code,
                              rerun_program=lambda s: 999.0)
    check(rec.stage == reconcile.UNRESOLVED,
          "a program repair that does NOT restore agreement is discarded")

    rec = reconcile.reconcile(bp, 20.0, 18.0, text, program_code=code,
                              rerun_program=lambda s: (_ for _ in ()).throw(RuntimeError("boom")))
    check(rec.stage == reconcile.UNRESOLVED,
          "a crashing re-execution does not crash reconciliation")


# =====================================================================
# PART 5 — adjudication by a third derivation
# =====================================================================

def part5():
    head("PART 5 — the third derivation adjudicates, it does not vote")

    ans, strat, cert = reconcile.adjudicate(76.0, 74.0, 74.0)
    check((ans, strat, cert) == (74.0, "third_confirms_blueprint", True),
          "an independent program confirming the equations certifies them")

    ans, strat, cert = reconcile.adjudicate(76.0, 74.0, 76.0)
    check(ans == 76.0 and strat == "third_confirms_program" and not cert,
          "two programs agreeing is corroboration but NOT a cross-language certificate")

    ans, strat, cert = reconcile.adjudicate(76.0, 74.0, 99.0)
    check(ans == 76.0 and strat == "program_default" and not cert,
          "three-way disagreement keeps the program, uncertified")

    ans, strat, cert = reconcile.adjudicate(76.0, 74.0, None)
    check(ans == 76.0 and strat == "program_default" and not cert,
          "a failed third derivation keeps the program")

    ans, _, cert = reconcile.adjudicate(None, 74.0, 74.0)
    check(ans == 74.0 and cert,
          "no program at all: the confirmed blueprint still ships")

    # The blueprint can never win on its own say-so.
    ans, _, _ = reconcile.adjudicate(76.0, 74.0, None)
    check(ans != 74.0, "the blueprint never overrides the program unconfirmed")


# =====================================================================
# PART 6 — solver wiring (stubbed clients; no models, no network)
# =====================================================================

def part6():
    head("PART 6 — wiring inside Mas_solver")

    from Mas_solver import (QualityEnhancedMultiAgentSolver, AgentResponse,
                            SOLVER_VERSION, _write_reconcile_sidecar)

    check(SOLVER_VERSION.startswith("17."), f"SOLVER_VERSION is {SOLVER_VERSION}")

    solver = object.__new__(QualityEnhancedMultiAgentSolver)
    solver.decoupled_programmer = True
    solver.enable_internal_baseline = False
    solver.enable_reconcile = True
    solver.enable_operand_repair = True
    solver.enable_third_derivation = True
    solver.third_derivation_route = "v2_backward"
    solver.legacy_sht = False
    solver.prog_temp = 0.05
    solver.enable_metamorphic_testing = False

    # --- the prompt must not contain the blueprint when decoupled ----------
    seen = {}

    class _Client:
        def call_model(self, msgs, **kw):
            seen["user"] = msgs[-1]["content"]
            seen["system"] = msgs[0]["content"]
            return "```python\ngivens = {'a': 2}\nanswer = givens['a'] + 3\nprint(answer)\n```\nANSWER: [[5]]"

        def __str__(self):
            return "stub"

    solver._get_client = lambda role: _Client()
    blueprint = {"givens": {"secret_operand": 123456},
                 "equations": ["answer = givens['secret_operand']"],
                 "unknown": "x", "solution_steps": ["step"]}

    resp = solver.run_programmer_solver("Add 2 and 3.", blueprint, show_blueprint=False)
    check("123456" not in seen["user"] and "COLLEAGUE" not in seen["user"],
          "decoupled: no part of the blueprint reaches the Programmer's prompt")
    check(resp.agent == "Programmer (independent)", "and it is labelled as independent")
    check(resp.quality_metrics.get("code", "").strip().endswith("print(answer)"),
          "the full program source is kept for the program-side repair")

    check("colleague" not in seen["user"].lower(), "no dangling draft rule in the user message")
    check("colleague" not in seen["system"].lower(),
          "and none in the system message either — telling the model to "
          "cross-check a draft that is not there cost 27 of 150 problems in the "
          "first v17 run")

    solver.run_programmer_solver("Add 2 and 3.", blueprint, show_blueprint=True)
    check("123456" in seen["user"], "coupled mode still shows the draft (ablation intact)")
    check("colleague" in seen["system"].lower(),
          "coupled mode DOES get the draft rule, so both ablation arms are "
          "internally consistent")

    # an empty blueprint must NOT stop an independent Programmer
    r = solver.run_programmer_solver("Add 2 and 3.", {}, show_blueprint=False)
    check(r.answer == "5.0" and r.agent == "Programmer (independent)",
          "a dead Architect costs one derivation, not both")
    r = solver.run_programmer_solver("Add 2 and 3.", {}, show_blueprint=True)
    check(r.agent == "Programmer (empty blueprint)",
          "coupled mode still fails fast on an empty blueprint")

    # --- the SymPy fallback must not re-couple the derivations -------------
    class _DeadClient:
        def call_model(self, msgs, **kw):
            return "no code here"

        def __str__(self):
            return "dead"

    solver._get_client = lambda role: _DeadClient()
    bp2 = {"givens": {"a": 2.0, "b": 3.0}, "equations": ["answer = givens['a'] + givens['b']"]}
    r = solver.run_programmer_solver("Add 2 and 3.", bp2, max_attempts=1, show_blueprint=False)
    check("SymPy" not in r.agent,
          "decoupled: a failed program never falls back to the blueprint's equations")
    r = solver.run_programmer_solver("Add 2 and 3.", bp2, max_attempts=1, show_blueprint=True)
    check("SymPy" in r.agent, "coupled: the historical SymPy fallback still runs")

    # [v17.1] A failure must leave something to diagnose. The first v17 run lost
    # 27 problems here and recorded nothing about why.
    solver._get_client = lambda role: _NoCodeClient()

    class _NoCodeClient:
        def call_model(self, msgs, **kw):
            return "The answer is probably around forty-two, no code though."

        def __str__(self):
            return "nocode"

    solver._get_client = lambda role: _NoCodeClient()
    r = solver.run_programmer_solver("Add 2 and 3.", {}, max_attempts=2,
                                     show_blueprint=False)
    check(r.agent == "Programmer (failed)" and "failure_reason" in r.quality_metrics
          and "failed_code" in r.quality_metrics
          and r.quality_metrics["all_calls_errored"] is False,
          "a failed Programmer records why it failed, not just that it did")

    # --- the LLM repair loop has its OWN flag ------------------------------
    # It must NOT ride on decoupled_programmer, or MAS_COUPLED_PROGRAMMER=1
    # would change two things at once and the coupling ablation would measure
    # their sum instead of the coupling.
    solver.enable_llm_blueprint_repair = False
    check(not solver._llm_blueprint_repair_allowed(),
          "the LLM blueprint-repair loop is off by default (it would forge agreement)")
    solver.decoupled_programmer = False
    check(not solver._llm_blueprint_repair_allowed(),
          "and coupling the Programmer does NOT switch it back on")
    solver.enable_llm_blueprint_repair = True
    check(solver._llm_blueprint_repair_allowed(), "only its own flag enables it")
    solver.enable_llm_blueprint_repair = False
    solver.decoupled_programmer = True

    # --- _reconcile_and_decide, end to end --------------------------------
    class _SIV:
        blueprint_answer = 74.0

    prog = AgentResponse(agent="Programmer (independent)", answer="76", parsed="76",
                         confidence=1.0, reasoning_trace="",
                         quality_metrics={"code": "givens = {'x': 4}\nprint(76)\n"})

    third_calls = [0]

    def _fake_third(problem, bp, show_blueprint=True, strategy_hint="", agent_label=None,
                    max_attempts=3):
        third_calls[0] += 1
        check(show_blueprint is False, "the third derivation is also decoupled")
        check(bool(strategy_hint), "the third derivation gets a different route")
        return AgentResponse(agent="Programmer (third derivation)", answer="74",
                             parsed="74", confidence=1.0, reasoning_trace="",
                             quality_metrics={"code": ""})

    solver.run_programmer_solver = _fake_third
    grounded = {"givens": {"a": 40.0, "b": 8.0},
                "equations": ["answer = givens['a'] * givens['b']"]}
    ans, fb, rec, third, calls = solver._reconcile_and_decide(
        "Add 40 and 8.", grounded, prog, _SIV(), 2)
    check(third_calls[0] == 1 and calls == 3,
          "a structural disagreement costs exactly one extra call")
    check(rec.final_strategy == "third_confirms_blueprint" and rec.certified
          and ans == "74.0", "and the confirmed blueprint ships")
    check(fb is False, "used_baseline_fallback is False -- there is no baseline")

    # agreement must cost nothing extra
    third_calls[0] = 0
    prog2 = AgentResponse(agent="Programmer (independent)", answer="320", parsed="320",
                          confidence=1.0, reasoning_trace="", quality_metrics={})

    class _SIV2:
        blueprint_answer = 320.0

    ans, _, rec, _, calls = solver._reconcile_and_decide(
        "Add 40 and 8.", grounded, prog2, _SIV2(), 2)
    check(third_calls[0] == 0 and calls == 2 and rec.certified and ans == "320.0",
          "agreement spends no further budget")

    # no SIV at all (the mas_no_siv ablation) degrades to the program alone
    ans, _, rec, _, calls = solver._reconcile_and_decide(
        "Add 40 and 8.", grounded, prog2, None, 2)
    check(rec.stage == reconcile.PROGRAM_ONLY and ans == "320.0" and calls == 2,
          "without SIV the system is one independent program, and says so")

    # --- Programmer liveness: v17's load-bearing agent ---------------------
    from Mas_solver import DeadAgentError, DEAD_AGENT_THRESHOLD

    class _ErrClient:
        def call_model(self, msgs, **kw):
            return "ERROR_GENERATION: model did not respond"

        def __str__(self):
            return "err"

    live = QualityEnhancedMultiAgentSolver.__new__(QualityEnhancedMultiAgentSolver)
    live.prog_temp = 0.05
    live.enable_metamorphic_testing = False
    live._get_client = lambda role: _ErrClient()
    raised = False
    for _ in range(DEAD_AGENT_THRESHOLD):
        try:
            live.run_programmer_solver("p", {}, max_attempts=1, show_blueprint=False)
        except DeadAgentError:
            raised = True
    check(raised, f"a Programmer erroring on {DEAD_AGENT_THRESHOLD} problems aborts the run")

    live._get_client = lambda role: _Client()
    live.run_programmer_solver("Add 2 and 3.", {}, show_blueprint=False)
    check(live._prog_dead_streak == 0, "one good answer clears the liveness streak")

    # --- the sidecar never raises -----------------------------------------
    import tempfile, os as _os, json as _json
    with tempfile.TemporaryDirectory() as d:
        path = _os.path.join(d, "trace.jsonl")
        _os.environ["RECONCILE_SIDECAR_PATH"] = path
        try:
            _write_reconcile_sidecar("problem text", "76", rec, 3, grounded, prog, None)
            with open(path, encoding="utf-8") as fh:
                rec_json = _json.loads(fh.readline())
            check(rec_json["reconcile"]["stage"] == reconcile.PROGRAM_ONLY
                  and rec_json["program_code"].startswith("givens ="),
                  "the sidecar records the stage AND the program source")
            _os.environ["RECONCILE_SIDECAR_PATH"] = _os.path.join(d, "nope", "x.jsonl")
            _write_reconcile_sidecar("p", "e", rec, 1, grounded, prog, None)
            check(True, "an unwritable sidecar path does not kill the run")
        finally:
            _os.environ.pop("RECONCILE_SIDECAR_PATH", None)


# =====================================================================
# PART 7 — the whole of solve(), with stubbed models
# =====================================================================
# Everything above tests a piece. This runs the real entry point end to end,
# because the failures that actually cost GPU-hours in this project were never
# logic errors in a mechanism -- they were a name that did not exist on a
# branch nobody executed offline first.

def part7():
    head("PART 7 — solve() end to end, and the CSV the notebook will write")

    import json as _json
    import os as _os
    import tempfile
    from Mas_solver import QualityEnhancedMultiAgentSolver, AgentRole

    TEXT = ("Two trains travel 80 miles westward, then 150 miles northward. "
            "What is the total distance?")
    # Structurally wrong equations whose operands are all in the text, so no
    # single-operand hypothesis can explain the disagreement.
    BP = {"reasoning": "r", "unknown": "d", "givens": {"west": 80, "north": 150},
          "equations": ["answer = givens['west'] * givens['north']"],
          "solution_steps": ["s"], "expected_answer": "12000",
          "distractor_check": "None"}

    class Stub:
        provider = "stub"
        model_name = "stub-7b"

        def __init__(self, kind):
            self.kind, self.calls, self.hinted = kind, 0, []

        def __str__(self):
            return "stub:" + self.kind

        def call_model(self, msgs, **kw):
            self.calls += 1
            user = msgs[-1]["content"]
            if self.kind == "math":
                return _json.dumps(BP)
            if self.kind == "prog":
                self.hinted.append("APPROACH:" in user)
                if "APPROACH:" in user:          # the third derivation
                    return ("```python\ngivens = {'w': 80, 'n': 150}\n"
                            "answer = givens['w'] * givens['n']\nprint(answer)\n```")
                return ("```python\ngivens = {'w': 80, 'n': 150}\n"
                        "answer = givens['w'] + givens['n']\nprint(answer)\n```")
            return "ANSWER: [[999]]"

    kinds = {AgentRole.MATHEMATICIAN: "math", AgentRole.PROGRAMMER: "prog"}

    def run(**attrs):
        clients = {r: Stub(kinds.get(r, "other")) for r in AgentRole}
        s = QualityEnhancedMultiAgentSolver(clients=clients)
        for k, v in attrs.items():
            setattr(s, k, v)
        return s.solve(TEXT, "230"), clients

    with tempfile.TemporaryDirectory() as d:
        _os.environ["RECONCILE_SIDECAR_PATH"] = _os.path.join(d, "t.jsonl")
        try:
            out, clients = run()
            mas, sht = out["mas"], out.get("sht", {})
            cands = {f"cand_{c['id']}": c["answer"] for c in sht.get("candidates", [])}

            check(out["baseline"]["answer"] is None,
                  "no internal baseline ran, and the key still exists for the notebook")
            check(clients[AgentRole.BASELINE].calls == 0,
                  "the baseline role is never called")
            check(mas["llm_calls"] == 3 and sht["api_calls_used"] == 3,
                  "a disagreement costs 3 calls: architect, program, third")
            check(clients[AgentRole.PROGRAMMER].hinted == [False, True],
                  "the third program is prompted along a different route")
            check(sht["triage_result"] == "unresolved"
                  and sht["final_strategy"] == "third_confirms_blueprint",
                  "the reconciliation decision reaches the sht_* columns")
            check(set(cands) == {"cand_program", "cand_blueprint", "cand_third"},
                  "the CSV gets the new candidate columns, not v16's names")
            check(cands["cand_program"] == 230.0 and cands["cand_blueprint"] == 12000.0,
                  "and they carry the two derivations' actual values")
            check(out["reconcile"]["certified"] is True,
                  "the reconcile block is serialisable and populated")

            line = _json.loads(open(_os.environ["RECONCILE_SIDECAR_PATH"],
                                    encoding="utf-8").readline())
            check(line["program_code"].startswith("givens ="),
                  "the sidecar captured the program source solve() actually ran")
            check(line["third_code"] is not None,
                  "and the third derivation's source too")

            # No third derivation: the program stands, uncertified.
            out2, c2 = run(enable_third_derivation=False)
            check(out2["mas"]["answer"] == "230.0"
                  and out2["mas"]["reconcile_strategy"] == "program_default"
                  and out2["mas"]["certified"] is False
                  and out2["mas"]["llm_calls"] == 2,
                  "MAS_NO_THIRD: 2 calls, the program ships, nothing is certified")

            # The COUPLING ABLATION (run 2): v17 in every respect except that
            # the Programmer reads the blueprint again. Everything else must
            # stay put, or the ablation measures a sum instead of the coupling.
            out4, c4 = run(decoupled_programmer=False)
            check(out4["baseline"]["answer"] is None
                  and out4["mas"]["decoupled_programmer"] is False
                  and out4["mas"]["reconcile_stage"] != "",
                  "the coupling ablation changes ONLY the coupling")
            check(c4[AgentRole.BASELINE].calls == 0,
                  "and still runs no internal baseline")

            # Legacy v16 path still runs, with its own columns.
            out3, c3 = run(decoupled_programmer=False, enable_internal_baseline=True,
                           legacy_sht=True, enable_reconcile=False)
            l_cands = {f"cand_{c['id']}" for c in out3.get("sht", {}).get("candidates", [])}
            check(out3["baseline"]["answer"] == "999"
                  and "cand_primary" in l_cands
                  and out3["mas"]["reconcile_stage"] == "",
                  "the v16 pipeline is still reachable for the ablation table")
        finally:
            _os.environ.pop("RECONCILE_SIDECAR_PATH", None)


def main() -> int:
    part1(); part2(); part3(); part4(); part5(); part6(); part7()
    print("\n" + "=" * 72)
    if FAILS:
        print(f"FAILURES ({len(FAILS)} of {CHECKS[0]} checks):")
        for f in FAILS:
            print("   " + f)
        return 1
    print(f"ALL {CHECKS[0]} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
