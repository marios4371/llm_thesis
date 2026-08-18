"""
Blueprint ensembling + coverage tests ([v14.8])
===============================================

Two independent changes are guarded here.

COVERAGE. The blueprint's CAS value is a property of the blueprint, not of the
Programmer, but solve() only computed it inside a block gated on the Programmer
having succeeded. Measured on mas_full_20260817: 98/150 problems ever produced
a blueprint value, and 28 of the 52 missing ones had a blueprint that evaluates
perfectly well -- every one blocked by `"SymPy" not in agent`. Those 28 also
lost their free `blueprint_eval` candidate.

ENSEMBLING. `run_mathematician_ensemble` derives k blueprints along different
reasoning routes and keeps the value the most routes agree on. Diversity comes
from the prompt because the local backend decodes greedily (do_sample=False).

No LLM is called: the Mathematician is replaced by a scripted client that
returns a chosen blueprint per route.

Run:
    python test_blueprint_ensemble.py
"""

import json
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, '.')

from Mas_solver import (
    QualityEnhancedMultiAgentSolver, AgentRole, BLUEPRINT_STRATEGY_HINTS,
)

_FAILURES = []


def check(label, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {label}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        _FAILURES.append(label)


def _bp_json(givens, equations):
    return json.dumps({
        "unknown": "x", "givens": givens, "solution_steps": [],
        "equations": equations, "expected_answer": "0", "distractor_check": "None",
    })


class ScriptedClient:
    """Returns the next scripted blueprint on each Mathematician call."""
    provider = "fake"
    model_name = "fake"

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0
        self.prompts = []

    def call_model(self, messages, **kw):
        self.prompts.append(messages[0]["content"])
        out = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return out


def solver_with(payloads):
    c = ScriptedClient(payloads)
    s = QualityEnhancedMultiAgentSolver(clients={r: c for r in AgentRole})
    return s, c


print("=" * 70)
print("BLUEPRINT ENSEMBLING")
print("=" * 70)

# Three routes agree on 12, two dissent. Support must be 3 and the chosen
# blueprint must be one of the agreeing ones.
AGREE = _bp_json({"a": 4, "b": 3}, ["answer = givens['a'] * givens['b']"])       # 12
OTHER = _bp_json({"a": 4, "b": 3}, ["answer = givens['a'] + givens['b']"])       # 7
THIRD = _bp_json({"a": 4, "b": 3}, ["answer = givens['a'] - givens['b']"])       # 1

print("\n[1] Majority agreement across routes")
s, c = solver_with([AGREE, OTHER, AGREE, AGREE, THIRD])
bp, tel = s.run_mathematician_ensemble("Q?", 5)
check("called the model once per route", c.calls == 5, f"calls={c.calls}")
check("all five evaluated", tel["n_evaluable"] == 5, f"{tel['n_evaluable']}")
check("support counts the agreeing routes", tel["support"] == 3, f"support={tel['support']}")
check("picked the agreed value", tel["value"] == 12.0, f"value={tel['value']}")
check("three distinct values seen", tel["n_distinct"] == 3, f"{tel['n_distinct']}")
check("chosen blueprint evaluates to the agreed value",
      "*" in bp["equations"][0], f"eqs={bp.get('equations')}")

print("\n[2] Every route disagrees -> support 1, still returns a usable blueprint")
s, c = solver_with([AGREE, OTHER, THIRD])
bp, tel = s.run_mathematician_ensemble("Q?", 3)
check("support is 1", tel["support"] == 1, f"support={tel['support']}")
check("three distinct values", tel["n_distinct"] == 3, f"{tel['n_distinct']}")
check("a blueprint is still returned", bool(bp.get("equations")))

print("\n[3] Nothing evaluable -> degrades to the production route")
EMPTY = json.dumps({"unknown": "x", "givens": {}, "solution_steps": [],
                    "equations": [], "expected_answer": "0", "distractor_check": ""})
s, c = solver_with([EMPTY, EMPTY, EMPTY])
bp, tel = s.run_mathematician_ensemble("Q?", 3)
check("no evaluable blueprints reported", tel["n_evaluable"] == 0)
check("support 0", tel["support"] == 0)
check("value is None", tel["value"] is None)
check("returns a dict rather than raising", isinstance(bp, dict))

print("\n[4] k is clamped to the number of defined routes")
s, c = solver_with([AGREE])
bp, tel = s.run_mathematician_ensemble("Q?", 99)
check("k clamped to the route table",
      tel["k"] == len(BLUEPRINT_STRATEGY_HINTS), f"k={tel['k']}")

print("\n[5] The routes really do send different prompts")
s, c = solver_with([AGREE])
s.run_mathematician_ensemble("Q?", len(BLUEPRINT_STRATEGY_HINTS))
check("every route produced a distinct system prompt",
      len(set(c.prompts)) == len(BLUEPRINT_STRATEGY_HINTS),
      f"{len(set(c.prompts))} distinct of {len(c.prompts)}")
check("route 0 is the untouched production prompt",
      "ADDITIONAL REQUIREMENT" not in c.prompts[0])

print("\n[6] Default is off — single-sample behaviour is unchanged")
s, c = solver_with([AGREE])
check("blueprint_samples defaults to 1", s.blueprint_samples == 1,
      f"got {s.blueprint_samples}")

print("\n" + "=" * 70)
print("COVERAGE: the blueprint evaluates even when the Programmer did not")
print("=" * 70)

# The v14.8 coverage path is a property of solve(); assert the invariant it
# relies on -- a blueprint's value never depends on who produced the answer.
from siv_module import SymbolicInverseVerifier as SIV

bp_ok = {"givens": {"a": 10, "b": 3}, "equations": ["answer = givens['a'] - givens['b']"]}
r_with = SIV.verify(bp_ok, 7.0)
r_without = SIV.verify(bp_ok, 0.0)
check("blueprint value is identical regardless of the probed answer",
      r_with.blueprint_answer == r_without.blueprint_answer == 7.0,
      f"{r_with.blueprint_answer} vs {r_without.blueprint_answer}")
check("audit still distinguishes the two",
      r_with.execution_audit_passed and not r_without.execution_audit_passed,
      f"{r_with.execution_audit_passed}/{r_without.execution_audit_passed}")

print("\n" + "=" * 70)
if _FAILURES:
    print(f"FAILED ({len(_FAILURES)}): {_FAILURES}")
    sys.exit(1)
print("ALL PASSED")
