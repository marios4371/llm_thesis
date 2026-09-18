"""
[v17.0] Reconciliation of two genuinely independent derivations.

WHY THIS MODULE EXISTS
----------------------
v16.4 found that SIV's audit was checking its own reflection: the Programmer
wrote its code FROM the blueprint being audited, so `execution_rel_error` was
exactly zero on 81% of rows. That is the VERIFICATION side of a coupling
defect. This module exists because the same defect sits on the GENERATION
side, and it is larger.

Measured on seed 44 (`mas_sht_math7b_20260912_094311.csv` paired row-for-row
with `b_pal_20260915.csv`, grader rel 1e-4):

    Programmer shown the blueprint as a "colleague's draft"   79.3%
      its answer equals the blueprint's CAS value on          141/150 rows
      oracle{programmer, blueprint}                           79.3%   (+0.0)

    same model, PAL prompt, blueprint NOT shown               84.0%
      oracle{program, blueprint}                              88.0%   (+4.0)
      both wrong on                                            18/150 rows

The v12.0 instruction ("use it to double-check, but if it conflicts with the
PROBLEM, follow the PROBLEM") did not survive contact with a 7B model: the
Programmer transcribes. So the system had ONE derivation wearing two hats, and
their agreement carried no information -- which is the sufficient explanation
for why 2880 selector rules and two classifiers all failed held-out
(v15.8) and why the SIV certificate tracked problem difficulty rather than
correctness (v14.6). There was nothing to select between.

Decoupling the Programmer restores a second opinion. This module is what to do
with it.

THE MECHANISM
-------------
Two derivations of the same problem in two different languages:

    declarative   Architect writes equations        -> evaluated by SymPy
    imperative    Programmer writes a program       -> executed by Python

Neither sees the other. When they produce the same number, that is a
certificate with real content. When they produce different numbers, the
literature asks "which one is right" -- by voting (defeated here by candidate
correlation, v15.8) or by an LLM judge (defeated here because a 7B judges
worse than it solves, v12.0/v13.0). This module asks a different question,
one that algebra can answer without any model:

    assuming the OTHER derivation is right, what value would each given
    inside the equations have had to take?

SIV Layer 2 already computes exactly that by inverting the equation chain. It
was useless before because the reference was the blueprint's own output, so
the inversion returned the declared values by construction. With an
independent reference the reconstructed values become real evidence, and the
problem text adjudicates:

    the problem says      20 broken lights
    the Architect wrote   22
    the program answered  76
    the equations give    74
    inverting from 76 asks for   20      <- a number the text contains

That is no longer opinion against opinion. It is a named finding: the
Architect miscopied one operand. Substituting it makes the two derivations
agree, and the agreement is then earned rather than assumed.

MEASURED, OFFLINE, ON THE FOUR STORED RUNS (510 evaluable rows)
---------------------------------------------------------------
Rows where exactly one given is absent from the problem text AND its
reconstruction from the independent reference IS a text number:

    n = 23      reference correct 95.7%      blueprint correct 0.0%
    substituting the reconstructed value: 23/23 reach exact agreement

These are rows the FORWARD snap of v15.0-15.2 could not reach -- the stored
givens are already post-repair, so the snap had abstained on them. Inversion
resolves the ambiguity the snap correctly refuses to guess at, because it
supplies the missing constraint: the value the rest of the chain needs.

The symmetric case (the PROGRAM misread an operand) is rare: 4 rows in 510
under a >=4-digit criterion, because the program's errors are structural
rather than transcriptional. It is implemented anyway -- it costs nothing and
is corroboration-gated -- but no accuracy is claimed for it.

WHAT THIS IS NOT
----------------
Not a selector. Nothing here ever overrides one derivation with the other on
the strength of a score. A repair only ships when it CREATES agreement between
two independently produced answers; if it does not, it is discarded and the
disagreement stands. The failure mode is therefore "no certificate", never a
silent overwrite -- which is the property every previous mechanism in this
project lacked and the reason they all had to be protected by a do-no-harm
anchor borrowed from the baseline.

Zero LLM calls. Everything here is arithmetic on values that already exist.

Prior art, delimited (WebSearch 2026-09-17):
  * MathPrompter (2303.05398) generates an algebraic expression AND Python
    from ONE model and accepts only exact agreement; on disagreement it
    resamples. No attribution, no repair, one model.
  * Automatic Model Selection (2305.14333) resolves CoT-vs-PAL disagreement
    with an LLM selector, which needs a GPT-4-class judge.
  * FOBAR (2308.07758) masks a number and asks the LLM to recover it. Same
    intuition, but the recovery is another sample from the model; here it is
    a symbolic solve, deterministic, over every given at once, and the
    recovered value is checked against the problem text rather than scored.
  * 2410.11781 shows LLM numeric errors are digit-level rather than
    magnitude-level, which is why a single-operand hypothesis is the right
    one to test.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import near_agreement
from blueprint_repair import (
    _STRUCTURAL_CONSTANTS,
    _snap_givens_to_text,
    _text_numbers,
)

logger = logging.getLogger(__name__)

# The agreement band. Identical to near_agreement.DEFAULT_TAU and to
# siv_module.TOLERANCE_REL -- one number, three places, deliberately.
# v16.3 measured the effect flat from 1e-6 to 1e-3 and broken at 1e-2.
DEFAULT_TAU = near_agreement.DEFAULT_TAU

# How close a reconstructed value must sit to a number in the problem text to
# count as "the text contains it". This is an IDENTITY test between two
# literals, not a comparison of two computations, so it is tight enough to
# admit only floating-point noise from the symbolic solve.
#
# It has to be this tight, and the reason is the whole safety argument of this
# module. At 1e-6 relative, two 9-digit numbers count as "the same" when they
# differ by nine units, and gsm-hard operands are 7-9 digits. Under that
# tolerance the attribution stops testing whether the text contains the
# reconstructed value and starts accepting any value near it -- which lets the
# repair fudge one operand into whatever makes the equations match the other
# derivation. That is the coupling defect of v16.4 rebuilt by hand: the
# blueprint would become conditioned on the program's answer, and the
# agreement that follows would prove nothing.
#
# Caught on a real row: gsm-hard_1266 reconstructs cable_package_cost as
# 9202358.4 while the problem states 9202361. At 1e-6 that passed and the
# "repair" shipped a number no one wrote down. At 1e-9 it is refused, which is
# correct -- a second given in that chain is also wrong, so no single-operand
# hypothesis explains the disagreement.
TEXT_MATCH_REL = 1e-9


# =====================================================================
# Stage names (stable strings -- they reach the CSV via sht_triage)
# =====================================================================
AGREE = "agree"                                   # certified, no repair needed
REPAIRED_BLUEPRINT = "repaired_blueprint_operand"  # certified after one substitution
REPAIRED_PROGRAM = "repaired_program_operand"      # certified after one substitution
UNRESOLVED = "unresolved"                          # genuine structural disagreement
PROGRAM_ONLY = "program_only"                      # the equations produced no value
BLUEPRINT_ONLY = "blueprint_only"                  # the program produced no value
NOTHING = "nothing"                                # neither side produced a value


@dataclass
class OperandRepair:
    """One named operand, on one named side, with the value the other
    derivation implies it should have had."""
    side: str                 # 'blueprint' | 'program'
    given: str
    declared: float
    reconstructed: float

    def as_dict(self) -> Dict[str, Any]:
        return {"side": self.side, "given": self.given,
                "declared": self.declared, "reconstructed": self.reconstructed}


@dataclass
class Reconciliation:
    """What the two derivations, taken together, licence us to say."""
    stage: str
    answer: Optional[float] = None
    certified: bool = False
    gap: Optional[float] = None               # relative gap BEFORE any repair
    repair: Optional[OperandRepair] = None
    blueprint_after: Optional[dict] = None    # repaired blueprint, if any
    program_code_after: Optional[str] = None  # repaired program source, if any
    rejected: List[OperandRepair] = field(default_factory=list)
    detail: str = ""
    # Filled in by the caller when a structural disagreement escalates to a
    # third derivation; `final_strategy` is what actually decided the answer
    # and is the string that reaches the CSV.
    third_value: Optional[float] = None
    final_strategy: str = ""

    @property
    def repaired(self) -> bool:
        return self.stage in (REPAIRED_BLUEPRINT, REPAIRED_PROGRAM)

    def as_dict(self) -> Dict[str, Any]:
        """Flat, JSON-safe view for the run sidecar."""
        return {
            "stage": self.stage,
            "final_strategy": self.final_strategy or self.stage,
            "answer": self.answer,
            "certified": self.certified,
            "gap": self.gap,
            "third_value": self.third_value,
            "repair": self.repair.as_dict() if self.repair else None,
            "rejected": [r.as_dict() for r in self.rejected],
            "detail": self.detail,
        }


# =====================================================================
# Numeric helpers
# =====================================================================

def _num(x: Any) -> Optional[float]:
    """float(x) for anything that is a finite real number, else None."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and v not in (float("inf"), float("-inf")) else None


def agrees(a: Optional[float], b: Optional[float], tau: float = DEFAULT_TAU) -> bool:
    """True when two derivations landed on the same number up to `tau`.

    Includes exact equality, unlike near_agreement.structurally_corroborated,
    whose strict `gap > 0` was needed because its two inputs came from the SAME
    derivation and exact agreement there was vacuous. Here the inputs come from
    different derivations in different languages, so exact agreement is the
    strongest evidence available, not the weakest.
    """
    gap = near_agreement.relative_gap(a, b)
    return gap is not None and gap < tau


def _in_text(value: Optional[float], text_nums: List[float]) -> bool:
    """Does the problem statement literally contain this number?"""
    v = _num(value)
    if v is None:
        return False
    return any(abs(v - t) <= max(1e-9, TEXT_MATCH_REL * abs(t)) for t in text_nums)


def _grounded(value: Optional[float], text_nums: List[float]) -> bool:
    """A given is grounded when the text contains it, or it is a structural
    constant the text never needed to state (a half, a dozen, 100 for a
    percentage). Same predicate blueprint_repair.operands_grounded uses, so
    the two cannot drift apart."""
    v = _num(value)
    if v is None:
        return False
    return _in_text(v, text_nums) or v in _STRUCTURAL_CONSTANTS


# =====================================================================
# Blueprint side: attribute a disagreement to one declared operand
# =====================================================================

def attribute_blueprint_slip(blueprint: dict,
                             reference: float,
                             problem_text: str,
                             verify: Optional[Callable] = None
                             ) -> Tuple[List[OperandRepair], List[OperandRepair]]:
    """Invert the equation chain from `reference` and look for a single
    operand the Architect miscopied.

    Returns (accepted, rejected). `accepted` holds the repairs that satisfy the
    whole conjunction below; `rejected` holds near-misses, kept only so a run
    can be audited afterwards.

    A candidate must satisfy ALL of:
      1. the given is actually USED by the chain (SymPy could solve for it);
      2. its declared value is NOT in the problem text and is not a structural
         constant -- i.e. the Architect wrote a number from nowhere;
      3. the value the inversion asks for IS in the problem text;
      4. the two differ.

    Condition 3 is what makes this attribution rather than speculation. An
    innocent given's reconstructed value is an arbitrary algebraic byproduct
    with no reason to coincide with a number the problem states; the guilty
    one's is the number that was there all along. This is the BACKWARD variant
    of text grounding that has been open in this project since v12.2 -- the
    forward variant shipped as v15.0 and could not reach these rows, because
    forward snapping abstains exactly when two text numbers are equally close,
    and the inversion supplies the constraint that breaks the tie.

    The given literally named 'answer' is skipped: some blueprints store the
    result as a given, and "reconstructing" it is circular.
    """
    if verify is None:                      # imported lazily: siv_module pulls in SymPy
        from siv_module import SymbolicInverseVerifier
        verify = SymbolicInverseVerifier.verify

    text_nums = _text_numbers(problem_text)
    accepted: List[OperandRepair] = []
    rejected: List[OperandRepair] = []
    if not text_nums:
        return accepted, rejected

    try:
        res = verify(blueprint, reference)
    except Exception as exc:                # a malformed chain is not our problem here
        logger.debug("attribution: SIV.verify raised %s", exc)
        return accepted, rejected

    for rec in getattr(res, "reconstructions", []) or []:
        if not getattr(rec, "solvable", False):
            continue
        if str(getattr(rec, "name", "")).strip().lower() == "answer":
            continue
        declared = _num(getattr(rec, "original_value", None))
        wanted = _num(getattr(rec, "reconstructed_value", None))
        if declared is None or wanted is None:
            continue
        if agrees(declared, wanted):
            continue                        # this given already reconstructs
        cand = OperandRepair("blueprint", str(rec.name), declared, wanted)
        if _grounded(declared, text_nums):
            continue                        # the Architect copied this one correctly
        if _in_text(wanted, text_nums):
            accepted.append(cand)
        else:
            rejected.append(cand)
    return accepted, rejected


def apply_blueprint_repair(blueprint: dict, repair: OperandRepair) -> dict:
    """A copy of the blueprint with one given replaced. Never mutates."""
    out = dict(blueprint)
    out["givens"] = dict(blueprint.get("givens") or {})
    out["givens"][repair.given] = repair.reconstructed
    fixes = list(out.get("_deterministic_fixes") or [])
    fixes.append(
        f"inverse-grounded operand {repair.given}: "
        f"{repair.declared!r} -> {repair.reconstructed!r}"
    )
    out["_deterministic_fixes"] = fixes
    return out


# =====================================================================
# Program side: the same repair, applied to the program's givens dict
# =====================================================================

def snap_program_givens(code: str, problem_text: str
                        ) -> Tuple[Optional[str], List[OperandRepair]]:
    """Rewrite `givens = {...}` in the program so every operand is a number the
    problem text contains, using the v15.0 single-digit-slip rule.

    Returns (new_source, repairs) or (None, []) when nothing is rewritten. The
    caller must RE-EXECUTE the result and keep it only if the new answer agrees
    with the other derivation -- this function proposes, it does not decide.

    Relies on the `givens = {...}` first-line convention the Programmer prompt
    has required since v12.0, which is also what SIV's givens-matching needs.
    """
    if not code or not problem_text:
        return None, []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, []

    target: Optional[ast.Dict] = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "givens"
                and isinstance(node.value, ast.Dict)):
            target = node.value
            break
    if target is None:
        return None, []

    givens: Dict[str, float] = {}
    for k_node, v_node in zip(target.keys, target.values):
        if not isinstance(k_node, ast.Constant) or not isinstance(k_node.value, str):
            return None, []                 # a computed key: refuse the whole rewrite
        val = _literal_number(v_node)
        if val is None:
            return None, []                 # an expression: refuse (values may be derived)
        givens[k_node.value] = val
    if not givens:
        return None, []

    snapped, fixes = _snap_givens_to_text(givens, problem_text)
    repairs = [OperandRepair("program", k, givens[k], snapped[k])
               for k in givens if snapped.get(k) != givens[k]]
    if not repairs:
        return None, []

    literal = "{" + ", ".join(f"{k!r}: {v!r}" for k, v in snapped.items()) + "}"
    new_code = _splice(code, target, literal)
    if new_code is None:
        return None, []
    logger.info("[v17.0] program operand snap: %s",
                "; ".join(f"{r.given} {r.declared!r}->{r.reconstructed!r}" for r in repairs))
    return new_code, repairs


def _literal_number(node: ast.AST) -> Optional[float]:
    """The value of a numeric literal, including a negated one. None for
    anything else -- a call, a name, an arithmetic expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal_number(node.operand)
        return None if inner is None else -inner
    return None


def _splice(code: str, node: ast.AST, replacement: str) -> Optional[str]:
    """Replace the source span of `node` with `replacement`."""
    lines = code.splitlines(keepends=True)
    start, end = getattr(node, "lineno", None), getattr(node, "end_lineno", None)
    col, end_col = getattr(node, "col_offset", None), getattr(node, "end_col_offset", None)
    if None in (start, end, col, end_col) or end > len(lines):
        return None
    prefix = "".join(lines[:start - 1]) + lines[start - 1][:col]
    suffix = lines[end - 1][end_col:] + "".join(lines[end:])
    return prefix + replacement + suffix


# =====================================================================
# The reconciliation itself
# =====================================================================

def reconcile(blueprint: dict,
              blueprint_value: Optional[float],
              program_value: Optional[float],
              problem_text: str,
              program_code: Optional[str] = None,
              rerun_program: Optional[Callable[[str], Optional[float]]] = None,
              tau: float = DEFAULT_TAU,
              enable_repair: bool = True,
              verify: Optional[Callable] = None) -> Reconciliation:
    """Compare two independent derivations and say what is certified.

    `rerun_program` executes a patched program and returns its number, or None.
    It is injected rather than imported so this module stays free of
    Mas_solver and testable with no models, no GPU and no network.

    On agreement the SYMBOLIC value is returned, not the program's. Both are
    machine-computed, so neither slips the way mental arithmetic does, but
    SymPy evaluates exactly while a Python float loses its last digits on the
    8-digit operands that dominate gsm-hard. Inside a band of 1e-4 the two can
    only differ there, so taking the exact one is free.
    """
    bp_v, pr_v = _num(blueprint_value), _num(program_value)
    gap = near_agreement.relative_gap(bp_v, pr_v)

    if bp_v is None and pr_v is None:
        return Reconciliation(NOTHING, None, False, None,
                              detail="neither derivation produced a number")
    if bp_v is None:
        return Reconciliation(PROGRAM_ONLY, pr_v, False, None,
                              detail="the equations did not evaluate")
    if pr_v is None:
        return Reconciliation(BLUEPRINT_ONLY, bp_v, False, None,
                              detail="the program produced no number")

    if agrees(bp_v, pr_v, tau):
        return Reconciliation(
            AGREE, bp_v, True, gap,
            detail=f"two independent derivations agree (rel {gap:.2e})")

    if not enable_repair:
        return Reconciliation(UNRESOLVED, None, False, gap,
                              detail="repair disabled")

    # --- the Architect miscopied one operand ---------------------------------
    accepted, rejected = attribute_blueprint_slip(blueprint, pr_v, problem_text, verify)
    if len(accepted) == 1:
        repair = accepted[0]
        patched = apply_blueprint_repair(blueprint, repair)
        new_bp = _evaluate_blueprint(patched, pr_v, verify)
        if agrees(new_bp, pr_v, tau):
            logger.info(
                "[v17.0] inverse-grounded repair: the problem text has %r where the "
                "blueprint declared %r for %s; the equations now agree with the "
                "program at %r", repair.reconstructed, repair.declared,
                repair.given, new_bp)
            return Reconciliation(
                REPAIRED_BLUEPRINT, new_bp, True, gap, repair, patched,
                rejected=rejected,
                detail=(f"{repair.given}: {repair.declared!r} -> "
                        f"{repair.reconstructed!r}, agreement restored"))
        rejected.append(repair)             # proposed but did not produce agreement
    elif len(accepted) > 1:
        rejected.extend(accepted)           # ambiguous: refuse rather than guess

    # --- the Programmer miscopied one operand --------------------------------
    if program_code and rerun_program is not None:
        new_code, prog_repairs = snap_program_givens(program_code, problem_text)
        if new_code and len(prog_repairs) >= 1:
            try:
                new_pr = _num(rerun_program(new_code))
            except Exception as exc:
                logger.debug("program re-execution failed: %s", exc)
                new_pr = None
            if agrees(bp_v, new_pr, tau):
                repair = prog_repairs[0]
                return Reconciliation(
                    REPAIRED_PROGRAM, bp_v, True, gap, repair,
                    program_code_after=new_code, rejected=rejected,
                    detail=(f"program operand {repair.given}: {repair.declared!r} "
                            f"-> {repair.reconstructed!r}, agreement restored"))
            rejected.extend(prog_repairs)

    return Reconciliation(
        UNRESOLVED, None, False, gap, rejected=rejected,
        detail=f"structural disagreement (rel {gap:.2e}), no single operand explains it")


def _evaluate_blueprint(blueprint: dict, reference: float,
                        verify: Optional[Callable] = None) -> Optional[float]:
    """What the equations evaluate to. `reference` only steers SIV's inverse
    layer, which we ignore here -- the forward value does not depend on it."""
    if verify is None:
        from siv_module import SymbolicInverseVerifier
        verify = SymbolicInverseVerifier.verify
    try:
        return _num(verify(blueprint, reference).blueprint_answer)
    except Exception as exc:
        logger.debug("re-evaluation failed: %s", exc)
        return None


# =====================================================================
# The third derivation's verdict
# =====================================================================

def adjudicate(program_value: Optional[float],
               blueprint_value: Optional[float],
               third_value: Optional[float],
               tau: float = DEFAULT_TAU) -> Tuple[Optional[float], str, bool]:
    """Resolve a structural disagreement with one additional independent
    derivation. Returns (answer, strategy, certified).

    The rule is corroboration, never arbitration: a value ships as CERTIFIED
    only when two derivations produced in DIFFERENT languages agree on it.
    The third derivation is another program, so when it confirms the
    equations we again have a cross-language agreement -- the same certificate
    stage 1 issues, reached the long way round. When it confirms the first
    program instead, two programs agree and we keep the program's answer, but
    that is same-language corroboration and is deliberately NOT certified.

    With no majority the program's answer stands. That is the stronger of the
    two derivations on exactly this population: on the structurally
    disagreeing rows of the four stored runs the program is right about 60% of
    the time and the equations about 20%. Defaulting to it is the do-no-harm
    floor of this design, and unlike v13.0's it is a component of the system
    rather than a baseline borrowed from the comparison table.
    """
    pr, bp, th = _num(program_value), _num(blueprint_value), _num(third_value)
    if th is not None and bp is not None and agrees(th, bp, tau):
        return bp, "third_confirms_blueprint", True
    if th is not None and pr is not None and agrees(th, pr, tau):
        return pr, "third_confirms_program", False
    return pr, "program_default", False
