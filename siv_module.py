"""
Symbolic Inverse Verification (SIV) Module
===========================================
Novel contribution for MAS-SHT v10.0

Core Idea — Symbolic Execution Audit:
    Given a forward equation chain  G → E₁ → E₂ → ... → Eₙ → A
    where G = givens, E = equations, A = answer,

    SIV operates in two complementary layers:

    LAYER 1 — Execution Audit (forward):
        Substitute all numeric givens into the symbolic chain → blueprint_answer.
        Compare blueprint_answer to computed_answer from the Programmer.
        If they match: the Programmer faithfully executed the blueprint.
        If they diverge: execution error detected (wrong code, wrong arithmetic).

    LAYER 2 — Given Relevance / Fault Set (inverse):
        For each given gᵢ ∈ G, treat gᵢ as unknown, substitute all other givens
        numerically (at their DECLARED values), set result = computed_answer,
        solve for gᵢ.
        If solved_gᵢ ≈ actual_gᵢ → this given is consistent with the answer.
        If solved_gᵢ ≠ actual_gᵢ → this given is INCONSISTENT with the answer.

        [v12.2 — SCOPE CORRECTION, controlled fault-injection results in
        test_siv_fault_injection.py] Because every MAS-SHT blueprint collapses
        to ONE scalar expression `answer = f(g1, ..., gn)` (equations always
        terminate in a single `answer` assignment), holding every OTHER given
        fixed at its declared value and inverting for one given at a time
        does NOT isolate a unique faulty variable when 2+ givens are used:
        perturbing any single input generically makes EVERY used given's
        one-at-a-time reconstruction disagree, not just the corrupted one.
        Measured (n=37 controlled single-given corruptions, blueprints with
        2-5 used givens): recall (the true fault is always present in
        failed_givens) = 100%; exact isolation (failed_givens names ONLY the
        corrupted variable) = 0% whenever 2+ givens are used; mean precision
        = exactly 1/n_used (50% at n=2, 33% at n=3, 25% at n=4, 20% at n=5).
        Isolation is only exact in the degenerate n_used=1 case.
        What DOES work reliably (100% in the same test): separating used
        (chain-connected) givens from UNUSED ones (see unused_givens below) —
        i.e. Layer 2 is a validated relevance/distractor filter, and — when
        the execution audit fails — a validated SUPERSET of implicated
        variables (the true fault is never missed), not a point localizer.

Known Limitation — Translation Layer:
    SIV operates on the math→math layer (blueprint equations → computed answer).
    It CANNOT detect errors in the NL→math layer (problem text → blueprint).
    If the Architect (Mathematician) modelled the problem incorrectly, the blueprint
    equations will be self-consistent but wrong — and SIV will not catch this.
    This is an explicit limitation: SIV verifies execution fidelity, not
    problem-modelling correctness.

Comparison with FOBAR (Jiang et al., ACL 2024):
    FOBAR: LLM-based backward verification → operates partially on NL layer,
           but is probabilistic and inherits LLM errors. Gives binary verdict.
    SIV:   CAS-based, deterministic, zero LLM calls. Operates on symbolic-execution
           layer. Gives a structured multi-variable diagnostic (implicated-variable
           set + validated distractor exclusion), not just pass/fail — but, per the
           v12.2 scope correction above, does not isolate a single faulty variable
           among several used ones.
    The two are ORTHOGONAL: FOBAR targets translation errors; SIV targets execution
    errors. Their union is strictly stronger than either alone.

Theoretical basis:
    If f(x) = y, then f⁻¹(y) = x (invertibility / cycle consistency).
    SIV checks whether the equation chain is invertible and whether inverting
    from computed_answer recovers the declared givens — exposing any arithmetic
    deviation between the Programmer's code and the Architect's blueprint.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("MAS_Pipeline")

# --- SymPy Import ---
try:
    import sympy
    from sympy import Symbol, symbols, solve, Eq, sympify, N, oo
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


# =====================================================================
# Data Structures
# =====================================================================

@dataclass
class GivenReconstruction:
    """Result of reconstructing a single given variable via inverse solve."""
    name: str
    original_value: float
    reconstructed_value: Optional[float]
    match: bool                    # True if |original - reconstructed| < tolerance
    relative_error: Optional[float]  # |diff| / max(|original|, 1e-9)
    solvable: bool                 # Whether SymPy could solve for this variable
    ambiguous: bool = False        # True if multiple real roots exist (not uniquely invertible)
    all_real_roots: List[float] = field(default_factory=list)  # All real-valued solutions
    error_message: str = ""


@dataclass
class SIVResult:
    """
    Complete result of Symbolic Inverse Verification.

    Execution Audit Fields (Layer 1 — forward):
        execution_audit_passed: True if blueprint forward evaluation ≈ computed_answer
        blueprint_answer:       What the blueprint equations evaluate to (all givens numeric)
        execution_rel_error:    |blueprint_answer - computed_answer| / max(|computed_answer|, 1e-9)

    Fault Localization Fields (Layer 2 — inverse):
        verified:        True if ALL solvable givens were reconstructed correctly
        confidence:      0.0 - 1.0 based on proportion of matched solvable givens
        givens_matched:  Number of givens that reconstructed correctly
        givens_total:    Total number of numeric givens
        reconstructions: Per-given reconstruction details
        failed_givens:   Names of givens that did NOT reconstruct. [v12.2] This is
                         an IMPLICATED SET, not a point localization: per
                         test_siv_fault_injection.py, when 2+ givens are used,
                         a single corrupted variable causes ALL used givens to
                         appear here (precision = 1/n_used; the true fault is
                         never missing — recall = 100% — but is not isolated).
        unused_givens:   Names of givens declared in blueprint but absent from
                         equation chain. This IS a reliably validated distinction
                         (100% in controlled testing) — the genuine, working part
                         of Layer 2's diagnostic value.
        invertible:      Whether the equation chain was symbolically invertible at all

    Structural Defect Fields (Layer 0 — pre-algebraic, [v14.0]):
        undefined_symbols:     Names appearing on the right-hand side of an equation
                               that are neither a declared given nor assigned by an
                               earlier equation. Their presence makes the chain
                               unevaluable: forward substitution leaves a free symbol
                               and float() raises. Measured on the v13.0 run, 30/109
                               audited problems died exactly here (blueprint_answer
                               None, invertible False) — the single largest silent
                               SIV failure mode, and a precisely reportable one.
        undeclared_given_keys: Keys used as givens['<key>'] that are absent from the
                               blueprint's givens dict, or declared non-numerically.
                               Detected by a deterministic pre-parse scan, so this is
                               reported even when SymPy cannot parse the chain at all.

    Meta:
        verifies_translation: Always False. SIV cannot verify NL→math correctness.
        skipped:              True when SIV never ran (no equations / no numeric
                              givens / no SymPy). Distinguishes "SIV had nothing to
                              audit" from "SIV audited and the audit failed" — the
                              two are NOT interchangeable for repair triggering or
                              for coverage statistics.
        tautological:         True when the audited answer was itself produced by
                              evaluating this blueprint's own equations (e.g. the
                              SymPy symbolic fallback). Layer 1 then passes by
                              construction and carries zero verification evidence;
                              such rows must be excluded from detector statistics.
        trace:                Human-readable audit trace
    """
    # Layer 1: Execution audit
    execution_audit_passed: bool = False
    blueprint_answer: Optional[float] = None
    execution_rel_error: Optional[float] = None

    # Layer 2: Fault localization
    verified: bool = False
    confidence: float = 0.0
    givens_matched: int = 0
    givens_total: int = 0
    reconstructions: List[GivenReconstruction] = field(default_factory=list)
    failed_givens: List[str] = field(default_factory=list)
    unused_givens: List[str] = field(default_factory=list)
    invertible: bool = False

    # Layer 0: Structural defects [v14.0]
    undefined_symbols: List[str] = field(default_factory=list)
    undeclared_given_keys: List[str] = field(default_factory=list)

    # Meta
    verifies_translation: bool = False   # Explicit limitation marker — always False
    skipped: bool = False                # [v14.0] SIV had nothing to audit
    tautological: bool = False           # [v14.0] answer came from these same equations
    computed_answer: Optional[float] = None  # The Programmer's answer that was audited
    trace: str = ""

    @property
    def has_structural_defect(self) -> bool:
        """[v14.0] The blueprint is malformed as *code*, before any algebra."""
        return bool(self.undefined_symbols or self.undeclared_given_keys)


# =====================================================================
# Core SIV Engine
# =====================================================================

class SymbolicInverseVerifier:
    """
    Symbolic Inverse Verification (SIV) — Symbolic Execution Audit.

    What SIV DOES guarantee:
        1. Execution Fidelity: whether the Programmer's numeric answer is
           consistent with the Architect's symbolic blueprint equations.
        2. Fault Localization: for each given variable, whether inverting the
           equation chain from the computed answer recovers the declared value.
        3. Ambiguity Detection: whether the equation chain is uniquely invertible
           (single root) or admits multiple solutions.
        4. Unused-Given Detection: whether any declared givens are absent from
           the equation chain (distractors or missing equations).

    What SIV CANNOT guarantee:
        Blueprint–Problem Translation correctness. If the Architect modelled the
        problem wrongly (NL→math error), SIV will audit a wrong blueprint faithfully
        and may report 'execution_audit_passed=True' for an incorrect solution.
        This is the fundamental operating limit of the math→math verification layer.

    Analogy:
        SIV is closer to a symbolic execution sanity check than to CycleGAN cycle
        consistency. The inverse is an algebraic rearrangement of the same expression
        (not an independently-learned inverse), so it provides an orthogonal signal
        only when execution diverges from the blueprint — not when the blueprint
        itself is wrong.
    """

    TOLERANCE_ABS = 1e-6    # Absolute tolerance for float comparison
    TOLERANCE_REL = 1e-4    # Relative tolerance (0.01%)

    @staticmethod
    def verify(blueprint: dict, computed_answer: float) -> SIVResult:
        """
        Main entry point: Symbolic Execution Audit + Fault Localization.

        Args:
            blueprint:       Mathematician's blueprint with 'givens' and 'equations'
            computed_answer: The numeric answer from the Programmer/SymPy

        Returns:
            SIVResult with two-layer audit outcome.
            Note: result.verifies_translation is always False — SIV cannot
            detect errors in the NL→math translation layer.
        """
        if not SYMPY_AVAILABLE:
            return SIVResult(
                trace="SIV skipped: SymPy not available",
                verifies_translation=False,
                skipped=True,
            )

        givens = blueprint.get("givens", {})
        equations = blueprint.get("equations", [])

        # Filter to numeric givens only
        numeric_givens = {
            k: v for k, v in givens.items()
            if isinstance(v, (int, float))
        }

        if not numeric_givens:
            return SIVResult(
                trace="SIV skipped: no numeric givens in blueprint",
                verifies_translation=False,
                skipped=True,
            )

        if not equations:
            return SIVResult(
                trace="SIV skipped: no equations in blueprint",
                verifies_translation=False,
                skipped=True,
            )

        trace_lines = ["[SIV] Symbolic Execution Audit"]
        trace_lines.append(f"  Computed answer (Programmer): {computed_answer}")
        trace_lines.append(f"  Givens declared: {list(numeric_givens.keys())}")
        trace_lines.append(
            "  NOTE: SIV audits math→math execution fidelity only. "
            "Blueprint–problem translation errors are outside SIV's detection scope."
        )

        # ── Step 0: Structural scan (pre-algebraic) [v14.0] ───────────────────
        # Runs BEFORE SymPy so a malformed blueprint is reported precisely even
        # when the chain cannot be parsed at all. Both defects below make the
        # chain unevaluable, and both are mechanically fixable by the
        # Mathematician if named — unlike Layer 2's implicated-variable set.
        undeclared_keys = SymbolicInverseVerifier._scan_undeclared_given_keys(
            equations, givens, numeric_givens
        )
        undefined_syms = SymbolicInverseVerifier._scan_undefined_names(
            equations, numeric_givens
        )

        # ── Step 1: Build the symbolic forward chain ──────────────────────────
        symbolic_expr, given_symbols, residual_syms = \
            SymbolicInverseVerifier._build_symbolic_chain(
                equations, numeric_givens, trace_lines
            )
        # The source scan names the culprit; the SymPy residual only backs it up
        # when the scan found nothing (see _scan_undefined_names on why the
        # residual's names are unreliable under implicit multiplication).
        if not undefined_syms and residual_syms:
            undefined_syms = residual_syms

        if symbolic_expr is None:
            if undeclared_keys:
                trace_lines.append(
                    f"  [Structural] Undeclared given keys referenced: {undeclared_keys}"
                )
            return SIVResult(
                trace="\n".join(trace_lines) + "\n  FAILED: Could not build symbolic chain",
                execution_audit_passed=False,
                verifies_translation=False,
                givens_total=len(numeric_givens),
                invertible=False,
                undefined_symbols=undefined_syms,
                undeclared_given_keys=undeclared_keys,
                computed_answer=computed_answer,
            )

        trace_lines.append(f"  Symbolic blueprint expression: {symbolic_expr}")

        # A stray free symbol makes forward substitution unevaluable, which in
        # turn makes every inverse solve fail. Return the structural verdict
        # directly instead of burning a SymPy solve per given to rediscover it.
        if undefined_syms or undeclared_keys:
            trace_lines.append(
                f"  [Structural] ✗ Chain is not evaluable — undefined names: "
                f"{undefined_syms or undeclared_keys}. "
                "Every right-hand-side name must be a declared numeric given or a "
                "variable assigned by an earlier equation."
            )
            return SIVResult(
                trace="\n".join(trace_lines),
                execution_audit_passed=False,
                verifies_translation=False,
                givens_total=len(numeric_givens),
                invertible=False,
                undefined_symbols=undefined_syms,
                undeclared_given_keys=undeclared_keys,
                computed_answer=computed_answer,
            )

        # ── Step 2: Execution Audit (Layer 1 — forward) ───────────────────────
        # Substitute ALL numeric givens into symbolic expr and compare to computed_answer.
        # This is the primary execution fidelity check.
        execution_audit_passed, blueprint_answer, exec_rel_error = \
            SymbolicInverseVerifier._forward_audit(
                symbolic_expr, given_symbols, numeric_givens, computed_answer, trace_lines
            )

        # ── Step 3: Unused-given detection ────────────────────────────────────
        unused_givens = SymbolicInverseVerifier._find_unused_givens(
            symbolic_expr, given_symbols, numeric_givens, trace_lines
        )

        # ── Step 4: Fault Localization (Layer 2 — inverse) ────────────────────
        reconstructions = []
        matched = 0
        failed_names = []

        for given_name, given_value in numeric_givens.items():
            recon = SymbolicInverseVerifier._reconstruct_single_given(
                symbolic_expr, given_name, given_value,
                computed_answer, numeric_givens, given_symbols, trace_lines
            )
            reconstructions.append(recon)

            if recon.match:
                matched += 1
            elif recon.solvable:
                failed_names.append(given_name)

        total = len(numeric_givens)
        solvable_count = sum(1 for r in reconstructions if r.solvable)

        # ── Step 5: Compute fault-localization confidence ─────────────────────
        if total == 0:
            confidence = 0.5
            verified = False
        elif solvable_count == 0:
            confidence = 0.4
            verified = False
            trace_lines.append("  WARNING: No givens were solvable (non-invertible chain)")
        else:
            confidence = matched / solvable_count
            verified = (matched == solvable_count) and (solvable_count >= 1)

        if verified and solvable_count == total:
            confidence = min(0.97, confidence)   # Cap below 1.0: translation layer untested
            trace_lines.append(
                "  ✓ EXECUTION AUDIT PASSED + ALL GIVENS LOCALIZED: "
                "Programmer faithfully executed blueprint. "
                "(Translation-layer correctness not verified by SIV.)"
            )
        elif verified:
            confidence = min(0.90, confidence)
            trace_lines.append(
                f"  ✓ PARTIAL LOCALIZATION: {matched}/{solvable_count} solvable givens matched "
                f"({total - solvable_count} non-invertible/unused)"
            )
        else:
            trace_lines.append(
                f"  ✗ FAULT LOCALIZATION: {matched}/{solvable_count} matched; "
                f"failed givens: {failed_names}"
            )

        if unused_givens:
            trace_lines.append(
                f"  ⚠ UNUSED GIVENS (declared but absent from equations): {unused_givens}. "
                "These may be distractors or indicate missing equations in the blueprint."
            )

        return SIVResult(
            # Layer 1
            execution_audit_passed=execution_audit_passed,
            blueprint_answer=blueprint_answer,
            execution_rel_error=exec_rel_error,
            # Layer 2
            verified=verified,
            confidence=confidence,
            givens_matched=matched,
            givens_total=total,
            reconstructions=reconstructions,
            failed_givens=failed_names,
            unused_givens=unused_givens,
            invertible=solvable_count > 0,
            # Layer 0 — empty by construction: a structural defect returns above
            undefined_symbols=[],
            undeclared_given_keys=[],
            # Meta
            verifies_translation=False,
            computed_answer=computed_answer,
            trace="\n".join(trace_lines),
        )

    # =========================================================================
    # Layer 0: Structural scan (pre-algebraic) [v14.0]
    # =========================================================================

    @staticmethod
    def _scan_undeclared_given_keys(
        equations: List[str],
        givens: Dict[str, Any],
        numeric_givens: Dict[str, float]
    ) -> List[str]:
        """
        Deterministic pre-parse scan: every givens['<key>'] reference in the
        equations whose key is not declared with a numeric value.

        Two sub-cases, both fatal to evaluation and both worth naming in a repair
        prompt: the key is absent from 'givens' entirely (hallucinated/typo'd), or
        it is declared with a non-numeric value (a string slipped into givens).

        Runs on the raw equation strings, so it works even when SymPy's parser
        cannot get past the reference (parse_expr treats an unresolved `givens`
        as a Symbol and raises "'Symbol' object is not subscriptable").
        """
        missing: List[str] = []
        for eq in equations:
            for key in re.findall(r"givens\[\s*['\"]([^'\"]+)['\"]\s*\]", str(eq)):
                if key not in numeric_givens and key not in missing:
                    suffix = "" if key not in givens else " (declared but not numeric)"
                    missing.append(f"{key}{suffix}")
        return missing

    # Names an equation's right-hand side may use without declaring them: the
    # callables _build_symbolic_chain injects into local_dict, plus the math
    # names SymbolicSolver exposes and SymPy's own constants.
    _ALLOWED_FREE_NAMES = frozenset({
        "abs", "max", "min", "ceil", "floor", "sqrt", "round", "int", "float",
        "sum", "len", "pow", "divmod", "log", "log10", "exp", "pi", "e", "E",
        "Abs", "Max", "Min", "N", "Rational", "math",
        "True", "False", "None", "and", "or", "not", "if", "else", "for", "in",
    })

    @staticmethod
    def _scan_undefined_names(
        equations: List[str],
        numeric_givens: Dict[str, float]
    ) -> List[str]:
        """
        [v14.0] Source-level scan for names used but never defined.

        A name on an equation's right-hand side is legitimate only if it is a
        declared numeric given (bare names resolve to the same Symbol the chain
        builder creates), a variable assigned by an EARLIER equation, or one of
        the callables/constants the evaluator injects. Anything else is a defect.

        Why this is a source scan and not a SymPy free_symbols check: the chain
        builder parses with `implicit_multiplication_application`, so an
        undefined name like `rate` comes back as the product r*a*t*e and its
        free symbols are the four letters. That is useless in a repair prompt —
        the whole point is to hand the Mathematician the actual name it forgot to
        define. The SymPy residual is kept in _build_symbolic_chain only as a
        last-resort fallback for defects this scan does not catch.
        """
        defined = set(numeric_givens)
        undefined: List[str] = []

        for raw_eq in equations:
            eq = str(raw_eq).strip()
            if not eq or eq.startswith("#") or "=" not in eq:
                continue
            lhs, rhs = eq.split("=", 1)

            # Drop givens[...] references (handled by the undeclared-key scan)
            # and any string literals, so their contents are not read as names.
            rhs = re.sub(r"givens\[\s*['\"][^'\"]*['\"]\s*\]", " ", rhs)
            rhs = re.sub(r"'[^']*'|\"[^\"]*\"", " ", rhs)
            # Drop attribute access (math.floor -> math) and call targets, so an
            # unknown *function* name is not reported as an unknown *variable*.
            rhs = re.sub(r"\.\s*[A-Za-z_][A-Za-z_0-9]*", " ", rhs)
            rhs = re.sub(r"[A-Za-z_][A-Za-z_0-9]*\s*\(", " ( ", rhs)

            for name in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", rhs):
                if (name not in defined
                        and name not in SymbolicInverseVerifier._ALLOWED_FREE_NAMES
                        and name not in undefined):
                    undefined.append(name)

            defined.add(lhs.strip())

        return undefined

    # =========================================================================
    # Layer 1: Execution Audit (forward)
    # =========================================================================

    @staticmethod
    def _forward_audit(
        symbolic_expr: Any,
        given_symbols: Dict[str, Any],
        numeric_givens: Dict[str, float],
        computed_answer: float,
        trace_lines: List[str]
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        Execution Audit (Layer 1):
        Substitute ALL numeric givens into the symbolic expression and evaluate.
        Compare the result to computed_answer.

        Returns:
            (passed, blueprint_answer, relative_error)

        This is the primary signal for whether the Programmer correctly executed
        the Architect's blueprint. If passed=True, the inverse verification
        (Layer 2) is guaranteed to pass as well — this is the tautology inherent
        to math→math verification. The value of Layer 2 is fault LOCALIZATION,
        not independent verification.
        """
        try:
            substituted = symbolic_expr
            for gname, gsym in given_symbols.items():
                val = numeric_givens.get(gname)
                if val is not None:
                    substituted = substituted.subs(gsym, val)

            blueprint_answer = float(N(substituted))
            abs_diff = abs(blueprint_answer - computed_answer)
            rel_err = abs_diff / max(abs(computed_answer), 1e-9)

            passed = (
                abs_diff < SymbolicInverseVerifier.TOLERANCE_ABS or
                rel_err < SymbolicInverseVerifier.TOLERANCE_REL
            )

            status = "✓ PASS" if passed else "✗ FAIL"
            trace_lines.append(
                f"  [Exec Audit] {status}: blueprint evaluates to {blueprint_answer:.6g}, "
                f"Programmer gave {computed_answer:.6g}, rel_error={rel_err:.2e}"
            )
            if not passed:
                trace_lines.append(
                    "    → Execution error: Programmer's answer deviates from blueprint equations."
                )
            return passed, blueprint_answer, rel_err

        except Exception as e:
            trace_lines.append(f"  [Exec Audit] ERROR during forward evaluation: {e}")
            return False, None, None

    # =========================================================================
    # Unused-Given Detection
    # =========================================================================

    @staticmethod
    def _find_unused_givens(
        symbolic_expr: Any,
        given_symbols: Dict[str, Any],
        numeric_givens: Dict[str, float],
        trace_lines: List[str]
    ) -> List[str]:
        """
        Detect givens that are declared in the blueprint but do not appear
        in the symbolic expression (equation chain). These are either:
          - Legitimate distractors correctly ignored by the Architect, or
          - Missing equations — a potential blueprint modelling error.
        SIV flags them but cannot distinguish between the two cases.
        """
        unused = []
        try:
            free_syms = symbolic_expr.free_symbols
            for gname, gsym in given_symbols.items():
                if gsym not in free_syms:
                    unused.append(gname)
        except Exception:
            pass
        return unused

    # =========================================================================
    # Symbolic Chain Builder
    # =========================================================================

    @staticmethod
    def _build_symbolic_chain(
        equations: List[str],
        numeric_givens: Dict[str, float],
        trace_lines: List[str]
    ) -> Tuple[Optional[Any], Dict[str, Any], List[str]]:
        """
        Build a single SymPy expression for 'answer' in terms of given symbols.

        Returns (answer_expression, given_symbols_dict, undefined_symbols).
        On failure the first element is None; undefined_symbols is still returned
        so the caller can report *why* rather than a bare "could not build chain".

        [v14.0] undefined_symbols are the free symbols left in the answer
        expression that are not declared givens — i.e. names the blueprint used
        without ever defining them. parse_expr silently mints a Symbol for such a
        name, so the defect stays invisible until float() fails during forward
        substitution; naming it here is what makes it repairable.
        """
        try:
            given_symbols = {}
            for name in numeric_givens:
                safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
                given_symbols[name] = Symbol(safe_name, real=True)

            computed = {}
            last_result = None

            for eq_str in equations:
                eq_str = eq_str.strip()
                if not eq_str or eq_str.startswith("#"):
                    continue
                if "=" not in eq_str:
                    continue

                parts = eq_str.split("=", 1)
                lhs_name = parts[0].strip()
                rhs_str = parts[1].strip()

                rhs_modified = rhs_str
                for gname, gsym in given_symbols.items():
                    placeholder = f"__given_{re.sub(r'[^a-zA-Z0-9_]', '_', gname)}__"
                    rhs_modified = rhs_modified.replace(f"givens['{gname}']", placeholder)
                    rhs_modified = rhs_modified.replace(f'givens["{gname}"]', placeholder)

                for var_name in computed:
                    rhs_modified = re.sub(
                        rf'\b{re.escape(var_name)}\b',
                        f"(__computed_{var_name}__)",
                        rhs_modified
                    )

                local_dict = {}
                for gname, gsym in given_symbols.items():
                    safe = re.sub(r'[^a-zA-Z0-9_]', '_', gname)
                    local_dict[f"__given_{safe}__"] = gsym
                for var_name, var_expr in computed.items():
                    local_dict[f"__computed_{var_name}__"] = var_expr

                local_dict['abs'] = sympy.Abs
                local_dict['max'] = sympy.Max
                local_dict['min'] = sympy.Min
                local_dict['ceil'] = sympy.ceiling
                local_dict['floor'] = sympy.floor
                local_dict['sqrt'] = sympy.sqrt
                local_dict['round'] = lambda x, n=0: sympy.floor(x * 10**n + sympy.Rational(1, 2)) / 10**n
                local_dict['int'] = sympy.floor
                local_dict['float'] = lambda x: x

                try:
                    transformations = standard_transformations + (
                        implicit_multiplication_application,
                        convert_xor,
                    )
                    expr = parse_expr(
                        rhs_modified,
                        local_dict=local_dict,
                        transformations=transformations
                    )
                    computed[lhs_name] = expr
                    last_result = expr

                except Exception as parse_err:
                    trace_lines.append(f"  WARNING: Could not parse '{eq_str}': {parse_err}")
                    try:
                        expr = SymbolicInverseVerifier._fallback_parse(
                            rhs_str, given_symbols, computed
                        )
                        if expr is not None:
                            computed[lhs_name] = expr
                            last_result = expr
                        else:
                            trace_lines.append(f"  Fallback parse also failed for: {eq_str}")
                            return None, {}, []
                    except Exception:
                        return None, {}, []

            answer_expr = computed.get("answer", last_result)
            undefined = SymbolicInverseVerifier._collect_undefined_symbols(
                answer_expr, given_symbols
            )
            return answer_expr, given_symbols, undefined

        except Exception as e:
            trace_lines.append(f"  FATAL in _build_symbolic_chain: {type(e).__name__}: {e}")
            return None, {}, []

    @staticmethod
    def _collect_undefined_symbols(
        answer_expr: Any,
        given_symbols: Dict[str, Any]
    ) -> List[str]:
        """
        Free symbols in the built expression that are not declared givens.

        A legitimately unused given is the opposite case (declared but absent from
        the expression) and is reported separately by _find_unused_givens.
        """
        if answer_expr is None:
            return []
        try:
            declared = set(given_symbols.values())
            return sorted(str(s) for s in answer_expr.free_symbols - declared)
        except Exception:
            return []

    @staticmethod
    def _fallback_parse(
        rhs_str: str,
        given_symbols: Dict[str, Any],
        computed: Dict[str, Any]
    ) -> Optional[Any]:
        """Fallback parser via sympify with manual substitution."""
        try:
            expr_str = rhs_str
            for gname, gsym in given_symbols.items():
                expr_str = expr_str.replace(f"givens['{gname}']", f"({gsym})")
                expr_str = expr_str.replace(f'givens["{gname}"]', f"({gsym})")
            for var_name, var_expr in computed.items():
                expr_str = re.sub(
                    rf'\b{re.escape(var_name)}\b',
                    f"({var_expr})",
                    expr_str
                )
            return sympify(expr_str)
        except Exception:
            return None

    # =========================================================================
    # Layer 2: Fault Localization (inverse)
    # =========================================================================

    @staticmethod
    def _reconstruct_single_given(
        answer_expr: Any,
        given_name: str,
        given_value: float,
        computed_answer: float,
        all_givens: Dict[str, float],
        given_symbols: Dict[str, Any],
        trace_lines: List[str]
    ) -> GivenReconstruction:
        """
        Fault Localization for one given variable.

        Substitute all OTHER givens numerically, then solve:
            answer_expr = computed_answer
        for the target given.

        Multi-root handling:
            All real-valued roots are collected and reported (ambiguity flag).
            Match is True if ANY root is within tolerance of original value.
            This is honest: we do not prefer the closest root circularly —
            we report all roots and let the caller see the full picture.
        """
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', given_name)
        target_symbol = given_symbols.get(given_name) or Symbol(safe_name, real=True)

        try:
            # Substitute all givens except the target
            substituted = answer_expr
            for other_name, other_value in all_givens.items():
                if other_name == given_name:
                    continue
                other_sym = given_symbols.get(other_name) or Symbol(
                    re.sub(r'[^a-zA-Z0-9_]', '_', other_name), real=True
                )
                substituted = substituted.subs(other_sym, other_value)

            # Check if expression actually depends on this given
            if target_symbol not in substituted.free_symbols:
                trace_lines.append(
                    f"  [{given_name}] UNUSED IN CHAIN — given not present in equation chain"
                )
                return GivenReconstruction(
                    name=given_name,
                    original_value=given_value,
                    reconstructed_value=None,
                    match=False,   # Cannot assess — not a pass
                    relative_error=None,
                    solvable=False,
                    ambiguous=False,
                    all_real_roots=[],
                    error_message="Variable absent from symbolic chain (unused given)"
                )

            equation = Eq(substituted, computed_answer)
            solutions = solve(equation, target_symbol)

            if not solutions:
                trace_lines.append(
                    f"  [{given_name}] No solution found by SymPy"
                )
                return GivenReconstruction(
                    name=given_name,
                    original_value=given_value,
                    reconstructed_value=None,
                    match=False,
                    relative_error=None,
                    solvable=False,
                    error_message="No solution found by SymPy"
                )

            # Collect ALL real-valued roots
            real_roots = []
            for sol in solutions:
                try:
                    sol_float = float(N(sol))
                    if not (sol_float != sol_float):  # NaN check
                        real_roots.append(sol_float)
                except (TypeError, ValueError, OverflowError):
                    continue  # Skip complex or non-numeric

            if not real_roots:
                trace_lines.append(f"  [{given_name}] Solutions exist but none are real-valued")
                return GivenReconstruction(
                    name=given_name,
                    original_value=given_value,
                    reconstructed_value=None,
                    match=False,
                    relative_error=None,
                    solvable=False,
                    error_message="No real-valued solution"
                )

            ambiguous = len(real_roots) > 1

            # Check if ANY root matches the original (honest multi-root handling)
            best_match_val = None
            best_rel_err = float('inf')
            found_match = False

            for root_val in real_roots:
                abs_diff = abs(root_val - given_value)
                rel_diff = abs_diff / max(abs(given_value), 1e-9)
                if (abs_diff < SymbolicInverseVerifier.TOLERANCE_ABS or
                        rel_diff < SymbolicInverseVerifier.TOLERANCE_REL):
                    found_match = True
                    if rel_diff < best_rel_err:
                        best_rel_err = rel_diff
                        best_match_val = root_val
                elif rel_diff < best_rel_err:
                    best_rel_err = rel_diff
                    best_match_val = root_val

            abs_diff_best = abs(best_match_val - given_value)
            rel_diff_best = abs_diff_best / max(abs(given_value), 1e-9)

            status = "✓ MATCH" if found_match else "✗ MISMATCH"
            ambig_note = f" [AMBIGUOUS: {len(real_roots)} roots]" if ambiguous else ""
            trace_lines.append(
                f"  [{given_name}] {status}{ambig_note}: "
                f"original={given_value}, "
                f"roots={[f'{r:.6g}' for r in real_roots]}, "
                f"rel_error={rel_diff_best:.2e}"
            )

            return GivenReconstruction(
                name=given_name,
                original_value=given_value,
                reconstructed_value=best_match_val,
                match=found_match,
                relative_error=rel_diff_best,
                solvable=True,
                ambiguous=ambiguous,
                all_real_roots=real_roots,
            )

        except Exception as e:
            trace_lines.append(
                f"  [{given_name}] ERROR during reconstruction: {type(e).__name__}: {e}"
            )
            return GivenReconstruction(
                name=given_name,
                original_value=given_value,
                reconstructed_value=None,
                match=False,
                relative_error=None,
                solvable=False,
                error_message=str(e)
            )

    # =========================================================================
    # Error Localization Report (for Critic / SHT)
    # =========================================================================

    @staticmethod
    def get_error_localization_report(result: SIVResult) -> str:
        """
        Generate a fault report for the Critic/SHT stage.

        Reports both execution-audit outcome (Layer 1) and the per-variable
        reconstruction results (Layer 2) so the Critic can focus on the
        implicated (used, non-distractor) variables rather than blind
        re-generation. [v12.2] When 2+ givens are used, this is a candidate
        set, not a single named culprit (see the SIVResult.failed_givens
        docstring and test_siv_fault_injection.py) — the report is phrased
        as "re-examine equations involving X, Y, Z", never as "X is the
        error", to stay honest about what was actually established.

        Explicitly notes that blueprint–problem translation errors are outside
        SIV's detection scope, so the Critic should also apply its own
        natural-language reasoning on whether the blueprint models the problem.
        """
        lines = ["SIV Fault Localization Report:"]

        # [v14.0] Layer 0 FIRST. Ordering is deliberate: a structural defect is a
        # single, named, mechanically fixable error, whereas Layer 2's implicated
        # set is at best 1/n_used precise. Leading with the weaker signal (as this
        # report did through v13.0) told the Mathematician "all your givens are
        # suspect" even when the real problem was one undefined name — which is
        # the diagnosed reason the repair loop's fix-rate sat at 6/39.
        if result.has_structural_defect:
            lines.append(
                "  [Layer 0 — Structure] ✗ The equation chain cannot be evaluated at all:"
            )
            for sym in result.undefined_symbols:
                lines.append(
                    f"    ✗ '{sym}' is used on the right-hand side of an equation but is "
                    f"never defined — it is not a key in 'givens' and no earlier equation "
                    f"assigns it."
                )
            for key in result.undeclared_given_keys:
                lines.append(
                    f"    ✗ givens['{key}'] is referenced but not declared as a numeric given."
                )
            lines.append(
                "    → FIX THIS FIRST: every name on the right-hand side of an equation must "
                "be either givens['<key>'] with that key declared to a number, or a variable "
                "assigned by an earlier equation in the list. Nothing else about the blueprint "
                "could be checked until this is resolved."
            )
            return "\n".join(lines)

        # Layer 1: Execution audit
        if result.blueprint_answer is not None:
            programmer_ans_str = (
                f"{result.computed_answer:.6g}" if result.computed_answer is not None else "?"
            )
            if result.execution_audit_passed:
                lines.append(
                    f"  [Layer 1 — Execution] ✓ PASS: Blueprint evaluates to "
                    f"{result.blueprint_answer:.6g}, matching Programmer output ({programmer_ans_str}). "
                    f"Execution fidelity confirmed."
                )
            else:
                lines.append(
                    f"  [Layer 1 — Execution] ✗ FAIL: Blueprint evaluates to "
                    f"{result.blueprint_answer:.6g}, but Programmer gave {programmer_ans_str} "
                    f"(rel_error={result.execution_rel_error:.2e}). "
                    f"Programmer deviated from blueprint equations."
                )
        else:
            lines.append("  [Layer 1 — Execution] Could not evaluate blueprint forward.")

        # Layer 2: Fault localization
        lines.append("  [Layer 2 — Fault Localization]:")
        any_fault = False
        for recon in result.reconstructions:
            if not recon.match and recon.solvable:
                any_fault = True
                roots_str = ", ".join(f"{r:.6g}" for r in recon.all_real_roots)
                lines.append(
                    f"    ⚠ INCONSISTENT: '{recon.name}' — "
                    f"declared value={recon.original_value}, "
                    f"inverse roots=[{roots_str}] "
                    f"(error={recon.relative_error:.2%})"
                )
            elif recon.ambiguous:
                roots_str = ", ".join(f"{r:.6g}" for r in recon.all_real_roots)
                lines.append(
                    f"    ~ AMBIGUOUS: '{recon.name}' — multiple real roots [{roots_str}]; "
                    f"original value matches, but solution is not uniquely invertible."
                )
            elif not recon.solvable:
                lines.append(
                    f"    ? UNVERIFIABLE: '{recon.name}' — {recon.error_message}"
                )

        if not any_fault and result.verified:
            lines.append("    ✓ All solvable givens reconstructed correctly.")

        if result.failed_givens:
            lines.append(
                f"\n  Recommended: Re-examine equations involving: "
                f"{', '.join(result.failed_givens)}"
            )

        if result.unused_givens:
            lines.append(
                f"\n  Unused givens (declared but absent from equation chain): "
                f"{result.unused_givens}. "
                "Verify whether these are intentional distractors or missing equations."
            )

        lines.append(
            "\n  ⚠ SIV Scope Limitation: This report covers execution fidelity only. "
            "Errors in how the problem was translated into equations (NL→math) "
            "are NOT detectable by SIV. Apply independent natural-language reasoning "
            "to verify the blueprint models the problem correctly."
        )

        return "\n".join(lines)
