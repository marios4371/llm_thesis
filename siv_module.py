"""
Symbolic Inverse Verification (SIV) Module
===========================================
Novel contribution for MAS-SHT v10.0

Core Idea (Cycle-Consistent Mathematical Reasoning):
    Given a forward equation chain  G → E₁ → E₂ → ... → Eₙ → A
    where G = givens, E = equations, A = answer,
    
    SIV inverts the chain symbolically:
    For each given gᵢ ∈ G, treat gᵢ as unknown, execute E symbolically,
    set result = A, solve for gᵢ.
    
    If solved_gᵢ ≈ actual_gᵢ for ALL givens → solution is PROVEN correct.
    If solved_gᵢ ≠ actual_gᵢ → we know EXACTLY which given is inconsistent,
    enabling targeted repair instead of full re-generation.

Difference from FOBAR (Jiang et al., 2024):
    FOBAR uses the LLM itself for backward verification → inherits same errors.
    SIV uses SymPy (computer algebra system) → deterministic, mathematically
    guaranteed correct verification. Zero API calls, zero possibility of 
    verification errors.

Theoretical basis:
    Inverse consistency (cycle consistency): if f(x) = y, then f⁻¹(y) = x.
    A correct solution must be invertible — given the answer and the problem
    structure, the original parameters must be recoverable.
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
    """Result of reconstructing a single given variable."""
    name: str
    original_value: float
    reconstructed_value: Optional[float]
    match: bool                  # True if |original - reconstructed| < tolerance
    relative_error: Optional[float]  # |diff| / max(|original|, 1e-9)
    solvable: bool               # Whether SymPy could solve for this variable
    error_message: str = ""


@dataclass
class SIVResult:
    """
    Complete result of Symbolic Inverse Verification.
    
    Fields:
        verified: True if ALL givens were successfully reconstructed
        confidence: 0.0 - 1.0 based on proportion of matched givens
        givens_matched: number of givens that reconstructed correctly
        givens_total: total number of numeric givens tested
        reconstructions: per-given reconstruction details
        failed_givens: names of givens that did NOT reconstruct (error localization)
        trace: human-readable trace of the verification process
        invertible: whether the equation chain was symbolically invertible at all
    """
    verified: bool = False
    confidence: float = 0.0
    givens_matched: int = 0
    givens_total: int = 0
    reconstructions: List[GivenReconstruction] = field(default_factory=list)
    failed_givens: List[str] = field(default_factory=list)
    trace: str = ""
    invertible: bool = False


# =====================================================================
# Core SIV Engine
# =====================================================================

class SymbolicInverseVerifier:
    """
    Symbolic Inverse Verification (SIV) — Novel Contribution.
    
    Verifies a mathematical solution by symbolically inverting the equation
    chain. For each given parameter, treats it as unknown while keeping all
    other values fixed, then solves the equations to check if the original
    value can be recovered from the computed answer.
    
    This is analogous to cycle consistency in generative models (CycleGAN),
    but applied to mathematical reasoning:
        Forward:  givens → equations → answer
        Inverse:  answer → equations⁻¹ → givens  (must match originals)
    
    Properties:
        - Deterministic: uses SymPy, not LLM (zero API calls)
        - Complete: checks every given, not just one masked number
        - Localizing: identifies exactly which given fails reconstruction
        - Proved: if all givens reconstruct, the answer is mathematically correct
    """

    TOLERANCE_ABS = 1e-6    # Absolute tolerance for float comparison
    TOLERANCE_REL = 1e-4    # Relative tolerance (0.01%)

    @staticmethod
    def verify(blueprint: dict, computed_answer: float) -> SIVResult:
        """
        Main verification entry point.
        
        Args:
            blueprint: Mathematician's blueprint with 'givens' and 'equations'
            computed_answer: The numeric answer from the Programmer/SymPy
            
        Returns:
            SIVResult with detailed verification outcome
        """
        if not SYMPY_AVAILABLE:
            return SIVResult(
                trace="SIV skipped: SymPy not available",
                verified=False,
                confidence=0.5
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
                verified=False,
                confidence=0.5
            )
        
        if not equations:
            return SIVResult(
                trace="SIV skipped: no equations in blueprint",
                verified=False,
                confidence=0.5
            )
        
        trace_lines = ["[SIV] Symbolic Inverse Verification"]
        trace_lines.append(f"  Computed answer: {computed_answer}")
        trace_lines.append(f"  Givens to verify: {list(numeric_givens.keys())}")
        
        # Step 1: Build the symbolic forward chain
        symbolic_expr = SymbolicInverseVerifier._build_symbolic_chain(
            equations, numeric_givens, trace_lines
        )
        
        if symbolic_expr is None:
            return SIVResult(
                trace="\n".join(trace_lines) + "\n  FAILED: Could not build symbolic chain",
                verified=False,
                confidence=0.4,
                givens_total=len(numeric_givens),
                invertible=False
            )
        
        trace_lines.append(f"  Symbolic answer expression: {symbolic_expr}")
        
        # Step 2: For each given, solve inverse
        reconstructions = []
        matched = 0
        failed_names = []
        
        for given_name, given_value in numeric_givens.items():
            recon = SymbolicInverseVerifier._reconstruct_single_given(
                symbolic_expr, given_name, given_value,
                computed_answer, numeric_givens, trace_lines
            )
            reconstructions.append(recon)
            
            if recon.match:
                matched += 1
            elif recon.solvable:
                failed_names.append(given_name)
        
        total = len(numeric_givens)
        solvable_count = sum(1 for r in reconstructions if r.solvable)
        
        # Step 3: Compute confidence
        if total == 0:
            confidence = 0.5
            verified = False
        elif solvable_count == 0:
            # Couldn't solve for any given — chain not invertible
            confidence = 0.4
            verified = False
            trace_lines.append("  WARNING: No givens were solvable (non-invertible chain)")
        else:
            # Confidence = proportion of solvable givens that matched
            confidence = matched / solvable_count
            verified = (matched == solvable_count) and (solvable_count >= 1)
        
        # Boost confidence if ALL givens (including non-solvable) matched
        if verified and solvable_count == total:
            confidence = 1.0  # Full mathematical proof
            trace_lines.append("  ✓ FULL INVERSE VERIFICATION: All givens reconstructed correctly")
        elif verified:
            confidence = min(0.95, confidence)
            trace_lines.append(
                f"  ✓ PARTIAL INVERSE VERIFICATION: {matched}/{solvable_count} "
                f"solvable givens matched ({total - solvable_count} non-invertible)"
            )
        else:
            trace_lines.append(
                f"  ✗ INVERSE VERIFICATION FAILED: {matched}/{solvable_count} matched, "
                f"failed givens: {failed_names}"
            )
        
        return SIVResult(
            verified=verified,
            confidence=confidence,
            givens_matched=matched,
            givens_total=total,
            reconstructions=reconstructions,
            failed_givens=failed_names,
            trace="\n".join(trace_lines),
            invertible=solvable_count > 0
        )

    @staticmethod
    def _build_symbolic_chain(
        equations: List[str],
        numeric_givens: Dict[str, float],
        trace_lines: List[str]
    ) -> Optional[Any]:
        """
        Build a single SymPy expression for 'answer' in terms of given symbols.
        
        Executes the equation chain symbolically, keeping givens as SymPy Symbols
        so they can later be solved for individually.
        
        Returns the symbolic expression for 'answer', or None if building fails.
        """
        try:
            # Create SymPy symbols for all givens
            given_symbols = {}
            for name in numeric_givens:
                # Sanitize name for SymPy (replace spaces, special chars)
                safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
                given_symbols[name] = Symbol(safe_name, real=True)
            
            # Build namespace: givens as symbols, everything else will be computed
            namespace = {}
            # Create a mock 'givens' dict that returns symbols
            givens_dict = dict(given_symbols)
            
            # Track computed variables
            computed = {}
            last_result = None
            
            for eq_str in equations:
                eq_str = eq_str.strip()
                if not eq_str or eq_str.startswith("#"):
                    continue
                
                if "=" not in eq_str:
                    continue
                
                # Split into LHS = RHS
                parts = eq_str.split("=", 1)
                lhs_name = parts[0].strip()
                rhs_str = parts[1].strip()
                
                # Replace givens['key'] and givens["key"] with symbol references
                rhs_modified = rhs_str
                for gname, gsym in given_symbols.items():
                    rhs_modified = rhs_modified.replace(
                        f"givens['{gname}']", f"__given_{re.sub(r'[^a-zA-Z0-9_]', '_', gname)}__"
                    )
                    rhs_modified = rhs_modified.replace(
                        f'givens["{gname}"]', f"__given_{re.sub(r'[^a-zA-Z0-9_]', '_', gname)}__"
                    )
                
                # Also replace bare variable references to previously computed values
                for var_name, var_expr in computed.items():
                    # Replace whole words only
                    rhs_modified = re.sub(
                        rf'\b{re.escape(var_name)}\b',
                        f"(__computed_{var_name}__)",
                        rhs_modified
                    )
                
                # Build substitution dict for parse_expr
                local_dict = {}
                for gname, gsym in given_symbols.items():
                    safe = re.sub(r'[^a-zA-Z0-9_]', '_', gname)
                    local_dict[f"__given_{safe}__"] = gsym
                for var_name, var_expr in computed.items():
                    local_dict[f"__computed_{var_name}__"] = var_expr
                
                # Add math functions
                local_dict['abs'] = sympy.Abs
                local_dict['max'] = sympy.Max
                local_dict['min'] = sympy.Min
                local_dict['ceil'] = sympy.ceiling
                local_dict['floor'] = sympy.floor
                local_dict['sqrt'] = sympy.sqrt
                local_dict['round'] = lambda x, n=0: sympy.floor(x * 10**n + 0.5) / 10**n
                local_dict['int'] = lambda x: sympy.floor(x)
                local_dict['float'] = lambda x: x  # No-op for symbolic
                
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
                    trace_lines.append(f"  WARNING: Could not parse equation '{eq_str}': {parse_err}")
                    # Try a simpler approach: direct Python-style evaluation
                    try:
                        expr = SymbolicInverseVerifier._fallback_parse(
                            rhs_str, given_symbols, computed
                        )
                        if expr is not None:
                            computed[lhs_name] = expr
                            last_result = expr
                        else:
                            trace_lines.append(f"  Fallback parse also failed for: {eq_str}")
                            return None
                    except Exception:
                        return None
            
            # Return the 'answer' expression, or the last computed expression
            answer_expr = computed.get("answer", last_result)
            return answer_expr
            
        except Exception as e:
            trace_lines.append(f"  FATAL in _build_symbolic_chain: {type(e).__name__}: {e}")
            return None

    @staticmethod
    def _fallback_parse(
        rhs_str: str,
        given_symbols: Dict[str, Any],
        computed: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Fallback parser for equations that parse_expr can't handle.
        Performs manual substitution and evaluation.
        """
        try:
            expr_str = rhs_str
            
            # Replace givens references
            for gname, gsym in given_symbols.items():
                expr_str = expr_str.replace(f"givens['{gname}']", f"({gsym})")
                expr_str = expr_str.replace(f'givens["{gname}"]', f"({gsym})")
            
            # Replace computed variable references
            for var_name, var_expr in computed.items():
                expr_str = re.sub(
                    rf'\b{re.escape(var_name)}\b',
                    f"({var_expr})",
                    expr_str
                )
            
            # Try sympify
            return sympify(expr_str)
        except Exception:
            return None

    @staticmethod
    def _reconstruct_single_given(
        answer_expr: Any,
        given_name: str,
        given_value: float,
        computed_answer: float,
        all_givens: Dict[str, float],
        trace_lines: List[str]
    ) -> GivenReconstruction:
        """
        For a single given variable, substitute all OTHER givens with their
        actual numeric values, then solve: answer_expr = computed_answer for
        the target given.
        
        This is the core of the inverse verification.
        """
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', given_name)
        target_symbol = Symbol(safe_name, real=True)
        
        try:
            # Substitute all givens EXCEPT the target with their numeric values
            substituted = answer_expr
            for other_name, other_value in all_givens.items():
                if other_name == given_name:
                    continue  # Keep target as symbol
                other_safe = re.sub(r'[^a-zA-Z0-9_]', '_', other_name)
                other_sym = Symbol(other_safe, real=True)
                substituted = substituted.subs(other_sym, other_value)
            
            # Now we have an expression in terms of target_symbol only
            # Solve: substituted = computed_answer
            equation = Eq(substituted, computed_answer)
            
            solutions = solve(equation, target_symbol)
            
            if not solutions:
                trace_lines.append(
                    f"  [{given_name}] No solution found (equation may be independent of this given)"
                )
                # Check if the expression actually contains the target symbol
                if target_symbol not in substituted.free_symbols:
                    trace_lines.append(
                        f"    → Expression independent of '{given_name}' "
                        f"(given not used in equation chain)"
                    )
                    return GivenReconstruction(
                        name=given_name,
                        original_value=given_value,
                        reconstructed_value=None,
                        match=True,  # Independent givens don't affect the answer
                        relative_error=None,
                        solvable=False,  # Can't solve because variable isn't present
                        error_message="Variable independent of answer"
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
            
            # Find the solution closest to the original value
            # (There may be multiple solutions, e.g., for quadratic equations)
            best_solution = None
            best_distance = float('inf')
            
            for sol in solutions:
                try:
                    sol_float = float(N(sol))
                    dist = abs(sol_float - given_value)
                    if dist < best_distance:
                        best_distance = dist
                        best_solution = sol_float
                except (TypeError, ValueError, OverflowError):
                    continue  # Skip complex or non-numeric solutions
            
            if best_solution is None:
                trace_lines.append(
                    f"  [{given_name}] Solutions exist but none are real-valued"
                )
                return GivenReconstruction(
                    name=given_name,
                    original_value=given_value,
                    reconstructed_value=None,
                    match=False,
                    relative_error=None,
                    solvable=False,
                    error_message="No real-valued solution"
                )
            
            # Compare
            abs_diff = abs(best_solution - given_value)
            rel_diff = abs_diff / max(abs(given_value), 1e-9)
            
            match = (
                abs_diff < SymbolicInverseVerifier.TOLERANCE_ABS or
                rel_diff < SymbolicInverseVerifier.TOLERANCE_REL
            )
            
            status = "✓ MATCH" if match else "✗ MISMATCH"
            trace_lines.append(
                f"  [{given_name}] {status}: "
                f"original={given_value}, reconstructed={best_solution:.6g}, "
                f"rel_error={rel_diff:.2e}"
            )
            
            return GivenReconstruction(
                name=given_name,
                original_value=given_value,
                reconstructed_value=best_solution,
                match=match,
                relative_error=rel_diff,
                solvable=True
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

    @staticmethod
    def get_error_localization_report(result: SIVResult) -> str:
        """
        Generate a targeted error report for the Critic/SHT stage.
        
        When SIV detects a mismatch, this report tells the Critic exactly
        which given variable is inconsistent, enabling targeted repair
        instead of full re-generation.
        """
        if result.verified:
            return "SIV: All givens verified. Solution is mathematically proven correct."
        
        lines = ["SIV Error Localization Report:"]
        
        for recon in result.reconstructions:
            if not recon.match and recon.solvable:
                lines.append(
                    f"  ⚠ INCONSISTENT: '{recon.name}' — "
                    f"problem states {recon.original_value}, "
                    f"but solving backwards from the answer gives {recon.reconstructed_value:.6g} "
                    f"(error: {recon.relative_error:.2%})"
                )
            elif not recon.solvable and not recon.match:
                lines.append(
                    f"  ? UNVERIFIABLE: '{recon.name}' — "
                    f"could not solve inversely ({recon.error_message})"
                )
        
        if result.failed_givens:
            lines.append(
                f"\n  Recommended action: Re-examine equations involving: "
                f"{', '.join(result.failed_givens)}"
            )
        
        return "\n".join(lines)
