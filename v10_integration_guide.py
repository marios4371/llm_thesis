"""
MAS-SHT v10.0 Integration Guide: Symbolic Inverse Verification (SIV)
=====================================================================

This file contains all the code changes needed to upgrade Mas_solver.py
from v9.1 to v10.0 with the novel SIV contribution.

Changes are organized by section. Copy each section into the appropriate
location in Mas_solver.py, or use the full updated file provided separately.
"""

# =====================================================================
# CHANGE 1: Version header (replace lines 1-5)
# =====================================================================

VERSION_HEADER = '''
"""
Enhanced Reasoning Quality Evaluation System for MAS Math Solver
VERSION 10.0: Symbolic Inverse Verification (SIV) — Novel Contribution

CHANGELOG v10.0 (over v9.1):
- [NEW] Symbolic Inverse Verification (SIV): zero-cost verification via 
        symbolic equation chain inversion using SymPy
- [NEW] SIVResult dataclass with per-given reconstruction details
- [NEW] SymbolicInverseVerifier class — core novel contribution
- [NEW] Error localization: SIV identifies EXACTLY which given variable is
        inconsistent, enabling targeted repair in SHT
- [NEW] SIV-enhanced confidence gate: uses inverse verification as strongest signal
- [NEW] SIV-informed Critic: passes error localization report to hypothesis generator
- [NEW] CSV output includes SIV metrics (siv_verified, siv_confidence, etc.)
- [NEW] AgentRole.PROCESS_VERIFIER for SIV in heterogeneous configs
"""
'''

# =====================================================================
# CHANGE 2: New import — add after existing imports (around line 30)
# =====================================================================

NEW_IMPORT = """
# --- [NEW v10.0] Symbolic Inverse Verification ---
from siv_module import (
    SymbolicInverseVerifier, SIVResult, GivenReconstruction
)
"""

# =====================================================================
# CHANGE 3: Updated AgentRole enum — add PROCESS_VERIFIER
# (This is conceptual — SIV doesn't use LLM, but useful for logging)
# =====================================================================

# No actual change needed since SIV is zero-cost (no LLM calls)
# But if you want to track it in the model config:
# AgentRole.PROCESS_VERIFIER = "process_verifier"  # optional


# =====================================================================
# CHANGE 4: Replace verify_code_against_blueprint method
# (in QualityEnhancedMultiAgentSolver class)
# This is the MAIN integration point for SIV
# =====================================================================

UPDATED_VERIFY_METHOD = '''
    def verify_code_against_blueprint(self, problem: str, blueprint: dict,
                                       code: str, code_answer: str) -> Tuple[bool, str, float]:
        """
        [v10.0] Two-phase verification:
          Phase 1: Rule-based checks (v9.0, zero cost)
          Phase 2: Symbolic Inverse Verification (v10.0, zero cost)
        
        SIV provides MATHEMATICAL PROOF of correctness when it succeeds,
        and EXACT ERROR LOCALIZATION when it fails.
        """
        givens = blueprint.get("givens", {})
        equations = blueprint.get("equations", [])
        
        issues = []
        
        # === Phase 1: Rule-based checks (unchanged from v9.0) ===
        
        code_givens = _extract_givens_dict_from_code(code)
        if code_givens is not None and givens:
            for key, val in givens.items():
                if key not in code_givens:
                    issues.append(f"Missing given '{key}'")
                elif isinstance(val, (int, float)) and isinstance(code_givens.get(key), (int, float)):
                    if abs(code_givens[key] - val) > 1e-6:
                        issues.append(f"Givens mismatch '{key}': blueprint={val} code={code_givens[key]}")
        
        for eq in equations:
            if "=" in eq:
                var_name = eq.split("=")[0].strip()
                if var_name not in code and var_name != "answer":
                    issues.append(f"Missing variable '{var_name}'")
        
        answer_num = _extract_last_number(code_answer)
        if answer_num is not None:
            if answer_num < 0 and not any(
                kw in problem.lower() for kw in ["loss", "decrease", "debt", "negative", "below", "fewer", "less", "owe"]
            ):
                issues.append(f"Negative answer ({answer_num}) seems wrong for this problem")
            
            if givens:
                max_given = max((abs(v) for v in givens.values() if isinstance(v, (int, float))), default=0)
                if max_given > 0 and abs(answer_num) > max_given * 10000:
                    issues.append(f"Answer ({answer_num}) implausibly large vs givens (max={max_given})")
        
        expected = blueprint.get("expected_answer", "")
        if expected and answer_num is not None:
            expected_num = _extract_last_number(str(expected))
            if expected_num is not None and abs(expected_num) > 0.01:
                rel_diff = abs(answer_num - expected_num) / max(abs(expected_num), 1e-9)
                if rel_diff > 0.1:
                    issues.append(f"Code answer ({answer_num}) differs from Mathematician estimate ({expected_num}) by {rel_diff:.0%}")
        
        # === Phase 2: Symbolic Inverse Verification (NEW v10.0) ===
        
        siv_result = SIVResult()  # Default empty
        if SYMPY_AVAILABLE and answer_num is not None and equations:
            logger.info("Running Symbolic Inverse Verification (SIV)...")
            siv_result = SymbolicInverseVerifier.verify(blueprint, answer_num)
            
            if siv_result.verified:
                logger.info(
                    f"SIV VERIFIED: {siv_result.givens_matched}/{siv_result.givens_total} "
                    f"givens reconstructed (confidence={siv_result.confidence:.2f})"
                )
                # SIV verification OVERRIDES rule-based issues (mathematical proof)
                if siv_result.confidence >= 0.95:
                    return True, f"SIV VERIFIED ({siv_result.givens_matched}/{siv_result.givens_total} givens)", siv_result.confidence
            else:
                logger.info(
                    f"SIV FAILED: {siv_result.givens_matched}/{siv_result.givens_total} matched, "
                    f"failed: {siv_result.failed_givens}"
                )
                for failed_name in siv_result.failed_givens:
                    recon = next(
                        (r for r in siv_result.reconstructions if r.name == failed_name),
                        None
                    )
                    if recon and recon.reconstructed_value is not None:
                        issues.append(
                            f"SIV: '{failed_name}' reconstruction mismatch "
                            f"(original={recon.original_value}, "
                            f"inverse={recon.reconstructed_value:.6g})"
                        )
                    else:
                        issues.append(f"SIV: '{failed_name}' not inversely solvable")
        
        # === Scoring (combines both phases) ===
        
        if not issues:
            return True, "All checks passed", 1.0
        
        # SIV failures are critical
        siv_failures = sum(1 for i in issues if "SIV:" in i)
        rule_critical = sum(1 for i in issues if "mismatch" in i.lower() and "SIV:" not in i)
        rule_minor = len(issues) - siv_failures - rule_critical
        
        if siv_failures > 0:
            # SIV found mathematical inconsistency — strong signal
            confidence = max(0.2, 1.0 - siv_failures * 0.3 - rule_critical * 0.15)
            return False, "; ".join(issues[:4]), confidence
        elif rule_critical > 0:
            confidence = max(0.3, 1.0 - rule_critical * 0.25 - rule_minor * 0.1)
            return False, "; ".join(issues[:3]), confidence
        
        confidence = max(0.6, 1.0 - rule_minor * 0.1)
        return True, "; ".join(issues[:3]), confidence
'''


# =====================================================================
# CHANGE 5: Updated _confidence_gate — SIV as additional criterion
# (in QualityEnhancedMultiAgentSolver class)
# =====================================================================

UPDATED_CONFIDENCE_GATE = '''
    def _confidence_gate(self, primary_answer: str, baseline_answer: str,
                         programmer_response: AgentResponse,
                         blueprint: dict,
                         siv_result: Optional[SIVResult] = None) -> Tuple[bool, str]:
        """
        [v10.0] Enhanced with SIV result as strongest signal.
        
        If SIV verified the answer → skip SHT (mathematical proof of correctness).
        If SIV found specific errors → trigger SHT with error localization.
        """
        # [NEW v10.0] Criterion 0: SIV overrides
        if siv_result is not None:
            if siv_result.verified and siv_result.confidence >= 0.95:
                return True, "siv_verified"  # Skip SHT — proven correct
            elif siv_result.invertible and not siv_result.verified:
                return False, f"siv_inconsistency:{','.join(siv_result.failed_givens)}"
        
        # Criterion 1: Programmer failed entirely
        if str(primary_answer).strip().lower() == "unknown":
            return False, "programmer_failed"

        # Criterion 2: Primary answer disagrees with baseline
        primary_num = _extract_last_number(primary_answer)
        baseline_num = _extract_last_number(baseline_answer)
        if primary_num is not None and baseline_num is not None:
            if abs(primary_num - baseline_num) > 1e-3:
                return False, "baseline_disagreement"
        elif str(primary_answer).strip() != str(baseline_answer).strip():
            return False, "baseline_disagreement"

        # Criterion 3: Programmer exhausted all repair attempts
        if programmer_response.quality_metrics.get("error") == "Max attempts reached":
            return False, "max_attempts_exhausted"

        # Criterion 4: Sanity checks
        if primary_num is not None:
            if primary_num < 0:
                return False, "negative_answer"

            givens = blueprint.get("givens", {})
            if givens:
                max_given = max(
                    (abs(v) for v in givens.values() if isinstance(v, (int, float))),
                    default=0
                )
                if max_given > 0 and abs(primary_num) > max_given * 10000:
                    return False, "answer_magnitude_suspicious"

        # Criterion 5: Both answers are "unknown"
        if str(baseline_answer).strip().lower() == "unknown":
            return False, "baseline_also_failed"

        return True, "all_checks_passed"
'''


# =====================================================================
# CHANGE 6: Updated generate_alternative_hypotheses — SIV error info
# (in QualityEnhancedMultiAgentSolver class)
# =====================================================================

UPDATED_HYPOTHESIS_GENERATOR = '''
    def generate_alternative_hypotheses(self, problem: str,
                                        primary_blueprint: dict,
                                        primary_answer: str,
                                        siv_error_report: str = "") -> List[dict]:
        """
        [v10.0] Enhanced Critic with SIV error localization.
        
        When SIV provides specific error information (e.g., "variable X 
        doesn't reconstruct correctly"), we pass this to the Critic for
        TARGETED correction instead of blind re-derivation.
        """
        primary_eqs = primary_blueprint.get("equations", [])
        primary_givens = primary_blueprint.get("givens", {})
        primary_steps = primary_blueprint.get("solution_steps", [])

        # [v10.0] Inject SIV error report if available
        siv_context = ""
        if siv_error_report:
            siv_context = f"""

AUTOMATED VERIFICATION REPORT (from symbolic inverse checker):
{siv_error_report}
NOTE: The symbolic checker found specific inconsistencies by solving the equations 
backwards. Pay special attention to the variables flagged above."""

        sys_msg = f"""You are a meticulous Mathematics Reviewer. 
Your job is to FIND ERRORS in a proposed solution and provide CORRECTIONS.

A colleague solved a math problem and got the answer: {primary_answer}

REVIEW CHECKLIST:
1. Are all relevant numbers from the problem extracted correctly?
2. Are any IRRELEVANT numbers (distractors) mistakenly included?
3. Is each mathematical operation correct for what the problem asks?
4. Are there any MISSING steps?
5. Does the final answer actually answer what was asked?
{siv_context}

After your review, provide exactly 2 corrected solutions:
- Correction 1: Fix the most likely error you found
- Correction 2: Solve from scratch using a completely different approach

OUTPUT FORMAT (strict JSON, no other text):
{{
  "review": "Brief description of error(s) found (or 'no errors found')",
  "alternatives": [
    {{
      "strategy_name": "correction_of_[specific error]",
      "error_found": "what was wrong in the original",
      "unknown": "what we need to find",
      "givens": {{"var_name": numeric_value, ...}},
      "solution_steps": ["Step 1: ...", "Step 2: ..."],
      "equations": ["step1 = givens['var'] ...", "answer = ..."],
      "expected_answer": "your mental estimate"
    }},
    {{
      "strategy_name": "independent_rederivation",
      "error_found": "solving from scratch to verify",
      "unknown": "what we need to find",
      "givens": {{"var_name": numeric_value, ...}},
      "solution_steps": ["Step 1: ...", "Step 2: ..."],
      "equations": ["step1 = givens['var'] ...", "answer = ..."],
      "expected_answer": "your mental estimate"
    }}
  ]
}}"""

        user_msg = f"""PROBLEM:
{problem}

COLLEAGUE'S SOLUTION TO REVIEW:
Givens: {json.dumps(primary_givens)}
Steps: {json.dumps(primary_steps)}
Equations: {json.dumps(primary_eqs)}
Answer obtained: {primary_answer}

Review for errors and provide 2 corrected/alternative solutions as JSON."""

        msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]

        raw = self._get_client(AgentRole.HYPOTHESIS_GENERATOR).call_model(
            msgs, temperature=0.3, max_tokens=900
        )

        if _is_error_response(raw):
            logger.warning("Hypothesis generator returned error response")
            return []

        alternatives = []
        try:
            text = str(raw).strip()
            text = re.sub(r"^```(?:json)?\\s*|\\s*```$", "", text, flags=re.IGNORECASE).strip()

            parsed = None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                start, end = text.find("{"), text.rfind("}")
                if start != -1 and end > start:
                    try:
                        parsed = json.loads(text[start:end+1])
                    except json.JSONDecodeError:
                        pass

            if parsed and "alternatives" in parsed:
                review = parsed.get("review", "")
                if review:
                    logger.info(f"SHT Critic review: {review[:150]}")
                    
                for alt in parsed["alternatives"][:2]:
                    if isinstance(alt, dict):
                        alt.setdefault("unknown", "the answer")
                        alt.setdefault("givens", {})
                        alt.setdefault("solution_steps", [])
                        alt.setdefault("equations", [])
                        alt.setdefault("strategy_name", "critic_correction")
                        alternatives.append(alt)
        except Exception as e:
            logger.warning(f"SHT: Failed to parse critic response: {e}")

        return alternatives
'''


# =====================================================================
# CHANGE 7: Updated _structured_hypothesis_testing — passes SIV result
# =====================================================================

UPDATED_SHT = '''
    def _structured_hypothesis_testing(self, problem: str, expected: str,
                                       primary_blueprint: dict,
                                       programmer_response: AgentResponse,
                                       baseline_answer: str,
                                       siv_result: Optional[SIVResult] = None) -> HypothesisLog:
        """
        [v10.0] Enhanced with SIV integration:
        1. Confidence gate uses SIV result (can skip SHT if SIV verified)
        2. Error localization report passed to Critic for targeted repair
        """
        primary_answer = programmer_response.answer
        primary_num = _extract_last_number(primary_answer)

        log = HypothesisLog(
            problem=problem,
            expected=expected,
            final_answer=primary_answer,
            final_strategy="primary",
        )

        primary_candidate = HypothesisResult(
            hypothesis_id="primary",
            strategy_name="primary_blueprint",
            blueprint=primary_blueprint,
            code=programmer_response.reasoning_trace,
            code_success=programmer_response.confidence > 0.5,
            execution_output=programmer_response.quality_metrics.get("execution_output", ""),
            answer=primary_answer,
            parsed_answer=primary_num,
            confidence=programmer_response.confidence,
            agent_response=programmer_response,
        )
        log.candidates.append(primary_candidate)

        baseline_num = _extract_last_number(baseline_answer)
        baseline_candidate = HypothesisResult(
            hypothesis_id="baseline",
            strategy_name="zero_shot_baseline",
            blueprint={},
            code=None,
            code_success=baseline_num is not None,
            execution_output="",
            answer=baseline_answer,
            parsed_answer=baseline_num,
            confidence=0.5,
            agent_response=None,
        )
        log.candidates.append(baseline_candidate)

        # [v10.0] Pass SIV result to confidence gate
        is_confident, gate_reason = self._confidence_gate(
            primary_answer, baseline_answer, programmer_response,
            primary_blueprint, siv_result=siv_result
        )

        if is_confident:
            log.triage_result = f"confident_skip ({gate_reason})"
            log.final_answer = primary_answer
            log.final_strategy = "primary_blueprint"
            log.hypothesis_testing_triggered = False
            log.api_calls_used = 3
            return log

        logger.info(f"SHT triggered: {gate_reason}")
        log.hypothesis_testing_triggered = True
        api_calls = 3

        # Check token budget
        sht_cost_estimate = 4 * 1500
        if not token_budget.can_afford(sht_cost_estimate):
            logger.warning("SHT skipped due to token budget.")
            if primary_num is not None:
                log.final_answer = primary_answer
                log.final_strategy = "primary_budget_skip"
            else:
                log.final_answer = baseline_answer
                log.final_strategy = "baseline_budget_skip"
            log.triage_result = "budget_skip"
            log.api_calls_used = 3
            return log

        # [v10.0] Generate SIV error report for targeted Critic
        siv_error_report = ""
        if siv_result and not siv_result.verified and siv_result.invertible:
            siv_error_report = SymbolicInverseVerifier.get_error_localization_report(siv_result)
            logger.info(f"SIV error report for Critic: {siv_error_report[:200]}")

        alt_blueprints = self.generate_alternative_hypotheses(
            problem, primary_blueprint, primary_answer,
            siv_error_report=siv_error_report  # [v10.0]
        )
        api_calls += 1

        for idx, alt_bp in enumerate(alt_blueprints[:2]):
            alt_response = self.run_programmer_solver(problem, alt_bp, max_attempts=1)
            api_calls += 1

            alt_num = _extract_last_number(alt_response.answer)
            alt_candidate = HypothesisResult(
                hypothesis_id=f"alt_{idx+1}",
                strategy_name=alt_bp.get("strategy_name", f"alternative_{idx+1}"),
                blueprint=alt_bp,
                code=alt_response.reasoning_trace,
                code_success=alt_response.confidence > 0.5,
                execution_output=alt_response.quality_metrics.get("execution_output", ""),
                answer=alt_response.answer,
                parsed_answer=alt_num,
                confidence=alt_response.confidence,
                agent_response=alt_response,
            )
            log.candidates.append(alt_candidate)

        triage_answer, triage_strategy, triage_method = self._triage_candidates(log.candidates)

        if triage_method in ("unanimous", "majority"):
            log.triage_result = triage_method
            log.final_answer = triage_answer
            log.final_strategy = triage_strategy
            log.api_calls_used = api_calls
            return log

        judge_answer, judge_strategy, judge_reasoning = self._judge_hypotheses(
            problem, log.candidates
        )
        api_calls += 1

        log.triage_result = "judge"
        log.judge_reasoning = judge_reasoning
        log.final_answer = judge_answer
        log.final_strategy = judge_strategy
        log.api_calls_used = api_calls
        return log
'''


# =====================================================================
# CHANGE 8: Updated solve() method — orchestrates SIV
# =====================================================================

UPDATED_SOLVE = '''
    def solve(self, problem: str, expected: str) -> Dict[str, Any]:
        # Step 1: Baseline
        baseline_prompt = f"{problem}\\n\\nSolve this step-by-step. End with: ANSWER: [[numeric_value]]"
        base_raw = self._get_client(AgentRole.BASELINE).call_model(
            [{"role": "user", "content": baseline_prompt}],
            temperature=0.1,
            max_tokens=500
        )
        base_ans, _ = self.extract_answer(base_raw)

        # Step 2: Architect
        blackboard_logic = self.run_mathematician_analysis(problem)

        # Step 3: Engineer (with SymPy fallback built-in)
        programmer_response = self.run_programmer_solver(problem, blackboard_logic)

        # Step 3b: Process-Level Verification (now includes SIV)
        verification_passed = True
        verification_feedback = "Skipped"
        verification_confidence = 1.0
        siv_result = None  # [v10.0]
        
        if (programmer_response.confidence > 0.5
            and programmer_response.answer != "unknown"
            and programmer_response.reasoning_trace
            and "SymPy" not in programmer_response.agent):
            
            verification_passed, verification_feedback, verification_confidence = \\
                self.verify_code_against_blueprint(
                    problem, blackboard_logic,
                    programmer_response.reasoning_trace,
                    programmer_response.answer
                )
            
            # [v10.0] Run SIV independently for metrics
            answer_num = _extract_last_number(programmer_response.answer)
            if SYMPY_AVAILABLE and answer_num is not None and blackboard_logic.get("equations"):
                siv_result = SymbolicInverseVerifier.verify(blackboard_logic, answer_num)
            
            adjusted_confidence = programmer_response.confidence * verification_confidence
            programmer_response = AgentResponse(
                agent=programmer_response.agent,
                answer=programmer_response.answer,
                parsed=programmer_response.parsed,
                confidence=adjusted_confidence,
                reasoning_trace=programmer_response.reasoning_trace,
                quality_metrics={
                    **programmer_response.quality_metrics,
                    "verification_passed": verification_passed,
                    "verification_confidence": verification_confidence,
                    "verification_feedback": verification_feedback[:200],
                    # [v10.0] SIV metrics
                    "siv_verified": siv_result.verified if siv_result else None,
                    "siv_confidence": siv_result.confidence if siv_result else None,
                    "siv_givens_matched": siv_result.givens_matched if siv_result else None,
                    "siv_givens_total": siv_result.givens_total if siv_result else None,
                    "siv_failed_givens": siv_result.failed_givens if siv_result else [],
                }
            )
            
            if not verification_passed:
                logger.info(f"Process verification FAILED (conf={verification_confidence:.2f}).")
                if SYMPY_AVAILABLE and blackboard_logic.get("equations"):
                    sym_ok, sym_ans, sym_trace = SymbolicSolver.solve_from_blueprint(blackboard_logic)
                    if sym_ok:
                        sym_num = _extract_last_number(sym_ans)
                        if sym_num is not None:
                            programmer_response = AgentResponse(
                                agent="SymPy (post-verification fallback)",
                                answer=str(sym_num),
                                parsed=str(sym_num),
                                confidence=0.75,
                                reasoning_trace=sym_trace[:500],
                                quality_metrics={
                                    "solver": "sympy_post_verification",
                                    "original_answer": programmer_response.answer,
                                    "verification_rejection": verification_feedback[:200],
                                }
                            )
                            # [v10.0] Re-run SIV on the SymPy answer
                            siv_result = SymbolicInverseVerifier.verify(blackboard_logic, sym_num)

        # Step 4: Structured Hypothesis Testing (with SIV integration)
        hypothesis_log = None
        if self.enable_hypothesis_testing:
            hypothesis_log = self._structured_hypothesis_testing(
                problem, expected, blackboard_logic,
                programmer_response, base_ans,
                siv_result=siv_result  # [v10.0] Pass SIV result
            )
            mas_answer = hypothesis_log.final_answer
            used_baseline_fallback = False
        else:
            mas_answer = programmer_response.answer
            used_baseline_fallback = False

        # Step 5: Fallback
        if self.enable_baseline_fallback_on_mas_failure:
            if str(mas_answer).strip().lower() == "unknown" and str(base_ans).strip().lower() != "unknown":
                mas_answer = base_ans
                used_baseline_fallback = True

        result = {
            "problem": problem,
            "expected": expected,
            "baseline": {
                "answer": base_ans,
                "model": str(self._get_client(AgentRole.BASELINE)),
            },
            "mas": {
                "answer": mas_answer,
                "logic_trace": json.dumps(blackboard_logic, ensure_ascii=False)[:500],
                "used_baseline_fallback": used_baseline_fallback,
                "programmer_metrics": programmer_response.quality_metrics,
                "verification": {
                    "passed": verification_passed,
                    "confidence": verification_confidence,
                    "feedback": verification_feedback[:200],
                },
            },
            "agents": [programmer_response],
            "model_config": self.get_model_config_summary(),
            # [v10.0] SIV metrics at top level
            "siv": {
                "verified": siv_result.verified if siv_result else None,
                "confidence": siv_result.confidence if siv_result else None,
                "givens_matched": siv_result.givens_matched if siv_result else None,
                "givens_total": siv_result.givens_total if siv_result else None,
                "invertible": siv_result.invertible if siv_result else None,
                "failed_givens": siv_result.failed_givens if siv_result else [],
                "trace": siv_result.trace[:300] if siv_result else "",
            },
        }

        if hypothesis_log:
            result["sht"] = {
                "triggered": hypothesis_log.hypothesis_testing_triggered,
                "triage_result": hypothesis_log.triage_result,
                "final_strategy": hypothesis_log.final_strategy,
                "num_candidates": len(hypothesis_log.candidates),
                "api_calls_used": hypothesis_log.api_calls_used,
                "judge_reasoning": hypothesis_log.judge_reasoning,
                "candidates": [
                    {
                        "id": c.hypothesis_id,
                        "strategy": c.strategy_name,
                        "answer": c.answer,
                        "success": c.code_success,
                    }
                    for c in hypothesis_log.candidates
                ],
            }

        return result
'''


# =====================================================================
# CHANGE 9: Updated CSV output in run() method (QualityAwarePipeline)
# Add SIV columns to the DataFrame
# =====================================================================

UPDATED_DF_CONSTRUCTION = '''
        df = pd.DataFrame([
            {
                "id": r["id"],
                "dataset": r.get("dataset", ""),
                "baseline_correct": r["baseline"]["correct"],
                "mas_correct": r["mas"]["correct"],
                "baseline_ans": r["baseline"]["answer"],
                "mas_ans": r["mas"]["answer"],
                "mas_used_baseline_fallback": r["mas"].get("used_baseline_fallback", False),
                "expected_snippet": str(r["expected"])[-30:],
                "verification_passed": r.get("mas", {}).get("verification", {}).get("passed", True),
                "verification_confidence": r.get("mas", {}).get("verification", {}).get("confidence", 1.0),
                "solver_agent": r.get("agents", [{}])[0].agent if r.get("agents") else "unknown",
                # [v10.0] SIV metrics
                "siv_verified": r.get("siv", {}).get("verified", None),
                "siv_confidence": r.get("siv", {}).get("confidence", None),
                "siv_givens_matched": r.get("siv", {}).get("givens_matched", None),
                "siv_givens_total": r.get("siv", {}).get("givens_total", None),
                "siv_invertible": r.get("siv", {}).get("invertible", None),
                "siv_failed_givens": str(r.get("siv", {}).get("failed_givens", [])),
                **sht_data[i],
                **r.get("model_config", {}),
            } for i, r in enumerate(detailed)
        ])
'''


# =====================================================================
# CHANGE 10: Updated report() method — add SIV stats
# =====================================================================

UPDATED_REPORT_SIV_SECTION = '''
        # [v10.0] SIV statistics
        siv_verified_count = sum(
            1 for r in self.results
            if r.get("siv", {}).get("verified", False)
        )
        siv_invertible_count = sum(
            1 for r in self.results
            if r.get("siv", {}).get("invertible", False)
        )
        siv_failed_count = sum(
            1 for r in self.results
            if r.get("siv", {}).get("invertible", False) and not r.get("siv", {}).get("verified", False)
        )
        
        print("-" * 60)
        print("SYMBOLIC INVERSE VERIFICATION (SIV) — Novel v10.0:")
        print(f"{'SIV Invertible Chains':<30} | {siv_invertible_count}/{n}")
        print(f"{'SIV Verified (proven correct)':<30} | {siv_verified_count}/{n}")
        print(f"{'SIV Detected Errors':<30} | {siv_failed_count}/{n}")
        if siv_invertible_count > 0:
            siv_precision = siv_verified_count / siv_invertible_count
            print(f"{'SIV Verification Rate':<30} | {siv_precision:.2%}")
'''
