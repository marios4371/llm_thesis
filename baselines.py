"""
Baseline Systems for MAS-SHT Comparative Evaluation
====================================================
Companion to Mas_solver.py v10.2.

Defines the simpler-than-MAS-SHT systems that the thesis compares against.
Every baseline lives behind a uniform interface (`BaselineResult`) so the
experiment runner in MAS_SHT_Experiments.ipynb can iterate over them
without per-system branching.

System inventory (mirrors the thesis):
    B1  direct_answer       — single LLM call, "Solve. Answer:"
    B2  chain_of_thought    — single LLM call, "Solve step by step. Answer:"
    B3  self_consistency    — 5 CoT samples at T=0.7, majority vote
    B4  baseline_only       — wraps the BASELINE agent of the existing pipeline
                              (re-uses the project's prompt — no new prompt)
    B5  MAS-NoSIV           — full pipeline with enable_siv=False
                              (constructed in the runner, not here)
    B6  MAS-NoSHT           — full pipeline with enable_sht=False
                              (constructed in the runner, not here)
    B7  MAS-SHT-Full        — current pipeline, unchanged
                              (constructed in the runner, not here)

Apples-to-apples policy:
    Each baseline returns a `BaselineResult` containing a parsed numeric
    answer (or None on failure), token estimate, latency, and the count of
    actual LLM calls. The experiment runner is responsible for calling each
    baseline on the SAME problem set with the SAME random seed.
"""

from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import Counter

# Re-use existing helpers from the main solver.
from Mas_solver import (
    UnifiedLLMClient,
    _extract_last_number,
    _is_error_response,
    token_budget,
    AgentRole,
)

logger = logging.getLogger("MAS_Pipeline")


# =====================================================================
# Uniform return type
# =====================================================================

@dataclass
class BaselineResult:
    """
    Per-problem result emitted by every baseline.

    Fields:
        answer:            Parsed numeric prediction. None on parse failure.
        raw:               Truncated model output (for debugging / CSV trace).
        num_llm_calls:     Number of LLM completions consumed by this run.
        tokens_estimated:  Approximate token cost (input chars/4 + output budget).
        time_s:            Wall-clock latency in seconds.
        error_type:        "" on success; otherwise an error category like
                           "api_error", "no_number_in_output", "all_samples_failed".
                           This is the field the metrics layer uses to
                           distinguish "wrong answer" from "API failure".
        meta:              Free-form per-baseline diagnostics (vote distribution
                           for SC, raw outputs for CoT, etc.).
    """
    answer: Optional[float]
    raw: str
    num_llm_calls: int
    tokens_estimated: int
    time_s: float
    error_type: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Internal helpers
# =====================================================================

def _estimate_tokens(prompt: str, max_tokens: int) -> int:
    """Mirror the heuristic used by Mas_solver.TokenBudget so reports line up."""
    return (len(prompt) // 4) + int(max_tokens * 0.35)


def _parse_numeric(text: str) -> Optional[float]:
    """Extract the last number; honours error responses.
    [v10.3] Priority: \\boxed{N} (Qwen/DeepSeek output format),
    then last number in text.
    """
    if _is_error_response(text):
        return None
    s = str(text)
    # Priority 1: \boxed{N} — format used by Qwen2.5-Math and DeepSeek-R1
    m = re.search(r'\\\\boxed\{(-?\d+(?:[.,]\d+)*)\}', s)
    if m:
        try:
            return float(m.group(1).replace(',', ''))
        except ValueError:
            pass
    # Priority 2: last number in text
    return _extract_last_number(s)


# =====================================================================
# B1 — Direct Answer (zero-shot, no reasoning)
# =====================================================================

DIRECT_PROMPT = (
    "Solve this math problem. Give only the final numeric answer, nothing else.\n\n"
    "Problem: {problem}\n\nAnswer:"
)


def direct_answer(client: UnifiedLLMClient, problem: str,
                  max_tokens: int = 512) -> BaselineResult:
    """B1 — single LLM call, zero reasoning, terse answer-only prompt.
    [v10.3] Default raised 64→512: local HF models need more tokens to reason
    before outputting a number even on zero-shot prompts.
    """
    t0 = time.time()
    prompt = DIRECT_PROMPT.format(problem=problem)
    msgs = [{"role": "user", "content": prompt}]
    raw = client.call_model(msgs, temperature=0.0, max_tokens=max_tokens)
    elapsed = time.time() - t0
    raw_str = str(raw)[:1000]

    if _is_error_response(raw):
        return BaselineResult(
            answer=None, raw=raw_str, num_llm_calls=1,
            tokens_estimated=_estimate_tokens(prompt, max_tokens),
            time_s=elapsed, error_type="api_error",
        )

    num = _parse_numeric(raw_str)
    return BaselineResult(
        answer=num, raw=raw_str, num_llm_calls=1,
        tokens_estimated=_estimate_tokens(prompt, max_tokens),
        time_s=elapsed,
        error_type="" if num is not None else "no_number_in_output",
    )


# =====================================================================
# B2 — Chain-of-Thought
# =====================================================================

COT_PROMPT = (
    "Solve this math problem step by step. After your reasoning, "
    "state the final numeric answer on a line starting with 'Answer:'.\n\n"
    "Problem: {problem}\n\nLet's think step by step."
)


def chain_of_thought(client: UnifiedLLMClient, problem: str,
                     max_tokens: int = 1024,  # [v10.6] 500→1024: math-model CoT on hard problems needs >500 tok; clipping the answer unfairly penalizes the baseline
                     temperature: float = 0.0) -> BaselineResult:
    """B2 — single CoT call. Temperature defaults to 0 for determinism;
    self_consistency() bumps it to 0.7 for diversity."""
    t0 = time.time()
    prompt = COT_PROMPT.format(problem=problem)
    msgs = [{"role": "user", "content": prompt}]
    raw = client.call_model(msgs, temperature=temperature, max_tokens=max_tokens)
    elapsed = time.time() - t0
    raw_str = str(raw)[:4000]  # [v10.3] raised 2000→4000 so \\boxed{} at end of output isn't clipped

    if _is_error_response(raw):
        return BaselineResult(
            answer=None, raw=raw_str, num_llm_calls=1,
            tokens_estimated=_estimate_tokens(prompt, max_tokens),
            time_s=elapsed, error_type="api_error",
        )

    # Prefer the explicit "Answer: X" line if present; fall back to last number.
    ans = None
    m = re.search(r"answer\s*[:=]\s*([\-+]?\d+(?:\.\d+)?)", raw_str, re.IGNORECASE)
    if m:
        try:
            ans = float(m.group(1))
        except ValueError:
            ans = None
    if ans is None:
        ans = _parse_numeric(raw_str)

    return BaselineResult(
        answer=ans, raw=raw_str, num_llm_calls=1,
        tokens_estimated=_estimate_tokens(prompt, max_tokens),
        time_s=elapsed,
        error_type="" if ans is not None else "no_number_in_output",
    )


# =====================================================================
# B3 — Self-Consistency (SC@n)
# =====================================================================

def self_consistency(client: UnifiedLLMClient, problem: str,
                     n: int = 5, temperature: float = 0.7,
                     max_tokens: int = 500,
                     inter_sample_sleep: float = 1.0) -> BaselineResult:
    """
    B3 — n independent CoT samples at temperature, then majority vote on the
    final numeric answer.

    Implementation notes:
        * Vote equality uses round(x, 6) to absorb float drift.
        * If all samples fail to produce a number → error_type='all_samples_failed'.
        * Tie-break: highest count wins; on equal counts, the answer with the
          smaller numeric value wins (deterministic — the alternative is
          first-seen-order which depends on dict iteration).
        * inter_sample_sleep adds extra slack on top of the client's own
          rate limiter to avoid 429 bursts on Groq.
    """
    t0 = time.time()
    raws: List[str] = []
    nums: List[float] = []
    errors = 0
    total_tokens = 0

    for i in range(n):
        sample = chain_of_thought(client, problem,
                                  max_tokens=max_tokens,
                                  temperature=temperature)
        raws.append(sample.raw)
        total_tokens += sample.tokens_estimated
        if sample.error_type == "":
            if sample.answer is not None:
                nums.append(round(sample.answer, 6))
        else:
            errors += 1
        if i < n - 1 and inter_sample_sleep > 0:
            time.sleep(inter_sample_sleep)

    elapsed = time.time() - t0
    raw_str = ("\n--- sample ---\n".join(raws))[:3000]

    if not nums:
        return BaselineResult(
            answer=None, raw=raw_str, num_llm_calls=n,
            tokens_estimated=total_tokens, time_s=elapsed,
            error_type="all_samples_failed" if errors == n else "no_majority",
            meta={"samples_failed": errors, "n": n},
        )

    counts = Counter(nums)
    # Sort by (count desc, value asc) for determinism on ties.
    best_val, best_cnt = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    return BaselineResult(
        answer=float(best_val), raw=raw_str, num_llm_calls=n,
        tokens_estimated=total_tokens, time_s=elapsed,
        error_type="",
        meta={
            "vote_distribution": dict(counts),
            "winner_votes": best_cnt,
            "samples_failed": errors,
            "n": n,
        },
    )


# =====================================================================
# B4 — Baseline-Only (current pipeline's BASELINE agent, standalone)
# =====================================================================

# Mirrors the prompt used inside QualityEnhancedMultiAgentSolver.solve() — the
# point of B4 is to measure that exact prompt in isolation, NOT a different
# direct/CoT prompt. If the pipeline's baseline prompt changes upstream,
# update this in lock-step.
#
# [v10.4] System+user split + 1024 max_tokens, mirroring the in-pipeline
# baseline prompt redesign. The old single-message prompt with 500 tokens
# was truncating CoT mid-stream on hard problems.
BASELINE_PIPELINE_SYS = (
    "You are a careful problem solver. "
    "Work through the problem step by step, showing arithmetic. "
    "On the LAST line, output EXACTLY one line of the form: "
    "ANSWER: [[<single numeric value, no units, no commas>]]"
)
BASELINE_PIPELINE_USER = (
    "{problem}\n\n"
    "Solve step by step. Do not skip arithmetic. "
    "End with the ANSWER line as instructed."
)


def baseline_only(client: UnifiedLLMClient, problem: str,
                  max_tokens: int = 1024) -> BaselineResult:
    """B4 — re-runs the pipeline's BASELINE agent prompt as a standalone
    system, so we can see how much of MAS-SHT's gain comes from prompt
    engineering alone vs the multi-agent scaffold.

    [v10.4] max_tokens default raised 500→1024 to match the in-pipeline
    baseline. Keep the parameter overridable so the experiment runner can
    still bound cost for very large model sweeps.
    """
    t0 = time.time()
    user = BASELINE_PIPELINE_USER.format(problem=problem)
    msgs = [
        {"role": "system", "content": BASELINE_PIPELINE_SYS},
        {"role": "user",   "content": user},
    ]
    raw = client.call_model(msgs, temperature=0.1, max_tokens=max_tokens)
    elapsed = time.time() - t0
    raw_str = str(raw)[:3000]   # [v10.4] +1000 chars so the ANSWER tag at end is preserved

    full_prompt_for_estimate = BASELINE_PIPELINE_SYS + "\n" + user

    if _is_error_response(raw):
        return BaselineResult(
            answer=None, raw=raw_str, num_llm_calls=1,
            tokens_estimated=_estimate_tokens(full_prompt_for_estimate, max_tokens),
            time_s=elapsed, error_type="api_error",
        )

    # Honour the ANSWER: [[X]] tag the prompt asks for; fall back to last number.
    ans = None
    m = re.search(r"ANSWER:\s*\[\[([^\]]+)\]\]", raw_str, re.IGNORECASE)
    if m:
        ans = _parse_numeric(m.group(1))
    if ans is None:
        ans = _parse_numeric(raw_str)

    return BaselineResult(
        answer=ans, raw=raw_str, num_llm_calls=1,
        tokens_estimated=_estimate_tokens(full_prompt_for_estimate, max_tokens),
        time_s=elapsed,
        error_type="" if ans is not None else "no_number_in_output",
    )


# =====================================================================
# B_pal — Program-Aided Language Model (Gao et al., ICML 2023)
# =====================================================================
#
# PAL is the canonical "ask the LLM to write a Python program that computes
# the answer, then execute the program" baseline. It is the closest non-MAS
# comparator to MAS-SHT's Architect+Engineer pipeline, so it is REQUIRED in
# any honest comparison: a positive MAS-SHT vs PAL delta is the only way to
# show that the multi-agent scaffold adds value on top of code execution.
#
# Implementation notes:
#   * One LLM call. No blueprint, no critic, no SymPy fallback.
#   * Code sandbox reuses Mas_solver.PythonExecutor → same forbidden-token
#     set, same answer/result variable resolution. This keeps the harness
#     identical to MAS-SHT and isolates the multi-agent contribution.
#   * If extraction fails, fall back to last-number from the raw text (some
#     models inline the result instead of using a code block on easy
#     problems — counting that as a failure would unfairly hurt PAL).
#   * No code repair loop (single-shot) — adding repair would conflate PAL
#     with MAS-SHT's repair mechanism.

PAL_SYSTEM = (
    "You are an expert Python programmer. To answer the math problem you will "
    "write a self-contained Python program that prints the final numeric answer. "
    "Use clear variable names, no external libraries beyond Python's math module. "
    "Do not print anything except the final answer.\n\n"
    "OUTPUT FORMAT: a single ```python ... ``` code block, then on a new line "
    "write: ANSWER: [[<numeric>]] using the value printed by your program."
)
PAL_USER = (
    "Problem:\n{problem}\n\n"
    "Write a Python program that computes the answer and prints it."
)


def pal(client: UnifiedLLMClient, problem: str,
        max_tokens: int = 1024) -> BaselineResult:  # [v10.6] 800→1024 headroom
    """Program-Aided Language model baseline (Gao et al., ICML 2023).
    Single LLM call → Python program → sandboxed execution → numeric answer.

    Returns BaselineResult with:
        error_type='code_extraction_failed' if no ```python block found
        error_type='code_execution_failed'  if sandbox rejected/raised
        error_type='no_number_in_output'    if neither code nor raw text yielded a number
        error_type=''                       on success
    """
    # Lazy import — PythonExecutor lives in the main solver. Doing this at
    # call time (not module top) avoids a circular import: Mas_solver imports
    # nothing from baselines, and baselines only needs the executor for PAL.
    from Mas_solver import PythonExecutor, _extract_code_from_response

    t0 = time.time()
    user = PAL_USER.format(problem=problem)
    msgs = [
        {"role": "system", "content": PAL_SYSTEM},
        {"role": "user",   "content": user},
    ]
    raw = client.call_model(msgs, temperature=0.0, max_tokens=max_tokens)
    elapsed = time.time() - t0
    raw_str = str(raw)[:3000]
    full_prompt = PAL_SYSTEM + "\n" + user

    if _is_error_response(raw):
        return BaselineResult(
            answer=None, raw=raw_str, num_llm_calls=1,
            tokens_estimated=_estimate_tokens(full_prompt, max_tokens),
            time_s=elapsed, error_type="api_error",
        )

    code = _extract_code_from_response(raw_str)
    code_ans, code_err = None, ""
    if code:
        ok, output = PythonExecutor.execute(code)
        if ok:
            code_ans = _parse_numeric(output)
            if code_ans is None:
                code_err = "code_no_number_in_output"
        else:
            code_err = f"code_execution_failed:{output[:120]}"

    # Prefer code answer; if the program failed, fall back to the inline
    # ANSWER tag or last-number-in-text. This avoids penalising PAL when the
    # model wrote a correct equation inline on a 1-step problem.
    ans = code_ans
    if ans is None:
        m = re.search(r"ANSWER:\s*\[\[([^\]]+)\]\]", raw_str, re.IGNORECASE)
        if m:
            ans = _parse_numeric(m.group(1))
    if ans is None:
        ans = _parse_numeric(raw_str)

    if ans is None:
        if not code:
            err = "code_extraction_failed"
        elif code_err:
            err = code_err
        else:
            err = "no_number_in_output"
    else:
        err = ""

    return BaselineResult(
        answer=ans, raw=raw_str, num_llm_calls=1,
        tokens_estimated=_estimate_tokens(full_prompt, max_tokens),
        time_s=elapsed,
        error_type=err,
        meta={
            "code_present":   code is not None,
            "code_succeeded": code_ans is not None,
        },
    )


# =====================================================================
# B_pot — Program-of-Thought (Chen et al., TMLR 2023)
# =====================================================================
#
# PoT differs from PAL in that the model is asked to INTERLEAVE natural-
# language reasoning with code, then the code is what produces the answer.
# In practice on grade-school math the two converge; we keep them separate
# so reviewers see a one-to-one mapping against the cited literature.

POT_SYSTEM = (
    "You are an expert at solving math problems using Python. "
    "First reason through the problem in 2-3 sentences. "
    "Then write a Python program that computes the answer and prints it.\n\n"
    "OUTPUT FORMAT:\n"
    "Reasoning: <brief 2-3 sentence reasoning>\n"
    "```python\n# code that prints the answer\n```\n"
    "ANSWER: [[<numeric>]]"
)


def pot(client: UnifiedLLMClient, problem: str,
        max_tokens: int = 1024) -> BaselineResult:
    """Program-of-Thought baseline (Chen et al., TMLR 2023).
    NL reasoning + Python code in a single call.
    """
    from Mas_solver import PythonExecutor, _extract_code_from_response

    t0 = time.time()
    user = PAL_USER.format(problem=problem)
    msgs = [
        {"role": "system", "content": POT_SYSTEM},
        {"role": "user",   "content": user},
    ]
    raw = client.call_model(msgs, temperature=0.0, max_tokens=max_tokens)
    elapsed = time.time() - t0
    raw_str = str(raw)[:3500]
    full_prompt = POT_SYSTEM + "\n" + user

    if _is_error_response(raw):
        return BaselineResult(
            answer=None, raw=raw_str, num_llm_calls=1,
            tokens_estimated=_estimate_tokens(full_prompt, max_tokens),
            time_s=elapsed, error_type="api_error",
        )

    code = _extract_code_from_response(raw_str)
    code_ans = None
    code_err = ""
    if code:
        ok, output = PythonExecutor.execute(code)
        if ok:
            code_ans = _parse_numeric(output)
        else:
            code_err = f"code_execution_failed:{output[:120]}"

    ans = code_ans
    if ans is None:
        m = re.search(r"ANSWER:\s*\[\[([^\]]+)\]\]", raw_str, re.IGNORECASE)
        if m:
            ans = _parse_numeric(m.group(1))
    if ans is None:
        ans = _parse_numeric(raw_str)

    err = "" if ans is not None else (code_err or "no_number_in_output")
    return BaselineResult(
        answer=ans, raw=raw_str, num_llm_calls=1,
        tokens_estimated=_estimate_tokens(full_prompt, max_tokens),
        time_s=elapsed, error_type=err,
        meta={
            "code_present":   code is not None,
            "code_succeeded": code_ans is not None,
        },
    )


# =====================================================================
# Convenience: registry for the runner
# =====================================================================

# Maps baseline_id → callable(client, problem) -> BaselineResult.
# B5/B6/B7 are pipeline-based and constructed in the runner with different
# enable_siv/enable_sht flags — they don't fit this signature.
#
# [v10.4] PAL and PoT added as strong code-execution baselines — these are
# the most direct competitors to MAS-SHT's Architect+Engineer scaffold and
# are MANDATORY in any honest comparison.
BASELINE_REGISTRY = {
    "b1_direct":          direct_answer,
    "b2_cot":             chain_of_thought,
    "b3_sc5":             lambda c, p: self_consistency(c, p, n=5),
    "b4_baseline_only":   baseline_only,
    "b_pal":              pal,
    "b_pot":              pot,
}
