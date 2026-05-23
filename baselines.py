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
                     max_tokens: int = 500,
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
# update this string in lock-step.
BASELINE_PIPELINE_PROMPT = (
    "{problem}\n\nSolve this step-by-step. End with: ANSWER: [[numeric_value]]"
)


def baseline_only(client: UnifiedLLMClient, problem: str,
                  max_tokens: int = 500) -> BaselineResult:
    """B4 — re-runs the pipeline's BASELINE agent prompt as a standalone
    system, so we can see how much of MAS-SHT's gain comes from prompt
    engineering alone vs the multi-agent scaffold."""
    t0 = time.time()
    prompt = BASELINE_PIPELINE_PROMPT.format(problem=problem)
    msgs = [{"role": "user", "content": prompt}]
    raw = client.call_model(msgs, temperature=0.1, max_tokens=max_tokens)
    elapsed = time.time() - t0
    raw_str = str(raw)[:2000]

    if _is_error_response(raw):
        return BaselineResult(
            answer=None, raw=raw_str, num_llm_calls=1,
            tokens_estimated=_estimate_tokens(prompt, max_tokens),
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
        tokens_estimated=_estimate_tokens(prompt, max_tokens),
        time_s=elapsed,
        error_type="" if ans is not None else "no_number_in_output",
    )


# =====================================================================
# Convenience: registry for the runner
# =====================================================================

# Maps baseline_id → callable(client, problem) -> BaselineResult.
# B5/B6/B7 are pipeline-based and constructed in the runner with different
# enable_siv/enable_sht flags — they don't fit this signature.
BASELINE_REGISTRY = {
    "b1_direct":          direct_answer,
    "b2_cot":             chain_of_thought,
    "b3_sc5":             lambda c, p: self_consistency(c, p, n=5),
    "b4_baseline_only":   baseline_only,
}
