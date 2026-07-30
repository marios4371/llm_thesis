"""
Enhanced Reasoning Quality Evaluation System for MAS Math Solver
VERSION 14.0: Structural Blueprint Repair (SIV Layer 0)

CHANGELOG v14.0 (over v13.0) — driven by a full re-analysis of the v13.0 run
AND its no-SIV ablation (`mas_full_20260712_02.csv` vs
`mas_no_siv_20260719.csv`, both n=150, same problems, same preset). The
ablation had already run but was never analysed: 72.67% with SIV vs 72.00%
without, paired 3 wins / 2 losses, McNemar p~1.0. SIV was causally NEUTRAL.
The diagnosis is not that SIV's logic is wrong — as a detector it is close
to perfect (audit_passed=False implies a wrong blueprint answer 66/66 times,
100% precision). The diagnosis is attrition, in three measured places:

- [FIX] LAYER-0 STRUCTURAL DEFECTS WERE INVISIBLE. 30 of the 109 audited
  problems returned blueprint_answer=None / invertible=False. Reproduced
  locally: this is what happens when an equation's right-hand side names
  something that is neither a declared given nor an earlier result
  (`t = givens['a'] * rate`) — parse_expr mints a free Symbol, and the
  defect only surfaces as a float() failure deep in forward substitution.
  SIV now scans for this BEFORE the algebra (new SIVResult fields
  `undefined_symbols`, `undeclared_given_keys`, `has_structural_defect`)
  and reports the offending name.

- [FIX] THE REPAIR LOOP SKIPPED EXACTLY THOSE 30 ROWS. The guard in
  _attempt_blueprint_repair read `not siv_result.invertible -> return`,
  so the worst blueprints in the run were the only ones never repaired.
  A chain that cannot be evaluated is a STRONGER reason to repair, not a
  reason to give up. Now gated on `not audit_passed and not skipped`, with
  the new `skipped` flag separating "SIV had nothing to audit" from "SIV
  audited and the audit failed" (previously conflated, since both left
  execution_audit_passed=False).

- [FIX] THE REPAIR PROMPT LED WITH SIV'S WEAKEST OUTPUT. It was built on
  Layer 2's `failed_givens`, whose precision is 1/n_used (v12.2 finding) —
  in effect telling the Mathematician "all your givens are suspect".
  get_error_localization_report now leads with the named structural defect,
  then Layer 1's concrete discrepancy (blueprint evaluates to X, code
  produced Y), and demotes the implicated set to last. Measured motivation:
  fix-rate was 6/39 (15.4%) despite 80.3% detection recall.

- [DESIGN] TAUTOLOGICAL AUDITS ARE NOW FLAGGED, NOT SILENTLY COUNTED. When
  the audited answer was itself produced by evaluating the blueprint's own
  equations (SymPy symbolic fallback), Layer 1 passes by construction. The
  main gate already excluded those rows by agent name; the repair path did
  NOT, and 4/4 such rows in the v13.0 run came back verified=True while 2
  of them were plainly wrong. New `tautological` flag marks them so they
  can be excluded from detector statistics instead of inflating them.

- [NEW] BLUEPRINT PROVENANCE LOGGING. givens/equations/provenance now reach
  the result dict, so blueprint quality is measurable offline and SIV can be
  replayed on past runs. Nothing about blueprint content was recorded before,
  which is why every diagnosis to date had to be inferred from a single
  numeric column.

Deliberately NOT changed: the do-no-harm invariant and the selection layer.
On this run the blueprint path is uniquely right on 2/150 problems while the
baseline is uniquely right on 90, so oracle{baseline, blueprint} = 76.7% vs
the baseline's 75.3% — any selection-layer work has a +1.4pp ceiling until
blueprint quality itself improves. Also unchanged: check_correctness's
absolute 1e-3 tolerance, which mis-grades ~5 gsm-hard rows (both systems
nearly equally) — left alone deliberately for consistency with runs already
reported.

VERSION 13.0: Baseline-Anchored Selection (Do-No-Harm Invariant)

CHANGELOG v13.0 (over v12.1) — driven by the first mixed-preset run
(qwen_math7b_mixed_mathematician_instruct, n=150, v12.2 dataset mix,
2026-07-11/12). That run achieved the coverage goal (blueprint_empty fell
from 62.7% to 27.3%; SIV audited 109/150) but accuracy COLLAPSED to 57.3%
vs 74.7% for plain CoT on the same mix — the selection layer, not the
models, destroyed the value: the run's own internal zero-shot baseline was
right 75.3% of the time. Three measured failures and their fixes:

- [DESIGN] THE DO-NO-HARM INVARIANT. The system may override its internal
  zero-shot baseline ONLY on positive, verifiable corroboration: >= 2
  honestly-independent derivations agreeing on the same value. With no
  corroboration anywhere, the final answer IS the baseline answer (new
  triage value 'baseline_default'). This makes "never lose to the simple
  baseline" a structural property instead of a hope. Measured basis: on
  the 55 judge_fallback rows the old path picked the primary and scored
  8/55 while the discarded baseline held 29/55; counterfactual replay of
  this run's own rows under the invariant = 113/150 (75.3%) vs the
  measured 86/150 (57.3%).

- [FIX] PSEUDO-VOTES. `blueprint_eval` (the CAS evaluation of the
  Architect's equations) can only co-group with the primary when the SIV
  execution audit passed — which certifies the Programmer's code computed
  THE SAME equations. That is one derivation counted twice, not two
  independent derivations, yet both `_triage_candidates` (majority) and
  `_evidence_score` (votes) counted it as 2 and outvoted the genuinely
  independent baseline. Measured: 36 'majority' rows ~= the 37
  audit-passed rows; baseline right on 30/36, the pair-vote answer on
  24/36 (8 correct baselines destroyed to save 2). Vote counting now
  dedupes the {primary, blueprint_eval} pair everywhere.

- [FIX] SIV VERIFICATION IS A CROSS-VALIDATOR, NOT AN AUTHORITY. Measured
  on this run: siv_verified with baseline AGREEMENT -> 22/22 correct
  (100%); siv_verified with baseline DISAGREEMENT -> 2/14 correct (14%).
  A verification that certifies blueprint-internal consistency inherits
  the blueprint's translation errors (the paper's SIV scope claim, now
  with field numbers — and the same phenomenon v10.1's gate ordering was
  built on: 3/3 regressions in the v10.0 n=50 run). `siv_full` therefore
  no longer auto-resolves the evidence ranking on its own; it remains a
  deterministic tie-breaker BETWEEN corroborated groups. The
  confident-skip gate is untouched — it already required baseline
  agreement before any SIV skip (97.9% accurate on this run).

- [FIX] `_deterministic_fallback` preferred the primary (the v12.1
  coin-flip bug's twin, in the one function v12.1 didn't reach because
  the judge never used to fail). It now prefers the baseline, per the
  invariant. The constrained judge itself is retained ONLY for ties
  between corroborated groups (rare); its output contract is tag-FIRST
  (`SELECTED_CANDIDATE: [[k]]` on line 1) so truncation can no longer
  destroy the tag — on this run it parsed 0/55 attempts (Math-7B judge
  wrote free CoT until the token cap).

- [PRESET] `qwen_math7b_mixed` (renamed from
  qwen_math7b_mixed_mathematician_instruct): Instruct now also serves
  HYPOTHESIS_GENERATOR and JUDGE — every role whose output must be
  parsed (JSON blueprints, alternative blueprints, constrained tag)
  runs on the format-following model; BASELINE and PROGRAMMER stay on
  Math-7B (raw arithmetic strength). Zero extra VRAM: roles share
  clients keyed by (provider, model, 4bit) — still exactly 2 loads.

- [FIX] local_hf multi-GPU OOM: on a T4 x2 session, loading the SECOND
  4-bit model OOM'd repeatedly (accelerate placed it on one GPU whose
  free memory did not survive materialization; observed 2026-07-12, the
  Mathematician never loaded and every blueprint died with
  ERROR_GENERATION while pre-flight still PASSed). Multi-GPU 4-bit loads
  now pass an explicit free-memory-aware max_memory map so placement
  reflects what is actually free. Pre-flight (notebook Cell 2.5) also
  gains a Mathematician-liveness check: a mixed-preset run with zero
  formed blueprints across the test problems now FAILS loudly instead
  of burning an 8h commit in fallback mode.

UPDATE (same v13.0, pre-first-run) — blueprint repair loop, and a second
do-no-harm gap closed:

- [NEW] `_attempt_blueprint_repair`: one repair attempt, gated on
  execution_audit_passed=False. Diagnostic basis (mas_full_20260712, the
  same mixed-preset run): among audited rows whose blueprint answer was
  wrong, this exact signal caught 49/61 (80.3% recall) with 0/25 false
  alarms on blueprints that were actually correct — by far the
  highest-leverage lever available, versus the ~20% (12/61) that are
  translation errors invisible to any equation-level check. On audit
  failure the Mathematician now gets SIV's own error-localization report
  and one chance to re-derive the blueprint BEFORE the primary answer
  reaches the confidence gate or SHT — SIV used as an in-loop repair
  signal, not only a post-hoc detector. `_structured_hypothesis_testing`
  takes a `pre_sht_calls` parameter (default 3, unchanged for every other
  caller) so the +2 real calls a repair spends are reflected in
  api_calls_used instead of silently undercounted. New CSV/result field
  `blueprint_repaired`.
- [FIX] The v11.4 SIV-anchor override (fully audited + verified +
  rel_error<1e-3 replaces the SHT answer outright) was the one place in
  `solve()` that still let self-consistency alone override a disagreeing
  baseline — exactly the pattern the rest of v13.0 was built to close, just
  not yet reached because it lives after `_structured_hypothesis_testing`
  returns. Measured on the same run: siv_verified WITH baseline agreement
  = 22/22 correct; WITH baseline disagreement = 2/14. The anchor now only
  fires when the baseline agrees or has itself failed; a disagreeing
  baseline suppresses it (logged, SHT's answer is kept).

CHANGELOG v12.1 (over v12.0) — driven by the first FULL v12.0 run
(qwen_math_7b_local, n=150, 2026-07-04). Result: 82.7% (124/150), up from
74.0% under the pre-v12 selection layer on the same problem set — the v12.0
fixes validated directly (+8.7pp). confident_skip 96.9% (96/150), unanimous
100% (18/150); SIV-verified answers 100% correct (12/12) vs 65.9% for
unverified, the first clean confirmation that SIV's signal is trustworthy.
Two remaining triage paths (evidence 34.6%, majority 40.0%) were still below
the confident-skip level, and one specific structural bug accounted for
most of the shortfall:
- [FIX] `_select_by_evidence` no longer auto-resolves a group ranking that
        has no real corroboration. v12.0 resolved the score ranking whenever
        exactly one group scored highest — but in the single most common
        no-majority case (a lone primary vs a lone baseline, no SIV signal,
        each its own group of size 1), the `executed` scoring component
        always favored the primary's group (it always has code; the
        zero-shot baseline never does). That is not evidence of correctness,
        just an artifact of which candidate happens to run code, and it
        resolved identically regardless of which side was actually right.
        Measured on the v12.0 run: of the 21 problems that reached exactly
        this tie, the primary was correct on 7 and the baseline was correct
        on a disjoint other 7 — a coin flip. Always picking the primary
        recovered its own 7 while converting all 7 baseline-correct cases
        into regressions (silent damage to already-correct answers, the
        outcome class the project treats as worst-case). `_select_by_evidence`
        now only auto-resolves when the winning group is genuinely
        corroborated — multiple independent derivations agree (votes >= 2,
        reachable here only via a size-tied 2-vs-2 "no majority" case) or SIV
        fully verified that exact value. A field of lone, unverified
        singletons (whichever candidates they are) is not decided
        deterministically at all; every one of them is now handed to the
        constrained judge, which is the mechanism designed for genuine ties.
- Not yet re-measured: this fix responds to a bug discovered by analyzing
  the v12.0 run's own CSV, not to a second full run. The next Kaggle
  execution is the first empirical measurement of v12.1.

NOTE — post-v12.1 scope correction (documentation/evaluation only, no
solve()/SHT behavior change, SOLVER_VERSION unchanged at "12.1"):
Analyzing the v12.0 run showed SIV Layer 2 (fault localization) genuinely
fires on real multi-variable blueprints only 1/150 times — far too rarely to
validate empirically from field data. A new controlled fault-injection suite
(test_siv_fault_injection.py) corrupts one given at a time across 12
synthetic blueprints (37 single-given corruptions, 1-5 used givens each) and
calls SymbolicInverseVerifier.verify() directly. Result: recall (the true
fault is always present in failed_givens) = 100%; exclusion (declared-but-
unused givens correctly separated from used ones) = 100%; but exact
isolation (failed_givens names ONLY the corrupted variable) = 0% whenever
2+ givens are used, with mean precision landing at exactly 1/n_used (50% at
n=2, down to 20% at n=5). This is a structural property, not a bug: every
MAS-SHT blueprint's equations collapse to one scalar expression
`answer = f(g1,...,gn)`, and holding all other givens fixed at their
declared values while inverting for one at a time generically shows EVERY
used given as inconsistent whenever the forward audit fails — not just the
truly corrupted one. siv_module.py's docstrings, SIVResult.failed_givens
field doc, and get_error_localization_report() are corrected to describe
Layer 2 honestly: a validated distractor/relevance filter and an implicated-
variable SUPERSET (never misses the true fault), not a point localizer. The
paper's claims are corrected to match (removed: "identifies precisely which
variable is inconsistent" / "per-variable fault localization" as a FOBAR
differentiator; added: the quantified fault-injection methodology and
numbers above as an honest, thesis-worthy validation of what Layer 2 does
and does not deliver).

CHANGELOG v12.0 (over v11.4) — driven by the n=150 Kaggle pilot
(qwen_math_7b_local, 2026-07-02). Pilot diagnosis: confident_skip path 96%
accurate (101/150 problems), unanimous 89%, but the judge triage path scored
15% on 40 problems where its own internal baseline scored 42.5% — the LLM
judge was the single component destroying the system (16 regressions vs 6
fixes; final 74.0% vs PAL 88.0%). Fixes:
- [FIX][CRITICAL] v11.4 SIV-anchor guard `(execution_rel_error or 1.0) < 1e-3`
        never fired on a PERFECT audit: rel_error == 0.0 is falsy, so
        `0.0 or 1.0` evaluated to 1.0 and the anchor was skipped exactly when
        SIV evidence was strongest. Confirmed in pilot: gsm-hard_1125
        (rel_err=5.5e-4) was anchored → fixed; gsm-hard_542 (rel_err=0.0,
        blueprint answer == gold) was NOT anchored → judge output "3".
        Now uses an explicit `is not None` check.
- [FIX][CRITICAL] _judge_hypotheses: judge is now CONSTRAINED to select one
        of the presented candidates by index (SELECTED_CANDIDATE: [[k]]).
        Previously, when the SELECTED_ANSWER tag regex failed, the code fell
        back to _extract_last_number(judge_free_text) — which returned
        candidate indexes, confidences and other trailing junk as "answers"
        (pilot regressions with predicted ∈ {1,2,3,4,5,8} against
        3-4 digit golds). A free-form number that matches no candidate is
        now rejected; on any parse failure the caller falls back to
        deterministic evidence-based selection, never to text scraping.
- [NEW] Evidence-based arbitration (_select_by_evidence): before the LLM
        judge is consulted, candidate answer groups are scored on
        deterministic evidence only: (#independent votes, CAS/blueprint
        agreement, SIV full verification, executed-code support, baseline
        support). A strict winner is selected without any LLM call
        (triage_result='evidence'). The LLM judge remains only as a
        tie-breaker among the top-scored groups.
- [NEW] Blueprint-evaluation candidate: when SIV's execution audit produced
        a numeric blueprint answer, that value enters SHT as a zero-cost
        candidate ('blueprint_eval'). Together with the PAL-style Programmer
        (below) this gives three INDEPENDENT derivations of the answer —
        free program, declarative blueprint (CAS-evaluated), zero-shot CoT —
        and majority/evidence voting over them is meaningful.
- [CHANGE] Programmer is now PAL-style: it solves the PROBLEM directly
        (the pilot's plain-PAL baseline scored 88.0% vs 74.0% for the full
        pipeline) and receives the Architect blueprint as a colleague's
        draft to cross-check, not as equations to transcribe. The
        `givens = {...}` first-line convention is kept so rule-based
        verification, metamorphic testing and SIV keep working. This makes
        SIV's execution audit a genuine independent cross-check (program vs
        blueprint) instead of a tautology (blueprint vs itself).
- [FIX] Primary-candidate validity in SHT is now based on actual code
        execution success, not on verification-adjusted confidence. The old
        `confidence > 0.5` check silently dropped the primary answer from
        arbitration whenever rule-based verification disagreed with the
        blueprint (confidence floor 0.3) — precisely the cases where
        arbitration matters most.
- [REMOVED] SymPy post-verification REPLACEMENT of the Programmer answer
        (pilot accuracy of that path: 33%, vs 50% for the answers it
        replaced). SymbolicSolver output now enters SHT as a candidate via
        the blueprint_eval mechanism instead of overwriting the primary.
        The programmer-hard-fail SymPy fallback (68.8% in pilot) is kept.
- [FIX] _confidence_gate criterion 5: a negative primary answer no longer
        triggers SHT when the baseline AGREES with it (two independent
        paths agreeing on a negative is evidence, not an anomaly — GSM-Hard
        golds are frequently negative; pilot: b_pal 83% vs MAS 17% on
        negative-gold problems). Negative answers trigger SHT only when the
        baseline also failed (no cross-check available).
- [NEW] SOLVER_VERSION constant exported for experiment provenance; the
        notebook runner stamps every CSV row and auto-discards checkpoints
        written by a different solver version.

CHANGELOG v10.2 (over v10.1):
- [NEW] HuggingFace Inference Providers Router support (provider="huggingface").
        Calls https://router.huggingface.co/{provider}/v1/chat/completions
        (OpenAI-compatible). Handles 503 cold-start with exponential backoff
        up to 60s. Auth via HF_API_KEY.
- [NEW] Together AI provider (provider="together"). OpenAI-compatible
        endpoint at https://api.together.xyz/v1. Auth via TOGETHER_API_KEY.
        Realistic path for serverless 7B math models (DeepSeek-R1-Distill,
        Qwen2.5-Math-7B) since they are not free on HF anymore.
- [NEW] Local HuggingFace provider (provider="local_hf"). Loads via
        transformers.AutoModelForCausalLM, runs on CUDA if available else CPU.
        Designed for 1.5B math models inside Colab T4. Zero API calls,
        no rate limit. Model is cached on the client instance.
- [NEW] Six new HETEROGENEOUS_PRESETS for small open math models:
        tiny_math_homogeneous, deepseek_distill_1_5b (local_hf, 1.5B);
        small_math_homogeneous, qwen_math_7b (together, 7B);
        phi4_mini (huggingface, 3.8B);
        small_vs_large (1.5B baseline + 70B everywhere else).
- [NEW] Pipeline-level ablation flags: enable_siv and enable_sht.
        These are kwargs on QualityAwarePipeline and threaded through to
        QualityEnhancedMultiAgentSolver. Used by baselines.py to implement
        B5 (MAS-NoSIV) and B6 (MAS-NoSHT) without duplicating the pipeline.
        v10.1 confidence gate ordering invariant is preserved unconditionally.
- [NEW] Per-provider rate limiters:
        hf_limiter (10 RPM), together_limiter (20 RPM).
        TokenBudget remains Groq-TPD-specific (not extended to other providers).

CHANGELOG v10.1 (over v10.0):
- [FIX] _confidence_gate: SIV "skip SHT" path no longer overrides baseline
        cross-check. Reordered criteria so baseline_disagreement is evaluated
        BEFORE the SIV-verified skip. Rationale: SIV cannot detect NL→math
        translation errors (it audits whatever blueprint the Architect produced),
        so a SIV pass is mathematical-consistency evidence only. The baseline
        answer comes from a different reasoning path and is the only translation-
        layer signal available — it must dominate.
        Validated on n=50 google homogeneous run: all 3 observed regressions
        (where baseline was correct and SIV-verified MAS answer was wrong) now
        correctly trigger SHT instead of confident_skip.
- [FIX] _confidence_gate: SIV-verified skip now requires baseline agreement;
        SIV's failure paths (siv_inconsistency, siv_execution_error) preserved.

CHANGELOG v10.0 (over v9.1):
- [NEW] Symbolic Inverse Verification (SIV): two-layer symbolic execution audit
        using SymPy CAS (zero LLM API calls)
        Layer 1 — Execution Audit: verifies that Programmer's answer matches
          what the blueprint equations actually evaluate to (forward check).
        Layer 2 — Fault Localization: per-variable inverse solve to identify
          which declared given is inconsistent with the computed answer.
- [NEW] Unused-given detection: flags givens declared in blueprint but absent
        from any equation (potential distractors or missing equations).
- [NEW] Ambiguity detection: reports multi-root cases (non-uniquely invertible chains).
- [NEW] SIV-gated SHT: when execution audit passes, SHT may be skipped;
        when SIV detects inconsistency, SHT is triggered with localization context.
- [NEW] SIV-informed Critic: fault localization report passed to hypothesis generator
        for targeted repair instead of blind re-generation.
- [NEW] CSV output: siv_verified, siv_confidence, siv_failed_givens,
        siv_execution_audit_passed, siv_unused_givens.
- [KNOWN LIMITATION — explicitly documented]:
    SIV operates on the math→math layer (blueprint→answer). It CANNOT detect
    errors in the NL→math layer (problem text→blueprint). If the Architect
    modelled the problem incorrectly, SIV will audit the wrong blueprint faithfully.
- [RELATIONSHIP TO FOBAR (Jiang et al., ACL 2024)]:
    FOBAR: LLM-based backward verification — probabilistic, operates partially on
           NL layer, gives binary verdict only.
    SIV:   CAS-based, deterministic, zero LLM calls — operates on execution layer,
           gives per-variable fault localization.
    The two methods are ORTHOGONAL: FOBAR targets translation errors; SIV targets
    execution errors. Their combination is strictly stronger than either alone.

CHANGELOG v9.1 (over v9.0):
- [FIX] RPM reduced from 30 → 12 to stay within Groq free tier TPM (~6K tokens/min for 70B)
- [FIX] Inter-problem cooldown (3s with SHT, 2s without) to prevent TPM bursts
- [FIX] Reduced max_tokens: baseline 800→500, hypothesis gen 1200→900, judge 800→500
- [FIX] Retry attempts reduced from 6 → 4 to avoid cascading backoffs

CHANGELOG v7.3 (over v7.2):
- [NEW] Heterogeneous Model Configuration: each agent role can use a different LLM
- [NEW] AgentRole enum (BASELINE, MATHEMATICIAN, PROGRAMMER, HYPOTHESIS_GENERATOR, JUDGE)
- [NEW] ModelConfig dataclass + HETEROGENEOUS_PRESETS (5 presets)
- [NEW] UnifiedLLMClient accepts model_override parameter
- [NEW] Solver uses _get_client(role) — dispatches to role-specific client
- [NEW] Pipeline supports heterogeneous_preset and custom_config params
- [NEW] CSV output includes model_config per role for experiment tracking
- [FIX] Client deduplication: same (provider, model) pair shares one connection

Inherits v7.1/v7.2 fixes:
- Error response detection, token budget, 429 handling, cache, auth detection
"""

from __future__ import annotations

import os
import re

# [v12.0] Experiment provenance: stamped into every CSV row by the notebook
# runner; checkpoints from a different solver version are auto-discarded so
# results never mix selection policies.
SOLVER_VERSION = "14.0"
import json
import time
import random
import hashlib
import threading
import pickle
import logging
import ast
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from dotenv import load_dotenv

# --- [NEW v10.0] Symbolic Inverse Verification ---
from siv_module import (
    SymbolicInverseVerifier, SIVResult, GivenReconstruction
)

# --- Statistical Libraries Check ---
try:
    from statsmodels.stats.contingency_tables import mcnemar
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from scipy.stats import chi2
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: 'datasets' library not found. Standard curated set will be used.")

# --- LLM Providers ---
# [v14.0] Guarded like every other provider import below. The only users are the
# Groq and Together clients (both OpenAI-compatible endpoints); local_hf and the
# offline test/analysis paths need none of it. Unguarded, this single line made
# the whole module unimportable without the package, which blocked running the
# offline test suites on a machine that has no API access.
try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False

    def OpenAI(*_args, **_kwargs):  # noqa: N802 - stands in for the real class
        raise ImportError(
            "The 'openai' package is required for the groq/together providers "
            "(pip install openai). Not needed for local_hf or offline analysis."
        )

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# --- [NEW v8.0] Symbolic Solver ---
try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("Warning: 'sympy' not found. Symbolic solver fallback disabled. pip install sympy")


# --------------------------- Configuration ---------------------------

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# [NEW v10.2] Optional keys for small-open-model providers — only required if
# the user selects a preset that targets the given provider. We do NOT crash
# at import time if these are missing.
HF_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# [NEW v10.2] Lazy-import flags for non-mandatory provider deps. We probe at
# call sites so that a Groq-only run never imports torch/transformers.
try:
    import requests as _requests  # used by huggingface + together providers
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# transformers / torch are heavy; do NOT import here. We import inside the
# local_hf provider branch on first call.
TRANSFORMERS_AVAILABLE = None  # tri-state: None=not probed, True/False=probed


# ========================== [NEW v7.3] Heterogeneous Model Config ==========================

class AgentRole(Enum):
    """
    Each agent role in the MAS-SHT pipeline can be assigned
    to a different provider + model combination.
    """
    BASELINE = "baseline"
    MATHEMATICIAN = "mathematician"
    PROGRAMMER = "programmer"
    HYPOTHESIS_GENERATOR = "hypothesis_generator"
    JUDGE = "judge"

@dataclass
class ModelConfig:
    """
    Configuration for a single model endpoint.

    provider:   "groq" | "google" | "huggingface" | "together" | "local_hf"
    model_name: specific model string; None = provider default.
    load_4bit:  [v10.3] local_hf only. Load model in 4-bit (BitsAndBytes NF4)
                so that 7B models fit on a 16 GB T4 GPU (~4 GB VRAM).
                Ignored for all non-local_hf providers.
    """
    provider: str = "groq"
    model_name: Optional[str] = None  # None = use provider default
    load_4bit: bool = False            # [v10.3] 4-bit quant for local_hf 7B models


# Pre-defined heterogeneous configurations
# Users can also build custom configs

HETEROGENEOUS_PRESETS: Dict[str, Dict[AgentRole, ModelConfig]] = {
    # All roles use the same model (backward compatible, same as v7.2)
    "homogeneous_groq": {
        AgentRole.BASELINE:              ModelConfig("groq", "qwen/qwen3-32b"),
        AgentRole.MATHEMATICIAN:         ModelConfig("groq", "qwen/qwen3-32b"),
        AgentRole.PROGRAMMER:            ModelConfig("groq", "qwen/qwen3-32b"),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("groq", "qwen/qwen3-32b"),
        AgentRole.JUDGE:                 ModelConfig("groq", "qwen/qwen3-32b"),
    },
    
    # Cross-architecture diversity: different model families per role
    # Key insight: diversity in model architecture → diversity in reasoning patterns
    "diverse_groq": {
        AgentRole.BASELINE:              ModelConfig("groq", "gemma2-9b-it"),              # Different family for baseline diversity
        AgentRole.MATHEMATICIAN:         ModelConfig("groq", "llama-3.3-70b-versatile"),   # Best reasoning for blueprint
        AgentRole.PROGRAMMER:            ModelConfig("groq", "llama-3.3-70b-versatile"),   # Needs precise instruction following
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("groq", "mixtral-8x7b-32768"),        # MoE architecture → diverse strategies
        AgentRole.JUDGE:                 ModelConfig("groq", "llama-3.3-70b-versatile"),   # Needs strong evaluation
    },
    
    # Cross-provider diversity: use both Groq and Google
    "cross_provider": {
        AgentRole.BASELINE:              ModelConfig("google", "gemini-2.5-flash-lite"),                       # Gemini as independent baseline
        AgentRole.MATHEMATICIAN:         ModelConfig("groq", "llama-3.3-70b-versatile"),   # LLaMA for structured JSON output
        AgentRole.PROGRAMMER:            ModelConfig("groq", "llama-3.3-70b-versatile"),   # LLaMA for code generation
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("google", "gemini-2.5-flash-lite"),                       # Gemini for diverse strategies
        AgentRole.JUDGE:                 ModelConfig("groq", "llama-3.3-70b-versatile"),   # LLaMA for final judgment
    },
    
    # Budget-optimized: small models where possible, large only where critical
    "budget_optimized": {
        AgentRole.BASELINE:              ModelConfig("groq", "llama-3.1-8b-instant"),      # Fast & cheap baseline
        AgentRole.MATHEMATICIAN:         ModelConfig("groq", "llama-3.3-70b-versatile"),   # Full power for blueprint
        AgentRole.PROGRAMMER:            ModelConfig("groq", "llama-3.1-8b-instant"),      # Small model can follow blueprints
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("groq", "llama-3.3-70b-versatile"),   # Needs creativity
        AgentRole.JUDGE:                 ModelConfig("groq", "llama-3.3-70b-versatile"),   # Needs strong judgment
    },
    
    # Homogeneous Google
    "homogeneous_google": {
        AgentRole.BASELINE:             ModelConfig("google", "gemini-3.1-flash-lite-preview"),
        AgentRole.MATHEMATICIAN:         ModelConfig("google", "gemini-3.1-flash-lite-preview"),
        AgentRole.PROGRAMMER:            ModelConfig("google", "gemini-3.1-flash-lite-preview"),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("google", "gemini-3.1-flash-lite-preview"),
        AgentRole.JUDGE:                 ModelConfig("google", "gemini-3.1-flash-lite-preview"),
    },

    # =====================================================================
    # [NEW v10.2] Small open math models — for comparison vs large LLMs.
    # These presets run the full MAS-SHT pipeline on small models so we can
    # measure whether the multi-agent scaffold compensates for raw model size.
    # =====================================================================

    # Tiny math model homogeneous — Qwen2.5-Math 1.5B running locally (Colab T4 friendly).
    # Zero API cost, zero rate limit. Best for full-GSM8K runs without hitting any quota.
    "tiny_math_homogeneous": {
        AgentRole.BASELINE:              ModelConfig("local_hf", "Qwen/Qwen2.5-Math-1.5B-Instruct"),
        AgentRole.MATHEMATICIAN:         ModelConfig("local_hf", "Qwen/Qwen2.5-Math-1.5B-Instruct"),
        AgentRole.PROGRAMMER:            ModelConfig("local_hf", "Qwen/Qwen2.5-Math-1.5B-Instruct"),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("local_hf", "Qwen/Qwen2.5-Math-1.5B-Instruct"),
        AgentRole.JUDGE:                 ModelConfig("local_hf", "Qwen/Qwen2.5-Math-1.5B-Instruct"),
    },

    # DeepSeek-R1-Distill 1.5B locally — reasoning-distilled tiny model.
    "deepseek_distill_1_5b": {
        AgentRole.BASELINE:              ModelConfig("local_hf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
        AgentRole.MATHEMATICIAN:         ModelConfig("local_hf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
        AgentRole.PROGRAMMER:            ModelConfig("local_hf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("local_hf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
        AgentRole.JUDGE:                 ModelConfig("local_hf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    },

    # [v10.3] Qwen2.5-Math 7B — math-specialist, loaded in 4-bit NF4 on T4 GPU.
    # ~4 GB VRAM footprint. Significantly better instruction-following than 1.5B:
    # can produce reliable JSON blueprints without the CoT-fallback workaround.
    # Expected to show the MAS architecture benefit that 1.5B models cannot.
    "qwen_math_7b_local": {
        AgentRole.BASELINE:              ModelConfig("local_hf", "Qwen/Qwen2.5-Math-7B-Instruct",          load_4bit=True),
        AgentRole.MATHEMATICIAN:         ModelConfig("local_hf", "Qwen/Qwen2.5-Math-7B-Instruct",          load_4bit=True),
        AgentRole.PROGRAMMER:            ModelConfig("local_hf", "Qwen/Qwen2.5-Math-7B-Instruct",          load_4bit=True),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("local_hf", "Qwen/Qwen2.5-Math-7B-Instruct",          load_4bit=True),
        AgentRole.JUDGE:                 ModelConfig("local_hf", "Qwen/Qwen2.5-Math-7B-Instruct",          load_4bit=True),
    },

    # [v10.3] DeepSeek-R1-Distill 7B — reasoning-chain distilled from R1 (671B).
    # Loaded in 4-bit NF4 on T4 GPU (~4 GB VRAM). Strong chain-of-thought reasoning.
    # Tests whether a reasoning-focused 7B model benefits from MAS decomposition.
    "deepseek_7b_local": {
        AgentRole.BASELINE:              ModelConfig("local_hf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", load_4bit=True),
        AgentRole.MATHEMATICIAN:         ModelConfig("local_hf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", load_4bit=True),
        AgentRole.PROGRAMMER:            ModelConfig("local_hf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", load_4bit=True),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("local_hf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", load_4bit=True),
        AgentRole.JUDGE:                 ModelConfig("local_hf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", load_4bit=True),
    },


    # [v10.3] Qwen2.5-Math 7B local — 4-bit NF4 on T4 GPU (~4 GB VRAM).
    # Reliably produces JSON blueprints (unlike 1.5B). Expected to show real MAS benefit.
    "qwen_math_7b_local": {
        AgentRole.BASELINE:              ModelConfig("local_hf", "Qwen/Qwen2.5-Math-7B-Instruct",          load_4bit=True),
        AgentRole.MATHEMATICIAN:         ModelConfig("local_hf", "Qwen/Qwen2.5-Math-7B-Instruct",          load_4bit=True),
        AgentRole.PROGRAMMER:            ModelConfig("local_hf", "Qwen/Qwen2.5-Math-7B-Instruct",          load_4bit=True),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("local_hf", "Qwen/Qwen2.5-Math-7B-Instruct",          load_4bit=True),
        AgentRole.JUDGE:                 ModelConfig("local_hf", "Qwen/Qwen2.5-Math-7B-Instruct",          load_4bit=True),
    },

    # [v10.6] Qwen2.5-Math 7B fp16 — NO bitsandbytes (bnb 4-bit materialization
    # crashes Kaggle kernels). fp16 weights ≈ 14.2 GB, sharded across all visible
    # GPUs via device_map="auto". Requires Kaggle accelerator "GPU T4 x2" (2×15 GB).
    "qwen_math_7b_fp16": {
        AgentRole.BASELINE:              ModelConfig("local_hf", "Qwen/Qwen2.5-Math-7B-Instruct"),
        AgentRole.MATHEMATICIAN:         ModelConfig("local_hf", "Qwen/Qwen2.5-Math-7B-Instruct"),
        AgentRole.PROGRAMMER:            ModelConfig("local_hf", "Qwen/Qwen2.5-Math-7B-Instruct"),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("local_hf", "Qwen/Qwen2.5-Math-7B-Instruct"),
        AgentRole.JUDGE:                 ModelConfig("local_hf", "Qwen/Qwen2.5-Math-7B-Instruct"),
    },

    # [v11.2] Qwen2.5-7B-INSTRUCT (general, NOT -Math) fp16 — the instruction-tuned
    # sibling. Unlike Qwen2.5-Math (RL-locked to chain-of-thought), this model
    # follows the JSON-blueprint prompt, so the full MAS pipeline (Mathematician →
    # Programmer → SIV → SHT) engages as designed. Same size/family as the Math
    # model = fair comparison. fp16 ≈ 14.2 GB sharded over T4 x2 (no bitsandbytes).
    "qwen_7b_instruct_fp16": {
        AgentRole.BASELINE:              ModelConfig("local_hf", "Qwen/Qwen2.5-7B-Instruct"),
        AgentRole.MATHEMATICIAN:         ModelConfig("local_hf", "Qwen/Qwen2.5-7B-Instruct"),
        AgentRole.PROGRAMMER:            ModelConfig("local_hf", "Qwen/Qwen2.5-7B-Instruct"),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("local_hf", "Qwen/Qwen2.5-7B-Instruct"),
        AgentRole.JUDGE:                 ModelConfig("local_hf", "Qwen/Qwen2.5-7B-Instruct"),
    },

    # [v13.0] Mixed by OUTPUT CONTRACT (renamed from v12.3's
    # qwen_math7b_mixed_mathematician_instruct, which swapped only the
    # Mathematician): every role whose output must be PARSED runs on the
    # instruction-following model, every role graded on raw math runs on the
    # math specialist. Math-7B is RL-tuned to emit free chain-of-thought
    # ending in \boxed{}, not structured output: as Mathematician it produced
    # 62.7% empty blueprints (v12.0/v12.1 run), and as Judge it emitted the
    # SELECTED_CANDIDATE tag 0/55 times (mixed run) — same pathology, three
    # roles. Both models 4-bit NF4, ~8 GB total, single T4; roles share
    # clients keyed by (provider, model, 4bit) so this is still exactly 2
    # model loads.
    #   Instruct : MATHEMATICIAN (JSON blueprint), HYPOTHESIS_GENERATOR
    #              (JSON alternative blueprints — feeds the only channel that
    #              may override the baseline under v13.0's do-no-harm rule),
    #              JUDGE (constrained tag)
    #   Math-7B  : BASELINE (zero-shot CoT anchor), PROGRAMMER (code)
    "qwen_math7b_mixed": {
        AgentRole.BASELINE:              ModelConfig("local_hf", "Qwen/Qwen2.5-Math-7B-Instruct", load_4bit=True),
        AgentRole.MATHEMATICIAN:         ModelConfig("local_hf", "Qwen/Qwen2.5-7B-Instruct",      load_4bit=True),
        AgentRole.PROGRAMMER:            ModelConfig("local_hf", "Qwen/Qwen2.5-Math-7B-Instruct", load_4bit=True),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("local_hf", "Qwen/Qwen2.5-7B-Instruct",      load_4bit=True),
        AgentRole.JUDGE:                 ModelConfig("local_hf", "Qwen/Qwen2.5-7B-Instruct",      load_4bit=True),
    },

    # [v10.3] DeepSeek-R1-Distill 7B local — 4-bit NF4 on T4 GPU (~4 GB VRAM).
    # Reasoning-chain distilled from R1 (671B). Tests whether reasoning-focused 7B
    # benefits from MAS decomposition.
    "deepseek_7b_local": {
        AgentRole.BASELINE:              ModelConfig("local_hf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", load_4bit=True),
        AgentRole.MATHEMATICIAN:         ModelConfig("local_hf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", load_4bit=True),
        AgentRole.PROGRAMMER:            ModelConfig("local_hf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", load_4bit=True),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("local_hf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", load_4bit=True),
        AgentRole.JUDGE:                 ModelConfig("local_hf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", load_4bit=True),
    },

    # 7B math-specialist via Together (serverless, has free credits).
    # DeepSeek-R1-Distill-Qwen-7B is a strong math reasoner.
    "small_math_homogeneous": {
        AgentRole.BASELINE:              ModelConfig("together", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        AgentRole.MATHEMATICIAN:         ModelConfig("together", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        AgentRole.PROGRAMMER:            ModelConfig("together", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("together", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        AgentRole.JUDGE:                 ModelConfig("together", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    },

    # Qwen2.5-Math 7B via Together — fine-tuned specifically for math.
    "qwen_math_7b": {
        AgentRole.BASELINE:              ModelConfig("together", "Qwen/Qwen2.5-Math-7B-Instruct"),
        AgentRole.MATHEMATICIAN:         ModelConfig("together", "Qwen/Qwen2.5-Math-7B-Instruct"),
        AgentRole.PROGRAMMER:            ModelConfig("together", "Qwen/Qwen2.5-Math-7B-Instruct"),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("together", "Qwen/Qwen2.5-Math-7B-Instruct"),
        AgentRole.JUDGE:                 ModelConfig("together", "Qwen/Qwen2.5-Math-7B-Instruct"),
    },

    # Phi-4-mini 3.8B via HF Router — Microsoft's small general model.
    # Tests whether a non-math-specific small model can ride the MAS scaffold.
    "phi4_mini": {
        AgentRole.BASELINE:              ModelConfig("huggingface", "microsoft/Phi-4-mini-instruct"),
        AgentRole.MATHEMATICIAN:         ModelConfig("huggingface", "microsoft/Phi-4-mini-instruct"),
        AgentRole.PROGRAMMER:            ModelConfig("huggingface", "microsoft/Phi-4-mini-instruct"),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("huggingface", "microsoft/Phi-4-mini-instruct"),
        AgentRole.JUDGE:                 ModelConfig("huggingface", "microsoft/Phi-4-mini-instruct"),
    },

    # Mixed: tiny baseline (1.5B local), large pipeline (70B Groq).
    # Question this preset answers: does a weak baseline + strong pipeline
    # outperform a strong baseline + nothing?
    "small_vs_large": {
        AgentRole.BASELINE:              ModelConfig("local_hf", "Qwen/Qwen2.5-Math-1.5B-Instruct"),
        AgentRole.MATHEMATICIAN:         ModelConfig("groq",     "llama-3.3-70b-versatile"),
        AgentRole.PROGRAMMER:            ModelConfig("groq",     "llama-3.3-70b-versatile"),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("groq",     "llama-3.3-70b-versatile"),
        AgentRole.JUDGE:                 ModelConfig("groq",     "llama-3.3-70b-versatile"),
    },

    # =====================================================================
    # [NEW v10.4] STRONG MODEL FAMILIES — for honest cross-family comparison.
    # The v9.1 paper run used ONLY Qwen3-32B; to make any claim about MAS-SHT
    # generalising across architectures, we need at least 2 more families.
    # Pick whichever preset matches the API keys you have.
    # =====================================================================

    # Llama 3.3 70B via Groq — different family than Qwen, similar size class.
    # Use this as a second main-experiment preset alongside homogeneous_groq.
    "homogeneous_llama70b_groq": {
        AgentRole.BASELINE:              ModelConfig("groq", "llama-3.3-70b-versatile"),
        AgentRole.MATHEMATICIAN:         ModelConfig("groq", "llama-3.3-70b-versatile"),
        AgentRole.PROGRAMMER:            ModelConfig("groq", "llama-3.3-70b-versatile"),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("groq", "llama-3.3-70b-versatile"),
        AgentRole.JUDGE:                 ModelConfig("groq", "llama-3.3-70b-versatile"),
    },

    # DeepSeek-Math via Together — strong open math specialist, NOT used in v9.1.
    # Gives an architecturally-diverse second data point on hard benchmarks.
    "homogeneous_deepseek_math": {
        AgentRole.BASELINE:              ModelConfig("together", "deepseek-ai/deepseek-math-7b-rl"),
        AgentRole.MATHEMATICIAN:         ModelConfig("together", "deepseek-ai/deepseek-math-7b-rl"),
        AgentRole.PROGRAMMER:            ModelConfig("together", "deepseek-ai/deepseek-math-7b-rl"),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("together", "deepseek-ai/deepseek-math-7b-rl"),
        AgentRole.JUDGE:                 ModelConfig("together", "deepseek-ai/deepseek-math-7b-rl"),
    },

    # Qwen2.5-Math 72B — top-of-line open math model. Use only if you have
    # the Together credit budget; one full run is ~50-100 USD on full GSM8K.
    "homogeneous_qwen_math_72b": {
        AgentRole.BASELINE:              ModelConfig("together", "Qwen/Qwen2.5-Math-72B-Instruct"),
        AgentRole.MATHEMATICIAN:         ModelConfig("together", "Qwen/Qwen2.5-Math-72B-Instruct"),
        AgentRole.PROGRAMMER:            ModelConfig("together", "Qwen/Qwen2.5-Math-72B-Instruct"),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("together", "Qwen/Qwen2.5-Math-72B-Instruct"),
        AgentRole.JUDGE:                 ModelConfig("together", "Qwen/Qwen2.5-Math-72B-Instruct"),
    },
}


# --------------------------- Cache & Logging ---------------------------

CACHE_FILE = "call_cache_v6.pkl"
CALL_CACHE: Dict[str, Any] = {}

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("MAS_Pipeline")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(ch)
    return logger

logger = setup_logging()

# --------------------------- Helpers ---------------------------

def _make_cache_key(provider: str, model_name: str, messages: List[Dict[str, str]], temperature: float) -> str:
    payload = {"provider": provider, "model": model_name, "messages": messages, "temperature": temperature}
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

class RateLimiter:
    """Enhanced rate limiter with 429-specific backoff and token tracking."""
    def __init__(self, requests_per_minute: int = 12):
        self.delay = 60.0 / max(1, requests_per_minute)
        self.last_call = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self.last_call = time.time()


class TokenBudget:
    """
    [NEW v7.2] Tracks estimated token usage against daily limit.
    Groq free tier: 100,000 tokens/day (TPD).
    """
    def __init__(self, daily_limit: int = 100_000):
        self.daily_limit = daily_limit
        self.tokens_used = 0
        self.lock = threading.Lock()
        self._warning_issued = False
    
    def estimate_tokens(self, messages: List[Dict[str, str]], max_tokens: int) -> int:
        """Realistic estimate: input_chars/4 + max_tokens*0.35 (models rarely use full max)."""
        input_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_input = input_chars // 4
        estimated_output = int(max_tokens * 0.35)
        return estimated_input + estimated_output
    
    def record_usage(self, estimated_tokens: int) -> None:
        with self.lock:
            self.tokens_used += estimated_tokens
    
    def can_afford(self, estimated_tokens: int) -> bool:
        with self.lock:
            remaining = self.daily_limit - self.tokens_used
            if remaining < estimated_tokens:
                if not self._warning_issued:
                    logger.warning(
                        f"TOKEN BUDGET: ~{self.tokens_used:,} used of {self.daily_limit:,} daily limit. "
                        f"Need ~{estimated_tokens:,} but only ~{remaining:,} remaining."
                    )
                    self._warning_issued = True
                return False
            return True
    
    def remaining(self) -> int:
        with self.lock:
            return max(0, self.daily_limit - self.tokens_used)
    
    def usage_report(self) -> str:
        pct = (self.tokens_used / self.daily_limit) * 100
        return (f"Token usage: ~{self.tokens_used:,} / {self.daily_limit:,} "
                f"({pct:.1f}%) | ~{self.remaining():,} remaining")

groq_limiter = RateLimiter(requests_per_minute=12)   # Conservative: Groq free tier TPM is ~6K for 70B models
google_limiter = RateLimiter(requests_per_minute=15)
# [NEW v10.2] Per-provider rate limiters for small-open-model paths.
# HF free tier is loose (~30 RPM nominal) but we keep it conservative because
# cold-starts on serverless inflate effective wall-clock.
hf_limiter = RateLimiter(requests_per_minute=10)
together_limiter = RateLimiter(requests_per_minute=20)
# local_hf has no rate limiter — it runs on the local device.
token_budget = TokenBudget(daily_limit=100_000)  # [NEW v7.2] Groq free tier
# NOTE: TokenBudget is intentionally NOT applied to HF/Together/local — the
# 100K/day cap is a Groq-specific TPD limit.

def _safe_json_load(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


# ==========================================================================
# [FIX v7.1] Error Detection Helper
# ==========================================================================

def _is_error_response(text: Any) -> bool:
    """
    Detect whether an LLM response is actually an error message.
    This prevents HTTP status codes (401, 429, 500, etc.) from being
    parsed as numeric answers.
    """
    if text is None:
        return True
    s = str(text).strip()
    if s.startswith("ERROR_GENERATION"):
        return True
    if s.startswith("ERROR_"):  # Catches ERROR_AUTH_401, ERROR_RATE_LIMIT_DAILY, ERROR_BUDGET_EXCEEDED
        return True
    # Check for common API error patterns
    error_patterns = [
        r"error.*(?:401|403|429|500|502|503)",
        r"(?:unauthorized|forbidden|rate.?limit|internal.?server)",
        r"authentication.*(?:failed|error|invalid)",
        r"api.?key.*(?:invalid|missing|expired)",
        r"token.?limit.*(?:reached|exceeded)",
        r"budget.*exceeded",
    ]
    s_lower = s.lower()
    for pattern in error_patterns:
        if re.search(pattern, s_lower):
            return True
    # Too short to be a real response (likely error)
    if len(s) < 5 and not re.match(r'^-?\d+\.?\d*$', s):
        return True
    return False


# [v11.1] JSON schema for the Mathematician blueprint, used for constrained
# decoding. The "reasoning" field comes FIRST and is required, so the enforcer
# lets the model think in free text before it must emit the structured fields —
# preserving its chain-of-thought scratchpad while still guaranteeing valid JSON.
_BLUEPRINT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning":       {"type": "string"},
        "unknown":         {"type": "string"},
        "givens":          {"type": "object", "additionalProperties": {"type": "number"}},
        "equations":       {"type": "array",  "items": {"type": "string"}},
        "expected_answer": {"type": "string"},
    },
    "required": ["reasoning", "givens", "equations", "expected_answer"],
}


def _after_blueprint_marker(text: str) -> str:
    """
    [v14.0] Return only what follows the last 'BLUEPRINT:' marker.

    In reasoning-first mode the Mathematician emits a plain-text derivation and
    then the JSON. _extract_blueprint_json scans from the first '{' to the last
    '}', so a derivation containing a brace (a set, an interval, LaTeX) would
    swallow the whole reply and fail to parse. Slicing at the marker keeps that
    from happening. No marker (or nothing after it) means the model ignored the
    format or was truncated mid-derivation: return the text unchanged so the
    normal extraction and the CoT->blueprint fallback still get their chance.
    """
    s = str(text)
    idx = s.rfind("BLUEPRINT:")
    if idx == -1:
        return s
    tail = s[idx + len("BLUEPRINT:"):]
    return tail if "{" in tail else s


# OPTIMIZED: Better blueprint extraction with structured fallback
def _extract_blueprint_json(text: str) -> dict:
    """
    Enhanced JSON extraction with better fallback handling.
    Ensures required keys exist even if parsing fails.
    """
    # [FIX v7.1] Check for error response first
    if _is_error_response(text):
        logger.warning(f"Mathematician returned error response: {str(text)[:200]}")
        return {
            "unknown": "the answer",
            "givens": {},
            "solution_steps": ["Error: LLM call failed"],
            "equations": [],
            "distractor_check": "",
            "metamorphic_tests": [],
            "notes": f"ERROR: {str(text)[:200]}"
        }
    
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    
    # Try direct parse
    try:
        bp = json.loads(text)
        if isinstance(bp, dict):
            bp.setdefault("unknown", "the answer")
            bp.setdefault("givens", {})
            bp.setdefault("solution_steps", [])
            bp.setdefault("equations", [])
            bp.setdefault("distractor_check", "")
            bp.setdefault("metamorphic_tests", [])
            bp.setdefault("notes", "")
            return bp
    except:
        pass
    
    # Try substring extraction
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            bp = json.loads(text[start:end+1])
            if isinstance(bp, dict):
                bp.setdefault("unknown", "the answer")
                bp.setdefault("givens", {})
                bp.setdefault("solution_steps", [])
                bp.setdefault("equations", [])
                bp.setdefault("distractor_check", "")
                bp.setdefault("metamorphic_tests", [])
                bp.setdefault("notes", "")
                return bp
        except:
            pass
    
    # Fallback: extract what we can
    givens = {}
    equations = []
    
    # Try to extract givens dict
    givens_match = re.search(r'"givens"\s*:\s*(\{[^}]+\})', text, re.DOTALL)
    if givens_match:
        try:
            givens = json.loads(givens_match.group(1))
        except:
            pass
    
    # Try to extract equations array
    eqs_match = re.search(r'"equations"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if eqs_match:
        try:
            equations = json.loads(f'[{eqs_match.group(1)}]')
        except:
            pass
    
    return {
        "unknown": "the answer",
        "givens": givens,
        "solution_steps": ["Solve step by step"],
        "equations": equations,
        "distractor_check": "",
        "metamorphic_tests": [],
        "notes": text[:800]
    }

def _extract_givens_dict_from_code(code_str: str) -> Optional[dict]:
    m = re.search(r"givens\s*=\s*(\{.*?\})\s*(?:\n|$)", code_str, re.DOTALL)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1))
    except Exception:
        return None

def _replace_givens_dict_in_code(code_str: str, new_givens: dict) -> str:
    m = re.search(r"(givens\s*=\s*)(\{.*?\})(\s*(?:\n|$))", code_str, re.DOTALL)
    if not m:
        return code_str
    prefix, _, suffix = m.group(1), m.group(2), m.group(3)
    return code_str[:m.start()] + prefix + repr(new_givens) + suffix + code_str[m.end():]

# --------------------------- Robust Code Extractor ---------------------------

def _extract_code_from_response(raw: str) -> Optional[str]:
    """
    Enhanced code extraction with multiple pattern matching strategies.
    """
    # [FIX v7.1] Don't try to extract code from error responses
    if _is_error_response(raw):
        return None
    
    s = str(raw)
    
    # Strategy 1: Standard markdown fences
    patterns = [
        r"```python\s+(.*?)```",
        r"```py\s+(.*?)```",
        r"```\s+(.*?)```",
        r"~~~python\s+(.*?)~~~",
        r"~~~\s+(.*?)~~~",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, s, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            code = re.sub(r"^python\s*\n", "", code, flags=re.IGNORECASE)
            return code
    
    # Strategy 2: Open fence (missing closing)
    open_patterns = [
        r"```(?:python|py)?\s+(.*?)$",
        r"~~~(?:python|py)?\s+(.*?)$",
    ]
    for pattern in open_patterns:
        match = re.search(pattern, s, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            code = re.split(r"\n\n(?:ANSWER|---|Note|Explanation)", code, maxsplit=1)[0]
            return code.strip()
    
    # Strategy 3: Code starts with "givens = " (no fence at all)
    givens_match = re.search(r"^(givens\s*=\s*\{.*)", s, re.DOTALL | re.MULTILINE)
    if givens_match:
        code = givens_match.group(1)
        code = re.split(r"\n\n(?:ANSWER|---)", code, maxsplit=1)[0]
        return code.strip()
    
    return None


def _extract_last_number(text: str) -> Optional[float]:
    """
    Extract the last numeric value from text, handling various formats.
    """
    # [FIX v7.1] Don't extract numbers from error responses
    if _is_error_response(text):
        return None
    
    text = str(text).strip()
    
    # Remove common non-numeric suffixes
    text = re.sub(r'\s*(dollars?|cents?|units?|items?|people|apples?|hours?|minutes?|days?|years?)\s*$', 
                  '', text, flags=re.IGNORECASE)
    
    # Find all numbers (including negatives, decimals, with commas)
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    
    if not numbers:
        return None
    
    last_num = numbers[-1].replace(',', '')
    try:
        return float(last_num)
    except:
        return None


def _format_blueprint_for_programmer(bp: dict) -> str:
    """
    Format blueprint in a clear, structured way for the Programmer to follow.
    """
    unknown = bp.get("unknown", "the answer")
    givens = bp.get("givens", {})
    solution_steps = bp.get("solution_steps", [])
    equations = bp.get("equations", [])
    distractor_check = bp.get("distractor_check", "")
    
    blueprint_text = f"""TARGET: {unknown}

GIVEN VALUES:
{json.dumps(givens, indent=2)}

SOLUTION STEPS:
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(solution_steps)])}

EQUATIONS TO IMPLEMENT (in order):
{chr(10).join([f"  {eq}" for eq in equations])}"""

    if distractor_check and distractor_check != "None":
        blueprint_text += f"\n\nDISTRACTORS TO IGNORE:\n{distractor_check}"
    
    return blueprint_text


# --------------------------- Code Executor ---------------------------

class PythonExecutor:
    @staticmethod
    def execute(code_str: str) -> Tuple[bool, str]:
        """Execute Python code safely with better error messages."""
        forbidden = [
            "import os", "import sys", "subprocess", "__import__",
            "eval(", "exec(", "compile(", "open(", "file(",
            "input(", "raw_input(", "rm -rf", "rmdir"
        ]
        
        code_lower = code_str.lower()
        for token in forbidden:
            if token in code_lower:
                return False, f"SecurityError: Forbidden token '{token}'"
        
        try:
            local_vars = {}
            import io
            from contextlib import redirect_stdout
            
            buf = io.StringIO()
            with redirect_stdout(buf):
                exec(code_str, {"__builtins__": __builtins__}, local_vars)
            
            output = buf.getvalue().strip()
            
            if not output:
                if 'answer' in local_vars:
                    return True, str(local_vars['answer'])
                elif 'result' in local_vars:
                    return True, str(local_vars['result'])
                else:
                    return False, "NoOutput: Code produced no output or answer variable"
            
            return True, output
            
        except NameError as e:
            return False, f"NameError: {str(e)}. Check variable definitions."
        except KeyError as e:
            return False, f"KeyError: {str(e)}. Check givens dict keys."
        except ZeroDivisionError:
            return False, "ZeroDivisionError: Cannot divide by zero"
        except Exception as e:
            return False, f"ExecutionError: {str(e)}"


# ==========================================================================
# [NEW v8.0] Symbolic Solver Fallback (SymPy)
# ==========================================================================

class SymbolicSolver:
    """
    When the Programmer's code execution fails, attempt to solve the
    blueprint equations symbolically using SymPy.
    
    This eliminates arithmetic errors entirely by delegating computation
    to a computer algebra system. Works best for problems expressible as
    algebraic equations (linear, polynomial, rate/proportion).
    
    Flow:
        1. Extract givens dict and equations from blueprint
        2. Substitute givens into equations
        3. Detect if there's an unknown variable to solve for
        4. Execute the equation chain symbolically
        5. Return numeric answer
    """

    @staticmethod
    def solve_from_blueprint(blueprint: dict) -> Tuple[bool, str, str]:
        """
        Attempt to solve a blueprint's equations using SymPy.
        
        Returns:
            (success: bool, answer: str, trace: str)
        """
        if not SYMPY_AVAILABLE:
            return False, "unknown", "SymPy not installed"
        
        givens = blueprint.get("givens", {})
        equations = blueprint.get("equations", [])
        
        if not equations:
            return False, "unknown", "No equations in blueprint"
        
        trace_lines = ["[SymPy Symbolic Solver]"]
        
        try:
            # Build a namespace with givens values
            namespace = {}
            givens_dict = {}
            
            for key, val in givens.items():
                if isinstance(val, (int, float)):
                    namespace[key] = val
                    givens_dict[key] = val
                    trace_lines.append(f"  Given: {key} = {val}")
            
            # Make givens accessible as dict too (for givens['key'] syntax)
            namespace['givens'] = givens_dict
            
            # Execute each equation in order using Python eval with restricted builtins
            safe_builtins = {
                "abs": abs, "round": round, "min": min, "max": max,
                "int": int, "float": float, "sum": sum, "len": len,
                "pow": pow, "divmod": divmod,
            }
            
            # Add math functions
            import math
            for fn_name in ['ceil', 'floor', 'sqrt', 'log', 'log10', 'exp', 'pi']:
                if hasattr(math, fn_name):
                    safe_builtins[fn_name] = getattr(math, fn_name)
            
            exec_globals = {"__builtins__": safe_builtins, "givens": givens_dict}
            exec_locals = dict(namespace)
            
            last_result = None
            for eq in equations:
                eq = eq.strip()
                if not eq or eq.startswith("#"):
                    continue
                
                trace_lines.append(f"  Exec: {eq}")
                
                try:
                    exec(eq, exec_globals, exec_locals)
                    # Track the last assigned variable
                    if "=" in eq and not eq.strip().startswith("if"):
                        var_name = eq.split("=")[0].strip()
                        if var_name in exec_locals:
                            last_result = exec_locals[var_name]
                except Exception as eq_err:
                    trace_lines.append(f"  ERROR in equation: {eq_err}")
                    # Try SymPy symbolic evaluation as last resort
                    sympy_result = SymbolicSolver._try_sympy_eval(eq, exec_locals, givens_dict)
                    if sympy_result is not None:
                        var_name = eq.split("=")[0].strip()
                        exec_locals[var_name] = sympy_result
                        last_result = sympy_result
                        trace_lines.append(f"  SymPy resolved: {var_name} = {sympy_result}")
                    else:
                        return False, "unknown", "\n".join(trace_lines)
            
            # Get final answer
            answer = exec_locals.get("answer", last_result)
            if answer is None:
                return False, "unknown", "\n".join(trace_lines) + "\n  No 'answer' variable found"
            
            # Convert to float
            try:
                answer_float = float(answer)
                trace_lines.append(f"  RESULT: {answer_float}")
                return True, str(answer_float), "\n".join(trace_lines)
            except (ValueError, TypeError):
                return False, "unknown", "\n".join(trace_lines) + f"\n  Non-numeric answer: {answer}"
            
        except Exception as e:
            trace_lines.append(f"  FATAL: {type(e).__name__}: {e}")
            return False, "unknown", "\n".join(trace_lines)

    @staticmethod
    def _try_sympy_eval(equation_str: str, local_vars: dict, givens: dict) -> Optional[float]:
        """
        Try to evaluate a single equation using SymPy when Python exec fails.
        Handles cases like division expressions, fractional arithmetic, etc.
        """
        if not SYMPY_AVAILABLE:
            return None
        
        try:
            # Extract RHS of assignment
            if "=" not in equation_str:
                return None
            
            parts = equation_str.split("=", 1)
            rhs = parts[1].strip()
            
            # Replace givens['key'] with actual values
            for key, val in givens.items():
                rhs = rhs.replace(f"givens['{key}']", str(val))
                rhs = rhs.replace(f'givens["{key}"]', str(val))
            
            # Replace known local variables
            for key, val in local_vars.items():
                if isinstance(val, (int, float)) and key != "givens":
                    # Only replace whole words
                    rhs = re.sub(rf'\b{re.escape(key)}\b', str(val), rhs)
            
            # Parse and evaluate with SymPy
            transformations = standard_transformations + (implicit_multiplication_application,)
            expr = parse_expr(rhs, transformations=transformations)
            result = float(expr.evalf())
            
            return result
        except Exception:
            return None


# ==========================================================================
# [FIX v7.1] Custom Exception for API Failures
# ==========================================================================

class LLMCallError(Exception):
    """Raised when all retries for an LLM API call are exhausted."""
    pass


# --------------------------- Unified LLM Client ---------------------------

class UnifiedLLMClient:
    # [NEW v10.2] HF Inference Providers Router — provider routing priority for
    # models that aren't pinned to a specific backend in HETEROGENEOUS_PRESETS.
    # Order matters: try the most reliable serverless host first.
    HF_ROUTER_PROVIDERS = ["together", "nebius", "fireworks-ai", "hf-inference"]

    def __init__(self, provider: str = "groq", use_cache: bool = False,
                 model_override: Optional[str] = None, load_4bit: bool = False):
        """
        [UPDATED v10.3] Supports five providers: groq, google, huggingface,
        together, local_hf. Backwards-compatible with v7.3+ (groq, google).
        load_4bit: [v10.3] local_hf only — loads model in 4-bit NF4 via BitsAndBytes.

        Provider semantics:
            groq:        OpenAI-compatible at api.groq.com, 12 RPM, 100K TPD.
            google:      google-generativeai SDK, 15 RPM.
            huggingface: HF Inference Providers Router (OpenAI-compatible),
                         10 RPM. Handles 503 cold-start with backoff.
            together:    OpenAI-compatible at api.together.xyz, 20 RPM.
            local_hf:    transformers.AutoModelForCausalLM in-process, no
                         rate limit, model cached on client instance.
        """
        self.provider = provider
        self.use_cache = use_cache
        self.model_name = "unknown"
        self.load_4bit = load_4bit  # [v10.3]
        self.load_4bit = load_4bit  # [v10.3] 4-bit NF4 quantization for local_hf 7B models
        # [v10.2] Pick the right limiter; local_hf gets a no-op limiter.
        self.limiter = {
            "groq":        groq_limiter,
            "google":      google_limiter,
            "huggingface": hf_limiter,
            "together":    together_limiter,
        }.get(provider, RateLimiter(requests_per_minute=600))  # local_hf → essentially unlimited

        # [v10.2] HF-specific state (cold-start retries, current routing provider)
        self._hf_route_idx = 0  # rotates through HF_ROUTER_PROVIDERS on 404
        # [v10.2] local_hf state (lazily populated; expensive)
        self._local_model = None
        self._local_tokenizer = None
        self._local_device = None

        if use_cache and os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "rb") as f:
                global CALL_CACHE
                try:
                    CALL_CACHE = pickle.load(f)
                except:
                    CALL_CACHE = {}

        if provider == "groq":
            if not GROQ_API_KEY: raise ValueError("Missing GROQ_API_KEY in .env file")
            self.client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
            # [v7.3] Use model_override if provided, else default
            self.model_name = model_override or "llama-3.3-70b-versatile"
        elif provider == "google":
            if not GOOGLE_API_KEY: raise ValueError("Missing GOOGLE_API_KEY in .env file")
            if not GOOGLE_AVAILABLE: raise ImportError("Google SDK missing. pip install google-generativeai")
            genai.configure(api_key=GOOGLE_API_KEY)
            if model_override:
                self.model_name = model_override
                self.client = genai.GenerativeModel(model_override)
            else:
                self._setup_google_model()
        elif provider == "huggingface":
            # [NEW v10.2] HF Inference Providers Router.
            if not HF_API_KEY:
                raise ValueError("Missing HF_API_KEY (or HUGGINGFACE_API_KEY) in .env file")
            if not REQUESTS_AVAILABLE:
                raise ImportError("'requests' library required for huggingface provider. pip install requests")
            # No persistent SDK client — we POST per-call. Just record the model.
            self.client = None
            self.model_name = model_override or "microsoft/Phi-4-mini-instruct"
        elif provider == "together":
            # [NEW v10.2] Together AI — OpenAI-compatible.
            if not TOGETHER_API_KEY:
                raise ValueError("Missing TOGETHER_API_KEY in .env file")
            self.client = OpenAI(base_url="https://api.together.xyz/v1", api_key=TOGETHER_API_KEY)
            self.model_name = model_override or "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        elif provider == "local_hf":
            # [NEW v10.2] Local transformers model. Lazy-load on first call.
            self.client = None  # Lazy init: see _ensure_local_model
            self.model_name = model_override or "Qwen/Qwen2.5-Math-1.5B-Instruct"
            logger.info(f"local_hf client created for {self.model_name} (lazy-load on first call)")
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def __repr__(self) -> str:
        return f"LLMClient({self.provider}/{self.model_name})"

    # [FIX v7.1] Validate API key at startup
    def validate_connection(self) -> bool:
        """Test that the API key works before running the full pipeline."""
        logger.info(f"Validating {self.provider} API connection...")
        try:
            test_response = self.call_model(
                [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
                temperature=0.0,
                max_tokens=50
            )
            if _is_error_response(test_response):
                logger.error(f"API validation FAILED. Response: {test_response}")
                return False
            logger.info(f"API validation OK. Test response: {str(test_response)[:100]}")
            return True
        except Exception as e:
            logger.error(f"API validation FAILED with exception: {e}")
            return False

    def call_model(self, messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 1200,
                   json_schema: Optional[dict] = None) -> Any:
        # [v11.1] json_schema: if set and provider is local_hf, force the output to
        # match this JSON schema via constrained decoding (token-level grammar).
        # Other providers ignore it (API models follow JSON prompts on their own).
        key = _make_cache_key(self.provider, self.model_name, messages, temperature)

        if self.use_cache and key in CALL_CACHE:
            return CALL_CACHE[key]

        # [FIX v7.2] Check token budget before calling
        estimated = token_budget.estimate_tokens(messages, max_tokens)
        if not token_budget.can_afford(estimated):
            return "ERROR_BUDGET_EXCEEDED: Daily token limit reached. Wait 24h or upgrade to Dev tier."

        last_err = None

        def _call_once():
            self.limiter.wait()
            if self.provider == "groq":
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                # [FIX v7.2] Track actual token usage from response
                if hasattr(resp, 'usage') and resp.usage:
                    actual_tokens = getattr(resp.usage, 'total_tokens', estimated)
                    token_budget.record_usage(actual_tokens)
                else:
                    token_budget.record_usage(estimated)
                return resp.choices[0].message.content
            if self.provider == "google":
                sys_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
                user_prompt = "\n\n".join([m["content"] for m in messages if m["role"] != "system"])
                full_prompt = f"System:\n{sys_prompt}\n\nTask:\n{user_prompt}"
                resp = self.client.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )
                )
                token_budget.record_usage(estimated)
                return getattr(resp, "text", "")
            # [NEW v10.2] HuggingFace Inference Providers Router
            if self.provider == "huggingface":
                return self._call_huggingface(messages, temperature, max_tokens)
            # [NEW v10.2] Together AI (OpenAI-compatible)
            if self.provider == "together":
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return resp.choices[0].message.content
            # [NEW v10.2] Local transformers model
            if self.provider == "local_hf":
                return self._call_local_hf(messages, temperature, max_tokens, json_schema=json_schema)

        for attempt in range(4):
            try:
                res = _call_once()

                if res is None or str(res).strip() == "":
                    last_err = "Empty response from API"
                    logger.warning(f"Attempt {attempt+1}/4: Empty response. Retrying...")
                    time.sleep(min(12.0, 1.5 * (attempt + 1)))
                    continue
                
                if self.use_cache:
                    CALL_CACHE[key] = res
                    with open(CACHE_FILE, "wb") as f:
                        pickle.dump(CALL_CACHE, f)
                return res
            except Exception as e:
                last_err = str(e)
                err_str = str(e).lower()
                logger.warning(f"Attempt {attempt+1}/4 failed: {type(e).__name__}: {str(e)[:200]}")
                
                # [FIX v7.1] Auth errors — no point retrying
                if "401" in err_str or "unauthorized" in err_str or "authentication" in err_str:
                    logger.error("AUTHENTICATION ERROR: API key is invalid or expired.")
                    return f"ERROR_AUTH_401: {last_err}"
                
                if "403" in err_str or "forbidden" in err_str:
                    logger.error("FORBIDDEN: API key does not have access to this model.")
                    return f"ERROR_AUTH_403: {last_err}"
                
                # [FIX v7.2] 429 Rate Limit — extract wait time from error message
                if "429" in err_str or "rate_limit" in err_str or "rate limit" in err_str:
                    # Try to extract wait time from Groq error (e.g., "try again in 8m27.168s")
                    wait_match = re.search(r'try again in (\d+)m([\d.]+)s', str(e))
                    if wait_match:
                        wait_minutes = int(wait_match.group(1))
                        wait_seconds = float(wait_match.group(2))
                        total_wait = wait_minutes * 60 + wait_seconds + 5  # +5s buffer
                        
                        if total_wait > 600:  # More than 10 minutes = daily limit hit
                            logger.error(
                                f"DAILY TOKEN LIMIT REACHED. Groq says wait {wait_minutes}m{wait_seconds:.0f}s. "
                                f"This usually means you've hit the 100K tokens/day free tier limit. "
                                f"Options: (1) Wait until tomorrow, (2) Upgrade to Dev tier at console.groq.com"
                            )
                            return f"ERROR_RATE_LIMIT_DAILY: {last_err}"
                        
                        logger.info(f"Rate limited. Waiting {total_wait:.0f}s as requested by Groq...")
                        time.sleep(total_wait)
                        continue
                    else:
                        # Generic 429 — exponential backoff with jitter
                        backoff = min(120, (2 ** attempt) * 5 + random.uniform(0, 5))
                        logger.info(f"Rate limited (429). Backing off {backoff:.0f}s...")
                        time.sleep(backoff)
                        continue
                
                # Other errors — standard backoff
                time.sleep(min(12.0, 1.5 * (attempt + 1)))

        return f"ERROR_GENERATION: {last_err or 'unknown_error'}"

    def _setup_google_model(self):
        try:
            available = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
            target = next((m for m in available if "flash" in m), available[0] if available else "models/gemini-1.5-flash")
            self.model_name = target
            self.client = genai.GenerativeModel(target)
        except Exception:
            self.model_name = "gemini-1.5-flash"
            self.client = genai.GenerativeModel(self.model_name)

    # =====================================================================
    # [NEW v10.2] HuggingFace Inference Providers Router
    # =====================================================================
    def _call_huggingface(self, messages: List[Dict[str, str]],
                          temperature: float, max_tokens: int) -> str:
        """
        Call the HF Inference Providers Router (OpenAI-compatible chat
        completions). Handles 503 cold-start with exponential backoff up to
        ~60s total. On 404 (model not on current routing provider), rotates
        through HF_ROUTER_PROVIDERS and retries once.

        Returns the assistant message content as a string. On unrecoverable
        failure raises an exception so the outer retry loop in call_model
        sees it.
        """
        import requests as _r
        # The router accepts both legacy 'hf-inference' (serverless) and any
        # of the third-party providers (together/nebius/fireworks). We start
        # at the index we last successfully used, fall through to others on
        # 404. This is per-instance, not global, to avoid cross-talk.
        backoff = 2.0
        total_waited = 0.0
        max_cold_start_wait = 60.0

        for attempt in range(6):
            route = self.HF_ROUTER_PROVIDERS[self._hf_route_idx % len(self.HF_ROUTER_PROVIDERS)]
            url = f"https://router.huggingface.co/{route}/v1/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                # Some HF-routed providers ignore unknown fields; OpenAI-compatible
                # ones accept these. Keep payload minimal.
            }
            headers = {
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json",
            }
            try:
                resp = _r.post(url, json=payload, headers=headers, timeout=120)
            except Exception as e:
                # Network-level failure. Treat as transient.
                logger.warning(f"HF route '{route}' network error: {e}. Retrying...")
                time.sleep(min(8.0, backoff))
                backoff *= 1.5
                continue

            if resp.status_code == 200:
                try:
                    j = resp.json()
                except Exception:
                    raise RuntimeError(f"HF returned 200 but body not JSON: {resp.text[:300]}")
                # OpenAI-compatible shape
                try:
                    return j["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    # Some legacy hf-inference responses return [{"generated_text": "..."}]
                    if isinstance(j, list) and j and isinstance(j[0], dict) and "generated_text" in j[0]:
                        return j[0]["generated_text"]
                    raise RuntimeError(f"HF unknown response shape: {str(j)[:300]}")

            # 503 — model loading. The router puts estimated_time in the body.
            if resp.status_code == 503:
                wait = 5.0
                try:
                    body = resp.json()
                    wait = float(body.get("estimated_time", 5.0))
                except Exception:
                    pass
                wait = min(wait, max_cold_start_wait - total_waited)
                if wait <= 0:
                    raise RuntimeError(f"HF cold-start exceeded {max_cold_start_wait}s budget for {self.model_name}")
                logger.info(f"HF cold-start ({route}): waiting {wait:.1f}s for {self.model_name}")
                time.sleep(wait)
                total_waited += wait
                continue

            # 404 — model not served by this routing provider; rotate.
            if resp.status_code == 404:
                logger.warning(f"HF model {self.model_name} not on '{route}' (404). Rotating provider.")
                self._hf_route_idx += 1
                if self._hf_route_idx >= len(self.HF_ROUTER_PROVIDERS) * 2:
                    raise RuntimeError(f"HF: no routing provider serves {self.model_name}")
                continue

            # 429 — let outer retry handle (it has the right backoff path).
            if resp.status_code == 429:
                raise RuntimeError(f"HF 429 rate_limit: {resp.text[:300]}")

            # 401/403 — auth issue, no point retrying.
            if resp.status_code in (401, 403):
                raise RuntimeError(f"HF 401/403 auth: {resp.text[:300]}")

            # Anything else — treat as transient with bounded backoff.
            logger.warning(f"HF route '{route}' status {resp.status_code}: {resp.text[:200]}")
            time.sleep(min(8.0, backoff))
            backoff *= 1.5

        raise RuntimeError(f"HF: exhausted attempts for {self.model_name}")

    # =====================================================================
    # [NEW v10.2] Local transformers (in-process) inference
    # =====================================================================
    def _ensure_local_model(self):
        """Load the local model + tokenizer on first use. Cached on instance."""
        global TRANSFORMERS_AVAILABLE
        if self._local_model is not None:
            return
        if TRANSFORMERS_AVAILABLE is False:
            raise ImportError("transformers/torch not installed. pip install transformers accelerate torch")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            TRANSFORMERS_AVAILABLE = True
        except ImportError as e:
            TRANSFORMERS_AVAILABLE = False
            raise ImportError(f"local_hf provider needs transformers+torch: {e}")

        use_cuda = torch.cuda.is_available()
        # [v10.7] bfloat16 on GPU, NOT float16. Qwen2.5(-Math) overflows fp16:
        # attention logits exceed fp16's 65504 max → Inf → NaN → the model emits
        # token 0 ('!') forever ('![](!!!!!!' garbage, exactly what pre-flight saw).
        # bf16 shares fp32's exponent range, so it cannot overflow. T4 (sm_75) runs
        # bf16 in PyTorch correctly (no tensor-core accel, but numerically fine).
        # Forced rather than gated on is_bf16_supported(), which returns a false
        # negative on some setups and would silently revert to broken fp16.
        dtype = torch.bfloat16 if use_cuda else torch.float32
        quant_tag = " [4-bit NF4]" if self.load_4bit else ""
        logger.info(
            f"local_hf: loading {self.model_name} "
            f"({'cuda' if use_cuda else 'cpu'}, dtype={dtype}){quant_tag}..."
        )

        tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token

        # [v10.3] KEY FIX for Kaggle cudaErrorNoKernelImageForDevice
        # Root cause: accelerate's device_map dispatch hooks invoke SM-specific CUDA
        # kernels during forward() that are not compiled for T4 (SM7.5) or P100 (SM6.0).
        # Fix: load to CPU, then .to("cuda") manually — uses only universal cublas.
        # attn_implementation="eager" additionally avoids FlashAttention2 (SM>=8.0 only).
        _lkw = dict(trust_remote_code=True, attn_implementation="eager")

        # [v10.3 FIX] Wrap the entire model-load block in a try/except so that
        # any failure (OOM, CUDA error, bnb error) cleans up partial VRAM
        # allocations before the exception propagates to call_model's retry loop.
        # Without this, each failed from_pretrained leaves weight shards in VRAM;
        # four retries × partial-7B = VRAM full of unreachable fragments → OOM spiral.
        import gc as _gc
        mdl = None
        # [v10.8] Reclaim VRAM from any prior/partial load before allocating.
        # Stale weights from an earlier cell run (or a failed retry) leave the
        # GPU near-full, so a fresh 14 GB load OOMs at ~86% then loops forever
        # (load fails -> _local_model stays None -> every next call reloads).
        if use_cuda:
            _gc.collect()
            torch.cuda.empty_cache()
        try:
            if self.load_4bit and use_cuda:
                # 4-bit (BitsAndBytes) requires device_map="auto" — no workaround exists.
                # attn_implementation="eager" still prevents the FA2 kernel mismatch.
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_cfg = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=dtype,   # [v10.7] bf16 — avoid fp16 overflow
                        bnb_4bit_use_double_quant=True,
                    )
                    # [v13.0] On multi-GPU boxes, hand accelerate an explicit
                    # free-memory-aware max_memory map. Observed 2026-07-12
                    # (T4 x2, mixed preset): with Math-7B already resident,
                    # "auto" placed the ENTIRE second model onto one GPU and
                    # its load transient filled all 14.5 GB → OOM on every
                    # retry, so the Mathematician never loaded. Capping each
                    # GPU at ~85% of its CURRENTLY-free memory forces the
                    # placement (and its materialization) to shard across
                    # devices that actually have room.
                    _load_kwargs = dict(
                        quantization_config=bnb_cfg,
                        device_map="auto", low_cpu_mem_usage=True, **_lkw,
                    )
                    if torch.cuda.device_count() > 1:
                        _mm = {}
                        for _gi in range(torch.cuda.device_count()):
                            _free_b, _ = torch.cuda.mem_get_info(_gi)
                            _mm[_gi] = int(_free_b * 0.85)
                        _mm["cpu"] = "24GiB"
                        _load_kwargs["max_memory"] = _mm
                        logger.info(f"local_hf: multi-GPU max_memory map = "
                                    f"{ {k: (v if isinstance(v, str) else f'{v/1e9:.1f}GB') for k, v in _mm.items()} }")
                    mdl = AutoModelForCausalLM.from_pretrained(
                        self.model_name, **_load_kwargs,
                    )
                    logger.info(f"local_hf: {self.model_name} loaded in 4-bit NF4 on cuda")
                except ImportError:
                    logger.warning("bitsandbytes not installed — falling back to fp16.")
                    mdl = AutoModelForCausalLM.from_pretrained(
                        self.model_name, torch_dtype=dtype, **_lkw,
                    ).to("cuda")
            else:
                # Standard fp16 path.
                # [v10.6] device_map="auto" shards weights across ALL visible GPUs —
                # required for 7B fp16 on Kaggle T4 x2 (14.2 GB over 2×15 GB cards).
                # Same dispatch machinery as the 4-bit path (proven on this env),
                # eager attention already prevents the FA2 kernel mismatch.
                if use_cuda:
                    n_gpu = torch.cuda.device_count()
                    max_mem = {
                        g: f"{max(1, int(torch.cuda.get_device_properties(g).total_memory / 2**30) - 2)}GiB"
                        for g in range(n_gpu)
                    }  # ~2 GiB headroom per GPU for KV cache + activations
                    mdl = AutoModelForCausalLM.from_pretrained(
                        self.model_name, torch_dtype=dtype,
                        device_map="auto", max_memory=max_mem,
                        low_cpu_mem_usage=True, **_lkw,
                    )
                    logger.info(f"local_hf: {str(dtype).split('.')[-1]} sharded over {n_gpu} GPU(s) {max_mem}")
                    devs = set(map(str, getattr(mdl, "hf_device_map", {}).values()))
                    if any(("cpu" in d) or ("disk" in d) for d in devs):
                        logger.warning(
                            f"local_hf: some layers offloaded to CPU/disk ({devs}) — "
                            "generation will be VERY slow. Switch Kaggle accelerator "
                            "to 'GPU T4 x2' for enough VRAM."
                        )
                else:
                    mdl = AutoModelForCausalLM.from_pretrained(
                        self.model_name, torch_dtype=dtype, **_lkw,
                    )

            mdl.eval()
            self._local_tokenizer = tok
            self._local_model     = mdl
            # Resolve actual device from params (correct for both plain and 4-bit).
            self._local_device    = next(mdl.parameters()).device
            logger.info(f"local_hf: {self.model_name} ready on {self._local_device}")

        except Exception as _load_err:
            # VRAM cleanup — prevent fragment accumulation across retries.
            logger.warning(
                f"local_hf: load FAILED ({type(_load_err).__name__}: "
                f"{str(_load_err)[:120]}). Purging VRAM before retry."
            )
            try:
                del mdl
            except Exception:
                pass
            mdl = None
            self._local_model = None
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            raise  # propagate to call_model's retry loop

    def _call_local_hf(self, messages: List[Dict[str, str]],
                       temperature: float, max_tokens: int,
                       json_schema: Optional[dict] = None) -> str:
        """Run a local HF causal LM. Uses the tokenizer's chat template if
        available (most modern instruct models ship one).

        [v11.1] If json_schema is given, applies token-level constrained decoding
        (lm-format-enforcer) so the output is GUARANTEED to be valid JSON matching
        the schema — the only reliable way to get structured output from a CoT-only
        model like Qwen2.5-Math, which ignores every prompt-level JSON instruction."""
        import torch
        self._ensure_local_model()
        tok = self._local_tokenizer
        mdl = self._local_model

        # Build the prompt. Prefer apply_chat_template for instruct models;
        # fall back to a plain concatenation if the tokenizer lacks one.
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            try:
                prompt = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt = self._fallback_prompt(messages)
        else:
            prompt = self._fallback_prompt(messages)

        # Trim if the prompt exceeds the model's context window.
        # We don't have a portable max_position_embeddings accessor across all
        # configs, so we use a conservative default of 4096 minus output budget.
        ctx = getattr(mdl.config, "max_position_embeddings", 4096) or 4096
        max_input = max(256, ctx - max_tokens - 64)
        # [v10.6] Truncate from the LEFT: losing prompt head degrades quality,
        # but right-truncation cuts the chat-template assistant tag and breaks
        # generation entirely (model continues the prompt instead of answering).
        tok.truncation_side = "left"
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=max_input)
        if ids["input_ids"].shape[1] >= max_input:
            logger.warning(
                f"local_hf: prompt hit context limit — truncated to {max_input} tokens "
                f"(ctx={ctx}, {max_tokens} reserved for output). Quality may degrade."
            )
        actual_device = next(mdl.parameters()).device
        ids = {k: v.to(actual_device) for k, v in ids.items()}

        # [v10.3] Greedy decoding only — avoids fp16 probability tensor
        # overflow (inf/nan) that causes CUDA device-side assert on sampling.
        # Greedy is standard for math problem solving (want most-likely token).
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=False,                         # greedy — no temperature division
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        # [v11.1] Token-level JSON-schema enforcement. The model literally cannot
        # emit a token that would break the schema, so prose-only models are forced
        # to produce a valid blueprint. Degrades gracefully to unconstrained
        # generation if the library is missing or errors.
        if json_schema is not None:
            try:
                from lmformatenforcer import JsonSchemaParser
                from lmformatenforcer.integrations.transformers import (
                    build_transformers_prefix_allowed_tokens_fn,
                )
                _parser = JsonSchemaParser(json_schema)
                gen_kwargs["prefix_allowed_tokens_fn"] = (
                    build_transformers_prefix_allowed_tokens_fn(tok, _parser)
                )
                logger.info("local_hf: JSON-schema constrained decoding ENABLED")
            except Exception as _enf_err:
                logger.warning(
                    f"local_hf: constrained decoding unavailable ({type(_enf_err).__name__}: "
                    f"{str(_enf_err)[:80]}) — generating unconstrained"
                )
        with torch.no_grad():
            try:
                out = mdl.generate(**ids, **gen_kwargs)
            except Exception as cuda_err:
                # [v10.3] CPU fallback for CUDA failures (kernel mismatch or
                # corrupted context after device-side assert).
                # After a CUDA assert the context is poisoned — do NOT try to
                # move model back to GPU, just stay on CPU for this session.
                if actual_device.type != "cpu":
                    logger.warning(
                        f"CUDA generate failed ({type(cuda_err).__name__}: "
                        f"{str(cuda_err)[:80]}). Switching to CPU permanently."
                    )
                    ids_cpu = {k: v.to("cpu") for k, v in ids.items()}
                    try:
                        mdl_cpu = mdl.to("cpu")
                    except Exception:
                        # CUDA context corrupted — reload model on CPU
                        from transformers import AutoModelForCausalLM as _AMCL
                        logger.warning("Reloading model on CPU (CUDA context corrupted).")
                        mdl_cpu = _AMCL.from_pretrained(
                            self.model_name, torch_dtype=torch.float32,
                            trust_remote_code=True, attn_implementation="eager",
                        )
                    self._local_model  = mdl_cpu
                    self._local_device = torch.device("cpu")
                    out = mdl_cpu.generate(**ids_cpu, **gen_kwargs)
                else:
                    raise
        # Decode only the newly-generated tokens.
        new_tokens = out[0, ids["input_ids"].shape[1]:]
        text = tok.decode(new_tokens, skip_special_tokens=True)
        return text

    @staticmethod
    def _fallback_prompt(messages: List[Dict[str, str]]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "user").upper()
            parts.append(f"{role}: {m.get('content', '')}")
        parts.append("ASSISTANT:")
        return "\n\n".join(parts)

# --------------------------- Dataset Manager ---------------------------

class EnhancedProblemManager:
    def __init__(self, random_seed: Optional[int] = None):
        random.seed(random_seed)

    def _maybe_harden(self, problem: str, hardener: Optional[str]) -> str:
        """
        [v10.4] Hardening modes:
          None / "" / "off"     → no perturbation.
          "distractor"          → SILENT distractors (default since v10.4). Adds
                                  1–3 unrelated factual sentences embedded in the
                                  problem text without any "ignore this" markers.
                                  Tests whether the solver actually reasons about
                                  relevance vs. surface-cueing on the word "ignore".
          "distractor_labeled"  → legacy v10.3 behaviour: distractors are tagged
                                  ("Unrelated note: …", "Extra context (ignore): …").
                                  Kept ONLY for reproducing the v10.3 paper run; not
                                  recommended for new experiments.
        Implementation note: silent distractors are syntactically valid English
        sentences that introduce names/quantities the solver could plausibly mistake
        for problem data, without any meta-marker betraying their status.
        """
        if not hardener or hardener == "off":
            return problem

        if hardener == "distractor":
            # ── SILENT distractors (default) ───────────────────────────────
            # Each template introduces a number that LOOKS like it could matter
            # but does not appear in the solution path. The framing is neutral
            # narrative — no "ignore", "unrelated", "irrelevant", "extra context".
            names = ["Alex", "Maria", "Nikos", "Elena", "Chris", "Sofia",
                     "Jordan", "Priya", "Aiko", "Mateo"]
            items = ["stickers", "marbles", "notebooks", "coins", "candies",
                     "tickets", "stamps", "buttons", "ribbons", "magnets"]
            locations = ["the next town", "a nearby school", "the warehouse",
                         "the depot", "another district"]
            n1 = random.randint(7, 99)
            n2 = random.randint(10, 250)
            n3 = random.randint(2, 60)
            n4 = random.randint(3, 80)
            who1 = random.choice(names)
            who2 = random.choice([n for n in names if n != who1])
            it1 = random.choice(items)
            it2 = random.choice([i for i in items if i != it1])
            loc = random.choice(locations)

            silent_distractors = [
                f"{who1} had collected {n1} {it1} the previous summer.",
                f"In {loc}, a similar store reported sales of {n2} units last month.",
                f"The temperature that day was {n3} degrees.",
                f"{who2} had {n4} {it2} stored in the attic.",
            ]
            random.shuffle(silent_distractors)
            k = random.choice([1, 2, 3])
            return problem.strip() + " " + " ".join(silent_distractors[:k])

        if hardener == "distractor_labeled":
            # ── Legacy labeled distractors (v10.3 behaviour) ───────────────
            names = ["Alex", "Maria", "Nikos", "Elena", "Chris", "Sofia"]
            items = ["stickers", "marbles", "notebooks", "coins", "candies", "tickets"]
            n1 = random.randint(7, 99)
            n2 = random.randint(10, 250)
            n3 = random.randint(2, 60)
            who = random.choice(names)
            it = random.choice(items)

            distractors = [
                f"Unrelated note: {who} counted {n1} {it} yesterday, but that does not affect the question.",
                f"Extra context (ignore): A different store sold {n2} items in total last week.",
                f"Reminder: The number {n3} appears in a separate example and is irrelevant here.",
            ]
            k = random.choice([1, 2, 3])
            return problem.strip() + "\n\n" + "\n".join(distractors[:k])

        return problem

    def load_random_problems(self, datasets_list: List[str], num_problems: int, hardener: Optional[str] = None) -> List[Dict[str, str]]:
        pool: List[Dict[str, str]] = []

        if not DATASETS_AVAILABLE:
            curated = [
                {"id": "c1", "puzzle": "If 2x + 3 = 15, what is x?", "answer": "6", "dataset": "curated"},
                {"id": "c2", "puzzle": "Jane has 5 apples. She eats 2. How many left?", "answer": "3", "dataset": "curated"},
                {"id": "c3", "puzzle": "Calculate 15% of 200.", "answer": "30", "dataset": "curated"},
                {"id": "c4", "puzzle": "Solve for x: x^2 - 4 = 0 (positive root)", "answer": "2", "dataset": "curated"},
                {"id": "c5", "puzzle": "A train travels 60 mph for 2 hours. Distance?", "answer": "120", "dataset": "curated"},
            ]
            for c in curated[:num_problems]:
                c["puzzle"] = self._maybe_harden(c["puzzle"], hardener)
                pool.append(c)
            return pool[:num_problems]

        if not datasets_list:
            datasets_list = ["gsm8k_test"]
        per_ds = max(1, (num_problems + len(datasets_list) - 1) // len(datasets_list))

        for ds_name in datasets_list:
            ds_name_norm = ds_name.strip().lower()

            try:
                if ds_name_norm in ["gsm8k", "gsm8k_test", "gsm8k-train", "gsm8k_train"]:
                    split = "test" if ds_name_norm in ["gsm8k", "gsm8k_test"] else "train"
                    ds = load_dataset("openai/gsm8k", "main", split=split)
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = ds[i].get("question", "")
                        raw_a = ds[i].get("answer", "")
                        # GSM8K stores full solution; extract numeric answer after "####"
                        a = raw_a.split("####")[-1].strip().replace(",", "") if "####" in raw_a else raw_a
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": f"gsm8k_{split}",
                            "id": f"gsm8k_{split}_{i}",
                        }
                        pool.append(item)

                elif ds_name_norm in ["gsm-hard", "gsm_hard"]:
                    ds = load_dataset("reasoning-machines/gsm-hard", split="train")
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = ds[i].get("input", "")
                        a = ds[i].get("target", "")
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": "gsm-hard",
                            "id": f"gsm-hard_{i}",
                        }
                        pool.append(item)

                elif ds_name_norm in ["gsm-plus", "gsm_plus", "gsmplus"]:
                    ds = load_dataset("qintongli/GSM-Plus", split="test")
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = ds[i].get("question", "")
                        a = ds[i].get("answer", "")
                        pt = ds[i].get("perturbation_type", "")
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": f"gsm-plus:{pt}" if pt else "gsm-plus",
                            "id": f"gsm-plus_{i}",
                        }
                        pool.append(item)

                elif ds_name_norm in ["gsm-symbolic", "gsm_symbolic", "gsm-symbolic-main", "gsm_symbolic_main"]:
                    ds = load_dataset("apple/GSM-Symbolic", name="main", split="test")
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = ds[i].get("question", "")
                        a = ds[i].get("answer", "")
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": "gsm-symbolic:main",
                            "id": f"gsm-symbolic_main_{i}",
                        }
                        pool.append(item)

                elif ds_name_norm in ["gsm-symbolic-p1", "gsm_symbolic_p1"]:
                    ds = load_dataset("apple/GSM-Symbolic", name="p1", split="test")
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = ds[i].get("question", "")
                        a = ds[i].get("answer", "")
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": "gsm-symbolic:p1",
                            "id": f"gsm-symbolic_p1_{i}",
                        }
                        pool.append(item)

                elif ds_name_norm in ["gsm-symbolic-p2", "gsm_symbolic_p2"]:
                    ds = load_dataset("apple/GSM-Symbolic", name="p2", split="test")
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = ds[i].get("question", "")
                        a = ds[i].get("answer", "")
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": "gsm-symbolic:p2",
                            "id": f"gsm-symbolic_p2_{i}",
                        }
                        pool.append(item)

                elif ds_name_norm in ["svamp"]:
                    ds = load_dataset("ChilleD/SVAMP", split="test")
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = ds[i].get("question_concat", None)
                        if not q:
                            body = ds[i].get("Body", "")
                            ques = ds[i].get("Question", ds[i].get("question", ""))
                            q = (str(body).strip() + "\n" + str(ques).strip()).strip()
                        a = ds[i].get("Answer", ds[i].get("answer", ""))
                        item = {
                            "puzzle": self._maybe_harden(str(q), hardener),
                            "answer": a,
                            "dataset": "svamp:test",
                            "id": f"svamp_test_{i}",
                        }
                        pool.append(item)

                # ----------------------------------------------------------------
                # [NEW 2026] MATH-500 / Hendrycks competition_math
                # Keys: "math", "math500", "math-500", "hendrycks_math"
                # HuggingFace: lighteval/MATH-Hard  (500 hardest problems)
                #              hendrycks/competition_math (full ~12 500)
                # Fields: problem → puzzle, solution → answer (last \boxed{})
                # ----------------------------------------------------------------
                elif ds_name_norm in ["math", "math500", "math-500", "hendrycks_math",
                                      "math_hard", "math-hard"]:
                    use_hard_subset = ds_name_norm in ["math500", "math-500",
                                                       "math_hard", "math-hard"]
                    if use_hard_subset:
                        ds = load_dataset("lighteval/MATH-Hard", split="test")
                    else:
                        ds = load_dataset("hendrycks/competition_math", split="test",
                                          trust_remote_code=True)
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = ds[i].get("problem", ds[i].get("question", ""))
                        raw_sol = ds[i].get("solution", ds[i].get("answer", ""))
                        # Extract the final boxed answer for numeric comparison
                        # Extract last \boxed{...} content, handling nested braces
                        # e.g. \boxed{\frac{1}{2}} → "\frac{1}{2}"
                        def _extract_last_boxed(s: str) -> str:
                            tag = r"\boxed{"
                            pos = s.rfind(tag)
                            if pos == -1:
                                return s.strip()
                            start = pos + len(tag)
                            depth, i = 1, start
                            while i < len(s) and depth:
                                if s[i] == "{":
                                    depth += 1
                                elif s[i] == "}":
                                    depth -= 1
                                i += 1
                            return s[start:i - 1].strip()
                        a = _extract_last_boxed(raw_sol)
                        level = ds[i].get("level", "")
                        subject = ds[i].get("type", ds[i].get("subject", ""))
                        tag = f"math-hard" if use_hard_subset else "math"
                        if subject:
                            tag += f":{subject}"
                        if level:
                            tag += f":{level}"
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": tag,
                            "id": f"{tag.replace(':', '_')}_{i}",
                        }
                        pool.append(item)

                # ----------------------------------------------------------------
                # [NEW 2026] AIME problems (2024 + historical)
                # Keys: "aime", "aime2024", "aime_2024"
                # HuggingFace: AI-MO/aimo-validation-aime  (AIME I & II 2024)
                #              Maxwell-Jia/AIME_1983_2024   (historical, all years)
                # Fields vary by source; we normalise to puzzle/answer.
                # Answers are integers 0-999.
                # ----------------------------------------------------------------
                elif ds_name_norm in ["aime", "aime2024", "aime_2024",
                                      "aime_historical", "aime-historical"]:
                    use_historical = ds_name_norm in ["aime_historical", "aime-historical"]
                    if use_historical:
                        ds = load_dataset("Maxwell-Jia/AIME_1983_2024", split="train",
                                          trust_remote_code=True)
                        q_key, a_key = "Problem", "Answer"
                    else:
                        ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
                        q_key, a_key = "problem", "answer"
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = str(ds[i].get(q_key, ds[i].get("problem", ""))).strip()
                        a = str(ds[i].get(a_key, ds[i].get("answer", ""))).strip()
                        year = ds[i].get("year", ds[i].get("Year", ""))
                        tag = "aime-historical" if use_historical else "aime2024"
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": f"{tag}:{year}" if year else tag,
                            "id": f"{tag}_{i}",
                        }
                        pool.append(item)

                # ----------------------------------------------------------------
                # [NEW 2026] OlympiadBench
                # Keys: "olympiadbench", "olympiad_bench", "olympiad-bench"
                # HuggingFace: GAIR/OlympiadBench
                # Subsets: OE_TO_maths_en_COMP (open-ended English math competition)
                # Fields: problem, final_answer (list → join)
                # ----------------------------------------------------------------
                elif ds_name_norm in ["olympiadbench", "olympiad_bench",
                                      "olympiad-bench", "olympiad"]:
                    ds = load_dataset(
                        "GAIR/OlympiadBench",
                        name="OE_TO_maths_en_COMP",
                        split="test",
                        trust_remote_code=True,
                    )
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = str(ds[i].get("problem", "")).strip()
                        raw_ans = ds[i].get("final_answer", ds[i].get("answer", ""))
                        if isinstance(raw_ans, list):
                            a = ", ".join(str(x) for x in raw_ans)
                        else:
                            a = str(raw_ans).strip()
                        subject = ds[i].get("subject", "")
                        tag = f"olympiadbench:{subject}" if subject else "olympiadbench"
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": tag,
                            "id": f"olympiadbench_{i}",
                        }
                        pool.append(item)

                # ----------------------------------------------------------------
                # [NEW 2026] MGSM — Multilingual Grade-School Math
                # Keys: "mgsm", "mgsm_en", "mgsm_de", "mgsm_es", "mgsm_fr",
                #       "mgsm_ja", "mgsm_zh", "mgsm_th", "mgsm_sw",
                #       "mgsm_bn", "mgsm_ru", "mgsm_te"
                # HuggingFace: juletxara/mgsm  (split = language code)
                # Fields: question, answer (numeric int)
                # ----------------------------------------------------------------
                elif ds_name_norm.startswith("mgsm"):
                    MGSM_LANGS = {"en", "de", "es", "fr", "ja", "zh",
                                  "th", "sw", "bn", "ru", "te"}
                    parts = ds_name_norm.split("_", 1)
                    lang = parts[1] if len(parts) > 1 and parts[1] in MGSM_LANGS else "en"
                    ds = load_dataset("juletxara/mgsm", lang, split="test")
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = str(ds[i].get("question", "")).strip()
                        a = str(ds[i].get("answer", "")).strip()
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": f"mgsm:{lang}",
                            "id": f"mgsm_{lang}_{i}",
                        }
                        pool.append(item)

                # ----------------------------------------------------------------
                # [2025] AIME 2025 / 2026 — MathArena (uncontaminated)
                # Keys: "aime_2025", "aime2025", "aime_2026", "aime2026"
                # HuggingFace: MathArena/aime_2025  |  MathArena/aime_2026
                # Fields: problem (str), answer (int)  — integers 0-999
                # Difficulty: ~AIME level, frontier models 70-97 %
                # ----------------------------------------------------------------
                elif ds_name_norm in ["aime_2025", "aime2025",
                                      "aime_2026", "aime2026",
                                      "aime_2025_i", "aime_2025_ii",
                                      "aime_2026_i", "aime_2026_ii"]:
                    year_map = {
                        "aime_2025": "MathArena/aime_2025",
                        "aime2025":  "MathArena/aime_2025",
                        "aime_2026": "MathArena/aime_2026",
                        "aime2026":  "MathArena/aime_2026",
                        "aime_2025_i":  "MathArena/aime_2025_I",
                        "aime_2025_ii": "MathArena/aime_2025_II",
                        "aime_2026_i":  "MathArena/aime_2026_I",
                        "aime_2026_ii": "MathArena/aime_2026_II",
                    }
                    hf_path = year_map.get(ds_name_norm, "MathArena/aime_2025")
                    ds = load_dataset(hf_path, split="train")
                    idxs = list(range(len(ds)))  # small dataset — use all
                    for i in idxs[:per_ds]:
                        q = str(ds[i].get("problem", "")).strip()
                        a = str(ds[i].get("answer", "")).strip()
                        tag = hf_path.replace("MathArena/", "").lower()
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": tag,
                            "id": f"{tag}_{i}",
                        }
                        pool.append(item)

                # ----------------------------------------------------------------
                # [2025] HMMT Feb/Nov 2025 + Feb 2026 — MathArena
                # Keys: "hmmt_feb_2025", "hmmt_nov_2025", "hmmt_feb_2026",
                #       "hmmt_2025", "hmmt_2026"
                # HuggingFace: MathArena/hmmt_{feb|nov}_{year}
                # Fields: problem (str), answer (str, can be expression)
                # Difficulty: harder than AIME; best models ~87 %
                # ----------------------------------------------------------------
                elif ds_name_norm.startswith("hmmt"):
                    hmmt_map = {
                        "hmmt_feb_2025": "MathArena/hmmt_feb_2025",
                        "hmmt_nov_2025": "MathArena/hmmt_nov_2025",
                        "hmmt_feb_2026": "MathArena/hmmt_feb_2026",
                        "hmmt_2025":     "MathArena/hmmt_nov_2025",
                        "hmmt_2026":     "MathArena/hmmt_feb_2026",
                    }
                    hf_path = hmmt_map.get(ds_name_norm, "MathArena/hmmt_feb_2026")
                    ds = load_dataset(hf_path, split="train")
                    idxs = list(range(len(ds)))
                    for i in idxs[:per_ds]:
                        q = str(ds[i].get("problem", "")).strip()
                        a = str(ds[i].get("answer", "")).strip()
                        tag = hf_path.replace("MathArena/", "").lower()
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": tag,
                            "id": f"{tag}_{i}",
                        }
                        pool.append(item)

                # ----------------------------------------------------------------
                # [2025] OlymMATH — RUC-AIBOX (Olympiad-level, bilingual)
                # Keys: "olymmath", "olymmath_hard", "olymmath_easy",
                #       "olymmath_en_hard", "olymmath_en_easy"
                # HuggingFace: RUC-AIBOX/OlymMATH  (configs: EN-HARD, EN-EASY, ZH-*)
                # Fields: problem, answer, subject, difficulty
                # Difficulty: HARD → frontier models ~58 %; EASY → AIME level
                # ----------------------------------------------------------------
                elif ds_name_norm.startswith("olymmath"):
                    # NOTE: HF config names are lowercase: en-hard / en-easy / zh-hard / zh-easy
                    if "easy" in ds_name_norm:
                        cfg = "en-easy"
                    else:
                        cfg = "en-hard"   # default to harder split
                    ds = load_dataset("RUC-AIBOX/OlymMATH",
                                      name=cfg, split="test")
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = str(ds[i].get("problem", "")).strip()
                        a = str(ds[i].get("answer", "")).strip()
                        subj = ds[i].get("subject", "")
                        tag = f"olymmath-{cfg.lower()}"
                        if subj:
                            tag += f":{subj}"
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": tag,
                            "id": f"olymmath_{cfg.lower()}_{i}",
                        }
                        pool.append(item)

                # ----------------------------------------------------------------
                # [2025] AMO-Bench — Meituan LongCat (original, contamination-free)
                # Keys: "amo_bench", "amo-bench", "amobench"
                # HuggingFace: meituan-longcat/AMO-Bench
                # Fields: problem, answer, category
                # Difficulty: IMO-level, hand-crafted; best model ~63 %
                # ----------------------------------------------------------------
                elif ds_name_norm in ["amo_bench", "amo-bench", "amobench"]:
                    ds = load_dataset("meituan-longcat/AMO-Bench", split="train",
                                      trust_remote_code=True)
                    idxs = list(range(len(ds)))  # only 50 problems
                    for i in idxs[:per_ds]:
                        q = str(ds[i].get("problem", "")).strip()
                        a = str(ds[i].get("answer", "")).strip()
                        cat = ds[i].get("category", "")
                        tag = f"amo-bench:{cat}" if cat else "amo-bench"
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": tag,
                            "id": f"amo_bench_{i}",
                        }
                        pool.append(item)

                # ----------------------------------------------------------------
                # [2025] Omni-MATH — KbsdJames (large olympiad collection)
                # Keys: "omni_math", "omnimath", "omni-math"
                # HuggingFace: KbsdJames/Omni-MATH  (train split, 4 428 problems)
                # Fields: problem, answer, domain (33 sub-domains), difficulty
                # Difficulty: olympiad-level; o1-mini ~60 %
                # ----------------------------------------------------------------
                elif ds_name_norm in ["omni_math", "omnimath", "omni-math"]:
                    ds = load_dataset("KbsdJames/Omni-MATH", split="train",
                                      trust_remote_code=True)
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = str(ds[i].get("problem", "")).strip()
                        a = str(ds[i].get("answer", "")).strip()
                        domain = ds[i].get("domain", "")
                        diff   = ds[i].get("difficulty", "")
                        tag = "omni-math"
                        if domain:
                            tag += f":{domain.split('/')[0].strip()}"
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": tag,
                            "id": f"omni_math_{i}",
                        }
                        pool.append(item)

                # ----------------------------------------------------------------
                # [2025] LiveMathBench — OpenCompass (anti-contamination)
                # Keys: "livemathbench", "live_math", "livemath"
                # HuggingFace: opencompass/LiveMathBench
                # Configs: v202412_CNMO_en | v202412_AMC_en | v202505_*
                # Fields: problem, answer, subject, competition
                # Difficulty: competition-level, updated post training cutoff
                # ----------------------------------------------------------------
                elif ds_name_norm in ["livemathbench", "live_math", "livemath",
                                      "livemathbench_amc", "livemathbench_cnmo"]:
                    if "amc" in ds_name_norm:
                        cfg = "v202412_AMC_en"
                    elif "cnmo" in ds_name_norm:
                        cfg = "v202412_CNMO_en"
                    else:
                        cfg = "v202505_CNMO_en"   # most recent by default
                    ds = load_dataset("opencompass/LiveMathBench",
                                      name=cfg, split="test",
                                      trust_remote_code=True)
                    idxs = list(range(len(ds)))
                    for i in idxs[:per_ds]:
                        q = str(ds[i].get("problem", "")).strip()
                        a = str(ds[i].get("answer", "")).strip()
                        comp = ds[i].get("competition", cfg)
                        tag = f"livemathbench:{comp}"
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": tag,
                            "id": f"livemathbench_{cfg}_{i}",
                        }
                        pool.append(item)

                else:
                    logger.warning(f"Unknown dataset key: {ds_name}. Skipping.")

            except Exception as e:
                logger.warning(f"Failed to load dataset '{ds_name}': {e}")

        if len(pool) < num_problems:
            curated = [
                {"id": "c1", "puzzle": "If 2x + 3 = 15, what is x?", "answer": "6", "dataset": "curated"},
                {"id": "c2", "puzzle": "Jane has 5 apples. She eats 2. How many left?", "answer": "3", "dataset": "curated"},
                {"id": "c3", "puzzle": "Calculate 15% of 200.", "answer": "30", "dataset": "curated"},
                {"id": "c4", "puzzle": "Solve for x: x^2 - 4 = 0 (positive root)", "answer": "2", "dataset": "curated"},
                {"id": "c5", "puzzle": "A train travels 60 mph for 2 hours. Distance?", "answer": "120", "dataset": "curated"},
            ]
            for c in curated:
                if len(pool) >= num_problems:
                    break
                c2 = dict(c)
                c2["puzzle"] = self._maybe_harden(c2["puzzle"], hardener)
                pool.append(c2)

        random.shuffle(pool)
        return pool[:num_problems]


# --------------------------- Solver (Architect-Engineer Pattern) ---------------------------

@dataclass
class AgentResponse:
    agent: str
    answer: str
    parsed: Any
    confidence: float
    reasoning_trace: str
    quality_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HypothesisResult:
    hypothesis_id: str
    strategy_name: str
    blueprint: dict
    code: Optional[str]
    code_success: bool
    execution_output: str
    answer: str
    parsed_answer: Optional[float]
    confidence: float
    agent_response: Optional[AgentResponse]

@dataclass
class HypothesisLog:
    problem: str
    expected: str
    candidates: List[HypothesisResult] = field(default_factory=list)
    triage_result: Optional[str] = None
    judge_reasoning: Optional[str] = None
    final_answer: str = "unknown"
    final_strategy: str = "none"
    hypothesis_testing_triggered: bool = False
    api_calls_used: int = 3

class QualityEnhancedMultiAgentSolver:
    
    def __init__(self, client: UnifiedLLMClient = None,
                 clients: Dict[AgentRole, UnifiedLLMClient] = None):
        """
        [UPDATED v7.3] Supports heterogeneous model configuration.
        
        Args:
            client: Single client for all roles (backward compatible).
            clients: Dict mapping AgentRole → UnifiedLLMClient.
                     If both provided, 'clients' takes precedence.
                     Missing roles in 'clients' fall back to 'client'.
        """
        # Build role→client mapping
        self._clients: Dict[AgentRole, UnifiedLLMClient] = {}
        
        if clients:
            self._clients = dict(clients)
        
        # Fill any missing roles with the default client
        if client:
            for role in AgentRole:
                if role not in self._clients:
                    self._clients[role] = client
        
        # Validate: every role must have a client
        for role in AgentRole:
            if role not in self._clients:
                raise ValueError(f"No client configured for role {role.value}. "
                                 "Provide either 'client' (default for all) or "
                                 "complete 'clients' dict.")
        
        self.math_temp = 0.0
        self.prog_temp = 0.05
        self.enable_baseline_fallback_on_mas_failure = True
        self.enable_metamorphic_testing = False
        self.enable_hypothesis_testing = True
        # [NEW v10.2] Ablation flag: when False, SIV is never invoked.
        #   - If both SIV and SHT are disabled → degenerate MAS (just Architect+Engineer)
        #   - If SIV disabled but SHT enabled  → confidence gate still runs but
        #     without SIV signal; baseline disagreement still triggers SHT.
        # This is what baselines.py B5 (MAS-NoSIV) flips.
        self.enable_siv = True

        # [v14.0] Blueprint reasoning mode. See run_mathematician_analysis.
        #   True  — 'DERIVATION:' free text, then 'BLUEPRINT:' JSON, one call
        #   False — v13.0 behaviour: JSON only, no scratchpad
        # Diagnostic basis (v13.0 run, n=150): on gsm8k_test the same model
        # answers 100% of problems correctly with free CoT, but only 41.7% of
        # its blueprints evaluate to the gold answer. The prompt demanded
        # "Return ONLY valid JSON, no preamble" AND "mentally trace through your
        # equations" — contradictory instructions, since constrained decoding
        # (which reserved a leading 'reasoning' field for exactly this) is
        # disabled on the Kaggle transformers build. The model had nowhere to
        # think. Zero extra LLM calls; falls back to the existing v11.0
        # CoT->blueprint extraction if the JSON section never arrives.
        self.math_reasoning_first = True

        # [v7.3] Log the configuration
        self._log_model_config()
    
    def _get_client(self, role: AgentRole) -> UnifiedLLMClient:
        """Get the client assigned to a specific agent role."""
        return self._clients[role]
    
    def _log_model_config(self):
        """Log which model is assigned to each role."""
        logger.info("=" * 50)
        logger.info("HETEROGENEOUS MODEL CONFIGURATION:")
        is_homogeneous = len(set(
            f"{c.provider}/{c.model_name}" for c in self._clients.values()
        )) == 1
        if is_homogeneous:
            c = list(self._clients.values())[0]
            logger.info(f"  [Homogeneous] All roles → {c.provider}/{c.model_name}")
        else:
            for role in AgentRole:
                c = self._clients[role]
                logger.info(f"  {role.value:<25} → {c.provider}/{c.model_name}")
        logger.info("=" * 50)
    
    def get_model_config_summary(self) -> Dict[str, str]:
        """Return a summary dict for logging/CSV output."""
        return {
            f"model_{role.value}": f"{self._clients[role].provider}/{self._clients[role].model_name}"
            for role in AgentRole
        }

    # -------------------------------------------------------------------------
    # Extract Answer (with error guard)
    # -------------------------------------------------------------------------
    
    def extract_answer(self, text: Any) -> Tuple[str, Any]:
        """
        Enhanced answer extraction with error response guard.
        """
        # [FIX v7.1] Check for error response FIRST
        if _is_error_response(text):
            logger.warning(f"extract_answer received error response: {str(text)[:150]}")
            return "unknown", None
        
        text = str(text)

        # Strategy 1: ANSWER: [[...]] tag
        match = re.search(r'ANSWER:\s*\[\[([^\]]+)\]\]', text, re.IGNORECASE)
        if match:
            val = match.group(1).strip()
            return val, val

        # Strategy 1b: \boxed{N} — output format of Qwen2.5-Math / DeepSeek-R1
        # [v10.3] Parse the LAST \boxed{} in case there are intermediate ones.
        boxed_matches = re.findall(r'\\boxed\{(-?\d+(?:[.,]\d+)*)\}', text)
        if boxed_matches:
            try:
                val = float(boxed_matches[-1].replace(',', ''))
                return str(val), val
            except ValueError:
                pass

        # Strategy 2: Extract last number
        num = _extract_last_number(text)
        if num is not None:
            return str(num), num
        
        # Strategy 3: Check for common answer patterns
        answer_patterns = [
            r'(?:answer|result|solution)\s*(?:is|=|:)\s*([0-9,.]+)',
            r'final\s+(?:answer|result)\s*(?:is|=|:)\s*([0-9,.]+)',
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    val = float(match.group(1).replace(',', ''))
                    return str(val), val
                except:
                    pass
        
        return "unknown", None

    def _last_number_from_text(self, s: str) -> Optional[float]:
        return _extract_last_number(s)

    # -------------------------------------------------------------------------
    # Mathematician Agent (Architect)
    # -------------------------------------------------------------------------
    
    def run_mathematician_analysis(self, problem: str, repair_context: str = "",
                                   prior_blueprint: Optional[dict] = None) -> dict:
        """
        [v9.0] Enhanced Mathematician with:
        1. Self-verification: asks the LLM to mentally compute the answer
           from its own equations and check if it's reasonable
        2. Retry on failure: if JSON parsing fails, retries once with
           a simpler prompt instead of returning empty blueprint

        [v13.0] repair_context: when non-empty (SIV's error-localization
        report from a FAILED execution audit on this Mathematician's own
        prior blueprint), the user message asks for a corrected re-derivation
        instead of a first attempt. See _attempt_blueprint_repair, the caller.

        [v14.0] prior_blueprint: the blueprint being repaired, echoed back into
        the prompt. v13.0 sent only the error report, so the model was asked to
        "re-derive from scratch" while being told which of its variables were
        implicated — a correction request without the thing to correct. Showing
        the givens/equations makes the repair a targeted edit, which is what the
        Layer-0 structural reports (a single named undefined symbol) actually
        call for.
        """
        
        sys_msg = """You are an expert Mathematician analyzing word problems.

Your task: Break down the problem into a structured solution plan.

OUTPUT FORMAT (strict JSON):
{
  "unknown": "what we need to find (one sentence)",
  "givens": {
    "variable_name_1": numeric_value,
    "variable_name_2": numeric_value
  },
  "solution_steps": [
    "Step 1: Clear description of first calculation",
    "Step 2: What to calculate next using Step 1 result",
    "Step 3: Final calculation to get the answer"
  ],
  "equations": [
    "step1_result = givens['variable_name_1'] + givens['variable_name_2']",
    "step2_result = step1_result * 2",
    "answer = step2_result"
  ],
  "expected_answer": "your mental estimate of what the numeric answer should be",
  "distractor_check": "List any numbers/info in the problem to IGNORE (if any)"
}

CRITICAL RULES:
1. Extract ONLY relevant numbers into 'givens'. Ignore irrelevant numbers.
2. Use descriptive variable names (e.g., 'initial_apples', 'eaten_apples')
3. Each equation must be valid Python code referencing givens['key']
4. The LAST equation must assign to 'answer'
5. SELF-CHECK: Before outputting, mentally trace through your equations with the actual numbers. Does the result match your expected_answer? If not, fix your equations.
6. Return ONLY valid JSON, no preamble or explanation

EXAMPLE:
Problem: "Jane has 10 apples. She eats 3 and buys 5 more. How many does she have?"
Output:
{
  "unknown": "total apples Jane has",
  "givens": {"initial_apples": 10, "eaten_apples": 3, "bought_apples": 5},
  "solution_steps": [
    "Step 1: Subtract eaten from initial: 10 - 3 = 7",
    "Step 2: Add bought: 7 + 5 = 12"
  ],
  "equations": [
    "remaining = givens['initial_apples'] - givens['eaten_apples']",
    "answer = remaining + givens['bought_apples']"
  ],
  "expected_answer": "12",
  "distractor_check": "None"
}
"""

        # [v14.0] Reasoning-first variant of the same contract. Rule 6 above
        # ("Return ONLY valid JSON, no preamble") is what removed the model's
        # scratchpad; here the derivation is explicitly requested FIRST and the
        # JSON follows a marker, so structuring becomes transcription of a
        # solution the model has already worked out rather than a single-shot
        # act of formalisation. The JSON contract itself is unchanged, so
        # everything downstream (_extract_blueprint_json, the Programmer's
        # prompt, SIV) sees exactly the same object.
        if self.math_reasoning_first and not repair_context:
            sys_msg = sys_msg.replace(
                "6. Return ONLY valid JSON, no preamble or explanation",
                "6. OUTPUT IN TWO PARTS, in this order:\n"
                "   DERIVATION:\n"
                "   <solve the problem step by step in plain text, doing the actual\n"
                "    arithmetic, until you reach the numeric answer. Be concise.>\n"
                "   BLUEPRINT:\n"
                "   <the JSON object above, and nothing after it>\n"
                "   The JSON must reproduce the derivation you just wrote: its\n"
                "   equations, evaluated in order, must yield the answer you reached."
            ).replace(
                "EXAMPLE:\nProblem:", "EXAMPLE (JSON part only):\nProblem:"
            )

        if repair_context:
            prior_txt = ""
            if prior_blueprint:
                prior_txt = (
                    "\nYour PREVIOUS blueprint for this exact problem was:\n"
                    + json.dumps(
                        {
                            "givens": prior_blueprint.get("givens", {}),
                            "equations": prior_blueprint.get("equations", []),
                        },
                        ensure_ascii=False, indent=1,
                    )[:1200]
                    + "\n"
                )
            user_msg = (
                f"Problem:\n{problem}\n"
                f"{prior_txt}\n"
                f"An automated symbolic checker rejected it:\n{repair_context}\n\n"
                "Produce a CORRECTED JSON blueprint. Keep whatever was already right; "
                "change what the report identifies. Requirements it must satisfy:\n"
                "- every name on the right-hand side of an equation is either "
                "givens['<key>'] with that key declared to a NUMBER in 'givens', or a "
                "variable assigned by an EARLIER equation in the list;\n"
                "- the last equation assigns to `answer`;\n"
                "- the values in 'givens' are the numbers the problem text actually "
                "states, used the way the problem describes them.\n"
                "Return ONLY the corrected JSON blueprint."
            )
        elif self.math_reasoning_first:
            user_msg = (
                f"Problem:\n{problem}\n\n"
                "Write DERIVATION: then BLUEPRINT: exactly as instructed."
            )
        else:
            user_msg = f"Problem:\n{problem}\n\nAnalyze and return the JSON blueprint."

        msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]

        # [v11.2] Plain call — instruction-tuned models (Qwen2.5-7B-Instruct, API)
        # follow the JSON-blueprint prompt directly. Constrained decoding (json_schema)
        # is left available in call_model for CoT-only models, but lm-format-enforcer
        # is incompatible with the Kaggle transformers build, so we rely on the model.
        # max_tokens 1536: room for the full blueprint without mid-object truncation.
        # [v14.0] reasoning-first needs room for the derivation AS WELL as the
        # JSON; too small a budget truncates before the BLUEPRINT: section and
        # silently sends every problem down the CoT-extraction fallback.
        res = self._get_client(AgentRole.MATHEMATICIAN).call_model(
            msgs, temperature=self.math_temp,
            max_tokens=2304 if (self.math_reasoning_first and not repair_context) else 1536,
        )
        blueprint = _extract_blueprint_json(_after_blueprint_marker(str(res)))

        res2 = None  # [v10.2] kept for CoT fallback scoping below
        # [v9.0] If blueprint is empty (JSON parse failed), retry with simpler prompt.
        # [v11.0] Skip this retry for local_hf CoT models — they re-emit prose, not
        # JSON, so the retry is a wasted ~40s call; the extraction step below handles
        # them directly off the primary CoT output.
        _math_provider = getattr(self._get_client(AgentRole.MATHEMATICIAN), "provider", "")
        if (not blueprint.get("equations") and not blueprint.get("givens")
                and _math_provider != "local_hf"):
            logger.info("Blueprint empty — retrying Mathematician with simplified prompt")
            retry_msg = f"""Solve this math problem step by step. Extract the numbers, write Python equations, and give the answer.

Problem: {problem}

Reply with ONLY this JSON (no other text):
{{"givens": {{"name": number}}, "equations": ["answer = ..."], "unknown": "what to find", "solution_steps": ["Step 1: ..."], "expected_answer": "number", "distractor_check": "None"}}"""

            res2 = self._get_client(AgentRole.MATHEMATICIAN).call_model(
                [{"role": "user", "content": retry_msg}],
                temperature=0.0, max_tokens=1000
            )  # [v10.6] 600→1000, same reason as the primary call
            blueprint2 = _extract_blueprint_json(str(res2))
            if blueprint2.get("equations") or blueprint2.get("givens"):
                logger.info("Retry succeeded — got valid blueprint")
                blueprint = blueprint2

        # [v11.0] Two-step blueprint extraction — THE fix for CoT-trained models.
        # ---------------------------------------------------------------------
        # Math-specialist models (Qwen2.5-Math, DeepSeek-Math) are RL-tuned to emit
        # free-form chain-of-thought ending in \boxed{}, NOT JSON, so the two attempts
        # above return empty. Instead of giving up (or building a tautological
        # passthrough that the Programmer can't act on and SIV can't meaningfully
        # verify), we make ONE focused extraction call: hand the model its OWN worked
        # solution and ask it to convert that into structured givens + equations.
        #
        # Extracting structure from an existing derivation is far easier for these
        # models than generating it from scratch — so this is where the Architect→
        # Engineer→Verifier pipeline (Programmer execution, SIV audit, SHT) finally
        # receives real material. Blueprints carry _extracted_from_cot=True for logging.
        if not blueprint.get("equations") and not blueprint.get("givens"):
            math_client = self._get_client(AgentRole.MATHEMATICIAN)
            raw_cot = str(res2 if res2 is not None else res)
            if raw_cot.strip() and not _is_error_response(raw_cot):
                # Keep setup (givens, stated early) AND conclusion (answer, at the end).
                if len(raw_cot) > 3000:
                    cot_excerpt = raw_cot[:1500] + "\n...\n" + raw_cot[-1500:]
                else:
                    cot_excerpt = raw_cot
                extract_prompt = f"""You are given a math problem and a worked solution. Convert the solution into a structured JSON blueprint a Python program can execute to reproduce the answer.

PROBLEM:
{problem}

WORKED SOLUTION:
{cot_excerpt}

Rules:
- "givens": JSON object mapping snake_case names to the NUMERIC input values (numbers only, no units, no text).
- "equations": ordered list of Python assignment strings using givens['name'] and earlier results; the LAST line MUST assign to `answer`.
- "expected_answer": the final numeric answer from the solution.

Output ONLY this JSON, nothing else:
{{"unknown": "what to find", "givens": {{"name": number}}, "equations": ["step1 = givens['a'] + givens['b']", "answer = step1 * 2"], "solution_steps": ["Step 1: ..."], "expected_answer": "number", "distractor_check": "None"}}"""
                res3 = math_client.call_model(
                    [{"role": "user", "content": extract_prompt}],
                    temperature=0.0, max_tokens=800,
                )
                bp3 = _extract_blueprint_json(str(res3))
                if bp3.get("equations") and bp3.get("givens"):
                    logger.info(
                        f"CoT→blueprint extraction SUCCESS: {len(bp3['givens'])} givens, "
                        f"{len(bp3['equations'])} equations (expected_answer={bp3.get('expected_answer')})"
                    )
                    bp3["_extracted_from_cot"] = True
                    blueprint = bp3
                else:
                    logger.info(
                        "CoT→blueprint extraction yielded no executable equations "
                        f"| raw head: {str(res3)[:160]!r}"
                    )

        # [v10.2] Last-resort tautological passthrough (local_hf only) — reached only
        # if extraction above also failed. Keeps an answer flowing (answer = givens
        # ['answer']) so the problem isn't a total loss, but SIV can only trivially
        # verify it. _local_hf_fallback=True marks this degraded path for logging.
        if not blueprint.get("equations") and not blueprint.get("givens"):
            math_client = self._get_client(AgentRole.MATHEMATICIAN)
            if getattr(math_client, "provider", "") == "local_hf":
                raw_cot = str(res2 if res2 is not None else res)
                extracted = _extract_last_number(raw_cot)
                if extracted is not None:
                    logger.info(
                        f"local_hf CoT fallback (tautological): answer={extracted} "
                        f"from {len(raw_cot)}-char reasoning text"
                    )
                    blueprint = {
                        "unknown": "the answer",
                        "givens": {"answer": extracted},
                        "solution_steps": [f"[CoT] {raw_cot[:400]}"],
                        "equations": ["answer = givens['answer']"],
                        "expected_answer": str(extracted),
                        "distractor_check": "",
                        "metamorphic_tests": [],
                        "_local_hf_fallback": True,
                    }
                else:
                    logger.warning(
                        "CoT fallback: no numeric answer found in CoT text"
                        " — blueprint stays empty, programmer_failed expected"
                        f" | raw head: {raw_cot[:220]!r}"
                    )

        # [v11.3] Log the final blueprint content so we can SEE what the model
        # produced (givens + equations) — the key diagnostic for whether the
        # structured pipeline is receiving real material or just falling back.
        _g = blueprint.get("givens", {}) or {}
        _eq = blueprint.get("equations", []) or []
        _src = ("tautological" if blueprint.get("_local_hf_fallback")
                else "extracted" if blueprint.get("_extracted_from_cot")
                else "primary")
        logger.info(
            f"Blueprint[{_src}]: {len(_g)} givens, {len(_eq)} equations, "
            f"expected_answer={blueprint.get('expected_answer')!r}"
        )
        if _eq:
            logger.info(f"  givens={_g}")
            logger.info(f"  equations={_eq}")

        return blueprint

    # -------------------------------------------------------------------------
    # [v13.0] Blueprint repair (one attempt, gated on execution-audit failure)
    # -------------------------------------------------------------------------

    def _attempt_blueprint_repair(self, problem: str, blueprint: dict,
                                  programmer_response: "AgentResponse",
                                  siv_result: "SIVResult"
                                  ) -> Tuple[dict, "AgentResponse", Optional["SIVResult"], bool]:
        """
        [v13.0] One repair attempt when SIV's execution audit fails.

        Measured motivation (mixed-preset run, n=150, see mas_full_20260712
        analysis): among audited rows whose blueprint answer was wrong,
        execution_audit_passed=False caught 49/61 (80.3% recall) with ZERO
        false alarms on correct blueprints (0/25). That is the single
        highest-leverage signal available: it targets exactly the failure
        mode this gate checks for (internally inconsistent equations), it
        never fires on a blueprint that was already fine, and it is
        available for free (SIV already computed it). This hands the
        Mathematician its own error report and asks for ONE corrected
        re-derivation BEFORE the answer ever reaches the confidence gate or
        SHT — SIV acting as an in-loop repair signal, not just a post-hoc
        detector. Capped at one attempt: this targets a specific, diagnosed
        failure, not a general retry-until-it-works loop.

        Returns (blueprint, programmer_response, siv_result, repaired) — the
        ORIGINAL triple unchanged whenever repair wasn't attempted or didn't
        produce something usable, so the caller never needs a separate
        repaired/not-repaired branch.

        [v14.0] The eligibility test changed. It used to be

            if ... or not siv_result.invertible: return   # skip

        which excluded every blueprint whose equation chain could not be
        inverted — i.e. precisely the 30/109 rows of the v13.0 run whose chain
        could not even be EVALUATED (undefined names on an equation's
        right-hand side, blueprint_answer=None). Those are the most clearly
        defective blueprints in the run and were the only ones never offered a
        repair. An unevaluable chain is a stronger reason to re-derive, not a
        reason to give up. Now: repair whenever the audit did not pass and SIV
        actually ran (`skipped` distinguishes "nothing to audit" from
        "audited and failed" — both leave execution_audit_passed=False).
        """
        if siv_result is None or siv_result.skipped or siv_result.execution_audit_passed:
            return blueprint, programmer_response, siv_result, False

        error_report = SymbolicInverseVerifier.get_error_localization_report(siv_result)
        _kind = "structural" if siv_result.has_structural_defect else "arithmetic"
        logger.info(
            f"Blueprint repair attempt ({_kind}) — SIV error report: {error_report[:300]}"
        )

        new_blueprint = self.run_mathematician_analysis(
            problem, repair_context=error_report, prior_blueprint=blueprint
        )
        if not new_blueprint.get("equations") or not new_blueprint.get("givens"):
            logger.info("Blueprint repair: Mathematician returned nothing usable — keeping original")
            return blueprint, programmer_response, siv_result, False

        new_response = self.run_programmer_solver(problem, new_blueprint)
        new_answer_num = _extract_last_number(new_response.answer)
        if new_answer_num is None:
            logger.info("Blueprint repair: repaired blueprint's code produced no answer — keeping original")
            return blueprint, programmer_response, siv_result, False

        new_siv_result = None
        if SYMPY_AVAILABLE and new_blueprint.get("equations"):
            new_siv_result = SymbolicInverseVerifier.verify(new_blueprint, new_answer_num)
            # [v14.0] If the repaired blueprint's answer came from the SymPy
            # symbolic fallback, that answer IS the evaluation of these same
            # equations (SymbolicSolver.solve_from_blueprint execs them), so
            # Layer 1 passes by construction and certifies nothing. The main SIV
            # gate in solve() excludes such answers by agent name; this path did
            # not, and on the v13.0 run all 4 rows that leaked through came back
            # verified=True with 2 of the 4 blueprint answers plainly wrong.
            # Flag rather than drop: the verdict stays visible in the logs and
            # CSV, but it is excluded from the confidence boost below and can be
            # filtered out of detector statistics.
            if "SymPy" in (new_response.agent or ""):
                new_siv_result.tautological = True
                logger.info(
                    "Post-repair audit marked TAUTOLOGICAL (answer produced by the "
                    "blueprint's own equations via SymPy fallback) — carries no "
                    "verification evidence."
                )

        logger.info(
            f"Blueprint repair: original audit_passed={siv_result.execution_audit_passed} -> "
            f"repaired audit_passed={new_siv_result.execution_audit_passed if new_siv_result else None}"
        )
        new_blueprint["_blueprint_repaired"] = True
        return new_blueprint, new_response, new_siv_result, True

    # -------------------------------------------------------------------------
    # Programmer Agent (Engineer)
    # -------------------------------------------------------------------------

    def run_programmer_solver(self, problem: str, blueprint: dict, max_attempts: int = 3) -> AgentResponse:
        
        givens = blueprint.get("givens", {})
        equations = blueprint.get("equations", [])
        solution_steps = blueprint.get("solution_steps", [])
        unknown = blueprint.get("unknown", "the answer")
        
        # [FIX v7.1] If blueprint has no equations (e.g., from error), fail fast
        if not equations and not givens:
            logger.warning("Programmer received empty blueprint (likely from API error)")
            return AgentResponse(
                agent="Programmer (empty blueprint)",
                answer="unknown",
                parsed="unknown",
                confidence=0.0,
                reasoning_trace="Blueprint was empty — likely API error",
                quality_metrics={"error": "empty_blueprint"}
            )
        
        blueprint_text = _format_blueprint_for_programmer(blueprint)
        
        # [v12.0] PAL-style prompt. The Programmer solves the PROBLEM directly
        # (single-call PAL scored 88.0% in the n=150 pilot vs 74.0% for the
        # blueprint-transcribing pipeline). The Architect blueprint is shown as
        # a colleague's draft to cross-check against — NOT as equations to
        # transcribe — so a mistranslated blueprint no longer poisons the
        # Programmer, and SIV's execution audit becomes a genuine independent
        # cross-check (free program vs declarative blueprint).
        # The `givens = {...}` first-line convention is kept: rule-based
        # verification, metamorphic testing and SIV givens-matching depend on it.
        sys_msg = """You are an expert Python programmer solving math word problems.

STRICT RULES:
1. Read the PROBLEM carefully and solve it yourself. Trust the PROBLEM text.
2. Start your code with: givens = {...}  — a dict of the numeric values you
   extract from the PROBLEM (clear snake_case names).
3. Compute the solution step by step from those givens using plain Python
   (math module allowed, nothing else).
4. Store the final result in a variable called 'answer'
5. Print ONLY the final numeric answer: print(answer) — no explanations, no units
6. A colleague's draft analysis may be provided. Use it to double-check your
   understanding, but if it conflicts with the PROBLEM, follow the PROBLEM.

EXAMPLE:
```python
givens = {"initial": 10, "used": 3}
remaining = givens['initial'] - givens['used']
answer = remaining
print(answer)
```

OUTPUT FORMAT:
- Python code in ```python ... ``` block
- After code, write: ANSWER: [[<number>]]
"""

        user_msg = f"""PROBLEM:
{problem}

COLLEAGUE'S DRAFT ANALYSIS (may contain errors — cross-check, do not transcribe):
{blueprint_text}

Write a Python program that solves the PROBLEM and prints the final numeric answer.
"""
        
        repair_feedback = ""
        best_answer = None
        last_code = None
        
        for attempt in range(max_attempts):
            msgs = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg + repair_feedback}
            ]
            
            raw_response = self._get_client(AgentRole.PROGRAMMER).call_model(
                msgs, 
                temperature=self.prog_temp, 
                max_tokens=1000
            )
            
            # [FIX v7.1] Check for error response from LLM
            if _is_error_response(raw_response):
                logger.warning(f"Programmer attempt {attempt+1}: API returned error")
                repair_feedback = f"\n\n[Attempt {attempt+1}] LLM call failed. Retrying..."
                continue
            
            # Extract code
            code = _extract_code_from_response(str(raw_response))
            if not code:
                repair_feedback = f"\n\n[Attempt {attempt+1}] ERROR: No code block found. Use ```python ... ```"
                continue
            
            last_code = code
            
            # Execute code
            success, output = PythonExecutor.execute(code)
            if not success:
                repair_feedback = f"\n\n[Attempt {attempt+1}] EXECUTION ERROR:\n{output}\n\nFix the code and try again."
                continue
            
            # Extract answer
            answer_num = _extract_last_number(output)
            if answer_num is None:
                repair_feedback = f"\n\n[Attempt {attempt+1}] FORMAT ERROR: Output was '{output}'. Print only a number."
                continue
            
            best_answer = str(answer_num)
            
            # Optional: Metamorphic testing
            gate_log = "Metamorphic testing disabled"
            if self.enable_metamorphic_testing:
                tests = blueprint.get("metamorphic_tests", [])
                if tests:
                    ok_gate, gate_log = self._metamorphic_gate(code, tests)
                    if not ok_gate:
                        gate_log = f"WARNING: {gate_log}"
            
            # Success!
            return AgentResponse(
                agent="Programmer (optimized)",
                answer=best_answer,
                parsed=best_answer,
                confidence=1.0,
                reasoning_trace=code[:500],
                quality_metrics={
                    "execution_output": output,
                    "metamorphic_gate": gate_log,
                    "attempts": attempt + 1
                }
            )
        
        # Failed after all attempts — try SymPy symbolic solver as fallback
        sympy_answer = None
        sympy_trace = ""
        if SYMPY_AVAILABLE and blueprint.get("equations"):
            logger.info("Programmer failed. Attempting SymPy symbolic solver fallback...")
            sym_ok, sym_ans, sym_trace = SymbolicSolver.solve_from_blueprint(blueprint)
            sympy_trace = sym_trace
            if sym_ok:
                sym_num = _extract_last_number(sym_ans)
                if sym_num is not None:
                    sympy_answer = str(sym_num)
                    logger.info(f"SymPy fallback SUCCESS: {sympy_answer}")

        if sympy_answer:
            return AgentResponse(
                agent="SymPy (symbolic fallback)",
                answer=sympy_answer,
                parsed=sympy_answer,
                confidence=0.8,  # High confidence (correct arithmetic) but no code verification
                reasoning_trace=sympy_trace[:500],
                quality_metrics={
                    "solver": "sympy_symbolic",
                    "programmer_failed_attempts": max_attempts,
                    "last_code_error": repair_feedback[:200] if repair_feedback else "N/A"
                }
            )

        fallback_answer = best_answer if best_answer else "unknown"
        return AgentResponse(
            agent="Programmer (failed)",
            answer=fallback_answer,
            parsed=fallback_answer,
            confidence=0.2,
            reasoning_trace=last_code[:500] if last_code else "No code generated",
            quality_metrics={
                "error": "Max attempts reached",
                "last_feedback": repair_feedback,
                "sympy_attempted": SYMPY_AVAILABLE,
                "sympy_trace": sympy_trace[:200] if sympy_trace else "N/A"
            }
        )

    # -------------------------------------------------------------------------
    # [NEW v8.0] Process-Level Verification
    # -------------------------------------------------------------------------
    
    def verify_code_against_blueprint(self, problem: str, blueprint: dict,
                                       code: str, code_answer: str) -> Tuple[bool, str, float]:
        """
        [v9.0] Purely rule-based verification (0 API calls).
        
        Cross-checks:
        1. Givens consistency: code uses same values as blueprint
        2. Equation coverage: all blueprint equations have corresponding code
        3. Answer sanity: sign, magnitude, and expected_answer match
        """
        givens = blueprint.get("givens", {})
        equations = blueprint.get("equations", [])
        
        issues = []
        
        # Check 1: Givens consistency
        code_givens = _extract_givens_dict_from_code(code)
        if code_givens is not None and givens:
            for key, val in givens.items():
                if key not in code_givens:
                    issues.append(f"Missing given '{key}'")
                elif isinstance(val, (int, float)) and isinstance(code_givens.get(key), (int, float)):
                    if abs(code_givens[key] - val) > 1e-6:
                        issues.append(f"Givens mismatch '{key}': blueprint={val} code={code_givens[key]}")
        
        # Check 2: Equation variables in code
        for eq in equations:
            if "=" in eq:
                var_name = eq.split("=")[0].strip()
                if var_name not in code and var_name != "answer":
                    issues.append(f"Missing variable '{var_name}'")
        
        # Check 3: Answer sanity
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
        
        # Check 4: [v9.0] Cross-check with Mathematician's expected_answer
        expected = blueprint.get("expected_answer", "")
        if expected and answer_num is not None:
            expected_num = _extract_last_number(str(expected))
            if expected_num is not None and abs(expected_num) > 0.01:
                rel_diff = abs(answer_num - expected_num) / max(abs(expected_num), 1e-9)
                if rel_diff > 0.1:  # More than 10% off from Mathematician's estimate
                    issues.append(f"Code answer ({answer_num}) differs from Mathematician estimate ({expected_num}) by {rel_diff:.0%}")
        
        # Score
        if not issues:
            return True, "All checks passed", 1.0
        
        critical = sum(1 for i in issues if "mismatch" in i.lower() or "negative" in i.lower() or "implausibly" in i.lower())
        minor = len(issues) - critical
        
        if critical > 0:
            confidence = max(0.3, 1.0 - critical * 0.25 - minor * 0.1)
            return False, "; ".join(issues[:3]), confidence
        
        confidence = max(0.6, 1.0 - minor * 0.1)
        return True, "; ".join(issues[:3]), confidence

    # -------------------------------------------------------------------------
    # Metamorphic Testing (Optional)
    # -------------------------------------------------------------------------
    
    def _metamorphic_gate(self, code_block: str, tests: list) -> Tuple[bool, str]:
        base_givens = _extract_givens_dict_from_code(code_block)
        if base_givens is None:
            return False, "No givens dict found"
        
        ok, base_out = PythonExecutor.execute(code_block)
        if not ok:
            return False, f"Base execution failed: {base_out}"
        
        base_val = _extract_last_number(base_out)
        if base_val is None:
            return False, f"Base output not numeric: {base_out}"
        
        logs = []
        for test in tests[:3]:
            name = test.get("name", "unnamed")
            muts = test.get("mutations", [])
            assertion = test.get("assert", {})
            
            mutated_givens = dict(base_givens)
            try:
                for mu in muts:
                    var = mu["var"]
                    op = mu["op"]
                    val = mu["value"]
                    
                    if var not in mutated_givens:
                        raise KeyError(f"Variable '{var}' not in givens")
                    
                    if op == "add":
                        mutated_givens[var] += val
                    elif op == "mul":
                        mutated_givens[var] *= val
                    else:
                        raise ValueError(f"Unknown op: {op}")
            except Exception as e:
                logs.append(f"[{name}] SKIP: {e}")
                continue
            
            mutated_code = _replace_givens_dict_in_code(code_block, mutated_givens)
            ok2, out2 = PythonExecutor.execute(mutated_code)
            if not ok2:
                logs.append(f"[{name}] SKIP: Execution failed")
                continue
            
            val2 = _extract_last_number(out2)
            if val2 is None:
                logs.append(f"[{name}] SKIP: Output not numeric")
                continue
            
            atype = assertion.get("type")
            aval = assertion.get("value")
            
            passed = False
            try:
                if atype == "delta":
                    passed = abs((val2 - base_val) - float(aval)) < 1e-6
                elif atype == "ratio":
                    if abs(base_val) > 1e-6:
                        passed = abs((val2 / base_val) - float(aval)) < 1e-4
                elif atype == "monotonic":
                    if aval == "increase":
                        passed = val2 > base_val
                    elif aval == "decrease":
                        passed = val2 < base_val
            except Exception as e:
                logs.append(f"[{name}] SKIP: Assertion error - {e}")
                continue
            
            status = "PASS" if passed else "FAIL"
            logs.append(f"[{name}] {status}: base={base_val}, mutated={val2}")
            
            if not passed:
                return False, "\n".join(logs)
        
        return True, "\n".join(logs) if logs else "No tests evaluated"

    # =========================================================================
    # STRUCTURED HYPOTHESIS TESTING (SHT)
    # =========================================================================

    def _confidence_gate(self, primary_answer: str, baseline_answer: str,
                         programmer_response: AgentResponse,
                         blueprint: dict,
                         siv_result: Optional[SIVResult] = None) -> Tuple[bool, str]:
        """
        [v10.1] SIV + baseline cross-validated SHT trigger.

        Ordering rationale (changed from v10.0):
          SIV CANNOT detect NL→math translation errors — it audits whatever
          blueprint the Architect produced. The baseline answer comes from a
          different reasoning path (zero-shot CoT) and is the only signal we
          have for translation faithfulness. Therefore the baseline-disagreement
          check MUST run BEFORE any SIV-driven "skip SHT" path; otherwise SIV's
          self-consistent-but-wrong verifications silently override the only
          translation-layer safety net (observed in v10.0: 3/3 regressions on
          n=50 had siv_verified=True with baseline disagreement, all skipped SHT).

        Trigger priority:
          1. Programmer hard-fail            → SHT
          2. Baseline disagreement           → SHT  (catches NL→math errors)
          3. Max repair attempts reached     → SHT
          4. SIV inconsistency / exec error  → SHT
          5. Sanity violations               → SHT
          6. Baseline itself failed          → SHT
          7. SIV verified + baseline agrees  → SKIP (mathematically consistent
                                                     AND cross-validated)
          8. Default                         → SKIP
        """
        # Criterion 1: Programmer hard-fail (nothing to verify)
        if str(primary_answer).strip().lower() == "unknown":
            return False, "programmer_failed"

        # Criterion 2: Baseline cross-check (catches NL→math errors SIV cannot see)
        primary_num = _extract_last_number(primary_answer)
        baseline_num = _extract_last_number(baseline_answer)
        baseline_failed = str(baseline_answer).strip().lower() == "unknown"
        baseline_disagreed = False
        if not baseline_failed:
            if primary_num is not None and baseline_num is not None:
                if abs(primary_num - baseline_num) > 1e-3:
                    baseline_disagreed = True
            elif str(primary_answer).strip() != str(baseline_answer).strip():
                baseline_disagreed = True
        if baseline_disagreed:
            return False, "baseline_disagreement"

        # Criterion 3: Programmer exhausted all repair attempts
        if programmer_response.quality_metrics.get("error") == "Max attempts reached":
            return False, "max_attempts_exhausted"

        # Criterion 4: SIV-driven SHT triggers (failures only — skip path comes last)
        if siv_result is not None:
            if siv_result.invertible and not siv_result.verified:
                return False, f"siv_inconsistency:{','.join(siv_result.failed_givens)}"
            if not siv_result.execution_audit_passed and siv_result.blueprint_answer is not None:
                return False, "siv_execution_error"

        # Criterion 5: Sanity checks
        if primary_num is not None:
            # [v12.0] A negative answer is only suspicious when there is no
            # cross-check: reaching this line means the baseline either agreed
            # (criterion 2 didn't fire) or failed. Two independent paths
            # agreeing on a negative value is evidence, not an anomaly —
            # GSM-Hard golds are frequently negative and the old unconditional
            # trigger sent every agreed-negative into SHT (pilot: 17% MAS vs
            # 83% PAL accuracy on negative-gold problems).
            if primary_num < 0 and baseline_failed:
                return False, "negative_answer_unverified"

            givens = blueprint.get("givens", {})
            if givens:
                max_given = max(
                    (abs(v) for v in givens.values() if isinstance(v, (int, float))),
                    default=0
                )
                if max_given > 0 and abs(primary_num) > max_given * 10000:
                    return False, "answer_magnitude_suspicious"

        # Criterion 6: Baseline itself failed — we have no cross-check, be cautious
        if baseline_failed:
            return False, "baseline_also_failed"

        # Criterion 7: SIV-verified skip ONLY when baseline already agrees
        # (baseline_disagreement was already short-circuited above; reaching here
        # means baseline_num is non-null, baseline didn't fail, and answers agree
        # within 1e-3, so a SIV pass is now genuinely cross-validated)
        if siv_result is not None:
            if (siv_result.execution_audit_passed
                    and siv_result.verified
                    and siv_result.confidence >= 0.90):
                return True, "siv_execution_consistent"

        return True, "all_checks_passed"

    STRATEGY_ARCHETYPES = [
        "Arithmetic-Sequential: chain of basic operations (add, subtract, multiply, divide) applied step by step",
        "Algebraic-Equational: set up one or more equations with unknowns and solve symbolically",
        "Unit-Rate: compute a per-unit rate first, then scale to the target quantity",
        "Working-Backwards: start from what the answer should look like and reverse the operations",
        "Partitioning: split the problem into independent sub-problems, solve each, then combine",
    ]

    def generate_alternative_hypotheses(self, problem: str,
                                        primary_blueprint: dict,
                                        primary_answer: str,
                                        siv_error_report: str = "") -> List[dict]:
        """
        [v10.0] Critic-based hypothesis generation with SIV error localization.
        
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
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()

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

    def _triage_candidates(self, candidates: List[HypothesisResult]) -> Tuple[Optional[str], Optional[str], str]:
        valid = [c for c in candidates if c.code_success and c.parsed_answer is not None]

        if not valid:
            return None, None, "no_valid_candidates"

        groups: Dict[str, List[HypothesisResult]] = {}
        for c in valid:
            matched = False
            for key in groups:
                if abs(c.parsed_answer - float(key)) < 1e-3:
                    groups[key].append(c)
                    matched = True
                    break
            if not matched:
                groups[str(c.parsed_answer)] = [c]

        if not groups:
            return None, None, "no_valid_candidates"

        # [v13.0] Rank and threshold by HONEST votes (the {primary,
        # blueprint_eval} pair counts once — see _honest_votes), so a
        # blueprint agreeing with its own faithful execution can no longer
        # outvote the independent zero-shot baseline.
        sorted_groups = sorted(
            groups.items(),
            key=lambda g: (self._honest_votes(g[1]),
                           sum(c.confidence for c in g[1]) / len(g[1])),
            reverse=True
        )

        best_answer_key, best_group = sorted_groups[0]

        if len(sorted_groups) == 1:
            winner = best_group[0]
            return winner.answer, winner.strategy_name, "unanimous"

        best_votes = self._honest_votes(best_group)
        runner_up_votes = self._honest_votes(sorted_groups[1][1])
        if best_votes >= 2 and best_votes > runner_up_votes:
            winner = best_group[0]
            return winner.answer, winner.strategy_name, "majority"

        return None, None, "no_majority"

    # -------------------------------------------------------------------------
    # [v12.0] Evidence-based arbitration
    # -------------------------------------------------------------------------

    @staticmethod
    def _answers_match(a: Optional[float], b: Optional[float]) -> bool:
        """Same tolerance as the triage grouping (abs 1e-3), plus a relative
        tolerance so large GSM-Hard magnitudes with float noise still match."""
        if a is None or b is None:
            return False
        return abs(a - b) < max(1e-3, 1e-9 * max(abs(a), abs(b)))

    @staticmethod
    def _honest_votes(members: List[HypothesisResult]) -> int:
        """[v13.0] Independent derivations agreeing on this answer.

        `blueprint_eval` is the CAS evaluation of the SAME equations the
        primary Programmer's code implements; the two can only land in one
        group when the SIV execution audit passed, i.e. when the code
        provably computed those equations. That is one derivation observed
        twice, not two independent confirmations — counting it as 2 let the
        pair outvote the genuinely independent zero-shot baseline (measured
        v12.1-mixed run: 36 such 'majority' rows, baseline right 30/36, the
        pair answer right 24/36). blueprint_eval still counts as a full vote
        in any group NOT containing the primary (e.g. agreeing with the
        baseline against a deviant program — genuinely independent there).
        """
        ids = {m.hypothesis_id for m in members}
        votes = len(members)
        if "primary" in ids and "blueprint_eval" in ids:
            votes -= 1
        return votes

    def _group_valid_candidates(self, candidates: List[HypothesisResult]
                                ) -> List[Tuple[float, List[HypothesisResult]]]:
        """Group executable/parsed candidates by numeric answer."""
        valid = [c for c in candidates if c.code_success and c.parsed_answer is not None]
        groups: List[Tuple[float, List[HypothesisResult]]] = []
        for c in valid:
            for rep, members in groups:
                if self._answers_match(c.parsed_answer, rep):
                    members.append(c)
                    break
            else:
                groups.append((c.parsed_answer, [c]))
        return groups

    def _evidence_score(self, rep: float, members: List[HypothesisResult],
                        siv_result: Optional[SIVResult]) -> Tuple[int, int, int, int, int]:
        """
        Deterministic evidence for one answer group. Lexicographic order:
          1. votes      — HONEST independent derivations agreeing on this
                          answer ([v13.0] the {primary, blueprint_eval} pair
                          counts once — the audit-passed CAS value IS the
                          primary's own equations; see _honest_votes)
          2. siv_full   — SIV inverse layer fully verified this answer
                          (audit passed + givens reconstruct, conf ≥ 0.9).
                          [v13.0] tie-breaker BETWEEN corroborated groups
                          only, never a standalone override: measured on the
                          mixed run, verified with baseline agreement was
                          22/22 correct but verified with baseline
                          DISAGREEMENT was 2/14 — self-consistency inherits
                          translation errors
          3. executed   — group contains an answer produced by executed code
                          (pilot: on program-vs-CoT disagreements the program
                          was right 15/29, the CoT 4/29)
          4. primary    — group contains the primary Programmer answer
          5. baseline   — group contains the zero-shot baseline
        """
        ids = {m.hypothesis_id for m in members}
        siv_full = 0
        if (siv_result is not None and siv_result.execution_audit_passed
                and siv_result.verified and siv_result.confidence >= 0.90
                and siv_result.blueprint_answer is not None):
            try:
                if self._answers_match(rep, float(siv_result.blueprint_answer)):
                    siv_full = 1
            except (TypeError, ValueError):
                pass
        executed = 1 if any(m.code for m in members) else 0
        return (self._honest_votes(members), siv_full, executed,
                1 if "primary" in ids else 0,
                1 if "baseline" in ids else 0)

    def _select_by_evidence(self, candidates: List[HypothesisResult],
                            siv_result: Optional[SIVResult]
                            ) -> Tuple[Optional[HypothesisResult], List[HypothesisResult], str]:
        """
        [v12.1] Arbitrate WITHOUT an LLM: rank answer groups by verifiable
        evidence and return a strict winner ONLY when the winning group is
        actually corroborated.

        [FIX v12.1] The v12.0 version resolved every group ranking as long as
        exactly one group scored highest — but in the single most common
        no-majority case (a lone primary vs a lone baseline, each in its own
        group of size 1, no SIV signal), the `executed` component of
        `_evidence_score` always favored the group containing the primary
        (it always has code; the baseline candidate never does). This is not
        evidence of correctness, just a structural artifact of who happens to
        run code — and it always resolved in favor of the primary regardless
        of whether the primary was actually right.
        Measured impact on the first full v12.0 run (n=150): of the 21
        problems that reached this exact primary-vs-baseline singleton tie,
        the primary was correct on 7 and the baseline was correct on a
        DIFFERENT 7 (a perfect, disjoint split — a coin flip). The old rule
        always picked the primary, so it recovered its own 7 while silently
        converting all 7 baseline-correct cases into regressions.
        We now only auto-resolve when the winning group carries genuine
        corroboration: either multiple independent derivations agree
        (votes >= 2 — reachable here only via a size-tied "no majority" case,
        e.g. 2-vs-2) or SIV fully verified this exact value. A lone,
        unverified singleton — whichever candidate it happens to be — is not
        evidence; it goes to the constrained judge instead, alongside every
        other unverified singleton (not just the ones that happened to tie
        on score), since none of them has more standing than any other.

        Returns (winner, tied_representatives, note):
          winner is None when no group is corroborated (or scores tie even
          after corroboration) — tied_representatives then holds one
          representative per remaining group for the LLM tie-break.
        """
        groups = self._group_valid_candidates(candidates)
        if not groups:
            return None, [], "no_valid_candidates"

        scored = sorted(
            ((self._evidence_score(rep, members, siv_result), rep, members)
             for rep, members in groups),
            key=lambda t: t[0], reverse=True,
        )

        def _representative(members: List[HypothesisResult]) -> HypothesisResult:
            rank = {"primary": 0, "baseline": 1, "blueprint_eval": 2}
            return sorted(members, key=lambda m: rank.get(m.hypothesis_id, 3))[0]

        note = "; ".join(
            f"{rep:.6g}: votes={s[0]} siv={s[1]} exec={s[2]} prim={s[3]} base={s[4]}"
            for s, rep, _ in scored[:4]
        )

        top_score = scored[0][0]
        top_votes = top_score[0]
        # [v13.0] Corroboration = honest votes >= 2, FULL STOP. siv_full no
        # longer qualifies a lone group on its own (measured: verified with
        # a disagreeing baseline was right 2/14 — SIV certifies blueprint-
        # internal consistency and inherits the translation error). It
        # still participates in the lexicographic score above, so between
        # two corroborated groups it breaks the tie deterministically.
        if top_votes >= 2:
            tied = [(rep, members) for score, rep, members in scored if score == top_score]
            if len(tied) == 1:
                return _representative(tied[0][1]), [], note
            return None, [_representative(members) for _, members in tied], note

        # [v13.0] No group is corroborated. Under the do-no-harm invariant
        # this is no longer the judge's problem: an uncorroborated field of
        # lone voices contains no signal a 7B judge has ever been measured
        # to extract (pilot free-form judge 15%; v12.1 constrained judge
        # parsed 0/55 attempts on the mixed run). The caller falls back to
        # the internal zero-shot baseline ('baseline_default').
        return None, [], note + "; uncorroborated -> baseline_default"

    def _deterministic_fallback(self, candidates: List[HypothesisResult],
                                siv_result: Optional[SIVResult]) -> HypothesisResult:
        """Last-resort pick when even the judge is unusable. Never scrapes
        free text. [v13.0] Preference is baseline → executed primary → any
        valid candidate → primary object regardless: the old primary-first
        order was the v12.1 coin-flip bug's twin living in the one function
        that fix never reached (it only fires when the judge fails, which
        never happened before the mixed run — where it then fired 55 times
        and picking the primary scored 8/55 against the baseline's 29/55)."""
        by_id = {c.hypothesis_id: c for c in candidates}
        primary = by_id.get("primary")
        baseline = by_id.get("baseline")
        if baseline is not None and baseline.parsed_answer is not None:
            return baseline
        if primary is not None and primary.code_success and primary.parsed_answer is not None:
            return primary
        for c in candidates:
            if c.code_success and c.parsed_answer is not None:
                return c
        return primary if primary is not None else candidates[0]

    def _judge_hypotheses(self, problem: str,
                          candidates: List[HypothesisResult]
                          ) -> Tuple[Optional[HypothesisResult], str]:
        """
        [v12.0] LLM judge as a CONSTRAINED tie-breaker.

        The judge sees only candidates that already tied on deterministic
        evidence and must reply with the INDEX of one of them
        (SELECTED_CANDIDATE: [[k]]). A free-form SELECTED_ANSWER is accepted
        only when it numerically matches one of the presented candidates.
        Anything else returns (None, reasoning) and the caller falls back to
        deterministic selection — the judge can no longer inject numbers that
        no solver produced (the pilot's free-form judge did exactly that,
        scoring 15% on 40 problems against its own baseline's 42.5%).
        """
        judgeable = [c for c in candidates if c.parsed_answer is not None]
        if not judgeable:
            return None, "no judgeable candidates"
        if len(judgeable) == 1:
            return judgeable[0], "single judgeable candidate"

        candidate_summaries = []
        for i, c in enumerate(judgeable):
            summary = f"""--- Candidate {i+1}: {c.strategy_name} ({c.hypothesis_id}) ---
Answer: {c.answer}
Equations: {json.dumps(c.blueprint.get('equations', []))}
Code (first 300 chars): {(c.code or 'N/A')[:300]}
"""
            candidate_summaries.append(summary)

        sys_msg = f"""You are a mathematical reasoning Judge. Several solution attempts for the same problem produced DIFFERENT answers. Exactly one candidate must be selected.

Evaluation criteria (in order of importance):
1. FAITHFULNESS: Does the candidate use the problem's numbers and conditions correctly?
2. MATHEMATICAL CORRECTNESS: Are the equations and logic sound?
3. COMPLETENESS: Does the approach account for ALL conditions in the problem?

OUTPUT FORMAT (strict):
The VERY FIRST line of your reply must be EXACTLY: SELECTED_CANDIDATE: [[k]]
where k is the number (1 to {len(judgeable)}) of the candidate you select.
After that line you may add a brief justification (2-3 sentences).
Do NOT invent a new answer. You MUST pick one of the presented candidates."""

        user_msg = f"""PROBLEM:
{problem}

CANDIDATES:
{''.join(candidate_summaries)}

Select the most reliable candidate."""

        msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]

        raw = self._get_client(AgentRole.JUDGE).call_model(msgs, temperature=0.0, max_tokens=500)

        # [FIX v7.1] Check for error response from judge
        if _is_error_response(raw):
            logger.warning("Judge returned error response")
            return None, "Judge API call failed"

        raw_text = str(raw)

        idx_match = re.search(r'SELECTED_CANDIDATE:\s*\[\[?\s*(\d+)\s*\]?\]', raw_text)
        if idx_match:
            k = int(idx_match.group(1))
            if 1 <= k <= len(judgeable):
                return judgeable[k - 1], raw_text[:500]

        # Tolerate the legacy tag, but ONLY if it names an existing candidate.
        ans_match = re.search(r'SELECTED_ANSWER:\s*\[\[([^\]]+)\]\]', raw_text)
        if ans_match:
            num = _extract_last_number(ans_match.group(1))
            for c in judgeable:
                if self._answers_match(num, c.parsed_answer):
                    return c, raw_text[:500]

        logger.warning("Judge output unparseable/invalid — deterministic fallback will be used")
        return None, raw_text[:500]

    def _structured_hypothesis_testing(self, problem: str, expected: str,
                                       primary_blueprint: dict,
                                       programmer_response: AgentResponse,
                                       baseline_answer: str,
                                       siv_result: Optional[SIVResult] = None,
                                       pre_sht_calls: int = 3) -> HypothesisLog:
        """[v10.0] Enhanced with SIV: can skip SHT if proven, or pass error info to Critic.

        [v13.0] pre_sht_calls: actual LLM calls already spent before this
        function runs (normally 3 — baseline, Mathematician, Programmer).
        The caller bumps this by 2 when a blueprint-repair attempt fired
        (see _attempt_blueprint_repair), so api_calls_used stays an honest
        total instead of silently under-counting repaired problems.
        """
        primary_answer = programmer_response.answer
        primary_num = _extract_last_number(primary_answer)

        log = HypothesisLog(
            problem=problem,
            expected=expected,
            final_answer=primary_answer,
            final_strategy="primary",
        )

        # [v12.0] Validity from ACTUAL execution success, not from
        # verification-adjusted confidence: the old `confidence > 0.5` check
        # silently dropped the primary answer from arbitration whenever
        # rule-based verification disagreed with the blueprint (floor 0.3) —
        # exactly the disagreement cases where arbitration matters most.
        primary_exec_ok = (
            str(primary_answer).strip().lower() != "unknown"
            and primary_num is not None
            and (bool(programmer_response.quality_metrics.get("execution_output"))
                 or str(programmer_response.agent).startswith("SymPy"))
        )
        primary_candidate = HypothesisResult(
            hypothesis_id="primary",
            strategy_name="primary_blueprint",
            blueprint=primary_blueprint,
            code=programmer_response.reasoning_trace,
            code_success=primary_exec_ok,
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

        # [v12.0] Blueprint-evaluation candidate: the CAS already evaluated the
        # Architect's equations during the SIV execution audit — that numeric
        # value is a third INDEPENDENT derivation (free program / declarative
        # blueprint / zero-shot CoT) and it costs zero LLM calls. Majority and
        # evidence voting over three independent paths is meaningful where
        # two-way primary-vs-baseline disagreement was not.
        if (self.enable_siv and siv_result is not None
                and siv_result.blueprint_answer is not None):
            bp_num = _extract_last_number(str(siv_result.blueprint_answer))
            if bp_num is not None:
                log.candidates.append(HypothesisResult(
                    hypothesis_id="blueprint_eval",
                    strategy_name="architect_blueprint_cas",
                    blueprint=primary_blueprint,
                    code=None,
                    code_success=True,
                    execution_output=f"CAS evaluation of blueprint equations = {bp_num}",
                    answer=str(bp_num),
                    parsed_answer=bp_num,
                    confidence=siv_result.confidence if siv_result.verified else 0.6,
                    agent_response=None,
                ))

        is_confident, gate_reason = self._confidence_gate(
            primary_answer, baseline_answer, programmer_response, primary_blueprint,
            siv_result=siv_result  # [v10.0]
        )

        if is_confident:
            log.triage_result = "confident_skip"
            log.final_answer = primary_answer
            log.final_strategy = "primary_blueprint"
            log.hypothesis_testing_triggered = False
            log.api_calls_used = pre_sht_calls
            return log

        logger.info(f"SHT triggered: {gate_reason}")
        log.hypothesis_testing_triggered = True
        api_calls = pre_sht_calls

        # [FIX v7.2] Check if we can afford SHT calls (~4 more calls × 1500 tokens)
        sht_cost_estimate = 4 * 1500  # hypothesis gen + 2 programmer + maybe judge
        if not token_budget.can_afford(sht_cost_estimate):
            logger.warning("SHT skipped due to token budget. Using best available answer.")
            # Fall back to whichever of primary/baseline looks better
            if primary_num is not None:
                log.final_answer = primary_answer
                log.final_strategy = "primary_budget_skip"
            else:
                log.final_answer = baseline_answer
                log.final_strategy = "baseline_budget_skip"
            log.triage_result = "budget_skip"
            log.api_calls_used = pre_sht_calls
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

        # [v12.0] Deterministic evidence-based arbitration BEFORE any LLM
        # judgment: rank answer groups by verifiable evidence (votes, SIV
        # verification, executed code, primary/baseline support). A strict
        # winner is selected with zero additional LLM calls.
        winner, tied_reps, evidence_note = self._select_by_evidence(
            log.candidates, siv_result
        )
        if winner is not None:
            log.triage_result = "evidence"
            log.judge_reasoning = f"evidence ranking: {evidence_note}"[:500]
            log.final_answer = winner.answer
            log.final_strategy = winner.strategy_name
            log.api_calls_used = api_calls
            return log

        # [v13.0] LLM judge ONLY between corroborated groups that tied on the
        # full evidence score (rare). On parse failure the deterministic
        # fallback decides. An empty tied_reps means NOTHING was corroborated
        # — do-no-harm applies: the answer is the internal zero-shot baseline,
        # with no judge call at all (the judge has never been measured to
        # beat the baseline on uncorroborated sets: pilot 15%, constrained
        # v12.1 parsed 0/55 on the mixed run).
        if tied_reps:
            judge_pick, judge_reasoning = self._judge_hypotheses(problem, tied_reps)
            api_calls += 1

            log.judge_reasoning = f"[{evidence_note}] {judge_reasoning}"[:500]
            if judge_pick is not None:
                log.triage_result = "judge"
                log.final_answer = judge_pick.answer
                log.final_strategy = judge_pick.strategy_name
            else:
                fallback = self._deterministic_fallback(log.candidates, siv_result)
                log.triage_result = "judge_fallback"
                log.final_answer = fallback.answer
                log.final_strategy = f"{fallback.strategy_name}_fallback"
            log.api_calls_used = api_calls
            return log

        log.judge_reasoning = f"[{evidence_note}]"[:500]
        baseline_c = next(
            (c for c in log.candidates
             if c.hypothesis_id == "baseline" and c.parsed_answer is not None),
            None,
        )
        if baseline_c is not None:
            log.triage_result = "baseline_default"
            log.final_answer = baseline_c.answer
            log.final_strategy = "baseline_do_no_harm"
        else:
            # Baseline itself unusable — the invariant has no anchor; fall
            # back deterministically (executed primary, then anything).
            fallback = self._deterministic_fallback(log.candidates, siv_result)
            log.triage_result = "fallback"
            log.final_answer = fallback.answer
            log.final_strategy = f"{fallback.strategy_name}_fallback"
        log.api_calls_used = api_calls
        return log

    # -------------------------------------------------------------------------
    # Main Solve Method
    # -------------------------------------------------------------------------

    def solve(self, problem: str, expected: str) -> Dict[str, Any]:
        # Step 1: Baseline
        # [v10.4] Baseline prompt redesign:
        #   - System role spells out the CoT contract so the model doesn't
        #     waste output budget restating it.
        #   - max_tokens raised 500→1024. Hard / GSM-Symbolic-P2 problems
        #     routinely need 700-900 tokens of reasoning before the final
        #     number; the old 500-token cap was truncating CoT mid-stream,
        #     which is what produced the implausible 5.4% baseline number
        #     in the v9.1 paper run.
        #   - "Show your reasoning, then on the LAST line write …" makes the
        #     ANSWER tag positionally deterministic so the extractor wins on
        #     truncated outputs too.
        baseline_sys = (
            "You are a careful problem solver. "
            "Work through the problem step by step, showing arithmetic. "
            "On the LAST line, output EXACTLY one line of the form: "
            "ANSWER: [[<single numeric value, no units, no commas>]]"
        )
        baseline_user = (
            f"{problem}\n\n"
            "Solve step by step. Do not skip arithmetic. "
            "End with the ANSWER line as instructed."
        )
        base_raw = self._get_client(AgentRole.BASELINE).call_model(
            [
                {"role": "system", "content": baseline_sys},
                {"role": "user",   "content": baseline_user},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        base_ans, _ = self.extract_answer(base_raw)

        # Step 2: Architect
        blackboard_logic = self.run_mathematician_analysis(problem)

        # Step 3: Engineer (with SymPy fallback built-in)
        programmer_response = self.run_programmer_solver(problem, blackboard_logic)

        # Step 3b: Process-Level Verification + SIV
        verification_passed = True
        verification_feedback = "Skipped"
        verification_confidence = 1.0
        siv_result = None  # [v10.0]
        blueprint_repaired = False  # [v13.0]

        if (programmer_response.confidence > 0.5
            and programmer_response.answer != "unknown"
            and programmer_response.reasoning_trace
            and "SymPy" not in programmer_response.agent):
            # Rule-based verification
            verification_passed, verification_feedback, verification_confidence = \
                self.verify_code_against_blueprint(
                    problem, blackboard_logic,
                    programmer_response.reasoning_trace,
                    programmer_response.answer
                )
            
            # [v10.0] Run Symbolic Execution Audit (SIV)
            # SIV operates on the math→math layer: it checks whether the Programmer's
            # numeric answer is consistent with the Architect's blueprint equations.
            # NOTE: SIV cannot detect NL→math translation errors (wrong blueprint).
            # [v10.2] enable_siv flag controls whether SIV runs at all (B5 ablation).
            answer_num = _extract_last_number(programmer_response.answer)
            if self.enable_siv and SYMPY_AVAILABLE and answer_num is not None and blackboard_logic.get("equations"):
                logger.info("Running SIV Symbolic Execution Audit...")
                siv_result = SymbolicInverseVerifier.verify(blackboard_logic, answer_num)
                if siv_result.execution_audit_passed and siv_result.verified:
                    logger.info(
                        f"SIV EXECUTION CONSISTENT: {siv_result.givens_matched}/"
                        f"{siv_result.givens_total} givens localized "
                        f"(conf={siv_result.confidence:.2f}). "
                        f"Unused givens: {siv_result.unused_givens}. "
                        f"Translation-layer correctness not verified."
                    )
                    if siv_result.confidence >= 0.90:
                        verification_passed = True
                        verification_confidence = siv_result.confidence
                        verification_feedback = (
                            f"SIV_EXEC_AUDIT_PASS ({siv_result.givens_matched}/"
                            f"{siv_result.givens_total} givens; "
                            f"unused={siv_result.unused_givens})"
                        )
                elif not siv_result.execution_audit_passed:
                    logger.info(
                        f"SIV EXECUTION ERROR: blueprint_answer={siv_result.blueprint_answer}, "
                        f"computed={answer_num}, rel_err={siv_result.execution_rel_error}"
                    )
                    # [v13.0] One repair attempt targeting exactly this signal
                    # (measured 80.3% recall / 0% false-alarm rate as a
                    # wrong-blueprint detector — see _attempt_blueprint_repair).
                    # Replaces blackboard_logic/programmer_response/siv_result
                    # in place so the confidence gate and SHT below see the
                    # repaired attempt, not the one that just failed audit.
                    blackboard_logic, programmer_response, siv_result, blueprint_repaired = (
                        self._attempt_blueprint_repair(
                            problem, blackboard_logic, programmer_response, siv_result
                        )
                    )
                    if blueprint_repaired:
                        answer_num = _extract_last_number(programmer_response.answer)
                        if siv_result is not None and siv_result.execution_audit_passed and siv_result.verified:
                            logger.info(
                                f"SIV EXECUTION CONSISTENT (post-repair): "
                                f"{siv_result.givens_matched}/{siv_result.givens_total} givens localized "
                                f"(conf={siv_result.confidence:.2f})"
                            )
                            # [v14.0] A tautological pass earns no confidence: the
                            # audited answer was produced by these very equations.
                            if siv_result.confidence >= 0.90 and not siv_result.tautological:
                                verification_passed = True
                                verification_confidence = siv_result.confidence
                                verification_feedback = (
                                    f"SIV_EXEC_AUDIT_PASS_POST_REPAIR ({siv_result.givens_matched}/"
                                    f"{siv_result.givens_total} givens)"
                                )
                        elif siv_result is not None and not siv_result.execution_audit_passed:
                            logger.info(
                                f"SIV EXECUTION ERROR (post-repair, unresolved): "
                                f"blueprint_answer={siv_result.blueprint_answer}, computed={answer_num}"
                            )
                        elif siv_result is not None:
                            logger.info(f"SIV LOCALIZATION FAILED (post-repair): {siv_result.failed_givens}")
                else:
                    logger.info(f"SIV LOCALIZATION FAILED: {siv_result.failed_givens}")

            # Adjust programmer confidence
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
                    "siv_verified": siv_result.verified if siv_result else None,
                    "siv_confidence": siv_result.confidence if siv_result else None,
                    "siv_execution_audit_passed": siv_result.execution_audit_passed if siv_result else None,
                    "siv_failed_givens": siv_result.failed_givens if siv_result else [],
                    "siv_unused_givens": siv_result.unused_givens if siv_result else [],
                }
            )
            
            # [v12.0] The SymPy post-verification REPLACEMENT is gone. In the
            # n=150 pilot the "SymPy (post-verification fallback)" answers were
            # 33% accurate while the Programmer answers they overwrote were 50%
            # accurate — replacing a working executed program because the
            # rule-based checker disagreed with the (possibly wrong) blueprint
            # was a net loss. The blueprint's CAS evaluation still participates
            # in arbitration as the 'blueprint_eval' SHT candidate, and the
            # verification verdict still lowers confidence / lands in the CSV;
            # it just no longer overwrites the primary answer pre-arbitration.
            if not verification_passed:
                logger.info(
                    f"Process verification FAILED (conf={verification_confidence:.2f}). "
                    f"Keeping Programmer answer; blueprint CAS value arbitrates in SHT."
                )

        # Step 4: Structured Hypothesis Testing (with SIV integration)
        hypothesis_log = None
        if self.enable_hypothesis_testing:
            hypothesis_log = self._structured_hypothesis_testing(
                problem, expected, blackboard_logic,
                programmer_response, base_ans,
                siv_result=siv_result,  # [v10.0]
                # [v13.0] +2 real calls (Mathematician + Programmer) already
                # spent above if a blueprint-repair attempt fired.
                pre_sht_calls=3 + (2 if blueprint_repaired else 0),
            )
            mas_answer = hypothesis_log.final_answer
            used_baseline_fallback = False

            # [v11.4] SIV anchor: when SIV fully audited the original blueprint
            # (execution_audit_passed + all givens verified + rel_error ≈ 0),
            # its CAS-computed answer is more reliable than SHT majority vote.
            # SHT candidates come from *alternative* blueprints that may diverge;
            # the SIV-anchored answer from the primary blueprint should win.
            # Guard: only override when SHT actually changed the answer (avoids
            # no-op writes) and SIV rel_error is essentially zero (< 0.1%).
            # [v12.0][CRITICAL FIX] `(rel_error or 1.0) < 1e-3` never fired on a
            # PERFECT audit: 0.0 is falsy, so `0.0 or 1.0` → 1.0 and the anchor
            # was skipped exactly when the evidence was strongest. Pilot proof:
            # gsm-hard_1125 (rel_err=5.5e-4) anchored → correct; gsm-hard_542
            # (rel_err=0.0, blueprint answer == gold) NOT anchored → judge
            # emitted "3". Explicit None check below.
            # [v13.0] Do-no-harm applies HERE too. A fully self-consistent
            # blueprint (audit passed, all givens reconstruct, rel_err≈0)
            # still only certifies that the equations are internally
            # coherent, not that they faithfully translate the problem —
            # exactly the scope limit the confidence gate's own criterion 7
            # already respects (SIV-verified skip only when baseline agrees).
            # Measured on the mixed run: siv_verified WITH baseline agreement
            # = 22/22 correct; siv_verified WITH baseline disagreement =
            # 2/14. Anchoring on this signal is safe only when the baseline
            # is not itself a contradicting, independent vote.
            _siv_rel_err = (siv_result.execution_rel_error
                            if siv_result is not None else None)
            if (siv_result is not None
                    and siv_result.execution_audit_passed
                    and siv_result.verified
                    and siv_result.blueprint_answer is not None
                    and _siv_rel_err is not None
                    and _siv_rel_err < 1e-3):
                _siv_str = str(siv_result.blueprint_answer)
                _siv_num = _extract_last_number(_siv_str)
                _mas_num = _extract_last_number(str(mas_answer))
                _base_num = _extract_last_number(str(base_ans))
                _base_failed = str(base_ans).strip().lower() == "unknown"
                _baseline_disagrees = (
                    not _base_failed and _base_num is not None and _siv_num is not None
                    and abs(_base_num - _siv_num) > 1e-3
                )
                if _siv_num is not None and _mas_num is not None and abs(_siv_num - _mas_num) > 1e-3:
                    if not _baseline_disagrees:
                        logger.info(
                            f"[v11.4] SIV-anchor override: SHT={mas_answer!r} → "
                            f"SIV={_siv_str!r} "
                            f"(audit_passed, all_givens_verified, "
                            f"rel_err={siv_result.execution_rel_error:.2e})"
                        )
                        mas_answer = _siv_str
                        if hypothesis_log is not None:
                            hypothesis_log.final_answer = mas_answer
                    else:
                        logger.info(
                            f"[v13.0] SIV-anchor SUPPRESSED: fully-verified blueprint="
                            f"{_siv_str!r} disagrees with baseline={base_ans!r} — "
                            f"do-no-harm invariant blocks the override "
                            f"(keeping {mas_answer!r})"
                        )
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
                "local_hf_fallback": bool(blackboard_logic.get("_local_hf_fallback", False)),  # [v10.2]
                "extracted_from_cot": bool(blackboard_logic.get("_extracted_from_cot", False)),  # [v11.0]
                "blueprint_repaired": bool(blackboard_logic.get("_blueprint_repaired", False)),  # [v13.0]
                # [v14.0] Blueprint content + provenance. Until now NOTHING about the
                # blueprint itself was recorded, so blueprint quality could only be
                # inferred from siv_blueprint_answer and SIV could not be replayed
                # offline on a finished run. These three fields are what make
                # "did this change improve the blueprints?" a measurable question.
                "blueprint_givens": json.dumps(
                    blackboard_logic.get("givens", {}), ensure_ascii=False)[:1000],
                "blueprint_equations": json.dumps(
                    blackboard_logic.get("equations", []), ensure_ascii=False)[:1500],
                "blueprint_provenance": (
                    "tautological" if blackboard_logic.get("_local_hf_fallback")
                    else "extracted_from_cot" if blackboard_logic.get("_extracted_from_cot")
                    else "primary_json"
                ),
                "programmer_metrics": programmer_response.quality_metrics,
                "verification": {
                    "passed": verification_passed,
                    "confidence": verification_confidence,
                    "feedback": verification_feedback[:200],
                },
            },
            "agents": [programmer_response],
            "model_config": self.get_model_config_summary(),
            # [v10.0] SIV metrics (two-layer execution audit + fault localization)
            "siv": {
                # Layer 1: Execution audit
                "execution_audit_passed": siv_result.execution_audit_passed if siv_result else None,
                "blueprint_answer": siv_result.blueprint_answer if siv_result else None,
                "execution_rel_error": siv_result.execution_rel_error if siv_result else None,
                # Layer 2: Fault localization
                "verified": siv_result.verified if siv_result else None,
                "confidence": siv_result.confidence if siv_result else None,
                "givens_matched": siv_result.givens_matched if siv_result else None,
                "givens_total": siv_result.givens_total if siv_result else None,
                "invertible": siv_result.invertible if siv_result else None,
                "failed_givens": siv_result.failed_givens if siv_result else [],
                "unused_givens": siv_result.unused_givens if siv_result else [],
                # Layer 0: structural defects [v14.0]
                "undefined_symbols": siv_result.undefined_symbols if siv_result else [],
                "undeclared_given_keys": siv_result.undeclared_given_keys if siv_result else [],
                # Meta
                "verifies_translation": False,  # Explicit limitation — always False
                "skipped": siv_result.skipped if siv_result else None,          # [v14.0]
                "tautological": siv_result.tautological if siv_result else None,  # [v14.0]
                "trace": siv_result.trace[:400] if siv_result else "",
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


# --------------------------- Main Pipeline ---------------------------

class QualityAwarePipeline:
    def __init__(self, provider: str = "groq", use_cache: bool = False,
                 heterogeneous_preset: Optional[str] = None,
                 custom_config: Optional[Dict[AgentRole, ModelConfig]] = None,
                 enable_siv: bool = True,
                 enable_sht: bool = True,
                 evaluation_mode: bool = False,
                 dataset_seed: Optional[int] = None,
                 math_reasoning_first: bool = True):
        """
        [UPDATED v10.4] Supports heterogeneous model configuration + ablation flags
        + evaluation_mode safety net.

        Args:
            provider: Default provider (used if no heterogeneous config).
            use_cache: Whether to cache API calls. SHOULD BE False during evaluation
                (cache hits make per-problem outputs depend on prior call order,
                violating apples-to-apples comparison across systems).
            heterogeneous_preset: Name of a preset from HETEROGENEOUS_PRESETS.
            custom_config: Custom Dict[AgentRole, ModelConfig] mapping.
            enable_siv: [v10.2] If False, the Symbolic Inverse Verifier never
                runs. Used by baselines.py B5 (MAS-NoSIV). The confidence gate
                still operates on baseline-vs-MAS disagreement; only the
                SIV-derived signal is missing.
            enable_sht: [v10.2] If False, Structured Hypothesis Testing never
                runs — the pipeline returns the Engineer's first answer as-is.
                Used by baselines.py B6 (MAS-NoSHT).
            evaluation_mode: [v10.4] If True:
                - Forces use_cache=False (with a loud warning if caller passed True),
                  so reported accuracy/cost numbers reflect the model, not the cache.
                - Logs the active config block at INFO so it lands in run logs.
                Set this for any reported / publishable run.
            dataset_seed: [v10.4] Seed for the problem sampler AND the distractor
                injector. Pass an int (e.g., 42) for reproducibility; pass a list
                of seeds to the experiment runner to do multi-seed evaluation
                (required for any error-bar story on accuracy deltas).
                None → non-deterministic (fine for debugging, NOT for reports).
        """
        # [v10.4] Evaluation safety net — refuse to silently cache.
        if evaluation_mode and use_cache:
            logger.warning(
                "evaluation_mode=True + use_cache=True is unsafe for reported "
                "results (cache hits make outputs depend on prior call order). "
                "Forcing use_cache=False."
            )
            use_cache = False
        self.evaluation_mode = evaluation_mode
        self.dataset_seed = dataset_seed
        # [v10.4] Pass the seed into the sampler. EnhancedProblemManager seeds
        # the module-global random state at construction; the distractor
        # injector and dataset sampler both draw from it.
        self.manager = EnhancedProblemManager(random_seed=dataset_seed)
        self.results: List[Dict[str, Any]] = []
        
        # Determine model configuration
        if custom_config:
            role_config = custom_config
        elif heterogeneous_preset and heterogeneous_preset in HETEROGENEOUS_PRESETS:
            role_config = HETEROGENEOUS_PRESETS[heterogeneous_preset]
        else:
            # Backward compatible: single provider for all roles
            role_config = {
                role: ModelConfig(provider, None)
                for role in AgentRole
            }
        
        # [v7.3] Build one LLMClient per unique (provider, model_name) pair
        # This avoids creating duplicate clients for the same model
        self._client_cache: Dict[str, UnifiedLLMClient] = {}
        clients: Dict[AgentRole, UnifiedLLMClient] = {}
        
        for role, mc in role_config.items():
            # [v10.3] Include load_4bit in cache key so 4-bit and fp16 variants
            # of the same model don't share a single client instance.
            cache_key = f"{mc.provider}:{mc.model_name or 'default'}:4bit={getattr(mc, 'load_4bit', False)}"
            if cache_key not in self._client_cache:
                self._client_cache[cache_key] = UnifiedLLMClient(
                    provider=mc.provider,
                    use_cache=use_cache,
                    model_override=mc.model_name,
                    load_4bit=getattr(mc, 'load_4bit', False),  # [v10.3]
                )
            clients[role] = self._client_cache[cache_key]
        
        # Store primary client for validation
        self.client = clients[AgentRole.MATHEMATICIAN]  # Use mathematician for validation
        
        # Create solver with heterogeneous clients
        self.solver = QualityEnhancedMultiAgentSolver(clients=clients)
        # [v10.2] Apply ablation flags from constructor.
        self.solver.enable_siv = enable_siv
        self.solver.enable_hypothesis_testing = enable_sht
        # [v14.0] Blueprint reasoning mode — see QualityEnhancedMultiAgentSolver
        # and pretest_blueprint_mode.py, which measures the two settings against
        # each other on a handful of problems before a full run commits to one.
        self.solver.math_reasoning_first = math_reasoning_first
        # Persist for logging/CSV.
        self.enable_siv = enable_siv
        self.enable_sht = enable_sht
        if not (enable_siv and enable_sht):
            logger.info(
                f"[v10.2 ablation] enable_siv={enable_siv}, enable_sht={enable_sht}"
            )
        logger.info(
            f"[v14.0] blueprint mode: "
            f"{'reasoning-first (DERIVATION -> BLUEPRINT)' if math_reasoning_first else 'json-only (v13.0)'}"
        )

    def _extract_gold_answer(self, text: Any) -> Optional[float]:
        text = str(text)
        if "####" in text:
            raw_gold = text.split("####")[-1].strip()
        else:
            raw_gold = text
        nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", raw_gold)
        if nums:
            try:
                return float(nums[-1].replace(",", ""))
            except ValueError:
                return None
        return None

    def check_correctness(self, pred: Any, gold_text: Any) -> bool:
        gold_val = self._extract_gold_answer(gold_text)
        if gold_val is None:
            return str(pred).strip() == str(gold_text).strip()
        try:
            pred_str = str(pred).replace("$", "").replace(",", "")
            pred_nums = re.findall(r"-?\d+(?:\.\d+)?", pred_str)
            if not pred_nums:
                return False
            return abs(float(pred_nums[-1]) - gold_val) < 1e-3
        except:
            return False

    def run(self, datasets_list=["gsm8k_test"], num_problems=10, hardener: Optional[str] = None) -> pd.DataFrame:
        
        # [FIX v7.1] Validate API connection before running
        if not self.client.validate_connection():
            logger.error("="*60)
            logger.error("FATAL: API connection failed! Cannot proceed.")
            logger.error("Please check:")
            logger.error("  1. Your .env file has the correct API key")
            logger.error("  2. The API key is not expired")
            logger.error("  3. You have sufficient API credits")
            logger.error("  4. The API service is not down")
            logger.error("="*60)
            raise ConnectionError(
                "API connection validation failed. Check your API key and .env file. "
                "All results would be errors (like the 401.0 issue)."
            )
        
        logger.info(f"Pipeline started. Fetching {num_problems} random problems from: {datasets_list} | hardener={hardener}")
        problems = self.manager.load_random_problems(datasets_list, num_problems, hardener=hardener)

        detailed = []
        # [FIX v7.1] Track consecutive API errors to detect persistent failures
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 5
        
        for i, p in enumerate(problems):
            # [FIX v9.1] Inter-problem cooldown to avoid TPM bursts
            if i > 0:
                cooldown = 3.0 if self.solver.enable_hypothesis_testing else 2.0
                logger.info(f"Cooldown {cooldown:.0f}s between problems...")
                time.sleep(cooldown)

            logger.info(f"Processing {i+1}/{len(problems)} (ID: {p['id']}, DS: {p['dataset']}) | {token_budget.usage_report()}")

            # [FIX v7.2] Check token budget before each problem
            tokens_per_problem = 9000 if self.solver.enable_hypothesis_testing else 4500
            if not token_budget.can_afford(tokens_per_problem):
                logger.error(
                    f"TOKEN BUDGET EXHAUSTED after {i} problems. "
                    f"{token_budget.usage_report()}. "
                    f"Stopping early to avoid silent failures."
                )
                break
            
            res = self.solver.solve(p["puzzle"], p["answer"])
            res["baseline"]["correct"] = self.check_correctness(res["baseline"]["answer"], p["answer"])
            res["mas"]["correct"] = self.check_correctness(res["mas"]["answer"], p["answer"])
            res["id"] = p["id"]
            res["dataset"] = p["dataset"]
            detailed.append(res)
            
            # [FIX v7.1] Check for persistent API failures
            if res["baseline"]["answer"] == "unknown" and res["mas"]["answer"] == "unknown":
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.error(f"ABORTING: {MAX_CONSECUTIVE_ERRORS} consecutive problems returned 'unknown'. "
                                 "API is likely down or key is invalid.")
                    break
            else:
                consecutive_errors = 0

        self.results = detailed
        sht_data = []
        for r in detailed:
            sht = r.get("sht", {})
            sht_data.append({
                "sht_triggered": sht.get("triggered", False),
                "sht_triage_result": sht.get("triage_result", "n/a"),
                "sht_winning_strategy": sht.get("final_strategy", "n/a"),
                "sht_num_candidates": sht.get("num_candidates", 0),
                "sht_api_calls": sht.get("api_calls_used", 3),
            })

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
                # [v8.0] Verification metrics
                "verification_passed": r.get("mas", {}).get("verification", {}).get("passed", True),
                "verification_confidence": r.get("mas", {}).get("verification", {}).get("confidence", 1.0),
                # [v8.0] Solver type (Programmer, SymPy, baseline fallback)
                "solver_agent": r.get("agents", [{}])[0].agent if r.get("agents") else "unknown",
                # [v10.0] SIV metrics — two-layer execution audit + fault localization
                # Layer 1
                "siv_execution_audit_passed": r.get("siv", {}).get("execution_audit_passed", None),
                "siv_blueprint_answer": r.get("siv", {}).get("blueprint_answer", None),
                "siv_execution_rel_error": r.get("siv", {}).get("execution_rel_error", None),
                # Layer 2
                "siv_verified": r.get("siv", {}).get("verified", None),
                "siv_confidence": r.get("siv", {}).get("confidence", None),
                "siv_givens_matched": r.get("siv", {}).get("givens_matched", None),
                "siv_givens_total": r.get("siv", {}).get("givens_total", None),
                "siv_invertible": r.get("siv", {}).get("invertible", None),
                "siv_failed_givens": str(r.get("siv", {}).get("failed_givens", [])),
                "siv_unused_givens": str(r.get("siv", {}).get("unused_givens", [])),
                # Layer 0 [v14.0]
                "siv_undefined_symbols": str(r.get("siv", {}).get("undefined_symbols", [])),
                "siv_undeclared_given_keys": str(r.get("siv", {}).get("undeclared_given_keys", [])),
                # Meta
                "siv_verifies_translation": False,  # Explicit limitation marker
                "siv_skipped": r.get("siv", {}).get("skipped", None),
                "siv_tautological": r.get("siv", {}).get("tautological", None),
                # [v14.0] blueprint provenance — see solve()'s result dict
                "blueprint_provenance": r.get("mas", {}).get("blueprint_provenance", ""),
                "blueprint_givens": r.get("mas", {}).get("blueprint_givens", ""),
                "blueprint_equations": r.get("mas", {}).get("blueprint_equations", ""),
                **sht_data[i],
                **r.get("model_config", {}),
            } for i, r in enumerate(detailed)
        ])
        return df

    def report(self):
        if not self.results:
            return
        rows = []
        for r in self.results:
            sht = r.get("sht", {})
            rows.append({
                "base": 1 if r["baseline"]["correct"] else 0,
                "mas": 1 if r["mas"]["correct"] else 0,
                "mas_fallback": 1 if r["mas"].get("used_baseline_fallback", False) else 0,
                "sht_triggered": 1 if sht.get("triggered", False) else 0,
                "sht_triage": sht.get("triage_result", "n/a"),
            })
        df = pd.DataFrame(rows)
        n = len(df)
        base_acc = df["base"].mean()
        mas_acc = df["mas"].mean()
        fb_rate = df["mas_fallback"].mean()
        sht_trigger_rate = df["sht_triggered"].mean()

        rescue_count = 0
        damage_count = 0
        for r in self.results:
            sht = r.get("sht", {})
            if not sht.get("triggered", False):
                continue
            primary_candidates = [c for c in sht.get("candidates", []) if c["id"] == "primary"]
            if primary_candidates:
                primary_ans = primary_candidates[0]["answer"]
                primary_correct = self.check_correctness(primary_ans, r["expected"])
                mas_correct = r["mas"]["correct"]
                if mas_correct and not primary_correct:
                    rescue_count += 1
                elif not mas_correct and primary_correct:
                    damage_count += 1

        sht_triggered_total = int(df["sht_triggered"].sum())

        print("\n" + "="*60)
        print("   PERFORMANCE REPORT (MAS + Structured Hypothesis Testing)")
        print("="*60)
        
        # [v7.3] Show model configuration
        if self.results and "model_config" in self.results[0]:
            mc = self.results[0]["model_config"]
            is_homogeneous = len(set(mc.values())) == 1
            if is_homogeneous:
                print(f"Model Config: Homogeneous ({list(mc.values())[0]})")
            else:
                print("Model Config: HETEROGENEOUS")
                for role_key, model_str in mc.items():
                    print(f"  {role_key:<25} → {model_str}")
            print("-" * 60)
        
        print(f"Total Examples: {n}")
        print("-" * 60)
        print(f"{'Metric':<30} | {'Value':<10}")
        print("-" * 60)
        print(f"{'Baseline Accuracy':<30} | {base_acc:.2%}")
        print(f"{'MAS+SHT Accuracy':<30} | {mas_acc:.2%}")
        print(f"{'Improvement over Baseline':<30} | {(mas_acc - base_acc):+.2%}")
        print(f"{'MAS->Baseline Fallback':<30} | {fb_rate:.2%}")
        print("-" * 60)
        
        # [v8.0] Verification and SymPy stats
        verif_failed = sum(
            1 for r in self.results
            if not r.get("mas", {}).get("verification", {}).get("passed", True)
        )
        sympy_used = sum(
            1 for r in self.results
            if r.get("agents") and "SymPy" in str(r["agents"][0].agent)
        )
        print(f"{'Verification Failures':<30} | {verif_failed}/{n}")
        print(f"{'SymPy Fallback Used':<30} | {sympy_used}/{n}")
        # [v10.0] SIV statistics
        siv_verified_count = sum(1 for r in self.results if r.get("siv", {}).get("verified", False))
        siv_invertible_count = sum(1 for r in self.results if r.get("siv", {}).get("invertible", False))
        siv_failed_count = siv_invertible_count - siv_verified_count
        siv_skipped_sht = sum(
            1 for r in self.results
            if r.get("siv", {}).get("verified", False)
            and not r.get("sht", {}).get("triggered", False)
        )
        print("-" * 60)
        print("SYMBOLIC INVERSE VERIFICATION (SIV) — Novel v10.0:")
        print(f"{'SIV Invertible Chains':<30} | {siv_invertible_count}/{n}")
        print(f"{'SIV Verified (proven correct)':<30} | {siv_verified_count}/{n}")
        print(f"{'SIV Detected Errors':<30} | {siv_failed_count}/{n}")
        print(f"{'SIV Skipped SHT (saved calls)':<30} | {siv_skipped_sht}/{n}")
        print("-" * 60)
        print(f"{'SHT Trigger Rate':<30} | {sht_trigger_rate:.2%} ({sht_triggered_total}/{n})")
        print(f"{'SHT Rescue (fixed wrong)':<30} | {rescue_count}")
        print(f"{'SHT Damage (broke correct)':<30} | {damage_count}")
        if sht_triggered_total > 0:
            print(f"{'SHT Net Benefit':<30} | {rescue_count - damage_count:+d} problems")
            triage_counts = df[df["sht_triggered"] == 1]["sht_triage"].value_counts()
            print("-" * 60)
            print("SHT Triage Breakdown:")
            for method, count in triage_counts.items():
                print(f"  {method:<26} | {count}")
        print("="*60 + "\n")


# --------------------------- Entrypoint ---------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Multi-Agent Math Solver - VERSION 10.0 (SIV + Critic-SHT)")
    print("  Symbolic Inverse Verification + Heterogeneous Models")
    print("  + Process Verification + SymPy Fallback + SHT")
    print("  [v10.0] Novel: deterministic backward verification via SymPy CAS")
    print("=" * 70)
    print()
    
    print("TOKEN BUDGET (Groq Free Tier = 100K tokens/day):")
    print("  10 problems + SHT  →  ~60K-90K tokens  (safe)")
    print("  20 problems + SHT  →  ~90K-120K tokens  (may hit limit)")
    print()

    print("Select Model Configuration:")
    print()
    print("  — Large models (API, needs key) —")
    print("  1) homogeneous_groq      — all roles: Qwen3-32B via Groq")
    print("  2) diverse_groq          — LLaMA 70B + Gemma 9B + Mixtral 8x7B (Groq)")
    print("  3) cross_provider        — Groq (LLaMA 70B) + Google (Gemini)")
    print("  4) budget_optimized      — LLaMA 8B cheap roles, 70B critical (Groq)")
    print("  5) homogeneous_google    — all roles: Gemini Flash (Google)")
    print()
    print("  — Small open models (needs HF_API_KEY or TOGETHER_API_KEY) —")
    print("  6) tiny_math_homogeneous  -- Qwen2.5-Math 1.5B local (zero API cost)")
    print("  7) deepseek_distill_1_5b  -- DeepSeek-R1-Distill-Qwen 1.5B local")
    print("  8) small_math_homogeneous -- DeepSeek-R1-Distill-Qwen 7B (Together)")
    print("  9) qwen_math_7b           -- Qwen2.5-Math 7B-Instruct (Together)")
    print(" 10) phi4_mini              -- Phi-4-mini 3.8B (HuggingFace Router)")
    print(" 11) small_vs_large         -- baseline: 1.5B local / rest: LLaMA 70B Groq")
    print(" 12) qwen_math_7b_local     -- Qwen2.5-Math 7B local 4-bit (T4 friendly)")
    print(" 13) deepseek_7b_local      -- DeepSeek-R1-Distill 7B local 4-bit (T4 friendly)")
    print()

    config_choice = input("Enter selection (1-13) [default=1]: ").strip()

    preset_map = {
        "1":  "homogeneous_groq",
        "2":  "diverse_groq",
        "3":  "cross_provider",
        "4":  "budget_optimized",
        "5":  "homogeneous_google",
        "6":  "tiny_math_homogeneous",
        "7":  "deepseek_distill_1_5b",
        "8":  "small_math_homogeneous",
        "9":  "qwen_math_7b",
        "10": "phi4_mini",
        "11": "small_vs_large",
        "12": "qwen_math_7b_local",
        "13": "deepseek_7b_local",
    }
    preset_name = preset_map.get(config_choice, "homogeneous_groq")
    
    print(f"\nSelected: {preset_name}")
    selected_config = HETEROGENEOUS_PRESETS[preset_name]
    print("Role assignments:")
    for role, mc in selected_config.items():
        print(f"  {role.value:<25} → {mc.provider}/{mc.model_name or 'default'}")
    print()
    
    # Number of problems
    num_input = input("Number of problems [default=10]: ").strip()
    num_problems = int(num_input) if num_input.isdigit() else 10
    
    # SHT toggle
    sht_input = input("Enable SHT hypothesis testing? (y/n) [default=y]: ").strip().lower()
    enable_sht = sht_input != "n"

    # [v10.4] Evaluation-mode prompt: opt-in to the cache only for dev/debug.
    eval_input = input(
        "Evaluation mode (disables cache for clean numbers)? (y/n) [default=y]: "
    ).strip().lower()
    evaluation_mode = eval_input != "n"
    if not evaluation_mode:
        print("WARNING: dev mode — response cache is ON. Numbers reported "
              "from this run are NOT publication-safe.")

    pipeline = QualityAwarePipeline(
        use_cache=not evaluation_mode,          # cache only when explicitly opted out of eval
        heterogeneous_preset=preset_name,
        evaluation_mode=evaluation_mode,
        enable_sht=enable_sht,
    )
    
    estimated_tokens = num_problems * (9000 if enable_sht else 4500)
    print(f"\nEstimated token usage: ~{estimated_tokens:,} tokens")
    
    # Check if cross-provider needs both keys
    providers_needed = set(mc.provider for mc in selected_config.values())
    if "groq" in providers_needed and not GROQ_API_KEY:
        print("ERROR: This config requires GROQ_API_KEY in .env")
        exit(1)
    if "google" in providers_needed and not GOOGLE_API_KEY:
        print("ERROR: This config requires GOOGLE_API_KEY in .env")
        exit(1)
    if "huggingface" in providers_needed and not HF_API_KEY:
        print("ERROR: This config requires HF_API_KEY (or HUGGINGFACE_API_KEY) in .env")
        exit(1)
    if "together" in providers_needed and not TOGETHER_API_KEY:
        print("ERROR: This config requires TOGETHER_API_KEY in .env")
        exit(1)

    # Dataset selection -- comment/uncomment to enable or disable.
    #
    # Grade-school level:
    #   "gsm8k_test"         GSM8K test split (standard baseline)
    #   "gsm8k_train"        GSM8K train split
    #   "gsm-hard"           GSM8K with larger numbers
    #   "gsm-plus"           GSM8K with 8 perturbation types
    #   "gsm-symbolic-main"  Apple GSM-Symbolic main
    #   "gsm-symbolic-p1"    GSM-Symbolic +1 extra clause
    #   "gsm-symbolic-p2"    GSM-Symbolic +2 extra clauses (hardest)
    #   "svamp"              SVAMP structural variation
    #
    # Competition / olympiad level (new 2026):
    #   "math500"            Hendrycks MATH-Hard 500
    #   "math"               Full Hendrycks competition_math (~12K)
    #   "aime2024"           AIME I & II 2024 (AI-MO/aimo-validation-aime)
    #   "aime_historical"    AIME 1983-2024 all years
    #   "olympiadbench"      OlympiadBench EN (IMO-level open-ended)
    #
    # Uncontaminated 2025/2026 benchmarks:
    #   "aime_2025"          AIME 2025 combined   (MathArena, 30 problems, int answers)
    #   "aime_2025_i"        AIME 2025 I only     (MathArena, 15 problems)
    #   "aime_2025_ii"       AIME 2025 II only    (MathArena, 15 problems)
    #   "aime_2026"          AIME 2026 combined   (MathArena, 30 problems, freshest)
    #   "aime_2026_i"        AIME 2026 I only     (MathArena, 15 problems)
    #   "aime_2026_ii"       AIME 2026 II only    (MathArena, 15 problems)
    #   "hmmt_feb_2025"      HMMT Feb 2025        (MathArena, ~35 problems)
    #   "hmmt_nov_2025"      HMMT Nov 2025        (MathArena, ~35 problems)
    #   "hmmt_feb_2026"      HMMT Feb 2026        (MathArena, ~35 problems, freshest)
    #   "hmmt_2025"          alias → hmmt_nov_2025
    #   "hmmt_2026"          alias → hmmt_feb_2026
    #   "olymmath_hard"      OlymMATH EN-HARD     (RUC-AIBOX, frontier ~58%)
    #   "olymmath_easy"      OlymMATH EN-EASY     (RUC-AIBOX, AIME level)
    #   "amo_bench"          AMO-Bench            (Meituan, 50 original IMO problems, ~63%)
    #   "omni_math"          Omni-MATH            (KbsdJames, 4 428 olympiad problems)
    #   "livemathbench"      LiveMathBench        (OpenCompass, anti-contamination, updated monthly)
    #
    # Multilingual (new 2026):
    #   "mgsm"   "mgsm_de"  "mgsm_es"  "mgsm_fr"
    #   "mgsm_zh"  "mgsm_ja"  "mgsm_ru"  "mgsm_th"
    DATASETS = [
        # --- Grade-school (existing) ---
        "gsm8k_test",
        "gsm-hard",
        "gsm-plus",
        "gsm-symbolic-p2",
        "svamp",
        # --- Competition / olympiad (existing) ---
        "math500",
        "aime2024",
        "olympiadbench",
        # --- Uncontaminated 2025/2026 (new) ---
        # "aime_2026",          # AIME 2026  — freshest, integer answers
        # "hmmt_feb_2026",      # HMMT Feb 2026 — harder than AIME
        # "olymmath_hard",      # OlymMATH EN-HARD — frontier ~58 %
        # "amo_bench",          # AMO-Bench — 50 original IMO problems
        # "omni_math",          # Omni-MATH — 4 428 olympiad problems
        # "livemathbench",      # LiveMathBench — anti-contamination, monthly updates
        # --- Multilingual (new 2026) ---
        # "mgsm",
    ]

    print()

    df_results = pipeline.run(
        datasets_list=DATASETS,
        num_problems=num_problems,
        hardener="distractor",
    )
    pipeline.report()

    print(f"\n{token_budget.usage_report()}")

    out_file = f"final_results_v73_{preset_name}_n{num_problems}.csv"
    df_results.to_csv(out_file, index=False)
    print(f"Results saved to '{out_file}'.")
