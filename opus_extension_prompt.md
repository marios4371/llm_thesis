# MAS-SHT v10.2 Extension — Prompt for Claude Opus 4.7

> Copy everything below the `---` line into Opus 4.7. Edit the three `<TODO>` placeholders first (GitHub repo URL, your decisions on the 3 design questions at the end, and any preset names you want changed).

---

## ROLE

You are a senior research engineer extending a diploma-thesis project called **MAS-SHT** (Multi-Agent Solver with Structured Hypothesis Testing), currently at version 10.1. Your job is to ship three interconnected extensions: (1) small open-weight math model support, (2) a Google Colab experiment harness, (3) a clean baseline suite + comparative metrics. Output must be **production-quality, integration-safe code** — not pseudocode, not snippets — and must coexist with the existing pipeline without rewriting it.

The thesis claim being defended is: *"MAS-SHT (Architect–Engineer + SIV + SHT) outperforms simpler approaches on GSM8K-style math word problems by a margin that is statistically significant after accounting for cost."* Your extensions exist to **make that claim falsifiable and defensible**. Treat the metrics layer as the most important deliverable — without it, the thesis is unverifiable.

## PROJECT FACTS (verified — do not re-derive these)

**Repo root:** `C:\Users\mario\gitproj\MAS_LLM_Thesis\`

**Key files:**
- `Mas_solver.py` — main pipeline, ~2000+ LOC, version 10.1
- `siv_module.py` — Symbolic Inverse Verifier (SymPy-based, novel contribution)
- `test_siv.py`, `test_integration.py` — existing test suite
- `.env` — holds `GROQ_API_KEY`, `GOOGLE_API_KEY`

**Architecture (must be preserved):**
- `class AgentRole(Enum)`: `BASELINE`, `MATHEMATICIAN`, `PROGRAMMER`, `HYPOTHESIS_GENERATOR`, `JUDGE`
- `@dataclass class ModelConfig`: `provider: str`, `model_name: Optional[str]`
- `HETEROGENEOUS_PRESETS: Dict[str, Dict[AgentRole, ModelConfig]]` — currently has `homogeneous_groq`, `diverse_groq`, `cross_provider`, `budget_optimized`, `homogeneous_google`
- `class UnifiedLLMClient(provider, use_cache, model_override)` — currently dispatches to `provider in {"groq", "google"}`. Groq path uses `OpenAI(base_url="https://api.groq.com/openai/v1")`. Google path uses `genai.GenerativeModel`.
- Rate limiting: module-level `groq_limiter = RateLimiter(12)`, `google_limiter = RateLimiter(15)`, `token_budget = TokenBudget(100_000)`
- Pipeline flow: Baseline → Mathematician (blueprint JSON) → Programmer (Python code) → SIV Layer 1 (forward audit) → SIV Layer 2 (per-variable inverse if Layer 1 fails) → Confidence Gate → SHT (if triggered) → Judge

**SIV's positioning vs FOBAR (must be preserved in any docstring you touch):** SIV is deterministic SymPy-based, zero LLM calls, per-variable fault localization, operates on math→math layer. FOBAR (Jiang et al. ACL 2024) is probabilistic LLM-based, binary verdict, partially NL-layer. The two are orthogonal.

**Constraint that bit the project before:** Groq free tier is 12 RPM and ~100K TPD. Inter-problem cooldown of 3s already exists. v10.1 added a fix where SIV-verified-skip cannot override baseline disagreement — preserve this ordering.

---

## EXTENSION 1 — Small Open Math Models

**Goal:** Enable head-to-head comparison of small math-specialist models against the large general models already in the pipeline.

**Provider additions to `UnifiedLLMClient`:**

1. **`provider="huggingface"`** — call HF Inference Providers router (`https://router.huggingface.co/{provider}/v1/chat/completions`, OpenAI-compatible). The legacy `api-inference.huggingface.co/models/{id}` endpoint is largely deprecated; use the router. Auth: `Authorization: Bearer {HF_API_KEY}`. Handle 503 "model loading" with exponential backoff up to 60s. Handle the OpenAI-compatible response shape from the router (not the legacy `[{"generated_text": "..."}]` shape).

2. **`provider="local_hf"`** — load via `transformers.AutoModelForCausalLM + AutoTokenizer`, run on whatever device is available (CUDA if present, else CPU). This path is for the 1.5B models so they can run inside Colab on a T4 with zero API dependency. Cache the loaded model on the client instance — do not reload per call. Support `torch.float16` on GPU, `torch.float32` on CPU.

3. **`provider="together"`** — Together AI's OpenAI-compatible endpoint (`https://api.together.xyz/v1`). This is the realistic path to actually running 7B math models in serverless mode. Auth: `TOGETHER_API_KEY`.

**Rate limiters & budgets (add module-level):**
- `hf_limiter = RateLimiter(10)` (conservative, free tier)
- `together_limiter = RateLimiter(20)` (Together free credits)
- Local HF: no rate limiter, no budget — just sequential execution
- Do NOT extend `TokenBudget` to non-Groq providers; it is Groq-TPD-specific.

**Models to wire (add to `HETEROGENEOUS_PRESETS`):**

```
"tiny_math_homogeneous":   all roles → Qwen/Qwen2.5-Math-1.5B-Instruct (local_hf)
"small_math_homogeneous":  all roles → deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (together)
"qwen_math_7b":            all roles → Qwen/Qwen2.5-Math-7B-Instruct (together)
"phi4_mini":               all roles → microsoft/Phi-4-mini-instruct (huggingface)
"small_vs_large":          baseline → Qwen2.5-Math-1.5B-Instruct (local_hf)
                           architect/programmer/hyp_gen/judge → llama-3.3-70b-versatile (groq)
"deepseek_distill_1_5b":   all roles → DeepSeek-R1-Distill-Qwen-1.5B (local_hf)
```

**HF-specific quirks you must handle:**
- Cold start: first call to a serverless model returns HTTP 503 with `estimated_time` in the body. Retry with backoff up to 60s total. After that, mark the call as failed (`ERROR_HF_COLD_START`) — do not block forever.
- Response truncation: many small models have 4K context. Trim system+user to fit within `model_max_context - max_tokens - 256` (safety margin). Use a simple `len(text) // 4` token estimator unless `tiktoken` is available.
- Some HF router models require a specific provider routing (e.g., `nebius`, `together`, `fireworks`). If the default route 404s, try the next provider in a small priority list `["together", "nebius", "fireworks"]` before giving up.
- Local HF: small math models often need the chat template applied (`tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`). Use it if the tokenizer exposes it; fall back to a plain concatenation otherwise.

**`.env` additions:** `HF_API_KEY`, `TOGETHER_API_KEY`. Loader must not crash if either is missing — only raise when a preset that needs it is selected.

---

## EXTENSION 2 — Google Colab Notebook

**Deliverable:** `MAS_SHT_Experiments.ipynb` — a single self-contained notebook, runnable end-to-end on a free Colab T4, that drives all experiments.

**Required cells, in order:**

1. **Setup**
   - `pip install` everything: `openai`, `google-generativeai`, `sympy`, `statsmodels`, `scipy`, `pandas`, `matplotlib`, `tqdm`, `datasets`, `transformers`, `accelerate`, `torch`, `python-dotenv`, `huggingface_hub`
   - Mount Google Drive
   - Clone the project repo: `git clone <TODO: paste your GitHub URL here>` into `/content/MAS_LLM_Thesis`. If the repo is private, use a token from Colab Secrets.
   - Load API keys from `google.colab.userdata` (Colab Secrets), not from a hardcoded `.env`. Required keys: `GROQ_API_KEY`, `GOOGLE_API_KEY`, `HF_API_KEY`, `TOGETHER_API_KEY`. Missing-key path: warn but continue (so a Groq-only run is possible).
   - Set `RESULTS_DIR = "/content/drive/MyDrive/MAS_SHT/results/"`, `CHECKPOINT_DIR = "/content/drive/MyDrive/MAS_SHT/checkpoints/"`. Create both.

2. **Dataset**
   - Use `datasets.load_dataset("openai/gsm8k", "main", split="test")`
   - Three sampling modes via a config dict: `{"mode": "full"}` (all 1319), `{"mode": "random", "n": 100, "seed": 42}`, `{"mode": "stratified", "n": 100, "seed": 42}` where stratification is by question word count into terciles (short/med/long).
   - Output: `problems: List[Dict]` with keys `id`, `question`, `gold_answer` (parsed numeric value from the GSM8K `####` line).

3. **Experiment runner**
   - Config dict at the top of the cell: `PRESETS_TO_RUN`, `BASELINES_TO_RUN`, `N_PROBLEMS`, `SEED`, `INTER_PROBLEM_DELAY`, `CHECKPOINT_EVERY` (default 5).
   - For each (system, preset) pair, iterate over `problems` with a `tqdm` bar.
   - Per-problem record: `{problem_id, system, preset, gold, predicted, correct, time_s, num_llm_calls, tokens_estimated, error_type}`. Always append, even on error.
   - **Checkpointing:** every `CHECKPOINT_EVERY` problems, pickle the current per-system results dict to `CHECKPOINT_DIR/{system}_{preset}_{timestamp}.pkl`. On cell re-run, load the latest checkpoint and resume from `len(checkpoint)` — do not re-run completed problems. This is mandatory; Colab will time out and the user must not lose progress.
   - **Per-provider rate limiting:** the cell must respect Groq 12 RPM / 100K TPD, HF 10 RPM, Google 15 RPM, Together 20 RPM. Use the module-level limiters from `Mas_solver.py` — do not invent new ones.
   - On Groq daily-budget exhaustion (`ERROR_RATE_LIMIT_DAILY` or `ERROR_BUDGET_EXCEEDED`), the cell pauses and prints `"Groq daily limit reached at problem X. Resume tomorrow."` then breaks out of that system's loop (other systems on other providers continue).
   - At the end of each system×preset run, write a CSV: `RESULTS_DIR/{system}_{preset}_{timestamp}.csv`.

4. **Aggregation**
   - Glob all CSVs in `RESULTS_DIR/`, merge into one DataFrame, dedupe on `(problem_id, system, preset)` keeping latest timestamp.
   - Pivot/groupby to produce a per-system summary: `accuracy`, `avg_time_s`, `avg_llm_calls`, `avg_tokens`, `n_problems`.

5. **Statistical analysis** — calls into `evaluation_metrics.py` (Extension 3).

6. **Plots & comparison table** — calls into `evaluation_metrics.py`.

The notebook must be one file with no out-of-band dependencies beyond what cell 1 installs. Do not assume the user has the project installed locally before opening Colab.

---

## EXTENSION 3 — Baselines & Comparative Evaluation

### Baselines (all run on the same problem set, same seed)

- **B1 Direct-Answer** — single LLM call, prompt: `"Solve this math problem. Give only the final numeric answer.\n\n{problem}\n\nAnswer:"`. Temperature 0.
- **B2 Chain-of-Thought** — single call, prompt: `"Solve step by step. State the final numeric answer on a line starting with 'Answer:'.\n\n{problem}"`. Temperature 0.
- **B3 Self-Consistency (SC@5)** — 5 independent CoT samples at temperature 0.7, parse the final number from each, **majority vote** by exact float equality (with rounding to 6 decimals to handle float noise). Add 1s extra inter-sample sleep to avoid RPM bursts. Record `num_llm_calls=5`.
- **B4 Baseline-Only** — wrap the existing `BASELINE` agent of the pipeline as a standalone (re-uses existing prompt and code path; no new prompt).
- **B5 MAS-NoSIV** — full pipeline, but `enable_siv=False`. The confidence gate must always trigger SHT. Measures SIV's contribution.
- **B6 MAS-NoSHT** — full pipeline, but `enable_sht=False`. SHT never runs regardless of gate. Measures SHT's contribution.
- **B7 MAS-SHT-Full** — current v10.1 pipeline, unchanged.

**Implementation rules:**
- B5 and B6 must be flags on the existing pipeline entry-point (`run_pipeline(..., enable_siv=True, enable_sht=True)`), **not** copies of the pipeline. Adding two boolean kwargs and gating the relevant blocks is the entire change.
- B1–B4 live in a new file `baselines.py` as standalone functions: `direct_answer(client, problem)`, `chain_of_thought(client, problem)`, `self_consistency(client, problem, n=5)`, `baseline_only(client, problem)`. Each returns `BaselineResult(answer: float | None, raw: str, num_llm_calls: int, tokens_estimated: int, time_s: float)`.

### Metrics — new file `evaluation_metrics.py`

```python
def compute_all_metrics(results_dict: Dict[str, pd.DataFrame],
                        reference_system: str = "mas_sht_full") -> pd.DataFrame:
    """
    For each system, return a row with columns:
      accuracy, n, delta_accuracy, error_reduction,
      avg_llm_calls, avg_tokens, avg_time_s, accuracy_per_call,
      siv_trigger_rate, siv_skip_rate, sht_trigger_rate,
      acc_when_sht_triggered, acc_when_sht_skipped
    SIV/SHT rates are NaN for non-MAS systems.
    """

def run_mcnemar_tests(results_dict: Dict[str, pd.DataFrame],
                      reference_system: str = "mas_sht_full",
                      alpha: float = 0.05) -> pd.DataFrame:
    """
    For each non-reference system, build the 2x2 contingency table over
    the SAME problem set (intersect problem_ids), run McNemar's test:
      - if discordant pairs (b+c) < 25 → exact=True, correction=False
      - else → exact=False, correction=True (Yates)
    Return columns: comparison, n_paired, b, c, statistic, p_value,
                    significant_at_alpha, test_used.
    """

def plot_comparison(metrics_df: pd.DataFrame, output_path: str = "comparison.png"):
    """
    Three panels in one figure:
      1. Accuracy bar chart, sorted, MAS-SHT highlighted.
      2. Efficiency scatter: x=avg_llm_calls, y=accuracy, one point per system,
         labeled. Pareto frontier dashed.
      3. Δ-accuracy bar chart with significance stars (* p<0.05, ** p<0.01,
         *** p<0.001) from McNemar.
    """
```

**Output artifacts the metrics module must produce:**
- `comparison_table.csv`
- `comparison_table.md` (via `to_markdown()`) for thesis appendix
- `mcnemar_results.csv`
- `comparison.png`

### Apples-to-apples enforcement

`run_mcnemar_tests` and `compute_all_metrics` must intersect `problem_id` sets across systems before computing. If a system ran on fewer problems (e.g., Groq daily limit hit), drop those `problem_id`s from all systems for the comparison and report `n_paired` clearly.

---

## CONSTRAINTS

1. **Edit `Mas_solver.py` in place.** Bump version to v10.2. Add a `CHANGELOG v10.2` block at the top covering: HF/Together/local_hf providers, new presets, `enable_siv`/`enable_sht` flags. Do not rewrite functions you are not changing.
2. **Preserve v10.1 invariants** — especially the `_confidence_gate` ordering fix where baseline disagreement is checked before SIV-verified skip.
3. **No silent failures.** Every error path must record an `error_type` on the result row so the metrics layer can distinguish "wrong answer" from "API failure".
4. **Same problem set across all systems.** The runner uses one shuffled `problems` list; every system iterates it. No per-system reshuffling.
5. **All tests still pass.** `python -m pytest test_siv.py test_integration.py` must remain green. Add `test_baselines.py` and `test_metrics.py` covering at least: each baseline returns a `BaselineResult`, McNemar handles n_discordant<25, `compute_all_metrics` handles empty DataFrames without crashing.

---

## DELIVERABLES (produce in this order, each as complete files or precise diffs)

1. **Diff for `Mas_solver.py`** — HF/Together/local_hf provider branches in `UnifiedLLMClient`, new rate limiters, new presets, `enable_siv`/`enable_sht` kwargs threaded through the pipeline entry-point. Show as unified diff or as before/after blocks for each touched function.
2. **`baselines.py`** — full file.
3. **`evaluation_metrics.py`** — full file with the three functions above.
4. **`MAS_SHT_Experiments.ipynb`** — full notebook as JSON (every cell, every line, no `...` placeholders).
5. **`test_baselines.py` + `test_metrics.py`** — full files.
6. **A short `EXTENSIONS_README.md`** (≤1 page) describing how to run a fresh experiment from scratch, where outputs land, and how to interpret the comparison table.

---

## BEFORE YOU WRITE ANY CODE

If any of the following are still ambiguous, **ask me first** rather than guess:

1. The GitHub URL for the Colab clone step (`<TODO>` above). If you don't get one, use placeholder `https://github.com/USERNAME/MAS_LLM_Thesis.git` and clearly mark it.
2. Whether SC@5 should drop to SC@3 to conserve Groq TPD on full-GSM8K runs. Default to SC@5 unless I say otherwise.
3. Whether the user expects `transformers` and `accelerate` pre-installed locally (for `local_hf`) or only in Colab. Default to "Colab only — local-HF-on-host is best-effort and warns at import time".

For everything else (response shapes, retry counts, file naming, plot styling), make a defensible choice and document it in a 1-line comment at the call site. Do not pepper me with low-stakes questions.

## QUALITY BAR (definition of done)

- A user can open the Colab notebook, paste their secrets, run all cells top-to-bottom on the GSM8K test split (random N=100, seed=42) and get back: a CSV per system, a merged comparison table, a McNemar table, and a 3-panel PNG — without manual intervention beyond clicking Run All.
- The metrics layer produces numbers that **could falsify the thesis** if MAS-SHT happens to be no better than B2 or B3. Do not bias the metric design toward MAS-SHT looking good.
- Total new+changed LOC budget: ~1500. If you exceed 2000, you are over-engineering — stop and reconsider scope.

Begin by reading `Mas_solver.py` and `siv_module.py` end-to-end. Then produce the deliverables in the order listed.
