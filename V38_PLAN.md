# v38: Symbolic Surgery — repair the false claim instead of dropping the sample

**Files:**
- `trace_surgery.py`: the operation (locate, propagate, cut for resume) and the ADAPT-RS rule. Pure functions, no model. It also holds the policy `surgery_policy`, frozen 2026-10-20 (section 3).
- `replay_v38.py`: the offline replay on the 680 stored rows. It reproduces the 2026-10-20 probes and gives every number in section 2.
- `pretest_v38.py`: the manifest, the GPU run with the surgery stage, and the pre-registered reading.
- `test_v38.py`: 53 offline checks. They include the frozen policy itself (a changed policy fails the test, so the notebooks stop) and the run's error guards.
- `pretest_data/v38_surgery.json`: 260 fresh rows, seed 38, built 2026-10-20 from metadata only.
- `MAS_SHT_Kaggle_v38.ipynb` runs forward for 8.5 h. `MAS_SHT_Colab_v38.ipynb` runs `--reverse` and saves to Drive.
- Unchanged and frozen: `trace_siv.py` (v35.1, sha256 483ed23b89b6) and `refutes()` (sha256 c6ddc9369e9a).
- Frozen for v38 (the run and the summary print these; values of the LF files as git checks them out on Linux):
  - `trace_surgery.py`: sha256 **42376931d94a**;
  - `surgery_policy()`: sha256 **8eb4df69f91b**.

The version number: v37 was left to the cross-family PRM session (see the shared notes' `VERSION_CLAIMS.md`).

## 1. Why

Trace-level SIV (v35/v36) audits the arithmetic claims the solver's chain of thought writes out, and it drops any sample with a provably false claim. That repairs **selection** but never **coverage**: if all five samples hold the same slip, there is nothing right left to pick.

The 2026-10-20 cluster analysis (one unit = one distinct false claim per row, 242 claims) found that the repeated slips are carry errors:
- **84%** of slips that recur in ≥ 2 samples are ±1 at a single digit position, against 24% of isolated slips (Fisher p = 4e-9).
- 72% of the repeated slips sit in multi-operand ± chains, against 24% of the isolated ones (p = 4e-6). This is the "lookahead" carry failure of multi-operand addition.
- The PRM gives ≥ 0.9 to 38% of carry errors (23% of other errors, p = 0.028).

Resampling repeats these slips and the verifier scores them as fine, so only an intervention can fix them. The CAS already knows the right value. v38 writes it in.

## 2. Offline evidence (`python replay_v38.py`; canonical grader |pred − gold| < 1e-3; no GPU)

### 2.1 Reproduction of the 2026-10-20 probes (their surgery copied verbatim into section A of the replay)

| quantity | value |
|---|---|
| wrong samples rescued by propagation alone | **69** |
| right samples harmed (both on MATH) | **2** |
| rows with no right sample that gain one | **6 / 79** |
| repair-then-vote vs SC@5, all 680 rows | **W17 L0** |
| the same on the pools the checker never saw (V21, v35, v36; 380 rows) | **W10 L0** (p = 0.002) |
| ADAPT-R with repair vs ADAPT-R | **84.1 vs 83.4, W7 L2**; by pool: P2a +1, P2b +1, v36 **+3** |

Every number above reproduces exactly.

### 2.2 The v38 operation (`trace_surgery.operate`)

It improves on the probe in three ways:
- It follows the audit's own rules: run-on chains and steps the trace itself calls false are not repaired.
- It writes the repaired text, including expressions on the right side and chain links, so the trace can be **re-audited and re-scored**.
- It never substitutes a corrected value as an exponent or an index.

| pool | refuted | correctable site | propagation reached the answer | rescued | harmed | repaired text passes the audit | no-right rows that gain one |
|---|---|---|---|---|---|---|---|
| P2a | 39 | 37 | 23 | 19 | 0 | 37 | 1 / 18 |
| P2b | 43 | 43 | 22 | 14 | 0 | 43 | 1 / 8 |
| V21 | 42 | 39 | 18 | 11 | 0 | 39 | 0 / 10 |
| MATH | 25 | 21 | 3 | 0 | 2 | 20 | 0 / 3 |
| v35 | 17 | 14 | 5 | 1 | 0 | 13 | 1 / 5 |
| v36 | 57 | 56 | 39 | 30 | 0 | 56 | 4 / 35 |
| **ALL** | **223** | **210** | **110** | **75** | **2** | **208** | **7 / 79** |

- All 210 resume cuts fall right after a relation sign.
- Every one of them is the audit's first FALSE claim.
- Both harmed samples are right samples the audit already refutes, so neither costs any rule a row.

**The table the policy has to weigh** (refuted WRONG samples with a correctable site, right after propagation / n):

| | propagation reached the final answer | it did NOT reach it |
|---|---|---|
| isolated slip (n = 1) | 37 / 62 | **0 / 61** |
| correlated slip (n ≥ 2) | **38 / 46** | **0 / 19** |

- When propagation does not reach the answer, it **never** rescues a sample. In that case the value was carried in prose, which numeric matching cannot follow.
- Of the 22 refuted RIGHT samples, propagation keeps 20 right. It reached the answer in only 2.

### 2.3 The rules (stored PRM scores of the ORIGINAL traces; a run re-scores the repaired ones)

| arm | P2a | P2b | V21 | MATH | v35 | v36 | ALL (680) | calls/row |
|---|---|---|---|---|---|---|---|---|
| SC@5 | 68 | 74 | 80 | 95 | 71 | 155 | 543 (79.9) | 5 |
| SC@5-R | 72 | 78 | 83 | 95 | 72 | 158 | 558 | 5 |
| SC@5-S (always propagate, then vote) | 71 | 77 | 83 | 95 | 71 | 163 | 560 | 5 |
| ADAPT | 76 | 82 | 85 | 90 | 70 | 154 | 557 (81.9) | 3.34 |
| ADAPT-R | 77 | 83 | 86 | 94 | 71 | 156 | 567 (83.4) | 3.32 |
| ADAPT-RS, always propagate | 78 | 84 | 86 | 92 | 70 | 160 | 570 | 3.33 |
| ADAPT-RS, propagate if it reached the answer, else refute | 78 | 84 | 86 | 94 | 71 | 160 | 573 (84.3) | 3.32 |
| **ADAPT-RS, the frozen policy (section 3)**, resume yields nothing (lower bound) | 78 | 84 | 86 | 94 | 71 | 160 | **573 (84.3)** | 3.34 |
| the same, resume = the propagated repair (proxy) | 78 | 84 | 86 | 94 | 71 | 160 | 573 (84.3) | 3.35 |

- The frozen policy is **W8 L2** vs ADAPT-R (untouched pools W5 L1) under both readings.
- Over the 210 refuted samples with a site it chooses 110 propagate, 24 resume and 76 refute.
- The offline replay cannot say what a resume does: only the run can.

Paired comparisons:
- ADAPT-RS (reach) vs ADAPT-R:
  - ALL: **W8 L2**, p = 0.11.
  - untouched pools: **W5 L1**.
  - v36: +4, all of it in v36's pool A (clean GSM-Hard ≥ 1e4: W5 L1; pools B and C have no opportunity).
- ADAPT-RS (always propagate) vs ADAPT-R: W8 L5. The 3 extra losses (math500_295, math500_272, math500_436) are non-reached repairs whose clean-looking text brings a wrong answer back into the pool.
- Repair-then-vote vs SC@5:
  - ALL: W17 L0.
  - untouched pools: **W11 L0** (p = 0.001).
  - vs SC@5-R it is only W9 L7: in voting, the repair adds little over refutation.

**The two remaining losses**, both limits of the method:
- **p2_1535.** Four samples share the decimal slip `4.32 + 4.95 + 64.80 = 73.07`. Repaired, they are arithmetically right but read the problem wrongly. They tie the right sample at a PRM score of 1.000, and the position tie-break picks a repaired one.
- **gsm-hard_272.** The first *checkable* false claim is not the first *error*: the repaired sum still carries an earlier, uncheckable wrong operand.

**Honesty notes:**
- P2a, P2b and MATH are in-sample for the checker.
- The "reached" rule was read off these same rows.
- No repaired trace was re-scored, and RESUME was never tried.

Hence the fresh run.

## 3. The method and the policy

**Operation** (per refuted sample; `trace_surgery.py`):
1. **Locate** the first false claim in reading order, under the audit's rules.
2. Then one of:
   - **Propagate.** Replace the claimed value with the CAS value. Re-evaluate later left-hand sides that reuse a corrected value, correct the claims that become false in turn, and map the final answer. Free.
   - **Resume.** Cut the trace right after that claim's relation sign, write the CAS value, and let Qwen2.5-Math-7B-Instruct continue. This is one partial call with the samples' sampling settings (T 0.8, top-p 0.95, top-k 50). It gets the **remainder** of the row's token budget: prefix plus continuation stay within the budget, with at least 128 new tokens. A resumed trace is therefore never longer than a sample could have been, and it stays inside the model's 4096-token context.
3. The repaired trace is **re-audited** (if still refuted, it is dropped) and **re-scored by the PRM**. It then returns to the pool.

**Rule ADAPT-RS:** ADAPT-R with the refuted samples passed through the policy instead of dropped. v22's gate (3 samples, min step reward ≥ 0.85, else 5) and its pick are unchanged.
- Calls: 3 or 5, plus one per resumed sample. A sample already resumed at the gate is not charged again.
- With a policy that only refutes, ADAPT-RS is exactly ADAPT-R (tested).

**The policy: `surgery_policy(sample, n_samples_sharing_the_false_claim) -> 'propagate' | 'resume' | 'refute'`.** It was frozen on 2026-10-20, before any v38 sample existed. The thesis author asked for it to be written, and it was written from the stored-row evidence below.

The inputs it may use:
- `sample['surgery']['site']` (the claim, its claimed and CAS values, its step);
- `sample['surgery']['prop']['reached']`;
- `sample['surgery']['slips']`;
- the sample's PRM rewards;
- n, the number of drawn samples holding the same slip.

The trade-offs:
- **Propagate** is free, but it misses values the trace carries in prose. Section 2.2: never a rescue when it does not reach the answer (0/80).
- **Resume** costs a call, and the model may re-derive the same wrong value after the splice, because a carry slip is a habit, not a typo. The run measures this directly ("answer unchanged" and "re-audit still refutes").
- **Refute** is v35, free and safe. But a correlated slip (the same false claim in ≥ 2 samples) cannot be fixed by resampling. Where every sample holds it, refutation leaves nothing right.

**FROZEN POLICY (2026-10-20):**

| condition | action | why |
|---|---|---|
| the propagated correction reached the final answer | **PROPAGATE** | free; it rescues 75 of 108 such wrong samples on the stored rows |
| it did not reach it, and the slip recurs (n ≥ 2) | **RESUME** | propagation rescues 0/80 when it does not reach the answer, and a correlated slip is the one error resampling repeats, so the call is worth paying |
| it did not reach it, and the slip is isolated (n = 1) | **REFUTE** | the other samples are independent draws without this slip; on the stored rows these slips are often symptoms of a derivation already off course (`30 - 35 = 3`, `40 + 80 = 220`), where a resume buys a call for little |

- Cost: +0.04 calls/row on v36's pool A (+0.02 over the 680 rows), against +0.13 for "resume whenever not reached" and +0.43 for "always resume".
- **What the decisive arm measures:** the primary arm's effect will come mostly from propagation (Wei 2022's mechanism, here inside verifier-routed sampling), with resume as the targeted remedy for correlated slips.
- **What the run still measures about resume:** the run stores a resume for EVERY refuted sample. "Always resume" and the resume diagnostics are therefore reported over all of them, as the non-decisive evidence on the novel mechanism.

## 4. Pre-registered (FROZEN 2026-10-20, before any sample exists; the policy was frozen the same day, section 3)

All readings use the canonical grader, |pred − gold| < 1e-3. It is pre-registered this time, not a correction.

| reading | bar |
|---|---|
| **PRIMARY** | ADAPT-RS vs ADAPT-R on the **200 A rows**. **SUPPORTED** if net ≥ **+3** with W ≥ 2L, labelled "significant" if the sign test gives p < 0.05. **REFUTED** if net ≤ 0. Otherwise INCONCLUSIVE. |
| **SECONDARY** | ADAPT-RS vs ADAPT-R on the **60 M rows** (MATH-500): net ≥ −1 → **NON-INFERIOR**, otherwise NOT MET |

**Reported, not decisive:**
- ADAPT-RS vs SC@5 (A and M).
- The fixed policies (propagate-if-reached, always-resume, always-propagate).
- Coverage (rows with no right sample that gain one).
- The resume diagnostics: how often the continuation re-commits the slip, the answer is unchanged, or the re-audit still refutes.
- Correlated-slip counts, calls/row, the audit statistics.

**Expectation, stated before the run:** v36's pool A gave W5 L1 per 100 rows offline (in-sample policy, original PRM scores), so 200 A rows would give about W10 L2 if the rate holds. The bar is +3 with W ≥ 2L. The re-scoring could move this either way.

At v36's rates the frozen policy resumes about 8 samples in the 200 A rows. The run, however, stores about 110 resumes (one per refuted sample with a site) for the diagnostics.

## 5. Rows, samples, cost, how to run

**Rows** (`pretest_data/v38_surgery.json`, seed 38, `python pretest_v38.py --build-manifest`, built 2026-10-20 from metadata only):

| pool | n | source and filter | eligible |
|---|---|---|---|
| **A** | 200 | GSM-Hard rows that pass `gsmhard_audit.py` (auditable, none of collision / never perturbed / Platinum-dropped), gold > 0, max number in the text ≥ 1e4, **not** in v36 nor the v18/v19 pretests | 519 |
| **M** | 60 | MATH-500 problems with a plain numeric answer, any level, **not** in v34/v35 | 137 |

- The M rows are evenly spaced: one near each end, and never more than 4 A rows in a row. Each half of a two-machine run therefore mixes both pools.
- Pool A is where v36 placed every surgery opportunity. M is where propagation has hurt and the audit is rarely triggered.

**Samples:** exactly v36 per row: 5 samples (v22's CoT prompt, T 0.8) with **1536** new tokens on A and **2048** on M, each scored by Qwen2.5-Math-PRM-7B.

Then the **surgery stage**, for every refuted sample with a correctable claim:
- the propagated repair, PRM-scored;
- **one** resume continuation, PRM-scored.

Both are stored, so every policy can be read off the same run. The audits are recomputed offline with the frozen checker.

**Guards (in `pretest_v38.py`, not in the notebooks, which do not sync with git):**
- The run refuses to start without the policy. `test_v38.py` also fails if the policy differs from the frozen one.
- `pretest_v22` (which holds the continuation code) is imported before any model loads.
- A **resume smoke test** runs right after the models load: one short continuation of a fixed corrected prefix must extend it. If the resume path is broken, the run fails in a minute, not after hours.
- An error in the audit or in a resume is recorded in the file (`surgery_error` / `resume_error`) and the run goes on. The 4th error stops it as systematic. The file is saved, so re-running resumes it. The summary prints the error count.

**Cost:**
- About 180-260 s per row (v36 Kaggle 182 s/row; MATH about 262 s/row).
- Plus one partial generation per refuted sample: about 0.56 per A row in v36.
- About **16-17 GPU-hours** in total.

**How to run:**
- **Kaggle** (`MAS_SHT_Kaggle_v38.ipynb`; T4 x2, Internet on, secret `HF_API_KEY`): Save & Run All. It works from the start, stops itself after 8.5 h and saves after every row.
- **Colab** (`MAS_SHT_Colab_v38.ipynb`; T4, secret `HF_API_KEY`): Run all. It works from the end (`--reverse`) and saves to `MyDrive/MAS_SHT_v38/`.
- A third short session may be needed. A second Kaggle session resumes it (Add Input → the first version's Output).
- **Reading:** upload Kaggle's file to Drive as `kaggle_pretest_v38.json` and Colab's Cell 4 merges it. Locally: `python pretest_v38.py --summary-only --out A.json --merge B.json`.

**Order of operations:**
1. ~~Write `surgery_policy`~~: done and frozen 2026-10-20.
2. `python test_v38.py`: 53/53 (verified 2026-10-20, also from a clean `git archive` plus the v38 files).
3. Commit and push ONLY the v38 files: `trace_surgery.py`, `replay_v38.py`, `pretest_v38.py`, `test_v38.py`, `V38_PLAN.md`, `pretest_data/v38_surgery.json`, `MAS_SHT_Kaggle_v38.ipynb`, `MAS_SHT_Colab_v38.ipynb`.
4. Run Kaggle and Colab at the same time.

**Offline verification done on 2026-10-20 (no GPU):**
- `operate` ran on all 3400 stored samples: 0 exceptions, slowest sample 0.33 s.
- A full 260-row stub run went end to end, through the pre-registered reading.
- The resume cut is right after the relation sign of the audit's first FALSE claim in 210/210 cases.

Do not tune the checker, the operation or the policy on these rows before the verdict.

## 6. What each outcome would mean

- **SUPPORTED.** Repair beats refutation where computation is the bottleneck: a CAS correction spliced into a sampled trace, inside verifier-routed sampling, fixes slips that neither voting nor the PRM can see. This is the first method in the thesis that adds **coverage**, not only better selection.
- **INCONCLUSIVE with few losses.** The repair is safe but rare. Report the opportunity count (refuted samples per row, correlated slips) next to it, as in v35/v36.
- **REFUTED.** Splicing the right value does not help once the repaired trace is re-scored. The resume diagnostics will say why: the slip is re-committed, or the PRM prefers other samples.
- **SECONDARY NOT MET.** On MATH the repair brings wrong answers back. The thesis then restricts surgery to arithmetic-heavy word problems.

## 7. Prior work and what is new (novelty checked 2026-10-20)

- **Wei et al. 2022**, *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (arXiv 2201.11903, appendix): a post-hoc external calculator. It evaluates every equation of a single CoT with Python and propagates results to later equations by string matching. This is the direct ancestor of PROPAGATE, so that part is **not new**.
- **Sun, Stolfo & Sachan 2025**, *Probing for Arithmetic Errors in Language Models* (arXiv 2507.12379): probes on hidden states detect arithmetic errors and guide selective re-prompting of the erroneous steps. It detects with the model's own internals and re-prompts. It does not splice an exact value or sit inside verifier-routed sampling.
- **Pan et al. 2025**, *LEMMA: Learning from Errors for MatheMatical Advancement in LLMs* (arXiv 2503.17439): fine-tunes the model to correct its own errors during generation ("Fix & Continue"). That needs training, and the fix comes from the model, not from an exact external value.
- Also related: T1 (arXiv 2504.04718; a tool filter before a reward model); AR-Sampling (arXiv 2504.01317; a PRM step threshold triggers a rethink).

**New here:**
- **Resume:** detect a false claim with a model-free CAS audit, truncate the sampled trace at that claim's relation sign, splice the exact value, and continue.
- Inside **verifier-routed sampling**: the repaired trace is re-audited and re-scored before it competes with the untouched samples.
- Aimed at the one error class shown to defeat both voting and the same-policy PRM, the correlated carry slip.

No published system was found that does this.
