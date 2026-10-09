# v32: Reading Jury (RJ), an extension of the v22 system

**Files:**
- `reading_jury.py`: the pools and the rule; pure code.
- `pretest_v32.py`: the dev screen.
- `test_v32.py`: offline checks, including that every notebook cell compiles and that the rule reproduces v22 when only Qwen's family is present.
- `MAS_SHT_Kaggle_v32.ipynb`, `MAS_SHT_Colab_v32.ipynb`.

The base is v22: Qwen2.5-Math-7B-Instruct samples, picked by the best last-step reward of Qwen2.5-Math-PRM-7B. On fresh rows it scored B5 78 vs SC@5 68, W11 L1.

RJ changes two things:
1. **Who writes the reading.** Two general instruct models of other families write it, not Qwen's own Reader.
2. **How the PRM's contested ties are broken.** Support is counted in model families, not in samples.

## 1. What the earlier screens established

| version | what was tried | result |
|---|---|---|
| v24 | a Qwen Reader writes sentence-anchored notes for the Solver | the Solver **adopts** the reading both ways: +19.6 pp on misread rows, −12.5 pp on well-read rows |
| v25 | the PRM picks between plain and Reader-view samples | RA 80 vs B5 78; on disputes the PRM saturates |
| v27, v28, v29, v30 | same-family agents check fidelity to the problem | all STOP: the PRM is problem-insensitive, and the same-family judge and questioner fail |
| v31 | DeepSeek-Math-7B-RL **solves** as a juror inside the ties | STOP, 75 vs 81. The juror is right on only 44.8% of samples, but repeats Qwen's self-consistent errors on only 1/13 rows |

**Stored-data facts behind v32** (dev rows, seed 45, no GPU):
- **The same-family Reader is not a second witness.**
  - On the 13 rows where ≥ 3/5 plain samples agree on a wrong answer, the Qwen-Reader view repeats it on **5/13**.
  - The DeepSeek juror repeats it on **1/13**.
  - Both are right on 5/13.
- **Counting views instead of families fails.** Treating the C and Q views as independent ("prefer the tied answer found in both views") gives B10 **74 vs 81** (W2 L9).
- **v31 lost on arithmetic, not on reading.**
  - Its losses were single stray votes and arithmetic scatter, e.g. p2_1824: 521.25, 24, 1.5, 104, 130.
  - Its confident votes (3–4 of 5) won p2_1972, 1268 and 890.

## 2. The published limitations it takes up

- **Tan et al., EMNLP 2025** (arXiv 2505.17656). Self-consistent errors are **model-specific**: cross-family overlap is 5–20%.
  - They only detect these errors, with a probe on a second model's hidden states, on TriviaQA and SciQ.
  - Mitigation is left open.
- **Li et al., "Rethinking Mixture-of-Agents"** (arXiv 2502.00674, TMLR 2026). Mixing model families trades **quality for diversity**, because the weaker models' answers are worse.
  - **RJ's answer:** the weaker family never answers. It only reads.
- **Yan et al., RoR-Bench** (arXiv 2504.00509). No fix exists for recitation "without over-reliance on user's clarifications".
  - **RJ's answer:** the foreign Reader supplies the clarification.
- **v31 (this thesis).** A cross-family juror helps only if it is as accurate as the generator.
  - **RJ's answer:** RJ's foreign views have the Solver's own arithmetic.

**Novelty check (2026-10-09).** The closest work is the two-step variant of Rephrase-and-Respond (Deng et al. 2023). There, ONE stronger model rephrases the question for a weaker responder, to give it better questions. It has no ensemble of readers, no error decorrelation and no verifier.

Not new:
- heterogeneous agents (X-MAS);
- PRM best-of-n;
- situation notes (MathWorld, v24).

Rejected in the same session, with reasons:
- **P(problem | solution) scoring:** covered by LEDOM's Reverse Reward (ACL 2026) and by Coder-Reviewer.
- **A cross-family hidden-state probe:** within-problem probe selection is about random ("Hidden Error Awareness", arXiv 2605.09502).
- **Defamiliarizing the story:** few residual errors are template-triggered.

## 3. The method

| agent | model | job |
|---|---|---|
| Reader F1 | microsoft/Phi-3.5-mini-instruct (family `phi`) | v24's sentence-anchored situation notes, greedy, once per row |
| Reader F2 | ibm-granite/granite-3.1-8b-instruct (family `granite`) | the same |
| Solver | Qwen2.5-Math-7B-Instruct (family `qwen`) | 5 plain samples (stored), plus **2 samples under each foreign reading** (v24's ASQ prompt; v22's sampler: T 0.8, top-p 0.95, top-k 50, 1024 tokens) |
| Verifier | Qwen2.5-Math-PRM-7B | scores every sample against the ORIGINAL text, last-step reward |

**Pool RJ** = C5 + F1×2 + F2×2: **9 solver samples and 2 reader calls**. The comparator B10 = C5 + Q5 has 10 solver samples and 1 reader call.

**Selection:**
- Outside a contested tie, RJ uses v22's rule: best last-step reward, then the lower pool position.
- Inside a contested tie, `jury_choice` decides. A contested tie means two or more answers within 1e-3 of the top reward.
- `jury_choice` sees, per tied answer, the **families** that produced it.
- A sample drawn after a failed reading is Qwen's own reading, and is counted as `qwen`.

## 4. Offline facts (test_v32.py)

- v22 is reproduced through the new code: B10 81, B5 78 (dev), B5 86 (held-out); B9 (C5 + Q4) 81.
- With only Qwen's family in the pool (all stored data), the rule must give v22 exactly, row by row.
- The pick never leaves the contested tie.
- An invalid choice raises an error.

## 5. Pre-registered (frozen before any foreign-reader sample exists)

| reading | rows | bar |
|---|---|---|
| **SCREEN** | 100 dev main (P2, seed 45) | RJ (jury rule) vs B10 (v22's rule): **GO** if net ≥ +3 with W ≥ 2L, so build a fresh-row confirmation; **STOP** if net ≤ 0; otherwise WEAK |
| GUARD | 20 dev GSM-Plus rows | RJ − B10 ≥ −1 |
| HELD (secondary) | 100 held-out P2 rows (seed 44, C5 only) | RJ with the jury rule vs the same pool with v22's rule ≥ 0, which means CONSISTENT |

**Reported, not decisive:**
- RJ vs the same pool with v22's rule: does the rule add anything beyond the samples?
- RJ vs B9: the same number of solver samples.
- RJ1 and RJ2: one foreign family each.
- Per-view accuracy.
- Readings usable per Reader.
- The **shared-error floor** per view: on rows where ≥ 3/5 plain samples agree on a wrong answer, how many of a view's samples repeat it. The Q view (same family) is compared with F1 and F2.

**The rule** (chosen by the author on 2026-10-10, before any foreign-reader sample existed; `reading_jury.jury_choice`):

```python
score = [len(g['families']) for g in groups]   # families over the WHOLE pool
best = max(score)
if best < 2 or score.count(best) > 1:
    return None          # no new witness, or a tie: keep v22
return score.index(best)
```

| families of the tied answers (v22's first) | choice |
|---|---|
| {qwen} vs {phi, granite} | the second (two foreign families against Qwen alone) |
| {qwen} vs {phi} | v22 (one witness against one) |
| {qwen, phi} vs {qwen, granite} | v22 (tie in family count) |
| {qwen, phi} vs {granite} | v22 (it already leads) |

## 6. Cost and how to run

- **Kaggle** (`MAS_SHT_Kaggle_v32.ipynb`, GPU T4 x2, Save & Run All), `pretest_v32.py --held --max-hours 8.0`:
  - dev readings: 2 Readers × 15 batches of 8 rows, about 30–40 min;
  - dev samples: one batched call per row (2 prompts × 2 samples) plus 4 PRM scores, about 3 min/row, so about 6–7 h for 120 rows.
- The held-out rows continue in a second session. Add the first session's Output as an input; the notebook resumes by itself.
- **Colab** (`MAS_SHT_Colab_v32.ipynb`, one T4): the result is kept on Drive. Expect two or three sessions.
- **Read the result:** `python pretest_v32.py --summary-only --out results_Octomber/pretest_v32.json`.

## 7. Risks, stated before the run

1. **Reader harm.** The foreign Readers may misread more than Qwen's Reader did. RJ's pool then holds wrong samples the PRM may rank first.
   - RJ vs RJ/v22 separates the rule's contribution from the pool's.
   - The guard rows bound the damage on well-read problems.
2. **Format.** A Reader that ignores the `S<n>: Q: … A: …` format falls back to the plain prompt. The fallback is counted as Qwen's reading, so the evidence weakens but is not corrupted.
   - The run stops if the first 8 readings of a Reader are all unusable.
3. **A shared GSM8K memory.** Every family has seen the GSM8K originals that P2 perturbs. If the foreign views repeat Qwen's self-consistent answers (high shared-error floor), the mechanism is refuted. That is reported as such.
4. **Power.** With 100 rows, +3 is about the noise floor. A GO only earns a fresh-row confirmation; it is not a thesis claim.
5. **Disk and time.** About 54 GB of checkpoints. A Reader's download is removed if less than 30 GB of disk is left. A session that stops at `--max-hours` resumes from the saved file.

## 8. Screen result

Not run yet.
