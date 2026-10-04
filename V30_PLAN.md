# v30: Re-Read Verification (RRV), an extension of the v22 system

**Files:**
- `reread_verification.py`: the verifier's input formats and the rule; pure, no model at import.
- `pretest_v30.py`: the dev screen (re-scores stored samples; generates nothing).
- `test_v30.py`: offline checks (38 with the rule written).
- `MAS_SHT_Kaggle_v30.ipynb`, `MAS_SHT_Colab_v30.ipynb`: the run notebooks.

The base is the thesis's one confirmed gain: **v22**, a Solver whose samples are chosen by a step verifier (Qwen2.5-Math-PRM-7B, best last-step reward). On 100 fresh seed-45 rows: B5 78 vs SC@5 68, W11 L1. RRV changes nothing in v22 except **the order in which the verifier reads the text**, and only on rows where the verifier cannot decide.

## 1. The real problem (stored data, no GPU)

| pool | right | right answer only in other samples | never reachable | selection loss |
|---|---|---|---|---|
| **B5** (5 plain samples) | 78 | 13 | 5 | 4 |
| **B10** (+ 5 Reader-view samples) | 81 | 4 | 5 | **10** (9 in a PRM tie) |

- B10 has a **contested tie** (different answers within 1e-3 of the top last-step reward) on 64 of 100 rows. 57 contain the right answer; v22's argmax wins 48 (**84%**).
- The lost ties hold **explicit** misreadings that the verifier scores ~1.000:
  - p2_1268: the picked solution writes *"Since there are two groups"*; the text says *"These 3 groups"*. The step reward **dips to 0.787**, then the last step is back at **1.00000**.
  - p2_734: the picked solution writes *"Time to return home from the library: 42 minutes"* (the trip is 13 + 42). The step **dips to 0.622**; the last step is **0.99998**.
  - p2_1972: 480 min/day divided by the 4-day total (2400) for "percentage of their day"; every step 1.000.
- So the verifier *notices* something mid-trace, but the last-step reward forgets it: it judges whether the last step follows from the steps before, not whether the trace fits the problem.
- v27 measured it directly: rewards under the familiar version of the problem correlate **r = 0.93** with rewards under the real one.
- v24, v27, v28, v29: no same-family **agent** checks fidelity better than the generator. RRV adds no agent and no generation.

## 2. The mechanism and the method

**Causal attention.** In the verifier's input the problem comes first, so every problem token is encoded before the solution exists. A misreading can be caught only by solution tokens looking back at it, and the last-step reward reads that comparison weakly.

**RRV:** after the solution, the verifier reads the problem **again**, then sees the answer once more; the reward of that final step is the RRV score. The second copy of the problem is encoded *with the solution's claims in view*, and the final judgment sits right after it.

| format | user turn | assistant turn (steps) |
|---|---|---|
| V0 = v22 | problem | s1 … sn |
| V1 = front copy (literature control) | problem + "Let me repeat that: " + problem | s1 … sn |
| **V2 = RRV** | problem | s1 … sn, "Let me read the problem again: " + problem, "So the answer is \boxed{a}." |
| V2p = mechanism check | problem | as V2, but the re-read copy is the **familiar version** (v26 Prototype agent) |

- Score: log-odds (logit_pos − logit_neg) of the last step, float32, so saturation near probability 1 does not erase differences.
- Rule: inside v22's contested tie only, the candidate with the largest `rrv_key` wins; everywhere else v22's pick stands. **The rule is written by the author in `rrv_key` before any RRV score exists** (see §5).
- V2p: a problem-aware verifier must move when the re-read copy changes. If V2 vs V2p still correlates near 0.93, the verifier did not become problem-aware, whatever the accuracy says.

## 3. The published ideas and their open windows

- **Xu et al. 2025, "Reward Models Identify Consistency, Not Causality"** (arXiv 2502.14619). Removing the question barely moves the reward; shuffling questions does. Tested on Skywork-PRM-1.5B/7B and Llama3.1-8B ORM/PRM, **not** Qwen2.5-Math-PRM-7B (v27 measures it here). Their remedies are all training: causality-augmented training, chain-of-thought awareness, human-in-the-loop refinement.
- **Leviathan et al. 2025, "Prompt Repetition Improves Non-Reasoning LLMs"** (arXiv 2512.14982). Repeating the prompt wins 47 of 70 tests with 0 losses, most when the material comes before the question (options-first), because the second copy attends to the first. With reasoning enabled the effect is neutral. Future work: partial repetition, attention analysis, fine-tuning with repetition. **Verifiers, reward models and judges are never mentioned.** A PRM is the extreme non-reasoning LLM: one forward pass, no chain of thought.
- **RE2, "Re-Reading Improves Reasoning"** (EMNLP 2024): re-reading for the generator, not the judge.
- **EVPV** (arXiv 2603.16253): premise verification for vision PRMs; it needs a policy-written checklist plus an independent constraint extractor.
- **R-PRM / ThinkPRM**: generative verifiers that restate the problem; they need training.

**RRV's increment:** a training-free, generation-free change to a discriminative PRM's input that targets problem-fidelity errors, tested where the PRM is measured problem-insensitive, with a mechanism check (V2p) and a literature control (V1).

Not new: PRM best-of-n, prompt repetition for generators, re-reading.

## 4. The ceiling (offline, test_v30.py)

A perfect tie-breaker on the same stored pools:

| pool | v22 | perfect tie-breaker |
|---|---|---|
| B5 | 78 | 81 |
| RA (2 plain + 2 read) | 80 | 87 |
| RA5 | 81 | 88 |
| **B10** | **81** | **90** |
| held-out B5 (v21 rows) | 86 | 90 |

## 5. Pre-registered (frozen 2026-10-04, before any RRV score exists)

| | rule |
|---|---|
| **SCREEN** (100 dev main rows) | RRV-B10 vs B10 (v22's rule, same 10 samples), format V2: net ≥ +3 with W ≥ 2L → **GO**; net ≤ 0 → **STOP**; otherwise WEAK |
| GUARD (20 GSM-Plus rows) | RRV-B10 − B10 ≥ −1 row |
| HELD (secondary) | the 100 v21 seed-44 main rows, read by no version from v24 to v29: RRV-B5 − B5 ≥ 0 → CONSISTENT, else INCONSISTENT. Only 4 ties are winnable there, so it is a transfer / no-harm check, not a second route to GO |
| DRIFT (gate) | V0 re-scored on 3 stored samples must reproduce the stored reward within 0.02, or the run stops |
| rule | `rrv_key = (rrv, real, -index)`: inside the contested tie the highest V2 log-odds wins; an exact RRV tie falls back to v22's order (higher last-step reward, then pool position). Frozen 2026-10-04 |

**Reported, not decisive:** RRV on B5/RA/RA5; RRV-RA vs B5; dispute-level accuracy against v22's 84%; the V1 control; the EXPLORATORY full re-rank by V2 alone; AUROC of V0 vs V2 inside the ties; the V2p mechanism line (r against v27's 0.93); every flip.

The dev rows motivated the idea, so a GO decides only whether a confirmation on **fresh** rows is worth running: the seed-48 rows (`pretest_data/v25_confirm_p2.json`) need C5 + Reader + Q5 + PRM + RRV, about 11-12 h.

## 6. Cost and how to run

- **No new solutions.** One verifier forward pass per scored sample, about 5 s on a T4.
- Pass 1: B10 ties of the 120 dev rows (556 scores). Pass 1b: B5 ties of the 140 held-out rows (226 scores). **About 1 h to the verdict.**
- Passes 2-4: the V1 control and other pools, the V2p mechanism check, every sample in V2 (`--all-passes`); about 2 h more.
- **Colab:** `MAS_SHT_Colab_v30.ipynb`, Run all; the result goes to `MyDrive/MAS_SHT_v30/pretest_v30.json`, saved after every row. **Kaggle:** `MAS_SHT_Kaggle_v30.ipynb`, Save & Run All; one T4 is enough.
- Offline reading: `python pretest_v30.py --summary-only --out <file>`.

## 7. Risks, stated before the run

- **The same-family bound.** v29's factored answers sided with the right reading 74% of the time. RRV gains only if re-reading lifts the verifier above the generator's own reading. V2p shows whether the verifier became problem-aware at all.
- **Saturation may persist** in V2 (every final step near +12 log-odds). Then the ranking is noise, W ≈ L, and the screen STOPs within an hour.
- **Literal-number pull.** The re-read copy contains the literal numbers; a solution that reuses one in the wrong role (734: 42) may look *more* consistent, not less.
- **Format shift.** A verbatim problem as a step is unusual for the verifier. The drift gate checks the plumbing, not this; V1 (which leaves the assistant turn untouched) is the comparison.
- **In-sample dev rows.** Hence the held-out reading and, on GO, the fresh confirmation.

## 8. Screen result

Not run yet.
