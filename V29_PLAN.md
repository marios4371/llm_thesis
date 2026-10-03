# v29: Dispute-to-Question (DQ), an extension of the v22 system

**Files:**
- `dispute_question.py`: the rule; pure, no model at import.
- `pretest_v29.py`: the dev screen.
- `test_v29.py`: 39 offline checks.
- `MAS_SHT_Kaggle_v29.ipynb`, `MAS_SHT_Colab_v29.ipynb`: the run notebooks.

The base is the thesis's one confirmed gain: **v22**, a Solver whose samples are chosen by a step verifier (Qwen2.5-Math-PRM-7B, best last-step reward). On 100 fresh seed-45 rows it scored B5 78 vs SC@5 68, with W11 L1. DQ changes nothing in v22 except on the rows where its verifier cannot decide.

## 1. The real problem (stored data, no GPU)

v22's errors on the 100 seed-45 main rows, by cause:

| pool | right | right answer only in other samples | never reachable | selection loss |
|---|---|---|---|---|
| **B5** (5 plain samples) | 78 | **13** | 5 | 4 |
| **B10** (+ 5 Reader-view samples) | 81 | 4 | 5 | **10** (9 in a PRM tie) |

Notes on the table:
- "Other samples" means the Reader-view Q samples or the v26 extras.
- "Never reachable" covers p2_2285 and 2297 (template 45), p2_338 and 323 (defective golds) and p2_1625 (ambiguous).
- **With 5 plain samples the problem is coverage.** Adding the Reader's reading fixes coverage (13 → 4). It also creates disputes between readings that the PRM scores alike (selection losses rise from 4 to 10).
- The PRM checks that a solution agrees with itself, not with the problem (Xu et al. 2025, arXiv 2502.14619). It is robust to removing the question but reacts to question–solution inconsistency. The disputed readings are self-consistent, so it cannot see the difference.

**Each lost dispute hinges on one local fact of the problem.** "Wrong" below is the solution v22 picked; "right" is the one it lost to.

| row | wrong solution | right solution | the question that settles it |
|---|---|---|---|
| 1268, 1263, 1250 | `3 × 2 groups = 6` | `3 groups × 3 = 9` | How many groups do the cheerleaders accompany? |
| 734 | return trip = 42 min (`55 + 42`) | return trip = 55 (`55 + 55`) | How long is the walk from the library back home? |
| 56 | "13 miles in the first hour" | 26 miles in the first hour | How far does the fog spread in the first hour? |
| 1972 | `480 / 2400` | `480 / 600` | How many minutes is one work day? |

About 3 of the 10 losses are ambiguous: p2_1697, 1630 and 890.

## 2. The method

DQ runs only when the PRM's top (last-step reward within 1e-3 of the max) holds two or more answers.

1. **Contenders.** v22's pick is the incumbent. Each other answer in the tie is a challenger, represented by its best candidate; at most 2, best first.
2. **The question.** The Questioner (Qwen2.5-7B-Instruct, greedy, 2 written demos) reads the two solutions. It finds the first point where they read or compute differently, and writes **one question about the problem with a numeric answer**, plus the value each solution uses for it.
   - A question is refused if it contains a disputed value the problem does not state.
3. **The factored answer.** The question is answered by fresh calls that see **only the problem and the question, never a solution**: 3 × Qwen2.5-Math-7B-Instruct + 2 × Qwen2.5-7B-Instruct, T = 0.8.
   - Showing no solution is the factored variant of Chain-of-Verification.
   - It is also required here: v24 measured that the Solver adopts any reading it is shown.
4. **The decision.** A challenger replaces the incumbent only if **≥ 4 of the 5** answers give its value. Challengers go one at a time, each against the current incumbent.

**Why it should work where v24/v26/v27/v28 did not.** Recitation happens when the model generates a whole solution, because the familiar schema pulls the reading. Here the model only answers a narrow question about a sentence. The text literally says "These 3 groups", and no schema gets a chance to override it.

## 3. The published idea and its open window

- **LMAD** (Localized Multi-Agent Debate, arXiv 2608.01463) extracts atomic claims, locates the earliest conflict between agents and debates only that segment.
  - In its debate each agent sees the competing claims.
  - It is evaluated on multi-hop QA only (HotpotQA, 2Wiki, MuSiQue, StrategyQA).
  - No verifier triggers it.
  - Its stated limitation is linear reasoning structures.
- **Chain-of-Verification** (Dhuliawala et al. 2023) answers verification questions independently of the draft (the "factored" variant), for factual long-form generation.
- **Re-Reading** (RE2, EMNLP 2024) has the generator re-read the question.

**DQ's increment:**
- the disagreement is between solutions a learned **verifier** cannot separate, and its tie is the trigger;
- it is resolved with **one factored question about the source text**, not a debate;
- it is applied to **math word problems with modified conditions** (GSM-Symbolic P2);
- it **selects** among existing candidates and never re-solves.

Not new: PRM best-of-n, sampling diversity, the Reader agent, localising a disagreement.

## 4. The ceiling (offline, test_v29.py)

The same stored pools, with DQ's knockout, given a perfect answerer:

| pool | v22 | perfect answerer |
|---|---|---|
| B5 | 78 | 81 |
| RA (2 plain + 2 read) | 80 | 87 |
| RA5 | 81 | 88 |
| **B10** | **81** | **90** |

- B10 has a dispute on 64 of the 100 rows.
- v22's rule already wins **48 of the 57** disputes that contain the right answer, i.e. **84%**. DQ must beat 84% on those disputes to gain anything.

## 5. Pre-registered (frozen 2026-10-03, before any question exists)

| | rule |
|---|---|
| **SCREEN** (100 main rows) | DQ-B10 vs B10 (v22's rule, same 10 samples): net ≥ +3 rows with W ≥ 2L → **GO**; net ≤ 0 → **STOP**; otherwise WEAK |
| GUARD (20 GSM-Plus rows) | DQ-B10 − B10 ≥ −1 row |
| rule | strict (≥ 4 of 5); the majority rule is exploratory |

**Reported, not decisive:**
- DQ on the RA, RA5 and B5 pools;
- DQ-RA vs B5, the confirmed v22 system;
- dispute-level accuracy against v22's 84%;
- the Questioner's usable rate and why questions fail;
- vote margins on pairs with exactly one right solution;
- every flip;
- extra calls per row.

These rows motivated DQ, because the error analysis read them. The screen decides whether a confirmation on **fresh** rows is worth running. On **GO**: the seed-48 rows (`pretest_data/v25_confirm_p2.json`) need C5 + Reader + Q5 + PRM + DQ, about 11-12 h.

## 6. Cost and how to run

- **No new solutions.** One question plus 5 short answers per disputed pair, about 75 s per pair on a T4.
- **Pass 1** makes every pair the verdict needs (B10, about 106 pairs): about 2-2.5 h.
- **Pass 2** makes the pairs of the other pools: about 1.5 h more.
- **Colab:** `MAS_SHT_Colab_v29.ipynb`, Run all. The result goes to `MyDrive/MAS_SHT_v29/pretest_v29.json`, saved after every pair; Run all again resumes.
- **Kaggle:** `MAS_SHT_Kaggle_v29.ipynb`, Save & Run All. To resume, add the previous Output as an input.
- **Offline reading:** `python pretest_v29.py --summary-only --out <file>`

## 7. Risks, stated before the run

- **The Questioner, not the answerers, is the weak link.** A 7B instruct model must spot the first divergence between two long solutions. The run stops at once if its first 8 questions are all unusable. The question report shows how many were usable and why the others were not.
- **A factored question can still be misread.** Example: "how many groups leave" when the text lists two clubs joined by "or". The 4-of-5 rule exists to protect the 48 disputes v22 already wins.
- **Ambiguous golds.** About 3 of the 10 lost disputes have ambiguous golds; no reading question fixes those.
