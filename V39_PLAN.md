# v39: Clause ablation — does a confidently wrong answer survive deleting the clause it ignored?

**Files:**
- `clause_ablation.py`: the detector, the analysis labels and the exact statistics. Pure functions, no model.
- `pretest_v39.py`: the manifest, the GPU run (greedy, solver only) and the pre-registered reading.
- `test_v39.py`: offline checks. They include the frozen sha256 of the detector and the manifest. A changed file fails the test, so the notebooks stop before the GPU.
- `pretest_data/v39_ablation.json`: 131 rows, seed 39, built 2026-10-21 from the stored samples and the GSM-Symbolic / GSM-Plus metadata. No generation existed when it was built.
- `MAS_SHT_Kaggle_v39.ipynb` (T4 x2, one worker per GPU) and `MAS_SHT_Colab_v39.ipynb` (one T4, saves to Drive).
- Frozen for v39 (the run and the summary print these; sha256 of the LF text, as git checks it out on Linux):
  - `clause_ablation.py`: sha256 **260bafc6b5bc**;
  - `pretest_data/v39_ablation.json`: sha256 **ef6833dabaed**.

The version number is claimed in the shared notes' `VERSION_CLAIMS.md`. Inverse Dispute Localization was not built: LMAD (arXiv 2608.01463) and Self-Contrast (ACL 2024) already cover its core.

## 1. Why

On GSM-Symbolic P2 most of the remaining errors are readings, not arithmetic. In the 2026-09-29 taxonomy, 12 of the 18 rows with no right sample apply the familiar GSM8K solution and override a condition that P2 **added**. RoR-Bench calls this *recitation* and reports no satisfactory remedy.

Two things are known from earlier versions:
- **No same-family check can see it.** The PRM gives 1.000 to 7 of 8 such errors. A Qwen Reader recites too. v27 (null-hypothesis verification), v29 (dispute to question), v30 (re-read verification) and v32 (reading jury) all stopped for this reason: the model confirms its own reading. Voting cannot help either, because the samples agree.
- **An intervention on the reading cuts both ways (v24).** The Solver adopts a Reader's reading: +19.6 pp per sample where it had misread, −12.5 pp where it had read correctly. What is missing is a detector that says *whether* the model misread, and *which* clause, without a gold answer.

**The textual check is too weak.** v21's premise ledger asked whether the derivation spent the stated quantities. It fired on 9% of rows. On this pilot's rows its simplest form (`clause_ablation.spent`: every number a unit states appears in the consensus samples' text) flags:
- 4 of 37 confidently-wrong P2 rows;
- 0 of 74 confidently-right P2 rows;
- 7 of 20 GSM-Plus distraction rows (there the distractor is *rightly* ignored).

The model writes the numbers out even when it ignores what they mean. Also, 9 of the 48 added clauses in the wrong rows hold no number at all, e.g. "they forgot their wallets at home, so they have to return … and walk all the way back".

**Deletion is a behavioural test.** If the model ignores clause X, the problem without X gets the same answer. The self-consistency that defeated v27–v32 becomes the signal: a model that recites ignores X in every variant.

## 2. The detector (`clause_ablation.py`)

1. **Units.** Every non-question sentence, every parenthetical (≥ 3 characters), and the question's leading clause ("Once they reach the library for the second time, how …") or trailing "if …" clause. They come from the project's sentence splitter (`renderings.split_sentences`, which protects decimals and abbreviations).
2. **Variants.** One per unit: the problem with that unit deleted. Commas and spacing are tidied, and a leading clause's deletion capitalises the question.
3. **Generation.** Qwen2.5-Math-7B-Instruct (4-bit, the paper's solver), v22's CoT prompt, greedy decoding, 1024 new tokens. Per row, the full problem and all its variants go into one left-padded batch. Answers are parsed as the stored samples were (`pretest_v21.parse_answer`). There is no PRM and no gold answer.
4. **Invariance.** A unit is invariant if its variant's answer equals the full problem's answer under the canonical grader, |a − b| < 1e-3. A variant with no parsable answer is never invariant.
5. **Flag (primary).** A row is flagged if some **informative** sentence or parenthetical is invariant. Informative means it holds a digit, a number word, or a condition word: `if`, `each`, `except`, `including`, `remaining`, `return`, `again`, `more`, `than`, `however`, … (the full list is in the code). Question clauses are kept out of the primary flag, because some of them restate a sentence (the wallet template's "for the second time"). They are reported separately.

The detector can only see a condition the model **ignores**. A condition the model *uses wrongly* changes the answer when deleted, so it is not flagged. This pilot measures how much of the error lives in the first kind.

## 3. Rows and labels (built 2026-10-21; frozen)

**Source pools.** These are the three stored P2 pools of the paper: P2a (seed 45), P2b (seed 48) and V21 (seed 44). Each row has 5 samples at T 0.8 with v22's CoT prompt. Answers are grouped by the canonical grader.

| group | definition | eligible | in the pilot |
|---|---|---|---|
| TARGET | P2 row whose plurality answer is **wrong**, with ≥ 3 of 5 samples agreeing | 37 | **37** (all) |
| CONTROL | P2 row whose plurality answer is **right**, with ≥ 4 of 5 agreeing | 159 | **74** (2 per TARGET, seed 39) |
| GUARD | GSM-Plus "distraction insertion" row (an irrelevant clause inserted by construction), plurality right, ≥ 4 of 5 | 59 | **20** (seed 39) |

**Size.** 131 rows, 560 deletion variants and 691 generations. Units per row: TARGET 5.05, CONTROL 3.97, GUARD 3.95.

**Clustering.** The 37 TARGET rows come from 20 templates, and one template (the house-budget problem) contributes 5 rows. The 74 CONTROL rows come from 31 templates. The P2 errors cluster by template (v24: 68% of the variance), so a one-row-per-template reading is reported.

**Analysis labels** (frozen in the manifest; the detector never sees them). Every unit is labelled **added**, **modified** or **same** against the problem it was derived from.
- **P2.** A GSM-Symbolic template has 50 instances with different names, places, items and numbers. Words present in ≥ 90% of the instances are template text (`CONST_SHARE`); the rest are slot fills. Only the unit's template words are compared with the base template's words: GSM-Symbolic `main`, same template id, every sentence of all 50 instances. A template sentence with new slot fills therefore stays **same**.
- **GSM-Plus.** The unit's content words are compared with the seed question.
- **Containment.** This is the best share of the unit's words found inside one reference sentence: < 0.5 → **added**, 0.5–0.75 → **modified**, ≥ 0.75 → **same**.

**Inspection.** One instance of each of the 50 P2 templates, 223 units, came out 131 same, 67 added and 25 modified. Read by hand, the labels match what P2 changed, with two known imperfections:
- A base sentence that P2 rewrote is labelled added. Template 37 turns "A lasagna has 30 pieces" into "A giant pizza has 87 servings".
- A change inside the question's core is not a unit. 6 of the 37 TARGET rows have no changed eligible unit, so for them a localization hit is impossible, by chance too.

**Eligible units by label.** TARGET: 48 added, 9 modified, 102 same. CONTROL: 79 added, 27 modified, 142 same.

## 4. Pre-registered (FROZEN 2026-10-21, before any generation exists)

**Rows read.** Every reading uses the canonical grader |pred − gold| < 1e-3.
- A row is **analysed** when its greedy full answer exists and agrees in correctness with its group: wrong for TARGET, right for CONTROL and GUARD.
- Rows where greedy decoding flips the stored consensus are counted and reported, not analysed.

**DETECTION.** f<sub>T</sub> and f<sub>C</sub> are the flag rates of the analysed TARGET and CONTROL rows.
- **MET** when f<sub>T</sub> − f<sub>C</sub> ≥ **0.20** and the one-sided Fisher exact p < **0.05**.
- Both conditions are required. A floating-point tolerance keeps 0.7 − 0.5 at the bar.

**LOCALIZATION.** This is computed over the flagged, analysed TARGET rows.
- A row is a *hit* when its flagged units include an **added or modified** unit.
- *Chance* is the probability of a hit if the same number of units had been drawn at random from the row's eligible units (hypergeometric, per row).
- **MET** when there are ≥ **10** such rows, the hit share exceeds the chance share by ≥ **0.15**, and the exact one-sided p < **0.05**. The p is the Poisson-binomial tail of the per-row chances: exact, no simulation.
- A detector that flags every unit hits by chance, so it does not localize.

**VERDICT.**
- **GO** if both are met: design the repair, v40, section 7.
- **WEAK** if exactly one is met.
- **NO-GO** if neither is met.
- **No verdict** on a partial run.

## 5. Reported, not decisive

- **Break-even ratio.** With v24's +19.6 / −12.5 pp and the eligible counts 37 and 159, a gated Reader breaks even at f<sub>T</sub>/f<sub>C</sub> ≈ 2.74 on these populations. This is a rough guide to what the gate must achieve, not a bar.
- **Other flag rules:** question clauses included; numeric units only; the textual ledger, with the overlap between deletion and ledger on the TARGET rows.
- **Invariant share of eligible units** per row: means, and the AUROC TARGET vs CONTROL.
- **Unit level:** invariance rate by label (added / modified / same) in each group. The recitation account predicts high invariance of added units in TARGET and low in CONTROL.
- **Per pool**, and **one row per template** (with its Fisher p).
- **GUARD:** the flag rate, and how often the flagged unit is the inserted distractor. Here ignoring is *right*, so this is the gate's exposure on well-read problems with irrelevant text.
- **Every flagged TARGET unit** (text and label), the input for designing the repair.
- **Exclusions:** missing full answers, greedy flips, unparsable variants. Also the cost (seconds per row, generations).

## 6. Cost and how to run

**Cost.** 691 greedy generations, at most 11 prompts per batch.
- A batch takes as long as its longest completion: about 1–2 minutes per row on a T4.
- In total, about **3–4 GPU-hours**. On Kaggle T4 x2 with `--parallel` (one worker per GPU, every other row) that is about **2 hours** of wall time.
- Each worker saves after every row. Re-running the identical command resumes it.

**Kaggle (recommended).** Use `MAS_SHT_Kaggle_v39.ipynb`:
1. Settings: Accelerator **GPU T4 x2**, Internet **On**, secret `HF_API_KEY`.
2. **Save Version → Save & Run All (Commit)**. The notebook runs `test_v39.py` first and stops on any failure. Before the first row, the run itself checks that a padded batch of two prompts solves both.
3. Download `pretest_v39_gpu0.json` and `pretest_v39_gpu1.json` from the Output tab.
4. If the session ends early, upload both files as a Kaggle **Dataset**, Add Input, and run again. The notebook copies them in and resumes.

**Colab (alternative, one T4, about 3–4 h).** `MAS_SHT_Colab_v39.ipynb` saves to `MyDrive/MAS_SHT_v39/` and resumes on **Run all**.
- It skips any row already finished in Kaggle files uploaded to the same folder under their own names.

**Reading** (no GPU):

    python pretest_v39.py --summary-only                 # pretest_v39.json + every pretest_v39_gpu*.json next to it
    python pretest_v39.py --summary-only --out A.json --merge B.json

## 7. What each outcome would mean

- **GO.** The deletion flag separates confident errors from confident successes, and it points at the clause P2 added. v40 would gate a reading intervention on the flag and target the flagged unit. Candidates:
  - re-solve with the flagged unit moved next to the question;
  - a Reader note on that unit only;
  - require the solution to state how the unit changes the answer.

  v40 would be measured against ADAPT and SC@k on **fresh** P2 rows. This pilot's rows were selected from stored samples the project has analysed many times: the selection is in-sample, even though the detector has never run.
- **WEAK, detection only.** A gate without a target: v24's Reader gated by the flag, judged against the break-even ratio.
- **WEAK, localization only.** The flagged units are the right ones, but the gate fires on too many good rows.
- **NO-GO.** Recorded among the unsuccessful attempts with its reason. The likely reasons, each readable from section 5:
  - the confident errors use the added clause wrongly instead of ignoring it;
  - deletion changes greedy answers chaotically;
  - right answers are also insensitive to a single deletion.

## 8. Prior work and what is new (checked 2026-10-21)

- **CogMath** (arXiv 2506.04481) removes a key condition. It reads an unchanged answer as reliance on a memorised problem: an **evaluation** of a model, aggregated over a benchmark.
- **TreeCut** (arXiv 2502.13442) and **MiP-GSM8K** (arXiv 2504.06514) build unanswerable or missing-premise problems to measure hallucination.
- **"Pause or Fabricate?"** (arXiv 2604.19656) masks premises to make training data.
- **ContextCite** (Cohen-Wang et al., NeurIPS 2024, arXiv 2409.00729) attributes a generation to context sources by ablating them and fitting a surrogate. Its applications include flagging statements misread from the context. Occlusion and leave-one-out attribution are the general family.
- **GSM-IC**, **I3C** (arXiv 2403.12744) and **System 2 Attention** (arXiv 2311.11829) handle irrelevant conditions. That is the opposite failure, and the one the GUARD rows measure.

**What is new is narrow.** Answer invariance under deleting a **stated condition** is used per problem, at test time and without labels, for three things:
- to flag a confident math answer as a misreading;
- to localize the condition it ignored;
- as the gate of a targeted repair, inside a pipeline where voting and a same-family verifier cannot see the error.

The probe itself is CogMath's, and ablation-based attribution is ContextCite's. Our contribution would be the use, together with the observation that deletion turns same-family self-consistency into the signal. This pilot tests whether the use works at all.

## 9. Caveats (known before the run)

- **In-sample selection.** The groups come from stored pools. A GO is a screen result and needs fresh rows (section 7).
- **Decoding.** The groups come from T 0.8 samples, but the detector runs greedy. Rows where greedy disagrees with the consensus's correctness are not analysed.
- **Clustering by template.** TARGET has 20 templates for 37 rows. The one-row-per-template reading is reported next to the primary.
- **Labels are heuristic** (section 3). They decide only LOCALIZATION:
  - a rewritten base sentence counted as added raises the hit share and its chance share together;
  - a change inside the question's core lowers both.
- **Only ignored conditions are detectable.** A condition the model uses wrongly changes the answer when deleted.
