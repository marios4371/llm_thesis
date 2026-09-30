# v25: Reading Arbitration (plan as of 2026-09-28)

A process verifier chooses between two agents' readings of the problem. It
does not choose among samples of one reading, which is what B5 does. The
experiment is pre-registered and has not been run.

| file | what it is |
|---|---|
| `reading_arbitration.py` | the method: the arms, the selection rule and the frozen verdicts. Deterministic, no model |
| `pretest_v25.py` | the runner: the dev screen on stored rows, the fresh confirmation, the summary |
| `test_v25.py` | 56 offline checks (`python test_v25.py`) |
| `pretest_data/v25_confirm_{p2,guard}.json` | 100 fresh P2 rows and 20 GSM-Plus distractor rows (seed 48, disjoint from v20-v24) |
| `MAS_SHT_Kaggle_v25.ipynb` | Cell 1 of the Kaggle notebook plus a run cell, with `MODE = 'dev'` or `'confirm'` |

## 1. Why: what B5 still gets wrong is a reading problem, and a vote cannot fix it

All numbers come from the stored v22 and v24 runs on the same 100 fresh P2
rows (seed 45), with no GPU. `python test_v25.py` reproduces them (part 4).

| | right / 100 |
|---|---|
| S5, plurality over 5 plain samples | 68 |
| **B5, verifier best-of-5 plain (v22, the current system)** | **78** |
| oracle over the 5 plain samples | 82 |
| oracle over **2 plain + 2 samples under the Reader's reading** | **88** |
| oracle over 3 plain + 2 read | 89 |
| oracle over all 10 | 91 |
| plurality over all 10 (both views) | 68 |
| plurality over 2 plain + 2 read | 68 |

1. **Selection is no longer B5's bottleneck; coverage is.** B5 is wrong on 22
   rows. On 18 of them none of the 5 plain samples is right, so no verifier,
   vote or restart policy that picks among them can reach those rows.
2. **The wrong samples agree.** Tan et al. (EMNLP 2025) call these
   self-consistent errors. They do not shrink with model scale, every
   consistency-based detector misses them, and that paper leaves their cause
   open.
3. **v24 found the cause on P2 by intervention.** Given the Reader's reading,
   the Solver adopts it in both directions:
   - +19.6 pp per sample on rows it had misread;
   - −12.5 pp on rows it had read well.
4. **The two readings fail on different rows.**
   - With one sample fewer, 2 plain + 2 read samples contain the right answer
     on 88 rows. The 5 plain samples contain it on 82.
   - On 11 of B5's 22 errors a read sample is right. On 9 of those, only a
     read sample is right.
5. **A vote throws this away.** Plurality over both views is 68, exactly S5.
   DivSampling's theory predicts this ("under majority voting, diversity may
   vanish"). So the missing piece is an arbiter that can tell a right reading
   from a wrong one, and the thesis already has one that works on P2 (v22:
   +10 over S5 on fresh rows).

## 2. The method

```
problem ─► Solver (Qwen2.5-Math-7B), plain CoT prompt ........ 2 samples   reading 1: the Solver's own
        ─► Reader (Qwen2.5-7B-Instruct, greedy) .............. situation Q/As per sentence (v24, unchanged)
        ─► Solver, the notes interleaved after each sentence . 2 samples   reading 2: the Reader's
        ─► Verifier (Qwen2.5-Math-PRM-7B) scores all 4 against the ORIGINAL text
        ─► answer = the sample with the best last-step reward (ties go to the plain sample)
```

Each design choice answers a measured failure:

- **The verifier never sees the notes.** A wrong note has to survive a check
  against the text it claims to describe. v24 failed because a wrong note was
  adopted without such a check.
- **The plain samples stay in the pool.** A wrong reading can win only if the
  verifier prefers it. This contains the fidelity cost (−12.5 pp on well-read
  rows) that sank v24 always-on.
- **The diversity is fresh samples under a different reading, not repair.**
  v22 showed that guided repair (V2P, rewind) loses to fresh samples. v20
  showed that re-rendering the problem adds no more diversity than
  temperature does (0.3968 vs 0.3968). v24 showed that a different reading
  does add it.
- **Equal calls.** RA is 2 + 1 + 2 = 5 LLM calls, the same as B5. The Reader
  call is greedy and takes ~31 s, less than one solver sample.

## 3. The literature, and the limitation each item leaves open

Checked by web search on 2026-09-28. How closely each source was read:
- **From the paper's own text:** Tan et al. (abstract and Limitations), the
  wrong-consensus paper (abstract), and DivSampling (the UAI 2026 abstract;
  the arXiv v1 was read through a fetch summary).
- **From the abstract only:** the four-stage paper.
- **From V24_PLAN.md:** MathWorld and Plan-and-Solve.
- **From search results only:** "Stop Overvaluing MAD", ReConcile and A-HMAD.
  Read these before quoting them in the thesis.

| work | what it shows | what it leaves open | v25's answer |
|---|---|---|---|
| Tan et al., *Too Consistent to Detect: A Study of Self-Consistent Errors in LLMs*, EMNLP 2025 | Self-consistent errors are stable or grow with scale, and all four detector families miss them. A cross-model probe on an external verifier's hidden states helps detection | Limitations: "The underlying causes of consistent errors still require deeper investigation." Detection only | On P2 the cause is the reading (the v24 intervention flips them). v25 **recovers** the right answer instead of only flagging the error |
| Wang et al., *On the Effect of Sampling Diversity in Scaling LLM Inference* (DivSampling), UAI 2026 | Diverse prompts (roles, injected ideas, rephrasings, including by a second model) lower Best-of-N error. "Under majority voting, diversity may vanish" | Selection uses the ground truth (oracle Best-of-N), not a verifier a system could run. The perturbations are generic, not aimed at an error type | A **learned process verifier** selects. The diversity is a situation reading aimed at self-consistent misreadings. The VRA arm tests their voting prediction on our data |
| Opedal et al., *World Models for Math Story Problems* (MathWorld), ACL Findings 2023 | Gold situation Q/As placed after their sentence: 70.8 → 78.6 | Needs gold world models: "an obvious limitation" | The Reader writes them (v24). v25 adds the arbiter v24 lacked |
| Wang et al., Plan-and-Solve, ACL 2023 | Planning fixes calculation and missing-step errors | "the semantic misunderstanding errors still remain" | These are the errors the second reading targets |
| Zhang et al., *Stop Overvaluing Multi-Agent Debate*, 2025 | Multi-agent debate rarely beats CoT or self-consistency at equal compute | Calls for model heterogeneity | Three different models in three different roles, at equal calls |
| *A Four-Stage Decomposition of Word-Problem Solving and Mechanistic Fragility*, arXiv 2609.17804 (EMNLP 2026 Findings) | Distractor failure localises to the operation-planning stage | No repair method | A repair that runs at inference time, with no access to model internals |
| Zhang et al., *Decomposing Wrong-Consensus Agreement in LLM Self-Consistency*, arXiv 2608.18795 | Wrong consensus is largely a stable per-question answer preference | "No new voting method is proposed" | Consistent with the template-level misreadings of v24 (68% of variance between templates) |

**Not new** (components or controls):
- PRM best-of-N (Lightman et al. 2023; Qwen2.5-Math-PRM);
- a second model that rephrases before a solver (RaR's two-step variant;
  DivSampling's dual-model rephrasing);
- heterogeneous agents (ReConcile; A-HMAD).

**New, as far as these searches found:**
1. A learned verifier arbitrating **between readings written by different
   agents**, at equal calls, with the plain reading kept in the pool.
2. Diversity aimed at **self-consistent errors**, with the mechanism
   (reading adoption) measured by intervention first.
3. Recovery of self-consistent errors, not only detection.

Do not claim "first". Claim "we found no prior work that...", and name these
papers.

## 4. Pre-registration (frozen in `pretest_v25.py` and `reading_arbitration.py`)

### Dev screen

`python pretest_v25.py --dev`, about 1-1.5 GPU-hours. It generates nothing.
- **Rows:** the 100 main + 20 guard rows of the v22 confirmation.
- **Plain samples:** the 5 that v22 stored, with v22's verifier scores.
- **Read samples:** the 5 ASQ samples that v24 stored, scored now by the same
  verifier code against the original text. RA takes the first two in draw
  order.

| outcome | condition |
|---|---|
| GO | RA − B5 ≥ +3 rows with W ≥ 2L |
| STOP | RA − B5 ≤ 0 |
| WEAK | anything in between (the user decides) |

These rows are fresh for the reader-view verifier scores, but not for the
idea: their oracle numbers are what suggested RA. The screen is never a
thesis claim.

It also re-scores the stored plain samples of 10 rows, to check this session's
verifier against v22's scores (drift).

### Confirmation

`python pretest_v25.py --max-hours 8.0`, about 11 h in two sessions.
- **Rows:** 100 fresh P2 + 20 guard.
- **Per row:** 5 plain samples, 1 Reader call and 2 read samples, with the
  v22 sampling settings. Every sample is scored by the verifier.

| reading | rule |
|---|---|
| **PRIMARY** | RA vs B5 (both 5 calls). **SUPPORTED** if net ≥ +4 rows, W ≥ 2L, and net ≥ +2 without the single template that contributes most. **REFUTED** if net ≤ 0. Otherwise INCONCLUSIVE |
| significance | The exact sign-test p is printed. A claim of significance needs p < 0.05 on its own |
| GUARD | RA − B5 ≥ −1 row → no harm |
| secondary | VRA vs S5 (predicted \|net\| ≤ 2: the diversity vanishes under a vote); RA vs B4 (same 4 solver samples); RA5 vs B5 (same 5 solver samples); where RA's wins come from (reading vs selection); pooled dev + confirmation sign test (n = 200, labelled as half not fresh for the idea) |
| exploratory, never used to choose | a cascade that stops when the first two plain samples agree; the verifier scoring the Reader's own notes ("reading vetting"); which view the verifier picks; the allocation table |

### Prediction, written before any read sample has a verifier score

- On plain samples the verifier converts 78 of 82 covered rows, i.e. 95%.
- If it arbitrates readings about as well, RA ≈ 0.95 × 88 ≈ 83-84 on dev,
  against B5 78: roughly +5 to +6 at equal calls.
- **The main risk is a verifier biased toward one reading's style.** The
  last-step reward saturates near 1.0, so near-ties are common; ties go to
  the plain sample, which is the conservative direction.
- The screen exists to catch this risk for about 1 GPU-hour instead of 11.

## 5. Run plan

1. Commit and push the v25 files, plus `results_September/preset_v24_dev.json`,
   which is untracked and is the dev screen's input. The Kaggle notebook pulls
   the .py files from git. It does not pull its own cells, so use the v25
   notebook as committed.
2. Run `MAS_SHT_Kaggle_v25.ipynb` with `MODE = 'dev'` on T4 x2 (Colab T4 also
   works, since it loads only the verifier). Read the SCREEN line.
3. On GO: set `MODE = 'confirm'`. It takes two sessions; the second one
   attaches the first session's Output and resumes.
4. Re-read offline:
   ```bash
   python pretest_v25.py --dev --summary-only --out results_September/pretest_v25_dev.json
   python pretest_v25.py --summary-only --out results_September/pretest_v25.json
   python pretest_v25.py --pooled results_September/pretest_v25_dev.json results_September/pretest_v25.json
   ```

## 6. What each outcome means for the thesis

- **SUPPORTED.** The thesis gets its own positive accuracy result on
  fresh rows: a three-agent system (Reader, Solver, Verifier) that beats the
  strongest single-reading system at equal calls. It is also the causal
  story, from v20 to v25, of why it works.
- **REFUTED at the screen.** A clean negative result, found in about one
  GPU-hour: a verifier trained on one model's solutions cannot judge another
  agent's reading. That tells PRM users something too. The "reading vetting"
  line will show whether the verifier can judge the notes themselves.
- **INCONCLUSIVE.** Report the direction and the pooled n = 200 reading,
  and state the power honestly.

## 7. Dev screen result (2026-09-28, `results_September/preset_V25_dev.json`)

**SCREEN: WEAK.** RA 80 vs B5 78 on the 100 main rows (W7 L5, net +2,
sign p = 0.77). Guard: −1, which the rule counts as no harm. Verifier drift
against v22's stored scores: max 0.0001, so the two views were scored on the
same scale.

| arm | calls | right / 100 | oracle |
|---|---|---|---|
| S5 | 5 | 68 | 82 |
| B4 | 4 | 75 | 79 |
| **B5** | 5 | **78** | 82 |
| **RA** | 5 | **80** | 88 |
| RA5 | 6 | 81 | 89 |
| RA7 | 8 | 81 | 91 |
| VRA (vote over RA's pool) | 5 | 68 | 88 |
| BQ5 (verifier, reading only) | 6 | 69 | 76 |

What held:
- **The pre-registered vote prediction held exactly.** VRA = S5 = 68: the
  diversity across readings vanishes under a vote.
- **RA − S5 = +12** at equal calls (W15 L3, p = 0.008).
- **The coverage mechanism works when it fires.** 5 of RA's 7 wins are
  rows where only the Reader's reading had the answer. Examples:
  - p2_1012: 424 on all 5 plain samples, 408 under the reading;
  - p2_1234: 260 twice under the reading, no plain sample right.

What did not, and why:
- **3 of the 5 losses are not verifier mistakes.** The right answer was only
  in plain samples 3-5, which RA drops (p2_2022, p2_169, p2_713).
- **The other 2 are near-ties at saturation.** 80% of all last-step scores
  are ≥ 0.999. On 42 of the 100 rows, two different answers share the top
  score within 0.001.
- **The verifier's resolution at the top, not coverage, is now the
  ceiling.** It turns the 88-91 covered rows into only 80-81 right, however
  the pool is arranged.

Explored after the screen, all reported. 10 variants were looked at, so
none of them is a claim:
- **Weighted votes are worse:** W-RA 73, W-B5 71.
- **Cascades:**
  - stop when the first two plain samples agree, else best of the 5 plain:
    78 at 3.29 calls;
  - the same, but best of 5 plain + 2 read: 80 at 4.58 calls.
- **Tie-breaks among near-tied samples** (min, mean, frequency): none takes
  RA above 81. The ones that "gain" +5 over B5 do so by lowering B5 to 75-76.

**Decision support.** If the dev discordance is the true effect (W 7%, L 5%),
the 100-row confirmation gives P(SUPPORTED) ≤ 0.29 and P(REFUTED) ≈ 0.33.
Recommendation: do not spend the 11 GPU-hours on it as registered.
