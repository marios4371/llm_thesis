# v24: the Reader agent, sentence-anchored situation QAs (plan as of 2026-09-27)

This is the next accuracy experiment. It is not a verifier: nothing is
scored, nothing is selected, and every problem gets the same treatment. It
builds on a published method, MathWorld, and takes up the limitation that
paper names. It is pre-registered and not yet run.

| file | what it is |
|---|---|
| `situation_reader.py` | the method: the Reader's prompt, sanitation, and the Solver's input for every arm. Deterministic, no model |
| `pretest_v24.py` | the pre-registered runner: dev pass on stored rows, fresh confirmation, summary |
| `v24_diagnosis.py` | the offline evidence that motivates it, and the power of the verdict rule (no GPU) |
| `test_v24.py` | 70 offline checks (`python test_v24.py`) |

## 1. Why: what is left is misreading, and it is systematic

Every number below comes from `python v24_diagnosis.py`, which reads the
stored CoT samples of the v21 and v22 main rows (200 P2 rows, 5 samples each).

- **Resampling cannot reach it.** On 28 of the 200 rows, none of the 5
  samples is right (v22: 18/100). No vote, verifier or restart policy that
  picks among samples can recover these rows.
- **The wrong samples agree.** On 12 of those 28 rows, at least 3 of the 5
  samples give the same wrong answer. The model is not guessing; it reads the
  problem one wrong way.
- **The errors follow the template, not the numbers.** GSM-Symbolic keeps a
  template's 50 instances together (row index // 50).
  - 68% of the row-level accuracy variance lies between templates.
  - The 10 worst of 49 templates hold 54% of all wrong samples.
  - 10 templates sit at 30% or less per sample, and 16 at 90% or more.

  The same situation is misread whatever the numbers. Examples:
  - "64 white animals, half of which were rabbits": the white animals that
    are not rabbits are dropped from the total. This is the modal answer in
    all three instances (3, 5 and 4 of the 5 samples).
  - "leave 12 feet between every plant (including the plant's width)": 0%
    over 5 instances.

This is what v21 called role errors, and what Plan-and-Solve and DUP call
semantic misunderstanding. The v22 post-mortem showed the failure of the one
repair aimed at it: pointing at the sentence (V2P) gets it re-read the same
wrong way.

**Consequence for evaluation.** A P2 result must count templates, not only
rows. Every reading below does.

## 2. The method

```
problem ─► split into sentences S1..Sn (deterministic, the ledger's splitter)
        ─► Reader  (Qwen2.5-7B-Instruct, greedy, 3 GSM8K-train demos):
             for each sentence, 1-3 Q/A pairs that make its situation explicit
             (whose each number is, every part of a split group incl. "the rest",
             the reference of each comparison, rates, the order of changes),
             then "Asked: <quantity, unit>". It never answers the question.
        ─► Solver  (Qwen2.5-Math-7B-Instruct, the unchanged CoT prompt, k=5, T=0.8):
             the problem slot holds each sentence followed by its notes
        ─► answer  = plurality vote over the 5 samples (SC@5), or any one sample
```

### What it builds on

The existing method is **MathWorld** (Opedal et al., Findings of ACL 2023).
It gives math story problems a world model: containers plus TRANSFER, RATE,
COMPARISON and PARTWHOLE relations. From *gold* world models it generated
question-answer pairs about the situation.

| GPT-3, 1 in-context example (their Table 3) | accuracy |
|---|---|
| the original problem | 70.8 |
| 2 gold situation Q/A pairs appended at the end | 71.8 |
| the same pairs placed after the sentence they describe | **78.6** |

The limitation is the gold. The automatic route, Codex parsing into world
models, solved about a third of the easiest dataset. Their Limitations
section says: *"An obvious limitation of this work is the low performance on
the task of solving MSPs"* and leaves stronger parsers to future work.

**Plan-and-Solve** (Wang et al., ACL 2023) names the same gap from the
solver's side. Its Limitations say that planning fixes calculation and
missing-step errors, *"but the semantic misunderstanding errors still
remain"*.

### What is new, and what is not

**Not new** (used as components or as controls):
- the relation types (MathWorld; schema-based instruction in education);
- situation Q/A placed at the sentence (MathWorld, gold only);
- "understand, then solve" in one model (PS+, DUP). The `Asked:` line is
  DUP's core-question stage;
- a problem-level schema label (SBI-RAG: one schema per problem, judged by an
  LLM rather than by accuracy);
- a decomposer model plus a solver model (Socratic CoT: trained, and its
  sub-questions are solution steps, not readings).

**New in v24:**
1. **No gold, no parser.** A Reader agent writes the sentence-anchored
   situation Q/As directly. Q/A pairs are the representation, as in QA-based
   meaning representations (QAMR), which people and LLMs produce reliably
   where formal parses fail. This turns MathWorld's gold-only effect into a
   stage a system can run.
2. **Reading and solving are split across two models of the same size.** The
   general-language model reads and the math model solves. This is the
   thesis's role-by-competence split (v12.3 preset).
3. **It is tested where semantic misunderstanding dominates.** That means P2,
   a 7B solver, and equal solver samples. The test carries MathWorld's
   placement control on self-generated notes (END) and a single-agent control
   (SELF).

**Why it is not one of the rejected ideas:**
- v20 re-rendered the problem, and the math model refused to paraphrase.
- v21 and v22 (V2P) quoted the problem's own sentences.
- v23 passed on a verifier-rejected step.

All of these either point at the text or depend on a failure signal. v24
adds explicit situation facts, written by a different agent, on every row.

**The closest negative evidence, and why it does not apply.** "Beyond
Pattern Recognition" (2025) revealed problems piece by piece with a JSON
state, and small models collapsed (Llama-3B from 0.64 to 0.05). v24 hides
nothing: the Reader sees the whole problem, and so does the Solver.

**Before writing related work, read these.** They were found by search and
only partly read here:
- "A Four-Stage Decomposition of Word-Problem Solving and Mechanistic
  Fragility in LLM Math Reasoning" (arXiv 2609.17804, EMNLP 2026 Findings).
  It localises distractor fragility to an internal operation-planning stage,
  and leaves a deployable repair to future work;
- SBI-RAG (arXiv 2410.13293);
- "Do Language Models Exhibit the Same Cognitive Biases in Problem Solving as
  Human Learners?" (arXiv 2401.18070).

None of these appeared to generate sentence-anchored situation Q/As with an
agent for a separate solver. That still has to be checked.

## 3. Pre-registration (frozen in `pretest_v24.py` before any ASQ sample exists)

**Arms.** Every arm uses k=5 solver samples with the v22 settings (T=0.8,
top-p 0.95, top-k 50, 1024 tokens).

| arm | what the Solver gets |
|---|---|
| C | the plain CoT prompt of every earlier version (on dev rows: the 5 samples v22 stored) |
| **ASQ** | **the Reader's notes interleaved after each sentence (v24)** |
| END | the same notes appended after the unchanged problem: the placement control |
| SELF | one call, told to read sentence by sentence first: the single-agent control |
| ASQM | optional: ASQ with the math model as the Reader: the heterogeneity control |

**Metric.** Per-sample accuracy is how often one reasoning chain is right, so
it is the direct measure of reasoning. SC@5 is the system metric.

**Dev pass** (`--dev`). It runs on the 100 main + 20 guard rows of the v22
confirmation, with their stored C samples, so no C is drawn.

| reading | rule |
|---|---|
| **PRIMARY** (main) | ASQ − C ≥ **+4.0 pp** per sample, **and** a paired sign-flip permutation test over rows gives p < 0.05, **and** the gain stays ≥ +2.0 pp after dropping the single template that helps most → **SUPPORTED** |
| | ASQ − C ≤ 0 → **REFUTED**; anything else → INCONCLUSIVE |
| SYSTEM (main) | SC@5 ASQ vs SC@5 C: net ≥ +4 rows with W ≥ 2L → the Reader–Solver system beats SC@5 at equal solver samples |
| MECHANISM | ASQ − END ≥ +2 pp → the position matters (MathWorld's effect holds for self-generated notes); END − C ≥ +2 pp with \|ASQ − END\| < 2 → the content matters, not the position |
| MULTI-AGENT | ASQ − SELF ≥ +2 pp → a separate Reader beats telling the Solver to read carefully |
| GUARD (20 GSM-Plus distractor rows) | SC@5 ASQ − SC@5 C ≥ −1 row → no harm |

The runner also prints:
- a 95% bootstrap interval that resamples templates;
- rows fixed (C ≤ 1/5 → ASQ ≥ 3/5) and rows broken (C ≥ 4/5 → ASQ ≤ 2/5);
- C → ASQ on the 8 templates hardest for C;
- how often a Reader note already contains the gold, to show whether the
  Reader is solving rather than reading.

**Fresh confirmation** (default mode). It runs on 100 new P2 rows (seed 47,
disjoint from every v20–v23 row) plus 20 guard rows. C is drawn in the same
session, and the same rules apply.

**Power of the PRIMARY rule** (`python v24_diagnosis.py --power`). ASQ's
samples are simulated from each row's stored C rate plus a true effect. C
stays the stored samples, as in the real dev pass. 300 simulations each:

| true effect | mean gain | SUPPORTED | INCONCLUSIVE | REFUTED |
|---|---|---|---|---|
| none (ASQ = C) | +0.0 pp | **0.00** | 0.48 | 0.52 |
| +0.25 on the misread templates (C ≤ 30%) | +5.3 pp | 0.77 | 0.23 | 0.00 |
| +0.40 on misread templates, −0.05 on easy rows | +6.0 pp | 0.72 | 0.28 | 0.00 |
| +0.06 on every row | +4.0 pp | 0.53 | 0.47 | 0.00 |
| −0.03 on every row | −2.4 pp | 0.00 | 0.05 | **0.95** |

The rule almost never reads SUPPORTED without a gain. It sees a +5–6 pp gain
about three times in four, and a small loss reads REFUTED.

The first version of this rule was replaced **before any ASQ sample
existed**. It used a row sign test plus "more templates improve than worsen".
The same simulation gave it 0.2 power on the most likely effect shape: big
gains on a few misread templates, small losses on easy rows. The reason is
that a sign test counts a row that goes from 0/5 to 4/5 the same as a row
that loses one sample.

**What it takes to pass, roughly (not part of the rule).** C is at 59.6% per
sample on these rows. +4 pp means about one misread template in eight is
read right, net of any row the notes break.

## 4. How to run (Kaggle T4×2 or Colab, the v22 setup)

**On Kaggle:**
1. Import `MAS_SHT_Kaggle_v24.ipynb` (File → Import Notebook). It is Cell 1 of
   `MAS_SHT_Kaggle.ipynb`, unchanged, plus one cell that runs the offline
   checks and then the dev pass.
2. Set GPU T4 ×2, Internet on, and the `HF_API_KEY` secret.
3. Run it with **Save & Run All**.

The commands it runs, and the rest of the sequence:

```bash
python test_v24.py                                    # offline, ~30 s
python v24_diagnosis.py --power                       # offline, section 1 + the power table, ~15 s

# 1) dev pass: Reader + ASQ on the 120 stored v22 rows, ~6.5 h, one session
python pretest_v24.py --dev --max-hours 8.0
python pretest_v24.py --dev --summary-only            # re-read, no GPU

# 1b) the controls, on the same rows and the same stored readings (optional, ~5 h each)
python pretest_v24.py --dev --arms ASQ,END --max-hours 8.0
python pretest_v24.py --dev --arms ASQ,END,SELF --max-hours 8.0

# 2) if the dev pass reads SUPPORTED: fresh confirmation, C + ASQ, ~11 h (two sessions)
python pretest_v24.py --build-manifest                # once, needs `datasets`; commit the 2 files
python pretest_v24.py --max-hours 8.0                 # re-run the identical command to resume
python pretest_v24.py --summary-only
```

Time estimate, from v22's measured ~193 s per row for 5 samples plus PRM:
- ~150 s for 5 solver samples;
- ~30–40 s for one greedy Reader call.

The Reader runs on the second GPU (on one GPU both 4-bit models fit).
Download `pretest_v24_dev.json` (or `pretest_v24.json`) after every session.
A stub run (`--stub`) writes `pretest_v24_stub.json`, never the real files.

## 5. Risks

- **The Reader may share the misreading.** It is a 7B model too. The bet is
  that a general model, asked only to read one sentence at a time, reads
  better than a math model in solving mode. If it does not, ASQ will copy the
  wrong reading into all 5 samples. That is why "rows broken" is printed.
- **Distractor amplification.** On GSM-Plus rows the Reader also annotates
  the distractor sentence, which makes it more salient. The guard reading
  exists for this.
- **Ceiling from ill-posed templates.** About 3 of the hard templates have
  ambiguous or defective golds (signatures; the house-fee order; integer
  division in the field templates). No reading fixes them.
- **Clustering.** P2 has only ~50 templates, so the effective sample size is
  closer to the template count than to 100. The template conditions are there
  so that a single template cannot carry the verdict.

## 6. How either outcome goes into the thesis

- **SUPPORTED.**
  - It is the thesis's original method contribution: an agent that turns
    MathWorld's gold-only situation model into a running stage, and measurably
    improves the reasoning of a 7B solver where semantic misunderstanding
    dominates.
  - With END: whether the position effect MathWorld found holds when the
    notes are self-generated.
- **REFUTED.** Together with v20 (paraphrase), v22 (quoting) and v21 (role
  errors), it closes the repair story at the reading level. Neither pointing
  at the text nor spelling out its situation helps a 7B solver, which locates
  the bottleneck in the reader's own situation model. The template-clustering
  finding stands either way, as a methodological point for GSM-Symbolic
  evaluation.

## References

- Opedal, Stoehr, Saparov, Sachan. World Models for Math Story Problems. Findings of ACL 2023. https://arxiv.org/abs/2306.04347
- Wang et al. Plan-and-Solve Prompting. ACL 2023. https://arxiv.org/abs/2305.04091
- Zhong et al. Achieving >97% on GSM8K: Deeply Understanding the Problems (DUP). Frontiers of Computer Science 2025. https://arxiv.org/abs/2404.14963
- Michael et al. Crowdsourcing Question-Answer Meaning Representations (QAMR). NAACL 2018.
- Shridhar, Stolfo, Sachan. Distilling Reasoning Capabilities into Smaller Language Models (Socratic CoT). Findings of ACL 2023.
- SBI-RAG. https://arxiv.org/abs/2410.13293
- Beyond Pattern Recognition: Probing Mental Representations of LMs. https://arxiv.org/abs/2502.16717
- Mirzadeh et al. GSM-Symbolic. ICLR 2025. https://arxiv.org/abs/2410.05229
- Hegarty, Mayer, Monk. Comprehension of arithmetic word problems. J. Educational Psychology 1995.
- A Four-Stage Decomposition of Word-Problem Solving. https://arxiv.org/abs/2609.17804
