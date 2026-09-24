"""
[v22.0] Phase 1: can a step-level VERIFIER agent see the errors that voting,
SIV and the premise ledger could not?

WHY A VERIFIER, AND WHY THIS ONE
--------------------------------
The v21 post-mortem (results_September/preset_V21.json, 700 CoT traces on
gsm-symbolic-p2) settled what the remaining errors are: ROLE errors, the right
numbers used in the wrong role. 84% of samples spend every stated quantity and
wrong samples are exactly as "complete" as right ones (47/54 mixed rows tie),
e.g. p2_2185 computes the wasps correctly and then counts the 32 rabbits where
the 64 white animals belong. Every check this project has built reads answers
(voting, v15.8's 2880 rules) or numbers (SIV, the ledger), and a role error is
invisible to both. It is visible only to something that reads the STEP in the
context of the problem.

Qwen2.5-Math-PRM-7B is a process reward model trained to score exactly that,
step by step, on the outputs of the SAME policy this thesis uses
(Qwen2.5-Math-7B-Instruct). It is a second agent with a different job: it never
solves, it judges each step. In the MAS it is the Verifier.

WHY THIS PHASE IS CHEAP AND CLEAN
---------------------------------
It generates nothing. It scores the 5 stored samples per row from the v21 run
(forward passes only), so every arm below is computed on the SAME samples that
produced S3 = 79 and S5 = 80. The comparison is paired by construction and
costs ~30 GPU-minutes instead of 8 hours.

ARMS (all views of the stored samples + their step rewards)
----
  S3, S5    plain self-consistency, recomputed (must match the v21 run)
  V3, V5    verifier-weighted vote: each sample votes with weight
            min(step rewards); the answer with the largest total wins
  B3, B5    best-of-N: the single sample with the highest min step reward

PRE-REGISTERED, written before any score exists
-----------------------------------------------
Primary, main set (P2, the same 100 rows):
    V3 - S3 >= +4 rows with wins >= 2x losses
        -> the verifier sees what voting cannot: build Phase 2
           (fresh rows + verifier-directed regeneration)
    V3 <= S3
        -> KILL: a matched step verifier adds nothing on P2 at 7B
Secondary (reported, not decisive): V5 vs S5, B3/B5, and on the 54 "mixed"
rows (right and wrong samples both present) the fraction where the best right
sample outscores the best wrong one. The guard set (GSM-Plus distractors) is
reported for harm only.

Aggregation is fixed in advance to min-over-steps (Lightman et al. 2023; a
chain is as good as its worst step). Other aggregations are printed as
exploratory and must not be picked from this run.

RUN
---
    python score_prm_v22.py                     # Kaggle, T4 x2, ~30 min
    python score_prm_v22.py --analyse-only      # offline, from the scores file
    python score_prm_v22.py --stub --limit 10   # offline plumbing check

The model is loaded with trust_remote_code=True, which the official model card
requires: it executes the model's own modelling file from the Qwen repo.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional

import premise_ledger as L

TRACES = 'pretest_data/v21_traces.json'
MANIFESTS = ('pretest_data/v20_problems.json', 'pretest_data/v21_guard_distraction.json')
OUT = 'prm_scores_v22.json'
PRM = 'Qwen/Qwen2.5-Math-PRM-7B'
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
SEP = '<extra_0>'


def correct(pred: Optional[float], gold: Optional[float]) -> bool:
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= max(1e-3, 1e-4 * abs(gold))


def split_steps(raw: str) -> List[str]:
    """The PRM's own convention: steps are separated by a blank line."""
    steps = [s.strip() for s in re.split(r'\n\s*\n', str(raw).strip()) if s.strip()]
    return steps or [str(raw).strip() or '(empty)']


# ---------------------------------------------------------------------------
# the verifier
# ---------------------------------------------------------------------------

class PRMScorer:
    def __init__(self, name: str, four_bit: bool = True):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        kw = dict(trust_remote_code=True, device_map='auto')
        if four_bit:
            from transformers import BitsAndBytesConfig
            # the reward head stays in full precision: it is two small linear
            # layers and it is the part that produces the number we read
            kw['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=['score', 'lm_head'])
        else:
            kw['torch_dtype'] = torch.float16
        t0 = time.time()
        self.model = AutoModel.from_pretrained(name, **kw).eval()
        self.sep_id = self.tok.encode(SEP)[0]
        print(f"  verifier {name} loaded in {time.time()-t0:.0f}s "
              f"({'4-bit NF4' if four_bit else 'fp16'})", flush=True)

    def score(self, problem: str, steps: List[str]) -> List[float]:
        torch = self.torch
        msgs = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': problem},
                {'role': 'assistant', 'content': SEP.join(steps) + SEP}]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=False)
        ids = self.tok.encode(text, return_tensors='pt')
        dev = next(self.model.parameters()).device
        ids = ids.to(dev)
        with torch.no_grad():
            logits = self.model(input_ids=ids)[0]
        probs = torch.softmax(logits.float(), dim=-1)[0]      # [seq, 2]
        mask = (ids[0] == self.sep_id)
        return probs[mask][:, 1].cpu().tolist()


class _StubScorer:
    """Offline stand-in: right-ish traces score a bit higher, with noise."""
    def __init__(self, golds: Dict[str, float]):
        self.rng = random.Random(0)

    def score(self, problem: str, steps: List[str]) -> List[float]:
        return [min(1.0, max(0.0, self.rng.gauss(0.8, 0.15))) for _ in steps]


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def agg(rewards: List[float], how: str) -> float:
    if not rewards:
        return 0.0
    if how == 'min':
        return min(rewards)
    if how == 'last':
        return rewards[-1]
    if how == 'prod':
        return math.exp(sum(math.log(max(r, 1e-9)) for r in rewards))
    if how == 'mean':
        return sum(rewards) / len(rewards)
    raise ValueError(how)


def plain_vote(answers: List[Optional[float]]) -> Optional[float]:
    g = L.clusters(answers)
    return answers[g[0][0]] if g else None


def weighted_vote(answers: List[Optional[float]], w: List[float]) -> Optional[float]:
    groups = L.clusters(answers)
    if not groups:
        return None
    best = max(groups, key=lambda g: (sum(w[i] for i in g), len(g), -g[0]))
    return answers[best[0]]


def best_of_n(answers: List[Optional[float]], w: List[float]) -> Optional[float]:
    idx = [i for i, a in enumerate(answers) if a is not None]
    if not idx:
        return None
    return answers[max(idx, key=lambda i: (w[i], -i))]


# ---------------------------------------------------------------------------

def analyse(scored: List[Dict], how_primary: str = 'min') -> int:
    def arms(r, how):
        ans = [s['answer'] for s in r['samples']]
        w = [agg(s['prm'], how) for s in r['samples']]
        return {'S3': plain_vote(ans[:3]), 'S5': plain_vote(ans[:5]),
                'V3': weighted_vote(ans[:3], w[:3]), 'V5': weighted_vote(ans[:5], w[:5]),
                'B3': best_of_n(ans[:3], w[:3]), 'B5': best_of_n(ans[:5], w[:5])}

    verdict = None
    for tag in ('main', 'guard'):
        rows = [r for r in scored if r.get('set') == tag]
        if not rows:
            continue
        n = len(rows)
        print('\n' + '=' * 72)
        print(f"  {tag.upper()} SET  n={n}   aggregation (pre-registered): {how_primary}")
        res = [{k: correct(v, r['gold']) for k, v in arms(r, how_primary).items()} for r in rows]
        mism = sum(1 for r, x in zip(rows, res)
                   if 'S3_correct' in r and x['S3'] != r['S3_correct'])
        if mism:
            print(f"  WARNING: recomputed S3 differs from the v21 run on {mism} rows")
        for a in ('S3', 'S5', 'V3', 'V5', 'B3', 'B5'):
            s = sum(x[a] for x in res)
            print(f"    {a}  {s:3d} = {100*s/n:5.1f}%")

        def paired(x, y):
            w = sum(1 for q in res if q[x] and not q[y])
            l = sum(1 for q in res if q[y] and not q[x])
            return w, l

        print()
        for x, y in (('V3', 'S3'), ('V5', 'S5'), ('B3', 'S3'), ('B5', 'S5'), ('V3', 'S5')):
            w, l = paired(x, y)
            print(f"    {x} vs {y}: W={w:2d} L={l:2d} net={w-l:+d}")

        mixed = right_higher = 0
        chance = 0.0
        for r in rows:
            R = [agg(s['prm'], how_primary) for s in r['samples'] if correct(s['answer'], r['gold'])]
            W = [agg(s['prm'], how_primary) for s in r['samples']
                 if s['answer'] is not None and not correct(s['answer'], r['gold'])]
            if R and W:
                mixed += 1
                right_higher += max(R) > max(W)
                # with scores unrelated to correctness, the best of |R| right
                # samples beats the best of |W| wrong ones with prob |R|/(|R|+|W|)
                chance += len(R) / (len(R) + len(W))
        if mixed:
            print(f"    mixed rows: best right sample outscores best wrong one in "
                  f"{right_higher}/{mixed} = {100*right_higher/mixed:.0f}%   "
                  f"(chance level {100*chance/mixed:.0f}%)")

        print('    exploratory aggregations (NOT for choosing -- report only):')
        for how in ('last', 'prod', 'mean'):
            rr = [{k: correct(v, r['gold']) for k, v in arms(r, how).items()} for r in rows]
            print(f"      {how:5s}: V3 {sum(x['V3'] for x in rr):3d}  V5 {sum(x['V5'] for x in rr):3d}"
                  f"  B3 {sum(x['B3'] for x in rr):3d}  B5 {sum(x['B5'] for x in rr):3d}")

        if tag == 'main':
            s3, v3 = sum(x['S3'] for x in res), sum(x['V3'] for x in res)
            w, l = paired('V3', 'S3')
            if v3 <= s3:
                verdict = "KILL: V3 <= S3 -- the matched verifier adds nothing on P2"
            elif v3 - s3 >= 4 and w >= 2 * l:
                verdict = "PASS: V3 beats S3 by >= 4 rows -> build Phase 2"
            else:
                verdict = "INCONCLUSIVE: V3 > S3 but short of the pass bar"
    print('\n  pre-registered reading:\n    ' + (verdict or 'no main-set rows'))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', default=TRACES)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--model', default=PRM)
    ap.add_argument('--fp16', action='store_true', help='skip 4-bit, load fp16 over both GPUs')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--analyse-only', action='store_true')
    args = ap.parse_args()

    if args.analyse_only:
        with open(args.out, encoding='utf-8') as fh:
            return analyse(json.load(fh)['rows'])

    texts: Dict[str, str] = {}
    for m in MANIFESTS:
        with open(m, encoding='utf-8') as fh:
            for r in json.load(fh)['rows']:
                texts[r['problem_id']] = r['text']
    with open(args.traces, encoding='utf-8') as fh:
        rows = json.load(fh)['rows']
    if args.limit:
        rows = rows[:args.limit]
    print(f"rows: {len(rows)}  samples: {sum(len(r['samples']) for r in rows)}")

    prior: Dict[str, Dict] = {}
    if os.path.exists(args.out) and not args.stub:
        with open(args.out, encoding='utf-8') as fh:
            prior = {r['pid']: r for r in json.load(fh)['rows']}
        print(f"resuming: {len(prior)} rows already scored")

    scorer = _StubScorer({}) if args.stub else PRMScorer(args.model, four_bit=not args.fp16)
    out, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        if r['pid'] in prior:
            out.append(prior[r['pid']])
            continue
        prob = texts[r['pid']]
        rec = {k: r[k] for k in ('pid', 'gold', 'set', 'S3_correct', 'S5_correct') if k in r}
        rec['samples'] = []
        for s in r['samples']:
            steps = split_steps(s['raw'])
            rec['samples'].append({'answer': s['answer'], 'n_steps': len(steps),
                                   'prm': scorer.score(prob, steps)})
        for extra in ('R', 'R0'):
            if extra in r:
                rec[extra] = {'answer': r[extra]['answer'],
                              'prm': scorer.score(prob, split_steps(r[extra]['raw']))}
        out.append(rec)
        if i % 10 == 0 or i == len(rows):
            print(f"  [{i:3d}/{len(rows)}] {(time.time()-t0)/max(1,i-len(prior)):.1f}s/row", flush=True)
            with open(args.out, 'w', encoding='utf-8') as fh:
                json.dump({'model': args.model, 'rows': out}, fh)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump({'model': args.model, 'rows': out}, fh)
    print(f"saved {args.out}")
    return analyse(out)


if __name__ == '__main__':
    sys.exit(main())
