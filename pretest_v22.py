"""
[v22.1] Phase 2: confirm the Verifier agent on rows it has never seen, and
test whether it can also tell the Solver WHERE to think again.

WHAT PHASE 1 FOUND (score_prm_v22.py on the 700 stored v21 traces)
------------------------------------------------------------------
Qwen2.5-Math-PRM-7B separates right from wrong samples: sample-level AUROC
0.85, and on the 54 mixed rows the best right sample outscores the best wrong
one 80% of the time against a 60% chance level (the premise ledger: 4/54).
The pre-registered aggregation (min over steps, weighted vote) turned that
into only V3 - S3 = +1 row: INCONCLUSIVE. Scoring the samples by the
verifier's LAST step did far better on both dev sets:

                      main (n=100)          guard (n=40)       all (n=140)
    S3                 79                    34                 113
    B3 by last step    85  (W7 L1)           35  (W1 L0)        120  (W8 L1)
    B5 by last step    86  (W7 L1 vs S5 80)  34  (W2 L0)        120  (W9 L1)

That choice was made AFTER seeing the dev scores, among 16 variants. So it is
a hypothesis, not a result, and this script is the test: the same rule, frozen
now, on 100 FRESH gsm-symbolic-p2 rows (seed 45, disjoint from the 100 dev
rows) plus 20 fresh distractor rows.

ARMS (5 sampled CoT traces per row, one batched generate, all verifier-scored)
----
  S3, S5   self-consistency over the first 3 / all 5 samples
  B3       best of the first 3 by the verifier's last-step score  <- PRIMARY
  V3       vote over the first 3 weighted by the last-step score
  B5       best of all 5 by the last-step score
  G        REWIND. When the verifier doubts every one of the first 3 samples
           (max over them of the min step score < 0.85), take the most trusted
           one, cut it just before the step the verifier doubts most, and let
           the Solver continue from there twice. G = best of {3 samples, 2
           continuations} by last step. Elsewhere G = B3.
  RST      RESTART, the control for G at the same extra cost: on the same gated
           rows, best of {3 samples, samples 4 and 5}. Elsewhere RST = B3.

G vs RST is the multi-agent question: does it help for the Verifier to tell
the Solver where the reasoning went wrong, or is a fresh attempt as good? The
0.85 gate is fixed from the dev rows (max-min < 0.85 flags 8/13 rows whose
three samples are all wrong, and ~10% of the others).

PRE-REGISTERED, frozen before any confirmation row is generated
---------------------------------------------------------------
Main set, n=100, paired:
    B3 - S3 >= +4 rows with wins >= 2x losses -> CONFIRMED: the Verifier agent
                                                  improves reasoning
    B3 - S3 <= 0                              -> REFUTED: the dev gain was
                                                  selection noise
    otherwise                                 -> INCONCLUSIVE at this n
Secondary, reported: B3 vs S5 (does verifying 3 beat sampling 5?), B5 vs S5,
V3 vs S3, and G vs RST on the gated rows (small n; descriptive only).
Guard set: B3 - S3 >= -1 on 20 distractor rows -> no harm.

COST
----
One batched generate of 5 samples per row (~190 s on a T4, as in v21), five
verifier forward passes (~25 s), and on gated rows one more batched generate
of 2 continuations. ~4 min/row -> ~8 h for 120 rows. --max-hours and resume
exactly as in v21. Both 4-bit models fit one T4 (~11 GB); with two GPUs the
verifier goes on the second.

    python pretest_v22.py --max-hours 8.0
    python pretest_v22.py --stub --limit 6      # offline, no GPU
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import premise_ledger as L
import pretest_v21 as V21
import score_prm_v22 as P

MANIFESTS = (('main', 'pretest_data/v22_confirm_p2.json'),
             ('guard', 'pretest_data/v22_confirm_guard.json'))
OUT = 'pretest_v22.json'
GATE = 0.85          # frozen from the dev rows; see the docstring
N_REWIND = 2


def last(prm: List[float]) -> float:
    return P.agg(prm, 'last')


def low(prm: List[float]) -> float:
    return P.agg(prm, 'min')


# ---------------------------------------------------------------------------
# continuation from a prefix (the rewind)
# ---------------------------------------------------------------------------

def continue_from(client, content: str, prefix: str, k: int, temperature: float,
                  max_tokens: int) -> List[str]:
    """k sampled continuations of an assistant turn that already begins with
    `prefix`. Returns the FULL texts (prefix + continuation)."""
    if isinstance(client, V21._StubClient):
        return [prefix + client.complete(content, temperature) for _ in range(k)]
    import torch
    from Mas_solver import LOCAL_HF_SAMPLING
    client._ensure_local_model()
    tok, mdl = client._local_tokenizer, client._local_model
    msgs = [{'role': 'user', 'content': content}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefix
    ids = tok(prompt, return_tensors='pt')
    dev = next(mdl.parameters()).device
    ids = {kk: v.to(dev) for kk, v in ids.items()}
    with torch.no_grad():
        out = mdl.generate(**ids, max_new_tokens=max_tokens, do_sample=True,
                           temperature=float(temperature),
                           top_p=float(LOCAL_HF_SAMPLING.get('top_p', 0.95)),
                           top_k=int(LOCAL_HF_SAMPLING.get('top_k', 50)),
                           renormalize_logits=True, num_return_sequences=k,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
    n_in = ids['input_ids'].shape[1]
    return [prefix + tok.decode(o[n_in:], skip_special_tokens=True) for o in out]


# ---------------------------------------------------------------------------

def run_row(client, scorer, r: Dict, k: int, temperature: float, max_tokens: int) -> Dict:
    text, gold = r['text'], r['gold']
    content = V21.COT_PROMPT.format(problem=text)
    t0 = time.time()
    raws = V21.sample(client, content, k, temperature, max_tokens)
    samples = []
    for raw in raws:
        steps = P.split_steps(raw)
        samples.append({'raw': str(raw)[:4000], 'answer': V21.parse_answer(raw),
                        'n_steps': len(steps), 'prm': scorer.score(text, steps)})
    ans = [s['answer'] for s in samples]
    w_last = [last(s['prm']) for s in samples]

    rec: Dict = {'pid': r['problem_id'], 'gold': gold, 'dataset': r.get('dataset')}

    def put(name, a):
        rec[f'{name}_answer'] = a
        rec[f'{name}_correct'] = V21.correct(a, gold)

    put('S3', P.plain_vote(ans[:3]))
    put('S5', P.plain_vote(ans[:k]))
    put('B3', P.best_of_n(ans[:3], w_last[:3]))
    put('V3', P.weighted_vote(ans[:3], w_last[:3]))
    put('B5', P.best_of_n(ans[:k], w_last[:k]))

    gate = max(low(s['prm']) for s in samples[:3])
    rec['gate'] = gate
    rec['gated'] = gate < GATE
    if rec['gated']:
        src = max(range(3), key=lambda i: (low(samples[i]['prm']), -i))
        prm = samples[src]['prm']
        cut = prm.index(min(prm)) if prm else 0
        steps = P.split_steps(raws[src])          # the untruncated text that was scored
        prefix = ('\n\n'.join(steps[:cut]) + '\n\n') if cut > 0 else ''
        conts = []
        for full in continue_from(client, content, prefix, N_REWIND, temperature, max_tokens):
            # the answer is read from the NEW text first: a prefix that already
            # stated an answer must not answer for the continuation
            a = V21.parse_answer(full[len(prefix):])
            if a is None:
                a = V21.parse_answer(full)
            conts.append({'raw': str(full)[:4000], 'answer': a,
                          'prm': scorer.score(text, P.split_steps(full))})
        rec['rewind'] = {'source': src, 'cut_step': cut, 'continuations': conts}
        pool = ans[:3] + [c['answer'] for c in conts]
        wts = w_last[:3] + [last(c['prm']) for c in conts]
        put('G', P.best_of_n(pool, wts))
        put('RST', P.best_of_n(ans[:k], w_last[:k]))
    else:
        put('G', rec['B3_answer'])
        put('RST', rec['B3_answer'])

    rec['samples'] = samples
    rec['seconds'] = round(time.time() - t0, 1)
    return rec


def summarise(out: List[Dict], args) -> int:
    arms = ['S3', 'S5', 'V3', 'B3', 'B5', 'RST', 'G']
    verdict = 'no main-set rows'

    def paired(rows, x, y):
        w = sum(1 for r in rows if r.get(f'{x}_correct') and not r.get(f'{y}_correct'))
        l = sum(1 for r in rows if r.get(f'{y}_correct') and not r.get(f'{x}_correct'))
        return w, l

    for tag in ('main', 'guard'):
        rows = [r for r in out if r.get('set') == tag]
        if not rows:
            continue
        n = len(rows)
        gated = [r for r in rows if r.get('gated')]
        print('\n' + '=' * 72)
        print(f"  {tag.upper()} SET  n={n}   gated (verifier doubts all 3) {len(gated)}/{n}")
        for a in arms:
            s = sum(1 for r in rows if r.get(f'{a}_correct'))
            print(f"    {a:4s} {s:3d} = {100*s/n:5.1f}%")
        print()
        for x, y in (('B3', 'S3'), ('B3', 'S5'), ('B5', 'S5'), ('V3', 'S3'), ('G', 'RST'), ('G', 'B3')):
            w, l = paired(rows, x, y)
            print(f"    {x:4s} vs {y:4s}: W={w:2d} L={l:2d} net={w-l:+d}")
        if gated:
            cov3 = sum(1 for r in gated if any(V21.correct(s['answer'], r['gold'])
                                               for s in r['samples'][:3]))
            covG = sum(1 for r in gated if any(V21.correct(c['answer'], r['gold'])
                                               for c in r['rewind']['continuations']))
            covR = sum(1 for r in gated if any(V21.correct(s['answer'], r['gold'])
                                               for s in r['samples'][3:5]))
            print(f"    on gated rows: a right answer among the 3 samples {cov3}/{len(gated)}, "
                  f"among the 2 rewinds {covG}, among the 2 restarts {covR}")
        if tag == 'main':
            s3 = sum(1 for r in rows if r.get('S3_correct'))
            b3 = sum(1 for r in rows if r.get('B3_correct'))
            w, l = paired(rows, 'B3', 'S3')
            if n != 100 and not args.stub:
                verdict = f"main n={n}, not the pre-registered 100 -- report as measured, no verdict"
            elif b3 - s3 <= 0:
                verdict = "REFUTED: B3 <= S3 -- the dev gain was selection noise"
            elif b3 - s3 >= 4 and w >= 2 * l:
                verdict = "CONFIRMED: the Verifier agent improves on self-consistency on fresh rows"
            else:
                verdict = "INCONCLUSIVE: B3 > S3 but short of the pre-registered bar"
        if tag == 'guard':
            w, l = paired(rows, 'B3', 'S3')
            print(f"    guard: B3 vs S3 net={w-l:+d} -> "
                  + ("no harm" if w - l >= -1 else "HARM on distractor rows"))
    print('\n  pre-registered reading:\n    ' + verdict)
    print(f"\n  saved: {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--preset', default='qwen_math7b_mixed')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--max-hours', type=float, default=0.0)
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()

    if args.summary_only:
        with open(args.out, encoding='utf-8') as fh:
            return summarise(json.load(fh)['rows'], args)

    allrows = []
    for tag, path in MANIFESTS:
        with open(path, encoding='utf-8') as fh:
            rows = json.load(fh)['rows']
        allrows += [dict(r, set=tag) for r in (rows[:args.limit] if args.limit else rows)]
    print(f"rows: {len(allrows)}   k={args.k} T={args.temperature}  gate={GATE}")

    if args.stub:
        client = V21._StubClient(allrows)
        scorer = P._StubScorer({})
    else:
        import torch
        two = torch.cuda.device_count() > 1
        client = V21.build_client(args.preset, None)
        scorer = P.PRMScorer(P.PRM, four_bit=True, device_map={'': 1 if two else 0})

    prior: Dict[str, Dict] = {}
    if os.path.exists(args.out) and args.resume and not args.stub:
        with open(args.out, encoding='utf-8') as fh:
            prior = {r['pid']: r for r in json.load(fh).get('rows', [])}
        print(f"resuming: {len(prior)} rows already in {args.out}")

    out: List[Dict] = []
    t0, ran, left = time.time(), 0, 0
    for i, r in enumerate(allrows, 1):
        if r['problem_id'] in prior and 'G_correct' in prior[r['problem_id']]:
            out.append(prior[r['problem_id']])
            continue
        if args.max_hours and time.time() - t0 > args.max_hours * 3600:
            left = len(allrows) - i + 1
            print(f"\n  --max-hours reached with {left} row(s) left; re-run the identical "
                  f"command to resume.")
            break
        rec = run_row(client, scorer, r, args.k, args.temperature, args.max_tokens)
        rec['set'] = r['set']
        out.append(rec)
        ran += 1
        el = time.time() - t0
        f = lambda a: 'OK ' if rec.get(f'{a}_correct') else 'BAD'
        print(f"[{i:3d}/{len(allrows)}] {r['set']:5s} {str(r['problem_id'])[:22]:22s} "
              f"S3={f('S3')} S5={f('S5')} B3={f('B3')} G={f('G')}"
              f"{' gated' if rec['gated'] else '      '}  {el/ran:5.0f}s/row", flush=True)
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump({'k': args.k, 'temperature': args.temperature, 'gate': GATE,
                       'rows': out}, fh, ensure_ascii=False)
    if left:
        print(f"\n  PARTIAL: {len(out)}/{len(allrows)} rows. No verdict is read off a partial run.")
    return summarise(out, args)


if __name__ == '__main__':
    sys.exit(main())
