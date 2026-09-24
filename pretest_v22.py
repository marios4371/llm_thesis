"""
[v22.2] Verifier-to-Problem grounding (V2P): the Verifier finds WHERE the
reasoning broke, the grounding layer finds WHICH SENTENCE OF THE PROBLEM that
step misread, and the Solver re-solves from scratch with that sentence in
front of it.

WHAT IS AND IS NOT NEW HERE (checked 2026-09-24)
-----------------------------------------------
Not new, and used here only as baselines:
  * a step verifier choosing among samples (best-of-N with a PRM): Lightman et
    al. 2023, Math-Shepherd, Qwen's own PRM paper;
  * rewinding to the verifier's doubted step and continuing from the model's
    own prefix: StepCo (ACL 2025), PRM-guided backtracking (2605.25143).
The gap V2P targets: every localise-then-repair method so far repairs on the
OUTPUT side -- it keeps the model's own prefix and asks it to continue or
revise. Two measured facts say that is the wrong side for this model:
  * the v21 post-mortem: the residual P2 errors are ROLE errors, a sentence of
    the problem read wrongly, not a wrong operation;
  * self-correction anchors on its own wrong solution ("LLMs cannot
    self-correct reasoning yet"; "The Self-Correction Illusion", 2606.05976).
And the one method that does repair on the INPUT side, RCoT, finds the
misread condition by asking the LLM to reconstruct the problem -- which this
thesis measured a math-specialised 7B cannot do (it refused to restate a
problem 2/2, v20 smoke test).
V2P closes that gap with parts this thesis already owns: the learned verifier
localises the step (solution side), the deterministic number grounding of
premise_ledger.py maps that step's numbers to the sentences they come from
(problem side), and the Solver never sees its failed attempt -- it gets the
problem again with those sentences quoted. No agent is ever asked to critique.

Offline evidence it can work (dev rows): on the 13 rows where all 3 samples
were wrong, the verifier's most-doubted step was the real error on ~7 (e.g.
"60 x 12 = 720 puppies" where the text says 11; "3 x (22+9) cheerleaders"
where it is 3 per group), and in v21 a quoted-sentence note fixed 7/9 rows
against 5/9 for a generic note.

ARMS
----
  S3, S5   self-consistency over the first 3 / all 5 samples
  B3       best of the first 3 by the verifier's last-step score
  B5       best of all 5
  Repair arms, only on GATED rows (the verifier doubts all 3 samples: max over
  them of the min step score < 0.85; elsewhere every repair arm = B3), each
  spending the same 2 extra samples and each = best of {3 samples + 2 new}:
  RST      restart: samples 4 and 5, blind
  RW       rewind: continue twice from just before the most-doubted step of
           the most-trusted sample (StepCo-style; the output-side baseline)
  V2P      re-solve twice from scratch with the grounded sentences quoted

PRE-REGISTERED, frozen before any confirmation row is generated
---------------------------------------------------------------
Confirmation set (100 fresh P2 rows, seed 45):
    B3 - S3 >= +4 rows, wins >= 2x losses -> CONFIRMED (verifier agent)
    B3 - S3 <= 0                          -> REFUTED
Novel claim, on the gated rows pooled over the dev pass and the confirmation
run (the V2P rule is fixed before either generates):
    V2P - RST >= +3 rows AND V2P >= RW    -> V2P supported
    V2P <= RST                            -> V2P refuted
Guard (distractor) rows: V2P - B3 >= -1   -> no harm from quoting a sentence

MODES
-----
    python pretest_v22.py --dev --max-hours 3.5     # ~20 gated dev rows, ~1.5 h
    python pretest_v22.py --max-hours 8.0           # confirmation, ~8-9 h
    python pretest_v22.py --stub --limit 6          # offline, no GPU
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Sequence

import premise_ledger as L
import pretest_v21 as V21
import score_prm_v22 as P

MANIFESTS = (('main', 'pretest_data/v22_confirm_p2.json'),
             ('guard', 'pretest_data/v22_confirm_guard.json'))
DEV_TRACES = 'pretest_data/v21_traces.json'
DEV_PRM = 'pretest_data/v22_dev_prm.json'
DEV_MANIFESTS = ('pretest_data/v20_problems.json', 'pretest_data/v21_guard_distraction.json')
OUT = 'pretest_v22.json'
OUT_DEV = 'pretest_v22_dev.json'
GATE = 0.85          # frozen from the dev rows; see the docstring
N_EXTRA = 2
V2P_MARKER = 'pay close attention to this part of the problem:'
_ANY_NUMBER = re.compile(r"(?<![A-Za-z_])\d+(?:,\d{3})*(?:\.\d+)?")


def last(prm: List[float]) -> float:
    return P.agg(prm, 'last')


def low(prm: List[float]) -> float:
    return P.agg(prm, 'min')


# ---------------------------------------------------------------------------
# grounding: doubted step -> problem sentences
# ---------------------------------------------------------------------------

def implicated_sentences(text: str, step: str, max_sentences: int = 2) -> List[str]:
    """The problem sentences whose stated quantities appear in `step`.

    Every number in the step counts, prose included: "the ratio is 6:6" is a
    reading of the problem even though it is not an operation. Sentences are
    ranked by how many of their quantities the step uses, ties in text order.
    """
    sents = L.split_sentences(text)
    nums = []
    for t in _ANY_NUMBER.findall(step or ''):
        try:
            nums.append(float(t.replace(',', '')))
        except ValueError:
            pass
    score: Dict[int, int] = {}
    for p in L.extract_premises(text):
        if any(L._close(f, n) for f in p.forms for n in nums):
            for s in p.sentences:
                score[s] = score.get(s, 0) + 1
    ranked = sorted(score, key=lambda s: (-score[s], s))[:max_sentences]
    return [sents[i] for i in sorted(ranked) if 0 <= i < len(sents)]


def v2p_note(sentences: Sequence[str]) -> str:
    if not sentences:
        return V21.GENERIC_NOTE
    return f"Note: {V2P_MARKER} " + ' '.join(f'"{s}"' for s in sentences)


# ---------------------------------------------------------------------------
# generation helpers
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


def _scored(scorer, text: str, full: str, answer_from: str = None) -> Dict:
    a = V21.parse_answer(answer_from) if answer_from is not None else None
    if a is None:
        a = V21.parse_answer(full)
    return {'raw': str(full)[:4000], 'answer': a,
            'prm': scorer.score(text, P.split_steps(full))}


# ---------------------------------------------------------------------------
# the repair arms, shared by the dev pass and the confirmation run
# ---------------------------------------------------------------------------

def repair(client, scorer, text: str, samples: List[Dict], raws: List[str],
           temperature: float, max_tokens: int, rewind: bool = True) -> Dict:
    """V2P (and the RW baseline) for one gated row. `samples` carry answer and
    prm; `raws` are the texts the prm was computed on."""
    content = V21.COT_PROMPT.format(problem=text)
    src = max(range(3), key=lambda i: (low(samples[i]['prm']), -i))
    prm = samples[src]['prm']
    steps = P.split_steps(raws[src])
    cut = prm.index(min(prm)) if prm else 0
    doubted = steps[cut] if cut < len(steps) else ''
    sents = implicated_sentences(text, doubted)
    note = v2p_note(sents)
    out: Dict = {'source': src, 'cut_step': cut, 'doubted_step': doubted[:800],
                 'implicated': sents, 'note': note}
    # V2P: a fresh solve of the problem with the grounded sentences quoted. The
    # verifier still judges against the ORIGINAL problem, without the note.
    v2p_content = V21.COT_PROMPT.format(problem=text + '\n\n' + note)
    out['v2p'] = [_scored(scorer, text, full)
                  for full in V21.sample(client, v2p_content, N_EXTRA, temperature, max_tokens)]
    if rewind:
        prefix = ('\n\n'.join(steps[:cut]) + '\n\n') if cut > 0 else ''
        out['rw'] = [_scored(scorer, text, full, answer_from=full[len(prefix):])
                     for full in continue_from(client, content, prefix, N_EXTRA,
                                               temperature, max_tokens)]
    return out


def arms_for(rec: Dict, samples: List[Dict], gold: float) -> None:
    ans = [s['answer'] for s in samples]
    wl = [last(s['prm']) for s in samples]

    def put(name, a):
        rec[f'{name}_answer'] = a
        rec[f'{name}_correct'] = V21.correct(a, gold)

    put('S3', P.plain_vote(ans[:3]))
    put('B3', P.best_of_n(ans[:3], wl[:3]))
    if len(samples) >= 5:
        put('S5', P.plain_vote(ans[:5]))
        put('B5', P.best_of_n(ans[:5], wl[:5]))
    rp = rec.get('repair')
    if rec.get('gated') and rp:
        put('RST', P.best_of_n(ans[:5], wl[:5]))
        for arm, key in (('V2P', 'v2p'), ('RW', 'rw')):
            if key in rp:
                extra = rp[key]
                put(arm, P.best_of_n(ans[:3] + [e['answer'] for e in extra],
                                     wl[:3] + [last(e['prm']) for e in extra]))
    else:
        for arm in ('RST', 'V2P', 'RW'):
            put(arm, rec['B3_answer'])


# ---------------------------------------------------------------------------
# one confirmation row
# ---------------------------------------------------------------------------

def run_row(client, scorer, r: Dict, k: int, temperature: float, max_tokens: int,
            rewind: bool) -> Dict:
    text, gold = r['text'], r['gold']
    t0 = time.time()
    raws = V21.sample(client, V21.COT_PROMPT.format(problem=text), k, temperature, max_tokens)
    samples = []
    for raw in raws:
        steps = P.split_steps(raw)
        samples.append({'raw': str(raw)[:4000], 'answer': V21.parse_answer(raw),
                        'n_steps': len(steps), 'prm': scorer.score(text, steps)})
    rec: Dict = {'pid': r['problem_id'], 'gold': gold, 'dataset': r.get('dataset')}
    rec['gate'] = max(low(s['prm']) for s in samples[:3])
    rec['gated'] = rec['gate'] < GATE
    if rec['gated']:
        rec['repair'] = repair(client, scorer, text, samples, raws, temperature,
                               max_tokens, rewind)
    arms_for(rec, samples, gold)
    rec['samples'] = samples
    rec['seconds'] = round(time.time() - t0, 1)
    return rec


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def _paired(rows, x, y):
    w = sum(1 for r in rows if r.get(f'{x}_correct') and not r.get(f'{y}_correct'))
    l = sum(1 for r in rows if r.get(f'{y}_correct') and not r.get(f'{x}_correct'))
    return w, l


def summarise_repair(rows: List[Dict], label: str) -> None:
    g = [r for r in rows if r.get('gated')]
    if not g:
        print(f"    {label}: no gated rows")
        return
    print(f"    {label}: gated rows n={len(g)}")
    for a in ('B3', 'RST', 'RW', 'V2P'):
        if all(f'{a}_correct' in r for r in g):
            print(f"      {a:4s} {sum(1 for r in g if r[f'{a}_correct']):3d}/{len(g)}")
    for x, y in (('V2P', 'RST'), ('V2P', 'RW'), ('RW', 'RST'), ('V2P', 'B3')):
        if all(f'{x}_correct' in r and f'{y}_correct' in r for r in g):
            w, l = _paired(g, x, y)
            print(f"      {x:4s} vs {y:4s}: W={w:2d} L={l:2d} net={w-l:+d}")
    grounded = sum(1 for r in g if (r.get('repair') or {}).get('implicated'))
    print(f"      grounding found a sentence on {grounded}/{len(g)} gated rows")


def summarise(out: List[Dict], args) -> int:
    verdict = 'no main-set rows'
    for tag in ('main', 'guard'):
        rows = [r for r in out if r.get('set') == tag]
        if not rows:
            continue
        n = len(rows)
        print('\n' + '=' * 72)
        print(f"  {tag.upper()} SET  n={n}")
        for a in ('S3', 'S5', 'B3', 'B5', 'RST', 'RW', 'V2P'):
            if all(f'{a}_correct' in r for r in rows):
                s = sum(1 for r in rows if r[f'{a}_correct'])
                print(f"    {a:4s} {s:3d} = {100*s/n:5.1f}%")
        print()
        for x, y in (('B3', 'S3'), ('B3', 'S5'), ('B5', 'S5'), ('V2P', 'S3')):
            if all(f'{x}_correct' in r and f'{y}_correct' in r for r in rows):
                w, l = _paired(rows, x, y)
                print(f"    {x:4s} vs {y:4s}: W={w:2d} L={l:2d} net={w-l:+d}")
        summarise_repair(rows, 'repair')
        if tag == 'main':
            s3 = sum(1 for r in rows if r['S3_correct'])
            b3 = sum(1 for r in rows if r['B3_correct'])
            w, l = _paired(rows, 'B3', 'S3')
            if n != 100 and not args.stub:
                verdict = f"main n={n}, not the pre-registered 100 -- report as measured"
            elif b3 - s3 <= 0:
                verdict = "REFUTED: B3 <= S3 -- the dev gain was selection noise"
            elif b3 - s3 >= 4 and w >= 2 * l:
                verdict = "CONFIRMED: the verifier agent improves on self-consistency on fresh rows"
            else:
                verdict = "INCONCLUSIVE: B3 > S3 but short of the pre-registered bar"
        if tag == 'guard':
            w, l = _paired(rows, 'V2P', 'B3')
            print(f"    guard: V2P vs B3 net={w-l:+d} -> "
                  + ("no harm" if w - l >= -1 else "HARM: quoting a distractor misleads"))
    print('\n  pre-registered reading (verifier):\n    ' + verdict)
    print("  V2P is read on the gated rows POOLED with the dev pass "
          "(pretest_v22_dev.json) -- see --pooled")
    return 0


def pooled(paths: Sequence[str]) -> int:
    rows = []
    for p in paths:
        if os.path.exists(p):
            with open(p, encoding='utf-8') as fh:
                rows += [r for r in json.load(fh)['rows'] if r.get('set') == 'main']
    g = [r for r in rows if r.get('gated')]
    print(f"POOLED gated main rows: {len(g)} (from {', '.join(paths)})")
    summarise_repair(rows, 'pooled')
    if g and all('V2P_correct' in r and 'RST_correct' in r for r in g):
        v = sum(r['V2P_correct'] for r in g)
        s = sum(r['RST_correct'] for r in g)
        rw = sum(r.get('RW_correct', False) for r in g)
        if v - s >= 3 and v >= rw:
            print("  pre-registered reading (V2P): SUPPORTED")
        elif v <= s:
            print("  pre-registered reading (V2P): REFUTED")
        else:
            print("  pre-registered reading (V2P): INCONCLUSIVE")
    return 0


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------

def _load_prior(path: str, resume: bool, stub: bool) -> Dict[str, Dict]:
    if os.path.exists(path) and resume and not stub:
        with open(path, encoding='utf-8') as fh:
            prior = {r['pid']: r for r in json.load(fh).get('rows', [])}
        print(f"resuming: {len(prior)} rows already in {path}")
        return prior
    return {}


def _save(path: str, rows: List[Dict], extra: Dict) -> None:
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(dict(extra, gate=GATE, rows=rows), fh, ensure_ascii=False)


def build(args, rows_for_stub):
    if args.stub:
        return V21._StubClient(rows_for_stub), P._StubScorer({})
    import torch
    two = torch.cuda.device_count() > 1
    client = V21.build_client(args.preset, None)
    scorer = P.PRMScorer(P.PRM, four_bit=True, device_map={'': 1 if two else 0})
    return client, scorer


def run_dev(args) -> int:
    """V2P and RW on the dev rows the verifier gates, reusing the stored
    samples and their stored verifier scores: no new samples are drawn for
    S3/B3/RST, which are already paired."""
    texts: Dict[str, str] = {}
    for m in DEV_MANIFESTS:
        with open(m, encoding='utf-8') as fh:
            for r in json.load(fh)['rows']:
                texts[r['problem_id']] = r['text']
    with open(DEV_TRACES, encoding='utf-8') as fh:
        traces = {r['pid']: r for r in json.load(fh)['rows']}
    with open(DEV_PRM, encoding='utf-8') as fh:
        prm = json.load(fh)['rows']
    rows = []
    for r in prm:
        gate = max(low(s['prm']) for s in r['samples'][:3])
        rows.append(dict(r, gate=gate, gated=gate < GATE))
    gated = [r for r in rows if r['gated']]
    if args.limit:
        gated = gated[:args.limit]
    print(f"dev rows: {len(rows)}  gated: {len(gated)}  (only these generate)")
    client, scorer = build(args, [{'problem_id': r['pid'], 'text': texts[r['pid']],
                                   'gold': r['gold']} for r in gated])
    out_path = args.out or OUT_DEV
    prior = _load_prior(out_path, args.resume, args.stub)
    out, t0, done = [], time.time(), 0
    for i, r in enumerate(gated, 1):
        if r['pid'] in prior and 'V2P_correct' in prior[r['pid']]:
            out.append(prior[r['pid']])
            continue
        if args.max_hours and time.time() - t0 > args.max_hours * 3600:
            print("\n  --max-hours reached; re-run the identical command to resume.")
            break
        text = texts[r['pid']]
        raws = [s['raw'] for s in traces[r['pid']]['samples']]
        samples = [{'answer': s['answer'], 'prm': s['prm']} for s in r['samples']]
        rec = {'pid': r['pid'], 'gold': r['gold'], 'set': r['set'],
               'gate': r['gate'], 'gated': True}
        rec['repair'] = repair(client, scorer, text, samples, raws,
                               args.temperature, args.max_tokens, not args.no_rewind)
        arms_for(rec, samples, r['gold'])
        out.append(rec)
        done += 1
        f = lambda a: 'OK ' if rec.get(f'{a}_correct') else 'BAD'
        print(f"[{i:3d}/{len(gated)}] {r['set']:5s} {r['pid'][:22]:22s} B3={f('B3')} "
              f"RST={f('RST')} RW={f('RW')} V2P={f('V2P')}  "
              f"{(time.time()-t0)/done:5.0f}s/row  implicated={len(rec['repair']['implicated'])}",
              flush=True)
        _save(out_path, out, {'mode': 'dev'})
    _save(out_path, out, {'mode': 'dev'})
    for tag in ('main', 'guard'):
        summarise_repair([r for r in out if r['set'] == tag], f'DEV {tag}')
    print(f"\n  saved: {out_path}")
    return 0


def run_confirm(args) -> int:
    allrows = []
    for tag, path in MANIFESTS:
        with open(path, encoding='utf-8') as fh:
            rows = json.load(fh)['rows']
        allrows += [dict(r, set=tag) for r in (rows[:args.limit] if args.limit else rows)]
    print(f"rows: {len(allrows)}   k={args.k} T={args.temperature}  gate={GATE}")
    client, scorer = build(args, allrows)
    out_path = args.out or OUT
    prior = _load_prior(out_path, args.resume, args.stub)
    out, t0, ran, left = [], time.time(), 0, 0
    for i, r in enumerate(allrows, 1):
        if r['problem_id'] in prior and 'V2P_correct' in prior[r['problem_id']]:
            out.append(prior[r['problem_id']])
            continue
        if args.max_hours and time.time() - t0 > args.max_hours * 3600:
            left = len(allrows) - i + 1
            print(f"\n  --max-hours reached with {left} row(s) left; re-run the identical "
                  f"command to resume.")
            break
        rec = run_row(client, scorer, r, args.k, args.temperature, args.max_tokens,
                      not args.no_rewind)
        rec['set'] = r['set']
        out.append(rec)
        ran += 1
        f = lambda a: 'OK ' if rec.get(f'{a}_correct') else 'BAD'
        print(f"[{i:3d}/{len(allrows)}] {r['set']:5s} {str(r['problem_id'])[:22]:22s} "
              f"S3={f('S3')} B3={f('B3')} V2P={f('V2P')}"
              f"{' gated' if rec['gated'] else '      '}  {(time.time()-t0)/ran:5.0f}s/row",
              flush=True)
        _save(out_path, out, {'mode': 'confirm', 'k': args.k, 'temperature': args.temperature})
    if left:
        print(f"\n  PARTIAL: {len(out)}/{len(allrows)} rows. No verdict is read off a partial run.")
    return summarise(out, args)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev', action='store_true', help='V2P/RW on the gated dev rows only')
    ap.add_argument('--pooled', nargs='*', default=None,
                    help='read V2P on gated main rows pooled over these result files')
    ap.add_argument('--out', default='')
    ap.add_argument('--preset', default='qwen_math7b_mixed')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--no-rewind', action='store_true', help='skip the RW baseline to save time')
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--max-hours', type=float, default=0.0)
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()

    if args.pooled is not None:
        return pooled(args.pooled or [OUT_DEV, OUT])
    if args.summary_only:
        with open(args.out or OUT, encoding='utf-8') as fh:
            return summarise(json.load(fh)['rows'], args)
    return run_dev(args) if args.dev else run_confirm(args)


if __name__ == '__main__':
    sys.exit(main())
