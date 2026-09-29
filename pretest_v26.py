"""
[v26.0] Pilot: Prototype-Contrastive Decoding (PCD). Does contrasting the
Solver against the familiar version of the problem stop it from reciting,
and so make each reasoning chain right more often on GSM-Symbolic P2?

The method is in prototype_contrast.py; the evidence and the literature are
in V26_PLAN.md. This is a DEV pilot on stored rows. It decides whether a
fresh-row confirmation is worth running; it is not a thesis claim by itself.

ROWS (fixed here, before any PCD sample exists)
----
The 100 main rows of the v22 confirmation (seed 45). The v22 run stored 5
plain samples per row. Samples 1-3 pick the rows; samples 4-5 are held out
to check the new sampler, so no row is chosen and scored on the same draws.
  hard      the 60 rows with at most 2 of samples 1-3 right
  easy      20 of the 40 rows with 3 of 3 right (seed 26): the harm check
The 80 rows run in one seeded shuffled order, so a run cut short is a random
subset and not "the hard ones first".

ARMS (k=3 samples each, all drawn in ONE batch per row; T=0.8, top-p 0.95,
top-k 50, 1024 tokens: the v22 settings)
----
  L0     plain sampling: the sampler of every earlier version
  MINP   L0 plus the plausibility constraint alone (b=0.1, no contrast)
  PCD    contrast against the Prototype agent's familiar version, a=1.0, b=0.1  <- v26
  PCD5   the same with a=0.5 (dose response)
  CADQ   contrast against the question alone, a=1.0, b=0.1 (a generic contrast)
The Prototype agent (Qwen2.5-7B-Instruct, greedy) writes one prototype per
row. A prototype it fails to write becomes the real text, and the summary
counts those rows.

PRE-REGISTERED, frozen 2026-09-29 before any PCD sample exists
--------------------------------------------------------------
Metric: per-sample accuracy (mean over rows of right/3), paired by row, with
a two-sided sign-flip permutation test over rows and a template bootstrap.
  VALIDITY   |L0 - stored samples 4-5| <= 10 pp over the 80 rows (the paired
             noise is about 4 pp; a broken sampler costs 20 pp or more).
             Otherwise the new sampler is suspect and nothing below is read.
  PRIMARY    hard rows: PCD - L0 >= +5 pp AND permutation p < 0.05 -> SUPPORTED
                        PCD - L0 <= 0                              -> REFUTED
                        otherwise                                   -> INCONCLUSIVE
  ATTRIBUTE  hard rows: PCD - MINP >= +3 pp -> the contrast is what works,
             not the truncation
  SPECIFIC   hard rows: PCD - CADQ >= +3 pp -> the PROTOTYPE is what works,
             not any contrast
  HARM       easy rows: PCD - L0 >= -5 pp -> no harm
  GO         SUPPORTED and no HARM -> run the fresh-row confirmation on the
             v25 seed-48 rows (pretest_data/v25_confirm_p2.json)
Reported, not decisive: PCD5 (dose response), SC@3 per arm, the 18 rows where
no stored sample is right, prototype statistics.

MODES
-----
    python pretest_v26.py --max-hours 8.0      # Kaggle T4 x2; resumes when re-run
    python pretest_v26.py --summary-only       # re-read, no GPU
    python pretest_v26.py --stub --limit 6     # offline plumbing, no GPU
"""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence

import pretest_v21 as V21
import pretest_v24 as V24
import prototype_contrast as PC
import situation_reader as SR

DEV_V22 = 'results_September/preset_v22.json'
DEV_MANIFESTS = ('pretest_data/v22_confirm_p2.json', 'pretest_data/v22_confirm_guard.json')
OUT = 'pretest_v26.json'
OUT_STUB = 'pretest_v26_stub.json'
SELECT_SEED = 26
N_EASY = 20
K = 3
PROTO_ABORT_AFTER = 5

VALID_MAX_PP = 10.0
PRIMARY_MIN_PP = 5.0
PRIMARY_ALPHA = 0.05
ATTR_MIN_PP = 3.0
SPEC_MIN_PP = 3.0
HARM_MIN_PP = -5.0


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def load_rows(v22_path: str = DEV_V22) -> List[Dict]:
    texts = V24._texts(DEV_MANIFESTS)
    rows = []
    for r in V24._read(v22_path)['rows']:
        if r['set'] != 'main':
            continue
        ans = [s.get('answer') for s in r['samples']]
        rows.append({'key': f"v26:{r['pid']}", 'pid': r['pid'], 'gold': r['gold'],
                     'template': V24.template_of(r['pid']), 'text': texts[r['pid']],
                     'stored': ans,
                     'sel': sum(V21.correct(a, r['gold']) for a in ans[:3])})
    hard = [r for r in rows if r['sel'] <= 2]
    easy = [r for r in rows if r['sel'] == 3]
    easy = random.Random(SELECT_SEED).sample(easy, min(N_EASY, len(easy)))
    for r in hard:
        r['group'] = 'hard'
    for r in easy:
        r['group'] = 'easy'
    picked = hard + easy
    random.Random(SELECT_SEED).shuffle(picked)
    return picked


def new_record(row: Dict) -> Dict:
    return {k: row[k] for k in ('key', 'pid', 'gold', 'template', 'group', 'stored', 'sel')} | {
        'arms': {}}


# ---------------------------------------------------------------------------
# agents
# ---------------------------------------------------------------------------

_PROTOS = {'n': 0, 'ok': 0, 'shown': False}


def write_prototype(reader, text: str) -> Dict:
    t0 = time.time()
    err = ''
    try:
        raw = str(reader.call_model(PC.proto_messages(text), temperature=0.0,
                                    max_tokens=PC.PROTO_MAX_TOKENS))
    except Exception as exc:
        raw, err = '', f'{type(exc).__name__}: {exc}'[:200]
    out = PC.parse_prototype(raw, text)
    out.update(error=err, seconds=round(time.time() - t0, 1))
    _PROTOS['n'] += 1
    _PROTOS['ok'] += bool(out['ok'])
    if out['ok'] and not _PROTOS['shown']:
        _PROTOS['shown'] = True
        print("  first prototype of this session:\n    REAL : " + text +
              "\n    PROTO: " + out['text'], flush=True)
    if _PROTOS['n'] >= PROTO_ABORT_AFTER and not _PROTOS['ok']:
        raise RuntimeError(f"the Prototype agent gave nothing usable on its first {_PROTOS['n']} "
                           f"rows; last error: {err or 'none'}; last output: {raw[:500]!r}")
    return out


class HFSampler:
    """PCD over the Solver's local HF model, with the chat template the
    plain samples of every earlier version used (pretest_v21._batched)."""

    def __init__(self, client):
        client._ensure_local_model()
        self.tok, self.model = client._local_tokenizer, client._local_model
        eos = self.model.generation_config.eos_token_id
        eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        if self.tok.eos_token_id is not None:
            eos.append(self.tok.eos_token_id)
        self.eos = sorted({int(e) for e in eos if e is not None})
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos[0]
        # The stored plain samples came from HF generate, which also applies any
        # processor set in the model's generation_config. Qwen2.5-Math-7B-Instruct
        # sets none (checked 2026-09-29); if a future config does, say so loudly.
        gc_ = self.model.generation_config
        extra = {k: getattr(gc_, k, None) for k in ('repetition_penalty', 'no_repeat_ngram_size',
                                                   'min_p', 'typical_p', 'epsilon_cutoff')}
        extra = {k: v for k, v in extra.items() if v not in (None, 0, 0.0, 1.0)}
        if extra:
            print(f"  WARNING: generation_config sets {extra}, which this sampler does not "
                  f"apply; L0 will not match the stored samples exactly", flush=True)
        self.config_note = extra

    def encode(self, content: str) -> List[int]:
        prompt = self.tok.apply_chat_template([{'role': 'user', 'content': content}],
                                              tokenize=False, add_generation_prompt=True)
        return self.tok(prompt)['input_ids']

    def run(self, prompts: List[str], streams: List[PC.Stream], args, seed: int) -> List[Dict]:
        import torch
        ids = [self.encode(p) for p in prompts]
        try:
            toks = PC.sample_streams(self.model, ids, streams, max_new_tokens=args.max_tokens,
                                     temperature=args.temperature, top_k=args.top_k,
                                     top_p=args.top_p, eos_ids=self.eos, pad_id=self.pad,
                                     seed=seed)
        except torch.cuda.OutOfMemoryError:
            # one arm at a time: same math per stream, smaller batches
            gc.collect()
            torch.cuda.empty_cache()
            print("    (out of memory with all arms in one batch; drawing arm by arm)", flush=True)
            toks = [None] * len(streams)
            for arm in dict.fromkeys(s.arm for s in streams):
                js = [j for j, s in enumerate(streams) if s.arm == arm]
                rows = sorted(r for j in js for r in ([streams[j].main] + (
                    [streams[j].contrast] if streams[j].contrast is not None else [])))
                remap = {r: i for i, r in enumerate(rows)}
                sub = [PC.Stream(arm=arm, main=remap[streams[j].main],
                                 contrast=(remap[streams[j].contrast]
                                           if streams[j].contrast is not None else None),
                                 alpha=streams[j].alpha, beta=streams[j].beta) for j in js]
                out = PC.sample_streams(self.model, [ids[r] for r in rows], sub,
                                        max_new_tokens=args.max_tokens,
                                        temperature=args.temperature, top_k=args.top_k,
                                        top_p=args.top_p, eos_ids=self.eos, pad_id=self.pad,
                                        seed=seed + len(js))
                for j, t in zip(js, out):
                    toks[j] = t
                gc.collect()
                torch.cuda.empty_cache()
        res = []
        for t in toks:
            raw = self.tok.decode(t, skip_special_tokens=True)
            res.append({'raw': raw[:6000], 'answer': V21.parse_answer(raw), 'n_tokens': len(t),
                        'eos': bool(t) and int(t[-1]) in self.eos})
        return res


class _StubSampler:
    """Offline stand-in: right with a probability that depends only on the arm."""
    P = {'L0': 0.5, 'MINP': 0.52, 'PCD': 0.62, 'PCD5': 0.58, 'CADQ': 0.5}

    def __init__(self):
        self.rng = random.Random(0)
        self.gold = {}

    def run(self, prompts, streams, args, seed):
        out = []
        for s in streams:
            g = self.gold[prompts[s.main]]
            a = g if self.rng.random() < self.P[s.arm] else g + self.rng.choice([1, 2, 3])
            out.append({'raw': f"We compute.\n\nAnswer: {a:g}", 'answer': float(a),
                        'n_tokens': 12, 'eos': True})
        return out


class _StubReader:
    def call_model(self, msgs, **kw):
        text = msgs[-1]['content'].split('Problem: ', 1)[-1]
        sents = SR.sentences(text)
        return ' '.join(sents[:-2] + sents[-1:]) if len(sents) > 2 else text


def seed_of(pid: str) -> int:
    return int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# one row
# ---------------------------------------------------------------------------

def run_row(sampler, reader, rec: Dict, text: str, arms: Sequence[str], args) -> bool:
    """Returns True if it drew anything."""
    need = [a for a in arms if len(rec['arms'].get(a, {}).get('samples', [])) < args.k]
    if not need:
        return False
    if 'prototype' not in rec:
        rec['prototype'] = write_prototype(reader, text)
    contexts = {'real': text, 'prototype': rec['prototype']['text'],
                'question': PC.question_of(text)}
    prompts, streams = PC.layout(need, args.k, contexts)
    if isinstance(sampler, _StubSampler):
        for p in prompts:
            sampler.gold[p] = rec['gold']
    t0 = time.time()
    res = sampler.run(prompts, streams, args, seed_of(rec['pid']))
    for s, r in zip(streams, res):
        rec['arms'].setdefault(s.arm, {'samples': []})['samples'].append(r)
    rec['seconds'] = round(rec.get('seconds', 0) + time.time() - t0, 1)
    return True


# ---------------------------------------------------------------------------
# reading the results
# ---------------------------------------------------------------------------

def acc(rec: Dict, arm: str) -> Optional[float]:
    ss = rec['arms'].get(arm, {}).get('samples', [])
    return sum(V21.correct(s['answer'], rec['gold']) for s in ss) / len(ss) if ss else None


def held_out(rec: Dict) -> float:
    a = rec['stored'][3:5]
    return sum(V21.correct(x, rec['gold']) for x in a) / len(a)


def sc(rec: Dict, arm: str) -> bool:
    import score_prm_v22 as P
    return V21.correct(P.plain_vote([s['answer'] for s in rec['arms'].get(arm, {}).get(
        'samples', [])]), rec['gold'])


def complete(rec: Dict, arms: Sequence[str], k: int) -> bool:
    return all(len(rec['arms'].get(a, {}).get('samples', [])) >= k for a in arms)


def diff(rows: List[Dict], x: str, y: str) -> Dict:
    d = [acc(r, x) - acc(r, y) for r in rows]
    if not d:
        return {'pp': 0.0, 'p': 1.0, 'w': 0, 'l': 0, 'ci': (0.0, 0.0), 'n': 0}
    by = collections.defaultdict(list)
    for r, v in zip(rows, d):
        by[r['template']].append(v)
    keys = list(by)
    rng = random.Random(0)
    boots = []
    for _ in range(2000):
        vals = [v for kk in (rng.choice(keys) for _ in keys) for v in by[kk]]
        boots.append(100 * sum(vals) / len(vals))
    boots.sort()
    return {'pp': 100 * sum(d) / len(d), 'p': V24.perm_p(d),
            'w': sum(1 for v in d if v > 1e-9), 'l': sum(1 for v in d if v < -1e-9),
            'ci': (boots[50], boots[1949]), 'n': len(d)}


def block(rows: List[Dict], arms: Sequence[str], title: str) -> None:
    if not rows:
        return
    n = len(rows)
    print('\n' + '=' * 78)
    print(f"  {title}  n={n}  templates={len({r['template'] for r in rows})}")
    print(f"    {'arm':5s} {'per-sample':>10s} {'SC@3':>5s} {'any right':>9s}")
    print(f"    {'held':5s} {100 * sum(held_out(r) for r in rows) / n:9.1f}%   (stored samples 4-5)")
    for a in arms:
        ps = 100 * sum(acc(r, a) for r in rows) / n
        anyr = sum(1 for r in rows if (acc(r, a) or 0) > 0)
        print(f"    {a:5s} {ps:9.1f}% {sum(sc(r, a) for r in rows):5d} {anyr:9d}")
    for x, y in (('PCD', 'L0'), ('PCD', 'MINP'), ('PCD', 'CADQ'), ('PCD5', 'L0'),
                 ('MINP', 'L0'), ('CADQ', 'L0')):
        if x in arms and y in arms:
            c = diff(rows, x, y)
            print(f"    {x:4s} - {y:4s}: {c['pp']:+5.1f} pp  [template bootstrap {c['ci'][0]:+.1f}, "
                  f"{c['ci'][1]:+.1f}]  perm p={c['p']:.4f}  rows W={c['w']} L={c['l']}")


def prototype_line(recs: List[Dict]) -> str:
    pr = [r['prototype'] for r in recs if 'prototype' in r]
    if not pr:
        return ''
    n = len(pr)
    return (f"  prototypes: usable {sum(p['ok'] for p in pr)}/{n}, identical to the problem "
            f"{sum(p['identical'] for p in pr)}, kept the question {sum(p['kept_question'] for p in pr)}, "
            f"mean length ratio {sum(p['ratio'] for p in pr) / n:.2f}, "
            f"{sum(p.get('seconds', 0) for p in pr) / n:.0f} s each")


def verdicts(recs: List[Dict], arms: Sequence[str]) -> List[str]:
    out = []
    v = 100 * (sum(acc(r, 'L0') for r in recs) - sum(held_out(r) for r in recs)) / len(recs)
    if abs(v) > VALID_MAX_PP:
        return [f"VALIDITY FAILED: L0 - stored samples 4-5 = {v:+.1f} pp (bar {VALID_MAX_PP}). "
                f"The new sampler is suspect; no verdict is read."]
    out.append(f"VALIDITY ok: L0 - stored samples 4-5 = {v:+.1f} pp")
    hard = [r for r in recs if r['group'] == 'hard']
    easy = [r for r in recs if r['group'] == 'easy']
    c = diff(hard, 'PCD', 'L0')
    facts = (f"{c['pp']:+.1f} pp per sample on {c['n']} hard rows, perm p={c['p']:.4f}, "
             f"template bootstrap [{c['ci'][0]:+.1f}, {c['ci'][1]:+.1f}], rows W={c['w']} L={c['l']}")
    if c['pp'] >= PRIMARY_MIN_PP and c['p'] < PRIMARY_ALPHA:
        out.append(f"PRIMARY  SUPPORTED: PCD makes each chain right more often ({facts})")
        supported = True
    elif c['pp'] <= 0:
        out.append(f"PRIMARY  REFUTED: PCD <= L0 ({facts})")
        supported = False
    else:
        out.append(f"PRIMARY  INCONCLUSIVE: {facts}")
        supported = False
    if 'MINP' in arms:
        m = diff(hard, 'PCD', 'MINP')['pp']
        out.append(f"ATTRIB   PCD - MINP = {m:+.1f} pp -> " + (
            "the contrast is what works" if m >= ATTR_MIN_PP else
            "not shown that the contrast, rather than the truncation, does the work"))
    if 'CADQ' in arms:
        s = diff(hard, 'PCD', 'CADQ')['pp']
        out.append(f"SPECIFIC PCD - CADQ = {s:+.1f} pp -> " + (
            "the prototype is what works" if s >= SPEC_MIN_PP else
            "a generic contrast does about as well (the prototype is not shown to matter)"))
    h = diff(easy, 'PCD', 'L0')['pp']
    harm = h < HARM_MIN_PP
    out.append(f"HARM     easy rows PCD - L0 = {h:+.1f} pp -> " + ("HARM" if harm else "no harm"))
    out.append("GO       " + ("run the fresh-row confirmation (v25 seed-48 rows)"
                              if supported and not harm else "do not run the confirmation as is"))
    return out


def summarise(recs: List[Dict], arms: Sequence[str], k: int, stub: bool) -> int:
    rows = [r for r in recs if complete(r, arms, k)]
    print(prototype_line(rows))
    block([r for r in rows if r['group'] == 'hard'], arms, 'HARD (<=2 of stored samples 1-3 right)')
    block([r for r in rows if r['group'] == 'easy'], arms, 'EASY (3 of 3 right)')
    zero = [r for r in rows if not any(V21.correct(a, r['gold']) for a in r['stored'])]
    if zero:
        print(f"\n  the {len(zero)} rows where none of the 5 stored samples is right "
              f"(rows with >=1 right sample / right samples):")
        for a in arms:
            ns = [sum(V21.correct(s['answer'], r['gold']) for s in r['arms'][a]['samples']) for r in zero]
            print(f"    {a:5s} {sum(1 for x in ns if x):3d} rows / {sum(ns):3d} samples")
    secs = [r.get('seconds', 0) for r in rows if r.get('seconds')]
    if secs:
        print(f"\n  seconds/row (all arms in one batch): {sum(secs) / len(secs):.0f}")
    print('\n  pre-registered reading:')
    if not rows:
        print('    none: no complete rows yet')
        return 0
    if len(rows) < len(recs) and not stub:
        print(f"    none: PARTIAL run, {len(rows)}/{len(recs)} rows complete. "
              f"Re-run the identical command to finish.")
        return 0
    if k != K or set(arms) != set(PC.DEFAULT_ARMS):
        print(f"    none: k={k}, arms={','.join(arms)} is not the pre-registered design (exploratory)")
        return 0
    for v in verdicts(rows, arms):
        print('    ' + v)
    return 0


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _save(path: str, recs: List[Dict], meta: Dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(dict(meta, rows=recs), fh, ensure_ascii=False)
    os.replace(tmp, path)


def run(args, arms: Sequence[str]) -> int:
    _PROTOS.update(n=0, ok=0, shown=False)
    rows = load_rows(args.v22)
    if args.limit:
        rows = rows[:args.limit]
    out_path = args.out or (OUT_STUB if args.stub else OUT)
    prior = {}
    if os.path.exists(out_path) and args.resume and not (args.stub and out_path == OUT):
        prior = {r['key']: r for r in V24._read(out_path).get('rows', [])}
        print(f"resuming: {len(prior)} rows in {out_path}")
    recs = [prior.get(r['key']) or new_record(r) for r in rows]
    texts = {r['key']: r['text'] for r in rows}
    need = [r for r in recs if not complete(r, arms, args.k)]
    print(f"rows: {len(recs)} (hard {sum(r['group'] == 'hard' for r in recs)}, "
          f"easy {sum(r['group'] == 'easy' for r in recs)})  arms={','.join(arms)}  k={args.k}  "
          f"to draw: {len(need)}")
    meta = {'k': args.k, 'arms': list(arms), 'temperature': args.temperature, 'top_p': args.top_p,
            'top_k': args.top_k, 'max_tokens': args.max_tokens, 'alpha': PC.ALPHA,
            'alpha_half': PC.ALPHA_HALF, 'beta': PC.BETA, 'select_seed': SELECT_SEED}
    if need:
        if args.stub:
            sampler, reader = _StubSampler(), _StubReader()
        else:
            solver, reader = V24.build(args, [])
            sampler = HFSampler(solver)
        t0, ran = time.time(), 0
        deadline = t0 + args.max_hours * 3600 if args.max_hours else 0.0
        for i, rec in enumerate(recs, 1):
            if complete(rec, arms, args.k):
                continue
            if deadline and time.time() > deadline:
                print("\n  --max-hours reached; re-run the identical command to resume.")
                break
            try:
                drew = run_row(sampler, reader, rec, texts[rec['key']], arms, args)
            except RuntimeError:
                _save(out_path, recs, meta)
                raise
            if drew:
                ran += 1
                f = lambda a: f"{a}={sum(V21.correct(s['answer'], rec['gold']) for s in rec['arms'][a]['samples'])}"
                pr = rec.get('prototype', {})
                tag = 'same' if pr.get('identical') else ('FAILED' if not pr.get('ok', True) else 'edited')
                print(f"[{i:3d}/{len(recs)}] {rec['group']:4s} {rec['pid'][:24]:24s} "
                      f"{' '.join(f(a) for a in arms)}  proto={tag}  "
                      f"{(time.time() - t0) / ran:5.0f}s/row", flush=True)
                _save(out_path, recs, meta)
    _save(out_path, recs, meta)
    print(f"\n  saved: {out_path}")
    return summarise(recs, arms, args.k, args.stub)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default=','.join(PC.DEFAULT_ARMS))
    ap.add_argument('--k', type=int, default=K)
    ap.add_argument('--v22', default=DEV_V22)
    ap.add_argument('--out', default='')
    ap.add_argument('--preset', default='qwen_math7b_mixed')
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--top-k', type=int, default=50)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--max-hours', type=float, default=0.0)
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()
    arms = [a.strip().upper() for a in args.arms.split(',') if a.strip()]
    bad = [a for a in arms if a not in PC.ARMS]
    if bad:
        print(f"unknown arm(s) {bad}; choose from {list(PC.ARMS)}")
        return 2
    if args.summary_only:
        data = V24._read(args.out or OUT)
        return summarise(data['rows'], data.get('arms', arms), data.get('k', args.k), args.stub)
    return run(args, arms)


if __name__ == '__main__':
    sys.exit(main())
