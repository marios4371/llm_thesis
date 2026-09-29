"""[v26.0] Guards for Prototype-Contrastive Decoding. Offline, no GPU.

What these tests exist to prevent is a pilot that measures a bug:
  * a hand-written decoding loop that is not the sampler of every earlier
    version (checked against HF `generate` and HF's own warpers on a tiny
    random Qwen2 model: greedy token-for-token, left padding, early EOS);
  * a contrast that changes anything when the two contexts are identical,
    or that picks a token the real problem finds implausible;
  * arms wired to the wrong prompts or the wrong a/b;
  * Prototype-agent demos taken from the benchmark;
  * rows chosen with the same samples they are later compared against;
  * a verdict read off a partial run, or bars that moved after freezing.

Run as `python test_v26.py`. The model tests need torch + transformers; they
SKIP without them, and `--require-model` turns a skip into a failure (the
Kaggle cell passes it).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
import tempfile

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')

import pretest_v21 as V21
import pretest_v24 as V24
import pretest_v26 as T
import prototype_contrast as PC
import situation_reader as SR

FAILS, SKIPS = [], []
N = [0]


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def _args(**kw):
    base = dict(arms=','.join(PC.DEFAULT_ARMS), k=3, v22=T.DEV_V22, out='', preset='qwen_math7b_mixed',
                temperature=0.8, top_p=0.95, top_k=50, max_tokens=1024, limit=6, stub=True,
                resume=True, max_hours=0.0, summary_only=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _torch():
    try:
        import torch
        import transformers  # noqa: F401
        return torch
    except Exception as exc:
        SKIPS.append(f'torch/transformers unavailable: {exc}')
        return None


def _tiny():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    return Qwen2ForCausalLM(cfg).eval()


def part1():
    print("\nPART 1 - the scorer (combine / warp)")
    torch = _torch()
    if torch is None:
        return
    g = torch.Generator().manual_seed(0)
    main = torch.randn(4, 50, generator=g) * 3
    con = torch.randn(4, 50, generator=g) * 3
    z = torch.zeros(4)
    lp = torch.log_softmax(main, -1)
    check(torch.allclose(PC.combine(main, con, z, z), lp, atol=1e-6),
          "a=0, b=0 is exactly log_softmax(real): plain sampling")
    check(torch.allclose(PC.combine(main, main, torch.full((4,), 1.0), z), lp, atol=1e-5),
          "a=1 against an IDENTICAL context changes nothing")
    s = PC.combine(main, con, torch.full((4,), 1.0), torch.full((4,), 0.1))
    p = lp.exp()
    ok = True
    for i in range(4):
        allowed = torch.isfinite(s[i])
        ok &= bool((p[i][allowed] >= 0.1 * p[i].max() - 1e-7).all())
        ok &= bool(allowed[p[i].argmax()])
    check(ok, "b=0.1 keeps only tokens with p(real) >= 0.1 * max, always including the argmax")
    s0 = PC.combine(main, con, torch.full((4,), 1.0), z)
    expect = 2 * lp - torch.log_softmax(con, -1)
    check(torch.allclose(s0, expect, atol=1e-5), "a=1 scores are 2*log p(real) - log p(prototype)")
    from transformers.generation.logits_process import (LogitsProcessorList, TemperatureLogitsWarper,
                                                        TopKLogitsWarper, TopPLogitsWarper)
    hf = LogitsProcessorList([TemperatureLogitsWarper(0.8), TopKLogitsWarper(50), TopPLogitsWarper(0.95)])
    logits = torch.randn(3, 200, generator=g) * 4
    ref = torch.log_softmax(hf(None, logits.clone()), -1)
    mine = PC.warp(torch.log_softmax(logits, -1), 0.8, 50, 0.95)
    same = torch.isfinite(ref) == torch.isfinite(mine)
    check(bool(same.all()) and torch.allclose(ref[torch.isfinite(ref)], mine[torch.isfinite(mine)], atol=1e-5),
          "warp() == HF Temperature -> TopK -> TopP warpers, renormalised (the v22 sampler)")


def part2():
    print("\nPART 2 - the sampler against HF generate (tiny random Qwen2)")
    torch = _torch()
    if torch is None:
        return
    m = _tiny()
    PAD, EOS = 0, 5
    rng = random.Random(1)
    prompts = [[rng.randrange(6, 64) for _ in range(n)] for n in (7, 3, 11, 5)]
    ref = []
    for p in prompts:
        out = m.generate(torch.tensor([p]), attention_mask=torch.ones(1, len(p), dtype=torch.long),
                         max_new_tokens=25, do_sample=False, eos_token_id=EOS, pad_token_id=PAD)
        ref.append(out[0, len(p):].tolist())
    strip = lambda t: t[:t.index(EOS) + 1] if EOS in t else t
    kw = dict(max_new_tokens=25, temperature=0, top_k=50, top_p=0.95, eos_ids=[EOS], pad_id=PAD)
    got = PC.sample_streams(m, prompts, [PC.Stream('L0', i) for i in range(4)], **kw)
    check(all(strip(a) == strip(b) for a, b in zip(got, ref)),
          "greedy loop == HF generate token for token (left padding, mixed lengths)")
    check(any(len(strip(t)) < 25 for t in got),
          "at least one stream stops early, so batch pruning is exercised")
    # PCD against an identical context == L0
    dup = prompts + [list(p) for p in prompts]
    st = [PC.Stream('PCD', i, contrast=4 + i, alpha=1.0, beta=0.1) for i in range(4)]
    got_same = PC.sample_streams(m, dup, st, **kw)
    check([strip(t) for t in got_same] == [strip(t) for t in got],
          "PCD against an identical prototype reproduces L0 exactly")
    # PCD against a different context: changes something, only plausible tokens
    other = [[rng.randrange(6, 64) for _ in range(9)] for _ in range(4)]
    st = [PC.Stream('PCD', i, contrast=4 + i, alpha=1.0, beta=0.1) for i in range(4)]
    got_c = PC.sample_streams(m, prompts + other, st, **kw)
    check([strip(t) for t in got_c] != [strip(t) for t in got],
          "PCD against a different prototype changes the output")
    plaus = True
    with torch.no_grad():
        for p, t in zip(prompts, got_c):
            seq = list(p)
            for tok in strip(t):
                pr = torch.softmax(m(torch.tensor([seq])).logits[0, -1].float(), -1)
                plaus &= bool(pr[tok] >= 0.1 * pr.max() - 1e-6)
                seq.append(tok)
    check(plaus, "every PCD token satisfies p(token | real) >= 0.1 * max, re-checked with fresh forwards")
    # sampling: seeded and reproducible
    kws = dict(kw, temperature=0.8)
    a1 = PC.sample_streams(m, prompts, [PC.Stream('L0', i) for i in range(4)], seed=7, **kws)
    a2 = PC.sample_streams(m, prompts, [PC.Stream('L0', i) for i in range(4)], seed=7, **kws)
    a3 = PC.sample_streams(m, prompts, [PC.Stream('L0', i) for i in range(4)], seed=8, **kws)
    check(a1 == a2 and a1 != a3, "sampling is reproducible for a seed and differs across seeds")
    try:
        PC.sample_streams(m, prompts[:2], [PC.Stream('L0', 0), PC.Stream('L0', 0)], **kw)
        check(False, "a prompt row shared by two streams is refused")
    except ValueError:
        check(True, "a prompt row shared by two streams is refused")


class _FakeTok:
    eos_token_id = 5
    pad_token_id = 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return msgs[-1]['content']

    def __call__(self, text):
        return {'input_ids': [6 + (ord(c) % 50) for c in text[:12]]}

    def decode(self, t, skip_special_tokens=True):
        return 'Answer: ' + str(sum(t) % 7)


class _FakeCfg:
    eos_token_id = [5]


class _FakeClient:
    def __init__(self, model):
        self._local_model, self._local_tokenizer = model, _FakeTok()

    def _ensure_local_model(self):
        pass


def part3():
    print("\nPART 3 - the arms, the Prototype agent, the OOM fallback")
    ctx = {'real': 'R has 3 apples. How many?', 'prototype': 'R has 3 apples. How many?x',
           'question': 'How many?'}
    prompts, streams = PC.layout(PC.DEFAULT_ARMS, 3, ctx)
    by = {a: [s for s in streams if s.arm == a] for a in PC.DEFAULT_ARMS}
    check(all(len(v) == 3 for v in by.values()) and len(streams) == 15,
          "3 samples per arm, 5 arms")
    check(all(prompts[s.main] == SR.cot_prompt(ctx['real']) for s in streams),
          "every arm's main context is the real problem in the CoT prompt of every earlier version")
    check(all(s.contrast is None and s.alpha == 0 and s.beta == 0 for s in by['L0']),
          "L0: no contrast, a=0, b=0 (plain sampling)")
    check(all(s.contrast is None and s.alpha == 0 and s.beta == 0.1 for s in by['MINP']),
          "MINP: the plausibility constraint alone")
    check(all(prompts[s.contrast] == SR.cot_prompt(ctx['prototype']) and s.alpha == 1.0 and s.beta == 0.1
              for s in by['PCD']), "PCD: contrast = the prototype, a=1.0, b=0.1")
    check(all(prompts[s.contrast] == SR.cot_prompt(ctx['prototype']) and s.alpha == 0.5 for s in by['PCD5']),
          "PCD5: the same prototype at a=0.5")
    check(all(prompts[s.contrast] == SR.cot_prompt('How many?') and s.alpha == 1.0 for s in by['CADQ']),
          "CADQ: contrast = the question alone, a=1.0")
    used = [r for s in streams for r in [s.main] + ([s.contrast] if s.contrast is not None else [])]
    check(sorted(used) == list(range(len(prompts))), "every prompt row belongs to exactly one stream")
    check((PC.ALPHA, PC.ALPHA_HALF, PC.BETA) == (1.0, 0.5, 0.1), "a and b are the literature defaults")

    texts = []
    for p in ('pretest_data/v22_confirm_p2.json', 'pretest_data/v25_confirm_p2.json',
              'pretest_data/v20_problems.json'):
        if os.path.exists(p):
            texts += [r['text'] for r in V24._read(p)['rows']]
    demo_words = [set(d[0].lower().split()) for d in PC.PROTO_DEMOS]
    overlap = max((len(w & set(t.lower().split())) / len(w) for w in demo_words for t in texts), default=0)
    check(texts and overlap < 0.6, f"no Prototype demo is a benchmark problem (max word overlap {overlap:.2f})")
    msgs = PC.proto_messages('Ana has 4 cats. How many?')
    check(msgs[0]['role'] == 'system' and msgs[-1]['content'] == 'Problem: Ana has 4 cats. How many?'
          and len(msgs) == 2 + 2 * len(PC.PROTO_DEMOS), "the Prototype agent sees the problem verbatim, after the demos")
    real = "Tom has 5 red balls and 3 blue balls. However, he loses 2 red balls. How many balls does Tom have?"
    ok = PC.parse_prototype("Rewritten problem: Tom has 5 red balls and 3 blue balls. How many balls does Tom have?", real)
    check(ok['ok'] and ok['text'].startswith('Tom has') and not ok['identical'] and ok['kept_question'],
          "a normal prototype parses, prefix stripped, question kept")
    same = PC.parse_prototype(real, real)
    check(same['ok'] and same['identical'], "an unchanged copy is marked identical (PCD then = MINP on that row)")
    for raw, why in (('', 'empty'), ("Let's solve it. Answer: 8", 'solved'),
                     (real * 3, 'too long'), ('Tom.', 'too short')):
        p = PC.parse_prototype(raw, real)
        check(not p['ok'] and p['text'] == real, f"a failed prototype ({why}) falls back to the real text")
    check(PC.question_of(real) == 'How many balls does Tom have?', "the question is the last '?' sentence")

    torch = _torch()
    if torch is None:
        return
    m = _tiny()
    m.generation_config = _FakeCfg()
    sampler = T.HFSampler(_FakeClient(m))
    args = _args(max_tokens=6, temperature=0.8)
    prompts, streams = PC.layout(PC.DEFAULT_ARMS, 2, ctx)
    real_ss = PC.sample_streams
    calls = []

    def flaky(model, ids, sts, **kw):
        calls.append(len(sts))
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError('fake')
        return [[10 + s.main, 11 + s.main] for s in sts]
    PC.sample_streams = flaky
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = sampler.run(prompts, streams, args, seed=1)
    finally:
        PC.sample_streams = real_ss
    check(calls[0] == len(streams) and calls[1:] == [2] * len(PC.DEFAULT_ARMS) and len(res) == len(streams),
          "on OOM the row is redrawn arm by arm, every stream still gets a sample")
    check(all(r['n_tokens'] == 2 for r in res), "each stream's result lands in its own slot")


def part4():
    print("\nPART 4 - the pilot rows (pre-registered)")
    rows = T.load_rows()
    hard = [r for r in rows if r['group'] == 'hard']
    easy = [r for r in rows if r['group'] == 'easy']
    check(len(hard) == 60 and len(easy) == 20, "60 hard rows + 20 easy rows")
    check(all(r['sel'] == sum(V21.correct(a, r['gold']) for a in r['stored'][:3]) for r in rows),
          "rows are chosen on stored samples 1-3 only")
    check(all(r['sel'] <= 2 for r in hard) and all(r['sel'] == 3 for r in easy), "hard <= 2/3, easy = 3/3")
    check([r['pid'] for r in T.load_rows()] == [r['pid'] for r in rows], "the selection and order are deterministic")
    check(len({r['group'] for r in rows[:20]}) == 2, "the run order mixes hard and easy rows")
    check(all(len(r['stored']) == 5 and r['text'] for r in rows), "each row keeps 5 stored answers and its text")
    check((T.VALID_MAX_PP, T.PRIMARY_MIN_PP, T.PRIMARY_ALPHA, T.ATTR_MIN_PP, T.SPEC_MIN_PP, T.HARM_MIN_PP, T.K)
          == (10.0, 5.0, 0.05, 3.0, 3.0, -5.0, 3), "the bars are the ones frozen on 2026-09-29")
    doc = T.__doc__
    check('frozen 2026-09-29' in doc and 'PCD - L0 >= +5 pp AND permutation p < 0.05' in doc
          and '<= 10 pp' in doc, "the docstring states the same bars")


def _rec(group, gold, arms_right, stored_right=(1, 1, 1, 1, 1), tpl=0, pid='x'):
    return {'key': pid, 'pid': pid, 'gold': gold, 'template': tpl, 'group': group,
            'stored': [gold if r else gold + 1 for r in stored_right], 'sel': 0,
            'arms': {a: {'samples': [{'answer': gold if i < n else gold + 1} for i in range(3)]}
                     for a, n in arms_right.items()}}


def part5():
    print("\nPART 5 - verdicts")
    base = {'L0': 1, 'MINP': 1, 'PCD5': 2, 'CADQ': 1}
    hard = [_rec('hard', 5.0, dict(base, PCD=3), stored_right=(0, 0, 0, 1, 0), tpl=i, pid=f'h{i}') for i in range(12)]
    hard += [_rec('hard', 5.0, dict(base, PCD=1), stored_right=(0, 0, 0, 0, 0), tpl=100 + i, pid=f'g{i}') for i in range(8)]
    easy = [_rec('easy', 5.0, dict(base, L0=3, PCD=3), stored_right=(1, 1, 1, 1, 1), tpl=200 + i, pid=f'e{i}') for i in range(5)]
    v = T.verdicts(hard + easy, PC.DEFAULT_ARMS)
    check(v[0].startswith('VALIDITY ok'), "validity passes when L0 matches the held-out samples")
    check(any(x.startswith('PRIMARY  SUPPORTED') for x in v), "a large, consistent gain is SUPPORTED")
    check(any('the contrast is what works' in x for x in v) and any('the prototype is what works' in x for x in v),
          "attribution and specificity read off PCD - MINP and PCD - CADQ")
    check(any(x.startswith('GO       run') for x in v), "SUPPORTED with no harm -> GO")
    flat = [_rec('hard', 5.0, dict(base, PCD=1), stored_right=(0, 0, 0, i % 2, 0), tpl=i, pid=f'f{i}') for i in range(10)]
    v = T.verdicts(flat + easy, PC.DEFAULT_ARMS)
    check(any(x.startswith('PRIMARY  REFUTED') for x in v), "no gain -> REFUTED")
    broken = [_rec('hard', 5.0, dict(base, L0=0, PCD=3), stored_right=(0, 0, 0, 1, 1), tpl=i, pid=f'b{i}') for i in range(10)]
    v = T.verdicts(broken, PC.DEFAULT_ARMS)
    check(len(v) == 1 and v[0].startswith('VALIDITY FAILED'), "a sampler far from the held-out samples blocks every verdict")
    harm = [_rec('easy', 5.0, dict(base, L0=3, PCD=1), tpl=300 + i, pid=f'z{i}') for i in range(5)]
    v = T.verdicts(hard + harm, PC.DEFAULT_ARMS)
    check(any(x.startswith('HARM     ') and x.endswith('HARM') for x in v) and any('do not run' in x for x in v),
          "harm on the easy rows blocks GO")


def part6():
    print("\nPART 6 - the runner (stub agents)")
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 'p.json')
        with contextlib.redirect_stdout(io.StringIO()):
            T.run(_args(out=out, limit=6), PC.DEFAULT_ARMS)
        data = V24._read(out)
        check(len(data['rows']) == 6 and all(T.complete(r, PC.DEFAULT_ARMS, 3) for r in data['rows']),
              "a stub run completes every arm on every row")
        check(all('prototype' in r and r['prototype']['ok'] for r in data['rows']), "each row stores its prototype")
        check(data['alpha'] == 1.0 and data['beta'] == 0.1 and data['select_seed'] == 26, "the run records a, b and the seed")
        before = json.dumps(data['rows'], sort_keys=True)
        with contextlib.redirect_stdout(io.StringIO()):
            T.run(_args(out=out, limit=6), PC.DEFAULT_ARMS)
        check(json.dumps(V24._read(out)['rows'], sort_keys=True) == before, "re-running a finished run changes nothing")
        out = os.path.join(d, 'cut.json')
        with contextlib.redirect_stdout(io.StringIO()):
            T.run(_args(out=out, limit=3), PC.DEFAULT_ARMS)           # 3 rows finished
            T.run(_args(out=out, limit=6, max_hours=-1.0), PC.DEFAULT_ARMS)  # 3 more, stopped at once
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T.summarise(V24._read(out)['rows'], PC.DEFAULT_ARMS, 3, stub=False)
        check('PARTIAL run' in buf.getvalue() and 'PRIMARY' not in buf.getvalue(),
              "a run stopped by --max-hours gives no verdict")
        with contextlib.redirect_stdout(io.StringIO()):
            T.run(_args(out=out, limit=6), PC.DEFAULT_ARMS)
        check(all(T.complete(r, PC.DEFAULT_ARMS, 3) for r in V24._read(out)['rows']), "the same command resumes it")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T.summarise(V24._read(out)['rows'], ['L0', 'PCD'], 3, stub=True)
        check('not the pre-registered design' in buf.getvalue(), "a run with other arms is labelled exploratory")
    check(T.OUT_STUB != T.OUT, "a stub run never writes the real output file")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--require-model', action='store_true')
    a = ap.parse_args()
    for p in (part1, part2, part3, part4, part5, part6):
        p()
    if SKIPS:
        print("\nSKIPPED model tests: " + '; '.join(sorted(set(SKIPS))))
        if a.require_model:
            FAILS.append('model tests skipped under --require-model')
    print(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed" + (" (model tests SKIPPED)" if SKIPS else ''))
    if FAILS:
        print("FAILED:\n  " + "\n  ".join(FAILS))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
