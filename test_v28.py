"""[v28.0] Guards for the Context-DPO screen. Offline; the model tests need
torch + transformers + peft and SKIP without them (`--require-model` turns a
skip into a failure; the Kaggle cell passes it).

What these tests exist to prevent is a screen that measures a bug:
  * a model scored on a template it was trained on (folds must be by
    template, disjoint, and each main row scored exactly once);
  * pairs that are not what the plan says: a chosen response that is wrong,
    a rejected one that is right, FV and RW differing in anything but the
    rejected text, a cut-off text taught as a response;
  * a DPO loss or schedule with a sign or scale error;
  * a prompt at training time that is not the prompt at sampling time;
  * LoRA training that does not move the model, or moves the reference;
  * a verdict read off a partial run, or bars that moved after freezing.

Run as `python test_v28.py`.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
import sys
import tempfile

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')

import context_dpo as CD
import pretest_v21 as V21
import pretest_v24 as V24
import pretest_v28 as T

FAILS, SKIPS = [], []
N = [0]


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def _args(**kw):
    base = dict(worker=0, device=0, arms=list(CD.ARMS), preset='qwen_math7b_mixed', temperature=0.8,
                max_tokens=1024, out='', out_dir='.', stub=True, smoke=False, resume=True, max_hours=0.0,
                summary_only=False, mode='stub')
    base.update(kw)
    return argparse.Namespace(**base)


def _s(answer, raw=None):
    return {'answer': answer, 'raw': raw if raw is not None else f"We compute.\nAnswer: {answer}"}


# ---------------------------------------------------------------------------

def part1():
    print("\nPART 1 - rows, folds, who scores what")
    rows = T.load_rows()
    main = [r for r in rows if r['set'] == 'main']
    guard = [r for r in rows if r['set'] != 'main']
    check(len(main) == 200 and len(guard) == 60, f"200 main + 60 guard rows (got {len(main)} + {len(guard)})")
    check(len({r['key'] for r in rows}) == len(rows), "row keys are unique")
    check(sum(r['src'] == 'v21' for r in main) == 100 and sum(r['src'] == 'v22' for r in main) == 100,
          "100 main rows from each stored run")
    check(all(len(r['base']) == 5 for r in main), "every main row has its 5 stored plain samples")
    check(all(r['template'] is not None and r['gold'] is not None for r in main),
          "every main row has a template and a gold")
    check(len({r['template'] for r in main}) == 49, "49 templates")
    v22 = [r for r in main if r['src'] == 'v22']
    check(all(r['prototype'] is not None for r in v22), "every v22 main row reuses its v27 prototype")
    check(all(r['prototype'] is None for r in main if r['src'] == 'v21'),
          "v21 rows have no prototype yet (the worker writes them)")
    f1, f2 = CD.split_folds(main), CD.split_folds(list(reversed(main)))
    check(f1 == f2, "the fold split does not depend on row order")
    check(set(f1) == {r['template'] for r in main} and set(f1.values()) == {0, 1},
          "every template is in exactly one of two folds")
    n = [sum(1 for r in main if f1[r['template']] == f) for f in (0, 1)]
    big = max(collections_count(main).values())
    check(abs(n[0] - n[1]) <= big, f"folds balanced within one template ({n[0]} vs {n[1]} rows)")
    P = [T.plan(rows, w, 'real') for w in (0, 1)]
    for w in (0, 1):
        tt = {r['template'] for r in P[w]['train']}
        et = {r['template'] for r in P[w]['eval']}
        check(not (tt & et), f"worker {w}: no template both trained on and scored")
        check({r['key'] for r in P[w]['valid']} <= {r['key'] for r in P[w]['eval']}
              and len(P[w]['valid']) == T.N_VALID, f"worker {w}: {T.N_VALID} validity rows, all scored rows")
    check({r['key'] for r in P[0]['eval']} == {r['key'] for r in P[1]['train']}
          and {r['key'] for r in P[1]['eval']} == {r['key'] for r in P[0]['train']},
          "each worker scores exactly the rows the other trains on")
    scored = [r['key'] for w in (0, 1) for r in P[w]['eval']]
    check(len(scored) == 200 and len(set(scored)) == 200, "each of the 200 main rows is scored exactly once")
    g = [r['key'] for w in (0, 1) for r in P[w]['guard']]
    check(len(g) == 60 and len(set(g)) == 60, "the 60 guard rows are split between the workers")
    S = T.plan(rows, 0, 'smoke')
    check(len(S['train']) == T.SMOKE['train'] and all(
        CD.candidates(r)['chosen'] and CD.candidates(r)['RW'] for r in S['train']),
          "the smoke trains on rows that can give pairs")


def collections_count(main):
    import collections
    return collections.Counter(r['template'] for r in main)


def part2():
    print("\nPART 2 - pairs")
    gold = 12.0
    row = {'key': 'x:1', 'pid': 'p', 'template': 3, 'text': 'A problem.', 'gold': gold,
           'prototype': {'ok': True, 'identical': False, 'text': 'A familiar problem.'},
           'base': [_s(12.0, 'right one'), _s(14.0, 'wrong one'), _s(12.0, 'right two'),
                    _s(15.0, 'wrong two'), _s(12.0, 'x' * 4000)],
           'extra': [_s(12.0, 'right three')],
           'familiar': [_s(14.0, 'recited one'), _s(14.0, 'recited two')]}
    c = CD.candidates(row)
    check([s['raw'] for s in c['chosen']] == ['right one', 'right two', 'right three'],
          "chosen = right samples, stored plain ones first, cut-off texts excluded")
    check([s['raw'] for s in c['RW']] == ['wrong one', 'wrong two'], "RW pool = wrong stored samples")
    check([s['raw'] for s in c['FV']] == ['recited one', 'recited two'], "FV pool = wrong familiar samples")
    fv, rw = CD.build_pairs(row, 'FV'), CD.build_pairs(row, 'RW')
    check(len(fv) == len(rw) == CD.MAX_CHOSEN * CD.MAX_NEG, "FV and RW: same number of pairs (2 x 2)")
    check([p['chosen'] for p in fv] == [p['chosen'] for p in rw], "FV and RW: the same chosen responses")
    check({p['rejected'] for p in fv} == {'recited one', 'recited two'}
          and {p['rejected'] for p in rw} == {'wrong one', 'wrong two'},
          "only the rejected text differs between the arms")
    check(all(V21.correct(p['chosen_answer'], gold) and not V21.correct(p['rejected_answer'], gold)
              for p in fv + rw), "every chosen is right and every rejected is wrong")
    one = dict(row, familiar=[_s(14.0, 'recited one')])
    check(len(CD.build_pairs(one, 'FV')) == len(CD.build_pairs(one, 'RW')) == 2,
          "one familiar negative -> one negative per chosen in BOTH arms")
    for label, bad in (("identical prototype", dict(row, prototype={'ok': True, 'identical': True})),
                       ("failed prototype", dict(row, prototype={'ok': False, 'identical': True})),
                       ("familiar version right for the real problem",
                        dict(row, familiar=[_s(12.0, 'fam right')])),
                       ("no wrong stored sample", dict(row, base=[_s(12.0, 'r')])),
                       ("no right sample", dict(row, base=[_s(1.0, 'w')], extra=[]))):
        check(CD.build_pairs(bad, 'FV') == [] and CD.build_pairs(bad, 'RW') == [],
              f"no pairs in either arm: {label}")
    check(CD.familiar_answer([_s(14.0), _s(15.0), _s(15.0)], gold) == 15.0,
          "familiar answer = the most frequent wrong answer on the prototype")
    check(CD.familiar_answer([_s(12.0), _s(12.0)], gold) is None, "no familiar answer when it is right")
    rows = T.load_rows()
    same = True
    for r in rows:
        if r['set'] != 'main':
            continue
        fake = dict(r, prototype={'ok': True, 'identical': False},
                    familiar=[s for s in r['base'] if not CD.right(s, r['gold'])][::-1])
        same &= len(CD.build_pairs(fake, 'FV')) == len(CD.build_pairs(fake, 'RW'))
    check(same, "on the 200 stored rows FV and RW always get the same number of pairs")
    try:
        CD.build_pairs(row, 'XX')
        check(False, "an unknown arm is refused")
    except ValueError:
        check(True, "an unknown arm is refused")
    check(CD.prompt_content('P?') == V21.COT_PROMPT.format(problem='P?'),
          "training prompt = the CoT prompt every sample was drawn with")


def part3():
    print("\nPART 3 - the loss and the schedule")
    t = CD.dpo_terms(-50.0, -60.0, -50.0, -60.0, 10)
    check(abs(t['margin']) < 1e-12 and abs(t['dpo'] - math.log(2)) < 1e-9,
          "policy = reference -> margin 0, DPO loss log 2")
    check(abs(t['nll'] - 5.0) < 1e-12, "NLL term = mean token NLL of the chosen response")
    up = CD.dpo_terms(-45.0, -60.0, -50.0, -60.0, 10)
    down = CD.dpo_terms(-50.0, -55.0, -50.0, -60.0, 10)
    check(up['margin'] > 0 and up['dpo'] < t['dpo'], "raising the chosen log-prob lowers the DPO loss")
    check(down['margin'] < 0 and down['dpo'] > t['dpo'], "raising the rejected log-prob raises it")
    check(abs(CD.dpo_terms(0.0, 0.0, 0.0, -1e4, 1)['dpo'] - 1e3) < 1e-6,
          "the float softplus is stable for large margins")
    torch = _torch()
    if torch is not None:
        tt = CD.dpo_terms(torch.tensor(-45.0), torch.tensor(-60.0), -50.0, -60.0, 10)
        check(abs(float(tt['loss']) - up['loss']) < 1e-5, "the torch loss equals the float loss")
    lrs = [CD.lr_at(s, 40) for s in range(41)]
    warm = int(round(CD.WARMUP_FRAC * 40))
    check(all(a < b for a, b in zip(lrs[:warm - 1], lrs[1:warm])), "warm-up rises")
    check(abs(max(lrs) - CD.LR) < 1e-15 and lrs[40] == 0.0, "peak = LR, zero after the last step")
    check(all(a >= b for a, b in zip(lrs[warm:], lrs[warm + 1:])), "then decays monotonically")
    o0, o1 = CD.epoch_order(9, 0), CD.epoch_order(9, 1)
    check(sorted(o0) == list(range(9)) and o0 != o1 and o0 == CD.epoch_order(9, 0),
          "each epoch is a fixed permutation, different per epoch")


class FakeTok:
    """Character-level tokenizer with a chat template (ids < 60)."""
    pad_token_id = 0
    eos_token_id = 1
    padding_side = 'right'

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return '<u>' + msgs[-1]['content'] + '<a>'

    def _ids(self, s):
        return [2 + (ord(ch) % 58) for ch in s]

    def __call__(self, text, add_special_tokens=True, return_tensors=None, padding=False):
        if isinstance(text, list):
            import torch
            ids = [self._ids(t) for t in text]
            m = max(len(x) for x in ids)
            left = self.padding_side == 'left'
            pad = lambda x: ([self.pad_token_id] * (m - len(x)) + x) if left else (x + [self.pad_token_id] * (m - len(x)))
            att = lambda x: ([0] * (m - len(x)) + [1] * len(x)) if left else ([1] * len(x) + [0] * (m - len(x)))
            return {'input_ids': torch.tensor([pad(x) for x in ids]),
                    'attention_mask': torch.tensor([att(x) for x in ids])}
        return {'input_ids': self._ids(text)}

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(40 + int(i) % 50) for i in ids if int(i) > 1)

    def convert_tokens_to_ids(self, t):
        return 1


def part4():
    print("\nPART 4 - encoding")
    tok = FakeTok()
    pair = {'problem': 'Two apples?', 'chosen': 'abc', 'rejected': 'abcdef'}
    e = CD.encode_pair(tok, pair, 1)
    prompt = tok._ids(tok.apply_chat_template([{'role': 'user', 'content': CD.prompt_content('Two apples?')}]))
    check(e['c_ids'][:e['n_prompt']] == prompt == e['r_ids'][:e['n_prompt']],
          "chosen and rejected share the chat-template prompt, token for token")
    check(e['n_tok_c'] == 4 and e['n_tok_r'] == 7 and e['c_ids'][-1] == 1,
          "responses closed with the end-of-turn token and counted")
    check(CD.encode_pair(tok, dict(pair, chosen='z' * CD.MAX_LEN_TOKENS), 1) is None,
          "a pair longer than the budget is dropped, not cut")


def part5():
    print("\nPART 5 - reading the results")
    check(CD.perm_p([0.0] * 10) == 1.0, "no difference -> p = 1")
    check(CD.perm_p([0.2] * 30) < 0.001, "a consistent difference -> small p")
    lo, hi = CD.template_ci([0.1] * 5 + [0.3] * 5, [1] * 5 + [2] * 5)
    check(10.0 - 1e-9 <= lo <= hi <= 30.0 + 1e-9, "template bootstrap interval within the template means")
    check(CD.vote([_s(3.0), _s(4.0), _s(4.0)]) == 4.0 and CD.vote([]) is None, "plurality vote")
    check(CD.recites([_s(14.0), _s(12.0)], 14.0) == 0.5 and CD.recites([_s(1.0)], None) is None,
          "recitation rate = share of samples giving the familiar answer")
    frozen = (T.VALID_MAX_PP, T.PRIMARY_MIN_PP, T.PRIMARY_ALPHA, T.ATTR_MIN_PP, T.HARM_MIN_PP,
              T.K_EVAL, T.K_GUARD, T.K_VALID, T.N_VALID, CD.ARMS, CD.BETA, CD.NLL_WEIGHT, CD.LR,
              CD.EPOCHS, CD.ACCUM, CD.LORA_R, CD.MAX_CHOSEN, CD.MAX_NEG, CD.K_FAMILIAR, CD.FOLD_SEED)
    check(frozen == (10.0, 4.0, 0.05, 2.0, -5.0, 4, 2, 4, 24, ('FV', 'RW'), 0.1, 1.0, 1e-4, 2, 4, 16,
                     2, 2, 2, 28), "the pre-registered bars and settings are the frozen ones")


def part6():
    print("\nPART 6 - the stub run end to end")
    with tempfile.TemporaryDirectory() as d:
        outs = [os.path.join(d, f'w{w}.json') for w in (0, 1)]
        with contextlib.redirect_stdout(io.StringIO()):
            for w in (0, 1):
                T.run_worker(_args(worker=w, out=outs[w]))
        st = {w: V24._read(outs[w]) for w in (0, 1)}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T.summarise(st)
        s = buf.getvalue()
        check('PRIMARY' in s and 'ATTRIBUTE' in s and 'HARM' in s and 'VALIDITY' in s,
              "a finished two-worker run prints every pre-registered line")
        check(all(len(st[w]['train'].get(a, {}).get('steps', [])) > 0 for w in (0, 1) for a in CD.ARMS),
              "both arms were trained on both workers")
        before = json.dumps(st[0]['rows'], sort_keys=True)
        with contextlib.redirect_stdout(io.StringIO()):
            T.run_worker(_args(worker=0, out=outs[0]))
        check(json.dumps(V24._read(outs[0])['rows'], sort_keys=True) == before,
              "re-running a finished worker draws nothing new")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T.summarise({0: st[0]})
        check('only one worker' in buf.getvalue() and 'PRIMARY' not in buf.getvalue(),
              "one worker alone gives no verdict")
        cut = os.path.join(d, 'cut.json')
        with contextlib.redirect_stdout(io.StringIO()):
            T.run_worker(_args(worker=1, out=cut, max_hours=-1.0))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T.summarise({0: st[0], 1: V24._read(cut)})
        check('PARTIAL' in buf.getvalue() and 'PRIMARY' not in buf.getvalue(),
              "a worker stopped by --max-hours gives no verdict")
        with contextlib.redirect_stdout(io.StringIO()):
            T.run_worker(_args(worker=1, out=cut))
        check(all(len(V24._read(cut)['rows'].get(k, {}).get('eval_FV', [])) == T.K_EVAL
                  for k in V24._read(cut)['meta']['eval_keys']), "the same command resumes it")
        # one GPU (Colab): FV on both workers first, RW in later sessions, one --out-dir
        od = os.path.join(d, 'drive')
        with contextlib.redirect_stdout(io.StringIO()):
            for w in (0, 1):
                T.run_worker(_args(worker=w, arms=['FV'], out_dir=od))
        seq = {w: V24._read(os.path.join(od, T.OUT['stub'].format(w=w))) for w in (0, 1)}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T.summarise(seq)
        s = buf.getvalue()
        check('PRIMARY' in s and 'ATTRIBUTE not read yet' in s,
              "FV-only sessions already give PRIMARY; ATTRIBUTE waits for RW")
        with contextlib.redirect_stdout(io.StringIO()):
            for w in (0, 1):
                T.run_worker(_args(worker=w, arms=['RW'], out_dir=od))
        seq = {w: V24._read(os.path.join(od, T.OUT['stub'].format(w=w))) for w in (0, 1)}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T.summarise(seq)
        check('ATTRIBUTE FV - RW' in buf.getvalue() and seq[0]['meta']['arms'] == ['FV', 'RW'],
              "the RW sessions complete the same files, and the summary merges both arms")
        check(all(len(seq[0]['rows'][k].get('eval_FV', [])) == T.K_EVAL for k in seq[0]['meta']['eval_keys']),
              "an RW session leaves the FV samples untouched")
        rec = st[0]['rows']
        fam = [k for k in st[0]['meta']['train_keys'] if rec.get(k, {}).get('familiar')]
        check(fam and all(k not in st[0]['meta']['eval_keys'] for k in fam),
              "familiar versions are drawn for training rows only")
    check(all(T.OUT['stub'] != T.OUT['real'] and T.OUT['smoke'] != T.OUT['real']
              and T.ADAPTERS['stub'] != T.ADAPTERS['real'] for _ in [0]),
          "stub and smoke runs never write the real files")


# ---------------------------------------------------------------------------
# model tests
# ---------------------------------------------------------------------------

def _torch():
    try:
        import torch
        import transformers  # noqa: F401
        return torch
    except Exception as exc:
        SKIPS.append(f'torch/transformers unavailable: {exc}')
        return None


def _tiny_engine(d):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=False)
    m = Qwen2ForCausalLM(cfg).eval()
    m.generation_config.eos_token_id = 1
    eng = object.__new__(T.HFEngine)
    eng.args = _args(mode='smoke', max_tokens=12)
    eng.client = object()
    eng.tok, eng.base, eng.pm, eng.loaded = FakeTok(), m, None, set()
    eng.end_id, eng.pad = 1, 0
    return eng


def part7():
    print("\nPART 7 - LoRA DPO on a tiny random Qwen2 (CPU)")
    torch = _torch()
    if torch is None:
        return
    try:
        import peft  # noqa: F401
    except Exception as exc:
        SKIPS.append(f'peft unavailable: {exc}')
        return
    old = dict(T.ADAPTERS)
    old_lr, old_drop = CD.LR, CD.LORA_DROPOUT
    with tempfile.TemporaryDirectory() as d:
        T.ADAPTERS['smoke'] = os.path.join(d, 'adapters')
        # a tiny model needs a larger step to move visibly in 5 optimizer steps,
        # and no dropout so that the direction checks are deterministic
        CD.LR, CD.LORA_DROPOUT = 5e-3, 0.0
        try:
            eng = _tiny_engine(d)
            pairs = [{'key': f'k{i}', 'problem': f'Problem {i}?', 'chosen': 'Answer: 7',
                      'rejected': 'We add. Answer: 9'} for i in range(6)]
            e = CD.encode_pair(eng.tok, pairs[0], eng.end_id)
            with torch.no_grad():
                lp0, n0 = eng.logp(e['c_ids'], e['n_prompt'])
                full = eng.base(input_ids=torch.tensor([e['c_ids']])).logits[0].float()
                ref = sum(torch.log_softmax(full[t - 1], -1)[e['c_ids'][t]] for t in range(e['n_prompt'], len(e['c_ids'])))
            check(n0 == e['n_tok_c'] and abs(float(lp0) - float(ref)) < 1e-4,
                  "logp() = sum of the response tokens' log-probs (logits_to_keep slice is right)")
            log = eng.train('FV_w0', pairs, 0.0, 3)
            check(log['status'] == 'done' and os.path.exists(eng.adapter_path('FV_w0')),
                  "training finishes and saves the adapter")
            # (peft's set_adapter marks the active adapter trainable again, so this
            # is checked before anything below activates it)
            check(not any(p.requires_grad for p in eng.pm.parameters()), "nothing is left trainable after training")
            st = log['steps']
            check(st[-1]['margin'] > st[0]['margin'] and st[-1]['reward_acc'] >= st[0]['reward_acc'],
                  f"the DPO margin grows ({st[0]['margin']:+.3f} -> {st[-1]['margin']:+.3f})")
            with torch.no_grad():
                with eng.use(None):
                    lp_ref, _ = eng.logp(e['c_ids'], e['n_prompt'])
                with eng.use('FV_w0'):
                    lp_new, _ = eng.logp(e['c_ids'], e['n_prompt'])
            check(abs(float(lp_ref) - float(lp0)) < 1e-4, "with the adapter disabled the model is untouched")
            check(float(lp_new) > float(lp0), "the adapter raises the chosen response's log-prob")
            out = eng.sample([CD.prompt_content('Problem 1?'), CD.prompt_content('A longer problem 2?')],
                             2, 0, 'FV_w0')
            check(len(out) == 2 and all(len(r) == 2 and 'raw' in r[0] for r in out),
                  "batched, left-padded sampling returns k samples per row")
            # a paused run resumes from its checkpoint and finishes
            eng2 = _tiny_engine(d)
            lg = eng2.train('RW_w0', pairs, -1.0, 3)       # deadline already passed
            check(lg['status'] == 'paused' and os.path.exists(eng2.adapter_path('RW_w0', ckpt=True)),
                  "a run past its deadline pauses with a checkpoint")
            eng3 = _tiny_engine(d)
            lg = eng3.train('RW_w0', pairs, 0.0, 3)
            check(lg['status'] == 'done' and lg['steps'][-1]['step'] == lg['total_opt']
                  and not os.path.exists(eng3.adapter_path('RW_w0', ckpt=True)),
                  "the re-run resumes from the checkpoint and finishes every step")
            eng4 = _tiny_engine(d)
            eng4.load_adapter('FV_w0')
            with torch.no_grad(), eng4.use('FV_w0'):
                lp_loaded, _ = eng4.logp(e['c_ids'], e['n_prompt'])
            check(abs(float(lp_loaded) - float(lp_new)) < 1e-4, "a saved adapter loads back exactly")
        finally:
            T.ADAPTERS.clear()
            T.ADAPTERS.update(old)
            CD.LR, CD.LORA_DROPOUT = old_lr, old_drop


def part8():
    print("\nPART 8 - the notebook")
    nb = json.load(open('MAS_SHT_Kaggle_v28.ipynb', encoding='utf-8'))
    src = '\n'.join(''.join(c['source']) for c in nb['cells'])
    check("BRANCH   = 'Context-DPO'" in src, "the notebook checks out the Context-DPO branch")
    check('test_v28.py' in src and '--require-model' in src, "it runs the tests, model tests required")
    check('--smoke' in src and '--summary-only' in src, "it runs the smoke check and the summary")
    check("'--worker', '0'" in src and "'--worker', '1'" in src, "it runs both workers")
    nb = json.load(open('MAS_SHT_Colab_v28.ipynb', encoding='utf-8'))
    src = '\n'.join(''.join(c['source']) for c in nb['cells'])
    check("BRANCH   = 'Context-DPO'" in src and "OUT_DIR = '/content/drive/MyDrive/MAS_SHT_v28'" in src,
          "Colab: the Context-DPO branch, every result on Google Drive")
    check("'--out-dir', OUT_DIR" in src and '--summary-only --out-dir' in src,
          "Colab: the workers and the summary read and write the same Drive folder")
    check(src.index("for arm in ('FV', 'RW')") > 0 and 'test_v28.py' in src and '--require-model' in src,
          "Colab: tests first, then FV on both workers before RW")
    check("'transformers>=4.44.0,<4.50'" in src and "'peft>=0.12,<0.15'" in src,
          "Colab: the same pinned stack as the Kaggle runs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--require-model', action='store_true')
    a = ap.parse_args()
    for p in (part1, part2, part3, part4, part5, part6, part7, part8):
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
