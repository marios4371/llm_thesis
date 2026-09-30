"""
[v28.0] Dev screen: Context-DPO against recitation. Does training the Solver to
prefer a solution of THIS problem over the solution of its familiar version
make each reasoning chain right more often, on templates it was never
trained on?

The method is in context_dpo.py; the evidence and the literature in
V28_PLAN.md. This is the thesis's first weight update: every earlier version
selected among a frozen model's outputs.

ROWS (all stored; nothing here is chosen after seeing a tuned sample)
----
  main   the 200 GSM-Symbolic P2 rows with 5 stored plain samples each:
         100 from the v21 dev run (seed 44) and 100 from the v22 confirmation
         (seed 45). 49 templates.
  guard  the 60 GSM-Plus distractor rows of the same runs (40 + 20).
The 49 templates are split in two folds (context_dpo.split_folds, seed 28).
Worker w trains on the rows of fold w and is evaluated on the rows of the
OTHER fold, so every one of the 200 rows is scored by a model that never saw
any instance of its template. The two workers run in parallel, one per T4.

PER WORKER
----------
  1 prototypes   the Prototype agent (Qwen2.5-7B-Instruct, greedy, the v26
                 prompt) writes the familiar version of each training row that
                 has none yet (the v22 rows reuse the v27 prototypes)
  2 familiar     the Solver samples the familiar version K_FAMILIAR=2 times
  3 validity     the untouched Solver re-samples 24 evaluation rows (k=4): the
                 new batched sampler must reproduce the stored samples
  4 per arm (FV, then RW): pairs (context_dpo.build_pairs), LoRA DPO+NLL on
                 the training fold, then k=4 samples on every evaluation row
                 and k=2 on this worker's 30 guard rows
Sampling: the v22 sampler (T=0.8, top-p 0.95, top-k 50, renormalised, 1024
new tokens, the CoT prompt of every earlier version), several rows per
generate call with left padding.

PRE-REGISTERED, frozen 2026-09-30 before any adapter exists
-----------------------------------------------------------
Metric: per-sample accuracy (mean over rows of right/k), paired by row with
the row's 5 stored plain samples (BASE); two-sided sign-flip permutation test
over rows; template bootstrap reported.
  VALIDITY   |re-sampled base - stored base| <= 10 pp on the 48 validity rows
             (both workers). Otherwise nothing below is read.
  PRIMARY    all 200 main rows: FV - BASE >= +4 pp AND perm p < 0.05
                -> SUPPORTED;  <= 0 -> REFUTED;  otherwise INCONCLUSIVE
  ATTRIBUTE  FV - RW >= +2 pp -> the familiar-version negatives are what
             works; otherwise any wrong sample does about as well
  HARM       guard rows: FV - BASE >= -5 pp
  GO         SUPPORTED, no HARM -> fresh-row confirmation: train on all 200
             rows, evaluate on the v25 seed-48 rows against a re-sampled base
Reported, not decisive: SC@4 against the first 4 stored samples, the
recitation rate (samples whose answer is the familiar version's answer), the
hard rows (stored per-sample <= 0.4), the training curves, RW - BASE.
These rows motivated v28, so this screen decides whether a confirmation is
worth running; it is not a thesis claim by itself.

MODES
-----
    python pretest_v28.py --worker 0 --device 0 --max-hours 8.5   # one per GPU
    python pretest_v28.py --worker 1 --device 1 --max-hours 8.5
    python pretest_v28.py --summary-only        # merges both workers, no GPU
    python pretest_v28.py --smoke --worker 0    # ~10 min GPU check, own files
    python pretest_v28.py --stub --worker 0     # offline plumbing, no GPU
Re-running the same command resumes (prototypes, samples and training
checkpoints are saved as they are made).
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence

import context_dpo as CD
import pretest_v21 as V21
import pretest_v24 as V24
import pretest_v26 as V26

V21_TRACES = 'pretest_data/v21_traces.json'
V21_MANIFESTS = ('pretest_data/v20_problems.json', 'pretest_data/v21_guard_distraction.json')
DEV_V25 = 'results_September/preset_V25_dev.json'
DEV_V26 = 'results_September/preset_V26_dev.json'
DEV_V27 = 'results_September/preset_V27_dev.json'

OUT = {'real': 'pretest_v28_w{w}.json', 'stub': 'pretest_v28_stub_w{w}.json',
       'smoke': 'pretest_v28_smoke_w{w}.json'}
ADAPTERS = {'real': 'v28_adapters', 'stub': 'v28_adapters_stub', 'smoke': 'v28_adapters_smoke'}

K_EVAL = 4
K_GUARD = 2
K_VALID = 4
N_VALID = 24
BATCH_SEQS = 16                # sequences per generate call
CKPT_EVERY = 8                 # optimizer steps between training checkpoints
SMOKE = {'train': 3, 'eval': 2, 'guard': 1, 'valid': 1, 'pairs': 2}

VALID_MAX_PP = 10.0
PRIMARY_MIN_PP = 4.0
PRIMARY_ALPHA = 0.05
ATTR_MIN_PP = 2.0
HARM_MIN_PP = -5.0
HARD_MAX = 0.4


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def _plain(samples) -> List[Dict]:
    return [{'raw': str(s.get('raw') or ''), 'answer': CD.as_float(s.get('answer'))}
            for s in samples]


def load_rows() -> List[Dict]:
    """main + guard rows with their stored plain samples. `base` is the
    comparator (5 plain samples); `extra` are further right-answer
    candidates for the chosen response, in order: v26 plain (L0), v26 other
    samplers, reader-view (Q) samples."""
    rows: List[Dict] = []
    texts = V24._texts(V21_MANIFESTS)
    for r in V24._read(V21_TRACES)['rows']:
        rows.append({'key': f"v21:{r['pid']}", 'pid': r['pid'], 'src': 'v21', 'set': r['set'],
                     'gold': CD.as_float(r['gold']), 'text': texts[r['pid']],
                     'template': V24.template_of(r['pid']), 'base': _plain(r['samples']),
                     'extra': [], 'prototype': None})
    v26 = {r['pid']: r for r in V24._read(DEV_V26)['rows']}
    v27 = {r['pid']: r for r in V24._read(DEV_V27)['rows']}
    for r in V24._read(DEV_V25)['rows']:
        pid = r['pid']
        arms = v26.get(pid, {}).get('arms', {})
        extra = _plain(arms.get('L0', {}).get('samples', []))
        for a in ('MINP', 'PCD', 'PCD5', 'CADQ'):
            extra += _plain(arms.get(a, {}).get('samples', []))
        extra += _plain(r.get('Q', []))
        proto = v27.get(pid, {}).get('prototype')
        rows.append({'key': f'v22:{pid}', 'pid': pid, 'src': 'v22', 'set': r['set'],
                     'gold': CD.as_float(r['gold']), 'text': r['text'],
                     'template': V24.template_of(pid), 'base': _plain(r['C']),
                     'extra': extra, 'prototype': proto})
    return rows


def plan(rows: List[Dict], w: int, mode: str) -> Dict[str, List[Dict]]:
    """Which rows worker w trains on, evaluates, guards and validates."""
    main = sorted([r for r in rows if r['set'] == 'main'], key=lambda r: r['key'])
    guard = sorted([r for r in rows if r['set'] != 'main'], key=lambda r: r['key'])
    folds = CD.split_folds(main)
    half = CD.split_list([r['key'] for r in guard], CD.FOLD_SEED)
    train = [r for r in main if folds[r['template']] == w]
    evalr = [r for r in main if folds[r['template']] != w]
    grd = [r for r in guard if half[r['key']] == w]
    valid = sorted(random.Random(CD.FOLD_SEED + w).sample(evalr, min(N_VALID, len(evalr))),
                   key=lambda r: r['key'])
    if mode == 'smoke':
        # rows that can give pairs (a right and a wrong stored sample); the v21
        # ones sort first, so the smoke also writes prototypes
        ok = [r for r in train if (lambda c: c['chosen'] and c['RW'])(CD.candidates(r))]
        train, evalr, grd = ok[:SMOKE['train']], evalr[:SMOKE['eval']], grd[:SMOKE['guard']]
        valid = evalr[:SMOKE['valid']]
    return {'train': train, 'eval': evalr, 'guard': grd, 'valid': valid, 'folds': folds}


def seed_of(*parts) -> int:
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# engines
# ---------------------------------------------------------------------------

class HFEngine:
    """The Solver (Qwen2.5-Math-7B-Instruct, 4-bit NF4, bf16 compute) pinned to
    one GPU, with LoRA adapters trained and swapped in place."""

    def __init__(self, args):
        import torch
        from Mas_solver import (AgentRole, HETEROGENEOUS_PRESETS, LOCAL_HF_SAMPLING,
                                UnifiedLLMClient)
        self.args = args
        cfg = HETEROGENEOUS_PRESETS[args.preset]
        self.solver_cfg = cfg[AgentRole.BASELINE]
        self.reader_cfg = cfg[AgentRole.MATHEMATICIAN]
        LOCAL_HF_SAMPLING.update(enabled=True)
        self.dev_index = args.device if torch.cuda.is_available() else None
        self._UC = UnifiedLLMClient
        self.client = None
        self.pm = None
        self.loaded = set()

    # -- models -------------------------------------------------------------
    def reader(self):
        m = self.reader_cfg
        print(f"  prototype agent: {m.model_name} (greedy) on cuda:{self.dev_index}", flush=True)
        return self._UC(provider=m.provider, use_cache=False, model_override=m.model_name,
                        load_4bit=getattr(m, 'load_4bit', False), device_index=self.dev_index)

    @staticmethod
    def free(client) -> None:
        if client is not None and hasattr(client, '_local_model'):
            client._local_model = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def solver(self):
        if self.client is None:
            m = self.solver_cfg
            print(f"  solver: {m.model_name} on cuda:{self.dev_index}", flush=True)
            self.client = self._UC(provider=m.provider, use_cache=False, model_override=m.model_name,
                                   load_4bit=getattr(m, 'load_4bit', False),
                                   device_index=self.dev_index)
            self.client._ensure_local_model()
            self.tok = self.client._local_tokenizer
            self.base = self.client._local_model
            ids = self.base.generation_config.eos_token_id
            ids = list(ids) if isinstance(ids, (list, tuple)) else [ids]
            im_end = self.tok.convert_tokens_to_ids('<|im_end|>')
            self.end_id = im_end if isinstance(im_end, int) and im_end >= 0 else ids[0]
            self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else ids[0]
        return self

    def model(self):
        return self.pm if self.pm is not None else self.base

    def device(self):
        return next(self.base.parameters()).device

    # -- adapters -----------------------------------------------------------
    def _lora_cfg(self):
        from peft import LoraConfig
        return LoraConfig(r=CD.LORA_R, lora_alpha=CD.LORA_ALPHA, lora_dropout=CD.LORA_DROPOUT,
                          target_modules=list(CD.LORA_TARGETS), bias='none', task_type='CAUSAL_LM')

    def new_adapter(self, name: str) -> None:
        from peft import get_peft_model
        if self.pm is None:
            self.pm = get_peft_model(self.base, self._lora_cfg(), adapter_name=name)
        else:
            self.pm.add_adapter(name, self._lora_cfg())
        self.pm.set_adapter(name)
        self.loaded.add(name)

    def use(self, name: Optional[str]):
        """Context: sample with adapter `name`, or the untouched model if None."""
        if name is None:
            return self.pm.disable_adapter() if self.pm is not None else contextlib.nullcontext()
        self.pm.set_adapter(name)
        return contextlib.nullcontext()

    def adapter_path(self, name: str, ckpt: bool = False) -> str:
        d = ADAPTERS[self.args.mode]
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f'{name}.ckpt.pt' if ckpt else f'{name}.pt')

    def has_adapter(self, name: str) -> bool:
        return os.path.exists(self.adapter_path(name))

    def load_adapter(self, name: str) -> None:
        import torch
        from peft import set_peft_model_state_dict
        if name in self.loaded:
            return
        self.new_adapter(name)
        sd = torch.load(self.adapter_path(name), map_location='cpu', weights_only=False)
        set_peft_model_state_dict(self.pm, sd, adapter_name=name)
        print(f"  loaded adapter {name} from {self.adapter_path(name)}", flush=True)

    # -- sampling -----------------------------------------------------------
    def sample(self, contents: List[str], k: int, seed: int, adapter: Optional[str]) -> List[List[Dict]]:
        """k samples for each content, several contents per generate call."""
        import torch
        self.solver()
        mdl = self.model()
        mdl.eval()
        out: List[List[Dict]] = []
        per = max(1, BATCH_SEQS // k)
        i = 0
        while i < len(contents):
            chunk = contents[i:i + per]
            try:
                with self.use(adapter), torch.no_grad():
                    out += self._generate(mdl, chunk, k, seed + i)
                i += len(chunk)
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                if per == 1:
                    raise
                per = max(1, per // 2)
                print(f"    (out of memory; {per} rows per generate call)", flush=True)
        return out

    def _generate(self, mdl, contents, k, seed):
        import torch
        tok = self.tok
        prompts = [tok.apply_chat_template([{'role': 'user', 'content': c}], tokenize=False,
                                           add_generation_prompt=True) for c in contents]
        side = tok.padding_side
        tok.padding_side = 'left'
        enc = tok(prompts, return_tensors='pt', padding=True)
        tok.padding_side = side
        dev = self.device()
        enc = {kk: v.to(dev) for kk, v in enc.items()}
        torch.manual_seed(seed)
        from Mas_solver import LOCAL_HF_SAMPLING
        gen = mdl.generate(**enc, max_new_tokens=self.args.max_tokens, do_sample=True,
                           temperature=float(self.args.temperature),
                           top_p=float(LOCAL_HF_SAMPLING.get('top_p', 0.95)),
                           top_k=int(LOCAL_HF_SAMPLING.get('top_k', 50)),
                           renormalize_logits=True, num_return_sequences=k,
                           pad_token_id=self.pad, use_cache=True)
        n_in = enc['input_ids'].shape[1]
        res = []
        for j in range(len(contents)):
            row = []
            for s in gen[j * k:(j + 1) * k]:
                raw = tok.decode(s[n_in:], skip_special_tokens=True)
                row.append({'raw': raw[:6000], 'answer': V21.parse_answer(raw)})
            res.append(row)
        return res

    # -- training -----------------------------------------------------------
    def _keep_kw(self, n: int) -> Dict[str, int]:
        import inspect
        base = self.pm.get_base_model() if self.pm is not None else self.base
        try:
            params = inspect.signature(type(base).forward).parameters
        except (TypeError, ValueError):
            return {}
        for name in ('logits_to_keep', 'num_logits_to_keep'):
            if name in params:
                return {name: n}
        return {}

    def logp(self, ids: List[int], n_prompt: int):
        """Sum of the response tokens' log-probs (a tensor, graph kept if
        grad is enabled) and their number."""
        import torch
        import torch.nn.functional as F
        mdl = self.model()
        x = torch.tensor([ids], device=self.device())
        n_resp = len(ids) - n_prompt
        out = mdl(input_ids=x, use_cache=False, **self._keep_kw(n_resp + 1))
        logits = out.logits[0, -(n_resp + 1):-1].float()
        lp = -F.cross_entropy(logits, x[0, n_prompt:], reduction='none')
        return lp.sum(), n_resp

    def train(self, name: str, pairs: List[Dict], deadline: float, epochs: int) -> Dict:
        import torch
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        self.solver()
        enc = [CD.encode_pair(self.tok, p, self.end_id) for p in pairs]
        dropped = sum(e is None for e in enc)
        enc = [e for e in enc if e is not None]
        if not enc:
            raise RuntimeError(f'{name}: no trainable pairs')
        ck_path = self.adapter_path(name, ckpt=True)
        ck = (torch.load(ck_path, map_location='cpu', weights_only=False)
              if os.path.exists(ck_path) else None)
        # reference log-probs: the untouched model
        if ck is not None:
            ref = ck['ref']
        else:
            ref = []
            with self.use(None), torch.no_grad():
                self.model().eval()
                t0 = time.time()
                for i, e in enumerate(enc):
                    rc, _ = self.logp(e['c_ids'], e['n_prompt'])
                    rr, _ = self.logp(e['r_ids'], e['n_prompt'])
                    ref.append((float(rc), float(rr)))
                    if i == 0:
                        print(f"    reference pass: {time.time() - t0:.1f} s for the first pair", flush=True)
        self.new_adapter(name)
        params = [p for n, p in self.pm.named_parameters() if 'lora_' in n and f'.{name}.' in n]
        mine = {id(p) for p in params}
        for p in self.pm.parameters():
            p.requires_grad = id(p) in mine
        opt = torch.optim.AdamW(params, lr=CD.LR, weight_decay=0.0)
        n = len(enc)
        total_micro = epochs * n
        total_opt = math.ceil(total_micro / CD.ACCUM)
        log = {'pairs': n, 'dropped_too_long': dropped, 'steps': [], 'total_opt': total_opt}
        start = 0
        if ck is not None:
            set_peft_model_state_dict(self.pm, ck['lora'], adapter_name=name)
            opt.load_state_dict(ck['opt'])
            start, log = ck['micro'], ck['log']
            print(f"    resuming {name} at pair-step {start}/{total_micro}", flush=True)
        base = self.pm.get_base_model()
        base.config.use_cache = False
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        base.enable_input_require_grads()
        self.pm.set_adapter(name)
        self.pm.train()
        acc = collections.defaultdict(float)
        t0 = time.time()
        status = 'done'
        for m in range(start, total_micro):
            epoch, pos = divmod(m, n)
            idx = CD.epoch_order(n, epoch)[pos]
            e = enc[idx]
            pc, nc = self.logp(e['c_ids'], e['n_prompt'])
            pr, _ = self.logp(e['r_ids'], e['n_prompt'])
            t = CD.dpo_terms(pc, pr, ref[idx][0], ref[idx][1], nc)
            (t['loss'] / CD.ACCUM).backward()
            for kk in ('loss', 'dpo', 'nll', 'margin'):
                acc[kk] += float(t[kk].detach())
            acc['win'] += float(t['margin'].detach()) > 0
            acc['dc'] += float(pc.detach()) - ref[idx][0]
            acc['n'] += 1
            if (m + 1) % CD.ACCUM == 0 or m + 1 == total_micro:
                step = m // CD.ACCUM
                for g in opt.param_groups:
                    g['lr'] = CD.lr_at(step, total_opt)
                torch.nn.utils.clip_grad_norm_(params, CD.MAX_GRAD_NORM)
                opt.step()
                opt.zero_grad(set_to_none=True)
                k = acc['n']
                rec = {'step': step + 1, 'epoch': epoch + 1, 'lr': CD.lr_at(step, total_opt),
                       'loss': acc['loss'] / k, 'dpo': acc['dpo'] / k, 'nll': acc['nll'] / k,
                       'margin': acc['margin'] / k, 'reward_acc': acc['win'] / k,
                       'chosen_logp_change': acc['dc'] / k,
                       's_per_pair': (time.time() - t0) / max(1, m + 1 - start)}
                log['steps'].append(rec)
                acc.clear()
                if (step + 1) % 4 == 0 or step + 1 == total_opt:
                    print(f"    {name} step {step + 1}/{total_opt} ep{epoch + 1} loss={rec['loss']:.3f} "
                          f"dpo={rec['dpo']:.3f} margin={rec['margin']:+.2f} acc={rec['reward_acc']:.2f} "
                          f"dlogp(chosen)={rec['chosen_logp_change']:+.1f} "
                          f"{rec['s_per_pair']:.1f}s/pair", flush=True)
                last = m + 1 == total_micro
                if not last and ((step + 1) % CKPT_EVERY == 0 or (deadline and time.time() > deadline)):
                    torch.save({'micro': m + 1, 'opt': opt.state_dict(), 'ref': ref, 'log': log,
                                'lora': get_peft_model_state_dict(self.pm, adapter_name=name, save_embedding_layers=False)}, ck_path)
                    if deadline and time.time() > deadline:
                        status = 'paused'
                        break
        base.gradient_checkpointing_disable()
        base.config.use_cache = True
        self.pm.eval()
        for p in self.pm.parameters():
            p.requires_grad = False
        if status == 'done':
            torch.save(get_peft_model_state_dict(self.pm, adapter_name=name, save_embedding_layers=False), self.adapter_path(name))
            if os.path.exists(ck_path):
                os.remove(ck_path)
            log['seconds'] = round(time.time() - t0, 1)
        log['status'] = status
        return log


class StubEngine:
    """Offline stand-in: right with a probability that depends on the adapter."""
    P = {None: 0.6, 'FV': 0.7, 'RW': 0.65}
    P_GUARD = 0.9

    def __init__(self, args, rows: List[Dict]):
        self.args = args
        self.gold = {}
        self.guard = set()
        for r in rows:
            self.gold[CD.prompt_content(r['text'])] = r['gold']
            if r['set'] != 'main':
                self.guard.add(CD.prompt_content(r['text']))
        self.adapters = set()
        self.tok = None

    def reader(self):
        return V26._StubReader()

    @staticmethod
    def free(client) -> None:
        return None

    def solver(self):
        return self

    def register(self, content: str, gold) -> None:
        self.gold[content] = gold

    def sample(self, contents, k, seed, adapter):
        rng = random.Random(seed)
        out = []
        for c in contents:
            g = self.gold.get(c)
            row = []
            for _ in range(k):
                if g is None:            # a prototype: recite, i.e. a wrong answer
                    a = 999.0
                else:
                    p = self.P_GUARD if c in self.guard else self.P[adapter.split('_')[0] if adapter else None]
                    a = g if rng.random() < p else g + rng.choice([1, 2, 3])
                row.append({'raw': f"We compute step by step.\nAnswer: {a:g}", 'answer': float(a)})
            out.append(row)
        return out

    def has_adapter(self, name):
        return name in self.adapters

    def load_adapter(self, name):
        return None

    def train(self, name, pairs, deadline, epochs):
        if not pairs:
            raise RuntimeError(f'{name}: no trainable pairs')
        self.adapters.add(name)
        return {'pairs': len(pairs), 'dropped_too_long': 0, 'status': 'done', 'total_opt': 1,
                'steps': [{'step': 1, 'loss': 0.6, 'dpo': 0.6, 'nll': 0.3, 'margin': 0.1,
                           'reward_acc': 0.6, 'chosen_logp_change': 0.5, 's_per_pair': 0.0}]}


# ---------------------------------------------------------------------------
# the worker
# ---------------------------------------------------------------------------

def _save(path: str, state: Dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(state, fh, ensure_ascii=False)
    os.replace(tmp, path)


def _rec(state: Dict, key: str) -> Dict:
    return state['rows'].setdefault(key, {})


def with_state(row: Dict, st: Dict) -> Dict:
    """The row as context_dpo sees it: stored samples + this worker's
    prototype and familiar samples."""
    out = dict(row)
    if st.get('prototype') is not None:
        out['prototype'] = st['prototype']
    out['familiar'] = st.get('familiar', [])
    return out


def run_worker(args) -> int:
    w = args.worker
    mode = args.mode
    out_path = args.out or OUT[mode].format(w=w)
    rows = load_rows()
    P = plan(rows, w, mode)
    state = {'worker': w, 'rows': {}, 'train': {}}
    if os.path.exists(out_path) and args.resume:
        state = V24._read(out_path)
        print(f"resuming worker {w} from {out_path}")
    state['meta'] = {'mode': mode, 'k_eval': K_EVAL, 'k_guard': K_GUARD, 'k_valid': K_VALID,
                     'k_familiar': CD.K_FAMILIAR, 'temperature': args.temperature,
                     'max_tokens': args.max_tokens, 'arms': list(args.arms),
                     'train_keys': [r['key'] for r in P['train']],
                     'eval_keys': [r['key'] for r in P['eval']],
                     'guard_keys': [r['key'] for r in P['guard']],
                     'valid_keys': [r['key'] for r in P['valid']]}
    print(f"worker {w} [{mode}]: train {len(P['train'])} rows "
          f"({len({r['template'] for r in P['train']})} templates), eval {len(P['eval'])} rows "
          f"({len({r['template'] for r in P['eval']})} templates), guard {len(P['guard'])}, "
          f"validity {len(P['valid'])}", flush=True)
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else 0.0

    def late() -> bool:
        if deadline and time.time() > deadline:
            _save(out_path, state)
            print("\n  --max-hours reached; re-run the identical command to resume.", flush=True)
            return True
        return False

    eng = StubEngine(args, rows) if mode == 'stub' else HFEngine(args)

    # 1 prototypes -----------------------------------------------------------
    need = [r for r in P['train'] if r['prototype'] is None
            and _rec(state, r['key']).get('prototype') is None]
    if need:
        V26._PROTOS.update(n=0, ok=0, shown=False)
        reader = eng.reader()
        t0 = time.time()
        for i, r in enumerate(need, 1):
            if late():
                return 0
            _rec(state, r['key'])['prototype'] = V26.write_prototype(reader, r['text'])
            _save(out_path, state)
            if i % 10 == 0 or i == len(need):
                print(f"  prototypes {i}/{len(need)}  {(time.time() - t0) / i:.1f} s each", flush=True)
        eng.free(reader)
        del reader
    eng.solver()

    # 2 familiar samples -----------------------------------------------------
    need = [r for r in P['train']
            if CD.edited(with_state(r, _rec(state, r['key']))['prototype'])
            and len(_rec(state, r['key']).get('familiar', [])) < CD.K_FAMILIAR]
    if need:
        t0 = time.time()
        step = max(1, BATCH_SEQS // CD.K_FAMILIAR)
        for i in range(0, len(need), step):
            if late():
                return 0
            chunk = need[i:i + step]
            texts = [with_state(r, _rec(state, r['key']))['prototype']['text'] for r in chunk]
            res = eng.sample([CD.prompt_content(t) for t in texts], CD.K_FAMILIAR,
                             seed_of('familiar', w, i), None)
            for r, ss in zip(chunk, res):
                _rec(state, r['key'])['familiar'] = ss
            _save(out_path, state)
            print(f"  familiar versions {min(i + step, len(need))}/{len(need)}  "
                  f"{(time.time() - t0) / min(i + step, len(need)):.0f} s/row", flush=True)

    # 3 validity: the untouched Solver, the new batched sampler -------------
    if not fill(eng, state, P['valid'], 'valid', None, K_VALID, out_path, late, seed_of('valid', w)):
        return 0
    print("  " + validity_line([(r, _rec(state, r['key'])['valid']) for r in P['valid']]), flush=True)

    # 4 arms ------------------------------------------------------------------
    for arm in args.arms:
        name = f'{arm}_w{w}'
        tr = state['train'].get(arm, {})
        if tr.get('status') != 'done' or not eng.has_adapter(name):
            pairs = [p for r in P['train'] for p in CD.build_pairs(with_state(r, _rec(state, r['key'])), arm)]
            if mode == 'smoke':
                if not pairs:       # plumbing only: the training code is the same for both arms
                    print(f"  smoke: no {arm} pairs on {len(P['train'])} rows; training on RW pairs", flush=True)
                    pairs = [p for r in P['train']
                             for p in CD.build_pairs(dict(with_state(r, _rec(state, r['key'])),
                                                          prototype={'ok': True, 'identical': False},
                                                          familiar=r['base']), 'RW')]
                pairs = pairs[:SMOKE['pairs']]
            rows_used = len({p['key'] for p in pairs})
            print(f"  {arm}: {len(pairs)} pairs from {rows_used} rows", flush=True)
            if not pairs:
                state['train'][arm] = {'status': 'no pairs', 'pairs': 0, 'rows': 0, 'steps': []}
                _save(out_path, state)
                print(f"  {arm}: NO training pairs on worker {w}; this arm cannot be scored", flush=True)
                continue
            log = eng.train(name, pairs, deadline, 1 if mode == 'smoke' else CD.EPOCHS)
            log['rows'] = rows_used
            state['train'][arm] = log
            _save(out_path, state)
            if log['status'] != 'done':
                late()
                return 0
        else:
            eng.load_adapter(name)
        if not fill(eng, state, P['eval'], f'eval_{arm}', name, K_EVAL, out_path, late,
                    seed_of('eval', arm, w)):
            return 0
        if not fill(eng, state, P['guard'], f'eval_{arm}', name, K_GUARD, out_path, late,
                    seed_of('guard', arm, w)):
            return 0
        print(f"  {arm}: evaluation done", flush=True)
    _save(out_path, state)
    print(f"\n  worker {w} finished; saved {out_path}", flush=True)
    return 0


def fill(eng, state, rows, field, adapter, k, out_path, late, seed) -> bool:
    """Draw k samples into state.rows[key][field] for every row still short."""
    need = [r for r in rows if len(_rec(state, r['key']).get(field, [])) < k]
    if not need:
        return True
    per = max(1, BATCH_SEQS // k)
    t0 = time.time()
    for i in range(0, len(need), per):
        if late():
            return False
        chunk = need[i:i + per]
        res = eng.sample([CD.prompt_content(r['text']) for r in chunk], k, seed + i, adapter)
        for r, ss in zip(chunk, res):
            _rec(state, r['key'])[field] = ss
        _save(out_path, state)
        done = min(i + per, len(need))
        right = sum(CD.right(s, r['gold']) for r in need[:done] for s in _rec(state, r['key'])[field])
        print(f"  {field:8s} {done:3d}/{len(need)} rows  per-sample {100 * right / (done * k):5.1f}%  "
              f"{(time.time() - t0) / done:.0f} s/row", flush=True)
    return True


# ---------------------------------------------------------------------------
# reading the results (both workers)
# ---------------------------------------------------------------------------

def validity_line(pairs) -> str:
    if not pairs:
        return 'validity: no rows'
    d = [CD.per_sample(v, r['gold']) - CD.per_sample(r['base'], r['gold']) for r, v in pairs]
    return (f"validity: re-sampled base - stored base = {100 * sum(d) / len(d):+.1f} pp "
            f"on {len(d)} rows (bar +/-{VALID_MAX_PP:.0f})")


def collect(states: Dict[int, Dict], rows: List[Dict]) -> Dict:
    """Per main/guard row: stored base, the tuned samples from the worker that
    did NOT train on its template, and the familiar answer from the worker
    that did."""
    by = {r['key']: r for r in rows}
    main, guard, valid, fam = {}, {}, [], {}
    for w, st in states.items():
        meta = st.get('meta', {})
        for key in meta.get('train_keys', []):
            f = CD.familiar_answer(st['rows'].get(key, {}).get('familiar', []), by[key]['gold'])
            fam[key] = f
        for key in meta.get('eval_keys', []):
            main[key] = st['rows'].get(key, {})
        for key in meta.get('guard_keys', []):
            guard[key] = st['rows'].get(key, {})
        for key in meta.get('valid_keys', []):
            v = st['rows'].get(key, {}).get('valid', [])
            if len(v) >= K_VALID:
                valid.append((by[key], v))
    return {'main': main, 'guard': guard, 'valid': valid, 'fam': fam, 'by': by}


def compare(rows: List[Dict], recs: Dict[str, Dict], arm: str, other: Optional[str], k: int) -> Dict:
    """arm - other (None = stored base), per-sample, paired by row."""
    d, t = [], []
    for r in rows:
        a = recs[r['key']].get(f'eval_{arm}', [])[:k]
        b = r['base'] if other is None else recs[r['key']].get(f'eval_{other}', [])[:k]
        d.append(CD.per_sample(a, r['gold']) - CD.per_sample(b, r['gold']))
        t.append(r['template'])
    if not d:
        return {'pp': 0.0, 'p': 1.0, 'w': 0, 'l': 0, 'ci': (0.0, 0.0), 'n': 0}
    return {'pp': 100 * sum(d) / len(d), 'p': CD.perm_p(d), 'ci': CD.template_ci(d, t),
            'w': sum(1 for v in d if v > 1e-9), 'l': sum(1 for v in d if v < -1e-9), 'n': len(d)}


def _complete(recs, rows, arm, k) -> bool:
    return all(len(recs.get(r['key'], {}).get(f'eval_{arm}', [])) >= k for r in rows)


def summarise(states: Dict[int, Dict], stub: bool = False) -> int:
    rows = load_rows()
    C = collect(states, rows)
    by = C['by']
    arms = [a for a in CD.ARMS if any(a in st.get('meta', {}).get('arms', []) for st in states.values())]
    main = [by[k] for k in sorted(C['main'])]
    guard = [by[k] for k in sorted(C['guard'])]
    print('\n' + '=' * 78)
    print(f"  v28 Context-DPO screen: {len(main)} main rows ({len({r['template'] for r in main})} "
          f"templates, each scored by the model that did not train on it), {len(guard)} guard rows")
    for w, st in sorted(states.items()):
        for arm, lg in sorted(st.get('train', {}).items()):
            last = lg.get('steps', [{}])[-1] if lg.get('steps') else {}
            first = lg.get('steps', [{}])[0] if lg.get('steps') else {}
            print(f"    train {arm}_w{w}: {lg.get('pairs', 0)} pairs / {lg.get('rows', '?')} rows, "
                  f"{lg.get('status')}, reward acc {first.get('reward_acc', 0):.2f} -> "
                  f"{last.get('reward_acc', 0):.2f}, margin {last.get('margin', 0):+.2f}, "
                  f"chosen dlogp {last.get('chosen_logp_change', 0):+.1f}, {lg.get('seconds', 0) / 60:.0f} min")
    print('  ' + validity_line(C['valid']))
    have = [a for a in arms if _complete(C['main'], main, a, K_EVAL)]
    print(f"\n    {'arm':5s} {'per-sample':>10s} {'SC@4':>5s} {'hard rows':>10s} {'recites':>8s}")
    hard = [r for r in main if CD.per_sample(r['base'], r['gold']) <= HARD_MAX]
    fam_rows = [r for r in main if C['fam'].get(r['key']) is not None]

    def line(label, get):
        ps = 100 * sum(CD.per_sample(get(r), r['gold']) for r in main) / max(1, len(main))
        sc = sum(V21.correct(CD.vote(get(r)[:4]), r['gold']) for r in main)
        hd = 100 * sum(CD.per_sample(get(r), r['gold']) for r in hard) / max(1, len(hard))
        rc = [CD.recites(get(r), C['fam'][r['key']]) for r in fam_rows]
        rc = 100 * sum(rc) / len(rc) if rc else float('nan')
        print(f"    {label:5s} {ps:9.1f}% {sc:5d} {hd:9.1f}% {rc:7.1f}%")
    line('BASE', lambda r: r['base'])
    for a in have:
        line(a, lambda r, a=a: C['main'][r['key']][f'eval_{a}'][:K_EVAL])
    print(f"    (hard rows: stored per-sample <= {HARD_MAX}, n={len(hard)}; recites: share of samples "
          f"giving the familiar version's answer, n={len(fam_rows)} rows)")
    for a in have:
        c = compare(main, C['main'], a, None, K_EVAL)
        print(f"    {a} - BASE: {c['pp']:+5.1f} pp  [template bootstrap {c['ci'][0]:+.1f}, {c['ci'][1]:+.1f}]"
              f"  perm p={c['p']:.4f}  rows W={c['w']} L={c['l']}")
        g = compare(guard, C['guard'], a, None, K_GUARD) if _complete(C['guard'], guard, a, K_GUARD) else None
        if g:
            print(f"    {a} - BASE on guard: {g['pp']:+5.1f} pp  rows W={g['w']} L={g['l']}")
    if set(have) >= {'FV', 'RW'}:
        c = compare(main, C['main'], 'FV', 'RW', K_EVAL)
        print(f"    FV - RW: {c['pp']:+5.1f} pp  perm p={c['p']:.4f}  rows W={c['w']} L={c['l']}")

    print('\n  pre-registered reading:')
    modes = {st.get('meta', {}).get('mode') for st in states.values()}
    if len(states) < 2:
        print('    none: only one worker has results; run both, then --summary-only')
        return 0
    empty = [w for w, st in states.items() if st.get('train', {}).get('FV', {}).get('status') == 'no pairs']
    if empty:
        print(f"    none: worker(s) {empty} had no FV training pairs, so their scored rows have no FV "
              f"samples. The screen cannot be read as designed.")
        return 0
    if 'FV' not in have or not _complete(C['guard'], guard, 'FV', K_GUARD):
        print('    none: PARTIAL run (FV not scored on every row yet). Re-run the workers to finish.')
        return 0
    if modes & {'smoke'}:
        print('    none: smoke run (plumbing only)')
        return 0
    for v in verdicts(C, main, guard, have):
        print('    ' + v)
    return 0


def verdicts(C: Dict, main: List[Dict], guard: List[Dict], have: Sequence[str]) -> List[str]:
    out = []
    v = C['valid']
    if len(v) < 2 * N_VALID:
        return [f"none: validity incomplete ({len(v)}/{2 * N_VALID} rows)"]
    dv = 100 * sum(CD.per_sample(s, r['gold']) - CD.per_sample(r['base'], r['gold']) for r, s in v) / len(v)
    if abs(dv) > VALID_MAX_PP:
        return [f"VALIDITY FAILED: re-sampled base - stored base = {dv:+.1f} pp (bar {VALID_MAX_PP:.0f}). "
                f"The sampler is suspect; no verdict is read."]
    out.append(f"VALIDITY ok: re-sampled base - stored base = {dv:+.1f} pp on {len(v)} rows")
    c = compare(main, C['main'], 'FV', None, K_EVAL)
    facts = (f"{c['pp']:+.1f} pp per sample on {c['n']} held-out-template rows, perm p={c['p']:.4f}, "
             f"template bootstrap [{c['ci'][0]:+.1f}, {c['ci'][1]:+.1f}], rows W={c['w']} L={c['l']}")
    if c['pp'] >= PRIMARY_MIN_PP and c['p'] < PRIMARY_ALPHA:
        out.append(f"PRIMARY   SUPPORTED: FV makes each chain right more often ({facts})")
        supported = True
    elif c['pp'] <= 0:
        out.append(f"PRIMARY   REFUTED: FV <= BASE ({facts})")
        supported = False
    else:
        out.append(f"PRIMARY   INCONCLUSIVE: {facts}")
        supported = False
    if 'RW' in have:
        a = compare(main, C['main'], 'FV', 'RW', K_EVAL)['pp']
        out.append(f"ATTRIBUTE FV - RW = {a:+.1f} pp -> " + (
            "the familiar-version negatives are what works" if a >= ATTR_MIN_PP else
            "any wrong sample does about as well (the familiar version is not shown to matter)"))
    else:
        out.append("ATTRIBUTE not read yet: RW has not been scored on every row")
    h = compare(guard, C['guard'], 'FV', None, K_GUARD)['pp']
    harm = h < HARM_MIN_PP
    out.append(f"HARM      guard FV - BASE = {h:+.1f} pp -> " + ("HARM" if harm else "no harm"))
    out.append("GO        " + ("run the fresh-row confirmation (train on all 200 rows, v25 seed-48 rows)"
                               if supported and not harm else "do not run the confirmation as is"))
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', type=int, default=0, choices=(0, 1))
    ap.add_argument('--device', type=int, default=-1, help='GPU index (default: the worker number '
                    'if that many GPUs exist, else 0)')
    ap.add_argument('--arms', default=','.join(CD.ARMS))
    ap.add_argument('--preset', default='qwen_math7b_mixed')
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--out', default='')
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--max-hours', type=float, default=0.0)
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()
    args.arms = [a.strip().upper() for a in args.arms.split(',') if a.strip()]
    bad = [a for a in args.arms if a not in CD.ARMS]
    if bad:
        print(f"unknown arm(s) {bad}; choose from {list(CD.ARMS)}")
        return 2
    args.mode = 'stub' if args.stub else ('smoke' if args.smoke else 'real')
    if args.summary_only:
        states = {}
        for w in (0, 1):
            p = OUT[args.mode].format(w=w)
            if os.path.exists(p):
                states[w] = V24._read(p)
        if not states:
            print(f"no results found ({OUT[args.mode].format(w='0/1')})")
            return 1
        return summarise(states, stub=args.stub)
    if args.device < 0:
        try:
            import torch
            n = torch.cuda.device_count()
        except Exception:
            n = 0
        args.device = args.worker if n > args.worker else 0
    return run_worker(args)


if __name__ == '__main__':
    sys.exit(main())
