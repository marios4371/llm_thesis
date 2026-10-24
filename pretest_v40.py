"""
[v40.0] Cross-model Symbolic Surgery (cross_model_surgery.py): does v38's
ADAPT-RS help solvers of OTHER families and sizes?

WHY
---
v38 measured ADAPT-RS (v22's verifier-gated sampling + Symbolic Surgery) on one
solver, Qwen2.5-Math-7B-Instruct: 131 vs ADAPT-R 122 on 200 clean GSM-Hard rows
(W9 L0, p = 0.004); replay_v40.py reproduces that exactly with v40's reading.
v40 draws fresh samples from solvers of OTHER families on those 200 rows and
measures the same system with the same verifier (Qwen2.5-Math-PRM-7B, scored on
Kaggle with --score-prm, because the laptop cannot hold it).

ROWS (pretest_data/v40_cross_model.json, built once from the v38/v36 manifests)
----------------------------------------------------------------------------
  main   the 200 clean GSM-Hard rows (max number >= 1e4) of v38's manifest, in
         its order: v38's Qwen-Math run is the reference row of every table
  pilot  10 clean GSM-Hard rows of v36's manifest (never in the reading): the
         speed / format / audit-coverage check that freezes the model list

PER ROW AND MODEL
-----------------
v38's sampler: 5 samples of v22's CoT prompt at T 0.8, top-p 0.95, top-k 50,
1536 new tokens, a fixed seed per (model, row, sample). Each sample is read by
cross_model_surgery (clean -> audit -> repair); every sample the FROZEN policy
sends to RESUME gets one partial call from the corrected prefix (same
sampling, the rest of the token budget). Backends: Ollama on the laptop
(--backend ollama), transformers 4-bit on Kaggle (--backend hf), a stub.

PRE-REGISTERED (V40_PLAN.md section 4; frozen before any main-row sample)
------------------------------------------------------------------------
Canonical grader |pred - gold| < 1e-3. A model enters the reading with >= 100
complete main rows; its rows are the complete ones (manifest order). The NEW
models are pooled; the Qwen-Math reference (in-sample for the method) is not.
  PRIMARY      ADAPT-RS vs ADAPT-R (v38's comparison), pooled W/L over the new
               models: net >= +5 with W >= 2L and two-sided sign p < 0.05 ->
               SUPPORTED; net <= 0 -> REFUTED; otherwise INCONCLUSIVE
  SECONDARY A  every included new model has net >= 0 -> CONSISTENT
  SECONDARY B  ADAPT-RS vs SC5 (self-consistency at 5 calls) pooled: net >= +5
               with W >= 2L -> MET
A model is read only once its samples, repairs and resumes carry PRM scores.
Reported, not decisive: ADAPT-RS vs ADAPT, the verifier-free SC5-R / SC5-RS,
per-model accuracies and W/L, audit coverage, refuted right samples, repairs,
resumes and calls per row, unparsed and truncated samples.

MODES
-----
    python pretest_v40.py --build-manifest                               # once; commit it
    python pretest_v40.py --backend ollama --pilot --models a,b,c        # Friday: 10 rows each
    python pretest_v40.py --backend ollama --models a,b,c --max-hours 44 # the weekend run; re-run resumes
    python pretest_v40.py --backend hf --models X,Y --parallel --max-hours 8.5   # Kaggle T4 x2
    python pretest_v40.py --score-prm --inputs F1 F2 ... [--parallel]    # Kaggle: PRM scores (needed)
    python pretest_v40.py --summary-only [--pilot]                       # every pretest_v40_*.json here
    python pretest_v40.py --stub --models s1,s2 --limit 4                # offline plumbing
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import zlib
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import cross_model_surgery as X
import pretest_v21 as V21
import trace_siv as T
import trace_surgery as S
from replay_v35 import sign_p

MANIFEST = 'pretest_data/v40_cross_model.json'
PREFIX = 'pretest_v40'
SEED = 40
N_PILOT = 10
K = 5
TEMPERATURE = 0.8
MAX_TOKENS = 1536            # v38's budget for these rows
NUM_CTX = 2560               # Ollama context: prompt (< 600 tokens) + 1536 new, with room
MIN_RESUME_TOKENS = 128
MAX_ERRORS = 3               # more failed resumes than this in one model's run: stop (it is systematic)
MIN_ROWS = 100               # a model enters the reading with at least this many complete main rows
V38_SOURCE = 'pretest_data/v38_surgery.json'
V36_SOURCE = 'pretest_data/v36_arith.json'

# pre-registered bars (V40_PLAN.md section 4)
PRI_MIN_NET, PRI_RATIO, PRI_ALPHA = 5, 2, 0.05
SECB_MIN_NET, SECB_RATIO = 5, 2


def _read(path: str) -> Dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def _save(path: str, recs: Dict, meta: Dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(dict(meta, rows=recs), fh, ensure_ascii=False)
    os.replace(tmp, path)


def sha(path: str) -> str:
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read().replace(b'\r\n', b'\n')).hexdigest()[:12]


def frozen() -> str:
    parts = [f"cross_model_surgery.py {sha(X.__file__)}", f"trace_siv.py {sha(T.__file__)}",
             f"trace_surgery.py {sha(S.__file__)}"]
    if os.path.exists(MANIFEST):
        parts.append(f"{os.path.basename(MANIFEST)} {sha(MANIFEST)}")
    return '; '.join(parts)


def tag(model: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '-', model.split('/')[-1]).strip('-')


def out_path(model: str, pilot: bool, directory: str = '.') -> str:
    return os.path.join(directory, f"{PREFIX}_{'pilot_' if pilot else ''}{tag(model)}.json")


def seed_for(model: str, pid: str, k: int) -> int:
    return zlib.crc32(f"{model}|{pid}|{k}".encode()) & 0x7fffffff


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def build_manifest(force: bool = False) -> int:
    if os.path.exists(MANIFEST) and not force:
        print(f"{MANIFEST} exists; refusing to overwrite pre-registered rows (--force to redo)")
        return 1
    main = [r for r in _read(V38_SOURCE)['rows'] if r['level'] == 'A']
    pool = [r for r in _read(V36_SOURCE)['rows'] if r.get('level') == 'A']
    pool.sort(key=lambda r: r['problem_id'])
    random.Random(SEED).shuffle(pool)
    rows = [{'pid': r['problem_id'], 'text': r['text'], 'gold': float(r['gold']), 'set': 'main'} for r in main]
    rows += [{'pid': r['problem_id'], 'text': r['text'], 'gold': float(r['gold']), 'set': 'pilot'}
             for r in pool[:N_PILOT]]
    with open(MANIFEST, 'w', encoding='utf-8') as fh:
        json.dump({'dataset': 'v40 cross-model surgery: v38 main rows (200 clean GSM-Hard >= 1e4) + 10 v36 pilot rows',
                   'seed': SEED, 'n': len(rows), 'sources': [V38_SOURCE, V36_SOURCE], 'rows': rows},
                  fh, ensure_ascii=False, indent=1)
    print(f"wrote {len(rows)} rows ({len(main)} main, {min(N_PILOT, len(pool))} pilot) -> {MANIFEST}")
    return 0


def load_rows(pilot: bool = False) -> List[Dict]:
    want = 'pilot' if pilot else 'main'
    return [r for r in _read(MANIFEST)['rows'] if r['set'] == want]


def complete(rec: Optional[Dict]) -> bool:
    rec = rec or {}
    return len(rec.get('C') or []) >= K and bool(rec.get('surgery_done'))


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

class OllamaBackend:
    def __init__(self, model: str, host: str, concurrency: int):
        import ollama_client as OC
        self.OC = OC
        self.model = model
        self.client = OC.OllamaClient(model, host=host, num_ctx=NUM_CTX, concurrency=concurrency)
        self.client.show()                      # fails now (model not pulled, server down), not after hours
        ok, tail = OC.smoke_prefill(self.client, V21.COT_PROMPT)
        self.can_resume = ok
        print(f"  {model}: {self.client.details()}; thinking={self.client.thinking()}; "
              f"prefill {'OK' if ok else 'NOT SUPPORTED (resumes off for this model)'}: {tail!r}", flush=True)

    def meta(self) -> Dict:
        return {'backend': 'ollama', 'ollama': self.client.version(), 'details': self.client.details(),
                'thinking': self.client.thinking(), 'can_resume': self.can_resume, 'num_ctx': NUM_CTX}

    def samples(self, content: str, k: int, seeds: Sequence[int]) -> List[Tuple[str, Dict]]:
        return self.client.sample(content, k, TEMPERATURE, MAX_TOKENS, seeds)

    def resume(self, content: str, prefix: str, sample: Dict, seed: int) -> str:
        used = int((sample.get('tokens') or MAX_TOKENS) * len(prefix) / max(1, len(sample.get('raw') or prefix)))
        return self.client.continue_from(content, prefix, TEMPERATURE, max(MIN_RESUME_TOKENS, MAX_TOKENS - used),
                                         seed)[0]

    def close(self) -> None:
        self.client.unload()


FALLBACK_TEMPLATE = ("{% for m in messages %}{% if m['role'] == 'user' %}User: {{ m['content'] }}\n\n"
                     "{% else %}{{ m['content'] }}\n\n{% endif %}{% endfor %}"
                     "{% if add_generation_prompt %}Assistant:{% endif %}")


class HFBackend:
    """Kaggle: transformers 4-bit through Mas_solver's local_hf client, pinned to
    the (single) visible GPU; v38's sampler (V21.sample) and resume (V22.continue_from)."""

    def __init__(self, model: str):
        import torch
        from Mas_solver import UnifiedLLMClient, LOCAL_HF_SAMPLING
        LOCAL_HF_SAMPLING.update(enabled=True)
        self.torch, self.model, self.can_resume = torch, model, True
        self.client = UnifiedLLMClient(provider='local_hf', use_cache=False, model_override=model, load_4bit=True,
                                       device_index=0 if torch.cuda.is_available() else None)
        self.client._ensure_local_model()
        tok = self.client._local_tokenizer
        if not getattr(tok, 'chat_template', None):
            tok.chat_template = FALLBACK_TEMPLATE
            print(f"  {model}: no chat template, using a plain User/Assistant one", flush=True)

    def meta(self) -> Dict:
        import transformers
        return {'backend': 'hf', 'transformers': transformers.__version__, 'load_4bit': True, 'can_resume': True}

    def samples(self, content: str, k: int, seeds: Sequence[int]) -> List[Tuple[str, Dict]]:
        self.torch.manual_seed(seeds[0])
        return [(str(x), {}) for x in V21.sample(self.client, content, k, TEMPERATURE, MAX_TOKENS)]

    def resume(self, content: str, prefix: str, sample: Dict, seed: int) -> str:
        import pretest_v22 as V22
        try:
            used = len(self.client._local_tokenizer(prefix, add_special_tokens=False)['input_ids'])
        except Exception:
            used = 0
        self.torch.manual_seed(seed)
        return V22.continue_from(self.client, content, prefix, 1, TEMPERATURE,
                                 max(MIN_RESUME_TOKENS, MAX_TOKENS - used))[0]

    def close(self) -> None:
        import gc
        self.client._local_model = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class StubBackend:
    """Offline: a sum written out step by step; the model 'slips' a carry in a
    third of the samples (some slips shared by several samples, so the policy
    resumes), and a resume finishes the corrected prefix."""

    def __init__(self, model: str, rows: Sequence[Dict]):
        self.model, self.can_resume = model, True
        self.gold = {V21.COT_PROMPT.format(problem=r['text']): r['gold'] for r in rows}

    def meta(self) -> Dict:
        return {'backend': 'stub', 'can_resume': True}

    def _text(self, g: float, slip: int, style: int) -> str:
        a = int(g) - 7
        c = a + 7 + slip
        if style == 0:
            return f"Add the parts:\n\\[ {a} + 7 = {c} \\]\n\nKeep it:\n\\[ {c} \\times 1 = {c} \\]\n\nAnswer: {c}"
        if style == 1:
            return f"1. **Add the parts:** {a} + 7 = **{c}**\n2. **Keep it:** {c} x 1 = {c}\n\n**Answer:** {c}"
        return (f"Add the parts: {a} + 7 = {c}.\nThen nothing more is added, so the total stays at {c - slip}.\n"
                f"Answer: {c - slip}")

    def samples(self, content: str, k: int, seeds: Sequence[int]) -> List[Tuple[str, Dict]]:
        g = self.gold.get(content, 0.0)
        out = []
        for s in seeds[:k]:
            rng = random.Random(s)
            slip = 10 if rng.random() < 0.34 else 0
            out.append((self._text(g, slip, s % 3), {'tokens': 40, 'done_reason': 'stop'}))
        return out

    def resume(self, content: str, prefix: str, sample: Dict, seed: int) -> str:
        v = re.findall(r'\d+(?:\.\d+)?', prefix[-40:])
        v = v[-1] if v else '0'
        return prefix + f"\n\nKeep it: {v} * 1 = {v}\n\nAnswer: {v}"

    def close(self) -> None:
        pass


def build_backend(args, model: str, rows: Sequence[Dict]):
    if args.stub:
        return StubBackend(model, rows)
    if args.backend == 'ollama':
        return OllamaBackend(model, args.host, args.concurrency)
    if args.backend == 'hf':
        return HFBackend(model)
    raise ValueError(f"unknown backend {args.backend!r}")


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

_ERRORS: List[str] = []


def do_row(backend, model: str, row: Dict) -> Dict:
    content = V21.COT_PROMPT.format(problem=row['text'])
    seeds = [seed_for(model, row['pid'], k) for k in range(K)]
    ts = time.time()
    outs = backend.samples(content, K, seeds)
    c = [{'raw': raw, 'tokens': info.get('tokens'), 'done': info.get('done_reason')} for raw, info in outs]
    resumes = 0
    if backend.can_resume:
        prep = X.prepare(c)
        for i in X.needs_resume(prep):
            prefix = prep[i]['surgery']['prefix']
            try:
                c[i]['resume'] = {'prefix': prefix,
                                  'raw': backend.resume(content, prefix, c[i], seed_for(model, row['pid'], 100 + i))}
                resumes += 1
            except Exception as exc:                         # recorded; more than MAX_ERRORS stop the run
                msg = f"{model} {row['pid']} resume: {type(exc).__name__}: {str(exc)[:200]}"
                c[i]['resume_error'] = msg
                _ERRORS.append(msg)
                print(f"  RESUME ERROR ({len(_ERRORS)}/{MAX_ERRORS} tolerated): {msg}", flush=True)
                if len(_ERRORS) > MAX_ERRORS:
                    raise RuntimeError(f"{len(_ERRORS)} resume errors: systematic, stopping (the file is saved; "
                                       f"re-run the identical command to resume)") from exc
    return {'C': c, 'surgery_done': True, 'resumes': resumes, 'seconds': round(time.time() - ts, 1)}


def run_model(args, model: str, rows: List[Dict], deadline: float) -> str:
    """One model through every row; returns 'done', 'deadline' or 'error'."""
    out = out_path(model, args.pilot, args.dir)
    recs = _read(out).get('rows', {}) if (args.resume and os.path.exists(out)) else {}
    todo = [r for r in rows if not complete(recs.get(r['pid']))]
    done = len(rows) - len(todo)
    if args.limit:
        todo = todo[:args.limit]
    print(f"\n=== {model}: {done}/{len(rows)} rows done, {len(todo)} to do now -> {out}", flush=True)
    if not todo:
        return 'done'
    try:
        backend = build_backend(args, model, rows)
    except Exception as exc:                                 # not pulled, server down, no GPU: the next model
        print(f"  {model}: cannot start ({type(exc).__name__}: {str(exc)[:300]})", flush=True)
        return 'error'
    meta = dict(backend.meta(), model=model, k=K, temperature=TEMPERATURE, top_p=0.95, top_k=50,
                max_tokens=MAX_TOKENS, prompt=V21.COT_PROMPT, frozen=frozen(), set='pilot' if args.pilot else 'main')
    t0 = time.time()
    status = 'done'
    try:
        for i, row in enumerate(todo, 1):
            if deadline and time.time() >= deadline:
                status = 'deadline'
                break
            rec = do_row(backend, model, row)
            recs[row['pid']] = rec
            _save(out, recs, meta)
            x = X.prepare(rec['C'])
            right = sum(X.canonical(s['answer'], row['gold']) for s in x)
            print(f"[{tag(model)} {i:3d}/{len(todo)}] {row['pid']:16s} right {right}/{K}  refuted "
                  f"{sum(s['refuted'] for s in x)}  repairs {sum(bool(s.get('surgery')) for s in x)}  resumes "
                  f"{rec['resumes']}  {rec['seconds']:5.0f}s  {(time.time() - t0) / i:5.0f}s/row", flush=True)
    except Exception as exc:
        print(f"\n  {model}: STOPPED by {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        status = 'error'
    finally:
        _save(out, recs, meta)
        backend.close()
    return status


def run(args) -> int:
    if not os.path.exists(MANIFEST):
        print(f"missing {MANIFEST}: run `python pretest_v40.py --build-manifest` first")
        return 2
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    if not models:
        print("--models is required (comma-separated)")
        return 2
    rows = load_rows(args.pilot)
    if args.reverse:
        rows = list(reversed(rows))
    if not args.stub and args.backend == 'hf':
        try:
            import pretest_v32 as T32
            T32.scrub_git_credentials(os.path.dirname(os.path.abspath(__file__)))
        except Exception:
            pass
    print(f"v40 {'PILOT' if args.pilot else 'main'} rows: {len(rows)}; models: {models}; frozen: {frozen()}",
          flush=True)
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else 0.0
    status = {}
    for m in models:
        status[m] = run_model(args, m, rows, deadline)
        if status[m] == 'deadline':
            print("\n  --max-hours reached; re-run the identical command to resume.", flush=True)
            break
    print(f"\n  status: {status}", flush=True)
    if args.pilot:
        pilot_report(args.dir)
    else:
        summarise(args.dir)
    return 1 if 'error' in status.values() else 0


def run_parallel(args) -> int:
    """Kaggle: one worker per visible GPU, worker i on models[i::n] in turn."""
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    if args.stub or n < 2 or len(models) < 2:
        return run(args)
    procs = []
    for i in range(min(n, len(models))):
        cmd = [sys.executable, '-u', os.path.abspath(__file__), '--backend', args.backend, '--models',
               ','.join(models[i::n]), '--max-hours', str(args.max_hours), '--dir', args.dir]
        for flag, on in (('--pilot', args.pilot), ('--no-resume', not args.resume)):
            if on:
                cmd.append(flag)
        if args.limit:
            cmd += ['--limit', str(args.limit)]
        procs.append(subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))))
        print(f"  worker {i} on GPU {i}: {models[i::n]}", flush=True)
    codes = [p.wait() for p in procs]
    print(f"  workers finished with exit codes {codes}", flush=True)
    summarise(args.dir)
    return max(codes)


# ---------------------------------------------------------------------------
# PRM scores (Kaggle; secondary)
# ---------------------------------------------------------------------------

def prm_path(result_file: str) -> str:
    d, b = os.path.split(result_file)
    return os.path.join(d, b.replace(f'{PREFIX}_', f'{PREFIX}_prm_', 1))


def score_prm(args) -> int:
    import score_prm_v22 as P
    files = [f for pat in args.inputs for f in sorted(glob.glob(pat))
             if '_prm_' not in os.path.basename(f) and '_pilot_' not in os.path.basename(f)]
    if args.parallel and not args.stub:
        try:
            import torch
            n = torch.cuda.device_count()
        except Exception:
            n = 0
        if n >= 2 and len(files) >= 2:
            procs = [subprocess.Popen([sys.executable, '-u', os.path.abspath(__file__), '--score-prm', '--inputs',
                                       *files[i::n]], env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i)))
                     for i in range(min(n, len(files)))]
            return max(p.wait() for p in procs)
    rows = {r['pid']: r for r in _read(MANIFEST)['rows']}
    scorer = None
    for f in files:
        out = prm_path(f)
        have = _read(out).get('scores', {}) if os.path.exists(out) else {}
        todo = []
        for pid, rec in _read(f).get('rows', {}).items():
            if not complete(rec):
                continue
            for x in X.prepare(rec['C']):
                texts = [x['raw']]
                sg = x.get('surgery') or {}
                texts += [sg[k]['raw'] for k in ('prop', 'resume') if sg.get(k)]
                todo += [(rows[pid]['text'], t) for t in texts if t and X.text_key(t) not in have]
        print(f"  {f}: {len(todo)} texts to score", flush=True)
        if todo and scorer is None:
            scorer = _StubPRM() if args.stub else P.PRMScorer(P.PRM, four_bit=True, device_map={'': 0})
        for i, (problem, text) in enumerate(todo, 1):
            have[X.text_key(text)] = scorer.score(problem, P.split_steps(text))
            if i % 50 == 0 or i == len(todo):
                with open(out, 'w', encoding='utf-8') as fh:
                    json.dump({'source': os.path.basename(f), 'prm': P.PRM, 'scores': have}, fh)
                print(f"    {i}/{len(todo)} scored", flush=True)
        if not todo and not os.path.exists(out):
            with open(out, 'w', encoding='utf-8') as fh:
                json.dump({'source': os.path.basename(f), 'prm': P.PRM, 'scores': have}, fh)
    return 0


class _StubPRM:
    def score(self, problem: str, steps: List[str]) -> List[float]:
        h = zlib.crc32('\n'.join(steps).encode()) % 100 / 100
        return [0.9] * max(0, len(steps) - 1) + [0.5 + h / 2]


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _wl(a: Sequence[bool], b: Sequence[bool]) -> Tuple[int, int]:
    return sum(x and not y for x, y in zip(a, b)), sum(y and not x for x, y in zip(a, b))


def model_reading(rows: Dict[str, Dict], recs: Dict, prm: Optional[Dict] = None) -> Dict:
    """Per-model numbers over its complete main rows."""
    done = [(rows[pid], X.prepare(rec['C'], prm)) for pid, rec in recs.items() if pid in rows and complete(rec)]
    order = {pid: i for i, pid in enumerate(rows)}
    done.sort(key=lambda t: order[t[0]['pid']])
    out = {'rows': len(done), 'right': {}, 'calls': {}}
    arms = dict(X.ARMS)
    if prm and all(x.get('prm') for _, c in done for x in c):
        import replay_v35 as R
        arms.update({'B5': lambda c: (c[T.best(c, range(5))]['answer'] if T.best(c, range(5)) is not None
                                      else None, 5),
                     'ADAPT': lambda c: R._adapt(c, False), 'ADAPT-R': T.adapt_r, 'ADAPT-RS': S.adapt_rs})
    for a, f in arms.items():
        res = [f(c) for _, c in done]
        out['right'][a] = [X.canonical(ans, r['gold']) for (r, _), (ans, _) in zip(done, res)]
        out['calls'][a] = sum(n for _, n in res) / max(1, len(done))
    s = [x for _, c in done for x in c]
    raws = [x for pid, rec in recs.items() if pid in rows and complete(rec) for x in rec['C']]
    out['stats'] = {
        'samples': len(s), 'right samples': sum(X.canonical(x['answer'], r['gold']) for r, c in done for x in c),
        'unparsed': sum(x['answer'] is None for x in s), 'truncated': sum(x.get('done') == 'length' for x in raws),
        'checked': sum(any(x['audit']) for x in s), 'refuted': sum(x['refuted'] for x in s),
        'refuted RIGHT': sum(x['refuted'] and X.canonical(x['answer'], r['gold']) for r, c in done for x in c),
        'repairs': sum(bool(x.get('surgery')) for x in s),
        'resumes stored': sum(bool((x.get('surgery') or {}).get('resume')) for x in s),
        'tokens/sample': sum(x.get('tokens') or 0 for x in raws) / max(1, len(raws)),
        'seconds/row': sum(rec.get('seconds', 0) for pid, rec in recs.items() if pid in rows and complete(rec))
                       / max(1, len(done))}
    return out


def results(directory: str, pilot: bool = False) -> Dict[str, Dict]:
    pat = os.path.join(directory, f"{PREFIX}_pilot_*.json" if pilot else f"{PREFIX}_*.json")
    out = {}
    for f in sorted(glob.glob(pat)):
        b = os.path.basename(f)
        if '_prm_' in b or (not pilot and b.startswith(f'{PREFIX}_pilot_')) or b.endswith('.tmp'):
            continue
        d = _read(f)
        out[d.get('model') or b] = {'file': f, 'recs': d.get('rows', {}), 'meta': {k: v for k, v in d.items()
                                                                                   if k != 'rows'}}
    return out


def verdicts(readings: Dict[str, Dict]) -> List[str]:
    """v38's system (ADAPT-RS: ADAPT with Symbolic Surgery) against v38's comparator (ADAPT-R),
    pooled over the new models; every included model needs PRM scores (--score-prm)."""
    new = {m: r for m, r in readings.items() if r['rows'] >= MIN_ROWS}
    if not new:
        return [f"none: no new model has >= {MIN_ROWS} complete main rows yet"]
    unscored = [m for m, r in new.items() if 'ADAPT-RS' not in r['right']]
    if unscored:
        return [f"none: no PRM scores yet for {', '.join(unscored)} (run MAS_SHT_Kaggle_v40_prm.ipynb)"]
    w = l = wb = lb = 0
    neg = []
    for m, r in new.items():
        a, b = _wl(r['right']['ADAPT-RS'], r['right']['ADAPT-R'])
        w, l = w + a, l + b
        if a - b < 0:
            neg.append(f"{m} ({a - b:+d})")
        a, b = _wl(r['right']['ADAPT-RS'], r['right']['SC5'])
        wb, lb = wb + a, lb + b
    p = sign_p(w, l)
    facts = f"ADAPT-RS vs ADAPT-R pooled over {len(new)} models: W{w} L{l} net {w - l:+d} (sign p = {p:.4f})"
    if w - l >= PRI_MIN_NET and w >= PRI_RATIO * l and p < PRI_ALPHA:
        out = [f"PRIMARY      SUPPORTED: {facts}"]
    elif w - l <= 0:
        out = [f"PRIMARY      REFUTED: {facts}"]
    else:
        out = [f"PRIMARY      INCONCLUSIVE: {facts}"]
    out.append(f"SECONDARY A  {'CONSISTENT: net >= 0 in every model' if not neg else 'NOT CONSISTENT: ' + ', '.join(neg)}")
    met = wb - lb >= SECB_MIN_NET and wb >= SECB_RATIO * lb
    out.append(f"SECONDARY B  {'MET' if met else 'NOT MET'}: ADAPT-RS vs SC5 (self-consistency) pooled "
               f"W{wb} L{lb} net {wb - lb:+d} (sign p = {sign_p(wb, lb):.4f})")
    return out


def _line(name: str, r: Dict) -> str:
    acc = '  '.join(f"{a} {sum(r['right'][a]):3d}" for a in r['right'])
    x = 'ADAPT-RS' if 'ADAPT-RS' in r['right'] else 'SC5-RS'
    y = 'ADAPT-R' if x == 'ADAPT-RS' else 'SC5'
    a, b = _wl(r['right'][x], r['right'][y])
    return (f"  {name[:30]:30s} {r['rows']:4d} rows  {acc}   {x} vs {y} W{a} L{b}   "
            f"calls/row {x} {r['calls'][x]:.2f}")


def summarise(directory: str = '.') -> int:
    rows = {r['pid']: r for r in load_rows(False)}
    res = results(directory)
    print('\n' + '=' * 100)
    print(f"  v40 cross-model surgery: main rows {len(rows)}; canonical grader abs < 1e-3; frozen: {frozen()}")
    readings = {}
    for m, d in res.items():
        p = prm_path(d['file'])
        prm = _read(p).get('scores') if os.path.exists(p) else None
        readings[m] = model_reading(rows, d['recs'], prm)
    try:
        import replay_v40 as RP
        q = RP.qwen_rows().get('A') or []
        ref = (model_reading(rows, {r['pid']: {'C': r['C'], 'surgery_done': True} for r in q}, RP.qwen_prm())
               if q else None)
    except Exception:
        ref = None
    if ref:
        print(_line('Qwen2.5-Math-7B (v38, reference)', ref))
    for m, r in readings.items():
        print(_line(m, r) + ('' if r['rows'] >= MIN_ROWS else f"   (< {MIN_ROWS} rows: not in the reading)"))
    print("\n  per model: samples / right / unparsed / truncated / checked / refuted / refuted RIGHT / repairs / "
          "resumes / tokens per sample / s per row")
    for m, r in ([('Qwen2.5-Math-7B (v38)', ref)] if ref else []) + list(readings.items()):
        st = r['stats']
        print(f"  {m[:30]:30s} " + ' / '.join(f"{v:.0f}" if isinstance(v, float) else str(v) for v in st.values()))
    complete_all = all(r['rows'] == len(rows) for r in readings.values())
    print('\n  pre-registered reading' + ('' if complete_all else ' (some models incomplete: each counts with its '
                                                               f'complete rows when >= {MIN_ROWS})') + ':')
    for line in verdicts({m: r for m, r in readings.items()}):
        print('    ' + line)
    return 0


def pilot_report(directory: str = '.') -> int:
    rows = {r['pid']: r for r in load_rows(True)}
    res = results(directory, pilot=True)
    print('\n' + '=' * 100)
    print("  v40 PILOT (speed / format / audit coverage; NOT read for accuracy)")
    print(f"  {'model':26s} {'rows':>4s} {'s/row':>6s} {'tok/s':>6s} {'tok/samp':>8s} {'trunc':>6s} {'unparsed':>8s} "
          f"{'checked':>8s} {'refuted':>8s} {'repairs':>8s} {'resume':>7s}  hours for 200 rows")
    for m, d in res.items():
        recs = {k: v for k, v in d['recs'].items() if complete(v)}
        if not recs:
            continue
        s = [x for rec in recs.values() for x in X.prepare(rec['C'])]
        raws = [x for rec in recs.values() for x in rec['C']]
        secs = sum(rec.get('seconds', 0) for rec in recs.values()) / len(recs)
        tok = sum(x.get('tokens') or 0 for x in raws)
        n = len(s)
        print(f"  {m[:26]:26s} {len(recs):4d} {secs:6.0f} {tok / max(1e-9, secs * len(recs)):6.1f} "
              f"{tok / max(1, len(raws)):8.0f} {sum(x.get('done') == 'length' for x in raws) / n:6.0%} "
              f"{sum(x['answer'] is None for x in s) / n:8.0%} {sum(any(x['audit']) for x in s) / n:8.0%} "
              f"{sum(x['refuted'] for x in s) / n:8.0%} {sum(bool(x.get('surgery')) for x in s) / n:8.0%} "
              f"{str(d['meta'].get('can_resume')):>7s}  {secs * 200 / 3600:5.1f}")
    print("  (tok/s counts every sample of a row over the row's wall time, i.e. the throughput the run will see)")
    return 0


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-manifest', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--backend', choices=('ollama', 'hf'), default='ollama')
    ap.add_argument('--models', default='')
    ap.add_argument('--host', default='http://localhost:11434')
    ap.add_argument('--concurrency', type=int, default=5, help='Ollama requests in flight per row')
    ap.add_argument('--pilot', action='store_true', help='the 10 pilot rows instead of the 200 main rows')
    ap.add_argument('--dir', default='.', help='where the result files live')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--reverse', action='store_true')
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--max-hours', type=float, default=0.0)
    ap.add_argument('--parallel', action='store_true', help='Kaggle: one worker per GPU')
    ap.add_argument('--score-prm', action='store_true')
    ap.add_argument('--inputs', nargs='*', default=[])
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if args.build_manifest:
        return build_manifest(args.force)
    if args.score_prm:
        return score_prm(args)
    if args.summary_only:
        return pilot_report(args.dir) if args.pilot else summarise(args.dir)
    if args.parallel:
        return run_parallel(args)
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
