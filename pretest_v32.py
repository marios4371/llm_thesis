"""
[v32.0] Dev screen: Reading Jury (RJ). Two readers from OTHER model families
write the situation notes; Qwen2.5-Math-7B-Instruct alone solves; v22's PRM
selects; inside the PRM's contested tie the answer backed by more FAMILIES
wins. Does that beat v22 with the same-family Reader (B10) at fewer solver
samples?

The method, the evidence and the literature are in reading_jury.py and
V32_PLAN.md. The rule is `jury_choice` in reading_jury.py.

ROWS
----
  dev        results_September/preset_V25_dev.json: 100 P2 main + 20 GSM-Plus
             guard rows (seed 45) with v22's stored C5 + Q5 and step rewards
  held-out   pretest_data/v21_traces.json + v22_dev_prm.json: 100 P2 main
             rows (seed 44), C5 with step rewards, read by no version since v24
New per row: one greedy reading from each foreign Reader, K_READ = 2 Solver
samples under each (v24's ASQ prompt; v22's sampler), every new sample scored
by v22's PRM against the ORIGINAL text.

PRE-REGISTERED (V32_PLAN.md section 5; frozen before any foreign sample exists)
-------------------------------------------------------------------------------
  SCREEN (100 dev main rows)  RJ = C5 + F1x2 + F2x2 with the jury rule
                              vs B10 = C5 + Q5 with v22's rule (the stored system,
                              ONE more solver sample than RJ):
                              net >= +3 with W >= 2L -> GO (fresh confirmation);
                              net <= 0 -> STOP; otherwise WEAK
  GUARD  (20 GSM-Plus rows)   RJ - B10 >= -1
  HELD   (secondary)          100 held-out rows: RJ with the jury rule vs the
                              same pool with v22's rule >= 0 -> CONSISTENT
Reported, not decisive: RJ vs the same pool under v22's rule (does the RULE
add anything beyond the samples?); B9 (iso solver samples); RJ1/RJ2 (one
foreign family each); per-view accuracy; the SHARED-ERROR FLOOR per view
(on rows where >= 3/5 plain samples agree on a wrong answer, how often a
view's samples repeat it: the Qwen Reader's Q view vs each foreign view).

MODES
-----
    python pretest_v32.py --held --max-hours 8.0    # Kaggle T4 x2: dev first, then held-out
    python pretest_v32.py --summary-only            # re-read, no GPU
    python pretest_v32.py --stub                    # offline plumbing, no GPU
Re-running the same command RESUMES: a row keeps every reading, sample and
score it already has. Order: dev readings -> dev samples (+ scores) -> held
readings -> held samples, so the pre-registered SCREEN is complete first.
With two GPUs the Solver (4-bit, split over both) and the PRM (4-bit, cuda:1)
run together and every row is scored as it is drawn; with one GPU all rows
of a part are drawn first, then scored.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import sys
import time
from typing import Dict, List, Optional, Sequence

import pretest_v21 as V21
import pretest_v24 as V24
import pretest_v30 as T30
import reading_jury as RJ
import score_prm_v22 as P
import situation_reader as SR

OUT = 'pretest_v32.json'
OUT_STUB = 'pretest_v32_stub.json'
READ_BATCH = 8                    # rows per batched greedy Reader call
READER_ABORT_AFTER = 8            # a Reader with nothing usable on its first 8 rows stops the run
POOL_ORDER = ('B10', 'B9', 'RJ', 'RJ1', 'RJ2', 'B5')


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def load_rows(args) -> List[Dict]:
    rows = T30.load_rows(args.v25, args.v26)
    if args.held:
        rows += [r for r in T30.load_held_rows(args.held_traces, args.held_prm) if r['set'] == 'held']
    return rows


def attach(row: Dict, rec: Dict) -> Dict:
    """The row with this run's readings and foreign-view samples on it."""
    for v in RJ.READER_VIEWS:
        row[v] = rec.get(v) or []
    row['readings'] = rec.get('readings') or {}
    return row


def parts(rows: Sequence[Dict]) -> List[List[Dict]]:
    dev = [r for r in rows if not T30.is_held(r)]
    held = [r for r in rows if T30.is_held(r)]
    return [p for p in (dev, held) if p]


def needs_reading(rec: Dict, view: str) -> bool:
    return view not in (rec.get('readings') or {})


def needs_samples(rec: Dict) -> bool:
    return any(len(rec.get(v) or []) < RJ.K_READ for v in RJ.READER_VIEWS)


def needs_scores(rec: Dict) -> bool:
    return any(s.get('prm') is None for v in RJ.READER_VIEWS for s in rec.get(v) or [])


def dev_complete(rows: Sequence[Dict], recs: Dict) -> bool:
    return all(not needs_samples(recs.get(r['key'], {})) and not needs_scores(recs.get(r['key'], {}))
               for r in rows if not T30.is_held(r))


# ---------------------------------------------------------------------------
# the foreign Readers
# ---------------------------------------------------------------------------

class ReaderLM:
    """A Reader of another family, loaded WITHOUT remote code (both readers
    are native transformers architectures), 4-bit NF4 with bf16 compute (the
    project's proven T4 path), pinned to one GPU. Greedy, batched over rows
    with left padding."""

    def __init__(self, name: str, device_index: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch, self.name = torch, name
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(name)
        tok.padding_side = 'left'
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=bnb, device_map={'': device_index},
            attn_implementation='sdpa', low_cpu_mem_usage=True).eval()
        self.tok = tok
        print(f"  reader {name} loaded on cuda:{device_index} in {time.time() - t0:.0f}s", flush=True)

    def generate(self, batch: List[List[Dict[str, str]]], max_new_tokens: int) -> List[str]:
        torch, tok = self.torch, self.tok
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        bos = tok.bos_token
        add_special = not (bos and all(p.startswith(bos) for p in prompts))
        try:
            enc = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=add_special)
            dev = next(self.model.parameters()).device
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                          temperature=None, top_p=None, top_k=None,
                                          pad_token_id=tok.pad_token_id)
            n = enc['input_ids'].shape[1]
            return [tok.decode(o[n:], skip_special_tokens=True) for o in out]
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            torch.cuda.empty_cache()
            h = len(batch) // 2
            print(f"    OOM on a batch of {len(batch)}; splitting", flush=True)
            return self.generate(batch[:h], max_new_tokens) + self.generate(batch[h:], max_new_tokens)


class _StubReaderLM:
    """Offline Reader: one note per sentence that holds a number; the second
    family 'fails' on every 7th row so the fallback path runs."""

    def __init__(self, name: str, device_index: int = 0):
        self.name = name
        self.n = 0

    def generate(self, batch, max_new_tokens):
        out = []
        for msgs in batch:
            self.n += 1
            if self.name == RJ.READER_MODEL['F2'] and self.n % 7 == 0:
                out.append("I will solve the problem directly. The answer is 12.")
                continue
            lines = []
            for line in msgs[-1]['content'].splitlines()[1:]:
                parts_ = line.split(': ', 1)
                if len(parts_) != 2 or not parts_[0].startswith('S'):
                    continue
                nums = SR.numbers(parts_[1])
                lines.append(f"{parts_[0]}: Q: What number does this sentence give? A: {nums[0]:g}."
                             if nums else f"{parts_[0]}: -")
            lines.append("Asked: the quantity the question asks for.")
            out.append('\n'.join(lines))
        return out


def build_reader(args, view: str):
    name = RJ.READER_MODEL[view]
    if args.stub:
        return _StubReaderLM(name)
    return ReaderLM(name, device_index=_roomiest_gpu())


def read_rows(reader, recs: Dict, todo: List[Dict], view: str, out_path: str, meta: Dict,
              deadline: float) -> bool:
    """Batched greedy readings for `todo`. Returns False if the deadline
    stopped it. Raises if the Reader writes nothing usable on its first
    READER_ABORT_AFTER rows (a model that will not follow the format would
    otherwise burn the session measuring plain samples twice)."""
    seen = usable = 0
    shown = False
    for b in range(0, len(todo), READ_BATCH):
        if deadline and time.time() > deadline:
            return False
        batch = todo[b:b + READ_BATCH]
        t0 = time.time()
        raws = reader.generate([SR.reader_messages(r['text']) for r in batch], SR.READER_MAX_TOKENS)
        secs = round((time.time() - t0) / len(batch), 1)
        for r, raw in zip(batch, raws):
            rd = SR.parse_reading(raw, len(SR.sentences(r['text'])))
            rd.update(raw=str(raw)[:4000], model=RJ.READER_MODEL[view], seconds=secs)
            recs[r['key']].setdefault('readings', {})[view] = rd
            seen += 1
            usable += bool(rd['ok'])
            if rd['ok'] and not shown:
                shown = True
                print(f"  first usable {view} reading ({r['pid']}); check it looks like the demos:\n    "
                      + str(raw).strip().replace('\n', '\n    '), flush=True)
        _save(out_path, recs, meta)
        print(f"  {view} read {min(b + READ_BATCH, len(todo))}/{len(todo)} rows, usable so far "
              f"{usable}/{seen}, {secs:.0f}s/row", flush=True)
        if seen >= READER_ABORT_AFTER and not usable:
            for r in batch:
                recs[r['key']]['readings'].pop(view, None)
            _save(out_path, recs, meta)
            raise RuntimeError(f"{RJ.READER_MODEL[view]} gave nothing usable on its first {seen} rows; "
                               f"last output: {str(raws[-1])[:600]!r}")
    return True


# ---------------------------------------------------------------------------
# the Solver and the verifier
# ---------------------------------------------------------------------------

def split_batched(texts: Sequence[str], n_prompts: int, k: int) -> List[List[str]]:
    """generate(num_return_sequences=k) on n prompts returns prompt 0's k
    sequences first, then prompt 1's, ..."""
    if len(texts) != n_prompts * k:
        raise ValueError(f"{len(texts)} sequences for {n_prompts} prompts x {k}")
    return [list(texts[i * k:(i + 1) * k]) for i in range(n_prompts)]


def sample_views(client, contents: List[str], k: int, temperature: float,
                 max_tokens: int) -> List[List[str]]:
    """k samples for EACH prompt in one generate call: v22's sampler
    (V21._batched's knobs) with left padding across the prompts. Falls back
    to one V21.sample call per prompt if the batched call fails."""
    if getattr(client, 'provider', '') != 'local_hf':
        return [V21.sample(client, c, k, temperature, max_tokens) for c in contents]
    try:
        import torch
        from Mas_solver import LOCAL_HF_SAMPLING
        client._ensure_local_model()
        tok, mdl = client._local_tokenizer, client._local_model
        prompts = [tok.apply_chat_template([{'role': 'user', 'content': c}], tokenize=False,
                                           add_generation_prompt=True) for c in contents]
        side = tok.padding_side
        tok.padding_side = 'left'
        try:
            enc = tok(prompts, return_tensors='pt', padding=True)
        finally:
            tok.padding_side = side
        dev = next(mdl.parameters()).device
        enc = {kk: v.to(dev) for kk, v in enc.items()}
        with torch.no_grad():
            out = mdl.generate(**enc, max_new_tokens=max_tokens, do_sample=True,
                               temperature=float(temperature),
                               top_p=float(LOCAL_HF_SAMPLING.get('top_p', 0.95)),
                               top_k=int(LOCAL_HF_SAMPLING.get('top_k', 50)),
                               renormalize_logits=True, num_return_sequences=k,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        n_in = enc['input_ids'].shape[1]
        return split_batched([tok.decode(o[n_in:], skip_special_tokens=True) for o in out], len(contents), k)
    except Exception as exc:
        print(f"  multi-prompt batch failed ({type(exc).__name__}: {str(exc)[:120]}); "
              f"one call per prompt", flush=True)
        gc.collect()
        return [V21.sample(client, c, k, temperature, max_tokens) for c in contents]


class _StubPRM:
    """Offline verifier: the last step scores high when the answer is right."""

    def __init__(self, rows: Sequence[Dict]):
        self.gold = {r['text']: r['gold'] for r in rows}

    def score(self, problem: str, steps: List[str]) -> List[float]:
        a = V21.parse_answer('\n\n'.join(steps))
        rng = random.Random(sum(map(ord, ''.join(steps)[-200:])))
        base = 0.99995 if V21.correct(a, self.gold.get(problem)) else rng.choice([0.99995, 0.6])
        return [0.97] * (len(steps) - 1) + [base]


def _n_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def _roomiest_gpu() -> int:
    import torch
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return max(range(len(free)), key=lambda i: free[i]) if free else 0


def build_solver(args, rows):
    if args.stub:
        return V24._StubSolver(rows)
    return V21.build_client(args.preset, None)       # v22's Solver, unchanged (4-bit, sampling on)


def build_verifier(args, rows):
    if args.stub:
        return _StubPRM(rows)
    return P.PRMScorer(P.PRM, four_bit=True, device_map={'': 1 if _n_gpus() > 1 else 0})


def free(*objs) -> None:
    for o in objs:
        if o is None:
            continue
        for attr in ('model', '_local_model'):
            if hasattr(o, attr):
                setattr(o, attr, None)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def drop_cache_if_tight(model_name: str, min_free_gb: float = 30.0) -> None:
    """Kaggle's disk holds four 7-8B checkpoints only just: after a Reader is
    freed, delete its download when less than `min_free_gb` is left."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        free_gb = shutil.disk_usage(HF_HUB_CACHE).free / 2 ** 30
        if free_gb >= min_free_gb:
            return
        path = os.path.join(HF_HUB_CACHE, 'models--' + model_name.replace('/', '--'))
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            print(f"  disk: {free_gb:.0f} GB free, removed the {model_name} download", flush=True)
    except Exception:
        pass


def draw(solver, row: Dict, rec: Dict, args) -> None:
    """The K_READ samples under each foreign reading this row still lacks, in
    one batched generate call. A failed reading falls back to the plain
    prompt and its samples are labelled Qwen's (they are Qwen's reading)."""
    views = [v for v in RJ.READER_VIEWS if len(rec.get(v) or []) < RJ.K_READ]
    if not views:
        return
    readings = [rec['readings'][v] for v in views]
    contents = [SR.asq_prompt(row['text'], rd) for rd in readings]
    k = RJ.K_READ
    t0 = time.time()
    raws = sample_views(solver, contents, k, args.temperature, args.max_tokens)
    for v, rd, rs in zip(views, readings, raws):
        fam = RJ.READER_FAMILY[v] if rd.get('ok') else RJ.OWN_FAMILY
        rec[v] = [{'raw': str(x), 'answer': V21.parse_answer(x), 'family': fam} for x in rs]
    rec.setdefault('seconds', {})['solver'] = round(time.time() - t0, 1)


def score(scorer, row: Dict, rec: Dict) -> None:
    t0 = time.time()
    for v in RJ.READER_VIEWS:
        for s in rec.get(v) or []:
            if s.get('prm') is None:
                s['prm'] = scorer.score(row['text'], P.split_steps(s['raw']))
    rec.setdefault('seconds', {})['prm'] = round(time.time() - t0, 1)


# ---------------------------------------------------------------------------
# reading the results
# ---------------------------------------------------------------------------

def _pct(x: int, n: int) -> str:
    return f"{x}/{n} = {100 * x / n:.1f}%" if n else '-'


def block(rows: List[Dict], title: str, rule=None) -> None:
    if not rows:
        return
    print('\n' + '=' * 78)
    print(f"  {title}: {len(rows)} rows, {len({r.get('template') for r in rows})} templates")
    use = rule is not None or _rule_written()
    print(f"    {'pool':5s} {'samples':>7s} {'n':>4s} {'v22 rule':>9s} {'jury rule':>10s} {'oracle':>7s}")
    for pool in POOL_ORDER:
        ok = [r for r in rows if RJ.available(r, pool)]
        if not ok:
            continue
        v = sum(RJ.right(r, pool, False) for r in ok)
        j = sum(RJ.right(r, pool, True, rule) for r in ok) if use else None
        o = sum(RJ.oracle(r, pool) for r in ok)
        print(f"    {pool:5s} {sum(n for _, n in RJ.POOLS[pool]):7d} {len(ok):4d} {v:9d} "
              f"{'-' if j is None else j:>10} {o:7d}")
    if not use:
        print("    (jury rule not written yet: reading_jury.jury_choice)")
        return
    for x, y in ((('RJ', True), ('B10', False)), (('RJ', True), ('RJ', False)),
                 (('RJ', True), ('B9', False)), (('RJ1', True), ('B10', False)),
                 (('RJ2', True), ('B10', False))):
        c = RJ.paired(rows, x, y, rule)
        if c['n']:
            name = lambda t: f"{t[0]}{'' if t[1] else '/v22'}"
            print(f"    {name(x):7s} vs {name(y):7s}: W={c['w']:2d} L={c['l']:2d} net={c['net']:+3d}  "
                  f"sign p={c['p']:.3f}")
    c = RJ.paired(rows, ('RJ', True), ('B10', False), rule)
    if c['wins'] or c['losses']:
        short = lambda ps: ', '.join(p.replace('gsm-symbolic_', '') for p in ps) or '-'
        print(f"    RJ vs B10 wins   {short(c['wins'])}")
        print(f"    RJ vs B10 losses {short(c['losses'])}")


def mechanism(rows: List[Dict]) -> None:
    print("\n  mechanism (reported, not decisive):")
    for view, who in (('C', 'plain, Qwen'), ('Q', 'Qwen2.5-7B-Instruct reading (stored)')) + tuple(
            (v, f"{RJ.READER_MODEL[v].split('/')[-1]} reading") for v in RJ.READER_VIEWS):
        k, n = RJ.view_accuracy(rows, view)
        if n:
            print(f"    per-sample accuracy  {view:2s} {who:45s} {_pct(k, n)}")
    for v in RJ.READER_VIEWS:
        rds = [(r.get('readings') or {}).get(v) for r in rows]
        rds = [x for x in rds if x is not None]
        if rds:
            print(f"    {v} readings usable {sum(bool(x.get('ok')) for x in rds)}/{len(rds)}, "
                  f"notes/row {sum(x.get('n_notes', 0) for x in rds) / len(rds):.1f}")
    for view in ('Q',) + RJ.READER_VIEWS:
        f = RJ.shared_error_floor(rows, view)
        if f['samples']:
            print(f"    shared-error floor {view:2s}: on {f['rows']} rows where >= 3/5 plain samples agree "
                  f"on a wrong answer, {f['repeat']}/{f['samples']} samples repeat it "
                  f"({100 * f['repeat'] / f['samples']:.0f}%), {f['right']} are right")


def _rule_written() -> bool:
    try:
        RJ.jury_choice([{'families': frozenset({'qwen'}), 'tie_families': frozenset({'qwen'}),
                         'n_pool': 1, 'n_tie': 1, 'real': 1.0, 'answer': 1.0, 'rep': 0}] * 2)
        return True
    except NotImplementedError:
        return False


def summarise(rows: List[Dict], recs: Dict, rule=None) -> int:
    rows = [attach(r, recs.get(r['key'], {})) for r in rows]
    dev = [r for r in rows if not T30.is_held(r)]
    main = [r for r in dev if r['set'] == 'main']
    guard = [r for r in dev if r['set'] == 'guard']
    held = [r for r in rows if r['set'] == 'held']
    done = dev_complete(rows, recs)
    print('\n' + '=' * 78)
    print(f"  v32 Reading Jury: readers {', '.join(RJ.READER_MODEL[v] for v in RJ.READER_VIEWS)}; "
          f"dev part {'complete' if done else 'PARTIAL'}")
    block(main, 'DEV MAIN (P2)', rule)
    block(guard, 'DEV GUARD (GSM-Plus)', rule)
    if main:
        mechanism(main)
    held_done = [r for r in held if RJ.available(r, 'RJ')]
    if held_done:
        block(held_done, 'HELD-OUT (v21 seed 44, C5 only: no B10/B9)', rule)
    print('\n  pre-registered reading:')
    if not done:
        print("    none: PARTIAL run (dev rows incomplete). Re-run the identical command.")
        return 0
    if not (rule is not None or _rule_written()):
        print("    none: reading_jury.jury_choice is not written")
        return 0
    for line in RJ.screen_verdict(main, guard, rule):
        print('    ' + line)
    if held and len(held_done) == len(held):
        print('    ' + RJ.held_verdict(held, rule))
    elif held:
        print(f"    HELD     not read yet: {len(held_done)}/{len(held)} held-out rows complete")
    return 0


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _save(path: str, recs: Dict, meta: Dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(dict(meta, rows=recs), fh, ensure_ascii=False)
    os.replace(tmp, path)


def _row_line(i: int, n: int, row: Dict, rec: Dict, t0: float, ran: int) -> str:
    r = attach(dict(row), rec)
    f = lambda v: f"{v}={sum(V21.correct(s.get('answer'), r['gold']) for s in r[v])}/{len(r[v])}"
    pool = 'RJ' if RJ.available(r, 'RJ') else None
    g = ''
    if pool:
        base = 'B5' if T30.is_held(r) else 'B10'
        b = 'R' if RJ.right(r, base, False) else '-'
        try:
            j = 'R' if RJ.right(r, 'RJ', True) else '-'
        except NotImplementedError:
            j = '?'
        g = f"{base} {b}  RJ {j}"
    rate = f"{(time.time() - t0) / ran:5.0f}s/row" if ran else ''
    return f"[{i:3d}/{n}] {row['set']:5s} {row['pid'][:24]:24s} {f('F1')} {f('F2')}  {g}  {rate}"


def run(args) -> int:
    rows = load_rows(args)
    out_path = args.out or (OUT_STUB if args.stub else OUT)
    recs: Dict[str, Dict] = {}
    if os.path.exists(out_path) and args.resume:
        recs = V24._read(out_path).get('rows', {})
        print(f"resuming: {len(recs)} rows in {out_path}")
    for r in rows:
        recs.setdefault(r['key'], {}).setdefault('readings', {})
    meta = {'readers': dict(RJ.READER_MODEL), 'families': dict(RJ.READER_FAMILY), 'k_read': RJ.K_READ,
            'temperature': args.temperature, 'max_tokens': args.max_tokens, 'verifier': P.PRM,
            'eps': RJ.EPS, 'reader_prompt': SR.READER_SYSTEM}
    print(f"rows: {sum(not T30.is_held(r) for r in rows)} dev, {sum(T30.is_held(r) for r in rows)} held-out",
          flush=True)
    t0 = time.time()
    deadline = t0 + args.max_hours * 3600 if args.max_hours else 0.0
    stop_msg = "\n  --max-hours reached; re-run the identical command to resume."
    for part in parts(rows):
        tag = 'held-out' if T30.is_held(part[0]) else 'dev'
        # 1. readings, one Reader at a time
        for view in RJ.READER_VIEWS:
            todo = [r for r in part if needs_reading(recs[r['key']], view)]
            if not todo:
                continue
            print(f"\n{tag}: {view} readings by {RJ.READER_MODEL[view]} on {len(todo)} rows", flush=True)
            reader = build_reader(args, view)
            try:
                ok = read_rows(reader, recs, todo, view, out_path, meta, deadline)
            finally:
                free(reader)
                if not args.stub:
                    drop_cache_if_tight(RJ.READER_MODEL[view])
            if not ok:
                _save(out_path, recs, meta)
                print(stop_msg)
                return summarise(rows, recs)
        # 2. Solver samples (+ scores)
        todo = [r for r in part if needs_samples(recs[r['key']]) or needs_scores(recs[r['key']])]
        if not todo:
            continue
        together = args.stub or _n_gpus() >= 2
        print(f"\n{tag}: Solver samples on {len(todo)} rows "
              f"({'scored as drawn' if together else 'scored after drawing'})", flush=True)
        need_gen = [r for r in todo if needs_samples(recs[r['key']])]
        solver = build_solver(args, rows) if need_gen else None
        scorer = build_verifier(args, rows) if together else None
        ran = 0
        for i, row in enumerate(todo, 1):
            if deadline and time.time() > deadline:
                _save(out_path, recs, meta)
                print(stop_msg)
                return summarise(rows, recs)
            rec = recs[row['key']]
            if solver is not None and needs_samples(rec):
                draw(solver, row, rec, args)
            if scorer is not None:
                score(scorer, row, rec)
            ran += 1
            _save(out_path, recs, meta)
            print(_row_line(i, len(todo), row, rec, t0, ran), flush=True)
        if not together:
            free(solver)
            solver = None
            scorer = build_verifier(args, rows)
            for row in todo:
                if deadline and time.time() > deadline:
                    _save(out_path, recs, meta)
                    print(stop_msg)
                    return summarise(rows, recs)
                score(scorer, row, recs[row['key']])
                _save(out_path, recs, meta)
        free(solver, scorer)
    _save(out_path, recs, meta)
    print(f"\n  saved: {out_path}")
    return summarise(rows, recs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--v25', default=T30.DEV_V25)
    ap.add_argument('--v26', default=T30.DEV_V26)
    ap.add_argument('--held-traces', default=T30.HELD_TRACES)
    ap.add_argument('--held-prm', default=T30.HELD_PRM)
    ap.add_argument('--held', action='store_true', help='after the dev rows, also the 100 held-out rows')
    ap.add_argument('--preset', default='qwen_math7b_mixed')
    ap.add_argument('--temperature', type=float, default=RJ.TEMPERATURE)
    ap.add_argument('--max-tokens', type=int, default=RJ.MAX_TOKENS)
    ap.add_argument('--out', default='')
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--max-hours', type=float, default=0.0)
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()
    if args.summary_only:
        path = args.out or (OUT_STUB if args.stub else OUT)
        args.held = True
        return summarise(load_rows(args), V24._read(path).get('rows', {}))
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
