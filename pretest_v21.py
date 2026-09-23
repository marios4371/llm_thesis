"""
[v21.0] Pre-test: can a deterministic premise ledger break the correlated
errors that self-consistency locks in?

WHERE v20 LEFT THIS
-------------------
v20 settled two things on gsm-symbolic-p2 (n=92, pretest_v20.json):
  * plain self-consistency@3 is +7.61pp over one greedy call -- the largest
    gain this project has measured, and the baseline any method must now beat;
  * of the 24 rows SC@3 still gets wrong, 13 have NO correct sample among the
    three and 6 more have a wrong 2-vs-1 majority. 19 of 24 failures are
    correlated errors, which no vote over answers can fix: that is the
    "tyranny of the majority" regime (Choi et al., Debate or Vote, 2025).

Reading those rows, the commonest correlated error is not a wrong operation
but a MISSING one -- a clause every sample skips (the "three trucks" of
p2_389, the "two times as many wasps" of p2_2185, the "four times as many
ants" of p2_2184). A skipped clause leaves a fingerprint no model is needed
to read: one of its quantities never becomes an operand. premise_ledger.py
reads it.

WHY THIS IS NOT ANOTHER SELECTION RULE
--------------------------------------
v15.8 closed selection-by-telemetry (2880 rules, all fail held out). This is
different in the two ways that matter:
  1. the signal is not fitted: one rule, written down before any CoT trace
     exists, with no threshold learned from outcomes;
  2. it validated offline on data it was not written for. On the v17 seed-44
     runs, where the Programmer and the Architect disagree and exactly one is
     right, "prefer the derivation that left fewer stated quantities unused"
     picks the right one 29/29 (v17.0) and 21/23 (v17.1). Zero LLM calls.
And it GENERATES, which selection cannot: when every sample skipped the same
clause, the ledger names the clause, and one extra call re-solves the problem
with that clause pointed out (arm R). That is the part that can reach rows
where the correct answer was never sampled.

THE ARMS (one pool of k=5 sampled CoT traces per row; arms are views of it)
--------
  S3, S5   self-consistency over the first 3 / all 5 samples. S5 is the
           compute ceiling the method must not need to exceed.
  L3, L5   premise vote: plurality among the samples that left the FEWEST
           stated quantities unused. Zero extra calls.
  R        targeted re-read. Triggered when some quantity is unused by all
           three of the first samples; one greedy call on the problem plus
           "Make sure your solution takes this into account: <sentence>".
  R0       the same call with a GENERIC note ("make sure you use all the
           relevant information"). Same cost as R. Separates "the ledger
           found the clause" from "re-reading helps" (Re2, Xu et al. 2024).
  L3R      premise vote over {3 samples, R}.    <- THE METHOD
  L3R0     premise vote over {3 samples, R0}.   <- its attribution control
  S4       plain majority over the first 4 samples, for a same-cost look.

PRE-REGISTERED, written before the run
--------------------------------------
Main set: gsm-symbolic-p2, the v20 manifest (same 100 rows, seed 44), so the
v20 greedy control is paired for free.

    L3R - S3 >= +5 rows, wins >= 2x losses, AND L3R >= S5
        -> the ledger improves reasoning at LESS compute than SC@5: full run
    L3R - L3R0 >= +3 rows
        -> the gain is the ledger's localisation, not re-reading
    L3R <= S3
        -> KILL: the ledger carries no usable signal on CoT; direction closes

Guard set: GSM-Plus "distraction insertion" (n=40), where a stated quantity
is irrelevant ON PURPOSE. This is the ledger's known failure mode -- "use
every number" is the shortcut SVAMP was built to punish -- so its harm is
measured, not assumed:
    L3R - S3 >= -2 rows on the guard set -> harm acceptable, report it
    otherwise -> the method needs a distractor gate before any full run

n=100 detects a large effect only; the summary reports discordant counts,
not p-values, so nobody reads it as a confirmation.

COST
----
The k samples are drawn in ONE batched generate (num_return_sequences=k),
which on a memory-bound T4 decode costs far less than k separate calls; it
falls back to k sequential calls if the batched path fails. R and R0 are
greedy and only run on triggered rows. Budget: measure it -- the first rows
print s/row. With --max-hours 8.0 a dead session resumes where it stopped.

    python pretest_v21.py --build-guard                   # offline, once
    python pretest_v21.py --max-hours 8.0                 # main + guard
    python pretest_v21.py --stub --limit 12               # offline, no GPU
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import zlib
from typing import Dict, List, Optional, Sequence

import premise_ledger as L

MAIN_MANIFEST = 'pretest_data/v20_problems.json'
GUARD_MANIFEST = 'pretest_data/v21_guard_distraction.json'
V20_RESULTS = 'results_September/pretest_v20.json'
OUT = 'pretest_v21.json'

COT_PROMPT = (
    "Solve this math problem step by step. After your reasoning, "
    "state the final numeric answer on a line starting with 'Answer:'.\n\n"
    "Problem: {problem}\n\nLet's think step by step."
)
NOTE_MARKER = 'make sure your solution takes this into account:'
GENERIC_NOTE = ("Note: read the problem carefully and make sure your solution "
                "takes all of the relevant information into account.")


def correct(pred: Optional[float], gold: Optional[float]) -> bool:
    """The repo's grading tolerance, unchanged."""
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= max(1e-3, 1e-4 * abs(gold))


def parse_answer(raw: str) -> Optional[float]:
    """Exactly baselines.chain_of_thought's parse, so S here means what S
    meant in v20: an 'Answer:' line first, then the shared numeric fallback."""
    import baselines
    s = str(raw)[:4000]
    if baselines._is_error_response(s):
        return None
    m = re.search(r"answer\s*[:=]\s*([\-+]?\d+(?:\.\d+)?)", s, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return baselines._parse_numeric(s)


# ---------------------------------------------------------------------------
# notes
# ---------------------------------------------------------------------------

def targeted_note(text: str, omitted: Sequence[L.Premise], max_sentences: int = 2) -> str:
    """Quote, verbatim, the sentences holding the quantities nobody spent."""
    sents = L.split_sentences(text)
    idx: List[int] = []
    for p in omitted:
        for s in p.sentences:
            if s not in idx and 0 <= s < len(sents):
                idx.append(s)
    idx = sorted(idx[:max_sentences])
    quoted = ' '.join(f'"{sents[i]}"' for i in idx)
    return f"Note: {NOTE_MARKER} {quoted}"


# ---------------------------------------------------------------------------
# model access
# ---------------------------------------------------------------------------

class _StubClient:
    """Offline stand-in. Writes a LaTeX CoT that spends every quantity and
    answers the gold -- except that, per row, it can 'forget' one clause and
    answer something else, which is the failure the ledger exists to see. A
    note naming the forgotten clause makes it remember."""
    provider = model_name = 'stub'

    def __init__(self, rows: List[Dict], seed: int = 0):
        self.rows = rows
        self.rng = random.Random(seed)
        self.calls = 0

    def _row(self, content: str) -> Dict:
        for r in self.rows:
            if r['text'] in content:
                return r
        return self.rows[0]

    def complete(self, content: str, temperature: float) -> str:
        self.calls += 1
        r = self._row(content)
        prem = L.extract_premises(r['text'])
        h = zlib.crc32(r['problem_id'].encode())
        forget = prem[h % len(prem)] if prem and h % 3 == 0 else None
        if forget is not None and temperature > 0 and self.rng.random() < 0.3:
            forget = None
        note = (content.split(NOTE_MARKER, 1)[-1]
                if NOTE_MARKER in content else '')
        if forget is not None and note and \
                L.split_sentences(r['text'])[forget.sentence] in note:
            forget = None
        used = [p for p in prem if p is not forget]
        expr = ' + '.join(str(p.value) for p in used) or '0'
        ans = r['gold'] if forget is None else r['gold'] + 1
        return f"Adding what matters: \\[ {expr} = {ans} \\]\nAnswer: {ans}"

    def call_model(self, msgs, temperature=0.0, max_tokens=0, **kw):
        return self.complete(msgs[-1]['content'], temperature)


def build_client(preset: str, stub_rows):
    if stub_rows is not None:
        return _StubClient(stub_rows)
    from Mas_solver import (AgentRole, UnifiedLLMClient, HETEROGENEOUS_PRESETS,
                            SOLVER_VERSION, LOCAL_HF_SAMPLING)
    cfg = HETEROGENEOUS_PRESETS[preset]
    m = cfg[AgentRole.BASELINE]
    print(f"solver_version={SOLVER_VERSION} preset={preset}")
    print(f"  solver: {m.provider}/{m.model_name}")
    LOCAL_HF_SAMPLING.update(enabled=True)
    print(f"  sampling ENABLED: {LOCAL_HF_SAMPLING}")
    # use_cache=False is load-bearing: k sampled calls on one prompt would all
    # be the same cache entry and SC@k would silently become SC@1.
    return UnifiedLLMClient(provider=m.provider, use_cache=False,
                            model_override=m.model_name,
                            load_4bit=getattr(m, 'load_4bit', False))


_BATCH_OK: Optional[bool] = None


def _batched(client, content: str, k: int, temperature: float,
             max_tokens: int) -> List[str]:
    """k samples in ONE generate call. Same chat template, same sampling knobs
    (renormalize_logits + top-k/top-p) as Mas_solver._call_local_hf."""
    import torch
    from Mas_solver import LOCAL_HF_SAMPLING
    client._ensure_local_model()
    tok, mdl = client._local_tokenizer, client._local_model
    msgs = [{'role': 'user', 'content': content}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
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
    return [tok.decode(o[n_in:], skip_special_tokens=True) for o in out]


def sample(client, content: str, k: int, temperature: float,
           max_tokens: int) -> List[str]:
    """k sampled completions; batched when the backend allows it."""
    global _BATCH_OK
    if isinstance(client, _StubClient):
        return [client.complete(content, temperature) for _ in range(k)]
    if _BATCH_OK is not False and getattr(client, 'provider', '') == 'local_hf':
        try:
            outs = _batched(client, content, k, temperature, max_tokens)
            if _BATCH_OK is None:
                uniq = len(set(o[:400] for o in outs))
                print(f"  batched sampling: {uniq}/{k} distinct continuations")
                if uniq < 2:
                    raise RuntimeError('batched samples are identical')
            _BATCH_OK = True
            return outs
        except Exception as exc:
            print(f"  batched sampling unavailable ({type(exc).__name__}: "
                  f"{str(exc)[:100]}); falling back to {k} sequential calls")
            _BATCH_OK = False
    return [str(client.call_model([{'role': 'user', 'content': content}],
                                  temperature=temperature, max_tokens=max_tokens))
            for _ in range(k)]


def greedy(client, content: str, max_tokens: int) -> str:
    if isinstance(client, _StubClient):
        return client.complete(content, 0.0)
    return str(client.call_model([{'role': 'user', 'content': content}],
                                 temperature=0.0, max_tokens=max_tokens))


# ---------------------------------------------------------------------------
# one row
# ---------------------------------------------------------------------------

def _trace(raw: str, premises) -> Dict:
    a = L.audit(premises, L.cot_operands(raw))
    return {'raw': str(raw)[:4000], 'answer': parse_answer(raw),
            'n_unconsumed': a.n_unconsumed,
            'unconsumed': [p.short() for p in a.unconsumed], '_audit': a}


def _plain(answers: Sequence[Optional[float]]) -> Optional[float]:
    g = L.clusters(answers)
    return answers[g[0][0]] if g else None


def run_row(client, r: Dict, k: int, temperature: float, max_tokens: int) -> Dict:
    text, gold = r['text'], r['gold']
    prem = L.extract_premises(text)
    content = COT_PROMPT.format(problem=text)
    t0 = time.time()
    traces = [_trace(raw, prem) for raw in sample(client, content, k, temperature, max_tokens)]
    calls = k
    ans = [t['answer'] for t in traces]
    aud = [t['_audit'] for t in traces]

    rec: Dict = {'pid': r['problem_id'], 'gold': gold, 'dataset': r.get('dataset'),
                 'n_premises': len(prem),
                 'premises': [p.short() for p in prem]}

    def put(name, a, info=None):
        rec[f'{name}_answer'] = a
        rec[f'{name}_correct'] = correct(a, gold)
        if info is not None:
            rec[f'{name}_vote'] = info

    put('S3', _plain(ans[:3]))
    put('S4', _plain(ans[:4]))
    put('S5', _plain(ans[:k]))
    put('L3', *L.premise_vote(ans[:3], aud[:3]))
    put('L5', *L.premise_vote(ans[:k], aud[:k]))

    shared = L.shared_omissions(prem, aud[:3])
    rec['triggered'] = bool(shared)
    rec['shared_omissions'] = [p.short() for p in shared]
    if shared:
        note = targeted_note(text, shared)
        rR = _trace(greedy(client, COT_PROMPT.format(problem=text + '\n\n' + note),
                           max_tokens), prem)
        rR0 = _trace(greedy(client, COT_PROMPT.format(problem=text + '\n\n' + GENERIC_NOTE),
                            max_tokens), prem)
        calls += 2
        rec['R_note'] = note
        rec['R'] = {kk: v for kk, v in rR.items() if kk != '_audit'}
        rec['R0'] = {kk: v for kk, v in rR0.items() if kk != '_audit'}
        put('R', rR['answer'])
        put('R0', rR0['answer'])
        put('L3R', *L.premise_vote(ans[:3] + [rR['answer']], aud[:3] + [rR['_audit']]))
        put('L3R0', *L.premise_vote(ans[:3] + [rR0['answer']], aud[:3] + [rR0['_audit']]))
    else:
        put('L3R', rec['L3_answer'], rec['L3_vote'])
        put('L3R0', rec['L3_answer'], rec['L3_vote'])

    rec['samples'] = [{kk: v for kk, v in t.items() if kk != '_audit'} for t in traces]
    rec['calls'] = calls
    rec['seconds'] = round(time.time() - t0, 1)
    return rec


# ---------------------------------------------------------------------------
# manifests
# ---------------------------------------------------------------------------

def build_guard(path: str, n: int, seed: int,
                perturbation: str = 'distraction insertion') -> int:
    """GSM-Plus rows whose perturbation is an inserted irrelevant quantity.
    Built offline, once, so every session scores the same rows."""
    from datasets import load_dataset
    ds = load_dataset('qintongli/GSM-Plus', split='test')
    idx = [i for i in range(len(ds)) if ds[i]['perturbation_type'] == perturbation]
    random.Random(seed).shuffle(idx)
    rows = []
    for i in idx:
        try:
            g = float(str(ds[i]['answer']).replace(',', '').strip())
        except ValueError:
            continue
        rows.append({'problem_id': f'gsm-plus_{i}', 'text': ds[i]['question'].strip(),
                     'gold': g, 'dataset': f'gsm-plus:{perturbation}'})
        if len(rows) >= n:
            break
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump({'dataset': f'gsm-plus:{perturbation}', 'seed': seed, 'n': len(rows),
                   'note': 'v21 guard set: the ledger\'s known failure mode is an '
                           'irrelevant quantity; this measures its harm.',
                   'rows': rows}, fh, ensure_ascii=False, indent=1)
    print(f"wrote {len(rows)} guard rows -> {path}")
    return 0 if rows else 1


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--main', default=MAIN_MANIFEST)
    ap.add_argument('--guard', default=GUARD_MANIFEST,
                    help="guard manifest; '' to skip the guard set")
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--preset', default='qwen_math7b_mixed')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--limit', type=int, default=0, help='rows per set (0 = all)')
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--max-hours', type=float, default=0.0,
                    help="stop cleanly after this many hours; Kaggle kills a "
                         "batch GPU session at 9h without saving")
    ap.add_argument('--build-guard', action='store_true')
    ap.add_argument('--guard-n', type=int, default=40)
    ap.add_argument('--seed', type=int, default=44)
    args = ap.parse_args()

    if args.build_guard:
        return build_guard(args.guard, args.guard_n, args.seed)
    if args.k < 4:
        print('--k must be >= 4: S4 and S5 are comparators')
        return 2

    sets = []
    for tag, path in (('main', args.main), ('guard', args.guard)):
        if not path:
            continue
        if not os.path.exists(path):
            print(f"{tag} manifest missing: {path}"
                  + ("  (build it: python pretest_v21.py --build-guard)" if tag == 'guard' else ''))
            return 2
        with open(path, encoding='utf-8') as fh:
            rows = json.load(fh)['rows']
        sets.append((tag, rows[:args.limit] if args.limit else rows))
    allrows = [dict(r, set=tag) for tag, rows in sets for r in rows]
    print('sets: ' + ', '.join(f'{t}={len(r)}' for t, r in sets)
          + f'   k={args.k} T={args.temperature}')

    client = build_client(args.preset, allrows if args.stub else None)

    prior: Dict[str, Dict] = {}
    if os.path.exists(args.out) and args.resume:
        try:
            with open(args.out, encoding='utf-8') as fh:
                prior = {r['pid']: r for r in json.load(fh).get('rows', [])}
            print(f"resuming: {len(prior)} rows already in {args.out}")
        except Exception as exc:
            print(f"could not read {args.out} ({exc}); starting fresh")

    out: List[Dict] = []
    t0, ran, left = time.time(), 0, 0
    for i, r in enumerate(allrows, 1):
        if r['problem_id'] in prior and 'L3R_correct' in prior[r['problem_id']]:
            out.append(prior[r['problem_id']])
            continue
        if args.max_hours and time.time() - t0 > args.max_hours * 3600:
            left = len(allrows) - i + 1
            print(f"\n  --max-hours reached with {left} row(s) left; re-run the "
                  f"identical command to resume.")
            break
        rec = run_row(client, r, args.k, args.temperature, args.max_tokens)
        rec['set'] = r['set']
        out.append(rec)
        ran += 1
        el = time.time() - t0
        flag = lambda a: 'OK ' if rec.get(f'{a}_correct') else 'BAD'
        print(f"[{i:3d}/{len(allrows)}] {r['set']:5s} {str(r['problem_id'])[:22]:22s} "
              f"S3={flag('S3')} S5={flag('S5')} L3={flag('L3')} L3R={flag('L3R')}"
              f"{' trig' if rec['triggered'] else '     '}  {el/ran:5.0f}s/row", flush=True)
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump({'main': args.main, 'guard': args.guard, 'preset': args.preset,
                       'k': args.k, 'temperature': args.temperature,
                       'rows': out}, fh, ensure_ascii=False, indent=1)

    if left:
        print(f"\n  PARTIAL: {len(out)}/{len(allrows)} rows. No verdict is read off a partial run.")
    return summarise(out, args)


def summarise(out: List[Dict], args) -> int:
    v20 = {}
    if os.path.exists(V20_RESULTS):
        with open(V20_RESULTS, encoding='utf-8') as fh:
            v20 = {r['pid']: r for r in json.load(fh).get('rows', [])}
    arms = ['S3', 'S4', 'S5', 'L3', 'L5', 'L3R0', 'L3R']
    cost = {'S3': '3', 'S4': '4', 'S5': '5', 'L3': '3', 'L5': '5',
            'L3R0': '3+2*trig', 'L3R': '3+1*trig'}

    def paired(rows, x, y):
        w = sum(1 for r in rows if r.get(f'{x}_correct') and not r.get(f'{y}_correct'))
        l = sum(1 for r in rows if r.get(f'{y}_correct') and not r.get(f'{x}_correct'))
        return w, l

    verdict_rows = {}
    for tag in ('main', 'guard'):
        rows = [r for r in out if r.get('set') == tag]
        if not rows:
            continue
        n = len(rows)
        trig = sum(1 for r in rows if r.get('triggered'))
        print('\n' + '=' * 72)
        print(f"  {tag.upper()} SET  n={n}   triggered {trig}/{n}")
        c_rows = [r for r in rows if r['pid'] in v20 and 'C_correct' in v20[r['pid']]]
        if c_rows:
            c = sum(1 for r in c_rows if v20[r['pid']]['C_correct'])
            print(f"    C   (v20 greedy, paired n={len(c_rows)})       {c:3d} = {100*c/len(c_rows):5.1f}%")
        for a in arms:
            s = sum(1 for r in rows if r.get(f'{a}_correct'))
            print(f"    {a:4s} calls {cost[a]:9s}               {s:3d} = {100*s/n:5.1f}%")
        mean_calls = 3 + trig / n
        print(f"    mean calls of L3R: {mean_calls:.2f}")
        print()
        for x, y in (('L3', 'S3'), ('L3R', 'S3'), ('L3R', 'S5'), ('L3R', 'L3R0'),
                     ('L3R', 'S4'), ('L5', 'S5')):
            w, l = paired(rows, x, y)
            print(f"    {x:4s} vs {y:4s}: W={w:2d} L={l:2d} net={w-l:+d}")
        moved = [r for r in rows if (r.get('L3_vote') or {}).get('moved')]
        if moved:
            good = sum(1 for r in moved if r['L3_correct'] and not r['S3_correct'])
            bad = sum(1 for r in moved if r['S3_correct'] and not r['L3_correct'])
            print(f"    ledger moved the S3 answer on {len(moved)} rows: fixed {good}, broke {bad}")
        tr = [r for r in rows if r.get('triggered')]
        if tr:
            for a in ('R', 'R0'):
                s = sum(1 for r in tr if r.get(f'{a}_correct'))
                print(f"    on triggered rows, {a:2s} alone: {s}/{len(tr)}")
        verdict_rows[tag] = rows

    print('\n  pre-registered reading:')
    rows = verdict_rows.get('main', [])
    if not rows:
        print('    no main-set rows')
    else:
        n = len(rows)
        sc = lambda a: sum(1 for r in rows if r.get(f'{a}_correct'))
        if n != 100 and not args.stub:
            print(f"    main n={n}, not the pre-registered 100 -- report as measured, no verdict")
        w, l = paired(rows, 'L3R', 'S3')
        if sc('L3R') <= sc('S3'):
            print("    KILL: L3R <= S3 -- the ledger carries no usable signal on CoT")
        elif sc('L3R') - sc('S3') >= 5 and w >= 2 * l and sc('L3R') >= sc('S5'):
            print("    PASS: L3R beats S3 by >= 5 rows and matches S5 at fewer calls -> full run")
        else:
            print("    INCONCLUSIVE at this n: L3R > S3 but short of the pass bar")
        wa, la = paired(rows, 'L3R', 'L3R0')
        print("    attribution: " + ("the gain is the ledger's localisation (L3R - L3R0 >= +3)"
                                     if wa - la >= 3 else
                                     "NOT shown -- targeted and generic re-reads are within 3 rows"))
    g = verdict_rows.get('guard', [])
    if g:
        w, l = paired(g, 'L3R', 'S3')
        print(f"    guard: L3R vs S3 on distractor rows net={w-l:+d} -> "
              + ("harm acceptable" if w - l >= -2 else "needs a distractor gate first"))
    print(f"\n  saved: {args.out}   (download it -- Kaggle loses the session)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
