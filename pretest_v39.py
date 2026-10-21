"""
[v39.0] Clause ablation pilot (clause_ablation.py): does a confidently WRONG
answer survive the deletion of a clause it should depend on?

WHY
---
P2's residual errors are recitations: the model solves the familiar GSM8K
problem and ignores a clause P2 added, the same way in every sample, so
voting and the PRM see a confident, self-consistent answer. If the model
ignores clause X, the problem WITHOUT X gets the same answer. Deleting one
unit at a time and re-solving greedily therefore tests, per problem and with
no gold answer, whether the answer depends on each stated condition. This
pilot measures that detector on stored rows whose 5 samples already show the
model confidently wrong or confidently right. No repair is tried here.

ROWS (pretest_data/v39_ablation.json, seed 39, built once by --build-manifest)
-----------------------------------------------------------------------------
From the three stored P2 pools of the paper (P2a seed 45, P2b seed 48, V21
seed 44; 5 samples each at T 0.8, v22's CoT prompt), with answers grouped by
the canonical grader (|a - b| < 1e-3):
  TARGET   every P2 row whose plurality answer is WRONG with >= 3 of 5 agreeing
  CONTROL  2 per TARGET row, drawn from the P2 rows whose plurality is RIGHT
           with >= 4 of 5 agreeing
  GUARD    20 GSM-Plus "distraction insertion" rows (an irrelevant clause was
           inserted by construction) with plurality right, >= 4 of 5 agreeing
Per row the manifest freezes the units, every deletion variant's text, and
each unit's label against the problem it was derived from (added / modified
/ same; GSM-Symbolic 'main' template for P2, the seed question for GSM-Plus).
The labels are for the analysis only; the detector never sees them.

GENERATION
----------
Qwen2.5-Math-7B-Instruct (4-bit, the paper's solver), v22's CoT prompt,
GREEDY, 1024 new tokens (the stored samples' budget). Per row the full
problem and every deletion variant go in one left-padded batch. No PRM.
Answers are parsed as the stored samples were (pretest_v21.parse_answer).

PRE-REGISTERED (V39_PLAN.md section 4; frozen before any generation exists)
--------------------------------------------------------------------------
A row is ANALYSED when its greedy full answer exists and agrees in
correctness with its group (TARGET wrong; CONTROL and GUARD right).
FLAG: some informative sentence or parenthetical whose deletion leaves the
greedy answer unchanged (clause_ablation.flagged, PRIMARY_KINDS).
  DETECTION     fT - fC >= 0.20 and one-sided Fisher p < 0.05
                (fT, fC = flag rates of the analysed TARGET / CONTROL rows)
  LOCALIZATION  over the >= 10 flagged analysed TARGET rows: the share whose
                flagged units include an ADDED or MODIFIED unit exceeds the
                chance share (same number of units drawn at random from the
                row's eligible units) by >= 0.15, exact one-sided p < 0.05
  VERDICT       both -> GO (design the repair, v40); one -> WEAK; none -> NO-GO
Reported, not decisive: the break-even ratio fT/fC a gated Reader needs
(v24's gains and losses), question clauses included, numeric units only,
AUROC of the invariant share, unit-level rates by label, per pool, one row
per template, the GUARD rows, exclusions and cost.

MODES
-----
    python pretest_v39.py --build-manifest            # once, offline (HF cache + stored pools); commit it
    python pretest_v39.py --parallel --max-hours 8    # Kaggle T4 x2: one worker per GPU; re-run to resume
    python pretest_v39.py --max-hours 11              # one GPU (Colab T4)
    python pretest_v39.py --summary-only              # re-read every result file, no GPU
    python pretest_v39.py --stub                      # offline plumbing, no GPU
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
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import clause_ablation as CA
import pretest_v21 as V21

MANIFEST = 'pretest_data/v39_ablation.json'
OUT = 'pretest_v39.json'
OUT_STUB = 'pretest_v39_stub.json'
SEED = 39
POOLS = ('P2a', 'P2b', 'V21')
TARGET_MIN_AGREE = 3
CONTROL_MIN_AGREE = 4
CONTROL_PER_TARGET = 2
N_GUARD = 20
MAX_TOKENS = 1024

# pre-registered bars (V39_PLAN.md section 4)
DET_MIN_DIFF, DET_ALPHA = 0.20, 0.05
LOC_MIN_EXCESS, LOC_ALPHA, LOC_MIN_ROWS = 0.15, 0.05, 10
CHANGED = ('added', 'modified')

# reported only: v24 measured, per sample, +19.6 pp where the Solver had misread
# and -12.5 pp where it had read correctly when it was given a Reader's reading
V24_GAIN, V24_LOSS = 0.196, 0.125

STORED_TEXTS = ('pretest_data/v22_confirm_p2.json', 'pretest_data/v20_problems.json',
                'pretest_data/v25_confirm_p2.json', 'pretest_data/v22_confirm_guard.json',
                'pretest_data/v21_guard_distraction.json', 'pretest_data/v25_confirm_guard.json')


def canonical(pred, gold) -> bool:
    """The project's grader, Mas_solver.check_correctness: |pred - gold| < 1e-3."""
    return pred is not None and gold is not None and abs(pred - gold) < 1e-3


def _read(path: str) -> Dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def sha(path: str) -> str:
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read().replace(b'\r\n', b'\n')).hexdigest()[:12]


# ---------------------------------------------------------------------------
# the manifest (offline, once)
# ---------------------------------------------------------------------------

def plurality(samples: Sequence[Dict]) -> Tuple[Optional[float], int]:
    """(plurality answer, its size) under the canonical grader; ties: first seen."""
    groups: List[List[float]] = []
    for s in samples:
        a = s.get('answer')
        if a is None:
            continue
        for g in groups:
            if canonical(a, g[0]):
                g.append(a)
                break
        else:
            groups.append([a])
    if not groups:
        return None, 0
    best = max(groups, key=len)
    return best[0], len(best)


def _index(pid: str) -> int:
    return int(pid.rsplit('_', 1)[1])


def _p2_annotator():
    """annotate(pid, text) for GSM-Symbolic P2 rows: template words against the
    base template ('main', same id), all 50 instances of each."""
    from datasets import load_dataset
    import renderings as RD
    p2 = load_dataset('apple/GSM-Symbolic', 'p2', split='test')
    mn = load_dataset('apple/GSM-Symbolic', 'main', split='test')
    by_p2, by_mn = defaultdict(list), defaultdict(list)
    for i, r in enumerate(p2):
        by_p2[r['id']].append(r['question'])
    for r in mn:
        by_mn[r['id']].append(r['question'])
    cache: Dict[int, Tuple[set, set, List[str]]] = {}

    def annotate(pid: str, text: str):
        rec = p2[_index(pid)]
        if rec['question'].strip() != text.strip():
            raise ValueError(f"{pid}: stored text differs from GSM-Symbolic p2[{_index(pid)}]")
        t = rec['id']
        if t not in cache:
            ref, seen = [], set()
            for q in by_mn[t]:
                for s in RD.split_sentences(q):
                    if s not in seen:
                        seen.add(s)
                        ref.append(s)
            cache[t] = (CA.template_words(by_p2[t]), CA.template_words(by_mn[t]), ref)
        keep, ref_keep, ref = cache[t]
        return t, CA.annotate(text, ref, keep, ref_keep)
    return annotate


def _plus_annotator():
    from datasets import load_dataset
    import renderings as RD
    gp = load_dataset('qintongli/GSM-Plus', split='test')

    def annotate(pid: str, text: str):
        rec = gp[_index(pid)]
        if rec['question'].strip() != text.strip():
            raise ValueError(f"{pid}: stored text differs from GSM-Plus[{_index(pid)}]")
        if rec['perturbation_type'] != 'distraction insertion':
            raise ValueError(f"{pid}: not a distraction-insertion row ({rec['perturbation_type']})")
        return None, CA.annotate(text, RD.split_sentences(rec['seed_question']))
    return annotate


def build_manifest(force: bool = False) -> int:
    if os.path.exists(MANIFEST) and not force:
        print(f"{MANIFEST} exists; refusing to overwrite pre-registered rows (--force to redo)")
        return 1
    import replay_v35 as R
    texts: Dict[str, str] = {}
    for f in STORED_TEXTS:
        for r in _read(f)['rows']:
            texts[r['problem_id']] = r['text']
    for r in _read(R.POOLS['P2b'])['rows']:
        if r.get('text'):
            if texts.get(r['pid'], r['text']) != r['text']:
                raise ValueError(f"{r['pid']}: two different stored texts")
            texts[r['pid']] = r['text']
    cand = {'TARGET': [], 'CONTROL': [], 'GUARD': []}
    for pool in POOLS:
        rows = R.load(pool)
        if not rows:
            raise FileNotFoundError(f"pool {pool} not found ({R.POOLS[pool]}): the manifest needs all three")
        for r in rows:
            a, n = plurality(r['C'][:5])
            right = canonical(a, r['gold'])
            x = {'pid': r['pid'], 'pool': pool, 'gold': r['gold'], 'text': texts[r['pid']],
                 'stored': {'plurality': a, 'agree': n, 'n_right': sum(canonical(s.get('answer'), r['gold'])
                                                                       for s in r['C'][:5])},
                 '_raws': [s.get('raw') or '' for s in r['C'][:5] if canonical(s.get('answer'), a)]}
            if r['set'] == 'main' and a is not None and not right and n >= TARGET_MIN_AGREE:
                cand['TARGET'].append(dict(x, group='TARGET'))
            elif r['set'] == 'main' and right and n >= CONTROL_MIN_AGREE:
                cand['CONTROL'].append(dict(x, group='CONTROL'))
            elif r['set'] == 'guard' and right and n >= CONTROL_MIN_AGREE:
                cand['GUARD'].append(dict(x, group='GUARD'))
    rng = random.Random(SEED)
    eligible = {g: len(v) for g, v in cand.items()}
    for g in ('CONTROL', 'GUARD'):
        cand[g].sort(key=lambda r: r['pid'])
        rng.shuffle(cand[g])
    cand['CONTROL'] = cand['CONTROL'][:CONTROL_PER_TARGET * len(cand['TARGET'])]
    cand['GUARD'] = cand['GUARD'][:N_GUARD]
    p2_ann, plus_ann = _p2_annotator(), _plus_annotator()
    rows = []
    for g in ('TARGET', 'CONTROL', 'GUARD'):
        for r in cand[g]:
            ann = p2_ann if r['pid'].startswith('gsm-symbolic_p2_') else plus_ann
            template, (sents, qi, us) = ann(r['pid'], r['text'])
            raws = r.pop('_raws')
            for u in us:
                u['variant'] = CA.delete(sents, u)
                u['spent'] = CA.spent(u['text'], raws)
            r['template'] = template
            r['units'] = us
            rows.append(r)
    rows.sort(key=lambda r: r['pid'])
    rng.shuffle(rows)
    meta = {'dataset': 'v39 clause ablation: stored P2 pools (TARGET confident-wrong, CONTROL confident-right) '
                       '+ GSM-Plus distraction GUARD',
            'seed': SEED, 'n': len(rows), 'eligible': eligible,
            'n_by_group': dict(Counter(r['group'] for r in rows)),
            'constants': {'ADDED_BELOW': CA.ADDED_BELOW, 'SAME_FROM': CA.SAME_FROM, 'CONST_SHARE': CA.CONST_SHARE,
                          'PRIMARY_KINDS': list(CA.PRIMARY_KINDS)},
            'rows': rows}
    with open(MANIFEST, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    n_units = sum(len(r['units']) for r in rows)
    print(f"  eligible: {eligible}")
    print(f"wrote {len(rows)} rows ({meta['n_by_group']}), {n_units} deletion variants -> {MANIFEST}")
    return 0


def load_rows() -> List[Dict]:
    return _read(MANIFEST)['rows']


def complete(rec: Optional[Dict], row: Dict) -> bool:
    rec = rec or {}
    return 'full' in rec and len(rec.get('units') or []) == len(row['units'])


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

_PROBLEM = re.compile(r"Problem: (.*)\n\nLet's think step by step\.$", re.S)


class _StubSolver:
    """Offline stand-in. Full problem: right for CONTROL/GUARD, wrong for TARGET
    (every 9th row the other way, so the exclusion rule runs). Variant: the full
    answer when the unit is one the 'model' ignores (a changed unit of a TARGET
    row, or every 5th unit), otherwise a different answer; every 13th variant
    has no parsable answer."""
    provider = 'stub'

    def __init__(self, rows: Sequence[Dict]):
        self.where: Dict[str, Tuple[Dict, Optional[int]]] = {}
        for r in rows:
            self.where.setdefault(r['text'], (r, None))
            for j, u in enumerate(r['units']):
                self.where.setdefault(u['variant'], (r, j))

    def complete(self, content: str) -> str:
        m = _PROBLEM.search(content)
        r, j = self.where.get(m.group(1) if m else '', (None, None))
        if r is None:
            return "Adding the parts: 17 + 25 = 42.\nAnswer: 42"
        h = zlib.crc32(r['pid'].encode())
        full = r['gold'] + (1.0 if (r['group'] == 'TARGET') != (h % 9 == 0) else 0.0)
        if j is None:
            return f"We combine the stated quantities.\nAnswer: {full:g}"
        u = r['units'][j]
        hu = zlib.crc32(u['text'].encode())
        if hu % 13 == 0:
            return "The problem does not say enough to determine this."
        ignored = (r['group'] == 'TARGET' and u.get('label') in CHANGED) or hu % 5 == 0
        return f"We combine the stated quantities.\nAnswer: {full if ignored else full + j + 2:g}"


_BATCH_FALLBACK = False


def greedy_batch(client, contents: List[str], max_tokens: int) -> List[str]:
    """One greedy completion per prompt, in ONE left-padded generate call (the
    chat template and the greedy settings of Mas_solver's local path). A batch
    that runs out of memory is halved; any other failure falls back, for the
    rest of the run, to one V21.greedy call per prompt."""
    global _BATCH_FALLBACK
    if isinstance(client, _StubSolver):
        return [client.complete(c) for c in contents]
    if _BATCH_FALLBACK or getattr(client, 'provider', '') != 'local_hf':
        return [V21.greedy(client, c, max_tokens) for c in contents]
    import torch
    try:
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
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out = mdl.generate(**enc, max_new_tokens=max_tokens, do_sample=False, temperature=None,
                               top_p=None, top_k=None, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        n_in = enc['input_ids'].shape[1]
        return [tok.decode(o[n_in:], skip_special_tokens=True) for o in out]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(contents) == 1:
            raise
        h = len(contents) // 2
        print(f"    OOM on a batch of {len(contents)}; splitting", flush=True)
        return greedy_batch(client, contents[:h], max_tokens) + greedy_batch(client, contents[h:], max_tokens)
    except Exception as exc:
        print(f"  batched greedy failed ({type(exc).__name__}: {str(exc)[:160]}); "
              f"one call per prompt from now on", flush=True)
        _BATCH_FALLBACK = True
        return [V21.greedy(client, c, max_tokens) for c in contents]


def smoke_batch(client) -> None:
    """Fail in a minute, not after hours: a padded batch of two prompts of
    different lengths must solve both (the short one is the left-padded one)."""
    short = V21.COT_PROMPT.format(problem='What is 17 + 25?')
    long_ = V21.COT_PROMPT.format(problem='A farmer keeps 17 cows in the north field and 25 cows in the south '
                                          'field. He has no other animals. How many cows does the farmer keep?')
    outs = greedy_batch(client, [short, long_], 256)
    if len(outs) != 2 or not all('42' in o for o in outs):
        raise RuntimeError(f"batch smoke test failed: {[o[-200:] for o in outs]}")
    print(f"  batch smoke test OK: {[V21.parse_answer(o) for o in outs]}", flush=True)


# ---------------------------------------------------------------------------
# reading the results
# ---------------------------------------------------------------------------

def reading(row: Dict, rec: Dict) -> Dict:
    """One row's detector reading (complete rows only)."""
    us = row['units']
    full = rec['full']['answer']
    ans = [u.get('answer') for u in rec['units']]
    full_ok = canonical(full, row['gold'])
    all_kinds = ('sentence', 'paren', 'qclause')
    elig = CA.eligible(us)
    flag = CA.flagged(us, full, ans)
    return {'pid': row['pid'], 'group': row['group'], 'pool': row['pool'], 'template': row.get('template'),
            'full': full, 'full_ok': full_ok,
            'analysed': full is not None and full_ok == (row['group'] != 'TARGET'),
            'same_as_stored': canonical(full, row['stored']['plurality']),
            'eligible': elig, 'flagged': flag, 'flag': bool(flag),
            'flag_all': bool(CA.flagged(us, full, ans, kinds=all_kinds)),
            'flag_numeric': bool(CA.flagged(us, full, ans, numeric=True)),
            'ledger': any(us[j].get('spent') is False for j in elig),
            'share': len(flag) / len(elig) if elig else 0.0,
            'labels': [u.get('label') for u in us], 'units': us, 'answers': ans,
            'unparsed': sum(a is None for a in ans)}


def localization(rows: Sequence[Dict]) -> Dict:
    """Over flagged TARGET readings: hits (a flagged unit is added/modified) vs
    the chance of a hit with the same number of units drawn at random."""
    hits, chance = 0, []
    for r in rows:
        elig, flag = r['eligible'], r['flagged']
        changed = [j for j in elig if r['labels'][j] in CHANGED]
        hits += any(j in changed for j in flag)
        chance.append(CA.chance_hit(len(elig), len(changed), len(flag)))
    n = len(rows)
    return {'n': n, 'hits': hits, 'H': hits / n if n else 0.0, 'H0': sum(chance) / n if n else 0.0,
            'p': CA.tail_at_least(chance, hits) if n else 1.0}


def analyse(read: Sequence[Dict]) -> Dict:
    a = {g: [r for r in read if r['group'] == g and r['analysed']] for g in ('TARGET', 'CONTROL', 'GUARD')}
    t, c = a['TARGET'], a['CONTROL']
    ft = sum(r['flag'] for r in t) / len(t) if t else 0.0
    fc = sum(r['flag'] for r in c) / len(c) if c else 0.0
    det_p = CA.fisher_greater(sum(r['flag'] for r in t), sum(not r['flag'] for r in t),
                              sum(r['flag'] for r in c), sum(not r['flag'] for r in c))
    loc = localization([r for r in t if r['flag']])
    eps = 1e-12                       # 0.7 - 0.5 is 0.19999999999999996 in floating point
    det_met = bool(t and c) and ft - fc >= DET_MIN_DIFF - eps and det_p < DET_ALPHA
    loc_met = loc['n'] >= LOC_MIN_ROWS and loc['H'] - loc['H0'] >= LOC_MIN_EXCESS - eps and loc['p'] < LOC_ALPHA
    return {'groups': a, 'fT': ft, 'fC': fc, 'det_p': det_p, 'det_met': det_met, 'loc': loc, 'loc_met': loc_met,
            'verdict': 'GO' if det_met and loc_met else 'WEAK' if det_met or loc_met else 'NO-GO'}


def _rate(rows: Sequence[Dict], key: str) -> str:
    return f"{sum(r[key] for r in rows)}/{len(rows)}" + (f" = {sum(r[key] for r in rows) / len(rows):.2f}"
                                                         if rows else '')


def _report(read: List[Dict], res: Dict, eligible: Dict) -> None:
    g = res['groups']
    t, c, gd = g['TARGET'], g['CONTROL'], g['GUARD']
    print("\n  rows read / analysed (greedy full answer exists and agrees in correctness with the group):")
    for name in ('TARGET', 'CONTROL', 'GUARD'):
        rr = [r for r in read if r['group'] == name]
        print(f"    {name:8s} read {len(rr):3d}  analysed {len(g[name]):3d}  "
              f"full answer missing {sum(r['full'] is None for r in rr)}  "
              f"greedy flipped correctness {sum(r['full'] is not None and not r['analysed'] for r in rr)}  "
              f"greedy = stored plurality {sum(r['same_as_stored'] for r in rr)}")
    print("\n  flag rates (analysed rows)      TARGET            CONTROL           GUARD")
    for key, name in (('flag', 'PRIMARY sentence+paren'), ('flag_all', 'with question clauses'),
                      ('flag_numeric', 'numeric units only'), ('ledger', 'textual ledger (no model)')):
        print(f"    {name:28s}" + ''.join(f"{_rate(x, key):18s}" for x in (t, c, gd)))
    both = sum(r['flag'] and r['ledger'] for r in t)
    print(f"    TARGET: deletion and ledger both {both}, deletion only {sum(r['flag'] for r in t) - both}, "
          f"ledger only {sum(r['ledger'] for r in t) - both}")
    au = CA.auroc([r['share'] for r in t], [r['share'] for r in c])
    print(f"    invariant share of eligible units: mean T {sum(r['share'] for r in t) / max(1, len(t)):.2f}, "
          f"C {sum(r['share'] for r in c) / max(1, len(c)):.2f}; AUROC T vs C "
          f"{au if au is None else round(au, 3)}")
    if res['fC'] > 0:
        need = V24_LOSS * eligible.get('CONTROL', 0) / max(1e-9, V24_GAIN * eligible.get('TARGET', 1))
        print(f"    fT/fC = {res['fT'] / res['fC']:.2f}; a gated Reader breaks even at about {need:.2f} "
              f"(v24's +{V24_GAIN:.3f}/-{V24_LOSS:.3f} over the eligible counts {eligible})")
    print("\n  unit level (eligible units of analysed rows): invariant / units by label")
    for name, rows in (('TARGET', t), ('CONTROL', c), ('GUARD', gd)):
        cnt = Counter()
        for r in rows:
            for j in r['eligible']:
                lab = r['labels'][j] or '-'
                cnt[(lab, 'n')] += 1
                cnt[(lab, 'inv')] += j in r['flagged']
        print(f"    {name:8s}" + ''.join(f"  {lab} {cnt[(lab, 'inv')]}/{cnt[(lab, 'n')]}"
                                         for lab in ('added', 'modified', 'same') if cnt[(lab, 'n')]))
    print("\n  per pool (analysed): flagged TARGET / flagged CONTROL")
    for pool in POOLS:
        print(f"    {pool:4s}  T {_rate([r for r in t if r['pool'] == pool], 'flag'):14s}"
              f"  C {_rate([r for r in c if r['pool'] == pool], 'flag')}")
    first = {}
    for name, rows in (('TARGET', t), ('CONTROL', c)):
        seen = {}
        for r in rows:
            seen.setdefault(r['template'], r)
        first[name] = list(seen.values())
    p1 = CA.fisher_greater(sum(r['flag'] for r in first['TARGET']), sum(not r['flag'] for r in first['TARGET']),
                           sum(r['flag'] for r in first['CONTROL']), sum(not r['flag'] for r in first['CONTROL']))
    print(f"  one row per template: T {_rate(first['TARGET'], 'flag')} ({len(first['TARGET'])} templates), "
          f"C {_rate(first['CONTROL'], 'flag')} ({len(first['CONTROL'])} templates), Fisher p {p1:.4f}")
    hit_g = sum(any(r['labels'][j] in CHANGED for j in r['flagged']) for r in gd if r['flag'])
    print(f"  GUARD: flagged {_rate(gd, 'flag')}; the flagged units include the inserted distractor "
          f"(a changed unit) in {hit_g} of {sum(r['flag'] for r in gd)}")
    print("\n  flagged TARGET rows, the units the answer ignored:")
    for r in t:
        for j in r['flagged']:
            u = r['units'][j]
            print(f"    {r['pid']:26s} {u['kind'][:5]:5s} {str(u.get('label')):8s} {u['text'][:90]}")
    print(f"\n  variants without a parsable answer: {sum(r['unparsed'] for r in read)} of "
          f"{sum(len(r['answers']) for r in read)}")


def summarise(rows: List[Dict], recs: Dict) -> int:
    read = [reading(r, recs[r['pid']]) for r in rows if complete(recs.get(r['pid']), r)]
    for x in read:
        x['seconds'] = recs[x['pid']].get('seconds', 0)
    print('\n' + '=' * 78)
    print(f"  v39 clause ablation pilot: {len(read)}/{len(rows)} rows complete; greedy; canonical grader abs < 1e-3")
    print(f"  frozen: {frozen()}")
    if not read:
        print("  nothing to read yet")
        return 0
    res = analyse(read)
    eligible = _read(MANIFEST).get('eligible', {}) if os.path.exists(MANIFEST) else {}
    _report(read, res, eligible)
    secs = [x['seconds'] for x in read if x['seconds']]
    if secs:
        print(f"  cost: {sum(secs) / len(secs):.0f} s/row, {sum(len(x['answers']) + 1 for x in read)} generations")
    loc = res['loc']
    print('\n  pre-registered reading:')
    print(f"    DETECTION     fT {res['fT']:.2f} - fC {res['fC']:.2f} = {res['fT'] - res['fC']:+.2f} "
          f"(bar {DET_MIN_DIFF:+.2f}), Fisher p = {res['det_p']:.4f} -> {'MET' if res['det_met'] else 'NOT MET'}")
    print(f"    LOCALIZATION  {loc['hits']}/{loc['n']} flagged TARGET rows hit a changed unit: "
          f"{loc['H']:.2f} vs chance {loc['H0']:.2f} (excess {loc['H'] - loc['H0']:+.2f}, bar "
          f"{LOC_MIN_EXCESS:+.2f}; needs >= {LOC_MIN_ROWS} rows), p = {loc['p']:.4f} -> "
          f"{'MET' if res['loc_met'] else 'NOT MET'}")
    if len(read) < len(rows):
        print(f"    VERDICT       none: PARTIAL run, {len(read)}/{len(rows)} rows. Re-run the identical command.")
    else:
        print(f"    VERDICT       {res['verdict']}")
    return 0


def frozen() -> str:
    parts = [f"clause_ablation.py {sha(CA.__file__)}"]
    if os.path.exists(MANIFEST):
        parts.append(f"{os.path.basename(MANIFEST)} {sha(MANIFEST)}")
    return '; '.join(parts)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def outputs(out: str) -> List[str]:
    """`out` and the per-GPU files a --parallel run writes next to it."""
    stem = out[:-5] if out.endswith('.json') else out
    return [out] + sorted(glob.glob(f"{glob.escape(stem)}_gpu*.json"))


def merged(paths: Sequence[str]) -> Dict:
    """Rows of several result files; a complete record wins over a partial one."""
    rows = {r['pid']: r for r in load_rows()}
    out: Dict = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        for k, rec in _read(p).get('rows', {}).items():
            if k not in out or (k in rows and complete(rec, rows[k]) and not complete(out[k], rows[k])):
                out[k] = rec
    return out


def _save(path: str, recs: Dict, meta: Dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(dict(meta, rows=recs), fh, ensure_ascii=False)
    os.replace(tmp, path)


def _shard(spec: str) -> Tuple[int, int]:
    i, n = (int(x) for x in spec.split('/'))
    if not 0 <= i < n:
        raise ValueError(f"--shard {spec}: need 0 <= i < n")
    return i, n


def run(args) -> int:
    if not os.path.exists(MANIFEST):
        print(f"missing {MANIFEST}: run `python pretest_v39.py --build-manifest` first")
        return 2
    rows = load_rows()
    mine = rows
    tag = ''
    if args.shard:
        i, n = _shard(args.shard)
        mine, tag = rows[i::n], f"s{i} "
    if args.reverse:
        mine = list(reversed(mine))
    if args.limit:
        mine = mine[:args.limit]
    out = args.out or (OUT_STUB if args.stub else OUT)
    recs: Dict[str, Dict] = {}
    if args.resume and os.path.exists(out):
        recs = _read(out).get('rows', {})
    # rows another worker or machine already finished (its file next to `out` or in the repo) are skipped
    done_elsewhere = merged(outputs(out) + outputs(OUT_STUB if args.stub else OUT)) if args.resume else {}
    todo = [r for r in mine if not complete(recs.get(r['pid']), r)
            and not complete(done_elsewhere.get(r['pid']), r)]
    meta = {'manifest': MANIFEST, 'frozen': frozen(), 'decoding': 'greedy', 'max_tokens': MAX_TOKENS,
            'prompt': V21.COT_PROMPT, 'preset': args.preset, 'shard': args.shard or 'all'}
    print(f"{tag}rows: {len(mine)}  to do: {len(todo)}  out: {out}  frozen: {meta['frozen']}", flush=True)
    if not todo:
        return 0 if args.shard else summarise(rows, merged(outputs(out)))
    if not args.stub:
        try:
            import pretest_v32 as T32
            T32.scrub_git_credentials(os.path.dirname(os.path.abspath(__file__)))
        except Exception:
            pass
    client = _StubSolver(rows) if args.stub else V21.build_client(args.preset, None)
    smoke_batch(client)
    t0 = time.time()
    deadline = t0 + args.max_hours * 3600 if args.max_hours else 0.0
    for k, row in enumerate(todo, 1):
        if deadline and time.time() >= deadline:
            _save(out, recs, meta)
            print(f"\n  {tag}--max-hours reached; re-run the identical command to resume.", flush=True)
            break
        ts = time.time()
        contents = [V21.COT_PROMPT.format(problem=row['text'])] + \
                   [V21.COT_PROMPT.format(problem=u['variant']) for u in row['units']]
        raws = greedy_batch(client, contents, MAX_TOKENS)
        parsed = [{'raw': str(x), 'answer': V21.parse_answer(x)} for x in raws]
        rec = {'full': parsed[0], 'units': parsed[1:], 'seconds': round(time.time() - ts, 1)}
        recs[row['pid']] = rec
        _save(out, recs, meta)
        x = reading(row, rec)
        print(f"[{tag}{k:3d}/{len(todo)}] {row['pid']:26s} {row['group']:7s} units {len(row['units'])}  "
              f"full {'right' if x['full_ok'] else 'WRONG'}  invariant {len(x['flagged'])}/{len(x['eligible'])}  "
              f"{rec['seconds']:5.0f}s  {(time.time() - t0) / k:5.0f}s/row", flush=True)
    _save(out, recs, meta)
    print(f"\n  {tag}saved: {out}", flush=True)
    if args.shard:
        return 0
    return summarise(rows, merged(outputs(out)))


def run_parallel(args) -> int:
    """One worker process per visible GPU (CUDA_VISIBLE_DEVICES pins it), each on
    every n-th row of the manifest and writing its own file; then one reading."""
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    if args.stub or n < 2:
        print(f"--parallel: {n} GPU(s) visible; running in this process")
        return run(args)
    out = args.out or OUT
    stem = out[:-5] if out.endswith('.json') else out
    procs = []
    for i in range(n):
        cmd = [sys.executable, '-u', os.path.abspath(__file__), '--shard', f'{i}/{n}', '--out',
               f'{stem}_gpu{i}.json', '--preset', args.preset, '--max-hours', str(args.max_hours)]
        if args.limit:
            cmd += ['--limit', str(args.limit)]
        if not args.resume:
            cmd.append('--no-resume')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        procs.append(subprocess.Popen(cmd, env=env))
        print(f"  worker {i} started on GPU {i}: {' '.join(cmd[2:])}", flush=True)
    codes = [p.wait() for p in procs]
    print(f"  workers finished with exit codes {codes}", flush=True)
    summarise(load_rows(), merged(outputs(out)))
    return max(codes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-manifest', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--preset', default='qwen_math7b_mixed')
    ap.add_argument('--out', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--max-hours', type=float, default=0.0)
    ap.add_argument('--reverse', action='store_true', help='work through the rows from the end')
    ap.add_argument('--shard', default='', help='i/n: only every n-th row from row i (set by --parallel)')
    ap.add_argument('--parallel', action='store_true', help='one worker per visible GPU')
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--merge', action='append', default=[], help='more result files (summary only)')
    args = ap.parse_args()
    if args.build_manifest:
        return build_manifest(args.force)
    if args.summary_only:
        out = args.out or (OUT_STUB if args.stub else OUT)
        return summarise(load_rows(), merged(outputs(out) + args.merge))
    if args.parallel:
        return run_parallel(args)
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
