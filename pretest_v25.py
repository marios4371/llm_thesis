"""
[v25.0] Pre-test: Reading Arbitration. Does a process verifier that chooses
between two agents' readings of the problem beat the same verifier choosing
among plain samples, at equal LLM calls, on GSM-Symbolic P2?

The method and its motivation are in reading_arbitration.py; the evidence,
the literature and the novelty check are in V25_PLAN.md. In one line: B5's
remaining errors are self-consistent misreadings that no plain sample fixes,
the Reader's reading fixes many of them but breaks others (v24), and a vote
cannot tell which is which (SC over both views = S5). The verifier decides.

ARMS (all read off the same samples; reading_arbitration.ARMS)
----
  S5    plurality over 5 plain samples                              5 calls
  B5    verifier best-of-5 plain samples (the v22 system)           5 calls
  RA    verifier best of {2 plain, 2 under the Reader's reading}    5 calls  <- v25
  B4    verifier best-of-4 plain (RA's solver samples, no Reader)   4 calls
  RA5   {3 plain, 2 read}                                           6 calls
  RA7   {5 plain, 2 read}                                           8 calls
  VRA   RA's pool with a plurality vote                             5 calls
  BQ2   verifier best-of-2 read                                     3 calls
Plain = the CoT prompt of every earlier version. Read = v24's ASQ prompt: the
Reader's sentence-anchored situation QAs interleaved after each sentence.
The verifier (Qwen2.5-Math-PRM-7B) always scores against the ORIGINAL
problem text and never sees the notes. Aggregation: last-step reward, the
rule v22 chose on dev and confirmed on fresh rows. Ties go to the plain
sample.

PRE-REGISTERED, frozen 2026-09-28, before any reader-view sample has a
verifier score
--------------------------------------------------------------------------
Dev screen (--dev): the 100 main + 20 guard rows of the v22 confirmation.
  C = the 5 plain samples v22 stored, with the verifier scores v22 stored.
  Q = the 5 ASQ samples v24 stored, scored here by the same verifier code.
  SCREEN  RA - B5 >= +3 rows with W >= 2L  -> GO: run the confirmation
          RA - B5 <= 0                     -> STOP
          otherwise                        -> WEAK (the user's call)
  These rows are fresh for the verifier scores, NOT for the idea: their
  oracle numbers are what motivated RA. The screen is never a thesis claim.

Confirmation (default mode): 100 fresh P2 rows + 20 GSM-Plus distractor
rows (seed 48; disjoint from every row of v20-v24). Per row: 5 plain
samples, one greedy Reader call, 2 read samples (T=0.8, top-p 0.95,
top-k 50, 1024 tokens: the v22 settings), every sample verifier-scored.
  PRIMARY (main)  RA vs B5, both 5 calls:
          net >= +4 rows AND W >= 2L AND net >= +2 without the single
          template that contributes most                -> SUPPORTED
          net <= 0                                      -> REFUTED
          otherwise                                     -> INCONCLUSIVE
          The exact sign-test p is printed with it. A claim of statistical
          significance needs p < 0.05 by itself.
  GUARD           RA - B5 >= -1 row                     -> no harm
  SECONDARY (reported, not decisive)
          VRA vs S5: the reading's diversity is predicted to VANISH under a
            vote (|net| <= 2). If it does, the verifier is what harvests it.
          RA vs B4 (same 4 solver samples) and RA5 vs B5 (same 5).
          Where RA's wins come from: 'reading' (the right answer is only in
            the read samples) vs 'selection' (it was among the 5 plain).
          Pooled dev + confirmation, RA vs B5, sign test (--pooled). The dev
            half is not fresh for the idea; say so wherever it is quoted.
  EXPLORATORY (printed, never used to choose anything): the cascade that
  stops when the first two plain samples agree; the verifier's score of the
  Reader's own notes ("reading vetting"); which view the verifier picks;
  the allocation table on dev; truncation sensitivity on dev.

MODES
-----
    python pretest_v25.py --dev                      # dev screen, verifier only, ~1 GPU-hour
    python pretest_v25.py --dev --summary-only       # re-read, no GPU
    python pretest_v25.py --build-manifest           # once (needs `datasets`); then commit
    python pretest_v25.py --max-hours 8.0            # confirmation, ~11 h: two sessions
    python pretest_v25.py --summary-only             # re-read the confirmation
    python pretest_v25.py --pooled pretest_v25_dev.json pretest_v25.json
    python pretest_v25.py --stub --limit 8 [--dev]   # offline plumbing, no GPU
With two GPUs the confirmation interleaves generation and scoring per row
(Solver split over both, Reader on cuda:1, verifier on the roomier GPU).
With one GPU it generates every row first, frees the Solver and Reader, then
scores; `--stage gen` / `--stage score` run one half on its own. Re-running
the identical command RESUMES: a row keeps every sample, reading and score it
already has.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import pretest_v21 as V21
import pretest_v24 as V24
import reading_arbitration as RA
import score_prm_v22 as P
import situation_reader as SR

DEV_V24 = 'results_September/preset_v24_dev.json'     # the v24 dev pass (ASQ samples, readings)
DEV_V22 = 'results_September/preset_v22.json'         # the v22 confirmation (C samples + scores)
DEV_MANIFESTS = ('pretest_data/v22_confirm_p2.json', 'pretest_data/v22_confirm_guard.json')
MANIFESTS = (('main', 'pretest_data/v25_confirm_p2.json'),
             ('guard', 'pretest_data/v25_confirm_guard.json'))
EARLIER_MANIFESTS = V24.EARLIER_MANIFESTS + ('pretest_data/v24_confirm_p2.json',
                                             'pretest_data/v24_confirm_guard.json')
OUT = 'pretest_v25.json'
OUT_DEV = 'pretest_v25_dev.json'
OUT_STUB = 'pretest_v25_stub.json'    # a stub run never writes the real files
CONFIRM_SEED = 48
KC, KQ = 5, 2                          # confirmation draws: plain, read
DRIFT_ROWS = 10                        # dev: re-score this many rows' plain samples
TRUNC = 3990                           # a stored raw this long may have lost its end
C, Q = RA.PLAIN, RA.READ


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def load_dev(v24_path: str = DEV_V24, v22_path: str = DEV_V22, limit: int = 0) -> List[Dict]:
    """The v24 dev rows: their plain samples and verifier scores come from the
    v22 run, their read samples and readings from the v24 run."""
    texts = V24._texts(DEV_MANIFESTS)
    v22 = {r['pid']: r for r in V24._read(v22_path)['rows']}
    rows = []
    for r in V24._read(v24_path)['rows']:
        old = v22[r['pid']]
        plain = [{'raw': str(s.get('raw', '')), 'answer': s.get('answer'),
                  'prm': s.get('prm'), 'stored_prm': True} for s in old['samples']]
        if [s['answer'] for s in plain] != [s.get('answer') for s in r['arms']['C']['samples']]:
            raise ValueError(f"{r['pid']}: the v24 C arm is not the v22 samples")
        read = [{'raw': str(s.get('raw', '')), 'answer': s.get('answer')}
                for s in r['arms'].get('ASQ', {}).get('samples', [])]
        rows.append({'key': f"dev:{r['pid']}", 'pid': r['pid'], 'source': 'dev', 'set': r['set'],
                     'gold': r['gold'], 'template': r.get('template'), 'text': texts[r['pid']],
                     C: plain, Q: read, 'reading': r.get('reading') or {}})
    return _limit(rows, limit)


def load_confirm(limit: int = 0) -> List[Dict]:
    rows = []
    for tag, path in MANIFESTS:
        for r in V24._read(path)['rows']:
            rows.append({'key': f"v25:{r['problem_id']}", 'pid': r['problem_id'], 'source': 'v25',
                         'set': tag, 'gold': r['gold'], 'template': V24.template_of(r['problem_id']),
                         'text': r['text'], C: [], Q: []})
    return _limit(rows, limit)


def _limit(rows: List[Dict], limit: int) -> List[Dict]:
    if not limit:
        return rows
    out = []
    for tag in ('main', 'guard'):
        out += [r for r in rows if r['set'] == tag][:max(1, limit // 2)]
    return out


def merge(row: Dict, prior: Optional[Dict]) -> Dict:
    """A resumed row keeps everything it already has. Dev rows keep their
    stored samples and take only the scores from the prior file."""
    if not prior:
        return row
    if row['source'] == 'dev':
        for view in (C, Q):
            for s, p in zip(row[view], prior.get(view, [])):
                if s.get('prm') is None and p.get('prm') is not None and p.get('answer') == s.get('answer'):
                    s['prm'] = p['prm']
                if 'prm_rescored' in p:
                    s['prm_rescored'] = p['prm_rescored']
        for k in ('reading_prm', 'drift', 'seconds'):
            if k in prior:
                row[k] = prior[k]
        return row
    return dict(prior, text=row['text'])


# ---------------------------------------------------------------------------
# agents
# ---------------------------------------------------------------------------

class _StubPRM:
    """Offline verifier: the last step scores high when the answer is right,
    with noise, so every code path runs and the arms can differ."""

    def __init__(self, rows: Sequence[Dict]):
        self.gold = {r['text']: r['gold'] for r in rows}

    def score(self, problem: str, steps: List[str]) -> List[float]:
        g = self.gold.get(problem)
        a = V21.parse_answer('\n\n'.join(steps))
        rng = random.Random(sum(map(ord, ''.join(steps)[-200:])))
        base = 0.85 if V21.correct(a, g) else 0.45
        return [0.97] * (len(steps) - 1) + [min(1.0, max(0.0, base + rng.uniform(-0.3, 0.3)))]


def n_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def roomiest_gpu() -> int:
    import torch
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return max(range(len(free)), key=lambda i: free[i]) if free else 0


def build_verifier(args, rows):
    if args.stub:
        return _StubPRM(rows)
    return P.PRMScorer(P.PRM, four_bit=True, device_map={'': roomiest_gpu()})


def build_generators(args, rows):
    if args.stub:
        return V24._StubSolver(rows), V24._StubReader()
    return V24.build(args, rows)


def free(*clients) -> None:
    for c in clients:
        if c is not None and hasattr(c, '_local_model'):
            c._local_model = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# one row
# ---------------------------------------------------------------------------

def _sample(raw: str) -> Dict:
    return {'raw': str(raw), 'answer': V21.parse_answer(raw)}


def needs_gen(rec: Dict, args) -> bool:
    return (len(rec[C]) < args.kc or 'reading' not in rec or len(rec[Q]) < args.kq)


def needs_score(rec: Dict) -> bool:
    if any(s.get('prm') is None for v in (C, Q) for s in rec[v]):
        return True
    return bool(rec.get('reading', {}).get('ok')) and 'reading_prm' not in rec


def generate(solver, reader, rec: Dict, args, deadline: float) -> bool:
    """Draw what the row still lacks, plain first. Returns False if the
    deadline stopped it part-way (the row is then resumed next session)."""
    text = rec['text']
    secs = rec.setdefault('seconds', {})
    if len(rec[C]) < args.kc:
        t0 = time.time()
        raws = V21.sample(solver, SR.cot_prompt(text), args.kc - len(rec[C]),
                          args.temperature, args.max_tokens)
        rec[C] += [_sample(r) for r in raws]
        secs[C] = round(secs.get(C, 0) + time.time() - t0, 1)
    if deadline and time.time() > deadline:
        return False
    if 'reading' not in rec:
        rec['reading'] = V24.read(reader, text)
        secs['reader'] = rec['reading'].get('seconds', 0)
    if len(rec[Q]) < args.kq:
        t0 = time.time()
        raws = V21.sample(solver, SR.asq_prompt(text, rec['reading']), args.kq - len(rec[Q]),
                          args.temperature, args.max_tokens)
        rec[Q] += [_sample(r) for r in raws]
        secs[Q] = round(secs.get(Q, 0) + time.time() - t0, 1)
    return True


def score(scorer, rec: Dict, drift: bool = False) -> None:
    """Every unscored sample, against the ORIGINAL problem text; then the
    Reader's notes (exploratory); on dev, optionally a drift re-score of the
    stored plain samples."""
    text = rec['text']
    t0 = time.time()
    for view in (C, Q):
        for s in rec[view]:
            if s.get('prm') is None:
                s['prm'] = scorer.score(text, P.split_steps(s['raw']))
    rd = rec.get('reading') or {}
    if rd.get('ok') and 'reading_prm' not in rec:
        steps = RA.note_steps(rd, len(SR.sentences(text)))
        rec['reading_prm'] = scorer.score(text, steps) if steps else []
    if drift and 'drift' not in rec:
        d = []
        for s in rec[C]:
            if s.get('stored_prm') and s.get('prm') and len(s['raw']) < TRUNC:
                s['prm_rescored'] = scorer.score(text, P.split_steps(s['raw']))
                if len(s['prm_rescored']) == len(s['prm']):
                    d.append(abs(s['prm_rescored'][-1] - s['prm'][-1]))
        rec['drift'] = d
    secs = rec.setdefault('seconds', {})
    secs['prm'] = round(secs.get('prm', 0) + time.time() - t0, 1)


# ---------------------------------------------------------------------------
# reading the results
# ---------------------------------------------------------------------------

def complete(rec: Dict, dev: bool) -> bool:
    need = ('RA', 'B5') + (('BQ5',) if dev else ())
    return all(RA.available(rec, a) for a in need)


def _pct(x: int, n: int) -> str:
    return f"{x:3d} = {100 * x / n:5.1f}%" if n else '  -'


def block(rows: List[Dict], title: str, dev: bool) -> None:
    n = len(rows)
    if not n:
        return
    arms = [a for a in list(RA.ARMS) + (list(RA.DEV_ARMS) if dev else [])
            if all(RA.available(r, a) for r in rows)]
    spec = {**RA.ARMS, **RA.DEV_ARMS}
    rd = [r.get('reading') or {} for r in rows]
    print('\n' + '=' * 78)
    print(f"  {title}  n={n}  templates={len({RA.template(r) for r in rows})}  "
          f"readings usable {sum(1 for x in rd if x.get('ok'))}/{n}  "
          f"notes/row {sum(x.get('n_notes', 0) for x in rd) / n:.1f}")
    print(f"    {'arm':4s} {'calls':>5s} {'right':>13s} {'oracle':>7s} {'picks read':>10s}")
    for a in arms:
        c, q, how = spec[a]
        k = sum(RA.right(r, a) for r in rows)
        orc = sum(RA.oracle(r, c, q) for r in rows)
        pk = (f"{sum(RA.chosen_view(r, a) == Q for r in rows):3d}/{n}" if how == 'best' and c and q
              else '')
        print(f"    {a:4s} {RA.calls(a):5d} {_pct(k, n):>13s} {orc:7d} {pk:>10s}")
    pairs = [('RA', 'B5'), ('RA', 'B4'), ('RA', 'S5'), ('RA5', 'B5'), ('RA7', 'B5'),
             ('VRA', 'S5'), ('BQ2', 'B5')] + ([('BQ5', 'B5'), ('SQ5', 'S5')] if dev else [])
    print()
    for x, y in pairs:
        if x in arms and y in arms:
            c = RA.paired(rows, x, y)
            print(f"    {x:4s} vs {y:3s}: W={c['w']:2d} L={c['l']:2d} net={c['net']:+3d}  "
                  f"w/o best template {c['loto']:+3d}  sign p={c['p']:.3f}")
    if 'RA' in arms and 'B5' in arms:
        src = RA.gain_source(rows)
        c = RA.paired(rows, 'RA', 'B5')
        print(f"\n    RA's wins over B5: {src.get('reading', 0)} only the reading had the answer, "
              f"{src.get('selection', 0)} it was among the 5 plain samples")
        print(f"      wins   {', '.join(p.replace('gsm-symbolic_', '') for p in c['wins']) or '-'}")
        print(f"      losses {', '.join(p.replace('gsm-symbolic_', '') for p in c['losses']) or '-'}")
        picked = collections.Counter((RA.chosen_view(r, 'RA'), RA.right(r, 'RA')) for r in rows)
        print(f"    RA picked a plain sample on {picked[(C, True)] + picked[(C, False)]} rows "
              f"({picked[(C, True)]} right), a read sample on {picked[(Q, True)] + picked[(Q, False)]} "
              f"({picked[(Q, True)]} right)")
    if dev:
        print("    allocation table, oracle (exploratory; this is what motivated RA):")
        for c in range(0, 6):
            cells = [f"C{c}+Q{q} {sum(RA.oracle(r, c, q) for r in rows):3d}"
                     for q in range(0, 6) if 2 <= c + q <= 7]
            print('      ' + '  '.join(cells))
    exploratory(rows, dev)


def exploratory(rows: List[Dict], dev: bool) -> None:
    n = len(rows)
    print("    EXPLORATORY (not for choosing anything):")
    # 1. cascade: stop when the first two plain samples agree
    right = calls = 0
    for r in rows:
        a = [s.get('answer') for s in r[C][:2]]
        if len(a) == 2 and len(P.L.clusters(a)) == 1 and a[0] is not None:
            right += V21.correct(a[0], r['gold'])
            calls += 2
        else:
            right += RA.right(r, 'RA')
            calls += 5
    print(f"      cascade (stop if C1 = C2, else RA): {right}/{n} at {calls / n:.2f} calls/row")
    # 2. reading vetting: does the verifier's score of the Reader's notes say
    #    when the reading hurts?
    helps, hurts = [], []
    for r in rows:
        rp = r.get('reading_prm')
        if not rp or not r[Q] or not r[C]:
            continue
        qa = sum(V21.correct(s.get('answer'), r['gold']) for s in r[Q]) / len(r[Q])
        ca = sum(V21.correct(s.get('answer'), r['gold']) for s in r[C]) / len(r[C])
        (hurts if qa < ca else helps).append(min(rp))
    au = RA.auroc(helps, hurts)
    print(f"      reading vetting: min verifier score of the notes, AUROC(reading does not hurt "
          f"> hurts) = {'-' if au is None else f'{au:.2f}'}  (n={len(helps)}/{len(hurts)})")
    # 3. dev: stored raws cut at 4000 chars may have lost their last step
    if dev:
        cut = [(r, i) for r in rows for i, s in enumerate(r[Q][:2]) if len(s['raw']) >= TRUNC]
        if cut:
            k0 = sum(RA.right(r, 'RA') for r in rows)
            saved = [(r, i, r[Q][i]['prm']) for r, i in cut]
            for r, i, _ in saved:
                r[Q][i]['prm'] = [-1.0]
            k1 = sum(RA.right(r, 'RA') for r in rows)
            for r, i, p in saved:
                r[Q][i]['prm'] = p
            print(f"      truncation: {len(cut)} of RA's read samples were stored cut; "
                  f"RA with them excluded {k1} (as run {k0})")
        d = [x for r in rows for x in r.get('drift', [])]
        if d:
            print(f"      verifier drift vs the stored v22 scores (last step, {len(d)} samples): "
                  f"max {max(d):.4f}, mean {sum(d) / len(d):.4f}")


def summarise(recs: List[Dict], dev: bool, stub: bool) -> int:
    rows = [r for r in recs if complete(r, dev)]
    main = [r for r in rows if r['set'] == 'main']
    guard = [r for r in rows if r['set'] == 'guard']
    block(main, 'DEV MAIN' if dev else 'MAIN', dev)
    block(guard, 'DEV GUARD' if dev else 'GUARD', dev)
    secs = [r.get('seconds', {}) for r in rows if r.get('seconds')]
    if secs and not dev:
        m = {k: sum(s.get(k, 0) for s in secs) / len(secs) for k in (C, 'reader', Q, 'prm')}
        print(f"\n  seconds/row: plain x{KC} {m[C]:.0f}, reader {m['reader']:.0f}, "
              f"read x{KQ} {m[Q]:.0f}, verifier {m['prm']:.0f}")
    print('\n  pre-registered reading:')
    if len(rows) < len(recs) and not stub:
        print(f"    none: PARTIAL run, {len(rows)}/{len(recs)} rows complete. "
              f"Re-run the identical command to finish.")
        return 0
    if not main:
        print('    none: no complete main rows')
        return 0
    if dev:
        print('    ' + RA.dev_verdict(main))
    else:
        print('    ' + RA.primary_verdict(main))
        c = RA.paired(main, 'RA', 'B5')
        print(f"    SIGNIF   sign p = {c['p']:.4f} -> "
              + ("significant at 0.05" if c['p'] < 0.05 else "not significant by itself"))
        v = RA.paired(main, 'VRA', 'S5')
        print(f"    VOTE     VRA - S5 = {v['net']:+d} -> "
              + ("the reading's diversity vanishes under a vote, as predicted" if abs(v['net']) <= 2
                 else "a vote does collect some of it (prediction failed)"))
    if guard:
        print('    ' + RA.guard_verdict(guard))
    return 0


def pooled(paths: Sequence[str]) -> int:
    rows = []
    for p in paths:
        d = V24._read(p)
        dev = d.get('mode') == 'dev'
        rows += [r for r in d['rows'] if r['set'] == 'main' and complete(r, dev)]
    c = RA.paired(rows, 'RA', 'B5')
    print(f"POOLED MAIN n={len(rows)} ({', '.join(paths)})")
    print(f"  RA {sum(RA.right(r, 'RA') for r in rows)}  B5 {sum(RA.right(r, 'B5') for r in rows)}  "
          f"W={c['w']} L={c['l']} net={c['net']:+d}  sign p={c['p']:.4f}")
    print("  (secondary: the dev rows are fresh for the verifier scores, not for the idea)")
    return 0


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------

def _load_prior(path: str, resume: bool, stub: bool) -> Dict[str, Dict]:
    if os.path.exists(path) and resume and not (stub and path in (OUT, OUT_DEV)):
        prior = {r['key']: r for r in V24._read(path).get('rows', [])}
        print(f"resuming: {len(prior)} rows in {path}")
        return prior
    return {}


def _save(path: str, recs: List[Dict], meta: Dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(dict(meta, rows=recs), fh, ensure_ascii=False)
    os.replace(tmp, path)


def _line(i: int, n: int, rec: Dict, t0: float, ran: int) -> str:
    f = lambda v: f"{v}={sum(V21.correct(s.get('answer'), rec['gold']) for s in rec[v])}/{len(rec[v])}"
    g = lambda a: ('OK ' if RA.right(rec, a) else 'bad') if RA.available(rec, a) else ' - '
    rd = rec.get('reading') or {}
    note = f"notes={rd.get('n_notes', 0)}" + ('' if rd.get('ok', True) else ' READER-FAILED')
    rate = f"{(time.time() - t0) / ran:5.0f}s/row" if ran else ''
    return (f"[{i:3d}/{n}] {rec['set']:5s} {rec['pid'][:24]:24s} {f(C)} {f(Q)}  "
            f"B5={g('B5')} RA={g('RA')}  {note}  {rate}")


def drift_keys(recs: Sequence[Dict], dev: bool) -> set:
    """Dev only: the first DRIFT_ROWS main rows get their stored plain samples
    re-scored, to check this session's verifier against v22's."""
    return {r['key'] for r in [r for r in recs if r['set'] == 'main'][:DRIFT_ROWS]} if dev else set()


def loop(recs, out_path, meta, args, dev, deadline, t0, solver=None, reader=None,
         scorer=None) -> bool:
    """One pass over the rows. Returns True if the deadline stopped it."""
    ran = 0
    dk = drift_keys(recs, dev)
    for i, rec in enumerate(recs, 1):
        want_drift = rec['key'] in dk and 'drift' not in rec
        gen = solver is not None and not dev and needs_gen(rec, args)
        sc = scorer is not None and (needs_score(rec) or want_drift)
        if not gen and not sc:
            continue
        if deadline and time.time() > deadline:
            print("\n  --max-hours reached; re-run the identical command to resume.")
            return True
        ts = time.time()
        if gen:
            try:
                if not generate(solver, reader, rec, args, deadline):
                    _save(out_path, recs, meta)
                    print("\n  --max-hours reached; re-run the identical command to resume.")
                    return True
            except RuntimeError:
                # the Reader guard fired: drop unusable readings and the read
                # samples drawn from them, so a resumed session redraws them
                for r in recs:
                    if not (r.get('reading') or {}).get('ok', True):
                        r.pop('reading', None)
                        r[Q] = []
                        r.pop('reading_prm', None)
                _save(out_path, recs, meta)
                raise
        if scorer is not None:
            score(scorer, rec, drift=want_drift)
        ran += 1
        rec['wall'] = round(rec.get('wall', 0) + time.time() - ts, 1)
        print(_line(i, len(recs), rec, t0, ran), flush=True)
        _save(out_path, recs, meta)
    return False


def run(args, dev: bool) -> int:
    if dev:
        for p in (args.v24, args.v22) + DEV_MANIFESTS:
            if not os.path.exists(p):
                print(f"missing {p}: the dev screen reads the stored v22 and v24 runs")
                return 2
        rows = load_dev(args.v24, args.v22, args.limit)
    else:
        for _, p in MANIFESTS:
            if not os.path.exists(p):
                print(f"missing {p}: run `python pretest_v25.py --build-manifest` first "
                      f"(needs the `datasets` package), then commit the two files.")
                return 2
        rows = load_confirm(args.limit)
    out_path = args.out or (OUT_STUB if args.stub else (OUT_DEV if dev else OUT))
    prior = _load_prior(out_path, args.resume, args.stub)
    recs = [merge(r, prior.get(r['key'])) for r in rows]
    meta = {'mode': 'dev' if dev else 'confirm', 'kc': KC if dev else args.kc,
            'kq': 5 if dev else args.kq, 'temperature': args.temperature,
            'max_tokens': args.max_tokens, 'verifier': P.PRM, 'aggregation': 'last'}
    todo_gen = [] if dev else [r for r in recs if needs_gen(r, args)]
    todo_sc = [r for r in recs if needs_score(r)]
    print(f"rows: {len(recs)} ({'dev screen' if dev else 'confirmation'})  "
          f"to generate: {len(todo_gen)}  to score: {len(todo_sc)}  stage={args.stage}")
    t0 = time.time()
    deadline = t0 + args.max_hours * 3600 if args.max_hours else 0.0
    stopped = False
    want_gen = bool(todo_gen) and args.stage in ('all', 'gen')
    want_sc = args.stage in ('all', 'score')
    if want_gen and want_sc and (args.stub or n_gpus() >= 2):
        scorer = build_verifier(args, recs)
        solver, reader = build_generators(args, todo_gen)
        stopped = loop(recs, out_path, meta, args, dev, deadline, t0, solver, reader, scorer)
    else:
        if want_gen:
            solver, reader = build_generators(args, todo_gen)
            stopped = loop(recs, out_path, meta, args, dev, deadline, t0, solver, reader)
            free(solver, reader)
            solver = reader = None
        dk = drift_keys(recs, dev)
        if want_sc and not stopped and any(needs_score(r) or (r['key'] in dk and 'drift' not in r)
                                           for r in recs):
            scorer = build_verifier(args, recs)
            stopped = loop(recs, out_path, meta, args, dev, deadline, t0, scorer=scorer)
    _save(out_path, recs, meta)
    print(f"\n  saved: {out_path}")
    return summarise(recs, dev, args.stub)


# ---------------------------------------------------------------------------
# fresh confirmation rows
# ---------------------------------------------------------------------------

def build_manifests(seed: int, n_main: int, n_guard: int, force: bool) -> int:
    """100 GSM-Symbolic P2 rows and 20 GSM-Plus distractor rows that no earlier
    version has drawn. Written once; the run only reads them."""
    for _, path in MANIFESTS:
        if os.path.exists(path) and not force:
            print(f"{path} exists; refusing to overwrite pre-registered rows (--force to redo)")
            return 1
    from datasets import load_dataset
    used = set()
    for p in EARLIER_MANIFESTS:
        if os.path.exists(p):
            used |= {r['problem_id'] for r in V24._read(p)['rows']}
    print(f"excluding {len(used)} rows drawn by earlier versions")
    ds = load_dataset('apple/GSM-Symbolic', name='p2', split='test')
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    main = []
    for i in idx:
        pid = f'gsm-symbolic_p2_{i}'
        g, text = V24._gold(ds[i].get('answer')), (ds[i].get('question') or '').strip()
        if pid in used or g is None or len(text) < 20:
            continue
        main.append({'problem_id': pid, 'text': text, 'gold': g, 'dataset': 'gsm-symbolic:p2'})
        if len(main) >= n_main:
            break
    gs = load_dataset('qintongli/GSM-Plus', split='test')
    gidx = [i for i in range(len(gs)) if gs[i]['perturbation_type'] == 'distraction insertion']
    random.Random(seed).shuffle(gidx)
    guard = []
    for i in gidx:
        pid = f'gsm-plus_{i}'
        try:
            g = float(str(gs[i]['answer']).replace(',', '').strip())
        except ValueError:
            continue
        if pid in used:
            continue
        guard.append({'problem_id': pid, 'text': gs[i]['question'].strip(), 'gold': g,
                      'dataset': 'gsm-plus:distraction insertion'})
        if len(guard) >= n_guard:
            break
    for (tag, path), rows, name in ((MANIFESTS[0], main, 'gsm-symbolic:p2'),
                                    (MANIFESTS[1], guard, 'gsm-plus:distraction insertion')):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'dataset': name, 'seed': seed, 'n': len(rows),
                       'note': f'v25 confirmation {tag} rows: fresh seed, disjoint from every '
                               f'row of v20-v24',
                       'rows': rows}, fh, ensure_ascii=False, indent=1)
        print(f"wrote {len(rows)} {tag} rows -> {path}")
    return 0 if len(main) == n_main and len(guard) == n_guard else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev', action='store_true', help='score the stored v24 read samples only')
    ap.add_argument('--v24', default=DEV_V24)
    ap.add_argument('--v22', default=DEV_V22)
    ap.add_argument('--kc', type=int, default=KC)
    ap.add_argument('--kq', type=int, default=KQ)
    ap.add_argument('--stage', choices=('all', 'gen', 'score'), default='all')
    ap.add_argument('--out', default='')
    ap.add_argument('--preset', default='qwen_math7b_mixed')
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--max-hours', type=float, default=0.0)
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--pooled', nargs='*', default=None)
    ap.add_argument('--build-manifest', action='store_true')
    ap.add_argument('--seed', type=int, default=CONFIRM_SEED)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    if args.build_manifest:
        return build_manifests(args.seed, 100, 20, args.force)
    if args.pooled is not None:
        return pooled(args.pooled or [OUT_DEV, OUT])
    if args.summary_only:
        path = args.out or (OUT_DEV if args.dev else OUT)
        return summarise(V24._read(path)['rows'], args.dev, args.stub)
    if not args.dev and (args.kc < 5 or args.kq < 2):
        print("the pre-registered arms need --kc >= 5 and --kq >= 2")
        return 2
    return run(args, dev=args.dev)


if __name__ == '__main__':
    sys.exit(main())
