"""
[v23.0] Pre-test: Verifier-guided Tabu Restarts (VTR). Does telling a fresh
Solver which STEP the Verifier rejected -- and nothing else -- produce better
restarts than blind restarts?

The method is in tabu_restart.py; the evidence that motivates it is in
v23_diagnosis.py (offline, no GPU). In one line: the residual errors of the
Solver+Verifier system are a mode of the small model's READING, one step
deep; fresh samples re-draw that mode, and every earlier repair either
pointed the Solver at the same sentence (V2P) or handed it its own failed
prefix (RW). VTR hands it one verifier-rejected step as negative evidence,
in a fresh context.

ARMS (a row is TRIGGERED when the leader's lowest step reward < TAU = 0.95;
on an untriggered row every repair arm IS B3, at zero extra cost)
----
  C1        one sampled CoT (the first draft): the "simple prompt"
  S3, S5    self-consistency over 3 / 5 samples
  B3, B5    verifier best-of-3 / best-of-5 (last-step reward), v22's system
  RST       B3 + 2 blind fresh samples on triggered rows (v22's best repair)
  TABU      B3 + 2 fresh samples told "this step was checked and is WRONG"  <- VTR
  ANS       B3 + 2 fresh samples told "the final answer X is WRONG": the same
            frame without the verifier's localisation (attribution control)
  MIX       B3 + 1 RST sample + 1 TABU sample (post hoc, exploratory, free)
  optional (--arms): TABU2 (two sequential restarts, the second also told the
  first restart's rejected step: tabu search with memory), V2P (v22's quote,
  re-run on these rows), RSTN (blind restarts re-drawn in this session)
Every repair arm spends the same 2 samples, and the Verifier always scores
against the ORIGINAL problem, without the note.

PRE-REGISTERED, frozen before any TABU sample exists (2026-09-26)
----------------------------------------------------------------
Dev pass (--dev): the stored drafts of the v21 dev rows (140) and the v22
confirmation rows (120). TAU was fixed on the v21 rows only; nothing about
TABU has been seen on either set. Read on the MAIN rows pooled (n=200):
    TABU - RST >= +3 rows AND TABU wins >= 2x its losses -> SUPPORTED: run
                                                          the fresh confirmation
    TABU <= RST                                          -> REFUTED
    otherwise                                            -> INCONCLUSIVE
    TABU - ANS >= +2 rows -> the verifier's LOCALISATION is what helps
    ANS >= TABU           -> excluding the answer is enough; localisation adds nothing
Guard rows (GSM-Plus distractors, pooled n=60):
    TABU - B3 >= -1 -> no harm
Fresh confirmation (default mode, 100 new P2 rows, seed 46), in addition to
the same TABU vs RST rule:
    TABU - S5 >= +4 rows AND wins >= 2x losses -> the full system beats
                                                  self-consistency@5 while
                                                  drawing fewer samples
Sample-level mechanism numbers (how often a restart is right, repeats the
leader's answer, or re-derives the doubted value) are reported, not decisive.

MODES
-----
    python pretest_v23.py --dev --max-hours 8.0            # stored rows, TABU+ANS, ~6 h
    python pretest_v23.py --dev --arms TABU --max-hours 8  # TABU only, ~3.5 h
    python pretest_v23.py --dev --summary-only             # re-read, no GPU
    python pretest_v23.py --build-manifest                 # needs `datasets`; once, then commit
    python pretest_v23.py --max-hours 8.0                  # fresh confirmation (two sessions)
    python pretest_v23.py --stub --dev --limit 6           # offline plumbing, no GPU
Re-running the identical command RESUMES; a row keeps the restarts it
already has, so adding an arm later only generates that arm.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence

import pretest_v21 as V21
import score_prm_v22 as P
import tabu_restart as T

V21_TRACES = 'pretest_data/v21_traces.json'
V21_PRM = 'pretest_data/v22_dev_prm.json'
V21_MANIFESTS = ('pretest_data/v20_problems.json', 'pretest_data/v21_guard_distraction.json')
V22_RESULTS = 'results_September/preset_v22.json'
V22_MANIFESTS = ('pretest_data/v22_confirm_p2.json', 'pretest_data/v22_confirm_guard.json')
MANIFESTS = (('main', 'pretest_data/v23_confirm_p2.json'),
             ('guard', 'pretest_data/v23_confirm_guard.json'))
OUT = 'pretest_v23.json'
OUT_DEV = 'pretest_v23_dev.json'
OUT_STUB = 'pretest_v23_stub.json'   # a stub run never writes the real files
REPAIR_ARMS = ('TABU', 'ANS', 'TABU2', 'V2P', 'RSTN')
DEFAULT_ARMS = 'TABU,ANS'
CONFIRM_SEED = 46


def _read(path: str) -> Dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def _texts(paths: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        for r in _read(p)['rows']:
            out[r['problem_id']] = r['text']
    return out


# ---------------------------------------------------------------------------
# the dev pool: stored drafts, stored rewards, nothing re-generated
# ---------------------------------------------------------------------------

def load_pool() -> List[Dict]:
    """Every stored row with its 5 samples, their texts and their rewards.

    v21 rows: raws from the v21 traces, rewards from the v22 Phase-1 scoring
    of exactly those raws. v22 rows: the confirmation run stored both."""
    rows: List[Dict] = []
    texts = _texts(V21_MANIFESTS)
    traces = {r['pid']: r for r in _read(V21_TRACES)['rows']}
    for r in _read(V21_PRM)['rows']:
        tr = traces[r['pid']]
        samples = [{'raw': s['raw'], 'answer': q['answer'], 'prm': q['prm']}
                   for s, q in zip(tr['samples'], r['samples'])]
        rows.append({'key': f"v21:{r['pid']}", 'pid': r['pid'], 'source': 'v21',
                     'set': r['set'], 'gold': r['gold'], 'text': texts[r['pid']],
                     'samples': samples})
    texts = _texts(V22_MANIFESTS)
    for r in _read(V22_RESULTS)['rows']:
        rows.append({'key': f"v22:{r['pid']}", 'pid': r['pid'], 'source': 'v22',
                     'set': r['set'], 'gold': r['gold'], 'text': texts[r['pid']],
                     'samples': [{'raw': s['raw'], 'answer': s['answer'], 'prm': s['prm']}
                                 for s in r['samples']]})
    return rows


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def _scored(scorer, text: str, raw: str) -> Dict:
    return {'raw': str(raw)[:4000], 'answer': V21.parse_answer(raw),
            'prm': scorer.score(text, P.split_steps(raw))}


def restarts(client, scorer, text: str, note: str, n: int, temperature: float,
             max_tokens: int) -> List[Dict]:
    """n fresh solves of the problem with `note` appended; each scored by the
    Verifier against the ORIGINAL problem."""
    content = T.restart_prompt(text, note)
    return [_scored(scorer, text, raw)
            for raw in V21.sample(client, content, n, temperature, max_tokens)]


def tabu2(client, scorer, text: str, p: Dict, temperature: float,
          max_tokens: int) -> Dict:
    """Tabu search with memory: restart 1 is told the leader's rejected step;
    if the Verifier also doubts restart 1, restart 2 is told both."""
    first = T.notes_for(p)['TABU']
    s1 = restarts(client, scorer, text, first, 1, temperature, max_tokens)[0]
    tabu = [p['doubted']] if p.get('doubted') else []
    q = T.plan([s1], k=1)
    added = bool(q['triggered'] and q['doubted'] and tabu
                 and T.compact(q['doubted']) != T.compact(tabu[0]))
    note2 = T.tabu_note(tabu + [q['doubted']]) if added else first
    s2 = restarts(client, scorer, text, note2, 1, temperature, max_tokens)[0]
    return {'samples': [s1, s2], 'memory_grew': added}


def generate(client, scorer, text: str, p: Dict, arm: str, temperature: float,
             max_tokens: int) -> Dict:
    notes = T.notes_for(p)
    if arm in ('TABU', 'ANS'):
        note = notes[arm]
        return {'note': note, 'samples': restarts(client, scorer, text, note, T.N_RESTARTS,
                                                  temperature, max_tokens)}
    if arm == 'RSTN':
        return {'note': '', 'samples': restarts(client, scorer, text, '', T.N_RESTARTS,
                                                temperature, max_tokens)}
    if arm == 'TABU2':
        out = tabu2(client, scorer, text, p, temperature, max_tokens)
        out['note'] = notes['TABU']
        return out
    if arm == 'V2P':
        import pretest_v22 as V22
        note = V22.v2p_note(V22.implicated_sentences(text, p.get('doubted') or ''))
        return {'note': note, 'samples': restarts(client, scorer, text, note, T.N_RESTARTS,
                                                  temperature, max_tokens)}
    raise ValueError(arm)


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

def arms_for(rec: Dict, arms: Sequence[str]) -> None:
    """Fill <ARM>_answer / <ARM>_correct from the drafts and the restarts."""
    d = rec['drafts']
    ans = [x['answer'] for x in d]
    wl = [x['last'] for x in d]
    gold = rec['gold']

    def put(name, a):
        rec[f'{name}_answer'] = a
        rec[f'{name}_correct'] = V21.correct(a, gold)

    put('C1', ans[0])
    put('S3', P.plain_vote(ans[:3]))
    put('B3', T.best_of(ans[:3], wl[:3]))
    if len(d) >= 5:
        put('S5', P.plain_vote(ans[:5]))
        put('B5', T.best_of(ans[:5], wl[:5]))
    trig = rec['triggered']
    b3 = rec['B3_answer']
    put('RST', T.best_of(ans[:5], wl[:5]) if trig and len(d) >= 5 else b3)
    extra = rec.get('extra') or {}
    for arm in arms:
        if not trig:
            put(arm, b3)
        elif arm in extra:
            ex = extra[arm]['samples']
            put(arm, T.best_of(ans[:3] + [e['answer'] for e in ex],
                               wl[:3] + [T.last(e['prm']) for e in ex]))
    if 'TABU' in arms and (not trig or 'TABU' in extra):
        if trig and len(d) >= 4:
            t0 = extra['TABU']['samples'][0]
            put('MIX', T.best_of(ans[:4] + [t0['answer']], wl[:4] + [T.last(t0['prm'])]))
        else:
            put('MIX', b3)


def new_record(row: Dict, samples: List[Dict], tau: float,
               raws: Optional[Sequence[str]] = None):
    """The row's record and the tabu keeper's plan. `raws` are the texts the
    rewards were computed on when they are longer than the stored copies."""
    raws = list(raws) if raws is not None else [s.get('raw', '') for s in samples]
    p = T.plan(samples, raws=raws, tau=tau)
    dv = T.doubted_value(p['doubted'], row['text']) if p['doubted'] else None
    rec = {'key': row['key'], 'pid': row['pid'], 'source': row.get('source', 'v23'),
           'set': row['set'], 'gold': row['gold'], 'tau': tau,
           'triggered': p['triggered'], 'leader': p['leader'],
           'leader_answer': p['leader_answer'], 'leader_min': p['leader_min'],
           'cut': p['cut'], 'doubted': (p['doubted'] or '')[:4000],
           'fallback': p['fallback'], 'doubted_value': dv,
           'drafts': [{'answer': s['answer'], 'last': T.last(s['prm']), 'min': T.low(s['prm'])}
                      for s in samples],
           'extra': {}}
    # samples 4-5 are the RST restarts; summarised like every other restart
    rec['rst'] = [{'answer': s['answer'], 'has_doubted_value': T.contains(raw, dv)}
                  for s, raw in zip(samples[3:5], raws[3:5])] if p['triggered'] else []
    return rec, p


def plan_of(rec: Dict) -> Dict:
    """The stored plan of a resumed row, so a restart is always told the step
    that was chosen when the row was first drafted."""
    return {'doubted': rec.get('doubted') or '', 'leader_answer': rec.get('leader_answer'),
            'triggered': rec.get('triggered'), 'cut': rec.get('cut')}


def complete(rec: Dict, arms: Sequence[str]) -> bool:
    return all(f'{a}_answer' in rec for a in arms)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _n(rows, arm):
    return sum(1 for r in rows if r.get(f'{arm}_correct'))


def _paired(rows, x, y):
    w = sum(1 for r in rows if r.get(f'{x}_correct') and not r.get(f'{y}_correct'))
    l = sum(1 for r in rows if r.get(f'{y}_correct') and not r.get(f'{x}_correct'))
    return w, l


def _pct(a, b):
    return f"{a:3d}/{b:<3d} {100*a/b:5.1f}%" if b else f"{a:3d}/0"


def mechanism(rows: List[Dict], arms: Sequence[str]) -> None:
    """Sample-level view of the restarts on triggered rows, split by whether
    the leader was right. More power than row counts; reported, not decisive."""
    trig = [r for r in rows if r['triggered']]
    for label, want in (('leader WRONG', False), ('leader right', True)):
        rr = [r for r in trig if V21.correct(r['leader_answer'], r['gold']) == want]
        if not rr:
            continue
        print(f"      restarts on {len(rr)} triggered rows, {label}:")
        print(f"        {'arm':5s} {'samples':>7s}  {'correct':>15s}  {'= leader answer':>15s}"
              f"  {'re-derive doubted value':>23s}")
        cols = [('RST', lambda r: r.get('rst') or [])]
        for a in arms:
            cols.append((a, lambda r, a=a: [
                {'answer': e['answer'], 'has_doubted_value': e.get('has_doubted_value')}
                for e in ((r.get('extra') or {}).get(a) or {}).get('samples', [])]))
        for name, get in cols:
            ss = [(r, s) for r in rr for s in get(r)]
            if not ss:
                continue
            ok = sum(1 for r, s in ss if V21.correct(s['answer'], r['gold']))
            same = sum(1 for r, s in ss if s['answer'] is not None
                       and V21.correct(s['answer'], r['leader_answer']))
            dv = [s['has_doubted_value'] for r, s in ss if s.get('has_doubted_value') is not None]
            print(f"        {name:5s} {len(ss):7d}  {_pct(ok, len(ss)):>15s}  "
                  f"{_pct(same, len(ss)):>15s}  {_pct(sum(dv), len(dv)):>23s}")


def block(rows: List[Dict], arms: Sequence[str], title: str) -> None:
    n = len(rows)
    if not n:
        return
    trig = [r for r in rows if r['triggered']]
    print('\n' + '=' * 72)
    print(f"  {title}  n={n}   triggered={len(trig)} "
          f"(leader wrong on {sum(1 for r in trig if not r['B3_correct'])})")
    show = ['C1', 'S3', 'S5', 'B3', 'B5', 'RST'] + list(arms) + (['MIX'] if 'TABU' in arms else [])
    for a in show:
        if all(f'{a}_correct' in r for r in rows):
            print(f"    {a:5s} {_n(rows, a):3d} = {100*_n(rows, a)/n:5.1f}%")
    cost = 3 + T.N_RESTARTS * len(trig) / n
    print(f"    samples per row: C1 1, S3/B3 3, S5/B5 5, every repair arm {cost:.2f}")
    print()
    pairs = [('B3', 'S3'), ('B5', 'S5'), ('RST', 'B3')]
    pairs += [(a, 'RST') for a in arms] + [(a, 'B3') for a in arms]
    if 'TABU' in arms:
        pairs += [('TABU', 'ANS'), ('TABU', 'S5'), ('TABU', 'B5'), ('TABU', 'C1')]
    for x, y in pairs:
        if all(f'{x}_correct' in r and f'{y}_correct' in r for r in rows):
            w, l = _paired(rows, x, y)
            print(f"    {x:5s} vs {y:4s}: W={w:2d} L={l:2d} net={w-l:+d}")
    mechanism(rows, arms)


def verdicts(main: List[Dict], guard: List[Dict], arms: Sequence[str], fresh: bool) -> List[str]:
    out = []
    if 'TABU' not in arms:
        return ['no verdict: the TABU arm was not requested']
    if not main:
        return ['no verdict: no complete main-set rows']
    w, l = _paired(main, 'TABU', 'RST')
    net = w - l
    if net >= 3 and w >= 2 * l:
        out.append(f"PRIMARY  SUPPORTED: TABU beats blind restarts (net {net:+d}, W={w} L={l})")
    elif net <= 0:
        out.append(f"PRIMARY  REFUTED: TABU <= RST (net {net:+d}, W={w} L={l})")
    else:
        out.append(f"PRIMARY  INCONCLUSIVE: TABU > RST but short of the bar (net {net:+d}, W={w} L={l})")
    if 'ANS' in arms:
        w, l = _paired(main, 'TABU', 'ANS')
        if w - l >= 2:
            out.append(f"ATTRIB   the verifier's LOCALISATION helps (TABU-ANS net {w-l:+d})")
        elif w - l <= 0:
            out.append(f"ATTRIB   answer exclusion is enough, localisation adds nothing "
                       f"(TABU-ANS net {w-l:+d})")
        else:
            out.append(f"ATTRIB   undecided (TABU-ANS net {w-l:+d})")
    if guard:
        w, l = _paired(guard, 'TABU', 'B3')
        out.append(f"GUARD    TABU vs B3 net {w-l:+d} -> " + ("no harm" if w - l >= -1 else "HARM"))
    if fresh and all('S5_correct' in r for r in main):
        w, l = _paired(main, 'TABU', 'S5')
        ok = w - l >= 4 and w >= 2 * l
        out.append(f"SYSTEM   TABU vs S5 net {w-l:+d} (W={w} L={l}) -> "
                   + ("the system beats SC@5 with fewer samples" if ok else "short of the bar"))
    return out


def summarise(rows: List[Dict], arms: Sequence[str], dev: bool, stub: bool,
              expected: int = 0, tau: float = T.TAU) -> int:
    rows = [r for r in rows if complete(r, arms)]
    if dev:
        for src in ('v21', 'v22'):
            for tag in ('main', 'guard'):
                block([r for r in rows if r['source'] == src and r['set'] == tag], arms,
                      f"{src.upper()} {tag.upper()}" + ("  (TAU chosen here)" if src == 'v21' else
                                                        "  (held out for TAU)"))
    main = [r for r in rows if r['set'] == 'main']
    guard = [r for r in rows if r['set'] == 'guard']
    block(main, arms, 'POOLED MAIN' if dev else 'MAIN')
    block(guard, arms, 'POOLED GUARD' if dev else 'GUARD')
    print('\n  pre-registered reading:')
    if tau != T.TAU:
        print(f"    none: --tau {tau} is not the pre-registered TAU={T.TAU} (exploratory run)")
        return 0
    if expected and len(rows) < expected and not stub:
        print(f"    none: PARTIAL run, {len(rows)}/{expected} rows complete. "
              f"Re-run the identical command to finish.")
        return 0
    for v in verdicts(main, guard, arms, fresh=not dev):
        print('    ' + v)
    return 0


# ---------------------------------------------------------------------------
# offline stub
# ---------------------------------------------------------------------------

class _StubClient:
    """Offline stand-in. Right with a probability that depends only on which
    note the prompt carries, so every code path runs and the arms differ."""
    provider = model_name = 'stub'

    def __init__(self, rows: List[Dict], seed: int = 0):
        self.rows = rows
        self.rng = random.Random(seed)
        self.calls = 0

    def complete(self, content: str, temperature: float) -> str:
        self.calls += 1
        r = next((x for x in self.rows if x['text'] in content), self.rows[0])
        g = float(r['gold'])
        if T.TABU_MARKER in content or T.TABU_MARKER_MULTI in content:
            p = 0.55
        elif T.ANS_MARKER in content:
            p = 0.4
        else:
            p = 0.3
        a = g if self.rng.random() < p else g + self.rng.choice([1, 2, 3])
        return (f"First we read the problem.\n\nThen we compute \\[ {T.fmt(g)} + 0 = {T.fmt(a)} \\]"
                f"\n\nAnswer: {T.fmt(a)}")

    def call_model(self, msgs, temperature=0.0, max_tokens=0, **kw):
        return self.complete(msgs[-1]['content'], temperature)


def build(args, rows):
    if args.stub:
        return _StubClient(rows), P._StubScorer({})
    import torch
    two = torch.cuda.device_count() > 1
    client = V21.build_client(args.preset, None)
    scorer = P.PRMScorer(P.PRM, four_bit=True, device_map={'': 1 if two else 0})
    return client, scorer


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------

def _load_prior(path: str, resume: bool, stub: bool) -> Dict[str, Dict]:
    # a stub run never resumes from, or into, the real result files
    if os.path.exists(path) and resume and not (stub and path in (OUT, OUT_DEV)):
        prior = {r['key']: r for r in _read(path).get('rows', [])}
        print(f"resuming: {len(prior)} rows already in {path}")
        return prior
    return {}


def _save(path: str, rows: List[Dict], extra: Dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(dict(extra, tau=extra.get('tau', T.TAU), rows=rows), fh, ensure_ascii=False)
    os.replace(tmp, path)


def _fill(client, scorer, rec: Dict, p: Dict, text: str, arms: Sequence[str], args) -> int:
    """Generate every requested arm this triggered row does not have yet."""
    made = 0
    for arm in arms:
        if arm in rec['extra']:
            continue
        g = generate(client, scorer, text, p, arm, args.temperature, args.max_tokens)
        for s in g['samples']:
            s['has_doubted_value'] = T.contains(s['raw'], rec['doubted_value'])
        rec['extra'][arm] = g
        made += 1
    return made


def _line(i, n, rec, arms, t0, ran):
    f = lambda a: 'OK ' if rec.get(f'{a}_correct') else 'BAD'
    shown = ' '.join(f"{a}={f(a)}" for a in ['B3', 'RST'] + list(arms))
    rate = f"{(time.time()-t0)/ran:5.0f}s/row" if ran else ''
    trig = f"cut={rec['cut']}" if rec['triggered'] else 'untriggered'
    return f"[{i:3d}/{n}] {rec['source']} {rec['set']:5s} {rec['pid'][:22]:22s} {shown}  {trig}  {rate}"


def run_dev(args, arms: Sequence[str]) -> int:
    pool = load_pool()
    if args.limit:
        # a small pool that still has every source x set, triggered and not
        keep = []
        for src in ('v21', 'v22'):
            for tag in ('main', 'guard'):
                grp = [r for r in pool if r['source'] == src and r['set'] == tag]
                trig = [r for r in grp if T.plan(r['samples'], tau=args.tau)['triggered']]
                keep += trig[:max(1, args.limit // 4)]
                keep += [r for r in grp if r not in trig][:1]
        pool = keep
    out_path = args.out or (OUT_STUB if args.stub else OUT_DEV)
    prior = _load_prior(out_path, args.resume, args.stub)
    recs = []
    for row in pool:
        rec, p = new_record(row, row['samples'], args.tau)
        old = prior.get(row['key'])
        if old and old.get('tau') == args.tau and old.get('doubted') == rec['doubted']:
            rec['extra'] = old.get('extra') or {}
        recs.append((row, rec, p))
    need = [x for x in recs if x[1]['triggered'] and any(a not in x[1]['extra'] for a in arms)]
    print(f"dev pool: {len(pool)} rows, triggered at TAU={args.tau}: "
          f"{sum(1 for x in recs if x[1]['triggered'])}; to generate: {len(need)} rows x "
          f"{len(arms)} arm(s) x {T.N_RESTARTS} samples")
    client = scorer = None
    if need:
        client, scorer = build(args, [x[0] for x in need])
    t0, ran, stopped = time.time(), 0, False
    for i, (row, rec, p) in enumerate(recs, 1):
        if rec['triggered'] and not stopped:
            if args.max_hours and time.time() - t0 > args.max_hours * 3600:
                print("\n  --max-hours reached; re-run the identical command to resume.")
                stopped = True
            elif _fill(client, scorer, rec, p, row['text'], arms, args):
                ran += 1
                arms_for(rec, arms)
                print(_line(i, len(recs), rec, arms, t0, ran), flush=True)
                _save(out_path, [x[1] for x in recs],
                      {'mode': 'dev', 'arms': list(arms), 'tau': args.tau})
        arms_for(rec, arms)
    _save(out_path, [x[1] for x in recs], {'mode': 'dev', 'arms': list(arms), 'tau': args.tau})
    print(f"\n  saved: {out_path}")
    return summarise([x[1] for x in recs], arms, dev=True, stub=args.stub,
                     expected=len(recs), tau=args.tau)


def run_confirm(args, arms: Sequence[str]) -> int:
    allrows = []
    for tag, path in MANIFESTS:
        if not os.path.exists(path):
            print(f"missing {path}: run `python pretest_v23.py --build-manifest` first "
                  f"(needs the `datasets` package), then commit the two files.")
            return 2
        rows = _read(path)['rows']
        allrows += [dict(r, set=tag, pid=r['problem_id'], key=f"v23:{r['problem_id']}",
                         source='v23')
                    for r in (rows[:args.limit] if args.limit else rows)]
    print(f"rows: {len(allrows)}  k=5 (3 drafts + 2 RST)  T={args.temperature}  "
          f"TAU={args.tau}  arms={','.join(arms)}")
    client, scorer = build(args, allrows)
    out_path = args.out or (OUT_STUB if args.stub else OUT)
    prior = _load_prior(out_path, args.resume, args.stub)
    out, t0, ran, left = [], time.time(), 0, 0
    for i, row in enumerate(allrows, 1):
        old = prior.get(row['key'])
        rec = old if old and old.get('tau') == args.tau and old.get('samples') else None
        done = rec is not None and (not rec['triggered'] or all(a in rec['extra'] for a in arms))
        if not done and args.max_hours and time.time() - t0 > args.max_hours * 3600:
            if not left:
                print("\n  --max-hours reached; re-run the identical command to resume.")
            left += 1
            if rec is not None:          # keep what an earlier session drew
                arms_for(rec, arms)
                out.append(rec)
            continue
        if done:
            arms_for(rec, arms)
            out.append(rec)
            continue
        ts = time.time()
        if rec is None:
            raws = V21.sample(client, V21.COT_PROMPT.format(problem=row['text']), 5,
                              args.temperature, args.max_tokens)
            samples = [_scored(scorer, row['text'], raw) for raw in raws]
            rec, _ = new_record(row, samples, args.tau, raws=raws)
            rec['samples'] = samples
        p = plan_of(rec)
        if rec['triggered']:
            _fill(client, scorer, rec, p, row['text'], arms, args)
        arms_for(rec, arms)
        rec['seconds'] = round(rec.get('seconds', 0) + time.time() - ts, 1)
        out.append(rec)
        ran += 1
        print(_line(i, len(allrows), rec, arms, t0, ran), flush=True)
        _save(out_path, out, {'mode': 'confirm', 'arms': list(arms), 'tau': args.tau,
                              'temperature': args.temperature})
    _save(out_path, out, {'mode': 'confirm', 'arms': list(arms), 'tau': args.tau,
                          'temperature': args.temperature})
    if left:
        print(f"\n  PARTIAL: {left} row(s) still to do. No verdict is read off a partial run.")
    return summarise(out, arms, dev=False, stub=args.stub, expected=len(allrows), tau=args.tau)


# ---------------------------------------------------------------------------
# fresh confirmation rows
# ---------------------------------------------------------------------------

def _gold(raw) -> Optional[float]:
    s = str(raw)
    if '####' in s:
        s = s.split('####')[-1]
    s = s.strip().replace(',', '').replace('$', '')
    try:
        return float(s.split()[0]) if s.split() else None
    except ValueError:
        return None


def build_manifests(seed: int, n_main: int, n_guard: int, force: bool) -> int:
    """100 GSM-Symbolic P2 rows and 20 GSM-Plus distractor rows that no
    earlier version has seen. Written once; the run only reads them."""
    for _, path in MANIFESTS:
        if os.path.exists(path) and not force:
            print(f"{path} exists; refusing to overwrite pre-registered rows (--force to redo)")
            return 1
    from datasets import load_dataset
    used = set()
    for p in V21_MANIFESTS + V22_MANIFESTS:
        used |= {r['problem_id'] for r in _read(p)['rows']}
    ds = load_dataset('apple/GSM-Symbolic', name='p2', split='test')
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    main = []
    for i in idx:
        pid = f'gsm-symbolic_p2_{i}'
        g, text = _gold(ds[i].get('answer')), (ds[i].get('question') or '').strip()
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
    for (tag, path), rows, ds_name in ((MANIFESTS[0], main, 'gsm-symbolic:p2'),
                                        (MANIFESTS[1], guard, 'gsm-plus:distraction insertion')):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'dataset': ds_name, 'seed': seed, 'n': len(rows),
                       'note': f'v23 confirmation {tag} rows: fresh seed, disjoint from every '
                               f'v20/v21/v22 row',
                       'rows': rows}, fh, ensure_ascii=False, indent=1)
        print(f"wrote {len(rows)} {tag} rows -> {path}")
    return 0 if len(main) == n_main and len(guard) == n_guard else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev', action='store_true',
                    help='TABU on the stored v21/v22 rows (no new drafts)')
    ap.add_argument('--arms', default=DEFAULT_ARMS,
                    help=f'comma list from {",".join(REPAIR_ARMS)}')
    ap.add_argument('--tau', type=float, default=T.TAU)
    ap.add_argument('--out', default='')
    ap.add_argument('--preset', default='qwen_math7b_mixed')
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--max-hours', type=float, default=0.0)
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--build-manifest', action='store_true')
    ap.add_argument('--seed', type=int, default=CONFIRM_SEED)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    arms = [a.strip().upper() for a in args.arms.split(',') if a.strip()]
    bad = [a for a in arms if a not in REPAIR_ARMS]
    if bad or not arms:
        print(f"unknown arm(s) {bad}; choose from {REPAIR_ARMS}")
        return 2
    if args.build_manifest:
        return build_manifests(args.seed, 100, 20, args.force)
    if args.summary_only:
        data = _read(args.out or (OUT_DEV if args.dev else OUT))
        return summarise(data['rows'], arms, dev=args.dev, stub=args.stub,
                         tau=data.get('tau', T.TAU))
    return run_dev(args, arms) if args.dev else run_confirm(args, arms)


if __name__ == '__main__':
    sys.exit(main())
