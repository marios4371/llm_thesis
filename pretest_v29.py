"""
[v29.0] Dev screen: Dispute-to-Question (DQ). When the v22 verifier's top is a
tie between different answers, does one factored question about the problem
text pick the right reading more often than v22's rule?

The rule is in dispute_question.py; the evidence and the literature in
V29_PLAN.md. This screen GENERATES NO SOLUTIONS. It reuses the stored v25 dev
pools (the 100 main + 20 guard rows of the v22 confirmation, seed 45):
  C  5 plain samples per row, with the PRM's step rewards
  Q  5 samples under the Reader's reading (v24 ASQ), same rewards
and adds, only on rows where a pool's PRM top is a dispute:
  a question     the Questioner (Qwen2.5-7B-Instruct, greedy) on each disputed
                 pair of solutions
  5 answers      the question answered from the problem text alone:
                 3 x Qwen2.5-Math-7B-Instruct + 2 x Qwen2.5-7B-Instruct, T=0.8

Offline, on the same stored pools: v22's rule gives B5 78, RA 80, RA5 81,
B10 81 (reproduced by test_v29.py), and a PERFECT answerer would give B5 81,
RA 87, RA5 88, B10 90. B10 has a dispute on 64 of the 100 rows; v22's rule
already wins 48 of the 57 disputes that contain the right answer (84%).

PRE-REGISTERED, frozen 2026-10-03 before any question exists
------------------------------------------------------------
  SCREEN (100 main rows)  DQ-B10 vs B10 (v22's rule on the same 10 samples):
                          net >= +3 rows with W >= 2L -> GO: build the
                          fresh-row confirmation;  net <= 0 -> STOP;  otherwise WEAK
  GUARD (20 GSM-Plus rows) DQ-B10 - B10 >= -1 row -> no harm
  Rule: strict (the challenger needs >= 4 of the 5 answers). The majority
  rule is exploratory.
Reported, not decisive: DQ on the RA, RA5 and B5 pools; DQ-RA vs B5 (the
confirmed v22 system); the dispute-level accuracy against v22's 84%; how
often the Questioner produced a usable question and why not; the vote
margins; every flip, right-to-wrong and wrong-to-right.
These rows motivated the idea (the error analysis looked at them), so the
screen decides whether a confirmation on FRESH rows is worth running; it is
never a thesis claim by itself.

MODES
-----
    python pretest_v29.py --max-hours 3.0      # Kaggle T4 x2 or Colab T4, ~2-3 h
    python pretest_v29.py --summary-only       # re-read, no GPU
    python pretest_v29.py --stub               # offline plumbing, no GPU
Re-running the same command resumes (every pair is saved as it is made).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional, Sequence

import dispute_question as DQ
import pretest_v21 as V21
import pretest_v24 as V24

DEV_V25 = 'results_September/preset_V25_dev.json'
OUT = 'pretest_v29.json'
OUT_STUB = 'pretest_v29_stub.json'
POOL_ORDER = ('B10', 'RA', 'RA5', 'B5')
RULES = ('strict', 'majority')
# Pass 1 makes every pair the pre-registered reading needs (B10, strict) on all
# 120 rows; pass 2 the pairs of the other pools and of the majority rule. A run
# cut short therefore still gives the verdict.
PRIMARY_SCOPE = ((DQ.PRIMARY_POOL, 'strict'),)
FULL_SCOPE = tuple((p, r) for p in POOL_ORDER for r in RULES)


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def load_rows(path: str = DEV_V25) -> List[Dict]:
    rows = []
    for r in V24._read(path)['rows']:
        rows.append({'key': f"v29:{r['pid']}", 'pid': r['pid'], 'set': r['set'], 'gold': r['gold'],
                     'template': r.get('template'), 'text': r['text'], 'C': r['C'], 'Q': r['Q']})
    return rows


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

class _StubReader:
    """Offline Questioner/answerer: the question asks for the final result, so
    each solution's 'value' is its answer; answers are right with p=0.85."""
    provider = model_name = 'stub-reader'

    def __init__(self, gold_of):
        self.gold_of = gold_of
        self.rng = random.Random(0)

    def call_model(self, msgs, temperature=0.0, max_tokens=0, **kw):
        content = msgs[-1]['content']
        if content.startswith('Problem:') and 'Solution 1:' in content:
            m = re.search(r'Solution 1:\n(.*?)\n\nSolution 2:\n(.*?)\n\nAnswer in exactly', content, re.S)
            a1, a2 = (V21.parse_answer(m.group(1)), V21.parse_answer(m.group(2))) if m else (None, None)
            return (f"QUESTION: What is the final result asked for?\nSOLUTION 1: {a1:g}\nSOLUTION 2: {a2:g}"
                    if a1 is not None and a2 is not None else "I cannot tell.")
        return self.answer(content)

    def answer(self, content):
        g = self.gold_of(content)
        a = g if self.rng.random() < 0.85 else g + 1
        return f"We read the problem.\nAnswer: {a:g}"


class _StubSolver(_StubReader):
    provider = model_name = 'stub-solver'


def build(args, rows):
    if args.stub:
        texts = {r['text']: float(r['gold']) for r in rows}

        def gold_of(content):
            for t, g in texts.items():
                if t in content:
                    return g
            return 0.0
        return _StubSolver(gold_of), _StubReader(gold_of)
    return V24.build(args, [])


def answers(client, content: str, k: int) -> List[str]:
    """k sampled answers in one batched generate (pretest_v21.sample: the
    sampler of every earlier version; sequential calls for a stub)."""
    return [str(x) for x in V21.sample(client, content, k, DQ.ANSWER_TEMPERATURE, DQ.A_MAX_TOKENS)]


# ---------------------------------------------------------------------------
# pairs
# ---------------------------------------------------------------------------

_Q = {'n': 0, 'ok': 0, 'shown': False}
Q_ABORT_AFTER = 8


def make_pair(solver, reader, row: Dict, ref_a: str, ref_b: str, args) -> Dict:
    """Question + factored answers for one pair of solutions (refs in sorted
    order are Solution 1 and Solution 2)."""
    r1, r2 = sorted((ref_a, ref_b))
    t0 = time.time()
    try:
        raw = str(reader.call_model(DQ.question_messages(row['text'], DQ.raw_of(row, r1),
                                                         DQ.raw_of(row, r2)),
                                    temperature=0.0, max_tokens=DQ.Q_MAX_TOKENS))
    except Exception as exc:
        raw = f'ERROR {type(exc).__name__}: {exc}'[:300]
    pair = DQ.parse_question(raw, row['text'])
    pair['refs'] = [r1, r2]
    _Q['n'] += 1
    _Q['ok'] += bool(pair['ok'])
    if pair['ok'] and not _Q['shown']:
        _Q['shown'] = True
        print(f"  first question of this session: {pair['question']}  "
              f"[{r1}: {pair['v1']:g} | {r2}: {pair['v2']:g}]", flush=True)
    if _Q['n'] >= Q_ABORT_AFTER and not _Q['ok']:
        raise RuntimeError(f"the Questioner gave nothing usable on its first {_Q['n']} pairs; "
                           f"last output: {raw[:500]!r}")
    if pair['ok']:
        content = DQ.answer_prompt(row['text'], pair['question'])
        raws = answers(solver, content, DQ.K_SOLVER) + answers(reader, content, DQ.K_READER)
        pair['answers'] = [{'raw': str(x)[:1500], 'answer': V21.parse_answer(x)} for x in raws]
        pair['votes'] = [a['answer'] for a in pair['answers']]
    else:
        pair['votes'] = []
    pair['seconds'] = round(time.time() - t0, 1)
    return pair


def needed(row: Dict, rec: Dict, scope=FULL_SCOPE) -> Optional[tuple]:
    """The first missing pair this row needs for the (pool, rule)s in scope, or None."""
    missing = []

    def gp(a, b):
        p = rec['pairs'].get(DQ.pair_key(a, b))
        if p is None:
            missing.append((a, b))
        return p
    for pool, rule in scope:
        DQ.resolve(row, pool, gp, rule)
        if missing:
            return missing[0]
    return None


# ---------------------------------------------------------------------------
# reading the results
# ---------------------------------------------------------------------------

def getter(rec: Dict):
    return lambda a, b: rec.get('pairs', {}).get(DQ.pair_key(a, b))


def dq_pick(row: Dict, rec: Dict, pool: str, rule: str = 'strict'):
    return DQ.resolve(row, pool, getter(rec), rule)


def complete(row: Dict, rec: Dict, scope=FULL_SCOPE) -> bool:
    return needed(row, rec, scope) is None


def paired(rows: Sequence[Dict], recs: Dict, pool: str, rule: str = 'strict',
           against: Optional[str] = None) -> Dict:
    """DQ on `pool` vs v22's rule on `against` (default: the same pool)."""
    w = l = right_dq = right_base = 0
    for r in rows:
        a = V21.correct(dq_pick(r, recs[r['key']], pool, rule)[0], r['gold'])
        b = V21.correct(DQ.v22_pick(r, against or pool), r['gold'])
        right_dq += a
        right_base += b
        w += a and not b
        l += b and not a
    return {'dq': right_dq, 'base': right_base, 'w': w, 'l': l, 'net': w - l, 'n': len(rows)}


def dispute_report(rows: Sequence[Dict], recs: Dict, pool: str = DQ.PRIMARY_POOL) -> List[str]:
    lines = []
    n_disp = with_right = v22_right = dq_right = 0
    flips = collections.Counter()
    for r in rows:
        cands = DQ.candidates(r, pool)
        inc, chs = DQ.contenders(cands)
        if not chs:
            continue
        n_disp += 1
        contenders = [inc] + chs
        if not any(V21.correct(cands[i]['answer'], r['gold']) for i in contenders):
            continue
        with_right += 1
        v = V21.correct(cands[inc]['answer'], r['gold'])
        d = V21.correct(dq_pick(r, recs[r['key']], pool)[0], r['gold'])
        v22_right += v
        dq_right += d
        flips[('right' if v else 'wrong') + ' -> ' + ('right' if d else 'wrong')] += 1
    lines.append(f"{pool}: {n_disp} rows with a dispute; {with_right} of them contain the right answer")
    if with_right:
        lines.append(f"  right after the dispute: v22 {v22_right}/{with_right} "
                     f"({100 * v22_right / with_right:.0f}%), DQ {dq_right}/{with_right} "
                     f"({100 * dq_right / with_right:.0f}%)   " +
                     ', '.join(f'{k}: {v}' for k, v in sorted(flips.items())))
    return lines


def question_report(recs: Dict, rows: Sequence[Dict]) -> List[str]:
    by = {r['key']: r for r in rows}
    pairs = [(by[k], p) for k, rec in recs.items() if k in by for p in rec.get('pairs', {}).values()]
    if not pairs:
        return ['no pairs yet']
    ok = [x for x in pairs if x[1].get('ok')]
    reasons = collections.Counter(p.get('reason') for _, p in pairs if not p.get('ok'))
    lines = [f"pairs: {len(pairs)}, usable questions {len(ok)} ({100 * len(ok) / len(pairs):.0f}%)"
             + (f"; not usable: {dict(reasons)}" if reasons else '')]
    # where exactly one solution of the pair is right: do the answers side with it?
    agree = total = 0
    hist = collections.Counter()
    for row, p in ok:
        r1, r2 = p['refs']
        a1 = row[r1[0]][int(r1[1:])]['answer']
        a2 = row[r2[0]][int(r2[1:])]['answer']
        ok1, ok2 = V21.correct(a1, row['gold']), V21.correct(a2, row['gold'])
        if ok1 == ok2:
            continue
        v_right, v_wrong = (p['v1'], p['v2']) if ok1 else (p['v2'], p['v1'])
        n_r, n_w = DQ.tally(p.get('votes', []), v_wrong, v_right)[::-1]
        hist[(n_r, n_w)] += 1
        total += 1
        agree += n_r > n_w
    if total:
        lines.append(f"pairs with one right solution: {total}; answers side with the right one on "
                     f"{agree} ({100 * agree / total:.0f}%). (right votes, wrong votes): "
                     + ', '.join(f'{k}:{v}' for k, v in sorted(hist.items(), reverse=True)))
    secs = [p.get('seconds', 0) for _, p in pairs]
    lines.append(f"seconds per pair: {sum(secs) / len(secs):.0f}")
    return lines


def screen_verdict(main: Sequence[Dict], guard: Sequence[Dict], recs: Dict) -> List[str]:
    p = paired(main, recs, DQ.PRIMARY_POOL)
    facts = f"DQ-B10 {p['dq']} vs B10 {p['base']}: W={p['w']} L={p['l']} net {p['net']:+d}"
    if p['net'] >= DQ.GO_NET and p['w'] >= DQ.GO_RATIO * p['l']:
        out = [f"SCREEN   GO: {facts} -> build the fresh-row confirmation"]
    elif p['net'] <= 0:
        out = [f"SCREEN   STOP: {facts}"]
    else:
        out = [f"SCREEN   WEAK: {facts}"]
    g = paired(guard, recs, DQ.PRIMARY_POOL)
    out.append(f"GUARD    DQ-B10 - B10 = {g['net']:+d} rows (W={g['w']} L={g['l']}) -> " +
               ('no harm' if g['net'] >= DQ.GUARD_MIN_NET else 'HARM'))
    return out


def summarise(rows: List[Dict], recs: Dict, stub: bool = False) -> int:
    for r in rows:
        recs.setdefault(r['key'], {'pairs': {}})
    main = [r for r in rows if r['set'] == 'main']
    guard = [r for r in rows if r['set'] != 'main']
    done = [r for r in rows if complete(r, recs[r['key']])]
    prim = [r for r in rows if complete(r, recs[r['key']], PRIMARY_SCOPE)]
    print('\n' + '=' * 78)
    print(f"  v29 Dispute-to-Question: {len(prim)}/{len(rows)} rows complete for the pre-registered "
          f"reading (B10, strict), {len(done)}/{len(rows)} for every pool and rule")
    for line in question_report(recs, rows):
        print('  ' + line)
    print(f"\n    {'pool':5s} {'v22':>4s} {'DQ':>4s} {'W':>3s} {'L':>3s}  {'DQ maj':>6s}   (100 main rows)")
    for pool in POOL_ORDER:
        s, m = paired(main, recs, pool), paired(main, recs, pool, 'majority')
        print(f"    {pool:5s} {s['base']:4d} {s['dq']:4d} {s['w']:3d} {s['l']:3d}  {m['dq']:6d}")
    sysc = paired(main, recs, 'RA', against='B5')
    print(f"    DQ-RA vs B5 (the v22 system, 4 vs 5 solver samples): {sysc['dq']} vs {sysc['base']}, "
          f"W={sysc['w']} L={sysc['l']}")
    for line in dispute_report(main, recs):
        print('  ' + line)
    calls = [len(recs[r['key']].get('pairs', {})) for r in main]
    print(f"  extra calls per main row: {6 * sum(calls) / len(main):.1f} on average "
          f"(1 question + 5 short answers per pair; {sum(1 for c in calls if c)} rows needed any)")
    if len(done) < len(rows):
        print("  (the other pools and the majority rule are still partial: their missing rows keep "
              "v22's pick)")
    print('\n  pre-registered reading:')
    if len(prim) < len(rows):
        print(f"    none: PARTIAL run, {len(prim)}/{len(rows)} rows complete for B10. "
              f"Re-run the same command.")
        return 0
    for v in screen_verdict(main, guard, recs):
        print('    ' + v)
    return 0


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _save(path: str, recs: Dict, meta: Dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(dict(meta, rows=recs), fh, ensure_ascii=False)
    os.replace(tmp, path)


def run(args) -> int:
    _Q.update(n=0, ok=0, shown=False)
    rows = load_rows(args.v25)
    out_path = args.out or (OUT_STUB if args.stub else OUT)
    recs: Dict[str, Dict] = {}
    if os.path.exists(out_path) and args.resume:
        recs = V24._read(out_path).get('rows', {})
        print(f"resuming: {len(recs)} rows in {out_path}")
    for r in rows:
        recs.setdefault(r['key'], {'pairs': {}})
    meta = {'pools': list(POOL_ORDER), 'rules': list(RULES), 'k_solver': DQ.K_SOLVER,
            'k_reader': DQ.K_READER, 'override_votes': DQ.OVERRIDE_VOTES,
            'temperature': DQ.ANSWER_TEMPERATURE, 'eps': DQ.EPS}
    todo = [r for r in rows if needed(r, recs[r['key']]) is not None]
    print(f"rows: {len(rows)} ({sum(r['set'] == 'main' for r in rows)} main), "
          f"rows still needing a pair: {len(todo)}", flush=True)
    solver = reader = None
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else 0.0
    t0, made = time.time(), 0
    for label, scope in (('pass 1 (B10, strict)', PRIMARY_SCOPE), ('pass 2 (all pools)', FULL_SCOPE)):
        todo = [r for r in rows if needed(r, recs[r['key']], scope) is not None]
        if not todo:
            continue
        print(f"\n{label}: {len(todo)} rows", flush=True)
        if solver is None:
            solver, reader = build(args, rows)
        for i, row in enumerate(todo, 1):
            rec = recs[row['key']]
            while True:
                nxt = needed(row, rec, scope)
                if nxt is None:
                    break
                if deadline and time.time() > deadline:
                    _save(out_path, recs, meta)
                    print("\n  --max-hours reached; re-run the identical command to resume.")
                    return summarise(rows, recs, args.stub)
                try:
                    pair = make_pair(solver, reader, row, nxt[0], nxt[1], args)
                except RuntimeError:
                    _save(out_path, recs, meta)
                    raise
                rec['pairs'][DQ.pair_key(*nxt)] = pair
                made += 1
                _save(out_path, recs, meta)
            p = dq_pick(row, rec, DQ.PRIMARY_POOL)[0]
            print(f"[{i:3d}/{len(todo)}] {row['set']:5s} {row['pid'][:24]:24s} pairs={len(rec['pairs'])} "
                  f"B10: v22 {'R' if V21.correct(DQ.v22_pick(row, 'B10'), row['gold']) else '-'} "
                  f"DQ {'R' if V21.correct(p, row['gold']) else '-'}  "
                  f"{(time.time() - t0) / max(1, made):5.0f}s/pair", flush=True)
    _save(out_path, recs, meta)
    print(f"\n  saved: {out_path}")
    return summarise(rows, recs, args.stub)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--v25', default=DEV_V25)
    ap.add_argument('--out', default='')
    ap.add_argument('--preset', default='qwen_math7b_mixed')
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--max-hours', type=float, default=0.0)
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()
    if args.summary_only:
        path = args.out or (OUT_STUB if args.stub else OUT)
        return summarise(load_rows(args.v25), V24._read(path).get('rows', {}), args.stub)
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
