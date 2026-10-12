"""
[v34.0] Generalization of verifier-routed sampling to a second benchmark:
MATH-500 (problems with a numeric answer).

WHY
---
Verifier-routed sampling (v22's ADAPT rule, unchanged since v22) is the
thesis's replicated positive on GSM-Symbolic P2: over 200 fresh rows from two
independent sets it scores 158 against SC@5's 142 (+16, W20 L4, p = 0.0015)
and B5's 159, at 3.33 calls per problem instead of 5. It has only ever been
measured on grade-school word problems. MATH-500 is harder and different in
kind (algebra, number theory, geometry, counting, precalculus; levels 1-5),
and it is closer to what the verifier (Qwen2.5-Math-PRM-7B) was trained on.

THE RULE (frozen since v22; gated_reading.ARMS['ADAPT5'])
---------------------------------------------------------
Draw C1..C3 and score them; if the min step reward of the best of them (by
last-step reward) is below 0.85, also draw C4, C5; answer with the best
last-step reward among the samples drawn. 3 or 5 calls.

ROWS AND SAMPLES
----------------
pretest_data/v34_math500.json: 100 of the 318 MATH-500 test problems whose
answer is a plain number, drawn with seed 34 (--build-manifest, once). Per row:
5 samples from Qwen2.5-Math-7B-Instruct with v22's sampler (T 0.8, top-p 0.95,
top-k 50) and v22's CoT prompt, max 2048 new tokens (MATH solutions are longer
than GSM's; 1024 truncates them), every sample scored by v22's PRM. Every arm
is read off the same samples; ADAPT5 is charged C4, C5 only where it gates.

PRE-REGISTERED (V34_PLAN.md; frozen 2026-10-12 before any sample exists)
------------------------------------------------------------------------
  PRIMARY 1  ADAPT5 vs SC@5: net >= +5 with W >= 2L and sign p < 0.05
             -> SUPPORTED; net <= 0 -> REFUTED; otherwise INCONCLUSIVE
  PRIMARY 2  ADAPT5 vs B5: net >= 0 at <= 4.0 calls/row -> NON-INFERIOR
             (net >= +3 with W >= 2L -> BEATS); otherwise NOT MET
Reported, not decisive: S3, B3, accuracy by MATH level, the gate rate.

MODES
-----
    python pretest_v34.py --build-manifest      # once (needs `datasets`); then commit
    python pretest_v34.py --max-hours 8.0       # Kaggle T4 x2; re-run to resume
    python pretest_v34.py --summary-only        # re-read, no GPU
    python pretest_v34.py --stub                # offline plumbing
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional

import gated_reading as G
import pretest_v21 as V21
import pretest_v24 as V24
import score_prm_v22 as P

MANIFEST = 'pretest_data/v34_math500.json'
OUT = 'pretest_v34.json'
OUT_STUB = 'pretest_v34_stub.json'
SEED = 34
N_ROWS = 100
K = 5
TEMPERATURE = 0.8
MAX_TOKENS = 2048
ARMS = ('ADAPT5', 'B5', 'S5', 'B3', 'S3')

# pre-registered bars (V34_PLAN.md, frozen before any sample exists)
SUP_MIN_NET, SUP_RATIO, SUP_ALPHA = 5, 2, 0.05
NI_MIN_NET, NI_MAX_CALLS, BEAT_MIN_NET = 0, 4.0, 3

_NUM = r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?'


# ---------------------------------------------------------------------------
# answers
# ---------------------------------------------------------------------------

def _boxed(s: str) -> List[str]:
    """Contents of every \\boxed{...}, braces balanced."""
    out, i = [], 0
    while True:
        j = s.find('\\boxed{', i)
        if j < 0:
            return out
        k, depth = j + 7, 1
        while k < len(s) and depth:
            depth += {'{': 1, '}': -1}.get(s[k], 0)
            k += 1
        out.append(s[j + 7:k - 1])
        i = k


def _to_num(t: str) -> Optional[float]:
    t = t.replace('\\!', '').replace('$', '').replace('\\,', '').strip()
    t = re.sub(r'\\text\{[^}]*\}|\^\{?\\circ\}?|\\%|%', '', t).strip()
    m = re.fullmatch(_NUM, t.replace(' ', ''))
    if m:
        return float(m.group(0).replace(',', ''))
    m = re.fullmatch(r'\\[dt]?frac\{(' + _NUM + r')\}\{(' + _NUM + r')\}', t.replace(' ', ''))
    if m:
        d = float(m.group(2).replace(',', ''))
        return float(m.group(1).replace(',', '')) / d if d else None
    return None


def parse_answer(raw: str) -> Optional[float]:
    """The LAST \\boxed{} that holds a number, else the last 'Answer:' line,
    else v21's parse. Read from the end: MATH solutions often run past the
    4000 characters v21's parser looks at."""
    s = str(raw)
    for b in reversed(_boxed(s)):
        v = _to_num(b)
        if v is not None:
            return v
    ms = re.findall(r'answer\s*(?:is)?\s*[:=]?\s*\$?(' + _NUM + r')', s, re.IGNORECASE)
    if ms:
        return float(ms[-1].replace(',', ''))
    return V21.parse_answer(s[-4000:])


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def build_manifest(force: bool = False) -> int:
    if os.path.exists(MANIFEST) and not force:
        print(f"{MANIFEST} exists; refusing to overwrite pre-registered rows (--force to redo)")
        return 1
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    pool = []
    for i, x in enumerate(ds):
        g = _to_num(x['answer'].strip())
        if g is not None and re.fullmatch(_NUM, x['answer'].strip().replace(' ', '')):
            pool.append({'problem_id': f"math500_{i}", 'text': x['problem'].strip(), 'gold': g,
                         'level': x['level'], 'subject': x['subject'], 'unique_id': x['unique_id']})
    random.Random(SEED).shuffle(pool)
    rows = pool[:N_ROWS]
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, 'w', encoding='utf-8') as fh:
        json.dump({'dataset': 'HuggingFaceH4/MATH-500 (numeric answers)', 'seed': SEED, 'n': len(rows),
                   'pool': len(pool), 'rows': rows}, fh, ensure_ascii=False, indent=1)
    print(f"wrote {len(rows)} of {len(pool)} numeric-answer rows -> {MANIFEST}")
    return 0


def load_rows() -> List[Dict]:
    return [{'key': r['problem_id'], 'pid': r['problem_id'], 'set': 'main', 'gold': r['gold'],
             'text': r['text'], 'level': r['level'], 'subject': r['subject'], 'template': r['subject']}
            for r in V24._read(MANIFEST)['rows']]


def complete(rec: Dict) -> bool:
    c = rec.get('C') or []
    return len(c) >= K and all(s.get('prm') is not None for s in c[:K])


# ---------------------------------------------------------------------------
# agents
# ---------------------------------------------------------------------------

class _StubSolver:
    provider = 'stub'

    def __init__(self, rows):
        self.gold = {r['text']: r['gold'] for r in rows}
        self.rng = random.Random(0)

    def call_model(self, msgs, temperature=0.0, max_tokens=0, **kw):
        content = msgs[-1]['content']
        g = next((v for t, v in self.gold.items() if t[:60] in content), 0.0)
        a = g if self.rng.random() < 0.6 else g + self.rng.choice([1, 2, 3])
        return f"Step one.\n\nStep two.\n\nSo the answer is $\\boxed{{{a:g}}}$."


class _StubPRM:
    def __init__(self, rows):
        self.gold = {r['text']: r['gold'] for r in rows}

    def score(self, problem, steps):
        a = parse_answer('\n\n'.join(steps))
        rng = random.Random(sum(map(ord, ''.join(steps)[-80:])))
        good = V21.correct(a, self.gold.get(problem))
        return [0.95 if good or rng.random() < 0.5 else 0.6] * (len(steps) - 1) + [0.999 if good else 0.7]


def _n_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# reading the results
# ---------------------------------------------------------------------------

def verdicts(main: List[Dict]) -> List[str]:
    out = []
    s = G.paired(main, 'ADAPT5', 'S5')
    c = G.calls(main, 'ADAPT5')
    facts = (f"ADAPT5 {s['x']} vs SC@5 {s['y']} of {s['n']}: W={s['w']} L={s['l']} net={s['net']:+d}, "
             f"sign p={s['p']:.4f}; calls/row {c:.2f} vs 5")
    if s['net'] >= SUP_MIN_NET and s['w'] >= SUP_RATIO * s['l'] and s['p'] < SUP_ALPHA:
        out.append(f"PRIMARY 1  SUPPORTED: verifier-routed sampling beats self-consistency on MATH-500 ({facts})")
    elif s['net'] <= 0:
        out.append(f"PRIMARY 1  REFUTED: {facts}")
    else:
        out.append(f"PRIMARY 1  INCONCLUSIVE: {facts}")
    b = G.paired(main, 'ADAPT5', 'B5')
    facts = (f"ADAPT5 {b['x']} vs B5 {b['y']}: W={b['w']} L={b['l']} net={b['net']:+d}, sign p={b['p']:.3f}; "
             f"calls/row {c:.2f} vs 5 ({100 * c / 5:.0f}% of B5's compute)")
    if b['net'] >= BEAT_MIN_NET and b['w'] >= SUP_RATIO * b['l'] and c <= NI_MAX_CALLS:
        out.append(f"PRIMARY 2  BEATS the PRM best-of-5 with less compute ({facts})")
    elif b['net'] >= NI_MIN_NET and c <= NI_MAX_CALLS:
        out.append(f"PRIMARY 2  NON-INFERIOR to the PRM best-of-5 with less compute ({facts})")
    else:
        out.append(f"PRIMARY 2  NOT MET: {facts}")
    return out


def summarise(rows: List[Dict], recs: Dict) -> int:
    for r in rows:
        r['C'] = (recs.get(r['key']) or {}).get('C') or []
    done = [r for r in rows if complete(r)]
    print('\n' + '=' * 78)
    print(f"  v34 MATH-500 (numeric answers): {len(done)}/{len(rows)} rows complete")
    if done:
        print(f"    {'arm':7s} {'right':>6s} {'calls/row':>10s}")
        for arm in ARMS:
            print(f"    {arm:7s} {sum(G.right(r, arm) for r in done):6d} {G.calls(done, arm):10.2f}")
        for x, y in (('ADAPT5', 'S5'), ('ADAPT5', 'B5'), ('ADAPT5', 'B3'), ('B5', 'S5')):
            c = G.paired(done, x, y)
            print(f"    {x:6s} vs {y:3s}: W={c['w']:2d} L={c['l']:2d} net={c['net']:+3d}  sign p={c['p']:.4f}")
        g = [r for r in done if G.gated(r)]
        print(f"    gated rows {len(g)}/{len(done)}")
        print("    by level:  " + "  ".join(
            f"L{lv} n={sum(r['level'] == lv for r in done)}: SC@5 {sum(G.right(r, 'S5') for r in done if r['level'] == lv)}"
            f" B5 {sum(G.right(r, 'B5') for r in done if r['level'] == lv)}"
            f" ADAPT5 {sum(G.right(r, 'ADAPT5') for r in done if r['level'] == lv)}"
            for lv in sorted({r['level'] for r in done})))
        k = [s for r in done for s in r['C'][:K]]
        print(f"    per-sample accuracy {100 * sum(V21.correct(s.get('answer'), r['gold']) for r in done for s in r['C'][:K]) / len(k):.1f}%"
              f", unparsed {sum(s.get('answer') is None for s in k)}/{len(k)}")
    print('\n  pre-registered reading:')
    if len(done) < len(rows):
        print(f"    none: PARTIAL run, {len(done)}/{len(rows)} rows complete. Re-run the identical command.")
        return 0
    for line in verdicts(done):
        print('    ' + line)
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
    if not os.path.exists(MANIFEST):
        print(f"missing {MANIFEST}: run `python pretest_v34.py --build-manifest` first")
        return 2
    rows = load_rows()
    if args.limit:
        rows = rows[:args.limit]
    out = args.out or (OUT_STUB if args.stub else OUT)
    recs: Dict[str, Dict] = {}
    if os.path.exists(out) and args.resume:
        recs = V24._read(out).get('rows', {})
        print(f"resuming: {len(recs)} rows in {out}")
    meta = {'manifest': MANIFEST, 'k': K, 'temperature': TEMPERATURE, 'max_tokens': MAX_TOKENS,
            'prompt': V21.COT_PROMPT, 'verifier': P.PRM, 'gate_min': G.GATE_MIN}
    todo = [r for r in rows if not complete(recs.get(r['key'], {}))]
    print(f"rows: {len(rows)}  to do: {len(todo)}", flush=True)
    if not todo:
        return summarise(rows, recs)
    if not args.stub:
        try:
            import pretest_v32 as T32
            T32.scrub_git_credentials(os.path.dirname(os.path.abspath(__file__)))
        except Exception:
            pass
    solver = _StubSolver(rows) if args.stub else V21.build_client(args.preset, None)
    scorer = (_StubPRM(rows) if args.stub
              else P.PRMScorer(P.PRM, four_bit=True, device_map={'': 1 if _n_gpus() > 1 else 0}))
    t0 = time.time()
    deadline = t0 + args.max_hours * 3600 if args.max_hours else 0.0
    for i, row in enumerate(todo, 1):
        if deadline and time.time() >= deadline:
            _save(out, recs, meta)
            print("\n  --max-hours reached; re-run the identical command to resume.")
            return summarise(rows, recs)
        rec = recs.setdefault(row['key'], {})
        c = rec.setdefault('C', [])
        ts = time.time()
        if len(c) < K:
            raws = V21.sample(solver, V21.COT_PROMPT.format(problem=row['text']), K - len(c),
                              TEMPERATURE, MAX_TOKENS)
            c += [{'raw': str(x), 'answer': parse_answer(x)} for x in raws]
        for s in c:
            if s.get('prm') is None:
                s['prm'] = scorer.score(row['text'], P.split_steps(s['raw']))
        rec['seconds'] = round(time.time() - ts, 1)
        _save(out, recs, meta)
        right = sum(V21.correct(s['answer'], row['gold']) for s in c)
        print(f"[{i:3d}/{len(todo)}] {row['pid']:12s} L{row['level']} {row['subject'][:14]:14s} right {right}/{K}  "
              f"{rec['seconds']:5.0f}s  {(time.time() - t0) / i:5.0f}s/row", flush=True)
    _save(out, recs, meta)
    print(f"\n  saved: {out}")
    return summarise(rows, recs)


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
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()
    if args.build_manifest:
        return build_manifest(args.force)
    if args.summary_only:
        path = args.out or (OUT_STUB if args.stub else OUT)
        rows = load_rows()
        if args.limit:
            rows = rows[:args.limit]
        return summarise(rows, V24._read(path).get('rows', {}))
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
