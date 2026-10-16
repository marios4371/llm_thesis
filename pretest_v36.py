"""
[v36.0] The large run of Trace-level SIV where it can act: arithmetic-heavy word
problems with large numbers, decimals and fractions.

WHY THESE ROWS
--------------
v35 showed that symbolic refutation helps only on SELECTION rows (the verifier
picks a wrong sample although a right one is in the pool) whose wrong pick
holds a checkable false claim. Fresh MATH L3-5 had 5 such opportunities in 80
rows. On the stored P2 pools the share of wrong samples carrying a provably
false claim grows with the size of the numbers in the problem:
    max number in the text   < 100   100-999   1e3-1e4   >= 1e4
    wrong samples refutable    13%      25%       19%       41%
so the run goes where the numbers are large and the arithmetic is long. The
rows are chosen from metadata only (dataset, perturbation type, magnitude of
the numbers in the TEXT), never from any outcome.

ROWS (pretest_data/v36_arith.json, seed 36, built once, then committed)
-----------------------------------------------------------------------
  A  100  GSM-Hard, CLEAN: auditable rows of gsmhard_audit.py with none of its
          three defect channels (collision, never perturbed, GSM8K-Platinum
          dropped), gold > 0, max number in the text >= 1e4; rows used by the
          v18/v19 pretests excluded.
  B   50  GSM-Plus "digit expansion" (human-verified golds, numeric answers),
          max number in the text >= 1e3.
  C   50  GSM-Plus "integer-decimal-fraction conversion" (same).
The manifest interleaves the pools (A, B, A, C, ...) so that a partial run, or
either half of a two-machine run, covers every pool evenly.

SAMPLES
-------
v22's sampler and CoT prompt, 5 samples per row from Qwen2.5-Math-7B-Instruct,
T 0.8, 1536 new tokens (large-number arithmetic runs longer than P2), every
sample scored by Qwen2.5-Math-PRM-7B. The audit (trace_siv.py, v35.1: the v35
checker plus two precision fixes found on the v35 rows, committed before this
manifest existed) runs offline on the saved traces.

PRE-REGISTERED (V36_PLAN.md section 4; frozen before any sample exists)
----------------------------------------------------------------------
  PRIMARY    ADAPT-R vs ADAPT, all 200 rows: net >= +5 with W >= 2L ->
             SUPPORTED (and "significant" if the sign test gives p < 0.05);
             net <= 0 -> REFUTED; otherwise INCONCLUSIVE
  SECONDARY  ADAPT-R vs SC@5, all 200 rows: net >= 0 -> NON-INFERIOR
  PRECISION  every FALSE verdict inside a RIGHT sample is printed for a hand check
Reported, not decisive: every arm per pool, B5-R vs B5, SC5-R vs SC5, calls/row,
the audit statistics, and the pooled fresh total with v35 (280 rows).

MODES
-----
    python pretest_v36.py --build-manifest     # once (needs `datasets`); then commit
    python pretest_v36.py --max-hours 8.5      # Kaggle T4 x2; re-run to resume
    python pretest_v36.py --reverse ...        # the second machine, from the end
    python pretest_v36.py --summary-only --out A.json --merge B.json
    python pretest_v36.py --stub               # offline plumbing
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from typing import Dict, List, Optional

import pretest_v21 as V21
import pretest_v24 as V24
import pretest_v34 as V34
import pretest_v35 as V35
import replay_v35 as R
import trace_siv as T

MANIFEST = 'pretest_data/v36_arith.json'
OUT = 'pretest_v36.json'
OUT_STUB = 'pretest_v36_stub.json'
SEED = 36
MAX_TOKENS = 1536
POOLS = (('A', 'gsm-hard-clean', 100), ('B', 'gsm-plus:digit expansion', 50),
         ('C', 'gsm-plus:integer-decimal-fraction conversion', 50))
MIN_MAG_A = 1e4
MIN_MAG_B = 1e3        # digit expansion: keep the rows whose numbers did grow
USED_BEFORE = ('pretest_data/large_number_rows.json', 'pretest_data/clean_large_number_rows.json')

# pre-registered bars (V36_PLAN.md section 4)
PRI_MIN_NET, PRI_RATIO, PRI_ALPHA = 5, 2, 0.05
SEC_MIN_NET = 0


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def max_number(text: str) -> float:
    vals = []
    for t in re.findall(r'\d[\d,]*(?:\.\d+)?', text or ''):
        try:
            vals.append(float(t.replace(',', '')))
        except ValueError:
            pass
    return max(vals) if vals else 0.0


def _num(x) -> Optional[float]:
    try:
        v = float(str(x).replace(',', '').strip())
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def build_manifest(force: bool = False) -> int:
    if os.path.exists(MANIFEST) and not force:
        print(f"{MANIFEST} exists; refusing to overwrite pre-registered rows (--force to redo)")
        return 1
    from datasets import load_dataset
    import gsmhard_audit as GA
    rng = random.Random(SEED)
    pools: Dict[str, List[Dict]] = {}

    # A: GSM-Hard, clean and large
    _, sets = GA.run()
    bad = set(sets['collision']) | set(sets['never_perturbed']) | set(sets['platinum_dropped'])
    used = set()
    for f in USED_BEFORE:
        if os.path.exists(f):
            used |= {r['problem_id'] for r in V24._read(f)['rows']}
    hard = load_dataset('reasoning-machines/gsm-hard', split='train')
    a = []
    for i, r in enumerate(hard):
        pid = f'gsm-hard_{i}'
        g = _num(r['target'])
        if (pid in sets['auditable'] and pid not in bad and pid not in used and g is not None and g > 0
                and max_number(r['input']) >= MIN_MAG_A):
            a.append({'problem_id': pid, 'text': r['input'].strip(), 'gold': g, 'level': 'A',
                      'subject': 'gsm-hard-clean'})
    pools['A'] = a

    # B, C: GSM-Plus arithmetic perturbations
    plus = load_dataset('qintongli/GSM-Plus', split='test')
    for key, name, _ in POOLS[1:]:
        ptype = name.split(':', 1)[1]
        rows = []
        for i, r in enumerate(plus):
            g = _num(r['answer'])
            if r['perturbation_type'] != ptype or g is None:
                continue
            if key == 'B' and max_number(r['question']) < MIN_MAG_B:
                continue
            rows.append({'problem_id': f'gsm-plus_{i}', 'text': r['question'].strip(), 'gold': g,
                         'level': key, 'subject': name})
        pools[key] = rows

    picked = {}
    for key, name, n in POOLS:
        rows = list(pools[key])
        rng.shuffle(rows)
        picked[key] = rows[:n]
        print(f"  pool {key} {name}: {len(pools[key])} eligible -> {len(picked[key])}")
    order = []                      # A, B, A, C, A, B, ...: every prefix and suffix mixes the pools
    ia, ib, ic = iter(picked['A']), iter(picked['B']), iter(picked['C'])
    for k in range(max(len(v) for v in picked.values())):
        for it in (ia, ib if k % 2 == 0 else ic):
            x = next(it, None)
            if x is not None:
                order.append(x)
    order += list(ia) + list(ib) + list(ic)
    with open(MANIFEST, 'w', encoding='utf-8') as fh:
        json.dump({'dataset': 'v36 arithmetic-heavy: GSM-Hard clean large + GSM-Plus arithmetic perturbations',
                   'seed': SEED, 'n': len(order), 'pools': {k: len(v) for k, v in pools.items()},
                   'rows': order}, fh, ensure_ascii=False, indent=1)
    print(f"wrote {len(order)} rows -> {MANIFEST}")
    return 0


# ---------------------------------------------------------------------------
# the reading
# ---------------------------------------------------------------------------

def _sign_p(w: int, l: int) -> float:
    return R.sign_p(w, l)


def verdicts(rows: List[Dict]) -> List[str]:
    out = []
    w = sum(R.right(r, 'ADAPT-R') and not R.right(r, 'ADAPT') for r in rows)
    l = sum(R.right(r, 'ADAPT') and not R.right(r, 'ADAPT-R') for r in rows)
    facts = R.wl(rows, 'ADAPT-R', 'ADAPT')
    if w - l >= PRI_MIN_NET and w >= PRI_RATIO * l:
        sig = ' and significant' if _sign_p(w, l) < PRI_ALPHA else ''
        out.append(f"PRIMARY    SUPPORTED{sig}: refutation improves verifier-routed sampling ({facts})")
    elif w - l <= 0:
        out.append(f"PRIMARY    REFUTED: {facts}")
    else:
        out.append(f"PRIMARY    INCONCLUSIVE: {facts}")
    w = sum(R.right(r, 'ADAPT-R') and not R.right(r, 'SC5') for r in rows)
    l = sum(R.right(r, 'SC5') and not R.right(r, 'ADAPT-R') for r in rows)
    out.append(f"SECONDARY  {'NON-INFERIOR' if w - l >= SEC_MIN_NET else 'NOT MET'} to SC@5: "
               f"{R.wl(rows, 'ADAPT-R', 'SC5')}")
    return out


def _rows(rows: List[Dict], recs: Dict) -> List[Dict]:
    done = []
    for r in rows:
        c = (recs.get(r['key']) or {}).get('C') or []
        if V34.complete({'C': c}):
            s = [dict(x) for x in c[:V34.K]]
            for x in s:
                x['ok'] = V21.correct(x.get('answer'), r['gold'])
                x['audit'] = T.audit(x.get('raw') or '')
            done.append({'set': 'main', 'pid': r['pid'], 'gold': r['gold'], 'pool': r['level'], 'C': s})
    return done


def summarise(rows: List[Dict], recs: Dict) -> int:
    done = _rows(rows, recs)
    print('\n' + '=' * 78)
    print(f"  v36 arithmetic-heavy rows (fresh): {len(done)}/{len(rows)} rows complete")
    print(f"  frozen: {V35._frozen()}")
    if done:
        groups = {f"{k}:{name.split(':')[-1][:9]}": [r for r in done if r['pool'] == k] for k, name, _ in POOLS}
        groups['ALL'] = done
        R.table(groups)
        for k in ('ADAPT', 'SC5'):
            print(f"  ALL  {R.wl(done, 'ADAPT-R', k)}")
        print(f"  ALL  {R.wl(done, 'B5-R', 'B5')}")
        print(f"  ALL  {R.wl(done, 'SC5-R', 'SC5')}")
        opp = [r for r in done if any(s['ok'] for s in r['C']) and not R.right(r, 'ADAPT')]
        print(f"  selection opportunities (ADAPT wrong, a right sample in the pool): {len(opp)}")
        R.audit_stats(done, 'fresh')
        print('\n  FALSE verdicts inside RIGHT samples (hand-check each):')
        n = 0
        for r in done:
            for j, s in enumerate(r['C']):
                if s['ok']:
                    for i, st in enumerate(s['audit']):
                        for t, v in st:
                            if v == T.FALSE:
                                n += 1
                                print(f"    {r['pid']} s{j} step{i}: {t}")
        print(f"    ({n} in total)")
    print('\n  pre-registered reading:')
    if len(done) < len(rows):
        print(f"    none: PARTIAL run, {len(done)}/{len(rows)} rows complete. Re-run the identical command.")
        return 0
    for line in verdicts(done):
        print('    ' + line)
    return 0


def _install(reverse: bool = False) -> None:
    """Point v34's runner at the v36 rows, the v36 token budget and the v36 reading."""
    V34.MANIFEST, V34.OUT, V34.OUT_STUB, V34.MAX_TOKENS = MANIFEST, OUT, OUT_STUB, MAX_TOKENS
    V34.summarise = summarise
    if reverse:
        load = V34.load_rows
        V34.load_rows = lambda: list(reversed(load()))


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
    ap.add_argument('--reverse', action='store_true', help='work through the rows from the end')
    ap.add_argument('--merge', action='append', default=[], help='more result files (summary only)')
    args = ap.parse_args()
    if args.build_manifest:
        return build_manifest(args.force)
    if args.summary_only:
        _install(False)
        path = args.out or (OUT_STUB if args.stub else OUT)
        rows = V34.load_rows()
        if args.limit:
            rows = rows[:args.limit]
        return summarise(rows, V35.merged([path] + args.merge))
    _install(args.reverse)
    return V34.run(args)


if __name__ == '__main__':
    sys.exit(main())
