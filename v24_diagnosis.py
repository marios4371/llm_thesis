"""
[v24.0] Why a Reader agent, read off the traces already on disk. No GPU, no
model, no generation. Every number below is recomputed from the stored CoT
samples of the v21 dev rows (pretest_data/v21_traces.json) and the v22
confirmation rows (results_September/preset_v22.json): 5 samples per row,
T=0.8, the prompt of every version since v20.

  1. RESAMPLING  How often none of the 5 samples is right, so that no vote,
                 verifier or restart policy that only picks among samples can
                 recover the row.
  2. THE MODE    On those rows, how often the wrong samples agree with each
                 other: a shared wrong answer is a shared wrong reading, not
                 noise.
  3. TEMPLATES   GSM-Symbolic keeps a template's 50 instances together
                 (row index // 50). How much of the row-level accuracy
                 variance is BETWEEN templates, and how concentrated the wrong
                 samples are. High numbers mean the errors are misreadings of
                 a problem STRUCTURE, which the numbers of an instance do not
                 change. They also mean that a reading of any P2 result must
                 count templates, not only rows.

  4. POWER       (--power) How often the pre-registered PRIMARY rule reads
                 SUPPORTED / INCONCLUSIVE / REFUTED on the 100 v22 main rows,
                 when ASQ samples are simulated from the stored C rates plus
                 a chosen true effect. It shows that the rule rarely fires
                 without a gain, and what gain it can see.

    python v24_diagnosis.py            # 1-3, a few seconds
    python v24_diagnosis.py --power    # also 4, about a minute
"""
from __future__ import annotations

import collections
import json
import random
import sys
from typing import Callable, Dict, List

import pretest_v21 as V21
import pretest_v24 as V24

ok = V21.correct


def main_rows() -> List[Dict]:
    rows = []
    for r in V24.load_pool(('v21', 'v22')):
        if r['set'] == 'main':
            rows.append({'src': r['source'], 'pid': r['pid'], 'gold': r['gold'],
                         'text': r['text'], 'template': r['template'],
                         'answers': [s['answer'] for s in r['stored_C']]})
    return rows


def resampling(rows: List[Dict]) -> None:
    print("\n1. RESAMPLING: rows where none of the 5 samples is right")
    for src in ('v21', 'v22'):
        rr = [r for r in rows if r['src'] == src]
        none = [r for r in rr if not any(ok(a, r['gold']) for a in r['answers'])]
        mean1 = sum(sum(ok(a, r['gold']) for a in r['answers']) for r in rr) / (5 * len(rr))
        print(f"   {src} main: {len(none)}/{len(rr)} rows; mean single-sample accuracy "
              f"{100 * mean1:.1f}%")


def mode(rows: List[Dict]) -> None:
    print("\n2. THE MODE: on rows with no right sample, the size of the largest group of "
          "identical wrong answers")
    none = [r for r in rows if not any(ok(a, r['gold']) for a in r['answers'])]
    size = collections.Counter()
    for r in none:
        c = collections.Counter(round(a, 4) for a in r['answers'] if a is not None)
        size[c.most_common(1)[0][1] if c else 0] += 1
    print("   " + "  ".join(f"{k} of 5: {size[k]} rows" for k in sorted(size)))
    shared = sum(v for k, v in size.items() if k >= 3)
    print(f"   {shared}/{len(none)} rows: at least 3 of the 5 samples give the SAME wrong answer")


def templates(rows: List[Dict]) -> None:
    print("\n3. TEMPLATES: GSM-Symbolic template = row index // 50")
    by = collections.defaultdict(list)
    for r in rows:
        by[r['template']].append(sum(ok(a, r['gold']) for a in r['answers']) / 5)
    allv = [v for vs in by.values() for v in vs]
    mu = sum(allv) / len(allv)
    tot = sum((v - mu) ** 2 for v in allv)
    between = sum(len(vs) * (sum(vs) / len(vs) - mu) ** 2 for vs in by.values())
    print(f"   {len(rows)} main rows fall in {len(by)} templates")
    print(f"   share of the row-level accuracy variance that lies BETWEEN templates: "
          f"{between / tot:.2f}")
    wrong = sorted(((sum(1 - v for v in vs), t) for t, vs in by.items()), reverse=True)
    total = sum(w for w, _ in wrong)
    print(f"   share of all wrong samples in the 10 worst templates: "
          f"{sum(w for w, _ in wrong[:10]) / total:.2f}")
    means = {t: sum(vs) / len(vs) for t, vs in by.items()}
    low = sum(1 for m in means.values() if m <= 0.3)
    high = sum(1 for m in means.values() if m >= 0.9)
    print(f"   templates at <= 30% per sample: {low};  at >= 90%: {high};  "
          f"in between: {len(means) - low - high}")
    print("   the 12 hardest templates (mean single-sample accuracy, rows):")
    for t, m in sorted(means.items(), key=lambda kv: kv[1])[:12]:
        ex = next(r for r in rows if r['template'] == t)
        print(f"     {t:2d}: {100 * m:5.1f}%  ({len(by[t])} rows)  {ex['text'][:70]}...")


def power(n_sim: int = 300, seed: int = 1) -> None:
    """Simulate the PRIMARY rule on the dev rows (v22 main). ASQ's 5 samples on
    a row are drawn with probability = the row's stored C rate + the effect;
    C stays the stored samples, as in the real dev pass."""
    print("\n4. POWER of the pre-registered PRIMARY rule on the 100 v22 main rows")
    rows = [r for r in main_rows() if r['src'] == 'v22']
    rate = {r['pid']: sum(ok(a, r['gold']) for a in r['answers']) / 5 for r in rows}
    by_t = collections.defaultdict(list)
    for r in rows:
        by_t[r['template']].append(rate[r['pid']])
    hard = {t for t, v in by_t.items() if sum(v) / len(v) <= 0.3}
    effects: Dict[str, Callable[[Dict], float]] = {
        'no gain (ASQ = C)': lambda r: 0.0,
        '+0.25 on the misread templates (C <= 30%)':
            lambda r: 0.25 if r['template'] in hard else 0.0,
        '+0.40 on misread templates, -0.05 on easy rows':
            lambda r: 0.40 if r['template'] in hard else (-0.05 if rate[r['pid']] >= 0.8 else 0.0),
        '+0.06 on every row': lambda r: 0.06,
        '-0.03 on every row': lambda r: -0.03,
    }
    rng = random.Random(seed)
    print(f"   {'true effect':48s} {'mean gain':>9s} {'SUPPORTED':>9s} {'INCONCL.':>8s} "
          f"{'REFUTED':>7s}")
    for name, eff in effects.items():
        tally = collections.Counter()
        gains = []
        for _ in range(n_sim):
            d, by = [], collections.defaultdict(list)
            for r in rows:
                p = min(1.0, max(0.0, rate[r['pid']] + eff(r)))
                v = sum(rng.random() < p for _ in range(5)) / 5 - rate[r['pid']]
                d.append(v)
                by[r['template']].append(v)
            pp = 100 * sum(d) / len(d)
            loto = min(100 * (sum(d) - sum(v)) / (len(d) - len(v)) for v in by.values())
            p = V24.perm_p(d, b=2000, seed=rng.randrange(1 << 30))
            gains.append(pp)
            if pp >= V24.PRIMARY_MIN_PP and p < V24.PRIMARY_ALPHA and loto >= V24.PRIMARY_LOTO_PP:
                tally['S'] += 1
            elif pp <= 0:
                tally['R'] += 1
            else:
                tally['I'] += 1
        print(f"   {name:48s} {sum(gains) / n_sim:+7.1f}pp {tally['S'] / n_sim:9.2f} "
              f"{tally['I'] / n_sim:8.2f} {tally['R'] / n_sim:7.2f}")


def main() -> int:
    rows = main_rows()
    print("=" * 76)
    print("v24 diagnosis: the Solver's remaining P2 errors are misreadings that resampling")
    print("re-draws (stored CoT samples, v21 + v22 main rows)")
    resampling(rows)
    mode(rows)
    templates(rows)
    if '--power' in sys.argv:
        power()
    return 0


if __name__ == '__main__':
    sys.exit(main())
