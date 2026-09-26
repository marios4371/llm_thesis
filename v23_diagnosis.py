"""
[v23.0] Why Verifier-guided Tabu Restarts, read off the traces already on
disk. No GPU, no model, no generation: every number below is recomputed from
the stored v21 dev rows (v21_traces + their v22 verifier scores) and the
stored v22 confirmation rows (preset_v22.json).

  1. HEADROOM   Selection is spent. B5 is within a few rows of oracle@5, so no
                new vote or verifier weighting can move the headline; the
                remaining errors are rows no sample gets right.
  2. TRIGGER    TAU is chosen on the v21 rows only, and read on the v22 rows
                as held out.
  3. THE MODE   On triggered rows with a wrong leader, fresh samples are
                rarely right and rarely repeat the leader's ANSWER, but about
                twice as often re-derive the value of the step the Verifier
                doubts: what they share is a step, not an answer.
  4. WORTH      What one restart is worth once it exists: how often a right
                restart displaces a wrong leader, and how often a wrong
                restart displaces a right one. With (3) this gives the
                break-even a restart policy has to clear.

    python v23_diagnosis.py
"""
from __future__ import annotations

import collections
import sys
from typing import Dict, List

import pretest_v21 as V21
import pretest_v23 as V23
import score_prm_v22 as P
import tabu_restart as T

ok = V21.correct


def headroom(rows: List[Dict], title: str) -> None:
    n = len(rows)
    if not n:
        return
    arms = collections.Counter()
    dist = collections.Counter()
    for r in rows:
        ans = [s['answer'] for s in r['samples']]
        wl = [T.last(s['prm']) for s in r['samples']]
        c = [ok(a, r['gold']) for a in ans]
        dist[sum(c)] += 1
        arms['C1'] += c[0]
        arms['S3'] += ok(P.plain_vote(ans[:3]), r['gold'])
        arms['S5'] += ok(P.plain_vote(ans[:5]), r['gold'])
        arms['B3'] += ok(T.best_of(ans[:3], wl[:3]), r['gold'])
        arms['B5'] += ok(T.best_of(ans[:5], wl[:5]), r['gold'])
        arms['oracle@3'] += any(c[:3])
        arms['oracle@5'] += any(c)
    mean1 = sum(k * v for k, v in dist.items()) / (5 * n)
    print(f"\n  {title}  n={n}")
    print("    " + "  ".join(f"{a} {arms[a]}" for a in
                             ('C1', 'S3', 'S5', 'B3', 'B5', 'oracle@3', 'oracle@5')))
    print(f"    mean single-sample accuracy {100*mean1:.1f}%")
    print("    rows by number of correct samples (of 5): "
          + ' '.join(f"{k}:{dist[k]}" for k in range(6)))
    b5_wrong = [r for r in rows if not ok(T.best_of([s['answer'] for s in r['samples']],
                                                    [T.last(s['prm']) for s in r['samples']]),
                                          r['gold'])]
    none = sum(1 for r in b5_wrong if not any(ok(s['answer'], r['gold']) for s in r['samples']))
    print(f"    B5 errors: {len(b5_wrong)} -- {none} with NO correct sample, "
          f"{len(b5_wrong) - none} selection misses")


def trigger_table(rows: List[Dict], title: str) -> None:
    print(f"\n  {title}  n={len(rows)}  (leader = B3's choice; triggered = leader's min step < tau)")
    allw = sum(1 for r in rows if not ok(r['samples'][T.leader(r['samples'])]['answer'], r['gold']))
    for tau in (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99):
        tr = [r for r in rows if T.plan(r['samples'], tau=tau)['triggered']]
        wr = [r for r in tr if not ok(r['samples'][T.leader(r['samples'])]['answer'], r['gold'])]
        mark = '   <- TAU' if tau == T.TAU else ''
        print(f"    tau {tau:.2f}: triggered {len(tr):3d}  wrong leaders caught {len(wr):2d}/{allw}"
              f"  right leaders triggered {len(tr)-len(wr):3d}{mark}")


def mode_table(rows: List[Dict], title: str) -> None:
    """What the fresh samples (4, 5) do on triggered rows, split by whether
    the leader is right; and whether the doubted value belongs to a correct
    solution at all."""
    print(f"\n  {title}")
    for want, label in ((False, 'leader WRONG'), (True, 'leader right')):
        n = fresh = same = right = 0
        dv_fresh = dv_fresh_n = dv_right = dv_right_n = given = 0
        for r in rows:
            p = T.plan(r['samples'])
            if not p['triggered'] or ok(p['leader_answer'], r['gold']) != want:
                continue
            n += 1
            for s in r['samples'][3:5]:
                fresh += 1
                same += s['answer'] is not None and ok(s['answer'], p['leader_answer'])
                right += ok(s['answer'], r['gold'])
            dv = T.doubted_value(p['doubted'], r['text']) if p['doubted'] else None
            if dv is None:
                given += 1
                continue
            for s in r['samples'][3:5]:
                dv_fresh_n += 1
                dv_fresh += bool(T.contains(s['raw'], dv))
            for s in r['samples']:
                if ok(s['answer'], r['gold']):
                    dv_right_n += 1
                    dv_right += bool(T.contains(s['raw'], dv))
        if not n:
            continue
        pct = lambda a, b: f"{a}/{b} = {100*a/b:.0f}%" if b else f"{a}/0"
        print(f"    {label}: {n} triggered rows")
        print(f"      fresh samples right                  {pct(right, fresh)}")
        print(f"      fresh samples repeating the leader's answer   {pct(same, fresh)}")
        print(f"      fresh samples re-deriving the doubted value   {pct(dv_fresh, dv_fresh_n)}"
              f"   ({given} rows: doubted value is a given, skipped)")
        print(f"      CORRECT samples containing the doubted value  {pct(dv_right, dv_right_n)}")


def worth_table(rows: List[Dict], title: str) -> None:
    """Selection is best-of by last-step reward, drafts included, so a
    restart matters only if it outscores the leader."""
    right_n = right_beats = wrong_n = wrong_beats = 0
    for r in rows:
        p = T.plan(r['samples'])
        if not p['triggered'] or p['leader'] is None:
            continue
        lead = T.last(r['samples'][p['leader']]['prm'])
        leader_ok = ok(p['leader_answer'], r['gold'])
        for i, s in enumerate(r['samples']):
            if i == p['leader'] or s['answer'] is None:
                continue
            if not leader_ok and ok(s['answer'], r['gold']):
                right_n += 1
                right_beats += T.last(s['prm']) > lead
            elif leader_ok and not ok(s['answer'], r['gold']):
                wrong_n += 1
                wrong_beats += T.last(s['prm']) > lead
    pct = lambda a, b: f"{a}/{b} = {100*a/b:.0f}%" if b else f"{a}/0"
    print(f"\n  {title}")
    print(f"    a RIGHT sample outscores a WRONG leader   {pct(right_beats, right_n)}")
    print(f"    a WRONG sample outscores a RIGHT leader   {pct(wrong_beats, wrong_n)}")


def main() -> int:
    pool = V23.load_pool()
    by = lambda src, tag: [r for r in pool if r['source'] == src and r['set'] == tag]
    print("=" * 76)
    print("1. HEADROOM: selection is spent")
    for src in ('v21', 'v22'):
        for tag in ('main', 'guard'):
            headroom(by(src, tag), f"{src.upper()} {tag}")
    print("\n" + "=" * 76)
    print("2. TRIGGER: TAU is read off the v21 rows; the v22 rows are held out")
    trigger_table(by('v21', 'main') + by('v21', 'guard'), 'V21 (dev)')
    trigger_table(by('v22', 'main') + by('v22', 'guard'), 'V22 (held out)')
    print("\n" + "=" * 76)
    print(f"3. THE MODE is a step, not an answer (TAU={T.TAU})")
    mode_table(by('v21', 'main') + by('v21', 'guard'), 'V21 (dev)')
    mode_table(by('v22', 'main') + by('v22', 'guard'), 'V22 (held out)')
    mode_table(pool, 'POOLED')
    print("\n" + "=" * 76)
    print("4. WORTH of one restart on a triggered row (any sample other than the leader)")
    worth_table(pool, 'POOLED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
