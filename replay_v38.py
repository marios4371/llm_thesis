"""
[v38.0] Offline replay of Symbolic Surgery (trace_surgery.py) on the 680 stored
rows (figures/make_paper_figures.load_all: P2a, P2b, V21, MATH, v35, v36).
No model, no GPU. Graded ONLY with the canonical grader |pred - gold| < 1e-3
(load_all installs it; the relative pretest grader hides the very slips this
version repairs, see V36_PLAN.md section 7.1).

    python replay_v38.py              # reproduction + the v38 operation + the rules
    python replay_v38.py --rows       # also every row where a surgery arm moves

What the replay can and cannot see:
  * PROPAGATE is fully offline, but the stored PRM scores are those of the
    ORIGINAL trace (a run re-scores the repaired one).
  * RESUME needs the solver. Policies that choose it are read twice: with the
    resume dropped (a lower bound) and with the propagated repair in its place
    (a proxy); each resume is charged one call either way.

Section A reproduces the 2026-10-20 probes (session 3c39c77f scratchpad:
trace_lift_probe.py, repair_vote_probe.py) with their surgery copied verbatim,
so the numbers the plan quotes can be re-derived from the repo.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'figures'))
os.chdir(ROOT)

import make_paper_figures as F       # noqa: E402  (installs the canonical grader on import)
import replay_v35 as R               # noqa: E402
import score_prm_v22 as P            # noqa: E402
import trace_siv as T                # noqa: E402
import trace_surgery as S            # noqa: E402

canon = F.canon
ORDER = ['P2a', 'P2b', 'V21', 'MATH', 'v35', 'v36']
UNTOUCHED = ('V21', 'v35', 'v36')    # pools the checker was never debugged on


# ---------------------------------------------------------------------------
# A. the 2026-10-20 probes, verbatim (only for reproduction)
# ---------------------------------------------------------------------------

def _p_subst(expr, fix):
    if not fix:
        return expr

    def rep(m):
        s = m.group(0)
        try:
            v = float(s.replace(',', ''))
        except ValueError:
            return s
        for wrong, right in fix:
            if S.close(v, wrong):
                return str(int(round(right))) if abs(right - round(right)) < 1e-9 else repr(right)
        return s
    return S.NUM.sub(rep, expr)


def probe_surgery(raw, answer, lift=False):
    """repaired_jury_probe.surgery (lift=False) / trace_lift_probe.surgery (lift=True)."""
    fix, changed = [], False
    for step in P.split_steps(raw or ''):
        for seg in T.segments(step):
            if r'\equiv' in seg:
                continue
            parts = re.split(r'(=|\\approx|≈)', seg)[0::2]
            carry = None
            for i in range(len(parts) - 1):
                a, b = parts[i], parts[i + 1]
                va = carry if carry is not None else T.value(_p_subst(a, fix))
                vb = T.value(b)
                carry = None
                if va is None or vb is None:
                    continue
                false = T.check_pair(_p_subst(a, fix), b) == T.FALSE
                if lift:
                    false = false or (fix and not S.close(va, vb) and T.value(a) is not None
                                      and S.close(T.value(a), vb))
                if false and not S.close(va, vb):
                    fix.append((vb, va))
                    changed = True
                    carry = va
    if not changed or answer is None:
        return None
    for wrong, right in fix:
        if S.close(answer, wrong):
            return right
    return answer


def reproduce(pools: Dict[str, List[Dict]]) -> None:
    print('=' * 78 + '\n  A. reproduction of the 2026-10-20 probes (canonical grader)')
    tot = Counter()
    for p in ORDER:
        resc = harm = cov = rows0 = 0
        for r in pools[p]:
            rep = []
            for x in r['C']:
                new = (probe_surgery(x.get('raw') or '', x.get('answer'), lift=True)
                       if T.refutes(x['audit'], None) or x['ok'] else None)
                ok = canon(new, r['gold']) if new is not None else x['ok']
                resc += (not x['ok']) and new is not None and ok
                harm += x['ok'] and new is not None and not ok
                rep.append(ok)
            if not any(x['ok'] for x in r['C']):
                rows0 += 1
                cov += any(rep)
        tot.update(rescued=resc, harmed=harm, cov=cov, rows0=rows0)
        print(f"    {p:5s} rescued {resc:3d}  harmed {harm}  no-right rows {rows0:3d} -> gain one {cov}")
    print(f"    TOTAL rescued {tot['rescued']} wrong samples, harmed {tot['harmed']} right ones; "
          f"{tot['cov']} of {tot['rows0']} rows with no right sample gain one")
    for r in (r for p in ORDER for r in pools[p]):
        for x in r['C']:
            new = probe_surgery(x.get('raw') or '', x.get('answer')) if T.refutes(x['audit'], None) else None
            x['rep'] = new if new is not None else x.get('answer')

    def with_rep(c):
        out = []
        for x in c:
            y = dict(x)
            y['answer'] = x['rep']
            y['audit'] = [] if x['rep'] is not None and x['rep'] != x.get('answer') else x['audit']
            out.append(y)
        return out
    R.ARMS['SC5-rep'] = lambda c: (R._plurality(with_rep(c), range(5)), 5)
    R.ARMS['ADAPT-R-rep'] = lambda c: R.ARMS['ADAPT-R'](with_rep(c))
    allr = [r for p in ORDER for r in pools[p]]
    unt = [r for p in UNTOUCHED for r in pools[p]]
    print(f"    ALL ({len(allr)})       {R.wl(allr, 'SC5-rep', 'SC5')}")
    print(f"    untouched ({len(unt)}) {R.wl(unt, 'SC5-rep', 'SC5')}")
    print(f"    ALL ({len(allr)})       {R.wl(allr, 'ADAPT-R-rep', 'ADAPT-R')}")
    print(f"    ADAPT-R-rep {F.acc(allr, 'ADAPT-R-rep'):.1f} vs ADAPT-R {F.acc(allr, 'ADAPT-R'):.1f}; "
          + '  '.join(f"{p} {sum(R.right(r, 'ADAPT-R-rep') for r in pools[p]) - sum(R.right(r, 'ADAPT-R') for r in pools[p]):+d}"
                      for p in ORDER))


# ---------------------------------------------------------------------------
# B. the v38 operation on every refuted sample
# ---------------------------------------------------------------------------

def prepare(pools: Dict[str, List[Dict]]) -> None:
    """sample['surgery'] = trace_surgery.operate(...) for every refuted sample;
    the propagated repair carries the ORIGINAL PRM scores and its own audit."""
    for r in (r for p in ORDER for r in pools[p]):
        for x in r['C']:
            x.pop('surgery', None)
            if x.get('answer') is None or not T.refutes(x['audit'], None):
                continue
            op = S.operate(x.get('raw') or '', x.get('answer'))
            if op is None:
                continue
            op['prop'].update(prm=x.get('prm'), audit=T.audit(op['prop']['raw']),
                              ok=canon(op['prop']['answer'], r['gold']))
            op['resume'] = None
            x['surgery'] = op


def operation_stats(pools: Dict[str, List[Dict]]) -> None:
    print('\n' + '=' * 78 + '\n  B. the v38 operation (trace_surgery.operate) on refuted samples')
    print(f"    {'pool':5s} {'refuted':>8s} {'site':>5s} {'reached':>8s} {'rescued':>8s} {'harmed':>7s} "
          f"{'re-audit clean':>15s} {'no-right rows':>14s} {'gain one':>9s}")
    tot = Counter()
    for p in ORDER:
        c = Counter()
        for r in pools[p]:
            gain = False
            for x in r['C']:
                if x.get('answer') is None or not T.refutes(x['audit'], None):
                    continue
                c['refuted'] += 1
                s = x.get('surgery')
                if not s:
                    continue
                c['site'] += 1
                pr = s['prop']
                c['reached'] += pr['reached']
                c['rescued'] += (not x['ok']) and pr['ok']
                c['harmed'] += x['ok'] and not pr['ok']
                c['clean'] += not T.refutes(pr['audit'], None)
                gain |= pr['ok']
            if not any(x['ok'] for x in r['C']):
                c['rows0'] += 1
                c['cov'] += gain
        tot.update(c)
        print(f"    {p:5s} {c['refuted']:8d} {c['site']:5d} {c['reached']:8d} {c['rescued']:8d} {c['harmed']:7d} "
              f"{c['clean']:15d} {c['rows0']:14d} {c['cov']:9d}")
    print(f"    {'ALL':5s} {tot['refuted']:8d} {tot['site']:5d} {tot['reached']:8d} {tot['rescued']:8d} "
          f"{tot['harmed']:7d} {tot['clean']:15d} {tot['rows0']:14d} {tot['cov']:9d}")

    # what the policy has to weigh: reached x right after, isolated vs correlated
    print('\n    refuted WRONG samples with a site, by what propagation did (right after / n):')
    cells = Counter()
    for r in (r for p in ORDER for r in pools[p]):
        idx = range(len(r['C']))
        for i, x in enumerate(r['C']):
            s = x.get('surgery')
            if not s or x['ok']:
                continue
            k = 'correlated (n>=2)' if S.n_sharing(r['C'], i, idx) >= 2 else 'isolated   (n=1)'
            reach = 'reached the answer' if s['prop']['reached'] else 'did NOT reach it  '
            cells[(k, reach, 'n')] += 1
            cells[(k, reach, 'ok')] += s['prop']['ok']
    for k in ('isolated   (n=1)', 'correlated (n>=2)'):
        print('      ' + k + '   ' + '   '.join(
            f"{reach.strip()}: {cells[(k, reach, 'ok')]}/{cells[(k, reach, 'n')]}"
            for reach in ('reached the answer', 'did NOT reach it  ')))
    rights = [(r, x) for r in (r for p in ORDER for r in pools[p]) for x in r['C'] if x.get('surgery') and x['ok']]
    print(f"    refuted RIGHT samples with a site: {len(rights)}; propagation keeps "
          f"{sum(x['surgery']['prop']['ok'] for _, x in rights)} right, reached the answer in "
          f"{sum(x['surgery']['prop']['reached'] for _, x in rights)}")


# ---------------------------------------------------------------------------
# C. the rules
# ---------------------------------------------------------------------------

def always(action):
    return lambda sample, n: action


def prop_if_reached(sample, n):
    return S.PROPAGATE if sample['surgery']['prop']['reached'] else S.REFUTE


def _user_policy_ready() -> bool:
    try:
        S.surgery_policy({'surgery': {'site': {}, 'prop': {'reached': False}, 'slips': []}, 'prm': [1.0],
                          'answer': 0.0, 'audit': []}, 1)
    except NotImplementedError:
        return False
    except Exception:
        return True                       # it runs on real samples; a toy dict may not satisfy it
    return True


def _with_resume(c: List[Dict], mode: str) -> List[Dict]:
    """mode 'drop': a resume yields nothing (lower bound); 'proxy': it yields
    the propagated repair (what an ideal continuation would compute)."""
    out = []
    for x in c:
        y = dict(x)
        if x.get('surgery'):
            y['surgery'] = dict(x['surgery'], resume=x['surgery']['prop'] if mode == 'proxy' else None)
        out.append(y)
    return out


def _vote(c, policy):
    cand = S.pool(c, range(5), policy)
    return R._plurality(cand, range(len(cand))) if cand else R._plurality(c, range(5))


def install_arms() -> List[str]:
    R.ARMS['SC5-S:prop'] = lambda c: (_vote(c, always(S.PROPAGATE)), 5)
    R.ARMS['ADAPT-RS:prop'] = lambda c: S.adapt_rs(c, always(S.PROPAGATE))
    R.ARMS['ADAPT-RS:reach'] = lambda c: S.adapt_rs(c, prop_if_reached)
    R.ARMS['ADAPT-RS:refute'] = lambda c: S.adapt_rs(c, always(S.REFUTE))
    arms = ['SC5', 'SC5-R', 'SC5-S:prop', 'ADAPT', 'ADAPT-R', 'ADAPT-RS:refute', 'ADAPT-RS:prop', 'ADAPT-RS:reach']
    if _user_policy_ready():
        R.ARMS['ADAPT-RS:user-lo'] = lambda c: S.adapt_rs(_with_resume(c, 'drop'))
        R.ARMS['ADAPT-RS:user-px'] = lambda c: S.adapt_rs(_with_resume(c, 'proxy'))
        arms += ['ADAPT-RS:user-lo', 'ADAPT-RS:user-px']
    return arms


def rules(pools: Dict[str, List[Dict]], show_rows: bool) -> None:
    arms = install_arms()
    print('\n' + '=' * 78 + '\n  C. the rules (right / rows; stored PRM scores; canonical grader)')
    if 'ADAPT-RS:user-lo' not in arms:
        print('    trace_surgery.surgery_policy() is not written yet: only the fixed policies are shown')
    groups = {p: pools[p] for p in ORDER}
    groups['P2 fresh'] = pools['P2a'] + pools['P2b']
    groups['untouched'] = [r for p in UNTOUCHED for r in pools[p]]
    groups['ALL'] = [r for p in ORDER for r in pools[p]]
    print(f"    {'arm':17s}" + ''.join(f"{g:>10s}" for g in groups) + '  calls/row')
    for a in arms:
        print(f"    {a:17s}" + ''.join(f"{sum(R.right(r, a) for r in rows):10d}" for rows in groups.values())
              + f"  {F.calls(groups['ALL'], a):9.2f}")
    print()
    for a in arms[5:]:
        for g in ('v36', 'untouched', 'ALL'):
            print(f"    {g:9s} {R.wl(groups[g], a, 'ADAPT-R')}")
    for g in ('untouched', 'ALL'):
        print(f"    {g:9s} {R.wl(groups[g], 'SC5-S:prop', 'SC5')}")
        print(f"    {g:9s} {R.wl(groups[g], 'SC5-S:prop', 'SC5-R')}")
    if show_rows:
        print('\n    rows where ADAPT-RS:prop and ADAPT-R differ:')
        for p in ORDER:
            for r in pools[p]:
                a, b = R.right(r, 'ADAPT-RS:prop'), R.right(r, 'ADAPT-R')
                if a != b:
                    print(f"      {p:5s} {r['pid']:26s} ADAPT-RS {'right' if a else 'WRONG'}  gold {r['gold']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', action='store_true')
    ap.add_argument('--skip-probe', action='store_true', help='skip section A (the reproduction)')
    args = ap.parse_args()
    pools = F.load_all()
    print(f"  rows: " + ', '.join(f"{p} {len(pools[p])}" for p in ORDER) + f"; total {sum(len(v) for v in pools.values())}")
    if not args.skip_probe:
        reproduce(pools)
    prepare(pools)
    operation_stats(pools)
    rules(pools, args.rows)
    return 0


if __name__ == '__main__':
    sys.exit(main())
