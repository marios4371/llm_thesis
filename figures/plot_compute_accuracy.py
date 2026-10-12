"""Accuracy vs compute on the 200 fresh GSM-Symbolic P2 rows (v22 + v25 confirmations).

Reads the two stored runs, recomputes every point (no GPU), and writes
figures/compute_accuracy.{png,pdf}. Run from the repo root:
    python figures/plot_compute_accuracy.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gated_reading as G
import null_hypothesis as NH
import pretest_v21 as V21
import pretest_v24 as V24
import pretest_v30 as T30

CONFIRM_B = 'results_Octomber/pretest_v25_2ndsession.json'
INK, MUTED, GRID, BASE, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
C_SC, C_BEST, C_ADAPT = '#2a78d6', '#eb6834', '#1baf7a'     # categorical slots 1-3 (validated)


def rows():
    a = [r for r in T30.load_rows() if r['set'] == 'main']
    b = [r for r in V24._read(CONFIRM_B)['rows'] if r['set'] == 'main']
    return a + b


def points(rs):
    ok = lambda a, r: V21.correct(a, r['gold'])
    bk = lambda r, k: G.v22_pick(G.cands(r, k, 0))
    sc = lambda r, k: G.plurality([s.get('answer') for s in r['C'][:k]])

    def adapt(r, t):
        cs = r['C'][:3]
        i = max(range(3), key=lambda j: (NH.last(cs[j]['prm']), -j))
        return (bk(r, 5), 5) if min(cs[i]['prm']) < t else (bk(r, 3), 3)

    out = {'sc': [], 'best': [], 'adapt': []}
    for k in range(1, 6):
        out['sc'].append((k, [ok(sc(r, k), r) for r in rs]))
        out['best'].append((k, [ok(bk(r, k), r) for r in rs]))
    for t in (0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99):
        res = [adapt(r, t) for r in rs]
        out['adapt'].append((t, sum(c for _, c in res) / len(rs), [ok(a, r) for (a, _), r in zip(res, rs)]))
    return out


def ci(v, rng, b=2000):
    n = len(v)
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return 100 * bs[int(0.025 * b)], 100 * bs[int(0.975 * b) - 1]


def main():
    rs = rows()
    n = len(rs)
    p = points(rs)
    rng = random.Random(0)
    acc = lambda v: 100 * sum(v) / n
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xs, ys = [k for k, _ in p['sc']], [acc(v) for _, v in p['sc']]
    ax.plot(xs, ys, color=C_SC, lw=2, marker='o', ms=7, mec=SURFACE, mew=1.5, label='Self-consistency (majority vote of k)')
    xs, ys = [k for k, _ in p['best']], [acc(v) for _, v in p['best']]
    ax.plot(xs, ys, color=C_BEST, lw=2, marker='s', ms=7, mec=SURFACE, mew=1.5, label='Verifier best-of-k (v22)')
    ax.plot([c for _, c, _ in p['adapt']], [acc(v) for _, _, v in p['adapt']], color=C_ADAPT, lw=2,
            marker='D', ms=6, mec=SURFACE, mew=1.5, label='Verifier-routed sampling (3 -> 5 when in doubt)')
    for name, x, v, col in (('SC@5', 4.95, p['sc'][4][1], C_SC), ('B5', 5.05, p['best'][4][1], C_BEST)):
        lo, hi = ci(v, rng)
        ax.errorbar([x], [acc(v)], yerr=[[acc(v) - lo], [hi - acc(v)]], color=col, capsize=4, lw=1.2, zorder=1)
    t, c, v = next(x for x in p['adapt'] if x[0] == 0.85)
    lo, hi = ci(v, rng)
    ax.errorbar([c], [acc(v)], yerr=[[acc(v) - lo], [hi - acc(v)]], color=C_ADAPT, capsize=4, lw=1.2, zorder=1)
    ax.plot([c], [acc(v)], marker='D', ms=10, color=C_ADAPT, mec=INK, mew=1.2, zorder=5)
    ax.annotate(f'pre-registered gate (0.85)\n{acc(v):.1f}% at {c:.2f} calls', xy=(c, acc(v)), xytext=(c - 1.45, acc(v) + 5.5),
                fontsize=8.5, color=INK, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    ax.annotate(f'SC@5 {acc(p["sc"][4][1]):.1f}%', xy=(5, acc(p['sc'][4][1])), xytext=(5.08, acc(p['sc'][4][1]) - 0.4),
                fontsize=8.5, color=INK, va='center')
    ax.annotate(f'B5 {acc(p["best"][4][1]):.1f}%', xy=(5, acc(p['best'][4][1])), xytext=(5.08, acc(p['best'][4][1]) + 0.6),
                fontsize=8.5, color=INK, va='center')
    ax.set_xlim(0.7, 5.75)
    ax.set_ylim(55, 90)
    ax.set_xlabel('LLM calls per problem', color=INK, fontsize=10)
    ax.set_ylabel('Accuracy (%)', color=INK, fontsize=10)
    ax.set_title(f'Accuracy vs. compute, {n} fresh GSM-Symbolic P2 problems (two independent sets)',
                 color=INK, fontsize=10.5, loc='left')
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    leg = ax.legend(loc='lower right', fontsize=8.5, frameon=False, labelcolor=INK)
    fig.text(0.01, 0.005, 'Error bars: 95% bootstrap CI (2,000 resamples). Qwen2.5-Math-7B-Instruct solver, '
             'Qwen2.5-Math-PRM-7B verifier.', fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join('figures', f'compute_accuracy.{ext}'), facecolor=SURFACE)
    print('wrote figures/compute_accuracy.png and .pdf')


if __name__ == '__main__':
    main()
