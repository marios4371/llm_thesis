"""
[v40.0] Offline evidence for cross-model Symbolic Surgery (no GPU, no model).

Reads the stored samples of three solvers and reports, per solver:
  A. the audit as v38 ran it (raw text, every FALSE verdict) against v40's
     (clean() + complete(), cross_model_surgery.py): samples with a checked
     claim, refuted samples, refuted samples that are RIGHT (false alarms);
  B. the verifier-free arms on the same five samples: SC5, SC5-R, SC5-RS
     (v38's frozen policy; a resume counts only where one was stored).
Solvers and files:
  Qwen2.5-Math-7B-Instruct   v38's fresh run: results_Octomber/pretest_v38*.json
                             (A = the 200 clean GSM-Hard rows v40 reuses, M = 60
                             MATH-500 rows); resumes stored for every repair
  DeepSeek-Math-7B-RL        v31's juror samples on 160 P2/GSM-Plus dev rows
  OLMo-2-1124-7B-Instruct    v31's second juror, 16 dev rows
                             (results_Octomber/preset_V31_dev.json; gold from
                             results_September/preset_V25_dev.json); no resumes
                             were ever drawn, so a RESUME decision yields nothing
                             there (a lower bound for SC5-RS)

    python replay_v40.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Dict, List

import cross_model_surgery as X
import trace_siv as T
from replay_v35 import sign_p

V38_FILES = ('results_Octomber/pretest_v38.json', 'results_Octomber/pretest_v38_2nd.json',
             'results_Octomber/pretest_v38_Colab.json')
V31_DEV = 'results_Octomber/preset_V31_dev.json'
V25_DEV = 'results_September/preset_V25_dev.json'


def _read(path: str) -> Dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def qwen_rows() -> Dict[str, List[Dict]]:
    """{'A': rows, 'M': rows}; a row is {'gold', 'C': stored samples in v40's format}."""
    import pretest_v38 as V38
    if not any(os.path.exists(f) for f in V38_FILES):
        return {}
    recs = V38.merged(list(V38_FILES))
    out: Dict[str, List[Dict]] = {'A': [], 'M': []}
    for r in V38.load_rows():
        rec = recs.get(r['key'])
        if not V38.complete(rec):
            continue
        c = []
        for s in rec['C'][:5]:
            op = s.get('surgery') or {}
            res = op.get('resume') or {}
            c.append({'raw': s.get('raw') or '',
                      'resume': {'prefix': op.get('prefix'), 'raw': res.get('raw')} if res.get('raw') else None})
        out[r['level']].append({'pid': r['pid'], 'gold': r['gold'], 'C': c})
    return out


def qwen_prm() -> Dict[str, List[float]]:
    """v38's stored PRM step rewards keyed by text (cross_model_surgery.text_key):
    every sample, propagated repair and resume that v38 scored."""
    import pretest_v38 as V38
    if not any(os.path.exists(f) for f in V38_FILES):
        return {}
    out: Dict[str, List[float]] = {}
    for rec in V38.merged(list(V38_FILES)).values():
        for s in (rec.get('C') or [])[:5]:
            if s.get('prm'):
                out[X.text_key(s.get('raw') or '')] = s['prm']
            op = s.get('surgery') or {}
            for k in ('prop', 'resume'):
                z = op.get(k) or {}
                if z.get('raw') and z.get('prm'):
                    out[X.text_key(z['raw'])] = z['prm']
            # v40 repairs the CLEANED trace, so its propagated text can differ from v38's by
            # markdown alone; it inherits the score v38 gave the same correction
            pz = op.get('prop') or {}
            if pz.get('prm'):
                mine = X.prepare([{'raw': s.get('raw') or ''}])[0].get('surgery')
                if mine and X.text_key(mine['prop']['raw']) not in out:
                    out[X.text_key(mine['prop']['raw'])] = pz['prm']
    return out


def foreign_rows() -> Dict[str, List[Dict]]:
    if not (os.path.exists(V31_DEV) and os.path.exists(V25_DEV)):
        return {}
    d = _read(V25_DEV)
    rows = d['rows'] if isinstance(d['rows'], list) else list(d['rows'].values())
    gold = {(r.get('pid') or r.get('problem_id')): r.get('gold') for r in rows}
    out: Dict[str, List[Dict]] = {}
    for pid, r in _read(V31_DEV)['rows'].items():
        g = gold.get(pid.split(':', 1)[-1])
        if g is None:
            continue
        for name, samples in r['jurors'].items():
            if len(samples) >= 5:
                out.setdefault(name.split('/')[-1], []).append(
                    {'pid': pid, 'gold': float(g), 'C': [{'raw': s.get('raw') or '', 'resume': None}
                                                         for s in samples[:5]]})
    return out


def audit_table(name: str, rows: List[Dict]) -> None:
    c = Counter()
    for r in rows:
        for s in r['C']:
            raw = s['raw']
            ok = X.canonical(X.parse_final(raw), r['gold'])
            old = T.audit(raw)
            new_text = X.clean(raw)
            new = X.audit(new_text)
            c['samples'] += 1
            c['right'] += ok
            c['checked (v38)'] += any(old)
            c['checked (v40)'] += any(new)
            c['refuted (v38)'] += T.refutes(old, None)
            c['refuted (v40)'] += T.refutes(new, None)
            c['refuted RIGHT (v38)'] += T.refutes(old, None) and ok
            c['refuted RIGHT (v40)'] += T.refutes(new, None) and ok
    print(f"  {name:28s} {c['samples']:5d} {c['right']:5d}   "
          f"{c['checked (v38)']:5d} -> {c['checked (v40)']:5d}   {c['refuted (v38)']:4d} -> {c['refuted (v40)']:4d}   "
          f"{c['refuted RIGHT (v38)']:3d} -> {c['refuted RIGHT (v40)']:3d}")


def arms_table(name: str, rows: List[Dict]) -> None:
    prepared = [(r, X.prepare(r['C'])) for r in rows]
    right = {a: [X.canonical(f(c)[0], r['gold']) for r, c in prepared] for a, f in X.ARMS.items()}
    calls = sum(X.sc5_rs(c)[1] for _, c in prepared) / max(1, len(prepared))

    def wl(a, b):
        w = sum(x and not y for x, y in zip(right[a], right[b]))
        l = sum(y and not x for x, y in zip(right[a], right[b]))
        return f"W{w} L{l} (p {sign_p(w, l):.3f})"
    ops = sum(bool(x.get('surgery')) for _, c in prepared for x in c)
    print(f"  {name:28s} {len(rows):4d}  {sum(right['SC5']):4d} {sum(right['SC5-R']):5d} {sum(right['SC5-RS']):6d}   "
          f"RS vs SC5 {wl('SC5-RS', 'SC5'):18s}  RS vs R {wl('SC5-RS', 'SC5-R'):18s}  repairs {ops:3d}  "
          f"calls/row {calls:.2f}")


def main() -> int:
    sys.stdout.reconfigure(encoding='utf-8')
    groups: Dict[str, List[Dict]] = {}
    q = qwen_rows()
    if q:
        groups['Qwen2.5-Math-7B  A (GSM-Hard)'] = q['A']
        groups['Qwen2.5-Math-7B  M (MATH)'] = q['M']
    for name, rows in foreign_rows().items():
        groups[f'{name[:16]} (P2 dev)'] = rows
    if not groups:
        print('no stored samples found (results_Octomber/pretest_v38*.json, preset_V31_dev.json)')
        return 1
    print('=' * 100)
    print('  A. the audit, v38 (raw text, every FALSE) -> v40 (clean + complete)')
    print(f"  {'solver':28s} {'samp':>5s} {'right':>5s}   {'checked':>13s}   {'refuted':>12s}   {'refuted RIGHT':>13s}")
    for name, rows in groups.items():
        audit_table(name, rows)
    print('\n  B. verifier-free arms (canonical grader; right rows out of rows)')
    print(f"  {'solver':28s} {'rows':>4s}  {'SC5':>4s} {'SC5-R':>5s} {'SC5-RS':>6s}")
    for name, rows in groups.items():
        arms_table(name, rows)
    return 0


if __name__ == '__main__':
    sys.exit(main())
