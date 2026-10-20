"""[v38.0] Guards for Symbolic Surgery (trace_surgery.py, pretest_v38.py). Offline, no GPU.

What these tests exist to prevent:
  * a repair that rewrites what the audit would never refute (run-on chains,
    claims the step itself calls false), corrupts exponents, or cuts the
    resume prefix anywhere but right after the first false claim's '=';
  * a correction that does not travel: chains, later left sides, the answer;
  * ADAPT-RS that is not ADAPT-R when the policy only refutes, or that charges
    a resumed sample twice;
  * a manifest that moved, overlaps v34/v35/v36 or the v18/v19 rows, or is not
    interleaved; bars or budgets that moved after freezing;
  * a GPU run before the policy is written, or with a policy other than the
    one frozen on 2026-10-20 (the notebooks run this file first and stop);
  * a resume path that fails silently (one failure is recorded, more than
    MAX_ERRORS stop the run; the smoke test runs before any row);
  * a notebook that starts the GPU before this file, or puts a token in a URL.

Run as `python test_v38.py`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pretest_v38 as V
import trace_siv as T
import trace_surgery as S

FAILS = []
N = [0]


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


CHAIN = ("First add the apples:\n\\[ 61 + 8 + 13 + 10 + 58 = 140 \\]\n\nThen double it:\n"
         "\\[ 140 \\times 2 = 280 \\]\n\nSo she has 280 apples. Then 280 + 20 = 300.\n\nAnswer: 300")


def part1():
    print("\nPART 1 - locate, propagate, cut")
    w = S.walk(CHAIN)
    check(w['fixes'] == [(140.0, 150.0), (280.0, 300.0), (300.0, 320.0)],
          "the correction travels: a later left side and a plain-text claim are re-evaluated")
    p = S.propagate(CHAIN, 300.0)
    check(p['answer'] == 320.0 and p['reached'], "the final answer is mapped (reached)")
    check('= 150' in p['raw'] and '150 \\times 2 = 300' in p['raw'] and 'Answer: 320' in p['raw'],
          "the repaired text carries every correction")
    check('she has 280 apples' in p['raw'], "prose is left as it is (the documented blind spot)")
    check(not T.refutes(T.audit(p['raw']), None), "the repaired text passes the audit")
    pre = S.resume_prefix(CHAIN)
    check(pre.endswith('58 = 150') and pre.startswith('First add'), "resume prefix = trace up to '=' + CAS value")
    check(S.first_false(CHAIN)['cut'] == CHAIN.index('= 140') + 1, "cut sits right after the relation sign")
    check(S.propagate("\\[ 361/400 = 0.9025 \\times 100 = 90.25 \\]\n\nAnswer: 90.25", 90.25) is None,
          "a run-on chain is not repaired (the audit does not refute it)")
    check(S.propagate("Wait, \\( 77 + 13 + 702 = 892 \\) is a mistake.\n\n\\( 77+13+702 = 792 \\)\n\nAnswer: 792",
                      792.0) is None, "a claim the step calls a mistake is not repaired")
    u = S.propagate("\\[ 5573040 \\text{ bales} - 18 \\text{ bales} = 5572922 \\text{ bales} \\]\n\n"
                    "The answer is \\boxed{5572922}", 5572922.0)
    check(u is not None and u['answer'] == 5573022.0 and '\\boxed{5573022}' in u['raw'],
          "a unit-bearing slip is repaired and the boxed answer rewritten")
    e = S.propagate("\\[ 1 + 1 = 3 \\]\n\n\\[ 3^3 + 3 = 30 \\]\n\nAnswer: 30", 30.0)
    check(e is not None and '2^3' in e['raw'] and '^2' not in e['raw'],
          "a corrected value is substituted as a base, never as an exponent")
    x = S.propagate("\\[ 131 + 64 + 55 \\times 11 = 131 + 64 + 565 = 760 \\]\n\nAnswer: 760", 760.0)
    check(x is not None and x['answer'] == 800.0 and not T.refutes(T.audit(x['raw']), None),
          "an expression on the right side is replaced by the value, and the chain carries it")
    check(S.fmt(150.0) == '150' and S.fmt(0.25) == '0.25' and 'e' not in S.fmt(1.8386e-05), "fmt")
    check(S.propagate("\\[ 2 + 2 = 4 \\]\n\nAnswer: 4", 4.0) is None, "a true trace has nothing to repair")


def _sample(ans, prm, false=False, prop_ans=None, reached=True, slip=(140.0, 150.0), resume_ans=None):
    x = {'answer': ans, 'prm': prm, 'audit': [[('a = b', T.FALSE if false else T.TRUE)]]}
    if false:
        x['surgery'] = {'site': {'claimed': slip[0], 'value': slip[1], 'claim': 'c', 'step': 0},
                        'slips': [slip],
                        'prop': {'answer': prop_ans, 'prm': prm, 'audit': [], 'reached': reached},
                        'resume': (None if resume_ans is None else {'answer': resume_ans, 'prm': prm, 'audit': []})}
    return x


def part2():
    print("\nPART 2 - the rule")
    c = [_sample(10.0, [0.99, 0.99], True, prop_ans=11.0), _sample(12.0, [0.99, 0.95]),
         _sample(13.0, [0.9, 0.9]), _sample(14.0, [0.5, 0.5]), _sample(15.0, [0.5, 0.5])]
    for x in c:
        x['audit'] = x['audit'] if x.get('surgery') else []
    refute = lambda s, n: S.REFUTE                                       # noqa: E731
    check(S.adapt_rs(c, refute) == T.adapt_r(c), "ADAPT-RS with a refute-only policy IS ADAPT-R")
    check(S.adapt_rs(c, lambda s, n: S.PROPAGATE) == (11.0, 3), "a propagated repair competes and can win the gate")
    c[0]['surgery']['resume'] = {'answer': 11.0, 'prm': [0.99, 0.99], 'audit': []}
    check(S.adapt_rs(c, lambda s, n: S.RESUME) == (11.0, 4), "a resume costs one call")
    c[0]['surgery']['resume']['prm'] = [0.5, 0.5]
    c[1]['prm'] = [0.5, 0.5]
    c[2]['prm'] = [0.5, 0.5]
    check(S.adapt_rs(c, lambda s, n: S.RESUME)[1] == 6, "a sample resumed at the gate is not charged again")
    c[0]['surgery']['prop']['audit'] = [[('x', T.FALSE)]]
    check(S.repaired(c[0], S.PROPAGATE) is None, "a repair the audit still refutes is dropped")
    try:
        S.adapt_rs(c, lambda s, n: 'maybe')
        check(False, "an unknown action is an error")
    except ValueError:
        check(True, "an unknown action is an error")
    d = [_sample(1.0, [0.9], True), _sample(2.0, [0.9], True), _sample(3.0, [0.9], True, slip=(7.0, 8.0))]
    check([S.n_sharing(d, i, range(3)) for i in range(3)] == [2, 2, 1], "n_sharing counts a shared slip")
    check((T.GATE_K, T.FULL_K, T.GATE_MIN) == (3, 5, 0.85), "v22's gate")


def part3():
    print("\nPART 3 - the manifest, the bars, the budgets")
    d = json.load(open(V.MANIFEST, encoding='utf-8'))
    rows = d['rows']
    ids = [r['problem_id'] for r in rows]
    check(d['seed'] == 38 and d['n'] == 260 and len(set(ids)) == 260, "seed 38, 260 distinct rows")
    a = [r for r in rows if r['level'] == 'A']
    m = [r for r in rows if r['level'] == 'M']
    check((len(a), len(m)) == (200, 60), "A/M = 200/60")
    check(all(r['problem_id'].startswith('gsm-hard_') and r['gold'] > 0 and V.V36.max_number(r['text']) >= 1e4
              for r in a), "A: GSM-Hard, gold > 0, max number >= 1e4")
    check(all(r['problem_id'].startswith('math500_') for r in m), "M: MATH-500")
    check(not (set(ids) & (V._ids(V.USED_A) | V._ids(V.USED_M))), "no row of v34, v35, v36, v18/v19")
    lv = [r['level'] for r in rows]
    check('M' in lv[:10] and 'M' in lv[-10:] and max(len(s) for s in ''.join(lv).split('M')) <= 4,
          "interleaved: an M row near each end, never more than 4 A rows in a row")
    try:
        au = json.load(open('gsmhard_audit_rows.json', encoding='utf-8'))['rows']
        bad = set(au['collision']) | set(au['never_perturbed']) | set(au['platinum_dropped'])
        aid = {r['problem_id'] for r in a}
        check(not (aid & bad) and aid <= set(au['auditable']), "A: every row auditable and clean")
    except FileNotFoundError:
        check(True, "(gsmhard_audit_rows.json absent in this checkout: skipped)")
    check((V.PRI_MIN_NET, V.PRI_RATIO, V.PRI_ALPHA, V.SEC_MIN_NET) == (3, 2, 0.05, -1), "bars frozen")
    check(V.MAX_TOKENS == {'A': 1536, 'M': 2048} and V.K == 5 and V.TEMPERATURE == 0.8, "budgets and sampling")
    check(V.canonical(5573022.0, 5573022.0) and not V.canonical(5573122.0, 5573022.0),
          "the canonical grader (an answer off by 100 on a 7-digit gold is wrong)")


def part4():
    print("\nPART 4 - offline plumbing (stub run, temp file)")
    out = os.path.join(tempfile.mkdtemp(), 'stub38.json')
    r = subprocess.run([sys.executable, 'pretest_v38.py', '--stub', '--limit', '5', '--out', out],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    check(r.returncode == 0 and os.path.exists(out), "the stub run completes")
    if os.path.exists(out):
        recs = json.load(open(out, encoding='utf-8'))['rows']
        ops = [s['surgery'] for rec in recs.values() for s in rec['C'] if s.get('surgery')]
        check(len(recs) == 5 and all(V.complete(rec) for rec in recs.values()), "5 complete records")
        check(ops and all(o['prop'].get('prm') and o['resume'] and o['resume'].get('prm') for o in ops),
              "every operated sample has a re-scored propagation and a re-scored resume")
        check(all(o['resume']['raw'].startswith(o['prefix']) for o in ops), "the resume continues the prefix")


def part5():
    print("\nPART 5 - the notebooks")
    for nb, run in (('MAS_SHT_Kaggle_v38.ipynb', 'pretest_v38.py --max-hours 8.5'),
                    ('MAS_SHT_Colab_v38.ipynb', "'pretest_v38.py', '--reverse'")):
        cells = json.load(open(nb, encoding='utf-8'))['cells']
        ok = True
        for c in cells:
            if c['cell_type'] != 'code':
                continue
            src = '\n'.join(('pass  # ' + l) if l.lstrip().startswith('!') else l
                            for l in ''.join(c['source']).split('\n'))
            try:
                compile(src, nb, 'exec')
            except SyntaxError as e:
                ok = False
                print(f"      {nb}: {e}")
        check(ok, f"every code cell of {nb} compiles")
        code = '\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'code')
        t, r = code.find("'test_v38.py'"), code.find(run)
        check(t >= 0 and r >= 0 and t < r, f"{nb} runs test_v38.py before `{run}`")
        check(not any(f'v{k}' in code for k in range(34, 38)), f"{nb} code cells mention no recent older version")
        check('_tok}@' not in code and '_token}@' not in code and '@github.com' not in code,
              f"{nb} never puts a token in a URL")


def part6():
    print("\nPART 6 - the policy is written, and it is the one frozen on 2026-10-20")
    ok = V.policy_written()
    check(ok, "trace_surgery.surgery_policy() returns an action (V38_PLAN.md section 3)")
    if ok:
        probes = {r: _sample(1.0, [0.9], True, prop_ans=2.0, reached=r, resume_ans=2.0) for r in (True, False)}
        for x in probes.values():
            x['surgery']['site'].update(rhs_literal=True, lhs='1 + 1', cut=0)
        check(all(S.surgery_policy(probes[True], n) == S.PROPAGATE for n in (1, 2, 5)),
              "frozen: the correction reached the answer -> propagate")
        check(all(S.surgery_policy(probes[False], n) == S.RESUME for n in (2, 3, 5)),
              "frozen: not reached and the slip recurs (n >= 2) -> resume")
        check(S.surgery_policy(probes[False], 1) == S.REFUTE, "frozen: not reached and isolated (n = 1) -> refute")


def part7():
    print("\nPART 7 - the run's error guards")
    keep, V._ERRORS[:] = V.resume_call, []

    def boom(*a, **k):
        raise ValueError('generation failed')
    V.resume_call = boom
    try:
        s = {'raw': CHAIN, 'answer': 300.0, 'prm': [0.9, 0.9, 0.9, 0.9]}
        stub = V._StubSolver([{'text': 'x', 'gold': 1.0}])
        V.operate_sample(s, 'problem', stub, V.V34._StubPRM([{'text': 'x', 'gold': 1.0}]), 64)
        check(s['surgery'] and s['surgery']['resume'] is None and 'generation failed' in s['surgery']['resume_error']
              and s['surgery']['prop']['prm'], "a failing resume is recorded and the propagated repair is kept")
        try:
            for _ in range(V.MAX_ERRORS + 1):
                fresh = {k: v for k, v in s.items() if k != 'surgery'}
                V.operate_sample(fresh, 'problem', stub, V.V34._StubPRM([{'text': 'x', 'gold': 1.0}]), 64)
            check(False, f"more than {V.MAX_ERRORS} errors stop the run")
        except RuntimeError:
            check(True, f"more than {V.MAX_ERRORS} errors stop the run")
    finally:
        V.resume_call, V._ERRORS[:] = keep, []
    V.smoke_resume(V._StubSolver([{'text': 'x', 'gold': 1.0}]))
    check(True, "the resume smoke test passes on the stub")

    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            return {'input_ids': list(range(len(text)))}             # one token per character

    class _Client:
        _local_tokenizer = _Tok()

        def _ensure_local_model(self):
            pass
    check(V.remaining_budget(_Client(), 'x' * 1000, 1536) == 536 and
          V.remaining_budget(_Client(), 'x' * 2000, 2048) == V.MIN_RESUME_TOKENS,
          "a continuation gets the rest of the row's budget (prefix + new <= budget), at least the floor")
    check(V.remaining_budget(object(), 'x', 1536) == 1536, "no tokenizer: the full budget")


def main():
    part1()
    part2()
    part3()
    part4()
    part5()
    part6()
    part7()
    print(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed" + (f"; FAILED: {FAILS}" if FAILS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
