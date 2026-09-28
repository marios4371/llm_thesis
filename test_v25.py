"""[v25.0] Guards for Reading Arbitration. Offline, no models.

What these tests exist to prevent is a comparison that is not the one it
names:
  * RA and B5 not costing the same number of calls;
  * the verifier seeing the Reader's notes (then it is not an independent
    check of the reading) or scoring against anything but the original text;
  * plain samples that are not the prompt of every earlier version, or read
    samples that are not v24's ASQ prompt;
  * dev rows whose B5 / S5 do not reproduce the stored v22 run;
  * confirmation rows that overlap an earlier version's rows;
  * a verdict read off a partial run, or bars that moved after they were frozen.

Run as `python test_v25.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import pretest_v21 as V21
import pretest_v24 as V24
import pretest_v25 as T
import reading_arbitration as RA
import situation_reader as SR

FAILS = []
N = [0]


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def _args(**kw):
    base = dict(dev=False, v24=T.DEV_V24, v22=T.DEV_V22, kc=5, kq=2, stage='all', out='',
                preset='qwen_math7b_mixed', temperature=0.8, max_tokens=1024, limit=8,
                stub=True, resume=True, max_hours=0.0, summary_only=False, pooled=None,
                build_manifest=False, seed=48, force=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _s(answer, last=None, raw=''):
    s = {'raw': raw or f'We compute.\n\nAnswer: {answer}', 'answer': answer}
    if last is not None:
        s['prm'] = [0.9, last]
    return s


def _row(pid, gold, plain, read, tpl=0, tag='main'):
    return {'key': pid, 'pid': pid, 'set': tag, 'gold': gold, 'template': tpl,
            RA.PLAIN: plain, RA.READ: read}


def part1():
    print("\nPART 1 - the arms")
    check(RA.calls('RA') == RA.calls('B5') == 5, "RA and B5 both cost 5 LLM calls")
    check(RA.calls('B4') == 4 and RA.calls('RA5') == 6 and RA.calls('RA7') == 8,
          "B4 / RA5 / RA7 cost 4 / 6 / 8 calls")
    check(RA.ARMS['RA'] == (2, 2, 'best') and RA.ARMS['B5'] == (5, 0, 'best'),
          "PRIMARY is best-of {2 plain, 2 read}; the comparator is best-of 5 plain")
    check((RA.PRIMARY, RA.COMPARATOR) == ('RA', 'B5'), "the primary pair is RA vs B5")
    r = _row('x', 10.0, [_s(3, .5), _s(10, .7), _s(4, .99), _s(4, .2), _s(4, .2)],
             [_s(10, .9), _s(7, .95)])
    ans, w, views = RA.pool(r, 2, 2)
    check(ans == [3, 10, 10, 7] and views == ['C', 'C', 'Q', 'Q'],
          "a pool takes the first plain samples, then the first read samples, in draw order")
    check(RA.choose(r, 'RA') == 7 and RA.chosen_view(r, 'RA') == 'Q',
          "best-of picks the highest last-step reward, whichever view it came from")
    check(RA.choose(r, 'B5') == 4 and not RA.right(r, 'B5'), "B5 reads only plain samples")
    check(RA.choose(r, 'VRA') == 10, "VRA votes over RA's pool")
    tie = _row('t', 1.0, [_s(1, .8), _s(2, .1)], [_s(5, .8), _s(6, .1)])
    check(RA.choose(tie, 'RA') == 1 and RA.chosen_view(tie, 'RA') == 'C',
          "a tie on the reward goes to the plain sample")
    unscored = _row('u', 1.0, [_s(1, .8)] * 5, [_s(1), _s(1, .5)])
    check(not RA.available(unscored, 'RA') and RA.available(unscored, 'B5')
          and RA.available(unscored, 'VRA'),
          "best-of arms need every pooled sample scored; votes do not")
    check(RA.oracle(r, 2, 2) and RA.oracle(r, 0, 1) and not RA.oracle(r, 1, 0), "oracle is 'the gold is in the pool'")
    rd = {'ok': True, 'notes': {'2': [['Q2?', 'a2']], '1': [['Q1?', 'a1'], ['Q1b?', 'b']]}}
    check(RA.note_steps(rd, 3) == ['Q1? a1', 'Q1b? b', 'Q2? a2'],
          "the notes become verifier steps in sentence order (exploratory only)")


def part2():
    print("\nPART 2 - the pre-registered bars and verdicts")
    check((RA.PRIMARY_MIN_NET, RA.PRIMARY_RATIO, RA.PRIMARY_LOTO, RA.GUARD_MIN_NET, RA.DEV_GO_NET)
          == (4, 2, 2, -1, 3), "the bars are the ones frozen on 2026-09-28")
    doc = T.__doc__
    check('frozen 2026-09-28' in doc and 'net >= +4 rows AND W >= 2L' in doc
          and 'RA - B5 >= +3 rows with W >= 2L' in doc and 'RA - B5 >= -1' in doc,
          "the docstring states the same bars")

    def rows(wins, losses, tpl_of_win=lambda i: i):
        out = []
        for i in range(wins):     # RA right, B5 wrong
            out.append(_row(f'w{i}', 1.0, [_s(2, .9)] * 5, [_s(1, .95), _s(1, .95)],
                            tpl=tpl_of_win(i)))
        for i in range(losses):   # B5 right, RA wrong
            out.append(_row(f'l{i}', 1.0, [_s(3, .5), _s(3, .5), _s(1, .9), _s(1, .9), _s(1, .9)],
                            [_s(4, .99), _s(4, .99)], tpl=100 + i))
        for i in range(10):       # both right
            out.append(_row(f'b{i}', 1.0, [_s(1, .9)] * 5, [_s(1, .9)] * 2, tpl=200 + i))
        return out

    r = rows(6, 1)
    check(sum(RA.right(x, 'RA') for x in r) == 16 and sum(RA.right(x, 'B5') for x in r) == 11,
          "the synthetic rows do what they say")
    check(RA.primary_verdict(r).startswith('PRIMARY  SUPPORTED'), "W6 L1 over many templates -> SUPPORTED")
    check(RA.primary_verdict(rows(5, 3)).startswith('PRIMARY  INCONCLUSIVE'),
          "net +2 -> INCONCLUSIVE")
    check(RA.primary_verdict(rows(6, 4)).startswith('PRIMARY  INCONCLUSIVE'),
          "net +2 even with more wins -> INCONCLUSIVE")
    check(RA.primary_verdict(rows(3, 3)).startswith('PRIMARY  REFUTED'), "net 0 -> REFUTED")
    check(RA.primary_verdict(rows(6, 1, lambda i: 0 if i < 4 else i)).startswith(
        'PRIMARY  INCONCLUSIVE'), "a gain that rests on one template is not SUPPORTED")
    check(RA.dev_verdict(rows(5, 1)).startswith('SCREEN   GO'), "dev net +4, W >= 2L -> GO")
    check(RA.dev_verdict(rows(3, 2)).startswith('SCREEN   WEAK'), "dev net +1 -> WEAK")
    check(RA.dev_verdict(rows(2, 2)).startswith('SCREEN   STOP'), "dev net 0 -> STOP")
    g = rows(0, 1)
    check('no harm' in RA.guard_verdict(g) and 'HARM' in RA.guard_verdict(rows(0, 2)),
          "guard: -1 row is no harm, -2 is harm")
    check(abs(RA.sign_p(11, 1) - 0.00635) < 1e-4 and RA.sign_p(0, 0) == 1.0,
          "the exact sign test matches the v22 W11 L1 figure")
    src = RA.gain_source(rows(2, 0) + [_row('s', 1.0, [_s(2, .9), _s(2, .9), _s(2, .1), _s(1, .1),
                                                      _s(2, .9)], [_s(1, .95), _s(2, .1)])])
    check(src == {'reading': 2, 'selection': 1}, "gain source splits reading from selection")


class _Recorder:
    """A verifier that records what it was shown."""

    def __init__(self):
        self.seen = []

    def score(self, problem, steps):
        self.seen.append((problem, list(steps)))
        return [0.5] * len(steps)


class _RecSolver:
    provider = model_name = 'rec'

    def __init__(self):
        self.prompts = []

    def call_model(self, msgs, temperature=0.0, max_tokens=0, **kw):
        self.prompts.append(msgs[-1]['content'])
        return "Step.\n\nAnswer: 7"


def part3():
    print("\nPART 3 - what each agent sees")
    text = ("Ana has 30 marbles. She gives a third of them to Ben. "
            "How many marbles does Ana have now?")
    reading = ("S1: Q: How many marbles does Ana start with? A: 30.\n"
               "S2: Q: How many does she give? A: A third of 30 = 10.\n"
               "S3: -\nAsked: the marbles Ana has left.")
    rec = {'key': 'k', 'pid': 'p', 'set': 'main', 'gold': 20.0, 'template': None, 'text': text,
           RA.PLAIN: [], RA.READ: []}
    solver = _RecSolver()

    class _Reader:
        def call_model(self, msgs, **kw):
            return reading

    V24._READS.update(n=0, ok=0, shown=True)
    ok = T.generate(solver, _Reader(), rec, _args(), 0.0)
    check(ok and len(rec[RA.PLAIN]) == 5 and len(rec[RA.READ]) == 2 and rec['reading']['ok'],
          "one row draws 5 plain samples, one reading, 2 read samples")
    check(solver.prompts[:5] == [SR.cot_prompt(text)] * 5 and SR.cot_prompt(text) == V21.COT_PROMPT.format(problem=text),
          "plain samples get exactly the CoT prompt of every earlier version")
    check(solver.prompts[5:] == [SR.asq_prompt(text, rec['reading'])] * 2
          and SR.ASQ_INTRO in solver.prompts[5],
          "read samples get v24's ASQ prompt with this row's reading")
    ver = _Recorder()
    T.score(ver, rec)
    problems = {p for p, _ in ver.seen}
    check(problems == {text}, "the verifier scores everything against the ORIGINAL problem text")
    sample_steps = [s for _, st in ver.seen[:7] for s in st]
    check(not any(SR.ASQ_INTRO in s or '(Q:' in s for s in sample_steps),
          "the verifier's input for a sample is the sample, never the notes block")
    check(all(s.get('prm') is not None for v in (RA.PLAIN, RA.READ) for s in rec[v])
          and 'reading_prm' in rec, "every sample and the reading are scored")
    n = len(ver.seen)
    T.score(ver, rec)
    check(len(ver.seen) == n, "a scored row is never re-scored")
    check(all(k not in rec for k in ('drift',)), "a confirmation row is never drift-checked")


def part4():
    print("\nPART 4 - dev rows reproduce the stored runs")
    rows = T.load_dev()
    main = [r for r in rows if r['set'] == 'main']
    guard = [r for r in rows if r['set'] == 'guard']
    check(len(main) == 100 and len(guard) == 20, "100 main + 20 guard dev rows")
    check(sum(RA.right(r, 'B5') for r in main) == 78 and sum(RA.right(r, 'S5') for r in main) == 68,
          "B5 = 78 and S5 = 68 on dev main, as the v22 run stored them")
    check(sum(RA.right(r, 'B5') for r in guard) == 18, "B5 = 18/20 on dev guard")
    check(sum(RA.right(r, 'SQ5') for r in main) == 66, "v24's ASQ SC@5 = 66 reproduces")
    check(sum(RA.oracle(r, 2, 2) for r in main) == 88 and sum(RA.oracle(r, 5, 0) for r in main) == 82,
          "the motivating oracle numbers: C2+Q2 88 vs C5 82")
    check(all(len(r[RA.READ]) == 5 and all('prm' not in s for s in r[RA.READ]) for r in rows),
          "the read samples arrive unscored: no reader-view verifier score exists yet")
    check(all(s.get('prm') for r in rows for s in r[RA.PLAIN]), "every plain sample carries its v22 score")


def part5():
    print("\nPART 5 - the runner")
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 'dev.json')
        rc = T.run(_args(dev=True, out=out, limit=6), dev=True)
        data = V24._read(out)
        check(rc == 0 and data['mode'] == 'dev' and len(data['rows']) == 6,
              "a stub dev pass writes where it is told")
        check(all(s.get('prm') for r in data['rows'] for s in r[RA.READ]),
              "the dev pass scores every read sample")
        check(T.OUT_STUB not in (T.OUT, T.OUT_DEV), "a stub run has its own default output file")
        seen = []
        orig = T.score
        T.score = lambda *a, **k: seen.append(1)
        T.run(_args(dev=True, out=out, limit=6), dev=True)
        T.score = orig
        check(not seen, "re-running the dev pass resumes: nothing is scored twice")

        out = os.path.join(d, 'conf.json')
        T.run(_args(out=out, stage='gen'), dev=False)
        data = V24._read(out)
        check(all(len(r[RA.PLAIN]) == 5 and len(r[RA.READ]) == 2 for r in data['rows'])
              and not any('prm' in s for r in data['rows'] for s in r[RA.PLAIN]),
              "--stage gen draws everything and scores nothing")
        T.run(_args(out=out, stage='score'), dev=False)
        data = V24._read(out)
        check(all(T.complete(r, False) for r in data['rows']), "--stage score then completes every row")
        before = json.dumps(data['rows'], sort_keys=True)
        T.run(_args(out=out), dev=False)
        check(json.dumps(V24._read(out)['rows'], sort_keys=True) == before,
              "a finished run re-run changes nothing")

        out = os.path.join(d, 'cut.json')
        T.run(_args(out=out, max_hours=-1.0, limit=8), dev=False)
        data = V24._read(out)
        check(not all(T.complete(r, False) for r in data['rows']),
              "--max-hours stops a run part-way")
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T.summarise(data['rows'], dev=False, stub=False)
        check('PARTIAL run' in buf.getvalue() and 'PRIMARY' not in buf.getvalue(),
              "no verdict is read off a partial run")
        T.run(_args(out=out, limit=8), dev=False)
        check(all(T.complete(r, False) for r in V24._read(out)['rows']),
              "the same command resumes it to completion")


def part6():
    print("\nPART 6 - the confirmation rows")
    ok = all(os.path.exists(p) for _, p in T.MANIFESTS)
    check(ok, "the v25 manifests exist (built once, committed before the run)")
    if not ok:
        return
    m = V24._read(T.MANIFESTS[0][1])
    g = V24._read(T.MANIFESTS[1][1])
    check(m['seed'] == 48 == T.CONFIRM_SEED and m['n'] == 100 and g['n'] == 20,
          "100 main + 20 guard rows at seed 48")
    ids = {r['problem_id'] for r in m['rows'] + g['rows']}
    used = set()
    for p in T.EARLIER_MANIFESTS:
        if os.path.exists(p):
            used |= {r['problem_id'] for r in V24._read(p)['rows']}
    used |= {r['pid'] for r in V24._read(T.DEV_V22)['rows']}
    check(not ids & used, "no confirmation row was drawn by v20-v24")
    check(all(r['dataset'] == 'gsm-symbolic:p2' for r in m['rows'])
          and all(r['dataset'].startswith('gsm-plus') for r in g['rows']),
          "main rows are P2, guard rows are GSM-Plus distractor rows")
    check(T.build_manifests(48, 100, 20, force=False) == 1, "the manifests refuse to be overwritten")


def main() -> int:
    for p in (part1, part2, part3, part4, part5, part6):
        p()
    print(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed")
    if FAILS:
        print("FAILED:\n  " + "\n  ".join(FAILS))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
