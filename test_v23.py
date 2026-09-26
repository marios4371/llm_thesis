"""[v23.0] Guards for Verifier-guided Tabu Restarts. Offline, no models.

What these tests exist to prevent is an arm that measures something other
than the mechanism it names:
  * a TABU restart that is not the RST prompt plus one note (then TABU vs RST
    is not a comparison of the note);
  * a note that leaks more of the failed solution than the one rejected step
    (then it is self-refinement, the anchoring VTR is defined against);
  * a leader that is not B3's choice (then "the answer we would have
    output" is not what the Verifier doubted);
  * a TAU or leader rule that silently changed after it was frozen.

Run as `python test_v23.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import pretest_v21 as V21
import pretest_v23 as V23
import score_prm_v22 as P
import tabu_restart as T

FAILS = []
N = [0]
HERE = os.path.dirname(os.path.abspath(__file__))


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def s(answer, prm, raw=''):
    return {'answer': answer, 'prm': prm, 'raw': raw}


RAW = ("To determine the answer, we need to follow these steps:\n\n"
       "1. **Find the wasps:** \\[ \\frac{2}{4} \\times 14 = 7 \\]\n\n"
       "2. **Count the animals:** \\[ 43 + 7 + 14 + 8 = 72 \\]\n\n"
       "Answer: 11.11")


def part1():
    print("\nPART 1 - the tabu keeper's decisions")
    ss = [s(None, [0.99]), s(5.0, [0.9, 0.97]), s(6.0, [0.99, 0.97])]
    check(T.leader(ss) == 1, "an unparsed draft is never the leader; ties go to the earliest")
    check(T.leader([s(None, [1.0])] * 3) is None, "no parsed draft -> no leader")
    check(T.leader(ss + [s(9.0, [0.1, 1.0])]) == 1, "only the first k=3 drafts compete")

    check(not T.substantive("To determine how many students leave, we follow these steps:"),
          "an introduction is not substantive")
    check(not T.substantive("1. Find the swimmers. 2. Find the others. 3. Add them."),
          "a numbered plan is not substantive")
    for step in ("\\[ 3 \\times 4 = 12 \\]", "the wasps are \\frac{2}{4} of the bees",
                 "So the total is = 72", "20 - 5 leaves 15"):
        check(T.substantive(step), f"substantive: {step!r}")

    steps = T.split_steps(RAW)
    cut, step = T.doubted_step(steps, [0.10, 0.99, 0.60, 0.99])
    check(cut == 2 and '72' in step,
          "the lowest-rewarded SUBSTANTIVE step is doubted, not the lower-rewarded intro")
    cut, step = T.doubted_step(steps[:2], [0.5, 0.9, 0.1])
    check(cut == 1, "rewards beyond the stored steps (a truncated trace) are ignored")

    p = T.plan([s(11.11, [0.99, 0.999, 0.999, 1.0], RAW)] * 3)
    check(not p['triggered'], "a leader with every step >= TAU is not triggered")
    p = T.plan([s(11.11, [0.99, 0.999, 0.90, 1.0], RAW)] * 3)
    check(p['triggered'] and p['cut'] == 2 and p['fallback'] is None,
          "a leader with a step < TAU is triggered, at that step")
    p = T.plan([s(3.0, [0.2, 1.0], "We follow these steps:\n\nAnswer: 3")] * 3)
    check(p['triggered'] and p['fallback'] == 'answer' and
          T.notes_for(p)['TABU'] == T.answer_note(3.0),
          "no computing step to forbid -> TABU degrades to the answer note, and says so")
    p = T.plan([s(None, [0.2])] * 3)
    check(p['triggered'] and p['fallback'] == 'no-leader' and T.notes_for(p)['TABU'] == '',
          "no parsed draft -> triggered, and the restarts are plain")

    check(T.restart_prompt('PROB') == V21.COT_PROMPT.format(problem='PROB'),
          "a restart with no note IS the RST/CoT prompt, byte for byte")
    note = T.tabu_note(['\\[ 43 + 7 + 14 + 8 = 72 \\]'])
    pr = T.restart_prompt('PROB', note)
    check(pr == V21.COT_PROMPT.format(problem='PROB\n\n' + note),
          "a TABU restart is the RST prompt plus the note, where V2P put its note")
    check(T.TABU_MARKER in note and '= 72' in note and T.DO_NOT in note,
          "the TABU note names the step as WRONG and asks for a fresh solve")
    two = T.tabu_note(['a = 1', 'b = 2'])
    check(T.TABU_MARKER_MULTI in two and '1. """' in two and '2. """' in two,
          "a tabu list of two steps is numbered")
    check(T.ANS_MARKER in T.answer_note(12.0) and '12,' in T.answer_note(12.0)
          and '12.5' in T.answer_note(12.5), "the ANS note carries the answer, formatted")
    check(T.answer_note(None) == '', "no answer -> no ANS note")

    long = 'x = 1\n\n\n' + 'y' * 3000 + '\nz = 2'
    c = T.compact(long)
    check(len(c) <= T.MAX_STEP_CHARS + 10 and '[...]' in c and c.endswith('z = 2')
          and '\n\n' not in c, "a long step is cut in the middle, keeping its conclusion")

    prob = "Beatriz saw 43 ants and 14 bees."
    check(T.doubted_value("43 + 14 = 57", prob) == 57.0, "the doubted value is the step's conclusion")
    check(T.doubted_value("there are 43 ants", prob) is None, "a stated given is not a conclusion")
    check(T.contains("so 57 animals", 57.0) is True and T.contains("x", None) is None,
          "value lookup")


def part2():
    print("\nPART 2 - the stored pool and the frozen numbers")
    pool = V23.load_pool()
    by = lambda src, tag: [r for r in pool if r['source'] == src and r['set'] == tag]
    check(len(pool) == 260 and len(by('v21', 'main')) == 100 and len(by('v21', 'guard')) == 40
          and len(by('v22', 'main')) == 100 and len(by('v22', 'guard')) == 20,
          "pool: 100+40 v21 rows, 100+20 v22 rows")
    check(all(len(r['samples']) == 5 and all(x['prm'] for x in r['samples']) for r in pool),
          "every stored row has 5 scored samples")
    check(len({r['key'] for r in pool}) == 260, "row keys are unique")
    same = all(r['samples'][T.leader(r['samples'])]['answer'] ==
               P.best_of_n([x['answer'] for x in r['samples'][:3]],
                           [T.last(x['prm']) for x in r['samples'][:3]])
               for r in pool)
    check(same, "the leader's answer IS B3's answer on all 260 rows")

    trig = lambda rows: sum(1 for r in rows if T.plan(r['samples'])['triggered'])
    check(T.TAU == 0.95, "TAU is the pre-registered 0.95")
    check(trig(by('v21', 'main') + by('v21', 'guard')) == 46
          and trig(by('v22', 'main') + by('v22', 'guard')) == 34,
          "at TAU, 46 v21 rows and 34 v22 rows are triggered (80 to generate)")

    recs = []
    for r in pool:
        rec, _ = V23.new_record(r, r['samples'], T.TAU)
        V23.arms_for(rec, [])
        recs.append(rec)
    v22m = [r for r in recs if r['source'] == 'v22' and r['set'] == 'main']
    n = lambda rows, a: sum(1 for r in rows if r[f'{a}_correct'])
    check(n(v22m, 'B5') == 78 and n(v22m, 'S5') == 68 and n(v22m, 'S3') == 66
          and n(v22m, 'B3') == 73, "v22 main reproduces V22_RESULTS.md: S3 66 S5 68 B3 73 B5 78")
    check(n(v22m, 'RST') == 77, "RST at TAU on v22 main: 77")

    # information flow: the note carries the one rejected step and nothing else
    leak = 0
    for r in pool:
        p = T.plan(r['samples'])
        if not p['triggered']:
            continue
        note = T.notes_for(p)['TABU']
        steps = T.split_steps(r['samples'][p['leader']]['raw'])
        for i, st in enumerate(steps):
            if i != p['cut'] and len(st) > 60 and st not in p['doubted'] and st[:60] in note:
                leak += 1
        if T.compact(p['doubted']) not in note:
            leak += 1
    check(leak == 0, "on all 80 triggered rows the TABU note holds the doubted step and no other")
    check(all(len(T.notes_for(T.plan(r['samples']))['TABU']) < 1000 for r in pool
              if T.plan(r['samples'])['triggered']), "no note exceeds 1000 characters")


def _rec(drafts, triggered, extra=None, gold=1.0, source='v22', tag='main'):
    rec = {'key': f'{source}:x', 'pid': 'x', 'source': source, 'set': tag, 'gold': gold,
           'triggered': triggered, 'leader_answer': drafts[0][0], 'cut': 1, 'doubted': 'a = 2',
           'drafts': [{'answer': a, 'last': w, 'min': w} for a, w in drafts],
           'extra': extra or {}, 'rst': []}
    return rec


def part3():
    print("\nPART 3 - arms and the pre-registered reading")
    d = [(2.0, 0.9), (3.0, 0.5), (2.0, 0.4), (1.0, 0.95), (4.0, 0.2)]
    r = _rec(d, False)
    V23.arms_for(r, ['TABU', 'ANS'])
    check(all(r[f'{a}_answer'] == r['B3_answer'] == 2.0 for a in ('RST', 'TABU', 'ANS', 'MIX')),
          "untriggered: every repair arm IS B3 (no extra samples)")
    ex = {'TABU': {'samples': [{'answer': 1.0, 'prm': [0.99]}, {'answer': 5.0, 'prm': [0.1]}]},
          'ANS': {'samples': [{'answer': 7.0, 'prm': [0.3]}, {'answer': 8.0, 'prm': [0.2]}]}}
    r = _rec(d, True, ex)
    V23.arms_for(r, ['TABU', 'ANS'])
    check(r['RST_answer'] == 1.0 and r['RST_correct'], "RST = best of the 5 drafts")
    check(r['TABU_answer'] == 1.0 and r['ANS_answer'] == 2.0,
          "TABU/ANS = best of the 3 drafts + their own 2 restarts")
    check(r['MIX_answer'] == 1.0, "MIX = best of 3 drafts + RST sample 4 + TABU sample 1")
    check(r['C1_answer'] == 2.0 and r['S3_answer'] == 2.0, "C1 is draft 1, S3 the plurality")
    r2 = _rec(d, True, {'TABU': ex['TABU']})
    V23.arms_for(r2, ['TABU', 'ANS'])
    check(not V23.complete(r2, ['TABU', 'ANS']) and V23.complete(r2, ['TABU']),
          "a triggered row missing an arm is incomplete, and is left out of the reading")

    main = [{'TABU_correct': True, 'RST_correct': False, 'ANS_correct': False}] * 4 + \
           [{'TABU_correct': False, 'RST_correct': True, 'ANS_correct': True}] * 1
    v = V23.verdicts(main, [], ['TABU', 'ANS'], fresh=False)
    check(v[0].startswith('PRIMARY  SUPPORTED') and 'net +3' in v[0], "net +3, W=4 L=1 -> SUPPORTED")
    check(v[1].startswith('ATTRIB   the verifier'), "TABU - ANS = +3 -> localisation matters")
    main = [{'TABU_correct': True, 'RST_correct': False}] * 4 + \
           [{'TABU_correct': False, 'RST_correct': True}] * 2
    v = V23.verdicts(main, [], ['TABU'], fresh=False)
    check(v[0].startswith('PRIMARY  INCONCLUSIVE'), "net +2 -> INCONCLUSIVE")
    main = [{'TABU_correct': True, 'RST_correct': False}] * 5 + \
           [{'TABU_correct': False, 'RST_correct': True}] * 3
    v = V23.verdicts(main, [], ['TABU'], fresh=False)
    check(v[0].startswith('PRIMARY  INCONCLUSIVE'), "net +2 with W < 2L -> INCONCLUSIVE")
    main = [{'TABU_correct': True, 'RST_correct': False}] * 6 + \
           [{'TABU_correct': False, 'RST_correct': True}] * 3
    v = V23.verdicts(main, [], ['TABU'], fresh=False)
    check(v[0].startswith('PRIMARY  SUPPORTED'), "net +3 with W = 2L -> SUPPORTED (the bar is inclusive)")
    main = [{'TABU_correct': False, 'RST_correct': False}] * 10
    v = V23.verdicts(main, [{'TABU_correct': False, 'B3_correct': True}] * 2, ['TABU'], fresh=False)
    check(v[0].startswith('PRIMARY  REFUTED') and 'HARM' in v[1], "net 0 -> REFUTED; guard -2 -> HARM")


def _args(**kw):
    base = dict(dev=False, arms='TABU,ANS', tau=T.TAU, out='', preset='qwen_math7b_mixed',
                temperature=0.8, max_tokens=64, limit=0, stub=True, resume=True,
                max_hours=0.0, summary_only=False, build_manifest=False, seed=46, force=False)
    base.update(kw)
    return argparse.Namespace(**base)


def part4():
    print("\nPART 4 - the runners, offline")
    tmp = tempfile.mkdtemp()
    out = os.path.join(tmp, 'dev.json')
    r = subprocess.run([sys.executable, 'pretest_v23.py', '--stub', '--dev', '--limit', '8',
                        '--arms', 'TABU', '--out', out], cwd=HERE, capture_output=True, text=True)
    check(r.returncode == 0 and os.path.exists(out), "stub dev pass (TABU only) runs")
    first = {x['key']: x for x in json.load(open(out))['rows']}
    r = subprocess.run([sys.executable, 'pretest_v23.py', '--stub', '--dev', '--limit', '8',
                        '--arms', 'TABU,ANS', '--out', out], cwd=HERE, capture_output=True, text=True)
    second = {x['key']: x for x in json.load(open(out))['rows']}
    trig = [k for k, x in first.items() if x['triggered']]
    check(r.returncode == 0 and trig and all(
        second[k]['extra']['TABU'] == first[k]['extra']['TABU'] and 'ANS' in second[k]['extra']
        for k in trig), "adding an arm on resume keeps the TABU restarts and only draws ANS")
    check('PRIMARY' in r.stdout and 'GUARD' in r.stdout, "the dev pass prints the pre-registered reading")
    r = subprocess.run([sys.executable, 'pretest_v23.py', '--dev', '--summary-only', '--out', out,
                        '--tau', '0.9'], cwd=HERE, capture_output=True, text=True)
    check(r.returncode == 0 and 'PRIMARY' in r.stdout,
          "--summary-only re-reads a file with the TAU stored in it, not the flag")

    # fresh confirmation, on temporary manifests
    pool = V23.load_pool()
    mk = lambda rows: {'rows': [{'problem_id': x['pid'], 'text': x['text'], 'gold': x['gold'],
                                 'dataset': 'x'} for x in rows]}
    paths = []
    for tag, rows in (('main', [x for x in pool if x['set'] == 'main'][:4]),
                      ('guard', [x for x in pool if x['set'] == 'guard'][:2])):
        p = os.path.join(tmp, f'{tag}.json')
        json.dump(mk(rows), open(p, 'w'))
        paths.append((tag, p))
    saved = V23.MANIFESTS
    V23.MANIFESTS = tuple(paths)
    try:
        cout = os.path.join(tmp, 'confirm.json')
        rc = V23.run_confirm(_args(out=cout, max_hours=1e-12), ['TABU', 'ANS'])
        part = json.load(open(cout))['rows']
        check(rc == 0 and len(part) == 0, "a session out of time draws nothing and reads no verdict")
        rc = V23.run_confirm(_args(out=cout, arms='TABU'), ['TABU'])
        full = json.load(open(cout))['rows']
        check(rc == 0 and len(full) == 6 and all('TABU_answer' in x for x in full),
              "confirmation: 6 rows, 5 drafts each, TABU where triggered")
        check(all(len(x['samples']) == 5 for x in full), "confirmation keeps the 5 drafts it drew")
        rc = V23.run_confirm(_args(out=cout, max_hours=1e-12), ['TABU', 'ANS'])
        kept = json.load(open(cout))['rows']
        check(len(kept) == 6 and all(x['samples'] == y['samples'] for x, y in zip(kept, full)),
              "out of time while adding an arm: every row drawn earlier is kept")
        rc = V23.build_manifests(46, 100, 20, force=False)
        check(rc == 1, "the manifest builder refuses to overwrite pre-registered rows")
    finally:
        V23.MANIFESTS = saved
    real = os.path.join(HERE, V23.OUT_DEV)
    before = os.path.getmtime(real) if os.path.exists(real) else None
    stub = os.path.join(HERE, V23.OUT_STUB)
    r = subprocess.run([sys.executable, 'pretest_v23.py', '--stub', '--dev', '--limit', '4',
                        '--arms', 'TABU'], cwd=HERE, capture_output=True, text=True)
    after = os.path.getmtime(real) if os.path.exists(real) else None
    check(r.returncode == 0 and os.path.exists(stub) and before == after,
          "a stub run with no --out writes pretest_v23_stub.json, never the real result file")
    if os.path.exists(stub):
        os.remove(stub)


def part5():
    print("\nPART 5 - the diagnosis reproduces the motivating numbers")
    r = subprocess.run([sys.executable, 'v23_diagnosis.py'], cwd=HERE, capture_output=True, text=True)
    out = r.stdout
    check(r.returncode == 0, "v23_diagnosis.py runs offline")
    check('B5 78  oracle@3 75  oracle@5 82' in out, "v22 main: B5 78, oracle@5 82")
    check('B5 errors: 22 -- 18 with NO correct sample' in out, "18 of B5's 22 errors have no right sample")
    check('wrong leaders caught 15/20' in out, "at TAU the v21 rows catch 15 of 20 wrong leaders")


def main() -> int:
    part1()
    part2()
    part3()
    part4()
    part5()
    print(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed")
    for f in FAILS:
        print("  FAILED:", f)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
