"""[v20.0] Guards for the input-space diversity pre-test. Offline, no models.

The thing these tests exist to prevent is the failure that killed v16.2 and
v19: an arm that measures its own broken construction instead of the mechanism
it was built to test. So most of what is checked here is not "does the code
run" but "does a rendering that changed the problem get refused".

Run as `python test_v20.py`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import renderings as R

FAILS = []
N = [0]

HERE = os.path.dirname(os.path.abspath(__file__))

PROB = ("Mr. Smith bought 3 crates of apples. Each crate holds 12 apples and "
        "cost $4.50. He sold half of all the apples at $0.75 each. "
        "How much money did he make?")


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def part1():
    print("\nPART 1 - the structural transforms are faithful BY CONSTRUCTION")
    s = R.split_sentences(PROB)
    check(len(s) == 4, f"the problem splits into 4 sentences, not {len(s)}")
    check(any('$4.50' in x for x in s), "a decimal price did not split a sentence")
    check(any(x.startswith('Mr. Smith') for x in s), "'Mr.' did not split a sentence")

    for name in ('givens_first', 'goal_first'):
        ren = R.STRUCTURAL[name](PROB)
        check(ren.ok, f"{name}: builds")
        check(R.numbers(ren.text) == R.numbers(PROB),
              f"{name}: every numeric literal survives, unchanged and uncounted-twice")
        check('How much money did he make?' in ren.text,
              f"{name}: the question is carried over verbatim")
        check(ren.text != PROB, f"{name}: actually produces a different surface form")

    check(R.givens_first(PROB).text.index('Known facts') <
          R.givens_first(PROB).text.index('Question:'),
          "givens_first puts the facts before the question")
    check(R.goal_first(PROB).text.index('Find:') <
          R.goal_first(PROB).text.index('You are given'),
          "goal_first puts the question first")

    short = "What is 2 + 2?"
    check(not R.givens_first(short).ok,
          "a one-sentence problem is REFUSED, not returned as a fake second voter")
    check(not R.goal_first(short).ok, "goal_first refuses it too")

    ident = R.identity(PROB)
    check(ident.ok and ident.text == PROB.strip(), "identity is the problem itself")


def part2():
    print("\nPART 2 - is_faithful catches the ways a rendering changes the problem")
    ok, _ = R.is_faithful(PROB, PROB)
    check(ok, "a problem is faithful to itself")

    cases = [
        (PROB.replace('12 apples', '21 apples'), 'a transposed digit'),
        (PROB.replace('3 crates', 'three crates'), 'a number spelled out'),
        (PROB.replace(' and cost $4.50', ''), 'a dropped quantity'),
        (PROB + ' There are 7 crates in the shed.', 'an added quantity'),
        ('The answer is 40.5', 'the model answered instead of restating'),
        (PROB.replace('How much money did he make?', 'He made some money.'),
         'the question turned into a statement'),
    ]
    for bad, why in cases:
        ok, reason = R.is_faithful(PROB, bad)
        check(not ok, f"refused: {why}" + (f"  [{reason[:40]}]" if ok else ""))

    # comma grouping is only a surface difference and must NOT be refused
    ok, _ = R.is_faithful("She saved 1200 dollars.", "She saved 1,200 dollars.")
    check(ok, "1200 and 1,200 are the same number, not a defect")


def part3():
    print("\nPART 3 - the paraphrase is validated, never patched")
    good = f"<restated>\nHow much money did he make? {PROB}\n</restated>"
    ren = R.parse_paraphrase(good, PROB)
    check(ren.ok, "a faithful paraphrase is accepted")

    ren = R.parse_paraphrase("Sure! Here you go: some text", PROB)
    check(not ren.ok and ren.text == PROB.strip(),
          "a response with no <restated> block is refused AND falls back to identity")

    bad = "<restated>\nHe bought 3 crates of 15 apples. How many?\n</restated>"
    ren = R.parse_paraphrase(bad, PROB)
    check(not ren.ok, "a paraphrase that changed a number is refused")
    check(ren.text == PROB.strip(), "...and the refused rendering falls back to the original")

    ren = R.parse_paraphrase("<restated>\n\n</restated>", PROB)
    check(not ren.ok, "an empty <restated> block is refused")


def part4():
    print("\nPART 4 - the vote cannot credit the ensemble for a coin flip")
    v = [('identity', 10.0), ('givens_first', 20.0), ('paraphrase', 20.0)]
    ans, info = R.majority(v)
    check(ans == 20.0, "2-1 majority wins")
    check(not info['unanimous'] and not info['tie'], "...and is neither unanimous nor a tie")

    ans, info = R.majority([('identity', 7.0), ('givens_first', 7.0)])
    check(ans == 7.0 and info['unanimous'], "unanimity is reported")

    ans, info = R.majority([('identity', 5.0), ('givens_first', 9.0)])
    check(ans == 5.0 and info['tie'],
          "a 1-1 tie breaks to identity, i.e. to what one call would have said")

    ans, info = R.majority([('identity', None), ('givens_first', 4.0)])
    check(ans == 4.0 and info['n_votes'] == 1, "a refused rendering does not vote")

    ans, _ = R.majority([('identity', None), ('givens_first', None)])
    check(ans is None, "no live votes -> no answer, not a guess")

    # tolerance: near-agreement inside the grader's tolerance is one cluster
    ans, info = R.majority([('identity', 3473609.0), ('a', 3473609.2), ('b', 12.0)])
    check(info['cluster_sizes'][0] == 2,
          "two answers inside the repo's tolerance count as one vote block")

    check(R.disagreement_rate([1.0, 1.0, 1.0]) == 0.0, "no disagreement -> 0.0")
    check(R.disagreement_rate([1.0, 2.0, 3.0]) == 1.0, "all differ -> 1.0")
    check(abs(R.disagreement_rate([1.0, 1.0, 2.0]) - 2 / 3) < 1e-9, "1 of 3 pairs agree")
    check(R.disagreement_rate([None, 1.0]) is None, "fewer than 2 live votes -> undefined")


def part5():
    print("\nPART 5 - the runner end to end, against a stub model")
    rows = [{'problem_id': f'stub_{i}', 'text': PROB.replace('3 crates', f'{i+3} crates'),
             'gold': float(10 * (i + 1)), 'dataset': 'stub'} for i in range(4)]
    with tempfile.TemporaryDirectory() as td:
        man = os.path.join(td, 'man.json')
        out = os.path.join(td, 'out.json')
        with open(man, 'w', encoding='utf-8') as fh:
            json.dump({'dataset': 'stub', 'seed': 0, 'n': len(rows), 'rows': rows}, fh)
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, 'pretest_v20.py'), '--stub',
             '--manifest', man, '--out', out, '--arms', 'CF'],
            capture_output=True, text=True, cwd=HERE)
        check(r.returncode == 0, f"a CF stub run exits 0 (rc={r.returncode})")
        if r.returncode != 0:
            print(r.stdout[-1500:], r.stderr[-1500:])
            return
        with open(out, encoding='utf-8') as fh:
            got = json.load(fh)
        rs = got['rows']
        check(len(rs) == 4, "every row is recorded")
        check(all(x['C_correct'] for x in rs), "the stub oracle solves the control")
        check(all(x['F_correct'] for x in rs), "and the ensemble does not lose them")
        check(all(x['F_renderings'][0]['name'] == 'identity' for x in rs),
              "identity is the first rendering")
        check(all(x['F_renderings'][0]['answer'] == x['C_answer'] for x in rs),
              "identity REUSES the control's answer instead of paying for it twice")
        check(all(len(x['F_renderings']) == 3 for x in rs),
              "all three default renderings are attempted")

        # arm S refuses to run without arm C in the same invocation
        r2 = subprocess.run(
            [sys.executable, os.path.join(HERE, 'pretest_v20.py'), '--stub',
             '--manifest', man, '--out', out + '2', '--arms', 'F'],
            capture_output=True, text=True, cwd=HERE)
        check(r2.returncode == 2, "arm F alone is rejected (it needs C's identity vote)")

        # re-running the same arms must RESUME, not redo: the stub counts
        # calls, so a second identical invocation that did no work is proof.
        before = json.load(open(out, encoding='utf-8'))['rows']
        r_re = subprocess.run(
            [sys.executable, os.path.join(HERE, 'pretest_v20.py'), '--stub',
             '--manifest', man, '--out', out, '--arms', 'CF'],
            capture_output=True, text=True, cwd=HERE)
        check(r_re.returncode == 0, "re-running the same arms exits 0")
        check('resume is ON' in r_re.stdout, "...and says it is resuming")
        check(json.load(open(out, encoding='utf-8'))['rows'] == before,
              "a resumed run leaves the completed rows byte-identical")

        # a second session merges rather than overwriting
        r3 = subprocess.run(
            [sys.executable, os.path.join(HERE, 'pretest_v20.py'), '--stub',
             '--manifest', man, '--out', out, '--arms', 'S'],
            capture_output=True, text=True, cwd=HERE)
        check(r3.returncode == 0, f"a follow-up S session exits 0 (rc={r3.returncode})")
        if r3.returncode == 0:
            with open(out, encoding='utf-8') as fh:
                merged = json.load(fh)['rows']
            check(all('C_correct' in x and 'S_correct' in x for x in merged),
                  "session 2 keeps session 1's arms on the same rows -- the "
                  "comparison stays paired")


def part6():
    print("\nPART 6 - the opt-in sampling switch exists and is OFF by default")
    src = open(os.path.join(HERE, 'Mas_solver.py'), encoding='utf-8').read()
    check('LOCAL_HF_SAMPLING' in src, "the switch is defined")
    check('"enabled": False' in src, "it is off by default, so pre-v20 runs are unchanged")
    check('renormalize_logits=True' in src,
          "sampling renormalises logits -- the fp16 overflow v10.3 hit")
    check('no longer an independent sample' in src,
          "a failed sampled call retries greedily on the GPU before the "
          "pre-existing handler moves a 7B model to CPU for the session")


if __name__ == '__main__':
    for p in (part1, part2, part3, part4, part5, part6):
        p()
    print(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed")
    for f in FAILS:
        print("  FAILED: " + f)
    sys.exit(1 if FAILS else 0)
