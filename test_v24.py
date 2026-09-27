"""[v24.0] Guards for the Reader agent. Offline, no models.

What these tests exist to prevent is an arm that measures something other
than the mechanism it names:
  * a C arm that is not the prompt of every earlier version (then ASQ vs C is
    not a comparison against the stored samples);
  * ASQ and END carrying different notes (then END is not a placement control);
  * a Reader prompt that contains P2 text (then the Reader was tuned on the
    rows it is judged on);
  * a failed reading that silently becomes some third prompt instead of C;
  * a verdict read off a partial run, or thresholds that moved after they
    were frozen.

Run as `python test_v24.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile

import premise_ledger as L
import pretest_v21 as V21
import pretest_v24 as V24
import situation_reader as SR

FAILS = []
N = [0]


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


TEXT = ("Ana has 30 marbles. She gives a third of them to Ben, who already had 4. "
        "Ben then buys twice as many as he was given. How many marbles does Ben have now?")
READ = ("S1: Q: How many marbles does Ana start with? A: 30.\n"
        "S2: Q: How many does she give Ben? A: A third of 30 = 10; she keeps the other 20.\n"
        "S2: Q: How many did Ben have before? A: 4.\n"
        "S3: Q: How many does Ben buy, relative to what? A: Twice the 10 he was given: 20.\n"
        "S4: -\n"
        "Asked: the number of marbles Ben has at the end.")


def _args(**kw):
    base = dict(dev=True, pool='v22', arms='', k=5, out='', preset='qwen_math7b_mixed',
                temperature=0.8, max_tokens=1024, limit=0, stub=True, resume=True,
                max_hours=0.0, summary_only=False, build_manifest=False, seed=47, force=False)
    base.update(kw)
    return argparse.Namespace(**base)


def part1():
    print("\nPART 1 - the Reader's input")
    check(SR.sentences(TEXT) == L.split_sentences(TEXT) and len(SR.sentences(TEXT)) == 4,
          "sentences come from the ledger's splitter, so indices match earlier versions")
    msgs = SR.reader_messages(TEXT)
    check(msgs[0]['role'] == 'system' and len(msgs) == 2 + 2 * len(SR.READER_DEMOS),
          "system prompt, one user/assistant pair per demo, then the problem")
    check([m['role'] for m in msgs[1:-1]] == ['user', 'assistant'] * len(SR.READER_DEMOS),
          "demos alternate user / assistant")
    last = msgs[-1]['content'].splitlines()
    check(last[0] == 'Problem sentences:' and
          last[1:] == [f"S{i}: {s}" for i, s in enumerate(SR.sentences(TEXT), 1)],
          "the problem reaches the Reader as numbered sentences, verbatim")
    for problem, reading in SR.READER_DEMOS:
        n = len(SR.sentences(problem))
        rd = SR.parse_reading(reading, n)
        check(rd['ok'] and rd['dropped'] == 0 and rd['asked'] and '=' not in rd['asked'],
              f"demo '{problem[:24]}...' parses cleanly under the Reader's own rules")
        check(max(int(i) for i in rd['notes']) <= n,
              f"demo '{problem[:24]}...' anchors every note to a real sentence")

    prompt = SR.READER_SYSTEM + ' '.join(p + ' ' + r for p, r in SR.READER_DEMOS)
    words = lambda t: re.findall(r"[a-z0-9']+", t.lower())
    grams = lambda t, n=6: {tuple(w[i:i + n]) for w in [words(t)] for i in range(len(w) - n + 1)}
    p2 = set()
    for path in ('pretest_data/v20_problems.json', 'pretest_data/v22_confirm_p2.json'):
        for r in json.load(open(path, encoding='utf-8'))['rows']:
            p2 |= grams(r['text'])
    shared = grams(prompt) & p2
    check(not shared, f"the Reader prompt shares no 6-word sequence with any stored P2 row "
                      f"({len(shared)} shared)")


def part2():
    print("\nPART 2 - parsing the Reader, sanitation only")
    rd = SR.parse_reading(READ, 4)
    check(rd['ok'] and rd['n_notes'] == 4 and rd['asked'].startswith('the number'),
          "the documented format parses: 4 notes and the Asked line")
    check(len(rd['notes']['2']) == 2 and '4' not in rd['notes'], "S2 has two notes, 'S4: -' has none")
    alt = ("- **S1:** Q: How many? A: 30.\n* S2 - Q: Given? A: 10. Q: Kept? A: 20.\n"
           "S3) Ben buys 20.\n**Asked:** Ben's marbles at the end")
    ra = SR.parse_reading(alt, 4)
    check(ra['notes'].get('1') == [['How many?', '30.']], "bullet and bold variants parse")
    check(ra['notes'].get('2') == [['Given?', '10.'], ['Kept?', '20.']],
          "two pairs on one line become two notes")
    check(ra['notes'].get('3') == [['', 'Ben buys 20.']], "a statement without Q/A is kept as a note")
    check(ra['asked'] == "Ben's marbles at the end", "a bold Asked line parses")
    rb = SR.parse_reading("S9: Q: x? A: 1.\nS1: Q: y? A: 2.", 4)
    check(rb['dropped'] == 1 and list(rb['notes']) == ['1'],
          "a note anchored to a sentence that does not exist is dropped and counted")
    many = '\n'.join(f"S1: Q: q{i}? A: {i}." for i in range(7))
    rc = SR.parse_reading(many, 2)
    check(len(rc['notes']['1']) == SR.MAX_QA_PER_SENTENCE and
          rc['dropped'] == 7 - SR.MAX_QA_PER_SENTENCE, "notes per sentence are capped and counted")
    rd2 = SR.parse_reading("S1: Q: a? A: 1.\nAsked: total = 45 marbles", 2)
    check(rd2['asked'] == '' and rd2['dropped'] == 1, "an Asked line that states a result is dropped")
    for junk in ('', 'To solve this, add 30 and 4.\nAnswer: 34', 'S1: -\nS2: -'):
        check(not SR.parse_reading(junk, 2)['ok'], f"nothing usable -> not ok: {junk[:20]!r}")
    long = SR.parse_reading("S1: Q: x? A: " + 'y' * 900, 1)
    check(len(long['notes']['1'][0][1]) <= SR.MAX_NOTE_CHARS, "a long note is cut to the cap")


def part3():
    print("\nPART 3 - the Solver's input, per arm")
    rd = SR.parse_reading(READ, 4)
    c = SR.cot_prompt(TEXT)
    check(c == V21.COT_PROMPT.format(problem=TEXT), "C is the prompt of every earlier version, byte for byte")
    bad = SR.parse_reading('Answer: 34', 4)
    check(SR.asq_prompt(TEXT, bad) == c and SR.end_prompt(TEXT, bad) == c,
          "an unusable reading leaves ASQ and END byte-identical to C")
    check(SR.asq_prompt(TEXT, None) == c, "a missing reading also falls back to C")
    head, tail = V21.COT_PROMPT.split('{problem}')
    a, e = SR.asq_prompt(TEXT, rd), SR.end_prompt(TEXT, rd)
    check(a.startswith(head) and a.endswith(tail) and e.startswith(head) and e.endswith(tail),
          "ASQ and END change only the problem slot of the CoT prompt")
    body = a[len(head):len(a) - len(tail)].splitlines()
    sents = SR.sentences(TEXT)
    pos = [body.index(s) for s in sents]
    check(pos == sorted(pos), "ASQ keeps every sentence, verbatim and in order")
    for i, s in enumerate(sents, 1):
        want = [SR.note_line(q, n) for q, n in (tuple(p) for p in rd['notes'].get(str(i), []))]
        got = body[pos[i - 1] + 1: pos[i - 1] + 1 + len(want)]
        check(got == want, f"S{i}'s notes sit directly under S{i}")
    check(e[len(head):].startswith(TEXT + '\n\n' + SR.END_HEADER),
          "END keeps the problem unchanged, then the notes")
    notes = lambda p: sorted(l for l in p.splitlines() if l.startswith('(Q:') or
                             l.startswith('(Asked:') or (l.startswith('(') and l != SR.ASQ_INTRO))
    check(notes(a) == notes(e) == sorted(SR.note_lines(TEXT, rd)),
          "ASQ and END carry exactly the same notes: only the position differs")
    s = SR.self_prompt(TEXT)
    check(TEXT in s and 'sentence by sentence' in s and "'Answer:'" in s,
          "SELF holds the problem verbatim and the single-agent reading instruction")


def part4():
    print("\nPART 4 - the stored pool and the frozen numbers")
    pool = V24.load_pool(('v22',))
    main = [r for r in pool if r['set'] == 'main']
    guard = [r for r in pool if r['set'] == 'guard']
    check(len(main) == 100 and len(guard) == 20, "v22 pool: 100 main + 20 guard rows")
    check(all(len(r['stored_C']) == 5 for r in pool), "every stored row has 5 C samples")
    check(len({r['key'] for r in pool}) == len(pool), "row keys are unique")
    check(all(r['template'] is not None for r in main) and all(r['template'] is None for r in guard),
          "every P2 row has a template; GSM-Plus rows have none")
    recs = []
    for r in main:
        rec = V24.new_record(r)
        rec['arms']['C'] = {'samples': r['stored_C']}
        recs.append(rec)
    first = sum(V24.first(r, 'C') for r in recs)
    s3 = sum(V24.sc(r, 'C', 3) for r in recs)
    s5 = sum(V24.sc(r, 'C', 5) for r in recs)
    ps = 100 * sum(V24.acc(r, 'C', 5) for r in recs) / len(recs)
    check((first, s3, s5) == (56, 66, 68) and abs(ps - 59.6) < 0.05,
          f"C on v22 main reproduces V22_RESULTS.md: first 56, S3 66, S5 68, 59.6% per sample "
          f"(got {first}, {s3}, {s5}, {ps:.1f}%)")
    v21 = V24.load_pool(('v21',))
    check(sum(r['set'] == 'main' for r in v21) == 100 and sum(r['set'] == 'guard' for r in v21) == 40,
          "v21 pool: 100 main + 40 guard rows")
    check(V24.template_of('gsm-symbolic_p2_2257') == 45 and V24.template_of('gsm-plus_7') is None,
          "template = row index // 50 for P2, none elsewhere")


def _rec(key, tmpl, gold, arms):
    return {'key': key, 'pid': key, 'source': 't', 'set': 'main', 'gold': gold, 'template': tmpl,
            'arms': {a: {'samples': [{'answer': x} for x in v]} for a, v in arms.items()}}


def _rows(n_up, n_down, n_same, up_by=5):
    """Rows where ASQ gains `up_by` correct samples of 5 on n_up rows and loses
    one on n_down, each row its own template."""
    rows, g = [], 1.0
    for i in range(n_up):
        rows.append(_rec(f'u{i}', i, g, {'C': [0] * 5, 'ASQ': [g] * up_by + [0] * (5 - up_by)}))
    for i in range(n_down):
        rows.append(_rec(f'd{i}', 100 + i, g, {'C': [g] * 5, 'ASQ': [g] * 4 + [0]}))
    for i in range(n_same):
        rows.append(_rec(f's{i}', 200 + i, g, {'C': [g] * 3 + [0] * 2, 'ASQ': [g] * 3 + [0] * 2}))
    return rows


def part5():
    print("\nPART 5 - statistics and the pre-registered reading")
    check((V24.PRIMARY_MIN_PP, V24.PRIMARY_ALPHA, V24.PRIMARY_LOTO_PP, V24.SYSTEM_MIN_NET,
           V24.MECH_MIN_PP, V24.GUARD_MIN_NET, V24.CONFIRM_SEED) == (4.0, 0.05, 2.0, 4, 2.0, -1, 47),
          "the pre-registered bars and the confirmation seed are the frozen ones")
    check(V24.sign_p(0, 0) == 1.0 and abs(V24.sign_p(10, 0) - 2 / 1024) < 1e-12 and
          V24.sign_p(3, 7) == V24.sign_p(7, 3), "exact two-sided sign test (reported)")
    d = [1.0] * 10
    check(V24.perm_p([]) == 1.0 and V24.perm_p([0.0, 0.0]) == 1.0 and V24.perm_p(d) < 0.01,
          "permutation test: no differences -> p=1; ten equal gains -> p ~ 2/1024")
    mixed = [0.8, 0.6, -0.2, 0.4, -0.2, 0.2]
    check(V24.perm_p(mixed) == V24.perm_p([-x for x in mixed]) and
          V24.perm_p(mixed) == V24.perm_p(mixed), "permutation test: seeded, sign-symmetric")
    check(V24.perm_p([1.0] * 3 + [-0.2] * 6) < V24.perm_p([0.2] * 3 + [-0.2] * 6),
          "permutation test weighs magnitudes: three big gains beat six small losses")
    rows = _rows(8, 1, 91)
    c = V24.compare(rows, 'ASQ', 'C', 5)
    check(abs(c['pp'] - (8 * 100 - 20) / 100) < 1e-9 and (c['w'], c['l']) == (8, 1) and
          (c['tw'], c['tl']) == (8, 1) and (c['sw'], c['sl']) == (8, 0),
          "compare: per-sample pp, row and template W/L, SC W/L")
    same = V24.cluster_ci(_rows(0, 0, 10), 'ASQ', 'C', 5)
    check(same == (0.0, 0.0), "identical arms have a zero-width template bootstrap interval")
    lo, hi = V24.cluster_ci(rows, 'ASQ', 'C', 5)
    check(lo <= c['pp'] <= hi, "the template bootstrap interval contains the estimate")

    v = V24.verdicts(rows, [], ['ASQ'], 5)
    check(v[0].startswith('PRIMARY  SUPPORTED') and c['p'] < 0.05 and c['loto'] >= 2.0,
          f"+7.8 pp, perm p={c['p']:.4f}, {c['loto']:+.1f} pp without the best template -> SUPPORTED")
    check('beats SC' in v[1], "SC@5 net +8 with W >= 2L -> the system beats SC")
    v = V24.verdicts(_rows(0, 0, 50), [], ['ASQ'], 5)
    check(v[0].startswith('PRIMARY  REFUTED'), "ASQ == C -> REFUTED")
    few = _rows(5, 2, 20)
    v = V24.verdicts(few, [], ['ASQ'], 5)
    check(v[0].startswith('PRIMARY  INCONCLUSIVE') and V24.compare(few, 'ASQ', 'C', 5)['p'] >= 0.05,
          "a large gain on too few rows (permutation p >= 0.05) -> INCONCLUSIVE")
    tied = _rows(8, 1, 91)
    for r in tied[:8]:
        r['template'] = 0                     # the 8 gains all come from one template
    for r in tied[8:9]:
        r['template'] = 1
    ct = V24.compare(tied, 'ASQ', 'C', 5)
    v = V24.verdicts(tied, [], ['ASQ'], 5)
    check(v[0].startswith('PRIMARY  INCONCLUSIVE') and ct['p'] < 0.05 and ct['loto'] < 2.0,
          f"a significant gain carried by ONE template does not pass "
          f"({ct['loto']:+.1f} pp without it)")
    harm = [_rec(f'g{i}', None, 1.0, {'C': [1.0] * 5, 'ASQ': [0] * 5}) for i in range(2)]
    v = V24.verdicts(rows, harm, ['ASQ'], 5)
    check(v[-1].startswith('GUARD') and v[-1].endswith('HARM'), "guard net -2 -> HARM")
    ok1 = [_rec('g', None, 1.0, {'C': [1.0] * 5, 'ASQ': [0] * 5})]
    check(V24.verdicts(rows, ok1, ['ASQ'], 5)[-1].endswith('no harm'), "guard net -1 -> no harm")
    for r in rows:
        r['arms']['END'] = {'samples': [{'answer': x['answer']} for x in r['arms']['C']['samples']]}
        r['arms']['SELF'] = {'samples': [{'answer': x['answer']} for x in r['arms']['ASQ']['samples']]}
    v = V24.verdicts(rows, [], ['ASQ', 'END', 'SELF'], 5)
    check(any(x.startswith('MECH     the position matters') for x in v),
          "ASQ - END >= 2 pp -> the position matters")
    check(any(x.startswith('MULTI') and 'no advantage' in x for x in v),
          "ASQ == SELF -> no advantage for a separate Reader")


def part6():
    print("\nPART 6 - the runners, offline")
    tmp = tempfile.mkdtemp()
    out = os.path.join(tmp, 'dev.json')
    rc = V24.run(_args(out=out, limit=6, arms='ASQ'), ['ASQ'], dev=True)
    d = json.load(open(out, encoding='utf-8'))
    rows = d['rows']
    check(rc == 0 and len(rows) == 6 and all(V24.has(r, 'ASQ', 5) for r in rows),
          "stub dev pass (ASQ) runs: every row has 5 ASQ samples")
    check(all(r['arms']['C'].get('stored') for r in rows), "on dev rows C is the stored samples, never drawn")
    check(all(r.get('reading', {}).get('ok') for r in rows), "every dev row has a reading")
    before = {r['key']: (r['reading'], r['arms']['ASQ']['samples']) for r in rows}
    rc = V24.run(_args(out=out, limit=6, arms='ASQ,END'), ['ASQ', 'END'], dev=True)
    rows2 = json.load(open(out, encoding='utf-8'))['rows']
    check(all(before[r['key']] == (r['reading'], r['arms']['ASQ']['samples']) for r in rows2) and
          all(V24.has(r, 'END', 5) for r in rows2),
          "adding END on resume keeps the reading and the ASQ samples, and draws only END")
    real_call = V24._StubReader.call_model
    V24._StubReader.call_model = lambda self, msgs, **kw: "To solve it, add them.\nAnswer: 3"
    try:
        dead = os.path.join(tmp, 'dead.json')
        try:
            V24.run(_args(out=dead, limit=12, arms='ASQ'), ['ASQ'], dev=True)
            aborted = False
        except RuntimeError as exc:
            aborted = 'nothing usable' in str(exc)
        left = json.load(open(dead, encoding='utf-8'))['rows']
        kept = sum('ASQ' in r['arms'] or 'reading' in r for r in left)
        check(aborted and kept == 0,
              f"a Reader that is unusable on its first {V24.READER_ABORT_AFTER} rows stops the "
              f"session, and the fallback samples it drew are dropped ({kept} kept)")
    finally:
        V24._StubReader.call_model = real_call
    out2 = os.path.join(tmp, 'late.json')
    rc = V24.run(_args(out=out2, limit=6, arms='ASQ', max_hours=-1.0), ['ASQ'], dev=True)
    late = json.load(open(out2, encoding='utf-8'))['rows']
    check(rc == 0 and not any('ASQ' in r['arms'] for r in late),
          "a session whose deadline has passed draws nothing (no platform clock involved)")

    saved = V24.MANIFESTS
    pool = [r for r in V24.load_pool(('v22',))]
    mk = lambda rr: {'rows': [{'problem_id': x['pid'], 'text': x['text'], 'gold': x['gold'],
                               'dataset': 'x'} for x in rr]}
    paths = []
    for tag, rr in (('main', [x for x in pool if x['set'] == 'main'][:3]),
                    ('guard', [x for x in pool if x['set'] == 'guard'][:2])):
        p = os.path.join(tmp, f'{tag}.json')
        json.dump(mk(rr), open(p, 'w', encoding='utf-8'))
        paths.append((tag, p))
    V24.MANIFESTS = tuple(paths)
    try:
        cout = os.path.join(tmp, 'confirm.json')
        rc = V24.run(_args(dev=False, out=cout), ['C', 'ASQ'], dev=False)
        conf = json.load(open(cout, encoding='utf-8'))['rows']
        check(rc == 0 and len(conf) == 5 and all(V24.has(r, 'C', 5) and V24.has(r, 'ASQ', 5)
                                                 and not r['arms']['C'].get('stored') for r in conf),
              "confirmation draws C in the session, beside ASQ")
        check(V24.build_manifests(47, 3, 2, force=False) == 1,
              "the manifest builder refuses to overwrite pre-registered rows")
    finally:
        V24.MANIFESTS = saved

    import contextlib
    import io
    part = os.path.join(tmp, 'part.json')
    d = json.load(open(out, encoding='utf-8'))
    del d['rows'][0]['arms']['ASQ']
    json.dump(d, open(part, 'w', encoding='utf-8'))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        V24.summarise(d['rows'], ['ASQ'], 5, dev=True, stub=False, expected=len(d['rows']))
    check('PARTIAL' in buf.getvalue() and 'PRIMARY' not in buf.getvalue(),
          "a partial file reads no verdict")
    d = json.load(open(out, encoding='utf-8'))            # ASQ complete, END partial
    del d['rows'][0]['arms']['END']
    json.dump(d, open(part, 'w', encoding='utf-8'))
    run_cli = lambda extra: __import__('subprocess').run(
        [sys.executable, 'pretest_v24.py', '--dev', '--summary-only', '--out', part] + extra,
        capture_output=True, text=True, encoding='utf-8', errors='replace').stdout
    both, asq = run_cli([]), run_cli(['--arms', 'ASQ'])
    check('PARTIAL' in both and 'PRIMARY' in asq and 'PARTIAL' not in asq,
          "--summary-only reads every stored arm by default, and only --arms when given")

    real_before = os.path.exists(V24.OUT_DEV) and os.path.getmtime(V24.OUT_DEV)
    stub_file = V24.OUT_STUB
    had_stub = os.path.exists(stub_file)
    rc = V24.run(_args(limit=2, arms='ASQ', resume=False), ['ASQ'], dev=True)
    real_after = os.path.exists(V24.OUT_DEV) and os.path.getmtime(V24.OUT_DEV)
    check(os.path.exists(stub_file) and real_before == real_after,
          "a stub run with no --out writes pretest_v24_stub.json, never the real result file")
    if not had_stub and os.path.exists(stub_file):
        os.remove(stub_file)


def main() -> int:
    for part in (part1, part2, part3, part4, part5, part6):
        part()
    print(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed")
    for f in FAILS:
        print("  FAILED:", f)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
