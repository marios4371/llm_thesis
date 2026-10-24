"""[v40.0] Guards for cross-model Symbolic Surgery (cross_model_surgery.py, ollama_client.py,
pretest_v40.py). Offline, no GPU, no Ollama server needed.

What these tests exist to prevent:
  * chat-model formatting the frozen audit cannot read (markdown bold, ÷, '4 x 6',
    '$8' money), or a cleaner that changes what Qwen-Math's LaTeX audit sees;
  * a FALSE verdict on a fragment the audit cut out of a longer expression
    ('2/4/2 = 1' from '= 1/4', the numeric tail of '6 + x + ...') refuting a
    right sample, while genuine slips stay refuted;
  * a final-answer parser that reads '1,376' as 1 or prefers an early 'answer';
  * arms that are not what V40_PLAN.md says (SC5, SC5-R, SC5-RS under v38's
    frozen policy; a resume costs one call only when it was drawn);
  * an Ollama request without the frozen sampling knobs or seeds, a resume that
    does not end the conversation with the assistant prefix, a thinking model
    allowed to think, a dropped connection that loses the row;
  * a manifest or a detector that moved after the pre-registration (sha256
    frozen in V40_PLAN.md), or a notebook that starts the GPU before this file.

Run as `python test_v40.py`.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error

import cross_model_surgery as X
import ollama_client as OC
import pretest_v40 as V
import trace_siv as T
import trace_surgery as S

FAILS = []
N = [0]


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def part1():
    print("\nPART 1 - the text the audit reads")
    check(X.clean("2. Total: 198 + 92 + 594 = **984**") == "2. Total: 198 + 92 + 594 = 984", "markdown bold goes")
    check(X.clean("so 2**3 = 8") == "so 2**3 = 8", "a power written 2**3 stays")
    check(X.clean("984 ÷ 2 = 492 and 4 x 6 = 24") == "984 / 2 = 492 and 4 * 6 = 24", "÷ and 'digit x digit'")
    check(X.clean("53 boxes at $8 each: 53 × $8 = $424") == "53 boxes at 8 each: 53 × 8 = 424",
          "money signs go in plain lines")
    latex = "\\[ 3 \\times \\$8 = \\$24 \\]"
    check(X.clean(latex) == latex, "a line with a LaTeX command is left alone")
    for s in ("**a** = **b**", "984 ÷ 2", "$5 + $3 = $8", latex):
        check(X.clean(X.clean(s)) == X.clean(s), f"idempotent on {s!r}")
    for raw, n in (("1. **Total:** 198 + 92 + 594 = **984**\nAnswer: 984", 1), ("Saturday: 984 ÷ 2 = 482", 1),
                   ("He has 4 x 6 = 26 apples.", 1), ("So 53 × $8 = $434 in all.", 1)):
        check(sum(v == T.FALSE for st in X.audit(X.clean(raw)) for _, v in st) == n and not any(T.audit(raw)),
              f"read only after cleaning: {raw[:34]!r}")


def part2():
    print("\nPART 2 - which FALSE verdicts count")
    fragments = ["Each tiny cube uses 2/4/2 = 1/4 ounce of water.",
                 "We have 6 + x + 2.5 + 12 + 2.5 = 30 hours in all.",
                 "Sunday is 4 + 60/60 = 4 + 1 = 5 hours.",
                 "V = 5*19 - 5 = 95 - 5 = 90",
                 "With overtime: 10 + 0.10 * 10 = 10 + 1 = 11 emails."]
    for t in fragments:
        txt = X.clean(t)
        check(T.refutes(T.audit(txt), None) and not X.refuted(txt), f"fragment not refuted: {t[:38]!r}")
    genuine = ["The total is 143 + 80 + 715 = 948 bananas.", "Total = 13 + 42 = 65 minutes",
               "\\[ 61 + 8 + 13 + 10 + 58 = 140 \\]", "So 53 * 8 = 434."]
    for t in genuine:
        check(X.refuted(X.clean(t)), f"genuine slip refuted: {t[:38]!r}")
    q = "First:\n\\[ 198 + 92 + 594 = 984 \\]\n\nThen:\n\\[ 984 / 2 = 492 \\]\n\nAnswer: 492"
    check(X.audit(X.clean(q)) == T.audit(q), "Qwen-Math LaTeX: v40's audit is v35's audit")
    check(not X.complete('2/4/2 = 1', 'uses 2/4/2 = 1/4 ounce') and X.complete('13 + 42 = 55', 'T = 13 + 42 = 55.')
          and X.complete('9 + 1 = 11', 'not in this text'), "complete(): fragment / whole / not found")


def part3():
    print("\nPART 3 - the final answer")
    cases = [("So the total is 1,376 watermelons.\nAnswer: 1,376", 1376.0), ("**Answer:** $424", 424.0),
             ("The answer is \\boxed{53}.", 53.0), ("Thus the answer is 2.5.", 2.5), ("Answer: 1/4", 0.25),
             ("I think the answer is 5. Then\n\\boxed{7}", 7.0), ("Answer: \\boxed{12}", 12.0),
             ("They have 42 apples left.", 42.0), ("Answer: -12 degrees", -12.0), ("no number here", None),
             ("Answer: 2 apples, then 3 pears. Answer: 30", 30.0)]
    for raw, want in cases:
        got = X.parse_final(raw)
        check(got == want, f"parse_final({raw[:30]!r}) = {want} (got {got})")


def _sample(answer, slip=None, reached=True):
    """A stored sample in v40's format: a trace with an optional carry slip."""
    if slip is None:
        raw = f"Add: 140 + 10 = 150\nAnswer: {answer}"
    elif reached:
        raw = f"Add: 140 + 10 = {150 + slip}\nAnswer: {150 + slip}"
    else:
        raw = f"Add: 140 + 10 = {150 + slip}\nSo, thinking it over, the total is {answer}.\nAnswer: {answer}"
    return {'raw': raw}


def part4():
    print("\nPART 4 - the arms")
    c = X.prepare([_sample(150), _sample(150), _sample(None, 10), _sample(None, 10), _sample(None, 10)])
    check([x['answer'] for x in c] == [150.0, 150.0, 160.0, 160.0, 160.0] and [x['refuted'] for x in c]
          == [False, False, True, True, True], "three samples share a carry slip (160 instead of 150)")
    check(all(x['surgery']['prop']['reached'] and x['surgery']['prop']['answer'] == 150.0 for x in c[2:]),
          "the correction reaches each slipped answer")
    check(X.sc5(c) == (160.0, 5) and X.sc5_r(c) == (150.0, 5) and X.sc5_rs(c) == (150.0, 5),
          "SC5 votes the slip, SC5-R drops it, SC5-RS repairs it (no call)")
    stored = [_sample(150), _sample(150), _sample(152, 10, reached=False), _sample(152, 10, reached=False),
              _sample(152, 10, reached=False)]
    c = X.prepare(stored)
    check(all(x['surgery'] and not x['surgery']['prop']['reached'] for x in c[2:])
          and X.needs_resume(c) == [2, 3, 4], "not reached and shared by 3 samples: the frozen policy resumes")
    check(X.sc5_rs(c) == (150.0, 5), "no resume drawn: the slipped samples drop out and cost no call")
    for i in (2, 3, 4):
        pre = c[i]['surgery']['prefix']
        stored[i]['resume'] = {'prefix': pre, 'raw': pre + "\nSo the total is 150.\nAnswer: 150"}
    c = X.prepare(stored)
    check(X.sc5_rs(c) == (150.0, 8) and all(x['surgery']['resume']['answer'] == 150.0 for x in c[2:]),
          "three drawn resumes: three calls, and they enter the vote")
    one = X.prepare([_sample(152, 10, reached=False)] + [_sample(151)] * 4)
    check(one[0].get('surgery') and X.needs_resume(one) == [] and S.surgery_policy(one[0], 1) == S.REFUTE,
          "an isolated slip that does not reach the answer is refuted only")
    allbad = X.prepare([{'raw': "Add: 2 + 2 = 5\nAnswer: 5"}] * 5)
    check(X.sc5_r(allbad)[0] == 5.0, "every sample refuted: SC5-R falls back to SC5")
    check(V.sha(T.__file__) == '483ed23b89b6' and V.sha(S.__file__) == '42376931d94a',
          "trace_siv.py and trace_surgery.py are the frozen v35 / v38 files (the policy is v38's)")


class FakeServer:
    """Records every request; answers like Ollama."""

    def __init__(self, thinking=False, prefill=True, fail_first=0):
        self.calls, self.thinking, self.prefill, self.fail = [], thinking, prefill, fail_first

    def __call__(self, url, payload, timeout):
        self.calls.append((url, payload))
        if self.fail:
            self.fail -= 1
            raise urllib.error.URLError('connection refused')
        if url.endswith('/api/show'):
            return {'capabilities': ['completion'] + (['thinking'] if self.thinking else []),
                    'details': {'family': 'llama', 'parameter_size': '3.2B', 'quantization_level': 'Q4_K_M'}}
        if url.endswith('/api/version'):
            return {'version': '0.40.3'}
        msgs = payload.get('messages') or []
        if msgs and msgs[-1]['role'] == 'assistant':
            text = ' 150.' if self.prefill else 'To find the sum we add the numbers.'
        else:
            text = f"Step.\nAnswer: {payload['options'].get('seed', 0) % 7}"
        return {'message': {'role': 'assistant', 'content': text}, 'eval_count': 12, 'eval_duration': 2e8,
                'done_reason': 'stop'}


def part5():
    print("\nPART 5 - the Ollama client (fake server)")
    srv = FakeServer(thinking=True)
    c = OC.OllamaClient('m', transport=srv, num_ctx=V.NUM_CTX)
    outs = c.sample('Q?', 5, 0.8, 1536, seeds=[11, 12, 13, 14, 15])
    chats = [p for u, p in srv.calls if u.endswith('/api/chat')]
    opts = [p['options'] for p in chats]
    check(len(outs) == 5 and len(chats) == 5, "five sampled completions, one request each")
    check(sorted(o['seed'] for o in opts) == [11, 12, 13, 14, 15] and all(
        (o['temperature'], o['top_p'], o['top_k'], o['num_predict'], o['num_ctx'], o['repeat_penalty'])
        == (0.8, 0.95, 50, 1536, V.NUM_CTX, 1.0) for o in opts), "v38's knobs and a seed per sample")
    check(all(p.get('think') is False and p['stream'] is False for p in chats), "a thinking model is asked not to think")
    srv2 = FakeServer(thinking=False)
    c2 = OC.OllamaClient('m', transport=srv2)
    full, _ = c2.continue_from('Q?', 'So the answer is', 0.8, 64, seed=3)
    last = [p for u, p in srv2.calls if u.endswith('/api/chat')][-1]
    check(full == 'So the answer is 150.' and last['messages'][-1] == {'role': 'assistant', 'content': 'So the answer is'}
          and 'think' not in last, "a resume ends the conversation with the assistant prefix and returns prefix + text")
    check(OC.smoke_prefill(OC.OllamaClient('m', transport=FakeServer()), V.V21.COT_PROMPT)[0]
          and not OC.smoke_prefill(OC.OllamaClient('m', transport=FakeServer(prefill=False)), V.V21.COT_PROMPT)[0],
          "the prefill smoke test tells a continuation from a fresh answer")
    sleep, OC.time.sleep = OC.time.sleep, lambda s: None
    try:
        c3 = OC.OllamaClient('m', transport=FakeServer(fail_first=2))
        check(c3.chat([{'role': 'user', 'content': 'x'}], 0.8, 8)[0].startswith('Step.'),
              "a dropped connection is retried (twice here) and the call succeeds")
        try:
            OC.OllamaClient('m', transport=FakeServer(fail_first=99)).chat([{'role': 'user', 'content': 'x'}], 0.8, 8)
            check(False, "an unreachable server is an error after the retries")
        except OC.OllamaError:
            check(True, "an unreachable server is an error after the retries")
    finally:
        OC.time.sleep = sleep


def part6():
    print("\nPART 6 - the manifest and the bars, frozen")
    d = json.load(open(V.MANIFEST, encoding='utf-8'))
    main = [r for r in d['rows'] if r['set'] == 'main']
    pilot = [r for r in d['rows'] if r['set'] == 'pilot']
    v38 = [r for r in json.load(open(V.V38_SOURCE, encoding='utf-8'))['rows'] if r['level'] == 'A']
    check(d['seed'] == 40 and len(main) == 200 and len(pilot) == 10, "seed 40: 200 main rows, 10 pilot rows")
    check([(r['pid'], r['text'], r['gold']) for r in main] == [(r['problem_id'], r['text'], float(r['gold']))
                                                              for r in v38], "main = v38's 200 A rows, same order")
    check(not ({r['pid'] for r in pilot} & {r['pid'] for r in main}) and all(r['pid'].startswith('gsm-hard_')
                                                                              for r in pilot),
          "pilot rows are other clean GSM-Hard rows (v36), never in the reading")
    check((V.PRI_MIN_NET, V.PRI_RATIO, V.PRI_ALPHA, V.SECB_MIN_NET, V.SECB_RATIO, V.MIN_ROWS) == (5, 2, 0.05, 5, 2, 100),
          "bars frozen")
    check((V.K, V.TEMPERATURE, V.MAX_TOKENS) == (5, 0.8, 1536), "v38's sampler and budget")
    plan = open('V40_PLAN.md', encoding='utf-8').read()
    for f in ('cross_model_surgery.py', V.MANIFEST):
        m = re.search(r'`' + re.escape(f) + r'`: sha256 \*\*([0-9a-f]{12})\*\*', plan)
        check(bool(m) and V.sha(f) == m.group(1), f"{f} is the file frozen in V40_PLAN.md")


def part7():
    print("\nPART 7 - offline plumbing (stub models, temp dir)")
    d = tempfile.mkdtemp()
    r = subprocess.run([sys.executable, 'pretest_v40.py', '--stub', '--models', 's1,s2', '--limit', '5', '--dir', d],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    files = sorted(os.listdir(d))
    check(r.returncode == 0 and files == ['pretest_v40_s1.json', 'pretest_v40_s2.json'],
          "two stub models, one file each, in turn")
    recs = json.load(open(os.path.join(d, 'pretest_v40_s1.json'), encoding='utf-8'))['rows']
    check(len(recs) == 5 and all(V.complete(x) for x in recs.values()), "5 complete records")
    check('no new model has >= 100 complete main rows' in r.stdout, "fewer than 100 rows: no verdict")
    r = subprocess.run([sys.executable, 'pretest_v40.py', '--stub', '--score-prm', '--inputs',
                        os.path.join(d, 'pretest_v40_s1.json')], capture_output=True, text=True, encoding='utf-8',
                       errors='replace')
    prm = json.load(open(os.path.join(d, 'pretest_v40_prm_s1.json'), encoding='utf-8'))['scores']
    rows = {x['pid']: x for x in V.load_rows()}
    reading = V.model_reading(rows, recs, prm)
    check(r.returncode == 0 and {'B5', 'ADAPT', 'ADAPT-R', 'ADAPT-RS'} <= set(reading['right']),
          "PRM scores attach by text and add the v38 arms")
    r = subprocess.run([sys.executable, 'pretest_v40.py', '--stub', '--pilot', '--models', 's1', '--dir', d],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    check(r.returncode == 0 and 'hours for 200 rows' in r.stdout and
          os.path.exists(os.path.join(d, 'pretest_v40_pilot_s1.json')), "the pilot writes its own file and report")
    fake = {'m': {'rows': 120, 'right': {'SC5': [False] * 120, 'ADAPT-R': [False] * 7 + [True] * 113,
                                         'ADAPT-RS': [True] * 7 + [False] + [True] * 112}}}
    out = V.verdicts(fake)
    check(out[0].startswith('PRIMARY      INCONCLUSIVE') and 'ADAPT-RS vs ADAPT-R' in out[0] and 'W7 L1' in out[0],
          "the primary is v38's ADAPT-RS vs ADAPT-R; net +6 but p = 0.07: INCONCLUSIVE")
    fake['m']['right']['ADAPT-RS'] = [True] * 120
    out = V.verdicts(fake)
    check(out[0].startswith('PRIMARY      SUPPORTED') and out[2].startswith('SECONDARY B  MET'),
          "W7 L0 (p = 0.016): SUPPORTED; ADAPT-RS beats self-consistency")
    check(V.verdicts({'m': {'rows': 150, 'right': {'SC5': [], 'SC5-RS': []}}})[0].startswith('none: no PRM scores'),
          "a model without PRM scores is not read")
    fake['m']['rows'] = 99
    check(V.verdicts(fake)[0].startswith('none'), "a model with 99 complete rows is not read")


def part7b():
    print("\nPART 7b - v40's reading reproduces v38 (where v38's result files exist)")
    import replay_v40 as RP
    q = RP.qwen_rows().get('A') or []
    if not q:
        check(True, "(v38 result files absent in this checkout: skipped)")
        return
    rows = {x['pid']: x for x in V.load_rows()}
    rd = V.model_reading(rows, {r['pid']: {'C': r['C'], 'surgery_done': True} for r in q}, RP.qwen_prm())
    s = {a: sum(v) for a, v in rd['right'].items()}
    check((s['ADAPT-RS'], s['ADAPT-R'], s['ADAPT'], s['SC5']) == (131, 122, 116, 118),
          "Qwen-Math on the 200 rows: ADAPT-RS 131, ADAPT-R 122, ADAPT 116, SC5 118 (v38's numbers)")


def part8():
    print("\nPART 8 - the notebooks")
    for nb, run in (('MAS_SHT_Kaggle_v40.ipynb', "pretest_v40.py --backend hf"),
                    ('MAS_SHT_Kaggle_v40_prm.ipynb', "pretest_v40.py --score-prm")):
        if not os.path.exists(nb):
            check(False, f"{nb} exists")
            continue
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
        t, r = code.find("'test_v40.py'"), code.find(run)
        check(t >= 0 and r >= 0 and t < r, f"{nb} runs test_v40.py before `{run}`")
        check('_tok}@' not in code and '_token}@' not in code and '@github.com' not in code,
              f"{nb} never puts a token in a URL")


def main():
    for part in (part1, part2, part3, part4, part5, part6, part7, part7b, part8):
        part()
    print(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed" + (f"; FAILED: {FAILS}" if FAILS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
