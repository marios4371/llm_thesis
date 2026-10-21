"""[v39.0] Guards for the clause ablation pilot (clause_ablation.py, pretest_v39.py). Offline, no GPU.

What these tests exist to prevent:
  * a deletion variant that is not the problem minus exactly one unit (a lost
    sentence, a dangling comma, a lower-case first word);
  * labels that call a template sentence ADDED because its names, places or
    numbers changed, or that miss a clause P2 really added;
  * a detector that counts an unparsable variant as "unchanged", fires on a
    unit with nothing to ignore, or reads the question clauses in the primary;
  * statistics that are approximate where they claim to be exact, or a bar
    missed by floating point (0.7 - 0.5 < 0.2);
  * a manifest or a detector that moved after the pre-registration (their
    sha256 are frozen in V39_PLAN.md: a changed file fails here and the
    notebooks stop before the GPU);
  * a batched greedy path that pads on the wrong side, mixes up the prompts'
    outputs, or loses a batch to an out-of-memory error;
  * a notebook that starts the GPU before this file, or puts a token in a URL.

Run as `python test_v39.py`.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

import clause_ablation as CA
import pretest_v21 as V21
import pretest_v39 as V

FAILS = []
N = [0]


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


TEXT = ("Kimia has a flower bed that is 126 feet long. Kimia wants to fill her flower bed with plants. "
        "Her flowers grow 7 feet wide so she needs to leave 14 feet between every plant (including the plants width). "
        "Each plant costs $4.50 at the store. Once she is back from the store, how much money will she spend "
        "if she buys 3 more plants?")


def part1():
    print("\nPART 1 - units and deletion")
    sents, qi, us = CA.units(TEXT)
    kinds = [u['kind'] for u in us]
    check(len(sents) == 5 and qi == 4, "five sentences, the question last ($4.50 does not split a sentence)")
    check(kinds == ['sentence', 'sentence', 'sentence', 'paren', 'sentence', 'qclause', 'qclause'],
          "units: every non-question sentence, the parenthetical, the leading and the trailing question clause")
    check([u['text'] for u in us if u['kind'] == 'qclause'] == ['Once she is back from the store',
                                                                 'if she buys 3 more plants'], "question clauses")
    v = [CA.delete(sents, u) for u in us]
    check(v[1] == TEXT.replace(' Kimia wants to fill her flower bed with plants.', ''),
          "a sentence deletion removes exactly that sentence")
    check(v[3].count('(') == 0 and 'every plant. Each' in v[3] and '  ' not in v[3],
          "a parenthetical deletion leaves no space before the full stop")
    check(v[5].endswith('at the store. How much money will she spend if she buys 3 more plants?'),
          "a leading question clause goes with its comma and the question starts upper-case")
    check(v[6].endswith('how much money will she spend?'), "a trailing 'if' clause goes with its comma")
    check(all(x != TEXT and len(x) < len(TEXT) for x in v), "every variant is shorter than the problem")
    check(not CA.informative('Kimia wants to fill her flower bed with plants.')
          and CA.informative('They forgot their wallets, so they have to return home.')
          and CA.informative('Half of them are sick.') and CA.informative('It costs $5.'),
          "informative = a digit, a number word or a condition word")


def part2():
    print("\nPART 2 - the added / modified / same labels")
    inst = [f"{n} has {k} apples. {n} buys {k + 2} pears at the {p}. How many fruits?"
            for n, k, p in (('Ann', 3, 'shop'), ('Bo', 4, 'market'), ('Cy', 5, 'store'), ('Di', 6, 'shop'),
                            ('Ed', 7, 'mall'), ('Flo', 8, 'stall'), ('Gus', 9, 'shop'), ('Hal', 2, 'store'),
                            ('Ivy', 3, 'market'), ('Jo', 4, 'kiosk'))]
    words = CA.template_words(inst)
    check({'apple', 'buy', 'pear', 'fruit'} <= words and not ({'ann', 'shop', 'market'} & words),
          "template words: the template's own words stay, names and places (slot fills) go")
    p2 = [s.replace(' How many fruits?', ' Then half of the pears rot. How many fruits?') for s in inst]
    keep = CA.template_words(p2)
    import renderings as RD
    _, _, us = CA.annotate(p2[0], RD.split_sentences(inst[5]), keep, words)
    lab = {u['text']: u['label'] for u in us}
    check(lab == {'Ann has 3 apples.': 'same', 'Ann buys 5 pears at the shop.': 'same',
                  'Then half of the pears rot.': 'added'},
          "a sentence P2 added is ADDED; kept sentences are SAME though their names, numbers and places differ")
    _, _, us = CA.annotate("Ann has 3 apples and a red hat. She buys 5 pears. How many fruits?",
                           ["Ann has 3 apples.", "She buys 5 pears.", "How many fruits?"])
    check([u['label'] for u in us] == ['modified', 'same'], "GSM-Plus style (plain words): a sentence with an "
                                                            "inserted clause is MODIFIED")
    check((CA.ADDED_BELOW, CA.SAME_FROM, CA.CONST_SHARE) == (0.5, 0.75, 0.9), "label thresholds frozen")
    check(CA.spent('It costs $1,250.50 and 3 more.', ['so 1250.5 + 3']) is True
          and CA.spent('It costs $7.', ['no number here']) is False and CA.spent('No digits.', ['x']) is None,
          "the textual ledger: commas, decimals, absent numbers, units without digits")


def part3():
    print("\nPART 3 - the detector and the statistics")
    us = [{'kind': 'sentence', 'text': 'A has 3.', 'informative': True},
          {'kind': 'sentence', 'text': 'A likes it.', 'informative': False},
          {'kind': 'paren', 'text': '(each one)', 'informative': True},
          {'kind': 'qclause', 'text': 'if 4 more', 'informative': True},
          {'kind': 'sentence', 'text': 'B has 5.', 'informative': True}]
    ans = [7.0, 7.0, 7.0004, 7.0, None]
    check(CA.flagged(us, 7.0, ans) == [0, 2], "flagged: informative sentence/paren units with the same answer")
    check(CA.flagged(us, 7.0, ans, kinds=('sentence', 'paren', 'qclause')) == [0, 2, 3], "question clauses on demand")
    check(CA.flagged(us, 7.0, ans, numeric=True) == [0], "numeric units only")
    check(CA.flagged(us, None, ans) == [] and not CA.same_answer(None, None),
          "no full answer: nothing is flagged; two missing answers are not the same answer")
    check(CA.eligible(us) == [0, 2, 4], "eligible units")
    ok = abs(CA.fisher_greater(8, 2, 1, 5) - 0.024475524475524483) < 1e-12 and \
        abs(CA.fisher_greater(21, 9, 30, 30) - 0.05615368801396418) < 1e-12
    check(ok, "Fisher one-sided p equals scipy's on two tables")
    check(abs(CA.chance_hit(4, 2, 2) - 5 / 6) < 1e-12 and CA.chance_hit(4, 0, 2) == 0.0
          and CA.chance_hit(3, 2, 2) == 1.0, "chance of a hit: exact hypergeometric")
    check(abs(CA.tail_at_least([0.5, 0.5], 1) - 0.75) < 1e-12
          and abs(CA.tail_at_least([0.2, 0.5, 1.0], 3) - 0.1) < 1e-12
          and CA.tail_at_least([0.3], 0) == 1.0, "Poisson-binomial tail: exact")
    check(CA.auroc([0.9, 0.5], [0.5, 0.1]) == 0.875 and CA.auroc([], [1.0]) is None, "AUROC with ties as halves")


def _reading(group, flag, flagged=(), labels=('added', 'same', 'same'), analysed=True):
    return {'group': group, 'analysed': analysed, 'flag': flag, 'flagged': list(flagged),
            'eligible': [0, 1, 2], 'labels': list(labels)}


def part4():
    print("\nPART 4 - the pre-registered decision")
    t = [_reading('TARGET', i < 21, [0] if i < 21 else []) for i in range(30)]
    c = [_reading('CONTROL', i < 30, [1] if i < 30 else []) for i in range(60)]
    res = V.analyse(t + c)
    check(abs(res['fT'] - res['fC'] - 0.2) < 1e-9 and not res['det_met'],
          "a difference of exactly 0.20 with p = 0.056 is NOT met (both conditions are required)")
    t = [_reading('TARGET', i < 28, [0] if i < 28 else []) for i in range(40)]
    c = [_reading('CONTROL', i < 40, [1] if i < 40 else []) for i in range(80)]
    res = V.analyse(t + c)
    check(abs(res['fT'] - res['fC'] - 0.2) < 1e-9 and res['det_p'] < 0.05 and res['det_met'],
          "exactly 0.20 with p < 0.05 IS met (floating point does not move the bar)")
    check(res['loc']['n'] == 28 and res['loc']['hits'] == 28 and abs(res['loc']['H0'] - 1 / 3) < 1e-12
          and res['loc_met'] and res['verdict'] == 'GO', "every flagged unit is the added one: localization met, GO")
    big_t = [_reading('TARGET', True, [1]) for _ in range(300)]
    big_c = [_reading('CONTROL', i < 243, [1] if i < 243 else []) for i in range(300)]
    res = V.analyse(big_t + big_c)
    check(res['det_p'] < 1e-6 and not res['det_met'] and not res['loc_met'] and res['verdict'] == 'NO-GO',
          "a 0.19 difference is not met however small p is; flags on SAME units do not localize: NO-GO")
    t9 = [_reading('TARGET', True, [0]) for _ in range(9)] + [_reading('TARGET', False) for _ in range(21)]
    res = V.analyse(t9 + [_reading('CONTROL', False) for _ in range(60)])
    check(res['det_met'] and res['loc']['n'] == 9 and not res['loc_met'] and res['verdict'] == 'WEAK',
          "fewer than 10 flagged TARGET rows: localization not testable, WEAK")
    all3 = [_reading('TARGET', True, [0, 1, 2]) for _ in range(20)]
    res = V.analyse(all3 + [_reading('CONTROL', False) for _ in range(20)])
    check(res['loc']['H'] == 1.0 and res['loc']['H0'] == 1.0 and not res['loc_met'],
          "flagging every unit hits by chance: no localization")
    out = [_reading('TARGET', True, [0], analysed=False)] + [_reading('CONTROL', False)]
    check(V.analyse(out)['groups']['TARGET'] == [], "a row whose greedy answer flipped correctness is not analysed")
    check((V.DET_MIN_DIFF, V.DET_ALPHA, V.LOC_MIN_EXCESS, V.LOC_ALPHA, V.LOC_MIN_ROWS) == (0.20, 0.05, 0.15, 0.05, 10)
          and V.CHANGED == ('added', 'modified') and CA.PRIMARY_KINDS == ('sentence', 'paren'), "bars frozen")
    check(V.MAX_TOKENS == 1024 and 'Answer:' in V21.COT_PROMPT and '{problem}' in V21.COT_PROMPT,
          "budget and prompt: the stored P2 samples' (v22's CoT prompt, 1024 tokens)")


def part5():
    print("\nPART 5 - the manifest, frozen")
    d = json.load(open(V.MANIFEST, encoding='utf-8'))
    rows = d['rows']
    g = {k: [r for r in rows if r['group'] == k] for k in ('TARGET', 'CONTROL', 'GUARD')}
    check(d['seed'] == 39 and d['n'] == 131 and len({r['pid'] for r in rows}) == 131, "seed 39, 131 distinct rows")
    check((len(g['TARGET']), len(g['CONTROL']), len(g['GUARD'])) == (37, 74, 20)
          and len(g['CONTROL']) == V.CONTROL_PER_TARGET * len(g['TARGET']), "37 TARGET, 2 CONTROL per TARGET, 20 GUARD")
    check(all(r['pid'].startswith('gsm-symbolic_p2_') and not V.canonical(r['stored']['plurality'], r['gold'])
              and r['stored']['agree'] >= 3 for r in g['TARGET']), "TARGET: P2, plurality wrong, >= 3 of 5 agree")
    check(all(r['pid'].startswith('gsm-symbolic_p2_') and V.canonical(r['stored']['plurality'], r['gold'])
              and r['stored']['agree'] >= 4 for r in g['CONTROL']), "CONTROL: P2, plurality right, >= 4 of 5 agree")
    check(all(r['pid'].startswith('gsm-plus_') and V.canonical(r['stored']['plurality'], r['gold'])
              and r['stored']['agree'] >= 4 for r in g['GUARD']), "GUARD: GSM-Plus, plurality right, >= 4 of 5")
    same = True
    for r in rows:
        sents, qi, us = CA.units(r['text'])
        mine = [(u['kind'], u['text'], u.get('span'), CA.delete(sents, u), CA.informative(u['text'])) for u in us]
        theirs = [(u['kind'], u['text'], u.get('span'), u['variant'], u['informative']) for u in r['units']]
        same &= mine == theirs
    check(same, "every unit, variant and informative flag in the manifest is what clause_ablation.py makes now")
    check(all(u['label'] in ('added', 'modified', 'same') for r in rows for u in r['units'])
          and all(CA.eligible(r['units']) for r in rows), "every unit labelled; every row has an eligible unit")
    check(d['constants'] == {'ADDED_BELOW': CA.ADDED_BELOW, 'SAME_FROM': CA.SAME_FROM, 'CONST_SHARE': CA.CONST_SHARE,
                             'PRIMARY_KINDS': list(CA.PRIMARY_KINDS)}, "the manifest was built with these constants")
    plan = open('V39_PLAN.md', encoding='utf-8').read()
    want = {f: re.search(r'`' + re.escape(f) + r'`: sha256 \*\*([0-9a-f]{12})\*\*', plan)
            for f in ('clause_ablation.py', V.MANIFEST)}
    check(all(want.values()), "V39_PLAN.md states the frozen sha256 of the detector and the manifest")
    for f, m in want.items():
        if m:
            check(V.sha(f) == m.group(1), f"{f} is the file frozen in V39_PLAN.md ({m.group(1)})")


def part6():
    print("\nPART 6 - offline plumbing (stub run, temp files)")
    tmp = tempfile.mkdtemp()
    out = os.path.join(tmp, 'stub39.json')
    r = subprocess.run([sys.executable, 'pretest_v39.py', '--stub', '--limit', '6', '--out', out, '--no-resume'],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    check(r.returncode == 0 and os.path.exists(out), "the stub run completes")
    rows = {x['pid']: x for x in V.load_rows()}
    if os.path.exists(out):
        recs = json.load(open(out, encoding='utf-8'))['rows']
        check(len(recs) == 6 and all(V.complete(rec, rows[k]) for k, rec in recs.items()), "6 complete records")
        check('PARTIAL run, 6/131' in r.stdout, "a partial run gives no verdict")
    order = V.load_rows()
    check(sorted(x['pid'] for x in order[0::2] + order[1::2]) == sorted(rows), "two shards cover every row once")
    a, b = os.path.join(tmp, 'a.json'), os.path.join(tmp, 'b.json')
    k = order[0]['pid']
    full = {'full': {'raw': '', 'answer': 1.0}, 'units': [{'raw': '', 'answer': 1.0}] * len(order[0]['units'])}
    V._save(a, {k: full}, {})
    V._save(b, {k: {'full': {'raw': '', 'answer': 1.0}, 'units': []}}, {})
    check(V.complete(V.merged([b, a])[k], order[0]), "merging keeps the complete record over a partial one")
    check(V.outputs('pretest_v39.json')[0] == 'pretest_v39.json', "the result files: the main one first")
    try:
        V._shard('2/2')
        check(False, "an impossible shard is refused")
    except ValueError:
        check(True, "an impossible shard is refused")


def part7():
    print("\nPART 7 - the batched greedy path (fake tokenizer and model)")
    try:
        import torch
    except Exception:
        check(True, "(torch absent: skipped)")
        return

    class Tok:
        padding_side = 'right'
        pad_token_id = 0
        eos_token_id = 0
        sides = []

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return '<' + msgs[0]['content'] + '>'

        def __call__(self, prompts, return_tensors='pt', padding=True):
            self.sides.append(self.padding_side)
            ids = [[ord(ch) for ch in p] for p in prompts]
            n = max(map(len, ids))
            pad = [[0] * (n - len(x)) + x if self.padding_side == 'left' else x + [0] * (n - len(x)) for x in ids]
            t = torch.tensor(pad)
            return {'input_ids': t, 'attention_mask': (t != 0).long()}

        def decode(self, ids, skip_special_tokens=True):
            return ''.join(chr(int(i)) for i in ids if int(i))

    class Model:
        calls = []

        def parameters(self):
            yield torch.zeros(1)

        def generate(self, input_ids, attention_mask, max_new_tokens, do_sample, **kw):
            self.calls.append((input_ids.shape[0], do_sample))
            if input_ids.shape[0] > 2:
                raise torch.cuda.OutOfMemoryError('fake')
            outs = []
            for row, m in zip(input_ids, attention_mask):
                real = ''.join(chr(int(i)) for i, k in zip(row, m) if k)
                outs.append(torch.cat([row, torch.tensor([ord(ch) for ch in f" Answer: {len(real)}"])]))
            return torch.stack(outs)

    class Client:
        provider = 'local_hf'
        _local_tokenizer = Tok()
        _local_model = Model()

        def _ensure_local_model(self):
            pass

    prompts = ['a', 'bbb', 'cc', 'dddd', 'e']
    outs = V.greedy_batch(Client(), prompts, 8)
    check(outs == [f" Answer: {len(p) + 2}" for p in prompts],
          "each prompt gets its own continuation, in order, prompt tokens cut off")
    check(Tok.sides and all(s == 'left' for s in Tok.sides) and Client._local_tokenizer.padding_side == 'right',
          "prompts are padded on the left; the tokenizer's own side is restored")
    check(Model.calls == [(5, False), (2, False), (3, False), (1, False), (2, False)],
          "greedy (do_sample=False); an out-of-memory batch is halved until it fits, order kept")


def part8():
    print("\nPART 8 - the notebooks")
    for nb, run in (('MAS_SHT_Kaggle_v39.ipynb', 'pretest_v39.py --parallel'),
                    ('MAS_SHT_Colab_v39.ipynb', "'pretest_v39.py', '--out'")):
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
        t, r = code.find("'test_v39.py'"), code.find(run)
        check(t >= 0 and r >= 0 and t < r, f"{nb} runs test_v39.py before `{run}`")
        check(not any(f'v{k}' in code for k in range(34, 39)), f"{nb} code cells mention no older version")
        check('_tok}@' not in code and '_token}@' not in code and '@github.com' not in code,
              f"{nb} never puts a token in a URL")


def main():
    part1()
    part2()
    part3()
    part4()
    part5()
    part6()
    part7()
    part8()
    print(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed" + (f"; FAILED: {FAILS}" if FAILS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
