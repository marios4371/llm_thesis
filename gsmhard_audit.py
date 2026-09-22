"""
[v20.0] A mechanical audit of how GSM-Hard was BUILT.

WHY THIS IS A SEPARATE FINDING AND NOT A FOOTNOTE
-------------------------------------------------
GSM-Hard is the standard benchmark for tool-augmented and program-aided math
reasoning, and this project has spent nineteen versions measuring selection,
verification, routing and reconciliation against it, every one of them null.
The audit below says why. GSM-Hard was made by taking GSM8K's test set and
replacing one small quantity with a large one, in two places: the problem TEXT
and the gold PROGRAM. When the original value occurred more than once in the
text -- and it often did, because small numbers repeat, and because "two" and
"2" are the same quantity -- the substitution hit every occurrence in the text
and only one assignment in the program. From that point the text and the gold
describe different problems, and any system that reads the text is graded
against the answer to a question nobody asked.

Worked example, gsm-hard_933. The text reads "He did his chores for 8404276
weeks. Then he bought ice cream cones for himself and 8404276 friends"; the
gold program has `weeks_chores_done = 8404276` and `ice_cream_friends = 3`.
The original had 3 weeks and 3 friends. A solver that reads 8404276 friends is
marked wrong for reading what it was given.

THE THREE CHANNELS
------------------
  A  collision        the perturbed value stands in >= 2 slots of the text and
                      exactly 1 of the program
  B  never perturbed  the perturbed value is in the text and nowhere in the
                      program -- the gold was left at its GSM8K original
  C  platinum dropped the underlying GSM8K question was removed by madrylab's
                      expert re-annotation as ambiguous or unanswerable

A and B are mechanical and need no judgement. C is human and entirely
independent of them, which is the point: it is not this project marking its own
homework. Measured 2026-09-22 over the 979 auditable rows of 1319:

    A  90  ( 9.2%)    B  55  (5.6%)    C  110  (8.4% of the 1316 matched)
    union, restricted to the auditable rows:  209 = 21.3%

Read the overlap carefully. A and C agree on only 11 rows, so the mechanical
channel and the human one are finding largely different broken items, and the
union is a floor rather than an estimate. The union is taken on one
population -- the auditable rows -- because C is defined on rows A and B
cannot see, and mixing the denominators would inflate it (24.5% instead of
21.3%).

WHAT FOLLOWS FROM IT
--------------------
Not "the earlier results were wrong". They were right about the systems and
wrong about what the number meant. On the gold-consistent subset the picture
is: CoT 80.0%, PAL 80.0%, v17.1 82.5%, and ORACLE(CoT, PAL) 85.0% -- with at
least 4 of the 6 rows both solvers miss still defective. True accuracy is near
90% and the true oracle near 92.5%, so **roughly 2.5pp of selection headroom
exists on this benchmark**, which is less than the noise any of the selection
experiments were trying to resolve. It also removes PAL's reported advantage
on large numbers: on clean rows the two tie, and their errors coincide on 6 of
8.

    python gsmhard_audit.py                 # the table
    python gsmhard_audit.py --json out.json # and the row ids
    python gsmhard_audit.py --show 5        # with worked examples

Needs `reasoning-machines/gsm-hard`, `madrylab/gsm8k-platinum` and
`openai/gsm8k`. The scoring functions below are pure and take a row dict, so
test_gsmhard_audit.py exercises them with no network.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

NUMBER_RE = re.compile(r'\d[\d,]*\.?\d*')

#: Spelled quantities count as occurrences. This is load-bearing: gsm-hard_266
#: perturbed "the first two kinds of beds" into "the first 736424 kinds",
#: which is only visible as a collision if "two" is read as 2 in the original.
WORDNUM = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
           'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
           'twelve': 12, 'dozen': 12, 'twice': 2, 'thrice': 3, 'double': 2,
           'triple': 3}

#: Below this, a changed number is not a GSM-Hard perturbation but ordinary
#: variation between the docstring and the text.
PERTURBATION_FLOOR = 1000.0


def numbers_in_text(text: str, words: bool = True) -> List[float]:
    out: List[float] = []
    for m in NUMBER_RE.finditer(text or ''):
        tok = m.group(0).replace(',', '').rstrip('.')
        try:
            out.append(float(tok))
        except ValueError:
            pass
    if words:
        for w, v in WORDNUM.items():
            out += [float(v)] * len(re.findall(r'\b' + w + r'\b', text or '', re.I))
    return out


def original_question(code: str) -> str:
    """GSM-Hard keeps the unperturbed GSM8K problem in the gold program's
    docstring, which is what makes this audit possible at all."""
    m = re.search(r'"""(.*?)"""', code or '', re.S)
    return m.group(1).strip() if m else ''


def code_literals(code: str) -> List[float]:
    """Numeric literals on the right-hand side of assignments -- the values the
    gold program actually computes with."""
    body = re.sub(r'""".*?"""', '', code or '', flags=re.S)
    out: List[float] = []
    for line in body.split('\n'):
        if '=' not in line:
            continue
        for m in re.finditer(r'(?<![\w.])\d+\.?\d*', line.split('=', 1)[1]):
            try:
                out.append(float(m.group(0)))
            except ValueError:
                pass
    return out


def perturbed_value(text: str, code: str) -> Optional[float]:
    """The large value GSM-Hard substituted in: present in the perturbed text,
    absent from the original docstring. None when the row is not auditable."""
    doc = original_question(code)
    if not doc:
        return None
    old = set(numbers_in_text(doc))
    cand = [v for v in numbers_in_text(text)
            if v not in old and v >= PERTURBATION_FLOOR]
    return max(cand) if cand else None


def audit_row(text: str, code: str) -> Dict[str, object]:
    """Classify one GSM-Hard row. Pure: no dataset, no network."""
    out: Dict[str, object] = {'auditable': False, 'defect': None,
                              'perturbed': None, 'n_text': 0, 'n_code': 0}
    p = perturbed_value(text, code)
    if p is None:
        out['reason'] = ('no docstring' if not original_question(code)
                         else 'no large substituted value found')
        return out
    n_text = sum(1 for v in numbers_in_text(text) if v == p)
    n_code = sum(1 for v in code_literals(code) if v == p)
    out.update(auditable=True, perturbed=p, n_text=n_text, n_code=n_code)
    if n_text >= 2 and n_code == 1:
        out['defect'] = 'collision'
    elif n_text >= 1 and n_code == 0:
        out['defect'] = 'never_perturbed'
    return out


def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or '').strip().lower())


def run(show: int = 0) -> Tuple[Dict[str, object], Dict[str, List[str]]]:
    from datasets import load_dataset
    hard = load_dataset('reasoning-machines/gsm-hard', split='train')
    plat = {_norm(r['question'])
            for r in load_dataset('madrylab/gsm8k-platinum', 'main', split='test')}
    gsm = {_norm(r['question'])
           for r in load_dataset('openai/gsm8k', 'main', split='test')}

    st: Counter = Counter()
    sets: Dict[str, List[str]] = {'collision': [], 'never_perturbed': [],
                                  'platinum_dropped': [], 'auditable': []}
    shown = 0
    for i, r in enumerate(hard):
        pid = f'gsm-hard_{i}'
        a = audit_row(r['input'], r['code'])
        if not a['auditable']:
            st['not_auditable'] += 1
        else:
            st['auditable'] += 1
            sets['auditable'].append(pid)
            if a['defect']:
                st[str(a['defect'])] += 1
                sets[str(a['defect'])].append(pid)
                if show and shown < show and a['defect'] == 'collision':
                    shown += 1
                    print(f"\n--- {pid}  perturbed {a['perturbed']:.0f} appears "
                          f"{a['n_text']}x in the text, {a['n_code']}x in the gold")
                    print('TEXT :', r['input'][:300])
                    print('ORIG :', original_question(r['code'])[:300])
        doc = _norm(original_question(r['code']))
        if doc and doc in gsm:
            st['matched_gsm8k'] += 1
            if doc not in plat:
                st['platinum_dropped'] += 1
                sets['platinum_dropped'].append(pid)
    st['total'] = len(hard)
    return dict(st), sets


def report(st: Dict[str, object], sets: Dict[str, List[str]]) -> None:
    A = int(st.get('auditable', 0)) or 1
    m = int(st.get('matched_gsm8k', 0)) or 1
    mech = set(sets['collision'])
    plt = set(sets['platinum_dropped'])
    print('\n' + '=' * 72)
    print(f"  GSM-Hard rows                              {st.get('total')}")
    print(f"  auditable (a large substitution is visible) {st.get('auditable')}")
    print()
    print(f"  A  collision         {st.get('collision', 0):4d}"
          f"  = {100*int(st.get('collision', 0))/A:5.1f}% of auditable")
    print(f"  B  never perturbed   {st.get('never_perturbed', 0):4d}"
          f"  = {100*int(st.get('never_perturbed', 0))/A:5.1f}%")
    print(f"  C  platinum dropped  {st.get('platinum_dropped', 0):4d}"
          f"  = {100*int(st.get('platinum_dropped', 0))/m:5.1f}% of {m} matched to GSM8K")
    print()
    print(f"  A and C overlap on {len(mech & plt)} rows -- they are finding "
          f"largely DIFFERENT broken items")
    # The union is reported on ONE denominator. C is a property of the
    # underlying GSM8K question and so is defined on rows A and B cannot see;
    # counting those in the numerator while dividing by the auditable rows
    # would inflate the headline. Restrict to the auditable set, which is the
    # population every channel is defined on.
    aud = set(sets.get('auditable', []))
    un = (mech | plt | set(sets['never_perturbed'])) & aud
    print(f"  union, on auditable  {len(un):4d}  = {100*len(un)/A:5.1f}%"
          f"   (C contributes {len(plt & aud)} of its {len(plt)} here)")
    print()
    print("  This is a floor, not an estimate: only defects a machine can see "
          "without\n  reading the problem are counted.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='', help='write the row ids here')
    ap.add_argument('--show', type=int, default=0,
                    help='print this many worked collision examples')
    args = ap.parse_args()
    st, sets = run(show=args.show)
    report(st, sets)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as fh:
            json.dump({'stats': st, 'rows': sets}, fh, indent=1)
        print(f"\n  ids written to {args.json}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
