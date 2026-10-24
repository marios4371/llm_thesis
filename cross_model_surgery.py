"""
[v40.0] Cross-model Symbolic Surgery: v38's repair, without a verifier, on other model families.

WHY THIS VERSION EXISTS
-----------------------
v38 measured Symbolic Surgery on ONE solver, Qwen2.5-Math-7B-Instruct: on 200
fresh clean GSM-Hard rows ADAPT-RS beat ADAPT-R W9 L0 (p = 0.004). A reader's
first question is whether that is a property of the method or of one model's
LaTeX habits. v40 runs the same 200 rows on models of other families and sizes
and compares the repair with the plain vote, with no verifier at all, so that
every model can be measured the same way (a local 3B model and a 7B on Kaggle).

WHAT HAD TO CHANGE FOR OTHER MODELS (measured 2026-10-22 on stored samples)
-------------------------------------------------------------------------
trace_siv.py and trace_surgery.py are unchanged (frozen since v35 / v38). This
module only decides which text they read and which of their verdicts count.
  1. clean(): chat models write markdown ('= **984**'), '÷', '4 x 6' and money
     as '$8' in plain text; the audit reads none of these, so they are
     rewritten before it runs ('984', '/', '4 * 6', '8'). Lines with a LaTeX
     command are left alone.
  2. complete(): the audit's plain-text path can cut a claim out of a longer
     expression: '2/4/2 = 1' out of '2/4/2 = 1/4', '2.5 + 12 + 2.5 = 30' out of
     '6 + x + 2.5 + 12 + 2.5 = 30', '4 + 60/60 = 4' out of '4 + 60/60 = 4 + 1'.
     On Qwen-Math's LaTeX that never happens; on DeepSeek-Math-7B-RL's plain
     text it refuted 25 of 250 right samples. A plain-text FALSE claim now
     counts only when no operator continues it on either side: DeepSeek 25 -> 1
     right samples refuted, OLMo-2-7B 4 -> 1, and the 1,300 v38 Qwen-Math
     samples are unchanged (123 refutations before and after; replay_v40.py).
  3. parse_final(): the later of the last 'Answer:' line (as the prompt asks)
     and the last \\boxed{}, else the last number; thousands separators and simple
     fractions are read ('1,376' is 1376, '1/4' is 0.25).

THE ARMS (no verifier; all read the same five samples per row)
--------------------------------------------------------------
  SC5     plurality of the five answers (canonical grouping |a - b| < 1e-3)
  SC5-R   plurality of the samples the audit does not refute (v35's refutation
          as a vote filter); if every sample is refuted, SC5
  SC5-RS  SC5-R plus each refuted sample's repair under v38's FROZEN policy
          (trace_surgery.surgery_policy: propagate when the correction reaches
          the answer, resume when the slip recurs in >= 2 samples, else drop);
          a repair the audit still refutes is dropped; empty pool: SC5.
          Every resume drawn is one partial call: calls = 5 + resumes.
With PRM scores (pretest_v40.py --score-prm, Kaggle) the v38 arms ADAPT,
ADAPT-R and ADAPT-RS are read off the same samples too (reported only).
"""
from __future__ import annotations

import hashlib
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import trace_siv as T
import trace_surgery as S

CANON_TOL = 1e-3


def canonical(a: Optional[float], b: Optional[float]) -> bool:
    """The project's grader, Mas_solver.check_correctness: |a - b| < 1e-3."""
    return a is not None and b is not None and abs(a - b) < CANON_TOL


# ---------------------------------------------------------------- 1. the text the audit reads
_BOLD = re.compile(r'(?<!\d)\*\*|\*\*(?!\d)')            # markdown bold; '2**3' (a power) stays
_TIMES_X = re.compile(r'(?<=\d)\s*[xX]\s*(?=\d)')         # '4 x 6' -> '4 * 6'


def clean(raw: str) -> str:
    """Chat-model formatting rewritten so the frozen audit can read it. Idempotent."""
    s = (raw or '').replace('\u00f7', '/').replace('\u2212', '-')
    s = _BOLD.sub('', s)
    s = _TIMES_X.sub(' * ', s)
    return '\n'.join(line if '\\' in line else line.replace('$', '') for line in s.split('\n'))


# ---------------------------------------------------------------- 2. which verdicts count
_OPS = '+-*/×^÷'
_MATH = re.compile(r'\\\[(.+?)\\\]|\\\((.+?)\\\)|\$\$(.+?)\$\$|\$(.+?)\$', re.S)


def complete(claim: str, text: str) -> bool:
    """False only for a plain-text claim that is a fragment of a longer
    expression: an operator right before its left side or right after its right
    side (or a variable after it). Claims inside LaTeX (parsed whole by the
    audit's math path) and claims not found verbatim are kept."""
    spans = [m.span() for m in _MATH.finditer(text)]
    found, start = False, 0
    while True:
        i = text.find(claim, start)
        if i < 0:
            return not found
        start = i + 1
        if any(a <= i < b for a, b in spans):
            continue
        found = True
        before = text[:i].rstrip()
        after = text[i + len(claim):].lstrip()
        bad_left = bool(before) and before[-1] in _OPS + '('
        bad_right = bool(after) and (after[0] in _OPS + '/(' or re.match(r'[A-Za-z]\b', after) is not None)
        if not bad_left and not bad_right:
            return True


def audit(text: str) -> List[List[T.Verdict]]:
    """trace_siv.audit on `text` (already cleaned) with the FALSE verdicts that
    are fragments removed: trace_siv.refutes() of it is v40's refutation."""
    return [[(c, v) for c, v in step if v != T.FALSE or complete(c, text)] for step in T.audit(text)]


def refuted(text: str) -> bool:
    return T.refutes(audit(text), None)


# ---------------------------------------------------------------- 3. the final answer
_NUM = r'[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?'
_ANSWER = re.compile(r'answer\s*(?:is)?\s*[:=]?\s*\$?\s*(' + _NUM + ')', re.I)
_BOXED = re.compile(r'\\boxed\s*\{\s*\\?\$?\s*(' + _NUM + r')')
_LAST = re.compile(r'(?<![\w.])(' + _NUM + ')')


def _number(s: str) -> Optional[float]:
    s = s.replace(' ', '')
    if s.count(',') and re.fullmatch(r'[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:/.*)?', s) is None:
        s = s.split(',')[0]                                  # '1,2' is not a thousands separator
    s = s.replace(',', '')
    try:
        if '/' in s:
            a, b = s.split('/', 1)
            return float(a) / float(b) if float(b) else None
        return float(s)
    except ValueError:
        return None


def parse_final(raw: str) -> Optional[float]:
    """The later of the last 'Answer: x' / 'the answer is x' and the last \\boxed{x};
    with neither, the last number of the text."""
    s = clean(raw)
    marked = [(m.start(), m.group(1)) for pat in (_ANSWER, _BOXED) for m in pat.finditer(s)]
    if marked:
        return _number(max(marked)[1])
    hits = _LAST.findall(s)
    return _number(hits[-1]) if hits else None


# ---------------------------------------------------------------- the surgery, model-free
def operate(text: str, answer: Optional[float]) -> Optional[Dict]:
    """trace_surgery.operate on a cleaned, refuted trace, kept only when its site
    (the first false claim) is itself a complete claim."""
    if answer is None or not refuted(text):
        return None
    try:
        op = S.operate(text, answer)
    except (ValueError, OverflowError, ZeroDivisionError, TypeError):    # e.g. a NaN value in fmt(): no repair
        return None
    if op is None or not complete(op['site']['claim'], text):
        return None
    return op


def text_key(text: str) -> str:
    """The key a PRM score is stored under (pretest_v40.py --score-prm): sha1 of the text."""
    return hashlib.sha1((text or '').encode('utf-8')).hexdigest()


def prepare(samples: Sequence[Dict], prm: Optional[Dict[str, List[float]]] = None) -> List[Dict]:
    """The reading-time view of stored samples ({'raw', 'resume': {'prefix', 'raw'} | None}):
    answers, refutations and repairs recomputed with this module, the stored
    resume attached only when its prefix is the one the frozen surgery cuts.
    With `prm` (text_key -> step rewards) every sample, propagation and resume
    that was scored carries its 'prm'."""
    prm = prm or {}
    out = []
    for s in samples:
        raw = s.get('raw') or ''
        text = clean(raw)
        x = {'raw': raw, 'answer': parse_final(raw), 'audit': audit(text), 'prm': prm.get(text_key(raw))}
        x['refuted'] = T.refutes(x['audit'], None)
        op = operate(text, x['answer']) if x['refuted'] else None
        if op:
            prop = dict(op['prop'])
            prop['audit'] = audit(prop['raw'])
            prop['prm'] = prm.get(text_key(prop['raw']))
            res = s.get('resume') or {}
            resume = None
            if res.get('raw') and op.get('prefix') and clean(res.get('prefix') or '') == op['prefix']:
                resume = {'raw': res['raw'], 'answer': parse_final(res['raw']), 'audit': audit(clean(res['raw'])),
                          'prm': prm.get(text_key(res['raw']))}
            x['surgery'] = {'site': op['site'], 'slips': op['slips'], 'prop': prop, 'prefix': op.get('prefix'),
                            'resume': resume}
        out.append(x)
    return out


def needs_resume(samples: Sequence[Dict], policy: Callable[[Dict, int], str] = None) -> List[int]:
    """Indices of prepared samples the frozen policy sends to RESUME (over all five):
    the only ones a run has to spend a partial call on."""
    policy = policy or S.surgery_policy
    idx = range(len(samples))
    return [i for i, x in enumerate(samples) if x.get('surgery') and x['surgery'].get('prefix')
            and policy(x, S.n_sharing(samples, i, idx)) == S.RESUME]


# ---------------------------------------------------------------- the arms
def plurality(answers: Sequence[Optional[float]]) -> Optional[float]:
    """Largest canonical group; ties go to the group seen first."""
    groups: List[List[float]] = []
    for a in answers:
        if a is None:
            continue
        for g in groups:
            if canonical(a, g[0]):
                g.append(a)
                break
        else:
            groups.append([a])
    return max(groups, key=len)[0] if groups else None


def sc5(samples: Sequence[Dict]) -> Tuple[Optional[float], int]:
    return plurality([x.get('answer') for x in samples]), len(samples)


def sc5_r(samples: Sequence[Dict]) -> Tuple[Optional[float], int]:
    keep = [x.get('answer') for x in samples if x.get('answer') is not None and not x.get('refuted')]
    return (plurality(keep) if keep else sc5(samples)[0]), len(samples)


def _repair(x: Dict, action: str) -> Optional[Dict]:
    s = x.get('surgery') or {}
    rep = s.get('prop') if action == S.PROPAGATE else s.get('resume') if action == S.RESUME else None
    if not rep or rep.get('answer') is None or T.refutes(rep.get('audit') or [], None):
        return None
    return rep


def sc5_rs(samples: Sequence[Dict], policy: Callable[[Dict, int], str] = None) -> Tuple[Optional[float], int]:
    policy = policy or S.surgery_policy
    idx = range(len(samples))
    pool, calls = [], len(samples)
    for i, x in enumerate(samples):
        if x.get('answer') is None:
            continue
        if not x.get('refuted'):
            pool.append(x['answer'])
            continue
        if not x.get('surgery'):
            continue
        action = policy(x, S.n_sharing(samples, i, idx))
        if action not in S.ACTIONS:
            raise ValueError(f'surgery_policy returned {action!r}, not one of {S.ACTIONS}')
        calls += action == S.RESUME and x['surgery'].get('resume') is not None      # a drawn resume is a call
        rep = _repair(x, action)
        if rep is not None:
            pool.append(rep['answer'])
    return (plurality(pool) if pool else sc5(samples)[0]), calls


ARMS: Dict[str, Callable[[Sequence[Dict]], Tuple[Optional[float], int]]] = {
    'SC5': sc5, 'SC5-R': sc5_r, 'SC5-RS': sc5_rs}
