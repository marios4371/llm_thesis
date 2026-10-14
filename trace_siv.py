"""
[v35.0] Trace-level SIV: symbolic refutation of the Solver's own claims,
composed with verifier-routed sampling (v22's ADAPT rule).

WHY THIS VERSION EXISTS
-----------------------
SIV (siv_module.py) was built as an execution audit: deterministic, CAS-based,
zero LLM calls, aimed at "wrong arithmetic". Every measurement since v14 says
the same two things about it:
  * when it REJECTS it is right (P2: 13/13 rejected blueprints were wrong;
    gsm-hard: 98.1%), but a PASS means little;
  * its bottleneck is the BLUEPRINT, a separate formalization the 7B Architect
    gets right on only 13/48 P2 rows (siv-blind-on-p2), at an extra call.
A Qwen-Math chain of thought already IS an execution trace: it writes its
arithmetic out as LaTeX claims (`61 + 8 + 13 + 10 + 58 = 140`). So this
version drops the blueprint and audits the trace itself, and it uses SIV only
in the role the data supports: REFUTATION, never certification.

WHAT THE STORED POOLS SHOW (no GPU; replay_v35.py reproduces every number)
-------------------------------------------------------------------------
  * Claim precision. On the untouched v21 P2 set every FALSE verdict inside a
    right sample (6/6) is a genuinely false claim the model later recovered
    from. Sample level: a FALSE claim sits in 0.3% of right P2 samples (2/631)
    and 2.5% of right MATH samples, against 19-28% of wrong ones.
  * The PRM is blind to a third of them: 29-40% of steps holding a provably
    false claim get a step reward >= 0.9.
  * Correlated slips. On P2 the SAME false sum recurs across independent
    samples (`61+8+13+10+58 = 140` in 4/5 samples, all scored 0.99-1.00), so
    majority voting amplifies it and the same-policy PRM cannot see it. 14 of
    SC@5's 58 P2 failures have a refutable wrong majority.
  * The regime problem. PRM best-of-n beats voting on P2 (self-consistent
    errors) but loses to it on MATH-500 (scattered errors: the PRM saturates at
    1.0 on wrong singletons, e.g. `64^2 = 64 (mod 11)`, `9999 = 3*3*11*11*11`).
    Those singletons are computational slips, exactly what the audit refutes.

THE RULE (ADAPT-R; the gate and the selection are v22's, unchanged)
------------------------------------------------------------------
  1. Draw C1..C3; audit each trace; drop the refuted ones.
  2. b = the best survivor by last-step PRM reward (ties: lower position).
  3. If b exists and min(step rewards of b) >= 0.85: answer b. 3 calls.
  4. Otherwise draw C4, C5, audit them, and answer with the best survivor of
     C1..C5 (if all five are refuted, v22's pick over all five). 5 calls.
The audit is pure Python + SymPy on the text: no model call, no training.

WHAT IS AND IS NOT NEW (checked 2026-10-13)
-------------------------------------------
Not new: filtering candidates with a tool before a reward model picks (T1,
Kang et al., ICLR 2026, arXiv 2504.04718: an LLM WRITES verification code per
candidate, distilled, about 2x compute); CAS/SMT step checking after an LLM
formalizer (MATH-VF, arXiv 2505.20869); PRM best-of-n; uncertainty-gated
sampling.
New here: (i) no LLM in the checker: the trace's own claims are parsed and
executed, so the audit is free and deterministic; (ii) correlated arithmetic
slips across samples, which neither voting nor a same-policy PRM can detect;
(iii) the error-structure account of when PRM selection beats voting, and the
evidence that refutation removes that dependence; (iv) composition with
verifier-routed compute.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import score_prm_v22 as P

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                            implicit_multiplication_application, convert_xor,
                                            factorial_notation)
    _TR = standard_transformations + (implicit_multiplication_application, convert_xor, factorial_notation)
    SYMPY = True
except ImportError:                                         # pragma: no cover
    SYMPY = False

TRUE, FALSE = 'TRUE', 'FALSE'
Verdict = Tuple[str, str]                                   # (claim text, TRUE / FALSE)

GATE_K = 3              # plain samples drawn first (v22)
FULL_K = 5              # samples after the gate fires (v22)
GATE_MIN = 0.85         # v22's ADAPT threshold on the min step reward
MAX_SEG = 400           # longer math segments are skipped (SymPy time, not precision)

_MATH_SEG = re.compile(r'\\\[(.+?)\\\]|\\\((.+?)\\\)|\$\$(.+?)\$\$|\$(.+?)\$', re.S)
_PLAIN = re.compile(r'(?<![\w.])(\d[\d.,]*(?:\s*[-+*/×]\s*\d[\d.,]*)+\s*=\s*\d[\d.,]*)(?![\w.])')
_CURRENCY = '¥€£₹₩'
# out of scope: anything whose meaning is not a plain equality of numbers
_BAD = ('\\overline', '\\ldots', '\\cdots', '\\dots', '\\vec', '\\begin', '\\int', '\\sum', '\\prod', '\\lim',
        '\\sin', '\\cos', '\\tan', '\\log', '\\ln', '\\pi', '\\infty', '\\angle', '\\triangle', '\\pm', '\\mp',
        '\\in ', '\\cup', '\\cap', '\\subset', '\\to', '\\rightarrow', '\\Rightarrow', '\\implies', '\\deg',
        '\\lfloor', '\\lceil', '\\max', '\\min', '\\gcd', '\\operatorname', '\\mathbf', '\\hat', '\\bar', '\\%', '%',
        '\\circ', '°', '\\prime', "'", '\\perp', '\\parallel', '\\sim', '\\cong', '\\neq', '\\ne ', '\\le', '\\ge',
        '<', '>', '\\lt', '\\gt', '|', '\\mid', '\\{', ':', ';', '?', '!', '\\choose', '\\over', '\\equiv', '\\mod',
        '\\pmod', '\\bmod', '\\approx', '\\boxed', '\\sqrt[')
_MOD = re.compile(r'\\(?:pmod|bmod)\s*\{?\s*(\d+)\s*\}?|\(\s*\\(?:text\{\s*mod\s*\}|mod)\s*(\d+)\s*\)'
                  r'|\\mod\s*\{?\s*(\d+)\s*\}?')
_SAFE = re.compile(r'[0-9A-Za-z_+\-*/(),.\s]*')


# ---------------------------------------------------------------------------
# LaTeX -> SymPy
# ---------------------------------------------------------------------------

def _brace(s: str, i: int) -> int:
    d = 0
    for j in range(i, len(s)):
        if s[j] == '{':
            d += 1
        elif s[j] == '}':
            d -= 1
            if d == 0:
                return j
    return -1


def _cmd(s: Optional[str], name: str, fmt: str, nargs: int) -> Optional[str]:
    """Rewrite \\name{a}{b} (nargs=2) or \\name{a} (nargs=1) with fmt, repeatedly."""
    for _ in range(20):
        if s is None:
            return None
        k = s.find('\\' + name + '{')
        if k < 0:
            return s
        i = k + len(name) + 1
        j = _brace(s, i)
        if j < 0:
            return None
        if nargs == 1:
            s = s[:k] + fmt.format(s[i + 1:j]) + s[j + 1:]
            continue
        if j + 1 >= len(s) or s[j + 1] != '{':
            return None
        m = _brace(s, j + 1)
        if m < 0:
            return None
        s = s[:k] + fmt.format(s[i + 1:j], s[j + 2:m]) + s[m + 1:]
    return s


def _unit(m) -> str:
    w = re.sub(r'[^A-Za-z0-9]', '', m.group(1))
    return f' u_{w} ' if w else ' '


def normalize(s: str) -> Optional[Tuple[str, frozenset]]:
    """One side of a claim -> (python expression, unit-symbol names), or None if
    out of scope. Words inside \\text{} become UNIT SYMBOLS, so `19 min + 18 min
    = 37 min` is checkable while `60 min = 1 hour` or `11 remainder 8` is not."""
    s = s.replace('\\left', '').replace('\\right', '').replace('\\!', '').replace('\\,', '').replace('\\;', ' ')
    s = s.replace('\\displaystyle', '').replace('\\$', '').replace('$', '')
    for c in _CURRENCY:
        s = s.replace(c, '')
    s = re.sub(r'\^\s*\{\s*\\circ\s*\}|\^\s*\\circ', '', s)
    s = re.sub(r'\\(?:text|mathrm|textbf|mbox|textit|mathit)\s*\{([^{}]*)\}', _unit, s)
    s = re.sub(r'\|\s*([A-Za-z])\s*\|', r'\1', s)          # |P| = the size of set P
    s = s.strip().rstrip('.,').strip()
    if not s or any(b in s for b in _BAD):
        return None
    if re.search(r'\d\s*\\[dt]?frac', s):                   # mixed numbers like 2\frac{1}{2}
        return None
    s = _cmd(s, 'dfrac', '(({0})/({1}))', 2)
    s = _cmd(s, 'tfrac', '(({0})/({1}))', 2)
    s = _cmd(s, 'frac', '(({0})/({1}))', 2)
    s = _cmd(s, 'binom', 'binomial({0}@{1})', 2)
    s = _cmd(s, 'sqrt', 'sqrt({0})', 1)
    if s is None or '\\' in s.replace('\\times', '').replace('\\cdot', '').replace('\\div', ''):
        return None
    s = s.replace('\\times', '*').replace('\\cdot', '*').replace('\\div', '/').replace('×', '*').replace('÷', '/')
    s = s.replace('·', '*').replace('−', '-').replace('–', '-')
    s = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', s)            # thousands separators
    if ',' in s:
        return None
    s = s.replace('@', ',')
    s = s.replace('{', '(').replace('}', ')').replace('[', '(').replace(']', ')').replace('^', '**')
    if not _SAFE.fullmatch(s) or re.search(r'(?<!\d)\.|\.(?!\d)', s):
        return None                                          # only numbers, names, operators reach SymPy
    if re.search(r'(?<![A-Za-z_0-9])[A-Za-z]\s*\(', re.sub(r'sqrt\(|binomial\(', '', s)):
        return None                                          # f(0), g(x): function notation
    units = frozenset(re.findall(r'u_\w+', s))
    if re.search(r'[A-Za-z]{2,}', re.sub(r'u_\w+|sqrt|binomial', '', s)):
        return None
    return s.strip(), units


def _parse(s: str, units: frozenset):
    loc = {c: sp.Symbol(c) for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'}
    loc.update({u: sp.Symbol(u) for u in units})
    try:
        return parse_expr(s, local_dict=loc, transformations=_TR, evaluate=True)
    except Exception:
        return None


def _real(e) -> Optional[float]:
    try:
        v = complex(sp.N(e, 30))
    except Exception:
        return None
    return None if abs(v.imag) > 1e-12 else v.real


def _decimals(lit: str) -> Optional[int]:
    m = re.fullmatch(r'\s*-?\d+(?:\.(\d+))?\s*(?:\*?\s*u_\w+)?\s*', lit)
    return None if not m else len(m.group(1) or '')


def _tol(a: str, b: str, va: float, vb: float, approx: bool) -> float:
    """The rounding the claim itself shows: a side written as 33.33 tolerates 0.005."""
    tol = 1e-9 * max(1.0, abs(va), abs(vb))
    for lit in (a, b):
        d = _decimals(lit)
        if d is not None:
            tol = max(tol, 0.5 * 10 ** (-d) + 1e-9)
    if approx:
        tol = max(tol, 0.01 * max(abs(va), abs(vb)))
    return tol


def value(raw: str) -> Optional[float]:
    """Numeric value of a unit-free side, else None."""
    n = normalize(raw)
    if not n or n[1]:
        return None
    e = _parse(*n)
    if e is None or getattr(e, 'free_symbols', {1}):
        return None
    return _real(e)


# ---------------------------------------------------------------------------
# claims
# ---------------------------------------------------------------------------

def check_pair(a_raw: str, b_raw: str, approx: bool = False) -> Optional[str]:
    """TRUE / FALSE / None (uncheckable) for the claim a = b. FALSE is returned
    only when both sides have the same shape (plain numbers, the same unit
    symbols linearly, or one variable) and differ beyond the shown rounding."""
    na, nb = normalize(a_raw), normalize(b_raw)
    if not na or not nb:
        return None
    (a, ua), (b, ub) = na, nb
    ea, eb = _parse(a, ua), _parse(b, ub)
    if ea is None or eb is None:
        return None
    try:
        fa, fb = ea.free_symbols, eb.free_symbols
    except Exception:
        return None
    if (ua and not fa) or (ub and not fb):
        return None                                          # a unit vanished (0 * unit)
    if not fa and not fb:
        if re.fullmatch(r'\s*\d+\s*/\s*\d+\s*', a) and re.fullmatch(r'\s*\d+\s*', b):
            n_, d_ = [int(x) for x in re.findall(r'\d+', a)]
            if d_ and n_ % d_ and int(b) == n_ // d_:
                return None                                  # `629 \div 7 = 89`: integer division
        va, vb = _real(ea), _real(eb)
        if va is None or vb is None:
            return None
        return TRUE if abs(va - vb) <= _tol(a, b, va, vb, approx) else FALSE
    if fa != fb:
        # a unit on one side only (`60 x 1/30 = 2 miles`) may confirm, never refute:
        # unit conversions (`1.25 hours = 1.25 x 60`) look exactly the same
        both = fa | fb
        if len(both) == 1 and str(next(iter(both))).startswith('u_') and (not fa or not fb):
            u = next(iter(both))
            va, vb = _real(ea.subs(u, 1)), _real(eb.subs(u, 1))
            if va is not None and vb is not None and abs(va - vb) <= _tol(a, b, va, vb, approx):
                return TRUE
        return None
    names = {str(x) for x in fa}
    if all(n.startswith('u_') for n in names):
        try:
            gens = sorted(fa, key=str)
            pa, pb = sp.Poly(ea, *gens), sp.Poly(eb, *gens)
        except Exception:
            return None
        if pa.total_degree() != 1 or pb.total_degree() != 1:
            return None
        ca, cb = dict(pa.terms()), dict(pb.terms())
        if set(ca) != set(cb):
            return None
        for k in ca:
            va, vb = _real(ca[k]), _real(cb[k])
            if va is None or vb is None:
                return None
            if abs(va - vb) > _tol(a, b, va, vb, approx):
                return FALSE
        return TRUE
    if len(fa) != 1 or any(n.startswith('u_') for n in names):
        return None
    try:
        d = sp.simplify(ea - eb)
    except Exception:
        return None
    if d == 0:
        return TRUE
    dv = _real(d) if not d.free_symbols else None
    if dv is not None and abs(dv) > 1e-9:
        return FALSE                                         # two expressions in x differing by a constant
    return None


def _congruences(seg: str) -> List[str]:
    """X = Y \\equiv Z \\pmod{m}: consecutive integer parts must agree mod m."""
    m = _MOD.search(seg)
    if not m:
        return []
    mod = int(next(g for g in m.groups() if g))
    vals = []
    for part in re.split(r'\\equiv|=', _MOD.sub('', seg)):
        if not part.strip():
            continue
        v = value(part)
        vals.append(int(round(v)) if v is not None and abs(v - round(v)) < 1e-9 else None)
    return [TRUE if (x - y) % mod == 0 else FALSE
            for x, y in zip(vals, vals[1:]) if x is not None and y is not None and mod]


def segments(step: str) -> List[str]:
    """The math pieces of one step: LaTeX segments, plain `12 + 15 = 27`, split on
    line breaks, \\quad and connective words (`x = 0 \\text{or} x = 5`)."""
    segs = [next(g for g in m.groups() if g is not None) for m in _MATH_SEG.finditer(step)]
    segs += [m.group(1) for m in _PLAIN.finditer(_MATH_SEG.sub(' ', step))]
    out = []
    for seg in segs:
        seg = seg.replace('\\\\', '\n').replace('&', '')
        for piece in re.split(r'\n|\\q?quad|\\text\{\s*(?:or|and|so|since|if|then|thus|hence)\b[^{}]*\}', seg):
            if piece and piece.strip() and len(piece) <= MAX_SEG:
                out.append(piece)
    return out


def _runon(parts: Sequence[str], i: int) -> bool:
    """`a = b op c = d`, the idiom of applying a new operation mid-chain
    (`361/400 = 0.9025 \\times 100 = 90.25`): parts[i] equals the head of parts[i+1]."""
    m = re.match(r'^(.*?)(\s*(?:[-+]|\\times|\\cdot|\\div|\*|/)\s*[\d.]+\s*(?:\\text\{[^{}]*\})?\s*)$',
                 parts[i + 1].strip())
    if not m:
        return False
    left, head = value(parts[i]), value(m.group(1))
    return left is not None and head is not None and abs(left - head) <= 1e-6 * max(1.0, abs(left))


def check_step(step: str) -> List[Verdict]:
    out: List[Verdict] = []
    for seg in segments(step):
        if '\\equiv' in seg:
            out += [(seg.strip()[:160], v) for v in _congruences(seg)]
            continue
        toks = re.split(r'(=|\\approx|≈)', seg)
        parts, rels = toks[0::2], toks[1::2]
        for i in range(len(parts) - 1):
            a, b = parts[i], parts[i + 1]
            if not a.strip() or not b.strip():
                continue
            v = check_pair(a, b, approx=rels[i] != '=')
            if v == FALSE and _runon(parts, i):
                v = None
            if v:
                out.append((f"{a.strip()} {rels[i]} {b.strip()}"[:160], v))
    return out


_ANCHOR = re.compile(r'^\s*([^=]+?)\s*=\s*([^=]+?)\s*$')


def check_anchors(steps: Sequence[str]) -> List[Tuple[int, str, str]]:
    """Consecutive one-variable equations that share a nonzero numeric side
    verbatim (`345 = 180 + |P| + 2|P| - 30 - 75 - 75 + 15`, then `345 = 180 +
    3|P| - 165 + 15`, math500_214) must agree: the other sides may differ by no
    constant. A zero shared side is skipped (`x = 0 or x - 5 = 0` is a case split)."""
    eqs = []
    for si, st in enumerate(steps):
        for seg in segments(st):
            if seg.count('=') == 1 and '\\approx' not in seg and '\\equiv' not in seg:
                m = _ANCHOR.match(seg)
                if m:
                    eqs.append((si, m.group(1).strip(), m.group(2).strip()))
    out = []
    for (_, l1, r1), (s2, l2, r2) in zip(eqs, eqs[1:]):
        for same, shared, x, y in ((l1 == l2, l1, r1, r2), (r1 == r2, r1, l1, l2)):
            if not same:
                continue
            sv = value(shared)
            if sv is None or abs(sv) < 1e-12:
                continue
            na, nb = normalize(x), normalize(y)
            if not na or not nb or na[1] or nb[1]:
                continue
            ea, eb = _parse(*na), _parse(*nb)
            if ea is None or eb is None or len(ea.free_symbols | eb.free_symbols) != 1:
                continue
            try:
                d = sp.simplify(ea - eb)
            except Exception:
                continue
            if d == 0:
                out.append((s2, f"{x} == {y}", TRUE))
            elif not d.free_symbols:
                out.append((s2, f"{x} == {y}", FALSE))
    return out


def audit(raw: str) -> List[List[Verdict]]:
    """Per PRM step (score_prm_v22.split_steps), the verdicts of its checkable claims."""
    if not SYMPY:
        return []
    steps = P.split_steps(raw or '')
    per: List[List[Verdict]] = []
    for st in steps:
        try:
            per.append(check_step(st))
        except Exception:
            per.append([])
    try:
        for si, txt, v in check_anchors(steps):
            per[si].append(('ANCHOR ' + txt, v))
    except Exception:
        pass
    return per


# ---------------------------------------------------------------------------
# the refutation policy and the rule
# ---------------------------------------------------------------------------

def refutes(verdicts: List[List[Verdict]], rewards: Optional[Sequence[float]]) -> bool:
    """Does this audit remove the sample from the pool?

    verdicts[i] is the list of (claim, TRUE/FALSE) of PRM step i; rewards[i]
    is the PRM's reward for that step (may be None or shorter than verdicts).

    Policy (frozen 2026-10-14, V35_PLAN.md section 3): ANY provably false claim,
    whatever the PRM thinks of its step. Restricting refutation to steps the
    PRM doubts loses exactly the correlated slips (`61+8+13+10+58 = 140`,
    scored 0.99-1.00 in 4/5 samples), the errors only the audit can see; and
    every FALSE verdict audited inside a right sample was a genuinely false
    claim, so recall costs no precision here. `rewards` is deliberately unused.
    """
    return any(v == FALSE for step in verdicts for _, v in step)


def survivors(samples: Sequence[Dict], idx: Sequence[int]) -> List[int]:
    """The indices in idx whose sample has an answer, a PRM score and no refutation."""
    return [i for i in idx if samples[i].get('answer') is not None and samples[i].get('prm')
            and not refutes(samples[i].get('audit') or [], samples[i].get('prm'))]


def best(samples: Sequence[Dict], idx: Sequence[int]) -> Optional[int]:
    """v22's pick: the highest last-step reward, ties to the lower position."""
    idx = [i for i in idx if samples[i].get('answer') is not None and samples[i].get('prm')]
    return None if not idx else max(idx, key=lambda i: (samples[i]['prm'][-1], -i))


def adapt_r(samples: Sequence[Dict]) -> Tuple[Optional[float], int]:
    """(answer, solver calls) of ADAPT-R. Each sample needs 'answer', 'prm' and
    'audit' (= audit(raw)). Reads C4, C5 only when the gate fires."""
    first = list(range(min(GATE_K, len(samples))))
    b = best(samples, survivors(samples, first))
    if b is not None and min(samples[b]['prm']) >= GATE_MIN:
        return samples[b]['answer'], GATE_K
    full = list(range(min(FULL_K, len(samples))))
    b = best(samples, survivors(samples, full))
    if b is None:
        b = best(samples, full)                              # everything refuted: v22's pick
    return (None if b is None else samples[b]['answer']), FULL_K
