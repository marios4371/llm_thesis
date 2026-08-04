"""
Deterministic blueprint repair — mechanical defects, fixed without an LLM.

WHY THIS EXISTS
---------------
v13.0 and v14.0 both answered a failed SIV audit the same way: hand the error
report back to the Mathematician and ask for a fresh derivation. Measured
fix-rate: 6/39 (15.4%) then 9/51 (18%). Asking a 7B model to re-derive throws
away a derivation that is usually *conceptually fine* and gambles on a better
one.

Classifying every structural defect in the v14.0 run (25 rows, the first run
that recorded blueprint contents at all) shows why that gamble is a bad trade —
almost none of them are reasoning errors:

  ~9  unbalanced subscript:  givens['hotel_nights'] * givens['hotel_cost'
                             -> the ']' is simply missing
  ~10 near-miss name:        givens['peach_bought']   vs declared 'peaches_bought'
                             givens['monthly_hanger'] vs declared 'honthly_hanger'
                             total_cost_pick          vs computed 'total_cost_picking'
  ~3  computed var read as a given:
                             number_of_intervals = ...          (computed on line 1)
                             total = givens['number_of_intervals']  (line 2)
  ~2  numeric value stored as a string: {"number_of_houses": "3"}

Every one of those is a *transcription* defect with a single mechanically
correct answer. This module fixes them in microseconds, deterministically,
before any LLM repair is considered — so the expensive path is reserved for
defects that genuinely need re-derivation.

SAFETY
------
A wrong fuzzy match would convert a loudly-broken blueprint into a silently-wrong
one, which is strictly worse than leaving it broken. So name resolution requires
BOTH a high similarity score AND a unique winner; ties and weak matches are left
alone for the LLM path. Every repair is returned in a log so the CSV records what
was changed and nothing happens invisibly.
"""

import difflib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

# A name must be at least this similar to a declared given/computed variable
# before we dare rewrite it. 0.82 accepts 'peach_bought' -> 'peaches_bought'
# and 'honthly_hanger_rental' -> 'monthly_hanger_rental' (single-character and
# short-suffix slips, which is what the model actually produces) while refusing
# genuinely different names.
_FUZZY_CUTOFF = 0.82

# The winner must beat the runner-up by this margin, otherwise the reference is
# ambiguous and we must not guess. Real case from the run: 'snails_aquariumt'
# sits between 'snails_aquarium1' and 'snails_aquarium2' -- picking either would
# be a coin flip dressed up as a repair.
_FUZZY_MARGIN = 0.05

_GIVENS_REF = re.compile(r"givens\[\s*(['\"])(.*?)\1\s*\]")


def _coerce_number(v: Any) -> Optional[float]:
    """A numeric value, or None if the value is not numerically meaningful."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", "").replace("$", "").replace("%", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _balance_subscripts(eq: str) -> Tuple[str, int]:
    """Close `givens['name'` references that never got their ']'.

    Scans for the opening form and inserts the bracket right after the closing
    quote when it is missing. Counted and returned so the caller can log it.
    """
    out, i, fixed = [], 0, 0
    pat = re.compile(r"givens\[\s*(['\"])(.*?)\1")
    while True:
        m = pat.search(eq, i)
        if not m:
            out.append(eq[i:])
            break
        end = m.end()
        rest = eq[end:]
        stripped = rest.lstrip()
        if stripped.startswith("]"):
            out.append(eq[i:end])
            i = end
        else:
            out.append(eq[i:end] + "]")
            fixed += 1
            i = end
    return "".join(out), fixed


def _resolve_name(name: str, candidates: List[str]) -> Optional[str]:
    """Unique, high-confidence match for `name` among `candidates`, else None."""
    if not candidates or name in candidates:
        return None
    scored = sorted(
        ((difflib.SequenceMatcher(None, name, c).ratio(), c) for c in candidates),
        reverse=True,
    )
    best_score, best = scored[0]
    if best_score < _FUZZY_CUTOFF:
        return None
    if len(scored) > 1 and best_score - scored[1][0] < _FUZZY_MARGIN:
        return None  # ambiguous — refuse to guess
    return best


def repair_blueprint(blueprint: dict) -> Tuple[dict, List[str]]:
    """Return (repaired_blueprint, fixes). The input is never mutated.

    `fixes` is empty when nothing mechanical was wrong, which is the caller's
    signal that the defect (if any) needs the LLM repair path instead.
    """
    givens = dict(blueprint.get("givens", {}) or {})
    equations = list(blueprint.get("equations", []) or [])
    if not equations:
        return blueprint, []

    fixes: List[str] = []

    # ── 1. Numeric values stored as strings ──────────────────────────────────
    # SIV filters givens to int/float, so {"number_of_houses": "3"} makes every
    # equation touching it unauditable even though the value is perfectly usable.
    for k, v in list(givens.items()):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            n = _coerce_number(v)
            if n is not None:
                givens[k] = n
                fixes.append(f"coerced givens[{k!r}] {v!r} -> {n}")

    numeric_keys = [k for k, v in givens.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]

    # ── 2. Unbalanced subscripts ─────────────────────────────────────────────
    for idx, eq in enumerate(equations):
        new_eq, n = _balance_subscripts(str(eq))
        if n:
            equations[idx] = new_eq
            fixes.append(f"closed {n} unbalanced givens[...] in eq{idx}")

    # ── 3 & 4. Name resolution, walking the chain in order ───────────────────
    # `computed` grows as we go so a reference can only resolve against names
    # that already exist at that point -- the same rule the evaluator uses.
    computed: List[str] = []
    for idx, eq in enumerate(equations):
        eq = str(eq)
        if "=" not in eq or eq.strip().startswith("#"):
            continue
        lhs, rhs = eq.split("=", 1)

        # 3. givens['key'] whose key is not a declared given.
        def _fix_ref(m):
            q, key = m.group(1), m.group(2)
            if key in numeric_keys:
                return m.group(0)
            # A computed variable read as if it were a given: drop the wrapper.
            if key in computed:
                fixes.append(f"eq{idx}: givens[{key!r}] -> {key} (computed above)")
                return key
            tgt = _resolve_name(key, numeric_keys)
            if tgt:
                fixes.append(f"eq{idx}: givens[{key!r}] -> givens[{tgt!r}]")
                return f"givens[{q}{tgt}{q}]"
            tgt = _resolve_name(key, computed)
            if tgt:
                fixes.append(f"eq{idx}: givens[{key!r}] -> {tgt} (computed above)")
                return tgt
            return m.group(0)

        rhs_new = _GIVENS_REF.sub(_fix_ref, rhs)

        # 4. Bare names that resolve to nothing: typo'd references to a given or
        #    to an earlier result. Protect string literals and call targets so a
        #    function name is never rewritten as a variable.
        masked = _GIVENS_REF.sub(lambda m: " " * len(m.group(0)), rhs_new)
        masked = re.sub(r"'[^']*'|\"[^\"]*\"", lambda m: " " * len(m.group(0)), masked)
        masked = re.sub(r"([A-Za-z_][A-Za-z_0-9]*)\s*\(",
                        lambda m: " " * len(m.group(0)), masked)
        known = set(numeric_keys) | set(computed) | _SAFE_NAMES
        edits = []
        for m in re.finditer(r"[A-Za-z_][A-Za-z_0-9]*", masked):
            name = m.group(0)
            if name in known:
                continue
            tgt = _resolve_name(name, computed) or _resolve_name(name, numeric_keys)
            if tgt:
                repl = tgt if tgt in computed else f"givens['{tgt}']"
                edits.append((m.start(), m.end(), repl))
                fixes.append(f"eq{idx}: undefined {name!r} -> {repl}")
        for start, end, repl in reversed(edits):
            rhs_new = rhs_new[:start] + repl + rhs_new[end:]

        equations[idx] = lhs + "=" + rhs_new
        computed.append(lhs.strip())

    if not fixes:
        return blueprint, []

    repaired = dict(blueprint)
    repaired["givens"] = givens
    repaired["equations"] = equations
    repaired["_deterministic_fixes"] = fixes
    return repaired, fixes


# Names an equation may use without declaring: what the SIV/SymPy evaluators
# inject, kept in sync with SymbolicInverseVerifier._ALLOWED_FREE_NAMES.
_SAFE_NAMES = frozenset({
    "abs", "max", "min", "ceil", "floor", "sqrt", "round", "int", "float",
    "sum", "len", "pow", "divmod", "log", "log10", "exp", "pi", "e", "E",
    "Abs", "Max", "Min", "N", "Rational", "math", "givens",
    "True", "False", "None", "and", "or", "not", "if", "else", "for", "in",
})
