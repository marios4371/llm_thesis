"""
SIV Evaluation — Quantitative analysis of Symbolic Inverse Verification
========================================================================
Companion to siv_module.py, Mas_solver.py and evaluation_metrics.py.

The paper (paper_mas_sht.tex, Future Work) explicitly defers the *empirical*
evaluation of SIV: "SIV metrics appear as null in the experiment logs ...
Quantitative evaluation of SIV's two-layer impact ... is reserved for future
experiments." This module fills exactly that gap.

It consumes the per-problem CSV(s) the Kaggle/Colab runner already emits — one
row per problem, with the SIV columns produced by `_row_from_mas`:

    problem_id, correct (bool),
    siv_execution_audit_passed (bool|None),   # Layer 1 verdict
    siv_verified (bool|None),                  # Layer 2: all solvable givens matched
    siv_confidence (float|None),
    siv_givens_matched (int|None), siv_givens_total (int|None),
    siv_invertible (bool|None),                # chain was symbolically invertible
    siv_failed_givens (str repr of list), siv_unused_givens (str repr of list),
    sht_triggered (bool), num_llm_calls (int)

`None`/NaN on every siv_* column means SIV did not run for that problem — almost
always because the Mathematician produced an *empty blueprint* (no equations /
no numeric givens). That regime is the single most important thing to surface,
so it is reported first as a coverage funnel.

Five public functions, each returning a tidy DataFrame (so they drop straight
into a thesis table), plus a `siv_report` that bundles them into markdown:

    siv_coverage(df)               -> the funnel: empty blueprint -> SIV ran ->
                                      invertible -> verified.
    siv_detector_metrics(df)       -> SIV's execution audit (and Layer-2 verdict)
                                      as a binary detector of answer correctness:
                                      confusion matrix + precision/recall/specificity.
    error_layer_decomposition(df)  -> among WRONG answers that SIV could audit,
                                      split execution-layer (SIV-detectable) vs
                                      translation-layer (NL->math, SIV-blind).
                                      This is the empirical version of the paper's
                                      central scope-limitation claim.
    siv_skip_efficacy(df)          -> does the confident-skip path save API calls
                                      without losing accuracy? (full system only)
    siv_ablation(full_df, no_siv_df) -> accuracy + cost delta of turning SIV on,
                                      with a paired McNemar test.

Design notes:
    * Robust to CSV round-tripping: booleans may come back as the strings
      "True"/"False", ints as floats, lists as "[...]" strings. `_to_bool` and
      `_parse_list` handle all of it.
    * Never raises on a missing column — returns an empty/0 row with a `note`
      so a half-finished run still produces a (clearly-flagged) table.
    * Pure pandas/numpy + (optional) the project's evaluation_metrics for the
      McNemar test. No new dependencies.
"""

from __future__ import annotations

import ast
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("MAS_Pipeline.siv_eval")

# Reuse the project's McNemar implementation for the ablation test.
try:
    from evaluation_metrics import run_mcnemar_tests
    _METRICS_AVAILABLE = True
except Exception:  # pragma: no cover - only when imported standalone
    _METRICS_AVAILABLE = False


# =====================================================================
# Coercion helpers (CSV round-tripping is lossy)
# =====================================================================

def _to_bool(s: pd.Series) -> pd.Series:
    """Coerce a column to nullable boolean, tolerating CSV string forms.

    Returns a pandas 'boolean' (nullable) Series so that "SIV did not run"
    (NaN/None) stays distinguishable from an explicit False.
    """
    def conv(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return pd.NA
        if isinstance(v, (int, float)):
            if isinstance(v, float) and np.isnan(v):
                return pd.NA
            return bool(v)
        t = str(v).strip().lower()
        if t in ("true", "1", "1.0", "yes"):
            return True
        if t in ("false", "0", "0.0", "no"):
            return False
        if t in ("", "nan", "none", "<na>", "null"):
            return pd.NA
        return pd.NA
    return s.map(conv).astype("boolean")


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _parse_list(v) -> list:
    """Parse a list that may have been stringified by CSV ('[a, b]')."""
    if isinstance(v, (list, tuple)):
        return list(v)
    if v is None:
        return []
    t = str(v).strip()
    if t in ("", "[]", "nan", "none", "<NA>"):
        return []
    try:
        out = ast.literal_eval(t)
        return list(out) if isinstance(out, (list, tuple)) else [out]
    except Exception:
        return []


def _siv_ran_mask(df: pd.DataFrame) -> pd.Series:
    """A problem is 'audited by SIV' iff SIV actually executed for it.

    SIV runs only when the blueprint has equations (see Mas_solver: enable_siv
    AND answer is not None AND blueprint['equations']). In that case the runner
    writes concrete siv_* values; otherwise every siv_* is None. We therefore
    OR three independent signals, because a *non-invertible* chain leaves the
    Layer-1 audit verdict null even though SIV did run:

        siv_invertible non-null  (set to a bool whenever SIV produced a result)
        siv_execution_audit_passed non-null
        siv_givens_total > 0
    """
    ran = pd.Series(False, index=df.index)
    if "siv_invertible" in df.columns:
        ran = ran | _to_bool(df["siv_invertible"]).notna()
    if "siv_execution_audit_passed" in df.columns:
        ran = ran | _to_bool(df["siv_execution_audit_passed"]).notna()
    if "siv_givens_total" in df.columns:
        ran = ran | (_to_num(df["siv_givens_total"]).fillna(0) > 0)
    return ran.astype(bool)


# =====================================================================
# 1) Coverage funnel
# =====================================================================

def siv_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """The SIV input funnel — the prerequisite for everything else.

    SIV can only verify problems whose blueprint produced equations + numeric
    givens. If the Mathematician returns an empty blueprint (the dominant
    failure mode on hard benchmarks, per the paper), SIV never runs. This funnel
    makes the available-evidence base explicit before any rate is reported.

    Columns: stage, n, pct_of_total, pct_of_audited.
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame([{"stage": "total", "n": 0,
                              "pct_of_total": float("nan"),
                              "pct_of_audited": float("nan"),
                              "note": "empty dataframe"}])

    ran = _siv_ran_mask(df)
    n_audited = int(ran.sum())

    invertible = _to_bool(df.get("siv_invertible", pd.Series(index=df.index)))
    n_invertible = int((invertible.fillna(False) & ran).sum())

    verified = _to_bool(df.get("siv_verified", pd.Series(index=df.index)))
    n_verified = int((verified.fillna(False) & ran).sum())

    def row(stage, k):
        return {
            "stage": stage,
            "n": k,
            "pct_of_total": k / n if n else float("nan"),
            "pct_of_audited": (k / n_audited) if n_audited else float("nan"),
        }

    return pd.DataFrame([
        row("total_problems", n),
        row("blueprint_empty (SIV could not run)", n - n_audited),
        row("audited_by_siv", n_audited),
        row("invertible_chain", n_invertible),
        row("verified (all solvable givens matched)", n_verified),
    ])


# =====================================================================
# 2) SIV as a correctness detector
# =====================================================================

def siv_detector_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Treat each SIV verdict as a binary classifier of answer correctness.

    Two verdicts are scored, over the SIV-audited subset only:

        execution_audit (Layer 1): predicts 'execution faithful to blueprint'.
            Positive verdict = execution_audit_passed == True.
        verified (Layer 2): predicts 'all solvable givens reconstruct'.

    For each we cross-tabulate verdict vs ground-truth `correct` and report
    precision/recall/specificity/FPR. The headline number for the thesis is
    `false_positive_rate`: how often SIV says 'execution-consistent' on an
    answer that is actually wrong. Because SIV is a math->math verifier, those
    false positives are exactly the translation-layer errors it cannot see —
    this quantifies the scope limitation rather than just asserting it.

    Columns: verdict, n_audited, tp, fp, tn, fn, precision, recall,
             specificity, false_positive_rate, accuracy_of_verdict.
    """
    ran = _siv_ran_mask(df)
    sub = df[ran].copy()
    out_rows = []

    if "correct" not in sub.columns or len(sub) == 0:
        return pd.DataFrame([{"verdict": v, "n_audited": len(sub), "tp": 0,
                              "fp": 0, "tn": 0, "fn": 0,
                              "precision": float("nan"), "recall": float("nan"),
                              "specificity": float("nan"),
                              "false_positive_rate": float("nan"),
                              "accuracy_of_verdict": float("nan"),
                              "note": "no audited rows / no 'correct' column"}
                             for v in ("execution_audit", "verified")])

    correct = _to_bool(sub["correct"]).fillna(False).to_numpy()

    verdict_cols = {
        "execution_audit": "siv_execution_audit_passed",
        "verified": "siv_verified",
    }
    for name, col in verdict_cols.items():
        if col not in sub.columns:
            continue
        verdict = _to_bool(sub[col]).fillna(False).to_numpy()
        tp = int(np.sum(verdict & correct))    # said ok, was correct
        fp = int(np.sum(verdict & ~correct))   # said ok, was WRONG  <- the key cell
        tn = int(np.sum(~verdict & ~correct))  # said suspect, was wrong
        fn = int(np.sum(~verdict & correct))   # said suspect, was correct
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) else float("nan")
        acc = (tp + tn) / len(sub) if len(sub) else float("nan")
        out_rows.append({
            "verdict": name,
            "n_audited": len(sub),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": prec,
            "recall": rec,
            "specificity": spec,
            "false_positive_rate": fpr,
            "accuracy_of_verdict": acc,
        })
    return pd.DataFrame(out_rows)


# =====================================================================
# 3) Error-layer decomposition (the central scope claim, empirically)
# =====================================================================

def error_layer_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Split wrong answers into the two layers SIV distinguishes.

    Restricted to problems where (a) SIV ran and (b) the final answer was wrong:

        execution_layer  : siv_execution_audit_passed == False.
            The Programmer's answer deviates from the blueprint's forward
            evaluation — SIV *catches* this. (SIV-detectable error.)
        translation_layer: siv_execution_audit_passed == True but answer wrong.
            The blueprint was executed faithfully but models the problem wrongly
            (NL->math error). SIV is blind here by construction (FOBAR's domain).
        non_invertible   : audit could not be computed (blueprint_answer is None
            / not invertible) — SIV abstained.

    This is the table that turns the paper's qualitative "SIV cannot detect
    NL->math errors" into a measured fraction, and motivates composing SIV with
    a FOBAR-style translation-layer check.

    Columns: layer, n, pct_of_siv_errors.
    """
    ran = _siv_ran_mask(df)
    if "correct" not in df.columns:
        return pd.DataFrame([{"layer": "—", "n": 0, "pct_of_siv_errors": float("nan"),
                              "note": "no 'correct' column"}])
    correct = _to_bool(df["correct"]).fillna(False)
    wrong = ran & ~correct
    sub = df[wrong].copy()
    n_err = len(sub)
    if n_err == 0:
        return pd.DataFrame([{"layer": "no_siv_audited_errors", "n": 0,
                              "pct_of_siv_errors": float("nan")}])

    audit = _to_bool(sub.get("siv_execution_audit_passed",
                             pd.Series(index=sub.index)))
    bp_ans = _to_num(sub.get("siv_blueprint_answer",
                             pd.Series(index=sub.index))) \
        if "siv_blueprint_answer" in sub.columns else pd.Series(np.nan, index=sub.index)

    # non-invertible / abstained: audit verdict is NA OR no blueprint_answer
    abstained = audit.isna()
    execution = (audit == False) & ~abstained        # noqa: E712 (nullable bool)
    translation = (audit == True) & ~abstained        # noqa: E712

    rows = [
        ("execution_layer (SIV-detectable)", int(execution.sum())),
        ("translation_layer (SIV-blind, NL->math)", int(translation.sum())),
        ("non_invertible (SIV abstained)", int(abstained.sum())),
    ]
    return pd.DataFrame([
        {"layer": name, "n": k, "pct_of_siv_errors": k / n_err if n_err else float("nan")}
        for name, k in rows
    ])


# =====================================================================
# 4) Confident-skip efficacy (cost without accuracy loss?)
# =====================================================================

def siv_skip_efficacy(df: pd.DataFrame) -> pd.DataFrame:
    """Does the SIV confident-skip path save calls while preserving accuracy?

    Splits the SIV-audited problems by whether SIV skipped SHT
    (siv_verified == True AND sht_triggered == False) vs the rest, and compares
    accuracy and average LLM calls. The thesis claim is that the skip group
    keeps high accuracy at strictly lower cost (≈3 calls vs ≈4.46, per the
    paper's Table on API budget).

    Columns: group, n, accuracy, avg_llm_calls.
    """
    ran = _siv_ran_mask(df)
    sub = df[ran].copy()
    if len(sub) == 0:
        return pd.DataFrame([{"group": "—", "n": 0, "accuracy": float("nan"),
                              "avg_llm_calls": float("nan"),
                              "note": "no SIV-audited rows"}])

    verified = _to_bool(sub.get("siv_verified", pd.Series(index=sub.index))).fillna(False)
    sht = _to_bool(sub.get("sht_triggered", pd.Series(index=sub.index))).fillna(False)
    skip = verified & ~sht

    correct = _to_bool(sub["correct"]).fillna(False).astype(int) \
        if "correct" in sub.columns else pd.Series(np.nan, index=sub.index)
    calls = _to_num(sub.get("num_llm_calls", pd.Series(index=sub.index)))

    def stat(mask, label):
        m = sub[mask]
        return {
            "group": label,
            "n": int(mask.sum()),
            "accuracy": float(correct[mask].mean()) if mask.any() else float("nan"),
            "avg_llm_calls": float(calls[mask].mean()) if mask.any() else float("nan"),
        }

    return pd.DataFrame([
        stat(skip, "siv_confident_skip"),
        stat(~skip, "went_to_sht"),
    ])


# =====================================================================
# 5) Ablation: SIV on vs off
# =====================================================================

def siv_ablation(full_df: pd.DataFrame,
                 no_siv_df: pd.DataFrame,
                 alpha: float = 0.05) -> pd.DataFrame:
    """Compare the full system against the enable_siv=False ablation.

    Both frames must carry problem_id + correct (+ num_llm_calls for the cost
    delta). Accuracy and cost are computed on the paired intersection of
    problem_ids; the McNemar p-value (from evaluation_metrics) tests whether the
    accuracy difference is significant.

    Columns: metric, full_siv, no_siv, delta, p_value (accuracy row only).
    """
    if "problem_id" not in full_df.columns or "problem_id" not in no_siv_df.columns:
        return pd.DataFrame([{"metric": "—", "full_siv": float("nan"),
                              "no_siv": float("nan"), "delta": float("nan"),
                              "p_value": float("nan"),
                              "note": "missing problem_id column"}])

    common = sorted(set(full_df["problem_id"].astype(str)) &
                    set(no_siv_df["problem_id"].astype(str)))
    if not common:
        return pd.DataFrame([{"metric": "—", "full_siv": float("nan"),
                              "no_siv": float("nan"), "delta": float("nan"),
                              "p_value": float("nan"),
                              "note": "no shared problem_ids"}])

    def restrict(d):
        d = d[d["problem_id"].astype(str).isin(common)].copy()
        return d.drop_duplicates("problem_id", keep="last") \
                .set_index(d["problem_id"].astype(str)).loc[common]

    f, g = restrict(full_df), restrict(no_siv_df)

    f_acc = float(_to_bool(f["correct"]).fillna(False).mean())
    g_acc = float(_to_bool(g["correct"]).fillna(False).mean())

    f_calls = float(_to_num(f.get("num_llm_calls", pd.Series(index=f.index))).mean())
    g_calls = float(_to_num(g.get("num_llm_calls", pd.Series(index=g.index))).mean())

    # McNemar p-value on the accuracy difference (paired).
    p_val = float("nan")
    if _METRICS_AVAILABLE:
        try:
            mc = run_mcnemar_tests(
                {"mas_sht_full": f.reset_index(drop=True),
                 "mas_no_siv": g.reset_index(drop=True)},
                reference_system="mas_sht_full", alpha=alpha,
            )
            if len(mc):
                p_val = float(mc.iloc[0]["p_value"])
        except Exception as e:
            logger.warning(f"siv_ablation McNemar failed: {e}")

    return pd.DataFrame([
        {"metric": "accuracy", "full_siv": f_acc, "no_siv": g_acc,
         "delta": f_acc - g_acc, "p_value": p_val},
        {"metric": "avg_llm_calls", "full_siv": f_calls, "no_siv": g_calls,
         "delta": f_calls - g_calls, "p_value": float("nan")},
        {"metric": "n_paired", "full_siv": len(common), "no_siv": len(common),
         "delta": 0, "p_value": float("nan")},
    ])


# =====================================================================
# Report bundle
# =====================================================================

def siv_report(full_df: pd.DataFrame,
               no_siv_df: Optional[pd.DataFrame] = None) -> str:
    """Assemble the full SIV evaluation as a markdown string for the thesis."""
    parts: List[str] = ["# SIV Evaluation Report\n"]

    def section(title, frame):
        parts.append(f"\n## {title}\n")
        try:
            parts.append(frame.to_markdown(index=False, floatfmt=".4f"))
        except Exception:
            parts.append(frame.to_string(index=False))
        parts.append("")

    section("1. Coverage funnel (SIV input availability)", siv_coverage(full_df))
    section("2. SIV as a correctness detector", siv_detector_metrics(full_df))
    section("3. Error-layer decomposition (execution vs translation)",
            error_layer_decomposition(full_df))
    section("4. Confident-skip efficacy", siv_skip_efficacy(full_df))
    if no_siv_df is not None:
        section("5. Ablation — SIV on vs off", siv_ablation(full_df, no_siv_df))
    else:
        parts.append("\n## 5. Ablation — SIV on vs off\n")
        parts.append("_No `enable_siv=False` run supplied. Add a `mas_no_siv` "
                     "variant to the runner to populate this section._\n")

    return "\n".join(parts)
