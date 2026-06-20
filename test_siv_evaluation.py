"""Offline unit tests for siv_evaluation.py.

Builds a small synthetic per-problem table that mirrors the columns the Kaggle
runner emits (`_row_from_mas`), with CSV-style stringified values, and checks
that every metric computes the hand-verified answer. No network / no model.
"""
import sys
import io

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console

import pandas as pd

import siv_evaluation as se


def _make_full_df() -> pd.DataFrame:
    """10 problems exercising every branch.

    Layout (full SIV+SHT system):
      p1 : empty blueprint        -> all siv_* None,  correct=True  (CoT fallback)
      p2 : empty blueprint        -> all siv_* None,  correct=False
      p3 : audited, audit PASS, verified, correct  -> confident skip (no SHT)
      p4 : audited, audit PASS, verified, correct  -> confident skip (no SHT)
      p5 : audited, audit PASS, verified, WRONG    -> translation-layer error (FP!)
      p6 : audited, audit FAIL,           WRONG    -> execution-layer error (TN, caught)
      p7 : audited, audit FAIL,           WRONG    -> execution-layer error (TN, caught)
      p8 : audited, audit PASS, not verified, correct, SHT triggered
      p9 : audited, audit None (non-invertible), WRONG -> abstained
      p10: audited, audit PASS, verified, correct  -> confident skip
    """
    rows = []

    def row(pid, correct, audit, verified, invertible, sht, calls,
            bp_ans="None", givens_total="2", siv_ran=True):
        # When SIV did not run (empty blueprint), the runner writes None to
        # every siv_* column — including siv_givens_total. Mirror that exactly.
        return {
            "problem_id": pid,
            "correct": str(correct),                       # CSV stringified bool
            "siv_execution_audit_passed": ("None" if not siv_ran else str(audit)),
            "siv_blueprint_answer": bp_ans,
            "siv_verified": ("None" if not siv_ran else str(verified)),
            "siv_invertible": ("None" if not siv_ran else str(invertible)),
            "siv_givens_total": ("None" if not siv_ran else givens_total),
            "sht_triggered": str(sht),
            "num_llm_calls": calls,
        }

    rows.append(row("p1", True,  None,  None,  None,  False, 3, siv_ran=False))
    rows.append(row("p2", False, None,  None,  None,  True,  5, siv_ran=False))
    rows.append(row("p3", True,  True,  True,  True,  False, 3, bp_ans="42", givens_total="2"))
    rows.append(row("p4", True,  True,  True,  True,  False, 3, bp_ans="10", givens_total="3"))
    rows.append(row("p5", False, True,  True,  True,  False, 3, bp_ans="7",  givens_total="2"))
    rows.append(row("p6", False, False, False, True,  True,  5, bp_ans="99", givens_total="2"))
    rows.append(row("p7", False, False, False, True,  True,  5, bp_ans="50", givens_total="3"))
    rows.append(row("p8", True,  True,  False, True,  True,  6, bp_ans="8",  givens_total="2"))
    # p9: audited (givens_total>0) but audit verdict None -> non-invertible/abstained
    r9 = row("p9", False, None, False, False, True, 5, bp_ans="None", givens_total="2")
    r9["siv_execution_audit_passed"] = "None"   # abstained, but SIV did run
    rows.append(r9)
    rows.append(row("p10", True, True,  True,  True,  False, 3, bp_ans="5", givens_total="2"))
    return pd.DataFrame(rows)


def _make_no_siv_df() -> pd.DataFrame:
    """Same 10 problems, SIV off. Accuracy slightly lower (p8 flips to wrong)."""
    base = _make_full_df()[["problem_id"]].copy()
    # full-system correctness: p1,p3,p4,p8,p10 correct (5/10)
    # no-siv: p8 now wrong -> 4/10
    correct_map = {"p1": True, "p2": False, "p3": True, "p4": True, "p5": False,
                   "p6": False, "p7": False, "p8": False, "p9": False, "p10": True}
    base["correct"] = base["problem_id"].map(correct_map).map(str)
    base["num_llm_calls"] = 4  # no skip path -> uniformly higher than full's skips
    return base


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise AssertionError(name)


def test_coverage():
    print("test_coverage")
    cov = se.siv_coverage(_make_full_df()).set_index("stage")["n"].to_dict()
    check("total = 10", cov["total_problems"] == 10)
    check("empty blueprint = 2", cov["blueprint_empty (SIV could not run)"] == 2)
    check("audited = 8", cov["audited_by_siv"] == 8)
    # invertible: p3,p4,p5,p6,p7,p8,p10 = 7 (p9 invertible=False)
    check("invertible = 7", cov["invertible_chain"] == 7)
    # verified: p3,p4,p5,p8?,p10 -> p8 verified=False -> p3,p4,p5,p10 = 4
    check("verified = 4", cov["verified (all solvable givens matched)"] == 4)


def test_detector():
    print("test_detector")
    det = se.siv_detector_metrics(_make_full_df()).set_index("verdict")
    ea = det.loc["execution_audit"]
    # audited = 8. execution_audit_passed True for p3,p4,p5,p8,p10 (5); of those
    # correct = p3,p4,p8,p10 (4) -> tp=4 ; p5 wrong -> fp=1
    check("execution_audit n_audited = 8", ea["n_audited"] == 8)
    check("execution_audit tp = 4", ea["tp"] == 4)
    check("execution_audit fp = 1 (the translation-layer FP)", ea["fp"] == 1)
    # verdict False/None for p6,p7 (audit False) and p9 (None->False): all wrong -> tn=3
    check("execution_audit tn = 3", ea["tn"] == 3)
    check("execution_audit fn = 0", ea["fn"] == 0)


def test_error_layers():
    print("test_error_layers")
    el = se.error_layer_decomposition(_make_full_df()).set_index("layer")["n"].to_dict()
    # SIV-audited wrong answers: p5 (audit True), p6,p7 (audit False), p9 (None)
    check("execution_layer = 2 (p6,p7)", el["execution_layer (SIV-detectable)"] == 2)
    check("translation_layer = 1 (p5)", el["translation_layer (SIV-blind, NL->math)"] == 1)
    check("non_invertible = 1 (p9)", el["non_invertible (SIV abstained)"] == 1)


def test_skip_efficacy():
    print("test_skip_efficacy")
    sk = se.siv_skip_efficacy(_make_full_df()).set_index("group")
    skip = sk.loc["siv_confident_skip"]
    # confident skip = verified & not sht = p3,p4,p10 (p5 verified but wrong still skips? )
    # verified True: p3,p4,p5,p10 ; sht False among them: p3,p4,p5,p10 all sht False -> 4
    check("skip n = 4", skip["n"] == 4)
    check("skip avg_calls = 3.0", abs(skip["avg_llm_calls"] - 3.0) < 1e-9)
    # accuracy of skip group: p3,p4,p10 correct, p5 wrong -> 3/4
    check("skip accuracy = 0.75", abs(skip["accuracy"] - 0.75) < 1e-9)


def test_ablation():
    print("test_ablation")
    ab = se.siv_ablation(_make_full_df(), _make_no_siv_df()).set_index("metric")
    check("full accuracy = 0.5", abs(ab.loc["accuracy", "full_siv"] - 0.5) < 1e-9)
    check("no_siv accuracy = 0.4", abs(ab.loc["accuracy", "no_siv"] - 0.4) < 1e-9)
    check("delta = +0.1", abs(ab.loc["accuracy", "delta"] - 0.1) < 1e-9)
    check("n_paired = 10", ab.loc["n_paired", "full_siv"] == 10)


def test_report_runs():
    print("test_report_runs")
    md = se.siv_report(_make_full_df(), _make_no_siv_df())
    check("report is non-empty markdown", "# SIV Evaluation Report" in md and len(md) > 200)


if __name__ == "__main__":
    test_coverage()
    test_detector()
    test_error_layers()
    test_skip_efficacy()
    test_ablation()
    test_report_runs()
    print("\nAll SIV-evaluation tests passed.")
