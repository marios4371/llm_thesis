"""
Unit tests for evaluation_metrics.py
=====================================
Run: python -m pytest test_metrics.py -v
"""

from __future__ import annotations

import math
import unittest
import pandas as pd

from evaluation_metrics import (
    compute_all_metrics, run_mcnemar_tests,
    _intersect_problem_ids, _restrict_to_ids,
    wilson_ci, bootstrap_ci, paired_delta_bootstrap_ci,
)
import numpy as np


def _make_df(rows):
    """Build a results DataFrame with the canonical schema."""
    return pd.DataFrame(rows)


class TestIntersection(unittest.TestCase):
    def test_intersection_basic(self):
        d = {
            "a": _make_df([{"problem_id": "p1"}, {"problem_id": "p2"}, {"problem_id": "p3"}]),
            "b": _make_df([{"problem_id": "p2"}, {"problem_id": "p3"}, {"problem_id": "p4"}]),
        }
        self.assertEqual(_intersect_problem_ids(d), ["p2", "p3"])

    def test_intersection_empty_input(self):
        self.assertEqual(_intersect_problem_ids({}), [])


class TestComputeAllMetrics(unittest.TestCase):
    def setUp(self):
        # Three systems on 4 shared problems.
        # mas_sht_full: 3/4 correct
        # b1_direct:    1/4 correct
        # b2_cot:       2/4 correct
        self.results = {
            "mas_sht_full": _make_df([
                {"problem_id": "p1", "correct": 1, "time_s": 5.0, "num_llm_calls": 3, "tokens_estimated": 1000,
                 "siv_invertible": True, "siv_verified": True, "sht_triggered": False},
                {"problem_id": "p2", "correct": 1, "time_s": 6.0, "num_llm_calls": 5, "tokens_estimated": 1500,
                 "siv_invertible": True, "siv_verified": False, "sht_triggered": True},
                {"problem_id": "p3", "correct": 1, "time_s": 4.0, "num_llm_calls": 3, "tokens_estimated": 900,
                 "siv_invertible": False, "siv_verified": False, "sht_triggered": False},
                {"problem_id": "p4", "correct": 0, "time_s": 7.0, "num_llm_calls": 5, "tokens_estimated": 1600,
                 "siv_invertible": True, "siv_verified": False, "sht_triggered": True},
            ]),
            "b1_direct": _make_df([
                {"problem_id": "p1", "correct": 1, "time_s": 1.0, "num_llm_calls": 1, "tokens_estimated": 200},
                {"problem_id": "p2", "correct": 0, "time_s": 1.0, "num_llm_calls": 1, "tokens_estimated": 200},
                {"problem_id": "p3", "correct": 0, "time_s": 1.0, "num_llm_calls": 1, "tokens_estimated": 200},
                {"problem_id": "p4", "correct": 0, "time_s": 1.0, "num_llm_calls": 1, "tokens_estimated": 200},
            ]),
            "b2_cot": _make_df([
                {"problem_id": "p1", "correct": 1, "time_s": 2.0, "num_llm_calls": 1, "tokens_estimated": 400},
                {"problem_id": "p2", "correct": 1, "time_s": 2.0, "num_llm_calls": 1, "tokens_estimated": 400},
                {"problem_id": "p3", "correct": 0, "time_s": 2.0, "num_llm_calls": 1, "tokens_estimated": 400},
                {"problem_id": "p4", "correct": 0, "time_s": 2.0, "num_llm_calls": 1, "tokens_estimated": 400},
            ]),
        }

    def test_accuracy_computation(self):
        m = compute_all_metrics(self.results, reference_system="mas_sht_full")
        m = m.set_index("system")
        self.assertAlmostEqual(m.loc["mas_sht_full", "accuracy"], 0.75)
        self.assertAlmostEqual(m.loc["b1_direct", "accuracy"], 0.25)
        self.assertAlmostEqual(m.loc["b2_cot", "accuracy"], 0.5)

    def test_delta_and_error_reduction(self):
        m = compute_all_metrics(self.results, reference_system="mas_sht_full")
        m = m.set_index("system")
        # Δ: per the implementation, delta_vs_ref = system_acc - ref_acc.
        # MAS gets 0 by definition.
        self.assertAlmostEqual(m.loc["mas_sht_full", "delta_vs_ref"], 0.0)
        # b1 gets 0.25 - 0.75 = -0.5
        self.assertAlmostEqual(m.loc["b1_direct", "delta_vs_ref"], -0.5)
        # error_reduction = (acc - ref) / (1 - ref) — for ref the row is NaN.
        self.assertTrue(math.isnan(m.loc["mas_sht_full", "error_reduction_vs_ref"]))

    def test_efficiency_metrics(self):
        m = compute_all_metrics(self.results, reference_system="mas_sht_full")
        m = m.set_index("system")
        # MAS uses 3,5,3,5 → mean 4.0
        self.assertAlmostEqual(m.loc["mas_sht_full", "avg_llm_calls"], 4.0)
        # b1 uses 1 each → mean 1.0
        self.assertAlmostEqual(m.loc["b1_direct", "avg_llm_calls"], 1.0)
        # accuracy_per_call: b1 = 0.25/1 = 0.25
        self.assertAlmostEqual(m.loc["b1_direct", "accuracy_per_call"], 0.25)

    def test_siv_and_sht_rates(self):
        m = compute_all_metrics(self.results, reference_system="mas_sht_full")
        m = m.set_index("system")
        # 3 of 4 rows have siv_invertible=True
        self.assertAlmostEqual(m.loc["mas_sht_full", "siv_trigger_rate"], 0.75)
        # 2 of 4 had sht_triggered=True
        self.assertAlmostEqual(m.loc["mas_sht_full", "sht_trigger_rate"], 0.5)
        # Non-MAS systems should have NaN here.
        self.assertTrue(math.isnan(m.loc["b1_direct", "sht_trigger_rate"]))

    def test_empty_dataframe_no_crash(self):
        m = compute_all_metrics({}, reference_system="mas_sht_full")
        self.assertEqual(len(m), 0)
        # Schema preserved (incl. v10.4 CI columns).
        self.assertIn("system", m.columns)
        self.assertIn("accuracy", m.columns)
        self.assertIn("accuracy_ci_lo", m.columns)
        self.assertIn("accuracy_ci_hi", m.columns)
        self.assertIn("delta_ci_lo", m.columns)
        self.assertIn("delta_ci_hi", m.columns)

    def test_ci_columns_are_populated(self):
        m = compute_all_metrics(self.results, reference_system="mas_sht_full")
        m = m.set_index("system")
        # CI for accuracy must straddle the point estimate.
        for sys_name in ["mas_sht_full", "b1_direct", "b2_cot"]:
            row = m.loc[sys_name]
            self.assertLessEqual(row["accuracy_ci_lo"], row["accuracy"])
            self.assertGreaterEqual(row["accuracy_ci_hi"], row["accuracy"])
        # Reference row has no paired-delta CI (NaN by design).
        self.assertTrue(math.isnan(m.loc["mas_sht_full", "delta_ci_lo"]))
        # Non-ref rows have finite delta CIs that bracket the point delta.
        for sys_name in ["b1_direct", "b2_cot"]:
            row = m.loc[sys_name]
            self.assertFalse(math.isnan(row["delta_ci_lo"]))
            self.assertLessEqual(row["delta_ci_lo"], row["delta_vs_ref"] + 1e-9)
            self.assertGreaterEqual(row["delta_ci_hi"], row["delta_vs_ref"] - 1e-9)


class TestConfidenceIntervals(unittest.TestCase):
    def test_wilson_n_zero(self):
        self.assertEqual(wilson_ci(0, 0), (0.0, 0.0, 0.0))

    def test_wilson_brackets_point(self):
        # For p=0.5, n=100, 95% CI should be roughly [0.40, 0.60].
        p, lo, hi = wilson_ci(50, 100, alpha=0.05)
        self.assertAlmostEqual(p, 0.5, places=6)
        self.assertGreater(lo, 0.39)
        self.assertLess(hi, 0.61)
        self.assertLessEqual(lo, p)
        self.assertGreaterEqual(hi, p)

    def test_wilson_extremes(self):
        # All correct: lower bound should still be > 0.95 for n=100.
        _, lo, hi = wilson_ci(100, 100, alpha=0.05)
        self.assertGreater(lo, 0.95)
        self.assertAlmostEqual(hi, 1.0, places=6)
        # None correct: upper bound should still be < 0.05 for n=100.
        _, lo, hi = wilson_ci(0, 100, alpha=0.05)
        self.assertAlmostEqual(lo, 0.0, places=6)
        self.assertLess(hi, 0.05)

    def test_bootstrap_empty(self):
        p, lo, hi = bootstrap_ci(np.array([]), n_boot=100)
        self.assertTrue(math.isnan(p))

    def test_bootstrap_brackets_point(self):
        arr = np.array([1] * 50 + [0] * 50)
        p, lo, hi = bootstrap_ci(arr, n_boot=1000, seed=42)
        self.assertAlmostEqual(p, 0.5, places=6)
        self.assertLessEqual(lo, p)
        self.assertGreaterEqual(hi, p)

    def test_paired_delta_unequal_lengths(self):
        a = np.array([1, 1, 0])
        b = np.array([0, 1])  # length mismatch → NaN
        d, lo, hi = paired_delta_bootstrap_ci(a, b)
        self.assertTrue(math.isnan(d))

    def test_paired_delta_brackets_point(self):
        # a is uniformly better than b by 0.3 → CI should bracket 0.3.
        a = np.array([1] * 80 + [0] * 20)
        b = np.array([1] * 50 + [0] * 50)
        d, lo, hi = paired_delta_bootstrap_ci(a, b, n_boot=1000, seed=42)
        self.assertAlmostEqual(d, 0.3, places=6)
        self.assertLessEqual(lo, d)
        self.assertGreaterEqual(hi, d)


class TestMcNemar(unittest.TestCase):
    def test_exact_when_few_discordant(self):
        # 4 problems, b+c = 2 → exact.
        ref = [1, 1, 1, 0]
        oth = [1, 1, 0, 1]
        d = {
            "mas_sht_full": _make_df([{"problem_id": f"p{i}", "correct": v}
                                       for i, v in enumerate(ref)]),
            "b1": _make_df([{"problem_id": f"p{i}", "correct": v}
                             for i, v in enumerate(oth)]),
        }
        out = run_mcnemar_tests(d, reference_system="mas_sht_full")
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["test_used"], "exact_binomial")
        self.assertEqual(out.iloc[0]["b"], 1)  # ref correct, other wrong (p3)
        self.assertEqual(out.iloc[0]["c"], 1)  # ref wrong, other correct (p4)

    def test_asymptotic_when_many_discordant(self):
        # 60 problems, big disagreement.
        ref = [1] * 30 + [0] * 30
        oth = [0] * 30 + [1] * 30
        d = {
            "mas_sht_full": _make_df([{"problem_id": f"p{i}", "correct": v}
                                       for i, v in enumerate(ref)]),
            "b1": _make_df([{"problem_id": f"p{i}", "correct": v}
                             for i, v in enumerate(oth)]),
        }
        out = run_mcnemar_tests(d, reference_system="mas_sht_full")
        self.assertEqual(out.iloc[0]["test_used"], "asymptotic_yates")

    def test_degenerate_no_discordant(self):
        # ref and other completely agree.
        ref = oth = [1, 0, 1, 0]
        d = {
            "mas_sht_full": _make_df([{"problem_id": f"p{i}", "correct": v}
                                       for i, v in enumerate(ref)]),
            "b1": _make_df([{"problem_id": f"p{i}", "correct": v}
                             for i, v in enumerate(oth)]),
        }
        out = run_mcnemar_tests(d, reference_system="mas_sht_full")
        self.assertEqual(out.iloc[0]["test_used"], "degenerate_no_discordant")
        self.assertAlmostEqual(out.iloc[0]["p_value"], 1.0)

    def test_intersect_drops_unmatched_ids(self):
        # ref has p1-p4, other has p3-p6 — intersection is p3, p4.
        d = {
            "mas_sht_full": _make_df([
                {"problem_id": "p1", "correct": 1},
                {"problem_id": "p2", "correct": 1},
                {"problem_id": "p3", "correct": 1},
                {"problem_id": "p4", "correct": 0},
            ]),
            "b1": _make_df([
                {"problem_id": "p3", "correct": 0},
                {"problem_id": "p4", "correct": 0},
                {"problem_id": "p5", "correct": 1},
                {"problem_id": "p6", "correct": 1},
            ]),
        }
        out = run_mcnemar_tests(d, reference_system="mas_sht_full")
        self.assertEqual(out.iloc[0]["n_paired"], 2)


if __name__ == "__main__":
    unittest.main()
