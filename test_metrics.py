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
)


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
        # Schema preserved.
        self.assertIn("system", m.columns)
        self.assertIn("accuracy", m.columns)


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
