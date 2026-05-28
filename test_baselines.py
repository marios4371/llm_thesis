"""
Unit tests for baselines.py
============================
Run: python -m pytest test_baselines.py -v

Uses a stub UnifiedLLMClient (no real API calls) so tests are deterministic
and CI-safe. The stub's responses are sequenced so we can verify SC vote
counting and parser fall-through paths.
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock
from typing import List

# Import the symbols under test. baselines.py imports from Mas_solver, which
# requires GROQ_API_KEY at construction time. We patch UnifiedLLMClient so
# the import side does not need a key.
import baselines
from baselines import (
    BaselineResult, direct_answer, chain_of_thought,
    self_consistency, baseline_only, BASELINE_REGISTRY,
    pal, pot,
)


class StubClient:
    """Minimal stand-in for UnifiedLLMClient. Returns scripted strings."""

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.calls = 0

    def call_model(self, messages, temperature=0.0, max_tokens=64):
        self.calls += 1
        if not self.responses:
            return ""
        return self.responses.pop(0)


class TestDirectAnswer(unittest.TestCase):
    def test_returns_baseline_result(self):
        c = StubClient(["42"])
        r = direct_answer(c, "What is 6 * 7?")
        self.assertIsInstance(r, BaselineResult)
        self.assertEqual(r.answer, 42.0)
        self.assertEqual(r.num_llm_calls, 1)
        self.assertEqual(r.error_type, "")

    def test_no_number_marks_error(self):
        c = StubClient(["I don't know"])
        r = direct_answer(c, "x?")
        self.assertIsNone(r.answer)
        self.assertEqual(r.error_type, "no_number_in_output")

    def test_api_error_marks_error(self):
        # baselines._is_error_response detects ERROR_ prefixes.
        c = StubClient(["ERROR_AUTH_401: bad key"])
        r = direct_answer(c, "x?")
        self.assertIsNone(r.answer)
        self.assertEqual(r.error_type, "api_error")


class TestChainOfThought(unittest.TestCase):
    def test_explicit_answer_tag_preferred(self):
        c = StubClient(["First, 6+1=7. Then, 7*6=42.\nAnswer: 42"])
        r = chain_of_thought(c, "?")
        self.assertEqual(r.answer, 42.0)
        self.assertEqual(r.num_llm_calls, 1)

    def test_falls_back_to_last_number(self):
        c = StubClient(["Reasoning ... result is 99"])
        r = chain_of_thought(c, "?")
        self.assertEqual(r.answer, 99.0)


class TestSelfConsistency(unittest.TestCase):
    def test_majority_vote(self):
        # 3 of 5 say 42, 2 say 41 — 42 wins.
        c = StubClient([
            "Answer: 42",
            "Answer: 42",
            "Answer: 41",
            "Answer: 42",
            "Answer: 41",
        ])
        r = self_consistency(c, "?", n=5, inter_sample_sleep=0.0)
        self.assertEqual(r.answer, 42.0)
        self.assertEqual(r.num_llm_calls, 5)
        self.assertEqual(r.meta["winner_votes"], 3)
        self.assertEqual(r.meta["vote_distribution"][42.0], 3)

    def test_tie_break_smaller_value(self):
        # Two answers tied at 2 votes — smaller wins for determinism.
        c = StubClient([
            "Answer: 10",
            "Answer: 5",
            "Answer: 10",
            "Answer: 5",
        ])
        r = self_consistency(c, "?", n=4, inter_sample_sleep=0.0)
        self.assertEqual(r.answer, 5.0)

    def test_all_samples_fail(self):
        c = StubClient(["ERROR_GENERATION: x"] * 5)
        r = self_consistency(c, "?", n=5, inter_sample_sleep=0.0)
        self.assertIsNone(r.answer)
        self.assertEqual(r.error_type, "all_samples_failed")
        self.assertEqual(r.num_llm_calls, 5)


class TestBaselineOnly(unittest.TestCase):
    def test_answer_tag_extracted(self):
        c = StubClient(["Step 1...\nANSWER: [[7]]"])
        r = baseline_only(c, "?")
        self.assertEqual(r.answer, 7.0)
        self.assertEqual(r.num_llm_calls, 1)

    def test_falls_back_to_number_extraction(self):
        c = StubClient(["So the answer is 12."])
        r = baseline_only(c, "?")
        self.assertEqual(r.answer, 12.0)


class TestRegistry(unittest.TestCase):
    def test_registry_keys_match_known_ids(self):
        # [v10.4] PAL + PoT added.
        self.assertEqual(set(BASELINE_REGISTRY.keys()),
                         {"b1_direct", "b2_cot", "b3_sc5", "b4_baseline_only",
                          "b_pal", "b_pot"})


class TestPAL(unittest.TestCase):
    """PAL must execute the generated code and prefer its output over the
    inline ANSWER tag — that's the whole point of the baseline."""

    def test_code_execution_wins(self):
        # Model emits a code block AND an ANSWER tag with a different value.
        # PAL should trust the code execution (42), not the tag (999).
        c = StubClient(["```python\nprint(6 * 7)\n```\nANSWER: [[999]]"])
        r = pal(c, "?")
        self.assertEqual(r.answer, 42.0)
        self.assertTrue(r.meta["code_succeeded"])

    def test_falls_back_to_tag_on_code_failure(self):
        c = StubClient(["```python\nundefined_var\n```\nANSWER: [[42]]"])
        r = pal(c, "?")
        self.assertEqual(r.answer, 42.0)
        self.assertFalse(r.meta["code_succeeded"])

    def test_no_code_block(self):
        c = StubClient(["The answer is 42."])
        r = pal(c, "?")
        self.assertEqual(r.answer, 42.0)
        self.assertFalse(r.meta["code_present"])

    def test_api_error(self):
        c = StubClient(["ERROR_AUTH_401: bad key"])
        r = pal(c, "?")
        self.assertIsNone(r.answer)
        self.assertEqual(r.error_type, "api_error")


class TestPoT(unittest.TestCase):
    def test_reasoning_plus_code(self):
        c = StubClient([
            "Reasoning: multiply.\n```python\nprint(6*7)\n```\nANSWER: [[42]]"
        ])
        r = pot(c, "?")
        self.assertEqual(r.answer, 42.0)


if __name__ == "__main__":
    unittest.main()
