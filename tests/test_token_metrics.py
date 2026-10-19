"""Offline checks for token measurement; no cloud calls, audio, or robot."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import llm_utils
import motion_gen


def response(prompt=10, completion=3, cached=None, **kwargs):
    usage = {"prompt_tokens": prompt, "completion_tokens": completion}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return SimpleNamespace(response_metadata={"token_usage": usage}, **kwargs)


class TokenMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.counter = llm_utils.TokenCounter(clock=lambda: self.now)

    def test_legacy_api_retains_return_value_and_cumulative_totals(self):
        self.assertEqual(self.counter.add(response()), (10, 3))
        self.assertEqual(self.counter.add(response(5, 2)), (5, 2))
        self.assertEqual((self.counter.prompt, self.counter.completion,
                          self.counter.calls, self.counter.total), (15, 5, 2, 20))
        self.assertEqual(self.counter.summary()["total_tokens"], 20)

    def test_modern_usage_metadata_is_supported_without_response_metadata(self):
        item = SimpleNamespace(usage_metadata={
            "input_tokens": 12, "output_tokens": 4, "total_tokens": 16,
            "input_token_details": {"cache_read": 8},
        })
        self.assertEqual(self.counter.add(item), (12, 4))
        self.assertEqual(self.counter.summary()["cached_tokens"], 8)
        self.assertTrue(self.counter.summary()["usage_known"])

    def test_zero_counts_are_known_and_do_not_fall_back_to_other_metadata(self):
        item = response(0, 0, cached=0, usage_metadata={"input_tokens": 100, "output_tokens": 50})
        self.assertEqual(self.counter.add(item), (0, 0))
        summary = self.counter.summary()
        self.assertEqual(summary["total_tokens"], 0)
        self.assertEqual(summary["cached_tokens"], 0)
        self.assertTrue(summary["usage_known"])

    def test_ollama_metadata_and_explicit_zero_are_known(self):
        item = SimpleNamespace(response_metadata={"prompt_eval_count": 0, "eval_count": 7})
        self.assertEqual(self.counter.add(item), (0, 7))
        self.assertTrue(self.counter.summary()["usage_known"])
        self.assertIsNone(self.counter.summary()["cached_tokens"])

    def test_unknown_usage_is_not_reported_as_zero(self):
        self.assertEqual(self.counter.add(SimpleNamespace()), (0, 0))
        record = self.counter.records_since(0)[0]
        for field in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
            self.assertIsNone(record[field])
            self.assertIsNone(self.counter.summary()[field])
        self.assertFalse(self.counter.summary()["usage_known"])
        self.assertEqual(self.counter.summary()["missing_usage_calls"], 1)

    def test_partial_summary_exposes_known_sums_without_claiming_complete_usage(self):
        self.counter.add(response(10, 3, cached=0))
        self.counter.add(response(None, 2))
        summary = self.counter.summary()
        self.assertIsNone(summary["prompt_tokens"])
        self.assertIsNone(summary["total_tokens"])
        self.assertIsNone(summary["cached_tokens"])
        self.assertEqual(summary["completion_tokens"], 5)
        self.assertEqual(summary["known_prompt_tokens"], 10)
        self.assertEqual(summary["known_completion_tokens"], 5)
        self.assertEqual(summary["known_total_tokens"], 13)
        self.assertEqual(summary["missing_usage_calls"], 1)

    def test_reported_total_can_be_known_when_breakdown_is_missing(self):
        item = SimpleNamespace(response_metadata={"token_usage": {"total_tokens": 30}})
        self.counter.add(item)
        summary = self.counter.summary()
        self.assertEqual(summary["total_tokens"], 30)
        self.assertIsNone(summary["prompt_tokens"])
        self.assertFalse(summary["usage_known"])

    def test_cache_unknown_and_explicit_zero_stay_distinct(self):
        self.counter.add(response(cached=0))
        checkpoint = self.counter.snapshot()
        self.counter.add(response())
        records = self.counter.records_since()
        self.assertEqual(records[0]["cached_tokens"], 0)
        self.assertIsNone(records[1]["cached_tokens"])
        self.assertIsNone(self.counter.delta(checkpoint)["cached_tokens"])
        self.assertIsNone(self.counter.summary()["cached_tokens"])

    def test_snapshot_separates_turns_and_copied_records_cannot_change_counter(self):
        self.counter.add(response(100, 20), source="earlier")
        checkpoint = self.counter.snapshot()
        self.counter.add(response(10, 2), source="dialogue", elapsed_s=1.5)
        self.counter.add(response(4, 1), source="head_planner", elapsed_s=0.6)
        summary = self.counter.delta(checkpoint)
        self.assertEqual((summary["calls"], summary["prompt_tokens"],
                          summary["completion_tokens"], summary["total_tokens"]), (2, 14, 3, 17))
        records = self.counter.records_since(checkpoint)
        self.assertEqual([item["source"] for item in records], ["dialogue", "head_planner"])
        self.assertEqual(records[0]["elapsed_s"], 1.5)
        records[0]["prompt_tokens"] = 999
        self.assertEqual(self.counter.delta(checkpoint)["prompt_tokens"], 14)

    def test_rolling_window_excludes_exact_lower_boundary_and_keeps_newer_calls(self):
        self.counter.add(response(10, 1, cached=0))
        self.now = 1.0
        self.counter.add(response(20, 2, cached=5))
        self.now = 60.0
        summary = self.counter.rolling_summary()
        self.assertEqual(summary["calls"], 1)
        self.assertEqual(summary["total_tokens"], 22)
        self.assertEqual(summary["cached_tokens"], 5)
        self.assertEqual(self.counter.summary()["calls"], 2)
        self.now = 61.0
        self.assertEqual(self.counter.rolling_summary()["calls"], 0)
        self.assertEqual(self.counter.summary()["calls"], 2)

    def test_empty_summary_is_known_zero_and_rolling_window_is_configurable(self):
        self.assertEqual(self.counter.summary()["total_tokens"], 0)
        self.assertEqual(self.counter.summary()["cached_tokens"], 0)
        self.assertTrue(self.counter.summary()["usage_known"])
        self.counter.add(response())
        self.now = 5.0
        self.assertEqual(self.counter.rolling_summary(10)["calls"], 1)
        self.assertEqual(self.counter.rolling_summary(2)["calls"], 0)

    def test_malformed_counts_fall_back_to_valid_modern_metadata(self):
        for malformed in (True, -1, "12", [], {}, 1.25, float("nan"), float("inf")):
            with self.subTest(value=malformed):
                counter = llm_utils.TokenCounter(clock=lambda: 0.0)
                item = response(malformed, malformed, usage_metadata={
                    "input_tokens": 9, "output_tokens": 2,
                    "input_token_details": {"cache_read": 0},
                })
                self.assertEqual(counter.add(item), (9, 2))
                self.assertTrue(counter.summary()["usage_known"])
                self.assertEqual(counter.summary()["cached_tokens"], 0)

    def test_malformed_nested_metadata_does_not_crash(self):
        for item in (
            SimpleNamespace(response_metadata=[1, 2], usage_metadata="invalid"),
            SimpleNamespace(response_metadata={"token_usage": [1, 2]}),
            SimpleNamespace(response_metadata={"token_usage": {
                "prompt_tokens": 10, "completion_tokens": 2,
                "prompt_tokens_details": "invalid",
            }}),
        ):
            with self.subTest(response=item):
                self.counter.add(item)
        self.assertEqual(self.counter.calls, 3)
        self.assertIsNone(self.counter.summary()["total_tokens"])

    def test_invalid_elapsed_is_unknown_and_records_contain_no_conversation(self):
        for elapsed in (-1, True, "1.0", float("nan"), float("inf"), None):
            with self.subTest(elapsed=elapsed):
                self.counter.add(response(content="private speech and key"), elapsed_s=elapsed)
                self.assertIsNone(self.counter.records_since()[-1]["elapsed_s"])
        serialized = json.dumps(self.counter.records_since(), allow_nan=False)
        self.assertNotIn("private speech", serialized)
        self.assertNotIn("content", serialized)

    def test_invalid_checkpoint_and_window_are_rejected(self):
        for checkpoint in (-1, 1, True, 0.5, None):
            with self.subTest(checkpoint=checkpoint), self.assertRaises(ValueError):
                self.counter.delta(checkpoint)
        for window in (0, -1, True, "60", float("nan"), float("inf")):
            with self.subTest(window=window), self.assertRaises(ValueError):
                self.counter.rolling_summary(window)

    def test_head_planner_records_additional_call_and_invoke_duration(self):
        self.counter.add(response(20, 5), source="dialogue", elapsed_s=1.0)
        checkpoint = self.counter.snapshot()
        head_response = response(10, 3, content='{"steps":[{"joints":{"0":30},"hold":1}]}')
        model = Mock(invoke=Mock(return_value=head_response))
        with (patch.object(llm_utils, "token_counter", self.counter),
              patch.object(motion_gen.time, "perf_counter", side_effect=[3.0, 3.75])):
            steps, warnings = motion_gen.plan_from_request(model, "tengok kanan sejauh 30 derajat")
        self.assertEqual(steps, [{"joints": {0: 30}, "hold": 1.0}])
        self.assertEqual(warnings, [])
        record = self.counter.records_since(checkpoint)[0]
        self.assertEqual(record["source"], "head_planner")
        self.assertEqual(record["elapsed_s"], 0.75)
        self.assertEqual(self.counter.delta(checkpoint)["calls"], 1)
        self.assertEqual(self.counter.summary()["calls"], 2)


if __name__ == "__main__":
    unittest.main()
