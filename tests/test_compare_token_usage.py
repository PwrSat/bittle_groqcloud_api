"""Paired token comparison uses mocked models and never executes robot actions."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import action_sequence
import llm_utils
import motion_gen
from tools import compare_token_usage as comparison


COMMANDS = {"walk": "kwkF", "sit": "ksit", "stand": "kup", "hello": "khi"}


def response(actions, *, prompt_tokens=100, completion_tokens=20):
    return SimpleNamespace(
        content=json.dumps({"reply": "OK", "actions": actions}), additional_kwargs={},
        response_metadata={"token_usage": {"prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens, "prompt_tokens_details": {"cached_tokens": 0}}},
    )


def head_response(angle=30, hold=1):
    result = response([])
    result.content = json.dumps({"steps": [{"joints": {"0": angle}, "hold": hold}]})
    return result


WALK_SIT = [{"type": "skill", "command": "kwkF", "seconds": 3},
            {"type": "skill", "command": "ksit", "seconds": 2}]


class TokenComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "project"
        self.snapshot = self.root / "logs" / "measurements" / "20261019-120000-token-optimization" / "baseline.json"
        self.snapshot.parent.mkdir(parents=True)
        self.baseline_data = {"model": "openai/gpt-oss-120b", "temperature": 0,
                              "max_history_messages": 20, "commands": COMMANDS,
                              "system_prompt": "SAVED_OLD_SYSTEM_PROMPT", "motion_prompt": "SAVED_OLD_HEAD_PROMPT"}
        self.snapshot.write_text(json.dumps(self.baseline_data), encoding="utf-8")
        patcher = patch.object(comparison, "PROJECT_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.baseline = comparison.load_baseline()

    def case(self, suffix):
        return next(case for case in comparison.CASES if case["id"].endswith(suffix))

    def test_fixtures_cover_ten_languages_mixed_repetition_custom_and_full_context(self):
        self.assertEqual(len(comparison.CASES), 18)
        self.assertEqual({case["language"] for case in comparison.CASES},
                         {"en", "zh", "hi", "es", "ar", "fr", "pt", "id", "ja", "ko"})
        self.assertEqual(len(self.case("context-repeat")["history"]), 20)
        self.assertEqual(self.case("repeat-head")["expected"], [comparison.head(1), comparison.head(1)])
        self.assertEqual(self.case("custom-head")["expected"], [comparison.head(1, angle=25, hold=0.8)])
        self.assertEqual(self.case("unsupported-legs")["expected"], [])
        self.assertEqual(self.case("chat")["expected"], [])

    def test_baseline_auto_latest_and_project_boundaries_are_validated_before_api(self):
        newest = self.snapshot.parent.with_name("20261020-120000-token-optimization") / "baseline.json"
        newest.parent.mkdir()
        newest.write_text(json.dumps(self.baseline_data), encoding="utf-8")
        self.assertEqual(Path(comparison.load_baseline()["source_path"]), newest)
        outside = self.base / "outside.json"
        outside.write_text(json.dumps(self.baseline_data), encoding="utf-8")
        with self.assertRaises(ValueError):
            comparison.load_baseline(outside)
        for field, value in (("motion_prompt", ""), ("model", "different-model"), ("max_history_messages", 10)):
            invalid = {**self.baseline_data, field: value}
            newest.write_text(json.dumps(invalid), encoding="utf-8")
            with self.subTest(field=field), self.assertRaises(ValueError):
                comparison.load_baseline()

    def test_default_dry_does_not_construct_model_or_use_motion_audio_devices(self):
        with (
            patch("llm_utils.get_model", side_effect=AssertionError("no API")) as model,
            patch("robot_link.create_robot", side_effect=AssertionError("no robot")),
            patch.object(action_sequence, "execute_actions", side_effect=AssertionError("no movement")),
            patch.object(motion_gen, "plan_from_request", side_effect=AssertionError("no planner")),
            redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(comparison.main([]), 0)
        model.assert_not_called()
        self.assertIn("18 pasangan siap", output.getvalue())

    def test_both_variants_receive_identical_twenty_message_context_without_expected_leakage(self):
        case = self.case("context-repeat")
        original_history = copy.deepcopy(case["history"])
        backend = Mock()
        backend.invoke.side_effect = [response(WALK_SIT), response(WALK_SIT)]
        model = comparison.APIPacer(backend, delay=0)
        results = [comparison.evaluate_variant(model, case, self.baseline, variant,
                    counter=llm_utils.TokenCounter()) for variant in ("before", "after")]

        before, after = [call.args[0] for call in backend.invoke.call_args_list]
        self.assertEqual(len(before), 22)
        self.assertEqual([message.content for message in before[1:]], [message.content for message in after[1:]])
        self.assertEqual(before[0].content, "SAVED_OLD_SYSTEM_PROMPT")
        self.assertNotEqual(after[0].content, before[0].content)
        self.assertEqual(before[-1].content, "[bahasa: ko] " + case["text"])
        self.assertEqual(case["history"], original_history)
        self.assertEqual(results[0]["history_digest"], results[1]["history_digest"])
        self.assertTrue(all(result["correct"] for result in results))

    def test_baseline_head_uses_saved_prompt_and_real_counter_while_after_uses_current_planner(self):
        case = self.case("mixed-right")
        actions = [{"type": "skill", "command": "ksit", "seconds": 2},
                   {"type": "motion", "request": "tengok kanan"},
                   {"type": "skill", "command": "kup", "seconds": 1}]
        backend = Mock()
        backend.invoke.side_effect = [response(actions), head_response(), response(actions)]
        model = comparison.APIPacer(backend, delay=0)
        original_counter = llm_utils.token_counter
        steps = [{"joints": {0: 30}, "hold": 1}]
        with patch.object(motion_gen, "plan_from_request", return_value=(steps, [])) as current:
            before = comparison.evaluate_variant(model, case, self.baseline, "before", counter=llm_utils.TokenCounter())
            after = comparison.evaluate_variant(model, case, self.baseline, "after", counter=llm_utils.TokenCounter())

        self.assertTrue(before["correct"] and after["correct"])
        self.assertEqual(before["tokens"]["calls"], 2)
        self.assertEqual(before["tokens"]["total_tokens"], 240)
        self.assertEqual(after["tokens"]["calls"], 1)
        self.assertEqual(after["tokens"]["total_tokens"], 120)
        self.assertEqual(backend.invoke.call_args_list[1].args[0][0].content, "SAVED_OLD_HEAD_PROMPT")
        current.assert_called_once_with(model, "tengok kanan")
        self.assertIs(llm_utils.token_counter, original_counter)

    def test_custom_head_fallback_uses_current_planner_and_counts_additional_call(self):
        case = self.case("custom-head")
        backend = Mock()
        backend.invoke.side_effect = [response([{"type": "motion", "request": case["text"]}]), head_response(25, 0.8)]
        model = comparison.APIPacer(backend, delay=0)

        result = comparison.evaluate_variant(model, case, self.baseline, "after", counter=llm_utils.TokenCounter())

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["correct"])
        self.assertEqual(result["attempted_calls"], 2)
        self.assertEqual([record["source"] for record in result["token_records"]], ["conversation", "head_planner"])

    def test_semantic_scoring_checks_direction_excursion_count_duration_and_custom_parameters(self):
        repeated = [{"type": "motion", "steps": [
            {"joints": {0: 10}, "hold": 0.2}, {"joints": {0: 30}, "hold": 1},
            {"joints": {0: 0}, "hold": 0.2}, {"joints": {0: 30}, "hold": 1}]}]
        self.assertTrue(comparison.score_actions(self.case("repeat-head"), "OK", repeated))
        self.assertFalse(comparison.score_actions(self.case("repeat-head"), "OK", [
            {"type": "motion", "steps": repeated[0]["steps"][:2]}]))
        custom = [{"type": "motion", "steps": [{"joints": {0: 25}, "hold": 0.8}]}]
        self.assertTrue(comparison.score_actions(self.case("custom-head"), "OK", custom))
        for angle, hold in ((-25, 0.8), (30, 0.8), (25, 1.0)):
            self.assertFalse(comparison.score_actions(self.case("custom-head"), "OK", [
                {"type": "motion", "steps": [{"joints": {0: angle}, "hold": hold}]}]))
        self.assertFalse(comparison.score_actions(comparison.CASES[0], "OK", [{**WALK_SIT[0], "seconds": 2}, WALK_SIT[1]]))
        self.assertFalse(comparison.score_actions(self.case("chat"), "", []))
        self.assertFalse(comparison.score_actions(self.case("unsupported-legs"), "OK", WALK_SIT))

    def test_pacer_waits_before_every_additional_api_call_in_bounded_sleeps(self):
        backend = Mock()
        model = comparison.APIPacer(backend, delay=31)
        with patch.object(comparison.time, "sleep") as sleep:
            for _ in range(3):
                model.invoke([])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [30, 1, 30, 1])
        self.assertEqual(model.calls, 3)

    def test_failed_head_request_counts_unknown_usage_and_never_saves_exception_text(self):
        case = self.case("mixed-right")
        error = RuntimeError("PRIVATE_API_SECRET")
        error.status_code = 429
        error.response = SimpleNamespace(headers={"retry-after": "7.5"})
        backend = Mock()
        backend.invoke.side_effect = [response([{"type": "motion", "request": "tengok kanan"}]), error]
        model = comparison.APIPacer(backend, delay=0)
        result = comparison.evaluate_variant(model, case, self.baseline, "before", counter=llm_utils.TokenCounter())

        self.assertEqual(result["status"], "llm_error")
        self.assertEqual(result["retry_after_s"], 7.5)
        self.assertEqual(result["tokens"]["calls"], 2)
        self.assertFalse(result["tokens"]["usage_known"])
        self.assertIsNone(result["tokens"]["total_tokens"])
        self.assertEqual(result["tokens"]["known_total_tokens"], 120)
        self.assertNotIn("PRIVATE_API_SECRET", json.dumps(result))

    def test_cli_stops_after_any_api_error_and_saves_partial_unknown_usage(self):
        for status_code in (None, 401, 403, 429):
            with self.subTest(status_code=status_code):
                error = RuntimeError("PRIVATE_API_SECRET")
                if status_code:
                    error.status_code = status_code
                backend = Mock()
                backend.invoke.side_effect = error
                with (
                    patch("llm_utils.get_model", return_value=backend), patch("dotenv.load_dotenv"),
                    patch.object(comparison.time, "sleep") as sleep,
                    redirect_stdout(io.StringIO()),
                ):
                    code = comparison.main(["--run", "--delay", "0"])
                self.assertEqual(code, 1)
                backend.invoke.assert_called_once()
                sleep.assert_not_called()
                report_file = sorted(self.root.glob("logs/measurements/*-token-comparison/summary.json"))[-1]
                report = json.loads(report_file.read_text(encoding="utf-8"))
                self.assertEqual(len(report["records"]), 1)
                self.assertEqual(len(report["incomplete_or_error_pairs"]), 1)
                self.assertIsNone(report["variants"]["before"]["tokens"]["total_tokens"])
                self.assertNotIn("PRIVATE_API_SECRET", json.dumps(report))

    def test_cli_reports_semantic_regression_snapshot_and_latency_caveat_with_nonzero_exit(self):
        backend = Mock()
        backend.invoke.side_effect = [response(WALK_SIT), response([{"type": "skill", "command": "kup", "seconds": 2}])]
        with (
            patch("llm_utils.get_model", return_value=backend), patch("dotenv.load_dotenv"),
            patch.object(action_sequence, "execute_actions", side_effect=AssertionError("no execution")),
            redirect_stdout(io.StringIO()),
        ):
            status = comparison.main(["--run", "--max-cases", "1", "--delay", "0"])

        self.assertEqual(status, 1)
        report = json.loads(next(self.root.glob("logs/measurements/*-token-comparison/summary.json")).read_text(encoding="utf-8"))
        self.assertEqual(report["semantic_regressions"], ["en-order-duration"])
        self.assertEqual(report["paired_before_tokens"], 120)
        self.assertEqual(report["paired_after_tokens"], 120)
        self.assertEqual(report["after_snapshot"]["system_prompt"], action_sequence.build_system_prompt(COMMANDS))
        self.assertIn("not production latency", report["timing_caveat"])
        self.assertIn("reply fluency", report["unscored"])


if __name__ == "__main__":
    unittest.main()
