"""Baseline workflow with fake microphone/model; no network or robot."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import llm_utils
from tools import measure_baseline as baseline


def response(content, prompt=100, completion=20):
    return SimpleNamespace(content=json.dumps(content), additional_kwargs={}, response_metadata={
        "token_usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    })


class PlanMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.counter = llm_utils.TokenCounter()
        self.patcher = patch.object(llm_utils, "token_counter", self.counter)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_counts_main_and_head_calls_without_executing_the_plan(self):
        chain = Mock()
        chain.invoke.return_value = response({"reply": "Siap!", "actions": [
            {"type": "skill", "command": "ksit"},
            {"type": "motion", "request": "tengok kanan sejauh 30 derajat"},
            {"type": "skill", "command": "kup"},
        ]})
        model = Mock()
        model.invoke.return_value = response({"steps": [
            {"joints": {"0": 30}, "hold": 0.5},
        ]}, prompt=50, completion=10)
        with patch.object(baseline.action_sequence, "execute_actions") as execute:
            result = baseline.measure_plan(model, chain, "duduk, tengok kanan sejauh 30 derajat, lalu berdiri", "id",
                                           {"ksit", "kup"}, baseline.CASES[4])
        self.assertEqual(result["tokens"]["total_tokens"], 180)
        self.assertEqual(result["tokens"]["calls"], 2)
        self.assertEqual([item["source"] for item in result["token_calls"]], ["main", "head_planner"])
        self.assertTrue(result["plan_correct"])
        self.assertEqual(result["plan_status"], "ok")
        execute.assert_not_called()

    def test_failed_head_request_keeps_main_usage_but_marks_total_unknown(self):
        chain = Mock()
        chain.invoke.return_value = response({"reply": "Siap!", "actions": [
            {"type": "motion", "request": "tengok kanan sejauh 30 derajat"},
        ]})
        model = Mock()
        model.invoke.side_effect = TimeoutError("not stored")
        result = baseline.measure_plan(model, chain, "tengok kanan sejauh 30 derajat", "id", {"kup"}, baseline.CASES[2])
        self.assertEqual(result["tokens"]["calls"], 1)
        self.assertIsNone(result["tokens"]["total_tokens"])
        self.assertFalse(result["tokens"]["usage_known"])
        self.assertEqual(result["token_calls"][0]["total_tokens"], 120)
        self.assertEqual(result["failed_requests"], 1)
        self.assertEqual(result["plan_status"], "llm_error")

    def test_invalid_later_action_is_not_sent_for_head_planning(self):
        chain = Mock()
        chain.invoke.return_value = response({"reply": "Siap!", "actions": [
            {"type": "motion", "request": "tengok kanan"},
            {"type": "skill", "command": "invalid"},
        ]})
        model = Mock()
        result = baseline.measure_plan(model, chain, "test", "id", {"kup"}, baseline.CASES[2])
        self.assertEqual(result["plan_status"], "invalid")
        self.assertFalse(result["plan_correct"])
        self.assertEqual(result["tokens"]["total_tokens"], 120)
        model.invoke.assert_not_called()

    def test_checks_negation_order_direction_and_duration(self):
        self.assertFalse(baseline.plan_matches(baseline.CASES[6], [{"type": "skill", "command": "ksit"}]))
        self.assertFalse(baseline.plan_matches(baseline.CASES[4], [
            {"type": "skill", "command": "kup"}, {"type": "skill", "command": "ksit"},
        ]))
        self.assertFalse(baseline.plan_matches(baseline.CASES[2], [
            {"type": "motion", "steps": [{"joints": {0: -30}}]},
        ]))
        self.assertFalse(baseline.plan_matches(baseline.CASES[5], [
            {"type": "skill", "command": "kwkF", "seconds": 2},
            {"type": "skill", "command": "ksit"},
        ]))
        self.assertTrue(baseline.plan_matches(baseline.CASES[7], []))


class BaselineCliTests(unittest.TestCase):
    def test_with_llm_uses_real_prompt_and_head_planner_but_never_moves_robot(self):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        case = {**baseline.CASES[4], "text": "Duduk, tengok kanan sejauh 30 derajat, lalu berdiri."}
        model = FakeListChatModel(responses=[
            json.dumps({"reply": "Siap!", "actions": [
                {"type": "skill", "command": "ksit"},
                {"type": "motion", "request": "tengok kanan sejauh 30 derajat"},
                {"type": "skill", "command": "kup"},
            ]}),
            json.dumps({"steps": [{"joints": {"0": 30}, "hold": 0.5}]}),
        ])
        with (
            TemporaryDirectory() as temp,
            patch.object(baseline, "PROJECT_ROOT", Path(temp)),
            patch.object(baseline, "CASES", (case,)),
            patch.object(llm_utils, "token_counter", llm_utils.TokenCounter()),
            patch.object(baseline.speechtotext, "initialize_transcriber"),
            patch.object(baseline.speechtotext, "get_model_load_seconds", return_value=1.0),
            patch.object(baseline.speechtotext, "listen_and_transcribe", return_value=(case["text"], "id")),
            patch.object(baseline.speechtotext, "get_last_measurement", return_value={
                "status": "ok", "raw_text": case["text"], "transcribe_s": 0.5,
            }),
            patch.object(llm_utils, "get_model", return_value=model),
            patch("robot_link.create_robot") as create_robot,
            patch.object(baseline.action_sequence, "execute_actions") as execute,
            patch("dotenv.load_dotenv"),
            patch("builtins.input", return_value=""), patch("builtins.print"),
        ):
            (Path(temp) / "Commands.json").write_text(json.dumps({"sit": "ksit", "stand": "kup"}))
            baseline.main(["--condition", "tenang", "--with-llm", "--no-audio"])
            directory = next((Path(temp) / "logs" / "measurements").iterdir())
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["tokens"]["calls"], 2)
        self.assertIsNone(summary["tokens"]["total_tokens"], "fake model has no usage metadata")
        self.assertEqual(summary["plan_correct_count"], 1)
        create_robot.assert_not_called()
        execute.assert_not_called()

    def test_stt_only_saves_every_result_and_never_uses_groq_or_robot(self):
        speeches = [{"status": "ok", "raw_text": case["text"], "capture_s": 2.0,
                     "audio_s": 1.0, "transcribe_s": 0.5} for case in baseline.CASES]
        transcripts = [(case["text"], "id") for case in baseline.CASES]
        with (
            TemporaryDirectory() as temp,
            patch.object(baseline, "PROJECT_ROOT", Path(temp)),
            patch.object(baseline.speechtotext, "initialize_transcriber"),
            patch.object(baseline.speechtotext, "get_model_load_seconds", return_value=1.2),
            patch.object(baseline.speechtotext, "listen_and_transcribe", side_effect=transcripts) as listen,
            patch.object(baseline.speechtotext, "get_last_measurement", side_effect=speeches),
            patch.object(llm_utils, "get_model") as get_model,
            patch("robot_link.create_robot") as create_robot,
            patch("builtins.input", return_value=""), patch("builtins.print"),
        ):
            baseline.main(["--condition", "tenang", "--no-audio"])
            directory = next((Path(temp) / "logs" / "measurements").iterdir())
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            session = json.loads((directory / "session.json").read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in (directory / "utterances.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(summary["trials"], 8)
        self.assertEqual(summary["accuracy"]["normalized"]["exact_count"], 8)
        self.assertEqual(summary["tokens"]["total_tokens"], 0)
        self.assertEqual(session["model_load_s"], 1.2)
        self.assertFalse(session["warmup_performed"])
        self.assertTrue(rows[0]["first_trial_after_load"])
        self.assertFalse(rows[1]["first_trial_after_load"])
        self.assertTrue(all(call.kwargs == {"audio_path": None} for call in listen.call_args_list))
        get_model.assert_not_called()
        create_robot.assert_not_called()

    def test_capture_failure_is_saved_before_the_session_stops(self):
        with (
            TemporaryDirectory() as temp,
            patch.object(baseline, "PROJECT_ROOT", Path(temp)),
            patch.object(baseline.speechtotext, "initialize_transcriber"),
            patch.object(baseline.speechtotext, "get_model_load_seconds", return_value=1.0),
            patch.object(baseline.speechtotext, "listen_and_transcribe", side_effect=OSError("missing mic")),
            patch.object(baseline.speechtotext, "get_last_measurement", return_value={
                "status": "capture_error", "raw_text": None, "error_type": "OSError",
            }),
            patch("builtins.input", return_value=""), patch("builtins.print"),
        ):
            with self.assertRaises(OSError):
                baseline.main(["--condition", "tenang", "--no-audio"])
            directory = next((Path(temp) / "logs" / "measurements").iterdir())
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["trials"], 1)
        self.assertEqual(summary["speech_status_counts"], {"capture_error": 1})
        self.assertEqual(summary["accuracy"]["normalized"]["wer"], 1.0)

    def test_quitting_before_first_sample_still_writes_readable_summary(self):
        with (
            TemporaryDirectory() as temp,
            patch.object(baseline, "PROJECT_ROOT", Path(temp)),
            patch.object(baseline.speechtotext, "initialize_transcriber"),
            patch.object(baseline.speechtotext, "get_model_load_seconds", return_value=1.0),
            patch.object(baseline.speechtotext, "listen_and_transcribe") as listen,
            patch("builtins.input", return_value="q"), patch("builtins.print"),
        ):
            baseline.main(["--condition", "tenang", "--no-audio"])
            directory = next((Path(temp) / "logs" / "measurements").iterdir())
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            session = json.loads((directory / "session.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["trials"], 0)
        self.assertTrue(session["interrupted"])
        listen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
