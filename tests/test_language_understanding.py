"""Offline checks of the text benchmark; never call Groq or connect a robot."""

from contextlib import redirect_stdout
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import action_sequence
from llm_utils import TokenCounter
from tools import check_language_understanding as benchmark


COMMANDS = {"sit": "ksit", "stand up": "kup", "walk": "kwkF"}


def response_for(case, actions=None):
    if actions is None:
        actions = [{"type": "skill", "command": command,
                    "seconds": case.get("first_seconds", 2) if index == 0 else 2}
                   for index, command in enumerate(case["expected"])]
    return SimpleNamespace(
        content=json.dumps({"reply": "OK", "actions": actions}), additional_kwargs={},
        response_metadata={"token_usage": {"prompt_tokens": 100, "completion_tokens": 20}},
    )


class FixtureTests(unittest.TestCase):
    def test_thirty_cases_cover_each_language_and_each_behavior(self):
        self.assertEqual(len(benchmark.CASES), 30)
        self.assertEqual(len({case["id"] for case in benchmark.CASES}), 30)
        for language in benchmark.LANGUAGES:
            cases = [case for case in benchmark.CASES if case["language"] == language]
            self.assertEqual([case["expected"] for case in cases],
                             [["ksit"], ["kup"], ["kwkF", "ksit"]])
            self.assertEqual(cases[2]["first_seconds"], 3)

    def test_expected_plans_pass_actual_application_parser_and_validator(self):
        for case in benchmark.CASES:
            with self.subTest(case=case["id"]):
                _, actions = action_sequence.parse_reply(response_for(case).content, set(COMMANDS.values()))
                self.assertTrue(benchmark.score_actions(case, actions))

    def test_negation_and_order_not_substring_matching(self):
        sit, negation, ordered = benchmark.CASES[:3]
        self.assertFalse(benchmark.score_actions(negation,
            [{"type": "skill", "command": "ksit"}, {"type": "skill", "command": "kup"}]))
        self.assertFalse(benchmark.score_actions(ordered,
            [{"type": "skill", "command": "ksit"}, {"type": "skill", "command": "kwkF", "seconds": 3}]))
        self.assertFalse(benchmark.score_actions(ordered,
            [{"type": "skill", "command": "kwkF", "seconds": 2}, {"type": "skill", "command": "ksit"}]))
        self.assertFalse(benchmark.score_actions(sit, [{"type": "motion", "request": "sit"}]))

    def test_native_scripts_and_language_tag_are_preserved(self):
        for language in ("zh", "hi", "ar", "ja", "ko"):
            case = next(case for case in benchmark.CASES if case["language"] == language)
            self.assertTrue(any(ord(char) > 127 for char in case["text"]))
            messages = benchmark.build_messages("system", case)
            self.assertEqual(messages[-1].content, f"[bahasa: {language}] {case['text']}")


class EvaluationTests(unittest.TestCase):
    def test_all_fixtures_use_one_model_call_and_record_actual_usage(self):
        for case in benchmark.CASES:
            with self.subTest(case=case["id"]):
                model = Mock()
                model.invoke.return_value = response_for(case)
                counter = TokenCounter()
                with patch.object(action_sequence, "prepare_actions", side_effect=AssertionError("no motion planning")):
                    result = benchmark.evaluate_case(model, case, COMMANDS, counter)
                self.assertTrue(result["correct"])
                self.assertEqual(result["tokens"]["total_tokens"], 120)
                model.invoke.assert_called_once()

    def test_invalid_skill_and_wrong_negation_are_distinguished(self):
        case = benchmark.CASES[1]
        for command, status in (("ksit", "ok"), ("invented", "invalid")):
            model = Mock()
            model.invoke.return_value = response_for(case, [{"type": "skill", "command": command}])
            result = benchmark.evaluate_case(model, case, COMMANDS, TokenCounter())
            self.assertFalse(result["correct"])
            self.assertEqual(result["status"], status)

    def test_provider_error_is_sanitized_and_failed_usage_unknown(self):
        error = RuntimeError("do not save secret payload")
        error.status_code = 429
        error.response = SimpleNamespace(headers={"retry-after": "12.5"})
        model = Mock()
        model.invoke.side_effect = error
        result = benchmark.evaluate_case(model, benchmark.CASES[0], COMMANDS, TokenCounter())
        self.assertEqual(result["retry_after_s"], 12.5)
        self.assertEqual(result["http_status"], 429)
        self.assertIsNone(result["tokens"]["total_tokens"])
        self.assertNotIn("secret", json.dumps(result))

    def test_default_and_explicit_dry_do_not_construct_any_model(self):
        for arguments in ([], ["--dry", "--languages", "fr,pt,ko", "--max-cases", "4"]):
            with patch("llm_utils.get_model", side_effect=AssertionError("no API")), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(benchmark.main(arguments), 0)
                self.assertIn("tidak memanggil API", output.getvalue())

    def test_import_is_safe_without_any_nonstdlib_packages(self):
        code = (
            "import sys; from tools import check_language_understanding; "
            "assert not any(k.startswith(('langchain', 'groq', 'whisper', 'robot_link', 'Text2Speech')) for k in sys.modules)"
        )
        result = subprocess.run([sys.executable, "-B", "-S", "-c", code],
                                cwd=benchmark.PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_run_stops_on_rate_limit_and_saves_unknown_usage(self):
        error = RuntimeError("provider text must not be written")
        error.status_code = 429
        error.response = SimpleNamespace(headers={"retry-after": "30"})
        model = Mock()
        model.invoke.side_effect = error
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "Commands.json").write_text(json.dumps(COMMANDS), encoding="utf-8")
            with patch.object(benchmark, "PROJECT_ROOT", root), patch("dotenv.load_dotenv"), \
                    patch("llm_utils.get_model", return_value=model), \
                    patch.object(benchmark.time, "sleep") as sleep, redirect_stdout(io.StringIO()):
                self.assertEqual(benchmark.main(["--run", "--languages", "fr,pt"]), 0)
            report = json.loads(next(root.glob("logs/measurements/*/summary.json")).read_text(encoding="utf-8"))
        model.invoke.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(report["planned"], 6)
        self.assertEqual(report["tested"], 1)
        self.assertEqual(report["failed_requests"], 1)
        self.assertIsNone(report["tokens"]["total_tokens"])
        self.assertIsNone(report["tokens"]["prompt_tokens"])
        self.assertNotIn("provider text", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
