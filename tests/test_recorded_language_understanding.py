"""Recorded ASR benchmark must use actual speech output without answer leakage."""

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
from llm_utils import TokenCounter
from tools import check_language_understanding as benchmark
from tools.check_multilingual import REFERENCE_CASES


COMMANDS = {"sit": "ksit", "stand": "kup", "walk": "kwkF"}


def guided_row(sample, *, text=None, actual_language=None):
    reference_language, reference = REFERENCE_CASES[(sample - 1) % len(REFERENCE_CASES)]
    return {
        "sample": sample,
        "reference_language": reference_language,
        "reference": reference,
        "speech": {
            "status": "ok", "normalized_text": text or f"ACTUAL_ASR_SAMPLE_{sample}",
            "language_returned": actual_language or reference_language,
        },
    }


class RecordedSessionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "project"
        self.root.mkdir()
        (self.root / "Commands.json").write_text(json.dumps(COMMANDS), encoding="utf-8")
        patcher = patch.object(benchmark, "PROJECT_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_session(self, rows=None, *, name="20261019-120000-multilingual"):
        directory = self.root / "logs" / "measurements" / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "utterances.jsonl"
        if rows is None:
            rows = [guided_row(sample) for sample in (2, 3, 5, 6)]
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_resolve_accepts_internal_session_folder_or_jsonl_file(self):
        path = self.write_session()

        self.assertEqual(benchmark.resolve_session(path.parent), path.resolve())
        self.assertEqual(benchmark.resolve_session(str(path)), path.resolve())

    def test_latest_uses_lexically_latest_multilingual_session_with_jsonl(self):
        self.write_session(name="20261019-100000-multilingual")
        newest = self.write_session(name="20261019-120000-multilingual")
        self.write_session(name="99999999-999999-language-understanding")
        (self.root / "logs" / "measurements" / "99999999-999999-multilingual").mkdir()

        self.assertEqual(benchmark.resolve_session("latest"), newest.resolve())

    def test_latest_missing_session_raises_value_error(self):
        with self.assertRaises(ValueError):
            benchmark.resolve_session("latest")

    def test_outside_project_folder_and_file_are_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()
        path = outside / "utterances.jsonl"
        path.write_text(json.dumps(guided_row(2)) + "\n", encoding="utf-8")

        for value in (outside, path, self.root / ".." / "outside" / "utterances.jsonl"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                benchmark.resolve_session(value)

    def test_cases_keep_actual_text_language_and_score_original_intention(self):
        rows = [guided_row(3, text="No camines; ponte de pie.", actual_language="es")]
        path = self.write_session(rows)

        cases = benchmark.load_session_cases(path)

        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case["text"], "No camines; ponte de pie.")
        self.assertEqual(case["language"], "es")
        self.assertEqual(case["reference_language"], "hi")
        self.assertEqual(case["expected"], ["kwkF", "ksit"])
        self.assertEqual(case["first_seconds"], 3.0)

    def test_model_messages_contain_only_actual_tag_and_asr_never_reference_or_source(self):
        path = self.write_session([guided_row(3, text="No camines; ponte de pie.", actual_language="es")])
        case = benchmark.load_session_cases(path)[0]
        case["source_marker"] = "SECRET_EVALUATION_SOURCE_METADATA"

        messages = benchmark.build_messages("SYSTEM_PROMPT_ONLY", case)

        self.assertEqual([message.content for message in messages],
                         ["SYSTEM_PROMPT_ONLY", "[bahasa: es] No camines; ponte de pie."])
        content = "\n".join(message.content for message in messages)
        self.assertNotIn(REFERENCE_CASES[2][1], content)
        self.assertNotIn("SECRET_EVALUATION_SOURCE_METADATA", content)
        self.assertNotIn(str(path), content)
        self.assertNotIn("[bahasa: hi]", content)

    def test_malformed_empty_or_non_guided_jsonl_is_rejected(self):
        path = self.write_session()
        for text in ("", "\n", "{not json}\n", "[]\n", "null\n", '{"sample": 2, "speech": {}}\n'):
            with self.subTest(text=text):
                path.write_text(text, encoding="utf-8")

                with self.assertRaises(ValueError):
                    benchmark.load_session_cases(path)

    def test_invalid_fields_never_fall_back_to_guided_answer_or_language(self):
        original = guided_row(3)
        invalid_rows = []
        for field, invalid in (
            ("status", "empty"), ("status", "decode_error"),
            ("normalized_text", ""), ("normalized_text", "   "),
            ("normalized_text", None), ("normalized_text", 123),
            ("language_returned", None), ("language_returned", ""),
            ("language_returned", "it"),
        ):
            row = copy.deepcopy(original)
            row["speech"][field] = invalid
            invalid_rows.append(row)
        for field, invalid in (
            ("sample", 0), ("sample", -1), ("sample", True), ("sample", 1.5), ("sample", "3"),
            ("reference", "Different guided answer."), ("reference_language", "ms"),
            ("speech", None),
        ):
            row = copy.deepcopy(original)
            row[field] = invalid
            invalid_rows.append(row)
        for row in invalid_rows:
            with self.subTest(row=row):
                path = self.write_session([row])

                with self.assertRaises(ValueError):
                    benchmark.load_session_cases(path)

    def test_known_reference_cannot_be_misattributed_to_another_reference_language(self):
        row = guided_row(3)
        row["reference_language"] = "fr"
        path = self.write_session([row])

        with self.assertRaises(ValueError):
            benchmark.load_session_cases(path)

    def test_duplicate_sample_and_more_than_thirty_rows_are_rejected(self):
        for rows in ([guided_row(2), guided_row(2)], [guided_row(sample) for sample in range(1, 32)]):
            with self.subTest(count=len(rows)):
                path = self.write_session(rows)

                with self.assertRaises(ValueError):
                    benchmark.load_session_cases(path)

    def test_default_session_dry_run_filters_reference_language_without_any_api_or_motion(self):
        self.write_session([guided_row(sample, actual_language="es") for sample in (1, 2, 3, 5, 6, 10)])

        with (
            patch("llm_utils.get_model", side_effect=AssertionError("no model in dry mode")) as get_model,
            patch.object(action_sequence, "prepare_actions", side_effect=AssertionError("no head planner")) as prepare,
            patch.object(action_sequence, "execute_actions", side_effect=AssertionError("no robot")) as execute,
            redirect_stdout(io.StringIO()) as output,
        ):
            status = benchmark.main(["--session", "latest", "--languages", "zh,hi,ar,fr"])

        self.assertEqual(status, 0)
        self.assertIn("4", output.getvalue())
        self.assertEqual(output.getvalue().count("ACTUAL_ASR_SAMPLE_"), 4)
        self.assertNotIn("ACTUAL_ASR_SAMPLE_1 ", output.getvalue())
        get_model.assert_not_called()
        prepare.assert_not_called()
        execute.assert_not_called()

    def test_invalid_session_run_is_rejected_before_constructing_a_model(self):
        invalid = guided_row(2)
        invalid["speech"]["normalized_text"] = ""
        path = self.write_session([invalid])

        with (
            patch("llm_utils.get_model", side_effect=AssertionError("invalid data must not call API")) as get_model,
            redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
            self.assertRaises((ValueError, SystemExit)),
        ):
            benchmark.main(["--run", "--session", str(path)])

        get_model.assert_not_called()

    def test_recorded_intention_scoring_rejects_wrong_walk_duration_order_and_head_motion(self):
        case = benchmark.load_session_cases(self.write_session())[0]
        walk = {"type": "skill", "command": "kwkF", "seconds": 3}
        sit = {"type": "skill", "command": "ksit", "seconds": 2}
        self.assertTrue(benchmark.score_actions(case, [walk, sit]))
        for actions in (
            [{**walk, "seconds": 2}, sit], [sit, walk],
            [{"type": "skill", "command": "kup", "seconds": 3}, sit],
            [walk, {"type": "motion", "request": "look right"}, sit],
            [{"type": "motion", "request": "walk"}],
        ):
            with self.subTest(actions=actions):
                self.assertFalse(benchmark.score_actions(case, actions))

    def test_evaluation_uses_actual_transcript_and_does_not_invoke_a_head_planner(self):
        case = benchmark.load_session_cases(self.write_session([
            guided_row(3, text="No camines; ponte de pie.", actual_language="es"),
        ]))[0]
        model = Mock()
        model.invoke.return_value = SimpleNamespace(
            content=json.dumps({"reply": "Vale", "actions": [{"type": "motion", "request": "look right"}]}),
            additional_kwargs={}, response_metadata={"token_usage": {"prompt_tokens": 100, "completion_tokens": 20}},
        )

        with (
            patch.object(action_sequence, "prepare_actions", side_effect=AssertionError("no head planner")) as prepare,
            patch.object(action_sequence, "execute_actions", side_effect=AssertionError("no robot")) as execute,
        ):
            result = benchmark.evaluate_case(model, case, COMMANDS, TokenCounter())

        self.assertFalse(result["correct"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(model.invoke.call_args.args[0][-1].content, "[bahasa: es] No camines; ponte de pie.")
        model.invoke.assert_called_once()
        prepare.assert_not_called()
        execute.assert_not_called()

    def test_recorded_session_stops_after_429_and_saves_unknown_usage(self):
        path = self.write_session()
        error = RuntimeError("PRIVATE_PROVIDER_ERROR")
        error.status_code = 429
        error.response = SimpleNamespace(headers={"retry-after": "15"})
        model = Mock()
        model.invoke.side_effect = error

        with (
            patch("llm_utils.get_model", return_value=model), patch("dotenv.load_dotenv"),
            patch.object(benchmark.time, "sleep") as sleep,
            redirect_stdout(io.StringIO()),
        ):
            status = benchmark.main(["--run", "--session", str(path), "--languages", "zh,hi,ar,fr"])

        self.assertEqual(status, 0)
        report_path = next(self.root.glob("logs/measurements/*language-understanding/summary.json"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["planned"], 4)
        self.assertEqual(report["tested"], 1)
        self.assertEqual(report["failed_requests"], 1)
        self.assertIsNone(report["tokens"]["total_tokens"])
        self.assertFalse(report["tokens"]["usage_known"])
        self.assertNotIn("PRIVATE_PROVIDER_ERROR", json.dumps(report))
        model.invoke.assert_called_once()
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
