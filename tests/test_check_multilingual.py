"""Manual multilingual checker control flow without a microphone or model."""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools import check_multilingual as checker


class MultilingualCheckerTests(unittest.TestCase):
    def test_guided_language_subset_deduplicates_and_preserves_canonical_order(self):
        references = [case for case in checker.REFERENCE_CASES if case[0] in {"zh", "hi", "ar", "fr"}]
        original_references = checker.REFERENCE_CASES
        with (
            TemporaryDirectory() as temp,
            patch.object(checker, "PROJECT_ROOT", Path(temp)),
            patch.object(checker.speechtotext, "initialize_transcriber") as initialize,
            patch.object(checker.speechtotext, "listen_and_transcribe",
                         return_value=("Actual speech, not a reference.", "en")) as listen,
            patch.object(checker.speechtotext, "get_last_measurement", return_value={"status": "ok"}),
            patch("builtins.input", return_value=""), patch("builtins.print"),
        ):
            checker.main(["--guided", "--languages", " fr, zh,fr, hi, ar ", "--no-audio"])
            report = next((Path(temp) / "logs" / "measurements").glob("*/utterances.jsonl"))
            rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 4)
        self.assertEqual([(row["reference_language"], row["reference"]) for row in rows], references)
        self.assertEqual([row["language_matches_reference"] for row in rows], [False] * 4)
        self.assertEqual(checker.REFERENCE_CASES, original_references)
        initialize.assert_called_once_with()
        self.assertEqual(listen.call_count, 4)
        self.assertTrue(all(call.kwargs == {"audio_path": None} and not call.args for call in listen.call_args_list))

    def test_explicit_samples_cycle_only_the_filtered_guided_references(self):
        selected = [case for case in checker.REFERENCE_CASES if case[0] in {"zh", "hi", "ar", "fr"}]
        references = selected * 2
        with (
            TemporaryDirectory() as temp,
            patch.object(checker, "PROJECT_ROOT", Path(temp)),
            patch.object(checker.speechtotext, "initialize_transcriber"),
            patch.object(checker.speechtotext, "listen_and_transcribe",
                         side_effect=[(text, language) for language, text in references]) as listen,
            patch.object(checker.speechtotext, "get_last_measurement", return_value={"status": "ok"}),
            patch("builtins.input", return_value=""), patch("builtins.print"),
        ):
            checker.main(["--guided", "--languages", "zh,hi,ar,fr", "--samples", "8"])
            report = next((Path(temp) / "logs" / "measurements").glob("*/utterances.jsonl"))
            rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([(row["reference_language"], row["reference"]) for row in rows], references)
        self.assertEqual([row["sample"] for row in rows], list(range(1, 9)))
        self.assertEqual(listen.call_count, 8)
        for number, invocation in enumerate(listen.call_args_list, 1):
            self.assertEqual(invocation.args, ())
            self.assertEqual(set(invocation.kwargs), {"audio_path"})
            self.assertEqual(invocation.kwargs["audio_path"], report.parent / f"{number:02}.wav")

    def test_language_filter_requires_guided_and_rejects_unknown_or_empty_codes_before_initializing(self):
        for arguments in (
            ["--languages", "zh,hi,ar,fr"],
            ["--guided", "--languages", "it"],
            ["--guided", "--languages", "zh,it"],
            ["--guided", "--languages", ""],
            ["--guided", "--languages", "zh,,hi"],
            ["--guided", "--languages", "zh, "],
            ["--guided", "--languages", "ZH"],
        ):
            with (
                self.subTest(arguments=arguments),
                patch.object(checker.speechtotext, "initialize_transcriber") as initialize,
                patch.object(checker.speechtotext, "listen_and_transcribe") as listen,
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as caught:
                    checker.main(arguments)

                self.assertEqual(caught.exception.code, 2)
                initialize.assert_not_called()
                listen.assert_not_called()

    def test_saves_automatic_language_switches_without_selecting_a_language(self):
        utterances = [("duduk", "id"), ("Hello", "en"), ("座って", "ja")]
        measurements = [{"language_returned": language, "normalized_text": text}
                        for text, language in utterances]
        with (
            TemporaryDirectory() as temp,
            patch.object(checker, "PROJECT_ROOT", Path(temp)),
            patch.object(checker.speechtotext, "initialize_transcriber"),
            patch.object(checker.speechtotext, "listen_and_transcribe", side_effect=utterances) as listen,
            patch.object(checker.speechtotext, "get_last_measurement", side_effect=measurements),
            patch("robot_link.create_robot") as robot,
            patch("builtins.input", return_value=""), patch("builtins.print"),
        ):
            checker.main(["--samples", "3", "--no-audio"])
            output = next((Path(temp) / "logs" / "measurements").glob("*/utterances.jsonl"))
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["speech"]["language_returned"] for row in rows], ["id", "en", "ja"])
        self.assertTrue(all("reference" not in row and "language_matches_reference" not in row for row in rows))
        self.assertTrue(all(call.kwargs == {"audio_path": None} for call in listen.call_args_list))
        robot.assert_not_called()

    def test_capture_error_is_saved_before_propagating(self):
        with (
            TemporaryDirectory() as temp,
            patch.object(checker, "PROJECT_ROOT", Path(temp)),
            patch.object(checker.speechtotext, "initialize_transcriber"),
            patch.object(checker.speechtotext, "listen_and_transcribe", side_effect=OSError("missing microphone")),
            patch.object(checker.speechtotext, "get_last_measurement", return_value={"status": "capture_error"}),
            patch("builtins.input", return_value=""), patch("builtins.print"),
        ):
            with self.assertRaises(OSError):
                checker.main(["--samples", "1", "--no-audio"])
            output = next((Path(temp) / "logs" / "measurements").glob("*/utterances.jsonl"))
            row = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(row["speech"]["status"], "capture_error")

    def test_guided_defaults_to_ten_and_never_passes_reference_to_recognizer(self):
        with (
            TemporaryDirectory() as temp,
            patch.object(checker, "PROJECT_ROOT", Path(temp)),
            patch.object(checker.speechtotext, "initialize_transcriber"),
            patch.object(checker.speechtotext, "listen_and_transcribe",
                         side_effect=[(text, language) for language, text in checker.REFERENCE_CASES]) as listen,
            patch.object(checker.speechtotext, "get_last_measurement", return_value={"status": "ok"}),
            patch("builtins.input", return_value=""), patch("builtins.print") as output,
        ):
            checker.main(["--guided", "--no-audio"])
            report = next((Path(temp) / "logs" / "measurements").glob("*/utterances.jsonl"))
            rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 10)
        self.assertEqual([(row["reference_language"], row["reference"]) for row in rows],
                         list(checker.REFERENCE_CASES))
        self.assertTrue(all(row["language_matches_reference"] is True for row in rows))
        self.assertTrue(all(call.kwargs == {"audio_path": None} and not call.args for call in listen.call_args_list))
        for language, reference in checker.REFERENCE_CASES:
            output.assert_any_call(f"\n[Contoh {language}] Ucapkan: {reference}")

    def test_guided_explicit_thirty_samples_cycles_all_references(self):
        references = checker.REFERENCE_CASES * 3
        with (
            TemporaryDirectory() as temp,
            patch.object(checker, "PROJECT_ROOT", Path(temp)),
            patch.object(checker.speechtotext, "initialize_transcriber"),
            patch.object(checker.speechtotext, "listen_and_transcribe",
                         side_effect=[(text, language) for language, text in references]) as listen,
            patch.object(checker.speechtotext, "get_last_measurement", return_value={"status": "ok"}),
            patch("builtins.input", return_value=""), patch("builtins.print"),
        ):
            checker.main(["--guided", "--samples", "30"])
            report = next((Path(temp) / "logs" / "measurements").glob("*/utterances.jsonl"))
            rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["reference_language"] for row in rows], [language for language, _ in references])
        self.assertEqual([row["sample"] for row in rows], list(range(1, 31)))
        self.assertEqual(listen.call_count, 30)
        self.assertTrue(all(set(call.kwargs) == {"audio_path"} for call in listen.call_args_list))

    def test_missing_empty_or_error_metadata_never_counts_as_language_pass(self):
        utterances = [("text", "en"), ("text", "zh"), ("text", "hi"), ("", "es"), ("text", "fr")]
        measurements = [None, {}, {"status": "capture_error"}, {"status": "ok"}, {"status": "ok"}]
        with (
            TemporaryDirectory() as temp,
            patch.object(checker, "PROJECT_ROOT", Path(temp)),
            patch.object(checker.speechtotext, "initialize_transcriber"),
            patch.object(checker.speechtotext, "listen_and_transcribe", side_effect=utterances),
            patch.object(checker.speechtotext, "get_last_measurement", side_effect=measurements),
            patch("builtins.input", return_value=""), patch("builtins.print"),
        ):
            checker.main(["--guided", "--samples", "5", "--no-audio"])
            report = next((Path(temp) / "logs" / "measurements").glob("*/utterances.jsonl"))
            rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["language_matches_reference"] for row in rows], [None, None, None, None, False])

    def test_guided_quit_keeps_completed_rows_only(self):
        with (
            TemporaryDirectory() as temp,
            patch.object(checker, "PROJECT_ROOT", Path(temp)),
            patch.object(checker.speechtotext, "initialize_transcriber"),
            patch.object(checker.speechtotext, "listen_and_transcribe", return_value=("Sit", "en")) as listen,
            patch.object(checker.speechtotext, "get_last_measurement", return_value={"status": "ok"}),
            patch("builtins.input", side_effect=["", "q"]), patch("builtins.print"),
        ):
            checker.main(["--guided", "--no-audio"])
            report = next((Path(temp) / "logs" / "measurements").glob("*/utterances.jsonl"))
            rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        listen.assert_called_once_with(audio_path=None)

    def test_guided_capture_error_saves_reference_and_cannot_pass_with_stale_ok(self):
        with (
            TemporaryDirectory() as temp,
            patch.object(checker, "PROJECT_ROOT", Path(temp)),
            patch.object(checker.speechtotext, "initialize_transcriber"),
            patch.object(checker.speechtotext, "listen_and_transcribe", side_effect=OSError("microphone")),
            patch.object(checker.speechtotext, "get_last_measurement", return_value={"status": "ok"}),
            patch("builtins.input", return_value=""), patch("builtins.print"),
        ):
            with self.assertRaises(OSError):
                checker.main(["--guided", "--no-audio"])
            report = next((Path(temp) / "logs" / "measurements").glob("*/utterances.jsonl"))
            row = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(row["reference_language"], "en")
        self.assertEqual(row["reference"], checker.REFERENCE_CASES[0][1])
        self.assertIsNone(row["language_matches_reference"])


if __name__ == "__main__":
    unittest.main()
