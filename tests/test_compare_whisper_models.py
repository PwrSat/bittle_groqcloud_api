"""Isolated Whisper comparison stays offline unless explicitly run/downloaded."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, sentinel
import wave
import weakref

import speechtotext as stt
from tools import check_language_understanding as sessions
from tools import compare_whisper_models as comparison
from tools.check_multilingual import REFERENCE_CASES


class WhisperModelComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "project"
        self.root.mkdir()
        self.session = self.root / "logs" / "measurements" / "20261021-120000-multilingual"
        self.session.mkdir(parents=True)
        self.rows = []
        self.wav_bytes = {}
        for sample, actual_language in ((2, "es"), (3, "hi")):
            language, reference = REFERENCE_CASES[sample - 1]
            self.rows.append({
                "sample": sample, "reference_language": language, "reference": reference,
                "speech": {"status": "ok", "normalized_text": f"ACTUAL_ASR_{sample}",
                           "language_returned": actual_language},
            })
            path = self.session / f"{sample:02}.wav"
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(bytes([sample, 0]) * 1600)
            self.wav_bytes[sample] = path.read_bytes()
        self.write_rows()
        for patcher in (
            patch.object(comparison, "PROJECT_ROOT", self.root),
            patch.object(sessions, "PROJECT_ROOT", self.root),
            patch.dict(os.environ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_rows(self):
        (self.session / "utterances.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.rows), encoding="utf-8",
        )

    def test_default_dry_never_initializes_a_model_api_microphone_or_backend(self):
        with (
            patch.object(comparison, "load_model", side_effect=AssertionError("no model in dry mode")) as load,
            patch.object(comparison, "_whisper_model_class", side_effect=AssertionError("no backend")) as backend,
            patch.object(stt, "initialize_transcriber", side_effect=AssertionError("no app model")),
            patch.object(stt, "_get_audio_devices", side_effect=AssertionError("no microphone")),
            patch("llm_utils.get_model", side_effect=AssertionError("no API")),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(comparison.main(["--session", "latest", "--languages", "zh,hi"]), 0)

        load.assert_not_called()
        backend.assert_not_called()
        with (
            patch.object(comparison, "load_model") as load,
            redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            comparison.main(["--download", "--session", "latest"])
        load.assert_not_called()

    def test_preparation_reads_actual_sample_wav_and_filters_reference_not_detected_language(self):
        source, cases = comparison.prepare_audio_cases("latest", languages={"zh"})

        self.assertEqual(source, (self.session / "utterances.jsonl").resolve())
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["sample"], 2)
        self.assertEqual(cases[0]["reference_language"], "zh")
        self.assertEqual(cases[0]["language"], "es")
        self.assertEqual(cases[0]["audio_bytes"], self.wav_bytes[2])
        self.assertEqual(Path(cases[0]["wav_path"]), self.session / "02.wav")
        self.assertAlmostEqual(cases[0]["audio_s"], 0.1)

    def test_invalid_session_or_wav_is_rejected_before_backend_loading(self):
        outside = self.base / "outside"
        outside.mkdir()
        outside_file = outside / "utterances.jsonl"
        outside_file.write_text(json.dumps(self.rows[0]) + "\n", encoding="utf-8")
        original_rows = copy.deepcopy(self.rows)
        for fault in ("missing_wav", "invalid_wav", "bad_status", "outside"):
            with self.subTest(fault=fault):
                self.rows = copy.deepcopy(original_rows)
                self.write_rows()
                wav = self.session / "02.wav"
                wav.write_bytes(self.wav_bytes[2])
                session = "latest"
                if fault == "missing_wav":
                    wav.unlink()
                elif fault == "invalid_wav":
                    wav.write_bytes(b"not a WAV file")
                elif fault == "bad_status":
                    self.rows[0]["speech"]["status"] = "decode_error"
                    self.write_rows()
                else:
                    session = str(outside_file)

                with (
                    patch.object(comparison, "load_model") as load,
                    patch.object(comparison, "_whisper_model_class") as backend,
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
                    self.assertRaises((ValueError, SystemExit)),
                ):
                    comparison.main(["--run", "--download", "--session", session])

                load.assert_not_called()
                backend.assert_not_called()

    def test_model_factory_uses_read_only_small_cache_and_project_medium_cache(self):
        factory = Mock(side_effect=[OSError("small not cached globally"), sentinel.small_model])
        with patch.object(comparison, "_whisper_model_class", return_value=factory):
            model = comparison.load_model("small", device="cpu", compute_type="int8", allow_download=False)
        self.assertIs(model, sentinel.small_model)
        self.assertEqual(factory.call_count, 2)
        first, fallback = factory.call_args_list
        self.assertTrue(first.kwargs["local_files_only"])
        self.assertNotIn("download_root", first.kwargs)
        self.assertTrue(fallback.kwargs["local_files_only"])
        self.assertEqual(Path(fallback.kwargs["download_root"]), self.root / "models")

        for allow_download in (False, True):
            with self.subTest(allow_download=allow_download):
                factory = Mock(return_value=sentinel.medium_model)
                with patch.object(comparison, "_whisper_model_class", return_value=factory):
                    model = comparison.load_model("medium", device="cpu", compute_type="int8", allow_download=allow_download)
                self.assertIs(model, sentinel.medium_model)
                factory.assert_called_once()
                self.assertEqual(factory.call_args.args, ("medium",))
                self.assertEqual(Path(factory.call_args.kwargs["download_root"]), self.root / "models")
                self.assertEqual(factory.call_args.kwargs["local_files_only"], not allow_download)
                self.assertEqual(factory.call_args.kwargs["device"], "cpu")
                self.assertEqual(factory.call_args.kwargs["compute_type"], "int8")

    def test_transcribe_case_passes_only_audio_model_and_preserves_real_output_metadata(self):
        decoded = {
            "status": "ok", "text": "मत बैठो।", "language": "hi", "language_probability": 0.63,
            "language_selected": "hi", "language_selected_probability": 0.63,
            "language_detected": "hi", "language_detected_probability": 0.63,
            "language_selection_reason": "supported_detection", "language_supported_mass": 0.9,
            "language_candidates": [],
            "segments": [SimpleNamespace(text="मत बैठो।", temperature=0.4, compression_ratio=1.2,
                                          avg_logprob=-0.4, no_speech_prob=0.01)],
            "info": SimpleNamespace(language="hi", language_probability=1.0),
        }
        with patch.object(stt, "transcribe_audio", return_value=decoded) as transcribe:
            result = comparison.transcribe_case(self.wav_bytes[3], sentinel.model)

        transcribe.assert_called_once_with(self.wav_bytes[3], model=sentinel.model)
        self.assertEqual(result["raw_text"], "मत बैठो।")
        self.assertEqual(result["normalized_text"], "मत बैठो।")
        self.assertEqual(result["decoder_temperatures"], [0.4])
        self.assertAlmostEqual(result["mean_compression_ratio"], 1.2)
        self.assertGreaterEqual(result["transcribe_s"], 0)
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_run_uses_same_bytes_for_both_models_and_unloads_before_loading_next(self):
        events, previous = [], []
        configuration = {key: getattr(stt, key) for key in
                         ("STT_LANGUAGE", "WHISPER_MODEL", "WHISPER_DEVICE", "WHISPER_COMPUTE_TYPE",
                          "WHISPER_TEMPERATURE")}

        class FakeModel:
            def __init__(self, name):
                self.name = name

        def load(name, **options):
            if previous:
                self.assertIsNone(previous[-1](), "previous model still held at next load")
            model = FakeModel(name)
            previous.append(weakref.ref(model))
            events.append(("load", name, options))
            return model

        def transcribe(audio, model):
            events.append(("decode", model.name, audio))
            return {"status": "ok", "raw_text": "REAL_DECODE", "normalized_text": "REAL_DECODE",
                    "language_returned": "en", "transcribe_s": 0.1, "decoder_temperatures": [0.0]}

        with (
            patch.object(comparison, "load_model", new=load),
            patch.object(comparison, "transcribe_case", new=transcribe),
            redirect_stdout(io.StringIO()),
        ):
            status = comparison.main(["--run", "--session", "latest", "--models", "small", "medium"])

        self.assertEqual(status, 0)
        self.assertEqual([event[1] for event in events if event[0] == "load"], ["small", "medium"])
        self.assertEqual([event[2] for event in events if event[0] == "decode"],
                         [self.wav_bytes[2], self.wav_bytes[3]] * 2)
        self.assertTrue(all(event[2]["allow_download"] is False for event in events if event[0] == "load"))
        self.assertEqual({key: getattr(stt, key) for key in configuration}, configuration)

    def test_failed_model_preserves_partial_report_and_unknown_failure_timing(self):
        def load(name, **options):
            if name == "medium":
                raise OSError("MODEL_NOT_CACHED")
            return sentinel.small_model

        with (
            patch.object(comparison, "load_model", new=load),
            patch.object(comparison, "transcribe_case", return_value={
                "status": "ok", "raw_text": "REAL_DECODE", "normalized_text": "REAL_DECODE",
                "language_returned": "en", "transcribe_s": 0.1,
            }),
            redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
        ):
            status = comparison.main(["--run", "--session", "latest", "--models", "small", "medium"])

        report_path = next(self.root.glob("logs/measurements/**/summary.json"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(status, 1)
        small, medium = report["models"]
        self.assertEqual(small["model"], "small")
        self.assertEqual(len(small["records"]), 2)
        self.assertTrue(all(record["raw_text"] == "REAL_DECODE" for record in small["records"]))
        self.assertEqual(small["records"][0]["previous_asr_text"], "ACTUAL_ASR_2")
        self.assertNotIn("text", small["records"][0])
        self.assertNotIn("language", small["records"][0])
        self.assertEqual(medium["model"], "medium")
        self.assertEqual(medium["status"], "error")
        self.assertIsNone(medium["model_load_s"])
        self.assertEqual(medium["records"], [])
        self.assertNotIn("passed", report)

    def test_failed_utterance_keeps_partial_result_without_fabricated_text_or_timing(self):
        successful = {"status": "ok", "raw_text": "REAL_DECODE", "normalized_text": "REAL_DECODE",
                      "language_returned": "en", "transcribe_s": 0.1}
        with (
            patch.object(comparison, "load_model", return_value=sentinel.model),
            patch.object(comparison, "transcribe_case", side_effect=[successful, RuntimeError("decode failure")]),
            redirect_stdout(io.StringIO()),
        ):
            status = comparison.main(["--run", "--session", "latest", "--models", "small"])

        report_path = next(self.root.glob("logs/measurements/**/summary.json"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(status, 1)
        model = report["models"][0]
        self.assertEqual(model["status"], "error")
        succeeded, failed = model["records"]
        self.assertEqual(succeeded["normalized_text"], "REAL_DECODE")
        self.assertEqual(failed["status"], "error")
        self.assertIsNone(failed["transcribe_s"])
        self.assertNotIn("raw_text", failed)
        self.assertNotIn("normalized_text", failed)
        self.assertEqual(failed["previous_asr_text"], "ACTUAL_ASR_3")
        self.assertNotIn("text", failed)
        self.assertEqual(failed["reference"], self.rows[1]["reference"])
        self.assertNotIn("passed", report)


if __name__ == "__main__":
    unittest.main()
