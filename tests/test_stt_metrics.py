"""Speech measurements without opening a microphone or loading Whisper."""

import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import speechtotext as stt


class Clock:
    def __init__(self):
        self.now = 0.0

    def tick(self, seconds):
        self.now += seconds

    def __call__(self):
        return self.now


class SpeechMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.audio = SimpleNamespace(
            frame_data=b"x" * 64000, sample_rate=16000, sample_width=2,
            get_wav_data=Mock(return_value=b"original-wave-bytes"),
        )
        self.recognizer = Mock(energy_threshold=300.0)

        def calibrate(*args, **kwargs):
            self.clock.tick(1.0)

        def capture(*args, **kwargs):
            self.clock.tick(3.0)
            return self.audio

        self.recognizer.adjust_for_ambient_noise.side_effect = calibrate
        self.recognizer.listen.side_effect = capture
        self.microphone = Mock()
        self.microphone.__enter__ = Mock(return_value=object())
        self.microphone.__exit__ = Mock(return_value=False)
        self.model = Mock()
        self.info = SimpleNamespace(language="id", language_probability=0.8)
        self.model.detect_language.return_value = ("id", 0.8, [("id", 0.8)])
        self.pcm_audio = object()

        def transcribe(*args, **kwargs):
            self.clock.tick(0.2)

            def segments():
                self.clock.tick(0.3)
                yield SimpleNamespace(
                    text="Duh Duk", avg_logprob=-0.2, no_speech_prob=0.1,
                )
                self.clock.tick(0.5)
                yield SimpleNamespace(
                    text="Duh Duk", avg_logprob=-0.6, no_speech_prob=0.3,
                )

            return segments(), self.info

        self.model.transcribe.side_effect = transcribe
        for patcher in (
            patch.object(stt, "_calibrated", False),
            patch.object(stt, "_last_measurement", None),
            patch.object(stt, "_model_load_seconds", 2.5),
            patch.object(stt, "STT_LANGUAGE", None),
            patch.object(stt, "_get_model", return_value=self.model),
            patch.object(
                stt, "_decode_audio_for_detection", return_value=self.pcm_audio,
            ),
            patch.object(
                stt, "_get_audio_devices",
                return_value=(self.recognizer, self.microphone),
            ),
            patch.object(stt.time, "perf_counter", side_effect=self.clock),
            patch("builtins.print"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_times_include_generator_decoding_and_separate_capture_from_audio(self):
        self.assertEqual(stt.listen_and_transcribe(), ("duduk", "id"))
        result = stt.get_last_measurement()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["load_model_s"], 2.5)
        self.assertEqual(result["calibration_s"], 1.0)
        self.assertEqual(result["capture_s"], 3.0)
        self.assertEqual(result["audio_s"], 2.0)
        self.assertAlmostEqual(result["transcribe_s"], 1.0)
        self.assertEqual(result["raw_text"], "Duh Duk Duh Duk")
        self.assertEqual(result["normalized_text"], "duduk")
        self.assertEqual(result["language_detected"], "id")
        self.assertEqual(result["language_returned"], "id")
        self.assertEqual(result["language_probability"], 0.8)
        self.assertEqual(result["language_probability_scope"], "language_only")
        self.assertAlmostEqual(result["mean_avg_logprob"], -0.4)
        self.assertAlmostEqual(result["mean_no_speech_prob"], 0.2)
        self.assertEqual(result["segment_count"], 2)
        self.assertEqual(result["energy_threshold"], 300.0)
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["compute_type"], "int8")
        self.assertIsNone(result["error_type"])
        json.dumps(result, allow_nan=False)

    def test_model_timing_is_exposed_without_reloading_or_recalibrating(self):
        stt.listen_and_transcribe()
        stt.listen_and_transcribe()
        self.assertEqual(stt.get_model_load_seconds(), 2.5)
        self.assertEqual(stt.get_last_measurement()["calibration_s"], 0.0)
        self.recognizer.adjust_for_ambient_noise.assert_called_once()

    def test_caller_cannot_change_stored_measurement(self):
        self.assertIsNone(stt.get_last_measurement())
        stt.listen_and_transcribe()
        result = stt.get_last_measurement()
        result["status"] = "changed"
        self.assertEqual(stt.get_last_measurement()["status"], "ok")

    def test_no_wav_is_written_in_normal_application_use(self):
        with patch.object(Path, "write_bytes") as write:
            stt.listen_and_transcribe()
        write.assert_not_called()

    def test_explicit_wav_file_is_the_exact_audio_given_to_whisper(self):
        with (
            patch.object(Path, "write_bytes") as write,
            patch.object(
                stt, "_decode_audio_for_detection", return_value=self.pcm_audio,
            ) as decode,
        ):
            stt.listen_and_transcribe(audio_path="explicit-benchmark.wav")
        write.assert_called_once_with(b"original-wave-bytes")
        self.audio.get_wav_data.assert_called_once_with()
        decode.assert_called_once_with(b"original-wave-bytes")
        self.assertIs(self.model.transcribe.call_args.args[0], self.pcm_audio)
        self.assertIs(
            self.model.detect_language.call_args.kwargs["audio"], self.pcm_audio,
        )

    def test_measurement_keeps_original_detection_after_decoder_is_given_language(self):
        self.info.language_probability = 1.0
        stt.listen_and_transcribe()
        result = stt.get_last_measurement()
        self.assertEqual(result["language_detected"], "id")
        self.assertEqual(result["language_probability"], 0.8)
        self.assertEqual(result["language_probability_scope"], "language_only")

    def test_explicit_language_has_no_automatic_detection_probability(self):
        self.info.language_probability = 1.0
        with patch.object(stt, "STT_LANGUAGE", "id"):
            stt.listen_and_transcribe()
        self.model.detect_language.assert_not_called()
        result = stt.get_last_measurement()
        self.assertIsNone(result["language_probability"])

    def test_missing_optional_audio_and_segment_metadata_does_not_break_pipeline(self):
        self.audio = SimpleNamespace(get_wav_data=lambda: b"fake-wave")
        self.model.transcribe.side_effect = None
        self.model.transcribe.return_value = (
            [SimpleNamespace(text="halo")], self.info,
        )
        self.assertEqual(stt.listen_and_transcribe(), ("halo", "id"))
        result = stt.get_last_measurement()
        self.assertIsNone(result["audio_s"])
        self.assertIsNone(result["mean_avg_logprob"])
        self.assertIsNone(result["mean_no_speech_prob"])

    def test_empty_transcript_still_records_raw_text_and_language(self):
        self.model.transcribe.side_effect = None
        self.model.transcribe.return_value = (
            [SimpleNamespace(text="... ...")], self.info,
        )
        self.assertEqual(stt.listen_and_transcribe(), (None, None))
        result = stt.get_last_measurement()
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["raw_text"], "... ...")
        self.assertEqual(result["normalized_text"], "")
        self.assertEqual(result["language_detected"], "id")
        self.assertIsNone(result["language_returned"])

    def test_decode_failure_keeps_type_without_error_message_and_resets_old_text(self):
        stt.listen_and_transcribe()

        def decode_error(*args, **kwargs):
            self.clock.tick(0.4)
            raise RuntimeError("private failure detail")

        self.model.transcribe.side_effect = decode_error
        self.assertEqual(stt.listen_and_transcribe(), (None, None))
        result = stt.get_last_measurement()
        self.assertEqual(result["status"], "decode_error")
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertEqual(result["error_stage"], "decode")
        self.assertAlmostEqual(result["transcribe_s"], 0.4)
        self.assertIsNone(result["raw_text"])
        self.assertEqual(result["segment_count"], 0)
        self.assertNotIn("private failure detail", json.dumps(result))

    def test_generator_failure_is_timed_and_reported(self):
        def failing_segments():
            self.clock.tick(0.6)
            raise ValueError("bad audio")
            yield  # Make this a generator; failure happens during iteration.

        self.model.transcribe.side_effect = None
        self.model.transcribe.return_value = (failing_segments(), self.info)
        self.assertEqual(stt.listen_and_transcribe(), (None, None))
        result = stt.get_last_measurement()
        self.assertEqual(result["status"], "decode_error")
        self.assertAlmostEqual(result["transcribe_s"], 0.6)

    def test_capture_error_is_recorded_before_it_reaches_the_caller(self):
        def capture_error(*args, **kwargs):
            self.clock.tick(1.2)
            raise OSError("microphone disconnected")

        self.recognizer.listen.side_effect = capture_error
        with self.assertRaises(OSError):
            stt.listen_and_transcribe()
        result = stt.get_last_measurement()
        self.assertEqual(result["status"], "capture_error")
        self.assertEqual(result["error_type"], "OSError")
        self.assertEqual(result["error_stage"], "capture")
        self.assertAlmostEqual(result["capture_s"], 1.2)
        self.assertIsNone(result["transcribe_s"])

    def test_model_error_is_recorded_without_opening_microphone(self):
        with patch.object(stt, "_get_model", side_effect=RuntimeError("unavailable")):
            with self.assertRaises(RuntimeError):
                stt.listen_and_transcribe()
        result = stt.get_last_measurement()
        self.assertEqual(result["status"], "capture_error")
        self.assertEqual(result["error_stage"], "model_load")
        self.recognizer.listen.assert_not_called()

    def test_nonfinite_optional_metadata_stays_json_safe(self):
        self.model.transcribe.side_effect = None
        self.model.transcribe.return_value = (
            [SimpleNamespace(text="halo", avg_logprob=float("nan"),
                             no_speech_prob=float("inf"))], self.info,
        )
        self.recognizer.energy_threshold = float("nan")
        stt.listen_and_transcribe()
        result = stt.get_last_measurement()
        self.assertIsNone(result["mean_avg_logprob"])
        self.assertIsNone(result["mean_no_speech_prob"])
        self.assertIsNone(result["energy_threshold"])
        json.dumps(result, allow_nan=False)


class ModelStartupMeasurementTests(unittest.TestCase):
    def test_model_load_time_is_measured_once_and_reused(self):
        clock = Clock()

        def load(*args, **kwargs):
            clock.tick(2.25)
            return object()

        module = ModuleType("faster_whisper")
        module.WhisperModel = Mock(side_effect=load)
        with (
            patch.object(stt, "_model", None),
            patch.object(stt, "_model_load_seconds", None),
            patch.dict(sys.modules, {"faster_whisper": module}),
            patch.object(stt.time, "perf_counter", side_effect=clock),
            patch("builtins.print"),
        ):
            self.assertIsNone(stt.get_model_load_seconds())
            self.assertIs(stt.initialize_transcriber(), stt.initialize_transcriber())
            self.assertEqual(stt.get_model_load_seconds(), 2.25)
            module.WhisperModel.assert_called_once_with(
                stt.WHISPER_MODEL, device="cpu", compute_type="int8",
            )


if __name__ == "__main__":
    unittest.main()
