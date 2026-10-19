"""Decoder and measurement regressions without a model, audio device, or API."""

import io
import json
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import speechtotext as stt


UTTERANCES = (
    ("en", "Don't sit down, stand up instead."),
    ("zh", "不要坐下，请站起来。"),
    ("hi", "मत बैठो, खड़े हो जाओ।"),
    ("es", "No te sientes, ponte de pie."),
    ("ar", "لا تجلس، قف فقط."),
    ("fr", "Ne t'assieds pas, reste debout."),
    ("pt", "Não te sentes, fica de pé."),
    ("id", "Jangan duduk, berdiri saja."),
    ("ja", "座らないで、立ってください。"),
    ("ko", "앉지 말고 서 주세요."),
)
NON_LATIN = {"zh", "hi", "ar", "ja", "ko"}


class FakeMicrophone:
    def __enter__(self):
        return sentinel.microphone_source

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class DecoderPolicyTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock(spec=["detect_language", "transcribe"])
        for patcher in (
            patch.object(stt, "STT_LANGUAGE", None),
            patch.object(stt, "_decode_audio_for_detection", return_value=sentinel.decoded_audio),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def configure_decoder(self, language, text, confidence=0.72):
        self.model.detect_language.return_value = (
            language, confidence, [(language, confidence)],
        )
        self.model.transcribe.return_value = (
            [SimpleNamespace(text=text)],
            SimpleNamespace(language=language, language_probability=1.0),
        )

    def assert_decoder_policy(self, language):
        kwargs = self.model.transcribe.call_args.kwargs
        self.assertEqual(kwargs["language"], language)
        self.assertEqual(kwargs["temperature"], 0.0)
        expected_hints = None if language in NON_LATIN else stt.COMMAND_HINTS
        self.assertEqual(kwargs["hotwords"], expected_hints)
        self.assertTrue(kwargs["vad_filter"])
        self.assertFalse(kwargs["condition_on_previous_text"])

    def test_automatic_decoder_disables_latin_hints_only_for_non_latin_languages(self):
        for language, text in UTTERANCES:
            with self.subTest(language=language):
                self.model.reset_mock()
                self.configure_decoder(language, text)

                result = stt.transcribe_audio(b"fake-wave", model=self.model)

                self.assert_decoder_policy(language)
                self.assertEqual(result["text"], text)
                self.assertEqual(result["language"], language)
                self.assertAlmostEqual(result["language_probability"], 0.72)
                self.assertIs(self.model.transcribe.call_args.args[0], sentinel.decoded_audio)
                self.model.detect_language.assert_called_once()
                self.model.transcribe.assert_called_once()

    def test_explicit_experiment_uses_same_decoder_policy_without_claiming_confidence(self):
        for language, text in UTTERANCES:
            with self.subTest(language=language):
                self.model.reset_mock()
                self.configure_decoder(language, text)

                with patch.object(stt, "STT_LANGUAGE", language):
                    result = stt.transcribe_audio(b"fake-wave", model=self.model)

                self.assert_decoder_policy(language)
                audio = self.model.transcribe.call_args.args[0]
                self.assertIsInstance(audio, io.BytesIO)
                self.assertEqual(audio.getvalue(), b"fake-wave")
                self.assertEqual(result["text"], text)
                self.assertIsNone(result["language_probability"])
                self.model.detect_language.assert_not_called()
                self.model.transcribe.assert_called_once()

    def test_existing_wrong_spanish_detection_can_still_recover_indonesian_command(self):
        self.configure_decoder("es", "tengok kanan", confidence=0.20)
        self.model.detect_language.return_value = (
            "es", 0.20, [("es", 0.20), ("id", 0.15), ("en", 0.10)],
        )

        result = stt.transcribe_audio(b"fake-wave", model=self.model)

        self.assert_decoder_policy("es")
        self.assertEqual(result["text"], "tengok kanan")
        self.assertEqual(result["language"], "id")
        self.assertAlmostEqual(result["language_probability"], 0.15)
        self.assertEqual(result["language_detected"], "es")
        self.assertAlmostEqual(result["language_detected_probability"], 0.20)

    def test_arabic_and_hindi_non_command_text_is_never_replaced_by_a_canned_command(self):
        examples = (
            ("ar", "هذا اختبار مختلف، لا تتحرك."),
            ("hi", "मुझे नहीं पता, अभी मत चलो।"),
        )
        for language, text in examples:
            with self.subTest(language=language):
                self.configure_decoder(language, text, confidence=0.42)

                result = stt.transcribe_audio(b"fake-wave", model=self.model)
                normalized = stt.normalize_transcript(
                    result["text"], result["language"], result["language_probability"],
                )

                self.assertEqual(result["text"], text)
                self.assertEqual(normalized, (text, language, False))
                self.assertAlmostEqual(result["language_probability"], 0.42)


class DecoderMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock(spec=["detect_language", "transcribe"])
        audio = SimpleNamespace(
            frame_data=b"\x00\x00" * 16000, sample_rate=16000, sample_width=2,
            get_wav_data=lambda: b"fake-wave",
        )
        self.recognizer = Mock()
        self.recognizer.energy_threshold = 300.0
        self.recognizer.listen.return_value = audio
        self.configure_segments([SimpleNamespace(text="لا تجلس، قف فقط.")])
        for patcher in (
            patch.object(stt, "STT_LANGUAGE", None),
            patch.object(stt, "_calibrated", True),
            patch.object(stt, "_last_measurement", None),
            patch.object(stt, "_get_model", return_value=self.model),
            patch.object(stt, "_decode_audio_for_detection", return_value=sentinel.decoded_audio),
            patch.object(stt, "_get_audio_devices", return_value=(self.recognizer, FakeMicrophone())),
            patch("builtins.print"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def configure_segments(self, segments):
        self.model.detect_language.return_value = ("ar", 0.68, [("ar", 0.68), ("en", 0.10)])
        self.model.transcribe.return_value = (
            segments, SimpleNamespace(language="ar", language_probability=1.0),
        )

    def test_segment_metrics_report_actual_temperatures_and_compression_not_defaults(self):
        self.configure_segments([
            SimpleNamespace(text="لا تجلس،", temperature=Decimal("0.0"), compression_ratio=Decimal("1.2")),
            SimpleNamespace(text="قف فقط.", temperature=Decimal("0.4"), compression_ratio=Decimal("2.4")),
            SimpleNamespace(text="", temperature=float("nan"), compression_ratio=float("inf")),
            SimpleNamespace(text="", temperature="invalid", compression_ratio="invalid"),
        ])

        self.assertEqual(stt.listen_and_transcribe(), ("لا تجلس، قف فقط.", "ar"))
        measurement = stt.get_last_measurement()

        self.assertEqual(measurement["decoder_temperatures"], [0.0, 0.4])
        self.assertAlmostEqual(measurement["mean_compression_ratio"], 1.8)
        self.assertAlmostEqual(measurement["language_probability"], 0.68)
        self.assertAlmostEqual(measurement["language_returned_probability"], 0.68)
        self.assertEqual(measurement["language_probability_scope"], "language_only")
        self.assertEqual(measurement["status"], "ok")
        serialized = json.dumps(measurement, allow_nan=False, ensure_ascii=False)
        restored = json.loads(serialized)
        self.assertEqual(restored["decoder_temperatures"], [0.0, 0.4])
        self.assertEqual(restored["normalized_text"], "لا تجلس، قف فقط.")

    def test_missing_segment_metrics_remain_unknown_instead_of_reporting_zero(self):
        self.assertEqual(stt.listen_and_transcribe(), ("لا تجلس، قف فقط.", "ar"))

        measurement = stt.get_last_measurement()

        self.assertEqual(measurement["decoder_temperatures"], [])
        self.assertIsNone(measurement["mean_compression_ratio"])
        self.assertIsNone(measurement["mean_avg_logprob"])
        self.assertIsNone(measurement["mean_no_speech_prob"])
        json.dumps(measurement, allow_nan=False)

    def test_invalid_only_segment_metrics_remain_unknown(self):
        self.configure_segments([
            SimpleNamespace(text="لا تجلس، قف فقط.", temperature=float("inf"), compression_ratio=float("nan")),
        ])

        stt.listen_and_transcribe()
        measurement = stt.get_last_measurement()

        self.assertEqual(measurement["decoder_temperatures"], [])
        self.assertIsNone(measurement["mean_compression_ratio"])
        json.dumps(measurement, allow_nan=False)

    def test_unsupported_audio_does_not_report_fabricated_decoder_metrics(self):
        self.model.detect_language.return_value = ("it", 0.99, [("it", 0.99), ("en", 0.01)])

        self.assertEqual(stt.listen_and_transcribe(), (None, None))
        measurement = stt.get_last_measurement()

        self.assertEqual(measurement["status"], "unsupported_language")
        self.assertEqual(measurement["decoder_temperatures"], [])
        self.assertIsNone(measurement["mean_compression_ratio"])
        self.model.transcribe.assert_not_called()


class CapturePausePolicyTests(unittest.TestCase):
    def test_new_recognizer_waits_through_a_short_multilingual_sentence_pause(self):
        recognizer = SimpleNamespace(pause_threshold=0.8)
        recognizer_factory = Mock(return_value=recognizer)
        microphone_factory = Mock(return_value=sentinel.fake_microphone)
        fake_sr = SimpleNamespace(Recognizer=recognizer_factory, Microphone=microphone_factory)

        with (
            patch.dict("sys.modules", {"speech_recognition": fake_sr}),
            patch.object(stt, "_recognizer", None),
            patch.object(stt, "_microphone", None),
        ):
            first = stt._get_audio_devices()
            second = stt._get_audio_devices()

        self.assertEqual(recognizer.pause_threshold, 1.3)
        self.assertEqual(first, (recognizer, sentinel.fake_microphone))
        self.assertEqual(second, first)
        recognizer_factory.assert_called_once_with()
        microphone_factory.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
