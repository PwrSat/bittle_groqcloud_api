"""Regression tests for speech handling, without a microphone or model download.

Run with: python -m unittest tests.test_stt_pipeline -v
"""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import speechtotext as stt


class NormalizeTranscriptTests(unittest.TestCase):
    def test_whitespace_is_cleaned_without_changing_the_utterance(self):
        self.assertEqual(
            stt.normalize_transcript("  halo\n  Bittle\t!  ", "id", 0.8),
            ("halo Bittle !", "id", False),
        )

    def test_empty_or_punctuation_only_transcripts_are_ignored(self):
        for text in ("", "   ", "..  ..  ..", "?! ...", "— … 。"):
            with self.subTest(text=text):
                self.assertEqual(
                    stt.normalize_transcript(text, "fr", 0.22),
                    ("", "fr", False),
                )

    def test_canonical_indonesian_command_needs_no_correction(self):
        self.assertEqual(
            stt.normalize_transcript("duduk", "id", 0.8),
            ("duduk", "id", False),
        )

    def test_known_whole_utterance_aliases_and_repetitions_are_corrected(self):
        examples = (
            ("Duh Duk", "id", 0.8),
            ("Duh Duk  Duh Duk", "id", 0.8),
            ("duduk duduk", "id", 0.8),
            ("Doudouk", "fr", 0.22),
            ("Doudouk Doudouk", "fr", 0.22),
        )
        for text, language, confidence in examples:
            with self.subTest(text=text, language=language):
                self.assertEqual(
                    stt.normalize_transcript(text, language, confidence),
                    ("duduk", "id", True),
                )

    def test_confident_foreign_language_is_not_rewritten(self):
        for text in ("Doudouk", "Duh Duk", "Salut!"):
            with self.subTest(text=text):
                self.assertEqual(
                    stt.normalize_transcript(text, "fr", 0.9),
                    (text, "fr", False),
                )

    def test_negations_and_conversations_are_not_reduced_to_commands(self):
        examples = (
            "jangan duduk",
            "jangan Duh Duk",
            "aku tadi duduk di kursi",
            "aku mendengar Duh Duk dari luar",
            "duduklah",
            "produk",
        )
        for text in examples:
            with self.subTest(text=text):
                self.assertEqual(
                    stt.normalize_transcript(text, "id", 0.8),
                    (text, "id", False),
                )

    def test_multilingual_text_is_preserved_even_with_low_language_confidence(self):
        examples = (
            ("歩いて", "ja", 0.36),
            ("Thank you.", "en", 0.8),
            ("Salut!", "fr", 0.22),
        )
        for text, language, confidence in examples:
            with self.subTest(text=text, language=language):
                self.assertEqual(
                    stt.normalize_transcript(text, language, confidence),
                    (text, language, False),
                )


class FakeMicrophone:
    def __init__(self):
        self.source = object()

    def __enter__(self):
        return self.source

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeRecognizer:
    def __init__(self, events):
        self.events = events
        self.calibration_calls = []
        self.listen_calls = []

    def adjust_for_ambient_noise(self, source, duration):
        self.events.append(("calibrate", duration))
        self.calibration_calls.append((source, duration))

    def listen(self, source, timeout=None, phrase_time_limit=None):
        self.events.append(("listen", phrase_time_limit))
        self.listen_calls.append((source, timeout, phrase_time_limit))
        return SimpleNamespace(get_wav_data=lambda: b"fake-wave-audio")


class SpeechPipelineTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.recognizer = FakeRecognizer(self.events)
        self.microphone = FakeMicrophone()
        self.model = Mock()
        self.set_transcript("halo Bittle")
        patches = (
            patch.object(stt, "_calibrated", False),
            patch.object(stt, "STT_LANGUAGE", None),
            patch.object(stt, "_get_model", return_value=self.model),
            patch.object(
                stt,
                "_get_audio_devices",
                return_value=(self.recognizer, self.microphone),
            ),
            patch("builtins.print", side_effect=self.record_print),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def set_transcript(self, text, language="id", confidence=0.8):
        self.model.transcribe.return_value = (
            [SimpleNamespace(text=text)],
            SimpleNamespace(language=language, language_probability=confidence),
        )

    def record_print(self, *args, **kwargs):
        self.events.append(("print", " ".join(str(arg) for arg in args)))

    def test_calibration_finishes_before_the_prompt_and_recording(self):
        self.assertEqual(stt.listen_and_transcribe(), ("halo Bittle", "id"))
        calibration_index = next(
            index for index, event in enumerate(self.events) if event[0] == "calibrate"
        )
        prompt_index = self.events.index(("print", "Talk..."))
        listen_index = next(
            index for index, event in enumerate(self.events) if event[0] == "listen"
        )
        self.assertLess(calibration_index, prompt_index)
        self.assertLess(prompt_index, listen_index)

    def test_calibration_is_reused_on_the_next_utterance(self):
        self.assertEqual(stt.listen_and_transcribe(), ("halo Bittle", "id"))
        self.assertEqual(stt.listen_and_transcribe(), ("halo Bittle", "id"))
        self.assertEqual(len(self.recognizer.calibration_calls), 1)
        self.assertEqual(len(self.recognizer.listen_calls), 2)
        self.assertEqual(self.model.transcribe.call_count, 2)

    def test_auto_language_and_command_hints_are_passed_to_whisper(self):
        stt.listen_and_transcribe()
        args, kwargs = self.model.transcribe.call_args
        self.assertIsInstance(args[0], io.BytesIO)
        self.assertEqual(args[0].getvalue(), b"fake-wave-audio")
        self.assertIsNone(kwargs["language"])
        self.assertEqual(kwargs["hotwords"], stt.COMMAND_HINTS)
        self.assertTrue(kwargs["vad_filter"])
        self.assertIn("min_silence_duration_ms", kwargs["vad_parameters"])
        self.assertFalse(kwargs["condition_on_previous_text"])

    def test_explicit_language_setting_is_passed_to_whisper(self):
        with patch.object(stt, "STT_LANGUAGE", "id"):
            stt.listen_and_transcribe()
        self.assertEqual(self.model.transcribe.call_args.kwargs["language"], "id")

    def test_explicit_language_does_not_claim_detection_confidence(self):
        self.set_transcript("halo", language="id", confidence=1.0)
        with patch.object(stt, "STT_LANGUAGE", "id"):
            self.assertEqual(stt.listen_and_transcribe(), ("halo", "id"))
        printed = " ".join(event[1] for event in self.events if event[0] == "print")
        self.assertIn("bahasa ditetapkan", printed)
        self.assertNotIn("100%", printed)

    def test_missing_microphone_is_reported_to_the_caller(self):
        with patch.object(stt, "_get_audio_devices", side_effect=OSError("no microphone")):
            with self.assertRaisesRegex(OSError, "no microphone"):
                stt.listen_and_transcribe()

    def test_decoder_failure_returns_no_command(self):
        self.model.transcribe.side_effect = RuntimeError("decoder failed")
        self.assertEqual(stt.listen_and_transcribe(), (None, None))

    def test_punctuation_is_ignored_by_the_pipeline(self):
        self.set_transcript("..  ..  ..  ..")
        self.assertEqual(stt.listen_and_transcribe(), (None, None))

    def test_known_mishearing_is_corrected_before_return(self):
        self.set_transcript("Duh Duk  Duh Duk")
        self.assertEqual(stt.listen_and_transcribe(), ("duduk", "id"))

    def test_low_confidence_japanese_stays_japanese(self):
        self.set_transcript("歩いて", language="ja", confidence=0.36)
        self.assertEqual(stt.listen_and_transcribe(), ("歩いて", "ja"))


if __name__ == "__main__":
    unittest.main()
