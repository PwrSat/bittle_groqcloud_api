"""Automatic language selection without audio devices, model loading or APIs."""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import speechtotext as stt


SUPPORTED_EXAMPLES = (
    ("en", "Please sit down."),
    ("zh", "请坐下。"),
    ("hi", "कृपया बैठो।"),
    ("es", "Siéntate, por favor."),
    ("ar", "اجلس من فضلك"),
    ("fr", "Assieds-toi, s'il te plaît."),
    ("pt", "Senta, por favor."),
    ("id", "Tolong duduk."),
    ("ja", "座ってください。"),
    ("ko", "앉아 주세요."),
)


def language_distribution(language, probability=0.8):
    """A detector result with all ten supported language candidates."""
    others = [code for code, _ in SUPPORTED_EXAMPLES if code != language]
    remaining = (1.0 - probability) / len(others)
    return language, probability, [
        (language, probability),
        *((code, remaining) for code in others),
    ]


def decoded_response(text, language):
    # Whisper returns probability 1.0 when language is supplied by the caller.
    # This is not confidence that the audio really uses that language.
    return (
        iter([SimpleNamespace(text=text)]),
        SimpleNamespace(language=language, language_probability=1.0),
    )


class AutomaticMultilingualPipelineTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock(spec=["detect_language", "transcribe"])
        self.decode_audio = Mock(return_value=sentinel.decoded_audio)
        for patcher in (
            patch.object(stt, "STT_LANGUAGE", None),
            patch.object(stt, "_decode_audio_for_detection", self.decode_audio),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def configure_response(self, language, text, probability=0.8):
        self.model.detect_language.return_value = language_distribution(
            language, probability,
        )
        self.model.transcribe.return_value = decoded_response(text, language)

    def test_all_ten_languages_are_selected_automatically(self):
        self.assertEqual(
            set(stt.SUPPORTED_LANGUAGES),
            {language for language, _ in SUPPORTED_EXAMPLES},
        )
        for language, text in SUPPORTED_EXAMPLES:
            with self.subTest(language=language):
                self.model.reset_mock()
                self.configure_response(language, text)

                result = stt.transcribe_audio(b"wave", model=self.model)

                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["language"], language)
                self.assertEqual(result["text"], text)
                self.assertEqual(result["language_detected"], language)
                self.assertAlmostEqual(result["language_probability"], 0.8)
                self.model.detect_language.assert_called_once()
                self.model.transcribe.assert_called_once()
                args, kwargs = self.model.transcribe.call_args
                self.assertIs(args[0], sentinel.decoded_audio)
                self.assertEqual(kwargs["language"], language)

    def test_every_utterance_can_switch_language_without_previous_language_lock(self):
        self.model.detect_language.side_effect = [
            language_distribution(language)
            for language, _ in SUPPORTED_EXAMPLES
        ]
        self.model.transcribe.side_effect = [
            decoded_response(text, language)
            for language, text in SUPPORTED_EXAMPLES
        ]

        results = [
            stt.transcribe_audio(b"wave", model=self.model)
            for _ in SUPPORTED_EXAMPLES
        ]

        self.assertEqual(
            [result["language"] for result in results],
            [language for language, _ in SUPPORTED_EXAMPLES],
        )
        self.assertEqual(self.model.detect_language.call_count, 10)
        self.assertEqual(self.model.transcribe.call_count, 10)
        self.assertEqual(self.decode_audio.call_count, 10)

    def test_selected_probability_is_not_decoder_forced_probability(self):
        self.configure_response("en", "Thank you.", probability=0.72)

        result = stt.transcribe_audio(b"wave", model=self.model)

        self.assertAlmostEqual(result["language_probability"], 0.72)
        self.assertAlmostEqual(result["language_detected_probability"], 0.72)
        self.assertEqual(result["info"].language_probability, 1.0)

    def test_missing_detector_probability_is_recovered_without_using_decoder_probability(self):
        self.model.detect_language.return_value = ("en", None, [("en", 0.7), ("id", 0.2)])
        self.model.transcribe.return_value = decoded_response("Hello there.", "en")

        result = stt.transcribe_audio(b"wave", model=self.model)

        self.assertEqual(result["language_detected_probability"], 0.7)
        self.assertEqual(result["language_selected_probability"], 0.7)
        self.assertEqual(result["language_probability"], 0.7)

    def test_low_confidence_japanese_remains_japanese(self):
        self.configure_response("ja", "歩いて", probability=0.36)

        result = stt.transcribe_audio(b"wave", model=self.model)

        self.assertEqual(result["text"], "歩いて")
        self.assertEqual(result["language"], "ja")
        self.assertAlmostEqual(result["language_probability"], 0.36)
        self.assertEqual(self.model.transcribe.call_args.kwargs["language"], "ja")

    def test_confident_french_is_preserved(self):
        self.configure_response("fr", "Regarde à droite.", probability=0.9)

        result = stt.transcribe_audio(b"wave", model=self.model)

        self.assertEqual(result["text"], "Regarde à droite.")
        self.assertEqual(result["language"], "fr")
        self.assertAlmostEqual(result["language_probability"], 0.9)

    def test_exact_command_evidence_uses_final_language_original_probability(self):
        self.model.detect_language.return_value = (
            "es", 0.20,
            [("es", 0.20), ("id", 0.15), ("en", 0.14), ("ja", 0.12),
             ("fr", 0.08), ("pt", 0.07), ("ar", 0.07), ("hi", 0.06),
             ("zh", 0.06), ("ko", 0.05)],
        )
        self.model.transcribe.return_value = decoded_response("tengok kanan", "es")

        result = stt.transcribe_audio(b"wave", model=self.model)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["text"], "tengok kanan")
        self.assertEqual(result["language"], "id")
        self.assertAlmostEqual(result["language_probability"], 0.15)
        self.assertEqual(result["language_detected"], "es")
        self.assertAlmostEqual(result["language_detected_probability"], 0.20)

    def test_full_negation_is_preserved_instead_of_reduced_to_a_command(self):
        examples = (
            ("id", "Jangan duduk, berdiri saja."),
            ("en", "Don't sit down, stand up instead."),
            ("zh", "不要坐下，请站起来。"),
            ("hi", "मत बैठो, खड़े हो जाओ।"),
            ("es", "No te sientes, ponte de pie."),
            ("ar", "لا تجلس، قف فقط."),
            ("fr", "Ne t'assieds pas, reste debout."),
            ("pt", "Não te sentes, fica de pé."),
            ("ja", "座らないで、立ってください。"),
            ("ko", "앉지 말고 서 주세요."),
        )
        for language, text in examples:
            with self.subTest(language=language):
                self.configure_response(language, text)

                result = stt.transcribe_audio(b"wave", model=self.model)

                self.assertEqual(result["text"], text)
                self.assertEqual(result["language"], language)

    def test_strongly_unsupported_language_does_not_decode_in_a_wrong_language(self):
        self.model.detect_language.return_value = (
            "it", 0.97,
            [("it", 0.97), ("en", 0.01), ("id", 0.005), ("ms", 0.005)],
        )

        result = stt.transcribe_audio(b"wave", model=self.model)

        self.assertEqual(result["status"], "unsupported_language")
        self.assertIsNone(result["language"])
        self.assertEqual(result["language_detected"], "it")
        self.assertAlmostEqual(result["language_detected_probability"], 0.97)
        self.model.transcribe.assert_not_called()

    def test_removed_malay_and_afrikaans_are_not_automatically_selected(self):
        for language in ("ms", "af"):
            with self.subTest(language=language):
                self.model.reset_mock()
                self.model.detect_language.return_value = (
                    language, 0.97,
                    [(language, 0.97), ("id", 0.02), ("en", 0.01)],
                )

                result = stt.transcribe_audio(b"wave", model=self.model)

                self.assertEqual(result["status"], "unsupported_language")
                self.assertIsNone(result["language"])
                self.assertEqual(result["language_detected"], language)
                self.model.transcribe.assert_not_called()

    def test_manual_legacy_language_bypasses_detection_without_claiming_confidence(self):
        self.model.transcribe.return_value = decoded_response("Halo.", "id")

        with patch.object(stt, "STT_LANGUAGE", "id"):
            result = stt.transcribe_audio(b"wave", model=self.model)

        self.model.detect_language.assert_not_called()
        self.decode_audio.assert_not_called()
        self.model.transcribe.assert_called_once()
        args, kwargs = self.model.transcribe.call_args
        self.assertIsInstance(args[0], io.BytesIO)
        self.assertEqual(args[0].getvalue(), b"wave")
        self.assertEqual(kwargs["language"], "id")
        self.assertEqual(result["language"], "id")
        self.assertIsNone(result["language_probability"])


if __name__ == "__main__":
    unittest.main()
