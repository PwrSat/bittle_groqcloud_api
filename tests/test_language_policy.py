"""Automatic supported-language policy, without models, microphones or APIs."""

import math
import unittest

from language_policy import (
    SUPPORTED_LANGUAGES, build_language_instruction, reconcile_text_language,
    select_language,
)


class SelectLanguageTests(unittest.TestCase):
    def test_exact_ten_languages_in_requested_order_without_previous_state(self):
        self.assertEqual(list(SUPPORTED_LANGUAGES), ["en", "zh", "hi", "es", "ar", "fr", "pt", "id", "ja", "ko"])
        self.assertEqual(SUPPORTED_LANGUAGES["ar"], "Arab Standar Modern")
        self.assertEqual(SUPPORTED_LANGUAGES["fr"], "Prancis")
        self.assertEqual(SUPPORTED_LANGUAGES["pt"], "Portugis")
        self.assertEqual(SUPPORTED_LANGUAGES["ko"], "Korea")
        for language in SUPPORTED_LANGUAGES:
            with self.subTest(language=language):
                result = select_language(language, 0.91, [(language, 0.91), ("it", 0.09)])
                self.assertEqual(result["language"], language)
                self.assertEqual(result["probability"], 0.91)

    def test_best_supported_acoustic_candidate_keeps_global_probability(self):
        result = select_language("th", 0.63, [("th", 0.63), ("id", 0.24), ("ko", 0.08), ("en", 0.05)])
        self.assertEqual(result["language"], "id")
        self.assertEqual(result["probability"], 0.24)
        self.assertAlmostEqual(result["supported_mass"], 0.37)
        self.assertEqual(result["detected_language"], "th")
        self.assertEqual(result["detected_probability"], 0.63)
        self.assertEqual(result["reason"], "supported_candidate")
        self.assertEqual(result["alternatives"], [
            {"language": "ko", "probability": 0.08},
            {"language": "en", "probability": 0.05},
        ])

    def test_confident_unsupported_speech_is_not_forced_to_supported_language(self):
        for language in ("ms", "af", "th", "it", "fa", "mr"):
            with self.subTest(language=language):
                result = select_language(language, 0.9, [(language, 0.9), ("en", 0.06), ("id", 0.04)])
                self.assertIsNone(result["language"])
                self.assertIsNone(result["probability"])
                self.assertEqual(result["reason"], "unsupported_confident")

    def test_unsupported_boundary_is_inclusive(self):
        result = select_language("it", 0.8, [("it", 0.8), ("en", 0.1), ("id", 0.1)])
        self.assertIsNone(result["language"])

    def test_missing_or_malformed_distribution_uses_only_original_supported_language(self):
        for distribution in (None, "bad", [], [None, ["id"], ["id", float("nan")]]):
            with self.subTest(distribution=distribution):
                self.assertEqual(select_language("ja", 0.7, distribution)["language"], "ja")
                self.assertIsNone(select_language("it", 0.7, distribution)["language"])

    def test_invalid_probabilities_and_duplicates_do_not_inflate_mass(self):
        result = select_language("th", 0.4, [
            ("id", 0.2), ("id", 0.3), ("fr", 0.1), ("en", math.nan),
            ("ja", math.inf), ("es", -1), ("hi", 2), ("pt", True),
            ("ar", "0.9"), None, ([], 0.8),
        ])
        self.assertEqual(result["language"], "id")
        self.assertEqual(result["probability"], 0.3)
        self.assertAlmostEqual(result["supported_mass"], 0.4)

    def test_mapping_and_normalized_codes_are_accepted(self):
        result = select_language(" ID ", 0.7, {" ID ": 0.7, "fr": 0.2})
        self.assertEqual(result["language"], "id")
        self.assertAlmostEqual(result["supported_mass"], 0.9)

    def test_impossible_mass_is_not_reported_as_probability(self):
        result = select_language("en", 0.8, [("en", 0.8), ("id", 0.7)])
        self.assertIsNone(result["supported_mass"])

    def test_missing_detection_and_all_data_are_safe(self):
        result = select_language(None, None, None)
        self.assertIsNone(result["language"])
        self.assertIsNone(result["probability"])
        self.assertIsNone(result["supported_mass"])

    def test_missing_distribution_does_not_invent_total_supported_mass(self):
        result = select_language("en", 0.7, None)
        self.assertEqual(result["probability"], 0.7)
        self.assertIsNone(result["supported_mass"])

    def test_missing_detector_score_can_be_recovered_from_actual_distribution(self):
        result = select_language("it", None, {"it": 0.95, "en": 0.05})
        self.assertEqual(result["detected_probability"], 0.95)
        self.assertIsNone(result["language"])

    def test_removed_languages_are_not_candidates_or_part_of_supported_mass(self):
        result = select_language("ms", 0.6, {"ms": 0.6, "af": 0.15, "id": 0.2, "en": 0.05})
        self.assertEqual(result["language"], "id")
        self.assertEqual(result["probability"], 0.2)
        self.assertAlmostEqual(result["supported_mass"], 0.25)
        self.assertEqual(result["alternatives"], [{"language": "en", "probability": 0.05}])

    def test_removed_languages_with_no_supported_candidates_are_rejected(self):
        for language in ("ms", "af"):
            with self.subTest(language=language):
                result = select_language(language, 0.7, {language: 0.7})
                self.assertIsNone(result["language"])
                self.assertEqual(result["reason"], "no_supported_candidate")


class ReconcileTextLanguageTests(unittest.TestCase):
    def test_exact_phrase_evidence_covers_each_supported_language(self):
        phrases = {
            "en": "Thank you!", "ja": "右を見て。", "es": "Siéntate.",
            "ar": "اجلس", "zh": "向右看", "id": "mundur", "hi": "बैठो",
            "fr": "Merci!", "pt": "Obrigada.", "ko": "고마워!",
        }
        for language, text in phrases.items():
            with self.subTest(language=language):
                original = "ja" if language == "en" else "en"
                result = reconcile_text_language(text, original, 0.2, [(original, 0.2), (language, 0.15)])
                self.assertEqual(result, (language, 0.15, "phrase_evidence"))

    def test_duduk_repairs_label_without_using_removed_malay_candidate(self):
        result = reconcile_text_language("duduk", "en", 0.61, [("id", 0.08), ("ms", 0.2)])
        self.assertEqual(result, ("id", 0.08, "phrase_evidence"))

    def test_correct_selected_phrase_preserves_its_actual_probability(self):
        result = reconcile_text_language("Tengok kanan.", "id", 0.08, [("id", 0.08), ("en", 0.2)])
        self.assertEqual(result, ("id", 0.08, "unchanged"))

    def test_removed_language_phrase_does_not_change_supported_label(self):
        for text in ("dankie", "kyk regs", "staan op"):
            with self.subTest(text=text):
                self.assertEqual(reconcile_text_language(text, "en", 0.3, {"af": 0.6}), ("en", 0.3, "unchanged"))

    def test_english_sit_uses_english_hint_even_with_removed_afrikaans_score(self):
        self.assertEqual(reconcile_text_language("sit", "id", 0.3, {"en": 0.1, "af": 0.6}), ("en", 0.1, "phrase_evidence"))

    def test_new_language_command_and_thanks_hints_keep_raw_scores(self):
        phrases = {
            "fr": ("assieds-toi", "levez-vous", "regarde à droite", "regarde à gauche", "merci"),
            "pt": ("sente-se", "levante-se", "olhe para direita", "olhe para esquerda", "obrigado", "obrigada"),
            "ko": ("앉아", "일어서", "오른쪽을 봐", "왼쪽을 봐", "고마워"),
        }
        for language, examples in phrases.items():
            for text in examples:
                with self.subTest(language=language, text=text):
                    result = reconcile_text_language(text, "id", 0.2, {language: 0.07, "id": 0.2})
                    self.assertEqual(result, (language, 0.07, "phrase_evidence"))

    def test_confident_supported_foreign_language_is_preserved(self):
        for selected in SUPPORTED_LANGUAGES:
            with self.subTest(selected=selected):
                self.assertEqual(reconcile_text_language("duduk", selected, 0.9, {"id": 0.05}), (selected, 0.9, "confident_language"))

    def test_rejected_and_unsupported_languages_are_not_revived_by_lexicon(self):
        self.assertEqual(reconcile_text_language("duduk", None, None, {"id": 0.05}), (None, None, "unchanged"))
        self.assertEqual(reconcile_text_language("duduk", "it", 0.9, {"id": 0.05}), ("it", 0.9, "unchanged"))

    def test_negation_counts_sentences_and_substrings_are_unchanged(self):
        for text in ("jangan duduk", "do not sit", "no te sientes", "ne regarde pas à droite", "não olhe para direita", "앉아 있지 마", "duduk dua kali", "duduk selama 3 detik", "aku tadi duduk di kursi", "duduk, lalu berdiri", "produk", "duduklah"):
            with self.subTest(text=text):
                self.assertEqual(reconcile_text_language(text, "en", 0.3, {"id": 0.2}), ("en", 0.3, "unchanged"))

    def test_scripts_alone_do_not_override_language(self):
        for text in ("東京", "山", "مرحبا بكم", "भारत", "stop"):
            with self.subTest(text=text):
                self.assertEqual(reconcile_text_language(text, "en", 0.3, {"ja": 0.2, "zh": 0.1}), ("en", 0.3, "unchanged"))

    def test_missing_language_probability_is_not_fabricated(self):
        self.assertEqual(reconcile_text_language("thank you", "es", 0.2, None), ("en", None, "phrase_evidence"))

    def test_rapid_language_switching_has_no_history_bias(self):
        for phrase, language in (("gracias", "es"), ("ありがとう", "ja"), ("thank you", "en"), ("merci", "fr"), ("obrigado", "pt"), ("고마워", "ko")):
            with self.subTest(language=language):
                result = reconcile_text_language(phrase, "id", 0.3, {language: 0.1})
                self.assertEqual(result[0], language)


class LanguageInstructionTests(unittest.TestCase):
    def test_prompt_names_all_ten_supported_codes_and_language_names(self):
        instruction = build_language_instruction()
        for code, name in SUPPORTED_LANGUAGES.items():
            with self.subTest(language=code):
                self.assertIn(f"{code}={name}", instruction)
        self.assertNotIn("ms=", instruction)
        self.assertNotIn("af=", instruction)

    def test_prompt_prioritizes_current_tag_and_preserves_input_semantics(self):
        instruction = build_language_instruction()
        for text in ("pesan terbaru", "bahasa riwayat", "teks input asli", "menerjemahkannya", "Jangan terjemahkan kode command", "larangan", "durasi", "urutan aksi"):
            with self.subTest(text=text):
                self.assertIn(text, instruction)

    def test_prompt_specifies_standard_arabic_and_relevant_writing_systems(self):
        instruction = build_language_instruction()
        for text in ("Arab Standar Modern (Fusha)", "Mandarin", "Tionghoa sederhana", "Devanagari", "bahasa Jepang", "Hangul"):
            with self.subTest(text=text):
                self.assertIn(text, instruction)


if __name__ == "__main__":
    unittest.main()
