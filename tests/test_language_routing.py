"""Language policy reaches action planning and speech without external services."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

from gtts.lang import tts_langs

import action_sequence
import language_policy
import llm_utils
import speechtotext
import Text2Speech


TARGET_LANGUAGES = {
    "en": "Inggris",
    "zh": "Mandarin",
    "hi": "Hindi",
    "es": "Spanyol",
    "ar": "Arab standar",
    "fr": "Perancis",
    "pt": "Portugis",
    "id": "Indonesia",
    "ja": "Jepang",
    "ko": "Korea",
}


class LanguageRoutingTests(unittest.TestCase):
    def test_voice_policy_has_exactly_the_ten_requested_codes(self):
        self.assertEqual(set(language_policy.SUPPORTED_LANGUAGES), set(TARGET_LANGUAGES))

    def test_action_prompt_uses_the_shared_language_instruction(self):
        with patch.object(
            action_sequence, "build_language_instruction",
            return_value="SHARED_LANGUAGE_POLICY_SENTINEL",
        ) as instruction:
            prompt = action_sequence.build_system_prompt({"ksit": "duduk", "kup": "berdiri"})

        instruction.assert_called_once_with()
        self.assertIn("SHARED_LANGUAGE_POLICY_SENTINEL", prompt)

    def test_shared_instruction_lists_every_code_and_language(self):
        instruction = language_policy.build_language_instruction().casefold()

        for code, name in TARGET_LANGUAGES.items():
            with self.subTest(language=code):
                self.assertIn(code, instruction)
                # Both Indonesian spellings of French are widely used.
                if code == "fr":
                    self.assertTrue("perancis" in instruction or "prancis" in instruction)
                else:
                    self.assertIn(name.casefold(), instruction)
        self.assertNotIn("afrikaans", instruction)
        self.assertNotIn("melayu", instruction)

    def test_latest_utterance_language_takes_priority_over_history(self):
        instruction = language_policy.build_language_instruction().casefold()

        self.assertIn("[bahasa:", instruction)
        self.assertIn("terbaru", instruction)
        self.assertIn("riwayat", instruction)

    def test_language_instruction_preserves_json_action_duration_and_negation_rules(self):
        prompt = action_sequence.build_system_prompt({"ksit": "duduk", "kup": "berdiri"})

        self.assertIn("HANYA satu objek JSON", prompt)
        self.assertIn("reply", prompt)
        self.assertIn("actions", prompt)
        self.assertIn('"command": "ksit"', prompt)
        self.assertIn("seconds", prompt)
        self.assertIn("DURASI:", prompt)
        self.assertIn("jangan duduk", prompt)
        self.assertIn("SEMUA aksi sesuai urutannya", prompt)

    def test_all_ten_languages_have_a_supported_local_tts_mapping(self):
        available = tts_langs()

        for code in TARGET_LANGUAGES:
            with self.subTest(language=code):
                expected = "zh-CN" if code == "zh" else code
                resolved = Text2Speech._resolve_gtts_lang(code)
                self.assertEqual(resolved, expected)
                self.assertIn(resolved, available)

    def test_low_confidence_duduk_correction_cannot_restore_removed_language_tags(self):
        for language in ("ms", "af"):
            with self.subTest(language=language):
                text, returned_language, _ = speechtotext.normalize_transcript(
                    "Duh Duk", language, 0.2,
                )
                self.assertEqual(text, "duduk")
                self.assertEqual(returned_language, "id")

    def test_groq_model_uses_low_variation_settings_without_changing_model(self):
        factory = Mock(return_value=sentinel.groq_model)
        module = SimpleNamespace(ChatGroq=factory)

        with patch.dict("sys.modules", {"langchain_groq": module}):
            model = llm_utils.get_model("groq")

        factory.assert_called_once_with(model="openai/gpt-oss-120b", temperature=0)
        self.assertIs(model, sentinel.groq_model)


if __name__ == "__main__":
    unittest.main()
