"""Guard protocol and behavioral constraints when shortening LLM instructions."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import action_sequence as sequence
import language_policy


class CompactPromptTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.commands = json.loads((root / "Commands.json").read_text(encoding="utf-8"))
        self.prompt = sequence.build_system_prompt(self.commands)

    def test_all_aliases_remain_in_one_lossless_command_map(self):
        payloads = [json.loads(line) for line in self.prompt.splitlines() if line.startswith("{")]
        self.assertEqual(payloads[0], self.commands)
        self.assertEqual(len(payloads[0]), len(self.commands))
        for alias in ("stand up", "stop", "berhenti", "diam"):
            self.assertEqual(payloads[0][alias], "kup")

    def test_mixed_example_passes_the_real_parser_and_retains_schema(self):
        example = next(line for line in reversed(self.prompt.splitlines()) if line.startswith("{"))
        reply, actions = sequence.parse_reply(example, set(self.commands.values()))
        self.assertTrue(reply)
        self.assertEqual([action["type"] for action in actions], ["skill", "motion", "skill"])
        self.assertEqual(actions[0], {"type": "skill", "command": "ksit", "seconds": 2})
        self.assertEqual(actions[1], {"type": "motion", "request": "tengok kepala ke kanan"})
        self.assertEqual(actions[2], {"type": "skill", "command": "kup", "seconds": 1})

    def test_limits_remain_dynamic_and_describe_post_send_waiting(self):
        with patch.multiple(sequence, MAX_ACTIONS=5, MAX_SEQUENCE_SECONDS=42.0,
                            SKILL_SECONDS_RANGE=(0.75, 7.0), DEFAULT_SKILL_SECONDS=1.5):
            prompt = sequence.build_system_prompt(self.commands)
        self.assertIn("Maksimal 5 aksi", prompt)
        self.assertIn("total maksimal 42 detik", prompt)
        self.assertIn("seconds adalah jeda setelah skill dikirim, antara 0.75 dan 7.0 detik; default 1.5", prompt)
        self.assertIn("Ikuti durasi yang diminta pengguna jika dalam batas", prompt)
        self.assertIn("misalnya 4 detik", prompt)
        self.assertIn("Berjalan/merangkak/mundur/berputar akan dihentikan otomatis sesudah jedanya", prompt)

    def test_sequence_negation_and_complete_rejection_are_explicit(self):
        for rule in ("SEMUA aksi sesuai urutannya", "Boleh mencampur skill dan motion",
                     "mengulang aksi dengan menuliskannya lagi", "Jangan menambah gerakan yang tidak diminta",
                     "Patuhi negasi seperti 'jangan duduk'", "gerakan yang tidak didukung",
                     "rangkaian terlalu panjang", "jelaskan singkat dalam reply dan gunakan actions: []",
                     "berhenti/diam/stop berarti skill kup, bukan mundur", "letakkan kup pada urutan itu"):
            with self.subTest(rule=rule):
                self.assertIn(rule, self.prompt)

    def test_head_scope_and_chat_capabilities_remain_explicit(self):
        for rule in ("Request hanya potongan permintaan gerakan kepala itu, dalam bahasa pengguna",
                     "bukan seluruh rangkaian", "Hanya menoleh kiri-kanan/geleng",
                     "tidak mengangguk atau merancang gerakan kaki baru",
                     "Kepala kembali netral setelah tiap aksi motion",
                     "Reply singkat (5-12 kata), kecuali diminta bercerita",
                     "actions boleh kosong", "sapaan boleh satu skill khi",
                     "HANYA satu objek JSON valid", "Jangan gunakan markdown atau format ##command##"):
            with self.subTest(rule=rule):
                self.assertIn(rule, self.prompt)

    def test_original_robot_role_skill_mapping_and_prose_categories_are_preserved(self):
        for rule in ("Kamu adalah Bittle, robot anjing peliharaan yang ramah",
                     "Pengguna berbicara lewat suara",
                     "Terjemahkan MAKSUD ucapan menjadi jawaban dan aksi yang berurutan",
                     "AKSI BAWAAN:", "Command harus salah satu KODE dalam daftar berikut, jangan diterjemahkan",
                     "AKSI KEPALA:", "URUTAN:", "DURASI:", "STOP:",
                     "CONTOH 'duduk, tengok kanan, lalu berdiri':"):
            with self.subTest(rule=rule):
                self.assertIn(rule, self.prompt)

    def test_custom_head_request_keeps_modifiers_instead_of_becoming_a_generic_alias(self):
        for rule in ("Salin utuh potongan ucapan itu, termasuk sudut, lama tahan",
                     "kecepatan, jumlah pengulangan, dan urutan internal"):
            with self.subTest(rule=rule):
                self.assertIn(rule, self.prompt)

    def test_language_policy_keeps_recent_tag_priority_native_scripts_and_meaning(self):
        policy = language_policy.build_language_instruction()
        for code, name in language_policy.SUPPORTED_LANGUAGES.items():
            self.assertIn(f"{code}={name}", policy)
        for rule in ("Tag [bahasa: ...] pada pesan terbaru dari pengguna menentukan bahasa reply",
                     "dahulukan tag tersebut daripada bahasa riwayat", "jangan tampilkan tag itu",
                     "Pahami teks input asli tanpa mengganti atau menerjemahkannya",
                     "Arab Standar Modern (Fusha)", "Mandarin dengan aksara Tionghoa sederhana",
                     "hi dengan aksara Devanagari", "ja dalam bahasa Jepang", "ko dengan Hangul",
                     "Jangan terjemahkan kode command", "larangan, jumlah, durasi, dan urutan aksi"):
            with self.subTest(rule=rule):
                self.assertIn(rule, policy)


if __name__ == "__main__":
    unittest.main()
