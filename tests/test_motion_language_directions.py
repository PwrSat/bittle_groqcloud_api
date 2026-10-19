"""Head direction checks across the ten supported voice languages, offline."""

import contextlib
import copy
import io
import unittest
from unittest.mock import Mock, call, patch

import motion_gen


DIRECTION_EXAMPLES = (
    ("en", "Look right.", "Look left."),
    ("zh", "请向右看。", "请向左看。"),
    ("hi", "दाईं ओर देखो।", "बाईं ओर देखो।"),
    ("es", "Mira a la derecha.", "Mira a la izquierda."),
    ("ar", "انظر إلى اليمين.", "انظر إلى اليسار."),
    ("fr", "Regarde à droite.", "Regarde à gauche."),
    ("pt", "Olhe para a direita.", "Olhe para a esquerda."),
    ("id", "Tengok kanan.", "Tengok kiri."),
    ("ja", "右を見てください。", "左を見てください。"),
    ("ko", "오른쪽으로 고개를 돌려 주세요.", "왼쪽으로 고개를 돌려 주세요."),
)


def plan(angle):
    return [
        {"joints": {0: 0}, "hold": 0.2},
        {"joints": {0: angle}, "hold": 0.4},
        {"joints": {0: angle // 3}, "hold": 0.2},
        {"joints": {0: 0}, "hold": 0.2},
    ]


class MotionLanguageDirectionTests(unittest.TestCase):
    def test_wrong_right_direction_is_corrected_in_all_ten_languages(self):
        for language, right, _ in DIRECTION_EXAMPLES:
            with self.subTest(language=language):
                original = plan(-30)
                before = copy.deepcopy(original)

                fixed, warnings = motion_gen.enforce_direction(original, right)

                self.assertEqual([step["joints"][0] for step in fixed], [0, 30, 10, 0])
                self.assertEqual([step["hold"] for step in fixed], [0.2, 0.4, 0.2, 0.2])
                self.assertTrue(warnings)
                self.assertEqual(original, before)

    def test_wrong_left_direction_is_corrected_in_all_ten_languages(self):
        for language, _, left in DIRECTION_EXAMPLES:
            with self.subTest(language=language):
                fixed, warnings = motion_gen.enforce_direction(plan(30), left)

                self.assertEqual([step["joints"][0] for step in fixed], [0, -30, -10, 0])
                self.assertTrue(warnings)

    def test_already_correct_direction_is_unchanged_in_all_ten_languages(self):
        for language, right, left in DIRECTION_EXAMPLES:
            for request, angle in ((right, 30), (left, -30)):
                with self.subTest(language=language, request=request):
                    original = plan(angle)

                    fixed, warnings = motion_gen.enforce_direction(original, request)

                    self.assertIs(fixed, original)
                    self.assertEqual(warnings, [])

    def test_both_directions_do_not_override_planned_order(self):
        for language, right, left in DIRECTION_EXAMPLES:
            with self.subTest(language=language):
                original = plan(-30)

                fixed, warnings = motion_gen.enforce_direction(original, right + " " + left)

                self.assertIs(fixed, original)
                self.assertEqual(warnings, [])

    def test_arabic_vowel_marks_and_hindi_spelling_variants(self):
        examples = (
            ("اُنْظُرْ يَمِينًا", 30),
            ("اُنْظُرْ يَسَارًا", -30),
            ("दाहिने देखो।", 30),
            ("बाएँ देखो।", -30),
        )
        for request, expected in examples:
            with self.subTest(request=request):
                fixed, warnings = motion_gen.enforce_direction(plan(-expected), request)

                self.assertEqual(fixed[1]["joints"][0], expected)
                self.assertTrue(warnings)

    def test_latin_direction_substrings_and_unrelated_text_do_not_flip(self):
        for request in (
            "The room is bright.",
            "Tell me about copyright.",
            "I am feeling alright.",
            "Show a clever movement.",
            "Κοίτα δεξιά.",
            "Κοίτα αριστερά.",
            "Gelengkan kepala.",
            None,
        ):
            with self.subTest(request=request):
                original = plan(-30)

                fixed, warnings = motion_gen.enforce_direction(original, request)

                self.assertIs(fixed, original)
                self.assertEqual(warnings, [])

    def test_romanized_japanese_and_latin_punctuation_keep_working(self):
        for request, expected in (
            ("migi wo mite", 30),
            ("hidari wo mite", -30),
            ("(RIGHT)!", 30),
            ("'left', please", -30),
        ):
            with self.subTest(request=request):
                fixed, warnings = motion_gen.enforce_direction(plan(-expected), request)

                self.assertEqual(fixed[1]["joints"][0], expected)
                self.assertTrue(warnings)

    def test_missing_or_neutral_head_does_not_invent_a_turn(self):
        for original in ([], [{"joints": {0: 0}, "hold": 0.3}]):
            fixed, warnings = motion_gen.enforce_direction(original, "look right")

            self.assertIs(fixed, original)
            self.assertEqual(warnings, [])

    def test_corrected_plans_keep_hardware_sign_and_return_to_neutral(self):
        for language, right, left in DIRECTION_EXAMPLES:
            for request, expected in ((right, 30), (left, -30)):
                with self.subTest(language=language, request=request):
                    robot = Mock(spec=["send", "send_joints"])
                    fixed, _ = motion_gen.enforce_direction(plan(-expected), request)

                    with (
                        patch.object(motion_gen, "SMOOTH", False),
                        patch.object(motion_gen.time, "sleep"),
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        motion_gen.execute_plan(robot, fixed, dry_run=False)

                    self.assertEqual(
                        robot.send_joints.call_args_list,
                        [call({0: 0}), call({0: expected * motion_gen.JOINT_SIGN[0]}),
                         call({0: expected // 3 * motion_gen.JOINT_SIGN[0]}),
                         call({0: 0}), call({0: 0})],
                    )
                    self.assertEqual(robot.send_joints.call_args, call({0: motion_gen.NEUTRAL_ANGLE}))


if __name__ == "__main__":
    unittest.main()
