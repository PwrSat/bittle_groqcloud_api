"""Common head requests avoid a second model call without changing fallback."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import head_templates
import llm_utils
import motion_gen


# Independent fixtures: right, left, explicit horizontal shake for each language.
COMMON_REQUESTS = {
    "en": ("look right", "look left", "shake your head"),
    "zh": ("向右看", "向左看", "摇头"),
    "hi": ("दाईं ओर देखो", "बाईं ओर देखो", "सिर को दाएं-बाएं हिलाओ"),
    "es": ("mira a la derecha", "mira a la izquierda", "mueve la cabeza de lado a lado"),
    "ar": ("انظر إلى اليمين", "انظر إلى اليسار", "هز رأسك من جانب إلى آخر"),
    "fr": ("regarde à droite", "regarde à gauche", "secoue la tête de gauche à droite"),
    "pt": ("olhe para a direita", "olhe para a esquerda", "balance a cabeça de um lado para o outro"),
    "id": ("tengok kepala ke kanan", "tengok kepala ke kiri", "gelengkan kepala"),
    "ja": ("右を見て", "左を見て", "首を横に振って"),
    "ko": ("오른쪽을 봐", "왼쪽을 봐", "고개를 좌우로 흔들어"),
}

CUSTOM_REQUESTS = (
    "do not look right", "look right for three seconds", "look right 30 degrees",
    "slowly look right", "look right twice", "look right then left",
    "if I say go, look right", "can you look right?", "look right...",
    "不要向右看", "向右看三秒钟", "慢慢向右看", "向右看两次", "向右看，然后向左看",
    "दाईं ओर मत देखो", "दाईं ओर तीन सेकंड देखो", "धीरे से दाईं ओर देखो", "दाईं ओर दो बार देखो",
    "no mires a la derecha", "mira a la derecha durante tres segundos", "mira a la derecha lentamente",
    "لا تنظر إلى اليمين", "انظر إلى اليمين لمدة ثلاث ثوان", "انظر إلى اليمين ببطء",
    "ne regarde pas à droite", "regarde à droite pendant trois secondes", "regarde à droite lentement",
    "não olhe para a direita", "olhe para a direita por três segundos", "olhe lentamente para a direita",
    "jangan tengok kanan", "tengok kanan selama tiga detik", "tengok kanan pelan-pelan",
    "tengok kanan 45 derajat", "tengok kanan dua kali", "tengok kanan lalu duduk",
    "bila kupanggil tengok kanan", "tengok kanan?", "tengok kanan...",
    "右を見ないで", "右を三秒間見て", "ゆっくり右を見て",
    "오른쪽을 보지 마", "오른쪽을 삼 초 동안 봐", "천천히 오른쪽을 봐",
    "nod your head", "look up", "gerakkan kepala", "सिर हिलाओ", "هز رأسك",
    "balance a cabeça", "guarda a destra", "copyright", '"look right"', "look-right",
)


def fake_model():
    response = SimpleNamespace(
        content=json.dumps({"steps": [{"joints": {"0": 10}, "hold": 0.25}]}),
        additional_kwargs={},
        response_metadata={"token_usage": {"prompt_tokens": 50, "completion_tokens": 10}},
    )
    return Mock(invoke=Mock(return_value=response))


class HeadTemplateTests(unittest.TestCase):
    def test_import_is_lightweight_and_never_imports_model_or_hardware_modules(self):
        code = (
            "import sys; import head_templates; "
            "assert not any(n.startswith(('motion_gen', 'llm_utils', 'langchain', "
            "'faster_whisper', 'ctranslate2', 'robot_link', 'bleak', 'serial')) "
            "for n in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-S", "-c", code],
            cwd=Path(head_templates.__file__).resolve().parent,
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_simple_requests_in_ten_languages_preserve_direction_and_head_only_limits(self):
        expected_angles = ((30,), (-30,), (-40, 40, -40, 40))
        for language, requests in COMMON_REQUESTS.items():
            for request, expected in zip(requests, expected_angles):
                with self.subTest(language=language, request=request):
                    plan = head_templates.plan_for_common_request(request)
                    self.assertIsNotNone(plan)
                    self.assertEqual(tuple(step["joints"][0] for step in plan["steps"]), expected)
                    self.assertTrue(all(set(step["joints"]) == {0} for step in plan["steps"]))
                    clean, warnings = motion_gen.validate_plan(plan)
                    self.assertEqual(warnings, [])
                    self.assertEqual(clean, plan["steps"])

    def test_case_spacing_and_one_sentence_terminator_do_not_change_the_request(self):
        for request in ("  LOOK   RIGHT! ", "tengok kanan.", "右を見て。", "向右看！"):
            with self.subTest(request=request):
                self.assertEqual(head_templates.plan_for_common_request(request)["steps"][0]["joints"], {0: 30})

    def test_unknown_or_modified_requests_never_match_a_command_fragment(self):
        for request in CUSTOM_REQUESTS + ("", None, 123, True, {}, []):
            with self.subTest(request=request):
                self.assertIsNone(head_templates.plan_for_common_request(request))

    def test_each_return_is_fresh_even_when_the_previous_plan_was_mutated(self):
        first = head_templates.plan_for_common_request("geleng kepala")
        first["steps"][0]["joints"][0] = 60
        first["steps"].pop()
        second = head_templates.plan_for_common_request("geleng kepala")
        self.assertEqual([step["joints"][0] for step in second["steps"]], [-40, 40, -40, 40])

    def test_observed_llm_head_wording_is_exact_and_does_not_absorb_modifiers(self):
        for direction, angle in (("right", 30), ("left", -30)):
            request = f"look head to the {direction}"
            with self.subTest(request=request):
                self.assertEqual(
                    head_templates.plan_for_common_request(request)["steps"],
                    [{"joints": {0: angle}, "hold": 1.0}],
                )
            for suffix in (" twice", " for 3 seconds", " slowly", " then sit"):
                with self.subTest(request=request + suffix):
                    self.assertIsNone(head_templates.plan_for_common_request(request + suffix))


class HeadPlannerIntegrationTests(unittest.TestCase):
    def setUp(self):
        counter_patch = patch.object(llm_utils, "token_counter", llm_utils.TokenCounter())
        counter_patch.start()
        self.addCleanup(counter_patch.stop)

    def test_thirty_simple_requests_save_all_thirty_additional_model_calls(self):
        model = Mock(invoke=Mock(side_effect=AssertionError("common request must remain local")))
        counter = llm_utils.TokenCounter()
        with patch.object(llm_utils, "token_counter", counter):
            for language, requests in COMMON_REQUESTS.items():
                for request in requests:
                    with self.subTest(language=language, request=request):
                        steps, warnings = motion_gen.plan_from_request(model, request)
                        self.assertTrue(steps)
                        self.assertEqual(warnings, [])
        model.invoke.assert_not_called()
        self.assertEqual(counter.summary()["total_tokens"], 0)
        self.assertEqual(counter.calls, 0)

    def test_every_custom_request_reaches_original_llm_unchanged_and_is_counted(self):
        counter = llm_utils.TokenCounter()
        with patch.object(llm_utils, "token_counter", counter):
            for request in CUSTOM_REQUESTS:
                with self.subTest(request=request):
                    model = fake_model()
                    steps, _ = motion_gen.plan_from_request(model, request)
                    self.assertTrue(steps)
                    model.invoke.assert_called_once()
                    messages = model.invoke.call_args.args[0]
                    self.assertEqual(messages[0].content, motion_gen.build_motion_prompt())
                    self.assertEqual(messages[-1].content, request)
        self.assertEqual(counter.calls, len(CUSTOM_REQUESTS))
        self.assertEqual({row["source"] for row in counter.records_since()}, {"head_planner"})

    def test_local_path_still_runs_both_existing_validation_and_direction_guard(self):
        with (
            patch.object(motion_gen, "validate_plan", wraps=motion_gen.validate_plan) as validate,
            patch.object(motion_gen, "enforce_direction", wraps=motion_gen.enforce_direction) as direction,
        ):
            steps, warnings = motion_gen.plan_from_request(Mock(), "tengok kanan")
        self.assertEqual(steps, [{"joints": {0: 30}, "hold": 1.0}])
        self.assertEqual(warnings, [])
        validate.assert_called_once()
        direction.assert_called_once()

    def test_changed_angle_limit_uses_llm_instead_of_silently_clipping_template(self):
        model = fake_model()
        with patch.object(motion_gen, "JOINT_LIMITS", {0: (-15, 15)}):
            steps, warnings = motion_gen.plan_from_request(model, "look right")
        self.assertEqual(steps, [{"joints": {0: 10}, "hold": 0.25}])
        self.assertEqual(warnings, [])
        model.invoke.assert_called_once()

    def test_changed_hold_limit_uses_llm_even_if_validator_clamps_without_warning(self):
        model = fake_model()
        with patch.object(motion_gen, "HOLD_RANGE", (0.2, 0.5)):
            steps, warnings = motion_gen.plan_from_request(model, "look right")
        self.assertEqual(steps, [{"joints": {0: 10}, "hold": 0.25}])
        self.assertEqual(warnings, [])
        model.invoke.assert_called_once()

    def test_changed_step_limit_does_not_silently_truncate_horizontal_shake(self):
        model = fake_model()
        with patch.object(motion_gen, "MAX_STEPS", 2):
            steps, warnings = motion_gen.plan_from_request(model, "shake your head")
        self.assertEqual(steps, [{"joints": {0: 10}, "hold": 0.25}])
        self.assertEqual(warnings, [])
        model.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
