"""Sequence parsing, preparation, and execution without services or hardware.

Run with: python -B -m unittest tests.test_action_sequence -v
"""

import json
import unittest
from unittest.mock import Mock, call, patch

import action_sequence as sequence


COMMANDS = {"kup", "ksit", "khi", "kwkF", "kbk", "kcrF", "kvtF", "kvtLX", "kphF"}


def skill(command="ksit", seconds=2.0):
    return {"type": "skill", "command": command, "seconds": seconds}


def motion(request="tengok kepala kanan"):
    return {"type": "motion", "request": request}


def head_steps(angle=35):
    return [{"joints": {0: angle}, "hold": 0.5}]


def prepared_motion(request="tengok kepala kanan", angle=35):
    return {**motion(request), "steps": head_steps(angle)}


class ReplyParsingTests(unittest.TestCase):
    def test_mixed_json_retains_order_and_repeated_skills(self):
        actions = [skill(), motion(), skill()]
        reply, parsed = sequence.parse_reply(
            json.dumps({"reply": "Aku duduk, menoleh, lalu duduk lagi.", "actions": actions}),
            COMMANDS,
        )
        self.assertEqual(reply, "Aku duduk, menoleh, lalu duduk lagi.")
        self.assertEqual(parsed, actions)

    def test_fenced_json_and_default_skill_duration(self):
        text = '```json\n{"reply":"Baik!","actions":[{"type":"skill","command":"ksit"}]}\n```'
        self.assertEqual(sequence.parse_reply(text, COMMANDS), ("Baik!", [skill()]))

    def test_plain_conversation_has_no_automatic_gesture(self):
        self.assertEqual(sequence.parse_reply("Sama-sama!", COMMANDS), ("Sama-sama!", []))

    def test_legacy_skill_reply_is_supported(self):
        self.assertEqual(
            sequence.parse_reply("Aku duduk. ##ksit##", COMMANDS),
            ("Aku duduk.", [skill()]),
        )

    def test_legacy_head_uses_original_request(self):
        self.assertEqual(
            sequence.parse_reply("Aku menoleh. ##motion##", COMMANDS, "tengok kiri"),
            ("Aku menoleh.", [motion("tengok kiri")]),
        )

    def test_multiple_legacy_tags_are_rejected_instead_of_taking_last(self):
        for text in ("Baik. ##ksit## ##kup##", "Baik. ##ksit## lalu ##motion##"):
            with self.subTest(text=text), self.assertRaises(sequence.ActionPlanError):
                sequence.parse_reply(text, COMMANDS, "duduk lalu tengok kanan")

    def test_malformed_or_incomplete_structured_reply_is_rejected(self):
        texts = [
            '{"reply": "Baik", "actions": [',
            '{"reply": "Baik"}',
            '{"reply": "", "actions": []}',
            '{"reply": true, "actions": []}',
            '{"reply": "Baik", "actions": "ksit"}',
            '{"reply": "Baik", "actions": [], "extra": "untrusted"}',
        ]
        for text in texts:
            with self.subTest(text=text), self.assertRaises(sequence.ActionPlanError):
                sequence.parse_reply(text, COMMANDS)


class ActionValidationTests(unittest.TestCase):
    def test_empty_actions_and_normalization(self):
        self.assertEqual(sequence.validate_actions([], COMMANDS), [])
        self.assertEqual(
            sequence.validate_actions([{"type": "skill", "command": "ksit"}], COMMANDS),
            [skill()],
        )

    def test_invalid_later_action_rejects_the_whole_plan(self):
        with self.assertRaises(sequence.ActionPlanError):
            sequence.validate_actions([skill(), skill("arbitrary serial text")], COMMANDS)

    def test_invalid_durations_cannot_reach_execution(self):
        for seconds in (True, False, float("nan"), float("inf"), -1, 0, 0.49, 10.01, "2", None):
            with self.subTest(seconds=seconds), self.assertRaises(sequence.ActionPlanError):
                sequence.validate_actions([skill(seconds=seconds)], COMMANDS)

    def test_rejects_unknown_fields_types_and_empty_motion_requests(self):
        actions = [
            {**skill(), "raw": "i 0 90"},
            {"type": "raw", "command": "i 0 90"},
            {"type": "skill"},
            motion(""),
            motion("   "),
            motion(True),
            {**motion(), "seconds": 1},
        ]
        for action in actions:
            with self.subTest(action=action), self.assertRaises(sequence.ActionPlanError):
                sequence.validate_actions([action], COMMANDS)

    def test_action_count_and_total_duration_are_bounded(self):
        for actions in ([skill()] * 9, [skill(seconds=10)] * 7):
            with self.subTest(actions=actions), self.assertRaises(sequence.ActionPlanError):
                sequence.validate_actions(actions, COMMANDS)


class ActionPreparationTests(unittest.TestCase):
    def test_each_head_request_is_prepared_separately_and_in_order(self):
        model = object()
        right, left = "tengok kepala kanan", "tengok kepala kiri"
        with patch.object(
            sequence.motion_gen, "plan_from_request",
            side_effect=[(head_steps(35), []), (head_steps(-35), ["arah diperbaiki"])],
        ) as planner:
            prepared, warnings = sequence.prepare_actions(
                model, [skill(), motion(right), motion(left), skill("kup")], COMMANDS
            )
        self.assertEqual(planner.call_args_list, [call(model, right), call(model, left)])
        self.assertEqual([action["type"] for action in prepared], ["skill", "motion", "motion", "skill"])
        self.assertEqual(prepared[1]["steps"], head_steps(35))
        self.assertEqual(prepared[2]["steps"], head_steps(-35))
        self.assertIn("arah diperbaiki", " ".join(warnings))

    def test_empty_head_plan_rejects_entire_sequence(self):
        with patch.object(sequence.motion_gen, "plan_from_request", return_value=([], ["invalid"])):
            with self.assertRaises(sequence.ActionPlanError):
                sequence.prepare_actions(object(), [skill(), motion(), skill("kup")], COMMANDS)

    def test_invalid_later_skill_is_rejected_before_any_head_planning(self):
        with patch.object(sequence.motion_gen, "plan_from_request") as planner:
            with self.assertRaises(sequence.ActionPlanError):
                sequence.prepare_actions(object(), [motion(), skill("bad")], COMMANDS)
            planner.assert_not_called()


class ActionExecutionTests(unittest.TestCase):
    def test_mixed_sequence_executes_in_order_and_preserves_final_sit(self):
        events = []
        robot = Mock()
        robot.send.side_effect = lambda command: events.append(("send", command))
        steps = head_steps()
        with (
            patch.object(sequence.time, "sleep", side_effect=lambda seconds: events.append(("wait", seconds))),
            patch.object(
                sequence.motion_gen, "execute_plan",
                side_effect=lambda robot, plan, **kwargs: events.append(("motion", plan)),
            ),
            patch("builtins.print"),
        ):
            sequence.execute_actions(robot, [skill("khi"), prepared_motion(), skill()], COMMANDS)
        self.assertEqual(events, [
            ("send", "khi"), ("wait", 2.0), ("motion", steps),
            ("send", "ksit"), ("wait", 2.0),
        ])

    def test_repeated_skills_are_all_executed(self):
        robot = Mock()
        with patch.object(sequence.time, "sleep"), patch("builtins.print"):
            sequence.execute_actions(robot, [skill("khi"), skill("khi")], COMMANDS)
        self.assertEqual(robot.send.call_args_list, [call("khi"), call("khi")])

    def test_each_continuous_gait_stops_after_its_timed_interval(self):
        for command in ("kwkF", "kbk", "kcrF", "kvtF", "kvtLX", "kphF"):
            with self.subTest(command=command):
                robot = Mock()
                with patch.object(sequence.time, "sleep") as sleep, patch("builtins.print"):
                    sequence.execute_actions(robot, [skill(command, 3)], COMMANDS)
                self.assertEqual(robot.send.call_args_list, [call(command), call("kup")])
                self.assertEqual(sleep.call_args_list, [call(3.0), call(0.5)])

    def test_gait_stops_and_settles_before_head_motion(self):
        events = []
        robot = Mock()
        robot.send.side_effect = lambda command: events.append(("send", command))
        with (
            patch.object(sequence.time, "sleep", side_effect=lambda seconds: events.append(("wait", seconds))),
            patch.object(sequence.motion_gen, "execute_plan", side_effect=lambda *args, **kwargs: events.append(("head",))),
            patch("builtins.print"),
        ):
            sequence.execute_actions(robot, [skill("kwkF", 3), prepared_motion()], COMMANDS)
        self.assertEqual(events, [
            ("send", "kwkF"), ("wait", 3.0), ("send", "kup"), ("wait", 0.5), ("head",),
        ])

    def test_sequence_duration_includes_gait_settling_and_head_interpolation(self):
        long_head = {**motion(), "steps": [
            {"joints": {0: angle}, "hold": 2.0}
            for angle in (60, -60, 60, -60, 60, -60)
        ]}
        # Requested gait intervals total exactly 60s; settling makes them exceed it.
        # Head holds total only 48s; movement and returning neutral exceed 60s.
        for actions in ([skill("kwkF", 10)] * 6, [long_head] * 4):
            with self.subTest(actions=actions):
                robot = Mock()
                with (
                    patch.object(sequence.time, "sleep") as sleep,
                    patch.object(sequence.motion_gen, "execute_plan") as execute_head,
                    self.assertRaises(sequence.ActionPlanError),
                ):
                    sequence.execute_actions(robot, actions, COMMANDS)
                self.assertEqual(robot.mock_calls, [])
                sleep.assert_not_called()
                execute_head.assert_not_called()

    def test_invalid_later_step_prevents_all_movement(self):
        invalid_actions = [
            [skill(), skill("not-in-command-list")],
            [skill(), motion()],  # not prepared; missing head steps
            [skill(), {**motion(), "steps": []}],
            [skill(), {**motion(), "steps": [{"joints": {8: 30}, "hold": 0.5}]}],
            [skill(), {**motion(), "steps": [{"joints": {0: float("nan")}, "hold": 0.5}]}],
        ]
        for actions in invalid_actions:
            with self.subTest(actions=actions):
                robot = Mock()
                with (
                    patch.object(sequence.time, "sleep") as sleep,
                    patch.object(sequence.motion_gen, "execute_plan") as execute_head,
                    self.assertRaises(sequence.ActionPlanError),
                ):
                    sequence.execute_actions(robot, actions, COMMANDS)
                robot.assert_not_called()
                robot.send.assert_not_called()
                robot.send_joints.assert_not_called()
                sleep.assert_not_called()
                execute_head.assert_not_called()

    def test_dry_run_does_not_move_or_wait(self):
        robot = Mock()
        with (
            patch.object(sequence.time, "sleep") as sleep,
            patch.object(sequence.motion_gen, "execute_plan") as execute_head,
            patch("builtins.print"),
        ):
            sequence.execute_actions(robot, [skill("kwkF"), prepared_motion(), skill()], COMMANDS, dry_run=True)
        robot.send.assert_not_called()
        robot.send_joints.assert_not_called()
        sleep.assert_not_called()
        for invocation in execute_head.call_args_list:
            self.assertTrue(invocation.kwargs.get("dry_run"))

    def test_send_failure_stops_remaining_actions_and_preserves_original_error(self):
        robot = Mock()
        error = RuntimeError("connection lost")
        robot.send.side_effect = [error, RuntimeError("cleanup also failed")]
        with patch.object(sequence.time, "sleep"), patch("builtins.print"):
            with self.assertRaises(RuntimeError) as caught:
                sequence.execute_actions(robot, [skill("khi"), skill()], COMMANDS)
        self.assertIs(caught.exception, error)
        self.assertEqual(robot.send.call_args_list, [call("khi"), call("kup")])

    def test_explicit_send_failure_status_aborts_remaining_actions(self):
        for failure_status in (False, -1):
            with self.subTest(status=failure_status):
                robot = Mock()
                robot.send.side_effect = [failure_status, None]
                with patch.object(sequence.time, "sleep") as sleep, patch("builtins.print"):
                    with self.assertRaises(RuntimeError):
                        sequence.execute_actions(robot, [skill("khi"), skill()], COMMANDS)
                self.assertEqual(robot.send.call_args_list, [call("khi"), call("kup")])
                sleep.assert_not_called()

    def test_head_failure_stops_remaining_actions(self):
        robot = Mock()
        error = RuntimeError("head send failed")
        with (
            patch.object(sequence.time, "sleep"),
            patch.object(sequence.motion_gen, "execute_plan", side_effect=error),
            patch("builtins.print"),
        ):
            with self.assertRaises(RuntimeError) as caught:
                sequence.execute_actions(robot, [prepared_motion(), skill()], COMMANDS)
        self.assertIs(caught.exception, error)
        robot.send.assert_called_once_with("kup")

    def test_keyboard_interrupt_stops_sequence_and_returns_to_standing(self):
        robot = Mock()
        with (
            patch.object(sequence.time, "sleep", side_effect=KeyboardInterrupt),
            patch("builtins.print"),
        ):
            with self.assertRaises(KeyboardInterrupt):
                sequence.execute_actions(robot, [skill("kwkF"), skill()], COMMANDS)
        self.assertEqual(robot.send.call_args_list, [call("kwkF"), call("kup")])

    def test_empty_conversational_sequence_never_touches_robot(self):
        robot = Mock()
        with patch.object(sequence.time, "sleep") as sleep:
            sequence.execute_actions(robot, [], COMMANDS)
        self.assertEqual(robot.mock_calls, [])
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
