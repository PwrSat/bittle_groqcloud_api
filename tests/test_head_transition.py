"""Exercise head cleanup and BLE packet boundaries without real hardware.

The firmware buffers incoming BLE writes until its main loop reads them.
Two writes received before that read can become one command, even if each
write ends with a newline. A successful GATT write alone cannot detect this.
"""

import unittest
from unittest.mock import Mock, patch

import action_sequence
import motion_gen
from robot_link import BleRobot


class BufferedFirmware:
    """Minimal single-buffer receiver with a simulated 40 ms processing delay."""

    def __init__(self):
        self.now = 0.0
        self.pending = b""
        self.ready_at = 0.0
        self.commands = []
        self.pose = None

    async def write_gatt_char(self, characteristic, data, response):
        if not self.pending:
            self.ready_at = self.now + 0.04
        self.pending += data

    def sleep(self, seconds):
        self.now += seconds
        if self.pending and self.now >= self.ready_at:
            command = self.pending.rstrip(b"\n").decode("ascii")
            self.commands.append((self.now, command))
            if command in {"ksit", "kup"}:
                self.pose = command
            self.pending = b""

    async def async_sleep(self, seconds):
        self.sleep(seconds)


class HeadTransitionTests(unittest.TestCase):
    @staticmethod
    def finish_write(coro, timeout=10):
        # All awaits use the virtual clock; no socket/event loop is needed.
        try:
            coro.send(None)
        except StopIteration as done:
            return done.value
        finally:
            coro.close()
        raise AssertionError("test BLE writer unexpectedly waited for real I/O")

    def execute(self, actions, smooth=True):
        firmware = BufferedFirmware()
        # Use the real BLE send and packet writer, bypassing discovery/connection.
        robot = BleRobot.__new__(BleRobot)
        robot._client = Mock(is_connected=True)
        robot._client.write_gatt_char = firmware.write_gatt_char
        robot._run = self.finish_write
        with (
            patch.object(motion_gen, "SMOOTH", smooth),
            patch.object(motion_gen.time, "sleep", side_effect=firmware.sleep),
            patch("robot_link.asyncio.sleep", side_effect=firmware.async_sleep),
            patch("builtins.print"),
        ):
            action_sequence.execute_actions(robot, actions, {"ksit", "kup"})
        return firmware

    def test_stand_after_head_is_received_as_a_separate_command(self):
        actions = [
            {"type": "skill", "command": "ksit", "seconds": 2},
            {"type": "motion", "request": "tengok kanan", "steps": [
                {"joints": {0: 30}, "hold": 1.0},
            ]},
            {"type": "skill", "command": "kup", "seconds": 1},
        ]
        for smooth in (True, False):
            with self.subTest(smooth=smooth):
                firmware = self.execute(actions, smooth=smooth)
                commands = [command for _, command in firmware.commands]
                self.assertEqual(firmware.pose, "kup", commands)
                self.assertIn("i", commands, "head release must be received separately")
                self.assertEqual(commands[-1], "kup")
                self.assertFalse(any("\n" in command for command in commands), commands)

    def test_next_head_movement_is_not_merged_with_previous_release(self):
        actions = [
            {"type": "motion", "request": request, "steps": [
                {"joints": {0: angle}, "hold": 0.5},
            ]}
            for request, angle in (("tengok kanan", 30), ("tengok kiri", -30))
        ]
        firmware = self.execute(actions)
        commands = [command for _, command in firmware.commands]
        self.assertEqual(commands.count("i"), 2, commands)
        self.assertIn("i 0 -30", commands)
        self.assertIn("i 0 30", commands)
        self.assertFalse(any("\n" in command for command in commands), commands)

    def test_release_wait_counts_toward_sequence_limit_before_any_movement(self):
        actions = [
            {"type": "motion", "request": "tengok kanan", "steps": [
                {"joints": {0: 30}, "hold": 0.2},
            ]},
            {"type": "skill", "command": "kup", "seconds": 0.5},
        ]
        robot = Mock()
        with (
            patch.object(motion_gen, "SMOOTH", False),
            patch.object(action_sequence, "MAX_SEQUENCE_SECONDS", 1.0),
            self.assertRaises(action_sequence.ActionPlanError),
        ):
            action_sequence.execute_actions(robot, actions, {"kup"})
        self.assertEqual(robot.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
