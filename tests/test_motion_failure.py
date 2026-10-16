"""Kegagalan gerakan kepala tidak boleh tertutup kegagalan cleanup."""

import unittest
from unittest.mock import Mock, patch

import motion_gen


class MotionFailureTests(unittest.TestCase):
    def test_cleanup_failure_preserves_the_original_send_error(self):
        robot = Mock()
        original = RuntimeError("pengiriman pertama gagal")
        robot.send_joints.side_effect = [original, RuntimeError("netral gagal")]
        with patch.object(motion_gen, "SMOOTH", False), patch("builtins.print"):
            with self.assertRaises(RuntimeError) as raised:
                motion_gen.execute_plan(robot, [{"joints": {0: 30}, "hold": 0.2}], dry_run=False)
        self.assertIs(raised.exception, original)
        self.assertEqual(robot.send_joints.call_count, 2)

    def test_cleanup_failure_is_reported_when_the_motion_succeeded(self):
        robot = Mock()
        cleanup = RuntimeError("netral gagal")
        robot.send_joints.side_effect = [None, cleanup]
        with (
            patch.object(motion_gen, "SMOOTH", False),
            patch.object(motion_gen.time, "sleep"),
            patch("builtins.print"),
        ):
            with self.assertRaises(RuntimeError) as raised:
                motion_gen.execute_plan(robot, [{"joints": {0: 30}, "hold": 0.2}], dry_run=False)
        self.assertIs(raised.exception, cleanup)


if __name__ == "__main__":
    unittest.main()
