"""Tes lapisan pengiriman tanpa Bluetooth, serial, atau robot fisik."""
import unittest
from concurrent.futures import TimeoutError as FutureTimeoutError
from unittest.mock import Mock, patch

from robot_link import BleRobot, SerialRobot


class SerialRobotSendTests(unittest.TestCase):
    def setUp(self):
        # Hindari constructor: ardSerial dapat membuka port dan membuat log.
        self.robot = SerialRobot.__new__(SerialRobot)
        self.robot._ard = Mock()
        self.robot._ports = {"port": "Bittle"}
        self.robot._ard.send.return_value = ["k\r\n", ""]

    def test_skill_uses_bounded_timeout_and_no_scheduler_delay(self):
        self.assertTrue(self.robot.send("ksit"))
        self.robot._ard.send.assert_called_once_with(
            self.robot._ports, ["ksit", 0], timeout=10
        )

    def test_skill_failure_raises(self):
        self.robot._ard.send.return_value = -1
        with self.assertRaisesRegex(RuntimeError, "ksit"):
            self.robot.send("ksit")

    def test_transport_exception_propagates(self):
        self.robot._ard.send.side_effect = TimeoutError("serial timeout")
        with self.assertRaisesRegex(TimeoutError, "serial timeout"):
            self.robot.send("kup")

    def test_joint_command_uses_bounded_timeout(self):
        self.assertTrue(self.robot.send_joints({0: 40, 1: -20}))
        self.robot._ard.send.assert_called_once_with(
            self.robot._ports, ["i", [0, 40, 1, -20], 0], timeout=5
        )

    def test_joint_failure_raises(self):
        self.robot._ard.send.return_value = -1
        with self.assertRaisesRegex(RuntimeError, "sudut motor"):
            self.robot.send_joints({0: 40})


class BleRobotSendTests(unittest.TestCase):
    def setUp(self):
        self.robot = BleRobot.__new__(BleRobot)
        self.robot._client = Mock(is_connected=True)
        self.robot._write = Mock(side_effect=self.make_write_coroutine)
        self.robot._connect = Mock(side_effect=self.make_connect_coroutine)
        self.robot._run = Mock(side_effect=self.close_coroutine)

    @staticmethod
    async def make_write_coroutine(data):
        return None

    @staticmethod
    async def make_connect_coroutine():
        return None

    @staticmethod
    def close_coroutine(coro, timeout=10):
        # _run normally executes the coroutine. Close it in this hardware-free
        # fake so a test failure cannot leave an unawaited coroutine warning.
        coro.close()

    def test_success_returns_true_after_write(self):
        self.assertTrue(self.robot.send("ksit"))
        self.robot._write.assert_called_once_with(b"ksit\n")
        self.robot._run.assert_called_once()
        self.assertEqual(self.robot._run.call_args.kwargs, {"timeout": 5})

    def test_write_failure_raises_with_command_and_original_cause(self):
        error = TimeoutError("BLE timeout")

        def fail(coro, timeout=10):
            coro.close()
            raise error

        self.robot._run.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, "ksit") as caught:
            self.robot.send("ksit")
        self.assertIs(caught.exception.__cause__, error)

    def test_disconnected_client_reconnects_before_write(self):
        self.robot._client.is_connected = False
        self.assertTrue(self.robot.send("kup"))
        self.robot._connect.assert_called_once_with()
        self.robot._write.assert_called_once_with(b"kup\n")
        self.assertEqual(
            [call.kwargs for call in self.robot._run.call_args_list],
            [{"timeout": 30}, {"timeout": 5}],
        )

    def test_failed_reconnect_never_attempts_write(self):
        self.robot._client.is_connected = False

        def fail(coro, timeout=10):
            coro.close()
            raise ConnectionError("connection lost")

        self.robot._run.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, "kup"):
            self.robot.send("kup")
        self.robot._write.assert_not_called()

    def test_joint_command_returns_send_result(self):
        self.robot.send = Mock(return_value=True)
        self.assertTrue(self.robot.send_joints({0: 40, 1: -20}))
        self.robot.send.assert_called_once_with("i 0 40 1 -20")

    def test_joint_send_failure_propagates(self):
        self.robot.send = Mock(side_effect=RuntimeError("write failed"))
        with self.assertRaisesRegex(RuntimeError, "write failed"):
            self.robot.send_joints({0: 40})


class BleRobotRunTests(unittest.TestCase):
    def setUp(self):
        self.robot = BleRobot.__new__(BleRobot)
        self.robot._loop = object()

    @staticmethod
    async def noop():
        return None

    def test_completed_future_returns_its_result(self):
        future = Mock()
        future.result.return_value = "sent"
        coro = self.noop()
        try:
            with patch("robot_link.asyncio.run_coroutine_threadsafe", return_value=future) as submit:
                self.assertEqual(self.robot._run(coro, timeout=5), "sent")
            submit.assert_called_once_with(coro, self.robot._loop)
            future.result.assert_called_once_with(5)
            future.cancel.assert_not_called()
        finally:
            coro.close()

    def test_timeout_cancels_pending_coroutine_and_preserves_error(self):
        error = FutureTimeoutError("write timed out")
        future = Mock()
        future.result.side_effect = error
        coro = self.noop()
        try:
            with patch("robot_link.asyncio.run_coroutine_threadsafe", return_value=future):
                with self.assertRaises(FutureTimeoutError) as caught:
                    self.robot._run(coro, timeout=5)
            self.assertIs(caught.exception, error)
            future.result.assert_called_once_with(5)
            future.cancel.assert_called_once_with()
        finally:
            coro.close()

    def test_non_timeout_exception_propagates_without_cancelling(self):
        error = ValueError("bad BLE data")
        future = Mock()
        future.result.side_effect = error
        coro = self.noop()
        try:
            with patch("robot_link.asyncio.run_coroutine_threadsafe", return_value=future):
                with self.assertRaises(ValueError) as caught:
                    self.robot._run(coro)
            self.assertIs(caught.exception, error)
            future.cancel.assert_not_called()
        finally:
            coro.close()


if __name__ == "__main__":
    unittest.main()
