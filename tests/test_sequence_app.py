"""Real LangChain prompt/history integration with fake audio, model, and robot.

Run with: python -B -m unittest tests.test_sequence_app -v
"""

import json
import os
from contextlib import contextmanager
from pathlib import Path
import runpy
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import action_sequence
import llm_utils
import robot_link
import speechtotext


ROOT = Path(__file__).resolve().parents[1]
REQUEST = "duduk, tengok kanan, lalu berdiri"
FAREWELL = "Baik bosku, sampai jumpa lagi!"


@contextmanager
def replace_module(name, module):
    """Replace only the audio module; retain LangChain's newly imported modules.

    Restoring all of sys.modules would discard Pydantic classes while retaining
    their validator registry, making a later LangChain import fail.
    """
    missing = object()
    previous = sys.modules.get(name, missing)
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous is missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class RecordingModelCallback(BaseCallbackHandler):
    def __init__(self, events):
        self.events = events

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.events.append(("model", messages[0]))


class RecordingRobot:
    def __init__(self, events):
        self.events = events

    def send(self, command):
        self.events.append(("send", command))

    def send_joints(self, angles):
        self.events.append(("joints", dict(angles)))

    def close(self):
        self.events.append(("close",))


class SequenceApplicationTests(unittest.TestCase):
    def run_application(self, responses, request=REQUEST):
        events = []
        robot = RecordingRobot(events)
        model = FakeListChatModel(
            responses=responses, callbacks=[RecordingModelCallback(events)]
        )
        utterances = iter([(request, "id"), ("udahan dulu", "id")])

        def listen():
            try:
                utterance = next(utterances)
            except StopIteration:
                raise AssertionError("application listened after the exit command")
            events.append(("listen", utterance[0]))
            return utterance

        def connect(*args, **kwargs):
            events.append(("connect",))
            return robot

        # Avoid importing audio playback libraries as well as playing sound.
        speech_module = ModuleType("Text2Speech")
        speech_module.text_to_speech_stream = Mock(
            side_effect=lambda text, **kwargs: events.append(("speech", text))
        )
        counter = llm_utils.TokenCounter()
        with (
            replace_module("Text2Speech", speech_module),
            patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false"}),
            patch("dotenv.load_dotenv"),
            patch.object(speechtotext, "initialize_transcriber", side_effect=lambda: events.append(("whisper",))),
            patch.object(speechtotext, "listen_and_transcribe", side_effect=listen),
            patch.object(llm_utils, "get_model", return_value=model),
            patch.object(llm_utils, "token_counter", counter),
            patch.object(robot_link, "create_robot", side_effect=connect),
            patch.object(action_sequence.time, "sleep"),
            patch("builtins.print"),
        ):
            namespace = runpy.run_path(str(ROOT / "BittleLLM.py"), run_name="__main__")
        return events, counter, namespace

    def test_main_prepares_head_then_executes_mixed_sequence_and_exits(self):
        reply = "Siap, aku duduk, tengok kanan, lalu berdiri!"
        outer = json.dumps({"reply": reply, "actions": [
            {"type": "skill", "command": "ksit", "seconds": 2},
            {"type": "motion", "request": "tengok kepala ke kanan"},
            {"type": "skill", "command": "kup", "seconds": 1},
        ]})
        head = json.dumps({"steps": [{"joints": {"0": 30}, "hold": 0.3}]})
        events, counter, namespace = self.run_application([outer, head])

        model_events = [(i, event[1]) for i, event in enumerate(events) if event[0] == "model"]
        self.assertEqual(len(model_events), 2)
        # Formatting the real prompt must preserve literal JSON braces.
        self.assertIn('"actions"', model_events[0][1][0].content)
        self.assertIn('"command": "ksit"', model_events[0][1][0].content)
        self.assertEqual(model_events[0][1][-1].content, f"[bahasa: id] {REQUEST}")
        self.assertEqual(model_events[1][1][-1].content, "tengok kepala ke kanan")

        first_send = next(i for i, event in enumerate(events) if event[0] == "send")
        self.assertLess(model_events[1][0], first_send, "head must be planned before the first skill")
        movements = [event for event in events if event[0] in {"send", "joints", "close"}]
        self.assertEqual(movements[0], ("send", "ksit"))
        self.assertEqual([event for event in movements if event[0] == "send"], [
            ("send", "ksit"), ("send", "i"), ("send", "kup"), ("send", "kup"),
        ])
        head_positions = [event[1][0] for event in movements if event[0] == "joints"]
        self.assertTrue(head_positions)
        self.assertIn(-30, head_positions)  # physical sign correction remains active
        self.assertEqual(head_positions[-1], 0)
        self.assertEqual(movements[-1], ("close",))
        self.assertEqual([event for event in events if event[0] in {"whisper", "connect"}], [
            ("whisper",), ("connect",),
        ])
        self.assertEqual(counter.calls, 2)
        self.assertIn(("speech", reply), events)
        self.assertIn(("speech", FAREWELL), events)
        # Exercise the real RunnableWithMessageHistory, not a stub implementation.
        messages = namespace["store"]["firstChat"].messages
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].content, f"[bahasa: id] {REQUEST}")
        self.assertEqual(messages[1].content, outer)

    def test_invalid_later_skill_prevents_all_actions_except_exit_stop(self):
        outer = json.dumps({"reply": "Baik!", "actions": [
            {"type": "skill", "command": "ksit"},
            {"type": "skill", "command": "arbitrary serial command"},
        ]})
        events, counter, _ = self.run_application([outer])
        self.assertEqual([event for event in events if event[0] in {"send", "joints", "close"}], [
            ("send", "kup"), ("close",),
        ])
        self.assertEqual(counter.calls, 1)

    def test_empty_head_plan_prevents_the_earlier_sit(self):
        outer = json.dumps({"reply": "Baik!", "actions": [
            {"type": "skill", "command": "ksit"},
            {"type": "motion", "request": "tengok kepala ke kanan"},
        ]})
        events, counter, _ = self.run_application([outer, '{"steps": []}'])
        self.assertEqual([event for event in events if event[0] in {"send", "joints", "close"}], [
            ("send", "kup"), ("close",),
        ])
        self.assertEqual(counter.calls, 2)

    def test_conversation_only_answers_without_a_gesture(self):
        reply = "Sama-sama, senang bisa menemanimu!"
        outer = json.dumps({"reply": reply, "actions": []})
        events, counter, _ = self.run_application([outer], request="terima kasih")
        self.assertEqual([event for event in events if event[0] in {"send", "joints", "close"}], [
            ("send", "kup"), ("close",),
        ])
        self.assertIn(("speech", reply), events)
        self.assertEqual(counter.calls, 1)


if __name__ == "__main__":
    unittest.main()
