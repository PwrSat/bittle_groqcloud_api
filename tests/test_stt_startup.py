"""Startup regressions without loading native libraries or using hardware.

Run with: python -m unittest tests.test_stt_startup -v
"""

import os
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, mock_open, patch

import speechtotext as stt


ROOT = Path(__file__).resolve().parents[1]


def fake_module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


class TranscriberInitializationTests(unittest.TestCase):
    def test_initialization_reuses_the_model_without_opening_audio_devices(self):
        model = object()
        constructor = Mock(return_value=model)
        with (
            patch.object(stt, "_model", None),
            patch.dict(sys.modules, {
                "faster_whisper": fake_module("faster_whisper", WhisperModel=constructor),
            }),
            patch.object(stt, "_get_audio_devices") as audio_devices,
            patch("builtins.print"),
        ):
            self.assertIs(stt.initialize_transcriber(), model)
            self.assertIs(stt.initialize_transcriber(), model)
            constructor.assert_called_once_with(
                stt.WHISPER_MODEL, device="cpu", compute_type="int8"
            )
            audio_devices.assert_not_called()

    def test_failed_construction_leaves_the_cache_empty_and_allows_retry(self):
        model = object()
        constructor = Mock(side_effect=[RuntimeError("model unavailable"), model])
        with (
            patch.object(stt, "_model", None),
            patch.dict(sys.modules, {
                "faster_whisper": fake_module("faster_whisper", WhisperModel=constructor),
            }),
            patch("builtins.print"),
        ):
            with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                stt.initialize_transcriber()
            self.assertIsNone(stt._model)
            self.assertIs(stt.initialize_transcriber(), model)
            self.assertEqual(constructor.call_count, 2)


class ApplicationStartupTests(unittest.TestCase):
    def run_startup(self, filename, events, initialization_error=None):
        """Execute each real entrypoint with external services replaced by fakes."""
        def initialize():
            events.append("whisper")
            if initialization_error is not None:
                raise initialization_error
            return object()

        def get_model(provider):
            events.append("llm")
            return object()

        def connect(*args, **kwargs):
            events.append("robot")
            return Mock()

        prompt = Mock()
        prompt.__or__ = Mock(return_value=object())
        modules = {
            "dotenv": fake_module("dotenv", load_dotenv=Mock()),
            "langchain_core": fake_module("langchain_core"),
            "langchain_core.messages": fake_module(
                "langchain_core.messages", HumanMessage=Mock(), AIMessage=Mock()
            ),
            "langchain_core.chat_history": fake_module(
                "langchain_core.chat_history",
                BaseChatMessageHistory=object, InMemoryChatMessageHistory=Mock(),
            ),
            "langchain_core.runnables": fake_module("langchain_core.runnables"),
            "langchain_core.runnables.history": fake_module(
                "langchain_core.runnables.history", RunnableWithMessageHistory=Mock()
            ),
            "langchain_core.prompts": fake_module(
                "langchain_core.prompts",
                ChatPromptTemplate=SimpleNamespace(from_messages=Mock(return_value=prompt)),
                MessagesPlaceholder=Mock(),
            ),
            "speechtotext": fake_module(
                "speechtotext", initialize_transcriber=initialize,
                listen_and_transcribe=Mock(side_effect=AssertionError("audio must stay closed")),
            ),
            "Text2Speech": fake_module("Text2Speech", text_to_speech_stream=Mock()),
            "robot_link": fake_module("robot_link", create_robot=connect),
            "ardSerial": fake_module("ardSerial", connectPort=connect),
            "speech_recognition": fake_module("speech_recognition"),
            "llm_utils": fake_module(
                "llm_utils", get_model=get_model, strip_thinking=Mock(),
                extract_final_reply=Mock(), is_exit_command=Mock(), token_counter=Mock(),
            ),
            "motion_gen": fake_module("motion_gen"),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ),
            patch("builtins.print"),
            patch("builtins.open", mock_open(read_data='{"sit": "ksit"}')),
        ):
            # Non-main execution exercises startup without entering the voice loop.
            runpy.run_path(str(ROOT / filename), run_name="__startup_test__")

    def test_whisper_initializes_before_llm_and_robot_in_both_entrypoints(self):
        for filename in ("BittleLLM.py", "BittleLLM_1.py"):
            with self.subTest(filename=filename):
                events = []
                self.run_startup(filename, events)
                self.assertEqual(events, ["whisper", "llm", "robot"])

    def test_whisper_failure_prevents_llm_and_robot_startup(self):
        for filename in ("BittleLLM.py", "BittleLLM_1.py"):
            with self.subTest(filename=filename):
                events = []
                with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                    self.run_startup(filename, events, RuntimeError("model unavailable"))
                self.assertEqual(events, ["whisper"])


if __name__ == "__main__":
    unittest.main()
