"""GPU profile wiring uses mocks; no CUDA backend, model, or robot is loaded."""

import builtins
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, sentinel

import cuda_runtime
import speechtotext as stt
from tools import check_multilingual as checker
from tools import compare_whisper_models as comparison
from tools import run_gpu


class TranscriberProfileTests(unittest.TestCase):
    def setUp(self):
        for name, value in (
            ("_model", None), ("_model_load_seconds", None),
            ("WHISPER_MODEL", "small"), ("WHISPER_DEVICE", "cpu"),
            ("WHISPER_COMPUTE_TYPE", "int8"), ("STT_LANGUAGE", None),
        ):
            patcher = patch.object(stt, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_configuring_gpu_changes_only_profile_and_keeps_language_automatic(self):
        stt.configure_transcriber(model="medium", device="cuda")

        self.assertEqual((stt.WHISPER_MODEL, stt.WHISPER_DEVICE, stt.WHISPER_COMPUTE_TYPE),
                         ("medium", "cuda", "int8_float16"))
        self.assertIsNone(stt.STT_LANGUAGE)
        self.assertIsNone(stt._model)

        stt.configure_transcriber(model="small", device="cpu")
        self.assertEqual((stt.WHISPER_MODEL, stt.WHISPER_DEVICE, stt.WHISPER_COMPUTE_TYPE),
                         ("small", "cpu", "int8"))

    def test_invalid_or_initialized_profile_is_rejected_without_partial_configuration(self):
        invalid = (
            {"model": "large-v3", "device": "cuda"},
            {"model": "medium", "device": "automatic"},
            {"model": "medium", "device": "cpu", "compute_type": "float16"},
            {"model": "medium", "device": "cuda", "compute_type": "unsupported"},
            {"model": "medium", "device": "cuda", "compute_type": ""},
            {"model": "medium", "device": "cuda", "compute_type": False},
            {"model": None, "device": "cuda"},
            {"model": "medium", "device": None},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                stt.configure_transcriber(**kwargs)
            self.assertEqual((stt.WHISPER_MODEL, stt.WHISPER_DEVICE, stt.WHISPER_COMPUTE_TYPE),
                             ("small", "cpu", "int8"))

        stt._model = sentinel.loaded_model
        with self.assertRaises(RuntimeError):
            stt.configure_transcriber(model="medium", device="cuda")
        self.assertEqual((stt.WHISPER_MODEL, stt.WHISPER_DEVICE, stt.WHISPER_COMPUTE_TYPE),
                         ("small", "cpu", "int8"))
        self.assertIs(stt._model, sentinel.loaded_model)

    def test_cuda_libraries_are_registered_before_backend_import_and_medium_uses_local_project_cache(self):
        stt.configure_transcriber(model="medium", device="cuda")
        events = []
        factory = Mock(return_value=sentinel.loaded_model)
        original_import = builtins.__import__

        def intercept_import(name, *args, **kwargs):
            if name == "faster_whisper":
                events.append("backend_import")
                return SimpleNamespace(WhisperModel=factory)
            return original_import(name, *args, **kwargs)

        with (
            patch.object(cuda_runtime, "configure_cuda_libraries", side_effect=lambda: events.append("dlls")),
            patch("builtins.__import__", side_effect=intercept_import),
            redirect_stdout(io.StringIO()),
        ):
            self.assertIs(stt._get_model(), sentinel.loaded_model)
            self.assertIs(stt._get_model(), sentinel.loaded_model)

        self.assertEqual(events, ["dlls", "backend_import"])
        factory.assert_called_once()
        self.assertEqual(factory.call_args.args, ("medium",))
        options = factory.call_args.kwargs
        self.assertEqual(options["device"], "cuda")
        self.assertEqual(options["compute_type"], "int8_float16")
        self.assertTrue(options["local_files_only"])
        self.assertEqual(Path(options["download_root"]), Path(stt.__file__).resolve().parent / "models")

    def test_cpu_small_keeps_the_previous_constructor_and_does_not_register_cuda(self):
        factory = Mock(return_value=sentinel.cpu_model)
        with (
            patch.dict("sys.modules", {"faster_whisper": SimpleNamespace(WhisperModel=factory)}),
            patch.object(cuda_runtime, "configure_cuda_libraries") as configure,
            redirect_stdout(io.StringIO()),
        ):
            self.assertIs(stt._get_model(), sentinel.cpu_model)

        factory.assert_called_once_with("small", device="cpu", compute_type="int8")
        configure.assert_not_called()

    def test_cuda_initialization_failure_propagates_without_silent_cpu_fallback(self):
        stt.configure_transcriber(model="medium", device="cuda")
        factory = Mock(side_effect=RuntimeError("CUDA not available"))
        with (
            patch.dict("sys.modules", {"faster_whisper": SimpleNamespace(WhisperModel=factory)}),
            patch.object(cuda_runtime, "configure_cuda_libraries"),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(RuntimeError) as caught:
                stt._get_model()

        self.assertIn("CUDA", str(caught.exception))
        self.assertEqual(str(caught.exception.__cause__), "CUDA not available")
        factory.assert_called_once()
        self.assertEqual(factory.call_args.kwargs["device"], "cuda")
        self.assertIsNone(stt._model)
        self.assertEqual(stt.WHISPER_DEVICE, "cuda")


class GpuToolProfileTests(unittest.TestCase):
    def test_comparison_registers_cuda_before_factory_but_leaves_cpu_path_alone(self):
        with tempfile.TemporaryDirectory() as temporary:
            for device in ("cuda", "cpu"):
                with self.subTest(device=device):
                    events = []

                    def factory_class():
                        events.append("factory_import")
                        return Mock(return_value=sentinel.comparison_model)

                    with (
                        patch.object(comparison, "PROJECT_ROOT", Path(temporary)),
                        patch.object(comparison, "_whisper_model_class", side_effect=factory_class),
                        patch.object(cuda_runtime, "configure_cuda_libraries", side_effect=lambda: events.append("dlls")),
                        patch.dict("os.environ"),
                    ):
                        model = comparison.load_model("medium", device=device, compute_type="int8", allow_download=False)

                    self.assertIs(model, sentinel.comparison_model)
                    self.assertEqual(events, ["dlls", "factory_import"] if device == "cuda" else ["factory_import"])

    def test_gpu_launcher_configures_before_running_the_application(self):
        events = []
        with (
            patch.object(stt, "configure_transcriber", side_effect=lambda **kwargs: events.append(("configure", kwargs))),
            patch("runpy.run_path", side_effect=lambda *args, **kwargs: events.append(("run", args, kwargs))),
            redirect_stdout(io.StringIO()),
        ):
            run_gpu.main()

        self.assertEqual(events[0], ("configure", {"model": "medium", "device": "cuda", "compute_type": "int8_float16"}))
        self.assertEqual(events[1][0], "run")
        self.assertEqual(Path(events[1][1][0]), Path(stt.__file__).resolve().parent / "BittleLLM.py")
        self.assertEqual(events[1][2]["run_name"], "__main__")
        self.assertEqual(len(events), 2)

    def test_guided_gpu_checker_configures_before_initialization_without_forcing_language(self):
        events = []
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(checker, "PROJECT_ROOT", Path(temporary)),
            patch.object(stt, "configure_transcriber", side_effect=lambda **kwargs: events.append(("configure", kwargs))),
            patch.object(stt, "initialize_transcriber", side_effect=lambda: events.append(("initialize",))),
            patch.object(stt, "listen_and_transcribe", return_value=("actual speech", "es")) as listen,
            patch.object(stt, "get_last_measurement", return_value={"status": "ok"}),
            patch("builtins.input", return_value=""), patch("builtins.print"),
        ):
            checker.main(["--gpu", "--guided", "--languages", "zh", "--samples", "1", "--no-audio"])

        self.assertEqual(events, [("configure", {"model": "medium", "device": "cuda", "compute_type": "int8_float16"}), ("initialize",)])
        listen.assert_called_once_with(audio_path=None)


if __name__ == "__main__":
    unittest.main()
