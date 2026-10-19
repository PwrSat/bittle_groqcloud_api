"""CUDA DLL discovery is scoped to the venv and never initializes hardware."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cuda_runtime as runtime


class CudaRuntimeTests(unittest.TestCase):
    def test_module_import_uses_only_standard_library_and_does_not_load_backend(self):
        code = (
            "import sys; import cuda_runtime; "
            "assert not any(name.startswith(('faster_whisper', 'ctranslate2', 'nvidia', 'torch')) "
            "for name in sys.modules); assert not cuda_runtime._DLL_DIRECTORY_HANDLES"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-S", "-c", code],
            cwd=Path(runtime.__file__).resolve().parent, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_non_windows_is_a_noop_without_discovery_or_path_changes(self):
        with (
            patch.object(runtime.sys, "platform", "linux"),
            patch.object(runtime, "_discover_cuda_directories", side_effect=AssertionError("no discovery")) as discover,
            patch.object(runtime, "_DLL_DIRECTORY_HANDLES", {}),
            patch.dict(os.environ, {"PATH": "LEGACY_PATH"}),
        ):
            self.assertEqual(runtime.configure_cuda_libraries(), [])
            self.assertEqual(os.environ["PATH"], "LEGACY_PATH")
            self.assertEqual(runtime._DLL_DIRECTORY_HANDLES, {})
        discover.assert_not_called()

    def test_registration_retains_handles_and_is_idempotent_for_legacy_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            directories = [Path(temporary) / "CUDA library one", Path(temporary) / "CUDA library two"]
            for directory in directories:
                directory.mkdir()
            handles = [Mock(), Mock()]
            legacy = os.pathsep.join(("LEGACY_ONE", "LEGACY_TWO"))
            with (
                patch.object(runtime.sys, "platform", "win32"),
                patch.object(runtime, "_discover_cuda_directories", return_value=directories),
                patch.object(runtime, "_DLL_DIRECTORY_HANDLES", {}),
                patch.object(runtime.os, "add_dll_directory", side_effect=handles, create=True) as register,
                patch.dict(os.environ, {"PATH": legacy}),
            ):
                self.assertEqual(runtime.configure_cuda_libraries(), directories)
                first_path = os.environ["PATH"]
                self.assertEqual(runtime.configure_cuda_libraries(), directories)
                self.assertEqual(os.environ["PATH"], first_path)
                self.assertEqual(first_path.split(os.pathsep), [str(path) for path in directories] + ["LEGACY_ONE", "LEGACY_TWO"])
                self.assertEqual(list(runtime._DLL_DIRECTORY_HANDLES.values()), handles)

            self.assertEqual(register.call_count, 2)
            self.assertEqual([call.args[0] for call in register.call_args_list], [str(path) for path in directories])
            for handle in handles:
                handle.close.assert_not_called()

    def test_discovery_uses_installed_manifests_and_excludes_dlls_outside_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            environment = base / "venv"
            purelib = environment / "Lib" / "site-packages"
            internal = purelib / "nvidia" / "cublas" / "bin" / "cublas64_12.dll"
            external = base / "other-environment" / "bin" / "cudnn64_9.dll"
            fallback = purelib / "nvidia" / "cuda_runtime" / "bin" / "cudart64_12.dll"
            nvrtc = purelib / "nvidia" / "cuda_nvrtc" / "bin" / "nvrtc64_120_0.dll"
            external_nvrtc = base / "other-environment" / "nvrtc" / "nvrtc64_120_0.dll"
            for path in (internal, external, fallback, nvrtc, external_nvrtc):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test placeholder, never loaded")
            internal_dist = SimpleNamespace(files=["nvidia/cublas/bin/cublas64_12.dll"],
                                            locate_file=lambda _: internal)
            external_dist = SimpleNamespace(files=["bin/cudnn64_9.dll"], locate_file=lambda _: external)
            nvrtc_dist = SimpleNamespace(files=["internal-nvrtc.dll", "external-nvrtc.dll"],
                                         locate_file=lambda entry: nvrtc if entry == "internal-nvrtc.dll" else external_nvrtc)

            def distribution(name):
                if name == "nvidia-cublas-cu12":
                    return internal_dist
                if name == "nvidia-cudnn-cu12":
                    return external_dist
                if name == "nvidia-cuda-nvrtc-cu12" and manifest_available:
                    return nvrtc_dist
                raise runtime.metadata.PackageNotFoundError(name)

            for manifest_available in (True, False):
                with (
                    self.subTest(nvrtc_manifest=manifest_available),
                    patch.object(runtime.sys, "prefix", str(environment)),
                    patch.object(runtime.metadata, "distribution", side_effect=distribution),
                    patch.object(runtime.sysconfig, "get_path", return_value=str(purelib)),
                ):
                    directories = runtime._discover_cuda_directories()

                self.assertEqual(set(directories), {internal.parent.resolve(), fallback.parent.resolve(), nvrtc.parent.resolve()})
                self.assertEqual(len(directories), 3)
                self.assertNotIn(external.parent, directories)
                self.assertNotIn(external_nvrtc.parent, directories)

    def test_registration_failure_is_reported_instead_of_silently_using_another_runtime(self):
        with (
            patch.object(runtime.sys, "platform", "win32"),
            patch.object(runtime, "_discover_cuda_directories", return_value=[Path("C:/fake-venv/cuda/bin")]),
            patch.object(runtime, "_DLL_DIRECTORY_HANDLES", {}),
            patch.object(runtime.os, "add_dll_directory", side_effect=OSError("DLL directory unavailable"), create=True),
            patch.dict(os.environ, {"PATH": "LEGACY_PATH"}),
        ):
            with self.assertRaises(RuntimeError):
                runtime.configure_cuda_libraries()
            self.assertEqual(os.environ["PATH"], "LEGACY_PATH")
            self.assertEqual(runtime._DLL_DIRECTORY_HANDLES, {})


if __name__ == "__main__":
    unittest.main()
