"""Expose CUDA DLLs from the active Python environment to this process only."""

from importlib import metadata
import os
from pathlib import Path
import sys
import sysconfig

_PACKAGES = ("nvidia-cublas-cu12", "nvidia-cudnn-cu12", "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12")
_DLL_DIRECTORY_HANDLES = {}


def _inside_environment(path):
    try:
        resolved = Path(path).resolve()
        resolved.relative_to(Path(sys.prefix).resolve())
        return resolved
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _discover_cuda_directories():
    """Read wheel manifests without importing NVIDIA packages or loading DLLs."""
    directories = []
    for package in _PACKAGES:
        try:
            distribution = metadata.distribution(package)
        except metadata.PackageNotFoundError:
            continue
        for entry in distribution.files or ():
            if str(entry).lower().endswith(".dll"):
                path = _inside_environment(distribution.locate_file(entry))
                if path is not None and path.is_file() and path.parent not in directories:
                    directories.append(path.parent)
    purelib = _inside_environment(sysconfig.get_path("purelib"))
    if purelib is not None:
        for component in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
            directory = _inside_environment(purelib / "nvidia" / component / "bin")
            if directory is not None and directory.is_dir() and directory not in directories:
                directories.append(directory)
    return directories


def configure_cuda_libraries():
    """Register wheel DLL folders and retain handles; never change system PATH."""
    if sys.platform != "win32":
        return []
    directories = _discover_cuda_directories()
    add_directory = getattr(os, "add_dll_directory", None)
    for directory in directories:
        key = os.path.normcase(str(directory))
        if add_directory is not None and key not in _DLL_DIRECTORY_HANDLES:
            try:
                _DLL_DIRECTORY_HANDLES[key] = add_directory(str(directory))
            except OSError as error:
                raise RuntimeError("Folder DLL CUDA dalam venv tidak dapat didaftarkan.") from error
    # CTranslate2 also uses legacy LoadLibraryA, which reads the process PATH.
    prefix = [str(directory) for directory in directories]
    keys = {os.path.normcase(item) for item in prefix}
    previous = [item for item in os.environ.get("PATH", "").split(os.pathsep)
                if item and os.path.normcase(item) not in keys]
    if prefix:
        os.environ["PATH"] = os.pathsep.join(prefix + previous)
    return directories
