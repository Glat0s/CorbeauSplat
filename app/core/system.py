import os
import platform
import shutil
from pathlib import Path


def resolve_project_root() -> Path:
    """Finds project root relative to this script (app/core/system.py)"""
    return Path(__file__).resolve().parent.parent.parent


def is_windows():
    """Detects if running on Windows"""
    return platform.system() == "Windows"


def is_windows_cuda():
    """Detects if running on Windows with an NVIDIA GPU (CUDA)"""
    return is_windows() and shutil.which("nvidia-smi") is not None


def get_optimal_threads():
    """Returns optimal thread count for the current platform"""
    return os.cpu_count() or 4


def resolve_binary(name):
    """
    Resolves a binary path, prioritising the local 'engines' directory.
    On Windows, also tries the .exe extension.
    Returns the absolute path string, or None if not found.
    """
    engines_dir = resolve_project_root() / "engines"

    # Candidate names: on Windows also try name.exe
    candidates = [name]
    if is_windows() and not name.endswith(".exe"):
        candidates.append(name + ".exe")

    # 1. Look inside engines/ directory (direct children)
    for candidate in candidates:
        local_path = engines_dir / candidate
        if local_path.exists() and (is_windows() or os.access(local_path, os.X_OK)):
            return str(local_path)

    # 2. On Windows, search engines/colmap/ subdirectory tree for colmap.exe
    if is_windows() and name == "colmap":
        colmap_dir = engines_dir / "colmap"
        if colmap_dir.exists():
            for root_dir, _dirs, files in os.walk(str(colmap_dir)):
                for f in files:
                    if f.lower() == "colmap.exe":
                        return str(Path(root_dir) / f)

    # 3. System PATH
    for candidate in candidates:
        result = shutil.which(candidate)
        if result:
            return result

    return None


def get_device():
    """Centralized device selection: cuda or cpu"""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    if shutil.which("nvidia-smi") is not None:
        return "cuda"
    return "cpu"


def get_memory_info():
    """Returns memory info for UMA/caching strategies"""
    import psutil

    mem = psutil.virtual_memory()
    return {"total": mem.total, "available": mem.available, "percent": mem.percent}


def check_dependencies():
    """Checks if required dependencies are installed"""
    missing = []

    if resolve_binary("ffmpeg") is None:
        missing.append("ffmpeg")

    if resolve_binary("colmap") is None:
        missing.append("colmap")

    try:
        import send2trash  # noqa: F401
    except ImportError:
        missing.append("send2trash")

    return missing
