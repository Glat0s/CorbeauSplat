import platform
import os
import shutil
import subprocess
from pathlib import Path

def resolve_project_root() -> Path:
    """Finds project root relative to this script (app/core/system.py)"""
    return Path(__file__).resolve().parent.parent.parent

def is_apple_silicon():
    """Détecte si on est sur Apple Silicon"""
    return platform.system() == 'Darwin' and platform.machine() == 'arm64'

def is_windows():
    """Detects if running on Windows"""
    return platform.system() == 'Windows'

def is_windows_cuda():
    """Detects if running on Windows with an NVIDIA GPU (CUDA)"""
    return is_windows() and shutil.which("nvidia-smi") is not None

def get_optimal_threads():
    """Returns optimal thread count for the current platform"""
    if is_apple_silicon():
        # Apple Silicon has heterogeneous P-cores (performance) + E-cores (efficiency).
        # For compute-heavy tasks (COLMAP, ffmpeg), we prefer P-cores only.
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                p_cores = int(result.stdout.strip())
                if p_cores > 0:
                    return p_cores
        except (ValueError, subprocess.SubprocessError, OSError):
            pass
        cpu_count = os.cpu_count() or 8
        return max(1, cpu_count // 2)
    # On Windows/Linux, use all logical cores for compute tasks
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

    # 1. Look inside engines/ directory
    for candidate in candidates:
        local_path = engines_dir / candidate
        if local_path.exists():
            if is_windows() or os.access(local_path, os.X_OK):
                return str(local_path)

    # 2. macOS .app bundle for COLMAP (non-Windows only)
    if not is_windows() and name == "colmap":
        colmap_app = engines_dir / "COLMAP.app" / "Contents" / "MacOS" / "colmap"
        if colmap_app.exists() and os.access(colmap_app, os.X_OK):
            return str(colmap_app)

    # 3. System PATH
    for candidate in candidates:
        result = shutil.which(candidate)
        if result:
            return result

    return None

def get_device():
    """Centralized device selection: mps, cuda, or cpu"""
    if is_apple_silicon():
        return "mps"
    # Prefer torch-based detection when available
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
    return {
        "total": mem.total,
        "available": mem.available,
        "percent": mem.percent
    }

def check_dependencies():
    """Vérifie si les dépendances nécessaires sont installées"""
    missing = []
    
    # Check ffmpeg
    if resolve_binary('ffmpeg') is None:
        missing.append('ffmpeg')
        
    # Check colmap
    if resolve_binary('colmap') is None:
        missing.append('colmap')

    # Check send2trash
    try:
        import send2trash  # noqa: F401
    except ImportError:
        missing.append('send2trash')

    return missing
