import os
import sys
from pathlib import Path

from .base_engine import BaseEngine
from .system import resolve_project_root


class SharpEngine(BaseEngine):
    """Moteur d'execution pour Apple ML Sharp"""

    def __init__(self, logger_callback=None):
        super().__init__("Sharp", logger_callback)
        self.process = None

    @staticmethod
    def _venv_dir():
        root_dir = resolve_project_root()
        bin_dir = "Scripts" if sys.platform == "win32" else "bin"
        return root_dir / ".venv_sharp" / bin_dir

    def _get_sharp_cmd(self):
        venv_bin = self._venv_dir()

        # 1. Look for sharp entry-point script in .venv_sharp
        for name in ["sharp.exe", "sharp"] if sys.platform == "win32" else ["sharp"]:
            sharp_bin = venv_bin / name
            if sharp_bin.exists() and (sys.platform == "win32" or os.access(sharp_bin, os.X_OK)):
                return [str(sharp_bin)]

        # 2. Fall back to running the module via the venv's python
        python_name = "python.exe" if sys.platform == "win32" else "python3"
        sharp_python = venv_bin / python_name
        if sharp_python.exists():
            return [str(sharp_python), "-m", "sharp.cli"]

        # 3. Check global PATH
        from shutil import which

        if which("sharp"):
            return ["sharp"]

        # 4. Last resort: current executable
        return [sys.executable, "-m", "sharp.cli"]

    def is_installed(self):
        """Vérifie si Sharp est disponible (venv_sharp ou local)"""
        import importlib.util
        from shutil import which

        # 1. Check .venv_sharp (platform-aware)
        venv_bin = self._venv_dir()
        for name in (
            ["sharp.exe", "sharp", "python.exe"]
            if sys.platform == "win32"
            else ["sharp", "python3"]
        ):
            if (venv_bin / name).exists():
                return True

        # 2. Check binary on PATH
        if which("sharp"):
            return True

        # 3. Check importable module
        return importlib.util.find_spec("sharp") is not None

    def predict(self, input_path, output_path, params=None):
        """
        Lance la prediction Sharp.
        params: dict of prediction parameters
        """
        params = params or {}
        cmd = self._get_sharp_cmd()

        cmd.extend(["predict"])
        # Prepare paths
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()

        cmd.extend(["-i", str(input_path)])
        cmd.extend(["-o", str(output_path)])

        checkpoint = params.get("checkpoint")
        if checkpoint:
            cmd.extend(["-c", str(Path(checkpoint).resolve())])

        device = params.get("device", self.device)
        if device and device != "default":
            cmd.extend(["--device", device])

        if params.get("verbose"):
            cmd.append("--verbose")

        # Environnement
        env = os.environ.copy()

        # Ensure all args are strings for Popen
        cmd = [str(arg) for arg in cmd]

        self.log(f"Launching Sharp: {' '.join(cmd)}")

        # [AUDIT] GoF-Template Method : Délégation au runner
        return self._execute_command(cmd, env=env)
