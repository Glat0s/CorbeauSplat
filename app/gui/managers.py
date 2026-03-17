import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.core.params import ColmapParams
from app.core.system import resolve_project_root

logger = logging.getLogger(__name__)


class SessionManager:
    """[AUDIT] SOLID-SRP : Gestion responsable uniquement de la persistance JSON"""

    def __init__(self, main_window):
        self.mw = main_window
        self._save_timer = QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._do_save)

    def get_session_file(self) -> Path:
        return resolve_project_root() / "config.json"

    def save(self, immediate=False):
        """[AUDIT] Optimisation Perf-IO : Debounce de la sauvegarde JSON pour ne pas geler l'UI"""
        if immediate:
            self._save_timer.stop()
            self._do_save()
        else:
            self._save_timer.start(1500)  # Debounce 1.5s

    def _do_save(self):
        state = {
            "language": self.mw.config_tab.combo_lang.currentData(),
        }

        tab_mapping = {
            "config": self.mw.config_tab,
            "colmap_params": self.mw.params_tab,
            "brush_params": self.mw.brush_tab,
            "sharp_params": self.mw.sharp_tab,
            "upscale_params": self.mw.upscale_tab,
            "extractor_360_params": self.mw.extractor_360_tab,
            "four_dgs_params": self.mw.four_dgs_tab,
            "superplat_params": self.mw.superplat_tab,
        }

        for key, tab in tab_mapping.items():
            if hasattr(tab, "get_state"):
                state[key] = tab.get_state()
            elif hasattr(tab, "get_params"):
                state[key] = tab.get_params()
                if hasattr(state[key], "to_dict"):
                    state[key] = state[key].to_dict()

        try:
            with open(self.get_session_file(), "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error("Failed to save session: %s", e)

    def load(self):
        session_file = self.get_session_file()
        if not session_file.exists():
            return

        try:
            with open(session_file) as f:
                state = json.load(f)

            tab_mapping = {
                "config": self.mw.config_tab,
                "colmap_params": self.mw.params_tab,
                "brush_params": self.mw.brush_tab,
                "sharp_params": self.mw.sharp_tab,
                "upscale_params": self.mw.upscale_tab,
                "extractor_360_params": self.mw.extractor_360_tab,
                "four_dgs_params": self.mw.four_dgs_tab,
                "superplat_params": self.mw.superplat_tab,
            }

            for key, tab in tab_mapping.items():
                if key in state:
                    if hasattr(tab, "set_state"):
                        tab.set_state(state[key])
                    elif hasattr(tab, "set_params"):
                        if key == "colmap_params":
                            tab.set_params(ColmapParams.from_dict(state[key]))
                        else:
                            tab.set_params(state[key])
        except Exception as e:
            logger.error("Failed to load session: %s", e)


class AppLifecycle:
    """[AUDIT] SOLID-SRP : Responsable du redemarrage OS et processus externes"""

    @staticmethod
    def restart(save_callback=None):
        if save_callback:
            try:
                save_callback()
            except Exception as e:
                logger.error("Error saving session before restart: %s", e)

        root_dir = resolve_project_root()
        python = sys.executable
        main_py = root_dir / "main.py"

        engines_dir = root_dir / "engines"
        needs_setup = not (engines_dir / "brush").exists()

        if needs_setup:
            logger.info("Reinstall detected: running setup before relaunch...")
            extra_argv = [a for a in sys.argv[1:] if a not in ("--gui",)]
            main_args = " ".join(f'"{a}"' for a in extra_argv)
            if sys.platform == "win32":
                cmd = (
                    f"timeout /t 1 /nobreak >nul && "
                    f'"{python}" -m app.scripts.setup_dependencies --startup && '
                    f'"{python}" "{main_py}" {main_args}'
                )
                subprocess.Popen(cmd, shell=True, cwd=str(root_dir))
            else:
                cmd = (
                    f"sleep 1 && "
                    f'"{python}" -m app.scripts.setup_dependencies --startup && '
                    f'"{python}" "{main_py}" {main_args}'
                )
                subprocess.Popen(cmd, shell=True, cwd=str(root_dir), start_new_session=True)
            QApplication.quit()
            sys.exit(0)

        # Normal relaunch
        args = [python, str(main_py)] + sys.argv[1:]
        logger.info("Relaunching: %s", args)

        if sys.platform != "win32":
            try:
                os.execv(python, args)
            except Exception as e:
                logger.warning("execv failed: %s — falling back to Popen.", e)
            subprocess.Popen(args, cwd=str(root_dir), start_new_session=True)
        else:
            subprocess.Popen(args, cwd=str(root_dir))

        QApplication.quit()
        sys.exit(0)

    @staticmethod
    def reset_factory(deep=False):
        QApplication.quit()

        root_dir = resolve_project_root()
        python = sys.executable
        main_py = root_dir / "main.py"

        to_delete = [
            root_dir / ".venv",
            root_dir / ".venv_sharp",
            root_dir / ".venv_360",
        ]

        if deep:
            to_delete.append(root_dir / "engines")
            to_delete.append(root_dir / "config.json")
            for p in root_dir.glob("config.sync-conflict-*"):
                to_delete.append(p)

        logger.info("Reset Factory %s on: %s", "DEEP" if deep else "LIGHT", root_dir)

        if sys.platform == "win32":
            # Build a cmd.exe command: wait 2 s, delete dirs, relaunch via python
            del_parts = " & ".join(
                f'if exist "{p}" (rmdir /s /q "{p}" 2>nul || del /f /q "{p}" 2>nul)'
                for p in to_delete
            )
            cmd = f"timeout /t 2 /nobreak >nul & " f"{del_parts} & " f'"{python}" "{main_py}"'
            subprocess.Popen(cmd, shell=True, cwd=str(root_dir))
        else:
            run_cmd = root_dir / "run.command"
            delete_cmd = " ".join(f'"{p}"' for p in to_delete)
            cmd = f'sleep 2 && rm -rf {delete_cmd} && "{run_cmd}" &'
            subprocess.Popen(cmd, shell=True, cwd=str(root_dir))

        sys.exit(0)
