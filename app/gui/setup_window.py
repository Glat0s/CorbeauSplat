"""
First-run setup window for CorbeauSplat.

Shown automatically when the app is launched without a completed setup.
Runs all downloads and engine installs in a background thread and shows
live per-step status.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.scripts.setup import STEP_IDS

_ICON: dict[str, str] = {
    "pending": "⏸",
    "running": "⏳",
    "done": "✅",
    "failed": "❌",
    "skip": "✅",
}
_COLOR: dict[str, str] = {
    "pending": "#888888",
    "running": "#7ec8e3",
    "done": "#44cc44",
    "failed": "#cc4444",
    "skip": "#44cc44",
}


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


class SetupWorker(QThread):
    step_update = Signal(str, str, str)  # step_id, display_name, status
    log_message = Signal(str)
    finished_setup = Signal(dict)  # {step_id: bool}

    def run(self) -> None:
        from app.scripts.setup import run_setup

        results = run_setup(
            log_cb=lambda msg: self.log_message.emit(msg),
            step_cb=lambda sid, name, status: self.step_update.emit(sid, name, status),
        )
        self.finished_setup.emit(results)


# ---------------------------------------------------------------------------
# Setup dialog
# ---------------------------------------------------------------------------


class SetupWindow(QDialog):
    """
    Blocking first-run setup dialog.  Call ``exec()`` — it returns only after
    setup is complete (or the user explicitly skips via the close button).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CorbeauSplat — First Run Setup")
        self.setMinimumWidth(520)
        self.setMinimumHeight(420)
        # Prevent accidental close while setup runs
        self.setWindowFlags(
            self.windowFlags()
            & ~Qt.WindowType.WindowCloseButtonHint
            & ~Qt.WindowType.WindowContextHelpButtonHint
        )
        self._step_labels: dict[str, QLabel] = {}
        self._completed = 0
        self._worker: SetupWorker | None = None
        self._init_ui()
        self._start()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("CorbeauSplat — First Run Setup")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 4px;")
        layout.addWidget(title)

        desc = QLabel(
            "Downloading models and installing required engines.\n"
            "This only runs once and may take a few minutes depending on your connection."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #aaaaaa; margin-bottom: 8px;")
        layout.addWidget(desc)

        # Step list
        steps_widget = QWidget()
        steps_layout = QVBoxLayout(steps_widget)
        steps_layout.setSpacing(3)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        for step_id, name in STEP_IDS:
            lbl = QLabel(f"{_ICON['pending']}  {name}")
            lbl.setStyleSheet(f"color: {_COLOR['pending']}; padding: 2px 6px; font-size: 12px;")
            self._step_labels[step_id] = lbl
            steps_layout.addWidget(lbl)
        layout.addWidget(steps_widget)

        # Overall progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, len(STEP_IDS))
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        # Current status
        self.status_lbl = QLabel("Starting setup…")
        self.status_lbl.setStyleSheet("color: #7ec8e3; font-size: 11px;")
        layout.addWidget(self.status_lbl)

        layout.addStretch()

        # Launch button — enabled only when done
        self.btn_launch = QPushButton("Launch CorbeauSplat")
        self.btn_launch.setEnabled(False)
        self.btn_launch.setMinimumHeight(38)
        self.btn_launch.setStyleSheet(
            "QPushButton { background-color: #2a82da; color: white; font-weight: bold;"
            " border-radius: 4px; font-size: 13px; }"
            "QPushButton:disabled { background-color: #444; color: #888; }"
        )
        self.btn_launch.clicked.connect(self.accept)
        layout.addWidget(self.btn_launch)

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    def _start(self) -> None:
        self._worker = SetupWorker()
        self._worker.step_update.connect(self._on_step_update)
        self._worker.log_message.connect(self._on_log)
        self._worker.finished_setup.connect(self._on_finished)
        self._worker.start()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_step_update(self, step_id: str, name: str, status: str) -> None:
        icon = _ICON.get(status, "?")
        color = _COLOR.get(status, "#888888")
        lbl = self._step_labels.get(step_id)
        if lbl:
            lbl.setText(f"{icon}  {name}")
            lbl.setStyleSheet(f"color: {color}; padding: 2px 6px; font-size: 12px;")
        if status == "running":
            self.status_lbl.setText(f"Running: {name}…")
        elif status in ("done", "failed", "skip"):
            self._completed += 1
            self.progress_bar.setValue(self._completed)

    def _on_log(self, msg: str) -> None:
        pass  # Could surface in a collapsible log widget in the future

    def _on_finished(self, results: dict) -> None:
        self.progress_bar.setValue(len(STEP_IDS))
        failed = [k for k, v in results.items() if not v]
        if failed:
            self.status_lbl.setText(
                f"⚠️  Some steps failed: {', '.join(failed)}. "
                "Re-run  uv run python -m app.scripts.setup  to retry."
            )
            self.status_lbl.setStyleSheet("color: #cc8844; font-size: 11px;")
        else:
            self.status_lbl.setText("✅  Setup complete!")
            self.status_lbl.setStyleSheet("color: #44cc44; font-size: 11px;")
        self.btn_launch.setEnabled(True)
        self.btn_launch.setFocus()
        # Re-enable close button now that it's safe
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowCloseButtonHint)
        self.show()  # re-apply flag change
