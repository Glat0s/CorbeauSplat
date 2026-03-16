"""
VR 180 green-screen segmentation settings tab.

Provides controls for:
  - VR format (SBS / Top-Bottom)
  - Eye selection (Left / Right)
  - Chroma-key tuning
  - SAM (Segment Anything) optional refinement
"""

from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core.i18n import add_language_observer, tr
from app.core.system import resolve_project_root


class VR180Tab(QWidget):
    """Settings panel for VR 180 green-screen processing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        add_language_observer(self.retranslate_ui)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ---- Header ----
        self.lbl_header = QLabel(tr("vr180_header", "VR 180 Green Screen Segmentation"))
        self.lbl_header.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 6px;")
        layout.addWidget(self.lbl_header)

        self.lbl_desc = QLabel(
            tr(
                "vr180_desc",
                "Process VR 180 side-by-side or top-bottom green-screen footage.\n"
                "Extracts one eye, removes the green background with chroma-key, "
                "and optionally refines the person mask with SAM.",
            )
        )
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setStyleSheet("color: #888; margin-bottom: 12px;")
        layout.addWidget(self.lbl_desc)

        # Workflow note
        lbl_note = QLabel(
            "Tip: This tab is a pre-processing tool — run it <i>before</i> the main pipeline "
            "(Steps 1-3). Select <b>VR 180 Green Screen (SAM)</b> as mode in Step 1 · Source, "
            "then use this tab to configure and run chroma-key extraction."
        )
        lbl_note.setWordWrap(True)
        lbl_note.setStyleSheet(
            "background-color: #1a3a1a; color: #7ec87e; "
            "border: 1px solid #2a6a2a; border-radius: 6px; "
            "padding: 8px 12px; font-size: 11px; margin-bottom: 8px;"
        )
        layout.addWidget(lbl_note)

        # ---- VR Format ----
        grp_format = QGroupBox(tr("vr180_grp_format", "VR 180 Format"))
        fmt_layout = QHBoxLayout(grp_format)

        self.fmt_group = QButtonGroup(self)
        self.radio_sbs = QRadioButton(tr("vr180_fmt_sbs", "Side-by-Side (SBS)"))
        self.radio_sbs.setChecked(True)
        self.radio_tb = QRadioButton(tr("vr180_fmt_tb", "Top-Bottom (TB)"))
        self.fmt_group.addButton(self.radio_sbs)
        self.fmt_group.addButton(self.radio_tb)
        fmt_layout.addWidget(self.radio_sbs)
        fmt_layout.addWidget(self.radio_tb)
        fmt_layout.addStretch()
        layout.addWidget(grp_format)

        # ---- Eye Selection ----
        grp_eye = QGroupBox(tr("vr180_grp_eye", "Eye"))
        eye_layout = QHBoxLayout(grp_eye)

        self.eye_group = QButtonGroup(self)
        self.radio_left = QRadioButton(tr("vr180_eye_left", "Left Eye"))
        self.radio_left.setChecked(True)
        self.radio_right = QRadioButton(tr("vr180_eye_right", "Right Eye"))
        self.eye_group.addButton(self.radio_left)
        self.eye_group.addButton(self.radio_right)
        eye_layout.addWidget(self.radio_left)
        eye_layout.addWidget(self.radio_right)
        eye_layout.addStretch()
        layout.addWidget(grp_eye)

        # ---- Chroma Key ----
        grp_chroma = QGroupBox(tr("vr180_grp_chroma", "Green Screen (Chroma Key)"))
        chroma_layout = QVBoxLayout(grp_chroma)

        row = QHBoxLayout()
        self.lbl_hue = QLabel(tr("vr180_lbl_hue", "Hue centre (HSV 0–180):"))
        row.addWidget(self.lbl_hue)
        self.spin_hue = QSpinBox()
        self.spin_hue.setRange(0, 180)
        self.spin_hue.setValue(60)
        self.spin_hue.setToolTip(tr("vr180_tip_hue", "60 = pure green in OpenCV HSV scale"))
        row.addWidget(self.spin_hue)
        row.addStretch()
        chroma_layout.addLayout(row)

        row = QHBoxLayout()
        self.lbl_hue_range = QLabel(tr("vr180_lbl_hue_range", "Hue tolerance (±):"))
        row.addWidget(self.lbl_hue_range)
        self.spin_hue_range = QSpinBox()
        self.spin_hue_range.setRange(1, 90)
        self.spin_hue_range.setValue(25)
        row.addWidget(self.spin_hue_range)
        row.addStretch()
        chroma_layout.addLayout(row)

        row = QHBoxLayout()
        self.lbl_sat = QLabel(tr("vr180_lbl_sat", "Min saturation (0–255):"))
        row.addWidget(self.lbl_sat)
        self.spin_sat = QSpinBox()
        self.spin_sat.setRange(0, 255)
        self.spin_sat.setValue(60)
        row.addWidget(self.spin_sat)
        row.addStretch()
        chroma_layout.addLayout(row)

        row = QHBoxLayout()
        self.lbl_val = QLabel(tr("vr180_lbl_val", "Min value/brightness (0–255):"))
        row.addWidget(self.lbl_val)
        self.spin_val = QSpinBox()
        self.spin_val.setRange(0, 255)
        self.spin_val.setValue(40)
        row.addWidget(self.spin_val)
        row.addStretch()
        chroma_layout.addLayout(row)

        layout.addWidget(grp_chroma)

        # ---- SAM Segmentation ----
        grp_sam = QGroupBox(tr("vr180_grp_sam", "SAM Person Segmentation (optional)"))
        sam_layout = QVBoxLayout(grp_sam)

        self.check_use_sam = QCheckBox(
            tr("vr180_check_sam", "Enable SAM (Segment Anything) refinement")
        )
        self.check_use_sam.setToolTip(
            tr(
                "vr180_tip_sam",
                "Uses Meta's Segment Anything Model to produce a cleaner person mask.\n"
                "Requires the SAM checkpoint (downloaded automatically on first setup).",
            )
        )
        self.check_use_sam.toggled.connect(self._update_sam_group)
        sam_layout.addWidget(self.check_use_sam)

        # Batch size
        batch_row = QHBoxLayout()
        self.lbl_batch = QLabel(tr("vr180_lbl_batch", "GPU batch size:"))
        batch_row.addWidget(self.lbl_batch)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 64)
        self.spin_batch.setValue(8)
        self.spin_batch.setToolTip(
            tr(
                "vr180_tip_batch",
                "Number of frames processed per GPU batch. Higher = faster but more VRAM.",
            )
        )
        batch_row.addWidget(self.spin_batch)
        batch_row.addStretch()
        sam_layout.addLayout(batch_row)

        # Model type
        row = QHBoxLayout()
        self.lbl_sam_model = QLabel(tr("vr180_lbl_sam_model", "SAM model:"))
        row.addWidget(self.lbl_sam_model)
        self.combo_sam_model = QComboBox()
        self.combo_sam_model.addItem("ViT-B (fastest, ~375 MB)", "vit_b")
        self.combo_sam_model.addItem("ViT-L (balanced, ~1.2 GB)", "vit_l")
        self.combo_sam_model.addItem("ViT-H (best quality, ~2.6 GB)", "vit_h")
        self.combo_sam_model.currentIndexChanged.connect(self._update_sam_status)
        row.addWidget(self.combo_sam_model)
        row.addStretch()
        sam_layout.addLayout(row)

        # Checkpoint status (read-only — downloaded by setup)
        self.lbl_sam_status = QLabel("❌ Not downloaded")
        self.lbl_sam_status.setStyleSheet("color: #cc4444; font-weight: bold; padding: 2px 0;")
        sam_layout.addWidget(self.lbl_sam_status)

        # Device
        row = QHBoxLayout()
        self.lbl_device = QLabel(tr("vr180_lbl_device", "Compute device:"))
        row.addWidget(self.lbl_device)
        self.combo_device = QComboBox()
        self.combo_device.addItem("CUDA (NVIDIA GPU)", "cuda")
        self.combo_device.addItem("CPU (slow)", "cpu")
        row.addWidget(self.combo_device)
        row.addStretch()
        sam_layout.addLayout(row)

        layout.addWidget(grp_sam)
        layout.addStretch()

        self._sam_widgets = [
            self.lbl_sam_model,
            self.combo_sam_model,
            self.lbl_sam_status,
            self.lbl_device,
            self.combo_device,
        ]
        self._update_sam_group(False)
        self._update_sam_status()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _update_sam_group(self, enabled: bool):
        for w in self._sam_widgets:
            w.setEnabled(enabled)

    def _update_sam_status(self):
        """Refresh the checkpoint status label for the selected model."""
        model_type = self.combo_sam_model.currentData()
        dest = resolve_project_root() / "engines" / f"sam_{model_type}.pth"
        if dest.exists():
            self.lbl_sam_status.setText("✅ Checkpoint ready")
            self.lbl_sam_status.setStyleSheet("color: #44cc44; font-weight: bold; padding: 2px 0;")
        else:
            self.lbl_sam_status.setText("❌ Not downloaded — re-run setup to download")
            self.lbl_sam_status.setStyleSheet("color: #cc4444; font-weight: bold; padding: 2px 0;")

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        model_type = self.combo_sam_model.currentData()
        checkpoint = str(resolve_project_root() / "engines" / f"sam_{model_type}.pth")
        return {
            "vr_format": "sbs" if self.radio_sbs.isChecked() else "tb",
            "eye": "left" if self.radio_left.isChecked() else "right",
            "hue_center": self.spin_hue.value(),
            "hue_range": self.spin_hue_range.value(),
            "sat_min": self.spin_sat.value(),
            "val_min": self.spin_val.value(),
            "use_sam": self.check_use_sam.isChecked(),
            "sam_model_type": model_type,
            "sam_checkpoint": checkpoint,
            "device": self.combo_device.currentData(),
            "batch_size": self.spin_batch.value(),
        }

    def set_params(self, params: dict):
        if not params:
            return
        fmt = params.get("vr_format", "sbs")
        self.radio_sbs.setChecked(fmt == "sbs")
        self.radio_tb.setChecked(fmt == "tb")

        eye = params.get("eye", "left")
        self.radio_left.setChecked(eye == "left")
        self.radio_right.setChecked(eye == "right")

        if "hue_center" in params:
            self.spin_hue.setValue(int(params["hue_center"]))
        if "hue_range" in params:
            self.spin_hue_range.setValue(int(params["hue_range"]))
        if "sat_min" in params:
            self.spin_sat.setValue(int(params["sat_min"]))
        if "val_min" in params:
            self.spin_val.setValue(int(params["val_min"]))
        if "use_sam" in params:
            self.check_use_sam.setChecked(bool(params["use_sam"]))
        if "sam_model_type" in params:
            idx = self.combo_sam_model.findData(params["sam_model_type"])
            if idx >= 0:
                self.combo_sam_model.setCurrentIndex(idx)
            self._update_sam_status()
        if "device" in params:
            idx = self.combo_device.findData(params["device"])
            if idx >= 0:
                self.combo_device.setCurrentIndex(idx)
        if "batch_size" in params:
            self.spin_batch.setValue(int(params["batch_size"]))

    def get_state(self):
        return self.get_params()

    def set_state(self, state):
        self.set_params(state)

    # ------------------------------------------------------------------
    # i18n
    # ------------------------------------------------------------------

    def retranslate_ui(self):
        self.lbl_header.setText(tr("vr180_header", "VR 180 Green Screen Segmentation"))
        self.lbl_desc.setText(
            tr(
                "vr180_desc",
                "Process VR 180 side-by-side or top-bottom green-screen footage.\n"
                "Extracts one eye, removes the green background with chroma-key, "
                "and optionally refines the person mask with SAM.",
            )
        )
        self.radio_sbs.setText(tr("vr180_fmt_sbs", "Side-by-Side (SBS)"))
        self.radio_tb.setText(tr("vr180_fmt_tb", "Top-Bottom (TB)"))
        self.radio_left.setText(tr("vr180_eye_left", "Left Eye"))
        self.radio_right.setText(tr("vr180_eye_right", "Right Eye"))
        self.lbl_hue.setText(tr("vr180_lbl_hue", "Hue centre (HSV 0–180):"))
        self.lbl_hue_range.setText(tr("vr180_lbl_hue_range", "Hue tolerance (±):"))
        self.lbl_sat.setText(tr("vr180_lbl_sat", "Min saturation (0–255):"))
        self.lbl_val.setText(tr("vr180_lbl_val", "Min value/brightness (0–255):"))
        self.check_use_sam.setText(
            tr("vr180_check_sam", "Enable SAM (Segment Anything) refinement")
        )
        self.lbl_sam_model.setText(tr("vr180_lbl_sam_model", "SAM model:"))
        self.lbl_device.setText(tr("vr180_lbl_device", "Compute device:"))
        self.lbl_batch.setText(tr("vr180_lbl_batch", "GPU batch size:"))
        self._update_sam_status()
