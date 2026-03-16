"""
VR 180 green-screen segmentation settings tab.

Shown when the user selects "VR 180" mode in the Config tab.
Provides controls for:
  - VR format (SBS / Top-Bottom)
  - Eye selection (Left / Right)
  - Chroma-key tuning
  - SAM (Segment Anything) optional refinement
  - SAM checkpoint download / path
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QRadioButton, QButtonGroup, QCheckBox,
    QDoubleSpinBox, QSpinBox, QComboBox, QProgressBar,
    QMessageBox, QFileDialog,
)
from PyQt6.QtCore import pyqtSignal, QThread

from app.core.i18n import tr, add_language_observer
from app.gui.widgets.drop_line_edit import DropLineEdit
from app.gui.widgets.dialog_utils import get_open_file_name


class SAMDownloadWorker(QThread):
    """Downloads the SAM checkpoint in a background thread."""
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str)

    # vit_b is the smallest/fastest model (~375 MB)
    CHECKPOINT_URLS = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }

    def __init__(self, model_type: str, dest_path: str):
        super().__init__()
        self.model_type = model_type
        self.dest_path = dest_path

    def run(self):
        import urllib.request
        from pathlib import Path

        url = self.CHECKPOINT_URLS.get(self.model_type)
        if not url:
            self.finished_signal.emit(False, f"Unknown model type: {self.model_type}")
            return

        dest = Path(self.dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        def _reporthook(block, block_size, total):
            if total > 0:
                pct = min(99, int(block * block_size / total * 100))
                self.progress_signal.emit(pct)

        try:
            urllib.request.urlretrieve(url, str(dest), reporthook=_reporthook)
            self.progress_signal.emit(100)
            self.finished_signal.emit(True, str(dest))
        except Exception as e:
            self.finished_signal.emit(False, str(e))


class VR180Tab(QWidget):
    """Settings panel for VR 180 green-screen processing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._download_worker = None
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
            tr("vr180_desc",
               "Process VR 180 side-by-side or top-bottom green-screen footage.\n"
               "Extracts one eye, removes the green background with chroma-key, "
               "and optionally refines the person mask with SAM.")
        )
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setStyleSheet("color: #888; margin-bottom: 12px;")
        layout.addWidget(self.lbl_desc)

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

        # Hue Centre
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

        # Hue Range
        row = QHBoxLayout()
        self.lbl_hue_range = QLabel(tr("vr180_lbl_hue_range", "Hue tolerance (±):"))
        row.addWidget(self.lbl_hue_range)
        self.spin_hue_range = QSpinBox()
        self.spin_hue_range.setRange(1, 90)
        self.spin_hue_range.setValue(25)
        row.addWidget(self.spin_hue_range)
        row.addStretch()
        chroma_layout.addLayout(row)

        # Saturation min
        row = QHBoxLayout()
        self.lbl_sat = QLabel(tr("vr180_lbl_sat", "Min saturation (0–255):"))
        row.addWidget(self.lbl_sat)
        self.spin_sat = QSpinBox()
        self.spin_sat.setRange(0, 255)
        self.spin_sat.setValue(60)
        row.addWidget(self.spin_sat)
        row.addStretch()
        chroma_layout.addLayout(row)

        # Value min
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
            tr("vr180_tip_sam",
               "Uses Meta's Segment Anything Model to produce a cleaner person mask.\n"
               "Requires the SAM checkpoint file and takes longer to process.")
        )
        self.check_use_sam.toggled.connect(self._update_sam_group)
        sam_layout.addWidget(self.check_use_sam)

        # XSeg fast segmentation (alternative to SAM)
        self.check_use_xseg = QCheckBox(
            tr("vr180_check_xseg", "Enable XSeg segmentation (fast, ~1.95ms/frame)")
        )
        self.check_use_xseg.setToolTip(
            tr("vr180_tip_xseg",
               "XSeg is a lightweight U-Net (256×256) — 5× faster than SAM.\n"
               "Best for single-person green-screen footage.\n"
               "Requires XSeg_model.pth checkpoint.")
        )
        self.check_use_xseg.toggled.connect(self._update_xseg_group)
        sam_layout.addWidget(self.check_use_xseg)

        # XSeg checkpoint
        xseg_row = QHBoxLayout()
        self.lbl_xseg_ckpt = QLabel(tr("vr180_lbl_xseg_ckpt", "XSeg checkpoint (.pth):"))
        xseg_row.addWidget(self.lbl_xseg_ckpt)
        self.edit_xseg_ckpt = DropLineEdit()
        self.edit_xseg_ckpt.setPlaceholderText(tr("vr180_ph_xseg_ckpt", "Path to XSeg_model.pth"))
        xseg_row.addWidget(self.edit_xseg_ckpt)
        self.btn_browse_xseg = QPushButton(tr("btn_browse", "Browse"))
        self.btn_browse_xseg.clicked.connect(self._browse_xseg_checkpoint)
        xseg_row.addWidget(self.btn_browse_xseg)
        sam_layout.addLayout(xseg_row)

        # Batch size
        batch_row = QHBoxLayout()
        self.lbl_batch = QLabel(tr("vr180_lbl_batch", "GPU batch size:"))
        batch_row.addWidget(self.lbl_batch)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 64)
        self.spin_batch.setValue(8)
        self.spin_batch.setToolTip(tr("vr180_tip_batch", "Number of frames processed per GPU batch. Higher = faster but more VRAM."))
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
        row.addWidget(self.combo_sam_model)
        row.addStretch()
        sam_layout.addLayout(row)

        # Checkpoint path
        row = QHBoxLayout()
        self.lbl_ckpt = QLabel(tr("vr180_lbl_ckpt", "Checkpoint (.pth):"))
        row.addWidget(self.lbl_ckpt)
        self.edit_ckpt = DropLineEdit()
        self.edit_ckpt.setPlaceholderText(tr("vr180_ph_ckpt", "Path to SAM checkpoint file"))
        row.addWidget(self.edit_ckpt)
        self.btn_browse_ckpt = QPushButton(tr("btn_browse", "Browse"))
        self.btn_browse_ckpt.clicked.connect(self._browse_checkpoint)
        row.addWidget(self.btn_browse_ckpt)
        sam_layout.addLayout(row)

        # Download button + progress
        dl_row = QHBoxLayout()
        self.btn_download_sam = QPushButton(tr("vr180_btn_download_sam", "Download SAM checkpoint"))
        self.btn_download_sam.clicked.connect(self._download_sam)
        dl_row.addWidget(self.btn_download_sam)
        dl_row.addStretch()
        sam_layout.addLayout(dl_row)

        self.sam_progress = QProgressBar()
        self.sam_progress.setVisible(False)
        sam_layout.addWidget(self.sam_progress)

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

        self._update_xseg_group(False)

        # Keep references to SAM-dependent widgets for enable/disable
        self._sam_widgets = [
            self.lbl_sam_model, self.combo_sam_model,
            self.lbl_ckpt, self.edit_ckpt, self.btn_browse_ckpt,
            self.btn_download_sam,
            self.lbl_device, self.combo_device,
        ]
        self._update_sam_group(False)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _update_sam_group(self, enabled: bool):
        for w in self._sam_widgets:
            w.setEnabled(enabled)

    def _browse_checkpoint(self):
        path, _ = get_open_file_name(
            self,
            tr("vr180_lbl_ckpt", "SAM Checkpoint"),
            "",
            "PyTorch checkpoint (*.pth);;All files (*.*)",
        )
        if path:
            self.edit_ckpt.setText(path)

    def _update_xseg_group(self, enabled: bool):
        self.lbl_xseg_ckpt.setEnabled(enabled)
        self.edit_xseg_ckpt.setEnabled(enabled)
        self.btn_browse_xseg.setEnabled(enabled)

    def _browse_xseg_checkpoint(self):
        path, _ = get_open_file_name(
            self,
            tr("vr180_lbl_xseg_ckpt", "XSeg Checkpoint"),
            "",
            "PyTorch checkpoint (*.pth);;All files (*.*)",
        )
        if path:
            self.edit_xseg_ckpt.setText(path)

    def _download_sam(self):
        model_type = self.combo_sam_model.currentData()
        from pathlib import Path
        from app.core.system import resolve_project_root

        dest = resolve_project_root() / "engines" / f"sam_{model_type}.pth"
        if dest.exists():
            QMessageBox.information(
                self,
                tr("msg_success", "OK"),
                tr("vr180_ckpt_exists", f"Checkpoint already downloaded:\n{dest}"),
            )
            self.edit_ckpt.setText(str(dest))
            return

        self.btn_download_sam.setEnabled(False)
        self.sam_progress.setVisible(True)
        self.sam_progress.setValue(0)

        self._download_worker = SAMDownloadWorker(model_type, str(dest))
        self._download_worker.progress_signal.connect(self.sam_progress.setValue)
        self._download_worker.finished_signal.connect(self._on_download_finished)
        self._download_worker.start()

    def _on_download_finished(self, success: bool, message: str):
        self.btn_download_sam.setEnabled(True)
        self.sam_progress.setVisible(False)
        if success:
            self.edit_ckpt.setText(message)
            QMessageBox.information(self, tr("msg_success", "OK"),
                                    tr("vr180_download_ok", f"SAM checkpoint downloaded:\n{message}"))
        else:
            QMessageBox.critical(self, tr("msg_error", "Error"),
                                 tr("vr180_download_err", f"Download failed:\n{message}"))

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        return {
            "vr_format": "sbs" if self.radio_sbs.isChecked() else "tb",
            "eye": "left" if self.radio_left.isChecked() else "right",
            "hue_center": self.spin_hue.value(),
            "hue_range": self.spin_hue_range.value(),
            "sat_min": self.spin_sat.value(),
            "val_min": self.spin_val.value(),
            "use_sam": self.check_use_sam.isChecked(),
            "sam_model_type": self.combo_sam_model.currentData(),
            "sam_checkpoint": self.edit_ckpt.text().strip(),
            "device": self.combo_device.currentData(),
            "use_xseg": self.check_use_xseg.isChecked(),
            "xseg_checkpoint": self.edit_xseg_ckpt.text().strip(),
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
        if "sam_checkpoint" in params:
            self.edit_ckpt.setText(params["sam_checkpoint"])
        if "device" in params:
            idx = self.combo_device.findData(params["device"])
            if idx >= 0:
                self.combo_device.setCurrentIndex(idx)
        if "use_xseg" in params:
            self.check_use_xseg.setChecked(bool(params["use_xseg"]))
        if "xseg_checkpoint" in params:
            self.edit_xseg_ckpt.setText(params["xseg_checkpoint"])
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
            tr("vr180_desc",
               "Process VR 180 side-by-side or top-bottom green-screen footage.\n"
               "Extracts one eye, removes the green background with chroma-key, "
               "and optionally refines the person mask with SAM.")
        )
        self.radio_sbs.setText(tr("vr180_fmt_sbs", "Side-by-Side (SBS)"))
        self.radio_tb.setText(tr("vr180_fmt_tb", "Top-Bottom (TB)"))
        self.radio_left.setText(tr("vr180_eye_left", "Left Eye"))
        self.radio_right.setText(tr("vr180_eye_right", "Right Eye"))
        self.lbl_hue.setText(tr("vr180_lbl_hue", "Hue centre (HSV 0–180):"))
        self.lbl_hue_range.setText(tr("vr180_lbl_hue_range", "Hue tolerance (±):"))
        self.lbl_sat.setText(tr("vr180_lbl_sat", "Min saturation (0–255):"))
        self.lbl_val.setText(tr("vr180_lbl_val", "Min value/brightness (0–255):"))
        self.check_use_sam.setText(tr("vr180_check_sam", "Enable SAM (Segment Anything) refinement"))
        self.lbl_sam_model.setText(tr("vr180_lbl_sam_model", "SAM model:"))
        self.lbl_ckpt.setText(tr("vr180_lbl_ckpt", "Checkpoint (.pth):"))
        self.btn_browse_ckpt.setText(tr("btn_browse", "Browse"))
        self.btn_download_sam.setText(tr("vr180_btn_download_sam", "Download SAM checkpoint"))
        self.lbl_device.setText(tr("vr180_lbl_device", "Compute device:"))
        self.check_use_xseg.setText(tr("vr180_check_xseg", "Enable XSeg segmentation (fast, ~1.95ms/frame)"))
        self.lbl_xseg_ckpt.setText(tr("vr180_lbl_xseg_ckpt", "XSeg checkpoint (.pth):"))
        self.lbl_batch.setText(tr("vr180_lbl_batch", "GPU batch size:"))
