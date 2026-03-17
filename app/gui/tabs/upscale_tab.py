from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core.i18n import add_language_observer, tr
from app.core.upscale_engine import UpscaleEngine
from app.gui.widgets.resettable import make_resettable


class UpscaleTab(QWidget):
    """
    Tab for Upscale configuration.
    Dependencies and models are installed by the first-run setup.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = UpscaleEngine()
        self.init_ui()
        add_language_observer(self.retranslate_ui)

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.lbl_title = QLabel(tr("upscale_title"))
        self.lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(self.lbl_title)

        self.lbl_desc = QLabel(tr("upscale_desc"))
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setStyleSheet("color: #aaa; margin-bottom: 5px;")
        layout.addWidget(self.lbl_desc)

        # Settings Group
        self.settings_group = QGroupBox(tr("upscale_group_settings"))
        form_layout = QFormLayout()

        # Model Selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["RealESRGAN_x4plus"])
        self.lbl_model = QLabel(tr("upscale_lbl_model"))
        form_layout.addRow(self.lbl_model, self.model_combo)

        # Model Description
        self.model_desc = QLabel("")
        self.model_desc.setStyleSheet("color: #888; font-style: italic; margin-bottom: 5px;")
        self.model_desc.setWordWrap(True)
        form_layout.addRow("", self.model_desc)

        self.model_combo.currentIndexChanged.connect(self.on_model_changed)

        # Status Label
        self.status_label = QLabel("")
        self.lbl_status = QLabel(tr("upscale_lbl_status"))
        form_layout.addRow(self.lbl_status, self.status_label)

        # Scale Factor
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(
            [tr("upscale_scale_x4"), tr("upscale_scale_x2"), tr("upscale_scale_x1")]
        )
        self.scale_combo.setCurrentIndex(2)  # Default: x1 (enhance only)
        self.scale_combo.setFixedWidth(120)
        self.scale_combo.setToolTip(tr("upscale_tip_scale"))
        self.lbl_scale = QLabel(tr("upscale_lbl_scale"))
        form_layout.addRow(
            self.lbl_scale,
            make_resettable(self.scale_combo, lambda: self.scale_combo.setCurrentIndex(2)),
        )

        # Performance Profile
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(
            [
                tr("upscale_profile_safe"),
                tr("upscale_profile_quality"),
                tr("upscale_profile_speed"),
                tr("upscale_profile_ultimate"),
                tr("upscale_profile_custom"),
            ]
        )
        self.profile_combo.setFixedWidth(140)
        self.profile_combo.setToolTip(tr("upscale_tip_profile"))
        self.lbl_profile = QLabel(tr("upscale_lbl_profile"))
        form_layout.addRow(
            self.lbl_profile,
            make_resettable(self.profile_combo, lambda: self.profile_combo.setCurrentIndex(0)),
        )
        self.profile_combo.currentIndexChanged.connect(self.on_profile_changed)

        # Tile Size
        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(0, 4096)
        self.tile_spin.setValue(512)
        self.tile_spin.setSingleStep(128)
        self.tile_spin.setSuffix(" px")
        self.tile_spin.setFixedWidth(85)
        self.tile_spin.setToolTip(tr("upscale_tip_tile"))
        self.tile_spin.valueChanged.connect(self.on_manual_change)
        self.lbl_tile = QLabel(tr("upscale_lbl_tile"))
        form_layout.addRow(self.lbl_tile, self.tile_spin)

        # Face Enhance
        self.face_enhance = QCheckBox(tr("upscale_check_face"))
        self.face_enhance.setToolTip(tr("upscale_tip_face"))
        form_layout.addRow("", self.face_enhance)

        # GFPGAN backend
        self.lbl_gfpgan_backend = QLabel(tr("upscale_lbl_gfpgan_backend", "GFPGAN backend:"))
        self.combo_gfpgan_backend = QComboBox()
        self.combo_gfpgan_backend.addItem("Triton + CUDA Graph (fastest)", "cuda_graph")
        self.combo_gfpgan_backend.addItem("Triton FP16 (fast)", "triton")
        self.combo_gfpgan_backend.addItem("PyTorch FP32 (fallback)", "pytorch")
        self.combo_gfpgan_backend.setFixedWidth(230)
        self.combo_gfpgan_backend.setToolTip(
            tr("upscale_tip_gfpgan_backend", "Select inference backend for GFPGAN face restoration")
        )
        form_layout.addRow(
            self.lbl_gfpgan_backend,
            make_resettable(
                self.combo_gfpgan_backend, lambda: self.combo_gfpgan_backend.setCurrentIndex(0)
            ),
        )

        # ESRGAN backend
        self.lbl_esrgan_backend = QLabel(tr("upscale_lbl_esrgan_backend", "ESRGAN backend:"))
        self.combo_esrgan_backend = QComboBox()
        self.combo_esrgan_backend.addItem("ORT TensorRT (fastest, requires build)", "ort_trt")
        self.combo_esrgan_backend.addItem("ORT CUDA EP (fast)", "ort_cuda")
        self.combo_esrgan_backend.addItem("PyTorch + torch.compile (default)", "torch_compile")
        self.combo_esrgan_backend.addItem("PyTorch FP16 (safe)", "torch_fp16")
        self.combo_esrgan_backend.setCurrentIndex(2)
        self.combo_esrgan_backend.setFixedWidth(260)
        self.combo_esrgan_backend.setToolTip(
            tr(
                "upscale_tip_esrgan_backend",
                "TRT builds engine on first run (~60s). Result is cached for future runs.",
            )
        )
        form_layout.addRow(
            self.lbl_esrgan_backend,
            make_resettable(
                self.combo_esrgan_backend, lambda: self.combo_esrgan_backend.setCurrentIndex(2)
            ),
        )

        # FP16
        self.fp16_check = QCheckBox(tr("upscale_lbl_fp16"))
        self.fp16_check.setToolTip(tr("upscale_tip_fp16"))
        self.fp16_check.setChecked(True)
        self.fp16_check.toggled.connect(self.on_manual_change)
        self.lbl_fp16 = QLabel(tr("upscale_lbl_fp16"))
        form_layout.addRow(self.lbl_fp16, self.fp16_check)

        self.settings_group.setLayout(form_layout)
        layout.addWidget(self.settings_group)
        layout.addStretch()

        self._updating_profile = False
        self.on_model_changed()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def on_profile_changed(self):
        if self._updating_profile:
            return
        self._updating_profile = True
        idx = self.profile_combo.currentIndex()
        if idx == 0:  # Safe
            self.tile_spin.setValue(512)
            self.fp16_check.setChecked(True)
        elif idx == 1:  # Quality Max
            self.tile_spin.setValue(512)
            self.fp16_check.setChecked(False)
        elif idx == 2:  # Speed
            self.tile_spin.setValue(0)
            self.fp16_check.setChecked(True)
        elif idx == 3:  # Ultimate
            self.tile_spin.setValue(0)
            self.fp16_check.setChecked(False)
        self._updating_profile = False

    def on_manual_change(self):
        if not self._updating_profile:
            self.profile_combo.setCurrentIndex(4)  # Custom

    def on_model_changed(self):
        self.update_model_desc()
        self.check_model_status()

    def update_model_desc(self):
        txt = self.model_combo.currentText()
        desc = ""
        if "x4plus_anime" in txt:
            desc = tr("upscale_desc_anime")
        elif "x4plus" in txt:
            desc = tr("upscale_desc_x4plus")
        elif "x4net" in txt:
            desc = tr("upscale_desc_x4net")
        self.model_desc.setText(desc)

    def check_model_status(self):
        model = self.model_combo.currentText()
        if self.engine.check_model_availability(model):
            self.status_label.setText(tr("upscale_status_available"))
            self.status_label.setStyleSheet("color: #44cc44;")
        else:
            self.status_label.setText(tr("upscale_status_missing_model"))
            self.status_label.setStyleSheet("color: #cc8844;")

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def get_params(self):
        return {
            "enabled": True,
            "model_name": self.model_combo.currentText(),
            "tile": self.tile_spin.value(),
            "target_scale": self.get_scale_factor(),
            "face_enhance": self.face_enhance.isChecked(),
            "fp16": self.fp16_check.isChecked(),
            "gfpgan_backend": self.combo_gfpgan_backend.currentData(),
            "esrgan_backend": self.combo_esrgan_backend.currentData(),
        }

    def get_scale_factor(self):
        idx = self.scale_combo.currentIndex()
        if idx == 1:
            return 2
        if idx == 2:
            return 1
        return 4

    def set_params(self, params):
        if not params:
            return
        if "tile" in params:
            self.tile_spin.setValue(params["tile"])
        if "fp16" in params:
            self.fp16_check.setChecked(params["fp16"])
        if "model_name" in params:
            self.model_combo.setCurrentText(params["model_name"])
        if "gfpgan_backend" in params:
            idx = self.combo_gfpgan_backend.findData(params["gfpgan_backend"])
            if idx >= 0:
                self.combo_gfpgan_backend.setCurrentIndex(idx)
        if "esrgan_backend" in params:
            idx = self.combo_esrgan_backend.findData(params["esrgan_backend"])
            if idx >= 0:
                self.combo_esrgan_backend.setCurrentIndex(idx)

    def get_state(self):
        return self.get_params()

    def set_state(self, state):
        self.set_params(state)

    def retranslate_ui(self):
        self.lbl_title.setText(tr("upscale_title"))
        self.lbl_desc.setText(tr("upscale_desc"))
        self.settings_group.setTitle(tr("upscale_group_settings"))
        self.lbl_model.setText(tr("upscale_lbl_model"))
        self.lbl_status.setText(tr("upscale_lbl_status"))
        self.lbl_scale.setText(tr("upscale_lbl_scale"))
        self.scale_combo.setToolTip(tr("upscale_tip_scale"))
        self.scale_combo.setItemText(0, tr("upscale_scale_x4"))
        self.scale_combo.setItemText(1, tr("upscale_scale_x2"))
        self.scale_combo.setItemText(2, tr("upscale_scale_x1"))
        self.lbl_profile.setText(tr("upscale_lbl_profile", "Performance Profile"))
        self.lbl_tile.setText(tr("upscale_lbl_tile"))
        self.tile_spin.setToolTip(tr("upscale_tip_tile"))
        self.face_enhance.setText(tr("upscale_check_face"))
        self.face_enhance.setToolTip(tr("upscale_tip_face"))
        self.lbl_fp16.setText(tr("upscale_lbl_fp16"))
        self.lbl_gfpgan_backend.setText(tr("upscale_lbl_gfpgan_backend", "GFPGAN backend:"))
        self.lbl_esrgan_backend.setText(tr("upscale_lbl_esrgan_backend", "ESRGAN backend:"))
        self.check_model_status()
        self.update_model_desc()
