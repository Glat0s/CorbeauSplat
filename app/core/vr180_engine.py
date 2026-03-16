"""
VR 180 green-screen processing engine.

Pipeline:
  1. Extract frames from a VR 180 video with FFmpeg, cropping to the
     requested eye half (SBS or Top-Bottom format).
  2. Remove the green screen using HSV chroma-key (OpenCV).
  3. Optionally refine the person mask with SAM (Segment Anything Model).
  4. Write masked PNG frames (RGBA) ready for COLMAP.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

from .base_engine import BaseEngine
from .system import resolve_binary, is_windows


class VR180Engine(BaseEngine):
    """Engine for VR 180 green-screen footage → segmented COLMAP images."""

    FORMATS = {"sbs": "Side-by-Side (SBS)", "tb": "Top-Bottom (TB)"}
    EYES = {"left": "Left Eye", "right": "Right Eye"}

    def __init__(self, logger_callback=None):
        super().__init__("VR180", logger_callback)
        self.ffmpeg_bin = resolve_binary("ffmpeg") or "ffmpeg"

    # ------------------------------------------------------------------
    # Dependency checks
    # ------------------------------------------------------------------

    def is_cv2_available(self) -> bool:
        try:
            import cv2  # noqa: F401
            return True
        except ImportError:
            return False

    def is_sam_available(self) -> bool:
        try:
            from segment_anything import sam_model_registry  # noqa: F401
            return True
        except ImportError:
            return False

    def is_installed(self) -> bool:
        """Minimum requirement: OpenCV must be importable."""
        return self.is_cv2_available()

    # ------------------------------------------------------------------
    # Step 1 – frame extraction
    # ------------------------------------------------------------------

    def _get_video_dimensions(self, video_path: str):
        """Returns (width, height) of the video, or (None, None) on failure."""
        null_sink = "NUL" if is_windows() else "/dev/null"
        cmd = [self.ffmpeg_bin, "-i", video_path, "-hide_banner", "-f", "null", null_sink]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            for line in result.stderr.splitlines():
                if "Video:" in line:
                    m = re.search(r"(\d{2,5})x(\d{2,5})", line)
                    if m:
                        return int(m.group(1)), int(m.group(2))
        except Exception as e:
            self.log(f"Could not read video dimensions: {e}")
        return None, None

    def _build_crop_filter(self, width: int, height: int, fmt: str, eye: str) -> str:
        """Returns an FFmpeg crop= filter string for the requested eye."""
        if fmt == "sbs":
            half_w = width // 2
            x_offset = 0 if eye == "left" else half_w
            return f"crop={half_w}:{height}:{x_offset}:0"
        elif fmt == "tb":
            half_h = height // 2
            y_offset = 0 if eye == "left" else half_h
            return f"crop={width}:{half_h}:0:{y_offset}"
        else:
            self.log(f"Unknown VR 180 format: {fmt}")
            return ""

    def extract_frames(
        self,
        video_path: str,
        output_dir: Path,
        fps: float = 5.0,
        fmt: str = "sbs",
        eye: str = "left",
        progress_callback=None,
        check_cancel=None,
    ) -> bool:
        """Extract and crop frames from a VR 180 video."""
        output_dir.mkdir(parents=True, exist_ok=True)

        width, height = self._get_video_dimensions(video_path)
        if not width:
            self.log("Failed to determine video dimensions.")
            return False

        crop_filter = self._build_crop_filter(width, height, fmt, eye)
        if not crop_filter:
            return False

        output_pattern = str(output_dir / "frame_%04d.png")

        cmd = [self.ffmpeg_bin]
        # Use CUDA hwaccel for decode only; keep software filters for crop+fps
        if is_windows():
            cmd.extend(["-hwaccel", "cuda"])
        cmd.extend([
            "-i", video_path,
            "-vf", f"fps={fps},{crop_filter}",
            "-pix_fmt", "rgb24",
            "-y",
            output_pattern,
        ])

        self.log(
            f"Extracting VR 180 frames "
            f"(eye={eye}, format={fmt.upper()}, fps={fps})..."
        )

        def _parser(line: str):
            if "frame=" in line:
                self.log(line)
                if progress_callback:
                    try:
                        f_num = int(line.split("frame=")[1].strip().split()[0])
                        progress_callback(min(28, max(1, f_num // 5)))
                    except Exception:
                        pass

        rc = self._execute_command(cmd, line_callback=_parser)
        if check_cancel and check_cancel():
            return False
        return rc == 0

    # ------------------------------------------------------------------
    # Step 2 – chroma-key green-screen removal
    # ------------------------------------------------------------------

    def remove_green_screen(
        self,
        image_path: Path,
        output_path: Path,
        hue_center: int = 60,
        hue_range: int = 25,
        sat_min: int = 60,
        val_min: int = 40,
        blur_px: int = 3,
    ) -> bool:
        """
        Remove the green screen using HSV chroma-keying.
        Writes an RGBA PNG where transparent pixels were green.

        hue_center  – HSV hue centre (0–180 in OpenCV). 60 = green.
        hue_range   – ± tolerance around hue_center.
        sat_min     – minimum saturation to be considered "coloured".
        val_min     – minimum value (brightness).
        blur_px     – Gaussian kernel size for edge smoothing (odd int).
        """
        try:
            import cv2
            import numpy as np

            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                self.log(f"Could not read {image_path.name}")
                return False

            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

            lo = np.array([max(0, hue_center - hue_range), sat_min, val_min], dtype=np.uint8)
            hi = np.array([min(180, hue_center + hue_range), 255, 255], dtype=np.uint8)
            green_mask = cv2.inRange(hsv, lo, hi)

            # Morphological cleanup
            k = np.ones((3, 3), np.uint8)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, k)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, k)

            # Edge smoothing
            if blur_px > 1:
                bk = blur_px if blur_px % 2 == 1 else blur_px + 1
                green_mask = cv2.GaussianBlur(green_mask, (bk, bk), 0)
                _, green_mask = cv2.threshold(green_mask, 127, 255, cv2.THRESH_BINARY)

            alpha = cv2.bitwise_not(green_mask)  # subject=255, background=0
            b, g, r = cv2.split(bgr)
            rgba = cv2.merge([b, g, r, alpha])
            cv2.imwrite(str(output_path.with_suffix(".png")), rgba)
            return True

        except Exception as e:
            self.log(f"Chroma-key error on {image_path.name}: {e}")
            return False

    # ------------------------------------------------------------------
    # Step 3 – SAM person segmentation (optional)
    # ------------------------------------------------------------------

    def segment_with_sam(
        self,
        image_path: Path,
        output_path: Path,
        sam_checkpoint: str,
        model_type: str = "vit_b",
        device: str = "cuda",
    ) -> bool:
        """
        Refine the person mask with SAM using centre-point prompts.
        The image at image_path may already have an alpha channel from the
        chroma-key step; SAM will AND its mask with the existing alpha.
        """
        if not self.is_sam_available():
            self.log("segment_anything not installed – skipping SAM.")
            return False

        checkpoint_path = Path(sam_checkpoint) if sam_checkpoint else None
        if not checkpoint_path or not checkpoint_path.exists():
            self.log(f"SAM checkpoint not found: {sam_checkpoint} – skipping.")
            return False

        try:
            import cv2
            import numpy as np
            import torch
            from segment_anything import sam_model_registry, SamPredictor

            img_data = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img_data is None:
                return False

            if img_data.shape[2] == 4:
                bgr = img_data[:, :, :3]
                prior_alpha = img_data[:, :, 3]
            else:
                bgr = img_data
                prior_alpha = None

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
            sam.to(device=device)

            predictor = SamPredictor(sam)
            predictor.set_image(rgb)

            # Centre-point prompts: centre, upper-centre, lower-centre
            pts = np.array([[w // 2, h // 2], [w // 2, h // 3], [w // 2, 2 * h // 3]])
            labels = np.ones(len(pts), dtype=np.int32)

            masks, scores, _ = predictor.predict(
                point_coords=pts, point_labels=labels, multimask_output=True
            )
            best = masks[int(np.argmax(scores))]
            sam_alpha = (best * 255).astype(np.uint8)

            # Combine with existing alpha (from chroma key)
            if prior_alpha is not None:
                combined = cv2.bitwise_and(prior_alpha, sam_alpha)
            else:
                combined = sam_alpha

            # Morphological cleanup
            k = np.ones((5, 5), np.uint8)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)

            b, g, r = cv2.split(bgr)
            result = cv2.merge([b, g, r, combined])
            out_png = output_path.with_suffix(".png")
            cv2.imwrite(str(out_png), result)
            return True

        except Exception as e:
            self.log(f"SAM error on {image_path.name}: {e}")
            return False

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        output_dir: Path,
        params: dict,
        progress_callback=None,
        log_callback=None,
        status_callback=None,
        check_cancel=None,
    ) -> bool:
        """
        Run the full VR 180 pipeline and write segmented frames to output_dir.
        params keys:
          vr_format       – "sbs" or "tb"
          eye             – "left" or "right"
          fps             – frame extraction rate (default 5)
          use_sam         – bool (default False)
          sam_checkpoint  – path string to SAM .pth file
          sam_model_type  – "vit_b" | "vit_l" | "vit_h"
          hue_center      – chroma-key HSV hue centre (default 60)
          hue_range       – chroma-key hue tolerance (default 25)
          sat_min         – chroma-key min saturation (default 60)
          val_min         – chroma-key min value (default 40)
          device          – "cuda" | "cpu"
        """
        def _log(msg):
            self.log(msg)
            if log_callback:
                log_callback(msg)

        def _status(msg):
            if status_callback:
                status_callback(msg)

        _log("=== VR 180 Processing Pipeline ===")
        _log(f"Video: {video_path}")

        fmt = params.get("vr_format", "sbs")
        eye = params.get("eye", "left")
        fps = float(params.get("fps", 5.0))
        use_sam = params.get("use_sam", False)
        sam_checkpoint = params.get("sam_checkpoint", "")
        sam_model_type = params.get("sam_model_type", "vit_b")
        hue_center = int(params.get("hue_center", 60))
        hue_range = int(params.get("hue_range", 25))
        sat_min = int(params.get("sat_min", 60))
        val_min = int(params.get("val_min", 40))
        device = params.get("device", "cuda" if is_windows() else "cpu")

        _log(f"Format: {fmt.upper()} | Eye: {eye} | FPS: {fps}")
        _log(f"Green screen removal: hue={hue_center}±{hue_range}, sat≥{sat_min}, val≥{val_min}")
        if use_sam:
            _log(f"SAM refinement: model={sam_model_type}, device={device}")

        # ---- Step 1: extract frames ----
        frames_dir = output_dir / "_frames_raw"
        _status("Extracting VR 180 frames...")

        if not self.is_cv2_available():
            _log("OpenCV not available. Install opencv-python.")
            return False

        ok = self.extract_frames(
            video_path, frames_dir, fps, fmt, eye,
            progress_callback=progress_callback,
            check_cancel=check_cancel,
        )
        if not ok:
            _log("Frame extraction failed.")
            return False

        if check_cancel and check_cancel():
            return False

        frame_files = sorted(frames_dir.glob("*.png"))
        total = len(frame_files)
        if total == 0:
            _log("No frames extracted from video.")
            return False

        _log(f"Extracted {total} frames. Starting segmentation...")
        if progress_callback:
            progress_callback(30)

        # ---- Steps 2 & 3: chroma-key + optional SAM ----
        for i, frame_path in enumerate(frame_files):
            if check_cancel and check_cancel():
                _log("Cancelled by user.")
                return False

            out_path = (output_dir / frame_path.stem).with_suffix(".png")

            chroma_ok = self.remove_green_screen(
                frame_path, out_path,
                hue_center=hue_center,
                hue_range=hue_range,
                sat_min=sat_min,
                val_min=val_min,
            )
            if not chroma_ok:
                _log(f"Chroma-key failed for {frame_path.name}; using original.")
                shutil.copy2(frame_path, out_path)

            if use_sam:
                sam_ok = self.segment_with_sam(
                    out_path, out_path,
                    sam_checkpoint=sam_checkpoint,
                    model_type=sam_model_type,
                    device=device,
                )
                if not sam_ok:
                    _log(f"SAM skipped for {frame_path.name}.")

            if progress_callback:
                pct = 30 + int((i + 1) / total * 65)
                progress_callback(pct)

            if i % 20 == 0 or i == total - 1:
                _status(f"Segmenting frame {i + 1}/{total}...")

        # Clean up raw frames
        shutil.rmtree(frames_dir, ignore_errors=True)

        if progress_callback:
            progress_callback(100)

        _log(f"✅ VR 180 processing complete. {total} frames saved to {output_dir}")
        return True
