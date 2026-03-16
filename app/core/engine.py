import json
import os
import platform
import shutil
from pathlib import Path

import send2trash

from .base_engine import BaseEngine
from .i18n import tr
from .system import get_optimal_threads, is_windows, resolve_binary

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class ColmapEngine(BaseEngine):
    """COLMAP execution engine independent of the graphical interface"""

    def __init__(
        self,
        params,
        input_path,
        output_path,
        input_type,
        fps,
        project_name="Untitled",
        logger_callback=None,
        progress_callback=None,
        status_callback=None,
        check_cancel_callback=None,
    ):
        super().__init__("COLMAP", logger_callback)
        self.params = params
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.input_type = input_type
        self.fps = fps
        self.project_name = project_name
        self.num_threads = get_optimal_threads()
        self._current_process = None
        self.progress = progress_callback if progress_callback else lambda x: None
        self.status = status_callback if status_callback else lambda x: None
        self.check_cancel = check_cancel_callback if check_cancel_callback else lambda: False

        # Resolve binaries
        self.ffmpeg_bin = resolve_binary("ffmpeg") or "ffmpeg"
        self.colmap_bin = resolve_binary("colmap") or "colmap"
        self.glomap_bin = resolve_binary("glomap") or "glomap"

        if is_windows():
            self.log(f"Windows detected — {self.num_threads} threads available")
        self.log(f"Binaries: {self.colmap_bin}, {self.ffmpeg_bin}, {self.glomap_bin}")

    # log method inherited from BaseEngine

    @property
    def project_path(self):
        """Alias for output_path used by Workers and UI"""
        return self.output_path

    def is_cancelled(self):
        return self.check_cancel()

    def run(self):
        """Runs the full pipeline"""
        try:
            # 1. Validation and Directory Setup
            setup_result = self._validate_and_setup_paths()
            if not setup_result:
                return False, "Path validation error"
            project_dir, images_dir, checkpoints_dir = setup_result

            # 2. Preparation Input (Extraction/Copy, Upscale, Normalize)
            if not self._process_input(project_dir, images_dir):
                if self.is_cancelled():
                    return False, tr("USER_CANCELLED")
                return False, "Error during input preparation"

            # 3. Pipeline COLMAP (Features, Matching, Mapper, Undistort)
            pipeline_result, msg = self._run_reconstruction_pipeline(project_dir, images_dir)
            return pipeline_result, msg

        except Exception as e:
            if self.is_cancelled():
                return False, "Stopped by user"
            return False, str(e)

    def _validate_and_setup_paths(self):
        # [AUDIT] OWASP-A01 : Path Traversal prevention
        safe_output = self.validate_path(str(self.output_path))
        if not safe_output:
            self.log("Unsafe output path")
            return None
        self.output_path = safe_output

        if ".." in self.project_name or "/" in self.project_name or "\\" in self.project_name:
            self.log("Invalid project name")
            return None

        project_dir = self.output_path / self.project_name
        images_dir = project_dir / "images"
        checkpoints_dir = project_dir / "checkpoints"

        project_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.log(f"Preparing project in: {project_dir}")

        raw_input = str(self.input_path)
        if "|" in raw_input:
            self.log("Validating multiple input paths...")
            for p in raw_input.split("|"):
                if not self.validate_path(p.strip()):
                    self.log(f"Unsafe input path: {p}")
                    return None

            first_path = Path(raw_input.split("|")[0].strip())
            if not first_path.exists():
                self.log(f"Input not found: {first_path}")
                return None
        else:
            if not self.validate_path(raw_input):
                self.log(f"Unsafe input path: {raw_input}")
                return None
            if not self.input_path.exists():
                self.log(f"Input not found: {self.input_path}")
                return None

        return project_dir, images_dir, checkpoints_dir

    def _process_input(self, project_dir, images_dir):
        self.status(tr("status_prep_images", "Preparing images..."))
        if not self._prepare_images(images_dir):
            return False

        upscale_conf = getattr(self, "upscale_config", None)
        if upscale_conf and upscale_conf.get("active", False):
            self.status(tr("status_upscaling", "Upscaling images..."))
            if not self._run_upscale(project_dir, images_dir):
                return False

        return self._check_and_normalize_resolution(images_dir)

    def _run_reconstruction_pipeline(self, project_dir, images_dir):
        database_path = project_dir / "database.db"
        sparse_dir = project_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)

        self.progress(25)

        if self.is_cancelled():
            return False, tr("USER_CANCELLED")
        self.status(tr("status_feature_extraction", "Feature extraction..."))
        if not self.feature_extraction(str(database_path), str(images_dir)):
            return False, "Feature extraction failed"

        self.progress(50)

        if self.is_cancelled():
            return False, tr("USER_CANCELLED")
        self.status(tr("status_feature_matching", "Feature matching..."))
        if not self.feature_matching(str(database_path)):
            return False, "Feature matching failed"

        self.progress(75)

        if self.is_cancelled():
            return False, tr("USER_CANCELLED")
        self.status(tr("status_reconstruction", "3D Reconstruction (Mapper)..."))
        if not self.mapper(str(database_path), str(images_dir), str(sparse_dir)):
            return False, "Reconstruction failed"

        self.progress(90)

        if self.params.undistort_images:
            if self.is_cancelled():
                return False, tr("USER_CANCELLED")
            dense_dir = project_dir / "dense"
            dense_dir.mkdir(exist_ok=True)
            self.status(tr("status_undistorting", "Undistorting images..."))
            if not self.image_undistorter(str(images_dir), str(sparse_dir), str(dense_dir)):
                return False, "Undistortion failed"

        self.progress(95)

        if not self.is_cancelled():
            self.status(tr("status_ready", "Processing complete!"))
            self.create_brush_config(project_dir, images_dir, sparse_dir)
            self.progress(100)
            return True, f"Dataset created: {project_dir}"

        return False, "Stopped by user"

    def _prepare_images(self, images_dir: Path):
        """Handles video extraction or image copying"""
        if self.input_type == "video":
            if self.is_cancelled():
                return False

            # Identify video paths (either pipe-separated files or a folder containing videos)
            video_paths = []
            if self.input_path.is_dir():
                # Search all videos in the folder
                supported_exts = {".mp4", ".mov", ".avi", ".mkv"}
                video_paths = [
                    f
                    for f in self.input_path.rglob("*")
                    if f.is_file() and f.suffix.lower() in supported_exts
                ]
                video_paths.sort()
            else:
                # Single or multiple files separated by '|'
                video_paths = [
                    Path(p.strip()) for p in str(self.input_path).split("|") if p.strip()
                ]

            total_videos = len(video_paths)

            if total_videos == 0:
                self.log(f"No video found in: {self.input_path}")
                return False

            for i, video_path in enumerate(video_paths):
                if self.is_cancelled():
                    return False

                if not video_path.exists():
                    self.log(f"Warning: Video not found: {video_path}")
                    continue

                # Prefix based on filename
                base_name = video_path.stem
                prefix = "".join([c for c in base_name if c.isalnum() or c in ("_", "-")])

                self.log(f"Extracting video ({i+1}/{total_videos}): {base_name}")

                if not self.extract_frames_from_video(str(video_path), images_dir, prefix=prefix):
                    self.log(f"Video extraction failed: {base_name}")
                    return False
            return True
        else:
            # Copy images (recursive to support extractor outputs)
            self.log("Copying source images to working folder...")

            if self.input_path.resolve() == images_dir.resolve():
                self.log("Images already in destination folder. Copy skipped.")
                return True

            try:
                # Recursive search for image files, ignoring masks (*.mask.png)
                src_files = [
                    f
                    for f in self.input_path.rglob("*")
                    if f.is_file()
                    and f.suffix.lower() in _IMAGE_EXTS
                    and not f.name.lower().endswith(".mask.png")
                ]

                total_files = len(src_files)
                self.log(f"{total_files} images found.")

                if total_files == 0:
                    return True  # Continue, maybe they are already there or handled otherwise

                # Copy
                for i, file_path in enumerate(src_files):
                    if self.is_cancelled():
                        return False

                    # Avoid using relative subfolder path to simplify COLMAP structure
                    # unless keeping hierarchy is desired. COLMAP often prefers a flat folder.
                    target_path = images_dir / file_path.name

                    # If name collision (e.g. frame_001.jpg in two different folders), add prefix
                    if target_path.exists():
                        target_path = images_dir / f"{file_path.parent.name}_{file_path.name}"

                    shutil.copy2(file_path, target_path)

                    if i % 10 == 0 or i == total_files - 1:
                        p = 5 + int((i / total_files) * 15)  # 5-20% range
                        self.progress(p)
                        self.status(f"Copying images: {i+1} / {total_files}")

                self.log(f"✅ {total_files} images copied to {images_dir}")
                return True
            except Exception as e:
                self.log(f"Image copy error: {e}")
                return False

    def _run_upscale(self, project_dir: Path, images_dir: Path):
        """Handles upscaling"""
        self.log(f"\n{'='*60}\nUpscaling (Super-Resolution)\n{'='*60}")
        if self.is_cancelled():
            return False

        try:
            from app.core.upscale_engine import UpscaleEngine

            upscaler = UpscaleEngine(logger_callback=self.log)

            if not upscaler.is_installed():
                self.log("WARNING: Upscale enabled but dependencies missing. Skipped.")
                return True  # Non-fatal

            # 1. Move original images to "images_src"
            images_sources_dir = project_dir / "images_src"

            if not images_sources_dir.exists():
                self.log(f"Moving originals to {images_sources_dir}...")
                shutil.move(str(images_dir), str(images_sources_dir))
                images_dir.mkdir(parents=True, exist_ok=True)

                # Perform Upscale
                model_name = self.upscale_config.get("model_name", "RealESRGAN_x4plus")
                tile_size = self.upscale_config.get("tile", 0)
                target_scale = self.upscale_config.get("target_scale", 4)
                face_enhance = self.upscale_config.get("face_enhance", False)
                fp16 = self.upscale_config.get("fp16", False)

                upsampler = upscaler.load_model(
                    model_name=model_name, tile=tile_size, target_scale=target_scale, half=fp16
                )
                if not upsampler:
                    self.log("Failed to load Upscale model")
                    return False

                self.log(f"Upscaling (x{target_scale})...")
                files = sorted(
                    [
                        f
                        for f in images_sources_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in (".jpg", ".png", ".jpeg")
                    ]
                )

                total = len(files)
                for i, f_path in enumerate(files):
                    if self.is_cancelled():
                        return False
                    out_p = images_dir / f_path.name

                    upscaler.upscale_image(
                        str(f_path), str(out_p), upsampler, face_enhance=face_enhance
                    )
                    if i % 5 == 0:
                        self.log(f"Upscale {i+1}/{total}...")

                self.log("Upscale complete.")
            else:
                self.log("'images_src' folder exists. Assuming upscale already done.")

            return True

        except Exception as e:
            self.log(f"Upscale error: {e}")
            return False

    def _check_and_normalize_resolution(self, images_dir: Path) -> bool:
        """
        Checks that all images have the same resolution.
        If not, resizes all to the smallest resolution found.
        """
        self.log(f"\n{'='*60}\nChecking image resolution\n{'='*60}")

        if not getattr(self, "_cv2_loaded", False):
            self.log("⚠️ OpenCV not available — resolution check skipped.")
            return True

        import cv2

        files = sorted(
            [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS]
        )

        if len(files) < 2:
            return True

        self.log(f"Analyzing {len(files)} images...")

        # First pass: read dimensions (grayscale = 1 channel, faster)
        sizes = {}  # Path -> (w, h)
        for f in files:
            if self.is_cancelled():
                return False
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                self.log(f"⚠️ Could not read: {f.name}")
                continue
            h, w = img.shape
            sizes[f] = (w, h)

        if not sizes:
            return True

        unique_sizes = set(sizes.values())
        if len(unique_sizes) == 1:
            w, h = next(iter(unique_sizes))
            self.log(f"✅ Uniform resolution: {w}×{h} px")
            return True

        # Target = smallest resolution
        min_w = min(s[0] for s in unique_sizes)
        min_h = min(s[1] for s in unique_sizes)
        images_to_resize = [(f, s) for f, s in sizes.items() if s != (min_w, min_h)]

        self.log(f"⚠️ {len(unique_sizes)} different resolutions detected.")
        self.log(f"Resizing {len(images_to_resize)} images → {min_w}×{min_h} px")

        # Second pass: resize only images that differ from target
        for i, (f, _) in enumerate(images_to_resize):
            if self.is_cancelled():
                return False
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is None:
                self.log(f"⚠️ Could not read: {f.name}")
                continue
            resized = cv2.resize(img, (min_w, min_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(f), resized)
            if (i + 1) % 10 == 0 or (i + 1) == len(images_to_resize):
                self.log(f"Resizing: {i+1}/{len(images_to_resize)}")
                self.status(f"Adjusting size: {i+1} / {len(images_to_resize)}")

        self.log(f"✅ {len(images_to_resize)} images resized to {min_w}×{min_h} px")
        return True

    def extract_frames_from_video(self, video_path: str, images_dir: Path, prefix=None):
        """Optimized video frame extraction via Template Method"""
        base_name = Path(video_path).stem
        self.log(f"\n{'='*60}\nExtracting frames: {Path(video_path).name}\n{'='*60}")
        images_dir.mkdir(parents=True, exist_ok=True)

        # Output pattern
        if prefix:
            output_pattern = images_dir / f"{prefix}_%04d.jpg"
        else:
            output_pattern = images_dir / "frame_%04d.jpg"

        cmd = [self.ffmpeg_bin]
        if is_windows():
            # CUDA hwaccel for decode only; software fps filter runs in CPU memory
            cmd.extend(["-hwaccel", "cuda"])

        cmd.extend(
            ["-i", video_path, "-vf", f"fps={self.fps}", "-qscale:v", "2", str(output_pattern)]
        )

        def _ffmpeg_parser(line_str):
            if "frame=" in line_str or "error" in line_str.lower():
                self.log(line_str)
                if "frame=" in line_str:
                    try:
                        f_num = line_str.split("frame=")[1].strip().split()[0]
                        self.status(f"Extracting {base_name}: frame {f_num}")
                    except Exception:  # noqa: BLE001
                        pass

        try:
            returncode = self._execute_command(cmd, line_callback=_ffmpeg_parser)
            if self.is_cancelled():
                return None

            if returncode == 0:
                num_frames = len([f for f in images_dir.iterdir() if f.suffix == ".jpg"])
                self.log(f"{num_frames} frames extracted")
                return True
            else:
                self.log("Extraction error")
                return None
        except Exception as e:
            self.log(f"Error: {str(e)}")
            return False

    def run_command(self, cmd, description, status_prefix=None):
        """Runs a command via centralized Template Method"""
        self.log(f"\n{'='*60}\n{description}\n{'='*60}")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.num_threads)
        env["OPENBLAS_NUM_THREADS"] = str(self.num_threads)

        def _colmap_parser(line_str):
            self.log(line_str)
            if status_prefix:
                if "Processed file" in line_str:
                    parts = line_str.split("Processed file")
                    if len(parts) > 1:
                        self.status(f"{status_prefix}: image {parts[1].strip()}")
                elif "Matching block" in line_str:
                    parts = line_str.split("Matching block")
                    if len(parts) > 1:
                        self.status(f"{status_prefix}: block {parts[1].strip()}")
                elif "Registering image" in line_str:
                    parts = line_str.split("Registering image")
                    if len(parts) > 1:
                        img_info = parts[1].split("(")[0].strip()
                        self.status(f"{status_prefix}: adding image {img_info}")
                elif "Bundle adjustment report" in line_str:
                    self.status(f"{status_prefix}: global optimization...")
                elif "Undistorting image" in line_str:
                    parts = line_str.split("Undistorting image")
                    if len(parts) > 1:
                        self.status(f"{status_prefix}: image {parts[1].strip()}")

        try:
            returncode = self._execute_command(cmd, env=env, line_callback=_colmap_parser)
            if self.is_cancelled():
                return False

            if returncode == 0:
                self.log(f"{description} complete")
                return True
            else:
                self.log(f"{description} failed")
                return False

        except FileNotFoundError:
            install_hint = "Download COLMAP from https://github.com/colmap/colmap/releases"
            self.log(f"COLMAP not found. Install with: {install_hint}")
            return False

    def feature_extraction(self, database_path, images_dir):
        cmd = [
            self.colmap_bin,
            "feature_extractor",
            "--database_path",
            database_path,
            "--image_path",
            images_dir,
            "--ImageReader.camera_model",
            self.params.camera_model,
            "--ImageReader.single_camera",
            "1" if self.params.single_camera else "0",
            "--FeatureExtraction.num_threads",
            str(self.num_threads),
            "--SiftExtraction.max_image_size",
            str(self.params.max_image_size),
            "--SiftExtraction.max_num_features",
            str(self.params.max_num_features),
            "--SiftExtraction.estimate_affine_shape",
            "1" if self.params.estimate_affine_shape else "0",
            "--SiftExtraction.domain_size_pooling",
            "1" if self.params.domain_size_pooling else "0",
        ]
        use_gpu = (
            is_windows()
            and getattr(self.params, "use_gpu_sift", True)
            and not getattr(self.params, "force_cpu", False)
        )
        if use_gpu:
            cmd.extend(["--SiftExtraction.use_gpu", "1", "--SiftExtraction.gpu_index", "0"])
        return self.run_command(cmd, "Feature Extraction", status_prefix="Analysis")

    def feature_matching(self, database_path):
        if self.params.matcher_type == "sequential":
            cmd = [
                self.colmap_bin,
                "sequential_matcher",
                "--database_path",
                database_path,
                "--FeatureMatching.num_threads",
                str(self.num_threads),
                "--SiftMatching.max_ratio",
                str(self.params.max_ratio),
                "--SiftMatching.max_distance",
                str(self.params.max_distance),
                "--SiftMatching.cross_check",
                "1" if self.params.cross_check else "0",
            ]
            description = "Sequential Matching"
        else:
            cmd = [
                self.colmap_bin,
                "exhaustive_matcher",
                "--database_path",
                database_path,
                "--FeatureMatching.num_threads",
                str(self.num_threads),
                "--SiftMatching.max_ratio",
                str(self.params.max_ratio),
                "--SiftMatching.max_distance",
                str(self.params.max_distance),
                "--SiftMatching.cross_check",
                "1" if self.params.cross_check else "0",
            ]
            description = "Exhaustive Matching"

        use_gpu = (
            is_windows()
            and getattr(self.params, "use_gpu_matching", True)
            and not getattr(self.params, "force_cpu", False)
        )
        if use_gpu:
            cmd.extend(["--SiftMatching.use_gpu", "1", "--SiftMatching.gpu_index", "0"])

        return self.run_command(cmd, description, status_prefix="Comparison")

    def mapper(self, database_path, images_dir, sparse_dir):
        if self.params.use_glomap:
            # GLOMAP Integration
            self.log("Using GLOMAP for reconstruction...")

            cmd = [
                self.glomap_bin,
                "mapper",
                "--database_path",
                database_path,
                "--image_path",
                images_dir,
                "--output_path",
                sparse_dir,
            ]

            # Note: GLOMAP output structure might need verification, typically creates/uses sparse/0
            # If glomap fails due to missing binary it will be caught by run_command exception handler
            return self.run_command(
                cmd, "3D Reconstruction (GLOMAP)", status_prefix="GLOMAP Reconstruction"
            )

        else:
            # Standard COLMAP Mapper
            cmd = [
                self.colmap_bin,
                "mapper",
                "--database_path",
                database_path,
                "--image_path",
                images_dir,
                "--output_path",
                sparse_dir,
                "--Mapper.num_threads",
                str(self.num_threads),
                "--Mapper.min_model_size",
                str(self.params.min_model_size),
                "--Mapper.multiple_models",
                "1" if self.params.multiple_models else "0",
                "--Mapper.ba_refine_focal_length",
                "1" if self.params.ba_refine_focal_length else "0",
                "--Mapper.ba_refine_principal_point",
                "1" if self.params.ba_refine_principal_point else "0",
                "--Mapper.ba_refine_extra_params",
                "1" if self.params.ba_refine_extra_params else "0",
                "--Mapper.min_num_matches",
                str(self.params.min_num_matches),
            ]
            return self.run_command(
                cmd, "3D Reconstruction (COLMAP)", status_prefix="3D Reconstruction"
            )

    def image_undistorter(self, images_dir: str, sparse_dir: str, output_dir: str):
        input_path = Path(sparse_dir) / "0"
        cmd = [
            self.colmap_bin,
            "image_undistorter",
            "--image_path",
            images_dir,
            "--input_path",
            str(input_path),
            "--output_path",
            output_dir,
            "--output_type",
            "COLMAP",
            "--max_image_size",
            str(self.params.max_image_size),
        ]
        return self.run_command(cmd, "Image Undistortion", status_prefix="Optical correction")

    def create_brush_config(self, output_dir: Path, images_dir: Path, sparse_dir: Path):
        # Determine actual paths to use (Undistorted vs Original)
        if self.params.undistort_images:
            final_images_path = output_dir / "dense" / "images"
            final_sparse_path = output_dir / "dense" / "sparse"
            self.log("Using undistorted images and reconstruction for Brush")
        else:
            final_images_path = images_dir
            final_sparse_path = sparse_dir / "0"

        optimized_for = "Windows/CUDA (RTX)"

        config = {
            "dataset_type": "colmap",
            "images_path": str(final_images_path),
            "sparse_path": str(final_sparse_path),
            "created_with": "CorbeauSplat",
            "architecture": platform.machine(),
            "optimized_for": optimized_for,
            "parameters": self.params.to_dict(),
        }
        config_path = output_dir / "brush_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        self.log(f"Brush config created: {config_path}")

    def stop(self):
        """Stops the current process via Template Method"""
        super().stop()

    @staticmethod
    def delete_project_content(target_path: Path):
        """Safely deletes a project folder's content (Trash)"""
        # [AUDIT] OWASP-A01 : Prevents deletion of root folder / entire disk on variable error
        safe_path = Path(target_path).resolve()
        if str(safe_path) == "/" or str(safe_path) == str(Path.home()):
            return False, "Critical deletion attempt blocked by safety check."

        if not target_path.exists():
            return False, "Folder does not exist"

        try:
            # Empty the directory by moving content to trash (except images)
            for item in target_path.iterdir():
                if item.name == "images":
                    continue

                try:
                    send2trash.send2trash(str(item))
                except Exception as e:
                    print(f"Failed to trash {item}. Reason: {e}")

            return True, "Content moved to trash"
        except Exception as e:
            return False, str(e)
