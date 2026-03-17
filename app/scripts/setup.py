"""
CorbeauSplat first-run setup.

Downloads all required model files and installs all external engine
dependencies. Run once before launching the application:

    uv run python -m app.scripts.setup

Or triggered automatically on first GUI launch.
"""

from __future__ import annotations

import logging
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Bump to force re-setup when the set of required assets changes.
SETUP_VERSION = "1"

_ESRGAN_URL = "https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/RealESRGAN_x4plus.fp16.onnx"
_GFPGAN_URL = (
    "https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/GFPGANv1.4.onnx"
)
# Ordered list of (step_id, display_name) — used by the UI and run_setup.
STEP_IDS: list[tuple[str, str]] = [
    ("colmap", "COLMAP (photogrammetry)"),
    ("brush", "Brush (3DGS trainer)"),
    ("supersplat", "SuperSplat (viewer)"),
    ("extractor360", "360° Extractor"),
    ("sharp", "Sharp (ML sharpening)"),
    ("esrgan", "ESRGAN model (upscaling)"),
    ("gfpgan", "GFPGAN model (face restore)"),
]


# ---------------------------------------------------------------------------
# Marker helpers
# ---------------------------------------------------------------------------


def get_marker_path() -> Path:
    from app.core.system import resolve_project_root

    return resolve_project_root() / "engines" / ".setup_complete"


def is_setup_complete() -> bool:
    m = get_marker_path()
    return m.exists() and m.read_text().strip() == SETUP_VERSION


def mark_setup_complete() -> None:
    m = get_marker_path()
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text(SETUP_VERSION)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _download_file(
    url: str,
    dest: Path,
    log_cb: Callable[[str], None] = logger.info,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> bool:
    """Download *url* → *dest*; skip silently if dest already exists."""
    if dest.exists():
        log_cb(f"  Already present: {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    log_cb(f"  Downloading {dest.name} …")

    def _hook(block: int, block_size: int, total: int) -> None:
        if total > 0 and progress_cb:
            progress_cb(min(99, int(block * block_size / total * 100)))

    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_hook)
        if progress_cb:
            progress_cb(100)
        log_cb(f"  ✅ {dest.name}")
        return True
    except Exception as e:
        log_cb(f"  ❌ Download failed: {e}")
        dest.unlink(missing_ok=True)
        return False


def _run_engine(
    dep,
    step_id: str,
    name: str,
    results: dict[str, bool],
    log_cb: Callable[[str], None],
    notify: Callable[[str, str, str], None],
) -> bool:
    """Install an engine dep, skipping if already present."""
    if dep.is_installed():
        log_cb(f"  ✅ {name} already installed")
        notify(step_id, name, "skip")
        results[step_id] = True
        return True

    log_cb(f"\n>>> {name}")
    notify(step_id, name, "running")
    try:
        dep.install()
    except Exception as e:
        log_cb(f"  Warning: {e}")
    # Engine steps are non-critical — don't fail the whole setup.
    results[step_id] = True
    notify(step_id, name, "done")
    return True


# ---------------------------------------------------------------------------
# Main setup runner
# ---------------------------------------------------------------------------


def run_setup(
    log_cb: Callable[[str], None] = logger.info,
    step_cb: Optional[Callable[[str, str, str], None]] = None,
) -> dict[str, bool]:
    """
    Run all setup steps in order.

    *step_cb(step_id, display_name, status)* where status ∈
    ``{'running', 'done', 'failed', 'skip'}``.

    Returns ``{step_id: success}``.
    """
    from app.core.system import resolve_project_root
    from app.scripts.setup_dependencies import (
        BrushEngineDep,
        Extractor360EngineDep,
        SharpEngineDep,
        SuperSplatEngineDep,
        _download_colmap_windows,
        _find_colmap_in_engines,
    )

    root = resolve_project_root()
    engines_dir = root / "engines"
    weights_dir = root / "app" / "weights"
    engines_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, bool] = {}

    def _notify(step_id: str, name: str, status: str) -> None:
        if step_cb:
            step_cb(step_id, name, status)

    def _run(step_id: str, name: str, fn: Callable[[], bool]) -> bool:
        log_cb(f"\n>>> {name}")
        _notify(step_id, name, "running")
        try:
            ok = bool(fn())
        except Exception as e:
            log_cb(f"  ❌ {e}")
            ok = False
        _notify(step_id, name, "done" if ok else "failed")
        results[step_id] = ok
        return ok

    # COLMAP binary
    _run(
        "colmap",
        "COLMAP (photogrammetry)",
        lambda: bool(_find_colmap_in_engines(engines_dir)) or _download_colmap_windows(engines_dir),
    )

    # External engines (non-critical — setup continues on failure)
    for dep_cls, step_id, name in [
        (BrushEngineDep, "brush", "Brush (3DGS trainer)"),
        (SuperSplatEngineDep, "supersplat", "SuperSplat (viewer)"),
        (Extractor360EngineDep, "extractor360", "360° Extractor"),
        (SharpEngineDep, "sharp", "Sharp (ML sharpening)"),
    ]:
        _run_engine(dep_cls(), step_id, name, results, log_cb, _notify)

    # Model weights
    _run(
        "esrgan",
        "ESRGAN model (upscaling)",
        lambda: _download_file(_ESRGAN_URL, weights_dir / "RealESRGAN_x4plus.fp16.onnx", log_cb),
    )
    _run(
        "gfpgan",
        "GFPGAN model (face restore)",
        lambda: _download_file(_GFPGAN_URL, weights_dir / "GFPGANv1.4.onnx", log_cb),
    )

    log_cb("\n=== Setup complete ===")
    mark_setup_complete()
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: ``uv run python -m app.scripts.setup``"""
    import argparse

    parser = argparse.ArgumentParser(description="CorbeauSplat first-run setup")
    parser.add_argument("--force", action="store_true", help="Re-run even if already complete")
    args = parser.parse_args()

    from app.core.logging_config import configure_logging

    configure_logging()

    if is_setup_complete() and not args.force:
        logger.info("Setup already complete. Use --force to re-run.")
        return

    results = run_setup()
    failed = [k for k, v in results.items() if not v]
    if failed:
        logger.warning("Some steps failed: %s — re-run setup to retry.", ", ".join(failed))
        sys.exit(1)
    else:
        logger.info("All setup steps completed successfully.")


if __name__ == "__main__":
    main()
