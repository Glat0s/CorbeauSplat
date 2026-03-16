#!/usr/bin/env python3
import argparse
import os
import sys
import time

from PyQt6.QtWidgets import QApplication

from app.core.brush_engine import BrushEngine
from app.core.engine import ColmapEngine
from app.core.i18n import tr
from app.core.logging_config import configure_logging, get_logger
from app.core.params import ColmapParams
from app.core.sharp_engine import SharpEngine
from app.core.superplat_engine import SuperSplatEngine
from app.core.system import check_dependencies
from app.gui.main_window import ColmapGUI

configure_logging()
logger = get_logger("main")


def get_parser():
    """Configure and return the argument parser"""
    parser = argparse.ArgumentParser(description=tr("cli_desc").replace("v0.8", "v0.9"))

    # GUI Mode
    parser.add_argument("--gui", action="store_true", help=tr("cli_gui_help"))

    # Operation Modes
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true", help=tr("cli_train_help"))
    group.add_argument("--predict", action="store_true", help=tr("cli_predict_help"))
    group.add_argument("--view", action="store_true", help=tr("cli_view_help"))

    # Common Arguments
    parser.add_argument("--input", "-i", help=tr("cli_input_help"))
    parser.add_argument("--output", "-o", help=tr("cli_output_help"))

    # --- COLMAP Arguments ---
    colmap_group = parser.add_argument_group("COLMAP Options")
    colmap_group.add_argument(
        "--type", choices=["images", "video"], default="images", help=tr("cli_type_help")
    )
    colmap_group.add_argument("--fps", type=int, default=5, help=tr("cli_fps_help"))
    colmap_group.add_argument("--camera_model", default="SIMPLE_RADIAL", help=tr("cli_cam_help"))
    colmap_group.add_argument("--undistort", action="store_true", help=tr("cli_undistort_help"))

    # --- BRUSH Arguments ---
    brush_group = parser.add_argument_group("BRUSH Options")
    brush_group.add_argument("--iterations", type=int, default=30000, help=tr("cli_iter_help"))
    brush_group.add_argument("--sh_degree", type=int, default=3, help=tr("cli_sh_degree_help"))
    brush_group.add_argument("--device", default="auto", help=tr("brush_tip_device"))

    # --- SHARP Arguments ---
    sharp_group = parser.add_argument_group("SHARP Options")
    sharp_group.add_argument("--checkpoint", help=tr("cli_checkpoint_help"))

    # --- SUPERSPLAT Arguments ---
    splat_group = parser.add_argument_group("SUPERSPLAT Options")
    splat_group.add_argument("--port", type=int, default=3000, help=tr("cli_port_help"))
    splat_group.add_argument("--data_port", type=int, default=8000, help=tr("cli_data_port_help"))

    return parser


def run_colmap(args):
    """Run the COLMAP photogrammetry pipeline."""
    if not args.input or not args.output:
        logger.error(tr("cli_err_colmap_args"))
        sys.exit(1)

    params = ColmapParams(camera_model=args.camera_model, undistort_images=args.undistort)

    logger.info(tr("cli_start_colmap"))
    logger.info(tr("cli_input", args.input))
    logger.info(tr("cli_output", args.output))

    engine = ColmapEngine(
        params,
        args.input,
        args.output,
        args.type,
        args.fps,
        logger_callback=lambda msg: logger.info(msg),
        progress_callback=lambda x: logger.info(tr("cli_progression", x)),
    )

    success, msg = engine.run()
    if success:
        logger.info(tr("cli_success", msg))
    else:
        logger.error(tr("cli_error", msg))
        sys.exit(1)


def run_brush(args):
    """Run Brush 3DGS training."""
    if not args.input or not args.output:
        logger.error(tr("cli_err_brush_args"))
        sys.exit(1)

    engine = BrushEngine()
    logger.info(tr("cli_start_brush"))
    logger.info(tr("cli_input", args.input))
    logger.info(tr("cli_output", args.output))

    params = {"total_steps": args.iterations, "sh_degree": args.sh_degree, "device": args.device}

    process = engine.train(args.input, args.output, params=params)

    try:
        for line in process.stdout:
            logger.info(line.rstrip())
        process.wait()
        if process.returncode == 0:
            logger.info(tr("msg_success"))
        else:
            logger.error(tr("msg_error"))
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info(tr("cli_stopping"))
        engine.stop()


def run_sharp(args):
    """Run Sharp ML sharpening prediction."""
    if not args.input or not args.output:
        logger.error(tr("cli_err_sharp_args"))
        sys.exit(1)

    engine = SharpEngine()
    logger.info(tr("cli_start_sharp"))

    params = {
        "checkpoint": args.checkpoint,
        "device": args.device if args.device != "auto" else "default",
        "verbose": True,
    }

    process = engine.predict(args.input, args.output, params=params)

    try:
        for line in process.stdout:
            logger.info(line.rstrip())
        process.wait()
        if process.returncode == 0:
            logger.info(tr("msg_success"))
        else:
            logger.error(tr("msg_error"))
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info(tr("cli_stopping"))
        engine.stop()


def run_supersplat(args):
    """Launch the SuperSplat viewer with a local data server."""
    if not args.input:
        logger.error(tr("cli_err_view_args"))
        sys.exit(1)

    engine = SuperSplatEngine()
    logger.info(tr("cli_start_view"))

    if os.path.isfile(args.input):
        data_dir = os.path.dirname(args.input)
        filename = os.path.basename(args.input)
    else:
        data_dir = args.input
        filename = ""

    ok, msg = engine.start_data_server(data_dir, port=args.data_port)
    if not ok:
        logger.error("Data server failed: %s", msg)
        sys.exit(1)
    logger.info(msg)

    ok, msg = engine.start_supersplat(port=args.port)
    if not ok:
        logger.error("SuperSplat server failed: %s", msg)
        engine.stop_all()
        sys.exit(1)
    logger.info(msg)

    url = f"http://localhost:{args.port}?url=http://localhost:{args.data_port}/{filename}"
    logger.info("Open in browser: %s", url)
    logger.info("Press Ctrl+C to stop the servers.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info(tr("cli_server_stop"))
        engine.stop_all()


def _launch_gui() -> None:
    """Create QApplication, run first-time setup if needed, then open main window."""
    from app.scripts.setup import is_setup_complete

    logger.info("Launching CorbeauSplat GUI...")
    app = QApplication(sys.argv)

    if not is_setup_complete():
        logger.info("First run detected — starting setup wizard.")
        from app.gui.setup_window import SetupWindow

        setup_win = SetupWindow()
        setup_win.exec()

    logger.info("Opening main window.")
    window = ColmapGUI()
    window.show()
    sys.exit(app.exec())


def main():
    parser = get_parser()
    args = parser.parse_args()

    missing_deps = check_dependencies()
    if missing_deps:
        logger.warning(
            "Missing dependencies: %s — some features may not work. "
            "Run: uv run python -m app.scripts.setup",
            ", ".join(missing_deps),
        )

    if args.gui:
        _launch_gui()
    elif args.train:
        run_brush(args)
    elif args.predict:
        run_sharp(args)
    elif args.view:
        run_supersplat(args)
    elif args.input and args.output:
        run_colmap(args)
    else:
        if len(sys.argv) == 1:
            _launch_gui()
        else:
            parser.print_help()
            sys.exit(0)


if __name__ == "__main__":
    main()
