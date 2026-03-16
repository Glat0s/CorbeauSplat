"""Centralised logging configuration for CorbeauSplat.

Call ``configure_logging()`` once at application startup (``main.py``).
Use ``get_logger(name)`` everywhere else to obtain a named logger.
"""

from __future__ import annotations

import logging
import logging.handlers


def configure_logging(log_level: int = logging.DEBUG) -> None:
    """Set up the root logger with a rotating file handler and a console handler.

    Safe to call more than once — subsequent calls are no-ops if handlers are
    already registered.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # Already configured

    root.setLevel(log_level)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler — keeps the last 5 × 5 MB rotations
    try:
        from app.core.system import resolve_project_root

        log_dir = resolve_project_root() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_dir / "corbeausplat.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        pass  # Don't crash if the project root is unavailable during early import

    # Console handler — INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  ``configure_logging()`` should be called first."""
    return logging.getLogger(name)
