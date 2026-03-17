from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


def set_dark_theme(app_instance=None):
    """Applique un thème sombre"""
    if app_instance is None:
        app_instance = QApplication.instance()

    dark_palette = QPalette()

    # Couleurs de base
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)

    app_instance.setPalette(dark_palette)
    app_instance.setStyleSheet(
        """
        QToolTip {
            color: #ffffff;
            background-color: #2a82da;
            border: 1px solid white;
        }
        QGroupBox {
            border: 1px solid #76797C;
            margin-top: 1.5em;
            border-radius: 4px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 3px;
            color: #ffffff;
        }
        QTabWidget::pane {
            border: 1px solid #76797C;
        }
        QTabBar::tab {
            background: #353535;
            color: #ffffff;
            padding: 5px;
            border: 1px solid #76797C;
            border-bottom-color: #76797C;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background: #505050;
            border-bottom-color: #505050;
        }
        QPushButton {
            background-color: #505050;
            border: 1px solid #76797C;
            border-radius: 4px;
            padding: 5px;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #606060;
        }
        QPushButton:pressed {
            background-color: #303030;
        }
        /* ── Reset-to-default buttons (must stay square, override min-width: 80px) ── */
        QPushButton#resetBtn {
            min-width: 22px;
            max-width: 22px;
            min-height: 22px;
            max-height: 22px;
            padding: 0px;
            border-radius: 3px;
            background-color: #404040;
            border: 1px solid #555555;
            color: #aaaaaa;
            font-size: 13px;
        }
        QPushButton#resetBtn:hover {
            background-color: #2a82da;
            color: #ffffff;
        }
        QPushButton#resetBtn:pressed {
            background-color: #1a62ba;
        }
        /* ── Input controls ── */
        QLineEdit, QTextEdit {
            background-color: #252525;
            border: 1px solid #76797C;
            border-radius: 4px;
            padding: 2px 4px;
            color: #ffffff;
            selection-background-color: #2a82da;
            selection-color: #ffffff;
        }
        QLineEdit:read-only {
            background-color: #1e1e1e;
            color: #888888;
        }
        QComboBox QAbstractItemView {
            background-color: #2b2b2b;
            border: 1px solid #76797C;
            color: #ffffff;
            selection-background-color: #2a82da;
            selection-color: #ffffff;
            outline: none;
        }

        QProgressBar {
            border: 1px solid #76797C;
            border-radius: 5px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #2a82da;
            width: 20px;
        }
    """
    )
