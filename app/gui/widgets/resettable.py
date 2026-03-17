from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget


def reset_button(reset_fn) -> QPushButton:
    """Return a standalone ↺ button for use inside an existing HBoxLayout.

    Styling is driven entirely by the app-level QPushButton#resetBtn rule in
    styles.py — no widget-level setStyleSheet so that rule is never overridden.
    """
    btn = QPushButton("↺")
    btn.setObjectName("resetBtn")
    btn.setFixedSize(22, 22)
    btn.setToolTip("Reset to default")
    btn.clicked.connect(reset_fn)
    return btn


def make_resettable(widget, reset_fn) -> QWidget:
    """Wrap *widget* in a container that includes a small ↺ reset button.

    The stretch at the end keeps the widget+button left-aligned so the pair
    stays compact inside any parent layout (QFormLayout, QVBoxLayout, etc.).
    """
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.addWidget(widget)
    layout.addWidget(reset_button(reset_fn))
    layout.addStretch()
    return container
