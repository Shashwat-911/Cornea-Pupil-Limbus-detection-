"""
Tool selection toolbar for the annotation tool.
"""

from __future__ import annotations

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QPushButton, QLabel, QButtonGroup,
    )
    from PyQt6.QtCore import pyqtSignal
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


if HAS_PYQT6:

    class AnnotationToolbar(QWidget):
        """Tool selection panel for annotation types.

        Signals
        -------
        tool_changed(str)
            Emitted when the user selects a different tool.
        """

        tool_changed = pyqtSignal(str)

        TOOLS = [
            ("1: Crypt", "crypt"),
            ("2: Furrow", "furrow"),
            ("3: Vessel", "vessel"),
            ("4: Ink Mark", "ink"),
            ("5: Collarette", "collarette_junction"),
        ]

        def __init__(self):
            super().__init__()
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("<b>Annotation Tools</b>"))

            self.button_group = QButtonGroup(self)
            self.button_group.setExclusive(True)

            for i, (label, tool_name) in enumerate(self.TOOLS):
                btn = QPushButton(label)
                btn.setCheckable(True)
                if i == 0:
                    btn.setChecked(True)
                btn.clicked.connect(
                    lambda checked, t=tool_name: self._on_tool_clicked(t)
                )
                layout.addWidget(btn)
                self.button_group.addButton(btn, i)

            layout.addStretch()

        def select_tool(self, tool_name: str):
            """Programmatically select a tool."""
            for i, (_, name) in enumerate(self.TOOLS):
                if name == tool_name:
                    btn = self.button_group.button(i)
                    if btn:
                        btn.setChecked(True)
                    self.tool_changed.emit(tool_name)
                    break

        def _on_tool_clicked(self, tool_name: str):
            self.tool_changed.emit(tool_name)

else:
    class AnnotationToolbar:
        """Stub — PyQt6 not available."""
        def __init__(self):
            raise ImportError("PyQt6 required")
