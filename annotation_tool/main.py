"""
IXcentai Standalone Iris Feature Annotation Tool - Main Entry Point.

Launch the standalone annotation GUI:
    python -m annotation_tool.main
or
    python annotation_tool/main.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path


def main() -> int:
    """Launch the IXcentai annotation tool GUI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("annotation_tool")

    try:
        from PyQt6.QtWidgets import QApplication
        from annotation_tool.gui.main_window import AnnotationMainWindow
    except ImportError as e:
        logger.error(
            "PyQt6 is required to run the annotation tool GUI.\n"
            "Please install dependencies with:\n"
            "    pip install -r annotation_tool/requirements.txt\n"
            f"Original error: {e}"
        )
        return 1

    app = QApplication(sys.argv)
    app.setApplicationName("IXcentai Iris Feature Annotator")
    app.setOrganizationName("Medevplus")

    window = AnnotationMainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
