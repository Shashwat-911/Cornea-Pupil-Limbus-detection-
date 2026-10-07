"""Unit tests for CyclotorsionReviewDialog and GUI adapter result conversion."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import pytest
import numpy as np

from pupil_tracking.utils.types import EllipseParams, DetectionQuality


def test_adapt_frame_result_ellipse_params():
    """Verify that _adapt_frame_result constructs valid EllipseParams objects."""
    from pupil_tracking.interface.gui_app import PupilTrackingGUI

    # Create dummy frame result namespace
    fr = SimpleNamespace(
        quality=SimpleNamespace(value="SURGICAL"),
        pupil_center=(300.0, 250.0),
        pupil_axes=(60.0, 58.0),
        pupil_angle=10.0,
        limbus_center=(302.0, 252.0),
        limbus_axes=(200.0, 198.0),
        limbus_angle=12.0,
        confidence=0.92,
        pupil_confidence=0.95,
        limbus_confidence=0.90,
        method="ML",
        corneal_center=(301.0, 251.0),
        offset_distance_px=2.5,
        offset_dx_px=1.5,
        offset_dy_px=2.0,
        offset_angle_deg=45.0,
        pixel_spacing=0.045,
        calibration_valid=True,
    )

    # Instantiate GUI helper method in headless mode
    gui = PupilTrackingGUI.__new__(PupilTrackingGUI)
    adapted = gui._adapt_frame_result(fr, (480, 640, 3))

    assert adapted is not None
    assert isinstance(adapted.pupil.ellipse, EllipseParams)
    assert isinstance(adapted.limbus.ellipse, EllipseParams)
    assert adapted.pupil.ellipse.center_x == 300.0
    assert adapted.pupil.ellipse.center_y == 250.0
    assert adapted.pupil.ellipse.semi_major == 30.0
    assert adapted.pupil.ellipse.semi_minor == 29.0
    assert adapted.limbus.ellipse.semi_major == 100.0
    assert adapted.limbus.ellipse.semi_minor == 99.0
    assert adapted.has_both is True


def test_cyclotorsion_review_dialog_toric_math(tmp_path):
    """Verify Alpins toric vector calculations in review dialog without UI display."""
    from pupil_tracking.interface.review_dialog import CyclotorsionReviewDialog
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    try:
        dialog = CyclotorsionReviewDialog(
            parent=root,
            patient_id="TEST_PX",
            patient_name="Test Patient",
            laterality="OD",
            storage_root=tmp_path,
        )

        # Test Alpins math with 3.0 deg cyclotorsion
        dialog._update_toric_math(3.0)
        assert dialog._corrected_axis_var.get() == "93.00°"
        # Loss for 3 deg: 2 * sin(3 deg) * 100% ~ 10.46%
        assert "10.5%" in dialog._residual_astig_var.get()

        # Test Alpins math with -2.5 deg cyclotorsion
        dialog._update_toric_math(-2.5)
        assert dialog._corrected_axis_var.get() == "87.50°"
        # Loss for 2.5 deg: 2 * sin(2.5 deg) * 100% ~ 8.7%
        assert "8.7%" in dialog._residual_astig_var.get()

        dialog.destroy()
    finally:
        root.destroy()
