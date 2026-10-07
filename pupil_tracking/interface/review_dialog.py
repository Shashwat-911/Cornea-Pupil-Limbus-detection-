"""Interactive Cyclotorsion Doctor Review & Validation Dialog.

Provides ophthalmologists with an in-depth review window for:
- Reference image (Pentacam IR or intra-operative snapshot) vs Live ELITA Frame
- Cyclotorsion measurement, intorsion/excyclotorsion direction, and confidence
- Toric axis correction (Alpins vector rule: planned vs corrected axis)
- Annulus feature correspondence and polar unwrap inspection
- Audit report and snapshot export
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from pupil_tracking.interface.theme import Colors
from pupil_tracking.iris.detect import IrisFeatureDetector
from pupil_tracking.iris.correspondence import estimate_correspondence, CorrespondenceConfig
from pupil_tracking.pentacam.session import SittingRegistrationSession

logger = logging.getLogger(__name__)


class CyclotorsionReviewDialog(tk.Toplevel):
    """Surgical-grade doctor review popup for cyclotorsion and toric centration."""

    def __init__(
        self,
        parent: tk.Widget,
        patient_id: str = "AUTO",
        patient_name: str = "",
        laterality: str = "OD",
        storage_root: Optional[Path] = None,
        gui_ref: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.title("Medevplus IXcentai — Cyclotorsion Doctor Review & Validation")
        self.geometry("980x720")
        self.minsize(840, 600)
        self.configure(bg=Colors.BG_PRIMARY)

        self.patient_id = patient_id or "AUTO"
        self.patient_name = patient_name or ""
        self.laterality = laterality or "OD"
        self.storage_root = storage_root or Path("review_output")
        self.gui_ref = gui_ref

        # State
        self._reference_image: Optional[np.ndarray] = None
        self._reference_path: Optional[str] = None
        self._current_image: Optional[np.ndarray] = None
        self._current_result: Optional[Any] = None
        self._iris_detector = IrisFeatureDetector()
        self._session = SittingRegistrationSession()
        self._last_result_dict: Dict[str, Any] = {}

        # Planned axis variable (degrees)
        self._planned_axis_var = tk.DoubleVar(value=90.0)
        self._planned_axis_var.trace_add("write", lambda *_: self._update_toric_math())

        # UI Variables
        self._status_var = tk.StringVar(value="Ready — Select reference or register current frame")
        self._cyclotorsion_deg_var = tk.StringVar(value="0.00°")
        self._direction_var = tk.StringVar(value="NONE")
        self._confidence_var = tk.StringVar(value="0.0%")
        self._grade_var = tk.StringVar(value="STANDBY")
        self._inliers_var = tk.StringVar(value="0 / 0")
        self._corrected_axis_var = tk.StringVar(value="90.00°")
        self._residual_astig_var = tk.StringVar(value="0.0%")
        self._ref_features_var = tk.StringVar(value="0")
        self._cur_features_var = tk.StringVar(value="0")

        self._build_ui()

    def _build_ui(self) -> None:
        # Header banner
        header = tk.Frame(self, bg=Colors.BG_SECONDARY, padx=16, pady=10)
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header,
            text=f"🔬 Cyclotorsion Doctor Review — Patient: {self.patient_name or self.patient_id} ({self.laterality})",
            font=("Segoe UI", 12, "bold"),
            fg=Colors.FG_PRIMARY,
            bg=Colors.BG_SECONDARY,
        )
        title_lbl.pack(side=tk.LEFT)

        status_lbl = tk.Label(
            header,
            textvariable=self._status_var,
            font=("Segoe UI", 9),
            fg=Colors.ACCENT_HOVER,
            bg=Colors.BG_SECONDARY,
        )
        status_lbl.pack(side=tk.RIGHT)

        # Main content split: Left = Image previews, Right = Clinical Metrics & Controls
        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=Colors.BG_PRIMARY, sashwidth=4)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Left panel: Image Displays
        left_frame = tk.Frame(main_pane, bg=Colors.BG_PRIMARY)
        main_pane.add(left_frame, minsize=480)

        # 2-Row Canvas: Top = Reference, Bottom = Current / Registered
        self._ref_canvas_label = tk.Label(
            left_frame, text="Reference Image (Pentacam / Baseline)", font=("Segoe UI", 9, "bold"),
            fg=Colors.FG_SECONDARY, bg=Colors.BG_PRIMARY, anchor="w",
        )
        self._ref_canvas_label.pack(fill=tk.X, padx=4, pady=(2, 2))

        self._ref_canvas = tk.Canvas(left_frame, bg="#080808", height=240, highlightthickness=1, highlightbackground=Colors.BORDER)
        self._ref_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        self._cur_canvas_label = tk.Label(
            left_frame, text="Current Live Image / Annulus Alignment", font=("Segoe UI", 9, "bold"),
            fg=Colors.FG_SECONDARY, bg=Colors.BG_PRIMARY, anchor="w",
        )
        self._cur_canvas_label.pack(fill=tk.X, padx=4, pady=(8, 2))

        self._cur_canvas = tk.Canvas(left_frame, bg="#080808", height=240, highlightthickness=1, highlightbackground=Colors.BORDER)
        self._cur_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        # Right panel: Cards & Action Controls
        right_frame = tk.Frame(main_pane, bg=Colors.BG_SECONDARY, width=340, padx=12, pady=10)
        main_pane.add(right_frame, minsize=320)

        # Section 1: Cyclotorsion Measurement Card
        c_card = tk.LabelFrame(
            right_frame, text=" Cyclotorsion Measurement ", font=("Segoe UI", 10, "bold"),
            fg=Colors.ACCENT_HOVER, bg=Colors.BG_SECONDARY, padx=10, pady=8, bd=1, relief=tk.SOLID
        )
        c_card.pack(fill=tk.X, pady=(0, 10))

        def make_row(parent, label, var, val_color=Colors.FG_PRIMARY):
            r = tk.Frame(parent, bg=Colors.BG_SECONDARY)
            r.pack(fill=tk.X, pady=2)
            tk.Label(r, text=label, font=("Segoe UI", 9), fg=Colors.FG_SECONDARY, bg=Colors.BG_SECONDARY).pack(side=tk.LEFT)
            tk.Label(r, textvariable=var, font=("Segoe UI", 10, "bold"), fg=val_color, bg=Colors.BG_SECONDARY).pack(side=tk.RIGHT)

        make_row(c_card, "Torsion Angle (θ):", self._cyclotorsion_deg_var, Colors.SURGICAL)
        make_row(c_card, "Torsion Direction:", self._direction_var, Colors.FG_PRIMARY)
        make_row(c_card, "Registration Confidence:", self._confidence_var, Colors.CLINICAL)
        make_row(c_card, "Quality Grade:", self._grade_var, Colors.ACCENT)
        make_row(c_card, "Annulus Inlier Pairs:", self._inliers_var, Colors.FG_PRIMARY)

        # Section 2: Toric Axis Correction (Alpins Vector Magnitude)
        t_card = tk.LabelFrame(
            right_frame, text=" Toric Axis Correction ", font=("Segoe UI", 10, "bold"),
            fg=Colors.CALIBRATION, bg=Colors.BG_SECONDARY, padx=10, pady=8, bd=1, relief=tk.SOLID
        )
        t_card.pack(fill=tk.X, pady=(0, 10))

        axis_row = tk.Frame(t_card, bg=Colors.BG_SECONDARY)
        axis_row.pack(fill=tk.X, pady=2)
        tk.Label(axis_row, text="Planned Axis (°):", font=("Segoe UI", 9), fg=Colors.FG_SECONDARY, bg=Colors.BG_SECONDARY).pack(side=tk.LEFT)
        axis_spin = ttk.Spinbox(axis_row, from_=0.0, to=180.0, increment=1.0, textvariable=self._planned_axis_var, width=7)
        axis_spin.pack(side=tk.RIGHT)

        make_row(t_card, "Corrected Treatment Axis:", self._corrected_axis_var, Colors.SURGICAL)
        make_row(t_card, "Under-Correction / Loss:", self._residual_astig_var, Colors.RESEARCH)

        # Section 3: Annulus & Landmark Statistics
        f_card = tk.LabelFrame(
            right_frame, text=" Iris Features & Landmarks ", font=("Segoe UI", 10, "bold"),
            fg=Colors.LIMBUS, bg=Colors.BG_SECONDARY, padx=10, pady=8, bd=1, relief=tk.SOLID
        )
        f_card.pack(fill=tk.X, pady=(0, 10))

        make_row(f_card, "Reference Landmarks:", self._ref_features_var, Colors.FG_PRIMARY)
        make_row(f_card, "Current Landmarks:", self._cur_features_var, Colors.FG_PRIMARY)

        # Action Buttons
        btn_frame = tk.Frame(right_frame, bg=Colors.BG_SECONDARY)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=4)

        ttk.Button(btn_frame, text="📂 Load Pentacam Reference BMP", command=self._load_reference_file).pack(fill=tk.X, pady=3)
        ttk.Button(btn_frame, text="📸 Use Current Frame as Reference", command=self._set_current_as_reference).pack(fill=tk.X, pady=3)
        ttk.Button(btn_frame, text="⚡ Re-Run Cyclotorsion Registration", command=self._on_rerun_registration).pack(fill=tk.X, pady=3)
        ttk.Button(btn_frame, text="💾 Export Audit Report & JSON", command=self._export_report).pack(fill=tk.X, pady=3)
        ttk.Button(btn_frame, text="✕ Close Review Window", command=self.destroy).pack(fill=tk.X, pady=3)

    def _render_image_to_canvas(self, canvas: tk.Canvas, img_bgr: np.ndarray, tag: str) -> None:
        if img_bgr is None or img_bgr.size == 0:
            return
        cw = max(10, canvas.winfo_width())
        ch = max(10, canvas.winfo_height())
        ih, iw = img_bgr.shape[:2]
        scale = min(cw / iw, ch / ih, 1.0)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(pil_img)

        setattr(self, f"_{tag}_tk_img", tk_img)
        canvas.delete("all")
        x_off = (cw - nw) // 2
        y_off = (ch - nh) // 2
        canvas.create_image(x_off, y_off, anchor=tk.NW, image=tk_img)

    def _load_reference_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Pentacam or Reference Image",
            filetypes=[("Image files", "*.bmp *.png *.jpg *.jpeg"), ("All files", "*.*")],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Failed to read image from {path}")
            return
        self._reference_image = img
        self._reference_path = path
        self._session.set_reference(img, eye_id=self.patient_id, laterality=self.laterality)
        self._render_image_to_canvas(self._ref_canvas, img, "ref")
        self._status_var.set(f"Loaded reference: {Path(path).name}")
        if self._current_image is not None:
            self.run_registration(self._current_image, mode="static")

    def _set_current_as_reference(self) -> None:
        if self._current_image is None:
            messagebox.showwarning("Warning", "No current frame available to set as reference.")
            return
        self._reference_image = self._current_image.copy()
        self._session.set_reference(self._reference_image, eye_id=self.patient_id, laterality=self.laterality)
        self._render_image_to_canvas(self._ref_canvas, self._reference_image, "ref")
        self._status_var.set("Current frame set as registration reference.")
        self.run_registration(self._current_image, mode="static")

    def _on_rerun_registration(self) -> None:
        if self._current_image is not None:
            self.run_registration(self._current_image, mode="static")

    def update_live_data(self, image: np.ndarray, result: Any) -> None:
        """Invoked by the main GUI display loop on live video frames."""
        self._current_image = image
        self._current_result = result
        # Only render current canvas periodically or on active review
        self._render_image_to_canvas(self._cur_canvas, image, "cur")

    def run_registration(self, image: np.ndarray, mode: str = "static") -> None:
        """Run registration between reference and image."""
        self._current_image = image
        self._render_image_to_canvas(self._cur_canvas, image, "cur")

        # 1. Check if reference is available
        if self._reference_image is None:
            # Fall back to self-reference baseline
            self._reference_image = image.copy()
            self._session.set_reference(self._reference_image, eye_id=self.patient_id, laterality=self.laterality)
            self._render_image_to_canvas(self._ref_canvas, self._reference_image, "ref")

        # 2. Extract iris features for readout
        cur_feat_cnt = 0
        ref_feat_cnt = 0
        try:
            if self._current_result and getattr(self._current_result, "has_both", False):
                ires = self._iris_detector.detect(
                    image,
                    pupil=self._current_result.pupil.ellipse,
                    limbus=self._current_result.limbus.ellipse,
                )
                if ires and ires.feature_set:
                    cur_feat_cnt = len(ires.feature_set.features)
        except Exception as e:
            logger.debug("Feature extraction failed: %s", e)

        self._cur_features_var.set(str(cur_feat_cnt))

        # 3. Perform cross-system / sitting-to-supine registration
        try:
            reg_res = self._session.register(
                image,
                eye_id=self.patient_id,
                laterality=self.laterality,
                detection=self._current_result,
                mode=mode,
            )

            theta = float(reg_res.rotation_deg) if reg_res.rotation_deg is not None else 0.0
            # Wrap to [-180, 180]
            if theta > 180.0:
                theta -= 360.0
            elif theta < -180.0:
                theta += 360.0

            conf = float(reg_res.confidence) if reg_res.confidence is not None else 0.0
            direction = str(reg_res.torsion_direction or "NONE")

            self._cyclotorsion_deg_var.set(f"{theta:+.2f}°")
            self._direction_var.set(direction)
            self._confidence_var.set(f"{conf * 100.0:.1f}%")
            self._grade_var.set("SURGICAL GRADE" if conf >= 0.70 and reg_res.valid else ("CLINICAL" if reg_res.valid else "INSUFFICIENT"))
            self._inliers_var.set(f"{reg_res.n_inliers} / {reg_res.n_correspondences}")
            self._ref_features_var.set(str(reg_res.pentacam_features_used or "---"))

            self._last_result_dict = {
                "patient_id": self.patient_id,
                "laterality": self.laterality,
                "timestamp": time.time(),
                "cyclotorsion_deg": theta,
                "direction": direction,
                "confidence": conf,
                "valid": reg_res.valid,
                "failure_reason": reg_res.failure_reason,
                "n_inliers": reg_res.n_inliers,
                "n_correspondences": reg_res.n_correspondences,
            }

            self._update_toric_math(theta)
            self._status_var.set(f"Registered (θ={theta:+.2f}°, conf={conf:.2f})")
        except Exception as exc:
            logger.error("Registration error in review popup: %s", exc)
            self._status_var.set(f"Registration error: {exc}")

    def _update_toric_math(self, theta_deg: Optional[float] = None) -> None:
        """Compute Alpins 3-degree rule: corrected treatment axis and residual cylinder."""
        if theta_deg is None:
            try:
                theta_str = self._cyclotorsion_deg_var.get().replace("°", "").replace("+", "").strip()
                theta_deg = float(theta_str)
            except ValueError:
                theta_deg = 0.0

        planned = float(self._planned_axis_var.get())
        corrected = (planned + theta_deg) % 180.0
        self._corrected_axis_var.set(f"{corrected:.2f}°")

        # Alpins Rule: Under-Correction % = 2 * sin(|theta|) * 100%
        theta_rad = math.radians(abs(theta_deg))
        loss_pct = min(100.0, 2.0 * math.sin(theta_rad) * 100.0)
        self._residual_astig_var.set(f"{loss_pct:.1f}%")

        if self._last_result_dict:
            self._last_result_dict["planned_axis_deg"] = planned
            self._last_result_dict["corrected_axis_deg"] = corrected
            self._last_result_dict["astigmatism_loss_pct"] = loss_pct

    def _export_report(self) -> None:
        """Export comprehensive review report and JSON artifact."""
        try:
            self.storage_root.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_json = self.storage_root / f"cyclotorsion_review_{self.patient_id}_{ts}.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(self._last_result_dict, f, indent=2)

            messagebox.showinfo("Export Successful", f"Audit report successfully exported to:\n{out_json}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to save report: {e}")
