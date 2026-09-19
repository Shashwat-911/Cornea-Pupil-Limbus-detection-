
"""
Surgical-grade pupil & limbus tracking GUI.

Features:
    - Load and analyse single images
    - Process video files with frame-by-frame playback
    - Live camera feed with real-time detection (optimised pipeline)
    - Smart circle/ellipse overlay (auto-selected per structure)
    - Diameter display in pixels AND millimetres
    - Semi-major/semi-minor display in pixels AND millimetres
    - Offset display in pixels AND millimetres
    - Auto-calibration from limbus (corneal diameter = 12.0 mm)
    - Detection overlay with pupil, limbus, corneal centre, offset
    - Export results to CSV / JSON / snapshot
    - Quality grade badge
    - FP16 / torch.compile / async capture / frame-skip / ROI / Kalman
    - Settings tab for full pipeline configuration
    - Progress bar with ETA for video-file processing
    - Pause / resume for video and camera
    - OptimizedVideoProcessor wired to video-file path
    - Robust None-safety: works even when model loading fails
    - Grayscale mode toggle (OFF / AUTO / FORCE) with [G] hotkey (NEW)
"""

from __future__ import annotations

import csv
import math
import re
import shutil
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.video.kalman_tracker import EyeKalmanTracker
from pupil_tracking.core.corneal_center import CornealCenterCalculator
from pupil_tracking.utils.types import (
    DetectionQuality,
    CalibrationInfo,
)
from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.logger import get_logger
from pupil_tracking.utils.runtime_profile import (
    apply_runtime_optimizations,
    detect_runtime_profile,
)
from pupil_tracking.interface.theme import DarkTheme

try:
    from pupil_tracking.ml.fast_inference import FastInference
    from pupil_tracking.video.optimized_processor import (
        OptimizedVideoProcessor,
        AsyncCapture,
        FrameResult,
        TrackingQuality,
    )

    _FAST_PIPELINE_AVAILABLE = True
except ImportError:
    _FAST_PIPELINE_AVAILABLE = False
    FastInference = None
    OptimizedVideoProcessor = None
    AsyncCapture = None
    FrameResult = None
    TrackingQuality = None

# ══════════════════════════════════════════════════════════════════
# RECORDING — Import FrameRecorder
# ══════════════════════════════════════════════════════════════════
from pupil_tracking.interface.frame_recorder import FrameRecorder


_CORNEAL_DIAMETER_MM = 12.0
_CIRCLE_DRAW_THRESHOLD = 0.95

_QUALITY_COLORS = {
    "SURGICAL": "#00e676",
    "CLINICAL": "#29b6f6",
    "RESEARCH": "#ffa726",
    "INSUFFICIENT": "#ef5350",
    "NO_DETECTION": "#616161",
}

# ══════════════════════════════════════════════════════════════════
# GRAYSCALE GUI 2 of 12 — Grayscale mode display labels & colours
# ══════════════════════════════════════════════════════════════════
_GRAYSCALE_LABELS = {
    "off": "RGB",
    "auto": "AUTO",
    "force": "GRAY",
}
_GRAYSCALE_COLORS = {
    "off": "#aaaaaa",
    "auto": "#00bcd4",
    "force": "#ffeb3b",
}
_GRAYSCALE_CYCLE = ["off", "auto", "force"]
# ══════════════════════════════════════════════════════════════════

_WINDOW_TITLE = "Medevplus IXcentai — Surgical Grade"
_MIN_WIDTH = 1280
_MIN_HEIGHT = 800
_DISPLAY_FPS_CAP = 30.0


class PupilTrackingGUI:
    def __init__(self, root: tk.Tk, colors=None) -> None:
        self.root = root
        self.root.withdraw()
        self.root.title(_WINDOW_TITLE)
        self.root.geometry(f"{_MIN_WIDTH}x{_MIN_HEIGHT}")
        self.root.minsize(_MIN_WIDTH, _MIN_HEIGHT)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if colors is None:
            colors = DarkTheme.apply(root)
        self._colors = colors

        self._runtime_profile = apply_runtime_optimizations(detect_runtime_profile())
        self.cfg = get_config()
        self.logger = get_logger()

        self._detector: Optional[UnifiedDetector] = None
        self._tracker: Optional[EyeKalmanTracker] = None
        self._corneal_calc: Optional[CornealCenterCalculator] = None

        self._current_image: Optional[np.ndarray] = None
        self._current_result: Optional[Any] = None
        self._display_image: Optional[ImageTk.PhotoImage] = None
        self._canvas_image_id: Optional[int] = None
        self._resize_after_id: Optional[str] = None
        self._display_scale: float = 1.0
        self._display_origin: Tuple[int, int] = (0, 0)
        self._display_size: Tuple[int, int] = (0, 0)

        self._video_cap: Optional[cv2.VideoCapture] = None
        self._video_running = False
        self._video_thread: Optional[threading.Thread] = None
        self._camera_mode = False
        self._frame_count = 0
        self._video_paused = False
        self._video_total_frames = 0
        self._video_start_time = 0.0
        self._last_display_update = 0.0
        self._active_source: Optional[Any] = None
        self._settings_apply_after_id: Optional[str] = None
        self._pending_live_apply_reasons: set[str] = set()
        self._suspend_live_settings_apply = False
        self._restart_in_progress = False

        self._fast_engine: Optional[Any] = None
        self._opt_processor: Optional[Any] = None
        self._async_capture: Optional[Any] = None
        self._using_optimized_camera: bool = False
        self._last_opt_stats: Dict[str, Any] = {}
        self._manual_roi: Optional[Dict[str, float]] = None
        # Detection is gated on the ROI: after a video/camera is loaded we show
        # the first frame and hold until the user confirms an ROI, at which point
        # the stored start callable spins up the detection thread.
        self._awaiting_roi: bool = False
        self._pending_detection_start: Optional[Any] = None
        self._roi_edit_active = False
        self._roi_drag_mode: Optional[str] = None
        self._roi_drag_offset: Tuple[float, float] = (0.0, 0.0)
        self._roi_original_before_edit: Optional[Dict[str, float]] = None
        self._roi_preview: Optional[Dict[str, float]] = None
        self._manual_ring: Optional[Dict[str, float]] = None
        self._ring_edit_active = False
        self._ring_drag_mode: Optional[str] = None
        self._ring_drag_offset: Tuple[float, float] = (0.0, 0.0)
        self._ring_original_before_edit: Optional[Dict[str, float]] = None
        self._ring_preview: Optional[Dict[str, float]] = None

        self._results_history: List[Dict[str, Any]] = []

        # ══════════════════════════════════════════════════════════
        # RECORDING — FrameRecorder instances (Overlay + Clean Raw Dual-Stream)
        # ══════════════════════════════════════════════════════════
        self._recorder = FrameRecorder(name="overlay")
        self._raw_recorder = FrameRecorder(name="raw")
        self._recorder.set_status_callback(self._on_recorder_status)
        self._recording_path: Optional[str] = None
        self._raw_recording_path: Optional[str] = None
        self._dual_recording_var = tk.BooleanVar(value=True)
        self._recording_default_path_var = tk.StringVar(value="")
        self._recording_timer_id: Optional[str] = None
        self._recording_fps_var = tk.DoubleVar(value=0.0)
        self._recording_dropped_var = tk.IntVar(value=0)
        # Patient/session storage. The root can be changed from Settings and
        # persists between launches; a patient must be selected before a
        # clinical recording is allowed.
        self._storage_settings_path = Path.home() / "AppData" / "Local" / "MedevplusIXcentai" / "settings.txt"
        self._storage_root_var = tk.StringVar(value=str(Path.home() / "Desktop" / "Centration"))
        self._patient_id_var = tk.StringVar(value="")
        self._patient_name_var = tk.StringVar(value="")
        self._selected_eye_var = tk.StringVar(value="")
        self._active_patient_dir: Optional[Path] = None
        self._add_px_dialog_open = False
        self._patient_summary_var = tk.StringVar(value="No patient selected")
        self._recording_location_var = tk.StringVar(value="Recording location: select a patient and eye")
        self._load_storage_settings()
        # ══════════════════════════════════════════════════════════

        self._init_settings_vars()

        self._build_menu()
        self._build_toolbar()
        self._build_status_bar()
        self._build_main_area()
        self._bind_live_setting_callbacks()
        self._install_crash_guards()

        self.root.update_idletasks()
        self.root.deiconify()
        self.root.after(100, self._init_detector)

    # ================================================================
    # Settings Tk Variables
    # ================================================================

    def _init_settings_vars(self) -> None:
        self._use_optimized_var = tk.BooleanVar(value=True)
        self._fp16_var = tk.BooleanVar(
            value=self._runtime_profile.recommended_fp16
        )
        self._compile_var = tk.BooleanVar(
            value=self._runtime_profile.recommended_compile
        )

        self._resolution_var = tk.IntVar(
            value=self._runtime_profile.recommended_resolution
        )
        self._stride_var = tk.IntVar(value=1)
        self._target_fps_var = tk.DoubleVar(
            value=self._runtime_profile.recommended_target_fps
        )
        self._performance_preset_var = tk.StringVar(value="balanced")

        self._roi_var = tk.BooleanVar(value=True)
        self._roi_cache_var = tk.IntVar(value=5)
        self._roi_status_var = tk.StringVar(value="Manual ROI: Off")
        self._ring_status_var = tk.StringVar(value="Manual Ring: Off")
        self._kalman_process_var = tk.DoubleVar(value=0.03)
        self._kalman_measure_var = tk.DoubleVar(value=0.1)

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 3 of 12 — Grayscale mode Tk variable
        # ══════════════════════════════════════════════════════════
        self._grayscale_mode_var = tk.StringVar(value="off")
        # ══════════════════════════════════════════════════════════

        self._pupil_fill_alpha_var = tk.IntVar(value=0)
        self._limbus_fill_alpha_var = tk.IntVar(value=0)

        # ── Modular Calibration Settings ──
        init_mode = getattr(self.cfg.calibration, "mode", "FIXED_PIXEL_SCALE") if hasattr(self.cfg, "calibration") else "FIXED_PIXEL_SCALE"
        self._calibration_mode_var = tk.StringVar(value=init_mode)
        init_manual_px = float(getattr(self.cfg.calibration, "manual_px_per_mm", 58.2) or 58.2) if hasattr(self.cfg, "calibration") else 58.2
        self._fixed_scale_var = tk.DoubleVar(value=init_manual_px)
        init_corneal = float(getattr(self.cfg.calibration, "corneal_diameter_mm", 12.0) or 12.0) if hasattr(self.cfg, "calibration") else 12.0
        self._corneal_ref_mm_var = tk.DoubleVar(value=init_corneal)
        init_ring = float(getattr(self.cfg.calibration, "suction_ring_diameter_mm", 9.4) or 9.4) if hasattr(self.cfg, "calibration") else 9.4
        self._ring_ref_mm_var = tk.DoubleVar(value=init_ring)

        # ── Interactive 2-Point Ruler Tool State ──
        self._ruler_calibration_active: bool = False
        self._ruler_points: list[tuple[float, float]] = []
        self._ruler_known_dist_mm_var = tk.DoubleVar(value=10.0)


    # ================================================================
    # Detector Initialisation
    # ================================================================

    def _init_detector(self) -> None:
        self._status_var.set("Loading model…")
        self.root.update()

        try:
            self._tracker = EyeKalmanTracker(config=self.cfg)
        except Exception as exc:
            self.logger.error("Failed to init tracker: %s", exc)
            self._tracker = None

        try:
            self._corneal_calc = CornealCenterCalculator(config=self.cfg)
        except Exception as exc:
            self.logger.error("Failed to init corneal calc: %s", exc)
            self._corneal_calc = None

        try:
            # ══════════════════════════════════════════════════════
            # GRAYSCALE GUI 4 of 12 — Pass grayscale_mode to detector
            # ══════════════════════════════════════════════════════
            self._detector = UnifiedDetector(
                config=self.cfg,
                grayscale_mode=self._grayscale_mode_var.get(),
            )
            # ══════════════════════════════════════════════════════

            if self._detector.ml_engine.available:
                tag = "GPU" if _FAST_PIPELINE_AVAILABLE else "GPU (classic)"
                self._model_status_var.set(f"Model: Ready ({tag})")
            else:
                self._model_status_var.set("Model: Classical Only (ML unavailable)")

            self._status_var.set("Ready — Load an image or start camera")
        except Exception as exc:
            self.logger.error("Failed to init detector: %s", exc)
            self._detector = None
            self._model_status_var.set(f"Model: ERROR — {exc}")
            self._status_var.set(
                "Model loading failed — classical detection unavailable"
            )

    # ================================================================
    # Grayscale Mode Control (NEW)
    # ================================================================

    # ══════════════════════════════════════════════════════════════════
    # GRAYSCALE GUI 5 of 12 — Grayscale toggle, apply, display convert
    # ══════════════════════════════════════════════════════════════════

    def _toggle_grayscale(self, event=None) -> None:
        """Cycle grayscale mode: OFF → AUTO → FORCE → OFF.

        Called by the [G] keyboard shortcut and the toolbar button.
        """
        current = self._grayscale_mode_var.get()
        try:
            idx = _GRAYSCALE_CYCLE.index(current)
        except ValueError:
            idx = 0
        next_mode = _GRAYSCALE_CYCLE[(idx + 1) % len(_GRAYSCALE_CYCLE)]
        self._grayscale_mode_var.set(next_mode)
        self._on_grayscale_changed()

    def _on_grayscale_changed(self, *_args) -> None:
        """Apply grayscale mode change to the detector and update UI.

        Called when the settings dropdown or toolbar button changes.
        """
        mode = self._grayscale_mode_var.get()

        # Apply to detector
        if self._detector is not None:
            self._detector.set_grayscale_mode(mode)

        # Update toolbar button appearance
        label = _GRAYSCALE_LABELS.get(mode, "RGB")
        color = _GRAYSCALE_COLORS.get(mode, "#aaaaaa")
        if hasattr(self, "_gray_btn"):
            self._gray_btn.config(text=f"🔲 {label}")
        if hasattr(self, "_gray_indicator"):
            self._gray_indicator.config(
                text=f"  {label}  ",
                foreground=color,
            )

        # Update status
        mode_names = {
            "off": "RGB (original)",
            "auto": "Auto-detect",
            "force": "Forced grayscale",
        }
        self._status_var.set(f"Grayscale mode: {mode_names.get(mode, mode)}")

        # Refresh display immediately
        self._refresh_display()

    def _convert_display_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale for display when mode is active.

        When grayscale mode is FORCE, the displayed image becomes
        grayscale (like an IR camera) with coloured overlays on top.

        When mode is AUTO, converts only if the detector detected
        the input as grayscale.

        When mode is OFF, returns the original frame unchanged.

        Parameters
        ----------
        frame : np.ndarray
            Original BGR frame.

        Returns
        -------
        np.ndarray
            Frame for display — 3-channel BGR uint8.
        """
        mode = self._grayscale_mode_var.get()

        if mode == "off":
            return frame

        if mode == "force":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if mode == "auto":
            # Convert only if detector applied grayscale processing
            if self._detector is not None:
                gs_info = self._detector.last_grayscale_info
                if gs_info is not None and gs_info.conversion_applied:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return frame

    # ══════════════════════════════════════════════════════════════════
    # END GRAYSCALE GUI 5
    # ══════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════
    # RECORDING — Professional recording with FrameRecorder
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _safe_folder_name(value: str) -> str:
        """Return a Windows-safe, readable folder name."""
        cleaned = re.sub(r'[<>:"/\\|?*]+', "_", value.strip())
        return cleaned.strip(". ") or "Unknown"

    def _load_storage_settings(self) -> None:
        try:
            if self._storage_settings_path.exists():
                for line in self._storage_settings_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("data_storage_root="):
                        root = line.split("=", 1)[1].strip()
                        if root:
                            self._storage_root_var.set(root)
                        break
        except (OSError, UnicodeError) as exc:
            self.logger.warning("Could not load storage settings: %s", exc)

    def _save_storage_settings(self) -> bool:
        location = self._storage_root_var.get().strip()
        if not location:
            messagebox.showwarning("Storage location required", "Choose a folder for patient data first.")
            return False
        try:
            root = Path(location).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            self._storage_root_var.set(str(root))
            self._storage_settings_path.parent.mkdir(parents=True, exist_ok=True)
            self._storage_settings_path.write_text(
                f"# Medevplus IXcentai - Settings\ndata_storage_root={root}\n",
                encoding="utf-8",
            )
            self._status_var.set(f"Patient data location saved → {root}")
            return True
        except OSError as exc:
            messagebox.showerror("Storage location error", f"Cannot use that folder:\n{exc}")
            return False

    def _choose_storage_root(self) -> None:
        selected = filedialog.askdirectory(title="Choose patient data storage folder", initialdir=self._storage_root_var.get())
        if selected:
            self._storage_root_var.set(selected)

    def _open_patient_dialog(self) -> None:
        if self._recorder.is_recording:
            return

        self._add_px_dialog_open = True
        self._update_patient_controls()
        c = self._colors

        # Dialog colours — white background, dark text
        DLG_BG      = "#FFFFFF"
        DLG_FG      = "#1A1A1A"
        DLG_FG_HINT = "#555555"
        DLG_BORDER  = "#CCCCCC"
        DLG_INPUT   = "#F5F5F5"
        DLG_ACCENT  = "#0071E3"
        DLG_BTN_SEC = "#E0E0E0"
        DLG_BTN_ACT = "#C8C8C8"

        dialog = tk.Toplevel(self.root)
        dialog.title("Add Patient")
        dialog.transient(self.root)
        dialog.grab_set()

        # Dialog window icon
        try:
            from PIL import Image, ImageTk as _ITk
            import pathlib as _pl
            _icon_path = _pl.Path(__file__).parent / "add-patient.png.png"
            _img = Image.open(_icon_path).resize((32, 32), Image.LANCZOS)
            _icon = _ITk.PhotoImage(_img)
            dialog.iconphoto(False, _icon)
            dialog._icon_ref = _icon  # keep reference alive
        except Exception:
            pass

        dialog.resizable(True, True)

        W, H = 660, 540
        dialog.minsize(W, H)

        self.root.update_idletasks()
        rx = self.root.winfo_rootx() + (self.root.winfo_width()  - W) // 2
        ry = self.root.winfo_rooty() + (self.root.winfo_height() - H) // 2
        dialog.geometry(f"{W}x{H}+{rx}+{ry}")
        dialog.configure(bg=DLG_BG)

        # ── Header ──────────────────────────────────────────────────
        hdr = tk.Frame(dialog, bg=DLG_BG)
        hdr.pack(fill=tk.X, padx=28, pady=(20, 12))
        tk.Label(
            hdr, text="Add Patient",
            font=("Segoe UI", 16, "bold"),
            bg=DLG_BG, fg=DLG_FG,
        ).pack(anchor=tk.W)
        tk.Label(
            hdr, text="Fields marked * are required",
            font=("Segoe UI", 10, "bold"),
            bg=DLG_BG, fg=DLG_FG_HINT,
        ).pack(anchor=tk.W, pady=(4, 0))

        tk.Frame(dialog, bg=DLG_BORDER, height=1).pack(fill=tk.X)

        # ── Body ────────────────────────────────────────────────────
        body = tk.Frame(dialog, bg=DLG_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=28, pady=16)
        body.columnconfigure(0, minsize=160)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, minsize=90)

        patient_id   = tk.StringVar()
        patient_name = tk.StringVar()
        hvid_od      = tk.StringVar()
        hvid_os      = tk.StringVar()
        pentacam_od  = tk.StringVar()
        pentacam_os  = tk.StringVar()

        LBL_FONT = ("Segoe UI", 11, "bold")
        ENT_FONT = ("Segoe UI", 11, "bold")
        BTN_FONT = ("Segoe UI", 10, "bold")

        def _field(grid_row, label_text, var, browse=False):
            tk.Label(
                body, text=label_text,
                font=LBL_FONT,
                bg=DLG_BG, fg=DLG_FG,
                anchor=tk.W,
            ).grid(row=grid_row, column=0, sticky=tk.W, pady=7, padx=(0, 14))

            ent = tk.Entry(
                body, textvariable=var,
                font=ENT_FONT,
                bg=DLG_INPUT, fg=DLG_FG,
                insertbackground=DLG_FG,
                relief=tk.FLAT, bd=0,
                highlightthickness=1,
                highlightbackground=DLG_BORDER,
                highlightcolor=DLG_ACCENT,
            )
            ent.grid(row=grid_row, column=1, sticky=tk.EW, pady=7, ipady=7)

            if browse:
                lbl_copy = label_text
                tk.Button(
                    body, text="Browse",
                    font=BTN_FONT,
                    bg=DLG_BTN_SEC, fg=DLG_FG,
                    activebackground=DLG_BTN_ACT, activeforeground=DLG_FG,
                    relief=tk.FLAT, bd=0, padx=10, pady=5, cursor="hand2",
                    command=lambda t=var, l=lbl_copy: t.set(
                        filedialog.askopenfilename(
                            title=l,
                            filetypes=[
                                ("Pentacam files", "*.jpg *.jpeg *.png *.pdf"),
                                ("All files", "*.*"),
                            ],
                        )
                    ),
                ).grid(row=grid_row, column=2, sticky=tk.EW, padx=(10, 0), pady=7)

        _field(0, "Patient ID *",       patient_id)
        _field(1, "Patient Name *",     patient_name)
        _field(2, "HVID OD (optional)", hvid_od)
        _field(3, "HVID OS (optional)", hvid_os)
        _field(4, "Pentacam OD (optional)", pentacam_od, browse=True)
        _field(5, "Pentacam OS (optional)", pentacam_os, browse=True)

        tk.Frame(dialog, bg=DLG_BORDER, height=1).pack(fill=tk.X)

        # ── Footer ──────────────────────────────────────────────────
        foot = tk.Frame(dialog, bg=DLG_BG)
        foot.pack(fill=tk.X, padx=28, pady=16)

        tk.Button(
            foot, text="Cancel",
            font=BTN_FONT,
            bg=DLG_BTN_SEC, fg=DLG_FG,
            activebackground=DLG_BTN_ACT, activeforeground=DLG_FG,
            relief=tk.FLAT, bd=0, padx=22, pady=8, cursor="hand2",
            command=dialog.destroy,
        ).pack(side=tk.RIGHT, padx=(10, 0))

        add_btn = tk.Button(
            foot, text="Add Patient",
            font=("Segoe UI", 10, "bold"),
            bg=DLG_BTN_SEC, fg=DLG_FG_HINT,
            activebackground=DLG_ACCENT, activeforeground="#FFFFFF",
            relief=tk.FLAT, bd=0, padx=22, pady=8, cursor="hand2",
            state=tk.DISABLED,
        )
        add_btn.pack(side=tk.RIGHT)

        def _update_enabled(*_):
            ok = bool(patient_id.get().strip() and patient_name.get().strip())
            add_btn.config(
                state=tk.NORMAL if ok else tk.DISABLED,
                bg=DLG_ACCENT      if ok else DLG_BTN_SEC,
                fg="#FFFFFF"       if ok else DLG_FG_HINT,
            )

        patient_id.trace_add("write", _update_enabled)
        patient_name.trace_add("write", _update_enabled)

        def create_patient():
            if not self._save_storage_settings():
                return
            try:
                root = Path(self._storage_root_var.get())
                patient_dir = (
                    root
                    / f"PX_{self._safe_folder_name(patient_id.get())}"
                      f"_{self._safe_folder_name(patient_name.get())}"
                )
                for eye in ("OD", "OS"):
                    (patient_dir / eye / "Pentacam").mkdir(parents=True, exist_ok=True)
                metadata = {
                    "patient_id":   patient_id.get().strip(),
                    "patient_name": patient_name.get().strip(),
                    "hvid":         {"OD": hvid_od.get().strip(), "OS": hvid_os.get().strip()},
                    "created_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                (patient_dir / "patient.txt").write_text(
                    "# Patient Record - Medevplus IXcentai\n"
                    f"patient_id={metadata['patient_id']}\n"
                    f"patient_name={metadata['patient_name']}\n"
                    f"hvid_od={metadata['hvid']['OD']}\n"
                    f"hvid_os={metadata['hvid']['OS']}\n"
                    f"created_at={metadata['created_at']}\n",
                    encoding="utf-8",
                )
                for eye, source in (("OD", pentacam_od.get()), ("OS", pentacam_os.get())):
                    (patient_dir / eye / "Pentacam" / "pentacam_values.txt").write_text(
                        f"# Pentacam Values - {eye}\nhvid={metadata['hvid'][eye]}\n",
                        encoding="utf-8",
                    )
                    if source:
                        src = Path(source)
                        if src.is_file():
                            shutil.copy2(src, patient_dir / eye / "Pentacam" / src.name)
                self._active_patient_dir = patient_dir
                self._patient_id_var.set(metadata["patient_id"])
                self._patient_name_var.set(metadata["patient_name"])
                self._selected_eye_var.set("")
                self._patient_summary_var.set(
                    f"PX: {metadata['patient_id']} \u2014 {metadata['patient_name']}"
                    " (choose OD or OS)"
                )
                self._recording_location_var.set(f"Patient folder: {patient_dir}")
                self._add_px_dialog_open = False
                self._update_patient_controls()
                self._status_var.set(f"Patient folder created \u2192 {patient_dir}")
                dialog.destroy()
            except OSError as exc:
                messagebox.showerror(
                    "Patient creation error",
                    f"Could not create patient folder:\n{exc}",
                )

        def close_dialog() -> None:
            self._add_px_dialog_open = False
            self._update_patient_controls()
            dialog.destroy()

        add_btn.config(command=create_patient)
        dialog.bind(
            "<Return>",
            lambda _e: create_patient() if add_btn.cget("state") == tk.NORMAL else None,
        )
        dialog.protocol("WM_DELETE_WINDOW", close_dialog)
        dialog.wait_visibility()
        dialog.focus_set()


    def _set_patient_btn_colors(self) -> None:
        active_eye = self._selected_eye_var.get()
        has_patient = self._active_patient_dir is not None
        add_px_active = bool(getattr(self, "_add_px_dialog_open", False))

        if hasattr(self, "_add_px_btn"):
            if isinstance(self._add_px_btn, tk.Button):
                self._add_px_btn.config(
                    state=tk.DISABLED if self._recorder.is_recording else tk.NORMAL,
                    bg=self._colors.BTN_ACTION if add_px_active else self._colors.BTN_BG,
                    fg=self._colors.BTN_PRIMARY if add_px_active else self._colors.FG_PRIMARY,
                    activebackground=self._colors.BTN_ACTION if add_px_active else self._colors.BTN_HOVER,
                    activeforeground=self._colors.BTN_PRIMARY,
                )
            else:
                self._add_px_btn.configure(
                    style="ToolbarActive.TButton" if add_px_active else "TButton",
                )

        for button in getattr(self, "_eye_buttons", []):
            text = str(button.cget("text"))
            is_active = has_patient and active_eye == text
            button.configure(style="ToolbarActive.TButton" if is_active else "TButton")
            button.config(state=tk.DISABLED if self._recorder.is_recording or not has_patient else tk.NORMAL)

    def _select_eye(self, eye: str) -> None:
        if self._recorder.is_recording or self._active_patient_dir is None:
            return
        self._selected_eye_var.set(eye)
        target = self._active_patient_dir / eye
        self._recording_location_var.set(f"Recording folder: {target}")
        self._patient_summary_var.set(f"PX: {self._patient_id_var.get()} — {self._patient_name_var.get()} | Eye: {eye}")
        self._update_patient_controls()

    def _update_patient_controls(self) -> None:
        locked = self._recorder.is_recording
        if hasattr(self, "_add_px_btn"):
            self._add_px_btn.config(state=tk.DISABLED if locked else tk.NORMAL)
        for button in getattr(self, "_eye_buttons", []):
            button.config(state=tk.DISABLED if locked or self._active_patient_dir is None else tk.NORMAL)
        self._set_patient_btn_colors()

    def _patient_recording_path(self) -> Optional[str]:
        if self._active_patient_dir is None or self._selected_eye_var.get() not in ("OD", "OS"):
            messagebox.showinfo("Patient and eye required", "Click Add PX, then select OD or OS before starting a recording.")
            return None
        eye_dir = self._active_patient_dir / self._selected_eye_var.get()
        eye_dir.mkdir(parents=True, exist_ok=True)
        return str(eye_dir / f"detection_{time.strftime('%Y%m%d_%H%M%S')}.mp4")

    def _choose_recording_path(self) -> Optional[str]:
        """Ask user where to save the recording."""
        default_name = self._recording_default_path_var.get()
        if not default_name:
            default_name = f"recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        path = filedialog.asksaveasfilename(
            title="Save Recording As",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 Video (H.264)", "*.mp4"),
                ("AVI Video", "*.avi"),
                ("All files", "*.*"),
            ],
            initialfile=default_name,
        )
        return path if path else None

    def _start_recording(self) -> None:
        """Start recording the current view (video/camera feed)."""
        if self._recorder.is_recording:
            return

        if self._current_image is None:
            messagebox.showinfo(
                "No Source",
                "Start a video or camera before recording.",
            )
            return

        path = self._patient_recording_path()
        if not path:
            return

        # The companion CSV should describe this recording only, never frames
        # viewed while preparing the camera or a prior patient session.
        self._results_history.clear()

        # Prepare a frame that includes the measurements panel so the
        # recording captures the full on-screen layout (image + table).
        display_image = self._prepare_recording_frame(
            self._current_image,
            self._current_result,
        )
        composite = self._compose_capture_frame(display_image, self._current_result)
        h, w = composite.shape[:2]

        target_fps = 30.0
        if self._video_cap is not None:
            fps = self._video_cap.get(cv2.CAP_PROP_FPS)
            if fps > 0 and fps <= 120:
                target_fps = fps

        if not self._recorder.start(path, w, h, target_fps):
            messagebox.showerror(
                "Recording Error",
                f"Cannot start recording. Check codec support.\nPath: {path}",
            )
            return

        self._recording_path = path

        # ── Start parallel clean RAW stream recording if enabled ──────
        self._raw_recording_path = None
        if self._dual_recording_var.get() and self._current_image is not None:
            raw_h, raw_w = self._current_image.shape[:2]
            p = Path(path)
            raw_path = str(p.with_name(f"{p.stem}_raw{p.suffix}"))
            if self._raw_recorder.start(raw_path, raw_w, raw_h, target_fps):
                self._raw_recording_path = raw_path
                self.logger.info("Clean RAW stream recording started → %s", raw_path)
            else:
                self.logger.warning("Failed to start RAW stream recorder at %s", raw_path)

        self._recording_default_path_var.set(
            f"recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        )

        self._update_recording_ui(started=True)
        self._update_patient_controls()
        status_msg = f"Recording started → {path}"
        if self._raw_recording_path:
            status_msg += f" (+ Clean RAW: {Path(self._raw_recording_path).name})"
        self._status_var.set(status_msg)
        self._start_recording_timer()

    def _stop_recording(self) -> None:
        """Stop recording and save the video file."""
        if not self._recorder.is_recording and not self._raw_recorder.is_recording:
            return

        path = self._recorder.stop()
        raw_path = self._raw_recorder.stop() if self._raw_recorder.is_recording else None
        self._stop_recording_timer()

        elapsed = (
            self._recorder.elapsed_time
            if hasattr(self._recorder, "elapsed_time")
            else 0
        )
        frame_count = (
            self._recorder.frame_count if hasattr(self._recorder, "frame_count") else 0
        )

        self._update_recording_ui(started=False)
        csv_path = None
        if path and self._results_history:
            csv_path = str(Path(path).with_suffix(".csv"))
            self._export_csv(csv_path, show_success=False)
        self._update_patient_controls()
        saved_message = f"Recording saved: {frame_count} frames in {elapsed:.1f}s → {path or 'unknown'}"
        if raw_path:
            saved_message += f"\nClean RAW stream → {raw_path}"
        if csv_path:
            saved_message += f"\nTelemetry CSV → {csv_path}"
        self._recording_location_var.set(saved_message)
        self._status_var.set(f"Saved: {frame_count} frames in {elapsed:.1f}s")
        if path:
            messagebox.showinfo("Dual Recording Saved", saved_message)

    def _toggle_recording(self) -> None:
        """Toggle recording on/off."""
        if self._recorder.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _on_recorder_status(self, status: dict) -> None:
        """Callback from FrameRecorder for status updates."""
        if status.get("is_recording"):
            elapsed = status.get("elapsed_time", 0)
            fps = status.get("fps", 0)
            dropped = status.get("dropped_frames", 0)

            mins, secs = divmod(int(elapsed), 60)
            self._recording_indicator.config(
                text=f"  REC {mins:02d}:{secs:02d} ({fps:.0f}fps)  ",
                foreground="#ff1744",
            )
            self._recording_fps_var.set(fps)
            self._recording_dropped_var.set(dropped)
        else:
            self._recording_indicator.config(text="  --:--  ", foreground="#616161")

    def _start_recording_timer(self) -> None:
        """Start the recording indicator update timer."""
        self._stop_recording_timer()
        self._update_recording_timer()

    def _update_recording_timer(self) -> None:
        """Periodically update the recording indicator."""
        if self._recorder.is_recording:
            elapsed = self._recorder.elapsed_time
            fps = self._recorder.frame_count / elapsed if elapsed > 0 else 0

            mins, secs = divmod(int(elapsed), 60)
            self._recording_indicator.config(
                text=f"  REC {mins:02d}:{secs:02d} ({fps:.0f}fps)  ",
                foreground="#ff1744",
            )
            self._recording_timer_id = self.root.after(
                500, self._update_recording_timer
            )
        else:
            self._stop_recording_timer()

    def _stop_recording_timer(self) -> None:
        """Stop the recording indicator update timer."""
        if self._recording_timer_id is not None:
            self.root.after_cancel(self._recording_timer_id)
            self._recording_timer_id = None

    def _update_recording_ui(self, started: bool = False) -> None:
        """Update recording button state."""
        if hasattr(self, "_rec_btn"):
            if started or self._recorder.is_recording:
                self._rec_btn.config(text="⏹ Stop Rec")
            else:
                self._rec_btn.config(text="⏺ Start Rec")

    def _write_frame_to_recorder(self, frame: np.ndarray) -> None:
        """Write a frame to the recorder (non-blocking)."""
        # 1. Write clean pristine raw frame to raw recorder for training
        try:
            if self._raw_recorder.is_recording and self._current_image is not None:
                self._raw_recorder.write(self._current_image)
        except Exception:
            pass

        # 2. Write composed UI overlay (image + measurements panel) to overlay recorder
        try:
            if self._recorder.is_recording and self._current_image is not None:
                display_image = self._prepare_recording_frame(
                    self._current_image, self._current_result
                )
                composite = self._compose_capture_frame(display_image, self._current_result)
                self._recorder.write(composite)
                return
        except Exception:
            # Fall back to naive write if composition fails
            pass

        if self._recorder.is_recording:
            self._recorder.write(frame)

    def _prepare_recording_frame(self, frame: np.ndarray, result: Any) -> np.ndarray:
        """Prepare a frame that mirrors the current on-screen display for recording."""
        mode = self._grayscale_mode_var.get()
        if mode == "off":
            image = frame.copy()
        else:
            image = self._convert_display_frame(frame.copy())

        if result is not None and self._show_overlay.get():
            self._draw_overlay_scaled(image, result, 1.0)

        self._draw_manual_roi_overlay(image, 1.0)
        self._draw_manual_ring_overlay(image, 1.0)
        if self._show_debug_overlay.get():
            self._draw_debug_overlay(image, 1.0)

        return image

    @staticmethod
    def _hex_to_bgr(value: str) -> Tuple[int, int, int]:
        value = value.lstrip("#")
        if len(value) != 6:
            return (200, 200, 200)
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
        return (b, g, r)

    def _measurement_capture_sections(self) -> List[Tuple[str, Tuple[int, int, int], List[Tuple[str, str]]]]:
        return [
            (
                "PUPIL",
                self._hex_to_bgr(self._colors.PUPIL),
                [
                    ("Center", self._pv["center"].get()),
                    ("Diameter (px)", self._pv["diameter_px"].get()),
                    ("Diameter (mm)", self._pv["diameter_mm"].get()),
                    ("Semi-Major (px)", self._pv["semi_major"].get()),
                    ("Semi-Major (mm)", self._pv["semi_major_mm"].get()),
                    ("Semi-Minor (px)", self._pv["semi_minor"].get()),
                    ("Semi-Minor (mm)", self._pv["semi_minor_mm"].get()),
                    ("Angle", self._pv["angle"].get()),
                    ("Fit Type", self._pv["fit_type"].get()),
                    ("Confidence", self._pv["confidence"].get()),
                    ("Quality", self._pv["quality"].get()),
                ],
            ),
            (
                "LIMBUS",
                self._hex_to_bgr(self._colors.LIMBUS),
                [
                    ("Center", self._lv["center"].get()),
                    ("Diameter (px)", self._lv["diameter_px"].get()),
                    ("Diameter (mm)", self._lv["diameter_mm"].get()),
                    ("Semi-Major (px)", self._lv["semi_major"].get()),
                    ("Semi-Major (mm)", self._lv["semi_major_mm"].get()),
                    ("Semi-Minor (px)", self._lv["semi_minor"].get()),
                    ("Semi-Minor (mm)", self._lv["semi_minor_mm"].get()),
                    ("Angle", self._lv["angle"].get()),
                    ("Fit Type", self._lv["fit_type"].get()),
                    ("Confidence", self._lv["confidence"].get()),
                    ("Quality", self._lv["quality"].get()),
                ],
            ),
            (
                "CORNEAL OFFSET",
                self._hex_to_bgr(self._colors.OFFSET),
                [
                    ("Corneal Centre", self._ov["corneal_center"].get()),
                    ("Offset (px)", self._ov["offset_px"].get()),
                    ("Offset (mm)", self._ov["offset_mm"].get()),
                    ("Offset dX,dY px", self._ov["offset_vec_px"].get()),
                    ("Offset dX,dY mm", self._ov["offset_vec_mm"].get()),
                    ("Offset Angle", self._ov["offset_angle"].get()),
                    ("Pupil/Limbus", self._ov["pupil_limbus_ratio"].get()),
                ],
            ),
            (
                "CORNEAL DIMENSIONS (WTW)",
                self._hex_to_bgr(self._colors.LIMBUS),
                [
                    ("Horizontal", self._wtw_vars["horizontal"].get()),
                    ("Vertical", self._wtw_vars["vertical"].get()),
                    ("Mean", self._wtw_vars["mean"].get()),
                ],
            ),
            (
                "PROCESSING",
                self._hex_to_bgr(self._colors.PROCESSING),
                [
                    ("Proc. Time", self._proc_time_var.get()),
                    ("Latency", self._latency_var.get()),
                    ("Latency Avg", self._latency_avg_var.get()),
                    ("Dropped/Stale", self._drop_var.get()),
                    ("Tracking", self._tracking_state_var.get()),
                    ("FPS", self._fps_var.get()),
                    ("Frame", self._frame_var.get()),
                    ("Image Size", self._image_size_var.get()),
                    ("Pipeline", self._pipeline_var.get()),
                    ("Grayscale", self._gray_mode_var_display.get()),
                ],
            ),
        ]

    # Map of non-ASCII glyphs used in the live Tk UI to ASCII equivalents
    # that OpenCV's cv2.putText (Hershey fonts) CAN render. Without this,
    # cv2.putText draws '?' for any codepoint outside its ASCII glyph table,
    # which is why the recording showed "155.5??" etc. The live UI is
    # unaffected — this only touches the composited/recorded panel.
    _CAPTURE_GLYPH_MAP = {
        "°": " deg",   # ° degree
        "×": "x",      # × multiplication
        "—": "-",      # — em dash
        "–": "-",      # – en dash
        "→": "->",     # → arrow
        "µ": "u",      # µ micro
        "≥": ">=",     # ≥
        "≤": "<=",     # ≤
        "±": "+/-",    # ±
        "•": "*",      # • bullet
        "⚠": "!",      # ⚠ warning
        "⚡": "",       # ⚡ lightning
        "✓": "OK",     # ✓ check
        "✗": "X",      # ✗ cross
    }

    def _ascii_for_capture(self, text: str) -> str:
        """Sanitize a UI string so cv2.putText renders it without '?' glyphs.

        Replaces known symbols with ASCII equivalents, then drops any
        remaining non-ASCII codepoint so nothing renders as a question mark.
        """
        if not text:
            return text
        for uni, ascii_rep in self._CAPTURE_GLYPH_MAP.items():
            if uni in text:
                text = text.replace(uni, ascii_rep)
        # Drop any leftover non-ASCII so OpenCV never emits '?'
        if any(ord(ch) > 127 for ch in text):
            text = text.encode("ascii", "ignore").decode("ascii")
        return text

    def _render_measurements_capture(self, height: int, width: int) -> np.ndarray:
        panel = np.full((height, width, 3), self._hex_to_bgr(self._colors.BG_SECONDARY), dtype=np.uint8)
        cv2.rectangle(
            panel,
            (0, 0),
            (width - 1, height - 1),
            self._hex_to_bgr(self._colors.BORDER),
            1,
        )
        pad = max(10, height // 72)
        gutter = max(10, width // 42)
        inner_w = width - pad * 2
        col_w = max(220, (inner_w - gutter) // 2)
        title_font = max(0.42, min(0.72, height / 900.0))
        body_font = max(0.34, min(0.54, height / 1080.0))
        line_h = max(16, int(height / 36))
        row_gap = max(3, int(line_h * 0.2))
        section_gap = max(8, int(line_h * 0.55))
        summary_box_h = max(62, int(height * 0.09))
        summary_gap = max(8, gutter // 2)
        summary_w = max(120, (inner_w - summary_gap) // 2)
        fg_primary = self._hex_to_bgr(self._colors.FG_PRIMARY)
        fg_secondary = self._hex_to_bgr(self._colors.FG_SECONDARY)
        card_bg = self._hex_to_bgr(self._colors.BG_TERTIARY)
        quality_color = self._hex_to_bgr(
            _QUALITY_COLORS.get(self._summary_quality_var.get(), self._colors.FG_PRIMARY)
        )
        tracking_color = self._hex_to_bgr(
            {
                "Tracking Stable": self._colors.SURGICAL,
                "Tracking Acquiring": self._colors.CLINICAL,
                "Tracking Degraded": self._colors.RESEARCH,
                "No Detection": self._colors.INSUFFICIENT,
                "Ready": self._colors.ACCENT,
                "Waiting": self._colors.FG_SECONDARY,
            }.get(self._summary_tracking_var.get(), self._colors.FG_PRIMARY)
        )

        summaries = [
            ("QUALITY", self._summary_quality_var.get(), quality_color),
            ("TRACKING", self._summary_tracking_var.get(), tracking_color),
            ("LATENCY", self._summary_latency_var.get(), fg_primary),
            ("PIPELINE", self._summary_pipeline_var.get(), fg_primary),
        ]
        for idx, (label, value, color) in enumerate(summaries):
            row = idx // 2
            col = idx % 2
            x0 = pad + col * (summary_w + summary_gap)
            y0 = pad + row * (summary_box_h + summary_gap)
            x1 = min(width - pad, x0 + summary_w)
            cv2.rectangle(panel, (x0, y0), (x1, y0 + summary_box_h), card_bg, -1)
            cv2.rectangle(panel, (x0, y0), (x1, y0 + summary_box_h), self._hex_to_bgr(self._colors.BORDER), 1)
            cv2.putText(panel, self._ascii_for_capture(label), (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, fg_secondary, 1, cv2.LINE_AA)
            cv2.putText(panel, self._ascii_for_capture(value or "---"), (x0 + 10, y0 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)

        sections = self._measurement_capture_sections()
        left_sections = sections[:2]
        right_sections = sections[2:]
        start_y = pad * 2 + summary_box_h * 2 + summary_gap

        def draw_section_column(items, x_start):
            y = start_y
            for title, accent, rows in items:
                box_h = max(60, 34 + len(rows) * (line_h + row_gap))
                if y + box_h > height - pad:
                    box_h = max(40, height - pad - y)
                cv2.rectangle(panel, (x_start, y), (x_start + col_w, min(height - pad, y + box_h)), card_bg, -1)
                cv2.rectangle(panel, (x_start, y), (x_start + col_w, min(height - pad, y + box_h)), self._hex_to_bgr(self._colors.BORDER), 1)
                cv2.putText(panel, self._ascii_for_capture(title), (x_start + 10, y + 24), cv2.FONT_HERSHEY_SIMPLEX, title_font, accent, 2, cv2.LINE_AA)
                row_y = y + 48
                for label, value in rows:
                    if row_y > y + box_h - 8:
                        break
                    clean_label = self._ascii_for_capture(label.replace("_", " ").title())
                    clean_value = self._ascii_for_capture((value or "---").replace("\n", " "))
                    if len(clean_value) > 32:
                        clean_value = clean_value[:29] + "..."
                    cv2.putText(panel, clean_label, (x_start + 10, row_y), cv2.FONT_HERSHEY_SIMPLEX, body_font, fg_secondary, 1, cv2.LINE_AA)
                    text_size = cv2.getTextSize(clean_value, cv2.FONT_HERSHEY_SIMPLEX, body_font, 1)[0]
                    value_x = max(x_start + 140, x_start + col_w - 10 - text_size[0])
                    cv2.putText(panel, clean_value, (value_x, row_y), cv2.FONT_HERSHEY_SIMPLEX, body_font, fg_primary, 1, cv2.LINE_AA)
                    row_y += line_h + row_gap
                y += box_h + section_gap

        draw_section_column(left_sections, pad)
        draw_section_column(right_sections, pad + col_w + gutter)
        return panel

    def _compose_capture_frame(self, image: np.ndarray, result: Any) -> np.ndarray:
        img_h, img_w = image.shape[:2]
        panel_w = max(700, int(img_w * 0.62))
        panel = self._render_measurements_capture(img_h, panel_w)
        divider = np.full((img_h, 3, 3), self._hex_to_bgr(self._colors.BORDER), dtype=np.uint8)
        return np.concatenate([image, divider, panel], axis=1)

    # ══════════════════════════════════════════════════════════════════
    # END RECORDING
    # ══════════════════════════════════════════════════════════════════

    # ================================================================
    # Menu
    # ================================================================

    def _build_menu(self) -> None:
        c = self._colors
        menu_cfg = dict(
            bg=c.BG_SECONDARY,
            fg=c.FG_PRIMARY,
            activebackground=c.ACCENT_DIM,
            activeforeground=c.FG_PRIMARY,
            relief="flat",
            borderwidth=0,
        )
        menubar = tk.Menu(self.root, **menu_cfg)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0, **menu_cfg)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="Open Image…",
            command=self._open_image,
            accelerator="Ctrl+O",
        )
        file_menu.add_command(
            label="Open Video…",
            command=self._open_video,
            accelerator="Ctrl+V",
        )
        file_menu.add_separator()
        file_menu.add_command(label="Open Folder…", command=self._open_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results CSV…", command=self._export_csv)
        file_menu.add_command(label="Export Results TXT…", command=self._export_txt)
        file_menu.add_command(label="Save Snapshot…", command=self._export_snapshot)
        file_menu.add_separator()
        file_menu.add_command(label="Save Snapshot…", command=self._export_snapshot)
        # ══════════════════════════════════════════════════════════
        # RECORDING 3 of 8 — Recording menu options
        # ══════════════════════════════════════════════════════════
        file_menu.add_separator()
        self._recording_menu_var = tk.StringVar(value="Start Recording")
        file_menu.add_command(
            label="Start Recording…",
            command=self._start_recording,
            accelerator="Ctrl+R",
        )
        file_menu.add_command(
            label="Stop Recording",
            command=self._stop_recording,
            accelerator="Ctrl+Shift+R",
        )
        file_menu.add_separator()
        # ══════════════════════════════════════════════════════════
        file_menu.add_command(
            label="Exit", command=self._on_close, accelerator="Ctrl+Q"
        )

        camera_menu = tk.Menu(menubar, tearoff=0, **menu_cfg)
        menubar.add_cascade(label="Camera", menu=camera_menu)
        camera_menu.add_command(label="Start Camera", command=self._start_camera)
        camera_menu.add_command(label="Stop Camera", command=self._stop_video)

        view_menu = tk.Menu(menubar, tearoff=0, **menu_cfg)
        menubar.add_cascade(label="View", menu=view_menu)
        self._show_overlay = tk.BooleanVar(value=True)
        self._show_pupil = tk.BooleanVar(value=True)
        self._show_limbus = tk.BooleanVar(value=True)
        self._show_offset = tk.BooleanVar(value=True)
        self._show_centers = tk.BooleanVar(value=True)
        self._show_ring_center = tk.BooleanVar(value=False)
        self._show_measurements = tk.BooleanVar(value=False)
        self._show_debug_overlay = tk.BooleanVar(value=False)
        view_menu.add_checkbutton(
            label="Show Overlay",
            variable=self._show_overlay,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show Pupil",
            variable=self._show_pupil,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show Limbus",
            variable=self._show_limbus,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show Offset Line",
            variable=self._show_offset,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show Centers",
            variable=self._show_centers,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show Ring Center",
            variable=self._show_ring_center,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show On-Image Measurements",
            variable=self._show_measurements,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show Debug Overlay",
            variable=self._show_debug_overlay,
            command=self._refresh_display,
        )

        self.root.bind("<Control-o>", lambda _e: self._open_image())
        self.root.bind("<Control-v>", lambda _e: self._open_video())
        self.root.bind("<Control-q>", lambda _e: self._on_close())
        self.root.bind("<space>", lambda _e: self._toggle_pause())
        # ══════════════════════════════════════════════════════════
        # RECORDING 4 of 8 — Recording keyboard shortcuts
        # ══════════════════════════════════════════════════════════
        self.root.bind("<Control-r>", lambda _e: self._start_recording())
        self.root.bind("<Control-R>", lambda _e: self._toggle_recording())
        # ══════════════════════════════════════════════════════════

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 6 of 12 — [G] keyboard shortcut
        # ══════════════════════════════════════════════════════════
        self.root.bind("<g>", self._toggle_grayscale)
        self.root.bind("<G>", self._toggle_grayscale)
        self.root.bind("<Return>", self._confirm_active_selection)
        self.root.bind("<Escape>", self._cancel_active_selection)
        self.root.bind("<Left>", lambda e: self._nudge_roi(-1, 0, e))
        self.root.bind("<Right>", lambda e: self._nudge_roi(1, 0, e))
        self.root.bind("<Up>", lambda e: self._nudge_roi(0, -1, e))
        self.root.bind("<Down>", lambda e: self._nudge_roi(0, 1, e))
        # ══════════════════════════════════════════════════════════

    # ================================================================
    # Toolbar
    # ================================================================

    def _toolbar_set_active(self, active_btn) -> None:
        """Highlight active_btn blue; reset all other tracked toolbar buttons."""
        for btn in getattr(self, "_toolbar_tracked_btns", []):
            try:
                is_active = btn is active_btn
                if isinstance(btn, tk.Button):
                    btn.config(
                        bg=self._colors.BTN_ACTION if is_active else self._colors.BTN_BG,
                        fg=self._colors.BTN_PRIMARY if is_active else self._colors.FG_PRIMARY,
                    )
                else:
                    btn.configure(style="ToolbarActive.TButton" if is_active else "TButton")
            except Exception:
                pass

    def _build_toolbar(self) -> None:
        c = self._colors
        # Outer bar spans the window: it holds the horizontally scrollable
        # ribbon on the left and a pinned, always-visible quality badge on
        # the right (the badge must never scroll out of view).
        bar = ttk.Frame(self.root, style="Primary.TFrame")
        bar.pack(side=tk.TOP, fill=tk.X, padx=0, pady=(0, 1))

        self._quality_label = ttk.Label(
            bar,
            text="  NO IMAGE  ",
            style="Quality.TLabel",
            anchor="center",
        )
        self._quality_label.pack(side=tk.RIGHT, padx=10)

        # Scrollable ribbon: a Canvas hosting an inner frame, with a slider
        # (horizontal scrollbar) that appears only when the ribbon overflows
        # the available width so every button stays reachable.
        scroll_wrap = ttk.Frame(bar, style="Primary.TFrame")
        scroll_wrap.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._toolbar_canvas = tk.Canvas(
            scroll_wrap,
            bg=c.BG_PRIMARY,
            highlightthickness=0,
            borderwidth=0,
            height=1,
        )
        self._toolbar_scroll = ttk.Scrollbar(
            scroll_wrap,
            orient=tk.HORIZONTAL,
            command=self._toolbar_canvas.xview,
        )
        self._toolbar_canvas.configure(xscrollcommand=self._toolbar_scroll.set)
        self._toolbar_canvas.pack(side=tk.TOP, fill=tk.X, expand=True)
        # The scrollbar is packed/unpacked on demand by _sync_toolbar().
        self._toolbar_scroll_shown = False

        # Inner frame kept as ``toolbar`` so all button rows below are hosted
        # inside the scrollable area unchanged.
        toolbar = ttk.Frame(self._toolbar_canvas, style="Primary.TFrame")
        self._toolbar_inner = toolbar
        self._toolbar_window = self._toolbar_canvas.create_window(
            (0, 0), window=toolbar, anchor="nw"
        )
        self._toolbar_tracked_btns: list = []

        _vid_btn = ttk.Button(toolbar, text="🏞️ Video", command=lambda: [self._open_video(), self._toolbar_set_active(_vid_btn)])
        _vid_btn.pack(side=tk.LEFT, padx=2)
        self._toolbar_tracked_btns.append(_vid_btn)
        _cam_btn = ttk.Button(toolbar, text="📷 Camera", command=lambda: [self._start_camera(), self._toolbar_set_active(_cam_btn)])
        _cam_btn.pack(side=tk.LEFT, padx=2)
        self._toolbar_tracked_btns.append(_cam_btn)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self._pause_btn = ttk.Button(
            toolbar,
            text="⏸ Pause",
            command=self._toggle_pause,
            state=tk.DISABLED,
        )
        self._pause_btn.pack(side=tk.LEFT, padx=2)
        self._pause_btn.pack(side=tk.LEFT, padx=2)
        self._toolbar_tracked_btns.append(self._pause_btn)
        _stop_btn = ttk.Button(toolbar, text="⏹ Stop", command=lambda: [self._stop_video(), self._toolbar_set_active(None)])
        _stop_btn.pack(side=tk.LEFT, padx=2)
        self._toolbar_tracked_btns.append(_stop_btn)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        # Load Add PX icon for toolbar button
        try:
            from PIL import Image as _PILImg, ImageTk as _PILITk
            import pathlib as _pl
            _apx_icon_path = _pl.Path(__file__).parent / "pupil_tracking" / "interface" / "add-patient.png.png"
            if not _apx_icon_path.exists():
                _apx_icon_path = _pl.Path("pupil_tracking") / "interface" / "add-patient.png.png"
            _apx_img = _PILImg.open(_apx_icon_path).resize((18, 18), _PILImg.LANCZOS)
            self._add_px_icon = _PILITk.PhotoImage(_apx_img)
        except Exception:
            self._add_px_icon = None

        if self._add_px_icon:
            self._add_px_btn = tk.Button(
                toolbar,
                text=" Add PX",
                image=self._add_px_icon,
                compound=tk.LEFT,
                font=("Segoe UI", 10),
                bg=self._colors.BTN_BG,
                fg=self._colors.FG_PRIMARY,
                activebackground=self._colors.BTN_ACTIVE,
                activeforeground=self._colors.FG_PRIMARY,
                relief=tk.FLAT, bd=0, padx=6, pady=3, cursor="hand2",
                command=lambda: [self._open_patient_dialog(), self._toolbar_set_active(self._add_px_btn)],
            )
        else:
            self._add_px_btn = ttk.Button(
                toolbar,
                text="⊕ Add PX",
                command=lambda: [self._open_patient_dialog(), self._toolbar_set_active(self._add_px_btn)],
            )
        self._toolbar_tracked_btns.append(self._add_px_btn)
        self._add_px_btn.pack(side=tk.LEFT, padx=2)
        self._eye_buttons: List[ttk.Button] = []
        for eye in ("OD", "OS"):
            button = ttk.Button(toolbar, text=eye, command=lambda current_eye=eye: self._select_eye(current_eye), state=tk.DISABLED)
            button.pack(side=tk.LEFT, padx=2)
            self._eye_buttons.append(button)

        # ══════════════════════════════════════════════════════════
        # RECORDING — Recording toolbar button
        # ══════════════════════════════════════════════════════════
        self._rec_btn = ttk.Button(
            toolbar,
            text="⏺ Start Rec",
            command=self._toggle_recording,
        )
        self._rec_btn.pack(side=tk.LEFT, padx=2)

        self._recording_indicator = ttk.Label(
            toolbar,
            text="  --:--  ",
            style="Quality.TLabel",
            foreground="#616161",
        )
        self._recording_indicator.pack(side=tk.LEFT, padx=(0, 4))

        self._dual_rec_check = ttk.Checkbutton(
            toolbar,
            text="Dual Rec (Raw+Overlay)",
            variable=self._dual_recording_var,
        )
        self._dual_rec_check.pack(side=tk.LEFT, padx=(2, 4))
        # ══════════════════════════════════════════════════════════

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 7 of 12 — Grayscale toggle button in toolbar
        #
        # Button cycles: RGB → AUTO → GRAY → RGB
        # Indicator label shows current mode with colour coding
        # ══════════════════════════════════════════════════════════
        self._gray_btn = ttk.Button(
            toolbar,
            text="🔲 RGB",
            command=self._toggle_grayscale,
        )
        self._gray_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self._roi_btn = ttk.Button(
            toolbar,
            text="ROI",
            command=self._toggle_roi,
        )
        self._roi_btn.pack(side=tk.LEFT, padx=2)

        self._ring_btn = ttk.Button(
            toolbar,
            text="Ring",
            command=self._toggle_ring,
        )
        self._ring_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        _csv_btn = ttk.Button(toolbar, text="Export CSV", command=lambda: [self._export_csv(), self._toolbar_set_active(_csv_btn)])
        _csv_btn.pack(side=tk.LEFT, padx=2)
        self._toolbar_tracked_btns.append(_csv_btn)
        _snap_btn = ttk.Button(toolbar, text="Snapshot", command=lambda: [self._export_snapshot(), self._toolbar_set_active(_snap_btn)])
        _snap_btn.pack(side=tk.LEFT, padx=2)
        self._toolbar_tracked_btns.append(_snap_btn)


        # ── Horizontal scroll plumbing ──────────────────────────────
        # Keeps the canvas exactly one ribbon-row tall, updates the
        # scrollable region, and shows the slider only when the ribbon is
        # wider than the visible area.
        def _sync_toolbar(_event=None) -> None:
            canvas = self._toolbar_canvas
            inner = self._toolbar_inner
            req_w = inner.winfo_reqwidth()
            req_h = inner.winfo_reqheight()
            # Match the canvas height to the ribbon only when it actually
            # changes, so we never fight the canvas <Configure> event.
            if req_h != getattr(self, "_toolbar_req_h", None):
                self._toolbar_req_h = req_h
                canvas.configure(height=req_h)
            canvas.configure(scrollregion=(0, 0, req_w, req_h))
            need = req_w > canvas.winfo_width() + 1
            if need and not self._toolbar_scroll_shown:
                self._toolbar_scroll.pack(side=tk.BOTTOM, fill=tk.X)
                self._toolbar_scroll_shown = True
            elif not need and self._toolbar_scroll_shown:
                self._toolbar_scroll.pack_forget()
                self._toolbar_scroll_shown = False
                canvas.xview_moveto(0.0)

        def _on_toolbar_wheel(event) -> str:
            # Wheel (and Shift+wheel) over the ribbon scrolls it sideways.
            if self._toolbar_scroll_shown and event.delta:
                self._toolbar_canvas.xview_scroll(
                    int(-event.delta / 120), "units"
                )
            return "break"

        self._toolbar_inner.bind("<Configure>", _sync_toolbar)
        self._toolbar_canvas.bind("<Configure>", _sync_toolbar)
        self._toolbar_canvas.bind("<MouseWheel>", _on_toolbar_wheel)
        self._toolbar_canvas.bind("<Shift-MouseWheel>", _on_toolbar_wheel)
        self._toolbar_canvas.after_idle(_sync_toolbar)

    # ================================================================
    # Main Area
    # ================================================================

    def _build_main_area(self) -> None:
        c = self._colors
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Default split ratio: 7:3 (image/loading on the left, panels on the right)
        left_frame = ttk.Frame(main, style="Primary.TFrame")
        main.add(left_frame, weight=65)

        self._build_progress_frame(left_frame)

        self._canvas = tk.Canvas(
            left_frame,
            bg=c.CANVAS_BG,
            cursor="crosshair",
            highlightthickness=0,
            borderwidth=0,
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self._canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self._canvas.bind("<MouseWheel>", self._on_canvas_wheel)

        right_frame = ttk.Frame(main, width=500)
        main.add(right_frame, weight=35)

        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        meas_frame = ttk.Frame(notebook, padding=8)
        notebook.add(meas_frame, text="Measurements")
        self._build_measurements_panel(meas_frame)

        detail_frame = ttk.Frame(notebook, padding=8)
        notebook.add(detail_frame, text="Details")
        self._build_details_panel(detail_frame)

        settings_frame = ttk.Frame(notebook, padding=8)
        notebook.add(settings_frame, text="⚙ Settings")
        self._build_settings_panel(settings_frame)

        # Ensure the PanedWindow sash initial position is set once layout is realized.
        # Without this, some Tk themes/OS window managers keep the sash at a default
        # position until the user resizes the window.
        def _set_initial_sash() -> None:
            w = main.winfo_width()
            if w is None or w <= 1:
                self.root.after(150, _set_initial_sash)
                return
            main.sashpos(0, int(w * 0.70))

        self.root.after(0, _set_initial_sash)


    def _build_progress_frame(self, parent: ttk.Frame) -> None:
        self._progress_outer = ttk.LabelFrame(
            parent,
            text="Video Progress",
            padding=4,
        )
        self._progress_outer.pack(side=tk.BOTTOM, fill=tk.X, padx=2, pady=(2, 0))

        self._progress_bar = ttk.Progressbar(
            self._progress_outer, mode="determinate", maximum=100
        )
        self._progress_bar.pack(fill=tk.X, padx=4, pady=(2, 2))

        info_row = ttk.Frame(self._progress_outer)
        info_row.pack(fill=tk.X, padx=4, pady=(0, 2))

        self._progress_label_var = tk.StringVar(value="No video loaded")
        ttk.Label(
            info_row,
            textvariable=self._progress_label_var,
            style="Muted.TLabel",
        ).pack(side=tk.LEFT)

        self._eta_label_var = tk.StringVar(value="")
        ttk.Label(
            info_row,
            textvariable=self._eta_label_var,
            style="Muted.TLabel",
        ).pack(side=tk.RIGHT)

    # ================================================================
    # Settings Panel
    # ================================================================

    def _build_settings_panel(self, parent: ttk.Frame) -> None:
        c = self._colors
        canvas = tk.Canvas(
            parent,
            highlightthickness=0,
            bg=c.BG_SECONDARY,
            borderwidth=0,
        )
        sb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        sf = ttk.Frame(canvas)
        sf.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sn = ("Consolas", 9)
        sw = 24

        storage_lf = ttk.LabelFrame(sf, text="Patient Data Storage", padding=8)
        storage_lf.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(storage_lf, text="Set this once during installation. You can change it later.", style="Muted.TLabel").pack(anchor=tk.W, pady=(0, 6))
        storage_row = ttk.Frame(storage_lf)
        storage_row.pack(fill=tk.X)
        ttk.Entry(storage_row, textvariable=self._storage_root_var, font=sn).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(storage_row, text="Browse…", command=self._choose_storage_root).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(storage_lf, text="Save storage location", command=self._save_storage_settings).pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(storage_lf, text="Structure: PX_ID_Name / OD or OS / detection video + CSV + Pentacam", style="Muted.TLabel").pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(storage_lf, textvariable=self._recording_location_var, style="Tiny.TLabel", wraplength=430, justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 8 of 12 — Grayscale section in Settings
        # ══════════════════════════════════════════════════════════
        g_lf = ttk.LabelFrame(sf, text="🔲 Grayscale Mode", padding=8)
        g_lf.pack(fill=tk.X, padx=4, pady=4)

        g_desc = ttk.Label(
            g_lf,
            text=(
                "Convert display to grayscale (like IR camera).\n"
                "Detection still works — overlays shown in colour.\n"
                "Press [G] to toggle quickly."
            ),
            style="Muted.TLabel",
            justify=tk.LEFT,
        )
        g_desc.pack(anchor=tk.W, pady=(0, 6))

        g_row = ttk.Frame(g_lf)
        g_row.pack(fill=tk.X, pady=2)
        ttk.Label(
            g_row,
            text="Mode:",
            font=sn,
            width=12,
        ).pack(side=tk.LEFT)

        for mode_val, mode_label, mode_desc in [
            ("off", "RGB (Original)", "Show original colour image"),
            ("auto", "Auto-Detect", "Grayscale only if input is grayscale"),
            ("force", "Force Grayscale", "Always show as grayscale (IR look)"),
        ]:
            rb = ttk.Radiobutton(
                g_lf,
                text=f"{mode_label}  —  {mode_desc}",
                variable=self._grayscale_mode_var,
                value=mode_val,
                command=self._on_grayscale_changed,
            )
            rb.pack(anchor=tk.W, padx=(20, 0), pady=1)

        self._gray_settings_status = tk.StringVar(value="Current: RGB")
        ttk.Label(
            g_lf,
            textvariable=self._gray_settings_status,
            font=sn,
            foreground=c.ACCENT,
        ).pack(anchor=tk.W, pady=(6, 0))
        # ══════════════════════════════════════════════════════════
        # END GRAYSCALE GUI 8
        # ══════════════════════════════════════════════════════════

        p_lf = ttk.LabelFrame(sf, text="Pipeline", padding=8)
        p_lf.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(
            p_lf,
            text="Live Preset:",
            font=sn,
        ).pack(anchor=tk.W)
        for preset_value, preset_label in [
            ("max_accuracy", "Max Accuracy"),
            ("balanced", "Balanced"),
            ("low_latency", "Low Latency"),
        ]:
            ttk.Radiobutton(
                p_lf,
                text=preset_label,
                variable=self._performance_preset_var,
                value=preset_value,
                command=self._apply_performance_preset,
            ).pack(anchor=tk.W, padx=(20, 0), pady=1)

        ttk.Checkbutton(
            p_lf,
            text="Use Optimised Pipeline (when available)",
            variable=self._use_optimized_var,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            p_lf,
            text="FP16 Half-Precision",
            variable=self._fp16_var,
            command=self._invalidate_fast_engine,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            p_lf,
            text="torch.compile (JIT)",
            variable=self._compile_var,
            command=self._invalidate_fast_engine,
        ).pack(anchor=tk.W)

        avail = "✓ Available" if _FAST_PIPELINE_AVAILABLE else "✗ Not installed"
        ttk.Label(
            p_lf,
            text=f"Fast pipeline: {avail}",
            style="Muted.TLabel",
            foreground=(c.SURGICAL if _FAST_PIPELINE_AVAILABLE else c.INSUFFICIENT),
        ).pack(anchor=tk.W, pady=(4, 0))

        v_lf = ttk.LabelFrame(sf, text="Video Processing", padding=8)
        v_lf.pack(fill=tk.X, padx=4, pady=4)

        r_row = ttk.Frame(v_lf)
        r_row.pack(fill=tk.X, pady=2)
        ttk.Label(r_row, text="Inference Resolution:", font=sn, width=sw).pack(
            side=tk.LEFT
        )
        self._res_display = tk.StringVar(value=str(self._resolution_var.get()))
        ttk.Scale(
            r_row,
            from_=192,
            to=512,
            variable=self._resolution_var,
            command=lambda v: self._res_display.set(str(int(float(v)))),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(r_row, textvariable=self._res_display, font=sn, width=4).pack(
            side=tk.LEFT
        )

        s_row = ttk.Frame(v_lf)
        s_row.pack(fill=tk.X, pady=2)
        ttk.Label(s_row, text="Frame Stride:", font=sn, width=sw).pack(side=tk.LEFT)
        ttk.Spinbox(
            s_row,
            from_=1,
            to=30,
            textvariable=self._stride_var,
            width=5,
            font=sn,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(s_row, text="(1 = every frame)", font=sn).pack(side=tk.LEFT)

        f_row = ttk.Frame(v_lf)
        f_row.pack(fill=tk.X, pady=2)
        ttk.Label(f_row, text="Camera Target FPS:", font=sn, width=sw).pack(
            side=tk.LEFT
        )
        ttk.Spinbox(
            f_row,
            from_=5.0,
            to=60.0,
            textvariable=self._target_fps_var,
            width=5,
            font=sn,
            increment=5.0,
        ).pack(side=tk.LEFT, padx=4)

        t_lf = ttk.LabelFrame(sf, text="ROI & Tracking", padding=8)
        t_lf.pack(fill=tk.X, padx=4, pady=4)

        ttk.Checkbutton(
            t_lf,
            text="Enable ROI Tracking",
            variable=self._roi_var,
        ).pack(anchor=tk.W)
        ttk.Label(
            t_lf,
            textvariable=self._roi_status_var,
            style="Muted.TLabel",
        ).pack(anchor=tk.W, pady=(2, 4))

        rc_row = ttk.Frame(t_lf)
        rc_row.pack(fill=tk.X, pady=2)
        ttk.Label(rc_row, text="ROI Cache (frames):", font=sn, width=sw).pack(
            side=tk.LEFT
        )
        ttk.Spinbox(
            rc_row,
            from_=1,
            to=30,
            textvariable=self._roi_cache_var,
            width=5,
            font=sn,
        ).pack(side=tk.LEFT, padx=4)

        kp_row = ttk.Frame(t_lf)
        kp_row.pack(fill=tk.X, pady=2)
        ttk.Label(kp_row, text="Kalman Process Noise:", font=sn, width=sw).pack(
            side=tk.LEFT
        )
        self._kp_display = tk.StringVar(value=f"{self._kalman_process_var.get():.3f}")
        ttk.Scale(
            kp_row,
            from_=0.001,
            to=0.5,
            variable=self._kalman_process_var,
            command=lambda v: self._kp_display.set(f"{float(v):.3f}"),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(kp_row, textvariable=self._kp_display, font=sn, width=6).pack(
            side=tk.LEFT
        )

        km_row = ttk.Frame(t_lf)
        km_row.pack(fill=tk.X, pady=2)
        ttk.Label(km_row, text="Kalman Measure Noise:", font=sn, width=sw).pack(
            side=tk.LEFT
        )
        self._km_display = tk.StringVar(value=f"{self._kalman_measure_var.get():.3f}")
        ttk.Scale(
            km_row,
            from_=0.01,
            to=1.0,
            variable=self._kalman_measure_var,
            command=lambda v: self._km_display.set(f"{float(v):.3f}"),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(km_row, textvariable=self._km_display, font=sn, width=6).pack(
            side=tk.LEFT
        )

        cal_lf = ttk.LabelFrame(sf, text="Calibration & Physical Units", padding=8)
        cal_lf.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(
            cal_lf,
            text="Select physical scale derivation method:",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 4))

        for mode_val, mode_text in [
            ("ANATOMICAL_ANCHOR", "Anatomical Anchor (Cornea ≈ 12.0 mm)"),
            ("FIXED_PIXEL_SCALE", "Fixed Pixel Scale (Manual / External Target)"),
            ("RING_REFLECTION", "Ring Reflection (Purkinje / Placido Ring)"),
        ]:
            ttk.Radiobutton(
                cal_lf,
                text=mode_text,
                variable=self._calibration_mode_var,
                value=mode_val,
                command=lambda: self._schedule_live_settings_apply("calibration"),
            ).pack(anchor=tk.W, padx=(8, 0), pady=1)

        px_mm_row = ttk.Frame(cal_lf)
        px_mm_row.pack(fill=tk.X, pady=2)
        ttk.Label(px_mm_row, text="Fixed Scale (px/mm):", font=sn, width=sw).pack(side=tk.LEFT)
        ttk.Spinbox(
            px_mm_row,
            from_=1.0,
            to=500.0,
            increment=0.5,
            textvariable=self._fixed_scale_var,
            width=8,
            font=sn,
        ).pack(side=tk.LEFT, padx=4)

        cornea_row = ttk.Frame(cal_lf)
        cornea_row.pack(fill=tk.X, pady=2)
        ttk.Label(cornea_row, text="Assumed Cornea (mm):", font=sn, width=sw).pack(side=tk.LEFT)
        ttk.Spinbox(
            cornea_row,
            from_=8.0,
            to=15.0,
            increment=0.1,
            textvariable=self._corneal_ref_mm_var,
            width=8,
            font=sn,
        ).pack(side=tk.LEFT, padx=4)

        ring_row = ttk.Frame(cal_lf)
        ring_row.pack(fill=tk.X, pady=2)
        ttk.Label(ring_row, text="Ref Ring Dia (mm):", font=sn, width=sw).pack(side=tk.LEFT)
        ttk.Spinbox(
            ring_row,
            from_=5.0,
            to=20.0,
            increment=0.1,
            textvariable=self._ring_ref_mm_var,
            width=8,
            font=sn,
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            cal_lf,
            text="⚡ Open Calibration & WTW Wizard…",
            command=self._open_calibration_wizard,
        ).pack(anchor=tk.W, pady=(6, 2))

        fill_lf = ttk.LabelFrame(sf, text="Circle Fill Shading", padding=8)

        fill_lf.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(
            fill_lf,
            text="Fill the pupil (green) and limbus (blue) circles.\n0 = no fill, 100 = fully coloured.",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        for label_text, var, disp_attr in [
            ("Pupil (green) %:", self._pupil_fill_alpha_var, "_pupil_fill_display"),
            ("Limbus (blue) %:", self._limbus_fill_alpha_var, "_limbus_fill_display"),
        ]:
            disp_var = tk.StringVar(value="0")
            setattr(self, disp_attr, disp_var)
            row = ttk.Frame(fill_lf)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label_text, font=sn, width=sw).pack(side=tk.LEFT)
            _v = var  # capture for lambda
            _d = disp_var
            ttk.Scale(
                row,
                from_=0,
                to=100,
                variable=_v,
                command=lambda v, d=_d: (
                    d.set(str(int(float(v)))),
                    self._refresh_display(),
                ),
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            ttk.Label(row, textvariable=disp_var, font=sn, width=4).pack(side=tk.LEFT)

        a_lf = ttk.LabelFrame(sf, text="Actions", padding=8)
        a_lf.pack(fill=tk.X, padx=4, pady=4)

        ttk.Button(
            a_lf,
            text="Rebuild Inference Engine",
            command=self._rebuild_engine_ui,
        ).pack(anchor=tk.W)

        self._engine_status_var = tk.StringVar(value="Engine: not initialised")
        ttk.Label(
            a_lf,
            textvariable=self._engine_status_var,
            font=sn,
        ).pack(anchor=tk.W, pady=(4, 0))

        note = (
            "Most settings apply live. Pipeline and resolution\n"
            "changes restart the active stream automatically\n"
            "when needed. Use 'Rebuild Inference Engine'\n"
            "to refresh the fast path immediately. Press [G]\n"
            "to toggle grayscale mode at any time."
        )
        ttk.Label(
            sf,
            text=note,
            style="Tiny.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=8, pady=(8, 4))

    # ================================================================
    # Measurements Panel
    # ================================================================

    def _build_measurements_panel(self, parent: ttk.Frame) -> None:
        c = self._colors

        canvas = tk.Canvas(
            parent,
            highlightthickness=0,
            bg=c.BG_SECONDARY,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        scroll_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(scroll_window, width=e.width),
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def add_card(
            pf,
            title: str,
            style_name: str,
            row: int,
            column: int,
            columnspan: int = 1,
        ) -> ttk.Frame:
            card = ttk.Frame(pf, style="MetricCard.TFrame", padding=6)
            card.grid(
                row=row,
                column=column,
                columnspan=columnspan,
                sticky="nsew",
                padx=3,
                pady=3,
            )
            ttk.Label(
                card,
                text=title,
                style=style_name,
            ).pack(anchor=tk.W, pady=(0, 4))
            body = ttk.Frame(card, style="MetricCard.TFrame")
            body.pack(fill=tk.X)
            body.columnconfigure(1, weight=1)
            return body

        def add_summary_card(pf, title: str, column: int) -> Tuple[tk.StringVar, ttk.Label]:
            card = ttk.Frame(pf, style="MetricCard.TFrame", padding=10)
            card.grid(row=0, column=column, sticky="nsew", padx=3, pady=2)
            card.pack_propagate(False)
            ttk.Label(card, text=title, style="CardKey.TLabel").pack(anchor=tk.W)
            var = tk.StringVar(value="---")
            label = ttk.Label(
                card,
                textvariable=var,
                style="CardValue.TLabel",
                width=18,
                anchor="w",
            )
            label.pack(anchor=tk.W, pady=(4, 0))
            return var, label

        def add_row(pf, label_text, width=16):
            row = ttk.Frame(pf, style="MetricCard.TFrame")
            row.pack(fill=tk.X, pady=1)
            row.columnconfigure(1, weight=1)
            ttk.Label(
                row,
                text=label_text,
                style="CardKey.TLabel",
                width=width,
                anchor="w",
            ).grid(row=0, column=0, sticky="w")
            var = tk.StringVar(value="---")
            ttk.Label(
                row,
                textvariable=var,
                style="CardValueSmall.TLabel",
                anchor="e",
                justify=tk.RIGHT,
            ).grid(row=0, column=1, sticky="ew")
            return var

        summary_outer = ttk.Frame(scroll_frame)
        summary_outer.pack(fill=tk.X, pady=(8, 6))
        summary_outer.columnconfigure(0, weight=1)
        summary_outer.columnconfigure(1, weight=1)
        summary_outer.columnconfigure(2, weight=1)
        summary_outer.columnconfigure(3, weight=1)
        self._summary_quality_var, self._summary_quality_label = add_summary_card(
            summary_outer, "Quality", 0
        )
        self._summary_tracking_var, self._summary_tracking_label = add_summary_card(
            summary_outer, "Tracking", 1
        )
        self._summary_latency_var, self._summary_latency_label = add_summary_card(
            summary_outer, "Latency", 2
        )
        self._summary_pipeline_var, self._summary_pipeline_label = add_summary_card(
            summary_outer, "Pipeline", 3
        )

        ttk.Label(
            scroll_frame,
            textvariable=self._patient_summary_var,
            style="Muted.TLabel",
            anchor="w",
        ).pack(fill=tk.X, padx=6, pady=(0, 6))

        cards_outer = ttk.Frame(scroll_frame, style="Primary.TFrame")
        cards_outer.pack(fill=tk.BOTH, expand=True)
        cards_outer.columnconfigure(0, weight=1, uniform="measurement_cards")
        cards_outer.columnconfigure(1, weight=1, uniform="measurement_cards")

        pupil_frame = add_card(cards_outer, "PUPIL", "PupilHeader.TLabel", 0, 0)
        self._pv: Dict[str, tk.StringVar] = {}
        self._pv["center"] = add_row(pupil_frame, "Center:")
        self._pv["diameter_px"] = add_row(pupil_frame, "Diameter (px):")
        self._pv["diameter_mm"] = add_row(pupil_frame, "Diameter (mm):")
        self._pv["semi_major"] = add_row(pupil_frame, "Semi-Major (px):")
        self._pv["semi_major_mm"] = add_row(pupil_frame, "Semi-Major (mm):")
        self._pv["semi_minor"] = add_row(pupil_frame, "Semi-Minor (px):")
        self._pv["semi_minor_mm"] = add_row(pupil_frame, "Semi-Minor (mm):")
        self._pv["angle"] = add_row(pupil_frame, "Angle:")
        self._pv["fit_type"] = add_row(pupil_frame, "Fit Type:")
        self._pv["confidence"] = add_row(pupil_frame, "Confidence:")
        self._pv["quality"] = add_row(pupil_frame, "Quality:")

        limbus_frame = add_card(cards_outer, "LIMBUS", "LimbusHeader.TLabel", 0, 1)
        self._lv: Dict[str, tk.StringVar] = {}
        self._lv["center"] = add_row(limbus_frame, "Center:")
        self._lv["diameter_px"] = add_row(limbus_frame, "Diameter (px):")
        self._lv["diameter_mm"] = add_row(limbus_frame, "Diameter (mm):")
        self._lv["semi_major"] = add_row(limbus_frame, "Semi-Major (px):")
        self._lv["semi_major_mm"] = add_row(limbus_frame, "Semi-Major (mm):")
        self._lv["semi_minor"] = add_row(limbus_frame, "Semi-Minor (px):")
        self._lv["semi_minor_mm"] = add_row(limbus_frame, "Semi-Minor (mm):")
        self._lv["angle"] = add_row(limbus_frame, "Angle:")
        self._lv["fit_type"] = add_row(limbus_frame, "Fit Type:")
        self._lv["confidence"] = add_row(limbus_frame, "Confidence:")
        self._lv["quality"] = add_row(limbus_frame, "Quality:")

        offset_frame = add_card(
            cards_outer, "CORNEAL CENTRE & OFFSET", "OffsetHeader.TLabel", 1, 0
        )
        self._ov: Dict[str, tk.StringVar] = {}
        self._ov["corneal_center"] = add_row(offset_frame, "Corneal Centre:")
        self._ov["corneal_reference"] = add_row(offset_frame, "Reference:")
        self._ov["ring_center"] = add_row(offset_frame, "Ring Centre:")
        self._ov["ring_diameter_px"] = add_row(offset_frame, "Ring Dia (px):")
        self._ov["ring_diameter_mm"] = add_row(offset_frame, "Ring Dia (mm):")
        self._ov["offset_px"] = add_row(offset_frame, "Offset (px):")
        self._ov["offset_mm"] = add_row(offset_frame, "Offset (mm):")
        self._ov["offset_vec_px"] = add_row(offset_frame, "Offset dX,dY px:")
        self._ov["offset_vec_mm"] = add_row(offset_frame, "Offset dX,dY mm:")
        self._ov["offset_angle"] = add_row(offset_frame, "Offset Angle:")
        self._ov["pupil_limbus_ratio"] = add_row(offset_frame, "Pupil/Limbus:")

        calib_frame = add_card(cards_outer, "CALIBRATION", "CalibHeader.TLabel", 1, 1)
        self._cv_vars: Dict[str, tk.StringVar] = {}
        self._cv_vars["source"] = add_row(calib_frame, "Source:")
        self._cv_vars["scale_px"] = add_row(calib_frame, "px/mm:")
        self._cv_vars["scale_mm"] = add_row(calib_frame, "mm/px:")
        self._cv_vars["reference"] = add_row(calib_frame, "Reference:")

        wtw_frame = add_card(
            cards_outer, "CORNEAL DIMENSIONS (WTW)", "LimbusHeader.TLabel", 2, 0, 2
        )
        self._wtw_vars: Dict[str, tk.StringVar] = {}
        self._wtw_vars["horizontal"] = add_row(wtw_frame, "Horizontal WTW:")
        self._wtw_vars["vertical"] = add_row(wtw_frame, "Vertical WTW:")
        self._wtw_vars["mean"] = add_row(wtw_frame, "Mean WTW:")

        proc_frame = add_card(cards_outer, "PROCESSING", "ProcHeader.TLabel", 3, 0, 2)
        self._proc_time_var = add_row(proc_frame, "Proc. Time:")
        self._latency_var = add_row(proc_frame, "Latency:")
        self._latency_avg_var = add_row(proc_frame, "Latency Avg:")
        self._drop_var = add_row(proc_frame, "Dropped/Stale:")
        self._tracking_state_var = add_row(proc_frame, "Tracking:")
        self._fps_var = add_row(proc_frame, "FPS:")
        self._frame_var = add_row(proc_frame, "Frame:")
        self._image_size_var = add_row(proc_frame, "Image Size:")
        self._pipeline_var = add_row(proc_frame, "Pipeline:")

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 9 of 12 — Grayscale info in measurements
        # ══════════════════════════════════════════════════════════
        self._gray_mode_var_display = add_row(proc_frame, "Grayscale:")


    def _build_details_panel(self, parent: ttk.Frame) -> None:
        c = self._colors
        self._details_text = tk.Text(
            parent,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg=c.BG_INPUT,
            fg=c.FG_PRIMARY,
            state=tk.DISABLED,
            height=30,
            insertbackground=c.FG_PRIMARY,
            selectbackground=c.ACCENT_DIM,
            selectforeground=c.FG_PRIMARY,
            borderwidth=0,
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(
            parent,
            orient=tk.VERTICAL,
            command=self._details_text.yview,
        )
        self._details_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._details_text.pack(fill=tk.BOTH, expand=True)

    def _build_status_bar(self) -> None:
        status = ttk.Frame(self.root, style="Primary.TFrame")
        status.pack(side=tk.BOTTOM, fill=tk.X)

        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(
            status,
            textvariable=self._status_var,
            style="Status.TLabel",
        ).pack(side=tk.LEFT, padx=5)

        self._model_status_var = tk.StringVar(value="Model: Loading…")
        ttk.Label(
            status,
            textvariable=self._model_status_var,
            style="Status.TLabel",
        ).pack(side=tk.RIGHT, padx=5)

    def _install_crash_guards(self) -> None:
        self.root.report_callback_exception = self._handle_tk_exception
        threading.excepthook = self._handle_thread_exception

    def _handle_tk_exception(self, exc_type, exc_value, exc_traceback) -> None:
        self.logger.exception(
            "Tk callback error",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        self._report_runtime_issue("UI callback error recovered")

    def _handle_thread_exception(self, args: threading.ExceptHookArgs) -> None:
        self.logger.exception(
            "Worker thread error in %s",
            args.thread.name if args.thread is not None else "unknown",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        self.root.after(
            0,
            lambda: self._report_runtime_issue("Background worker recovered"),
        )

    def _report_runtime_issue(self, message: str) -> None:
        try:
            self._status_var.set(message)
        except Exception:
            pass

    def _bind_live_setting_callbacks(self) -> None:
        callbacks = (
            (self._use_optimized_var, "pipeline"),
            (self._fp16_var, "engine"),
            (self._compile_var, "engine"),
            (self._resolution_var, "resolution"),
            (self._stride_var, "stride"),
            (self._target_fps_var, "display"),
            (self._roi_var, "roi"),
            (self._roi_cache_var, "roi"),
            (self._kalman_process_var, "tracking"),
            (self._kalman_measure_var, "tracking"),
            (self._calibration_mode_var, "calibration"),
            (self._fixed_scale_var, "calibration"),
            (self._corneal_ref_mm_var, "calibration"),
            (self._ring_ref_mm_var, "calibration"),
        )
        for var, reason in callbacks:
            var.trace_add(
                "write",
                lambda *_args, _reason=reason: self._schedule_live_settings_apply(
                    _reason
                ),
            )

    def _schedule_live_settings_apply(self, reason: str) -> None:
        if self._suspend_live_settings_apply:
            return
        self._pending_live_apply_reasons.add(reason)
        if self._settings_apply_after_id is not None:
            self.root.after_cancel(self._settings_apply_after_id)
        self._settings_apply_after_id = self.root.after(
            180,
            self._apply_live_settings,
        )

    def _apply_live_settings(self) -> None:
        self._settings_apply_after_id = None
        reasons = set(self._pending_live_apply_reasons)
        self._pending_live_apply_reasons.clear()
        if not reasons:
            return
        if self._restart_in_progress:
            self._pending_live_apply_reasons.update(reasons)
            self._settings_apply_after_id = self.root.after(250, self._apply_live_settings)
            return

        self._res_display.set(str(int(self._resolution_var.get())))
        self._kp_display.set(f"{float(self._kalman_process_var.get()):.3f}")
        self._km_display.set(f"{float(self._kalman_measure_var.get()):.3f}")

        runtime_restart_reasons = {"pipeline", "engine", "resolution", "stride", "roi", "tracking"}
        restart_required = bool(reasons.intersection(runtime_restart_reasons))

        if hasattr(self.cfg, "video"):
            self.cfg.video.kalman_process_noise = float(self._kalman_process_var.get())
            self.cfg.video.kalman_measurement_noise = float(
                self._kalman_measure_var.get()
            )

        # Apply calibration settings
        if "calibration" in reasons or not restart_required:
            cal_mode = self._calibration_mode_var.get()
            manual_px = float(self._fixed_scale_var.get())
            corneal_mm = float(self._corneal_ref_mm_var.get())
            ring_mm = float(self._ring_ref_mm_var.get())
            if hasattr(self.cfg, "calibration"):
                self.cfg.calibration.mode = cal_mode
                self.cfg.calibration.manual_px_per_mm = manual_px
                self.cfg.calibration.corneal_diameter_mm = corneal_mm
                self.cfg.calibration.suction_ring_diameter_mm = ring_mm
            if self._detector is not None:
                self._detector.set_calibration_mode(
                    mode=cal_mode,
                    manual_px_per_mm=manual_px,
                    corneal_diameter_mm=corneal_mm,
                    ring_diameter_mm=ring_mm,
                )

        if self._tracker is not None and not self._video_running:
            self._tracker = EyeKalmanTracker(config=self.cfg)

        if self._opt_processor is not None and not restart_required:
            try:
                self._opt_processor.update_runtime_settings(
                    enable_auto_roi=self._roi_var.get(),
                    roi_cache_ttl=self._roi_cache_var.get(),
                    process_noise=self._kalman_process_var.get(),
                    measurement_noise=self._kalman_measure_var.get(),
                )
                self._apply_manual_roi_to_processor()
            except Exception as exc:
                self.logger.warning("Live optimized settings apply failed: %s", exc)

        if restart_required and self._video_running and self._active_source is not None:
            self._restart_active_stream(
                f"Applied {' / '.join(sorted(reasons))} settings",
                rebuild_engine=bool(
                    reasons.intersection({"pipeline", "engine", "resolution"})
                ),
            )
            return

        if (
            reasons.intersection({"pipeline", "engine", "resolution"})
            and not self._video_running
        ):
            self._invalidate_fast_engine()
            self._get_fast_engine()

        self._status_var.set(
            "Settings applied live: "
            + ", ".join(r.replace("_", " ").title() for r in sorted(reasons))
        )
        self._refresh_display()

        # When calibration changes, re-compute mm values on the
        # currently displayed result so the measurement panel updates
        # immediately (without requiring a new image load / detection).
        if "calibration" in reasons and self._current_result is not None:
            res = self._current_result
            new_cal = self._detector._calibration if self._detector is not None else None
            if new_cal is not None and new_cal.calibrated:
                res.calibration = new_cal
            elif (
                new_cal is not None
                and not new_cal.calibrated
                and getattr(res, "limbus", None) is not None
                and getattr(res.limbus, "detected", False)
                and getattr(res.limbus, "ellipse", None) is not None
            ):
                # ANATOMICAL_ANCHOR after reset: _current_best() returns
                # uncalibrated because _ema_px_per_mm is None.  Compute
                # the tautological calibration from the current result's
                # pixel geometry so the panel shows correct values.
                ep = res.limbus.ellipse
                corneal_mm = float(self._corneal_ref_mm_var.get())
                dia_px = ep.semi_major * 2.0
                if dia_px > 10:
                    res.calibration = CalibrationInfo(
                        calibrated=True,
                        px_per_mm=dia_px / corneal_mm,
                        mm_per_px=corneal_mm / dia_px,
                        source="anatomical_on_switch",
                        method="anatomical",
                        corneal_diameter_assumed_mm=corneal_mm,
                    )
            # Clear stale pre-computed mm attributes that were set by
            # _add_mm_values / evaluate_clinical_wtw during the original
            # detection.  These would otherwise be read by the display
            # code and show values computed with the OLD calibration.
            # (LimbusDetection/PupilDetection are dataclasses, so we
            # reset fields to None rather than delattr which does not
            # truly remove dataclass fields.)
            for target in (
                getattr(res, "limbus", None),
                getattr(res, "pupil", None),
            ):
                if target is None:
                    continue
                for attr in (
                    "wtw_horizontal_mm",
                    "wtw_vertical_mm",
                    "wtw_mean_mm",
                    "wtw_astigmatism_mm",
                    "is_wtw_measured",
                    "wtw_validity_status",
                    "radius_mm",
                    "center_mm",
                ):
                    if hasattr(target, attr):
                        try:
                            setattr(target, attr, None)
                        except Exception:
                            pass
            self._update_measurements(res)

    def _restart_active_stream(
        self, reason: str, rebuild_engine: bool = False
    ) -> None:
        if self._restart_in_progress:
            return
        source = self._active_source
        if source is None:
            return
        self._restart_in_progress = True
        self._status_var.set(f"{reason} - restarting stream...")
        try:
            if self._recorder.is_recording:
                self._stop_recording()
            if rebuild_engine:
                self._invalidate_fast_engine()
            self._stop_video()
        except Exception as exc:
            self.logger.exception("Safe stream restart failed during stop: %s", exc)
            self._restart_in_progress = False
            self._report_runtime_issue("Restart failed while stopping the active stream")
            return

        self.root.after(120, lambda src=source: self._complete_stream_restart(src))

    def _complete_stream_restart(self, source: Any) -> None:
        try:
            self._start_video(source)
        except Exception as exc:
            self.logger.exception("Safe stream restart failed during start: %s", exc)
            self._report_runtime_issue("Restart failed while starting the active stream")
        finally:
            self._restart_in_progress = False
            if self._pending_live_apply_reasons:
                self._schedule_live_settings_apply("restart")

    # ================================================================
    # Fast Inference Engine
    # ================================================================

    def _find_model_path(self) -> Optional[str]:
        candidates: List[str] = []
        if isinstance(self.cfg, dict):
            cfg_path = self.cfg.get("model_path")
            if cfg_path:
                candidates.append(str(cfg_path))
        if self._detector is not None:
            eng = getattr(self._detector, "ml_engine", None)
            if eng is not None:
                for attr in ("model_path", "_model_path"):
                    p = getattr(eng, attr, None)
                    if p:
                        candidates.append(str(p))
        candidates += [
            "models/best_model.pth",
            "model/best_model.pth",
            "pupil_tracking/models/best_model.pth",
        ]
        for c in candidates:
            if Path(c).is_file():
                found_path = str(Path(c).resolve())
                self.logger.info(f"Loaded model from: {found_path}")
                return found_path
        return None

    def _get_fast_engine(self) -> Optional[Any]:
        if not _FAST_PIPELINE_AVAILABLE:
            return None
        if self._fast_engine is not None:
            return self._fast_engine
        model_path = self._find_model_path()
        if model_path is None:
            self.logger.warning(
                "Cannot locate model file — optimised pipeline disabled"
            )
            return None
        try:
            self._fast_engine = FastInference(
                model_path=model_path,
                device="auto",
                input_size=self._resolution_var.get(),
                use_half=self._fp16_var.get(),
                use_compile=self._compile_var.get(),
                reflection_removal=True,
                suction_ring_removal=True,
            )
            self.logger.info(
                "FastInference ready (%s)",
                self._fast_engine.device,
            )
            self._engine_status_var.set(f"Engine: ready ({self._fast_engine.device})")
            return self._fast_engine
        except Exception as exc:
            self.logger.error("FastInference init failed: %s", exc)
            self._engine_status_var.set(f"Engine: error — {exc}")
            return None

    def _invalidate_fast_engine(self) -> None:
        if self._fast_engine is None:
            return
        try:
            import torch

            if (
                hasattr(self._fast_engine, "model")
                and self._fast_engine.model is not None
            ):
                del self._fast_engine.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._fast_engine = None
        self._engine_status_var.set("Engine: invalidated (will rebuild on next use)")

    def _rebuild_engine_ui(self) -> None:
        self._invalidate_fast_engine()
        engine = self._get_fast_engine()
        if engine is not None:
            engine.warmup()
            msg = (
                f"Engine: rebuilt ({engine.device}, "
                f"res={self._resolution_var.get()}, "
                f"fp16={self._fp16_var.get()})"
            )
            self._engine_status_var.set(msg)
            self._status_var.set(msg)
        else:
            self._engine_status_var.set("Engine: rebuild FAILED")
            self._status_var.set("Engine rebuild failed — check model path and logs")

    def _apply_performance_preset(self) -> None:
        preset = self._performance_preset_var.get()
        base_res = int(self._runtime_profile.recommended_resolution)
        base_fps = float(self._runtime_profile.recommended_target_fps)
        self._suspend_live_settings_apply = True
        try:
            if preset == "max_accuracy":
                self._resolution_var.set(
                    384 if self._runtime_profile.has_cuda else max(320, base_res)
                )
                self._target_fps_var.set(20.0 if self._runtime_profile.has_cuda else min(18.0, base_fps))
                self._stride_var.set(1)
                self._roi_var.set(True)
            elif preset == "low_latency":
                self._resolution_var.set(256 if self._runtime_profile.has_cuda else 288)
                self._target_fps_var.set(30.0 if self._runtime_profile.has_cuda else max(20.0, base_fps))
                self._stride_var.set(1)
                self._roi_var.set(True)
            else:
                self._resolution_var.set(base_res)
                self._target_fps_var.set(base_fps)
                self._stride_var.set(1)
                self._roi_var.set(True)
        finally:
            self._suspend_live_settings_apply = False
        self._res_display.set(str(self._resolution_var.get()))
        self._pending_live_apply_reasons.update({"resolution", "display", "stride", "roi"})
        self._schedule_live_settings_apply("preset")
        self._status_var.set(f"Preset: {preset.replace('_', ' ').title()} applied")

    def _get_display_interval(self) -> float:
        preset = self._performance_preset_var.get()
        preset_cap = {
            "max_accuracy": 20.0,
            "balanced": 28.0,
            "low_latency": 36.0,
        }.get(preset, _DISPLAY_FPS_CAP)
        target = float(self._target_fps_var.get() or _DISPLAY_FPS_CAP)
        cap = max(5.0, min(_DISPLAY_FPS_CAP, preset_cap, target))
        return 1.0 / cap

    def _derive_tracking_state(
        self, result: Any, stats: Optional[Dict[str, Any]] = None
    ) -> str:
        conf = float(getattr(result, "overall_confidence", 0.0) or 0.0)
        has_both = bool(getattr(result, "has_both", False))
        stats = stats or {}
        stale = int(stats.get("stale_frames", 0))
        dropped = int(stats.get("dropped_frames", 0))
        recent_latency = float(stats.get("latency_avg_ms", 0.0) or 0.0)

        if not has_both:
            return "No Detection"
        if stale > 10 or dropped > 10 or recent_latency > 250.0:
            return "Tracking Degraded"
        if conf >= 0.75:
            return "Tracking Stable"
        if conf >= 0.35:
            return "Tracking Acquiring"
        return "Tracking Degraded"

    def _set_summary_tracking_state(self, tracking_text: str) -> None:
        self._summary_tracking_var.set(tracking_text)
        tracking_color = {
            "Tracking Stable": self._colors.SURGICAL,
            "Tracking Acquiring": self._colors.CLINICAL,
            "Tracking Degraded": self._colors.RESEARCH,
            "No Detection": self._colors.INSUFFICIENT,
            "Ready": self._colors.ACCENT,
            "Waiting": self._colors.FG_SECONDARY,
        }.get(tracking_text, self._colors.FG_PRIMARY)
        self._summary_tracking_label.config(foreground=tracking_color)

    def _toggle_pause(self) -> None:
        if self._awaiting_roi or not self._video_running:
            return
        self._video_paused = not self._video_paused
        if self._video_paused:
            self._pause_btn.config(text="▶ Resume")
            self._status_var.set("Paused")
        else:
            self._pause_btn.config(text="⏸ Pause")
            self._status_var.set("Resumed")

    def _sync_roi_ring_button_styles(self) -> None:
        if hasattr(self, "_roi_btn"):
            is_active = self._roi_edit_active or self._manual_roi is not None
            self._roi_btn.configure(style="ROIActive.TButton" if is_active else "TButton")
            self._roi_btn.config(text="Clear ROI" if is_active else "ROI")
        if hasattr(self, "_ring_btn"):
            is_active = self._ring_edit_active or self._manual_ring is not None
            self._ring_btn.configure(style="RingActive.TButton" if is_active else "TButton")
            self._ring_btn.config(text="Clear Ring" if is_active else "Ring")

    def _toggle_roi(self) -> None:
        """Single toggle: if ROI is active/editing → clear it; otherwise begin selection."""
        if self._roi_edit_active or self._manual_roi is not None:
            self._clear_manual_roi()
        else:
            self._begin_roi_selection()

    def _toggle_ring(self) -> None:
        """Single toggle: if ring is active/editing → clear it; otherwise begin selection."""
        if self._ring_edit_active or self._manual_ring is not None:
            self._clear_manual_ring()
        else:
            self._begin_ring_selection()

    def _begin_roi_selection(self) -> None:
        if self._current_image is None:
            self._status_var.set("Start the camera, then drag a circular ROI on the image")
            return
        if self._ring_edit_active:
            self._cancel_ring_selection()
        self._roi_edit_active = True
        self._roi_drag_mode = None
        self._roi_drag_offset = (0.0, 0.0)
        active_roi = self._active_manual_roi()
        self._roi_original_before_edit = (
            dict(active_roi) if active_roi is not None else None
        )
        if active_roi is not None:
            self._roi_preview = dict(active_roi)
        else:
            h, w = self._current_image.shape[:2]
            radius = 939.0 / 2.0
            self._roi_preview = {
                "center_x": w / 2.0,
                "center_y": h / 2.0,
                "radius": radius,
                "frame_width": float(w),
                "frame_height": float(h),
            }
        self._canvas.configure(cursor="tcross")
        self._sync_roi_ring_button_styles()
        self._roi_status_var.set("Manual ROI: Editing")
        self._status_var.set(
            "ROI edit mode: drag inside to move, drag rim to resize, Enter to apply, Esc to cancel"
        )
        self._refresh_display()

    def _begin_ring_selection(self) -> None:
        if self._current_image is None:
            self._status_var.set("Start the camera, then drag the docked red ring on the image")
            return
        if self._roi_edit_active:
            self._cancel_roi_selection()
        self._ring_edit_active = True
        self._ring_drag_mode = None
        self._ring_drag_offset = (0.0, 0.0)
        active_ring = self._active_manual_ring()
        self._ring_original_before_edit = (
            dict(active_ring) if active_ring is not None else None
        )
        if active_ring is not None:
            self._ring_preview = dict(active_ring)
        else:
            self._ring_preview = self._suggest_manual_ring_preview()
        self._canvas.configure(cursor="tcross")
        self._sync_roi_ring_button_styles()
        self._ring_status_var.set("Manual Ring: Editing")
        self._status_var.set(
            "Ring edit mode: use only on docked frames, drag circle to match red dots, Enter to lock"
        )
        self._refresh_display()

    def _clear_manual_roi(self) -> None:
        self._manual_roi = None
        self._roi_preview = None
        self._roi_drag_mode = None
        self._roi_drag_offset = (0.0, 0.0)
        self._roi_edit_active = False
        self._roi_original_before_edit = None
        self._canvas.configure(cursor="crosshair")
        self._sync_roi_ring_button_styles()
        self._roi_status_var.set("Manual ROI: Off")
        previous_suspend = self._suspend_live_settings_apply
        self._suspend_live_settings_apply = True
        try:
            self._roi_var.set(False)
        finally:
            self._suspend_live_settings_apply = previous_suspend
        if self._video_running:
            self._awaiting_roi = True
            self._status_var.set("Video waiting — set an ROI to resume detection")
        if self._current_result is not None:
            self._current_result = None
        if self._opt_processor is not None:
            self._opt_processor.clear_manual_roi()
            self._opt_processor.update_runtime_settings(enable_auto_roi=False)
        self._refresh_display()

    def _clear_manual_ring(self) -> None:
        self._manual_ring = None
        self._ring_preview = None
        self._ring_drag_mode = None
        self._ring_drag_offset = (0.0, 0.0)
        self._ring_edit_active = False
        self._ring_original_before_edit = None
        self._canvas.configure(cursor="crosshair")
        self._sync_roi_ring_button_styles()
        self._ring_status_var.set("Manual Ring: Off")
        if self._opt_processor is not None:
            self._opt_processor.clear_manual_ring()
        if self._current_result is not None:
            self._apply_manual_ring_policy(self._current_result)
            self._update_measurements(self._current_result)
            self._current_result.ring_status = "ring_absent"
            self._current_result.ring_center = None
            self._current_result.ring_radius = None
        self._refresh_display()

    def _cancel_roi_selection(self, event: Any = None) -> None:
        # Abandon the in-progress edit and restore whatever ROI existed before
        # it (may be None). If detection is still gated on the ROI, keep the
        # frame frozen and re-prompt instead of starting.
        self._manual_roi = (
            dict(self._roi_original_before_edit)
            if self._roi_original_before_edit is not None
            else None
        )
        self._roi_preview = None
        self._roi_drag_mode = None
        self._roi_drag_offset = (0.0, 0.0)
        self._roi_edit_active = False
        self._roi_original_before_edit = None
        self._canvas.configure(cursor="crosshair")
        self._sync_roi_ring_button_styles()
        if self._manual_roi is not None:
            self._roi_status_var.set(
                f"Manual ROI: On ({int(round(self._manual_roi['radius']))} px)"
            )
        else:
            self._roi_status_var.set("Manual ROI: Off")
        self._apply_manual_roi_to_processor()
        if self._awaiting_roi:
            self._status_var.set("ROI required — set an ROI to start detection")
        else:
            self._status_var.set("ROI selection cancelled")
        self._refresh_display()

    def _cancel_ring_selection(self, event: Any = None) -> None:
        # Abandon the in-progress ring edit and restore the prior ring (or None).
        self._manual_ring = (
            dict(self._ring_original_before_edit)
            if self._ring_original_before_edit is not None
            else None
        )
        self._ring_preview = None
        self._ring_drag_mode = None
        self._ring_drag_offset = (0.0, 0.0)
        self._ring_edit_active = False
        self._ring_original_before_edit = None
        self._canvas.configure(cursor="crosshair")
        self._sync_roi_ring_button_styles()
        if self._manual_ring is not None:
            self._ring_status_var.set(
                f"Manual Ring: Locked ({int(round(self._manual_ring['radius'] * 2.0))} px)"
            )
        else:
            self._ring_status_var.set("Manual Ring: Off")
        self._apply_manual_ring_to_processor()
        if self._current_result is not None:
            self._apply_manual_ring_policy(self._current_result)
            self._update_measurements(self._current_result)
        self._status_var.set("Ring selection cancelled")
        self._refresh_display()

    def _on_canvas_press(self, event: Any) -> None:
        if getattr(self, "_ruler_calibration_active", False):
            self._handle_ruler_canvas_click(event)
            return
        if self._ring_edit_active:
            self._handle_ring_canvas_press(event)
            return
        if not self._roi_edit_active:
            return
        point = self._canvas_to_image_point(event.x, event.y)

        if point is None or self._current_image is None:
            return
        if self._roi_preview is None:
            h, w = self._current_image.shape[:2]
            self._roi_preview = {
                "center_x": point[0],
                "center_y": point[1],
                "radius": 939.0 / 2.0,
                "frame_width": float(w),
                "frame_height": float(h),
            }

        cx = self._roi_preview["center_x"]
        cy = self._roi_preview["center_y"]
        radius = self._roi_preview["radius"]
        distance = math.hypot(point[0] - cx, point[1] - cy)
        rim_threshold = max(10.0, radius * 0.18)

        if abs(distance - radius) <= rim_threshold:
            self._roi_drag_mode = "resize"
        elif distance < radius:
            self._roi_drag_mode = "move"
            self._roi_drag_offset = (point[0] - cx, point[1] - cy)
        else:
            self._roi_drag_mode = "resize"
            self._roi_preview["center_x"] = point[0]
            self._roi_preview["center_y"] = point[1]
            self._roi_preview["radius"] = max(12.0, radius * 0.5)
        self._refresh_display()

    def _on_canvas_drag(self, event: Any) -> None:
        if self._ring_edit_active:
            self._handle_ring_canvas_drag(event)
            return
        if (
            not self._roi_edit_active
            or self._roi_drag_mode is None
            or self._roi_preview is None
        ):
            return
        point = self._canvas_to_image_point(event.x, event.y)
        if point is None or self._current_image is None:
            return
        h, w = self._current_image.shape[:2]
        if self._roi_drag_mode == "move":
            radius = self._roi_preview["radius"]
            cx = point[0] - self._roi_drag_offset[0]
            cy = point[1] - self._roi_drag_offset[1]
            self._roi_preview["center_x"] = float(np.clip(cx, radius, w - radius))
            self._roi_preview["center_y"] = float(np.clip(cy, radius, h - radius))
        else:
            cx = self._roi_preview["center_x"]
            cy = self._roi_preview["center_y"]
            max_radius = min(cx, cy, w - cx, h - cy)
            radius = max(8.0, math.hypot(point[0] - cx, point[1] - cy))
            self._roi_preview["radius"] = float(max(8.0, min(radius, max_radius)))
        self._refresh_display()

    def _on_canvas_release(self, event: Any) -> None:
        if self._ring_edit_active:
            self._handle_ring_canvas_release(event)
            return
        if not self._roi_edit_active:
            return
        self._roi_drag_mode = None
        self._roi_drag_offset = (0.0, 0.0)
        self._canvas.configure(cursor="fleur")
        if self._roi_preview is not None:
            self._status_var.set(
                "ROI ready. Drag to refine, press Enter to apply, or Esc to cancel"
            )
        self._refresh_display()

    def _confirm_active_selection(self, event: Any = None) -> None:
        if self._ring_edit_active:
            self._confirm_ring_selection(event)
            return
        self._confirm_roi_selection(event)

    def _confirm_roi_selection(self, event: Any = None) -> None:
        if not self._roi_edit_active:
            return
        preview = self._roi_preview
        if preview is None or preview["radius"] < 8.0:
            self._status_var.set("ROI selection cancelled")
            self._cancel_roi_selection()
            return
        self._manual_roi = dict(preview)
        self._roi_preview = None
        self._roi_drag_mode = None
        self._roi_drag_offset = (0.0, 0.0)
        self._roi_edit_active = False
        self._roi_original_before_edit = None
        self._canvas.configure(cursor="crosshair")
        self._sync_roi_ring_button_styles()
        self._roi_status_var.set(
            f"Manual ROI: On ({int(round(self._manual_roi['radius']))} px)"
        )
        self._status_var.set("Manual ROI applied to live detection")
        self._apply_manual_roi_to_processor()
        self._refresh_display()
        # If detection was gated on the ROI (freshly loaded video/camera),
        # confirming the ROI is what actually starts the processing thread.
        if self._awaiting_roi:
            self._awaiting_roi = False
            if self._pending_detection_start is not None:
                start = self._pending_detection_start
                self._pending_detection_start = None
                self._pause_btn.config(state=tk.NORMAL, text="⏸ Pause")
                start()

    def _confirm_ring_selection(self, event: Any = None) -> None:
        if not self._ring_edit_active:
            return
        preview = self._ring_preview
        if preview is None or preview["radius"] < 8.0:
            self._status_var.set("Ring selection cancelled")
            self._cancel_ring_selection()
            return
        self._manual_ring = dict(preview)
        self._ring_preview = None
        self._ring_drag_mode = None
        self._ring_drag_offset = (0.0, 0.0)
        self._ring_edit_active = False
        self._ring_original_before_edit = None
        self._canvas.configure(cursor="crosshair")
        self._sync_roi_ring_button_styles()
        self._ring_status_var.set(
            f"Manual Ring: Locked ({int(round(self._manual_ring['radius'] * 2.0))} px)"
        )
        self._status_var.set("Manual docked ring locked and applied to offset calculation")
        self._show_ring_center.set(True)
        self._apply_manual_ring_to_processor()
        if self._current_result is not None:
            self._apply_manual_ring_policy(self._current_result)
            self._update_measurements(self._current_result)
        self._refresh_display()

    def _nudge_roi(self, dx: int, dy: int, event: Any = None) -> None:
        if not self._roi_edit_active or self._roi_preview is None or self._current_image is None:
            return
        step = 10.0 if (event is not None and (event.state & 0x0001)) else 2.0
        h, w = self._current_image.shape[:2]
        radius = self._roi_preview["radius"]
        self._roi_preview["center_x"] = float(
            np.clip(self._roi_preview["center_x"] + dx * step, radius, w - radius)
        )
        self._roi_preview["center_y"] = float(
            np.clip(self._roi_preview["center_y"] + dy * step, radius, h - radius)
        )
        self._status_var.set(
            "ROI edit mode: arrows nudge, drag move/resize, Enter apply, Esc cancel"
        )
        self._refresh_display()

    def _on_canvas_wheel(self, event: Any) -> None:
        if self._ring_edit_active:
            self._handle_ring_canvas_wheel(event)
            return
        if not self._roi_edit_active or self._roi_preview is None or self._current_image is None:
            return
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        step = 10.0 if (getattr(event, "state", 0) & 0x0001) else 4.0
        direction = 1.0 if delta > 0 else -1.0
        h, w = self._current_image.shape[:2]
        cx = self._roi_preview["center_x"]
        cy = self._roi_preview["center_y"]
        max_radius = min(cx, cy, w - cx, h - cy)
        radius = self._roi_preview["radius"] + direction * step
        self._roi_preview["radius"] = float(max(8.0, min(radius, max_radius)))
        self._status_var.set(
            "ROI edit mode: wheel resizes, arrows nudge, Enter apply, Esc cancel"
        )
        self._refresh_display()

    def _cancel_active_selection(self, event: Any = None) -> None:
        if getattr(self, "_ruler_calibration_active", False):
            self._cancel_ruler_calibration(event)
            return
        if self._ring_edit_active:
            self._cancel_ring_selection(event)
            return
        self._cancel_roi_selection(event)

    def _cancel_ruler_calibration(self, event: Any = None) -> None:
        self._ruler_calibration_active = False
        self._ruler_points.clear()
        self._canvas.configure(cursor="crosshair")
        self._status_var.set("Ruler calibration cancelled")
        self._refresh_display()

    def _start_ruler_calibration(self, known_dist_mm: float = 10.0) -> None:
        self._ruler_calibration_active = True
        self._ruler_points.clear()
        self._ruler_known_dist_mm_var.set(known_dist_mm)
        self._canvas.configure(cursor="tcross")
        self._status_var.set(
            f"Ruler Calibration: Click Point 1 on image (known span: {known_dist_mm:.1f} mm)"
        )
        self._refresh_display()

    def _handle_ruler_canvas_click(self, event: Any) -> None:
        pt = self._canvas_to_image_point(event.x, event.y)
        if pt is None:
            return
        self._ruler_points.append(pt)
        if len(self._ruler_points) == 1:
            self._status_var.set(
                f"Ruler point 1 set at ({pt[0]:.1f}, {pt[1]:.1f}) px. Now click Point 2..."
            )
            self._refresh_display()
        elif len(self._ruler_points) >= 2:
            p1 = self._ruler_points[0]
            p2 = self._ruler_points[1]
            dist_px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            known_mm = float(self._ruler_known_dist_mm_var.get())
            if known_mm <= 0:
                known_mm = 10.0
            if dist_px < 2.0:
                self._status_var.set("Selected points too close together (<2px). Click 2 distinct points.")
                self._ruler_points.clear()
                self._refresh_display()
                return

            from pupil_tracking.calibration.spatial_calibration import calculate_ruler_scale
            px_per_mm, mm_per_px = calculate_ruler_scale(p1, p2, known_mm)
            self._fixed_scale_var.set(round(px_per_mm, 2))
            self._calibration_mode_var.set("FIXED_PIXEL_SCALE")
            self._schedule_live_settings_apply("calibration")
            self._ruler_calibration_active = False
            self._canvas.configure(cursor="crosshair")
            msg = f"Ruler Calibrated: {px_per_mm:.2f} px/mm ({dist_px:.1f} px = {known_mm:.1f} mm)"
            self._status_var.set(msg)
            self._refresh_display()
            messagebox.showinfo(
                "Ruler Calibration Applied",
                f"Calibration Successful!\n\nScale: {px_per_mm:.2f} px/mm ({mm_per_px:.4f} mm/px)\n"
                f"Measured Span: {dist_px:.1f} px = {known_mm:.1f} mm\n"
                f"Mode: Fixed Pixel Scale (Active)",
            )

    def _open_calibration_wizard(self) -> None:
        """Open a modern, comprehensive Calibration & Corneal Sizing Wizard dialog."""
        wizard = tk.Toplevel(self.root)
        wizard.title("Scale & Corneal Sizing (WTW) Calibration Wizard")
        wizard.geometry("540x580")
        wizard.resizable(False, False)
        wizard.transient(self.root)
        wizard.grab_set()

        c = self._colors
        wizard.configure(bg=c.BG_PRIMARY)

        # Header
        header = ttk.Frame(wizard, padding=16)
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text="Optical Scale & WTW Calibration",
            font=("Segoe UI", 14, "bold"),
            foreground=c.FG_PRIMARY,
        ).pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Establish an accurate pixel-to-millimeter ratio for genuine patient-specific\nWhite-to-White corneal diameter measurements.",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(4, 0))

        # Main cards container
        container = ttk.Frame(wizard, padding=(16, 0, 16, 16))
        container.pack(fill=tk.BOTH, expand=True)

        # Method 1: Interactive 2-Point Ruler Card
        m1 = ttk.LabelFrame(container, text="Method 1: Interactive 2-Point Ruler Tool", padding=10)
        m1.pack(fill=tk.X, pady=4)
        ttk.Label(
            m1,
            text="Click two points on an image with a known physical distance\n(e.g., surgical marker, ruler, or target grid).",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        ruler_row = ttk.Frame(m1)
        ruler_row.pack(fill=tk.X, pady=2)
        ttk.Label(ruler_row, text="Known Distance (mm):", width=20).pack(side=tk.LEFT)
        known_mm_ent = ttk.Spinbox(
            ruler_row,
            from_=0.5,
            to=100.0,
            increment=0.5,
            textvariable=self._ruler_known_dist_mm_var,
            width=8,
        )
        known_mm_ent.pack(side=tk.LEFT, padx=4)

        def _start_ruler():
            wizard.destroy()
            self._start_ruler_calibration(self._ruler_known_dist_mm_var.get())

        ttk.Button(
            ruler_row,
            text="🎯 Click 2 Points on Canvas",
            command=_start_ruler,
        ).pack(side=tk.LEFT, padx=8)

        # Method 2: Direct Optical Scale Input
        m2 = ttk.LabelFrame(container, text="Method 2: Direct Optical / Microscope Scale", padding=10)
        m2.pack(fill=tk.X, pady=4)
        ttk.Label(
            m2,
            text="Directly enter known magnification / scale factor in pixels per millimeter.",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        scale_row = ttk.Frame(m2)
        scale_row.pack(fill=tk.X, pady=2)
        ttk.Label(scale_row, text="Fixed Scale (px/mm):", width=20).pack(side=tk.LEFT)
        scale_ent = ttk.Spinbox(
            scale_row,
            from_=1.0,
            to=500.0,
            increment=0.5,
            textvariable=self._fixed_scale_var,
            width=8,
        )
        scale_ent.pack(side=tk.LEFT, padx=4)

        def _apply_fixed():
            self._calibration_mode_var.set("FIXED_PIXEL_SCALE")
            self._schedule_live_settings_apply("calibration")
            wizard.destroy()
            messagebox.showinfo("Scale Set", f"Fixed optical scale set to {self._fixed_scale_var.get():.2f} px/mm.")

        ttk.Button(scale_row, text="Apply Scale", command=_apply_fixed).pack(side=tk.LEFT, padx=8)

        # Method 3: Ring Fiducial Reflection
        m3 = ttk.LabelFrame(container, text="Method 3: Suction / Placido Ring Fiducial", padding=10)
        m3.pack(fill=tk.X, pady=4)
        ttk.Label(
            m3,
            text="Calibrate dynamically from a known surgical suction ring or Placido ring.",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        ring_row = ttk.Frame(m3)
        ring_row.pack(fill=tk.X, pady=2)
        ttk.Label(ring_row, text="Ring Outer Dia (mm):", width=20).pack(side=tk.LEFT)
        ring_ent = ttk.Spinbox(
            ring_row,
            from_=5.0,
            to=25.0,
            increment=0.1,
            textvariable=self._ring_ref_mm_var,
            width=8,
        )
        ring_ent.pack(side=tk.LEFT, padx=4)

        def _apply_ring():
            self._calibration_mode_var.set("RING_REFLECTION")
            self._schedule_live_settings_apply("calibration")
            wizard.destroy()
            messagebox.showinfo("Ring Mode Set", f"Ring reflection calibration activated ({self._ring_ref_mm_var.get():.1f} mm standard).")

        ttk.Button(ring_row, text="Apply Ring Mode", command=_apply_ring).pack(side=tk.LEFT, padx=8)

        # Method 4: Anatomical Baseline Anchor
        m4 = ttk.LabelFrame(container, text="Method 4: Anatomical Baseline Anchor (12.0 mm)", padding=10)
        m4.pack(fill=tk.X, pady=4)
        ttk.Label(
            m4,
            text="Assumes horizontal corneal diameter is 12.0 mm (not patient-specific).",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        def _apply_anatomical():
            self._calibration_mode_var.set("ANATOMICAL_ANCHOR")
            self._schedule_live_settings_apply("calibration")
            wizard.destroy()
            messagebox.showinfo("Baseline Set", "Calibration set to 12.0 mm Anatomical Anchor.")

        ttk.Button(m4, text="Reset to 12.0 mm Baseline Anchor", command=_apply_anatomical).pack(anchor=tk.W)

        # Footer close button
        btn_bar = ttk.Frame(wizard, padding=(16, 0, 16, 16))
        btn_bar.pack(fill=tk.X)
        ttk.Button(btn_bar, text="Close", command=wizard.destroy).pack(side=tk.RIGHT)


    def _canvas_to_image_point(
        self, canvas_x: float, canvas_y: float
    ) -> Optional[Tuple[float, float]]:
        if self._current_image is None:
            return None
        ox, oy = self._display_origin
        dw, dh = self._display_size
        if dw <= 0 or dh <= 0:
            return None
        if not (ox <= canvas_x <= ox + dw and oy <= canvas_y <= oy + dh):
            return None
        x = (canvas_x - ox) / max(self._display_scale, 1e-6)
        y = (canvas_y - oy) / max(self._display_scale, 1e-6)
        h, w = self._current_image.shape[:2]
        return (float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1)))

    def _active_manual_roi(self) -> Optional[Dict[str, float]]:
        if self._manual_roi is None or self._current_image is None:
            return None
        h, w = self._current_image.shape[:2]
        if (
            int(round(self._manual_roi.get("frame_width", w))) != w
            or int(round(self._manual_roi.get("frame_height", h))) != h
        ):
            return None
        return self._manual_roi

    def _active_manual_ring(self) -> Optional[Dict[str, float]]:
        if self._manual_ring is None or self._current_image is None:
            return None
        h, w = self._current_image.shape[:2]
        if (
            int(round(self._manual_ring.get("frame_width", w))) != w
            or int(round(self._manual_ring.get("frame_height", h))) != h
        ):
            return None
        return self._manual_ring

    def _apply_manual_ring_policy(self, result: Any) -> Any:
        """Only allow ring data when a manual ring has been confirmed."""
        if result is None:
            return None
        ring = self._active_manual_ring()
        if ring is None:
            setattr(result, "ring_status", "ring_absent")
            setattr(result, "ring_center", None)
            setattr(result, "ring_radius", None)
            setattr(result, "ring_contour", None)
            setattr(result, "ring_dot_count", 0)
            setattr(result, "ring_confidence", 0.0)
            if "ring" in str(getattr(result, "corneal_reference_source", "")):
                setattr(result, "corneal_reference_source", "limbus")
            return result

        center = (float(ring["center_x"]), float(ring["center_y"]))
        radius = float(ring["radius"])
        setattr(result, "ring_status", "ring_present")
        setattr(result, "ring_center", center)
        setattr(result, "ring_radius", radius)
        setattr(result, "ring_contour", None)
        setattr(result, "ring_dot_count", int(round(ring.get("dot_count", 12))))
        setattr(result, "ring_confidence", 1.0)
        setattr(result, "corneal_reference_source", "manual_ring")
        return result

    def _apply_manual_roi_to_processor(self) -> None:
        roi = self._active_manual_roi()
        if self._opt_processor is None:
            return
        if roi is None:
            self._opt_processor.clear_manual_roi()
            return
        self._opt_processor.set_manual_roi(
            center_x=roi["center_x"],
            center_y=roi["center_y"],
            radius=roi["radius"],
            frame_shape=(
                self._current_image.shape if self._current_image is not None else None
            ),
        )

    def _apply_manual_ring_to_processor(self) -> None:
        ring = self._active_manual_ring()
        if self._opt_processor is None:
            return
        if ring is None:
            self._opt_processor.clear_manual_ring()
            return
        self._opt_processor.set_manual_ring(
            center_x=ring["center_x"],
            center_y=ring["center_y"],
            radius=ring["radius"],
            dot_count=int(round(ring.get("dot_count", 12))),
            frame_shape=(
                self._current_image.shape if self._current_image is not None else None
            ),
        )

    def _suggest_manual_ring_preview(self) -> Dict[str, float]:
        if self._current_result is not None:
            ring_status = getattr(self._current_result, "ring_status", "unknown")
            ring_center = getattr(self._current_result, "ring_center", None)
            ring_radius = getattr(self._current_result, "ring_radius", None)
            if (
                ring_status == "ring_present"
                and ring_center is not None
                and ring_radius is not None
            ):
                h, w = self._current_image.shape[:2]
                return {
                    "center_x": float(ring_center[0]),
                    "center_y": float(ring_center[1]),
                    "radius": float(ring_radius),
                    "frame_width": float(w),
                    "frame_height": float(h),
                    "dot_count": float(getattr(self._current_result, "ring_dot_count", 12)),
                }
        if self._opt_processor is not None and self._current_image is not None:
            suggestion = self._opt_processor.get_manual_ring(self._current_image.shape)
            if suggestion is not None:
                h, w = self._current_image.shape[:2]
                return {
                    "center_x": float(suggestion["center_x"]),
                    "center_y": float(suggestion["center_y"]),
                    "radius": float(suggestion["radius"]),
                    "frame_width": float(w),
                    "frame_height": float(h),
                    "dot_count": float(suggestion.get("dot_count", 12)),
                }
        h, w = self._current_image.shape[:2]
        roi = self._active_manual_roi()
        if roi is not None:
            center_x = float(roi["center_x"])
            center_y = float(roi["center_y"])
            radius = max(12.0, float(roi["radius"]) * 0.92)
        else:
            center_x = w / 2.0
            center_y = h / 2.0
            if self._current_result is not None and hasattr(self._current_result, "calibration") and self._current_result.calibration.calibrated:
                px_per_mm = self._current_result.calibration.px_per_mm
                radius = max(12.0, (14.1 * px_per_mm) / 2.0)
            else:
                radius = max(12.0, min(w, h) * 0.42)
        return {
            "center_x": center_x,
            "center_y": center_y,
            "radius": radius,
            "frame_width": float(w),
            "frame_height": float(h),
            "dot_count": 12.0,
        }

    def _handle_ring_canvas_press(self, event: Any) -> None:
        point = self._canvas_to_image_point(event.x, event.y)
        if point is None or self._current_image is None:
            return
        if self._ring_preview is None:
            self._ring_preview = self._suggest_manual_ring_preview()
        cx = self._ring_preview["center_x"]
        cy = self._ring_preview["center_y"]
        radius = self._ring_preview["radius"]
        distance = math.hypot(point[0] - cx, point[1] - cy)
        rim_threshold = max(10.0, radius * 0.15)
        if abs(distance - radius) <= rim_threshold:
            self._ring_drag_mode = "resize"
        elif distance < radius:
            self._ring_drag_mode = "move"
            self._ring_drag_offset = (point[0] - cx, point[1] - cy)
        else:
            self._ring_drag_mode = "resize"
            self._ring_preview["center_x"] = point[0]
            self._ring_preview["center_y"] = point[1]
            self._ring_preview["radius"] = max(12.0, radius * 0.5)
        self._refresh_display()

    def _handle_ring_canvas_drag(self, event: Any) -> None:
        if (
            not self._ring_edit_active
            or self._ring_drag_mode is None
            or self._ring_preview is None
            or self._current_image is None
        ):
            return
        point = self._canvas_to_image_point(event.x, event.y)
        if point is None:
            return
        h, w = self._current_image.shape[:2]
        if self._ring_drag_mode == "move":
            radius = self._ring_preview["radius"]
            cx = point[0] - self._ring_drag_offset[0]
            cy = point[1] - self._ring_drag_offset[1]
            self._ring_preview["center_x"] = float(np.clip(cx, radius, w - radius))
            self._ring_preview["center_y"] = float(np.clip(cy, radius, h - radius))
        else:
            cx = self._ring_preview["center_x"]
            cy = self._ring_preview["center_y"]
            max_radius = min(cx, cy, w - cx, h - cy)
            radius = max(8.0, math.hypot(point[0] - cx, point[1] - cy))
            self._ring_preview["radius"] = float(max(8.0, min(radius, max_radius)))
        self._refresh_display()

    def _handle_ring_canvas_release(self, event: Any) -> None:
        if not self._ring_edit_active:
            return
        self._ring_drag_mode = None
        self._ring_drag_offset = (0.0, 0.0)
        self._canvas.configure(cursor="fleur")
        if self._ring_preview is not None:
            self._status_var.set(
                "Ring ready. Match the red-dot circle, then press Enter to lock or Esc to cancel"
            )
        self._refresh_display()

    def _handle_ring_canvas_wheel(self, event: Any) -> None:
        if (
            not self._ring_edit_active
            or self._ring_preview is None
            or self._current_image is None
        ):
            return
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        step = 10.0 if (getattr(event, "state", 0) & 0x0001) else 4.0
        direction = 1.0 if delta > 0 else -1.0
        h, w = self._current_image.shape[:2]
        cx = self._ring_preview["center_x"]
        cy = self._ring_preview["center_y"]
        max_radius = min(cx, cy, w - cx, h - cy)
        radius = self._ring_preview["radius"] + direction * step
        self._ring_preview["radius"] = float(max(8.0, min(radius, max_radius)))
        self._status_var.set(
            "Ring edit mode: wheel resizes, drag move/resize, Enter lock, Esc cancel"
        )
        self._refresh_display()

    def _get_manual_roi_crop(
        self, frame: np.ndarray
    ) -> Optional[Tuple[np.ndarray, float, float]]:
        roi = self._active_manual_roi()
        if roi is None:
            return None
        h, w = frame.shape[:2]
        cx = float(np.clip(roi["center_x"], 0, w - 1))
        cy = float(np.clip(roi["center_y"], 0, h - 1))
        radius = max(1.0, float(roi["radius"]))
        x0 = max(0, int(np.floor(cx - radius)))
        y0 = max(0, int(np.floor(cy - radius)))
        x1 = min(w, int(np.ceil(cx + radius)))
        y1 = min(h, int(np.ceil(cy + radius)))
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        return crop, float(x0), float(y0)

    # ================================================================
    # Image Operations
    # ================================================================

    def _open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image files", "*.jpeg *.jpg *.png *.bmp *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._stop_video()
        self._load_and_detect_image(path)

    def _open_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder:
            return
        self._stop_video()
        images = (
            sorted(Path(folder).glob("*.jpeg"))
            + sorted(Path(folder).glob("*.jpg"))
            + sorted(Path(folder).glob("*.png"))
        )
        if not images:
            messagebox.showwarning("No Images", f"No images found in {folder}")
            return
        self._results_history.clear()
        for i, img_path in enumerate(images):
            self._status_var.set(f"Processing {i + 1}/{len(images)}: {img_path.name}")
            self.root.update()
            self._load_and_detect_image(str(img_path))
        self._status_var.set(f"Processed {len(images)} images")

    # ================================================================
    # Synchronous calibration sync — runs BEFORE every detection
    # to guarantee the detector uses the current GUI settings
    # (eliminates race with the 180 ms debounce in
    # _schedule_live_settings_apply).
    # ================================================================

    def _sync_calibration_to_detector(self) -> None:
        """Push current GUI calibration settings into the detector immediately.

        Called synchronously before every detection so that the detector
        always reflects the user's latest calibration choice, even when
        the 180 ms debounced ``_apply_live_settings`` has not fired yet.
        """
        if self._detector is None:
            return
        cal_mode = self._calibration_mode_var.get()
        manual_px = float(self._fixed_scale_var.get())
        corneal_mm = float(self._corneal_ref_mm_var.get())
        ring_mm = float(self._ring_ref_mm_var.get())
        if hasattr(self.cfg, "calibration"):
            self.cfg.calibration.mode = cal_mode
            self.cfg.calibration.manual_px_per_mm = manual_px
            self.cfg.calibration.corneal_diameter_mm = corneal_mm
            self.cfg.calibration.suction_ring_diameter_mm = ring_mm
        self._detector.set_calibration_mode(
            mode=cal_mode,
            manual_px_per_mm=manual_px,
            corneal_diameter_mm=corneal_mm,
            ring_diameter_mm=ring_mm,
        )

    def _load_and_detect_image(self, path: str) -> None:
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Error", f"Cannot read image: {path}")
            return
        self._current_image = image
        self._frame_count += 1
        if self._detector is None:
            self._status_var.set("Detector not ready — showing raw image")
            self._current_result = None
            self._refresh_display()
            return
        self._sync_calibration_to_detector()
        self._status_var.set(f"Detecting: {Path(path).name}…")
        self.root.update()
        result = self._detector.detect(
            image, frame_number=self._frame_count, source=path
        )
        result = self._apply_manual_ring_policy(result)
        self._current_result = result
        self._results_history.append(result.to_dict())
        self._update_measurements(result)
        self._refresh_display()
        self._status_var.set(
            f"{Path(path).name} — {result.overall_quality.value} "
            f"({result.overall_confidence:.3f}) — "
            f"{result.metadata.processing_time_ms:.0f} ms"
        )

    # ================================================================
    # Video Operations
    # ================================================================

    def _open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._stop_video()
        self._start_video(path)

    def _arm_detection_gate(
        self, first_frame: np.ndarray, start_callable: Any, label: str
    ) -> None:
        """Show the first frame and hold detection until an ROI is confirmed.

        Detection does not begin until the user confirms an ROI; at that point
        ``start_callable`` is invoked to spin up the processing thread.
        """
        self._current_image = first_frame
        self._awaiting_roi = True
        self._pending_detection_start = start_callable
        self._pause_btn.config(state=tk.DISABLED, text="⏸ Pause")
        self._progress_label_var.set("Waiting for ROI")
        self._refresh_display()
        # Auto-enter ROI drawing so the circle is immediately draggable.
        self._begin_roi_selection()
        self._status_var.set(
            f"{label} loaded — drag the ROI and press Enter to start detection"
        )

    def _start_video(self, source: Any) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open: {source}")
            return
        self._active_source = source
        self._video_cap = cap
        self._video_running = True
        self._video_paused = False
        self._using_optimized_camera = False
        self._camera_mode = isinstance(source, int)
        if self._tracker is not None:
            self._tracker.reset()
        self._results_history.clear()
        self._frame_count = 0
        self._video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._video_start_time = time.monotonic()
        self._last_display_update = 0.0
        if self._video_total_frames > 0:
            self._progress_bar.config(
                mode="determinate",
                maximum=self._video_total_frames,
            )
        else:
            self._progress_bar.config(mode="indeterminate", maximum=100)
        self._progress_bar["value"] = 0
        self._progress_label_var.set("Starting…")
        self._eta_label_var.set("")
        src_name = "Camera" if isinstance(source, int) else Path(str(source)).name
        use_opt = self._use_optimized_var.get() and _FAST_PIPELINE_AVAILABLE
        engine = self._get_fast_engine() if use_opt else None
        if engine is not None:
            start = lambda: self._start_video_optimized(engine, src_name)
        else:
            start = lambda: self._start_video_classic(src_name)
        # A restart (settings change mid-stream) preserves the already-confirmed
        # ROI, so resume detection immediately instead of re-prompting.
        if self._restart_in_progress:
            start()
            return
        # Fresh load: grab the first frame for ROI drawing, then hold detection
        # until the user confirms an ROI (rewind files so frame 0 is not consumed).
        ok, first = cap.read()
        if not ok or first is None:
            messagebox.showerror("Error", f"No frames to read from: {source}")
            self._stop_video()
            return
        if not self._camera_mode:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._arm_detection_gate(first, start, src_name)

    def _start_video_classic(self, src_name: str) -> None:
        # Reset the ETA/FPS clock now that processing actually begins — the
        # time spent drawing the ROI must not count against the frame rate.
        self._video_start_time = time.monotonic()
        self._status_var.set(f"Playing (classic): {src_name}")
        self._pipeline_var.set("Classic")
        self._video_thread = threading.Thread(
            target=self._video_loop_classic,
            args=(src_name,),
            daemon=True,
            name="VideoLoopClassic",
        )
        self._video_thread.start()

    def _video_loop_classic(self, source_name: str) -> None:
        raw_frame_idx = 0
        consecutive_read_failures = 0
        max_read_failures = 10  # tolerate transient camera glitches
        while self._video_running and self._video_cap is not None:
            stride = max(1, self._stride_var.get())
            if self._video_paused or self._awaiting_roi:
                time.sleep(0.05)
                continue
            ret, frame = self._video_cap.read()
            if not ret:
                if not self._camera_mode:
                    self._video_running = False
                    self.root.after(
                        0,
                        lambda: self._status_var.set(
                            f"Video complete: {len(self._results_history)} frames"
                        ),
                    )
                    self.root.after(0, self._on_video_complete)
                    break
                # Camera mode: tolerate transient failures
                consecutive_read_failures += 1
                if consecutive_read_failures >= max_read_failures:
                    self.root.after(
                        0,
                        lambda: self._status_var.set(
                            "Camera read failed repeatedly — stopping"
                        ),
                    )
                    self._video_running = False
                    break
                time.sleep(0.01)
                continue
            consecutive_read_failures = 0
            raw_frame_idx += 1
            if stride > 1 and (raw_frame_idx % stride) != 0:
                self.root.after(0, self._update_progress, raw_frame_idx, False)
                continue
            self._frame_count += 1
            self._current_image = frame
            if self._detector is None:
                self.root.after(0, self._update_progress, raw_frame_idx, True)
                continue
            manual_crop = self._get_manual_roi_crop(frame)
            if manual_crop is None:
                self._current_result = None
                self.root.after(0, self._update_progress, raw_frame_idx, True)
                continue
            crop, roi_x, roi_y = manual_crop
            self._sync_calibration_to_detector()
            result = self._detector.detect_video_frame(
                crop,
                frame_number=self._frame_count,
                roi_x=roi_x,
                roi_y=roi_y,
            )
            result = self._apply_manual_ring_policy(result)
            result.metadata.source = source_name
            if self._tracker is not None:
                smoothed = self._tracker.update(result)
            else:
                smoothed = result
            if smoothed.has_both and self._corneal_calc is not None:
                smoothed.corneal_center = self._corneal_calc.calculate(
                    smoothed.pupil,
                    smoothed.limbus,
                    result.calibration,
                )
            smoothed.calibration = result.calibration
            smoothed.ring_status = getattr(result, "ring_status", "unknown")
            smoothed.ring_center = getattr(result, "ring_center", None)
            smoothed.ring_radius = getattr(result, "ring_radius", None)
            smoothed.ring_contour = getattr(result, "ring_contour", None)
            smoothed.ring_dot_count = getattr(result, "ring_dot_count", 0)
            smoothed.corneal_reference_source = getattr(
                result,
                "corneal_reference_source",
                "limbus",
            )
            if (
                smoothed.ring_status == "ring_present"
                and smoothed.ring_center is not None
                and getattr(smoothed, "pupil", None) is not None
                and getattr(smoothed.pupil, "ellipse", None) is not None
            ):
                px = smoothed.pupil.ellipse.center_x
                py = smoothed.pupil.ellipse.center_y
                points = [(px, py, "pupil")]
                weights = [max(getattr(smoothed.pupil, "confidence", 0.0), 1e-3)]
                if getattr(smoothed, "limbus", None) is not None and getattr(smoothed.limbus, "ellipse", None) is not None:
                    points.append(
                        (
                            smoothed.limbus.ellipse.center_x,
                            smoothed.limbus.ellipse.center_y,
                            "limbus",
                        )
                    )
                    weights.append(max(getattr(smoothed.limbus, "confidence", 0.0), 1e-3))
                points.append((smoothed.ring_center[0], smoothed.ring_center[1], "ring"))
                weights.append(max(getattr(result, "ring_confidence", 0.0), 1e-3))
                total_w = sum(weights)
                rcx = sum(pt[0] * w for pt, w in zip(points, weights)) / total_w
                rcy = sum(pt[1] * w for pt, w in zip(points, weights)) / total_w
                smoothed.corneal_reference_source = "+".join(name for _, _, name in points)
                smoothed.corneal_center.center_px = (rcx, rcy)
                smoothed.corneal_center.offset_px = (px - rcx, py - rcy)
                smoothed.corneal_center.offset_magnitude_px = math.hypot(
                    px - rcx,
                    py - rcy,
                )
                smoothed.corneal_center.offset_angle_deg = math.degrees(
                    math.atan2(py - rcy, px - rcx)
                )
                smoothed.corneal_center.valid = True
                if result.calibration.calibrated:
                    smoothed.corneal_center.center_mm = result.calibration.point_px_to_mm((rcx, rcy))
                    dx_mm = (px - rcx) * result.calibration.mm_per_px
                    dy_mm = (py - rcy) * result.calibration.mm_per_px
                    smoothed.corneal_center.offset_mm = (dx_mm, dy_mm)
                    smoothed.corneal_center.offset_magnitude_mm = math.hypot(dx_mm, dy_mm)
            self._current_result = smoothed
            self._results_history.append(smoothed.to_dict())

            # ══════════════════════════════════════════════════════════
            # RECORDING — Write frame to recorder at full resolution
            # ══════════════════════════════════════════════════════════
            if self._recorder.is_recording:
                annotated = self._prepare_recording_frame(frame, smoothed)
                self._write_frame_to_recorder(annotated)
            # ══════════════════════════════════════════════════════════

            now = time.monotonic()
            display_interval = self._get_display_interval()
            if (now - self._last_display_update) >= display_interval:
                self._last_display_update = now
                self.root.after(0, self._on_classic_frame, smoothed)
            self.root.after(0, self._update_progress, raw_frame_idx, True)

    def _on_classic_frame(self, result: Any) -> None:
        self._update_measurements(result)
        self._fps_var.set("---")
        self._refresh_display()

    def _start_video_optimized(self, engine: Any, src_name: str) -> None:
        model_path = self._find_model_path() or "models/best_model.pth"
        try:
            self._opt_processor = OptimizedVideoProcessor(
                model_path=model_path,
                device="auto",
                input_size=self._resolution_var.get(),
                half_precision=self._fp16_var.get(),
                use_compile=self._compile_var.get(),
                enable_auto_roi=self._roi_var.get(),
                roi_cache_ttl=self._roi_cache_var.get(),
                fast_mode=True,
                skip_quality_check=False,
                batch_size=self._runtime_profile.recommended_batch_size,
            )
            self._apply_manual_roi_to_processor()
            self._apply_manual_ring_to_processor()
            engine.warmup()
        except Exception as exc:
            self.logger.error("Optimized video startup failed, falling back to classic: %s", exc)
            self._opt_processor = None
            self._using_optimized_camera = False
            self._engine_status_var.set(f"Engine: fallback to classic - {exc}")
            self._status_var.set("Optimized video startup failed - using classic pipeline")
            self._start_video_classic(src_name)
            return
        self._using_optimized_camera = True
        preset_label = self._performance_preset_var.get().replace("_", " ").title()
        self._status_var.set(f"Playing (optimised): {src_name}")
        self._pipeline_var.set(f"Optimised [{preset_label}]")
        # Reset the ETA/FPS clock now that processing actually begins — the ROI
        # drawing wait and one-time engine warmup must not skew the frame rate.
        self._video_start_time = time.monotonic()
        self._video_thread = threading.Thread(
            target=self._video_loop_optimized,
            args=(src_name,),
            daemon=True,
            name="VideoLoopOptimised",
        )
        self._video_thread.start()

    def _video_loop_optimized(self, source_name: str) -> None:
        stride = max(1, self._stride_var.get())
        raw_frame_idx = 0
        fps_counter = 0
        fps_timer = time.monotonic()
        current_fps = 0.0

        # Use decode-ahead threading for video files (not camera)
        if not self._camera_mode and self._video_cap is not None:
            import queue as _queue

            frame_queue = _queue.Queue(
                maxsize=max(6, self._runtime_profile.recommended_capture_buffer * 3)
            )
            from pupil_tracking.video.optimized_processor import _FrameReader

            reader = _FrameReader(
                self._video_cap,
                frame_queue,
                stride=stride,
                pause_predicate=lambda: self._video_paused or self._awaiting_roi,
            )
            reader.start()

            try:
                while self._video_running:
                    if self._video_paused or self._awaiting_roi:
                        time.sleep(0.05)
                        continue
                    try:
                        item = frame_queue.get(timeout=5.0)
                    except _queue.Empty:
                        break
                    if item is None:
                        # Video complete
                        self._video_running = False
                        self.root.after(
                            0,
                            lambda: self._status_var.set(
                                f"Video complete (optimised): "
                                f"{len(self._results_history)} frames"
                            ),
                        )
                        self.root.after(0, self._on_video_complete)
                        break
                    pending_end = False
                    if (
                        self._opt_processor is not None
                        and self._opt_processor.should_shed_input_frames()
                        and frame_queue.qsize() > 3
                    ):
                        try:
                            queued_item = frame_queue.get_nowait()
                        except _queue.Empty:
                            queued_item = None
                        if queued_item is None:
                            pending_end = True
                        else:
                            item = queued_item
                    raw_frame_idx, frame = item
                    if self._opt_processor is not None:
                        self._opt_processor.note_source_frame(raw_frame_idx)
                    self._frame_count += 1
                    self._current_image = frame
                    try:
                        frame_result = self._opt_processor.process_frame(
                            frame, self._frame_count
                        )
                    except Exception as exc:
                        self.logger.error("Optimised video frame error: %s", exc)
                        continue
                    try:
                        fr_ns = self._dict_to_frame_ns(frame_result)
                        adapted = self._adapt_frame_result(fr_ns, frame.shape)
                    except Exception as exc:
                        self.logger.error("Optimised adapt error: %s", exc)
                        continue
                    self._current_result = adapted
                    try:
                        self._results_history.append(adapted.to_dict())
                    except Exception as exc:
                        self.logger.error("Optimised to_dict error: %s", exc)

                    # ══════════════════════════════════════════════════════════
                    # RECORDING — Write frame to recorder at full resolution
                    # ══════════════════════════════════════════════════════════
                    if self._recorder.is_recording:
                        annotated = self._prepare_recording_frame(frame, adapted)
                        self._write_frame_to_recorder(annotated)
                    # ══════════════════════════════════════════════════════════

                    fps_counter += 1
                    now = time.monotonic()
                    display_interval = self._get_display_interval()
                    elapsed_fps = now - fps_timer
                    if elapsed_fps >= 1.0:
                        current_fps = fps_counter / elapsed_fps
                        fps_counter = 0
                        fps_timer = now
                    if (now - self._last_display_update) >= display_interval:
                        self._last_display_update = now
                        _fps = current_fps
                        self.root.after(
                            0,
                            self._on_optimized_video_frame,
                            adapted,
                            _fps,
                        )
                    self.root.after(0, self._update_progress, raw_frame_idx, True)
                    if pending_end:
                        self._video_running = False
                        self.root.after(
                            0,
                            lambda: self._status_var.set(
                                f"Video complete (optimised): "
                                f"{len(self._results_history)} frames"
                            ),
                        )
                        self.root.after(0, self._on_video_complete)
                        break
            finally:
                reader.stop()
                reader.join(timeout=3.0)

            # If loop ended naturally (not via stop button), signal completion
            if not self._video_running:
                return
            self._video_running = False
            self.root.after(
                0,
                lambda: self._status_var.set(
                    f"Video complete (optimised): {len(self._results_history)} frames"
                ),
            )
            self.root.after(0, self._on_video_complete)
            return

        # Fallback: synchronous read (camera via _start_video path)
        while self._video_running and self._video_cap is not None:
            stride = max(1, self._stride_var.get())
            if self._video_paused or self._awaiting_roi:
                time.sleep(0.05)
                continue
            ret, frame = self._video_cap.read()
            if not ret:
                self._video_running = False
                self.root.after(
                    0,
                    lambda: self._status_var.set(
                        f"Video complete (optimised): "
                        f"{len(self._results_history)} frames"
                    ),
                )
                self.root.after(0, self._on_video_complete)
                break
            raw_frame_idx += 1
            if stride > 1 and (raw_frame_idx % stride) != 0:
                self.root.after(0, self._update_progress, raw_frame_idx, False)
                continue
            if self._opt_processor is not None:
                self._opt_processor.note_source_frame(raw_frame_idx)
            self._frame_count += 1
            self._current_image = frame
            try:
                frame_result = self._opt_processor.process_frame(
                    frame, self._frame_count
                )
            except Exception as exc:
                self.logger.error("Optimised video frame error: %s", exc)
                continue
            try:
                fr_ns = self._dict_to_frame_ns(frame_result)
                adapted = self._adapt_frame_result(fr_ns, frame.shape)
            except Exception as exc:
                self.logger.error("Optimised adapt error: %s", exc)
                continue
            self._current_result = adapted
            try:
                self._results_history.append(adapted.to_dict())
            except Exception as exc:
                self.logger.error("Optimised to_dict error: %s", exc)

            # ══════════════════════════════════════════════════════════
            # RECORDING — Write frame to recorder at full resolution
            # ══════════════════════════════════════════════════════════
            if self._recorder.is_recording:
                annotated = self._prepare_recording_frame(frame, adapted)
                self._write_frame_to_recorder(annotated)
            # ══════════════════════════════════════════════════════════

            fps_counter += 1
            now = time.monotonic()
            display_interval = self._get_display_interval()
            elapsed_fps = now - fps_timer
            if elapsed_fps >= 1.0:
                current_fps = fps_counter / elapsed_fps
                fps_counter = 0
                fps_timer = now
            if (now - self._last_display_update) >= display_interval:
                self._last_display_update = now
                _fps = current_fps
                self.root.after(
                    0,
                    self._on_optimized_video_frame,
                    adapted,
                    _fps,
                )
            self.root.after(0, self._update_progress, raw_frame_idx, True)

    def _on_optimized_video_frame(self, adapted: Any, fps: float) -> None:
        self._update_measurements(adapted)
        if self._opt_processor is not None:
            stats = self._opt_processor.get_stats()
            self._last_opt_stats = dict(stats)
            res = stats.get("resolution", "?")
            skip = stats.get("frame_skip", 0)
            roi_mode = stats.get("roi_mode", "off")
            roi = {"manual": "M", "auto": "A", "off": "N"}.get(roi_mode, "Y")
            lat_avg = stats.get("latency_recent_ms", stats.get("latency_avg_ms", 0.0))
            dropped = stats.get("dropped_frames", 0)
            stale = stats.get("stale_frames", 0)
            tracking_state = self._derive_tracking_state(adapted, stats)
            self._fps_var.set(f"{fps:.1f}  (res {res}, skip {skip}, ROI {roi})")
            self._latency_avg_var.set(f"{lat_avg:.1f} ms")
            self._drop_var.set(f"{dropped} / {stale}")
            self._tracking_state_var.set(tracking_state)
            self._set_summary_tracking_state(tracking_state)
            overload_suffix = "  |  Overload Protection Active" if stats.get("overload_active") else ""
            self._status_var.set(
                f"Video ({tracking_state.lower()}) — {lat_avg:.0f} ms avg latency{overload_suffix}"
            )
        else:
            self._fps_var.set(f"{fps:.1f}")
            self._latency_avg_var.set("---")
            self._drop_var.set("---")
            self._tracking_state_var.set("---")
        self._refresh_display()

    def _update_progress(self, raw_frame_idx: int, processed: bool) -> None:
        total = self._video_total_frames
        if total <= 0:
            self._progress_label_var.set(f"Frame {raw_frame_idx}")
            return
        self._progress_bar["value"] = raw_frame_idx
        pct = raw_frame_idx / total * 100.0
        self._progress_label_var.set(
            f"Frame {raw_frame_idx}/{total}  ({pct:.1f}%)  —  "
            f"{len(self._results_history)} processed"
        )
        elapsed = time.monotonic() - self._video_start_time
        if raw_frame_idx > 0 and elapsed > 0.5:
            remaining_frames = total - raw_frame_idx
            rate = raw_frame_idx / elapsed
            if rate > 0:
                eta_sec = remaining_frames / rate
                if eta_sec < 60:
                    self._eta_label_var.set(f"ETA: {eta_sec:.0f}s")
                elif eta_sec < 3600:
                    m, s = divmod(int(eta_sec), 60)
                    self._eta_label_var.set(f"ETA: {m}m {s}s")
                else:
                    h, rem = divmod(int(eta_sec), 3600)
                    m, s = divmod(rem, 60)
                    self._eta_label_var.set(f"ETA: {h}h {m}m")
            else:
                self._eta_label_var.set("")
        else:
            self._eta_label_var.set("")

    def _on_video_complete(self) -> None:
        total = self._video_total_frames
        if total > 0:
            self._progress_bar["value"] = total
        elapsed = time.monotonic() - self._video_start_time
        n = len(self._results_history)
        avg_fps = n / elapsed if elapsed > 0 else 0
        self._progress_label_var.set(
            f"Done — {n} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS)"
        )
        self._eta_label_var.set("✓ Complete")
        self._pause_btn.config(state=tk.DISABLED, text="⏸ Pause")

    # ================================================================
    # Camera
    # ================================================================

    def _start_camera(self) -> None:
        self._stop_video()
        self._camera_mode = True
        engine = self._get_fast_engine() if self._use_optimized_var.get() else None
        if engine is not None:
            self._start_camera_optimized(engine)
        else:
            self._start_video(0)

    def _start_camera_optimized(self, engine: Any) -> None:
        try:
            self._async_capture = AsyncCapture(
                0,
                buffer_size=self._runtime_profile.recommended_capture_buffer,
            )
            self._async_capture.start()
        except RuntimeError as exc:
            messagebox.showerror("Camera Error", str(exc))
            return
        self._active_source = 0
        try:
            self._opt_processor = OptimizedVideoProcessor(
                model_path=self._find_model_path() or "models/best_model.pth",
                device="auto",
                input_size=self._resolution_var.get(),
                half_precision=self._fp16_var.get(),
                use_compile=self._compile_var.get(),
                enable_auto_roi=self._roi_var.get(),
                roi_cache_ttl=self._roi_cache_var.get(),
                fast_mode=True,
                skip_quality_check=False,
                batch_size=self._runtime_profile.recommended_batch_size,
            )
            self._apply_manual_roi_to_processor()
            self._apply_manual_ring_to_processor()
            engine.warmup()
        except Exception as exc:
            self.logger.error("Optimized camera startup failed, falling back to classic: %s", exc)
            self._engine_status_var.set(f"Engine: fallback to classic - {exc}")
            self._status_var.set("Optimized camera startup failed - using classic pipeline")
            self._opt_processor = None
            if self._async_capture is not None:
                try:
                    self._async_capture.stop()
                except Exception:
                    pass
                self._async_capture = None
            self._using_optimized_camera = False
            self._start_video(0)
            return
        self._using_optimized_camera = True
        # Grab a preview frame so the operator can place the ROI before any
        # detection runs. The AsyncCapture is already streaming, so poll it
        # briefly for the first decoded frame.
        first = None
        for _ in range(40):
            data = self._async_capture.read(timeout=0.05)
            if data is not None:
                first = data[1]
                break
        if first is None:
            messagebox.showerror("Camera Error", "No frames received from camera")
            self._stop_video()
            return
        self._arm_detection_gate(first, self._launch_camera_optimized, "Camera")

    def _launch_camera_optimized(self) -> None:
        self._using_optimized_camera = True
        self._video_running = True
        self._video_paused = False
        self._frame_count = 0
        self._results_history.clear()
        self._pause_btn.config(state=tk.NORMAL, text="⏸ Pause")
        preset_label = self._performance_preset_var.get().replace("_", " ").title()
        self._status_var.set("Camera (optimised) — starting…")
        self._pipeline_var.set(f"Optimised [{preset_label}]")
        self._progress_bar.config(mode="indeterminate")
        self._progress_label_var.set("Live camera")
        self._eta_label_var.set("")
        self._video_thread = threading.Thread(
            target=self._camera_loop_optimized,
            daemon=True,
            name="OptCameraLoop",
        )
        self._video_thread.start()

    def _camera_loop_optimized(self) -> None:
        fps_counter = 0
        fps_timer = time.monotonic()
        current_fps = 0.0

        while self._video_running and self._async_capture is not None:
            if self._video_paused or self._awaiting_roi:
                time.sleep(0.05)
                continue

            # Check for capture errors
            cap_err = self._async_capture.get_error()
            if cap_err is not None:
                self.logger.error("Camera capture error: %s", cap_err)
                self.root.after(
                    0,
                    lambda e=str(cap_err): self._status_var.set(f"Camera error: {e}"),
                )
                break

            data = self._async_capture.read(timeout=0.05)
            if data is None:
                continue
            fnum, frame, _ts = data
            if self._opt_processor is not None:
                self._opt_processor.note_source_frame(fnum)

            # Skip stale frames aggressively to keep surgical UI responsive.
            frame_age = time.time() - _ts
            stale_threshold = (
                self._opt_processor.get_stale_frame_threshold_s()
                if self._opt_processor is not None
                else 0.12
            )
            if frame_age > stale_threshold:
                if self._opt_processor is not None:
                    self._opt_processor.note_stale_frame()
                continue

            self._frame_count = fnum
            self._current_image = frame
            try:
                frame_result = self._opt_processor.process_frame(frame, fnum)
            except Exception as exc:
                self.logger.error("Optimised frame error: %s", exc)
                continue
            try:
                fr_ns = self._dict_to_frame_ns(frame_result)
                adapted = self._adapt_frame_result(fr_ns, frame.shape)
            except Exception as exc:
                self.logger.error("Optimised camera adapt error: %s", exc)
                continue
            self._current_result = adapted
            try:
                self._results_history.append(adapted.to_dict())
            except Exception as exc:
                self.logger.error("Optimised camera to_dict error: %s", exc)

            # ══════════════════════════════════════════════════════════
            # RECORDING — Write frame to recorder at full resolution
            # ══════════════════════════════════════════════════════════
            if self._recorder.is_recording:
                annotated = self._prepare_recording_frame(frame, adapted)
                self._write_frame_to_recorder(annotated)
            # ══════════════════════════════════════════════════════════

            fps_counter += 1
            now = time.monotonic()
            display_interval = self._get_display_interval()
            elapsed_fps = now - fps_timer
            if elapsed_fps >= 1.0:
                current_fps = fps_counter / elapsed_fps
                fps_counter = 0
                fps_timer = now

            if (now - self._last_display_update) >= display_interval:
                self._last_display_update = now
                self.root.after(0, self._on_optimized_frame, adapted, current_fps)

    def _on_optimized_frame(self, adapted: SimpleNamespace, fps: float = 0.0) -> None:
        self._update_measurements(adapted)
        if self._opt_processor is not None:
            stats = self._opt_processor.get_stats()
            self._last_opt_stats = dict(stats)
            res = stats.get("resolution", "?")
            skip = stats.get("frame_skip", 0)
            roi_mode = stats.get("roi_mode", "off")
            roi = {"manual": "M", "auto": "A", "off": "N"}.get(roi_mode, "Y")
            lat_avg = stats.get("latency_recent_ms", stats.get("latency_avg_ms", 0.0))
            dropped = stats.get("dropped_frames", 0)
            stale = stats.get("stale_frames", 0)
            tracking_state = self._derive_tracking_state(adapted, stats)
            self._fps_var.set(f"{fps:.1f}  (res {res}, skip {skip}, ROI {roi})")
            self._latency_avg_var.set(f"{lat_avg:.1f} ms")
            self._drop_var.set(f"{dropped} / {stale}")
            self._tracking_state_var.set(tracking_state)
            self._set_summary_tracking_state(tracking_state)
            overload_suffix = "  |  Overload Protection Active" if stats.get("overload_active") else ""
            self._status_var.set(
                f"Camera ({tracking_state.lower()}) — {lat_avg:.0f} ms avg latency{overload_suffix}"
            )
        else:
            self._fps_var.set("---")
            self._latency_avg_var.set("---")
            self._drop_var.set("---")
            self._tracking_state_var.set("---")
        self._progress_label_var.set(f"Frame {self._frame_count}")
        self._refresh_display()

    # ================================================================
    # Dict → namespace bridge for optimized pipeline results
    # ================================================================

    @staticmethod
    def _dict_to_frame_ns(d: dict) -> SimpleNamespace:
        """Convert a flat detection dict (from OptimizedVideoProcessor /
        FastInference / result_to_dict) into the SimpleNamespace that
        ``_adapt_frame_result`` expects."""
        pupil_det = d.get("pupil_detected", False)
        limbus_det = d.get("limbus_detected", False)

        if pupil_det:
            px, py = d.get("pupil_x", 0.0), d.get("pupil_y", 0.0)
            pr = d.get("pupil_radius", 0.0)
            # pupil_major/minor are already full-axis diameters from FastInference
            p_major = d.get("pupil_major", pr * 2)
            p_minor = d.get("pupil_minor", pr * 2)
            p_angle = d.get("pupil_angle", 0.0)
            pupil_center = (px, py)
            pupil_axes = (p_major, p_minor)
            pupil_angle = p_angle
        else:
            pupil_center = None
            pupil_axes = None
            pupil_angle = 0.0

        if limbus_det:
            lx, ly = d.get("limbus_x", 0.0), d.get("limbus_y", 0.0)
            lr = d.get("limbus_radius", 0.0)
            # limbus_major/minor are already full-axis diameters from FastInference
            l_major = d.get("limbus_major", lr * 2)
            l_minor = d.get("limbus_minor", lr * 2)
            l_angle = d.get("limbus_angle", 0.0)
            limbus_center = (lx, ly)
            limbus_axes = (l_major, l_minor)
            limbus_angle = l_angle
        else:
            limbus_center = None
            limbus_axes = None
            limbus_angle = 0.0

        conf = d.get(
            "overall_confidence",
            d.get("pupil_confidence", d.get("confidence", 0.0)),
        )

        q_str = d.get("overall_quality", "")
        quality = SimpleNamespace(value=q_str) if q_str else None

        return SimpleNamespace(
            pupil_center=pupil_center,
            pupil_axes=pupil_axes,
            pupil_angle=pupil_angle,
            limbus_center=limbus_center,
            limbus_axes=limbus_axes,
            limbus_angle=limbus_angle,
            confidence=conf,
            quality=quality,
            pupil_fit_type=d.get("pupil_fit_type"),
            limbus_fit_type=d.get("limbus_fit_type"),
            processing_ms=d.get("processing_time_ms", d.get("latency_ms", 0.0)),
            latency_ms=d.get("latency_ms", d.get("processing_time_ms", 0.0)),
            frame_number=d.get("frame_idx", 0),
            is_interpolated=not d.get("pupil_detected", False),
            ring_status=d.get("ring_status", "unknown"),
            ring_center=(
                (d.get("ring_center_x"), d.get("ring_center_y"))
                if d.get("ring_center_x") is not None and d.get("ring_center_y") is not None
                else None
            ),
            ring_radius=d.get("ring_radius"),
            ring_dot_count=d.get("ring_dot_count", 0),
            corneal_reference_source=d.get("corneal_reference_source", "limbus"),
            reuse_cached_result=bool(d.get("reuse_cached_result", False)),
            reuse_reason=d.get("reuse_reason"),
            _eye_result=d.get("_eye_result"),
        )

    # ================================================================
    # FrameResult adapter (unchanged from original)
    # ================================================================

    def _adapt_frame_result(
        self, fr: Any, frame_shape: Tuple[int, ...]
    ) -> SimpleNamespace:
        H, W = frame_shape[:2]
        eye_result = getattr(fr, "_eye_result", None)
        if eye_result is not None:
            eye_result.metadata.image_width = W
            eye_result.metadata.image_height = H
            eye_result.metadata.frame_number = getattr(fr, "frame_number", 0)
            eye_result.metadata.latency_ms = getattr(fr, "latency_ms", fr.processing_ms)
            # ── Override detector-internal calibration with GUI mode ──
            # The internal UnifiedDetector's StabilizedCalibrator is stuck
            # on ANATOMICAL_ANCHOR (never receives set_calibration_mode).
            # Build the correct calibration from the GUI dropdown, then
            # re-compute pre-computed mm attributes so to_dict() / CSV
            # export reflect the user's selection.
            _cal_mode = self._calibration_mode_var.get() if hasattr(self, "_calibration_mode_var") else "ANATOMICAL_ANCHOR"
            _corneal_mm = float(self._corneal_ref_mm_var.get() if hasattr(self, "_corneal_ref_mm_var") else _CORNEAL_DIAMETER_MM)
            _fixed_scale = float(self._fixed_scale_var.get() if hasattr(self, "_fixed_scale_var") else 58.2)
            _ring_ref_mm = float(self._ring_ref_mm_var.get() if hasattr(self, "_ring_ref_mm_var") else 9.4)
            if _cal_mode in ("FIXED_PIXEL_SCALE", "fixed_manual", "manual"):
                _px = max(0.1, _fixed_scale)
                new_cal = CalibrationInfo(
                    calibrated=True,
                    px_per_mm=_px,
                    mm_per_px=1.0 / _px,
                    source="fixed_manual",
                    method="fixed_manual",
                    reference_diameter_mm=0.0,
                    reference_diameter_px=0.0,
                    confidence=1.0,
                    corneal_diameter_assumed_mm=None,
                )
            elif _cal_mode == "RING_REFLECTION":
                _ring_r = getattr(fr, "ring_radius", None)
                if _ring_r is not None and _ring_r > 10:
                    _dia = _ring_r * 2.0
                    _px = _dia / _ring_ref_mm
                    new_cal = CalibrationInfo(
                        calibrated=True,
                        px_per_mm=_px,
                        mm_per_px=1.0 / _px,
                        source=f"ring_reflection_{_ring_ref_mm:.1f}mm",
                        method="ring_reflection",
                        reference_diameter_mm=_ring_ref_mm,
                        reference_diameter_px=_dia,
                        confidence=0.95,
                        corneal_diameter_assumed_mm=None,
                    )
                elif (
                    getattr(eye_result, "limbus", None) is not None
                    and getattr(eye_result.limbus, "detected", False)
                    and getattr(eye_result.limbus, "ellipse", None) is not None
                ):
                    _lsm = eye_result.limbus.ellipse.semi_major * 2.0
                    _px = _lsm / _corneal_mm if _corneal_mm > 0 else 0.0
                    new_cal = CalibrationInfo(
                        calibrated=True,
                        px_per_mm=_px,
                        mm_per_px=1.0 / _px if _px > 0 else 0.0,
                        source="limbus_semi_major (fallback)",
                        method="anatomical",
                        reference_diameter_mm=_corneal_mm,
                        reference_diameter_px=_lsm,
                        confidence=0.85,
                        corneal_diameter_assumed_mm=_corneal_mm,
                    )
                else:
                    new_cal = CalibrationInfo(
                        calibrated=False,
                        px_per_mm=0.0,
                        mm_per_px=0.0,
                        source="none",
                        method="ring_reflection",
                        reference_diameter_mm=0.0,
                        reference_diameter_px=0.0,
                        confidence=0.0,
                        corneal_diameter_assumed_mm=None,
                    )
            else:
                # ANATOMICAL_ANCHOR
                if (
                    getattr(eye_result, "limbus", None) is not None
                    and getattr(eye_result.limbus, "detected", False)
                    and getattr(eye_result.limbus, "ellipse", None) is not None
                ):
                    _lsm = eye_result.limbus.ellipse.semi_major * 2.0
                    _px = _lsm / _corneal_mm if _corneal_mm > 0 else 0.0
                    new_cal = CalibrationInfo(
                        calibrated=True,
                        px_per_mm=_px,
                        mm_per_px=1.0 / _px if _px > 0 else 0.0,
                        source="limbus_semi_major (optimised)",
                        method="anatomical",
                        reference_diameter_mm=_corneal_mm,
                        reference_diameter_px=_lsm,
                        confidence=min(0.95, getattr(eye_result, "overall_confidence", 0.0) + 0.05),
                        corneal_diameter_assumed_mm=_corneal_mm,
                    )
                else:
                    new_cal = CalibrationInfo(
                        calibrated=False,
                        px_per_mm=0.0,
                        mm_per_px=0.0,
                        source="none",
                        method="anatomical",
                        reference_diameter_mm=0.0,
                        reference_diameter_px=0.0,
                        confidence=0.0,
                        corneal_diameter_assumed_mm=_corneal_mm,
                    )
            eye_result.calibration = new_cal
            # Clear stale pre-computed mm attributes set by
            # _add_mm_values / evaluate_clinical_wtw during the
            # original detection with the wrong calibration.
            for target in (
                getattr(eye_result, "limbus", None),
                getattr(eye_result, "pupil", None),
            ):
                if target is None:
                    continue
                for attr in (
                    "wtw_horizontal_mm",
                    "wtw_vertical_mm",
                    "wtw_mean_mm",
                    "wtw_astigmatism_mm",
                    "is_wtw_measured",
                    "wtw_validity_status",
                    "radius_mm",
                    "center_mm",
                ):
                    if hasattr(target, attr):
                        try:
                            setattr(target, attr, None)
                        except Exception:
                            pass
            # Re-compute mm values with the correct calibration
            if new_cal.calibrated:
                if (
                    getattr(eye_result, "pupil", None) is not None
                    and eye_result.pupil.detected
                    and eye_result.pupil.ellipse is not None
                ):
                    pe = eye_result.pupil.ellipse
                    eye_result.pupil.radius_mm = pe.radius * new_cal.mm_per_px
                    eye_result.pupil.center_mm = (
                        pe.center_x * new_cal.mm_per_px,
                        pe.center_y * new_cal.mm_per_px,
                    )
                if (
                    getattr(eye_result, "limbus", None) is not None
                    and eye_result.limbus.detected
                    and eye_result.limbus.ellipse is not None
                ):
                    le = eye_result.limbus.ellipse
                    eye_result.limbus.radius_mm = le.radius * new_cal.mm_per_px
                    eye_result.limbus.center_mm = (
                        le.center_x * new_cal.mm_per_px,
                        le.center_y * new_cal.mm_per_px,
                    )
                    from pupil_tracking.calibration.spatial_calibration import (
                        evaluate_clinical_wtw,
                    )
                    h, v, m, astig, is_m, status = evaluate_clinical_wtw(
                        eye_result.limbus, new_cal,
                    )
                    eye_result.limbus.wtw_horizontal_mm = h
                    eye_result.limbus.wtw_vertical_mm = v
                    eye_result.limbus.wtw_mean_mm = m
                    eye_result.limbus.wtw_astigmatism_mm = astig
                    eye_result.limbus.is_wtw_measured = is_m
                    eye_result.limbus.wtw_validity_status = status
            return eye_result

        cal_mode = self._calibration_mode_var.get() if hasattr(self, "_calibration_mode_var") else "ANATOMICAL_ANCHOR"
        corneal_mm = float(self._corneal_ref_mm_var.get() if hasattr(self, "_corneal_ref_mm_var") else _CORNEAL_DIAMETER_MM)
        fixed_scale = float(self._fixed_scale_var.get() if hasattr(self, "_fixed_scale_var") else 58.2)
        ring_ref_mm = float(self._ring_ref_mm_var.get() if hasattr(self, "_ring_ref_mm_var") else 9.4)

        if cal_mode in ("FIXED_PIXEL_SCALE", "fixed_manual", "manual"):
            px_per_mm = max(0.1, fixed_scale)
            cal = SimpleNamespace(
                calibrated=True,
                px_per_mm=px_per_mm,
                mm_per_px=1.0 / px_per_mm,
                source="fixed_manual",
                method="fixed_manual",
                reference_diameter_mm=0.0,
                reference_diameter_px=0.0,
                confidence=1.0,
                corneal_diameter_assumed_mm=None,
            )
        elif cal_mode == "RING_REFLECTION":
            ring_radius = getattr(fr, "ring_radius", None)
            if ring_radius is not None and ring_radius > 10:
                dia_px = ring_radius * 2.0
                px_per_mm = dia_px / ring_ref_mm
                cal = SimpleNamespace(
                    calibrated=True,
                    px_per_mm=px_per_mm,
                    mm_per_px=1.0 / px_per_mm,
                    source=f"ring_reflection_{ring_ref_mm:.1f}mm",
                    method="ring_reflection",
                    reference_diameter_mm=ring_ref_mm,
                    reference_diameter_px=dia_px,
                    confidence=0.95,
                    corneal_diameter_assumed_mm=None,
                )
            elif fr.limbus_axes is not None:
                limbus_semi_major_dia = float(max(fr.limbus_axes))
                px_per_mm = limbus_semi_major_dia / corneal_mm
                cal = SimpleNamespace(
                    calibrated=True,
                    px_per_mm=px_per_mm,
                    mm_per_px=1.0 / px_per_mm if px_per_mm > 0 else 0.0,
                    source="limbus_semi_major (fallback)",
                    method="anatomical",
                    reference_diameter_mm=corneal_mm,
                    reference_diameter_px=limbus_semi_major_dia,
                    confidence=min(0.85, fr.confidence),
                    corneal_diameter_assumed_mm=corneal_mm,
                )
            else:
                cal = SimpleNamespace(
                    calibrated=False,
                    px_per_mm=0.0,
                    mm_per_px=0.0,
                    source="none",
                    method="ring_reflection",
                    reference_diameter_mm=0.0,
                    reference_diameter_px=0.0,
                    confidence=0.0,
                    corneal_diameter_assumed_mm=None,
                )
        else:  # ANATOMICAL_ANCHOR
            if fr.limbus_axes is not None:
                limbus_semi_major_dia = float(max(fr.limbus_axes))
                px_per_mm = limbus_semi_major_dia / corneal_mm
                mm_per_px = 1.0 / px_per_mm if px_per_mm > 0 else 0.0
                cal = SimpleNamespace(
                    calibrated=True,
                    px_per_mm=px_per_mm,
                    mm_per_px=mm_per_px,
                    source="limbus_semi_major (optimised)",
                    method="anatomical",
                    reference_diameter_mm=corneal_mm,
                    reference_diameter_px=limbus_semi_major_dia,
                    confidence=min(0.95, fr.confidence + 0.05),
                    corneal_diameter_assumed_mm=corneal_mm,
                )
            else:
                cal = SimpleNamespace(
                    calibrated=False,
                    px_per_mm=0.0,
                    mm_per_px=0.0,
                    source="none",
                    method="anatomical",
                    reference_diameter_mm=0.0,
                    reference_diameter_px=0.0,
                    confidence=0.0,
                    corneal_diameter_assumed_mm=corneal_mm,
                )
        _MAP = {
            "SURGICAL": "SURGICAL",
            "CLINICAL": "CLINICAL",
            "INTERPOLATED": "RESEARCH",
            "PREDICTED": "RESEARCH",
            "FAILED": "NO_DETECTION",
        }
        if fr.quality:
            raw_quality = fr.quality.value
            if raw_quality in _MAP:
                q_str = _MAP[raw_quality]
            elif fr.pupil_center is None and fr.limbus_center is None:
                q_str = "NO_DETECTION"
            else:
                q_str = "INSUFFICIENT"
        else:
            q_str = "NO_DETECTION"
        try:
            overall_q = DetectionQuality(q_str)
        except (ValueError, KeyError):
            overall_q = SimpleNamespace(value=q_str)
        pupil_fit_type = getattr(fr, "pupil_fit_type", None)
        limbus_fit_type = getattr(fr, "limbus_fit_type", None)

        def _make_ellipse(center, axes, angle, fit_type=None):
            if center is None or axes is None:
                return None
            full_a, full_b = float(max(axes)), float(min(axes))
            semi_a, semi_b = full_a / 2.0, full_b / 2.0
            mean_radius = (semi_a + semi_b) / 2.0  # match EllipseParams convention
            ecc = (
                math.sqrt(max(0.0, 1.0 - (semi_b / semi_a) ** 2)) if semi_a > 0 else 0.0
            )
            circ = (semi_b / semi_a) if semi_a > 0 else 1.0
            return SimpleNamespace(
                center_x=center[0],
                center_y=center[1],
                radius=mean_radius,
                semi_major=semi_a,
                semi_minor=semi_b,
                angle_deg=angle,
                eccentricity=ecc,
                circularity=circ,
                fit_quality=fr.confidence,
                fit_rms_residual=0.0,
                num_contour_points=0,
                uncertainty_center_x=1.0,
                uncertainty_center_y=1.0,
                fit_type=fit_type,
            )

        p_ell = _make_ellipse(
            fr.pupil_center, fr.pupil_axes, fr.pupil_angle, pupil_fit_type
        )
        pupil = SimpleNamespace(
            detected=p_ell is not None,
            ellipse=p_ell,
            confidence=fr.confidence if p_ell else 0.0,
            quality=overall_q,
            method=SimpleNamespace(value="ML_optimised"),
            fit_type=pupil_fit_type,
        )
        l_ell = _make_ellipse(
            fr.limbus_center, fr.limbus_axes, fr.limbus_angle, limbus_fit_type
        )
        limbus = SimpleNamespace(
            detected=l_ell is not None,
            ellipse=l_ell,
            confidence=(min(0.95, fr.confidence + 0.05) if l_ell else 0.0),
            quality=overall_q,
            method=SimpleNamespace(value="ML_optimised"),
            fit_type=limbus_fit_type,
        )
        ref_source = getattr(fr, "corneal_reference_source", "limbus")
        has_both = pupil.detected and limbus.detected
        if has_both:
            pe, le = pupil.ellipse, limbus.ellipse
            ring_center = getattr(fr, "ring_center", None)
            use_ring_reference = (
                getattr(fr, "ring_status", "unknown") == "ring_present"
                and ring_center is not None
            )
            pts = [(pe.center_x, pe.center_y, "pupil")]
            weights = [max(pupil.confidence, 1e-3)]
            pts.append((le.center_x, le.center_y, "limbus"))
            weights.append(max(limbus.confidence, 1e-3))
            if use_ring_reference:
                pts.append((ring_center[0], ring_center[1], "ring"))
                weights.append(max(getattr(fr, "confidence", 0.0), 1e-3))
            total_w = sum(weights)
            ref_x = sum(pt[0] * w for pt, w in zip(pts, weights)) / total_w
            ref_y = sum(pt[1] * w for pt, w in zip(pts, weights)) / total_w
            ref_source = "+".join(name for _, _, name in pts)
            dx = pe.center_x - ref_x
            dy = pe.center_y - ref_y
            mag_px = math.hypot(dx, dy)
            ang = math.degrees(math.atan2(dy, dx))
            if cal.calibrated:
                dx_mm, dy_mm = dx * cal.mm_per_px, dy * cal.mm_per_px
                mag_mm = mag_px * cal.mm_per_px
                off_mm = (dx_mm, dy_mm)
            else:
                mag_mm, off_mm = None, None
            cc = SimpleNamespace(
                valid=True,
                center_px=(ref_x, ref_y),
                offset_px=(dx, dy),
                offset_magnitude_px=mag_px,
                offset_magnitude_mm=mag_mm,
                offset_mm=off_mm,
                offset_angle_deg=ang,
            )
        else:
            cc = SimpleNamespace(
                valid=False,
                center_px=(0.0, 0.0),
                offset_px=(0.0, 0.0),
                offset_magnitude_px=0.0,
                offset_magnitude_mm=None,
                offset_mm=None,
                offset_angle_deg=0.0,
            )
        meta = SimpleNamespace(
            processing_time_ms=fr.processing_ms,
            latency_ms=getattr(fr, "latency_ms", fr.processing_ms),
            frame_number=fr.frame_number,
            image_width=W,
            image_height=H,
            source="camera (optimised)",
            reuse_cached_result=bool(getattr(fr, "reuse_cached_result", False)),
            reuse_reason=getattr(fr, "reuse_reason", None),
        )
        alerts: List[str] = []
        if fr.is_interpolated:
            alerts.append("⚡ Interpolated frame (Kalman prediction)")
        if fr.quality is not None and fr.quality.value == "FAILED":
            alerts.append("⚠ Detection failed this frame")

        result = SimpleNamespace(
            pupil=pupil,
            limbus=limbus,
            corneal_center=cc,
            calibration=cal,
            metadata=meta,
            overall_quality=overall_q,
            overall_confidence=fr.confidence,
            has_both=has_both,
            alerts=alerts,
            ring_status=getattr(fr, "ring_status", "unknown"),
            ring_center=getattr(fr, "ring_center", None),
            ring_radius=getattr(fr, "ring_radius", None),
            ring_dot_count=getattr(fr, "ring_dot_count", 0),
            corneal_reference_source=ref_source,
        )
        result.to_dict = lambda _r=result, _fr=fr, _cal=cal: (
            self._frame_result_to_export_dict(_fr, _cal, _r)
        )
        return result

    def _frame_result_to_export_dict(
        self,
        fr: Any,
        cal: SimpleNamespace,
        adapted: SimpleNamespace,
    ) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "metadata": {
                "frame_number": fr.frame_number,
                "processing_time_ms": fr.processing_ms,
                "latency_ms": getattr(fr, "latency_ms", fr.processing_ms),
                "source": "camera (optimised)",
            },
            "overall_quality": (
                adapted.overall_quality.value
                if hasattr(adapted.overall_quality, "value")
                else str(adapted.overall_quality)
            ),
            "overall_confidence": fr.confidence,
            "calibration": {
                "calibrated": cal.calibrated,
                "mm_per_px": cal.mm_per_px,
                "px_per_mm": cal.px_per_mm,
                "source": getattr(cal, "source", "none"),
                "method": getattr(cal, "method", "anatomical"),
                "corneal_diameter_assumed_mm": getattr(cal, "corneal_diameter_assumed_mm", None),
            },
        }
        if fr.pupil_center is not None and fr.pupil_axes is not None:
            semi_a, semi_b = max(fr.pupil_axes) / 2.0, min(fr.pupil_axes) / 2.0
            # Use the full major-axis diameter for exported diameters. The
            # radius-only convention is wrong for elliptical fits and leads to
            # inconsistent mm values when exported to CSV.
            mm = cal.mm_per_px if cal.calibrated else 0.0
            major_diameter_px = semi_a * 2.0
            d["pupil"] = {
                "detected": True,
                "confidence": fr.confidence,
                "fit_type": getattr(fr, "pupil_fit_type", None),
                "radius_mm": (semi_a * mm) if cal.calibrated else None,
                "center_mm": (
                    (fr.pupil_center[0] * mm, fr.pupil_center[1] * mm)
                    if cal.calibrated else None
                ),
                "ellipse": {
                    "center_x": fr.pupil_center[0],
                    "center_y": fr.pupil_center[1],
                    "radius": semi_a,
                    "semi_major": semi_a,
                    "semi_minor": semi_b,
                    "angle_deg": float(getattr(fr, "pupil_angle", 0.0) or 0.0),
                    "diameter_mm": (major_diameter_px * mm) if cal.calibrated else None,
                    "semi_major_mm": (semi_a * mm) if cal.calibrated else None,
                    "semi_minor_mm": (semi_b * mm) if cal.calibrated else None,
                },
            }
        else:
            d["pupil"] = {"detected": False, "ellipse": {}}
        if fr.limbus_center is not None and fr.limbus_axes is not None:
            semi_a, semi_b = max(fr.limbus_axes) / 2.0, min(fr.limbus_axes) / 2.0
            mm = cal.mm_per_px if cal.calibrated else 0.0
            major_diameter_px = semi_a * 2.0
            wtw_h = (2.0 * semi_a * mm) if cal.calibrated else None
            wtw_v = (2.0 * semi_b * mm) if cal.calibrated else None
            wtw_m = (major_diameter_px * mm) if cal.calibrated else None
            wtw_astig = (abs(wtw_h - wtw_v)) if (wtw_h is not None and wtw_v is not None) else None
            is_wtw_m = bool(cal.calibrated and getattr(cal, "method", "anatomical") != "anatomical")
            if not cal.calibrated:
                wtw_status = "UNAVAILABLE"
            elif not is_wtw_m:
                wtw_status = "ANCHORED_BASELINE"
            else:
                wtw_status = "VALID_CLINICAL_RANGE" if (wtw_m is not None and 9.5 <= wtw_m <= 13.5) else "OUT_OF_BOUNDS_WARNING"

            d["limbus"] = {
                "detected": True,
                "confidence": min(0.95, fr.confidence + 0.05),
                "fit_type": getattr(fr, "limbus_fit_type", None),
                "radius_mm": (semi_a * mm) if cal.calibrated else None,
                "center_mm": (
                    (fr.limbus_center[0] * mm, fr.limbus_center[1] * mm)
                    if cal.calibrated else None
                ),
                "wtw_horizontal_mm": wtw_h,
                "wtw_vertical_mm": wtw_v,
                "wtw_mean_mm": wtw_m,
                "wtw_astigmatism_mm": wtw_astig,
                "is_wtw_measured": is_wtw_m,
                "wtw_validity_status": wtw_status,
                "ellipse": {
                    "center_x": fr.limbus_center[0],
                    "center_y": fr.limbus_center[1],
                    "radius": semi_a,
                    "semi_major": semi_a,
                    "semi_minor": semi_b,
                    "angle_deg": float(getattr(fr, "limbus_angle", 0.0) or 0.0),
                    # In anatomical mode, mean_r*2*mm ≡ wtw_m (algebraically
                    # identical).  Use wtw_m explicitly so the CSV semantically
                    # agrees with the WTW card rather than re-deriving from px.
                    "diameter_mm": wtw_m if (cal.calibrated and wtw_m is not None) else ((mean_r * 2.0 * mm) if cal.calibrated else None),
                    "semi_major_mm": (semi_a * mm) if cal.calibrated else None,
                    "semi_minor_mm": (semi_b * mm) if cal.calibrated else None,
                },
            }
        else:
            d["limbus"] = {
                "detected": False,
                "ellipse": {},
                "is_wtw_measured": False,
                "wtw_validity_status": "UNAVAILABLE",
            }

        ring_center = getattr(fr, "ring_center", None)
        use_ring_reference = (
            getattr(fr, "ring_status", "unknown") == "ring_present"
            and ring_center is not None
        )
        if fr.pupil_center is not None and (
            use_ring_reference or fr.limbus_center is not None
        ):
            points = [(fr.pupil_center[0], fr.pupil_center[1], "pupil")]
            weights = [max(fr.confidence, 1e-3)]
            if fr.limbus_center is not None:
                points.append((fr.limbus_center[0], fr.limbus_center[1], "limbus"))
                weights.append(max(min(0.95, fr.confidence + 0.05), 1e-3))
            if use_ring_reference:
                points.append((ring_center[0], ring_center[1], "ring"))
                weights.append(max(fr.confidence, 1e-3))
            total_w = sum(weights)
            ref_center = (
                sum(pt[0] * w for pt, w in zip(points, weights)) / total_w,
                sum(pt[1] * w for pt, w in zip(points, weights)) / total_w,
            )
            dx = fr.pupil_center[0] - ref_center[0]
            dy = fr.pupil_center[1] - ref_center[1]
            mag_px = math.hypot(dx, dy)
            d["corneal_center"] = {
                "center_px": ref_center,
                "offset_magnitude_px": mag_px,
                "offset_magnitude_mm": (
                    mag_px * cal.mm_per_px if cal.calibrated else None
                ),
                "offset_angle_deg": math.degrees(math.atan2(dy, dx)),
            }
        else:
            d["corneal_center"] = {}
        d["ring_status"] = getattr(fr, "ring_status", "unknown")
        d["corneal_reference_source"] = getattr(fr, "corneal_reference_source", "limbus")
        if ring_center is not None:
            d["ring_center_x"] = ring_center[0]
            d["ring_center_y"] = ring_center[1]
        d["ring_radius"] = getattr(fr, "ring_radius", None)
        d["ring_dot_count"] = getattr(fr, "ring_dot_count", 0)
        return d

    # ================================================================
    # Stop
    # ================================================================

    def _stop_video(self) -> None:
        self._video_running = False
        self._video_paused = False
        self._camera_mode = False
        if not self._restart_in_progress:
            self._active_source = None
        self._roi_edit_active = False
        self._roi_drag_mode = None
        self._roi_drag_offset = (0.0, 0.0)
        self._roi_original_before_edit = None
        self._roi_preview = None
        # Clear any pending ROI-gated detection start.
        self._awaiting_roi = False
        self._pending_detection_start = None
        self._canvas.configure(cursor="crosshair")
        if hasattr(self, "_roi_btn"):
            self._roi_btn.config(text="ROI")
        if self._video_thread is not None:
            try:
                if threading.current_thread() is not self._video_thread:
                    self._video_thread.join(timeout=2.5)
            except Exception:
                pass
            self._video_thread = None
        if self._async_capture is not None:
            try:
                self._async_capture.stop()
            except Exception:
                pass
            self._async_capture = None
        if self._video_cap is not None:
            try:
                self._video_cap.release()
            except Exception:
                pass
            self._video_cap = None
        if self._opt_processor is not None:
            try:
                self._opt_processor.reset()
            except Exception:
                pass
            self._opt_processor = None
        self._last_opt_stats = {}
        self._using_optimized_camera = False
        self._pipeline_var.set("---")
        self._fps_var.set("---")
        self._pause_btn.config(state=tk.DISABLED, text="⏸ Pause")
        self._progress_bar.config(mode="determinate")
        self._progress_bar["value"] = 0
        # ══════════════════════════════════════════════════════════
        # RECORDING — Auto-stop recording when video stops
        # ══════════════════════════════════════════════════════════
        if self._recorder.is_recording:
            self._stop_recording()
        # ══════════════════════════════════════════════════════════

    # ================================================================
    # Display
    # ================================================================

    def _refresh_display(self) -> None:
        try:
            if self._current_image is None:
                self._draw_welcome_screen()
                return
            if not self._canvas.winfo_ismapped():
                return
        except Exception as exc:
            self.logger.exception("Display pre-check error: %s", exc)
            self._report_runtime_issue("Display recovered from an internal error")
            return

        if self._current_image is None:
            self._draw_welcome_screen()
            return
        if not self._canvas.winfo_ismapped():
            return

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 10 of 12 — Convert display frame
        #
        # The detector already processed the original image.
        # Now convert the DISPLAY copy to grayscale if mode
        # is active.  Overlays (green/blue circles) are drawn
        # on this grayscale background — IR camera look.
        # ══════════════════════════════════════════════════════════
        try:
            mode = self._grayscale_mode_var.get()
            if mode == "off":
                image = self._current_image
            else:
                image = self._convert_display_frame(self._current_image.copy())

            canvas_w = self._canvas.winfo_width()
            canvas_h = self._canvas.winfo_height()
            if canvas_w < 10 or canvas_h < 10:
                return
            h, w = image.shape[:2]
            scale = min(canvas_w / w, canvas_h / h, 1.0)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            offset_x = (canvas_w - new_w) // 2
            offset_y = (canvas_h - new_h) // 2
            self._display_scale = scale
            self._display_origin = (offset_x, offset_y)
            self._display_size = (new_w, new_h)

            display = cv2.resize(image, (new_w, new_h))

            result = self._current_result
            if result is not None and self._show_overlay.get():
                self._draw_overlay_scaled(display, result, scale)
            self._draw_manual_roi_overlay(display, scale)
            self._draw_manual_ring_overlay(display, scale)
            if self._show_debug_overlay.get():
                self._draw_debug_overlay(display, scale)
            self._show_image_fast(display, canvas_w, canvas_h, new_w, new_h)
        except Exception as exc:
            self.logger.exception("Display refresh error: %s", exc)
            self._report_runtime_issue("Display recovered from an internal error")

    def _draw_welcome_screen(self) -> None:
        c = self._colors
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        self._canvas.delete("all")
        self._canvas_image_id = None
        cx, cy = cw // 2, ch // 2
        self._canvas.create_text(
            cx,
            cy - 60,
            text="Medevplus IXcentai",
            fill=c.FG_PRIMARY,
            font=("Segoe UI", 22, "bold"),
            anchor="center",
        )
        self._canvas.create_text(
            cx,
            cy - 20,
            text="surgical grade",
            fill=c.ACCENT,
            font=("Segoe UI", 13),
            anchor="center",
        )
        self._canvas.create_text(
            cx,
            cy + 20,
            text="Open an image, video, or start the camera to begin",
            fill=c.FG_SECONDARY,
            font=("Segoe UI", 11),
            anchor="center",
        )
        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 11 of 12 — Updated welcome shortcuts
        # RECORDING — Updated shortcuts to include recording
        # ══════════════════════════════════════════════════════════
        shortcuts = (
            "Ctrl+O  Image    Ctrl+V  Video    Space  Pause"
            "    G  Grayscale    Ctrl+R  Record    Ctrl+Q  Quit"
        )
        # ══════════════════════════════════════════════════════════
        self._canvas.create_text(
            cx,
            cy + 60,
            text=shortcuts,
            fill=c.FG_TERTIARY,
            font=("Consolas", 9),
            anchor="center",
        )

    def _show_image_fast(
        self,
        image_bgr: np.ndarray,
        canvas_w: int,
        canvas_h: int,
        img_w: int,
        img_h: int,
    ) -> None:
        """Display an already-resized BGR image, reusing the canvas item."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self._display_image = ImageTk.PhotoImage(pil_img)
        x = (canvas_w - img_w) // 2
        y = (canvas_h - img_h) // 2
        if self._canvas_image_id is not None:
            try:
                self._canvas.coords(self._canvas_image_id, x, y)
                self._canvas.itemconfig(
                    self._canvas_image_id, image=self._display_image
                )
                return
            except tk.TclError:
                self._canvas_image_id = None
        self._canvas.delete("all")
        self._canvas_image_id = self._canvas.create_image(
            x, y, anchor=tk.NW, image=self._display_image
        )

    @staticmethod
    def _scale_ellipse(e: Any, scale: float) -> SimpleNamespace:
        """Return ellipse namespace with coordinates scaled for display."""
        return SimpleNamespace(
            center_x=e.center_x * scale,
            center_y=e.center_y * scale,
            radius=e.radius * scale,
            semi_major=e.semi_major * scale,
            semi_minor=e.semi_minor * scale,
            angle_deg=e.angle_deg,
        )

    def _get_ellipse_intersection(
        self, ellipse: Any, px: float, py: float, dx: float, dy: float
    ) -> Tuple[float, float]:
        """Compute the intersection of a ray from (px, py) in direction (dx, dy) with an ellipse."""
        cx = ellipse.center_x
        cy = ellipse.center_y
        a = max(1.0, ellipse.semi_major)
        b = max(1.0, ellipse.semi_minor)
        angle_rad = math.radians(ellipse.angle_deg)

        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Local coordinates of start point relative to ellipse center
        x_loc = (px - cx) * cos_a + (py - cy) * sin_a
        y_loc = -(px - cx) * sin_a + (py - cy) * cos_a

        # Local direction vector
        dx_loc = dx * cos_a + dy * sin_a
        dy_loc = -dx * sin_a + dy * cos_a

        # Quadratic equation coefficients for distance t: A * t^2 + 2 * B * t + C = 0
        A_coef = (dx_loc / a) ** 2 + (dy_loc / b) ** 2
        B_coef = (x_loc * dx_loc) / (a ** 2) + (y_loc * dy_loc) / (b ** 2)
        C_coef = (x_loc / a) ** 2 + (y_loc / b) ** 2 - 1.0

        disc = B_coef ** 2 - A_coef * C_coef
        if disc < 0 or A_coef == 0:
            # Fallback to a simple default bounding box boundary if math fails
            return px + dx * a, py + dy * b

        t = (-B_coef + math.sqrt(disc)) / A_coef
        return px + t * dx, py + t * dy

    def _draw_cross_section(self, out: np.ndarray, result: Any, scale: float) -> None:
        """Draw intersecting horizontal/vertical cross section lines between pupil and limbus."""
        if not (
            result.pupil.detected
            and result.pupil.ellipse is not None
            and result.limbus.detected
            and result.limbus.ellipse is not None
        ):
            return

        p_ellipse = self._scale_ellipse(result.pupil.ellipse, scale)
        l_ellipse = self._scale_ellipse(result.limbus.ellipse, scale)

        p_cx = p_ellipse.center_x
        p_cy = p_ellipse.center_y

        # Calculate intersection points on the pupil boundary
        p_up_pt = self._get_ellipse_intersection(p_ellipse, p_cx, p_cy, 0.0, -1.0)
        p_down_pt = self._get_ellipse_intersection(p_ellipse, p_cx, p_cy, 0.0, 1.0)
        p_left_pt = self._get_ellipse_intersection(p_ellipse, p_cx, p_cy, -1.0, 0.0)
        p_right_pt = self._get_ellipse_intersection(p_ellipse, p_cx, p_cy, 1.0, 0.0)

        l_cx = l_ellipse.center_x
        l_cy = l_ellipse.center_y

        # Calculate intersection points on the limbus boundary
        l_up_pt = self._get_ellipse_intersection(l_ellipse, l_cx, l_cy, 0.0, -1.0)
        l_down_pt = self._get_ellipse_intersection(l_ellipse, l_cx, l_cy, 0.0, 1.0)
        l_left_pt = self._get_ellipse_intersection(l_ellipse, l_cx, l_cy, -1.0, 0.0)
        l_right_pt = self._get_ellipse_intersection(l_ellipse, l_cx, l_cy, 1.0, 0.0)

        # Colors (BGR)
        green_color = (0, 255, 0)      # Pupil green
        blue_color = (255, 100, 0)     # Limbus blue

        # Draw pupil cross
        cv2.line(out, (int(round(p_left_pt[0])), int(round(p_left_pt[1]))), (int(round(p_right_pt[0])), int(round(p_right_pt[1]))), green_color, 1, cv2.LINE_AA)
        cv2.line(out, (int(round(p_up_pt[0])), int(round(p_up_pt[1]))), (int(round(p_down_pt[0])), int(round(p_down_pt[1]))), green_color, 1, cv2.LINE_AA)

        # Draw limbus cross
        cv2.line(out, (int(round(l_left_pt[0])), int(round(l_left_pt[1]))), (int(round(l_right_pt[0])), int(round(l_right_pt[1]))), blue_color, 1, cv2.LINE_AA)
        cv2.line(out, (int(round(l_up_pt[0])), int(round(l_up_pt[1]))), (int(round(l_down_pt[0])), int(round(l_down_pt[1]))), blue_color, 1, cv2.LINE_AA)

        # Draw ASCII degree labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_sz = max(0.3, 0.4 * scale)
        lbl_color = (220, 220, 220)

        # UP: 270 degrees
        up_x, up_y = int(round(l_up_pt[0])), int(round(l_up_pt[1]))
        cv2.putText(out, "270", (up_x - int(10 * scale), up_y - int(5 * scale)), font, font_sz, lbl_color, 1, cv2.LINE_AA)

        # DOWN: 90 degrees
        down_x, down_y = int(round(l_down_pt[0])), int(round(l_down_pt[1]))
        cv2.putText(out, "90", (down_x - int(7 * scale), down_y + int(12 * scale)), font, font_sz, lbl_color, 1, cv2.LINE_AA)

        # LEFT: 0 degrees
        left_x, left_y = int(round(l_left_pt[0])), int(round(l_left_pt[1]))
        cv2.putText(out, "0", (left_x - int(15 * scale), left_y + int(4 * scale)), font, font_sz, lbl_color, 1, cv2.LINE_AA)

        # RIGHT: 180 degrees
        right_x, right_y = int(round(l_right_pt[0])), int(round(l_right_pt[1]))
        cv2.putText(out, "180", (right_x + int(5 * scale), right_y + int(4 * scale)), font, font_sz, lbl_color, 1, cv2.LINE_AA)


    def _draw_overlay_scaled(self, out: np.ndarray, result: Any, scale: float) -> None:
        """Draw overlays on an already-resized image with scaled coords."""
        h, w = out.shape[:2]
        cal = result.calibration

        ring_status = getattr(result, "ring_status", "unknown")
        if ring_status == "ring_present":
            ring_center = getattr(result, "ring_center", None)
            ring_radius = getattr(result, "ring_radius", None)
            ring_contour = getattr(result, "ring_contour", None)
            if ring_center is not None and ring_radius is not None:
                cx = int(round(ring_center[0] * scale))
                cy = int(round(ring_center[1] * scale))
                rr = int(round(ring_radius * scale))
                if ring_contour is not None and len(ring_contour) >= 5:
                    scaled = np.round(ring_contour.astype(np.float32) * scale).astype(np.int32)
                    cv2.drawContours(out, [scaled], -1, (0, 0, 255), 2)
                else:
                    cv2.circle(out, (cx, cy), rr, (0, 0, 255), 2, cv2.LINE_AA)
                if self._show_ring_center.get():
                    _base = max(4, int(10 * scale))
                    _cal_mm_per_px = getattr(result.calibration, "mm_per_px", 0.0) or 0.0
                    if _cal_mm_per_px > 0:
                        _ring_cross_size = int(max(10, round(_base + (0.5 / _cal_mm_per_px) * scale)))
                    else:
                        _ring_cross_size = int(max(12, min(_base, 26)))
                    cv2.drawMarker(
                        out,
                        (cx, cy),
                        (0, 0, 255),
                        cv2.MARKER_CROSS,
                        _ring_cross_size,
                        2,
                        cv2.LINE_AA,
                    )
                if self._show_measurements.get():
                    label = f"R={ring_radius * 2.0:.0f}px"
                    if cal.calibrated:
                        label += f" ({ring_radius * 2.0 * cal.mm_per_px:.2f}mm)"
                    cv2.putText(
                        out,
                        label,
                        (cx + 10, cy - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.3, 0.45 * scale),
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

        # Draw limbus first (larger) so pupil renders on top
        if (
            self._show_limbus.get()
            and result.limbus.detected
            and result.limbus.ellipse is not None
        ):
            e_orig = result.limbus.ellipse
            e = self._scale_ellipse(e_orig, scale)
            limbus_color = (255, 100, 0)
            limbus_alpha = self._limbus_fill_alpha_var.get() / 100.0
            if limbus_alpha > 0:
                self._draw_filled_structure(out, e, limbus_color, limbus_alpha)
            ct = self._draw_structure(out, e, limbus_color)
            if self._show_centers.get():
                cv2.circle(out, ct, max(2, int(4 * scale)), limbus_color, -1)
            if self._show_measurements.get():
                dia_px = e_orig.radius * 2.0
                label = f"D={dia_px:.0f}px"
                if cal.calibrated:
                    dia_mm = dia_px * cal.mm_per_px
                    smaj_mm = e_orig.semi_major * cal.mm_per_px
                    smin_mm = e_orig.semi_minor * cal.mm_per_px
                    label += f" ({dia_mm:.2f}mm  {smaj_mm:.2f}x{smin_mm:.2f})"
                ft = getattr(e_orig, "fit_type", None) or getattr(
                    result.limbus, "fit_type", None
                )
                if ft:
                    label += f" [{ft}]"
                font_scale = max(0.3, 0.45 * scale)
                cv2.putText(
                    out,
                    label,
                    (ct[0] + 10, ct[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    limbus_color,
                    1,
                    cv2.LINE_AA,
                )

        # Draw pupil second so it always appears on top of the limbus fill
        if (
            self._show_pupil.get()
            and result.pupil.detected
            and result.pupil.ellipse is not None
        ):
            e_orig = result.pupil.ellipse
            e = self._scale_ellipse(e_orig, scale)
            pupil_color = (0, 255, 0)
            pupil_alpha = self._pupil_fill_alpha_var.get() / 100.0
            if pupil_alpha > 0:
                self._draw_filled_structure(out, e, pupil_color, pupil_alpha)
            ct = self._draw_structure(out, e, pupil_color)
            if self._show_centers.get():
                cv2.circle(out, ct, max(2, int(4 * scale)), pupil_color, -1)
            if self._show_measurements.get():
                dia_px = e_orig.radius * 2.0
                label = f"D={dia_px:.0f}px"
                if cal.calibrated:
                    label += f" ({dia_px * cal.mm_per_px:.2f}mm)"
                ft = getattr(e_orig, "fit_type", None) or getattr(
                    result.pupil, "fit_type", None
                )
                if ft:
                    label += f" [{ft}]"
                font_scale = max(0.3, 0.45 * scale)
                cv2.putText(
                    out,
                    label,
                    (ct[0] + 10, ct[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    pupil_color,
                    1,
                    cv2.LINE_AA,
                )

        cc = getattr(result, "corneal_center", None)
        if (
            self._show_centers.get()
            and cc is not None
            and getattr(cc, "valid", False)
            and getattr(cc, "center_px", None)
        ):
            center_pt = (
                int(round(cc.center_px[0] * scale)),
                int(round(cc.center_px[1] * scale)),
            )
            # Marker size is specified in *display* pixels. To keep the marker
            # size physically meaningful, we scale it using the current
            # calibration when available.
            #
            # Marker size is specified in *display* pixels.
            # Keep the marker size physically meaningful by converting
            # a desired physical offset (+3mm relative to the previous
            # baseline marker) into image pixels using calibration.
            _base = max(4, int(10 * scale))
            cal_mm_per_px = getattr(result.calibration, "mm_per_px", 0.0) or 0.0
            if cal_mm_per_px > 0:
                cursor_size = int(max(10, round(_base + (0.5 / cal_mm_per_px) * scale)))
            else:
                cursor_size = int(max(12, min(_base, 26)))
            cv2.drawMarker(
                out,
                center_pt,
                (255, 255, 255),
                cv2.MARKER_CROSS,
                cursor_size,
                2,
                cv2.LINE_AA,
            )
            if self._show_measurements.get():
                ref_name = getattr(result, "corneal_reference_source", "cornea")
                cv2.putText(
                    out,
                    f"Corneal Centre [{ref_name}]",
                    (center_pt[0] + 12, center_pt[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.35, 0.46 * scale),
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        if self._show_offset.get() and result.has_both:
            p = result.pupil.ellipse
            p_pt = (int(round(p.center_x * scale)), int(round(p.center_y * scale)))
            if cc is not None and getattr(cc, "valid", False) and getattr(cc, "center_px", None):
                ref_pt = (
                    int(round(cc.center_px[0] * scale)),
                    int(round(cc.center_px[1] * scale)),
                )
                dx = p.center_x - cc.center_px[0]
                dy = p.center_y - cc.center_px[1]
            else:
                l = result.limbus.ellipse
                ref_pt = (int(round(l.center_x * scale)), int(round(l.center_y * scale)))
                dx = p.center_x - l.center_x
                dy = p.center_y - l.center_y
            cv2.line(out, p_pt, ref_pt, (0, 255, 255), 2, cv2.LINE_AA)
            if self._show_centers.get():
                _base = max(4, int(10 * scale))
                _cal_mm_per_px = getattr(result.calibration, "mm_per_px", 0.0) or 0.0
                if _cal_mm_per_px > 0:
                    _offset_cross_size = int(max(10, round(_base + (0.5 / _cal_mm_per_px) * scale)))
                else:
                    _offset_cross_size = int(max(12, min(_base, 26)))
                cv2.drawMarker(
                    out,
                    ref_pt,
                    (255, 255, 255),
                    cv2.MARKER_CROSS,
                    _offset_cross_size,
                    2,
                    cv2.LINE_AA,
                )
            if self._show_measurements.get():
                offset_px = math.hypot(dx, dy)
                mid = ((p_pt[0] + ref_pt[0]) // 2, (p_pt[1] + ref_pt[1]) // 2)
                label = f"{offset_px:.1f}px"
                if cal.calibrated:
                    label += f" ({offset_px * cal.mm_per_px:.2f}mm)"
                font_scale = max(0.25, 0.4 * scale)
                cv2.putText(
                    out,
                    label,
                    (mid[0] + 5, mid[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        self._draw_cross_section(out, result, scale)

        # (Removed) On-image quality badge text (e.g. "SURGICAL (0.94)") that
        # was drawn top-left on the video.  The quality/confidence is still
        # shown in the ribbon badge and the measurements panel.
        font_scale_t = max(0.3, 0.5 * scale)
        cv2.putText(
            out,
            f"{result.metadata.processing_time_ms:.0f}ms",
            (w - max(80, int(100 * scale)), 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale_t,
            (180, 180, 180),
            1,
        )
        # (Removed) On-image "OVERLOAD PROTECTION" overlay that blinked
        # top-left while adaptive overload protection was active.  Overload
        # protection itself still runs; only the on-video text was removed.

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 12 of 12 — Grayscale mode badge on image
        # ══════════════════════════════════════════════════════════
        # (Removed) Grayscale mode badge text overlay (top-right).
        # Grayscale processing (OFF/AUTO/FORCE) and overlays remain active; 
        # only the on-image text label was removed to reduce clutter.
        # ══════════════════════════════════════════════════════════

        font_scale_a = max(0.25, 0.4 * scale)
        for i, alert in enumerate(result.alerts[:3]):
            cv2.putText(
                out,
                alert[:80],
                (10, h - 15 - i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_a,
                (0, 100, 255),
                1,
            )

        self._draw_ruler_overlay(out, scale)

    def _draw_debug_overlay(self, out: np.ndarray, scale: float) -> None:

        stats = self._last_opt_stats
        if not stats:
            return
        h, w = out.shape[:2]
        pad = max(8, int(10 * scale))
        line_gap = max(16, int(18 * scale))
        font_scale = max(0.32, 0.42 * scale)
        lines = [
            f"Preset: {self._performance_preset_var.get().replace('_', ' ').title()}",
            f"Pipeline: {stats.get('backend', self._pipeline_var.get())}",
            f"Latency avg: {float(stats.get('latency_avg_ms', 0.0) or 0.0):.1f} ms",
            f"Proc avg: {float(stats.get('processing_avg_ms', 0.0) or 0.0):.1f} ms",
            f"ROI avg: {float(stats.get('roi_avg_ms', 0.0) or 0.0):.1f} ms",
            f"ROI mode: {str(stats.get('roi_mode', 'off')).title()}",
            f"Tracking: {self._tracking_state_var.get() or '---'}",
            (
                "Adaptive quality: "
                + (
                    f"ON (stable={int(stats.get('stable_tracking_streak', 0))}, "
                    f"skips={int(stats.get('quality_check_skips', 0))})"
                    if stats.get("adaptive_quality_active")
                    else "OFF"
                )
            ),
            (
                f"Dropped/Stale: {int(stats.get('dropped_frames', 0))}/"
                f"{int(stats.get('stale_frames', 0))}"
            ),
            (
                "Overload protection: "
                + (
                    f"ACTIVE (reuse={int(stats.get('cached_reuse_total', 0))})"
                    if stats.get("overload_active")
                    else f"Idle (reuse={int(stats.get('cached_reuse_total', 0))})"
                )
            ),
        ]
        box_width = max(220, int(w * 0.34))
        box_height = pad * 2 + line_gap * len(lines)
        x0 = max(0, w - box_width - pad)
        y0 = max(0, h - box_height - pad)
        cv2.rectangle(out, (x0, y0), (x0 + box_width, y0 + box_height), (18, 18, 18), -1)
        cv2.rectangle(out, (x0, y0), (x0 + box_width, y0 + box_height), (70, 70, 70), 1)
        for idx, line in enumerate(lines):
            y = y0 + pad + line_gap * (idx + 1) - 4
            cv2.putText(
                out,
                line,
                (x0 + pad, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

    def _draw_manual_roi_overlay(self, out: np.ndarray, scale: float) -> None:
        roi = (
            self._roi_preview
            if self._roi_preview is not None
            else self._active_manual_roi()
        )
        if roi is None:
            return
        cx = int(round(roi["center_x"] * scale))
        cy = int(round(roi["center_y"] * scale))
        radius = max(1, int(round(roi["radius"] * scale)))
        is_editing = self._roi_preview is not None and self._roi_edit_active
        color = (0, 255, 255) if is_editing else (0, 220, 255)
        original = out.copy()
        mask = np.zeros(out.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius, 255, -1, cv2.LINE_AA)
        shaded = out.copy()
        shaded[:] = (15, 15, 15)
        outside = cv2.bitwise_not(mask)
        out[:] = cv2.addWeighted(out, 0.45, shaded, 0.55, 0.0)
        inside_original = cv2.bitwise_and(original, original, mask=mask)
        outside_dimmed = cv2.bitwise_and(out, out, mask=outside)
        out[:] = cv2.add(inside_original, outside_dimmed)
        cv2.circle(out, (cx, cy), radius, color, 2, cv2.LINE_AA)
        if is_editing:
            cv2.circle(out, (cx, cy), 3, color, -1)
        handle_x = cx + radius
        handle_y = cy
        cv2.circle(out, (handle_x, handle_y), max(4, int(6 * scale)), color, -1)
        
        dia_px = roi["radius"] * 2.0
        dia_str = f"Dia: {dia_px:.0f}px"
        if self._current_result is not None and getattr(self._current_result, "calibration", None) is not None and self._current_result.calibration.calibrated:
            dia_mm = dia_px * self._current_result.calibration.mm_per_px
            dia_str += f" ({dia_mm:.2f}mm)"
        
        label = f"ROI {dia_str} (Enter=lock)" if is_editing else f"ROI {dia_str}"
        font_scale = max(0.4, 0.5 * scale)
        (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        text_x = max(10, cx - radius - text_w - 10)
        if is_editing:
            cv2.putText(
                out,
                label,
                (text_x, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )
        
            caption = "ROI Edit: drag move/resize, arrows nudge, Enter apply, Esc cancel"
            cv2.putText(
                out,
                caption,
                (max(10, cx - radius), max(20, cy - radius - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.35, 0.48 * scale),
                color,
                1,
                cv2.LINE_AA,
            )

    def _draw_manual_ring_overlay(self, out: np.ndarray, scale: float) -> None:
        ring = (
            self._ring_preview
            if self._ring_preview is not None
            else self._active_manual_ring()
        )
        if ring is None:
            return
        cx = int(round(ring["center_x"] * scale))
        cy = int(round(ring["center_y"] * scale))
        radius = max(1, int(round(ring["radius"] * scale)))
        is_editing = self._ring_preview is not None and self._ring_edit_active
        if (
            not is_editing
            and self._current_result is not None
            and getattr(self._current_result, "ring_status", "unknown") == "ring_present"
        ):
            return
        color = (40, 110, 255) if is_editing else (0, 0, 255)
        thickness = 2 if is_editing else 3
        cv2.circle(out, (cx, cy), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 3, color, -1)
        handle_x = cx + radius
        handle_y = cy
        cv2.circle(out, (handle_x, handle_y), max(4, int(6 * scale)), color, -1)
        
        dia_px = ring["radius"] * 2.0
        dia_str = f"Dia: {dia_px:.0f}px"
        if self._current_result is not None and getattr(self._current_result, "calibration", None) is not None and self._current_result.calibration.calibrated:
            dia_mm = dia_px * self._current_result.calibration.mm_per_px
            dia_str += f" ({dia_mm:.2f}mm)"
            
        label = f"Manual Ring {dia_str} (Enter=lock)" if is_editing else f"Manual Ring {dia_str}"
        cv2.putText(
            out,
            label,
            (cx + 10, cy - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.35, 0.45 * scale),
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.circle(out, (handle_x, handle_y), max(5, int(7 * scale)), (20, 20, 20), 1)
        
        if is_editing:
            caption = "Ring Edit: drag move/resize, arrows nudge, Enter apply, Esc cancel"
            cv2.putText(
                out,
                caption,
                (max(10, cx - radius), max(20, cy - radius - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.35, 0.48 * scale),
                color,
                1,
                cv2.LINE_AA,
            )

    def _draw_ruler_overlay(self, out: np.ndarray, scale: float) -> None:
        if not getattr(self, "_ruler_calibration_active", False) and not getattr(self, "_ruler_points", None):
            return
        pts = getattr(self, "_ruler_points", [])
        color = (0, 255, 255)
        for i, pt in enumerate(pts):
            cx = int(round(pt[0] * scale))
            cy = int(round(pt[1] * scale))
            cv2.circle(out, (cx, cy), 5, color, -1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 9, color, 2, cv2.LINE_AA)
            cv2.putText(
                out,
                f"P{i+1}",
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.4, 0.5 * scale),
                color,
                1,
                cv2.LINE_AA,
            )

        if len(pts) >= 2:
            p1 = (int(round(pts[0][0] * scale)), int(round(pts[0][1] * scale)))
            p2 = (int(round(pts[1][0] * scale)), int(round(pts[1][1] * scale)))
            cv2.line(out, p1, p2, color, 2, cv2.LINE_AA)
            dist_px = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
            known_mm = float(self._ruler_known_dist_mm_var.get())
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            cv2.putText(
                out,
                f"{dist_px:.1f} px = {known_mm:.1f} mm ({dist_px/known_mm:.2f} px/mm)",
                (mid_x + 10, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.4, 0.52 * scale),
                color,
                2,
                cv2.LINE_AA,
            )

    @staticmethod

    def _draw_filled_structure(
        out: np.ndarray,
        ellipse_data: Any,
        color: Tuple[int, int, int],
        alpha: float,
    ) -> None:
        """Fill the circle/ellipse area with a semi-transparent colour overlay."""
        if alpha <= 0.0:
            return
        overlay = out.copy()
        ct = (int(round(ellipse_data.center_x)), int(round(ellipse_data.center_y)))
        ratio = (ellipse_data.semi_minor / ellipse_data.semi_major
                 if ellipse_data.semi_major > 0 else 1.0)
        if ratio > _CIRCLE_DRAW_THRESHOLD:
            r = int(round((ellipse_data.semi_major + ellipse_data.semi_minor) / 2.0))
            cv2.circle(overlay, ct, r, color, -1, cv2.LINE_AA)
        else:
            axes = (int(round(ellipse_data.semi_major)),
                    int(round(ellipse_data.semi_minor)))
            cv2.ellipse(overlay, ct, axes, int(round(ellipse_data.angle_deg)),
                        0, 360, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)

    @staticmethod
    def _draw_structure(
        out: np.ndarray,
        ellipse_data: Any,
        color: Tuple[int, int, int],
        thickness: int = 2,
    ) -> Tuple[int, int]:
        ct = (int(round(ellipse_data.center_x)), int(round(ellipse_data.center_y)))
        if ellipse_data.semi_major > 0:
            ratio = ellipse_data.semi_minor / ellipse_data.semi_major
        else:
            ratio = 1.0
        if ratio > _CIRCLE_DRAW_THRESHOLD:
            r = int(round((ellipse_data.semi_major + ellipse_data.semi_minor) / 2.0))
            cv2.circle(out, ct, r, color, thickness, cv2.LINE_AA)
        else:
            axes = (
                int(round(ellipse_data.semi_major)),
                int(round(ellipse_data.semi_minor)),
            )
            angle = int(round(ellipse_data.angle_deg))
            cv2.ellipse(out, ct, axes, angle, 0, 360, color, thickness, cv2.LINE_AA)
        return ct

    def _on_canvas_resize(self, _event: Any) -> None:
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(50, self._debounced_resize)

    def _debounced_resize(self) -> None:
        self._resize_after_id = None
        self._canvas_image_id = None
        self._refresh_display()

    # ================================================================
    # Measurements
    # ================================================================

    def _update_measurements(self, result: Any) -> None:
        try:
            cal = result.calibration
            has_cal = cal.calibrated if cal else False
            mm_per_px = cal.mm_per_px if has_cal else 0.0
            quality = (
                result.overall_quality.value
                if hasattr(result.overall_quality, "value")
                else str(result.overall_quality)
            )
            color = _QUALITY_COLORS.get(quality, "#888888")
            self._quality_label.config(
                text=f"  {quality} ({result.overall_confidence:.3f})  ",
                foreground=color,
            )
            self._summary_quality_var.set(quality)
            self._summary_quality_label.config(foreground=color)

            if result.pupil.detected and result.pupil.ellipse is not None:
                e = result.pupil.ellipse
                dia_px = e.radius * 2.0
                self._pv["center"].set(f"({e.center_x:.1f}, {e.center_y:.1f}) px")
                self._pv["diameter_px"].set(f"{dia_px:.1f} px")
                self._pv["diameter_mm"].set(
                    f"{dia_px * mm_per_px:.2f} mm" if has_cal else "— (no calibration)"
                )
                self._pv["semi_major"].set(f"{e.semi_major:.1f} px")
                self._pv["semi_major_mm"].set(
                    f"{e.semi_major * mm_per_px:.2f} mm"
                    if has_cal
                    else "— (no calibration)"
                )
                self._pv["semi_minor"].set(f"{e.semi_minor:.1f} px")
                self._pv["semi_minor_mm"].set(
                    f"{e.semi_minor * mm_per_px:.2f} mm"
                    if has_cal
                    else "— (no calibration)"
                )
                self._pv["angle"].set(f"{e.angle_deg:.1f}°")
                ft = getattr(e, "fit_type", None) or getattr(
                    result.pupil, "fit_type", None
                )
                self._pv["fit_type"].set(ft if ft else "—")
                self._pv["confidence"].set(f"{result.pupil.confidence:.3f}")
                q_val = (
                    result.pupil.quality.value
                    if hasattr(result.pupil.quality, "value")
                    else str(result.pupil.quality)
                )
                self._pv["quality"].set(q_val)
            else:
                for var in self._pv.values():
                    var.set("---")

            if result.limbus.detected and result.limbus.ellipse is not None:
                e = result.limbus.ellipse
                dia_px = e.radius * 2.0
                self._lv["center"].set(f"({e.center_x:.1f}, {e.center_y:.1f}) px")
                self._lv["diameter_px"].set(f"{dia_px:.1f} px")
                self._lv["diameter_mm"].set(
                    f"{dia_px * mm_per_px:.2f} mm" if has_cal else "— (no calibration)"
                )
                self._lv["semi_major"].set(f"{e.semi_major:.1f} px")
                self._lv["semi_major_mm"].set(
                    f"{e.semi_major * mm_per_px:.2f} mm"
                    if has_cal
                    else "— (no calibration)"
                )
                self._lv["semi_minor"].set(f"{e.semi_minor:.1f} px")
                self._lv["semi_minor_mm"].set(
                    f"{e.semi_minor * mm_per_px:.2f} mm"
                    if has_cal
                    else "— (no calibration)"
                )
                self._lv["angle"].set(f"{e.angle_deg:.1f}°")
                ft = getattr(e, "fit_type", None) or getattr(
                    result.limbus, "fit_type", None
                )
                self._lv["fit_type"].set(ft if ft else "—")
                self._lv["confidence"].set(f"{result.limbus.confidence:.3f}")
                q_val = (
                    result.limbus.quality.value
                    if hasattr(result.limbus.quality, "value")
                    else str(result.limbus.quality)
                )
                self._lv["quality"].set(q_val)
            else:
                for var in self._lv.values():
                    var.set("---")

            cc = result.corneal_center
            if cc.valid and result.has_both:
                pe = result.pupil.ellipse
                le = result.limbus.ellipse
                self._ov["corneal_center"].set(
                    f"({cc.center_px[0]:.1f}, {cc.center_px[1]:.1f}) px"
                )
                self._ov["corneal_reference"].set(
                    getattr(result, "corneal_reference_source", "limbus")
                )
                ring_center = getattr(result, "ring_center", None)
                if ring_center is not None:
                    self._ov["ring_center"].set(
                        f"({ring_center[0]:.1f}, {ring_center[1]:.1f}) px"
                    )
                else:
                    self._ov["ring_center"].set("---")
                dx, dy = cc.offset_px
                offset_px = cc.offset_magnitude_px
                offset_angle = cc.offset_angle_deg
                self._ov["offset_px"].set(f"{offset_px:.1f} px")
                self._ov["offset_vec_px"].set(f"({dx:.1f}, {dy:.1f}) px")
                if has_cal:
                    dx_mm, dy_mm = dx * mm_per_px, dy * mm_per_px
                    offset_mm = offset_px * mm_per_px
                    self._ov["offset_mm"].set(f"{offset_mm:.3f} mm")
                    self._ov["offset_vec_mm"].set(f"({dx_mm:.3f}, {dy_mm:.3f}) mm")
                else:
                    self._ov["offset_mm"].set("— (no calibration)")
                    self._ov["offset_vec_mm"].set("— (no calibration)")
                self._ov["offset_angle"].set(f"{offset_angle:.1f}°")
                ring_radius = getattr(result, "ring_radius", None)
                if ring_radius is not None:
                    ring_dia_px = ring_radius * 2.0
                    self._ov["ring_diameter_px"].set(f"{ring_dia_px:.1f} px")
                    if has_cal:
                        self._ov["ring_diameter_mm"].set(
                            f"{ring_dia_px * mm_per_px:.3f} mm"
                        )
                    else:
                        self._ov["ring_diameter_mm"].set("â€” (no calibration)")
                else:
                    self._ov["ring_diameter_px"].set("---")
                    self._ov["ring_diameter_mm"].set("---")
                if le.radius > 0:
                    self._ov["pupil_limbus_ratio"].set(f"{pe.radius / le.radius:.3f}")
                else:
                    self._ov["pupil_limbus_ratio"].set("---")
            else:
                for var in self._ov.values():
                    var.set("---")

            if has_cal:
                method_tag = getattr(cal, "method", "anatomical")
                self._cv_vars["source"].set(f"{cal.source} [{method_tag}]")
                self._cv_vars["scale_px"].set(f"{cal.px_per_mm:.2f} px/mm")
                self._cv_vars["scale_mm"].set(f"{cal.mm_per_px:.4f} mm/px")
                if getattr(cal, "corneal_diameter_assumed_mm", None):
                    self._cv_vars["reference"].set(
                        f"Assumed {cal.corneal_diameter_assumed_mm:.1f}mm"
                    )
                elif cal.reference_diameter_mm > 0:
                    self._cv_vars["reference"].set(
                        f"{cal.reference_diameter_mm:.1f}mm ({cal.reference_diameter_px:.0f}px)"
                    )
                else:
                    self._cv_vars["reference"].set("Fixed External Scale")
            else:
                self._cv_vars["source"].set("not calibrated")
                self._cv_vars["scale_px"].set("---")
                self._cv_vars["scale_mm"].set("---")
                self._cv_vars["reference"].set("---")

            # ── Corneal Dimensions (WTW) Card ──
            limbus_res = getattr(result, "limbus", None)
            if (
                hasattr(self, "_wtw_vars")
                and limbus_res is not None
                and getattr(limbus_res, "detected", False)
                and getattr(limbus_res, "ellipse", None) is not None
                and has_cal
            ):
                le = limbus_res.ellipse
                # Always recompute WTW from current pixel geometry and
                # calibration.  Pre-computed attributes (set by
                # _add_mm_values during detection) become stale when the
                # calibration mode changes without a new detection.
                #
                # Horizontal WTW = 2 x Limbus Semi-Major (mm)
                # Vertical   WTW = 2 x Limbus Semi-Minor (mm)
                # i.e. the full limbus axes in mm — exactly twice the
                # Semi-Major (mm) / Semi-Minor (mm) rows shown in the Limbus
                # panel.  Reported as the true measured geometry (no clamping).
                h_wtw = 2.0 * le.semi_major * mm_per_px
                v_wtw = 2.0 * le.semi_minor * mm_per_px
                m_wtw = (h_wtw + v_wtw) / 2.0

                self._wtw_vars["horizontal"].set(f"{h_wtw:.2f} mm")
                self._wtw_vars["vertical"].set(f"{v_wtw:.2f} mm")
                self._wtw_vars["mean"].set(f"{m_wtw:.2f} mm")
            elif hasattr(self, "_wtw_vars"):
                for var in self._wtw_vars.values():
                    var.set("---")


            proc_ms = float(getattr(result.metadata, "processing_time_ms", 0.0) or 0.0)
            reused = bool(getattr(result.metadata, "reuse_cached_result", False))
            if not reused and proc_ms > 0.5:
                self._last_real_proc_time_ms = proc_ms
            shown_proc_ms = getattr(self, "_last_real_proc_time_ms", proc_ms)
            if not reused:
                shown_proc_ms = proc_ms
            display_proc_prev = getattr(self, "_display_proc_time_ms", shown_proc_ms)
            proc_alpha = 0.18 if reused else 0.32
            display_proc_ms = display_proc_prev + proc_alpha * (shown_proc_ms - display_proc_prev)
            self._display_proc_time_ms = display_proc_ms
            self._proc_time_var.set(f"{display_proc_ms:.1f} ms")
            latency_ms = float(getattr(
                result.metadata, "latency_ms", result.metadata.processing_time_ms
            ) or 0.0)
            display_latency_prev = getattr(self, "_display_latency_ms", latency_ms)
            latency_alpha = 0.20 if reused else 0.34
            display_latency_ms = (
                display_latency_prev + latency_alpha * (latency_ms - display_latency_prev)
            )
            self._display_latency_ms = display_latency_ms
            self._latency_var.set(f"{display_latency_ms:.1f} ms")
            if not self._using_optimized_camera:
                self._latency_avg_var.set("---")
                self._drop_var.set("---")
                self._tracking_state_var.set("---")
            self._frame_var.set(str(result.metadata.frame_number))
            self._image_size_var.set(
                f"{result.metadata.image_width} × {result.metadata.image_height}"
            )
            self._summary_latency_var.set(f"{display_latency_ms:.1f} ms")
            self._summary_pipeline_var.set(self._pipeline_var.get())
            tracking_text = self._tracking_state_var.get()
            if not tracking_text or tracking_text == "---":
                tracking_text = "Ready" if result.has_both else "Waiting"
            self._set_summary_tracking_state(tracking_text)

            mode = self._grayscale_mode_var.get()
            mode_labels = {
                "off": "RGB (Original)",
                "auto": "Auto-Detect",
                "force": "Forced Grayscale",
            }
            gs_label = mode_labels.get(mode, mode)
            if self._detector is not None:
                gs_info = self._detector.last_grayscale_info
                if gs_info is not None and gs_info.conversion_applied:
                    gs_label += " ✓ applied"
            self._gray_mode_var_display.set(gs_label)

            if hasattr(self, "_gray_settings_status"):
                self._gray_settings_status.set(f"Current: {gs_label}")

            self._update_details(result)
        except Exception as exc:
            self.logger.exception("Measurement update error: %s", exc)
            self._report_runtime_issue(
                "Measurement panel recovered from an internal error"
            )

    def _update_details(self, result: Any) -> None:
        cal = result.calibration
        has_cal = cal.calibrated if cal else False
        mm_per_px = cal.mm_per_px if has_cal else 0.0

        self._details_text.config(state=tk.NORMAL)
        self._details_text.delete("1.0", tk.END)

        lines: List[str] = []
        lines.append(f"Source:  {result.metadata.source}")
        lines.append(
            f"Image:   {result.metadata.image_width}×{result.metadata.image_height}"
        )
        q_val = (
            result.overall_quality.value
            if hasattr(result.overall_quality, "value")
            else str(result.overall_quality)
        )
        lines.append(f"Quality: {q_val} ({result.overall_confidence:.4f})")
        lines.append(f"Time:    {result.metadata.processing_time_ms:.1f} ms")
        latency_ms = getattr(result.metadata, "latency_ms", None)
        if latency_ms is not None:
            lines.append(f"Latency: {latency_ms:.1f} ms")

        # Grayscale info in details
        mode = self._grayscale_mode_var.get()
        mode_names = {"off": "OFF (RGB)", "auto": "AUTO", "force": "FORCE (Grayscale)"}
        lines.append(f"Gray:    {mode_names.get(mode, mode)}")
        if self._detector is not None:
            gs_info = self._detector.last_grayscale_info
            if gs_info is not None:
                lines.append(
                    f"         applied={gs_info.conversion_applied}, "
                    f"input={'gray' if gs_info.was_grayscale else 'RGB'}"
                )
                if gs_info.conversion_applied:
                    lines.append(
                        f"         contrast {gs_info.contrast_before:.1f} "
                        f"→ {gs_info.contrast_after:.1f}"
                    )
        lines.append("")

        ring_status = getattr(result, "ring_status", "unknown")
        ring_center = getattr(result, "ring_center", None)
        ring_radius = getattr(result, "ring_radius", None)
        if ring_status == "ring_present" and ring_center is not None and ring_radius is not None:
            lines.append("=== SUCTION RING ===")
            lines.append(f"  Status:     {ring_status}")
            lines.append(f"  Method:     {getattr(result, 'ring_method', 'unknown')}")
            lines.append(
                f"  Center:     ({ring_center[0]:.2f}, {ring_center[1]:.2f}) px"
            )
            lines.append(f"  Diameter:   {ring_radius * 2.0:.2f} px")
            if has_cal:
                lines.append(f"  Diameter:   {ring_radius * 2.0 * mm_per_px:.3f} mm")
            lines.append(f"  Dots:       {getattr(result, 'ring_dot_count', 0)}")
            lines.append(
                f"  Reference:  {getattr(result, 'corneal_reference_source', 'limbus')}"
            )
            lines.append("")

        if result.pupil.detected and result.pupil.ellipse is not None:
            e = result.pupil.ellipse
            dia_px = e.radius * 2.0
            m_val = (
                result.pupil.method.value
                if hasattr(result.pupil.method, "value")
                else str(result.pupil.method)
            )
            ft = getattr(e, "fit_type", None) or getattr(result.pupil, "fit_type", None)
            lines.append("=== PUPIL ===")
            lines.append(f"  Method:     {m_val}")
            lines.append(f"  Fit Type:   {ft or '—'}")
            lines.append(f"  Center:     ({e.center_x:.2f}, {e.center_y:.2f}) px")
            lines.append(f"  Diameter:   {dia_px:.2f} px")
            if has_cal:
                lines.append(f"  Diameter:   {dia_px * mm_per_px:.3f} mm")
            lines.append(f"  Semi-axes:  {e.semi_major:.2f} × {e.semi_minor:.2f} px")
            if has_cal:
                lines.append(
                    f"  Semi-axes:  {e.semi_major * mm_per_px:.3f}"
                    f" × {e.semi_minor * mm_per_px:.3f} mm"
                )
            lines.append(f"  Angle:      {e.angle_deg:.2f}°")
            lines.append(f"  Eccentric:  {e.eccentricity:.4f}")
            lines.append(f"  Circular:   {e.circularity:.4f}")
            lines.append(f"  Fit qual:   {e.fit_quality:.4f}")
            lines.append(f"  RMS resid:  {e.fit_rms_residual:.4f}")
            lines.append(f"  Contour:    {e.num_contour_points} pts")
            lines.append(
                f"  Uncert:     ±({e.uncertainty_center_x:.2f},"
                f" {e.uncertainty_center_y:.2f}) px"
            )
            lines.append("")

        if result.limbus.detected and result.limbus.ellipse is not None:
            e = result.limbus.ellipse
            dia_px = e.radius * 2.0
            m_val = (
                result.limbus.method.value
                if hasattr(result.limbus.method, "value")
                else str(result.limbus.method)
            )
            ft = getattr(e, "fit_type", None) or getattr(
                result.limbus, "fit_type", None
            )
            lines.append("=== LIMBUS ===")
            lines.append(f"  Method:     {m_val}")
            lines.append(f"  Fit Type:   {ft or '—'}")
            lines.append(f"  Center:     ({e.center_x:.2f}, {e.center_y:.2f}) px")
            lines.append(f"  Diameter:   {dia_px:.2f} px")
            if has_cal:
                lines.append(f"  Diameter:   {dia_px * mm_per_px:.3f} mm")
            lines.append(f"  Semi-axes:  {e.semi_major:.2f} × {e.semi_minor:.2f} px")
            if has_cal:
                lines.append(
                    f"  Semi-axes:  {e.semi_major * mm_per_px:.3f}"
                    f" × {e.semi_minor * mm_per_px:.3f} mm"
                )
            lines.append(f"  Angle:      {e.angle_deg:.2f}°")
            lines.append(f"  Eccentric:  {e.eccentricity:.4f}")
            lines.append(f"  Circular:   {e.circularity:.4f}")
            lines.append(f"  Fit qual:   {e.fit_quality:.4f}")
            lines.append(f"  RMS resid:  {e.fit_rms_residual:.4f}")
            lines.append(f"  Contour:    {e.num_contour_points} pts")
            lines.append("")

        if result.corneal_center.valid:
            cc = result.corneal_center
            lines.append("=== CORNEAL CENTRE & OFFSET ===")
            lines.append(
                f"  Centre:     ({cc.center_px[0]:.2f}, {cc.center_px[1]:.2f}) px"
            )
            lines.append(
                f"  Offset:     ({cc.offset_px[0]:.2f}, {cc.offset_px[1]:.2f}) px"
            )
            lines.append(f"  Magnitude:  {cc.offset_magnitude_px:.2f} px")
            if cc.offset_magnitude_mm is not None:
                lines.append(
                    f"  Offset:     ({cc.offset_mm[0]:.3f}, {cc.offset_mm[1]:.3f}) mm"
                )
                lines.append(f"  Magnitude:  {cc.offset_magnitude_mm:.3f} mm")
            lines.append(f"  Angle:      {cc.offset_angle_deg:.2f}°")
            lines.append("")

        if has_cal:
            lines.append("=== CALIBRATION ===")
            lines.append(f"  Source:     {cal.source}")
            lines.append(f"  px/mm:      {cal.px_per_mm:.4f}")
            lines.append(f"  mm/px:      {cal.mm_per_px:.6f}")
            lines.append(
                f"  Ref diam:   {cal.reference_diameter_mm:.1f}"
                f" mm = {cal.reference_diameter_px:.0f} px"
            )
            lines.append(f"  Confidence: {cal.confidence:.3f}")
            lines.append("")

        if result.alerts:
            lines.append("=== ALERTS ===")
            for alert in result.alerts:
                lines.append(f"  ! {alert}")
            lines.append("")

        self._details_text.insert("1.0", "\n".join(lines))
        self._details_text.config(state=tk.DISABLED)

    # ================================================================
    # Export
    # ================================================================

    def _export_csv(self, path: Optional[str] = None, show_success: bool = True) -> None:
        if not self._results_history:
            messagebox.showinfo("No Data", "No results to export")
            return
        if path is None:
            path = filedialog.asksaveasfilename(
                title="Export CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
            )
        if not path:
            return
        def _round(v: Any, nd: int = 4) -> Any:
            """Round numeric values; pass through blanks/None as ''."""
            if v is None or v == "":
                return ""
            try:
                fv = float(v)
                if not math.isfinite(fv):
                    return ""
                return round(fv, nd)
            except (TypeError, ValueError):
                return v

        def _diam_mm(ell: Dict[str, Any], mm_px: float) -> Any:
            """Prefer the pre-computed diameter_mm; else derive from mean
            radius. NEVER recompute from semi-major alone (that collapses to
            the calibration constant). This matches the Measurements panel,
            which shows diameter = ellipse.radius * 2 * mm_per_px."""
            if ell.get("diameter_mm") not in (None, ""):
                return _round(ell.get("diameter_mm"))
            r = ell.get("radius")
            if r not in (None, "") and mm_px:
                return _round(float(r) * 2.0 * mm_px)
            return ""

        rows: List[Dict[str, Any]] = []
        for r in self._results_history:
            pupil = r.get("pupil", {})
            limbus = r.get("limbus", {})
            pe = pupil.get("ellipse", {})
            le = limbus.get("ellipse", {})
            cc = r.get("corneal_center", {})
            meta = r.get("metadata", {})
            cal_info = r.get("calibration", {})
            is_cal = bool(cal_info.get("calibrated", False))
            mm_px = float(cal_info.get("mm_per_px", 0) or 0) if is_cal else 0.0
            cal_method = cal_info.get("method", "") or ("anatomical" if is_cal else "")
            assumed_mm = cal_info.get("corneal_diameter_assumed_mm", "")
            if assumed_mm is None or not is_cal or cal_method != "anatomical":
                assumed_mm = ""

            # Diameter in px = mean radius * 2 (matches the panel exactly).
            pupil_dia_px = (pe.get("radius", 0) or 0) * 2 if pe.get("radius") else ""
            limbus_dia_px = (le.get("radius", 0) or 0) * 2 if le.get("radius") else ""

            # Semi-axes in mm: dynamically computed from semi_major/minor px * mm_per_px
            pupil_major_mm = _round(float(pe.get("semi_major", 0)) * mm_px) if (is_cal and pe.get("semi_major")) else ""
            pupil_minor_mm = _round(float(pe.get("semi_minor", 0)) * mm_px) if (is_cal and pe.get("semi_minor")) else ""
            limbus_major_mm = _round(float(le.get("semi_major", 0)) * mm_px) if (is_cal and le.get("semi_major")) else ""
            limbus_minor_mm = _round(float(le.get("semi_minor", 0)) * mm_px) if (is_cal and le.get("semi_minor")) else ""

            rows.append(
                {
                    "frame": meta.get("frame_number", ""),
                    "source": meta.get("source", ""),
                    "processing_time_ms": _round(meta.get("processing_time_ms", ""), 2),
                    "latency_ms": _round(meta.get("latency_ms", ""), 2),
                    # ── Pupil ─────────────────────────────────────────────
                    "pupil_detected": pupil.get("detected", False),
                    "pupil_cx_px": _round(pe.get("center_x", "")),
                    "pupil_cy_px": _round(pe.get("center_y", "")),
                    "pupil_diameter_px": _round(pupil_dia_px),
                    "pupil_diameter_mm": _diam_mm(pe, mm_px) if is_cal else "",
                    "pupil_radius_mm": _round(pupil.get("radius_mm", "")) if is_cal else "",
                    "pupil_semi_major_px": _round(pe.get("semi_major", "")),
                    "pupil_semi_minor_px": _round(pe.get("semi_minor", "")),
                    "pupil_semi_major_mm": pupil_major_mm,
                    "pupil_semi_minor_mm": pupil_minor_mm,
                    "pupil_angle_deg": _round(pe.get("angle_deg", ""), 2),
                    "pupil_fit_type": pupil.get("fit_type", "") or "",
                    "pupil_confidence": _round(pupil.get("confidence", ""), 3),
                    # ── Limbus ────────────────────────────────────────────
                    "limbus_detected": limbus.get("detected", False),
                    "limbus_cx_px": _round(le.get("center_x", "")),
                    "limbus_cy_px": _round(le.get("center_y", "")),
                    "limbus_diameter_px": _round(limbus_dia_px),
                    "limbus_diameter_mm": _diam_mm(le, mm_px) if is_cal else "",
                    "limbus_radius_mm": _round(limbus.get("radius_mm", "")) if is_cal else "",
                    "limbus_semi_major_px": _round(le.get("semi_major", "")),
                    "limbus_semi_minor_px": _round(le.get("semi_minor", "")),
                    "limbus_semi_major_mm": limbus_major_mm,
                    "limbus_semi_minor_mm": limbus_minor_mm,
                    "limbus_angle_deg": _round(le.get("angle_deg", ""), 2),
                    "limbus_fit_type": limbus.get("fit_type", "") or "",
                    "limbus_confidence": _round(limbus.get("confidence", ""), 3),
                    # ── Clinical Corneal WTW Dimensions ───────────────────
                    "measured_wtw_horizontal_mm": _round(limbus.get("wtw_horizontal_mm", "")) if is_cal else "",
                    "measured_wtw_vertical_mm": _round(limbus.get("wtw_vertical_mm", "")) if is_cal else "",
                    "measured_wtw_mean_mm": _round(limbus.get("wtw_mean_mm", "")) if is_cal else "",
                    "limbus_astigmatic_difference_mm": _round(limbus.get("wtw_astigmatism_mm", "")) if is_cal else "",
                    "wtw_validity_status": limbus.get("wtw_validity_status", "") if is_cal else "",
                    # ── Corneal center / offset ───────────────────────────
                    "corneal_center_x_px": _round(
                        (cc.get("center_px") or ["", ""])[0]
                    ),

                    "corneal_center_y_px": _round(
                        (cc.get("center_px") or ["", ""])[1]
                    ),
                    "offset_px": _round(cc.get("offset_magnitude_px", "")),
                    "offset_mm": _round(cc.get("offset_magnitude_mm", "")) if is_cal else "",
                    "offset_angle_deg": _round(cc.get("offset_angle_deg", ""), 2),
                    "corneal_reference_source": r.get("corneal_reference_source", ""),
                    # ── Ring ──────────────────────────────────────────────
                    "ring_status": r.get("ring_status", ""),
                    "ring_center_x": _round(r.get("ring_center_x", "")),
                    "ring_center_y": _round(r.get("ring_center_y", "")),
                    "ring_radius_px": _round(r.get("ring_radius", "")),
                    "ring_diameter_mm": (
                        _round((r.get("ring_radius") or 0) * 2.0 * mm_px)
                        if is_cal and r.get("ring_radius") and mm_px else ""
                    ),
                    "ring_dot_count": r.get("ring_dot_count", ""),
                    # ── Calibration / quality ─────────────────────────────
                    "calibrated": is_cal,
                    "calibration_method": cal_method if is_cal else "",
                    "corneal_diameter_assumed_mm": _round(assumed_mm, 2) if assumed_mm != "" else "",
                    "px_per_mm": _round(cal_info.get("px_per_mm", "")) if is_cal else "",
                    "mm_per_px": _round(cal_info.get("mm_per_px", ""), 6) if is_cal else "",
                    "quality": r.get("overall_quality", ""),
                    "overall_confidence": _round(r.get("overall_confidence", ""), 3),
                    "grayscale_mode": r.get("grayscale_mode", ""),
                }
            )

        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        if show_success:
            self._status_var.set(f"Exported {len(rows)} rows → {path}")

    def _export_txt(self) -> None:
        if not self._results_history:
            messagebox.showinfo("No Data", "No results to export")
            return
        path = filedialog.asksaveasfilename(
            title="Export Results TXT",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
        )
        if not path:
            return
        lines = [
            "# Pupil-Limbus Detection Results",
            "",
            "[export_info]",
            "version=2.3",
            f"total_frames={len(self._results_history)}",
            f"export_timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"corneal_diameter_assumption_mm={_CORNEAL_DIAMETER_MM}",
            "",
            "[results]",
        ]
        for index, result in enumerate(self._results_history, start=1):
            lines.append(f"frame_{index}={result!s}")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._status_var.set(f"Exported {len(self._results_history)} results → {path}")

    def _export_snapshot(self) -> None:
        if self._current_image is None:
            messagebox.showinfo("No Image", "No image to export")
            return
        path = filedialog.asksaveasfilename(
            title="Save Snapshot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")],
        )
        if not path:
            return
        image = self._prepare_recording_frame(
            self._current_image,
            self._current_result,
        )
        cv2.imwrite(path, image)
        self._status_var.set(f"Snapshot saved → {path}")

    # ================================================================
    # Lifecycle
    # ================================================================

    def _on_close(self) -> None:
        self._stop_video()
        # ══════════════════════════════════════════════════════════
        # RECORDING — Cleanup recording on close
        # ══════════════════════════════════════════════════════════
        if self._recorder.is_recording:
            self._stop_recording()
        # ══════════════════════════════════════════════════════════
        if self._fast_engine is not None:
            try:
                import torch

                if (
                    hasattr(self._fast_engine, "model")
                    and self._fast_engine.model is not None
                ):
                    del self._fast_engine.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self._fast_engine = None
        if _FAST_PIPELINE_AVAILABLE and FastInference is not None:
            try:
                FastInference.reset_cache()
            except Exception:
                pass
        try:
            self.logger.close()
        except Exception:
            pass
        self.root.destroy()


def launch_gui() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        import os
        import sys
        # gui_app.py is in pupil_tracking/interface/, so base is 3 levels up
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        icon_path = os.path.join(base_dir, "build", "assets", "icon.ico")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except Exception:
        pass
        
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    colors = DarkTheme.apply(root)
    _app = PupilTrackingGUI(root, colors=colors)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
