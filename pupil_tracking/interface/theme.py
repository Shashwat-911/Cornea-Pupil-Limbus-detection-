"""
Professional dark theme for the surgical-grade pupil tracking GUI.

Provides a complete dark colour scheme inspired by VSCode, Linear, and
Figma dark modes — adapted for clinical readability and surgical-tool
aesthetics.

Usage
-----
>>> from pupil_tracking.interface.theme import DarkTheme
>>> colors = DarkTheme.apply(root)
>>> label = ttk.Label(root, text="Hello", style="Header.TLabel")
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class Colors:
    """Central colour palette — every UI colour lives here."""

    # ── Backgrounds ──────────────────────────────────────────────
    BG_PRIMARY = "#121212"
    BG_SECONDARY = "#1D1D1F"
    BG_TERTIARY = "#1D1D1F"
    BG_INPUT = "#121212"
    CANVAS_BG = "#121212"

    # ── Borders ──────────────────────────────────────────────────
    BORDER = "#323232"
    BORDER_FOCUS = "#0071E3"

    # ── Foregrounds ──────────────────────────────────────────────
    FG_PRIMARY = "#e4e4e8"
    FG_SECONDARY = "#9898a6"
    FG_TERTIARY = "#606070"

    # ── Accent ───────────────────────────────────────────────────
    ACCENT = "#0071E3"
    ACCENT_HOVER = "#3395FF"
    ACCENT_DIM = "#003A75"

    # ── Quality grades ───────────────────────────────────────────
    SURGICAL = "#00e676"
    CLINICAL = "#29b6f6"
    RESEARCH = "#ffa726"
    INSUFFICIENT = "#ef5350"
    NO_DETECTION = "#616161"

    # ── Measurement section headers ──────────────────────────────
    PUPIL = "#00e676"
    LIMBUS = "#5c8aff"
    OFFSET = "#00e5cc"
    CALIBRATION = "#ffb74d"
    PROCESSING = "#9e9e9e"

    # ── Buttons ──────────────────────────────────────────────────
    BTN_PRIMARY = "#FFFFFF"
    BTN_SECONDARY = "#323232"
    BTN_ACTION = "#0071E3"
    BTN_BG = "#323232"
    BTN_HOVER = "#3D3D3D"
    BTN_ACTIVE = "#1D1D1F"

    # ── Scrollbar ────────────────────────────────────────────────
    SCROLLBAR = "#1D1D1F"
    SCROLLBAR_THUMB = "#323232"

    # ── Quality colour map (for badge / labels) ──────────────────
    QUALITY_MAP = {
        "SURGICAL": "#00e676",
        "CLINICAL": "#29b6f6",
        "RESEARCH": "#ffa726",
        "INSUFFICIENT": "#ef5350",
        "NO_DETECTION": "#616161",
    }


class DarkTheme:
    """Apply the dark theme to a Tk root window."""

    @staticmethod
    def apply(root: tk.Tk) -> Colors:
        """Configure all styles and return the colour palette.

        This replaces the default ttk theme setup. Call once at
        application startup, before constructing any widgets.
        """
        colors = Colors

        # ── Root window ──────────────────────────────────────────
        root.configure(bg=colors.BG_PRIMARY)
        root.option_add("*background", colors.BG_PRIMARY)
        root.option_add("*foreground", colors.FG_PRIMARY)
        root.option_add("*highlightBackground", colors.BG_PRIMARY)
        root.option_add("*selectBackground", colors.ACCENT_DIM)
        root.option_add("*selectForeground", colors.FG_PRIMARY)

        # ── Set base ttk theme ───────────────────────────────────
        style = ttk.Style(root)
        available = style.theme_names()
        for preferred in ("clam", "alt", "default"):
            if preferred in available:
                style.theme_use(preferred)
                break

        # ── TFrame ───────────────────────────────────────────────
        style.configure("TFrame", background=colors.BG_SECONDARY)
        style.configure(
            "Primary.TFrame", background=colors.BG_PRIMARY,
        )
        style.configure(
            "Card.TFrame", background=colors.BG_TERTIARY,
        )
        style.configure(
            "MetricCard.TFrame", background=colors.BG_TERTIARY,
        )

        # ── TLabel ───────────────────────────────────────────────
        style.configure(
            "TLabel",
            background=colors.BG_SECONDARY,
            foreground=colors.FG_PRIMARY,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Muted.TLabel",
            background=colors.BG_SECONDARY,
            foreground=colors.FG_SECONDARY,
            font=("Consolas", 9),
        )
        style.configure(
            "Tiny.TLabel",
            background=colors.BG_SECONDARY,
            foreground=colors.FG_TERTIARY,
            font=("Consolas", 8),
        )
        # Section headers
        style.configure(
            "PupilHeader.TLabel",
            background=colors.BG_SECONDARY,
            foreground=colors.PUPIL,
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "LimbusHeader.TLabel",
            background=colors.BG_SECONDARY,
            foreground=colors.LIMBUS,
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "OffsetHeader.TLabel",
            background=colors.BG_SECONDARY,
            foreground=colors.OFFSET,
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "CalibHeader.TLabel",
            background=colors.BG_SECONDARY,
            foreground=colors.CALIBRATION,
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "ProcHeader.TLabel",
            background=colors.BG_SECONDARY,
            foreground=colors.PROCESSING,
            font=("Segoe UI", 11, "bold"),
        )
        # Value labels in measurements
        style.configure(
            "Value.TLabel",
            background=colors.BG_SECONDARY,
            foreground=colors.FG_PRIMARY,
            font=("Consolas", 10),
        )
        style.configure(
            "ValueKey.TLabel",
            background=colors.BG_SECONDARY,
            foreground=colors.FG_SECONDARY,
            font=("Consolas", 10),
        )
        style.configure(
            "CardKey.TLabel",
            background=colors.BG_TERTIARY,
            foreground=colors.FG_SECONDARY,
            font=("Segoe UI", 9, "bold"),
        )
        style.configure(
            "CardValue.TLabel",
            background=colors.BG_TERTIARY,
            foreground=colors.FG_PRIMARY,
            font=("Consolas", 12, "bold"),
        )
        style.configure(
            "CardValueSmall.TLabel",
            background=colors.BG_TERTIARY,
            foreground=colors.FG_PRIMARY,
            font=("Consolas", 9),
        )
        # Quality badge
        style.configure(
            "Quality.TLabel",
            background=colors.BG_PRIMARY,
            foreground=colors.FG_PRIMARY,
            font=("Consolas", 11, "bold"),
        )
        # Status bar
        style.configure(
            "Status.TLabel",
            background=colors.BG_PRIMARY,
            foreground=colors.FG_SECONDARY,
            font=("Consolas", 9),
        )

        # ── TButton ─────────────────────────────────────────────
        style.configure(
            "TButton",
            background=colors.BTN_SECONDARY,
            foreground=colors.FG_PRIMARY,
            bordercolor=colors.BORDER,
            focuscolor=colors.ACCENT,
            font=("Segoe UI", 10),
            padding=(10, 4),
        )
        style.map(
            "TButton",
            background=[
                ("active", colors.BTN_ACTIVE),
                ("!disabled", colors.BTN_SECONDARY),
            ],
            foreground=[
                ("disabled", colors.FG_TERTIARY),
            ],
        )
        # Primary button (white, high contrast)
        style.configure(
            "Primary.TButton",
            background=colors.BTN_PRIMARY,
            foreground=colors.BG_PRIMARY,
            bordercolor=colors.BTN_PRIMARY,
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#E0E0E0")],
            foreground=[("active", colors.BG_PRIMARY)],
        )
        # Action/Accent button (blue)
        style.configure(
            "Accent.TButton",
            background=colors.BTN_ACTION,
            foreground=colors.BTN_PRIMARY,
            bordercolor=colors.BTN_ACTION,
        )
        style.map(
            "Accent.TButton",
            background=[("active", colors.ACCENT_HOVER)],
            foreground=[("active", colors.BTN_PRIMARY)],
        )


        # Toolbar active (selected) button — blue highlight
        style.configure(
            "ToolbarActive.TButton",
            background=colors.BTN_ACTION,
            foreground=colors.BTN_PRIMARY,
            bordercolor=colors.BTN_ACTION,
            font=("Segoe UI", 10, "bold"),
            padding=(10, 4),
        )
        style.map(
            "ToolbarActive.TButton",
            background=[
                ("active",    colors.ACCENT_HOVER),
                ("!disabled", colors.BTN_ACTION),
            ],
            foreground=[
                ("active",    colors.BTN_PRIMARY),
                ("!disabled", colors.BTN_PRIMARY),
            ],
        )
        # ROI/Ring interaction buttons — green while active/editing
        style.configure(
            "ROIActive.TButton",
            background=colors.SURGICAL,
            foreground=colors.BG_PRIMARY,
            bordercolor=colors.SURGICAL,
            font=("Segoe UI", 10, "bold"),
            padding=(10, 4),
        )
        style.map(
            "ROIActive.TButton",
            background=[("active", colors.SURGICAL), ("!disabled", colors.SURGICAL)],
            foreground=[("active", colors.BG_PRIMARY), ("!disabled", colors.BG_PRIMARY)],
        )
        style.configure(
            "RingActive.TButton",
            background=colors.SURGICAL,
            foreground=colors.BG_PRIMARY,
            bordercolor=colors.SURGICAL,
            font=("Segoe UI", 10, "bold"),
            padding=(10, 4),
        )
        style.map(
            "RingActive.TButton",
            background=[("active", colors.SURGICAL), ("!disabled", colors.SURGICAL)],
            foreground=[("active", colors.BG_PRIMARY), ("!disabled", colors.BG_PRIMARY)],
        )
        # ── TNotebook ───────────────────────────────────────────
        style.configure(
            "TNotebook",
            background=colors.BG_PRIMARY,
            bordercolor=colors.BORDER,
            tabmargins=(2, 4, 2, 0),
        )
        style.configure(
            "TNotebook.Tab",
            background=colors.BG_TERTIARY,
            foreground=colors.FG_SECONDARY,
            padding=(14, 6),
            font=("Segoe UI", 10),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", colors.ACCENT_DIM)],
            foreground=[("selected", colors.FG_PRIMARY)],
        )

        # ── TLabelframe ─────────────────────────────────────────
        style.configure(
            "TLabelframe",
            background=colors.BG_SECONDARY,
            foreground=colors.ACCENT,
            bordercolor=colors.BORDER,
            relief="flat",
        )
        style.configure(
            "TLabelframe.Label",
            background=colors.BG_SECONDARY,
            foreground=colors.ACCENT,
            font=("Segoe UI", 10, "bold"),
        )

        # ── TCheckbutton ────────────────────────────────────────
        style.configure(
            "TCheckbutton",
            background=colors.BG_SECONDARY,
            foreground=colors.FG_PRIMARY,
            indicatorcolor=colors.BG_INPUT,
            font=("Segoe UI", 10),
        )
        style.map(
            "TCheckbutton",
            indicatorcolor=[("selected", colors.ACCENT)],
            background=[("active", colors.BG_TERTIARY)],
        )

        # ── TScale ──────────────────────────────────────────────
        style.configure(
            "TScale",
            background=colors.BG_SECONDARY,
            troughcolor=colors.BG_INPUT,
            bordercolor=colors.BORDER,
        )
        style.configure(
            "Horizontal.TScale",
            background=colors.BG_SECONDARY,
            troughcolor=colors.BG_INPUT,
        )

        # ── TSpinbox ────────────────────────────────────────────
        style.configure(
            "TSpinbox",
            fieldbackground=colors.BG_INPUT,
            foreground=colors.FG_PRIMARY,
            bordercolor=colors.BORDER,
            arrowcolor=colors.FG_SECONDARY,
            background=colors.BG_TERTIARY,
        )
        style.map(
            "TSpinbox",
            fieldbackground=[("focus", colors.BG_INPUT)],
            bordercolor=[("focus", colors.BORDER_FOCUS)],
        )

        # ── TScrollbar ──────────────────────────────────────────
        style.configure(
            "TScrollbar",
            troughcolor=colors.SCROLLBAR,
            background=colors.SCROLLBAR_THUMB,
            bordercolor=colors.SCROLLBAR,
            arrowcolor=colors.FG_TERTIARY,
        )
        style.map(
            "TScrollbar",
            background=[("active", colors.FG_TERTIARY)],
        )
        style.configure(
            "Vertical.TScrollbar",
            troughcolor=colors.SCROLLBAR,
            background=colors.SCROLLBAR_THUMB,
        )

        # ── TProgressbar ────────────────────────────────────────
        style.configure(
            "TProgressbar",
            troughcolor=colors.BG_INPUT,
            background=colors.ACCENT,
            bordercolor=colors.BORDER,
        )
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=colors.BG_INPUT,
            background=colors.ACCENT,
        )

        # ── TSeparator ──────────────────────────────────────────
        style.configure(
            "TSeparator",
            background=colors.BORDER,
        )

        # ── TPanedwindow ────────────────────────────────────────
        style.configure(
            "TPanedwindow",
            background=colors.BG_PRIMARY,
        )
        style.configure(
            "Sash",
            sashthickness=4,
            handlesize=8,
        )

        return colors
