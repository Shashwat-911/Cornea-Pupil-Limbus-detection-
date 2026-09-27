#!/usr/bin/env python3
"""
Live Video Annotation Tool for Pupil/Limbus Tracking.

Controls:
    SPACE  - Pause/resume video and enter annotation mode
    P      - Switch to pupil annotation mode (circular bias)
    L      - Switch to limbus annotation mode (elliptical)
    ENTER  - Confirm current annotation
    R      - Refine current fit using image edges (manual only)
    U      - Undo last point
    C      - Clear current annotation
    T      - Trigger incremental retrain
    S      - Save all annotations to disk
    N      - Next frame (when paused)
    B      - Previous frame (when paused)
    F      - Toggle fit-to-screen / original size
    +/-    - Zoom in / out
    G      - Toggle edge-snap mode (auto-snap clicks to nearest edge)
    D      - Toggle between ellipse / circle constraint
    ESC/Q  - Quit
"""

import cv2
import json
import numpy as np
import os
import time
import sys
import shutil
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple
from enum import Enum, auto
from datetime import datetime


# ─── Custom JSON encoder that handles numpy types ───────────────────────────

class NumpySafeEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy types to native Python types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _to_python_float(val) -> float:
    """Safely convert any numeric value to native Python float."""
    if isinstance(val, (np.floating, np.integer)):
        return float(val)
    return float(val)


def _to_python_int(val) -> int:
    """Safely convert any numeric value to native Python int."""
    if isinstance(val, (np.floating, np.integer)):
        return int(val)
    return int(val)


# ─── Data Models ────────────────────────────────────────────────────────────

class AnnotationMode(Enum):
    IDLE = auto()
    PUPIL = auto()
    LIMBUS = auto()


class FitConstraint(Enum):
    ELLIPSE = auto()
    CIRCLE = auto()


@dataclass
class EllipseParams:
    cx: float
    cy: float
    semi_major: float
    semi_minor: float
    angle_deg: float

    def __post_init__(self):
        """Ensure all values are native Python floats, never numpy types."""
        self.cx = float(self.cx)
        self.cy = float(self.cy)
        self.semi_major = float(self.semi_major)
        self.semi_minor = float(self.semi_minor)
        self.angle_deg = float(self.angle_deg)
        # Ensure semi_major >= semi_minor
        if self.semi_major < self.semi_minor:
            self.semi_major, self.semi_minor = self.semi_minor, self.semi_major

    @property
    def aspect_ratio(self) -> float:
        return self.semi_minor / max(self.semi_major, 1e-6)

    @property
    def area(self) -> float:
        return math.pi * self.semi_major * self.semi_minor

    @property
    def perimeter_approx(self) -> float:
        a, b = self.semi_major, self.semi_minor
        h = ((a - b) ** 2) / ((a + b) ** 2 + 1e-12)
        return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))

    def point_on_boundary(self, theta_rad: float) -> Tuple[float, float]:
        cos_a = math.cos(math.radians(self.angle_deg))
        sin_a = math.sin(math.radians(self.angle_deg))
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        x = self.cx + self.semi_major * cos_t * cos_a - self.semi_minor * sin_t * sin_a
        y = self.cy + self.semi_major * cos_t * sin_a + self.semi_minor * sin_t * cos_a
        return float(x), float(y)

    def sample_boundary(self, n_points: int = 72) -> List[Tuple[float, float]]:
        return [
            self.point_on_boundary(2.0 * math.pi * i / n_points)
            for i in range(n_points)
        ]

    def distance_to_boundary_sampled(self, px: float, py: float,
                                     n_samples: int = 72) -> float:
        """Distance from point to nearest sampled boundary point."""
        best = float('inf')
        for t in range(n_samples):
            theta = 2.0 * math.pi * t / n_samples
            bx, by = self.point_on_boundary(theta)
            d = math.hypot(px - bx, py - by)
            if d < best:
                best = d
        return best

    def to_cv2(self) -> Tuple:
        return (
            (self.cx, self.cy),
            (self.semi_major * 2, self.semi_minor * 2),
            self.angle_deg,
        )

    def to_cv2_int(self) -> Tuple:
        return (
            (int(round(self.cx)), int(round(self.cy))),
            (int(round(self.semi_major * 2)), int(round(self.semi_minor * 2))),
            self.angle_deg,
        )

    def to_cv2_scaled(self, scale: float) -> Tuple:
        return (
            (int(round(self.cx * scale)), int(round(self.cy * scale))),
            (int(round(self.semi_major * 2 * scale)),
             int(round(self.semi_minor * 2 * scale))),
            self.angle_deg,
        )

    @classmethod
    def from_cv2(cls, ellipse_tuple) -> "EllipseParams":
        center, axes, angle = ellipse_tuple
        a_half = float(axes[0]) / 2.0
        b_half = float(axes[1]) / 2.0
        semi_major = max(a_half, b_half)
        semi_minor = min(a_half, b_half)
        angle = float(angle)
        if b_half > a_half:
            angle = (angle + 90.0) % 180.0
        return cls(
            cx=round(float(center[0]), 2),
            cy=round(float(center[1]), 2),
            semi_major=round(semi_major, 2),
            semi_minor=round(semi_minor, 2),
            angle_deg=round(angle % 180.0, 2),
        )

    @classmethod
    def from_circle(cls, cx, cy, radius) -> "EllipseParams":
        return cls(
            cx=round(float(cx), 2),
            cy=round(float(cy), 2),
            semi_major=round(float(radius), 2),
            semi_minor=round(float(radius), 2),
            angle_deg=0.0,
        )

    def to_dict(self) -> dict:
        return {
            "cx": float(self.cx),
            "cy": float(self.cy),
            "semi_major": float(self.semi_major),
            "semi_minor": float(self.semi_minor),
            "angle_deg": float(self.angle_deg),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EllipseParams":
        return cls(
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            semi_major=float(d["semi_major"]),
            semi_minor=float(d["semi_minor"]),
            angle_deg=float(d["angle_deg"]),
        )


@dataclass
class FitResult:
    params: EllipseParams
    method: str
    residual: float
    edge_score: float = 0.0
    combined: float = 0.0


@dataclass
class FrameAnnotation:
    pupil: Optional[EllipseParams] = None
    limbus: Optional[EllipseParams] = None
    timestamp_sec: float = 0.0
    frame_index: int = 0
    annotated_at: str = ""

    def to_dict(self) -> dict:
        d = {}
        if self.pupil:
            d["pupil"] = self.pupil.to_dict()
        if self.limbus:
            d["limbus"] = self.limbus.to_dict()
        d["timestamp_sec"] = float(self.timestamp_sec)
        d["frame_index"] = int(self.frame_index)
        d["annotated_at"] = str(self.annotated_at)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FrameAnnotation":
        pupil = EllipseParams.from_dict(d["pupil"]) if "pupil" in d else None
        limbus = EllipseParams.from_dict(d["limbus"]) if "limbus" in d else None
        return cls(
            pupil=pupil,
            limbus=limbus,
            timestamp_sec=float(d.get("timestamp_sec", 0)),
            frame_index=int(d.get("frame_index", 0)),
            annotated_at=str(d.get("annotated_at", "")),
        )


# ─── Annotation Store ───────────────────────────────────────────────────────

class AnnotationStore:
    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.annotations: Dict[str, FrameAnnotation] = {}
        self._dirty = False
        self._load()

    def _load(self):
        if self.save_path.exists():
            try:
                with open(self.save_path, "r") as f:
                    raw = json.load(f)
                for key, val in raw.items():
                    self.annotations[key] = FrameAnnotation.from_dict(val)
                print(
                    f"[AnnotationStore] Loaded {len(self.annotations)} "
                    f"annotations from {self.save_path}"
                )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[AnnotationStore] Warning: {e}")

    def add(self, frame_name: str, annotation: FrameAnnotation):
        self.annotations[frame_name] = annotation
        self._dirty = True

    def save(self):
        if not self._dirty and self.save_path.exists():
            return
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        out = {k: v.to_dict() for k, v in self.annotations.items()}

        # Write to temp file first
        tmp = self.save_path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(out, f, indent=2, cls=NumpySafeEncoder)
        except Exception as e:
            print(f"[AnnotationStore] Error writing tmp: {e}")
            # Try direct write as fallback
            try:
                with open(self.save_path, "w") as f:
                    json.dump(out, f, indent=2, cls=NumpySafeEncoder)
                self._dirty = False
                print(f"[AnnotationStore] Saved {len(self.annotations)} "
                      f"annotations (direct)")
            except Exception as e2:
                print(f"[AnnotationStore] FAILED to save: {e2}")
            return

        # Atomic replace (works on Windows and Linux)
        try:
            os.replace(str(tmp), str(self.save_path))
        except OSError:
            try:
                # Windows fallback: delete target then rename
                if self.save_path.exists():
                    bak = self.save_path.with_suffix(".bak")
                    try:
                        if bak.exists():
                            bak.unlink()
                        shutil.move(str(self.save_path), str(bak))
                    except OSError:
                        self.save_path.unlink()
                shutil.move(str(tmp), str(self.save_path))
            except OSError:
                try:
                    shutil.copy2(str(tmp), str(self.save_path))
                    tmp.unlink()
                except OSError as final_e:
                    print(f"[AnnotationStore] Save fallback error: {final_e}")

        self._dirty = False
        print(f"[AnnotationStore] Saved {len(self.annotations)} annotations")

    def __len__(self):
        return len(self.annotations)

    def __contains__(self, key):
        return key in self.annotations


# ─── Edge Detector ───────────────────────────────────────────────────────────

class EdgeProcessor:
    def __init__(self):
        self._cache_frame_id = -1
        self._edges = None
        self._grad_mag = None
        self._grad_x = None
        self._grad_y = None
        self._gray = None

    def process(self, frame: np.ndarray, frame_id: int = 0):
        if self._cache_frame_id == frame_id and self._edges is not None:
            return
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        self._gray = gray

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.2)

        self._grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        self._grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        self._grad_mag = np.sqrt(self._grad_x ** 2 + self._grad_y ** 2)

        med = float(np.median(blurred))
        lo = int(max(0, 0.5 * med))
        hi = int(min(255, 1.5 * med))
        edges1 = cv2.Canny(blurred, lo, hi)
        edges2 = cv2.Canny(blurred, max(1, lo // 2), max(2, hi // 2))
        self._edges = cv2.bitwise_or(edges1, edges2)
        self._cache_frame_id = frame_id

    @property
    def edges(self) -> Optional[np.ndarray]:
        return self._edges

    @property
    def gradient_magnitude(self) -> Optional[np.ndarray]:
        return self._grad_mag

    def snap_to_edge(self, x: int, y: int,
                     search_radius: int = 15) -> Tuple[int, int]:
        if self._edges is None or self._grad_mag is None:
            return x, y

        h, w = self._edges.shape[:2]
        x0 = max(0, x - search_radius)
        y0 = max(0, y - search_radius)
        x1 = min(w, x + search_radius + 1)
        y1 = min(h, y + search_radius + 1)

        roi_edges = self._edges[y0:y1, x0:x1]
        roi_grad = self._grad_mag[y0:y1, x0:x1]

        edge_ys, edge_xs = np.where(roi_edges > 0)
        if len(edge_xs) == 0:
            return self._snap_to_gradient(x, y, search_radius)

        abs_xs = edge_xs.astype(np.float64) + x0
        abs_ys = edge_ys.astype(np.float64) + y0
        dists = np.sqrt((abs_xs - x) ** 2 + (abs_ys - y) ** 2)
        grads = roi_grad[edge_ys, edge_xs]

        max_g = float(grads.max()) if grads.max() > 0 else 1.0
        max_d = float(search_radius)

        scores = (grads / max_g) * 0.5 + \
                 (1.0 - np.minimum(dists / max_d, 1.0)) * 0.5

        best = int(np.argmax(scores))
        return int(abs_xs[best]), int(abs_ys[best])

    def _snap_to_gradient(self, x: int, y: int,
                          radius: int) -> Tuple[int, int]:
        if self._grad_mag is None:
            return x, y

        h, w = self._grad_mag.shape[:2]
        x0 = max(0, x - radius)
        y0 = max(0, y - radius)
        x1 = min(w, x + radius + 1)
        y1 = min(h, y + radius + 1)

        roi = self._grad_mag[y0:y1, x0:x1]
        if roi.size == 0:
            return x, y

        yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
        cx_local = x - x0
        cy_local = y - y0
        dist = np.sqrt((xx - cx_local) ** 2 + (yy - cy_local) ** 2)
        dist_weight = np.exp(-dist / (radius * 0.5))
        weighted = roi * dist_weight

        peak = np.unravel_index(int(np.argmax(weighted)), weighted.shape)
        return int(peak[1] + x0), int(peak[0] + y0)

    def edge_score_along_ellipse(self, params: EllipseParams,
                                 n_samples: int = 90) -> float:
        if self._grad_mag is None:
            return 0.0

        h, w = self._grad_mag.shape[:2]
        boundary = params.sample_boundary(n_samples)
        total = 0.0
        valid = 0

        for bx, by in boundary:
            ix, iy = int(round(bx)), int(round(by))
            if 1 <= ix < w - 1 and 1 <= iy < h - 1:
                patch = self._grad_mag[iy - 1:iy + 2, ix - 1:ix + 2]
                total += float(patch.max())
                valid += 1

        if valid == 0:
            return 0.0

        avg_grad = total / valid
        ref = float(np.percentile(self._grad_mag, 95))
        if ref < 1e-6:
            return 0.0

        return min(1.0, avg_grad / (ref * 0.5))

    def sample_edge_points_near_ellipse(
        self, params: EllipseParams,
        band_width: int = 8,
        n_angular: int = 72
    ) -> List[Tuple[int, int]]:
        if self._edges is None or self._grad_mag is None:
            return []

        h, w = self._edges.shape[:2]
        edge_pts = []
        boundary = params.sample_boundary(n_angular)

        for bx, by in boundary:
            best_pt = None
            best_score = 0.0

            dx = bx - params.cx
            dy = by - params.cy
            length = math.hypot(dx, dy)
            if length < 1:
                continue
            nx = dx / length
            ny = dy / length

            for r in range(-band_width, band_width + 1):
                px = bx + r * nx
                py = by + r * ny
                ix, iy = int(round(px)), int(round(py))

                if 0 <= ix < w and 0 <= iy < h:
                    if self._edges[iy, ix] > 0:
                        g = float(self._grad_mag[iy, ix])
                        dist_penalty = abs(r) * 0.1
                        score = g - dist_penalty
                        if score > best_score:
                            best_score = score
                            best_pt = (ix, iy)

            if best_pt is not None:
                edge_pts.append(best_pt)

        return edge_pts


# ─── Ellipse Fitter ─────────────────────────────────────────────────────────

class EllipseFitter:
    MIN_POINTS = 5

    @staticmethod
    def fit(points: List[Tuple[int, int]],
            constraint: FitConstraint = FitConstraint.ELLIPSE,
            image: Optional[np.ndarray] = None,
            edge_proc: Optional[EdgeProcessor] = None
            ) -> Optional[EllipseParams]:
        """Fit ellipse/circle to user-clicked points.

        The fit is ranked purely by how close each candidate passes
        through the clicked points (boundary residual).  Edge scores
        are computed for display feedback only and never influence
        which candidate is chosen.
        """
        if len(points) < EllipseFitter.MIN_POINTS:
            return None

        pts_arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        candidates: List[FitResult] = []

        # Method 1: Standard fitEllipse
        try:
            e = cv2.fitEllipse(pts_arr)
            p = EllipseParams.from_cv2(e)
            if EllipseFitter._validate(p, points):
                res = EllipseFitter._boundary_residual(p, points)
                candidates.append(FitResult(p, "standard", res))
        except cv2.error:
            pass

        # Method 2: fitEllipseDirect
        if len(points) >= 5:
            try:
                e = cv2.fitEllipseDirect(pts_arr)
                p = EllipseParams.from_cv2(e)
                if EllipseFitter._validate(p, points):
                    res = EllipseFitter._boundary_residual(p, points)
                    candidates.append(FitResult(p, "direct", res))
            except (cv2.error, Exception):
                pass

        # Method 3: fitEllipseAMS
        if len(points) >= 5:
            try:
                e = cv2.fitEllipseAMS(pts_arr)
                p = EllipseParams.from_cv2(e)
                if EllipseFitter._validate(p, points):
                    res = EllipseFitter._boundary_residual(p, points)
                    candidates.append(FitResult(p, "ams", res))
            except (cv2.error, Exception):
                pass

        # Method 4: Kasa circle
        try:
            p = EllipseFitter._fit_circle_kasa(points)
            if p and EllipseFitter._validate(p, points):
                res = EllipseFitter._boundary_residual(p, points)
                candidates.append(FitResult(p, "circle_kasa", res))
        except Exception:
            pass

        # Method 5: Pratt circle
        try:
            p = EllipseFitter._fit_circle_pratt(points)
            if p and EllipseFitter._validate(p, points):
                res = EllipseFitter._boundary_residual(p, points)
                candidates.append(FitResult(p, "circle_pratt", res))
        except Exception:
            pass

        # Method 6: Huber robust circle
        try:
            p = EllipseFitter._fit_circle_huber(points)
            if p and EllipseFitter._validate(p, points):
                res = EllipseFitter._boundary_residual(p, points)
                candidates.append(FitResult(p, "circle_huber", res))
        except Exception:
            pass

        # Method 7: RANSAC ellipse
        if len(points) >= 7:
            try:
                p = EllipseFitter._fit_ransac(points)
                if p and EllipseFitter._validate(p, points):
                    res = EllipseFitter._boundary_residual(p, points)
                    candidates.append(FitResult(p, "ransac", res))
            except Exception:
                pass

        # Method 8: MinEnclosingCircle-seeded
        try:
            p = EllipseFitter._fit_enclosing_circle(points)
            if p and EllipseFitter._validate(p, points):
                res = EllipseFitter._boundary_residual(p, points)
                candidates.append(FitResult(p, "enclosing", res))
        except Exception:
            pass

        if not candidates:
            return None

        # ── Edge score: informational only, never used for ranking ──
        if edge_proc is not None and image is not None:
            edge_proc.process(image)
            for c in candidates:
                c.edge_score = edge_proc.edge_score_along_ellipse(c.params)

        # ── Scoring: YOUR POINTS are the sole authority ─────────────
        for c in candidates:
            # Combined score = residual to your clicked points
            c.combined = c.residual

            # Mild shape preference only as a tiebreaker
            if constraint == FitConstraint.CIRCLE:
                ar = c.params.aspect_ratio
                # Gently penalize non-circular fits
                c.combined *= (1.0 + (1.0 - ar) * 0.1)
            # ELLIPSE constraint: no bias at all — fit your points

        candidates.sort(key=lambda c: c.combined)
        best = candidates[0]

        print(f"  [FIT] Best: {best.method} "
              f"residual={best.residual:.2f} "
              f"edge={best.edge_score:.2f} "
              f"({best.params.semi_major:.1f}x{best.params.semi_minor:.1f} "
              f"ar={best.params.aspect_ratio:.2f})")

        return best.params

    @staticmethod
    def refine_with_edges(params: EllipseParams,
                          image: np.ndarray,
                          edge_proc: EdgeProcessor,
                          iterations: int = 3) -> Optional[EllipseParams]:
        """Edge-based refinement — only called manually via R key."""
        edge_proc.process(image)
        current = params
        original_score = edge_proc.edge_score_along_ellipse(params)

        for iteration in range(iterations):
            band = max(4, int(
                min(current.semi_major, current.semi_minor) * 0.12
            ))
            n_ang = max(36, min(120, int(current.perimeter_approx / 4)))

            edge_pts = edge_proc.sample_edge_points_near_ellipse(
                current, band_width=band, n_angular=n_ang
            )

            if len(edge_pts) < 8:
                break

            pts_arr = np.array(
                edge_pts, dtype=np.float32
            ).reshape(-1, 1, 2)

            best_p = None
            best_score = -1.0

            for fit_fn in [cv2.fitEllipse, cv2.fitEllipseDirect,
                           cv2.fitEllipseAMS]:
                try:
                    e = fit_fn(pts_arr)
                    p = EllipseParams.from_cv2(e)
                    if not EllipseFitter._validate(p, edge_pts):
                        continue

                    drift = math.hypot(
                        p.cx - params.cx, p.cy - params.cy
                    )
                    max_drift = min(
                        params.semi_major, params.semi_minor
                    ) * 0.4
                    if drift > max_drift:
                        continue

                    size_ratio = p.semi_major / max(params.semi_major, 1)
                    if not (0.6 < size_ratio < 1.6):
                        continue

                    score = edge_proc.edge_score_along_ellipse(p)
                    if score > best_score:
                        best_score = score
                        best_p = p
                except cv2.error:
                    pass

            if best_p is not None:
                cur_score = edge_proc.edge_score_along_ellipse(current)
                if best_score > cur_score * 0.9:
                    current = best_p
                else:
                    break
            else:
                break

        final_score = edge_proc.edge_score_along_ellipse(current)
        if final_score >= original_score * 0.85:
            print(f"  [REFINE] edge: "
                  f"{original_score:.2f} -> {final_score:.2f}")
            return current
        else:
            print(f"  [REFINE] rejected "
                  f"({original_score:.2f} -> {final_score:.2f})")
            return params

    @staticmethod
    def _validate(p: EllipseParams,
                  points: List[Tuple[int, int]] = None) -> bool:
        if p.semi_major < 2 or p.semi_minor < 2:
            return False
        if p.semi_major > 5000 or p.semi_minor > 5000:
            return False
        ratio = p.semi_major / max(p.semi_minor, 0.1)
        if ratio > 5:
            return False
        if math.isnan(p.cx) or math.isnan(p.cy):
            return False
        if math.isnan(p.semi_major) or math.isnan(p.semi_minor):
            return False
        if math.isinf(p.cx) or math.isinf(p.cy):
            return False

        if points and len(points) >= 3:
            pts = np.array(points, dtype=np.float64)
            centroid = pts.mean(axis=0)
            spread = max(float(pts.std(axis=0).max()) * 2, 10)
            dist = math.hypot(
                p.cx - float(centroid[0]),
                p.cy - float(centroid[1])
            )
            if dist > spread * 4:
                return False
            max_span = float(np.ptp(pts, axis=0).max())
            if p.semi_major > max_span * 3:
                return False
            if p.semi_major < max_span * 0.1:
                return False

        return True

    @staticmethod
    def _boundary_residual(p: EllipseParams,
                           points: List[Tuple[int, int]]) -> float:
        if not points:
            return float('inf')
        total = 0.0
        for px, py in points:
            d = p.distance_to_boundary_sampled(
                float(px), float(py), n_samples=72
            )
            total += d * d
        return math.sqrt(total / len(points))

    @staticmethod
    def _fit_circle_kasa(
        points: List[Tuple[int, int]]
    ) -> Optional[EllipseParams]:
        n = len(points)
        if n < 3:
            return None
        pts = np.array(points, dtype=np.float64)
        x, y = pts[:, 0], pts[:, 1]
        A = np.column_stack([2 * x, 2 * y, np.ones(n)])
        b = x ** 2 + y ** 2
        try:
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        cx = float(result[0])
        cy = float(result[1])
        r_sq = float(result[2]) + cx ** 2 + cy ** 2
        if r_sq <= 0:
            return None
        r = math.sqrt(r_sq)
        if r < 2:
            return None
        return EllipseParams.from_circle(cx, cy, r)

    @staticmethod
    def _fit_circle_pratt(
        points: List[Tuple[int, int]]
    ) -> Optional[EllipseParams]:
        n = len(points)
        if n < 3:
            return None
        pts = np.array(points, dtype=np.float64)
        x, y = pts[:, 0], pts[:, 1]
        mx, my = float(x.mean()), float(y.mean())
        u, v = x - mx, y - my

        Suu = float((u * u).sum())
        Svv = float((v * v).sum())
        Suv = float((u * v).sum())
        Suuu = float((u ** 3).sum())
        Svvv = float((v ** 3).sum())
        Suvv = float((u * v * v).sum())
        Svuu = float((v * u * u).sum())

        A = np.array([[Suu, Suv], [Suv, Svv]])
        B = np.array([(Suuu + Suvv) / 2.0, (Svvv + Svuu) / 2.0])

        try:
            uc, vc = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            return None

        cx = float(uc) + mx
        cy = float(vc) + my
        r_sq = float(uc) ** 2 + float(vc) ** 2 + (Suu + Svv) / n
        if r_sq <= 0:
            return None
        r = math.sqrt(r_sq)
        if r < 2:
            return None
        return EllipseParams.from_circle(cx, cy, r)

    @staticmethod
    def _fit_circle_huber(
        points: List[Tuple[int, int]], max_iter: int = 20
    ) -> Optional[EllipseParams]:
        initial = EllipseFitter._fit_circle_pratt(points)
        if initial is None:
            initial = EllipseFitter._fit_circle_kasa(points)
        if initial is None:
            return None

        cx, cy, r = initial.cx, initial.cy, initial.semi_major
        pts = np.array(points, dtype=np.float64)

        for _ in range(max_iter):
            dx = pts[:, 0] - cx
            dy = pts[:, 1] - cy
            dists = np.sqrt(dx ** 2 + dy ** 2)
            residuals = np.abs(dists - r)
            med_res = max(float(np.median(residuals)), 1e-6)

            weights = np.where(
                residuals < 1.5 * med_res,
                1.0,
                1.5 * med_res / (residuals + 1e-12),
            )

            wx = pts[:, 0] * weights
            wy = pts[:, 1] * weights
            A = np.column_stack([2 * wx, 2 * wy, weights])
            b = (pts[:, 0] ** 2 + pts[:, 1] ** 2) * weights

            try:
                result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                break

            cx_new = float(result[0])
            cy_new = float(result[1])
            r_sq = float(result[2]) + cx_new ** 2 + cy_new ** 2
            if r_sq <= 0:
                break
            r_new = math.sqrt(r_sq)

            if abs(cx_new - cx) < 0.01 and abs(cy_new - cy) < 0.01:
                cx, cy, r = cx_new, cy_new, r_new
                break
            cx, cy, r = cx_new, cy_new, r_new

        if r < 2:
            return None
        return EllipseParams.from_circle(cx, cy, r)

    @staticmethod
    def _fit_ransac(
        points: List[Tuple[int, int]], n_iter: int = 300
    ) -> Optional[EllipseParams]:
        n = len(points)
        if n < 6:
            return None

        pts_arr = np.array(points, dtype=np.float32)

        spread = max(float(pts_arr.std(axis=0).max()), 5)
        inlier_thresh = spread * 0.2

        best_params = None
        best_inliers = 0
        best_res = float('inf')

        for _ in range(n_iter):
            idx = np.random.choice(n, 5, replace=False)
            sample = pts_arr[idx].reshape(-1, 1, 2)

            try:
                e = cv2.fitEllipse(sample)
                p = EllipseParams.from_cv2(e)
            except cv2.error:
                continue

            if not EllipseFitter._validate(p, points):
                continue

            inlier_pts = []
            for px, py in points:
                d = p.distance_to_boundary_sampled(
                    float(px), float(py), n_samples=36
                )
                if d < inlier_thresh:
                    inlier_pts.append((px, py))

            ni = len(inlier_pts)
            if ni > best_inliers or (ni == best_inliers and ni > 0):
                res = (
                    EllipseFitter._boundary_residual(p, inlier_pts)
                    if inlier_pts
                    else float('inf')
                )
                if ni > best_inliers or res < best_res:
                    best_inliers = ni

                    if len(inlier_pts) >= 5:
                        try:
                            inp = np.array(
                                inlier_pts, dtype=np.float32
                            ).reshape(-1, 1, 2)
                            e2 = cv2.fitEllipse(inp)
                            p2 = EllipseParams.from_cv2(e2)
                            if EllipseFitter._validate(p2, inlier_pts):
                                best_params = p2
                                best_res = EllipseFitter._boundary_residual(
                                    p2, inlier_pts
                                )
                            else:
                                best_params = p
                                best_res = res
                        except cv2.error:
                            best_params = p
                            best_res = res
                    else:
                        best_params = p
                        best_res = res

        return best_params

    @staticmethod
    def _fit_enclosing_circle(
        points: List[Tuple[int, int]]
    ) -> Optional[EllipseParams]:
        pts = np.array(points, dtype=np.float32)
        center, radius = cv2.minEnclosingCircle(pts)

        dists = np.sqrt(
            (pts[:, 0] - float(center[0])) ** 2 +
            (pts[:, 1] - float(center[1])) ** 2
        )
        mean_r = float(dists.mean())
        if mean_r < 2:
            return None

        return EllipseParams.from_circle(
            float(center[0]), float(center[1]), mean_r
        )


# ─── Display Manager ────────────────────────────────────────────────────────

class DisplayManager:
    SCREEN_MARGIN_X = 80
    SCREEN_MARGIN_Y = 120

    def __init__(self, frame_w, frame_h,
                 max_display_w=0, max_display_h=0):
        self.frame_w = frame_w
        self.frame_h = frame_h

        if max_display_w <= 0 or max_display_h <= 0:
            sw, sh = self._detect_screen_size()
            self.max_w = max_display_w if max_display_w > 0 else sw
            self.max_h = max_display_h if max_display_h > 0 else sh
        else:
            self.max_w = max_display_w
            self.max_h = max_display_h

        self.scale = self._compute_fit_scale()
        self._update()
        print(
            f"[Display] {frame_w}x{frame_h} -> "
            f"{self.display_w}x{self.display_h} "
            f"(scale={self.scale:.3f})"
        )

    def _update(self):
        self.display_w = max(1, int(self.frame_w * self.scale))
        self.display_h = max(1, int(self.frame_h * self.scale))

    @staticmethod
    def _detect_screen_size():
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            w = root.winfo_screenwidth() - DisplayManager.SCREEN_MARGIN_X
            h = root.winfo_screenheight() - DisplayManager.SCREEN_MARGIN_Y
            root.destroy()
            if w > 400 and h > 300:
                return w, h
        except Exception:
            pass
        try:
            import subprocess
            out = subprocess.check_output(
                ["xrandr", "--current"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in out.splitlines():
                if "*" in line:
                    res = line.split()[0].split("x")
                    return (
                        int(res[0]) - DisplayManager.SCREEN_MARGIN_X,
                        int(res[1]) - DisplayManager.SCREEN_MARGIN_Y,
                    )
        except Exception:
            pass
        return 1280, 720

    def _compute_fit_scale(self):
        s = min(self.max_w / self.frame_w, self.max_h / self.frame_h)
        if self.frame_w >= 320 and self.frame_h >= 240:
            s = min(s, 1.0)
        return max(0.1, s)

    def set_scale(self, s):
        self.scale = max(0.1, min(s, 3.0))
        self._update()

    def zoom(self, delta):
        self.set_scale(self.scale + delta)

    def fit_to_screen(self):
        self.set_scale(self._compute_fit_scale())

    def screen_to_image(self, sx, sy):
        ix = max(0, min(int(round(sx / self.scale)), self.frame_w - 1))
        iy = max(0, min(int(round(sy / self.scale)), self.frame_h - 1))
        return ix, iy

    def image_to_screen(self, ix, iy):
        return int(round(ix * self.scale)), int(round(iy * self.scale))

    def resize_for_display(self, frame):
        if self.display_w == self.frame_w and \
                self.display_h == self.frame_h:
            return frame.copy()
        interp = (
            cv2.INTER_AREA if self.scale < 1.0 else cv2.INTER_LINEAR
        )
        return cv2.resize(
            frame, (self.display_w, self.display_h),
            interpolation=interp,
        )


# ─── Main Application ───────────────────────────────────────────────────────

class LiveAnnotationApp:
    WINDOW_NAME = "Pupil Tracker - Live Annotation"
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    DOT_RADIUS = 3
    DOT_BORDER = 1
    LINE_THICK = 1
    ELLIPSE_THICK = 2
    CROSS_SIZE = 10
    CROSS_THICK = 1
    CURSOR_GAP = 3
    CURSOR_LEN = 12

    C_PUPIL = (0, 255, 0)
    C_LIMBUS = (255, 130, 0)
    C_SAVED_P = (0, 255, 220)
    C_SAVED_L = (255, 220, 0)
    C_WHITE = (255, 255, 255)
    C_BLACK = (0, 0, 0)
    C_BG = (30, 30, 30)
    C_WARN = (0, 140, 255)
    C_OK = (0, 255, 100)
    C_SNAP = (255, 255, 0)
    C_QUALITY_GOOD = (0, 255, 0)
    C_QUALITY_MED = (0, 200, 255)
    C_QUALITY_BAD = (0, 0, 255)

    def __init__(
        self,
        video_source,
        output_dir="clinical_data",
        annotations_file="annotations.json",
        auto_save_interval=10,
        max_display_w=0,
        max_display_h=0,
    ):
        self.video_source = video_source
        self.output_dir = Path(output_dir)
        self.image_dir = self.output_dir / "training_data" / "images"
        self.mask_dir = self.output_dir / "training_data" / "masks"
        self.ann_path = (
            self.output_dir / "annotations" / annotations_file
        )
        self.auto_save_interval = auto_save_interval

        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.ann_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_video()
        self.display = DisplayManager(
            self.frame_w, self.frame_h,
            max_display_w, max_display_h,
        )

        self.store = AnnotationStore(self.ann_path)
        self.edge_proc = EdgeProcessor()
        self.mode = AnnotationMode.IDLE
        self.fit_constraint = FitConstraint.ELLIPSE
        self.edge_snap = True
        self.paused = False
        self.current_points: List[Tuple[int, int]] = []
        self.current_pupil: Optional[EllipseParams] = None
        self.current_limbus: Optional[EllipseParams] = None
        self.preview_fit: Optional[EllipseParams] = None
        self.preview_score: float = 0.0
        self.frame_idx = 0
        self.current_frame: Optional[np.ndarray] = None
        self.message = ""
        self.message_expire = 0.0
        self.unsaved_count = 0
        self.mouse_x = -1
        self.mouse_y = -1
        self.snap_preview: Optional[Tuple[int, int]] = None
        self.needs_redraw = True
        self.dragging_point: int = -1

        # Curation & Active Learning frame jumping
        self.clean_base = ""
        self.curated_frames = []
        if isinstance(self.video_source, str) and self.video_source.lower() not in ("camera", "cam", "0"):
            base = Path(self.video_source).stem
            self.clean_base = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base)
            curated_path = self.output_dir / "annotations" / f"{self.clean_base}_curated_frames.json"
            if curated_path.exists():
                try:
                    with open(curated_path, "r") as f:
                        self.curated_frames = sorted(list(set(json.load(f))))
                    if self.curated_frames:
                        self._seek(self.curated_frames[0])
                        self._flash(f"Loaded {len(self.curated_frames)} curated hard frames! Starting at frame {self.curated_frames[0]}.", 6.0)
                except Exception as e:
                    print(f"[Curation] Error loading curated frames: {e}")

    def _init_video(self):
        src = self.video_source
        if src.lower() in ("camera", "cam", "0"):
            self.cap = cv2.VideoCapture(0)
            self.total_frames = -1
            self.fps = 30.0
        else:
            path = Path(src)
            if not path.exists():
                raise FileNotFoundError(f"Video not found: {path}")
            self.cap = cv2.VideoCapture(str(path))
            self.total_frames = int(
                self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open: {src}")
        ret, test = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame")
        self.frame_h, self.frame_w = test.shape[:2]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print(
            f"[Video] {self.frame_w}x{self.frame_h}, "
            f"{self.total_frames} frames, {self.fps:.1f} fps"
        )

    def _flash(self, msg, dur=2.5):
        self.message = msg
        self.message_expire = time.time() + dur
        self.needs_redraw = True
        print(f"  >> {msg}")

    def _frame_name(self):
        if isinstance(self.video_source, str) and self.video_source.lower() not in ("camera", "cam", "0"):
            base = Path(self.video_source).stem
            clean_base = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base)
            return f"{clean_base}_frame_{self.frame_idx:06d}.jpg"
        return f"frame_{self.frame_idx:06d}.jpg"

    def _i2s(self, ix, iy):
        return self.display.image_to_screen(ix, iy)

    def _update_preview(self):
        if len(self.current_points) < 5:
            self.preview_fit = None
            self.preview_score = 0.0
            return

        constraint = self.fit_constraint
        if (self.mode == AnnotationMode.PUPIL and
                constraint == FitConstraint.ELLIPSE):
            constraint = FitConstraint.CIRCLE

        self.preview_fit = EllipseFitter.fit(
            self.current_points,
            constraint=constraint,
            image=self.current_frame,
            edge_proc=self.edge_proc,
        )

        if self.preview_fit and self.current_frame is not None:
            self.edge_proc.process(
                self.current_frame, self.frame_idx
            )
            self.preview_score = (
                self.edge_proc.edge_score_along_ellipse(
                    self.preview_fit
                )
            )
        else:
            self.preview_score = 0.0

    def _update_snap_preview(self, screen_x, screen_y):
        if not self.edge_snap or self.current_frame is None:
            self.snap_preview = None
            return
        img_x, img_y = self.display.screen_to_image(
            screen_x, screen_y
        )
        self.edge_proc.process(self.current_frame, self.frame_idx)
        sx, sy = self.edge_proc.snap_to_edge(
            img_x, img_y, search_radius=15
        )
        if (sx, sy) != (img_x, img_y):
            self.snap_preview = (sx, sy)
        else:
            self.snap_preview = None

    # ── Mouse ────────────────────────────────────────────────────────

    def _find_nearest_point(self, img_x, img_y, threshold=15):
        best_idx = -1
        best_dist = float(threshold)
        for i, (px, py) in enumerate(self.current_points):
            d = math.hypot(px - img_x, py - img_y)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x, self.mouse_y = x, y
            if (self.mode != AnnotationMode.IDLE and self.paused):
                self._update_snap_preview(x, y)

                if 0 <= self.dragging_point < len(self.current_points):
                    img_x, img_y = self.display.screen_to_image(x, y)
                    if self.edge_snap and self.current_frame is not None:
                        self.edge_proc.process(
                            self.current_frame, self.frame_idx
                        )
                        img_x, img_y = self.edge_proc.snap_to_edge(
                            img_x, img_y
                        )
                    self.current_points[self.dragging_point] = (
                        img_x, img_y
                    )
                    self._update_preview()

                self.needs_redraw = True
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_x, self.mouse_y = x, y
            if self.mode == AnnotationMode.IDLE:
                self._flash("Press P for pupil or L for limbus")
                return
            if not self.paused:
                self._flash("Pause first (SPACE)")
                return

            img_x, img_y = self.display.screen_to_image(x, y)

            near_idx = self._find_nearest_point(
                img_x, img_y,
                threshold=int(12 / self.display.scale),
            )
            if near_idx >= 0:
                self.dragging_point = near_idx
                self.needs_redraw = True
                return

            if self.edge_snap and self.current_frame is not None:
                self.edge_proc.process(
                    self.current_frame, self.frame_idx
                )
                snapped_x, snapped_y = self.edge_proc.snap_to_edge(
                    img_x, img_y, search_radius=15
                )
                snap_dist = math.hypot(
                    snapped_x - img_x, snapped_y - img_y
                )
                if snap_dist > 0:
                    print(
                        f"  [SNAP] ({img_x},{img_y}) -> "
                        f"({snapped_x},{snapped_y}) "
                        f"d={snap_dist:.1f}"
                    )
                img_x, img_y = snapped_x, snapped_y

            self.current_points.append((img_x, img_y))
            self._update_preview()
            self.needs_redraw = True

            n = len(self.current_points)
            label = (
                "Pupil"
                if self.mode == AnnotationMode.PUPIL
                else "Limbus"
            )
            if n >= 5:
                quality = ""
                if self.preview_score > 0.5:
                    quality = " [GOOD]"
                elif self.preview_score > 0.25:
                    quality = " [OK]"
                else:
                    quality = " [weak - add more]"
                self._flash(
                    f"{label}: {n} pts{quality} - ENTER to confirm"
                )
            else:
                self._flash(
                    f"{label}: {n}/5 pts (need {5 - n} more)"
                )

        if event == cv2.EVENT_LBUTTONUP:
            if self.dragging_point >= 0:
                self.dragging_point = -1
                self._update_preview()
                self.needs_redraw = True

        if event == cv2.EVENT_RBUTTONDOWN:
            if (self.mode != AnnotationMode.IDLE and
                    self.current_points):
                img_x, img_y = self.display.screen_to_image(x, y)
                near_idx = self._find_nearest_point(
                    img_x, img_y,
                    threshold=int(20 / self.display.scale),
                )
                if near_idx >= 0:
                    self.current_points.pop(near_idx)
                    self._update_preview()
                    self._flash(
                        f"Removed point - "
                        f"{len(self.current_points)} left"
                    )
                    self.needs_redraw = True

    # ── Annotation ───────────────────────────────────────────────────

    def _confirm(self):
        """Confirm current annotation using only clicked points.

        No automatic edge refinement — the fit follows your points
        exactly. Press R manually if you want edge-based refinement.
        """
        constraint = self.fit_constraint
        if (self.mode == AnnotationMode.PUPIL and
                constraint == FitConstraint.ELLIPSE):
            constraint = FitConstraint.CIRCLE

        params = EllipseFitter.fit(
            self.current_points,
            constraint=constraint,
            image=self.current_frame,
            edge_proc=self.edge_proc,
        )

        if params is None:
            self._flash(
                "Fit failed! Need 5+ well-placed points", 3
            )
            return

        # No auto-refinement: the shape matches your clicked points.
        # Use R key manually if you want edge snapping.

        if self.mode == AnnotationMode.PUPIL:
            self.current_pupil = params
            self._flash(
                f"Pupil OK ({params.cx:.0f},{params.cy:.0f}) "
                f"r={params.semi_major:.0f}x{params.semi_minor:.0f}"
                f" ar={params.aspect_ratio:.2f}"
                f" - now L for LIMBUS"
            )
            self.mode = AnnotationMode.LIMBUS
            self.current_points = []
            self.preview_fit = None
            self.fit_constraint = FitConstraint.ELLIPSE
            self.needs_redraw = True

        elif self.mode == AnnotationMode.LIMBUS:
            self.current_limbus = params
            self._flash(
                f"Limbus OK ({params.cx:.0f},{params.cy:.0f}) "
                f"r={params.semi_major:.0f}x{params.semi_minor:.0f}"
                f" - saved!"
            )
            self._save_annotation()
            self.mode = AnnotationMode.IDLE
            self.current_points = []
            self.preview_fit = None
            self.needs_redraw = True

    def _save_annotation(self):
        name = self._frame_name()
        cv2.imwrite(
            str(self.image_dir / name), self.current_frame
        )

        ann = FrameAnnotation(
            pupil=self.current_pupil,
            limbus=self.current_limbus,
            timestamp_sec=round(self.frame_idx / self.fps, 3),
            frame_index=self.frame_idx,
            annotated_at=datetime.now().isoformat(),
        )
        self.store.add(name, ann)
        self.unsaved_count += 1
        self._gen_mask(name, ann)

        if self.unsaved_count >= self.auto_save_interval:
            self.store.save()
            self.unsaved_count = 0
            self._flash(f"Auto-saved! Total: {len(self.store)}")

        self.current_pupil = None
        self.current_limbus = None
        print(f"  [SAVED] {name} (total: {len(self.store)})")

    def _gen_mask(self, name, ann):
        mask = np.zeros(
            (self.frame_h, self.frame_w), dtype=np.uint8
        )
        if ann.limbus:
            cv2.ellipse(mask, ann.limbus.to_cv2_int(), 1, -1)
        if ann.pupil:
            cv2.ellipse(mask, ann.pupil.to_cv2_int(), 2, -1)
        cv2.imwrite(
            str(
                self.mask_dir /
                name.replace(".jpg", "_mask.png")
            ),
            mask,
        )

    def _clear_state(self):
        self.mode = AnnotationMode.IDLE
        self.current_points = []
        self.current_pupil = None
        self.current_limbus = None
        self.preview_fit = None
        self.preview_score = 0.0
        self.dragging_point = -1
        self.needs_redraw = True

    # ── Drawing ──────────────────────────────────────────────────────

    def _draw_dot(self, vis, sx, sy, color, radius=None):
        r = radius or self.DOT_RADIUS
        cv2.circle(
            vis, (sx, sy), r + self.DOT_BORDER,
            self.C_BLACK, -1, cv2.LINE_AA,
        )
        cv2.circle(
            vis, (sx, sy), r, color, -1, cv2.LINE_AA,
        )

    def _draw_points(self, vis, color):
        if not self.current_points:
            return
        scr = [
            self._i2s(px, py)
            for px, py in self.current_points
        ]
        n = len(scr)

        for i in range(1, n):
            cv2.line(
                vis, scr[i - 1], scr[i], color,
                self.LINE_THICK, cv2.LINE_AA,
            )
        if n >= 5:
            cv2.line(
                vis, scr[-1], scr[0], color,
                self.LINE_THICK, cv2.LINE_AA,
            )

        for i, (sx, sy) in enumerate(scr):
            if i == self.dragging_point:
                self._draw_dot(vis, sx, sy, self.C_WHITE, radius=5)
            else:
                self._draw_dot(vis, sx, sy, color)
            cv2.putText(
                vis, str(i + 1), (sx + 5, sy - 5),
                self.FONT, 0.3, self.C_WHITE, 1, cv2.LINE_AA,
            )

    def _draw_ellipse(self, vis, params, color, thick=0,
                      center=True, label="", dashed=False):
        t = thick if thick > 0 else self.ELLIPSE_THICK
        scaled = params.to_cv2_scaled(self.display.scale)

        center_pt, axes, angle = scaled
        if axes[0] < 1 or axes[1] < 1:
            return
        if axes[0] > 10000 or axes[1] > 10000:
            return

        if dashed:
            boundary = params.sample_boundary(120)
            pts_scr = [
                self._i2s(int(round(bx)), int(round(by)))
                for bx, by in boundary
            ]
            for i in range(0, len(pts_scr) - 1, 4):
                end = min(i + 2, len(pts_scr) - 1)
                cv2.line(
                    vis, pts_scr[i], pts_scr[end],
                    color, t, cv2.LINE_AA,
                )
        else:
            cv2.ellipse(vis, scaled, color, t, cv2.LINE_AA)

        if center:
            cx, cy = self._i2s(
                int(params.cx), int(params.cy)
            )
            s = self.CROSS_SIZE
            ct = self.CROSS_THICK
            cv2.line(
                vis, (cx - s, cy), (cx + s, cy),
                color, ct, cv2.LINE_AA,
            )
            cv2.line(
                vis, (cx, cy - s), (cx, cy + s),
                color, ct, cv2.LINE_AA,
            )

        if label:
            cx, cy = self._i2s(
                int(params.cx), int(params.cy)
            )
            lx = cx + self.CROSS_SIZE + 4
            ly = cy - 5
            text = (
                f"{label} "
                f"{params.semi_major:.0f}x{params.semi_minor:.0f}"
            )
            cv2.putText(
                vis, text, (lx, ly),
                self.FONT, 0.4, color, 1, cv2.LINE_AA,
            )

    def _draw_preview(self, vis, color, label):
        if self.preview_fit is None:
            return

        if self.preview_score > 0.5:
            q_color = self.C_QUALITY_GOOD
            q_text = "GOOD"
        elif self.preview_score > 0.25:
            q_color = self.C_QUALITY_MED
            q_text = "OK"
        else:
            q_color = self.C_QUALITY_BAD
            q_text = "WEAK"

        self._draw_ellipse(
            vis, self.preview_fit, color, thick=1,
            center=True,
            label=(
                f"{label} [{q_text} {self.preview_score:.0%}]"
            ),
            dashed=True,
        )

        overlay = vis.copy()
        scaled = self.preview_fit.to_cv2_scaled(self.display.scale)
        center_pt, axes, angle = scaled
        if 0 < axes[0] < 10000 and 0 < axes[1] < 10000:
            cv2.ellipse(
                overlay, scaled, color, -1, cv2.LINE_AA,
            )
            cv2.addWeighted(overlay, 0.08, vis, 0.92, 0, vis)

        cx, cy = self._i2s(
            int(self.preview_fit.cx),
            int(self.preview_fit.cy),
        )
        bar_w = 40
        bar_h = 4
        bar_x = cx - bar_w // 2
        bar_y = cy + self.CROSS_SIZE + 8
        cv2.rectangle(
            vis, (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h),
            (80, 80, 80), -1,
        )
        fill_w = int(bar_w * min(1.0, self.preview_score))
        cv2.rectangle(
            vis, (bar_x, bar_y),
            (bar_x + fill_w, bar_y + bar_h),
            q_color, -1,
        )

    def _draw_snap_preview(self, vis):
        if (self.snap_preview is None or
                self.mode == AnnotationMode.IDLE):
            return
        sx, sy = self._i2s(
            self.snap_preview[0], self.snap_preview[1]
        )
        d = 4
        pts = np.array([
            [sx, sy - d], [sx + d, sy],
            [sx, sy + d], [sx - d, sy],
        ], dtype=np.int32)
        cv2.polylines(
            vis, [pts], True, self.C_SNAP, 1, cv2.LINE_AA,
        )

    def _draw_cursor(self, vis):
        if self.mouse_x < 0 or self.mode == AnnotationMode.IDLE:
            return
        if not self.paused:
            return

        x, y = self.mouse_x, self.mouse_y
        h, w = vis.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return

        color = (
            self.C_PUPIL
            if self.mode == AnnotationMode.PUPIL
            else self.C_LIMBUS
        )
        g = self.CURSOR_GAP
        s = self.CURSOR_LEN

        cv2.line(
            vis, (x - g - s, y), (x - g, y),
            color, 1, cv2.LINE_AA,
        )
        cv2.line(
            vis, (x + g, y), (x + g + s, y),
            color, 1, cv2.LINE_AA,
        )
        cv2.line(
            vis, (x, y - g - s), (x, y - g),
            color, 1, cv2.LINE_AA,
        )
        cv2.line(
            vis, (x, y + g), (x, y + g + s),
            color, 1, cv2.LINE_AA,
        )

    def _draw_hud(self, vis):
        dh, dw = vis.shape[:2]
        fs = max(0.35, min(dw / 1600.0, 0.6))

        bar_h = 30
        ov = vis.copy()
        cv2.rectangle(ov, (0, 0), (dw, bar_h), self.C_BG, -1)
        cv2.addWeighted(ov, 0.7, vis, 0.3, 0, vis)

        mt = {
            AnnotationMode.IDLE: "VIEW",
            AnnotationMode.PUPIL: "PUPIL",
            AnnotationMode.LIMBUS: "LIMBUS",
        }[self.mode]
        mc = {
            AnnotationMode.IDLE: self.C_WHITE,
            AnnotationMode.PUPIL: self.C_PUPIL,
            AnnotationMode.LIMBUS: self.C_LIMBUS,
        }[self.mode]
        cv2.putText(
            vis, mt, (6, bar_h - 8),
            self.FONT, fs, mc, 1, cv2.LINE_AA,
        )

        cst = (
            "CIRCLE"
            if self.fit_constraint == FitConstraint.CIRCLE
            else "ELLIPSE"
        )
        cv2.putText(
            vis, cst, (80, bar_h - 8),
            self.FONT, fs * 0.6, self.C_WHITE, 1, cv2.LINE_AA,
        )

        snap_t = "SNAP:ON" if self.edge_snap else "SNAP:OFF"
        snap_c = (
            self.C_SNAP if self.edge_snap else (100, 100, 100)
        )
        cv2.putText(
            vis, snap_t, (160, bar_h - 8),
            self.FONT, fs * 0.5, snap_c, 1, cv2.LINE_AA,
        )

        if self.total_frames > 0:
            if self.curated_frames and self.frame_idx in self.curated_frames:
                pos = self.curated_frames.index(self.frame_idx) + 1
                ft = f"CURATED {pos}/{len(self.curated_frames)} | Frame {self.frame_idx}/{self.total_frames}"
            else:
                ft = f"{self.frame_idx}/{self.total_frames}"
        else:
            ft = f"F:{self.frame_idx}"
        fts = cv2.getTextSize(ft, self.FONT, fs * 0.8, 1)[0]
        cv2.putText(
            vis, ft, (dw // 2 - fts[0] // 2, bar_h - 8),
            self.FONT, fs * 0.8, self.C_WHITE, 1, cv2.LINE_AA,
        )

        st = "PAUSED" if self.paused else "PLAY"
        sts = cv2.getTextSize(st, self.FONT, fs * 0.7, 1)[0]
        cv2.putText(
            vis, st, (dw - sts[0] - 8, bar_h - 8),
            self.FONT, fs * 0.7, self.C_WHITE, 1, cv2.LINE_AA,
        )

        if self.mode != AnnotationMode.IDLE:
            n = len(self.current_points)
            need = max(0, 5 - n)
            if n == 0:
                pt = "Click boundary points (right-click remove)..."
                pc = self.C_WARN
            elif need > 0:
                pt = f"{n}/5 pts (need {need} more)"
                pc = self.C_WARN
            else:
                score_text = ""
                if self.preview_score > 0:
                    score_text = (
                        f" | quality: {self.preview_score:.0%}"
                    )
                pt = (
                    f"{n} pts - ENTER confirm | "
                    f"R refine{score_text}"
                )
                pc = self.C_OK
            cv2.putText(
                vis, pt, (6, bar_h + 18),
                self.FONT, fs * 0.5, pc, 1, cv2.LINE_AA,
            )

        bot_h = 24
        bot_y = dh - bot_h
        ov2 = vis.copy()
        cv2.rectangle(
            ov2, (0, bot_y), (dw, dh), self.C_BG, -1,
        )
        cv2.addWeighted(ov2, 0.7, vis, 0.3, 0, vis)

        cv2.putText(
            vis, f"Ann:{len(self.store)}", (6, dh - 6),
            self.FONT, fs * 0.5, self.C_OK, 1, cv2.LINE_AA,
        )

        zt = f"{self.display.scale:.0%}"
        zts = cv2.getTextSize(zt, self.FONT, fs * 0.45, 1)[0]
        cv2.putText(
            vis, zt, (dw // 2 - zts[0] // 2, dh - 6),
            self.FONT, fs * 0.45, self.C_WHITE, 1, cv2.LINE_AA,
        )

        if self.total_frames > 0:
            frac = self.frame_idx / max(self.total_frames, 1)
            pw = max(1, int(frac * dw))
            cv2.rectangle(
                vis, (0, bot_y - 2), (dw, bot_y),
                (60, 60, 60), -1,
            )
            cv2.rectangle(
                vis, (0, bot_y - 2), (pw, bot_y),
                self.C_PUPIL, -1,
            )

        if self.message and time.time() <= self.message_expire:
            cv2.putText(
                vis, self.message, (6, bot_y - 8),
                self.FONT, fs * 0.45, self.C_WARN,
                1, cv2.LINE_AA,
            )

        ctrl = (
            "SPC:pause P:pupil L:limbus RET:confirm "
            "R:refine U:undo G:snap D:shape S:save Q:quit"
        )
        cv2.putText(
            vis, ctrl, (6, bot_y - 22),
            self.FONT, max(0.25, fs * 0.28),
            (120, 120, 120), 1, cv2.LINE_AA,
        )

    # ── Render ───────────────────────────────────────────────────────

    def _render(self):
        if self.current_frame is None:
            blank = np.zeros(
                (
                    max(100, self.display.display_h),
                    max(100, self.display.display_w),
                    3,
                ),
                dtype=np.uint8,
            )
            cv2.putText(
                blank, "No frame",
                (20, blank.shape[0] // 2),
                self.FONT, 1, self.C_WHITE, 2,
            )
            return blank

        vis = self.display.resize_for_display(self.current_frame)

        fn = self._frame_name()
        if fn in self.store:
            s = self.store.annotations[fn]
            if s.pupil:
                self._draw_ellipse(
                    vis, s.pupil, self.C_SAVED_P,
                    thick=2, center=True, label="P(saved)",
                )
            if s.limbus:
                self._draw_ellipse(
                    vis, s.limbus, self.C_SAVED_L,
                    thick=2, center=False, label="L(saved)",
                )

        if self.current_pupil:
            self._draw_ellipse(
                vis, self.current_pupil, self.C_PUPIL,
                thick=2, center=True, label="Pupil OK",
            )
        if self.current_limbus:
            self._draw_ellipse(
                vis, self.current_limbus, self.C_LIMBUS,
                thick=2, center=True, label="Limbus OK",
            )

        if self.current_points:
            c = (
                self.C_PUPIL
                if self.mode == AnnotationMode.PUPIL
                else self.C_LIMBUS
            )
            lbl = (
                "Pupil"
                if self.mode == AnnotationMode.PUPIL
                else "Limbus"
            )
            self._draw_points(vis, c)
            self._draw_preview(vis, c, lbl)

        self._draw_snap_preview(vis)
        self._draw_cursor(vis)
        self._draw_hud(vis)

        return vis

    # ── Seek ─────────────────────────────────────────────────────────

    def _seek(self, idx):
        if self.total_frames > 0:
            idx = max(0, min(idx, self.total_frames - 1))
        else:
            idx = max(0, idx)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = self.cap.read()
        if ret:
            self.frame_idx = idx
            self.current_frame = f
            self.needs_redraw = True

    # ── Keys ─────────────────────────────────────────────────────────

    def _handle_key(self, key):
        if key == -1:
            return True
        c = chr(key & 0xFF).lower()

        if c == "q" or key == 27:
            return False

        if key == 32:  # SPACE
            self.paused = not self.paused
            if self.paused:
                if self.current_frame is not None:
                    self.edge_proc.process(
                        self.current_frame, self.frame_idx
                    )
                self._flash(
                    "PAUSED - P for pupil, L for limbus"
                )
            else:
                self._clear_state()
                self._flash("Playing...")
            return True

        if c == "p":
            if not self.paused:
                self._flash("Pause first (SPACE)")
            else:
                self._clear_state()
                self.mode = AnnotationMode.PUPIL
                self.fit_constraint = FitConstraint.CIRCLE
                self._flash(
                    "PUPIL (circle): click 5+ boundary points"
                )
            return True

        if c == "l":
            if not self.paused:
                self._flash("Pause first (SPACE)")
            else:
                self.mode = AnnotationMode.LIMBUS
                self.current_points = []
                self.preview_fit = None
                self.fit_constraint = FitConstraint.ELLIPSE
                self._flash(
                    "LIMBUS (ellipse): click 5+ boundary points"
                )
                self.needs_redraw = True
            return True

        if key == 13:  # ENTER
            if self.mode in (
                AnnotationMode.PUPIL, AnnotationMode.LIMBUS
            ):
                self._confirm()
            else:
                self._flash("Press P or L first")
            return True

        if c == "r":
            if (self.preview_fit and
                    self.current_frame is not None):
                self.edge_proc.process(
                    self.current_frame, self.frame_idx
                )
                refined = EllipseFitter.refine_with_edges(
                    self.preview_fit,
                    self.current_frame,
                    self.edge_proc,
                    iterations=5,
                )
                if refined:
                    boundary = refined.sample_boundary(
                        len(self.current_points)
                    )
                    self.current_points = [
                        (int(round(bx)), int(round(by)))
                        for bx, by in boundary
                    ]
                    self._update_preview()
                    self._flash(
                        f"Refined! score: "
                        f"{self.preview_score:.0%}"
                    )
                else:
                    self._flash("Refinement failed")
            elif len(self.current_points) < 5:
                self._flash("Need 5+ points first")
            return True

        if c == "u":
            if self.current_points:
                self.current_points.pop()
                self._update_preview()
                self._flash(
                    f"Undo - {len(self.current_points)} left"
                )
            return True

        if c == "c":
            self.current_points = []
            self.preview_fit = None
            self.preview_score = 0.0
            self._flash("Cleared")
            return True

        if c == "s":
            self.store.save()
            self.unsaved_count = 0
            self._flash(
                f"Saved {len(self.store)} annotations"
            )
            return True

        if c == "g":
            self.edge_snap = not self.edge_snap
            self._flash(
                f"Edge snap: "
                f"{'ON' if self.edge_snap else 'OFF'}"
            )
            return True

        if c == "d":
            if self.fit_constraint == FitConstraint.ELLIPSE:
                self.fit_constraint = FitConstraint.CIRCLE
                self._flash("Constraint: CIRCLE")
            else:
                self.fit_constraint = FitConstraint.ELLIPSE
                self._flash("Constraint: ELLIPSE")
            if self.current_points:
                self._update_preview()
            return True

        if c == "f":
            self.display.fit_to_screen()
            cv2.resizeWindow(
                self.WINDOW_NAME,
                self.display.display_w,
                self.display.display_h,
            )
            self._flash(f"Fit {self.display.scale:.0%}")
            return True

        if c in ("+", "="):
            self.display.zoom(0.1)
            cv2.resizeWindow(
                self.WINDOW_NAME,
                self.display.display_w,
                self.display.display_h,
            )
            self._flash(f"Zoom {self.display.scale:.0%}")
            return True

        if c == "-":
            self.display.zoom(-0.1)
            cv2.resizeWindow(
                self.WINDOW_NAME,
                self.display.display_w,
                self.display.display_h,
            )
            self._flash(f"Zoom {self.display.scale:.0%}")
            return True

        if c == "n" and self.paused:
            if self.curated_frames:
                next_indices = [idx for idx in self.curated_frames if idx > self.frame_idx]
                if next_indices:
                    target_idx = next_indices[0]
                    self._seek(target_idx)
                    pos = self.curated_frames.index(target_idx) + 1
                    self._flash(f"Curated frame {pos}/{len(self.curated_frames)} (frame {target_idx})")
                else:
                    self._flash("No more curated frames forward!")
            else:
                self._seek(self.frame_idx + 1)
            self._clear_state()
            return True

        if c == "b" and self.paused:
            if self.curated_frames:
                prev_indices = [idx for idx in self.curated_frames if idx < self.frame_idx]
                if prev_indices:
                    target_idx = prev_indices[-1]
                    self._seek(target_idx)
                    pos = self.curated_frames.index(target_idx) + 1
                    self._flash(f"Curated frame {pos}/{len(self.curated_frames)} (frame {target_idx})")
                else:
                    self._flash("No more curated frames backward!")
            else:
                self._seek(self.frame_idx - 1)
            self._clear_state()
            return True

        if key in (83, 3, 100) and self.paused:
            self._seek(self.frame_idx + 10)
            self._flash(f">> frame {self.frame_idx}")
            return True

        if key in (81, 2, 97) and self.paused:
            self._seek(self.frame_idx - 10)
            self._flash(f"<< frame {self.frame_idx}")
            return True

        if c == "t":
            self._trigger_retrain()
            return True

        return True

    def _trigger_retrain(self):
        self.store.save()
        n = len(self.store)
        if n < 10:
            self._flash(
                f"Need 10+ annotations (have {n})", 3
            )
            return
        self._flash(
            f"Retraining ({n} annotations)...", 10
        )
        cv2.imshow(self.WINDOW_NAME, self._render())
        cv2.waitKey(100)
        try:
            retrain_model(
                self.image_dir,
                self.mask_dir,
                self.ann_path,
                epochs=min(50, n),
                batch_size=min(8, max(2, n // 2)),
            )
            self._flash("Retrain done!", 5)
        except Exception as e:
            self._flash(f"Retrain failed: {e}", 5)
            print(f"[ERROR] {e}")

    # ── Main Loop ────────────────────────────────────────────────────

    def run(self):
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.WINDOW_NAME,
            self.display.display_w,
            self.display.display_h,
        )
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_cb)

        print(f"\n{'=' * 60}")
        print(f"  LIVE ANNOTATION TOOL")
        print(f"  Source : {self.video_source}")
        print(f"  Frame  : {self.frame_w}x{self.frame_h}")
        print(
            f"  Display: "
            f"{self.display.display_w}x{self.display.display_h} "
            f"({self.display.scale:.0%})"
        )
        print(f"  Output : {self.output_dir}")
        print(f"  Saved  : {len(self.store)} annotations")
        print(f"{'=' * 60}")
        print(__doc__)

        delay = max(1, int(1000 / self.fps))
        running = True

        try:
            while running:
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        if self.total_frames > 0:
                            self._flash(
                                "End of video. B to go back.",
                                5,
                            )
                            self.paused = True
                        else:
                            break
                    else:
                        self.current_frame = frame
                        self.frame_idx = int(
                            self.cap.get(
                                cv2.CAP_PROP_POS_FRAMES
                            )
                        )
                        self.needs_redraw = True

                if self.needs_redraw or not self.paused:
                    cv2.imshow(
                        self.WINDOW_NAME, self._render()
                    )
                    self.needs_redraw = False

                key = cv2.waitKey(
                    30 if self.paused else delay
                )
                running = self._handle_key(key)

        except KeyboardInterrupt:
            print("\n[Interrupted]")
        finally:
            if self.unsaved_count > 0:
                self.store.save()
            self.cap.release()
            cv2.destroyAllWindows()
            print(
                f"\nDone. {len(self.store)} annotations saved."
            )


# ─── Standalone utilities ────────────────────────────────────────────────

def generate_masks_from_annotations(
    annotations_path, image_dir, mask_dir
):
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    with open(annotations_path, "r") as f:
        raw = json.load(f)
    gen = 0
    for name, data in raw.items():
        ip = Path(image_dir) / name
        if not ip.exists():
            continue
        img = cv2.imread(str(ip))
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        ann = FrameAnnotation.from_dict(data)
        if ann.limbus:
            cv2.ellipse(mask, ann.limbus.to_cv2_int(), 1, -1)
        if ann.pupil:
            cv2.ellipse(mask, ann.pupil.to_cv2_int(), 2, -1)
        cv2.imwrite(
            str(mask_dir / name.replace(".jpg", "_mask.png")),
            mask,
        )
        gen += 1
    print(f"[MaskGen] {gen} masks generated")
    return gen


def retrain_model(
    image_dir, mask_dir, annotations_path,
    epochs=50, batch_size=8,
    model_save_path="models/best_model.pth",
):
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("[ERROR] pip install torch torchvision")
        return

    model_path = Path(model_save_path)
    if model_path.exists():
        backup = model_path.with_name(
            f"backup_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        )
        shutil.copy2(model_path, backup)

    class DS(Dataset):
        def __init__(self, idir, mdir, subset_pairs=None):
            if subset_pairs is not None:
                self.pairs = subset_pairs
            else:
                self.pairs = []
                for p in sorted(Path(idir).glob("*.jpg")):
                    mp = Path(mdir) / (p.stem + "_mask.png")
                    if mp.exists():
                        self.pairs.append((p, mp))
            print(f"[Dataset] {len(self.pairs)} pairs")

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, i):
            ip, mp = self.pairs[i]
            img = cv2.resize(
                cv2.cvtColor(
                    cv2.imread(str(ip)), cv2.COLOR_BGR2RGB
                ),
                (256, 256),
            ).astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            mask = cv2.resize(
                cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE),
                (256, 256),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int64)
            return torch.from_numpy(img), torch.from_numpy(mask)

    ds = DS(image_dir, mask_dir)
    if len(ds) == 0:
        print("[Retrain] No data!")
        return

    # ── Video-wise Train/Val Splitting ──
    video_to_pairs = {}
    for p, mp in ds.pairs:
        if "_frame_" in p.name:
            vname = p.name.split("_frame_")[0]
        else:
            vname = "legacy_single_images"
        video_to_pairs.setdefault(vname, []).append((p, mp))

    video_names = list(video_to_pairs.keys())
    import random
    rng = random.Random(42)  # reproducible seed
    rng.shuffle(video_names)

    # 15% validation videos, at least 1 if we have multiple videos
    n_val_vids = max(1, int(len(video_names) * 0.15)) if len(video_names) > 1 else 0

    val_vids = set(video_names[:n_val_vids])
    train_vids = set(video_names[n_val_vids:])

    train_pairs = []
    val_pairs = []
    for vname, pairs in video_to_pairs.items():
        if vname in val_vids:
            val_pairs.extend(pairs)
        else:
            train_pairs.extend(pairs)

    if not val_pairs and len(ds.pairs) >= 2:
        nv = max(1, len(ds.pairs) // 10)
        nt = len(ds.pairs) - nv
        tds, vds = torch.utils.data.random_split(ds, [nt, nv])
    else:
        tds = DS(image_dir, mask_dir, subset_pairs=train_pairs)
        vds = DS(image_dir, mask_dir, subset_pairs=val_pairs)

    print(f"[Split] Videos: {len(video_names)} total. Train vids: {len(train_vids)} ({len(tds)} frames), Val vids: {len(val_vids)} ({len(vds)} frames)")

    tl = DataLoader(
        tds, batch_size=batch_size, shuffle=True, drop_last=True if len(tds) > batch_size else False,
    )
    vl = DataLoader(vds, batch_size=batch_size)

    try:
        from pupil_tracking.ml.model import PupilSegmentationModel
        model = PupilSegmentationModel(num_classes=3)
    except ImportError:
        model = _build_unet(3)

    dev = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(dev)
    if model_path.exists():
        try:
            model.load_state_dict(
                torch.load(str(model_path), map_location=dev),
                strict=False,
            )
        except Exception:
            pass

    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs
    )
    best = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        tloss = 0
        for imgs, masks in tl:
            imgs, masks = imgs.to(dev), masks.to(dev)
            opt.zero_grad()
            loss = crit(model(imgs), masks)
            loss.backward()
            opt.step()
            tloss += loss.item()
        tloss /= max(len(tl), 1)

        model.eval()
        vloss = 0
        with torch.no_grad():
            for imgs, masks in vl:
                imgs, masks = imgs.to(dev), masks.to(dev)
                vloss += crit(model(imgs), masks).item()
        vloss /= max(len(vl), 1)
        sched.step()

        if ep % 10 == 0 or ep == 1:
            print(
                f"  Ep {ep}/{epochs}: "
                f"t={tloss:.4f} v={vloss:.4f}"
            )
        if vloss < best:
            best = vloss
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(model_path))

    print(f"[Retrain] Done. best_val={best:.4f}")


def _build_unet(nc=3):
    import torch
    import torch.nn as nn

    class DC(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.n = nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(True),
                nn.Conv2d(o, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(True),
            )

        def forward(self, x):
            return self.n(x)

    class U(nn.Module):
        def __init__(self, nc):
            super().__init__()
            self.e1 = DC(3, 64)
            self.e2 = DC(64, 128)
            self.e3 = DC(128, 256)
            self.p = nn.MaxPool2d(2)
            self.b = DC(256, 512)
            self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.d3 = DC(512, 256)
            self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.d2 = DC(256, 128)
            self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.d1 = DC(128, 64)
            self.o = nn.Conv2d(64, nc, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.p(e1))
            e3 = self.e3(self.p(e2))
            b = self.b(self.p(e3))
            d3 = self.d3(torch.cat([self.u3(b), e3], 1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.o(d1)

    return U(nc)


def check_training_data(image_dir, mask_dir):
    imgs = sorted(Path(image_dir).glob("*.jpg"))
    msks = sorted(Path(mask_dir).glob("*_mask.png"))
    matched = sum(
        1
        for i in imgs
        if (Path(mask_dir) / (i.stem + "_mask.png")).exists()
    )
    print(
        f"\nImages:{len(imgs)} Masks:{len(msks)} "
        f"Matched:{matched}"
    )
    print(
        f"Status: "
        f"{'READY' if matched >= 10 else 'Need more'}\n"
    )
    return matched >= 10


def main():
    import argparse

    p = argparse.ArgumentParser(
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd")

    a = sub.add_parser("annotate")
    a.add_argument("source")
    a.add_argument("--output-dir", default="clinical_data")
    a.add_argument("--auto-save", type=int, default=10)
    a.add_argument("--max-width", type=int, default=0)
    a.add_argument("--max-height", type=int, default=0)
    a.add_argument(
        "--corrected-output",
        default=None,
        help="Path to write corrected annotations without overwriting originals",
    )

    m = sub.add_parser("generate-masks")
    m.add_argument(
        "--annotations",
        default="clinical_data/annotations/annotations.json",
    )
    m.add_argument(
        "--image-dir",
        default="clinical_data/training_data/images",
    )
    m.add_argument(
        "--mask-dir",
        default="clinical_data/training_data/masks",
    )

    t = sub.add_parser("train")
    t.add_argument(
        "--image-dir",
        default="clinical_data/training_data/images",
    )
    t.add_argument(
        "--mask-dir",
        default="clinical_data/training_data/masks",
    )
    t.add_argument(
        "--annotations",
        default="clinical_data/annotations/annotations.json",
    )
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument(
        "--model-path", default="models/best_model.pth"
    )

    c = sub.add_parser("check")
    c.add_argument(
        "--image-dir",
        default="clinical_data/training_data/images",
    )
    c.add_argument(
        "--mask-dir",
        default="clinical_data/training_data/masks",
    )

    args = p.parse_args()

    if args.cmd == "annotate":
        LiveAnnotationApp(
            args.source,
            args.output_dir,
            auto_save_interval=args.auto_save,
            max_display_w=args.max_width,
            max_display_h=args.max_height,
        ).run()
    elif args.cmd == "generate-masks":
        generate_masks_from_annotations(
            args.annotations, args.image_dir, args.mask_dir,
        )
    elif args.cmd == "train":
        retrain_model(
            args.image_dir,
            args.mask_dir,
            args.annotations,
            args.epochs,
            args.batch_size,
            args.model_path,
        )
    elif args.cmd == "check":
        check_training_data(args.image_dir, args.mask_dir)
    else:
        p.print_help()


if __name__ == "__main__":
    main()