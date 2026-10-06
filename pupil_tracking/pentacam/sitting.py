"""CPU utilities for seated iris images; no trained weights or GPU required.

Scores are engineering quality gates, not calibrated clinical probabilities.
Positive angles denote counter-clockwise image rotation (OpenCV convention).
"""
from dataclasses import dataclass
import hashlib

import cv2
import numpy as np


def valid_geometry(pupil, limbus, shape):
    """Reject malformed or physically unusable externally supplied geometry."""
    if pupil is None or limbus is None:
        return False
    h, w = shape[:2]
    for e in (pupil, limbus):
        values = (e.center_x, e.center_y, e.semi_major, e.semi_minor, e.angle_deg)
        if (not np.isfinite(values).all() or min(e.semi_major, e.semi_minor) <= 1
                or max(e.semi_major, e.semi_minor) > max(h, w)
                or not (0 <= e.center_x < w and 0 <= e.center_y < h)):
            return False
    return limbus.radius > pupil.radius * 1.1


def registration_mask(image):
    """Exclude black margins, saturated glints, and vivid rendered overlays."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    bad = (gray < 6) | (gray > 245)
    if image.ndim == 3:
        blue, green, red = cv2.split(image)
        high = cv2.max(cv2.max(blue, green), red)
        low = cv2.min(cv2.min(blue, green), red)
        bad |= (cv2.subtract(high, low) > 90) & (high > 170)
    bad = cv2.dilate(bad.astype(np.uint8), np.ones((5, 5), np.uint8))
    return (bad == 0).astype(np.uint8) * 255


def ocular_viewport(gray):
    """Locate a large dark image panel, preserving raw ocular images unchanged."""
    h, w = gray.shape
    small = cv2.resize(gray, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    binary = (small < 230).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > small.shape[1] * .45 and ch > small.shape[0] * .45:
            # Panel border must actually surround the region, not just be a dark pupil.
            if cv2.contourArea(c) / (cw * ch) > .88:
                candidates.append((cw * ch, x, y, cw, ch))
    if not candidates:
        return 0, 0, w, h
    _, x, y, cw, ch = max(candidates)
    sx, sy = w / small.shape[1], h / small.shape[0]
    x0, y0 = int(x * sx), int(y * sy)
    return x0, y0, min(w - x0, int(cw * sx)), min(h - y0, int(ch * sy))


@dataclass(frozen=True)
class AngularMatch:
    angle_deg: float = 0.0
    valid: bool = False
    score: float = 0.0
    overlap: float = 0.0
    peak_margin: float = 0.0
    band_spread_deg: float = 0.0
    reason: str = "insufficient_texture"


def masked_angular_match(ref, curr, mask_ref=None, mask_curr=None, max_degrees=30.0,
                         *, reference_cache=None):
    """Shift-dependent masked ZNCC, evaluated at all angles using six batched FFTs.

Unlike multiplying both strips by a shared stationary mask, each candidate
aligns the masks with the iris texture. Retains radial information rather than
averaging it away. Radial bands provide a second consistency check.
"""
    a, b = np.asarray(ref, dtype=np.float32), np.asarray(curr, dtype=np.float32)
    if a.ndim != 2 or a.shape != b.shape or min(a.shape) < 8:
        return AngularMatch(reason="invalid_shape")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return AngularMatch(reason="nonfinite_input")
    if not 0 < max_degrees < 180:
        raise ValueError("max_degrees must be between 0 and 180")
    ma = np.ones_like(a) if mask_ref is None else (np.asarray(mask_ref) > 0).astype(np.float32)
    mb = np.ones_like(b) if mask_curr is None else (np.asarray(mask_curr) > 0).astype(np.float32)
    if ma.shape != a.shape or mb.shape != b.shape:
        return AngularMatch(reason="invalid_mask_shape")
    # Remove broad illumination while preserving angular crypt/furrow texture.
    # Circular padding avoids a false feature at the 0/360-degree seam.
    def highpass(x, m):
        k = max(5, (x.shape[1] // 24) | 1)
        p = k // 2
        weight = cv2.blur(np.pad(m, ((0, 0), (p, p)), mode="wrap"), (k, 1))[:, p:-p]
        mean = cv2.blur(np.pad(x * m, ((0, 0), (p, p)), mode="wrap"), (k, 1))[:, p:-p]
        return (x - mean / np.maximum(weight, 1e-6)) * m
    n = a.shape[1]
    # One bounded reference cache, keyed by content rather than object identity.
    # Callers may safely reuse mutable acquisition arrays; changes invalidate it.
    key = None
    fa = None
    if reference_cache is not None:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.ascontiguousarray(a))
        digest.update(np.ascontiguousarray(ma))
        key = (a.shape, digest.digest())
        if reference_cache.get("key") == key:
            fa = reference_cache["spectrum"]
    if fa is None:
        a = highpass(a, ma)
        fa = np.fft.rfft(np.stack((ma, a, a * a)), axis=-1)
        if reference_cache is not None:
            reference_cache.clear()
            reference_cache.update(key=key, spectrum=fa)
    b = highpass(b, mb)
    fb = np.fft.rfft(np.stack((mb, b, b * b)), axis=-1)
    products = np.stack((
        fa[0] * fb[0].conj(), fa[1] * fb[0].conj(),
        fa[0] * fb[1].conj(), fa[2] * fb[0].conj(),
        fa[0] * fb[2].conj(), fa[1] * fb[1].conj(),
    ))
    # Correlation is linear: sum spectra within each radial band BEFORE
    # inverse FFT. Only 18 inverse transforms instead of 6 * num_radial,
    # while retaining every pixel and the same three-band consistency test.
    groups = np.array_split(products, 3, axis=1)
    band_sizes = [group.shape[1] for group in groups]
    terms = np.fft.irfft(np.stack([group.sum(axis=1, dtype=np.complex128)
                                  for group in groups], axis=1), n=n, axis=-1)
    shifts = np.arange(n, dtype=float)
    shifts[shifts > n / 2] -= n
    allowed = np.abs(shifts * 360 / n) <= max_degrees

    def scores(t, num_rows):
        count, sa, sb, saa, sbb, sab = t.sum(axis=1)
        count = np.maximum(count, 1e-6)
        va, vb = saa - sa * sa / count, sbb - sb * sb / count
        denom = np.sqrt(np.maximum(va * vb, 0))
        corr = (sab - sa * sb / count) / np.maximum(denom, 1e-8)
        overlap = count / (num_rows * n)
        ok = allowed & (overlap >= .30) & (va / count > 1e-5) & (vb / count > 1e-5)
        return np.where(ok, np.clip(corr, -1, 1), -1), overlap

    score, overlap = scores(terms, a.shape[0])
    peak = int(np.argmax(score))
    if score[peak] < .35:
        return AngularMatch(score=float(score[peak]), reason="weak_correlation")
    # Reject boundary solutions: the true maximum may lie outside the search.
    if abs(shifts[peak]) >= np.floor(max_degrees * n / 360):
        return AngularMatch(reason="search_boundary")
    dist = np.abs((np.arange(n) - peak + n / 2) % n - n / 2)
    competing = allowed & (dist > max(2, n * 2 / 360))
    margin = float(score[peak] - np.max(score[competing])) if competing.any() else 0.0
    bands = []
    for i, band_size in enumerate(band_sizes):
        bs, _ = scores(terms[:, i:i+1], band_size)
        j = int(np.argmax(bs))
        if bs[j] >= .30:
            bands.append(float(shifts[j] * 360 / n))
    spread = float(np.ptp(bands)) if len(bands) >= 2 else 180.0
    yl, yc, yr = score[(peak - 1) % n], score[peak], score[(peak + 1) % n]
    denominator = yl - 2 * yc + yr
    delta = float(np.clip(.5 * (yl - yr) / denominator, -.5, .5)) if denominator < -1e-8 else 0.
    valid = margin >= .035 and spread <= 2.0
    return AngularMatch(
        angle_deg=float((shifts[peak] + delta) * 360 / n), valid=bool(valid),
        score=float(score[peak]), overlap=float(overlap[peak]), peak_margin=margin,
        band_spread_deg=spread, reason="" if valid else "ambiguous_or_inconsistent_bands",
    )
