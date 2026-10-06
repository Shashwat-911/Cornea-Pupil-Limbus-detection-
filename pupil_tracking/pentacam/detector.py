"""Pentacam iris and anterior segment detector.

Provides robust, additive anatomical feature detection on Pentacam (seated IR)
anterior segment and Scheimpflug eye imagery:
    1. Rejects Pentacam UI chrome, overlays, reticles, and annotation text.
    2. Localizes anatomical pupil and limbus boundaries.
    3. Normalizes iris stroma between pupil and limbus.
    4. Detects spatially separated texture keypoints, not verified anatomical labels.
    5. Extracts contrast-invariant multi-scale descriptors for cross-modality matching.

Implementation goals (not clinical validation):
    - Purely additive, non-mutating.
    - Zero interference with existing ELITA centration pipeline.
    - Rejects UI feathers and overlays.
    - Runtime depends on image size, hardware, and OpenCV build; benchmark on target data.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pupil_tracking.pentacam.sitting import ocular_viewport, valid_geometry, registration_mask
from pupil_tracking.pentacam.refinement import refine_limbus, anatomy_candidates

from pupil_tracking.pentacam.types import (
    PentacamDetectionResult,
    PentacamDetectionStatus,
    PentacamFeature,
    PentacamFeatureSet,
    PentacamGeometry,
    PentacamImageType,
    PentacamQuality,
)
from pupil_tracking.utils.types import EllipseParams

logger = logging.getLogger(__name__)


class PentacamIrisDetector:
    """Detects pupil, limbus, and distinctive iris landmarks on Pentacam imagery.

    Designed for cross-modality iris registration against ELITA RGB/IR images.
    """

    def __init__(
        self,
        num_angles: int = 72,
        num_radii: int = 8,
        min_contrast: float = 3.5,
        min_features: int = 6,
        inner_inset_frac: float = 0.10,
        outer_inset_frac: float = 0.08,
        max_working_size: int = 640,
    ) -> None:
        self.num_angles = num_angles
        self.num_radii = num_radii
        self.min_contrast = min_contrast
        self.min_features = min_features
        self.inner_inset_frac = inner_inset_frac
        self.outer_inset_frac = outer_inset_frac
        self.max_working_size = max_working_size

    def detect(
        self,
        image: np.ndarray,
        geometry: Optional[PentacamGeometry] = None,
        image_type: PentacamImageType = PentacamImageType.ANTERIOR_SEGMENT,
        extract_features: bool = True,
        review_anatomy: bool = False,
        refine_boundary: bool = True,
    ) -> PentacamDetectionResult:
        """Detect iris structure and landmarks from a Pentacam image.

        Parameters
        ----------
        image : np.ndarray
            Pentacam image (BGR or grayscale uint8).
        geometry : Optional[PentacamGeometry]
            Pre-computed pupil/limbus geometry, if available.
        image_type : PentacamImageType
            Type of Pentacam image.

        Returns
        -------
        PentacamDetectionResult
            Validated detection result with geometry and accepted iris features.
        """
        t0 = time.perf_counter()

        if image is None or image.size == 0:
            return PentacamDetectionResult(
                valid=False,
                status=PentacamDetectionStatus.NO_IMAGE,
                failure_reason="Empty or null image input",
                processing_time_ms=(time.perf_counter() - t0) * 1000.0,
            )
        if image.dtype != np.uint8 or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
            raise ValueError("Expected uint8 grayscale or BGR image")

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape[:2]

        # Process the ocular panel at bounded resolution, then map all geometry
        # and descriptors back to original screenshot coordinates.
        if geometry is None and max(h, w) > self.max_working_size > 0:
            x, y, cw, ch = ocular_viewport(gray)
            scale = min(1.0, self.max_working_size / max(cw, ch))
            rw, rh = max(1, round(cw * scale)), max(1, round(ch * scale))
            crop = cv2.resize(image[y:y + ch, x:x + cw], (rw, rh), interpolation=cv2.INTER_AREA)
            worker = PentacamIrisDetector(self.num_angles, self.num_radii, self.min_contrast,
                                         self.min_features, self.inner_inset_frac,
                                         self.outer_inset_frac, max_working_size=0)
            result = worker.detect(crop, image_type=image_type, extract_features=extract_features,
                                   review_anatomy=review_anatomy, refine_boundary=refine_boundary)
            # Resize dimensions can differ by <1 pixel. Use the actual ratios.
            sx, sy = cw / rw, ch / rh
            for ellipse in (result.geometry.pupil, result.geometry.limbus, result.geometry.refined_limbus):
                if ellipse is not None:
                    ellipse.center_x = (ellipse.center_x + .5) * sx - .5 + x
                    ellipse.center_y = (ellipse.center_y + .5) * sy - .5 + y
                    ellipse.semi_major *= (sx + sy) / 2
                    ellipse.semi_minor *= (sx + sy) / 2
            for feature in result.feature_set.features + result.anatomy_candidates:
                feature.x = (feature.x + .5) * sx - .5 + x
                feature.y = (feature.y + .5) * sy - .5 + y
            result.geometry.limbus_support_points = [
                ((px + .5) * sx - .5 + x, (py + .5) * sy - .5 + y)
                for px, py in result.geometry.limbus_support_points]
            if result.geometry.limbus_fit_residual_px is not None:
                result.geometry.limbus_fit_residual_px *= (sx + sy) / 2
            if result.geometry.pupil is not None:
                result.geometry.pupil_radius_px = result.geometry.pupil.radius
            if result.geometry.limbus is not None:
                result.geometry.limbus_radius_px = result.geometry.limbus.radius
            result.image_width, result.image_height = w, h
            result.processing_time_ms = (time.perf_counter() - t0) * 1000
            return result

        # 1. Build UI / Overlay exclusion mask
        ui_mask = self._build_ui_mask(gray)
        if image.ndim == 3:
            blue, green, red = cv2.split(image)
            high = cv2.max(cv2.max(blue, green), red)
            low = cv2.min(cv2.min(blue, green), red)
            vivid = ((cv2.subtract(high, low) > 90) & (high > 170)).astype(np.uint8)
            ui_mask[cv2.dilate(vivid, np.ones((5, 5), np.uint8)) > 0] = 0

        # 2. Localize or validate pupil and limbus
        if geometry is None or not (geometry.pupil_detected and geometry.limbus_detected):
            detected_geom, geom_err = self._localize_geometry(gray, ui_mask, refine_boundary)
            if detected_geom is None:
                return PentacamDetectionResult(
                    valid=False,
                    status=PentacamDetectionStatus.NO_PUPIL if "pupil" in geom_err else PentacamDetectionStatus.NO_LIMBUS,
                    failure_reason=geom_err,
                    image_width=w,
                    image_height=h,
                    processing_time_ms=(time.perf_counter() - t0) * 1000.0,
                )
            geom = detected_geom
        else:
            geom = geometry

        if not valid_geometry(geom.pupil, geom.limbus, gray.shape):
            return PentacamDetectionResult(status=PentacamDetectionStatus.DEGENERATE,
                failure_reason="Invalid pupil/limbus geometry", image_width=w, image_height=h,
                processing_time_ms=(time.perf_counter() - t0) * 1000)

        if not extract_features:
            # Current-frame angular registration validates texture itself; it
            # does not consume these descriptors. Geometry-only is explicit.
            return PentacamDetectionResult(valid=True, status=PentacamDetectionStatus.OK,
                image_type=image_type, geometry=geom, image_width=w, image_height=h,
                quality=PentacamQuality.MARGINAL,
                confidence=min(geom.pupil.fit_quality, geom.limbus.fit_quality),
                processing_time_ms=(time.perf_counter() - t0) * 1000)

        # 3. Extract iris annulus mask
        annulus_mask = self._build_annulus_mask(gray.shape, geom, ui_mask)
        annulus_mask = cv2.bitwise_and(annulus_mask, registration_mask(image))

        # 4. Detect distinctive iris landmarks inside the annulus
        features, feature_metrics = self._extract_features(gray, geom, annulus_mask)

        # 5. Determine quality grade and status
        n_accepted = len(features)
        if n_accepted < self.min_features:
            status = PentacamDetectionStatus.INSUFFICIENT_FEATURES
            quality = PentacamQuality.POOR
            valid = False
            fail_reason = f"Only {n_accepted} features accepted (minimum {self.min_features})"
        else:
            status = PentacamDetectionStatus.OK
            valid = True
            fail_reason = ""
            if n_accepted >= 40 and feature_metrics["angular_coverage"] >= 0.70:
                quality = PentacamQuality.GOOD
            elif n_accepted >= 20:
                quality = PentacamQuality.ACCEPTABLE
            else:
                quality = PentacamQuality.MARGINAL

        confidence = float(np.clip(
            (n_accepted / 50.0) * 0.6 + feature_metrics["angular_coverage"] * 0.4,
            0.0, 1.0,
        )) if valid else 0.0

        feature_set = PentacamFeatureSet(
            features=features,
            num_candidates=feature_metrics["num_candidates"],
            num_accepted=n_accepted,
            angular_coverage_ratio=feature_metrics["angular_coverage"],
            largest_angular_gap_deg=feature_metrics["largest_gap_deg"],
        )

        hypotheses = anatomy_candidates(gray, geom, annulus_mask) if review_anatomy else []
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return PentacamDetectionResult(
            valid=valid,
            status=status,
            image_type=image_type,
            geometry=geom,
            feature_set=feature_set,
            image_width=w,
            image_height=h,
            coordinate_system="pentacam_pixel",
            quality=quality,
            confidence=confidence,
            failure_reason=fail_reason,
            processing_time_ms=dt_ms,
            anatomy_candidates=hypotheses,
        )

    def _build_ui_mask(self, gray: np.ndarray) -> np.ndarray:
        """Create a mask where valid ocular image is 255 and UI chrome/text is 0.

        Detects and suppresses:
            - Sharp horizontal/vertical UI line overlays and crosshairs
            - Bright annotation text (characters, numbers)
            - Pure black letterboxing/chrome margins
        """
        h, w = gray.shape[:2]
        usable = np.ones((h, w), dtype=np.uint8) * 255

        # Pure black border/chrome margin
        black_thresh = 5
        usable[gray <= black_thresh] = 0

        # Highly saturated synthetic pure white text / graphics (e.g. > 252)
        white_thresh = 252
        white_pixels = gray >= white_thresh
        if np.any(white_pixels):
            # Dilate text/lines to eliminate anti-aliased feather edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated_white = cv2.dilate(white_pixels.astype(np.uint8), kernel)
            usable[dilated_white > 0] = 0

        # Detect horizontal and vertical crosshair / reticle lines
        # Using morphological opening with thin long kernels
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        edges = cv2.Canny(gray, 100, 200)
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)

        lines_mask = cv2.bitwise_or(h_lines, v_lines)
        if np.any(lines_mask):
            lines_dilated = cv2.dilate(lines_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            usable[lines_dilated > 0] = 0

        return usable

    def _localize_geometry(
        self,
        gray: np.ndarray,
        ui_mask: np.ndarray,
        refine_boundary: bool = True,
    ) -> Tuple[Optional[PentacamGeometry], str]:
        """Automatically find pupil and limbus on Pentacam IR imagery.

        Uses Otsu thresholding + contour fitting + concentric circular search.
        """
        h, w = gray.shape[:2]

        # In IR Scheimpflug images, the pupil is the darkest central circular cavity
        blurred = cv2.GaussianBlur(gray, (9, 9), 2.0)

        # 1. Pupil localization
        # Pupil is dark: search lowest 15th percentile of intensity inside valid mask
        histogram = cv2.calcHist([blurred], [0], ui_mask, [256], [0, 256]).ravel()
        count = int(histogram.sum())
        if count == 0:
            return None, "No valid ocular pixels outside UI"
        # Exact linear percentile of uint8 values via a 256-bin histogram;
        # avoids allocating/sorting the valid pixel array on every frame.
        cumulative = np.cumsum(histogram, dtype=np.float64)
        rank = .15 * (count - 1)
        lo, hi = math.floor(rank), math.ceil(rank)
        low_value = np.searchsorted(cumulative, lo, side="right")
        high_value = np.searchsorted(cumulative, hi, side="right")
        p15 = float(low_value + (rank - lo) * (high_value - low_value))
        # Several dark thresholds prevent a low-contrast iris from joining the
        # pupil while retaining pupils above an absolute camera black level.
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        usable = cv2.compare(ui_mask, 0, cv2.CMP_GT)
        contours = []
        for threshold in sorted(set((20.0, 30.0, 40.0, max(p15, 25.0)))):
            pupil_bin = cv2.bitwise_and(cv2.compare(blurred, math.ceil(threshold) - 1, cv2.CMP_LE), usable)
            pupil_bin = cv2.morphologyEx(pupil_bin, cv2.MORPH_CLOSE, k_close)
            found, _ = cv2.findContours(pupil_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(found)
        best_pupil = None
        best_score = -1.0
        min_pupil_area = math.pi * (min(h, w) * 0.05) ** 2
        max_pupil_area = math.pi * (min(h, w) * 0.35) ** 2

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_pupil_area <= area <= max_pupil_area):
                continue
            if len(cnt) < 5:
                continue

            (cx, cy), (d1, d2), angle = cv2.fitEllipse(cnt)
            smaj = max(d1, d2) / 2.0
            smin = min(d1, d2) / 2.0
            eccentricity = math.sqrt(max(0.0, 1.0 - (smin / (smaj + 1e-6)) ** 2))
            if eccentricity > 0.65:
                continue  # Pupil should not be extremely oblong
            if smaj > min(h, w) * .30 or abs(area / (math.pi * smaj * smin) - 1) > .20:
                continue

            # Distance to image center
            center_dist = math.hypot(cx - w / 2.0, cy - h / 2.0)
            center_penalty = center_dist / (max(h, w) * 0.5)

            score = area / (1.0 + center_penalty * 3.0)
            if score > best_score:
                best_score = score
                best_pupil = EllipseParams(
                    center_x=float(cx),
                    center_y=float(cy),
                    semi_major=float(smaj),
                    semi_minor=float(smin),
                    angle_deg=float((angle + (90 if d1 < d2 else 0)) % 180),
                    fit_quality=0.90,
                    eccentricity=float(eccentricity),
                    circularity=float(smin / smaj),
                )

        if best_pupil is None:
            return None, "No measured pupil contour"

        # 2. Limbus localization
        # Limbus is approximately concentric with pupil, with radius 2.2x to 3.5x pupil radius
        pcx, pcy = best_pupil.center_x, best_pupil.center_y
        pr = best_pupil.radius

        min_limbus_r = max(pr * 1.45, min(h, w) * .25)
        max_limbus_r = min(min(pcx, w - pcx) * .95, min(h, w) * .57)

        if min_limbus_r >= max_limbus_r:
            return None, "Insufficient image extent for limbus"
        else:
            # Radial derivative search around pupil center
            # Lateral arcs are less occluded by eyelids than superior/inferior
            # arcs. The median rejects localized crypts and thin overlay lines.
            a = np.linspace(-.60, .60, 40)
            test_angles = np.concatenate((a, a + np.pi))
            r_samples = np.linspace(min_limbus_r, max_limbus_r, 128)
            xs = (pcx + r_samples[:, None] * np.cos(test_angles)).astype(np.float32)
            ys = (pcy + r_samples[:, None] * np.sin(test_angles)).astype(np.float32)
            samples = cv2.remap(blurred.astype(np.float32), xs, ys, cv2.INTER_LINEAR)
            samples = cv2.GaussianBlur(samples, (1, 9), 0)
            diffs = np.median(np.gradient(samples, axis=0), axis=1)
            best_r_idx = int(np.argmax(diffs[2:-2])) + 2
            if diffs[best_r_idx] < .12:
                return None, "No supported limbus intensity transition"
            limbus_r = float(r_samples[best_r_idx])

        best_limbus = EllipseParams(
            center_x=float(pcx),
            center_y=float(pcy),
            semi_major=float(limbus_r),
            semi_minor=float(limbus_r),
            angle_deg=0.0,
            fit_quality=0.40,
        )

        refinement = refine_limbus(gray, ui_mask, best_limbus) if refine_boundary else None
        refined = None
        support, residual = [], None
        method = "coarse_circle"
        if refinement is not None:
            refined, points, residual = refinement
            support = [tuple(map(float, point)) for point in points]
            method = "supported_ellipse"
        else:
            # This remains a coarse estimate, not a high-confidence full edge.
            best_limbus.fit_quality = .40

        geom = PentacamGeometry(
            pupil=best_pupil,
            limbus=best_limbus,
            refined_limbus=refined,
            pupil_detected=True,
            limbus_detected=True,
            pupil_radius_px=float(best_pupil.radius),
            limbus_radius_px=float(best_limbus.radius),
            pupil_limbus_ratio=float(best_pupil.radius / max(best_limbus.radius, 1e-6)),
            limbus_method=method, limbus_support_points=support,
            limbus_fit_residual_px=residual,
        )

        return geom, ""

    def _build_annulus_mask(
        self,
        shape: Tuple[int, int],
        geom: PentacamGeometry,
        ui_mask: np.ndarray,
    ) -> np.ndarray:
        """Create a boolean mask for the annular iris region, excluding UI overlays."""
        h, w = shape[:2]
        annulus = np.zeros((h, w), dtype=np.uint8)

        le = geom.refined_limbus or geom.limbus
        pe = geom.pupil

        if le is None or pe is None:
            return annulus

        # Outer limbus ellipse (inset)
        outer_smaj = int(round(le.semi_major * (1.0 - self.outer_inset_frac)))
        outer_smin = int(round(le.semi_minor * (1.0 - self.outer_inset_frac)))
        cv2.ellipse(
            annulus,
            (int(round(le.center_x)), int(round(le.center_y))),
            (max(1, outer_smaj), max(1, outer_smin)),
            le.angle_deg,
            0, 360,
            255, -1,
        )

        # Inner pupil ellipse (inset outwards)
        inner_smaj = int(round(pe.semi_major * (1.0 + self.inner_inset_frac)))
        inner_smin = int(round(pe.semi_minor * (1.0 + self.inner_inset_frac)))
        cv2.ellipse(
            annulus,
            (int(round(pe.center_x)), int(round(pe.center_y))),
            (max(1, inner_smaj), max(1, inner_smin)),
            pe.angle_deg,
            0, 360,
            0, -1,
        )

        # Combine with UI exclusion mask
        annulus = cv2.bitwise_and(annulus, ui_mask)
        return annulus

    def _extract_features(
        self,
        gray: np.ndarray,
        geom: PentacamGeometry,
        annulus_mask: np.ndarray,
    ) -> Tuple[List[PentacamFeature], Dict]:
        """Detect spatially separated texture keypoints; no anatomical classification."""
        h, w = gray.shape[:2]
        pe = geom.pupil
        le = geom.refined_limbus or geom.limbus

        if pe is None or le is None:
            return [], {"num_candidates": 0, "angular_coverage": 0.0, "largest_gap_deg": 360.0}

        pcx, pcy = pe.center_x, pe.center_y
        lcx, lcy = le.center_x, le.center_y

        # Enhance local contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Texture response: Laplacian of Gaussian
        laplacian = cv2.Laplacian(enhanced, cv2.CV_32F, ksize=3)
        abs_lap = np.abs(laplacian)

        # Precompute gradient magnitude and orientation once for the entire image
        gx_full = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
        gy_full = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
        mag_full, ori_full = cv2.cartToPolar(gx_full, gy_full, angleInDegrees=True)

        # Dynamic local contrast threshold derived from iris stroma
        iris_pixels = enhanced[annulus_mask > 0]
        if len(iris_pixels) < 50:
            return [], {"num_candidates": 0, "angular_coverage": 0.0, "largest_gap_deg": 360.0}

        p10 = float(np.percentile(iris_pixels, 10))
        p90 = float(np.percentile(iris_pixels, 90))
        iris_contrast_span = max(p90 - p10, 10.0)

        # Detect actual two-dimensional texture extrema instead of accepting
        # arbitrary lattice positions. Erosion keeps entire descriptor patches
        # away from glints, annotations and uncertain annulus edges.
        safe = cv2.erode(annulus_mask, np.ones((9, 9), np.uint8))
        safe[:max(0, int(le.center_y - .65 * le.semi_minor)), :] = 0
        corner_image = cv2.GaussianBlur(gray, (5, 5), 1.0)
        corner_response = cv2.cornerMinEigenVal(corner_image, blockSize=5)
        corners = cv2.goodFeaturesToTrack(corner_image,
            maxCorners=min(240, self.num_angles * 3), qualityLevel=.025,
            minDistance=max(5., le.radius * .035), mask=safe, blockSize=5)
        if corners is None:
            return [], {"num_candidates": 0, "angular_coverage": 0., "largest_gap_deg": 360.}
        fx, fy = corners[:, 0, 0], corners[:, 0, 1]
        angles = np.degrees(np.arctan2(fy-pcy, fx-pcx)) % 360
        distance = np.hypot(fx-pcx, fy-pcy)
        radial = np.clip((distance - pe.radius) / max(le.radius-pe.radius, 1), 0, 1)
        ix, iy = np.rint(fx).astype(int), np.rint(fy).astype(int)
        inside = (ix >= 3) & (iy >= 3) & (ix < w - 3) & (iy < h - 3)
        indices = np.flatnonzero(inside)
        num_candidates = len(corners)
        accepted_features = []
        if len(indices):
            dy, dx = np.mgrid[-3:4, -3:4]
            xx = ix[indices, None] + dx.ravel()
            yy = iy[indices, None] + dy.ravel()
            patches = enhanced[yy, xx]
            mean_lap = abs_lap[yy, xx].mean(axis=1)
            local_std = patches.std(axis=1)
            keep = (mean_lap >= self.min_contrast) & (local_std >= self.min_contrast * .7)
            # Reject patches that straddle glints, UI or annulus boundaries.
            keep &= (annulus_mask[yy, xx] > 0).mean(axis=1) >= .85
            bins = np.minimum((ori_full[yy, xx] / 22.5).astype(np.int32), 15)
            bins += np.arange(len(indices))[:, None] * 16
            hist = np.bincount(bins.ravel(), weights=mag_full[yy, xx].ravel(),
                               minlength=len(indices) * 16).reshape(-1, 16)
            hist /= np.linalg.norm(hist, axis=1, keepdims=True) + 1e-7
            confidence = np.clip(mean_lap / 40 + local_std / (2 * iris_contrast_span), .1, 1)
            for j in np.flatnonzero(keep):
                i = indices[j]
                accepted_features.append(PentacamFeature(
                    id=len(accepted_features), x=float(fx[i]), y=float(fy[i]),
                    angle_deg=float(angles[i]), radial_norm=float(radial[i]),
                    response=float(corner_response[iy[i], ix[i]] * 10000), confidence=float(confidence[j]),
                    valid=True, descriptor=hist[j].astype(np.float32)))

        # Compute coverage metrics
        metrics = self._compute_coverage_metrics(accepted_features)
        metrics["num_candidates"] = num_candidates

        return accepted_features, metrics

    def _compute_coverage_metrics(self, features: List[PentacamFeature]) -> Dict:
        """Compute angular coverage and largest angular gap."""
        if not features:
            return {"angular_coverage": 0.0, "largest_gap_deg": 360.0}

        angles = sorted([f.angle_deg for f in features])
        gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
        gaps.append((angles[0] + 360.0) - angles[-1])
        largest_gap = float(max(gaps))

        # 30-degree bin occupancy (12 bins)
        bins = set(int(a // 30.0) for a in angles)
        coverage_ratio = float(len(bins) / 12.0)

        return {
            "angular_coverage": coverage_ratio,
            "largest_gap_deg": largest_gap,
        }
