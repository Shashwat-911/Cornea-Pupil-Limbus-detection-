"""Bounded-resolution CPU session for one seated reference and one eye.

One instance per capture stream. Not thread-safe. Caller must supply confirmed
eye identity and an unmirrored camera coordinate convention.
"""
import copy
import time

import cv2
import numpy as np

from pupil_tracking.pentacam.cross_registration import CrossModalityRegistrationEngine
from pupil_tracking.pentacam.cross_system import CrossSystemRegistrationResult, RegistrationFailureKind
from pupil_tracking.pentacam.sitting import ocular_viewport
from pupil_tracking.utils.types import EyeDetectionResult, PupilDetection, LimbusDetection


class SittingRegistrationSession:
    def __init__(self, max_size=640, num_angles=720):
        if max_size < 256 or num_angles < 180:
            raise ValueError("max_size >= 256 and num_angles >= 180 required")
        self.max_size = max_size
        self.engine = CrossModalityRegistrationEngine(num_angles=num_angles)
        self.reference = None
        self.reference_detection = None
        self.eye_id = None
        self.laterality = None

    def _prepare(self, image, crop=True):
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0
                or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3)):
            raise ValueError("Expected nonempty uint8 grayscale or BGR image")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        x, y, w, h = ocular_viewport(gray) if crop else (0, 0, image.shape[1], image.shape[0])
        scale = min(1., self.max_size / max(w, h))
        size = (round(w * scale), round(h * scale))
        return cv2.resize(image[y:y+h, x:x+w], size, interpolation=cv2.INTER_AREA), (x, y, size[0]/w, size[1]/h)

    def set_reference(self, image, *, eye_id, laterality):
        self.reference = self.reference_detection = None
        self.eye_id = self.laterality = None
        if not eye_id or laterality not in ("OD", "OS"):
            raise ValueError("A nonempty eye_id and OD/OS laterality are required")
        prepared, _ = self._prepare(image)
        detection = self.engine.pentacam_detector.detect(prepared, refine_boundary=False)
        if not detection.valid:
            raise ValueError(f"Reference rejected: {detection.failure_reason}")
        self.reference = prepared.copy()
        self.reference.setflags(write=False)
        self.reference_detection = detection
        self.eye_id, self.laterality = str(eye_id), laterality
        self.engine._polar_cache_key = None
        self.engine._last_smooth_theta = None
        return copy.deepcopy(detection)

    def register(self, image, *, eye_id, laterality, detection=None, dynamic=False):
        """Detect seated frames automatically, or pass validated ELITA geometry.

        Supplied detection coordinates must match the full input image. Pure
        seated input is cropped automatically; ELITA cropping is caller-owned.
        Return time includes preprocessing and automatic current detection.
        """
        start = time.perf_counter()
        if self.reference is None:
            return CrossSystemRegistrationResult(failure=RegistrationFailureKind.MISSING_METADATA,
                                                   failure_reason="Set a valid seated reference first")
        if str(eye_id) != self.eye_id or laterality != self.laterality:
            return CrossSystemRegistrationResult(failure=RegistrationFailureKind.COORDINATE_MISMATCH,
                                                   failure_reason="Reference and current eye identity/laterality differ")
        current, (x, y, sx, sy) = self._prepare(image, crop=detection is None)
        if detection is None:
            detected = self.engine.pentacam_detector.detect(current, extract_features=False,
                                                           refine_boundary=False)
            if not detected.valid:
                return CrossSystemRegistrationResult(
                    failure=RegistrationFailureKind.NO_ELITA, failure_reason=detected.failure_reason,
                    processing_time_ms=(time.perf_counter() - start) * 1000)
            detection = EyeDetectionResult(
                pupil=PupilDetection(detected=True, ellipse=detected.geometry.pupil),
                limbus=LimbusDetection(detected=True, ellipse=detected.geometry.limbus))
        else:
            detection = copy.deepcopy(detection)
            for e in (detection.pupil.ellipse, detection.limbus.ellipse):
                if e is not None:
                    e.center_x = (e.center_x - x + .5) * sx - .5
                    e.center_y = (e.center_y - y + .5) * sy - .5
                    e.semi_major *= (sx + sy) / 2
                    e.semi_minor *= (sx + sy) / 2
        result = self.engine.register(self.reference, current, self.reference_detection,
                                      detection, laterality=laterality,
                                      mode="dynamic" if dynamic else "static")
        result.processing_time_ms = (time.perf_counter() - start) * 1000
        return result
