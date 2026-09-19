# pupil_tracking/core/validation.py
"""Cross-validation and rejection logic for pupil/limbus detection.

Validates physically impossible combinations (pupil outside limbus,
pupil larger than limbus, structures outside ring opening) and
rejects the less confident detection when violations are found.

Extracted from :class:`UnifiedDetector` during the Phase-3 refactoring.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from pupil_tracking.ml.postprocess import (
    validate_pupil_limbus_pair,
    RingSegmentationResult,
)
from pupil_tracking.core.deterministic_ring_detector import (
    RingDetectionResult,
    RingStatus,
)
from pupil_tracking.utils.types import (
    EyeDetectionResult,
    PupilDetection,
    LimbusDetection,
)

logger = logging.getLogger(__name__)


def cross_validate_and_reject(
    result: EyeDetectionResult,
    ring_result: Optional[RingDetectionResult] = None,
) -> EyeDetectionResult:
    """Cross-validate pupil and limbus.

    REJECT physically impossible combinations:
    * pupil centre outside the limbus
    * pupil radius larger than limbus radius
    * pupil or limbus outside ring opening

    Plan change: ratio warning threshold relaxed 0.80 -> 0.85
    for dilated surgical pupils under anaesthesia.
    """
    pe = result.pupil.ellipse
    le = result.limbus.ellipse

    if pe is None or le is None:
        return result

    # --- Centre-offset check ---
    dx = pe.center_x - le.center_x
    dy = pe.center_y - le.center_y
    offset = math.sqrt(dx * dx + dy * dy)

    if le.radius > 0:
        offset_ratio = offset / le.radius

        if offset_ratio > 1.0:
            logger.warning(
                "Pupil centre outside limbus (ratio=%.2f). "
                "Rejecting less confident.",
                offset_ratio,
            )
            if result.pupil.confidence > result.limbus.confidence:
                result.limbus = LimbusDetection()
                result.alerts.append("Limbus rejected: pupil centre outside limbus")
            else:
                result.pupil = PupilDetection()
                result.alerts.append("Pupil rejected: centre outside limbus")
            return result

        if offset_ratio > 0.5:
            result.alerts.append(
                f"Large pupil-limbus offset: {offset_ratio:.2f} of limbus radius"
            )
            result.pupil.confidence *= 0.8
            result.limbus.confidence *= 0.8

    # --- Size-ratio check ---
    if le.radius > 0 and pe.radius > 0:
        ratio = pe.radius / le.radius

        if ratio > 1.0:
            logger.warning(
                "Pupil larger than limbus (ratio=%.2f).",
                ratio,
            )
            if result.pupil.confidence > result.limbus.confidence:
                result.limbus = LimbusDetection()
                result.alerts.append("Limbus rejected: smaller than pupil")
            else:
                result.pupil = PupilDetection()
                result.alerts.append("Pupil rejected: larger than limbus")
            return result

        # PLAN CHANGE: 0.80 -> 0.85 for dilated surgical pupils
        if ratio > 0.85:
            result.alerts.append(f"Unusual pupil/limbus ratio: {ratio:.2f}")
            result.pupil.confidence *= 0.7
            result.limbus.confidence *= 0.7

    # --- Ring containment check ---
    if (
        ring_result is not None
        and ring_result.status in (RingStatus.PRESENT, RingStatus.PARTIAL)
        and ring_result.ring_center is not None
        and ring_result.ring_radius is not None
    ):
        ring_cx, ring_cy = ring_result.ring_center
        ring_r = ring_result.ring_radius

        pupil_dist = math.sqrt(
            (pe.center_x - ring_cx) ** 2 + (pe.center_y - ring_cy) ** 2
        )
        if pupil_dist > ring_r * 0.85:
            result.alerts.append(
                f"Pupil centre near/outside ring opening: "
                f"dist={pupil_dist:.1f} vs ring_r={ring_r:.1f}"
            )
            result.pupil.confidence *= 0.7

        limbus_extent = (
            math.sqrt((le.center_x - ring_cx) ** 2 + (le.center_y - ring_cy) ** 2)
            + le.radius
        )

        if limbus_extent > ring_r * 1.1:
            result.alerts.append(
                f"Limbus extends outside ring: "
                f"extent={limbus_extent:.1f} vs ring_r={ring_r:.1f}"
            )
            result.limbus.confidence *= 0.7

        if le.radius > ring_r * 0.85:
            result.alerts.append(
                f"Limbus radius ({le.radius:.1f}) close to ring "
                f"radius ({ring_r:.1f}) — may be detecting ring as limbus"
            )
            result.limbus.confidence *= 0.5

    # --- Geometric cross-validation from postprocess module ---
    ring_seg = None
    if ring_result is not None and ring_result.status == RingStatus.PRESENT:
        ring_seg = RingSegmentationResult(
            detected=True,
            center=ring_result.ring_center,
            radius=ring_result.ring_radius,
        )

    valid, issues = validate_pupil_limbus_pair(pe, le, ring=ring_seg)
    for issue in issues:
        if issue not in result.alerts:
            result.alerts.append(f"Cross-validation: {issue}")

    return result
