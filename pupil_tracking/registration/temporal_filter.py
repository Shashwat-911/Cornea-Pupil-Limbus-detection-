"""
Temporal Kalman filtering for cyclotorsion tracking in surgical video streams.

Stabilizes intra-operative torsion measurements:
    - State: [torsion_angle_deg, angular_velocity_deg_per_s]
    - Rejects single-frame outliers (surgical instrument occlusions, partial blinks)
    - Coasting across temporary dropouts without discontinuous jumps
    - Sub-degree real-time tracking with low phase lag
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CyclotorsionKalmanFilter:
    """1D Constant-Velocity Kalman Filter for cyclotorsion tracking.

    Parameters
    ----------
    process_noise_std : float
        Expected physiological ocular torsional acceleration (deg/s^2).
    measurement_noise_std : float
        Baseline measurement uncertainty at 1.0 confidence (degrees).
    outlier_threshold_sigma : float
        Innovation Mahalanobis distance threshold to reject outliers.
    """

    def __init__(
        self,
        process_noise_std: float = 2.0,
        measurement_noise_std: float = 0.25,
        outlier_threshold_sigma: float = 3.5,
    ):
        self.q_std = process_noise_std
        self.r_std = measurement_noise_std
        self.outlier_sigma = outlier_threshold_sigma

        # State: [angle, velocity]
        self.x = np.zeros(2, dtype=np.float64)
        # Covariance: P
        self.P = np.diag([1.0, 4.0]).astype(np.float64)

        self._last_timestamp: Optional[float] = None
        self._initialized = False
        self._consecutive_rejects = 0

    def reset(self) -> None:
        """Reset filter state."""
        self.x = np.zeros(2, dtype=np.float64)
        self.P = np.diag([1.0, 4.0]).astype(np.float64)
        self._last_timestamp = None
        self._initialized = False
        self._consecutive_rejects = 0

    def update(
        self,
        measured_deg: Optional[float],
        confidence: float,
        timestamp: Optional[float] = None,
    ) -> Tuple[float, float, bool]:
        """Update tracker with a new cyclotorsion measurement.

        Parameters
        ----------
        measured_deg : float or None
            Raw measured torsion in degrees (or None if detection failed).
        confidence : float
            Measurement confidence in [0, 1].
        timestamp : float, optional
            Monotonic frame timestamp in seconds (defaults to time.perf_counter()).

        Returns
        -------
        (filtered_deg, angular_velocity, accepted) : tuple
            filtered_deg: smoothed cyclotorsion angle
            angular_velocity: estimated rate of twist in deg/s
            accepted: True if measurement was accepted, False if rejected/coasted
        """
        now = timestamp if timestamp is not None else time.perf_counter()

        if self._last_timestamp is None:
            dt = 1.0 / 30.0  # Assume nominal 30 fps for first frame
        else:
            dt = max(now - self._last_timestamp, 0.001)
            dt = min(dt, 0.5)  # Cap large gaps to prevent divergence
        self._last_timestamp = now

        # ── 1. Predict Step ─────────────────────────────────────────────
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=np.float64)

        # Continuous white noise acceleration model for Q
        q = self.q_std ** 2
        Q = np.array([
            [0.25 * (dt ** 4) * q, 0.5 * (dt ** 3) * q],
            [0.5 * (dt ** 3) * q, (dt ** 2) * q]
        ], dtype=np.float64)

        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        # First frame initialization
        if not self._initialized:
            if measured_deg is not None and confidence >= 0.3:
                self.x = np.array([measured_deg, 0.0], dtype=np.float64)
                self.P = np.diag([self.r_std ** 2, 4.0]).astype(np.float64)
                self._initialized = True
                return float(self.x[0]), float(self.x[1]), True
            else:
                return 0.0, 0.0, False

        # If measurement is missing or zero confidence, coast on prediction
        if measured_deg is None or confidence < 0.15:
            self.x = x_pred
            self.P = P_pred
            return float(self.x[0]), float(self.x[1]), False

        # ── 2. Update Step ──────────────────────────────────────────────
        H = np.array([[1.0, 0.0]], dtype=np.float64)

        # Adaptive measurement noise: R scales inversely with confidence
        safe_conf = max(confidence, 0.1)
        R_val = (self.r_std / safe_conf) ** 2
        R = np.array([[R_val]], dtype=np.float64)

        # Innovation (residual)
        y = measured_deg - (H @ x_pred)[0]

        # Circular angle difference wrapping [-180, 180]
        y = (y + 180.0) % 360.0 - 180.0

        # Innovation covariance
        S = (H @ P_pred @ H.T + R)[0, 0]

        # Mahalanobis outlier rejection
        mahalanobis = abs(y) / math.sqrt(max(S, 1e-8))

        if mahalanobis > self.outlier_sigma and self._consecutive_rejects < 5:
            # Reject outlier: coast on prediction
            self._consecutive_rejects += 1
            self.x = x_pred
            self.P = P_pred
            logger.debug("Torsion outlier rejected: resid=%.2f°, sigma=%.2f", y, mahalanobis)
            return float(self.x[0]), float(self.x[1]), False

        # Accepted measurement
        self._consecutive_rejects = 0
        K = (P_pred @ H.T) / S
        self.x = x_pred + (K.flatten() * y)
        self.P = (np.eye(2) - K @ H) @ P_pred

        return float(self.x[0]), float(self.x[1]), True
