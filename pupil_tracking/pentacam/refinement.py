"""Image-supported limbus refinement; no anatomical feature classification."""
import cv2
import numpy as np

from pupil_tracking.utils.types import EllipseParams
from pupil_tracking.pentacam.types import PentacamFeature


def refine_limbus(gray, mask, seed):
    """Fit a constrained ellipse to independent radial transitions.

    The superior sector is excluded because eyelid edges can look like limbus.
    Missing sectors are extrapolated, never reported as observed boundaries.
    Returns None when the image cannot support an independent fit.
    """
    radius = seed.radius
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    offsets = np.arange(-round(.22 * radius), round(.22 * radius) + 1, dtype=np.float32)
    radii = radius + offsets[:, None]
    xs = (seed.center_x + radii * np.cos(angles)).astype(np.float32)
    ys = (seed.center_y + radii * np.sin(angles)).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 2.5)
    samples = cv2.remap(smooth.astype(np.float32), xs, ys, cv2.INTER_LINEAR)
    usable = cv2.remap(mask, xs, ys, cv2.INTER_NEAREST) > 0
    step = max(3, round(radius * .025))
    gradient = (samples[2 * step:] - samples[:-2 * step]) / (2 * step)
    eligible = usable[2 * step:] & usable[:-2 * step] & usable[step:-step]
    # A broad proximity prior prevents isolated distant scleral edges winning.
    prior = np.exp(-.5 * (offsets[step:-step] / (.14 * radius)) ** 2)[:, None]
    response = np.where(eligible, gradient * prior, -1)
    indices = np.argmax(response, axis=0)
    cols = np.arange(len(angles))
    strength = gradient[indices, cols]
    rr = radius + offsets[indices + step]
    selected = (strength > .45) & (response[indices, cols] > .30)
    selected &= (np.sin(angles) > -.55)
    selected &= (indices > 0) & (indices < len(gradient)-1)
    points = np.column_stack((seed.center_x + rr * np.cos(angles),
                              seed.center_y + rr * np.sin(angles))).astype(np.float32)
    keep = selected.copy()
    ellipse = None
    for _ in range(4):
        if keep.sum() < 35:
            return None
        (cx, cy), (d1, d2), angle = cv2.fitEllipse(points[keep])
        major, minor = max(d1, d2) / 2, min(d1, d2) / 2
        angle = (angle + (90 if d1 < d2 else 0)) % 180
        if (not np.isfinite([cx, cy, major, minor, angle]).all()
                or major > radius * 1.22 or minor < radius * .78
                or minor / major < .78
                or np.hypot(cx-seed.center_x, cy-seed.center_y) > radius * .18):
            return None
        t = np.deg2rad(angle)
        dx, dy = points[:, 0]-cx, points[:, 1]-cy
        normalized = np.sqrt(((dx*np.cos(t)+dy*np.sin(t))/major)**2
                             + ((-dx*np.sin(t)+dy*np.cos(t))/minor)**2)
        residual = np.abs(normalized - 1) * np.sqrt(major * minor)
        keep = selected & (residual < max(2., radius * .025))
        ellipse = EllipseParams(center_x=float(cx), center_y=float(cy),
                                semi_major=float(major), semi_minor=float(minor),
                                angle_deg=float(angle), fit_quality=0.,
                                eccentricity=float(np.sqrt(1-(minor/major)**2)),
                                circularity=float(minor/major))
    # Require support on both sides and the inferior arc to constrain centre.
    if (keep.sum() < 45 or sum(keep & (np.cos(angles) > .65)) < 10
            or sum(keep & (np.cos(angles) < -.65)) < 10
            or sum(keep & (np.sin(angles) > .65)) < 10):
        return None
    error = float(np.median(residual[keep]))
    ellipse.fit_quality = float(np.clip(keep.sum() / selected.sum() * np.exp(-error/3), 0, 1))
    ellipse.fit_rms_residual = float(np.sqrt(np.mean(residual[keep]**2)))
    ellipse.num_contour_points = int(keep.sum())
    return ellipse, points[keep], error


def anatomy_candidates(gray, geometry, mask):
    """Unverified morphology hypotheses for expert review, not registration.

    Dark compact depressions suggest crypt-like candidates. Peripheral dark
    ridges with approximately tangential orientation suggest furrow-like
    candidates. IR appearance alone cannot establish anatomical identity.
    """
    pupil, limbus = geometry.pupil, geometry.refined_limbus or geometry.limbus
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    theta = np.arctan2(yy-limbus.center_y, xx-limbus.center_x)
    radial = (np.hypot(xx-pupil.center_x, yy-pupil.center_y)-pupil.radius) / max(1, limbus.radius-pupil.radius)
    safe = cv2.erode(mask, np.ones((15, 15), np.uint8)) > 0
    safe &= yy > limbus.center_y - .65 * limbus.semi_minor
    responses = {"crypt_like_candidate": np.zeros(gray.shape, np.float32),
                 "furrow_like_candidate": np.zeros(gray.shape, np.float32)}
    for sigma in (1.5, 3., 4.5):
        smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
        coarse = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma * 2)
        hxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3) * sigma**2 / 4
        hyy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3) * sigma**2 / 4
        hxy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3) * sigma**2 / 4
        spread = np.sqrt((hxx-hyy)**2+4*hxy*hxy)
        high, low = (hxx+hyy+spread)/2, (hxx+hyy-spread)/2
        direction = .5 * np.arctan2(2*hxy, hxx-hyy)
        dark = (coarse-smooth > 3) & safe
        compact = dark & (low > high * .25) & (high > 2) & (radial < .85)
        ridge = dark & (np.abs(low) < high * .25) & (high > 2) & (radial > .50)
        ridge &= np.abs(np.cos(direction-theta)) > .85
        for kind, valid in (("crypt_like_candidate", compact), ("furrow_like_candidate", ridge)):
            responses[kind] = np.maximum(responses[kind], np.where(valid, high, 0))
    features = []
    for kind, response in responses.items():
        peaks = (response == cv2.dilate(response, np.ones((9, 9), np.uint8))) & (response > 2)
        ys, xs = np.nonzero(peaks)
        selected = []
        for i in np.argsort(response[ys, xs])[::-1]:
            x, y = int(xs[i]), int(ys[i])
            if any((x-px)**2+(y-py)**2 < 12**2 for px, py in selected):
                continue
            selected.append((x, y))
            features.append(PentacamFeature(id=len(features), x=float(x), y=float(y),
                angle_deg=float(np.degrees(theta[y, x]) % 360), radial_norm=float(radial[y, x]),
                response=float(response[y, x]), confidence=0., kind=kind, anatomy_verified=False))
            if len(selected) == 24:
                break
    return features
