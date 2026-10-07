"""Framework-independent, leakage-checked manifests for paired-eye training.

JSONL rows use relative image paths and pseudonymous patient/eye IDs. Splits
are assigned by patient, not frame, so both eyes and all visits stay together.
Synthetic examples are explicitly labelled and are not clinical test evidence.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import cv2


@dataclass(frozen=True)
class TrainingPair:
    patient_id: str
    eye_id: str
    laterality: str
    reference: Path
    current: Path
    rotation_deg: float
    split: str
    label_source: str

    def load_images(self):
        """Return original BGR pixels; apply training augmentation afterwards."""
        images = tuple(cv2.imread(str(p), cv2.IMREAD_COLOR)
                       for p in (self.reference, self.current))
        if any(image is None for image in images):
            raise ValueError("Training pair contains an unreadable image")
        return images


def load_pair_manifest(path):
    """Validate all rows and split boundaries before returning training pairs.

    rotation_deg uses CCW image rotation, with unmirrored reference/current
    images of the same eye. Angles must be verified labels, never predictions.
    Paths are resolved relative to the manifest. No patient data is uploaded.
    """
    path = Path(path).resolve()
    pairs, patient_splits, image_splits, eye_owners = [], {}, {}, {}
    image_hashes = {}
    required = set(TrainingPair.__dataclass_fields__)
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != required:
                raise ValueError('Expected exactly the documented TrainingPair fields')
            for field in ('patient_id', 'eye_id', 'reference', 'current'):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError(f'{field} must be a nonempty string')
            if row['laterality'] not in ('OD', 'OS'):
                raise ValueError('laterality must be OD or OS')
            if row['split'] not in ('train', 'validation', 'test'):
                raise ValueError('Unknown split')
            if row['label_source'] not in ('expert', 'synthetic'):
                raise ValueError('label_source must be expert or synthetic')
            angle = row['rotation_deg']
            if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not math.isfinite(angle) or abs(angle) >= 180:
                raise ValueError('rotation_deg must be finite and between -180 and 180')
            owner = (row['patient_id'], row['laterality'])
            if eye_owners.setdefault(row['eye_id'], owner) != owner:
                raise ValueError('Eye identity has conflicting patient or laterality')
            if patient_splits.setdefault(row['patient_id'], row['split']) != row['split']:
                raise ValueError('Patient leakage across splits')
            for field in ('reference', 'current'):
                image = (path.parent / row[field]).resolve()
                if not image.is_file():
                    raise ValueError(f'{field} image is missing')
                if image not in image_hashes:
                    image_hashes[image] = hashlib.sha256(image.read_bytes()).digest()
                if image_splits.setdefault(image_hashes[image], row['split']) != row['split']:
                    raise ValueError('Image leakage across splits')
                row[field] = image
            pairs.append(TrainingPair(**row))
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Manifest line {line_number}: {exc}') from exc
    if not pairs:
        raise ValueError('Pair manifest is empty')
    return pairs


class CyclotorsionAugmentor:
    """Clinical data augmentations for polar iris strips."""

    def __init__(
        self,
        max_delta_deg: float = 10.0,
        illumination_prob: float = 0.5,
        occlusion_prob: float = 0.5,
        noise_prob: float = 0.4,
        dilation_prob: float = 0.4,
    ) -> None:
        self.max_delta_deg = max_delta_deg
        self.illumination_prob = illumination_prob
        self.occlusion_prob = occlusion_prob
        self.noise_prob = noise_prob
        self.dilation_prob = dilation_prob

    def augment_strip_pair(
        self,
        ref_strip,
        curr_strip,
        mask_ref,
        mask_curr,
        rotation_deg: float,
        rng=None,
    ):
        """Apply physiologically plausible augmentations to a pair of polar strips."""
        import numpy as np

        if rng is None:
            rng = np.random.default_rng()

        ref = ref_strip.copy()
        curr = curr_strip.copy()
        m_ref = mask_ref.copy()
        m_curr = mask_curr.copy()
        h, w = curr.shape[:2]

        # 1. Additional circular shift along angular dimension
        delta_deg = float(rng.uniform(-self.max_delta_deg, self.max_delta_deg))
        shift_px = int(round(delta_deg * w / 360.0))
        if shift_px != 0:
            curr = np.roll(curr, shift_px, axis=1)
            m_curr = np.roll(m_curr, shift_px, axis=1)
            rotation_deg = float((rotation_deg + delta_deg + 180.0) % 360.0 - 180.0)

        # 2. Illumination / contrast variation
        if rng.random() < self.illumination_prob:
            gamma = float(rng.uniform(0.75, 1.35))
            gain = float(rng.uniform(0.8, 1.2))
            bias = float(rng.uniform(-20, 20))
            norm = (curr.astype(np.float32) / 255.0) ** gamma
            curr = np.clip(norm * 255.0 * gain + bias, 0, 255).astype(np.uint8)

        # 3. Radial stretch (simulating pupillary dilation)
        if rng.random() < self.dilation_prob:
            stretch = float(rng.uniform(0.9, 1.1))
            new_h = max(8, int(round(h * stretch)))
            res = cv2.resize(curr, (w, new_h), interpolation=cv2.INTER_LINEAR)
            m_res = cv2.resize(m_curr, (w, new_h), interpolation=cv2.INTER_NEAREST)
            if new_h >= h:
                curr = res[:h, :]
                m_curr = m_res[:h, :]
            else:
                curr = np.pad(res, ((0, h - new_h), (0, 0)), mode="edge")
                m_curr = np.pad(m_res, ((0, h - new_h), (0, 0)), mode="constant", constant_values=0)

        # 4. Sector / eyelid occlusions
        if rng.random() < self.occlusion_prob:
            occ_len = int(rng.uniform(0.1, 0.35) * w)
            occ_start = int(rng.integers(0, w))
            occ_indices = (np.arange(occ_start, occ_start + occ_len)) % w
            m_curr[:, occ_indices] = 0

        # 5. Sensor noise
        if rng.random() < self.noise_prob:
            noise_sigma = float(rng.uniform(1.0, 5.0))
            noise = rng.normal(0, noise_sigma, curr.shape).astype(np.float32)
            curr = np.clip(curr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return ref, curr, m_ref, m_curr, rotation_deg


def get_pytorch_dataset(
    manifest_path,
    split: str = "train",
    *,
    augment: bool = False,
    num_angles: int = 720,
    num_radial: int = 64,
):
    """Load a PyTorch Dataset for paired cyclotorsion training.

    PyTorch is imported lazily inside this function to ensure runtime inference
    remains completely independent of deep learning frameworks.
    """
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for training dataset support. Install via: pip install torch"
        ) from exc

    import numpy as np
    from pupil_tracking.pentacam.detector import PentacamIrisDetector
    from pupil_tracking.pentacam.sitting import registration_mask
    from pupil_tracking.registration.polar import PolarUnwrapper

    all_pairs = load_pair_manifest(manifest_path)
    pairs = [p for p in all_pairs if p.split == split]
    if not pairs:
        raise ValueError(f"No pairs found for split '{split}' in manifest")

    detector = PentacamIrisDetector()
    unwrapper = PolarUnwrapper(num_angles=num_angles, num_radial=num_radial)
    augmentor = CyclotorsionAugmentor() if augment else None

    class _CyclotorsionPairDataset(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            pair = self.items[idx]
            ref_bgr, curr_bgr = pair.load_images()

            # Unwrap reference
            det_ref = detector.detect(ref_bgr, refine_boundary=False)
            det_curr = detector.detect(curr_bgr, refine_boundary=False)
            if not det_ref.valid or not det_curr.valid:
                # Blank fallback if detection fails
                strip_ref = np.zeros((num_radial, num_angles), dtype=np.uint8)
                mask_ref = np.zeros((num_radial, num_angles), dtype=np.uint8)
                strip_curr = np.zeros((num_radial, num_angles), dtype=np.uint8)
                mask_curr = np.zeros((num_radial, num_angles), dtype=np.uint8)
                angle = pair.rotation_deg
            else:
                p_ref = unwrapper.unwrap(
                    ref_bgr,
                    pupil_center=(det_ref.geometry.pupil.center_x, det_ref.geometry.pupil.center_y),
                    pupil_axes=(det_ref.geometry.pupil.semi_major, det_ref.geometry.pupil.semi_minor),
                    pupil_angle_deg=det_ref.geometry.pupil.angle_deg,
                    limbus_center=(det_ref.geometry.limbus.center_x, det_ref.geometry.limbus.center_y),
                    limbus_axes=(det_ref.geometry.limbus.semi_major, det_ref.geometry.limbus.semi_minor),
                    limbus_angle_deg=det_ref.geometry.limbus.angle_deg,
                    validity_mask=registration_mask(ref_bgr),
                )
                p_curr = unwrapper.unwrap(
                    curr_bgr,
                    pupil_center=(det_curr.geometry.pupil.center_x, det_curr.geometry.pupil.center_y),
                    pupil_axes=(det_curr.geometry.pupil.semi_major, det_curr.geometry.pupil.semi_minor),
                    pupil_angle_deg=det_curr.geometry.pupil.angle_deg,
                    limbus_center=(det_curr.geometry.limbus.center_x, det_curr.geometry.limbus.center_y),
                    limbus_axes=(det_curr.geometry.limbus.semi_major, det_curr.geometry.limbus.semi_minor),
                    limbus_angle_deg=det_curr.geometry.limbus.angle_deg,
                    validity_mask=registration_mask(curr_bgr),
                )
                strip_ref, mask_ref = p_ref.image, p_ref.mask
                strip_curr, mask_curr = p_curr.image, p_curr.mask
                angle = pair.rotation_deg

            if augmentor is not None:
                strip_ref, strip_curr, mask_ref, mask_curr, angle = augmentor.augment_strip_pair(
                    strip_ref, strip_curr, mask_ref, mask_curr, angle
                )

            # Convert to float32 tensors (1, H, W) normalized [-1, 1]
            t_ref = torch.from_numpy(strip_ref.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
            t_curr = torch.from_numpy(strip_curr.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
            t_mask_ref = torch.from_numpy((mask_ref > 0).astype(np.float32)).unsqueeze(0)
            t_mask_curr = torch.from_numpy((mask_curr > 0).astype(np.float32)).unsqueeze(0)

            return {
                "ref": t_ref,
                "curr": t_curr,
                "mask_ref": t_mask_ref,
                "mask_curr": t_mask_curr,
                "target_deg": torch.tensor(angle, dtype=torch.float32),
                "patient_id": pair.patient_id,
                "eye_id": pair.eye_id,
                "laterality": pair.laterality,
            }

    return _CyclotorsionPairDataset(pairs)

