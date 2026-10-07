import json

import cv2
import numpy as np
import pytest

from pupil_tracking.pentacam.training_data import load_pair_manifest


def manifest(tmp_path, **override):
    cv2.imwrite(str(tmp_path / 'eye.png'), np.full((32, 32, 3), 120, np.uint8))
    row = dict(patient_id='p1', eye_id='p1_OD', laterality='OD',
               reference='eye.png', current='eye.png', rotation_deg=3.5,
               split='train', label_source='expert')
    row.update(override)
    path = tmp_path / 'pairs.jsonl'
    path.write_text(json.dumps(row)+'\n', encoding='utf-8')
    return path, row


def test_load_images_and_labels(tmp_path):
    path, _ = manifest(tmp_path)
    pair, = load_pair_manifest(path)
    assert pair.rotation_deg == 3.5
    assert all(image.shape == (32, 32, 3) for image in pair.load_images())


@pytest.mark.parametrize('override', [dict(rotation_deg=float('nan')),
    dict(rotation_deg=True), dict(label_source='prediction'), dict(laterality='left'),
    dict(reference='missing.png'), dict(split='unknown')])
def test_reject_invalid_training_labels(tmp_path, override):
    path, _ = manifest(tmp_path, **override)
    with pytest.raises(ValueError, match='Manifest line 1'):
        load_pair_manifest(path)


@pytest.mark.parametrize('change', [dict(split='test'),
    dict(split='test', patient_id='p2', eye_id='p2_OD'), dict(laterality='OS')])
def test_reject_split_leakage_and_eye_conflicts(tmp_path, change):
    path, row = manifest(tmp_path)
    row.update(change)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='Manifest line 2'):
        load_pair_manifest(path)
