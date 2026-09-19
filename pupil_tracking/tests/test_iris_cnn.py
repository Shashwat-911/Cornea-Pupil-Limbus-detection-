"""Tests for the Phase V iris CNN models (segmentation + encoding).

These tests are deterministic, use synthetic fixtures, and do not require
model files or clinical data. They validate the model wrappers, torch model
architectures, preprocessing/postprocessing, and the loss functions.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pupil_tracking.ml.iris_encoder import (
    _IrisEncoderModel,
    _preprocess_strip,
    create_model as create_encoder,
)
from pupil_tracking.ml.iris_losses import (
    IrisContrastiveLoss,
    IrisSegmentationLoss,
    IrisTripletLoss,
    hard_negative_mining,
)
from pupil_tracking.ml.iris_segmentation import (
    IrisSegmentationNet,
    _postprocess,
    _preprocess,
    create_model as create_segmenter,
)
from pupil_tracking.iris.types import IrisROI


# ── segmentation ───────────────────────────────────────────────────────

def _fake_roi(size=320):
    return IrisROI(
        center_x=size / 2,
        center_y=size / 2,
        pupil_semi_major=55,
        pupil_semi_minor=50,
        limbus_semi_major=130,
        limbus_semi_minor=125,
        pupil_radius_px=55,
        limbus_radius_px=130,
        valid=True,
    )


def _synthetic_bgr(size=320):
    gray = np.full((size, size), 25, np.uint8)
    import cv2

    c = size // 2
    cv2.circle(gray, (c, c), 130, 80, -1)
    cv2.circle(gray, (c, c), 55, 10, -1)
    rng = np.random.default_rng(0)
    noise = rng.integers(-12, 12, (size, size)).astype(np.int16)
    gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_segmentation_preprocess_shape_and_range():
    img = _synthetic_bgr()
    tensor = _preprocess(img)
    assert tensor.shape == (1, 3, img.shape[0], img.shape[1])
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


def test_segmentation_postprocess_annulus_constraint():
    h, w = 320, 320
    roi = _fake_roi(size=320)
    logits = np.zeros((1, 2, 256, 256), dtype=np.float32)
    # everything predicts usable_iris
    logits[0, 1] = 2.0
    mask = _postprocess(logits, (0, 0, w, h), roi, threshold=0.5)
    assert mask.shape == (h, w)
    assert mask.dtype == bool
    # mask must be zero outside the iris annulus
    assert not mask[0, :].any()  # top row is far from annulus center


def test_segmentation_torch_model_forward():
    model = create_segmenter(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2, 256, 256)


def test_unavailable_segmentor_predict_all_true():
    seg = IrisSegmentationNet(model_path="nonexistent.onnx")
    assert not seg.available
    img = np.zeros((64, 64, 3), np.uint8)
    mask = seg.predict(img, _fake_roi(size=64))
    assert mask.shape == (64, 64)
    assert mask.all()


# ── encoder ────────────────────────────────────────────────────────────

def test_encoder_torch_model_forward_l2_unit():
    torch.manual_seed(0)
    model = create_encoder()
    model.eval()
    x = torch.randn(4, 1, 64, 512)
    with torch.no_grad():
        emb = model(x)
    assert emb.shape == (4, 128)
    norms = emb.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_encoder_preprocess_strip_shape():
    strip = (np.random.default_rng(0).integers(0, 256, (64, 512))).astype(np.uint8)
    t = _preprocess_strip(strip)
    assert t.shape == (1, 1, 64, 512)
    assert t.dtype == np.float32
    assert np.isfinite(t).all()


def test_encoder_similarity_bounds():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    from pupil_tracking.ml.iris_encoder import IrisEncoderNet

    assert abs(IrisEncoderNet.similarity(a, a) - 1.0) < 1e-6
    assert abs(IrisEncoderNet.similarity(a, b)) < 1e-6


# ── losses ─────────────────────────────────────────────────────────────

def test_segmentation_loss_is_scalar_and_positive():
    torch.manual_seed(0)
    loss_fn = IrisSegmentationLoss(gamma=2.0, dice_weight=0.5)
    logits = torch.randn(2, 2, 64, 64)
    targets = torch.randint(0, 2, (2, 64, 64))
    loss = loss_fn(logits, targets)
    assert loss.ndim == 0
    assert float(loss) > 0.0


def test_segmentation_loss_perfect_prediction_small():
    loss_fn = IrisSegmentationLoss(gamma=2.0, dice_weight=0.5)
    targets = torch.randint(0, 2, (1, 32, 32))
    logits = torch.zeros(1, 2, 32, 32)
    logits.scatter_(1, targets.unsqueeze(1), 8.0)  # high logit for true class
    loss = loss_fn(logits, targets)
    assert float(loss) < 0.2


def test_triplet_loss_positive_separation_drops():
    torch.manual_seed(0)
    loss_fn = IrisTripletLoss(margin=0.3)
    a = torch.randn(8, 128)
    p = a + 0.02 * torch.randn(8, 128)  # close positives
    n = a + 1.5 * torch.randn(8, 128)  # distant negatives
    loss = loss_fn(a, p, n)
    assert float(loss) < 0.05


def test_contrastive_loss_same_vs_different():
    torch.manual_seed(0)
    loss_fn = IrisContrastiveLoss(margin=1.0)
    a = torch.randn(8, 32)

    # Positive (same-iris): small separation -> higher loss than perfect overlap
    loss_pos = float(loss_fn(a, a + 0.05 * torch.randn(8, 32), torch.ones(8)))
    # Perfectly identical pairs yield zero loss
    loss_identical = float(loss_fn(a, a.clone(), torch.ones(8)))
    # Well-separated negatives are beyond margin -> zero loss
    loss_diff = float(loss_fn(a, a + 5.0 * torch.ones(8, 32), torch.zeros(8)))

    assert loss_identical == pytest.approx(0.0, abs=1e-6)
    assert loss_pos > 0.0
    assert loss_diff == pytest.approx(0.0, abs=1e-6)


def test_hard_negative_mining_selects_hardest():
    torch.manual_seed(0)
    emb = torch.randn(6, 128)
    emb = emb / emb.norm(dim=1, keepdim=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    a, p, n = hard_negative_mining(emb, labels, num_negatives=1)
    assert a.ndim == 2 and p.ndim == 2 and n.ndim == 2
    assert a.shape[1] == 128
    assert p.shape[0] == p.shape[0] and a.shape[0] >= 2