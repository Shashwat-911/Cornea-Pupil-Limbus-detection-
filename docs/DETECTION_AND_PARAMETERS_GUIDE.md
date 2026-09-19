# 👁️ Ophthalmic Pupil, Limbus & Cyclotorsion Tracking Guide
### *The Engineering & Clinical Onboarding Manual for New Team Members*

---

## 1. The 30-Second Mental Model

When performing laser eye surgery (such as refractive corneal lenticule extraction or LASIK):
1. The **Pupil** is the dark aperture in the center that dilates and constricts under surgical lighting.
2. The **Limbus** is the outer anatomical boundary where the colored **Iris** meets the white **Sclera**.
3. The **Corneal Center** is defined anatomically as the geometric center of the **Limbus**.
4. **Centration Offset** is the vector $(\Delta x, \Delta y)$ between the Pupil Center and the Corneal (Limbus) Center. Surgeries align their laser shots relative to this physiological offset.
5. **Cyclotorsion** is the ocular rotational twist (torsion in degrees, $\pm \theta$) that occurs when a patient transitions from upright sitting (pre-op diagnostic scans like Pentacam) to lying flat on their back under the surgical microscope.

```
       [Sclera (White)]
             │
     ┌───────┴───────┐
    ╱   Limbus (Outer)╲   <─── Corneal Center (x_L, y_L)
   │   ┌───────────┐   │
   │  │    Iris     │  │
   │  │  ┌───────┐  │  │
   │  │  │ Pupil │  │  │  <─── Pupil Center (x_P, y_P)
   │  │  └───────┘  │  │
   │   └───────────┘   │       Centration Vector:
    ╲                 ╱        Δx = x_P - x_L,  Δy = y_P - y_L
     └───────────────┘         Offset (mm) = sqrt(Δx² + Δy²) / px_per_mm
```

---

## 2. End-to-End Pipeline Architecture

The pipeline takes a raw camera frame (from a live surgical video feed or clinical image) and outputs verified sub-millimeter clinical parameters.

```
Raw Camera Frame (BGR / Grayscale)
             │
             ▼
  [1. Preprocessing & Normalization]
     ├── Specular Reflection Masking (removes bright surgical lamp glares)
     └── Multi-scale Dual-Pass CLAHE (contrast enhancement for dark/light eyes)
             │
             ▼
  [2. Suction Ring Detector]
     ├── Checks if surgical suction ring is present ("docked" vs "pre-docked")
     └── Automatically selects tight or wide search boundaries
             │
             ▼
  [3. Primary AI Segmentation (ResNet-34 + U-Net via ONNX)]
     ├── Class 0: Background / Sclera
     ├── Class 1: Pupil
     └── Class 2: Iris / Limbus
     └── (Fallback): Classical Starburst & Adaptive Thresholding if ML confidence < 0.25
             │
             ▼
  [4. Contour Extraction & Sub-Pixel Ellipse Fitting]
     ├── RANSAC Outlier Rejection (filters out eyelashes, eyelid borders, speculum)
     ├── Scharr Sub-Pixel Gradient Refinement (±0.1 px boundary localization)
     └── Auto Circle vs. Ellipse selection
             │
             ▼
  [5. Clinical Centration & Spatial Calibration]
     ├── Anatomical Anchor: 12.0 mm Horizontal Visible Iris Diameter (HVID)
     ├── Compute Pupil Center, Limbus Center, Offset (mm), and Angle (deg)
     └── Kalman Filter Smoothing (removes high-frequency tremor across video frames)
             │
             ▼
  [6. Iris Registration & Cyclotorsion (When Reference Image Active)]
     ├── Daugman Rubber-Sheet Polar Unwrapping (Iris Annulus -> 64 x 360 strip)
     ├── 5 Concurrent Detection Streams:
     │     ├── Stream A: Phase Correlation (Fourier Shift Theorem)
     │     ├── Stream B: Deep Feature Matcher (Keypoint descriptors)
     │     ├── Stream C: Surgical Ink Tracker (Gentian violet mark detection)
     │     ├── Stream D: Conjunctival Vessel Tracker (Limbal bifurcations)
     │     └── Stream E: Custom DINOv2+LoRA Iris Feature Model
     └── Consensus Fusion Engine (Weighted median outlier rejection -> Final Torsion Δθ)
             │
             ▼
  [7. Surgeon UI HUD & Concurrent Dual-Stream Recording]
     ├── Screen Overlay Video (*.mp4 with crosshairs, metrics & angle)
     └── Pristine Clean Raw Video (*_raw.mp4 for AI model training)
```

---

## 3. Step-by-Step Subsystem Walkthrough

### Step 1: Preprocessing (`pupil_tracking/preprocessing/`)
* **Problem**: Surgical operating microscopes have blinding LED glares (Purkinje reflections) and variable lighting. Dark brown irises have faint pupil boundaries, while light blue irises have low-contrast limbus margins.
* **Solution**:
  - `reflection_removal.py`: Spots pixels above the 97th brightness percentile and masks them out so the reflection's centroid does not pull the pupil center off-target.
  - `grayscale_handler.py`: Applies two passes of Contrast-Limited Adaptive Histogram Equalization (CLAHE):
    - Low-clip pass (1.5) preserving delicate iris patterns.
    - High-clip pass (4.0) revealing deep pupil edges.

### Step 2: Suction Ring Awareness (`pupil_tracking/preprocessing/ring_aware.py`)
* **Problem**: When the surgeon docks a femtosecond laser suction ring onto the eye, the ring's dark metallic or plastic circular edge looks identical to a pupil or limbus to standard algorithms!
* **Solution**:
  - `deterministic_ring_detector.py` checks for the suction ring circle.
  - If **Docked**: The search radius is clamped strictly inside the ring (`inner_margin = 15 px`), eliminating false detections on the suction device.
  - If **Pre-docked**: The detector searches the broader field of view and softly suppresses eyelid margins (`eyelid_margin_frac = 0.15`).

### Step 3: AI Segmentation & Classical Fallback (`pupil_tracking/ml/` & `core/`)
* **Model**: `EyeSegmentationModel` (ResNet-34 encoder + U-Net decoder with skip connections).
* **Speed**: Quantized into ONNX INT8 (`models/onnx/segmentation_quantized.onnx`), running in under **15–25 ms** on modern CPUs/GPUs.
* **Classical Fallback**: If the eye is partially obscured or confidence drops below `min_pupil_confidence = 0.25`, the system automatically shifts to `classical_fallback.py` using adaptive thresholding and Starburst ray-casting, applying a 15% confidence penalty.

### Step 4: Sub-Pixel Ellipse Fitting (`pupil_tracking/core/smart_fitter.py`)
* **Problem**: Segmented contours are jagged pixel steps (integer coordinates), and eyelids or eyelashes occlude the top and bottom of the eye.
* **Solution**:
  - **RANSAC** (500 iterations): Randomly samples points along the contour and votes on the best mathematical ellipse, throwing away points caused by eyelid occlusions or glints.
  - **Sub-Pixel Peak Refinement**: Casts rays normal to the boundary and uses the Scharr gradient operator with parabolic 3-point peak interpolation to locate edge boundaries with **sub-pixel accuracy (0.1 pixel)**.

### Step 5: Corneal Centration & Spatial Calibration (`pupil_tracking/core/corneal_center.py`)
* **Anatomical Definition**:
  $$\text{Corneal Center} = (x_{\text{limbus}}, y_{\text{limbus}})$$
  $$\text{Pupil Center} = (x_{\text{pupil}}, y_{\text{pupil}})$$
  $$\text{Decentration Vector} = (x_P - x_L, y_P - y_L)$$
* **Pixel-to-Millimeter Conversion**:
  - Default mode: `ANATOMICAL_ANCHOR`.
  - The human horizontal white-to-white (WTW) corneal diameter averages **12.0 mm**.
  - System sets: $\text{px\_per\_mm} = \frac{2 \cdot r_{\text{limbus}}}{12.0}$.
  - Every displacement is therefore converted to millimeters: $\text{Offset (mm)} = \frac{\sqrt{\Delta x^2 + \Delta y^2}}{\text{px\_per\_mm}}$.
  - *Clinical Norm*: Typical physiological decentration is **0.1 to 0.5 mm** (usually slightly nasal/inferior). Values $>0.6\text{ mm}$ trigger warnings.

### Step 6: Iris Registration & Cyclotorsion (`pupil_tracking/registration/`)
* **The Clinical Need**: Astigmatic corneal incisions (e.g. toric ablation or lenticule alignment) require accurate rotational orientation. If the eye rotates by just $3^\circ$, up to **10% of astigmatism correction power is lost**!
* **How it Works**:
  1. **Rubber-Sheet Unwrapping (`polar.py`)**: Based on Daugman’s model, the annular iris region between pupil radius $r_P(\theta)$ and limbus radius $r_L(\theta)$ is projected into a normalized polar strip of size $64 \times 360$ pixels ($1^\circ$ per column).
  2. **Multi-Stream Torsion Detection**:
     - **Stream A (Phase Correlation)**: Computes 2D Fast Fourier Transform cross-power spectrum between the unwrapped reference and live frame. The horizontal shift corresponds directly to rotation angle $\Delta \theta$.
     - **Stream B (Deep Keypoint Matcher)**: Detects persistent natural iris landmarks (crypts and furrows) using local feature descriptors and RANSAC homography.
     - **Stream C (Surgical Ink Tracker)**: Detects purple/blue surgical gentian violet marks placed by the surgeon at 3 and 9 o'clock using HSV color gating (`[120, 50, 50]` to `[160, 255, 255]`).
     - **Stream D (Conjunctival Vessel Tracker)**: Tracks blood vessel branchings (bifurcations) near the limbus.
     - **Stream E (Custom DINOv2 Feature Model)**: Uses our custom fine-tuned LoRA neural network for dense patch feature correspondence.
  3. **Consensus Fusion (`fusion.py`)**: Uses a weighted median algorithm. If at least 2 streams agree within $1.0^\circ$, it emits the high-confidence fused cyclotorsion angle $\Delta \theta$.

### Step 7: Temporal Kalman Smoothing (`pupil_tracking/video/kalman_tracker.py`)
* Maintains a state vector $[x, y, r_a, r_b, \theta, v_x, v_y, \dots]$ across video frames.
* Dampens video jitter and hand/microscope tremor.
* **Blink Handling**: If pupil confidence falls below $0.30$, the system flags a "Blink" and coast-forwards the previous stable state for up to $5$ frames (`max_carry_forward_frames = 5`) without dropping the surgical overlay.

---

## 4. Master Parameter Cheat Sheet

All parameters reside centrally in `pupil_tracking/utils/config.py`. Here are the ones you will most commonly inspect or tune:

### A. Detection Thresholds (`DetectionConfig`)
| Parameter | Default | Why it matters | How to tune |
| :--- | :--- | :--- | :--- |
| `min_pupil_confidence` | `0.25` | Minimum ML probability required to accept a pupil mask. | Lower to `0.15` for dim video feeds; raise to `0.40` for pristine still photos. |
| `min_limbus_confidence` | `0.25` | Minimum ML probability required to accept an iris/limbus mask. | Lower if outer iris boundary is faint. |
| `min_pupil_area` | `200 px` | Ignores dark specks smaller than this area. | Increase if shadows or lashes create false pupil spots. |
| `min_limbus_area` | `3000 px` | Ignores circular iris artifacts smaller than this area. | Set based on camera zoom (3000 is safe for 512px). |
| `morph_kernel_size` | `5` | Morphological filter size used to close small mask holes. | Use 3 for fine detail; 5 or 7 for noisy frames. |

### B. Ellipse & RANSAC Fitting (`FittingConfig` & `SubPixelConfig`)
| Parameter | Default | Why it matters | How to tune |
| :--- | :--- | :--- | :--- |
| `ransac_iterations` | `500` | Number of random candidate subsets tested. | 500 gives sub-pixel precision in ~2 ms. Reduce to 200 for extreme low-power CPUs. |
| `ransac_threshold` | `1.5 px` | Maximum distance a contour point can deviate from the ellipse to count as an inlier. | Tighter (1.0) rejects eyelids; looser (2.5) tolerates irregular pupils. |
| `ransac_min_inlier_ratio`| `0.60` | At least 60% of points must fit the ellipse curve. | If heavily occluded by eyelids, drop to 0.45. |
| `use_scharr` | `True` | Uses Scharr 3x3 filter instead of Sobel for better rotational symmetry. | Keep `True` for surgical-grade angle fidelity. |

### C. Spatial Calibration & Units (`CalibrationConfig`)
| Parameter | Default | Why it matters | How to tune |
| :--- | :--- | :--- | :--- |
| `mode` | `ANATOMICAL_ANCHOR` | Calibration strategy (`ANATOMICAL_ANCHOR`, `FIXED_PIXEL_SCALE`, `RING_REFLECTION`). | Use `ANATOMICAL_ANCHOR` when no calibration target is present. |
| `corneal_diameter_mm` | `12.0 mm` | Population standard Horizontal Visible Iris Diameter (HVID). | Modify if working with pediatric eyes or specific animal studies. |
| `ema_alpha` | `0.15` | Exponential Moving Average smoothing factor for pixel-to-mm ratio. | 0.15 prevents single-frame noise from making millimeter readings jump around. |

### D. Cyclotorsion & Iris Registration (`RegistrationConfig`)
| Parameter | Default | Why it matters | How to tune |
| :--- | :--- | :--- | :--- |
| `polar_num_angles` | `360` | Angular resolution of unwrapped iris ($360 = 1^\circ$ per column). | Keep at 360 for $0.1^\circ$ sub-degree precision. |
| `polar_num_radial` | `64` | Radial bands sampled between pupil and limbus. | 64 captures crypts and furrows without oversampling. |
| `fusion_method` | `weighted_median` | Algorithm combining the 5 registration streams. | Robust against single-stream outlier failures. |
| `fusion_agreement_threshold_deg` | `1.0°` | Max angular discrepancy between streams to declare consensus. | Relax to $1.5^\circ$ if patient has high corneal haze. |
| `ink_hsv_lower` / `upper` | `[120,50,50]` to `[160,255,255]` | HSV color gate for gentian violet surgical marking pens. | Adjust if clinical team switches from purple to blue or green ink. |

### E. Video & Temporal Smoothing (`VideoConfig`)
| Parameter | Default | Why it matters | How to tune |
| :--- | :--- | :--- | :--- |
| `enable_kalman` | `True` | Applies Kalman filtering across video stream. | Always keep `True` for live video; disable for individual image inspection. |
| `kalman_process_noise`| `0.1` | How fast the Kalman filter responds to genuine rapid eye movements. | Lower = smoother; Higher = tracks fast saccades quicker. |
| `max_carry_forward_frames`| `5` | Frames to coast on prediction during a blink before clearing display. | 5 frames at 30 fps $\approx 160$ ms, matching natural human blink duration. |

---

## 5. Frequently Asked Questions (New Joinee Q&A)

### Q1: Does iris color (dark brown vs. light blue/green) affect detection?
**No.** 
1. **Centration is geometric**: The corneal center is the geometric center of the limbal ellipse $(x_L, y_L)$, and the pupil center is $(x_P, y_P)$. This depends on spatial contours, not color values.
2. **Grayscale dual-pass CLAHE**: `grayscale_handler.py` runs dual CLAHE curves. For dark eyes with low pigment difference, it boosts micro-gradients. For light blue eyes, it handles the strong pupil boundary without blowing out the limbus.
3. **Ink tracking is color-isolated**: The surgical ink tracker isolates the gentian violet marker using HSV hues ($120^\circ-160^\circ$), which sits outside the melanin hue range of natural eyes.

### Q2: What is the difference between Pupil Center and Corneal Center?
They are **not** the same point!
- The **Corneal Center** is the center of the **Limbus** (the cornea's physical perimeter).
- The **Pupil Center** is the center of the pupil aperture, which is anatomically offset (decentered) physiologically by 0.1 to 0.4 mm towards the nasal and inferior quadrant.
- During eye surgery, lasers do not simply shoot at the pupil (which changes center when the pupil dilates or constricts under drugs); they rely on the fixed corneal/limbal frame of reference.

### Q3: Why do we record two video files at the same time ("Dual-Stream Recording")?
- When the surgeon clicks Record, the system saves:
  1. `patient_case_overlay.mp4`: Contains the surgeon's crosshairs, bounding ellipses, cyclotorsion dial, and millimeter measurements. Great for surgery review and auditing.
  2. `patient_case_raw.mp4`: Contains the **unmodified, pristine camera frames** without any visual graphics drawn on top.
- *Why?* Computer vision and AI models cannot be trained on images that have green crosshairs or white text burned into them! The raw video allows continuous training and validation of future AI models without losing surgical context.

---

## 6. Developer Quickstart

### 1. Run Detection on a Single Clinical Image
```python
import cv2
from pupil_tracking.core.detector import UnifiedDetector

# 1. Initialize detector (loads ONNX model and config automatically)
detector = UnifiedDetector()

# 2. Read input eye image
image = cv2.imread("clinical_data/clean/eye_01.jpeg")

# 3. Perform detection
result = detector.detect(image, frame_number=0, source="eye_01.jpeg")

# 4. Access clinical outputs
print(f"Pupil Center: ({result.pupil.center.x:.1f}, {result.pupil.center.y:.1f}) px")
print(f"Limbus Center: ({result.limbus.center.x:.1f}, {result.limbus.center.y:.1f}) px")
print(f"Decentration Offset: {result.corneal_center.offset_mm:.3f} mm")
print(f"Offset Angle: {result.corneal_center.offset_angle_deg:.1f}°")
print(f"White-to-White (WTW): {result.calibration.limbus_diameter_mm:.2f} mm")
print(f"Quality Rating: {result.quality.grade.value}")
```

### 2. Measure Cyclotorsion Relative to a Reference Image
```python
from pupil_tracking.registration.engine import RegistrationEngine
from pupil_tracking.utils.config import get_config

# 1. Initialize registration engine
cfg = get_config()
reg_engine = RegistrationEngine(cfg.registration)

# 2. Set reference (e.g. seated pre-op capture)
ref_image = cv2.imread("preop_reference.jpeg")
ref_result = detector.detect(ref_image)
reg_engine.set_reference(ref_image, ref_result)

# 3. In live surgical loop:
live_image = cv2.imread("surgical_frame_042.jpeg")
live_result = detector.detect(live_image)

# 4. Measure cyclotorsion
torsion_result = reg_engine.process_frame(live_image, live_result)
print(f"Cyclotorsion Angle: {torsion_result.torsion_deg:+.2f}° (Conf: {torsion_result.confidence:.2f})")
print(f"Agreement status: {torsion_result.streams_in_agreement} streams agreed")
```

---
*Maintained by the Medevplus / IXcentai Engineering Team. For questions or modifications, consult `pupil_tracking/utils/config.py`.*
