"""Deep Forensic Analysis of the Three New Patient Recordings."""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path

base = Path(r"C:\Users\Shashwat\Desktop\Centration")
datasets = [
    ("PX_normal_download (OD)", base / "PX_normal_download" / "OD"),
    ("PX_ru_od (OD)", base / "PX_ru_od" / "OD"),
    ("PX_ru_os (OS)", base / "PX_ru_os" / "OS"),
]

for label, p in datasets:
    print("=" * 80)
    print(f"DATASET: {label}")
    print("=" * 80)
    if not p.exists():
        print(f"Directory {p} does not exist!")
        continue

    for vid_p in sorted(p.glob("*.mp4")):
        cap = cv2.VideoCapture(str(vid_p))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur_sec = n_frames / fps if fps > 0 else 0
        print(f"Video: {vid_p.name:<38} | Res: {w}x{h} | FPS: {fps:5.2f} | Frames: {n_frames:4d} | Duration: {dur_sec:5.1f}s")
        cap.release()

    for csv_p in sorted(p.glob("*.csv")):
        df = pd.read_csv(csv_p)
        print(f"\nCSV File: {csv_p.name} ({len(df)} rows)")
        pupil_det = df["pupil_detected"].mean() * 100
        limbus_det = df["limbus_detected"].mean() * 100
        print(f"Pupil Detection Rate:  {pupil_det:.2f}% ({df['pupil_detected'].sum()}/{len(df)})")
        print(f"Limbus Detection Rate: {limbus_det:.2f}% ({df['limbus_detected'].sum()}/{len(df)})")

        # Processing time
        p_mean = df["processing_time_ms"].mean()
        p_p95 = df["processing_time_ms"].quantile(0.95)
        print(f"Proc Time (ms):        mean={p_mean:.1f}ms, p95={p_p95:.1f}ms, max={df['processing_time_ms'].max():.1f}ms")

        # Calibration & WTW
        cal_method = df["calibration_method"].iloc[0]
        px_mm = df["px_per_mm"].iloc[0]
        wtw_h_mean = df["measured_wtw_horizontal_mm"].mean()
        wtw_status_counts = df["wtw_validity_status"].value_counts().to_dict()
        print(f"Calibration:           Method={cal_method}, Scale={px_mm:.2f} px/mm")
        print(f"Measured WTW (Horiz):  {wtw_h_mean:.2f} mm (Status: {wtw_status_counts})")

        # Corneal Offset
        off_px = df["offset_px"].mean()
        off_mm = df["offset_mm"].mean()
        print(f"Corneal Offset:        {off_px:.2f} px | {off_mm:.3f} mm")

        # Cyclotorsion
        cyclo_deg = df["cyclotorsion_deg"].dropna()
        if len(cyclo_deg) > 0:
            c_mean, c_std = cyclo_deg.mean(), cyclo_deg.std()
            c_min, c_max = cyclo_deg.min(), cyclo_deg.max()
            c_conf_mean = df["cyclotorsion_confidence"].mean()
            c_qual = df["cyclotorsion_quality"].value_counts().to_dict()
            c_dir = df["cyclotorsion_direction"].value_counts().to_dict()
            c_streams = df["cyclotorsion_agreeing_streams"].value_counts().to_dict()
            base_frame = df["cyclotorsion_baseline_frame"].iloc[0] if "cyclotorsion_baseline_frame" in df.columns else "N/A"
            print(f"Cyclo baseline lock:   Frame {base_frame}")
            print(f"Cyclo Angle theta:     mean={c_mean:+.2f} deg, std={c_std:.2f} deg, range=[{c_min:+.2f} deg, {c_max:+.2f} deg]")
            print(f"Cyclo Confidence:      mean={c_conf_mean:.3f}")
            print(f"Cyclo Qualities:       {c_qual}")
            print(f"Cyclo Directions:      {c_dir}")
            print(f"Agreeing Streams:      {c_streams}")
            deg_diffs = cyclo_deg.diff().abs().dropna()
            print(f"Frame-to-Frame Jitter: mean_abs_jump={deg_diffs.mean():.3f} deg, p95_jump={deg_diffs.quantile(0.95):.3f} deg, max_jump={deg_diffs.max():.3f} deg")
        else:
            print("No cyclotorsion data recorded!")
    print()
