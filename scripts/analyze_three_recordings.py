"""Comprehensive Analysis of the Three Patient Recordings (PX_er_12, PX_ru_od, PX_ra_os)."""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path

datasets = [
    ('PX_er_12 (OD)', Path(r'C:\Users\Shashwat\Desktop\Centration\PX_er_12\OD')),
    ('PX_ru_od (OD)', Path(r'C:\Users\Shashwat\Desktop\Centration\PX_ru_od\OD')),
    ('PX_ra_os (OS)', Path(r'C:\Users\Shashwat\Desktop\Centration\PX_ra_os\OS')),
]

for label, p in datasets:
    print('='*75)
    print(f'DATASET: {label}')
    print('='*75)
    
    # 1. Video file properties
    for vid_p in sorted(p.glob('*.mp4')):
        cap = cv2.VideoCapture(str(vid_p))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur_sec = n_frames / fps if fps > 0 else 0
        print(f'Video: {vid_p.name:<35} | Res: {w}x{h} | FPS: {fps:5.2f} | Frames: {n_frames:4d} | Duration: {dur_sec:5.1f}s')
        cap.release()
        
    # 2. CSV analysis
    csv_files = list(p.glob('*.csv'))
    if not csv_files:
        print('No CSV found!')
        continue
    df = pd.read_csv(csv_files[0])
    print(f'\nCSV File: {csv_files[0].name} ({len(df)} logged rows)')
    
    # Detection rates
    pupil_det = df['pupil_detected'].mean() * 100
    limbus_det = df['limbus_detected'].mean() * 100
    print(f'Pupil Detection Rate: {pupil_det:.2f}% ({df["pupil_detected"].sum()}/{len(df)})')
    print(f'Limbus Detection Rate: {limbus_det:.2f}% ({df["limbus_detected"].sum()}/{len(df)})')
    
    # Latency & processing time
    p_mean = df['processing_time_ms'].mean()
    p_std = df['processing_time_ms'].std()
    p_max = df['processing_time_ms'].max()
    p_p95 = df['processing_time_ms'].quantile(0.95)
    print(f'Processing Time (ms): mean={p_mean:.1f}ms, std={p_std:.1f}ms, p95={p_p95:.1f}ms, max={p_max:.1f}ms')
    
    l_mean = df['latency_ms'].mean()
    l_std = df['latency_ms'].std()
    l_max = df['latency_ms'].max()
    l_p95 = df['latency_ms'].quantile(0.95)
    print(f'Latency (ms):         mean={l_mean:.1f}ms, std={l_std:.1f}ms, p95={l_p95:.1f}ms, max={l_max:.1f}ms')
    
    # Effective frame processing rate
    frame_diffs = df['frame'].diff().dropna()
    print(f'Frame step in video:  mean={frame_diffs.mean():.2f}, min={frame_diffs.min():.0f}, max={frame_diffs.max():.0f}')
    
    # Pupil geometry
    p_cx_mean, p_cx_std = df['pupil_cx_px'].mean(), df['pupil_cx_px'].std()
    p_cy_mean, p_cy_std = df['pupil_cy_px'].mean(), df['pupil_cy_px'].std()
    p_dia_px_mean, p_dia_px_std = df['pupil_diameter_px'].mean(), df['pupil_diameter_px'].std()
    p_dia_mm_mean, p_dia_mm_std = df['pupil_diameter_mm'].mean(), df['pupil_diameter_mm'].std()
    print(f'Pupil Center (px):    X={p_cx_mean:.1f} +/- {p_cx_std:.2f} (span={df["pupil_cx_px"].max()-df["pupil_cx_px"].min():.1f}px) | Y={p_cy_mean:.1f} +/- {p_cy_std:.2f} (span={df["pupil_cy_px"].max()-df["pupil_cy_px"].min():.1f}px)')
    print(f'Pupil Diameter:       {p_dia_px_mean:.1f} +/- {p_dia_px_std:.2f} px  |  {p_dia_mm_mean:.2f} +/- {p_dia_mm_std:.2f} mm')
    
    # Limbus geometry
    l_cx_mean, l_cx_std = df['limbus_cx_px'].mean(), df['limbus_cx_px'].std()
    l_cy_mean, l_cy_std = df['limbus_cy_px'].mean(), df['limbus_cy_px'].std()
    l_dia_px_mean, l_dia_px_std = df['limbus_diameter_px'].mean(), df['limbus_diameter_px'].std()
    l_dia_mm_mean, l_dia_mm_std = df['limbus_diameter_mm'].mean(), df['limbus_diameter_mm'].std()
    print(f'Limbus Center (px):   X={l_cx_mean:.1f} +/- {l_cx_std:.2f} (span={df["limbus_cx_px"].max()-df["limbus_cx_px"].min():.1f}px) | Y={l_cy_mean:.1f} +/- {l_cy_std:.2f} (span={df["limbus_cy_px"].max()-df["limbus_cy_px"].min():.1f}px)')
    print(f'Limbus Diameter:      {l_dia_px_mean:.1f} +/- {l_dia_px_std:.2f} px  |  {l_dia_mm_mean:.2f} +/- {l_dia_mm_std:.2f} mm')
    
    # Calibration & WTW
    cal_method = df['calibration_method'].iloc[0]
    px_mm = df['px_per_mm'].iloc[0]
    wtw_h_mean = df['measured_wtw_horizontal_mm'].mean()
    wtw_status_counts = df['wtw_validity_status'].value_counts().to_dict()
    print(f'Calibration:          Method={cal_method}, Scale={px_mm:.2f} px/mm')
    print(f'Measured WTW (Horiz): {wtw_h_mean:.2f} mm (Status: {wtw_status_counts})')
    
    # Corneal Offset
    off_px_mean, off_px_std = df['offset_px'].mean(), df['offset_px'].std()
    off_mm_mean, off_mm_std = df['offset_mm'].mean(), df['offset_mm'].std()
    off_ang_mean, off_ang_std = df['offset_angle_deg'].mean(), df['offset_angle_deg'].std()
    print(f'Corneal Offset:       {off_px_mean:.2f} +/- {off_px_std:.2f} px  |  {off_mm_mean:.3f} +/- {off_mm_std:.3f} mm  |  Angle: {off_ang_mean:+.1f} deg +/- {off_ang_std:.1f} deg')
    
    # Cyclotorsion
    cyclo_deg = df['cyclotorsion_deg'].dropna()
    if len(cyclo_deg) > 0:
        c_mean, c_std = cyclo_deg.mean(), cyclo_deg.std()
        c_min, c_max = cyclo_deg.min(), cyclo_deg.max()
        c_conf_mean = df['cyclotorsion_confidence'].mean()
        c_qual = df['cyclotorsion_quality'].value_counts().to_dict()
        c_dir = df['cyclotorsion_direction'].value_counts().to_dict()
        c_streams = df['cyclotorsion_agreeing_streams'].value_counts().to_dict()
        base_frame = df['cyclotorsion_baseline_frame'].iloc[0]
        print(f'\n--- CYCLOTORSION STREAM METRICS ---')
        print(f'Baseline Lock Frame:  Frame {base_frame:.0f}')
        print(f'Angle theta:          mean={c_mean:+.2f} deg, std={c_std:.2f} deg, range=[{c_min:+.2f} deg, {c_max:+.2f} deg]')
        print(f'Confidence:           mean={c_conf_mean:.3f} (min={df["cyclotorsion_confidence"].min():.3f}, max={df["cyclotorsion_confidence"].max():.3f})')
        print(f'Quality Ratings:      {c_qual}')
        print(f'Directions:           {c_dir}')
        print(f'Agreeing Streams:     {c_streams}')
        
        # Frame-to-frame angular stability / jitter
        deg_diffs = cyclo_deg.diff().abs().dropna()
        print(f'Frame-to-Frame Jitter: mean_abs_jump={deg_diffs.mean():.3f} deg, p95_jump={deg_diffs.quantile(0.95):.3f} deg, max_jump={deg_diffs.max():.3f} deg')
    else:
        print('No cyclotorsion data recorded!')
    print()
