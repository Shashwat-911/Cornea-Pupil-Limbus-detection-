"""Generate crisp, executive-grade visual diagrams for the Cyclotorsion Non-Tech KT Document."""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Output directory
output_dir = Path("docs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Set global styles
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

# ----------------------------------------------------------------------
# Diagram 1: The Cyclotorsion Problem & Solution (The Clinical Reality)
# ----------------------------------------------------------------------
def generate_diagram_1():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor='#F8FAFC')
    
    # Common eye drawing helper
    def draw_eye(ax, center=(0, 0), rot_deg=0, label=""):
        # Sclera
        ax.add_patch(patches.Ellipse(center, 4.4, 3.2, angle=0, facecolor='#FFFFFF', edgecolor='#CBD5E1', lw=2))
        # Limbus (outer iris)
        ax.add_patch(patches.Circle(center, 1.25, facecolor='#93C5FD', edgecolor='#1E3A8A', lw=2.5, alpha=0.85))
        # Iris crypts/texture lines
        for ang in np.linspace(0, 360, 24, endpoint=False):
            rad = np.radians(ang + rot_deg)
            x1 = center[0] + 0.5 * np.cos(rad)
            y1 = center[1] + 0.5 * np.sin(rad)
            x2 = center[0] + 1.15 * np.cos(rad)
            y2 = center[1] + 1.15 * np.sin(rad)
            ax.plot([x1, x2], [y1, y2], color='#1D4ED8', lw=1.2, alpha=0.6)
        # Pupil (black center)
        ax.add_patch(patches.Circle(center, 0.45, facecolor='#0F172A', edgecolor='#334155', lw=1.5))
        
        # 12 o'clock anatomical reference marker
        rad_ref = np.radians(90 + rot_deg)
        xr = center[0] + 1.25 * np.cos(rad_ref)
        yr = center[1] + 1.25 * np.sin(rad_ref)
        ax.plot([center[0], xr], [center[1], yr], color='#EF4444', lw=2.8, linestyle='--')
        ax.plot(xr, yr, 'o', color='#EF4444', markersize=7)

    # Panel 1: Sitting Upright (Pre-Op Diagnostic Scan)
    ax1 = axes[0]
    ax1.set_facecolor('#FFFFFF')
    draw_eye(ax1, (0, 0), 0)
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.0, 2.2)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title("1. SITTING UPRIGHT\n(Diagnostic Exam / Pentacam)", fontsize=12, fontweight='bold', color='#0F2C59', pad=10)
    ax1.text(0, -1.8, "• Patient sits upright in chair\n• Reference axis = 90.0° (True Vertical)\n• Planned laser angle = 90.0°", 
             ha='center', fontsize=9.5, color='#334155', bbox=dict(boxstyle='round,pad=0.5', facecolor='#F1F5F9', edgecolor='#CBD5E1'))
    ax1.annotate("12 o'clock (90°)", xy=(0, 1.25), xytext=(0, 1.7),
                 arrowprops=dict(arrowstyle="->", color='#EF4444', lw=1.5),
                 ha='center', fontsize=9.5, fontweight='bold', color='#EF4444')

    # Panel 2: Lying Down (Eye naturally rolls!)
    ax2 = axes[1]
    ax2.set_facecolor('#FFFFFF')
    draw_eye(ax2, (0, 0), 7) # 7 degrees cyclotorsion
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.0, 2.2)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title("2. LYING DOWN (SUPINE)\n(On Operating Table)", fontsize=12, fontweight='bold', color='#B91C1C', pad=10)
    
    # Show rotation arc
    arc = patches.Arc((0, 0), 3.0, 3.0, angle=0, theta1=83, theta2=90, color='#EA580C', lw=2.5)
    ax2.add_patch(arc)
    ax2.annotate("Cyclotorsion\n+7.0° Twist", xy=(-0.35, 1.45), xytext=(-1.5, 1.6),
                 arrowprops=dict(arrowstyle="->", color='#EA580C', lw=1.5),
                 ha='center', fontsize=9, fontweight='bold', color='#EA580C')
    ax2.text(0, -1.8, "• Inner ear gravity sensors shift\n• Eye rolls sideways naturally by +7°\n• Target zone is now at 97.0°!", 
             ha='center', fontsize=9.5, color='#334155', bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF2F2', edgecolor='#FECACA'))

    # Panel 3: Our Solution (Real-Time Automated Tracking & Shift)
    ax3 = axes[2]
    ax3.set_facecolor('#FFFFFF')
    draw_eye(ax3, (0, 0), 7)
    # Corrected treatment axis drawn
    rad_corr = np.radians(90 + 7)
    xc1 = -1.8 * np.cos(rad_corr)
    yc1 = -1.8 * np.sin(rad_corr)
    xc2 = 1.8 * np.cos(rad_corr)
    yc2 = 1.8 * np.sin(rad_corr)
    ax3.plot([xc1, xc2], [yc1, yc2], color='#059669', lw=3, linestyle='-')
    ax3.set_xlim(-2.5, 2.5)
    ax3.set_ylim(-2.0, 2.2)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title("3. OUR SOFTWARE AT WORK\n(Dynamic Laser Compensation)", fontsize=12, fontweight='bold', color='#065F46', pad=10)
    ax3.annotate("Corrected Laser Axis\nShifted +7.0° to 97°", xy=(xc2, yc2), xytext=(1.4, 1.7),
                 arrowprops=dict(arrowstyle="->", color='#059669', lw=1.8),
                 ha='center', fontsize=9.5, fontweight='bold', color='#059669')
    ax3.text(0, -1.8, "• AI tracks iris shift instantly (<100ms)\n• Laser automatically re-aims to 97.0°\n• 100% Surgical precision maintained", 
             ha='center', fontsize=9.5, color='#334155', bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECFDF5', edgecolor='#A7F3D0'))

    plt.suptitle("WHAT IS CYCLOTORSION? WHY SURGERY NEEDS AUTOMATIC COMPENSATION", 
                 fontsize=14, fontweight='bold', color='#0F2C59', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "diagram1_cyclotorsion_concept.png", bbox_inches='tight')
    plt.close()
    print("Saved diagram 1.")

# ----------------------------------------------------------------------
# Diagram 2: Clinical Impact / The Alpins 3-Degree Rule Curve
# ----------------------------------------------------------------------
def generate_diagram_2():
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor='#F8FAFC')
    ax.set_facecolor('#FFFFFF')
    
    angles = np.linspace(0, 32, 200)
    # Alpins formula: Under-correction % = 2 * sin(theta) * 100%
    loss_percent = 2 * np.sin(np.radians(angles)) * 100
    loss_percent = np.clip(loss_percent, 0, 100)
    
    # Colored clinical severity bands
    ax.axvspan(0, 1.0, color='#DCFCE7', alpha=0.6, label='Optimal Surgical Accuracy (0° - 1°)')
    ax.axvspan(1.0, 3.0, color='#FEF3C7', alpha=0.6, label='Acceptable Clinical Tolerance (1° - 3°)')
    ax.axvspan(3.0, 32.0, color='#FEE2E2', alpha=0.6, label='Severe Loss & Distorted Vision (> 3°)')
    
    # Plot curve
    ax.plot(angles, loss_percent, color='#1E40AF', lw=3.5, label='Astigmatism Treatment Loss %')
    
    # Key callout points
    points = [
        (1.0, 3.49, "1° Error\n3.5% Loss", '#059669'),
        (3.0, 10.47, "3° Error\n10.5% Loss\n(Threshold)", '#D97706'),
        (5.0, 17.43, "5° Error\n17.4% Loss", '#EA580C'),
        (10.0, 34.73, "10° Error\n34.7% Loss", '#DC2626'),
        (30.0, 100.0, "30° Error\n100% Loss (Total Failure!)", '#991B1B'),
    ]
    
    for ang, loss, txt, col in points:
        ax.plot(ang, loss, 'o', color=col, markersize=8)
        offset_y = 10 if loss < 80 else -18
        offset_x = 0.5 if ang < 25 else -5.5
        ax.annotate(txt, xy=(ang, loss), xytext=(ang + offset_x, loss + offset_y),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
                    fontweight='bold', fontsize=9, color=col)
    
    ax.set_title("WHY PRECISION MATTERS: THE 3-DEGREE RULE (ALPINS FORMULA)\nEvery 1° of uncorrected eye rotation wastes ~3.5% of astigmatism correction", 
                 fontsize=12.5, fontweight='bold', color='#0F2C59', pad=14)
    ax.set_xlabel("Uncorrected Eye Cyclotorsion Angle (Degrees)", fontsize=11, fontweight='bold', color='#334155')
    ax.set_ylabel("Astigmatism Treatment Lost (%)", fontsize=11, fontweight='bold', color='#334155')
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle='--', alpha=0.4, color='#94A3B8')
    ax.legend(loc='lower right', frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=9.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "diagram2_alpins_rule.png", bbox_inches='tight')
    plt.close()
    print("Saved diagram 2.")

# ----------------------------------------------------------------------
# Diagram 3: Polar Unwrapping Analogy ("The Donut to a Ribbon")
# ----------------------------------------------------------------------
def generate_diagram_3():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), facecolor='#F8FAFC', gridspec_kw={'width_ratios': [1, 1.4]})
    
    # Left: The Donut (Cartesian Iris Ring)
    ax1 = axes[0]
    ax1.set_facecolor('#FFFFFF')
    
    # Draw circular iris
    circle_limbus = patches.Circle((0, 0), 1.6, facecolor='#93C5FD', edgecolor='#1D4ED8', lw=2.5)
    circle_pupil = patches.Circle((0, 0), 0.65, facecolor='#0F172A', edgecolor='#334155', lw=2.0)
    ax1.add_patch(circle_limbus)
    ax1.add_patch(circle_pupil)
    
    # Features inside iris
    angles = [30, 75, 120, 190, 240, 310]
    radii = [1.0, 1.25, 0.9, 1.35, 1.1, 1.2]
    colors = ['#DC2626', '#EA580C', '#16A34A', '#9333EA', '#D97706', '#2563EB']
    
    for a, r, c in zip(angles, radii, colors):
        rad = np.radians(a)
        ax1.plot(r * np.cos(rad), r * np.sin(rad), 'o', color=c, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
        ax1.text(r * np.cos(rad)*1.18, r * np.sin(rad)*1.18, f"{a}°", color='#1E293B', fontsize=8.5, fontweight='bold', ha='center', va='center')
        
    ax1.annotate("", xy=(1.85, 0.5), xytext=(1.85, -0.5),
                 arrowprops=dict(arrowstyle="->", color='#2563EB', lw=2.5, connectionstyle="arc3,rad=-0.4"))
    ax1.text(2.1, 0, "Eye Rotates\nCircularly (θ)", color='#2563EB', fontsize=9.5, fontweight='bold', va='center')
    
    ax1.set_xlim(-2.4, 2.5)
    ax1.set_ylim(-2.0, 2.0)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title("STEP A: CIRCULAR IRIS ('THE DONUT')\nRotating eye moves features in circles", fontsize=11, fontweight='bold', color='#0F2C59')

    # Right: The Ribbon (Polar Unwrapped Strip)
    ax2 = axes[1]
    ax2.set_facecolor('#FFFFFF')
    
    # Draw unwrapped rectangle strip
    rect = patches.Rectangle((0, 0.2), 360, 1.0, facecolor='#E0F2FE', edgecolor='#0284C7', lw=2)
    ax2.add_patch(rect)
    
    # Place same features on unwrapped strip (Angle = X, Radius = Y)
    for a, r, c in zip(angles, radii, colors):
        norm_r = 0.2 + (r - 0.65) / (1.6 - 0.65) * 1.0
        ax2.plot(a, norm_r, 'o', color=c, markersize=11, markeredgecolor='white', markeredgewidth=1.5)
        ax2.text(a, norm_r + 0.18, f"{a}°", color=c, fontsize=8.5, fontweight='bold', ha='center')
        
    # Show horizontal shift arrow
    ax2.annotate("", xy=(300, 1.4), xytext=(220, 1.4),
                 arrowprops=dict(arrowstyle="->", color='#059669', lw=3))
    ax2.text(260, 1.55, "Eye Rotation becomes a SIMPLE HORIZONTAL SHIFT (Δx)!", 
             color='#059669', fontsize=9.5, fontweight='bold', ha='center')
    
    ax2.set_xlim(-15, 375)
    ax2.set_ylim(-0.2, 1.8)
    ax2.set_xticks([0, 90, 180, 270, 360])
    ax2.set_xticklabels(['0°', '90° (Top)', '180°', '270°', '360°'], fontsize=9, fontweight='bold')
    ax2.set_yticks([0.2, 1.2])
    ax2.set_yticklabels(['Pupil\n(Inner)', 'Limbus\n(Outer)'], fontsize=9, fontweight='bold')
    ax2.set_title("STEP B: UNWRAPPED POLAR STRIP ('THE RIBBON')\nCircular math turns into 1D horizontal shift!", fontsize=11, fontweight='bold', color='#0F2C59')
    ax2.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.suptitle("HOW WE SOLVE ROTATION: 'UNWRAPPING THE DONUT INTO A RIBBON'", 
                 fontsize=13, fontweight='bold', color='#0F2C59', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "diagram3_polar_unwrapping_analogy.png", bbox_inches='tight')
    plt.close()
    print("Saved diagram 3.")

# ----------------------------------------------------------------------
# Diagram 4: Complete System Architecture Flowchart
# ----------------------------------------------------------------------
def generate_diagram_4():
    fig, ax = plt.subplots(figsize=(14, 8.5), facecolor='#F8FAFC')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # Helper to draw box
    def draw_box(x, y, w, h, title, subtitle="", bg='#EFF6FF', border='#3B82F6', title_col='#1E3A8A'):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.2",
                                     facecolor=bg, edgecolor=border, lw=2)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=title_col)
            ax.text(x + w/2, y + h*0.30, subtitle, ha='center', va='center', fontsize=8, color='#475569')
        else:
            ax.text(x + w/2, y + h*0.5, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=title_col)
            
    def draw_arrow(x1, y1, x2, y2, col='#64748B'):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=col, lw=2.0, shrinkA=3, shrinkB=3))

    # STAGE 1: Inputs
    draw_box(0.5, 6.8, 3.8, 1.4, "Pre-Op Diagnostic Image", "Pentacam Scan (Patient Sitting Up)", '#F0FDF4', '#22C55E', '#14532D')
    draw_box(0.5, 4.8, 3.8, 1.4, "Live Surgical Video Feed", "Microscope Camera (Patient Lying Down)", '#EFF6FF', '#3B82F6', '#1E3A8A')
    
    # STAGE 2: Preprocessing & Iris Isolation
    draw_box(4.8, 5.5, 2.5, 2.0, "AI Eye Normalization", "1. Detect Pupil & Limbus\n2. Extract Iris Ring\n3. Polar Ribbon Unwrap\n4. CLAHE Contrast Boost", '#F8FAFC', '#64748B', '#0F172A')
    
    draw_arrow(4.3, 7.5, 4.8, 6.8)
    draw_arrow(4.3, 5.5, 4.8, 6.2)
    
    # STAGE 3: The 5 Parallel Streams (The Council of Experts)
    stream_w = 2.2
    stream_h = 1.05
    sx = 7.9
    
    draw_box(sx, 7.5, stream_w, stream_h, "Stream A: Phase Correlation", "Polar Texture Frequency FFT", '#FDF4FF', '#D946EF', '#701A75')
    draw_box(sx, 6.2, stream_w, stream_h, "Stream B: Deep Matcher", "Iris Crypts & Freckles", '#F5F3FF', '#8B5CF6', '#4C1D95')
    draw_box(sx, 4.9, stream_w, stream_h, "Stream C: Ink Tracker", "Gentian Violet Pen Marks", '#FFFBEB', '#F59E0B', '#78350F')
    draw_box(sx, 3.6, stream_w, stream_h, "Stream D: Vessel Tracker", "Scleral Blood Vessel Forks", '#FEF2F2', '#EF4444', '#7F1D1D')
    draw_box(sx, 2.3, stream_w, stream_h, "Stream E: Pentacam Match", "Diagnostic Feature Anchor", '#ECFDF5', '#10B981', '#064E3B')

    for sy in [8.0, 6.7, 5.4, 4.1, 2.8]:
        draw_arrow(7.3, 6.5, sx, sy)
        
    # STAGE 4: Fusion & Safety Engine
    draw_box(10.7, 4.4, 2.8, 2.4, "Consensus Fusion &\nSafety Interlock", "• Weighted Median of Streams\n• Reject Eyelid/Glare Outliers\n• Safety Grade: SURGICAL / CLINICAL\n• Interlock: Halt laser if unsafe!", '#FEF9C3', '#CA8A04', '#713F12')
    
    for sy in [8.0, 6.7, 5.4, 4.1, 2.8]:
        draw_arrow(sx + stream_w, sy, 10.7, 5.6)
        
    # STAGE 5: Final Output & Action
    draw_box(10.7, 1.2, 2.8, 2.2, "Real-Time Laser Alignment", "θ = +7.2° Detected!\nTreatment Axis: 90° → 97.2°\nExecution: < 100 ms\nZero Astigmatism Loss!", '#DCFCE7', '#16A34A', '#14532D')
    
    draw_arrow(12.1, 4.4, 12.1, 3.4, col='#16A34A')
    
    # Section Labels along the top
    ax.text(2.4, 8.6, "1. INPUTS", fontsize=11, fontweight='bold', color='#475569', ha='center')
    ax.text(6.0, 8.6, "2. PREPROCESSING", fontsize=11, fontweight='bold', color='#475569', ha='center')
    ax.text(9.0, 8.6, "3. 5-STREAM DETECTION", fontsize=11, fontweight='bold', color='#475569', ha='center')
    ax.text(12.1, 8.6, "4. FUSION & ACTION", fontsize=11, fontweight='bold', color='#475569', ha='center')

    plt.suptitle("END-TO-END CYCLOTORSION DETECTION & REAL-TIME LASER CORRECTION ARCHITECTURE", 
                 fontsize=13, fontweight='bold', color='#0F2C59', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "diagram4_pipeline_flowchart.png", bbox_inches='tight')
    plt.close()
    print("Saved diagram 4.")

if __name__ == '__main__':
    generate_diagram_1()
    generate_diagram_2()
    generate_diagram_3()
    generate_diagram_4()
    print("All 4 diagrams successfully generated in docs/figures!")
