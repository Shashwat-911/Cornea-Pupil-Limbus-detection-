"""
Registration pipeline for cyclotorsion detection and iris registration.

Provides multi-stream torsion estimation from paired (reference + current)
eye images using phase correlation, deep matching, ink markers, limbal
vessels, and custom-trained iris feature models.
"""

from pupil_tracking.registration.engine import RegistrationEngine
from pupil_tracking.registration.polar import PolarUnwrapper
from pupil_tracking.registration.enhancement import IrisEnhancer

__all__ = [
    "RegistrationEngine",
    "PolarUnwrapper",
    "IrisEnhancer",
]
