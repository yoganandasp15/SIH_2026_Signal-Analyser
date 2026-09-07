"""
Open-Set Automatic Modulation Classification (AMC) Package
==========================================================
Combines physical DSP invariants with statistical distance (Mahalanobis)
and an explicit Out-Of-Distribution (OOD) rejection boundary.
"""

from .classifier import classify_modulation_open_set

__all__ = ["classify_modulation_open_set"]
