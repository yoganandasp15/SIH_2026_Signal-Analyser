"""
DSP Signal Conditioning Package
===============================
Provides deterministic channel impairment conditioning:
- Blind DC / LO Leakage Removal
- Conditional IQ Imbalance Estimation & Compensation
- Power Normalization
- Rational Polyphase Resampling
- Optional Automatic Gain Control (AGC)
"""

from typing import Tuple, Dict, Any
import numpy as np

from .dc import remove_dc_offset
from .iq_imbalance import (
    estimate_iq_imbalance,
    compensate_iq_imbalance,
    conditional_iq_conditioning
)
from .normalization import normalize_signal_power
from .resampler import resample_signal_sps
from .agc import apply_dual_speed_agc


def condition_signal(
    signal: np.ndarray,
    apply_iq: bool = True,
    apply_agc: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Standard V3 Conditioning Pipeline:
    1. DC offset removal
    2. Conditional IQ imbalance compensation (only if detected)
    3. Scale-invariant power normalization
    4. Optional AGC (disabled by default)
    """
    y = remove_dc_offset(signal)
    if apply_iq:
        y, report = conditional_iq_conditioning(y)
    else:
        report = estimate_iq_imbalance(y)
        report["compensation_applied"] = False
    y, _ = normalize_signal_power(y, target_power=1.0)
    if apply_agc:
        y, _ = apply_dual_speed_agc(y)
    return y, report


__all__ = [
    "remove_dc_offset",
    "estimate_iq_imbalance",
    "compensate_iq_imbalance",
    "conditional_iq_conditioning",
    "normalize_signal_power",
    "resample_signal_sps",
    "apply_dual_speed_agc",
    "condition_signal"
]
