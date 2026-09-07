"""
Instantaneous Phase and Frequency Dynamics Module
=================================================
Extracts phase jitter, derivative frequency variance, and unrolled phase
stability for modulation family discrimination (e.g. constant phase PSK vs continuous phase FSK/FM).
"""

from typing import Dict, Any, Optional
import numpy as np


def compute_instantaneous_phase_features(
    signal: np.ndarray,
    fs: Optional[float] = None
) -> Dict[str, float]:
    """
    Computes variance of instantaneous frequency and phase.
    """
    if len(signal) < 8:
        return {
            "instantaneous_freq_variance": 0.0,
            "instantaneous_phase_variance": 0.0
        }

    use_fs = float(fs) if (fs is not None and fs > 0) else 1.0

    # Phase unrolling
    if np.iscomplexobj(signal):
        phase = np.unwrap(np.angle(signal))
    else:
        from scipy.signal import hilbert
        analytic = hilbert(signal)
        phase = np.unwrap(np.angle(analytic))

    # Instantaneous frequency: derivative of phase
    diff_phase = np.diff(phase)
    inst_freq = diff_phase * (use_fs / (2.0 * np.pi))

    # Variance of detrended instantaneous frequency
    freq_detrend = inst_freq - np.mean(inst_freq)
    freq_var = float(np.var(freq_detrend))

    # Variance of detrended phase
    t = np.arange(len(phase))
    if len(t) > 2:
        # Fit linear carrier slope and subtract
        poly = np.polyfit(t, phase, 1)
        phase_detrend = phase - (poly[0] * t + poly[1])
        phase_var = float(np.var(phase_detrend))
    else:
        phase_var = 0.0

    return {
        "instantaneous_freq_variance": freq_var,
        "instantaneous_phase_variance": phase_var
    }
