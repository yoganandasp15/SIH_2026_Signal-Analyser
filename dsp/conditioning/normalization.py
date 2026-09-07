"""
Scale-Invariant Power Normalization
===================================
Normalizes baseband waveforms to unit average power E[|y|^2] = 1.0.
"""

from typing import Tuple
import numpy as np


def normalize_signal_power(
    signal: np.ndarray,
    target_power: float = 1.0,
    eps: float = 1e-12
) -> Tuple[np.ndarray, float]:
    """
    Normalizes complex baseband signal to target average power.

    Formula:
    --------
    P_avg = (1/N) * sum |x[k]|^2
    y[n] = x[n] * sqrt(target_power / P_avg)
    """
    if len(signal) == 0:
        return signal, 0.0

    avg_power = float(np.mean(np.abs(signal) ** 2))
    if avg_power < eps:
        return signal, avg_power

    scale_factor = np.sqrt(target_power / (avg_power + eps))
    return signal * scale_factor, avg_power
