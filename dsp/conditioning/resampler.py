"""
Signal Resampling Module
========================
Provides rational polyphase resampling to adjust samples per symbol (SPS)
for downstream clock recovery and symbol slicing.
"""

from typing import Tuple
import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction


def resample_signal_sps(
    signal: np.ndarray,
    current_sps: float,
    target_sps: float = 2.0,
    max_samples: int = 200_000
) -> Tuple[np.ndarray, float]:
    """
    Resamples baseband signal to target samples-per-symbol (SPS).

    Parameters:
    -----------
    signal : np.ndarray
        Input complex or real baseband array.
    current_sps : float
        Current samples per symbol.
    target_sps : float
        Target samples per symbol (typically 2.0 for Gardner TED).
    max_samples : int
        Memory ceiling.

    Returns:
    --------
    Tuple[resampled_signal, new_sps]
    """
    if len(signal) == 0 or current_sps <= 0.0:
        return signal, current_sps

    if abs(current_sps - target_sps) < 0.01:
        return signal[:max_samples], current_sps

    # Find rational ratio up / down ~ target_sps / current_sps
    ratio = float(target_sps / current_sps)
    frac = Fraction(ratio).limit_denominator(64)
    up = frac.numerator
    down = frac.denominator

    sig_slice = signal[:min(len(signal), max_samples)]

    # Polyphase resampling
    resampled_real = resample_poly(np.real(sig_slice), up, down)
    if np.iscomplexobj(signal):
        resampled_imag = resample_poly(np.imag(sig_slice), up, down)
        resampled = resampled_real + 1j * resampled_imag
    else:
        resampled = resampled_real

    effective_sps = current_sps * (up / down)
    return resampled, effective_sps
