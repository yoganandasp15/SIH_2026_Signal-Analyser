"""
Automatic Gain Control (AGC) - Optional Synchronization Aid
============================================================
Provides a dual-speed root-mean-square tracking loop.
CRITICAL DESIGN INVARIANT: AGC is NOT a default conditioning stage.
It must never be applied blindly to amplitude-modulated waveforms (QAM, AM, PAPR analysis)
as it eliminates informative envelope dynamics. It is strictly an optional aid for phase loops.
"""

from typing import Tuple
import numpy as np


def apply_dual_speed_agc(
    signal: np.ndarray,
    target_rms: float = 1.0,
    attack_alpha: float = 0.05,
    decay_alpha: float = 0.005,
    max_gain: float = 40.0,
    min_gain: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies dual-speed RMS envelope tracking loop.

    Parameters:
    -----------
    signal : np.ndarray
        Input signal.
    target_rms : float
        Target output RMS level.
    attack_alpha : float
        Fast attack rate when power rises.
    decay_alpha : float
        Slow decay rate when power drops.
    max_gain, min_gain : float
        Gain clamping boundaries.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (gain_adjusted_signal, tracking_gain_history)
    """
    if len(signal) == 0:
        return signal, np.ones(0)

    n = len(signal)
    gain_out = np.zeros(n, dtype=np.float32)
    output = np.zeros(n, dtype=signal.dtype)

    current_gain = 1.0
    for i in range(n):
        s = signal[i]
        out_val = s * current_gain
        inst_mag = abs(out_val)
        err = target_rms - inst_mag

        if err < 0:  # Output too large -> fast attack
            alpha = attack_alpha
        else:        # Output too small -> slow release
            alpha = decay_alpha

        current_gain += alpha * err
        current_gain = float(np.clip(current_gain, min_gain, max_gain))

        gain_out[i] = current_gain
        output[i] = s * current_gain

    return output, gain_out
