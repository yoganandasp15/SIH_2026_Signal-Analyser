"""
DC Offset and Local Oscillator (LO) Leakage Cancellation
========================================================
Removes carrier leakage spikes and stationary bias from baseband signals.
"""

import numpy as np


def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
    """
    Removes Local Oscillator (LO) leakage and DC center frequency spike.

    Formula:
    --------
    x_tilde[n] = x[n] - (1/N) * sum_{k=0}^{N-1} x[k]
    """
    if len(signal) == 0:
        return signal
    dc_val = np.mean(signal)
    return signal - dc_val
