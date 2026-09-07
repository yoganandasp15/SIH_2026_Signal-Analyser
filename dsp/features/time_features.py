"""
Envelope and Time-Domain Features Module
========================================
Extracts statistical moments from the instantaneous envelope and time series:
- Peak-to-Average Power Ratio (PAPR)
- Envelope Variance Ratio
- Zero-Crossing Rate
- Amplitude Kurtosis
"""

from typing import Dict, Any
import numpy as np


def compute_envelope_and_time_features(
    signal: np.ndarray,
    eps: float = 1e-12
) -> Dict[str, float]:
    """
    Computes time-domain statistical metrics on signal envelope and instantaneous power.
    """
    if len(signal) == 0:
        return {
            "papr_db": 0.0,
            "envelope_variance": 0.0,
            "amplitude_skewness": 0.0,
            "zero_crossing_rate": 0.0,
            "amplitude_kurtosis": 0.0
        }

    env = np.abs(signal)
    mean_env = float(np.mean(env))
    var_env = float(np.var(env))
    std_env = float(np.std(env))

    pwr = env ** 2
    avg_pwr = float(np.mean(pwr))
    peak_pwr = float(np.max(pwr)) if len(pwr) > 0 else 0.0

    papr_db = 10.0 * np.log10(max(peak_pwr / (avg_pwr + eps), 1.0))
    env_var_ratio = std_env / (mean_env + eps)

    # Zero-crossing rate on in-phase component
    real_sig = np.real(signal) - np.mean(np.real(signal))
    zero_crossings = np.sum(np.diff(np.signbit(real_sig).astype(int)) != 0)
    zcr = float(zero_crossings / (len(real_sig) - 1)) if len(real_sig) > 1 else 0.0

    # Kurtosis and Skewness of envelope
    if var_env > eps and std_env > eps:
        kurt = float(np.mean((env - mean_env) ** 4) / (var_env ** 2))
        skew = float(np.mean((env - mean_env) ** 3) / (std_env ** 3))
    else:
        kurt = 0.0
        skew = 0.0

    return {
        "papr_db": papr_db,
        "envelope_variance": env_var_ratio,
        "amplitude_skewness": skew,
        "zero_crossing_rate": zcr,
        "amplitude_kurtosis": kurt
    }
