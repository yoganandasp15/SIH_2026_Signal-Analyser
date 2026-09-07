"""
Cyclostationary Baud Clock Feature Extraction
=============================================
Detects periodic cyclostationary statistical structure arising from symbol transitions.
Computes cyclic autocorrelation and spectral correlation peak prominence.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np


def compute_cyclic_clock_features(
    signal: np.ndarray,
    fs: Optional[float] = None,
    max_pts: int = 16384
) -> Dict[str, float]:
    """
    Computes cyclostationary peak prominence by analyzing the spectrum of the
    instantaneous power / non-linear transform |y[n]|^2 or y[n]^2.
    """
    if len(signal) < 64:
        return {
            "cyclic_peak_prominence": 0.0,
            "snr_m2m4_db": 0.0
        }

    use_pts = min(len(signal), max_pts)
    sig_slice = signal[:use_pts]

    # Non-linear squared-magnitude operation to expose symbol transition lines
    pwr = np.abs(sig_slice) ** 2
    pwr = pwr - np.mean(pwr)

    n_fft = 2 ** int(np.ceil(np.log2(use_pts)))
    fft_mag = np.abs(np.fft.rfft(pwr, n=n_fft))

    # Exclude DC and very low frequency bins
    skip_bins = max(5, int(n_fft * 0.005))
    valid_mag = fft_mag[skip_bins:]

    if len(valid_mag) == 0:
        prominence = 0.0
    else:
        median_floor = float(np.median(valid_mag))
        if median_floor < 1e-6:
            median_floor = float(np.mean(valid_mag)) + 1e-6
        max_peak = float(np.max(valid_mag))
        prominence = float(np.clip(max_peak / median_floor, 0.0, 50.0))

    # M2M4 split-moment SNR estimate
    r_sig = np.real(sig_slice)
    m2 = float(np.mean(r_sig ** 2))
    m4 = float(np.mean(r_sig ** 4))
    arg = max(2.0 * (m2 ** 2) - m4, 1e-12)
    s_est = np.sqrt(arg)
    n_est = max(m2 - s_est, 1e-12)
    snr_m2m4_db = float(10.0 * np.log10(max(s_est / n_est, 1e-3)))

    return {
        "cyclic_peak_prominence": float(prominence),
        "snr_m2m4_db": float(snr_m2m4_db)
    }
