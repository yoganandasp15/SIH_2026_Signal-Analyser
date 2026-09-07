"""
Spectral Distribution Features Module
=====================================
Extracts frequency-domain moments and energy distributions:
- Spectral Centroid
- Spectral Spread / Variance
- Spectral Flatness Measure (SFM)
- Spectral Rolloff (95% energy boundary)
- Occupied Bandwidth (99% power integration)
"""

from typing import Dict, Any, Optional
import numpy as np
from scipy.signal import welch


def compute_spectral_distribution_features(
    signal: np.ndarray,
    fs: Optional[float] = None,
    nperseg: int = 1024,
    eps: float = 1e-12
) -> Dict[str, float]:
    """
    Computes spectral distribution features.
    If fs is None, normalized frequency [-0.5, 0.5] cycles/sample is used.
    """
    if len(signal) < 32:
        return {
            "spectral_centroid": 0.0,
            "spectral_spread": 0.0,
            "spectral_flatness": 0.0,
            "spectral_rolloff": 0.0,
            "obw_99": 0.0
        }

    use_fs = float(fs) if (fs is not None and fs > 0) else 1.0
    sig_slice = signal[:min(len(signal), 65536)]

    if np.iscomplexobj(sig_slice):
        freqs, psd = welch(
            sig_slice,
            fs=use_fs,
            nperseg=min(len(sig_slice), nperseg),
            return_onesided=False
        )
        freqs = np.fft.fftshift(freqs)
        psd = np.fft.fftshift(psd)
    else:
        freqs, psd = welch(
            sig_slice,
            fs=use_fs,
            nperseg=min(len(sig_slice), nperseg),
            return_onesided=True
        )

    psd = psd + eps
    total_energy = float(np.sum(psd))

    # Normalized spectral density
    p_norm = psd / total_energy

    # Spectral Centroid
    centroid = float(np.sum(freqs * p_norm))

    # Spectral Spread
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * p_norm)))

    # Spectral Flatness Measure (SFM)
    geom_mean = float(np.exp(np.mean(np.log(psd))))
    arith_mean = float(np.mean(psd))
    sfm = float(geom_mean / (arith_mean + eps))

    # Spectral Rolloff (95% of cumulative energy)
    cum_energy = np.cumsum(psd)
    idx_95 = int(np.searchsorted(cum_energy, 0.95 * total_energy))
    idx_95 = min(idx_95, len(freqs) - 1)
    rolloff = float(abs(freqs[idx_95]))

    # Occupied Bandwidth (99% power integration)
    idx_low = int(np.searchsorted(cum_energy, 0.005 * total_energy))
    idx_high = int(np.searchsorted(cum_energy, 0.995 * total_energy))
    idx_low = max(0, min(idx_low, len(freqs) - 1))
    idx_high = max(0, min(idx_high, len(freqs) - 1))
    obw_99 = float(abs(freqs[idx_high] - freqs[idx_low]))

    return {
        "spectral_centroid": centroid,
        "spectral_spread": spread,
        "spectral_flatness": sfm,
        "spectral_rolloff": rolloff,
        "obw_99": obw_99
    }
