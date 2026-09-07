"""
Spectral Analysis & Time-Frequency Transform Module
===================================================
Implements Welch Power Spectral Density (PSD) and Short-Time Fourier Transform
(STFT) Spectrogram waterfall analysis for complex baseband and analytic signals.
"""

from typing import Tuple, Optional
import numpy as np
from scipy.signal import welch, spectrogram


def compute_welch_psd(
    signal: np.ndarray,
    fs: float,
    nperseg: int = 2048,
    noverlap: Optional[int] = None,
    scaling: str = 'density'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the Welch Power Spectral Density (PSD) for a complex signal,
    with frequencies centered around zero baseband [-fs/2, +fs/2].

    Parameters:
    -----------
    signal : np.ndarray
        1D complex baseband array.
    fs : float
        Sampling frequency (Hz).
    nperseg : int
        Length of each segment (default: 2048).
    noverlap : Optional[int]
        Overlap between segments (default: nperseg // 2).
    scaling : str
        'density' (V^2/Hz) or 'spectrum' (V^2).

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - f_shifted: Frequency array in Hz [-fs/2, +fs/2]
        - psd_db: Logarithmic PSD in dB/Hz
        - psd_linear: Linear PSD array
    """
    if len(signal) == 0:
        return np.array([]), np.array([]), np.array([])

    actual_nperseg = min(len(signal), nperseg)
    if noverlap is None:
        actual_noverlap = actual_nperseg // 2
    else:
        actual_noverlap = min(noverlap, actual_nperseg - 1)

    # Welch computation for complex input (return_onesided=False)
    f, pxx = welch(
        signal,
        fs=fs,
        window='hann',
        nperseg=actual_nperseg,
        noverlap=actual_noverlap,
        return_onesided=False,
        scaling=scaling
    )

    # Shift zero-frequency component to center of spectrum
    f_shifted = np.fft.fftshift(f)
    pxx_shifted = np.fft.fftshift(pxx)

    # Avoid log of zero / negative values
    eps = 1e-18
    psd_linear = np.maximum(pxx_shifted, eps)
    psd_db = 10.0 * np.log10(psd_linear)

    return f_shifted, psd_db, psd_linear


def compute_spectrogram(
    signal: np.ndarray,
    fs: float,
    nperseg: int = 1024,
    noverlap: Optional[int] = None,
    max_time_bins: int = 500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the 2D Short-Time Fourier Transform (STFT) waterfall spectrogram.

    Parameters:
    -----------
    signal : np.ndarray
        Complex baseband signal.
    fs : float
        Sampling frequency (Hz).
    nperseg : int
        Segment FFT window length.
    noverlap : Optional[int]
        Window overlap (default: nperseg // 2).
    max_time_bins : int
        Maximum time bins to downsample for GUI performance.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - t_bins: Time bin array (seconds)
        - f_bins_shifted: Frequency array (Hz) [-fs/2, +fs/2]
        - sxx_db_shifted: 2D matrix (time x frequency) in dB
    """
    if len(signal) == 0:
        return np.array([]), np.array([]), np.zeros((0, 0))

    actual_nperseg = min(len(signal), nperseg)
    if noverlap is None:
        actual_noverlap = actual_nperseg // 2
    else:
        actual_noverlap = min(noverlap, actual_nperseg - 1)

    f, t, sxx = spectrogram(
        signal,
        fs=fs,
        window='hann',
        nperseg=actual_nperseg,
        noverlap=actual_noverlap,
        return_onesided=False,
        scaling='density',
        mode='psd'
    )

    # Shift frequencies to [-fs/2, +fs/2]
    f_shifted = np.fft.fftshift(f)
    sxx_shifted = np.fft.fftshift(sxx, axes=0)

    # Convert to dB
    eps = 1e-18
    sxx_db = 10.0 * np.log10(np.maximum(sxx_shifted, eps))

    # Downsample time bins if too dense for web visualization
    if len(t) > max_time_bins:
        step = len(t) // max_time_bins
        t = t[::step]
        sxx_db = sxx_db[:, ::step]

    # Transpose so shape is (len(t), len(f)) for standard plotting
    return t, f_shifted, sxx_db.T
