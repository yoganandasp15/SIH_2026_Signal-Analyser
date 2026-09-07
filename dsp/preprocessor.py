"""
Signal Preprocessing and Conditioning Module
============================================
Provides mathematical conditioning for complex baseband signals:
- Local Oscillator (LO) / DC offset cancellation
- Scale-invariant unit average power normalization (E[|x|^2] = 1.0)
- Configurable Butterworth bandpass / lowpass filtering
- Signal statistical moments and Peak-to-Average Power Ratio (PAPR)
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.signal import butter, sosfiltfilt


def validate_input_signal(
    signal: Any,
    fs: float,
    min_samples: int = 16
) -> Tuple[bool, str, np.ndarray]:
    """
    Robust verification layer for blind input signals.
    Validates:
    - Signal is non-None and convertible to numeric array
    - 1-D array structure (or single-channel squeeze)
    - Finite samples (no NaN or Inf)
    - Non-empty with len >= min_samples
    - fs is finite and > 0

    Returns:
    --------
    Tuple[bool, str, np.ndarray]
        (is_valid, error_message, validated_1d_array)
    """
    if signal is None:
        return False, "Signal input is None", np.array([], dtype=np.complex64)

    try:
        sig_arr = np.asarray(signal)
    except Exception as e:
        return False, f"Cannot convert input to numpy array: {e}", np.array([], dtype=np.complex64)

    if not np.issubdtype(sig_arr.dtype, np.number):
        return False, f"Signal must have numeric dtype, got {sig_arr.dtype}", np.array([], dtype=np.complex64)

    if sig_arr.ndim != 1:
        squeezed = sig_arr.squeeze()
        if squeezed.ndim == 1:
            sig_arr = squeezed
        else:
            return False, f"Signal must be 1-dimensional, got shape {sig_arr.shape}", np.array([], dtype=np.complex64)

    if len(sig_arr) < min_samples:
        return False, f"Signal length {len(sig_arr)} is less than minimum {min_samples} samples", sig_arr

    if not np.all(np.isfinite(sig_arr)):
        return False, "Signal contains non-finite values (NaN or Inf)", sig_arr

    if not isinstance(fs, (int, float, np.number)) or not np.isfinite(fs) or fs <= 0:
        return False, f"Invalid sampling rate fs={fs}; must be finite and > 0", sig_arr

    return True, "", sig_arr


def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
    """
    Removes Local Oscillator (LO) leakage and DC center frequency spike.

    Formula:
    --------
    x_tilde[n] = x[n] - (1/N) * sum_{k=0}^{N-1} x[k]
    """
    if signal is None or len(signal) == 0:
        return signal
    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains non-finite values (NaN or Inf)")
    dc_val = np.mean(signal)
    return signal - dc_val


def normalize_signal_power(
    signal: np.ndarray,
    target_power: float = 1.0,
    eps: float = 1e-12
) -> Tuple[np.ndarray, float]:
    """
    Normalizes complex baseband signal to unit average power E[|y|^2] = 1.0.

    Formula:
    --------
    P_avg = (1/N) * sum |x[k]|^2
    y[n] = x[n] / sqrt(P_avg)

    Returns:
    --------
    Tuple[np.ndarray, float]
        (normalized_signal, measured_average_power)
    """
    if target_power <= 0:
        raise ValueError("target_power must be positive and non-zero")
    if signal is None or len(signal) == 0:
        return signal, 0.0
    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains non-finite values (NaN or Inf)")

    avg_power = float(np.mean(np.abs(signal) ** 2))
    if avg_power < eps:
        # Avoid zero division on pure silence
        return signal, avg_power

    scale_factor = np.sqrt(target_power / (avg_power + eps))
    normalized_signal = signal * scale_factor
    return normalized_signal, avg_power


def apply_bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int = 4
) -> np.ndarray:
    """
    Applies a zero-phase Butterworth bandpass filter to complex baseband.

    Parameters:
    -----------
    signal : np.ndarray
        Complex baseband signal.
    fs : float
        Sample rate (Hz).
    lowcut : float
        Lower cutoff frequency (Hz).
    highcut : float
        Upper cutoff frequency (Hz).
    order : int
        Filter order.
    """
    if fs <= 0 or not np.isfinite(fs):
        raise ValueError(f"Invalid sampling rate fs={fs}; must be finite and > 0")
    if signal is None or len(signal) <= order * 3:
        return signal

    nyq = 0.5 * fs
    low = max(1e-5, lowcut) / nyq
    high = min(nyq - 1e-5, highcut) / nyq

    if low >= high or high >= 1.0:
        return signal

    try:
        sos = butter(order, [low, high], btype='bandpass', output='sos')
        i_filtered = sosfiltfilt(sos, np.real(signal))
        q_filtered = sosfiltfilt(sos, np.imag(signal))
        return i_filtered + 1j * q_filtered
    except Exception:
        return signal


def compute_signal_stats(signal: np.ndarray) -> Dict[str, float]:
    """
    Computes summary statistical properties of the complex envelope.

    Returns:
    --------
    Dict containing:
    - mean_amplitude
    - std_amplitude
    - envelope_variance_ratio (std / mean)
    - peak_amplitude
    - papr_db (Peak-to-Average Power Ratio in dB)
    """
    if signal is None or len(signal) == 0 or not np.all(np.isfinite(signal)):
        return {
            "mean_amplitude": 0.0,
            "std_amplitude": 0.0,
            "envelope_variance_ratio": 0.0,
            "peak_amplitude": 0.0,
            "papr_db": 0.0
        }

    envelope = np.abs(signal)
    mean_amp = float(np.mean(envelope))
    std_amp = float(np.std(envelope))
    peak_amp = float(np.max(envelope))
    avg_power = float(np.mean(envelope ** 2))
    peak_power = peak_amp ** 2

    papr_db = 10.0 * np.log10(max(peak_power / (avg_power + 1e-12), 1.0))
    var_ratio = std_amp / (mean_amp + 1e-12)

    return {
        "mean_amplitude": mean_amp,
        "std_amplitude": std_amp,
        "envelope_variance_ratio": var_ratio,
        "peak_amplitude": peak_amp,
        "papr_db": papr_db
    }
