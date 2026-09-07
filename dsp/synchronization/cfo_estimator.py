"""
Carrier Frequency Offset (CFO) and Phase Recovery Module
========================================================
Implements coarse carrier estimation via M-th power spectral search with parabolic
interpolation, followed by 2nd-order Digital Costas Loop fine phase tracking.
Extracts coarse CFO, fine CFO, residual CFO, PLL lock metric, and phase ambiguity.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def estimate_coarse_cfo_mth_power(
    signal: np.ndarray,
    fs: float,
    m_order: int = 4,
    search_range_hz: Optional[float] = None
) -> Tuple[float, float]:
    """
    Estimates coarse Carrier Frequency Offset (CFO) using the M-th power non-linear method.
    Applies parabolic interpolation around the spectral peak for sub-bin precision.

    Parameters:
    -----------
    signal : np.ndarray
        Complex baseband signal.
    fs : float
        Sampling frequency in Hz.
    m_order : int
        Modulation order M (e.g. 2 for BPSK, 4 for QPSK, 1 for unmodulated).
    search_range_hz : Optional[float]
        Maximum expected frequency offset range in Hz.

    Returns:
    --------
    Tuple[cfo_hz, peak_prominence]
    """
    if len(signal) < 64 or fs <= 0 or m_order < 1:
        return 0.0, 0.0

    n_pts = min(len(signal), 32768)
    sig_slice = signal[:n_pts]

    # Non-linear M-th power transformation collapses modulation phase modulation
    if m_order == 1:
        y_m = sig_slice
    else:
        y_m = sig_slice ** m_order

    n_fft = 2 ** int(np.ceil(np.log2(n_pts * 2)))
    fft_vals = np.abs(np.fft.fft(y_m, n=n_fft))
    freqs = np.fft.fftfreq(n_fft, d=1.0 / fs)

    fft_shifted = np.fft.fftshift(fft_vals)
    freqs_shifted = np.fft.fftshift(freqs)

    # Search window
    max_range = search_range_hz if search_range_hz is not None else 0.45 * fs
    mask = np.abs(freqs_shifted) <= float(m_order * max_range)

    if not np.any(mask):
        return 0.0, 0.0

    masked_fft = fft_shifted[mask]
    masked_freqs = freqs_shifted[mask]

    peak_local_idx = int(np.argmax(masked_fft))
    peak_mag = float(masked_fft[peak_local_idx])
    median_mag = float(np.median(masked_fft)) + 1e-12
    prominence = peak_mag / median_mag

    if peak_local_idx > 0 and peak_local_idx < len(masked_fft) - 1:
        # Parabolic interpolation for sub-bin accuracy
        alpha = masked_fft[peak_local_idx - 1]
        beta = masked_fft[peak_local_idx]
        gamma = masked_fft[peak_local_idx + 1]
        denom = 2.0 * (2.0 * beta - alpha - gamma)
        delta = (alpha - gamma) / denom if abs(denom) > 1e-12 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        df = masked_freqs[1] - masked_freqs[0]
        fine_peak_freq = masked_freqs[peak_local_idx] + delta * df
    else:
        fine_peak_freq = masked_freqs[peak_local_idx]

    cfo_hz = float(fine_peak_freq / m_order)
    return cfo_hz, float(prominence)


def track_carrier_costas_loop(
    signal: np.ndarray,
    fs: float,
    m_order: int = 4,
    loop_bw_hz: float = 200.0,
    damping: float = 0.707
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    2nd-Order Digital Costas Loop / Decision-Directed Phase Lock Loop.
    Performs continuous phase tracking and fine residual CFO cancellation.

    Parameters:
    -----------
    signal : np.ndarray
        Input symbols or oversampled signal.
    fs : float
        Sample rate.
    m_order : int
        2 for BPSK, 4 for QPSK/QAM.
    loop_bw_hz : float
        Closed-loop bandwidth in Hz.
    damping : float
        Damping factor (default 0.707 critically damped).

    Returns:
    --------
    Tuple[derotated_signal, tracking_telemetry]
    """
    n = len(signal)
    if n < 16 or fs <= 0:
        return signal, {
            "pll_locked": False,
            "lock_metric": 0.0,
            "residual_cfo_hz": 0.0,
            "phase_error_var": 1.0,
            "final_phase_rad": 0.0
        }

    # Standard 2nd-order loop filter coefficients
    theta_bw = loop_bw_hz / fs
    denom = 1.0 + 2.0 * damping * theta_bw + theta_bw ** 2
    alpha = (4.0 * damping * theta_bw) / denom
    beta = (4.0 * theta_bw ** 2) / denom

    phase = 0.0
    freq_acc = 0.0
    output = np.zeros(n, dtype=np.complex64)
    error_history = np.zeros(n, dtype=np.float32)
    cycle_slips = 0
    prev_phase = 0.0

    for k in range(n):
        s = signal[k]
        # Derotate
        s_rot = s * np.exp(-1j * phase)
        output[k] = s_rot

        i_val = np.real(s_rot)
        q_val = np.imag(s_rot)

        # Phase Error Detector (PED)
        if m_order == 2:
            # BPSK error detector
            e = i_val * q_val
        elif m_order == 4:
            # QPSK error detector
            e = np.sign(i_val) * q_val - np.sign(q_val) * i_val
        elif m_order == 8:
            # 8-PSK decision directed error detector
            angle_val = np.angle(s_rot)
            quant_angle = np.round(angle_val / (np.pi / 4.0)) * (np.pi / 4.0)
            e = np.sin(angle_val - quant_angle)
        else:
            e = np.sign(i_val) * q_val - np.sign(q_val) * i_val

        # Normalize error
        norm_factor = (i_val ** 2 + q_val ** 2) + 1e-6
        e_norm = float(e / norm_factor)
        e_norm = float(np.clip(e_norm, -2.0, 2.0))
        error_history[k] = e_norm

        # 2nd order loop filter update
        freq_acc += beta * e_norm
        phase += freq_acc + alpha * e_norm

        if k > 0 and abs(phase - prev_phase) > np.pi:
            cycle_slips += 1
        prev_phase = phase

    # Evaluate loop lock over last 25% of symbols
    tail_len = max(16, int(n * 0.25))
    tail_errors = error_history[-tail_len:]
    error_var = float(np.var(tail_errors))

    # Lock metric: normalized inverse variance
    lock_metric = float(np.clip(1.0 / (1.0 + 10.0 * error_var), 0.0, 1.0))
    is_locked = bool(lock_metric > 0.70 and error_var < 0.15)

    residual_cfo_hz = float((freq_acc * fs) / (2.0 * np.pi))

    telemetry = {
        "pll_locked": is_locked,
        "lock_metric": lock_metric,
        "residual_cfo_hz": residual_cfo_hz,
        "phase_error_var": error_var,
        "final_phase_rad": float(phase % (2.0 * np.pi)),
        "phase_offset_rad": float(phase % (2.0 * np.pi)),
        "cycle_slips": cycle_slips
    }

    return output, telemetry


def track_carrier_decision_directed_qam(
    signal: np.ndarray,
    fs: float,
    qam_order: int = 16,
    loop_bw_hz: float = 150.0,
    damping: float = 0.707
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Decision-Directed Carrier Recovery Loop for 16-QAM and 64-QAM.
    Slices received symbols to the nearest standard QAM constellation grid point,
    computes decision-directed phase error, and tracks phase offset and cycle slips.
    """
    n = len(signal)
    if n < 16 or fs <= 0:
        return signal, {
            "pll_locked": False,
            "lock_metric": 0.0,
            "residual_cfo_hz": 0.0,
            "phase_error_var": 1.0,
            "final_phase_rad": 0.0,
            "phase_offset_rad": 0.0,
            "cycle_slips": 0
        }

    pwr = float(np.mean(np.abs(signal) ** 2))
    scale = np.sqrt(pwr) if pwr > 1e-12 else 1.0
    sig_norm = signal / scale

    if qam_order == 64:
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=float) / np.sqrt(42.0)
    else:  # 16-QAM
        levels = np.array([-3, -1, 1, 3], dtype=float) / np.sqrt(10.0)

    theta_bw = loop_bw_hz / fs
    denom = 1.0 + 2.0 * damping * theta_bw + theta_bw ** 2
    alpha = (4.0 * damping * theta_bw) / denom
    beta = (4.0 * theta_bw ** 2) / denom

    phase = 0.0
    freq_acc = 0.0
    output = np.zeros(n, dtype=np.complex64)
    error_history = np.zeros(n, dtype=np.float32)
    cycle_slips = 0
    prev_phase = 0.0

    for k in range(n):
        s = sig_norm[k]
        s_rot = s * np.exp(-1j * phase)
        output[k] = s_rot * scale

        i_val = float(np.real(s_rot))
        q_val = float(np.imag(s_rot))

        # Decision slicer
        i_hat = float(levels[int(np.argmin(np.abs(levels - i_val)))])
        q_hat = float(levels[int(np.argmin(np.abs(levels - q_val)))])

        # Decision-directed error: Im(s_rot * a_hat*)
        e = q_val * i_hat - i_val * q_hat
        norm_ref = (i_hat ** 2 + q_hat ** 2) + 1e-6
        e_norm = float(np.clip(e / norm_ref, -2.0, 2.0))
        error_history[k] = e_norm

        freq_acc += beta * e_norm
        phase += freq_acc + alpha * e_norm

        if k > 0 and abs(phase - prev_phase) > np.pi:
            cycle_slips += 1
        prev_phase = phase

    tail_len = max(16, int(n * 0.25))
    tail_errors = error_history[-tail_len:]
    error_var = float(np.var(tail_errors))

    lock_metric = float(np.clip(1.0 / (1.0 + 12.0 * error_var), 0.0, 1.0))
    is_locked = bool(lock_metric > 0.65 and error_var < 0.18)
    residual_cfo_hz = float((freq_acc * fs) / (2.0 * np.pi))

    telemetry = {
        "pll_locked": is_locked,
        "lock_metric": lock_metric,
        "residual_cfo_hz": residual_cfo_hz,
        "phase_error_var": error_var,
        "final_phase_rad": float(phase % (2.0 * np.pi)),
        "phase_offset_rad": float(phase % (2.0 * np.pi)),
        "cycle_slips": cycle_slips
    }

    return output, telemetry


def resolve_phase_ambiguity(
    symbols: np.ndarray,
    m_order: int = 4
) -> Tuple[np.ndarray, float]:
    """
    Rotates the constellation to align with standard quadrant grid
    and resolves residual M-fold rotational phase ambiguity.
    """
    if len(symbols) == 0:
        return symbols, 0.0

    angles = np.angle(symbols)
    sector = 2.0 * np.pi / m_order
    nominal_phase = np.pi / m_order if m_order == 4 else 0.0

    wrapped = (angles - nominal_phase + sector / 2.0) % sector - (sector / 2.0)
    median_offset = float(np.median(wrapped))

    corrected_symbols = symbols * np.exp(-1j * median_offset)
    return corrected_symbols, median_offset
