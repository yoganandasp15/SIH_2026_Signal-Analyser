"""
Conditional IQ Imbalance Estimation and Compensation Module
===========================================================
Estimates amplitude mismatch and quadrature phase skew prior to compensation.
Only applies compensation when empirical evidence (Image Rejection Ratio < threshold)
confirms genuine hardware front-end imbalance, protecting valid signals from distortion.
"""

from typing import Tuple, Dict, Any
import numpy as np


def estimate_iq_imbalance(
    signal: np.ndarray,
    eps: float = 1e-12
) -> Dict[str, Any]:
    """
    Estimates IQ amplitude mismatch factor (g) and phase skew (phi in radians),
    and computes the estimated Image Rejection Ratio (IRR in dB).

    For an ideal balanced complex signal:
      E[I^2] == E[Q^2] and E[I * Q] == 0.

    Parameters:
    -----------
    signal : np.ndarray
        Complex baseband signal (zero-mean).
    eps : float
        Numerical regularizer.

    Returns:
    --------
    Dict containing:
      - amplitude_mismatch: float (g = sqrt(Var(Q) / Var(I)))
      - phase_error_rad: float (phi = arcsin(E[I*Q] / sqrt(Var(I)*Var(Q))))
      - phase_error_deg: float
      - estimated_irr_db: float (Image Rejection Ratio in dB)
      - imbalance_detected: bool
    """
    if len(signal) < 128:
        return {
            "amplitude_mismatch": 1.0,
            "phase_error_rad": 0.0,
            "phase_error_deg": 0.0,
            "estimated_irr_db": 100.0,
            "imbalance_detected": False
        }

    i = np.real(signal) - np.mean(np.real(signal))
    q = np.imag(signal) - np.mean(np.imag(signal))

    var_i = float(np.mean(i ** 2))
    var_q = float(np.mean(q ** 2))

    if var_i < eps or var_q < eps:
        return {
            "amplitude_mismatch": 1.0,
            "phase_error_rad": 0.0,
            "phase_error_deg": 0.0,
            "estimated_irr_db": 100.0,
            "imbalance_detected": False
        }

    # Amplitude imbalance factor
    g = float(np.sqrt(var_q / (var_i + eps)))

    # Phase error via normalized cross-correlation
    cov_iq = float(np.mean(i * q))
    sin_phi = cov_iq / (np.sqrt(var_i * var_q) + eps)
    sin_phi = float(np.clip(sin_phi, -0.999, 0.999))
    phi = float(np.arcsin(sin_phi))

    # Image Rejection Ratio (IRR) approximation
    # IRR = 10 * log10( (1 + 2*g*cos(phi) + g^2) / (1 - 2*g*cos(phi) + g^2) )
    cos_phi = np.cos(phi)
    num = 1.0 + 2.0 * g * cos_phi + (g ** 2)
    denom = max(1.0 - 2.0 * g * cos_phi + (g ** 2), 1e-9)
    irr_db = float(10.0 * np.log10(num / denom))

    # Strong evidence test: IRR < 28 dB and (|g - 1| > 0.08 or |phi| > 0.08 rad (~4.5 deg))
    amp_dev = abs(g - 1.0)
    phase_dev_deg = abs(np.degrees(phi))
    detected = bool(irr_db < 28.0 and (amp_dev > 0.08 or phase_dev_deg > 4.5))

    return {
        "amplitude_mismatch": g,
        "phase_error_rad": phi,
        "phase_error_deg": float(np.degrees(phi)),
        "estimated_irr_db": irr_db,
        "imbalance_detected": detected
    }


def compensate_iq_imbalance(
    signal: np.ndarray,
    g: float,
    phi: float
) -> np.ndarray:
    """
    Applies Gram-Schmidt orthogonalization to correct amplitude mismatch and phase skew.
    """
    if len(signal) == 0:
        return signal

    i = np.real(signal)
    q = np.imag(signal)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    if abs(cos_phi) < 1e-4 or g < 1e-4:
        return signal

    i_corr = i
    q_corr = (q - i * sin_phi) / (g * cos_phi)

    return i_corr + 1j * q_corr


def conditional_iq_conditioning(
    signal: np.ndarray,
    force_compensation: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Evaluates signal for IQ imbalance and conditionally compensates only when justified.
    Reports:
    - IRR before (dB)
    - estimated imbalance parameters (gain, phase)
    - IRR after (dB)
    - compensation applied = "YES" or "NO"
    """
    metrics = estimate_iq_imbalance(signal)
    irr_before = float(metrics["estimated_irr_db"])
    metrics["irr_before"] = irr_before

    if force_compensation or metrics["imbalance_detected"]:
        compensated = compensate_iq_imbalance(
            signal,
            g=metrics["amplitude_mismatch"],
            phi=metrics["phase_error_rad"]
        )
        post_metrics = estimate_iq_imbalance(compensated)
        irr_after = float(post_metrics["estimated_irr_db"])
        metrics["irr_after"] = irr_after
        metrics["compensation_applied"] = "YES"
        metrics["is_compensated"] = True
        return compensated, metrics
    else:
        metrics["irr_after"] = irr_before
        metrics["compensation_applied"] = "NO"
        metrics["is_compensated"] = False
        return signal, metrics
