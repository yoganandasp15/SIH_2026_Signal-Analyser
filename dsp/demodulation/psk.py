"""
Phase Shift Keying (PSK) Demodulator Module
===========================================
Demodulates BPSK, QPSK, and 8-PSK symbols with Gray mapping, producing both
hard-decision binary bitstreams and soft Log-Likelihood Ratios (LLRs).
"""

from typing import Tuple, Dict
import numpy as np

from .llr import compute_maxlog_llrs


def get_bpsk_constellation() -> Dict[Tuple[int, ...], complex]:
    """BPSK mapping: 0 -> +1.0, 1 -> -1.0"""
    return {
        (0,): 1.0 + 0.0j,
        (1,): -1.0 + 0.0j
    }


def get_qpsk_constellation() -> Dict[Tuple[int, ...], complex]:
    """Gray-coded QPSK mapping normalized to unit energy."""
    scale = 1.0 / np.sqrt(2.0)
    return {
        (0, 0): complex(scale, scale),
        (0, 1): complex(-scale, scale),
        (1, 1): complex(-scale, -scale),
        (1, 0): complex(scale, -scale)
    }


def get_8psk_constellation() -> Dict[Tuple[int, ...], complex]:
    """Gray-coded 8-PSK mapping on the unit circle."""
    gray_order = [
        (0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0),
        (1, 1, 0), (1, 1, 1), (1, 0, 1), (1, 0, 0)
    ]
    mapping = {}
    for i, b_tuple in enumerate(gray_order):
        phase = 2.0 * np.pi * i / 8.0
        mapping[b_tuple] = np.exp(1j * phase)
    return mapping


def demodulate_psk(
    symbols: np.ndarray,
    order: int = 4,
    noise_variance: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Demodulates M-PSK symbols (M = 2, 4, or 8).

    Returns:
    --------
    Tuple[hard_bits, soft_llrs, evm_pct]
    """
    if len(symbols) == 0:
        return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.float32), 0.0

    if order == 2:
        constellation = get_bpsk_constellation()
    elif order == 8:
        constellation = get_8psk_constellation()
    else:
        constellation = get_qpsk_constellation()

    hard_bits, soft_llrs = compute_maxlog_llrs(
        symbols,
        constellation_map=constellation,
        noise_variance=noise_variance
    )

    # Compute EVM (Error Vector Magnitude) %
    pts = np.array(list(constellation.values()))
    # Map each symbol to closest constellation point
    closest_pts = pts[np.argmin(np.abs(symbols[:, np.newaxis] - pts[np.newaxis, :]), axis=1)]
    error_vec = symbols - closest_pts
    evm_pct = float(np.sqrt(np.mean(np.abs(error_vec) ** 2)) / (np.sqrt(np.mean(np.abs(closest_pts) ** 2)) + 1e-12) * 100.0)

    return hard_bits, soft_llrs, evm_pct
