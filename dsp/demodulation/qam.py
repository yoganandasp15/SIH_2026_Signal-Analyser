"""
Quadrature Amplitude Modulation (QAM) Demodulator Module
========================================================
Demodulates 16-QAM and 64-QAM square constellations with Gray mapping,
producing hard bits, soft LLRs, and EVM telemetry.
"""

from typing import Tuple, Dict
import numpy as np

from .llr import compute_maxlog_llrs


def get_16qam_constellation() -> Dict[Tuple[int, ...], complex]:
    """Standard Gray-coded 16-QAM constellation normalized to unit energy."""
    levels = [-3.0, -1.0, 1.0, 3.0]
    gray_map_2bit = {
        -3.0: (0, 0),
        -1.0: (0, 1),
         1.0: (1, 1),
         3.0: (1, 0)
    }
    scale = 1.0 / np.sqrt(10.0)

    mapping = {}
    for i_val in levels:
        for q_val in levels:
            bits = gray_map_2bit[i_val] + gray_map_2bit[q_val]
            mapping[bits] = complex(i_val * scale, q_val * scale)
    return mapping


def get_64qam_constellation() -> Dict[Tuple[int, ...], complex]:
    """Standard Gray-coded 64-QAM constellation normalized to unit energy."""
    levels = [-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]
    gray_map_3bit = {
        -7.0: (0, 0, 0),
        -5.0: (0, 0, 1),
        -3.0: (0, 1, 1),
        -1.0: (0, 1, 0),
         1.0: (1, 1, 0),
         3.0: (1, 1, 1),
         5.0: (1, 0, 1),
         7.0: (1, 0, 0)
    }
    scale = 1.0 / np.sqrt(42.0)

    mapping = {}
    for i_val in levels:
        for q_val in levels:
            bits = gray_map_3bit[i_val] + gray_map_3bit[q_val]
            mapping[bits] = complex(i_val * scale, q_val * scale)
    return mapping


def demodulate_qam(
    symbols: np.ndarray,
    order: int = 16,
    noise_variance: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Demodulates 16-QAM or 64-QAM symbols.
    """
    if len(symbols) == 0:
        return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.float32), 0.0

    if order == 64:
        constellation = get_64qam_constellation()
    else:
        constellation = get_16qam_constellation()

    hard_bits, soft_llrs = compute_maxlog_llrs(
        symbols,
        constellation_map=constellation,
        noise_variance=noise_variance
    )

    pts = np.array(list(constellation.values()))
    closest_pts = pts[np.argmin(np.abs(symbols[:, np.newaxis] - pts[np.newaxis, :]), axis=1)]
    error_vec = symbols - closest_pts
    evm_pct = float(np.sqrt(np.mean(np.abs(error_vec) ** 2)) / (np.sqrt(np.mean(np.abs(closest_pts) ** 2)) + 1e-12) * 100.0)

    return hard_bits, soft_llrs, evm_pct
