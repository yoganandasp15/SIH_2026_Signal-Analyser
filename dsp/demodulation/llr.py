"""
Log-Likelihood Ratio (LLR) Computation Module
=============================================
Computes soft bits using the Max-Log approximation over constellation points
parameterized by the channel noise variance sigma^2:

LLR(b_i) = ln( P(b_i = 0 | y) / P(b_i = 1 | y) )
         ~= (1 / (2 * sigma^2)) * [ min_{s in S_i^(1)} |y - s|^2 - min_{s in S_i^(0)} |y - s|^2 ]

Convention:
-----------
LLR > 0 -> Bit 0 more likely (hard bit = 0)
LLR < 0 -> Bit 1 more likely (hard bit = 1)
"""

from typing import Dict, List, Tuple
import numpy as np


def compute_maxlog_llrs(
    symbols: np.ndarray,
    constellation_map: Dict[Tuple[int, ...], complex],
    noise_variance: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes soft LLRs and hard bits for arbitrary constellations.

    Parameters:
    -----------
    symbols : np.ndarray
        Received complex symbols (normalized).
    constellation_map : Dict[Tuple[int, ...], complex]
        Mapping from bit tuple (b_0, b_1, ...) to complex constellation point.
    noise_variance : float
        Estimated noise variance sigma^2.

    Returns:
    --------
    Tuple[hard_bits, soft_llrs]
    """
    if len(symbols) == 0 or len(constellation_map) == 0:
        return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.float32)

    sigma2 = max(1e-4, float(noise_variance))
    bits_per_symbol = len(next(iter(constellation_map.keys())))
    num_symbols = len(symbols)

    all_points = np.array(list(constellation_map.values()), dtype=np.complex64)
    all_bit_tuples = list(constellation_map.keys())

    # Pre-partition constellation indices for each bit position
    bit_indices_0: List[List[int]] = [[] for _ in range(bits_per_symbol)]
    bit_indices_1: List[List[int]] = [[] for _ in range(bits_per_symbol)]

    for idx, b_tuple in enumerate(all_bit_tuples):
        for bit_pos, b in enumerate(b_tuple):
            if b == 0:
                bit_indices_0[bit_pos].append(idx)
            else:
                bit_indices_1[bit_pos].append(idx)

    llrs = np.zeros(num_symbols * bits_per_symbol, dtype=np.float32)

    # Vectorized distance computation for batches
    batch_size = 2048
    for start in range(0, num_symbols, batch_size):
        end = min(start + batch_size, num_symbols)
        batch_syms = symbols[start:end, np.newaxis]  # (B, 1)

        # Squared Euclidean distance: |y - s|^2 -> (B, M)
        sq_dist = np.abs(batch_syms - all_points[np.newaxis, :]) ** 2

        for bit_pos in range(bits_per_symbol):
            # Min distance to symbols where bit_pos == 0
            min_d0 = np.min(sq_dist[:, bit_indices_0[bit_pos]], axis=1)
            # Min distance to symbols where bit_pos == 1
            min_d1 = np.min(sq_dist[:, bit_indices_1[bit_pos]], axis=1)

            # LLR = (min_d1 - min_d0) / (2 * sigma^2)
            llr_batch = (min_d1 - min_d0) / (2.0 * sigma2)
            # Clamp to prevent extreme overflows
            llr_batch = np.clip(llr_batch, -20.0, 20.0)

            out_indices = np.arange(start, end) * bits_per_symbol + bit_pos
            llrs[out_indices] = llr_batch

    hard_bits = (llrs < 0.0).astype(np.uint8)
    return hard_bits, llrs
