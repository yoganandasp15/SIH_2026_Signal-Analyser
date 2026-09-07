"""
Low-Density Parity-Check (LDPC) Decoder Module
==============================================
Implements Normalized Min-Sum iterative soft decoding over GF(2) sparse parity-check matrices.
Tracks iteration count, syndrome weight trajectory, and enforces strict algebraic validation
np.all(H * c^T == 0 mod 2).
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np


def create_quasi_cyclic_ldpc_matrix(
    block_size: int = 16,
    num_row_blocks: int = 4,
    num_col_blocks: int = 8
) -> np.ndarray:
    """
    Constructs a standard Quasi-Cyclic LDPC (QC-LDPC) parity-check matrix H (M x N)
    using circulant permutation submatrices (IEEE 802.16 / WiMAX standard structure).
    """
    Z = block_size
    M = num_row_blocks * Z
    N = num_col_blocks * Z
    H = np.zeros((M, N), dtype=np.uint8)

    # Standard prototype shift matrix for rate 1/2
    shifts = [
        [0, -1,  2, -1,  1,  0, -1, -1],
        [-1, 0, -1,  1, -1,  1,  0, -1],
        [1, -1,  0, -1,  2, -1,  1,  0],
        [-1, 1, -1,  0, -1,  2, -1,  1]
    ]

    for r_idx in range(num_row_blocks):
        for c_idx in range(num_col_blocks):
            shift_val = shifts[r_idx % len(shifts)][c_idx % len(shifts[0])]
            if shift_val >= 0:
                sub = np.eye(Z, dtype=np.uint8)
                sub = np.roll(sub, shift_val % Z, axis=1)
                H[r_idx * Z : (r_idx + 1) * Z, c_idx * Z : (c_idx + 1) * Z] = sub

    return H


def get_standard_ldpc_matrix(profile_name: str = "WIMAX_R12_128") -> np.ndarray:
    """Returns standard LDPC parity check matrix."""
    return create_quasi_cyclic_ldpc_matrix(block_size=16, num_row_blocks=4, num_col_blocks=8)


def decode_ldpc_minsum(
    llrs: np.ndarray,
    H: Optional[np.ndarray] = None,
    max_iter: int = 20,
    alpha: float = 0.80
) -> Tuple[np.ndarray, bool, int, List[int]]:
    """
    Normalized Min-Sum LDPC Decoder.

    Parameters:
    -----------
    llrs : np.ndarray
        Received soft LLRs (length matching N columns of H).
    H : Optional[np.ndarray]
        Parity-check matrix (M x N). If None, standard WIMAX matrix (64 x 128) is used.
    max_iter : int
        Maximum decoding iterations.
    alpha : float
        Min-sum normalization scaling factor (0.75 - 0.85).

    Returns:
    --------
    Tuple[decoded_bits, syndrome_valid, iterations_run, syndrome_weight_trajectory]
    """
    if H is None:
        H = get_standard_ldpc_matrix()

    M, N = H.shape
    n_rx = len(llrs)

    if n_rx < N:
        # Pad with neutral zero LLRs
        input_llrs = np.zeros(N, dtype=np.float32)
        input_llrs[:n_rx] = llrs
    else:
        input_llrs = llrs[:N].astype(np.float32)

    # Pre-extract non-zero indices for variable and check nodes
    check_neighbors: List[np.ndarray] = [np.where(H[m, :] == 1)[0] for m in range(M)]
    var_neighbors: List[np.ndarray] = [np.where(H[:, n] == 1)[0] for n in range(N)]

    # Messages from check to variable: L_c2v[m, n]
    # Messages from variable to check: L_v2c[n, m]
    L_c2v = { (m, n): 0.0 for m in range(M) for n in check_neighbors[m] }
    L_v2c = { (n, m): float(input_llrs[n]) for n in range(N) for m in var_neighbors[n] }

    syndrome_weights: List[int] = []
    decoded_bits = (input_llrs < 0.0).astype(np.uint8)

    # Initial syndrome check
    init_syndrome = np.mod(np.dot(H, decoded_bits), 2)
    init_weight = int(np.sum(init_syndrome))
    syndrome_weights.append(init_weight)

    if np.all(init_syndrome == 0):
        return decoded_bits, True, 0, syndrome_weights

    for iteration in range(1, max_iter + 1):
        # 1. Check Node Update (Normalized Min-Sum)
        for m in range(M):
            v_nodes = check_neighbors[m]
            if len(v_nodes) == 0:
                continue

            for n in v_nodes:
                # Other variable nodes connected to check m
                other_v = [v for v in v_nodes if v != n]
                if not other_v:
                    L_c2v[(m, n)] = 0.0
                    continue

                vals = [L_v2c[(v, m)] for v in other_v]
                # Sign product
                prod_sign = 1.0
                min_mag = 1e9
                for v_val in vals:
                    if v_val < 0.0:
                        prod_sign = -prod_sign
                    mag = abs(v_val)
                    if mag < min_mag:
                        min_mag = mag

                L_c2v[(m, n)] = alpha * prod_sign * min_mag

        # 2. Variable Node Update & A Posteriori LLRs
        total_llrs = input_llrs.copy()
        for n in range(N):
            c_nodes = var_neighbors[n]
            sum_c2v = sum(L_c2v[(m, n)] for m in c_nodes)
            total_llrs[n] += sum_c2v

            for m in c_nodes:
                # Exclude check m
                L_v2c[(n, m)] = float(input_llrs[n] + (sum_c2v - L_c2v[(m, n)]))

        # 3. Hard Decision & Syndrome Parity Satisfaction Check
        decoded_bits = (total_llrs < 0.0).astype(np.uint8)
        syndrome = np.mod(np.dot(H, decoded_bits), 2)
        weight = int(np.sum(syndrome))
        syndrome_weights.append(weight)

        # STRICT VALIDATION: np.all(syndrome == 0)
        if np.all(syndrome == 0):
            return decoded_bits, True, iteration, syndrome_weights

    return decoded_bits, False, max_iter, syndrome_weights
