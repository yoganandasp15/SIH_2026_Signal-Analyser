"""
Pseudo-Random De-interleaving Module
====================================
Implements pseudo-random interleaving via seeded Galois LFSR permutations.
Includes a strictly bounded search space over canonical polynomials, seeds,
and block sizes to prevent search-space explosion.

Candidate Library:
------------------
- Polynomial degrees: 5 (0x12), 7 (0x48), 9 (0x110)
- Seeds: 1, 3, 5, 7, 9, 11, 13, 15
- Block sizes: 64, 128, 256, 512
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


CANONICAL_LFSRS = [
    {"degree": 5, "poly": 0x12, "name": "x^5 + x^2 + 1"},
    {"degree": 7, "poly": 0x48, "name": "x^7 + x^3 + 1"},
    {"degree": 9, "poly": 0x110, "name": "x^9 + x^4 + 1"}
]

ALLOWED_SEEDS = [1, 3, 5, 7, 9, 11, 13, 15]
ALLOWED_BLOCK_SIZES = [64, 128, 256, 512]


def generate_lfsr_permutation(
    block_size: int,
    poly: int = 0x48,
    degree: int = 7,
    seed: int = 1
) -> np.ndarray:
    """
    Generates a deterministic pseudo-random permutation of length block_size
    using an LFSR sequence with duplicate rejection and guaranteed termination.
    """
    # Ensure degree is large enough to cover the block size
    eff_degree = max(degree, int(np.ceil(np.log2(max(block_size, 2)))))
    eff_poly = poly
    if eff_degree > 7:
        eff_degree = 9
        eff_poly = 0x110
    elif eff_degree < 5:
        eff_degree = 5
        eff_poly = 0x12

    mask = (1 << eff_degree) - 1
    state = seed & mask
    if state == 0:
        state = 1

    seen = set()
    perm: List[int] = []
    max_steps = (1 << eff_degree) * 2

    step = 0
    while len(perm) < block_size and step < max_steps:
        step += 1
        lsb = state & 1
        state >>= 1
        if lsb:
            state ^= eff_poly

        candidate = (state - 1) % block_size
        if 0 <= candidate < block_size and candidate not in seen:
            seen.add(candidate)
            perm.append(candidate)

    # Append any remaining unvisited indices to guarantee a complete bijection
    for idx in range(block_size):
        if idx not in seen:
            seen.add(idx)
            perm.append(idx)

    return np.array(perm, dtype=int)


def invert_permutation(perm: np.ndarray) -> np.ndarray:
    """Computes the inverse permutation pi^-1 such that pi^-1(pi(i)) == i."""
    inv = np.zeros(len(perm), dtype=int)
    inv[perm] = np.arange(len(perm))
    return inv


def interleave_pseudorandom(
    data: np.ndarray,
    block_size: int = 128,
    poly: int = 0x48,
    degree: int = 7,
    seed: int = 1,
    length: Optional[int] = None
) -> np.ndarray:
    """Interleaves data using an LFSR permutation."""
    if length is not None:
        block_size = length

    n = len(data)
    num_blocks = n // block_size
    if num_blocks == 0:
        return data.copy()

    perm = generate_lfsr_permutation(block_size, poly, degree, seed)
    truncated = num_blocks * block_size
    out = np.zeros(truncated, dtype=data.dtype)

    for b in range(num_blocks):
        blk = data[b * block_size : (b + 1) * block_size]
        out[b * block_size : (b + 1) * block_size] = blk[perm]

    if n > truncated:
        return np.concatenate([out, data[truncated:]])
    return out


def deinterleave_pseudorandom(
    data: np.ndarray,
    block_size: int = 128,
    poly: int = 0x48,
    degree: int = 7,
    seed: int = 1,
    length: Optional[int] = None
) -> np.ndarray:
    """De-interleaves data by applying the inverse LFSR permutation."""
    if length is not None:
        block_size = length

    n = len(data)
    num_blocks = n // block_size
    if num_blocks == 0:
        return data.copy()

    perm = generate_lfsr_permutation(block_size, poly, degree, seed)
    inv_perm = invert_permutation(perm)
    truncated = num_blocks * block_size
    out = np.zeros(truncated, dtype=data.dtype)

    for b in range(num_blocks):
        blk = data[b * block_size : (b + 1) * block_size]
        out[b * block_size : (b + 1) * block_size] = blk[inv_perm]

    if n > truncated:
        return np.concatenate([out, data[truncated:]])
    return out
