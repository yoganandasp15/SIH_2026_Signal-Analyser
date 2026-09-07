"""
Diagonal / Helical De-interleaving Module
=========================================
Implements diagonal skew interleaving and de-interleaving.
Data is organized in (rows x cols) frames, read out along diagonal trajectories
to disperse adjacent burst errors in both time and frequency axes.
"""

from typing import Optional
import numpy as np


def interleave_diagonal(
    data: np.ndarray,
    rows: int = 8,
    cols: int = 16,
    n: Optional[int] = None
) -> np.ndarray:
    """
    Applies diagonal interleaver: fills rows x cols row-by-row,
    reads out along skewed diagonals: index (r, (c + r) % cols).
    """
    if n is not None:
        rows = n
        cols = n
    block_len = rows * cols
    data_len = len(data)
    num_blocks = data_len // block_len
    if num_blocks == 0:
        return data.copy()

    truncated = num_blocks * block_len
    out_blocks = []

    for b in range(num_blocks):
        block = data[b * block_len : (b + 1) * block_len].reshape((rows, cols))
        interleaved_block = np.zeros((rows, cols), dtype=data.dtype)
        for r in range(rows):
            for c in range(cols):
                skew_c = (c + r) % cols
                interleaved_block[r, c] = block[r, skew_c]
        out_blocks.append(interleaved_block.reshape(-1))

    res = np.concatenate(out_blocks)
    if data_len > truncated:
        return np.concatenate([res, data[truncated:]])
    return res


def deinterleave_diagonal(
    data: np.ndarray,
    rows: int = 8,
    cols: int = 16,
    n: Optional[int] = None
) -> np.ndarray:
    """
    Inverts diagonal interleaver by placing elements back into their original coordinates:
    index (r, (c + r) % cols).
    """
    if n is not None:
        rows = n
        cols = n
    block_len = rows * cols
    data_len = len(data)
    num_blocks = data_len // block_len
    if num_blocks == 0:
        return data.copy()

    truncated = num_blocks * block_len
    out_blocks = []

    for b in range(num_blocks):
        block = data[b * block_len : (b + 1) * block_len].reshape((rows, cols))
        deinterleaved_block = np.zeros((rows, cols), dtype=data.dtype)
        for r in range(rows):
            for c in range(cols):
                orig_c = (c + r) % cols
                deinterleaved_block[r, orig_c] = block[r, c]
        out_blocks.append(deinterleaved_block.reshape(-1))

    res = np.concatenate(out_blocks)
    if data_len > truncated:
        return np.concatenate([res, data[truncated:]])
    return res
