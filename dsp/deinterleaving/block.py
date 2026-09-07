"""
Block De-interleaving Module
============================
Reverses row-column rectangular matrix interleaving.
Interleaver writes data by rows and reads by columns.
De-interleaver writes data by columns and reads by rows.
"""

from typing import Union
import numpy as np


def interleave_block(
    data: np.ndarray,
    rows: int,
    cols: int
) -> np.ndarray:
    """
    Interleaves data by filling an (rows x cols) matrix row-by-row
    and reading it out column-by-column.
    """
    block_len = rows * cols
    n = len(data)
    num_blocks = n // block_len
    if num_blocks == 0:
        return data.copy()

    truncated_len = num_blocks * block_len
    reshaped = data[:truncated_len].reshape((num_blocks, rows, cols))
    # Transpose rows and cols -> (num_blocks, cols, rows)
    transposed = np.transpose(reshaped, (0, 2, 1))
    interleaved = transposed.reshape(-1)

    if n > truncated_len:
        return np.concatenate([interleaved, data[truncated_len:]])
    return interleaved


def deinterleave_block(
    data: np.ndarray,
    rows: int,
    cols: int
) -> np.ndarray:
    """
    De-interleaves data by filling an (cols x rows) matrix row-by-row
    (equivalent to filling rows x cols by columns) and reading row-by-row.
    """
    block_len = rows * cols
    n = len(data)
    num_blocks = n // block_len
    if num_blocks == 0:
        return data.copy()

    truncated_len = num_blocks * block_len
    reshaped = data[:truncated_len].reshape((num_blocks, cols, rows))
    # Transpose back: (num_blocks, rows, cols)
    transposed = np.transpose(reshaped, (0, 2, 1))
    deinterleaved = transposed.reshape(-1)

    if n > truncated_len:
        return np.concatenate([deinterleaved, data[truncated_len:]])
    return deinterleaved
