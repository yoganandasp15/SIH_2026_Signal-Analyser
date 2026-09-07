"""
Convolutional De-interleaving Module
====================================
Implements Ramsey (Type III) / Forney convolutional shift-register interleaving
and de-interleaving.
Number of branches: B
Delay step: M
Interleaver branch j delay: j * M
De-interleaver branch j delay: (B - 1 - j) * M
Total end-to-end delay through every branch is constant: (B - 1) * M.
"""

from typing import List, Optional, Any
import numpy as np


def interleave_convolutional(
    data: np.ndarray,
    branches: int = 4,
    delay_step: int = 2,
    M: Optional[int] = None
) -> np.ndarray:
    """
    Applies Forney convolutional interleaver across B branches with delay increment M.
    """
    if M is not None:
        delay_step = M
    n = len(data)
    if n == 0 or branches <= 1:
        return data.copy()

    # Shift registers for each branch
    delay_lines: List[List[Any]] = [
        [0.0] * (j * delay_step) for j in range(branches)
    ]

    out = np.zeros(n, dtype=data.dtype)
    for i in range(n):
        branch_idx = i % branches
        val = data[i]
        delays = delay_lines[branch_idx]
        if len(delays) > 0:
            out[i] = delays.pop(0)
            delays.append(val)
        else:
            out[i] = val

    return out


def deinterleave_convolutional(
    data: np.ndarray,
    branches: int = 4,
    delay_step: int = 2,
    M: Optional[int] = None,
    shift_latency: Optional[bool] = None
) -> np.ndarray:
    """
    Applies Forney convolutional de-interleaver with complementary branch delays:
    Delay for branch j is: (branches - 1 - j) * delay_step.
    """
    if M is not None:
        delay_step = M

    n = len(data)
    if n == 0 or branches <= 1:
        return data.copy()

    total_delay = branches * (branches - 1) * delay_step
    delay_lines: List[List[Any]] = [
        [0.0] * ((branches - 1 - j) * delay_step) for j in range(branches)
    ]

    out = np.zeros(n, dtype=data.dtype)
    for i in range(n):
        branch_idx = i % branches
        val = data[i]
        delays = delay_lines[branch_idx]
        if len(delays) > 0:
            out[i] = delays.pop(0)
            delays.append(val)
        else:
            out[i] = val

    if shift_latency and n > total_delay:
        return out[total_delay:]
    return out
