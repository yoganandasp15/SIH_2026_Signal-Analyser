"""
Frequency Shift Keying (FSK) Demodulator Module
===============================================
Implements statistical soft demodulation for 2-FSK and 4-FSK using a quadrature
frequency discriminator with variance-normalized Max-Log log-likelihood ratios.

Statistical Model:
------------------
Let d[k] be the integrated discriminator decision variable around symbol k:
d[k] ~ N(f_m, sigma_f^2)
where f_m is the nominal tone frequency and sigma_f^2 is the discriminator noise variance.

For 2-FSK:
LLR = ( (d[k] - f_1)^2 - (d[k] - f_0)^2 ) / (2 * sigma_f^2)

For 4-FSK:
Max-Log soft bit likelihoods over the 4 tone hypotheses using standard Gray mapping:
+3: (0, 1),  +1: (0, 0),  -1: (1, 0),  -3: (1, 1)
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def demodulate_fsk(
    signal: np.ndarray,
    order: int = 2,
    sps: float = 2.0,
    noise_variance: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Demodulates 2-FSK or 4-FSK baseband signals into hard bits and soft LLRs.

    Parameters:
    -----------
    signal : np.ndarray
        Complex baseband signal.
    order : int
        2 for 2-FSK, 4 for 4-FSK.
    sps : float
        Samples per symbol.
    noise_variance : float
        Estimated noise variance.

    Returns:
    --------
    Tuple[hard_bits, soft_llrs, ser_est]
    """
    n = len(signal)
    if n < 4:
        return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.float32), 0.0

    # Instantaneous frequency discriminator: diff of unrolled phase
    if np.iscomplexobj(signal):
        inst_phase_diff = np.angle(signal[1:] * np.conj(signal[:-1]))
    else:
        from scipy.signal import hilbert
        an = hilbert(signal)
        inst_phase_diff = np.angle(an[1:] * np.conj(an[:-1]))

    stride = max(1, int(round(sps)))
    # Integrate/average discriminator over each symbol interval
    num_symbols = len(inst_phase_diff) // stride
    if num_symbols == 0:
        return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.float32), 0.0

    reshaped = inst_phase_diff[:num_symbols * stride].reshape((num_symbols, stride))
    d_syms = np.mean(reshaped, axis=1)

    # Estimate discriminator noise variance
    detrended = d_syms - np.median(d_syms)
    sigma_f2 = max(1e-4, float(np.var(detrended) * (noise_variance / (1.0 + noise_variance))))

    if order == 2:
        # 2-FSK: Estimate mark and space tone deviations
        pos_mask = d_syms > np.median(d_syms)
        neg_mask = ~pos_mask

        f_1 = float(np.mean(d_syms[pos_mask])) if np.any(pos_mask) else 0.5
        f_0 = float(np.mean(d_syms[neg_mask])) if np.any(neg_mask) else -0.5

        # LLR = ((d - f_1)^2 - (d - f_0)^2) / (2 * sigma_f^2)
        # Note: LLR > 0 means bit 0 is closer; LLR < 0 means bit 1 is closer
        d0_sq = (d_syms - f_0) ** 2
        d1_sq = (d_syms - f_1) ** 2
        llrs = ((d1_sq - d0_sq) / (2.0 * sigma_f2)).astype(np.float32)
        llrs = np.clip(llrs, -20.0, 20.0)

        hard_bits = (llrs < 0.0).astype(np.uint8)
        ser_est = float(0.5 * np.exp(-abs(f_1 - f_0) / (2.0 * np.sqrt(sigma_f2) + 1e-6)))

    else:
        # 4-FSK: 4 tones spaced symmetrically around center frequency
        p95 = float(np.percentile(d_syms, 95))
        p05 = float(np.percentile(d_syms, 5))
        center = (p95 + p05) / 2.0
        delta_f = max(0.02, (p95 - p05) / 3.0)

        # Tone hypotheses: Tone 0 (+3), Tone 1 (+1), Tone 2 (-1), Tone 3 (-3)
        tones = [
            center + 1.5 * delta_f,  # Tone 0 -> (0, 1)
            center + 0.5 * delta_f,  # Tone 1 -> (0, 0)
            center - 0.5 * delta_f,  # Tone 2 -> (1, 0)
            center - 1.5 * delta_f   # Tone 3 -> (1, 1)
        ]

        d0 = (d_syms - tones[0]) ** 2
        d1 = (d_syms - tones[1]) ** 2
        d2 = (d_syms - tones[2]) ** 2
        d3 = (d_syms - tones[3]) ** 2

        # Bit 0: Tone 0,1 has bit0=0; Tone 2,3 has bit0=1
        min_d_b0_0 = np.minimum(d0, d1)
        min_d_b0_1 = np.minimum(d2, d3)
        llr_b0 = (min_d_b0_1 - min_d_b0_0) / (2.0 * sigma_f2)

        # Bit 1: Tone 1,2 has bit1=0; Tone 0,3 has bit1=1
        min_d_b1_0 = np.minimum(d1, d2)
        min_d_b1_1 = np.minimum(d0, d3)
        llr_b1 = (min_d_b1_1 - min_d_b1_0) / (2.0 * sigma_f2)

        llrs = np.zeros(num_symbols * 2, dtype=np.float32)
        llrs[0::2] = np.clip(llr_b0, -20.0, 20.0)
        llrs[1::2] = np.clip(llr_b1, -20.0, 20.0)

        hard_bits = (llrs < 0.0).astype(np.uint8)
        ser_est = float(0.25 * np.exp(-delta_f / (2.0 * np.sqrt(sigma_f2) + 1e-6)))

    return hard_bits, llrs, ser_est
