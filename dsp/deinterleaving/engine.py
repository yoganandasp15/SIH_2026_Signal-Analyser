"""
De-interleaving Hypothesis Engine
=================================
Performs candidate generation and ranking across the four interleaver topologies:
Periodicity Detector -> Candidate Frame Lengths -> Candidate Parameter Library -> Structural Ranking.
Marks hypotheses as HYPOTHESIZED until closed-loop FEC or CRC validation confirms them.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

from dsp.contracts import (
    InterleaverType,
    InterleaverHypothesis,
    EpistemicStatus
)
from .block import deinterleave_block
from .convolutional import deinterleave_convolutional
from .diagonal import deinterleave_diagonal
from .pseudorandom import deinterleave_pseudorandom, CANONICAL_LFSRS


def detect_candidate_periods(
    bits: np.ndarray,
    max_lag: int = 1024,
    max_candidates: int = 5
) -> List[int]:
    """
    Detects potential periodic structures in the bitstream via autocorrelation.
    Returns candidate block lengths without claiming exact blind identification.
    """
    n = len(bits)
    if n < 64:
        return []

    # Map binary [0, 1] to bipolar [-1, 1] for zero-mean autocorrelation
    bipolar = 2.0 * bits.astype(float) - 1.0
    limit_lag = min(max_lag, n // 2)

    # Compute biased autocorrelation
    r_xx = np.correlate(bipolar, bipolar, mode='full')
    r_xx = r_xx[n - 1 : n - 1 + limit_lag]
    r_norm = r_xx / (r_xx[0] + 1e-12)

    # Exclude lag 0 to 7 to prevent small-word bias
    if len(r_norm) < 16:
        return []

    search_slice = r_norm[8:]
    mean_val = float(np.mean(search_slice))
    std_val = float(np.std(search_slice))
    thresh = mean_val + 1.8 * std_val

    candidate_lags = []
    for lag in range(8, len(r_norm) - 1):
        val = r_norm[lag]
        if val > thresh and val > r_norm[lag - 1] and val > r_norm[lag + 1]:
            candidate_lags.append((val, lag))

    candidate_lags.sort(key=lambda x: x[0], reverse=True)
    return [lag for _, lag in candidate_lags[:max_candidates]]


def search_interleaver_candidates(
    llrs: np.ndarray,
    hard_bits: np.ndarray,
    max_candidates: int = 8
) -> List[InterleaverHypothesis]:
    """
    Generates and structurally scores candidate de-interleaving hypotheses.
    Always includes a baseline NONE hypothesis.
    """
    candidates: List[InterleaverHypothesis] = []

    # 1. Baseline: No interleaving (NONE)
    candidates.append(InterleaverHypothesis(
        topology=InterleaverType.NONE,
        depth=1,
        span=1,
        parameters={},
        deinterleaved_llrs=llrs.copy(),
        confidence=0.70,
        status=EpistemicStatus.HYPOTHESIZED
    ))

    if len(llrs) < 64:
        return candidates

    detected_periods = detect_candidate_periods(hard_bits, max_lag=512)

    # 2. Block De-interleaver Candidates
    standard_grids = [(8, 16), (16, 16), (16, 32), (32, 32)]
    # Add divisors of detected periods if applicable
    for p in detected_periods:
        for r in [8, 16, 32]:
            if p % r == 0 and (r, p // r) not in standard_grids:
                standard_grids.append((r, p // r))

    for r, c in standard_grids[:4]:
        if len(llrs) >= r * c:
            deint_llrs = deinterleave_block(llrs, rows=r, cols=c)
            candidates.append(InterleaverHypothesis(
                topology=InterleaverType.BLOCK,
                depth=r,
                span=c,
                parameters={"rows": r, "cols": c},
                deinterleaved_llrs=deint_llrs,
                confidence=0.60 if (r * c) in detected_periods else 0.45,
                status=EpistemicStatus.HYPOTHESIZED
            ))

    # 3. Convolutional De-interleaver Candidates
    conv_profiles = [(4, 2), (6, 2), (8, 4)]
    for b, m in conv_profiles:
        if len(llrs) >= b * m * 4:
            deint_llrs = deinterleave_convolutional(llrs, branches=b, delay_step=m)
            candidates.append(InterleaverHypothesis(
                topology=InterleaverType.CONVOLUTIONAL,
                depth=b,
                span=m,
                parameters={"branches": b, "delay_step": m},
                deinterleaved_llrs=deint_llrs,
                confidence=0.48,
                status=EpistemicStatus.HYPOTHESIZED
            ))

    # 4. Diagonal De-interleaver Candidates
    diag_profiles = [(8, 16), (16, 16)]
    for r, c in diag_profiles:
        if len(llrs) >= r * c:
            deint_llrs = deinterleave_diagonal(llrs, rows=r, cols=c)
            candidates.append(InterleaverHypothesis(
                topology=InterleaverType.DIAGONAL,
                depth=r,
                span=c,
                parameters={"rows": r, "cols": c},
                deinterleaved_llrs=deint_llrs,
                confidence=0.42,
                status=EpistemicStatus.HYPOTHESIZED
            ))

    # 5. Pseudo-Random LFSR Candidates (Strictly bounded)
    for lfsr in CANONICAL_LFSRS[:2]:
        for seed in [1, 3]:
            for blk in [128, 256]:
                if len(llrs) >= blk:
                    deint_llrs = deinterleave_pseudorandom(
                        llrs,
                        block_size=blk,
                        poly=lfsr["poly"],
                        degree=lfsr["degree"],
                        seed=seed
                    )
                    candidates.append(InterleaverHypothesis(
                        topology=InterleaverType.PSEUDO_RANDOM,
                        depth=blk,
                        span=1,
                        parameters={
                            "block_size": blk,
                            "poly_name": lfsr["name"],
                            "poly": lfsr["poly"],
                            "seed": seed
                        },
                        deinterleaved_llrs=deint_llrs,
                        confidence=0.40,
                        status=EpistemicStatus.HYPOTHESIZED
                    ))
                    break  # Bounded to 1 per seed

    # Sort candidates by confidence
    candidates.sort(key=lambda x: x.confidence, reverse=True)
    return candidates[:max_candidates]
