"""
De-interleaving and Permutation Inversion Package
================================================
Implements the four mandatory de-interleaving topologies:
1. Block De-interleaver (Row/Column Matrix Inversion)
2. Convolutional De-interleaver (Ramsey / Forney Shift Register Inversion)
3. Diagonal De-interleaver (Helical / Skew Matrix Inversion)
4. Pseudo-Random De-interleaver (Galois LFSR Permutation Inversion)
With bounded search space and algebraic periodicity detection.
"""

from .block import deinterleave_block, interleave_block
from .convolutional import deinterleave_convolutional, interleave_convolutional
from .diagonal import deinterleave_diagonal, interleave_diagonal
from .pseudorandom import deinterleave_pseudorandom, interleave_pseudorandom
from .engine import search_interleaver_candidates

# Compatibility alias
evaluate_interleaver_candidates = search_interleaver_candidates

__all__ = [
    "deinterleave_block",
    "interleave_block",
    "deinterleave_convolutional",
    "interleave_convolutional",
    "deinterleave_diagonal",
    "interleave_diagonal",
    "deinterleave_pseudorandom",
    "interleave_pseudorandom",
    "search_interleaver_candidates",
    "evaluate_interleaver_candidates"
]

