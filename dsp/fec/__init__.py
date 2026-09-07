"""
Forward Error Correction (FEC) Decoding & Validation Package
============================================================
Implements defense and aerospace standard FEC algorithms:
1. Convolutional Viterbi Decoder (Soft & Hard, K=3..7, standard octal generators)
2. Reed-Solomon Algebraic Decoder over GF(2^8) (Berlekamp-Massey, Chien, Forney)
3. Low-Density Parity-Check (LDPC) Normalized Min-Sum Decoder (DVB-S2 / WiMAX)
4. Concatenated Decoder (Inner Viterbi -> De-interleaver -> Outer Reed-Solomon)
5. Strict Algebraic Validation Engine (H * c^T == 0 mod 2, RS GF(256) zero syndrome)
"""

from .viterbi import (
    encode_convolutional,
    decode_viterbi_soft,
    decode_viterbi_hard,
    CONV_GENERATORS
)
from .reed_solomon import (
    GF256,
    encode_reed_solomon,
    decode_reed_solomon,
    RS_PROFILES
)
from .ldpc import (
    decode_ldpc_minsum,
    get_standard_ldpc_matrix
)
from .concatenated import decode_concatenated_chain
from .profiles import (
    STANDARD_RS_PROFILES,
    STANDARD_CONV_PROFILES,
    STANDARD_LDPC_PROFILES,
    ReedSolomonProfile,
    ConvolutionalProfile,
    LDPCProfile
)
from .ranker import rank_fec_candidates
from .engine import evaluate_fec_hypotheses

# Compatibility alias
evaluate_fec_candidates = rank_fec_candidates

__all__ = [
    "encode_convolutional",
    "decode_viterbi_soft",
    "decode_viterbi_hard",
    "CONV_GENERATORS",
    "GF256",
    "encode_reed_solomon",
    "decode_reed_solomon",
    "RS_PROFILES",
    "STANDARD_RS_PROFILES",
    "STANDARD_CONV_PROFILES",
    "STANDARD_LDPC_PROFILES",
    "ReedSolomonProfile",
    "ConvolutionalProfile",
    "LDPCProfile",
    "decode_ldpc_minsum",
    "get_standard_ldpc_matrix",
    "decode_concatenated_chain",
    "evaluate_fec_hypotheses",
    "evaluate_fec_candidates",
    "rank_fec_candidates"
]

