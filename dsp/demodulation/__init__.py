"""
Demodulation and Soft Information (LLR) Engine Package
======================================================
Demodulates synchronized baseband symbols into both:
- Hard-decision binary bitstreams (uint8 arrays)
- Soft Log-Likelihood Ratios (LLRs) normalized by noise variance
Supported families: 2-FSK, 4-FSK, BPSK, QPSK, 8-PSK, 16-QAM, 64-QAM
"""

from .llr import compute_maxlog_llrs
from .psk import demodulate_psk
from .qam import demodulate_qam
from .fsk import demodulate_fsk
from .demodulator import demodulate_symbols

# Compatibility alias
demodulate_signal = demodulate_symbols

__all__ = [
    "compute_maxlog_llrs",
    "demodulate_psk",
    "demodulate_qam",
    "demodulate_fsk",
    "demodulate_symbols",
    "demodulate_signal"
]

