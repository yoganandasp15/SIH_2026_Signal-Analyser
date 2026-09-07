"""
Framing, Synchronization, and CRC Validation Package
====================================================
Implements:
1. Complete CRC engine parameterized by (poly, init, refin, refout, xorout, width)
2. Sync-word cross-correlation (Barker 7/11/13, CCSDS 0x1ACFFC1D, HDLC 0x7E, DMR)
3. Generic frame repetition and structure discovery
4. Header and payload extraction with ASCII/Hex representations
"""

from .crc import (
    CRCProfile,
    CRC_PROFILES,
    compute_crc,
    verify_crc_stream
)
from .correlator import (
    correlate_sync_words,
    PREAMBLE_CATALOG
)
from .framing_engine import analyze_frames

# Compatibility alias
analyze_frame_structure = analyze_frames

__all__ = [
    "CRCProfile",
    "CRC_PROFILES",
    "compute_crc",
    "verify_crc_stream",
    "correlate_sync_words",
    "PREAMBLE_CATALOG",
    "analyze_frames",
    "analyze_frame_structure"
]

