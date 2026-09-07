"""
Preamble and Sync-Word Cross-Correlation Module
===============================================
Correlates incoming bitstreams against standard defense and aerospace preambles:
- Barker Codes (Barker-7, Barker-11, Barker-13)
- CCSDS 32-Bit Synchronization Marker (0x1ACFFC1D)
- HDLC / AX.25 Framing Flag (0x7E = 01111110)
- ETSI DMR Synchronization Patterns
Supports automatic polarity inversion detection (+/- 180 deg phase ambiguity).
"""

from typing import List, Dict, Any, Tuple
import numpy as np


PREAMBLE_CATALOG: Dict[str, Dict[str, Any]] = {
    "Barker-7": {
        "bits": np.array([1, 1, 1, 0, 0, 1, 0], dtype=np.uint8),
        "hex": "0x72",
        "name": "Barker-7 Sequence"
    },
    "Barker-11": {
        "bits": np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=np.uint8),
        "hex": "0x712",
        "name": "Barker-11 Sequence (DSSS / 802.11)"
    },
    "Barker-13": {
        "bits": np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=np.uint8),
        "hex": "0x1F35",
        "name": "Barker-13 Radar / Telemetry Preamble"
    },
    "CCSDS-32": {
        "bits": np.unpackbits(np.array([0x1A, 0xCF, 0xFC, 0x1D], dtype=np.uint8)),
        "hex": "0x1ACFFC1D",
        "name": "CCSDS Telemetry Attached Sync Marker (ASM)"
    },
    "HDLC-8": {
        "bits": np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8),
        "hex": "0x7E",
        "name": "HDLC / AX.25 Frame Delimiter (0x7E)"
    },
    "DMR-BS-Sync": {
        "bits": np.unpackbits(np.array([0x77, 0xD5, 0x5F, 0x7D, 0x2D], dtype=np.uint8))[:36],
        "hex": "0x77D55F7D2D",
        "name": "DMR Base Station Voice/Data Sync"
    }
}


def correlate_sync_words(
    bits: np.ndarray,
    max_bit_errors: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Scans the bitstream for matches against the preamble catalog.
    Checks both normal and inverted polarities.
    Enforces exact match on short patterns (<= 11 bits) to eliminate false positives.
    """
    n = len(bits)
    matches: List[Dict[str, Any]] = []
    if n < 8:
        return matches

    for name, info in PREAMBLE_CATALOG.items():
        pattern = info["bits"]
        p_len = len(pattern)
        if n < p_len:
            continue

        # Scale error tolerance with pattern length
        if max_bit_errors is not None:
            allowed_errors = max_bit_errors
        else:
            if p_len <= 11:
                allowed_errors = 0
            elif p_len <= 16:
                allowed_errors = 1
            else:
                allowed_errors = 2

        # Sliding window correlation
        for offset in range(n - p_len + 1):
            window = bits[offset : offset + p_len]

            # Normal polarity
            err_norm = int(np.sum(window != pattern))
            if err_norm <= allowed_errors:
                matches.append({
                    "name": name,
                    "hex": info["hex"],
                    "offset": offset,
                    "errors": err_norm,
                    "inverted": False,
                    "length": p_len
                })
                continue

            # Inverted polarity
            err_inv = int(np.sum(window != (1 - pattern)))
            if err_inv <= allowed_errors:
                matches.append({
                    "name": name,
                    "hex": info["hex"],
                    "offset": offset,
                    "errors": err_inv,
                    "inverted": True,
                    "length": p_len
                })

    # Sort matches by earliest bit offset
    matches.sort(key=lambda x: x["offset"])
    return matches
