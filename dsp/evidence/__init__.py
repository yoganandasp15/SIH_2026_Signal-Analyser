"""
Multi-Stage Gated Evidence Fusion Package
=========================================
Implements gated evidence fusion across the 5 validation stages:
Modulation Plausibility -> Synchronization Convergence -> Demodulation Stability -> FEC Validity -> Framing/CRC.
Partitions all intelligence into the strict epistemic hierarchy:
OBSERVED, ESTIMATED, HYPOTHESIZED, VALIDATED, and UNKNOWN.
"""

from .fusion import fuse_evidence

__all__ = ["fuse_evidence"]
