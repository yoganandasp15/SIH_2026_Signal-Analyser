"""
Physical Synchronization Framework Package
===========================================
Provides closed-loop digital synchronization:
- Coarse Carrier Frequency Offset (CFO) estimation via M-th power spectral search
- Fine phase and carrier tracking via Digital Costas Loop & Decision-Directed PLL
- Symbol timing recovery via Gardner, Mueller-Muller, and Early-Late detectors
- Polyphase clock interpolation and constellation alignment
"""

from .cfo_estimator import (
    estimate_coarse_cfo_mth_power,
    track_carrier_costas_loop,
    track_carrier_decision_directed_qam,
    resolve_phase_ambiguity
)
from .timing_recovery import (
    gardner_timing_recovery,
    mueller_muller_timing_recovery,
    early_late_timing_recovery
)
from .synchronizer import synchronize_signal

__all__ = [
    "estimate_coarse_cfo_mth_power",
    "track_carrier_costas_loop",
    "track_carrier_decision_directed_qam",
    "resolve_phase_ambiguity",
    "gardner_timing_recovery",
    "mueller_muller_timing_recovery",
    "early_late_timing_recovery",
    "synchronize_signal"
]
