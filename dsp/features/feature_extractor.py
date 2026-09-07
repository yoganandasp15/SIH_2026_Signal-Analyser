"""
Unified Feature Extraction Engine
=================================
Aggregates statistical moments, higher-order cumulants, spectral distributions,
and phase dynamics into the canonical 20-Dimensional SignalFeatures data contract.
"""

from typing import Optional
import numpy as np

from dsp.contracts import SignalFeatures
from .cumulants import compute_reference_cumulants
from .time_features import compute_envelope_and_time_features
from .spectral_features import compute_spectral_distribution_features
from .phase_features import compute_instantaneous_phase_features
from .cyclostationary import compute_cyclic_clock_features


def extract_20d_features(
    signal: np.ndarray,
    fs: Optional[float] = None
) -> SignalFeatures:
    """
    Computes all 20 standardized physical features and returns a typed SignalFeatures instance.

    Features:
    ---------
    1. c20_mag
    2. c20_angle
    3. c21
    4. c40_mag
    5. c40_angle
    6. c42_real
    7. c63_norm
    8. spectral_centroid
    9. spectral_spread
    10. spectral_flatness
    11. spectral_rolloff
    12. obw_99
    13. envelope_variance
    14. amplitude_skewness
    15. amplitude_kurtosis
    16. instantaneous_freq_variance
    17. instantaneous_phase_variance
    18. cyclic_peak_prominence
    19. zero_crossing_rate
    20. snr_m2m4_db
    """
    cumulants = compute_reference_cumulants(signal)
    time_feats = compute_envelope_and_time_features(signal)
    spec_feats = compute_spectral_distribution_features(signal, fs=fs)
    phase_feats = compute_instantaneous_phase_features(signal, fs=fs)
    cyclo_feats = compute_cyclic_clock_features(signal, fs=fs)

    return SignalFeatures(
        c20_mag=float(cumulants["c20_mag"]),
        c20_angle=float(cumulants["c20_angle"]),
        c40_mag=float(cumulants["c40_mag"]),
        c40_angle=float(cumulants["c40_angle"]),
        c42_real=float(cumulants["c42_real"]),
        c63_norm=float(cumulants["c63_norm"]),
        spectral_centroid=float(spec_feats["spectral_centroid"]),
        spectral_spread=float(spec_feats["spectral_spread"]),
        spectral_flatness=float(spec_feats["spectral_flatness"]),
        spectral_rolloff=float(spec_feats["spectral_rolloff"]),
        obw_99=float(spec_feats["obw_99"]),
        papr_db=float(time_feats["papr_db"]),
        envelope_variance=float(time_feats["envelope_variance"]),
        amplitude_skewness=float(time_feats["amplitude_skewness"]),
        amplitude_kurtosis=float(time_feats["amplitude_kurtosis"]),
        instantaneous_freq_variance=float(phase_feats["instantaneous_freq_variance"]),
        instantaneous_phase_variance=float(phase_feats["instantaneous_phase_variance"]),
        cyclic_peak_prominence=float(cyclo_feats["cyclic_peak_prominence"]),
        zero_crossing_rate=float(time_feats["zero_crossing_rate"]),
        snr_m2m4_db=float(cyclo_feats["snr_m2m4_db"])
    )
