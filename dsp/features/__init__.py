"""
DSP Feature Extraction Package
==============================
Provides deterministic feature extractors for blind signal analysis:
- Higher-Order Moments & Cumulants (C20, C21, C40, C42, C63)
- Spectral moments (centroid, spread, flatness, rolloff, OBW)
- Envelope and time-domain statistics (PAPR, variance, kurtosis, zero-crossing)
- Instantaneous phase & frequency dynamics
- Cyclostationary Baud clock features
- Concrete 20-Dimensional SignalFeatures Schema
"""

from .cumulants import compute_reference_cumulants, compute_c63_cumulant
from .time_features import compute_envelope_and_time_features
from .spectral_features import compute_spectral_distribution_features
from .phase_features import compute_instantaneous_phase_features
from .cyclostationary import compute_cyclic_clock_features
from .feature_extractor import extract_20d_features

# Alias for compatibility
extract_signal_features = extract_20d_features

__all__ = [
    "compute_reference_cumulants",
    "compute_c63_cumulant",
    "compute_envelope_and_time_features",
    "compute_spectral_distribution_features",
    "compute_instantaneous_phase_features",
    "compute_cyclic_clock_features",
    "extract_20d_features",
    "extract_signal_features"
]

