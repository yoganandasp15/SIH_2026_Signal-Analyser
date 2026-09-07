"""
Open-Set Hybrid Modulation Classifier Module
============================================
Implements a dual-branch open-set AMC:
- Branch A: Deterministic physical invariants (Cumulants, Envelope Variance, Frequency Deviations, SFM)
- Branch B: Statistical Mahalanobis distance metric with calibrated OOD threshold
- Explicit "UNKNOWN_OOD" outcome when signals do not match modeled characteristics
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from dsp.contracts import (
    ModulationFamily,
    ConfidenceLevel,
    SignalHypothesis,
    SignalFeatures
)
from dsp.features.feature_extractor import extract_20d_features


# Reference feature centroids for modeled modulation families (20 dimensions aligned with SignalFeatures.to_vector())
# Dimensions:
# 0: c20_mag, 1: c20_angle, 2: c40_mag, 3: c40_angle, 4: c42_real, 5: c63_norm,
# 6: spectral_centroid, 7: spectral_spread, 8: spectral_flatness, 9: spectral_rolloff,
# 10: obw_99, 11: papr_db, 12: envelope_variance, 13: amplitude_skewness, 14: amplitude_kurtosis,
# 15: instantaneous_freq_variance, 16: instantaneous_phase_variance, 17: cyclic_peak_prominence,
# 18: zero_crossing_rate, 19: snr_m2m4_db
_MOD_CENTROIDS: Dict[ModulationFamily, np.ndarray] = {
    ModulationFamily.BPSK: np.array([
        1.0, 0.0, 2.0, np.pi, -2.0, 16.0, 0.0, 0.14, 0.21, 0.20, 0.87,
        2.8, 0.12, 0.0, 0.0, 0.03, 60.0, 3.1, 0.12, 15.0
    ], dtype=np.float64),
    ModulationFamily.QPSK: np.array([
        0.05, 0.0, 1.0, 0.0, -1.0, 12.0, 0.0, 0.14, 0.21, 0.21, 0.87,
        2.8, 0.12, 0.0, 0.0, 0.02, 50.0, 3.1, 0.12, 15.0
    ], dtype=np.float64),
    ModulationFamily.PSK_8: np.array([
        0.05, 0.0, 0.05, 0.0, -1.0, 12.0, 0.0, 0.14, 0.21, 0.21, 0.87,
        2.8, 0.12, 0.0, 0.0, 0.02, 50.0, 3.1, 0.12, 15.0
    ], dtype=np.float64),
    ModulationFamily.QAM_16: np.array([
        0.05, 0.0, 0.68, 0.0, -0.68, 8.0, 0.0, 0.14, 0.21, 0.21, 0.87,
        3.5, 0.28, 0.1, 0.0, 0.02, 40.0, 3.1, 0.12, 15.0
    ], dtype=np.float64),
    ModulationFamily.QAM_64: np.array([
        0.05, 0.0, 0.62, 0.0, -0.62, 6.0, 0.0, 0.14, 0.21, 0.21, 0.87,
        4.5, 0.35, 0.1, 0.0, 0.02, 40.0, 3.1, 0.12, 15.0
    ], dtype=np.float64),
    ModulationFamily.FSK_2: np.array([
        0.05, 0.0, 0.05, 0.0, -0.9, 12.0, 0.0, 0.06, 0.05, 0.05, 0.70,
        2.8, 0.12, 0.0, 0.0, 0.002, 10.0, 3.2, 0.07, 15.0
    ], dtype=np.float64),
    ModulationFamily.FSK_4: np.array([
        0.05, 0.0, 0.05, 0.0, -0.9, 12.0, 0.0, 0.08, 0.05, 0.08, 0.80,
        2.8, 0.12, 0.0, 0.0, 0.005, 10.0, 3.2, 0.07, 15.0
    ], dtype=np.float64),
    ModulationFamily.ANALOG_AM: np.array([
        0.1, 0.0, 0.2, 0.0, 0.1, 0.5, 0.0, 0.15, 0.08, 0.3, 0.2,
        4.5, 0.55, 0.2, 0.2, 0.05, 2.0, 0.3, 0.5, 15.0
    ], dtype=np.float64),
    ModulationFamily.ANALOG_FM: np.array([
        0.05, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.25, 0.08, 0.45, 0.5,
        0.2, 0.03, 0.0, 0.0, 0.1, 2.0, 0.5, 0.5, 15.0
    ], dtype=np.float64)
}

# Empirical variance scaling weights across the 20 dimensions for Mahalanobis-like distance
_FEATURE_VARIANCE_WEIGHTS = np.array([
    0.3,   # c20_mag
    0.5,   # c20_angle
    0.4,   # c40_mag
    0.8,   # c40_angle
    0.4,   # c42_real
    3.0,   # c63_norm
    0.1,   # spectral_centroid
    0.1,   # spectral_spread
    0.1,   # spectral_flatness
    0.1,   # spectral_rolloff
    0.2,   # obw_99
    0.8,   # papr_db
    0.15,  # envelope_variance
    0.8,   # amplitude_skewness
    0.8,   # amplitude_kurtosis
    0.5,   # instantaneous_freq_variance
    25.0,  # instantaneous_phase_variance
    0.5,   # cyclic_peak_prominence
    0.8,   # zero_crossing_rate
    5.0    # snr_m2m4_db
], dtype=np.float64)

# Calibrated OOD rejection threshold
_OOD_DISTANCE_THRESHOLD = 15.0


def classify_modulation_open_set(
    signal: np.ndarray,
    fs: Optional[float] = None,
    features: Optional[SignalFeatures] = None
) -> SignalHypothesis:
    """
    Classifies the signal into a modulation hypothesis using DSP rules and statistical distance,
    with an explicit UNKNOWN_OOD safety boundary.

    Parameters:
    -----------
    signal : np.ndarray
        Complex baseband signal.
    fs : Optional[float]
        Sampling rate in Hz (or None if normalized).
    features : Optional[SignalFeatures]
        Pre-extracted 20-D features. If None, extracted automatically.

    Returns:
    --------
    SignalHypothesis
    """
    if len(signal) < 32:
        return SignalHypothesis(
            modulation=ModulationFamily.UNKNOWN_OOD,
            symbol_rate=None,
            carrier_offset=0.0,
            samples_per_symbol=2.0,
            pulse_model=None,
            confidence=0.0,
            confidence_level=ConfidenceLevel.UNKNOWN,
            evidence=["Signal length too short for reliable classification (< 32 samples)"],
            rejected_hypotheses=[]
        )

    if features is None:
        features = extract_20d_features(signal, fs=fs)

    use_fs = float(fs) if (fs is not None and fs > 0) else 1.0
    feat_vec = features.to_vector()
    norm_feat_vec = feat_vec.copy()
    if use_fs > 1.0:
        norm_feat_vec[6] /= use_fs   # spectral_centroid
        norm_feat_vec[7] /= use_fs   # spectral_spread
        norm_feat_vec[9] /= use_fs   # spectral_rolloff
        norm_feat_vec[10] /= use_fs  # obw_99
        norm_feat_vec[15] /= (use_fs ** 2)  # instantaneous_freq_variance

    evidence: List[str] = []
    rejected: List[str] = []

    # Branch A: Physics-Based Deterministic Invariant Gating
    env_var = features.envelope_variance
    c20_mag = features.c20_mag
    c40_mag = features.c40_mag
    c42_real = features.c42_real
    c63_norm = features.c63_norm
    pvar = features.instantaneous_phase_variance
    sfm = features.spectral_flatness
    norm_freq_var = features.instantaneous_freq_variance / (use_fs ** 2) if use_fs > 1.0 else features.instantaneous_freq_variance
    norm_spec_spread = features.spectral_spread / use_fs if use_fs > 1.0 else features.spectral_spread

    rule_candidate: Optional[ModulationFamily] = None
    rule_conf = 0.0

    # 1. Constant envelope test & chirp/sweep anomaly check
    is_constant_envelope = env_var < 0.18
    is_chirp_or_sweep = (
        is_constant_envelope and
        c20_mag < 0.35 and c40_mag < 0.35 and
        (pvar > 100.0 or sfm > 0.6 or norm_freq_var > 0.05)
    )

    if is_constant_envelope and not is_chirp_or_sweep:
        if c20_mag > 0.65 or c40_mag > 1.4:
            rule_candidate = ModulationFamily.BPSK
            rule_conf = 0.95
            evidence.append(f"Strong 2-fold symmetry: |C20|={c20_mag:.2f} > 0.65, C63={c63_norm:.1f} -> BPSK")
        elif c40_mag > 0.6 and c42_real < -0.6:
            rule_candidate = ModulationFamily.QPSK
            rule_conf = 0.93
            evidence.append(f"4-fold symmetry: |C40|={c40_mag:.2f} > 0.6, C42={c42_real:.2f} < -0.6, C63={c63_norm:.1f} -> QPSK")
        elif norm_spec_spread < 0.12 and sfm < 0.15 and c20_mag < 0.35 and c40_mag < 0.35:
            if norm_spec_spread > 0.068 or norm_freq_var > 0.0022:
                rule_candidate = ModulationFamily.FSK_4
                rule_conf = 0.90
                evidence.append(f"Constant envelope (var={env_var:.3f}) with multi-tone spread ({norm_spec_spread:.3f}) -> 4-FSK")
            else:
                rule_candidate = ModulationFamily.FSK_2
                rule_conf = 0.92
                evidence.append(f"Constant envelope (var={env_var:.3f}) with 2-tone frequency variance -> 2-FSK")
        elif c42_real < -0.7 and norm_spec_spread >= 0.10 and sfm >= 0.15 and pvar <= 100.0:
            rule_candidate = ModulationFamily.PSK_8
            rule_conf = 0.85
            evidence.append(f"Constant envelope with circular constellation (C42={c42_real:.2f}) -> 8-PSK")
    elif not is_constant_envelope:
        # Non-constant envelope: QAM, AM, or multicarrier
        if c42_real < -0.4 and 0.30 < c40_mag < 0.85:
            if features.papr_db > 3.2:
                rule_candidate = ModulationFamily.QAM_64
                rule_conf = 0.86
                evidence.append(f"Multi-ring constellation: PAPR={features.papr_db:.1f} dB, C42={c42_real:.2f} -> 64-QAM")
            else:
                rule_candidate = ModulationFamily.QAM_16
                rule_conf = 0.91
                evidence.append(f"2-ring constellation: PAPR={features.papr_db:.1f} dB, C42={c42_real:.2f} -> 16-QAM")
        elif sfm < 0.12 and features.papr_db > 3.0 and norm_freq_var < 0.001 and pvar < 15.0:
            rule_candidate = ModulationFamily.ANALOG_AM
            rule_conf = 0.88
            evidence.append(f"Low spectral flatness (SFM={sfm:.3f}) and envelope modulation -> AM")

    # Branch B: Statistical Weighted Distance against Class Centroids
    distances: Dict[ModulationFamily, float] = {}
    for mod_fam, centroid in _MOD_CENTROIDS.items():
        diff = (norm_feat_vec - centroid) / (_FEATURE_VARIANCE_WEIGHTS + 1e-6)
        diff[1] = np.angle(np.exp(1j * (norm_feat_vec[1] - centroid[1]))) / (_FEATURE_VARIANCE_WEIGHTS[1] + 1e-6)
        diff[3] = np.angle(np.exp(1j * (norm_feat_vec[3] - centroid[3]))) / (_FEATURE_VARIANCE_WEIGHTS[3] + 1e-6)
        dist = float(np.sqrt(np.sum(diff ** 2)))
        distances[mod_fam] = dist

    best_dist_fam = min(distances, key=distances.get)
    min_dist = distances[best_dist_fam]

    # OOD Rejection Check
    is_ood = (min_dist > _OOD_DISTANCE_THRESHOLD and (rule_candidate is None or rule_conf < 0.80)) or is_chirp_or_sweep
    if is_ood:
        if is_chirp_or_sweep:
            evidence.append("Chirp / frequency sweep dynamics detected -> anomalous/radar waveform")
        else:
            evidence.append(f"OOD Distance ({min_dist:.2f}) exceeds rejection threshold ({_OOD_DISTANCE_THRESHOLD})")
        return SignalHypothesis(
            modulation=ModulationFamily.UNKNOWN_OOD,
            symbol_rate=None,
            carrier_offset=0.0,
            samples_per_symbol=2.0,
            pulse_model=None,
            confidence=0.30,
            confidence_level=ConfidenceLevel.LOW,
            is_ood=True,
            mahalanobis_distance=min_dist,
            evidence=evidence,
            rejected_hypotheses=[f"{m.value} (d={d:.2f})" for m, d in sorted(distances.items(), key=lambda x: x[1])[:3]]
        )

    # Fusion of Rule Candidate and Distance Candidate
    if rule_candidate is not None:
        final_mod = rule_candidate
        final_conf = rule_conf
    else:
        final_mod = best_dist_fam
        # Calibrate distance to probability
        final_conf = float(np.clip(1.0 - (min_dist / _OOD_DISTANCE_THRESHOLD) * 0.5, 0.4, 0.92))
        evidence.append(f"Centroid distance minimum: {final_mod.value} (d={min_dist:.2f})")

    # Estimate symbol rate hint from cyclic clock or spectral bandwidth
    symbol_rate_est: Optional[float] = None
    if fs is not None and fs > 0:
        if final_mod in [ModulationFamily.BPSK, ModulationFamily.QPSK, ModulationFamily.PSK_8, ModulationFamily.QAM_16, ModulationFamily.QAM_64]:
            symbol_rate_est = max(100.0, float(min(features.obw_99 * 0.8, max(features.spectral_spread * 1.5, 100.0))))
        elif final_mod in [ModulationFamily.FSK_2, ModulationFamily.FSK_4]:
            symbol_rate_est = max(50.0, float(min(features.obw_99 * 0.5, max(features.spectral_spread * 1.2, 50.0))))

    sps_est = float(fs / symbol_rate_est) if (fs and symbol_rate_est and symbol_rate_est > 0) else 2.0

    conf_level = ConfidenceLevel.HIGH if final_conf >= 0.85 else (ConfidenceLevel.MEDIUM if final_conf >= 0.65 else ConfidenceLevel.LOW)

    for m, d in sorted(distances.items(), key=lambda x: x[1]):
        if m != final_mod and len(rejected) < 4:
            rejected.append(f"{m.value} (d={d:.2f})")

    return SignalHypothesis(
        modulation=final_mod,
        symbol_rate=symbol_rate_est,
        carrier_offset=features.spectral_centroid if (fs is not None) else 0.0,
        samples_per_symbol=sps_est,
        pulse_model="Square Root Raised Cosine" if "QAM" in final_mod.value or "PSK" in final_mod.value else "Gaussian / Nyquist",
        confidence=final_conf,
        confidence_level=conf_level,
        is_ood=False,
        mahalanobis_distance=min_dist,
        evidence=evidence,
        rejected_hypotheses=rejected
    )
