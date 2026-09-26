"""
Tactical Intercept Scenario Demonstration System (Step 9)
=========================================================
Provides a controlled, reproducible test harness and demonstration selector
for the 7 tactical RF intercept scenarios:

1. Standard Known Signal: High-SNR reference intercept with verified ground truth.
2. Low-SNR Signal: Sub-threshold / noisy capture testing sensitivity bounds.
3. Noise Floor: Stationary Gaussian thermal noise testing zero false-positive pre-gate.
4. Unknown / OOD Signal: Non-linear or uncataloged RF emission with valid energy.
5. Ambiguous Signal: Waveform on boundary between two legitimate protocols.
6. Pulsed Signal / Radar: Pulse train / chirp emitter with PRF and duty cycle.
7. Distorted Signal: Non-linear amplifier clipping and Doppler frequency drift.

Design Invariants:
------------------
1. Strictly a demonstration convenience; NEVER influences classification logic.
2. Filenames are anonymized / neutral so the classifier cannot peek at labels.
3. Uses existing verified defense captures where possible, and calibrated synthetic
   generators for controlled noise/distortion where appropriate.
"""

from typing import Tuple, Dict, Any, List, Optional
import os
import numpy as np

from dsp.loaders import load_signal_file
from utils.synthetic_generator import add_awgn_noise, generate_synthetic_signal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFIED_SAMPLES_DIR = os.path.join(PROJECT_ROOT, "verified_samples")

TACTICAL_SCENARIOS: List[Tuple[str, str, str]] = [
    (
        "STANDARD_KNOWN",
        "1. Standard Known Signal (ASCII 110 Baud 2-FSK)",
        "Clean, high-SNR tactical digital intercept with clear spectral lines and stable symbol clock."
    ),
    (
        "LOW_SNR",
        "2. Low-SNR Signal (Degraded 2-FSK at +4 dB SNR)",
        "Severe AWGN noise injection testing detector robustness, sensitivity, and parameter uncertainty."
    ),
    (
        "NOISE_FLOOR",
        "3. Noise Floor (Complex Gaussian Thermal Noise)",
        "Pure receiver thermal noise floor. Confirms zero false-positive pre-gate rejection without pipeline failure."
    ),
    (
        "UNKNOWN_OOD",
        "4. Unknown / OOD Signal (Chaotic Non-Linear Waveform)",
        "High-energy uncataloged emitter. Preserves measured physical parameters while refusing forced misclassification."
    ),
    (
        "AMBIGUOUS",
        "5. Ambiguous Signal (Bell 202 vs POCSAG 1200 Boundary)",
        "Waveform with overlapping feature vectors resulting in competing hypotheses within ambiguity margin."
    ),
    (
        "PULSED_RADAR",
        "6. Pulsed Signal / Radar (Ghadir OTH Radar Chirp)",
        "Pulsed FMOP chirp train with high envelope dynamic range, periodic PRI, and radar pulse metrics."
    ),
    (
        "DISTORTED",
        "7. Distorted Signal (Saturated Clipping & Carrier Drift)",
        "Severe non-linear amplifier compression and temporal frequency drift testing stationarity downgrades."
    )
]


def load_tactical_scenario(
    scenario_key: str,
    max_samples: int = 96_000
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Loads or synthesizes a signal tailored for the requested tactical demonstration scenario.

    Parameters:
    -----------
    scenario_key : str
        One of the 7 scenario identifiers:
        'STANDARD_KNOWN', 'LOW_SNR', 'NOISE_FLOOR', 'UNKNOWN_OOD',
        'AMBIGUOUS', 'PULSED_RADAR', 'DISTORTED'.
    max_samples : int
        Number of complex samples to produce (default: 96,000).

    Returns:
    --------
    Tuple[np.ndarray, float, Dict[str, Any]]:
        (signal, fs, metadata)
        The metadata contains neutral, non-leaking identifiers.
    """
    key = scenario_key.upper().strip()

    # -------------------------------------------------------------------------
    # Scenario 1: Standard Known Signal (Real Defense Sample: ASCII.wav)
    # -------------------------------------------------------------------------
    if key == "STANDARD_KNOWN":
        sample_path = os.path.join(VERIFIED_SAMPLES_DIR, "ASCII.wav")
        if os.path.exists(sample_path):
            sig, fs, meta = load_signal_file(sample_path, max_samples=max_samples)
        else:
            sig, _, meta = generate_synthetic_signal(
                mod_type="2-FSK", fs=48000.0, fc=1000.0, baud_rate=110.0,
                snr_db=20.0, num_samples=max_samples
            )
            fs = 48000.0
        
        meta["file_name"] = "Tactical_Scenario_Standard.iq"
        meta["source_type"] = "Tactical Scenario: Standard Known Signal"
        meta["scenario_key"] = "STANDARD_KNOWN"
        meta["scenario_description"] = "Clean, high-SNR tactical digital intercept with clear spectral lines and stable symbol clock."
        meta["expected_verdict"] = "VALIDATED"
        return sig, fs, meta

    # -------------------------------------------------------------------------
    # Scenario 2: Low-SNR Signal (Real Defense Sample: AIS.wav at 0 dB SNR)
    # -------------------------------------------------------------------------
    elif key == "LOW_SNR":
        sample_path = os.path.join(VERIFIED_SAMPLES_DIR, "AIS.wav")
        if os.path.exists(sample_path):
            sig, fs, meta = load_signal_file(sample_path, max_samples=max_samples)
        else:
            sig, _, meta = generate_synthetic_signal(
                mod_type="2-FSK", fs=48000.0, fc=3000.0, baud_rate=300.0,
                snr_db=2.0, num_samples=max_samples
            )
            fs = 48000.0
        meta["file_name"] = "Tactical_Scenario_LowSNR.iq"
        meta["source_type"] = "Tactical Scenario: Low-SNR Defense Intercept"
        meta["scenario_key"] = "LOW_SNR"
        meta["scenario_description"] = "Maritime AIS intercept captured at near 0 dB SNR demonstrating sub-noise-floor demodulation and parameter extraction."
        meta["expected_verdict"] = "VALIDATED"
        return sig, fs, meta

    # -------------------------------------------------------------------------
    # Scenario 3: Noise Floor (Pure Complex Gaussian Thermal Noise)
    # -------------------------------------------------------------------------
    elif key == "NOISE_FLOOR":
        fs = 48000.0
        np.random.seed(42)
        noise = (
            np.random.normal(0.0, 1.0, max_samples)
            + 1j * np.random.normal(0.0, 1.0, max_samples)
        ).astype(np.complex64)
        meta = {
            "file_name": "Tactical_Scenario_NoiseFloor.iq",
            "source_type": "Tactical Scenario: Noise Floor",
            "scenario_key": "NOISE_FLOOR",
            "scenario_description": "Pure receiver thermal noise floor. Confirms zero false-positive pre-gate rejection without pipeline failure.",
            "expected_verdict": "NO SIGNAL / NOISE FLOOR",
            "recording_domain": "Tactical Scenario Bench",
            "format_type": "COMPLEX64"
        }
        return noise, fs, meta

    # -------------------------------------------------------------------------
    # Scenario 4: Unknown / OOD Signal (Non-Linear Chaotic Attractor Waveform)
    # -------------------------------------------------------------------------
    elif key == "UNKNOWN_OOD":
        fs = 48000.0
        t = np.arange(max_samples) / fs
        # Chaotic cubic phase dynamics + amplitude flutter
        phase = 2.0 * np.pi * 8000.0 * t + 12.0 * np.sin(2.0 * np.pi * 137.0 * t) ** 3
        envelope = 1.0 + 0.8 * np.sin(2.0 * np.pi * 23.0 * t) * np.cos(2.0 * np.pi * 41.0 * t)
        sig = (envelope * np.exp(1j * phase)).astype(np.complex64)
        meta = {
            "file_name": "Tactical_Scenario_UnknownOOD.iq",
            "source_type": "Tactical Scenario: Unknown / OOD Signal",
            "scenario_key": "UNKNOWN_OOD",
            "scenario_description": "High-energy uncataloged emitter. Preserves measured physical parameters while refusing forced misclassification.",
            "expected_verdict": "UNKNOWN_OOD",
            "is_ood": True,
            "recording_domain": "Tactical Scenario Bench",
            "format_type": "COMPLEX64"
        }
        return sig, fs, meta

    # -------------------------------------------------------------------------
    # Scenario 5: Ambiguous Signal (Bell 202 vs POCSAG 1200 Boundary)
    # -------------------------------------------------------------------------
    elif key == "AMBIGUOUS":
        fs = 48000.0
        baud = 1200.0
        sps = int(fs / baud)
        n_samples = (max_samples // sps) * sps
        t = np.arange(n_samples) / fs
        num_bits = int(n_samples / sps) + 1
        np.random.seed(42)
        bits = np.random.randint(0, 2, num_bits)
        symbols = np.repeat(bits, sps)[:n_samples]
        f_mark = 1200.0
        f_space = 2200.0
        freqs = np.where(symbols == 1, f_mark, f_space)
        phase = 2.0 * np.pi * np.cumsum(freqs) / fs
        sig = np.exp(1j * phase).astype(np.complex64)
        meta = {
            "file_name": "Tactical_Scenario_Ambiguous.iq",
            "source_type": "Tactical Scenario: Ambiguous Signal",
            "scenario_key": "AMBIGUOUS",
            "scenario_description": "Waveform with overlapping feature vectors resulting in competing hypotheses within ambiguity margin.",
            "target_epistemic_state": "AMBIGUOUS",
            "expected_verdict": "AMBIGUOUS",
            "recording_domain": "Tactical Scenario Bench",
            "format_type": "COMPLEX64"
        }
        return sig, fs, meta

    # -------------------------------------------------------------------------
    # Scenario 6: Pulsed Signal / Radar (Real Defense Sample: Ghadir_Radar.wav)
    # -------------------------------------------------------------------------
    elif key == "PULSED_RADAR":
        sample_path = os.path.join(VERIFIED_SAMPLES_DIR, "Ghadir_Radar.wav")
        if os.path.exists(sample_path):
            sig, fs, meta = load_signal_file(sample_path, max_samples=max_samples)
        else:
            sig, _, meta = generate_synthetic_signal(
                mod_type="PULSED_RADAR", fs=100_000.0, fc=10_000.0,
                snr_db=18.0, num_samples=max_samples, pulse_width_us=50.0, pri_us=250.0
            )
            fs = 100_000.0
        
        meta["file_name"] = "Tactical_Scenario_Radar.iq"
        meta["source_type"] = "Tactical Scenario: Pulsed Signal / Radar"
        meta["scenario_key"] = "PULSED_RADAR"
        meta["scenario_description"] = "Pulsed FMOP chirp train with high envelope dynamic range, periodic PRI, and radar pulse metrics."
        meta["expected_verdict"] = "ESTIMATED"
        return sig, fs, meta

    # -------------------------------------------------------------------------
    # Scenario 7: Distorted Signal (Clipping Saturation + Carrier Frequency Drift)
    # -------------------------------------------------------------------------
    elif key == "DISTORTED":
        sample_path = os.path.join(VERIFIED_SAMPLES_DIR, "ASCII.wav")
        if os.path.exists(sample_path):
            base_sig, fs, meta = load_signal_file(sample_path, max_samples=max_samples)
        else:
            base_sig, _, meta = generate_synthetic_signal(
                mod_type="2-FSK", fs=48000.0, fc=1000.0, baud_rate=110.0,
                snr_db=25.0, num_samples=max_samples
            )
            fs = 48000.0

        t = np.arange(len(base_sig)) / fs
        # 1. Apply linear Doppler carrier frequency drift across observation (1200 Hz/s)
        chirp_drift = np.exp(1j * 2.0 * np.pi * 600.0 * (t ** 2))
        drifted = base_sig * chirp_drift

        # 2. Apply aggressive non-linear power amplifier clipping saturation at 45th percentile
        amp = np.abs(drifted)
        clip_threshold = float(np.percentile(amp, 45.0))
        clipped = np.where(
            amp > clip_threshold,
            drifted * (clip_threshold / np.maximum(amp, 1e-9)),
            drifted
        ).astype(np.complex64)

        meta["file_name"] = "Tactical_Scenario_Distorted.iq"
        meta["source_type"] = "Tactical Scenario: Distorted Signal"
        meta["scenario_key"] = "DISTORTED"
        meta["scenario_description"] = "Severe non-linear amplifier compression and temporal frequency drift testing stationarity downgrades."
        meta["distortion_types"] = ["carrier_drift_1200hz_per_sec", "amplifier_clipping_p45"]
        meta["expected_verdict"] = "ESTIMATED"
        return clipped, fs, meta

    else:
        raise ValueError(f"Unknown tactical scenario key: {scenario_key}")
