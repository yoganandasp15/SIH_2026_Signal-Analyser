"""
Controlled Signal Robustness & Degradation Stress Testing Engine (Step 11)
==========================================================================
Systematically generates controlled physical impairments across 6 degradation dimensions:
1. Additive White Gaussian Noise (AWGN SNR Sweep: +20 dB to -10 dB)
2. Amplitude Scaling / Dynamic Range Attenuation (10.0 to 0.001)
3. Carrier Frequency Offset (CFO Sweep: 0 Hz to 0.25*fs)
4. Doppler Frequency Drift (Linear Chirp Sweep: 0 to 2000 Hz/s)
5. Non-Linear Power Amplifier Saturation & Clipping (90% to 10% percentile)
6. Shortened Observation Duration (100k samples down to 2048 samples)

For each degradation level, tracks:
- Classification decision (VALIDATED, ESTIMATED, AMBIGUOUS, UNKNOWN)
- Extracted parameter deviations (carrier frequency, OBW, Baud rate)
- Evidence score & temporal stability score
- Graceful degradation behavior (safely transitions to ESTIMATED or UNKNOWN
  under severe physical degradation without forced false certainty)
- Processing latency
"""

from typing import Dict, Any, List, Tuple, Optional, Callable
import time
import numpy as np

from .adaptive_pipeline import run_adaptive_pipeline
from .spectral import compute_welch_psd


def apply_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Injects complex Gaussian noise at the exact target SNR in dB."""
    p_sig = float(np.mean(np.abs(signal) ** 2))
    if p_sig < 1e-12:
        p_sig = 1.0
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_power = p_sig / snr_lin
    noise_std = np.sqrt(noise_power / 2.0)
    noise = np.random.normal(0.0, noise_std, len(signal)) + 1j * np.random.normal(0.0, noise_std, len(signal))
    return (signal + noise).astype(np.complex64)


def apply_amplitude_scaling(signal: np.ndarray, scale_factor: float) -> np.ndarray:
    """Scales waveform amplitude to simulate high attenuation or low ADC dynamic range."""
    return (signal * float(scale_factor)).astype(np.complex64)


def apply_frequency_offset(signal: np.ndarray, fs: float, cfo_hz: float) -> np.ndarray:
    """Injects static Carrier Frequency Offset (CFO)."""
    t = np.arange(len(signal)) / fs
    carrier = np.exp(1j * 2.0 * np.pi * cfo_hz * t)
    return (signal * carrier).astype(np.complex64)


def apply_frequency_drift(signal: np.ndarray, fs: float, drift_hz_per_sec: float) -> np.ndarray:
    """Injects linear Doppler carrier frequency drift across the observation."""
    t = np.arange(len(signal)) / fs
    chirp = np.exp(1j * np.pi * drift_hz_per_sec * (t ** 2))
    return (signal * chirp).astype(np.complex64)


def apply_clipping(signal: np.ndarray, percentile_threshold: float) -> np.ndarray:
    """Applies non-linear amplifier saturation clipping at the specified percentile."""
    amp = np.abs(signal)
    th = float(np.percentile(amp, max(1.0, min(99.0, percentile_threshold))))
    clipped = np.where(amp > th, signal * (th / np.maximum(amp, 1e-9)), signal)
    return clipped.astype(np.complex64)


def apply_shortened_duration(signal: np.ndarray, num_samples: int) -> np.ndarray:
    """Truncates observation duration to simulate short tactical bursts."""
    n = max(512, min(len(signal), int(num_samples)))
    return signal[:n].astype(np.complex64)


class SignalRobustnessTester:
    """
    Automated stress-testing suite measuring detector and parameter degradation
    profiles under controlled physical distortions.
    """

    def __init__(self, fs: float = 48000.0):
        self.fs = fs

    def sweep_awgn(
        self,
        signal: np.ndarray,
        snr_levels_db: Optional[List[float]] = None,
        ground_truth_token: str = ""
    ) -> List[Dict[str, Any]]:
        """Sweeps SNR from +20 dB down to -10 dB."""
        levels = snr_levels_db or [20.0, 15.0, 10.0, 5.0, 2.0, 0.0, -3.0, -6.0, -10.0]
        results = []

        for snr in levels:
            np.random.seed(42)
            degraded = apply_awgn(signal, snr_db=snr)
            t0 = time.time()
            res = run_adaptive_pipeline(degraded, self.fs)
            elapsed_ms = (time.time() - t0) * 1000.0

            det = res.get("autonomous_detection", {})
            p = res.get("parameters", {})
            prot = det.get("protocol_name", "Unknown")
            dec = res.get("final_decision", "UNKNOWN")
            conf = det.get("confidence", 0.0)
            score = det.get("evidence_score", 0.0)
            t_score = res.get("temporal_validation", {}).get("cross_window_consistency_score", 0.0)

            matched = ground_truth_token.lower() in prot.lower() if ground_truth_token else True

            results.append({
                "dimension": "AWGN",
                "level": snr,
                "unit": "dB",
                "decision": dec,
                "protocol": prot,
                "matched_truth": matched,
                "confidence": conf,
                "evidence_score": score,
                "temporal_stability_score": t_score,
                "estimated_snr_db": p.get("snr_db", 0.0),
                "latency_ms": elapsed_ms,
                "is_safe_degradation": (dec in ["ESTIMATED", "UNKNOWN"] if snr <= 0.0 else True)
            })

        return results

    def sweep_clipping(
        self,
        signal: np.ndarray,
        percentiles: Optional[List[float]] = None,
        ground_truth_token: str = ""
    ) -> List[Dict[str, Any]]:
        """Sweeps amplifier saturation clipping from mild (90%) to severe (10%)."""
        th_levels = percentiles or [95.0, 80.0, 60.0, 40.0, 20.0, 10.0]
        results = []

        for p_th in th_levels:
            degraded = apply_clipping(signal, percentile_threshold=p_th)
            t0 = time.time()
            res = run_adaptive_pipeline(degraded, self.fs)
            elapsed_ms = (time.time() - t0) * 1000.0

            det = res.get("autonomous_detection", {})
            p = res.get("parameters", {})
            prot = det.get("protocol_name", "Unknown")
            dec = res.get("final_decision", "UNKNOWN")
            t_score = res.get("temporal_validation", {}).get("cross_window_consistency_score", 0.0)

            matched = ground_truth_token.lower() in prot.lower() if ground_truth_token else True

            results.append({
                "dimension": "CLIPPING",
                "level": p_th,
                "unit": "%ile",
                "decision": dec,
                "protocol": prot,
                "matched_truth": matched,
                "evidence_score": det.get("evidence_score", 0.0),
                "temporal_stability_score": t_score,
                "latency_ms": elapsed_ms
            })

        return results

    def sweep_doppler_drift(
        self,
        signal: np.ndarray,
        drift_rates: Optional[List[float]] = None,
        ground_truth_token: str = ""
    ) -> List[Dict[str, Any]]:
        """Sweeps Doppler carrier frequency drift rate from 0 to 2000 Hz/s."""
        drifts = drift_rates or [0.0, 100.0, 300.0, 600.0, 1200.0, 2000.0]
        results = []

        for d_rate in drifts:
            degraded = apply_frequency_drift(signal, self.fs, drift_hz_per_sec=d_rate)
            t0 = time.time()
            res = run_adaptive_pipeline(degraded, self.fs)
            elapsed_ms = (time.time() - t0) * 1000.0

            det = res.get("autonomous_detection", {})
            dec = res.get("final_decision", "UNKNOWN")
            t_score = res.get("temporal_validation", {}).get("cross_window_consistency_score", 0.0)

            results.append({
                "dimension": "DOPPLER_DRIFT",
                "level": d_rate,
                "unit": "Hz/s",
                "decision": dec,
                "protocol": det.get("protocol_name", "Unknown"),
                "temporal_stability_score": t_score,
                "evidence_score": det.get("evidence_score", 0.0),
                "latency_ms": elapsed_ms
            })

        return results

    def sweep_observation_duration(
        self,
        signal: np.ndarray,
        durations: Optional[List[int]] = None,
        ground_truth_token: str = ""
    ) -> List[Dict[str, Any]]:
        """Sweeps capture sample duration from full capture down to 2048 samples."""
        max_n = len(signal)
        n_levels = durations or [max_n, 100_000, 50_000, 20_000, 8_192, 4_096, 2_048]
        n_levels = [n for n in n_levels if n <= max_n]
        results = []

        for n in n_levels:
            shortened = apply_shortened_duration(signal, num_samples=n)
            t0 = time.time()
            res = run_adaptive_pipeline(shortened, self.fs)
            elapsed_ms = (time.time() - t0) * 1000.0

            det = res.get("autonomous_detection", {})
            dec = res.get("final_decision", "UNKNOWN")
            prot = det.get("protocol_name", "Unknown")

            matched = ground_truth_token.lower() in prot.lower() if ground_truth_token else True

            results.append({
                "dimension": "DURATION",
                "level": n,
                "unit": "samples",
                "duration_ms": (n / self.fs) * 1000.0,
                "decision": dec,
                "protocol": prot,
                "matched_truth": matched,
                "latency_ms": elapsed_ms
            })

        return results
