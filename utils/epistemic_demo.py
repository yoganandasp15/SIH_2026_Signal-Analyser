"""
Epistemic State Verification Bench & Demo Signal Generator (Step 7)
===================================================================
Provides synthesized waveforms and presets specifically designed to demonstrate
and rigorously test each of the 6 first-class application states in the NTRO Signal Analyzer:
1. VALIDATED: Closed-loop mathematical/physical proof confirmed with high stationarity.
2. ESTIMATED: Continuous physical parameters estimated; temporal drift or lack of code proof.
3. AMBIGUOUS: Competing candidate hypotheses within ambiguity threshold (|delta| <= 0.05).
4. UNKNOWN: Uncataloged modulation with valid RF energy; physical observations preserved.
5. UNKNOWN_OOD: Out-of-Distribution waveform deviating from catalog feature manifolds.
6. NO SIGNAL / NOISE FLOOR: Input rejected by zero false-positive Gaussian noise pre-gate.
"""

from typing import Tuple, Dict, Any, List
import numpy as np

EPISTEMIC_DEMO_PRESETS: List[Tuple[str, str]] = [
    ("VALIDATED", "1. VALIDATED: Mathematical & Physical Proof Confirmed (Stable CW Carrier)"),
    ("ESTIMATED", "2. ESTIMATED: Continuous Physical Parameters Estimated (Drifting/Hopping Signal)"),
    ("AMBIGUOUS", "3. AMBIGUOUS: Competing Hypotheses Active (Bell 202 vs POCSAG 1200)"),
    ("UNKNOWN", "4. UNKNOWN: Uncataloged Modulation Profile (Non-Standard RF Emission)"),
    ("UNKNOWN_OOD", "5. UNKNOWN_OOD: Out-of-Distribution Waveform (Non-Linear Chaotic Dynamics)"),
    ("NO SIGNAL / NOISE FLOOR", "6. NO SIGNAL / NOISE FLOOR: Stationary Gaussian Noise Floor Rejection")
]


def generate_epistemic_test_signal(
    state: str,
    fs: float = 48000.0,
    duration_s: float = 2.0
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Generates an authentic synthetic test signal tailored to trigger the designated
    epistemic application state in the NTRO Signal Analyzer pipeline.

    Parameters:
    -----------
    state : str
        Target epistemic state: 'VALIDATED', 'ESTIMATED', 'AMBIGUOUS', 'UNKNOWN',
        'UNKNOWN_OOD', or 'NO SIGNAL / NOISE FLOOR'.
    fs : float
        Sampling frequency in Hz (default: 48000.0).
    duration_s : float
        Duration of the test signal in seconds (default: 2.0).

    Returns:
    --------
    Tuple[np.ndarray, float, Dict[str, Any]]:
        (signal, fs, metadata)
    """
    n_samples = int(fs * duration_s)
    t = np.arange(n_samples) / fs

    s_upper = (state or "VALIDATED").upper()

    if "VALIDATED" in s_upper and "UNVALIDATED" not in s_upper:
        # State 1: Highly stable pure CW tone with SNR > 30 dB
        # Perfect stationarity across all 4 windows, 0 contradictions -> VALIDATED
        carrier_fc = 10000.0
        sig = (np.exp(1j * 2.0 * np.pi * carrier_fc * t)).astype(np.complex64)
        meta = {
            "file_name": "Epistemic_Demo_VALIDATED.iq",
            "source_type": "Epistemic State Bench (VALIDATED)",
            "preset_label": "VALIDATED: Pure CW Carrier Tone (10.0 kHz)",
            "target_epistemic_state": "VALIDATED",
            "recording_domain": "Epistemic Verification Bench",
            "format_type": "COMPLEX64"
        }

    elif "ESTIMATED" in s_upper:
        # State 2: Stepped carrier frequency and amplitude across the 4 temporal windows
        # Cross-window parameter consistency drops below the 0.60 stability threshold
        win_len = n_samples // 4
        t_w = np.arange(win_len) / fs
        w1 = 1.0 * np.exp(1j * 2.0 * np.pi * 2000.0 * t_w)
        w2 = 3.0 * np.exp(1j * 2.0 * np.pi * 18000.0 * t_w)
        w3 = 0.5 * np.exp(1j * 2.0 * np.pi * 5000.0 * t_w)
        w4 = 2.0 * np.exp(1j * 2.0 * np.pi * 12000.0 * t_w)
        sig = np.concatenate([w1, w2, w3, w4])
        if len(sig) < n_samples:
            pad = np.zeros(n_samples - len(sig), dtype=np.complex64)
            sig = np.concatenate([sig, pad])
        sig = sig[:n_samples].astype(np.complex64)
        meta = {
            "file_name": "Epistemic_Demo_ESTIMATED.iq",
            "source_type": "Epistemic State Bench (ESTIMATED)",
            "preset_label": "ESTIMATED: Multi-Window Temporal Parameter Drift",
            "target_epistemic_state": "ESTIMATED",
            "recording_domain": "Epistemic Verification Bench",
            "format_type": "COMPLEX64"
        }

    elif "AMBIGUOUS" in s_upper:
        # State 3: Dual-tone FSK waveform (1200 Hz / 2200 Hz) with 1000 Hz tone shift
        # Competing modulation candidates (Bell 202 vs POCSAG 1200) within ambiguity margin
        baud = 1200.0
        sps = int(fs / baud)
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
            "file_name": "Epistemic_Demo_AMBIGUOUS.iq",
            "source_type": "Epistemic State Bench (AMBIGUOUS)",
            "preset_label": "AMBIGUOUS: Competing FSK Hypotheses (Bell 202 vs POCSAG 1200)",
            "target_epistemic_state": "AMBIGUOUS",
            "recording_domain": "Epistemic Verification Bench",
            "format_type": "COMPLEX64"
        }

    elif "UNKNOWN_OOD" in s_upper or "OUT_OF_DISTRIBUTION" in s_upper:
        # State 5: Highly non-linear / chaotic waveform with extreme polynomial phase & kurtosis
        # Out-of-Distribution Mahalanobis distance manifold violation
        phase = 2.0 * np.pi * 8000.0 * t + 12.0 * np.sin(2.0 * np.pi * 137.0 * t) ** 3
        envelope = 1.0 + 0.8 * np.sin(2.0 * np.pi * 23.0 * t) * np.cos(2.0 * np.pi * 41.0 * t)
        sig = (envelope * np.exp(1j * phase)).astype(np.complex64)
        meta = {
            "file_name": "Epistemic_Demo_UNKNOWN_OOD.iq",
            "source_type": "Epistemic State Bench (UNKNOWN_OOD)",
            "preset_label": "UNKNOWN_OOD: Out-of-Distribution Non-Linear Waveform",
            "target_epistemic_state": "UNKNOWN_OOD",
            "is_ood": True,
            "recording_domain": "Epistemic Verification Bench",
            "format_type": "COMPLEX64"
        }

    elif "NO SIGNAL" in s_upper or "NOISE" in s_upper:
        # State 6: Pure complex Gaussian white noise floor rejected by zero false-positive pre-gate
        np.random.seed(42)
        noise = (np.random.normal(0, 1, n_samples) + 1j * np.random.normal(0, 1, n_samples)).astype(np.complex64)
        sig = noise
        meta = {
            "file_name": "Epistemic_Demo_NOISE_FLOOR.iq",
            "source_type": "Epistemic State Bench (NO SIGNAL / NOISE FLOOR)",
            "preset_label": "NO SIGNAL / NOISE FLOOR: Gaussian Thermal Noise",
            "target_epistemic_state": "NO SIGNAL / NOISE FLOOR",
            "recording_domain": "Epistemic Verification Bench",
            "format_type": "COMPLEX64"
        }

    else:
        # State 4: UNKNOWN (Uncataloged Modulation Profile)
        # RF energy is clearly present (high SNR, discrete spectral peaks), but violates all cataloged protocols
        tones = (
            np.exp(1j * 2.0 * np.pi * 3750.0 * t) +
            0.7 * np.exp(1j * 2.0 * np.pi * 8210.0 * t) +
            0.5 * np.exp(1j * 2.0 * np.pi * 14330.0 * t)
        )
        sig = tones.astype(np.complex64)
        meta = {
            "file_name": "Epistemic_Demo_UNKNOWN.iq",
            "source_type": "Epistemic State Bench (UNKNOWN)",
            "preset_label": "UNKNOWN: Uncataloged Modulation Profile",
            "target_epistemic_state": "UNKNOWN",
            "recording_domain": "Epistemic Verification Bench",
            "format_type": "COMPLEX64"
        }

    return sig, fs, meta
