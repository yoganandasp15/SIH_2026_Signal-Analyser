"""
Automated Modulation Classification (AMC) Engine
=================================================
Implements the Physics-Based Multi-Stage Triage Architecture:
1. Gate 1: Pulsed Radar / OTH Intercept Gating (Linear FMOP R^2 trajectory, matched radar frames)
2. Gate 2: TDMA Cellular / Digital Burst Gating (GSM GMSK / DMR 4-FSK / AIS / APRS frame matching)
3. Gate 3: Analog Audio / Voice & NFM Pre-Classifier (Speech formants, telephony BW >= 1200 Hz, SFM < 0.15)
4. Gate 3.5: Frequency Shift Keying (FSK) Engine (Baseband bipolar eye, NAVTEX 170 Hz shift, APRS AFSK, FT8 M-FSK)
5. Gate 4: Continuous Digital Communication AMC (Fine Frequency Lock & Higher-Order Cumulants: BPSK, QPSK, 8-PSK, 16-QAM, AM, FM, CW)
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from scipy.signal import find_peaks, medfilt, spectrogram
from .spectral import compute_welch_psd


def estimate_fine_carrier_offset(
    signal: np.ndarray,
    fs: float,
    max_offset_hz: float = 100_000.0
) -> float:
    """
    Estimates residual carrier frequency offset using 2nd and 4th power spectral line recovery.
    """
    if len(signal) < 512 or fs <= 0:
        return 0.0

    n_pts = min(len(signal), 32768)
    sig_slice = signal[:n_pts]

    # 1. 2nd power (BPSK / 2-phase)
    y2 = sig_slice ** 2
    y2 = y2 - np.mean(y2)
    n_fft = 2 ** int(np.ceil(np.log2(n_pts)))
    spec2 = np.abs(np.fft.fft(y2, n=n_fft))
    freqs2 = np.fft.fftfreq(n_fft, d=1.0 / fs)

    spec2_s = np.fft.fftshift(spec2)
    freqs2_s = np.fft.fftshift(freqs2)

    mask2 = np.abs(freqs2_s) <= 2.0 * max_offset_hz
    if np.any(mask2):
        peak2_idx = np.argmax(spec2_s[mask2])
        peak2_freq = freqs2_s[mask2][peak2_idx]
        delta_f2 = peak2_freq / 2.0
        peak2_prom = spec2_s[mask2][peak2_idx] / (np.median(spec2_s[mask2]) + 1e-12)
    else:
        delta_f2 = 0.0
        peak2_prom = 0.0

    # 2. 4th power (QPSK / 4-phase)
    y4 = sig_slice ** 4
    y4 = y4 - np.mean(y4)
    spec4 = np.abs(np.fft.fft(y4, n=n_fft))
    spec4_s = np.fft.fftshift(spec4)

    mask4 = np.abs(freqs2_s) <= 4.0 * max_offset_hz
    if np.any(mask4):
        peak4_idx = np.argmax(spec4_s[mask4])
        peak4_freq = freqs2_s[mask4][peak4_idx]
        delta_f4 = peak4_freq / 4.0
        peak4_prom = spec4_s[mask4][peak4_idx] / (np.median(spec4_s[mask4]) + 1e-12)
    else:
        delta_f4 = 0.0
        peak4_prom = 0.0

    if peak2_prom > 8.0 and peak2_prom >= peak4_prom:
        return float(delta_f2)
    elif peak4_prom > 8.0:
        return float(delta_f4)
    return 0.0


def compute_spectral_flatness(signal: np.ndarray, fs: float) -> Tuple[float, int]:
    """
    Computes Spectral Flatness Measure (SFM) and number of formant / harmonic peaks.
    SFM = Geometric Mean(PSD) / Arithmetic Mean(PSD).
    Audio speech / tone signals have low SFM (< 0.10) with harmonic formant spikes.
    Digital communication signals (QPSK/QAM) have high SFM (> 0.25).
    """
    f_s, psd_db, psd_lin = compute_welch_psd(signal, fs, nperseg=1024)
    pos_mask = (f_s >= 50.0) & (f_s <= min(8000.0, 0.48 * fs))
    if not np.any(pos_mask):
        return 0.5, 0

    active_psd = psd_lin[pos_mask] + 1e-18
    geom_mean = float(np.exp(np.mean(np.log(active_psd))))
    arith_mean = float(np.mean(active_psd))
    sfm = float(geom_mean / (arith_mean + 1e-18))

    peaks, _ = find_peaks(active_psd, height=2.0 * np.median(active_psd))
    num_peaks = len(peaks)

    return sfm, num_peaks


def compute_higher_order_cumulants(
    signal: np.ndarray,
    eps: float = 1e-12
) -> Dict[str, float]:
    """
    Computes 2nd and 4th order statistical moments and cumulants for complex baseband signal.
    """
    if len(signal) == 0:
        return {
            "c20_mag": 0.0,
            "c21": 0.0,
            "c40_mag": 0.0,
            "c42_real": 0.0,
            "c40_real": 0.0,
            "c40_imag": 0.0,
            "c42_mag": 0.0
        }

    y = signal - np.mean(signal)
    avg_pwr = np.mean(np.abs(y) ** 2)
    if avg_pwr > eps:
        y = y / np.sqrt(avg_pwr)

    mu_20 = np.mean(y ** 2)
    mu_21 = np.mean(np.abs(y) ** 2)
    mu_40 = np.mean(y ** 4)
    mu_42 = np.mean(np.abs(y) ** 4)

    c_20 = mu_20
    c_21 = float(np.real(mu_21))
    c_40 = mu_40 - 3.0 * (mu_20 ** 2)
    c_42 = mu_42 - (np.abs(mu_20) ** 2) - 2.0 * (mu_21 ** 2)

    denom = (c_21 ** 2) + eps
    norm_c40 = c_40 / denom
    norm_c42 = c_42 / denom

    return {
        "c20_mag": float(np.abs(c_20)),
        "c21": float(c_21),
        "c40_mag": float(np.abs(norm_c40)),
        "c40_real": float(np.real(norm_c40)),
        "c40_imag": float(np.imag(norm_c40)),
        "c42_real": float(np.real(norm_c42)),
        "c42_mag": float(np.abs(norm_c42))
    }


def detect_8mfsk_ale(signal: np.ndarray, fs: float) -> Optional[Dict[str, Any]]:
    """
    Detects 8-Tone MFSK Automatic Link Establishment (2G ALE / MIL-STD-188-141).
    Verifies 8 discrete tones spaced by ~250 Hz in the voice band [600, 2700] Hz.
    """
    if len(signal) < 2048 or fs <= 0:
        return None
    from scipy.signal import welch
    n_pts = min(len(signal), 131072)
    nperseg = 4096 if fs <= 48000 else 8192
    f_w, psd_w = welch(signal.real[:n_pts], fs=fs, nperseg=nperseg)
    voice_mask = (f_w >= 600.0) & (f_w <= 2700.0)
    if not np.any(voice_mask):
        return None
    f_v = f_w[voice_mask]
    psd_v = psd_w[voice_mask]
    pk_thresh = 0.08 * np.max(psd_v)
    bin_dist = max(3, int(180.0 / (f_w[1] - f_w[0])))
    pks_v, _ = find_peaks(psd_v, height=pk_thresh, distance=bin_dist)
    pk_freqs = f_v[pks_v]
    if len(pk_freqs) in [7, 8, 9]:
        base_f = pk_freqs[0]
        harm_errs = [abs((f - base_f) - round((f - base_f) / 250.0) * 250.0) for f in pk_freqs]
        avg_err = float(np.mean(harm_errs))
        max_err = float(np.max(harm_errs))
        if avg_err < 18.0 and max_err < 35.0:
            return {
                "modulation_type": "8-Tone MFSK (Automatic Link Establishment / 2G ALE / MIL-STD-188-141)",
                "confidence": 0.98,
                "decision_path": f"8-Tone MFSK Detected ({len(pk_freqs)} Tones Spaced ~250 Hz, Avg Harmonic Err = {avg_err:.2f} Hz) -> 2G ALE (MIL-STD-188-141)",
                "fsk_shift_hz": 250.0,
                "baud_rate_hint": 125.0,
                "is_ale": True
            }
    return None


def detect_frequency_shift_keying(
    signal: np.ndarray,
    fs: float,
    bw_hz: float = 0.0,
    fc_est: float = 0.0
) -> Optional[Dict[str, Any]]:
    """
    Discriminates Frequency Shift Keying (FSK) modulations:
    1. 8-Tone MFSK (2G ALE / MIL-STD-188-141)
    2. Baseband bipolar FSK (POCSAG Paging)
    3. Passband 2-FSK (NAVTEX 170 Hz shift, APRS AFSK 1000 Hz shift)
    4. M-FSK / 8-FSK (FT8 tone step plateaus in narrowband passband)
    """
    if len(signal) < 1024 or fs <= 0:
        return None

    # Sub-case 0: 8-Tone MFSK (2G ALE)
    ale_res = detect_8mfsk_ale(signal, fs)
    if ale_res is not None:
        return ale_res

    # Sub-case A: Bipolar Baseband Eye Pattern (Demodulated FSK / POCSAG)
    c_hist, _ = np.histogram(signal.real[:65536], bins=30)
    pk_bipolar, _ = find_peaks(c_hist, height=0.15 * np.max(c_hist), distance=6)
    if len(pk_bipolar) == 2 and c_hist[15] < 0.40 * np.max(c_hist):
        return {
            "modulation_type": "2-FSK (Baseband Paging / POCSAG)",
            "confidence": 0.95,
            "decision_path": "Gate 3.5: Bipolar Baseband Eye Pattern -> 2-FSK (POCSAG / Paging)",
            "fsk_shift_hz": 4500.0
        }

    # Sub-case B: Narrowband Carrier Discriminator (PSK31 vs FT8)
    if bw_hz > 0.0 and bw_hz < 300.0 and len(signal) > 4000:
        sig_sub = signal[:min(len(signal), 65536)]
        y2 = sig_sub ** 2
        n_fft = 131072
        spec2 = np.abs(np.fft.fft(y2, n=n_fft))
        freqs2 = np.fft.fftfreq(n_fft, d=1.0 / fs)
        pos_mask = (freqs2 >= 200.0) & (freqs2 <= 0.48 * fs)
        if np.any(pos_mask):
            pk_idx = np.argmax(spec2[pos_mask])
            best_f2 = freqs2[pos_mask][pk_idx]
            pk_val = spec2[pos_mask][pk_idx]

            local_mask = (freqs2 >= best_f2 - 100.0) & (freqs2 <= best_f2 + 100.0)
            above_half = freqs2[local_mask][spec2[local_mask] >= 0.5 * pk_val]
            bw_half = np.max(above_half) - np.min(above_half)
            if bw_half >= 15.0:
                return {
                    "modulation_type": "8-FSK / M-FSK (Amateur Weak-Signal Mode / FT8)",
                    "confidence": 0.96,
                    "decision_path": f"Gate 3.5: Multi-Tone Frequency Steps (BW={bw_hz:.1f} Hz, Squared BW={bw_half:.1f} Hz) -> FT8 / M-FSK",
                    "fsk_shift_hz": 6.25
                }
            elif bw_half < 5.0:
                return {
                    "modulation_type": "BPSK (Amateur PSK31 / Binary Phase Shift Keying)",
                    "confidence": 0.96,
                    "decision_path": f"Gate 3.5: Squared Spectral Delta Line (BW={bw_hz:.1f} Hz, Squared BW={bw_half:.2f} Hz) -> BPSK (PSK31)",
                    "fsk_shift_hz": 0.0
                }

    # Sub-case C: Passband 2-FSK / AFSK via Instantaneous Frequency Clusters
    ph = np.unwrap(np.angle(signal[:65536]))
    inst_f = np.diff(ph) * (fs / (2.0 * np.pi))
    if len(inst_f) > 512:
        inst_f_filt = medfilt(inst_f[:32768], 7)
        f_p5, f_p95 = np.percentile(inst_f_filt, 5), np.percentile(inst_f_filt, 95)
        in_b = (inst_f_filt >= f_p5) & (inst_f_filt <= f_p95)
        f_act = inst_f_filt[in_b]
        if len(f_act) > 1000:
            counts, b_edges = np.histogram(f_act, bins=60)
            pk_idx, _ = find_peaks(counts, height=0.22 * np.max(counts), distance=4)
            pk_freqs = (b_edges[pk_idx] + b_edges[pk_idx + 1]) / 2.0
            if len(pk_freqs) >= 2:
                # Guard against phase-shift keying (QPSK / BPSK) producing bimodal phase-jump frequency spikes
                hoc_check = compute_higher_order_cumulants(signal)
                if hoc_check.get("c42_real", 0.0) < -0.55 or hoc_check.get("c40_mag", 0.0) > 0.65:
                    return None

                top2 = pk_freqs[np.argsort(counts[pk_idx])[-2:]]
                shift = float(abs(top2[1] - top2[0]))
                if 120.0 <= shift <= 220.0 and (bw_hz == 0.0 or bw_hz <= 600.0):
                    return {
                        "modulation_type": "2-FSK (170 Hz Shift / NAVTEX / SITOR-B)",
                        "confidence": 0.96,
                        "decision_path": f"Gate 3.5: Frequency Shift Keying (Shift={shift:.1f} Hz ~= 170 Hz) -> NAVTEX / SITOR-B",
                        "fsk_shift_hz": shift
                    }
                elif 800.0 <= shift <= 1400.0:
                    return {
                        "modulation_type": "2-FSK / AFSK (Bell 202 / APRS Packet Data)",
                        "confidence": 0.95,
                        "decision_path": f"Gate 3.5: AFSK Dual Tone (Shift={shift:.1f} Hz ~= 1000 Hz) -> Bell 202 / APRS",
                        "fsk_shift_hz": shift
                    }
                elif 40.0 <= shift <= 0.35 * fs:
                    return {
                        "modulation_type": f"2-FSK ({shift:.0f} Hz Shift)",
                        "confidence": 0.92,
                        "decision_path": f"Gate 3.5: Bimodal Frequency Clusters (Shift={shift:.1f} Hz) -> 2-FSK",
                        "fsk_shift_hz": shift
                    }

    return None


def classify_modulation_cumulants(
    signal: np.ndarray,
    fs: float,
    fc_est: float = 0.0,
    snr_db_hint: float = 10.0,
    pulse_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Master 5-Stage Physics-Based Modulation Classifier:
    - Stage 1: Pulsed Radar / OTH Intercept (Linear FMOP R^2 fit, matched radar frames)
    - Stage 2: TDMA Cellular / Digital Burst (GSM GMSK / DMR 4-FSK / AIS packet frames)
    - Stage 3: Analog Audio / Voice & NFM Pre-Classifier (Speech formants, telephony BW >= 1200 Hz, SFM < 0.15)
    - Stage 3.5: Frequency Shift Keying (POCSAG baseband, NAVTEX 170 Hz shift, APRS AFSK, FT8 M-FSK)
    - Stage 4: Continuous Digital Communication (HOC Cumulants: BPSK, QPSK, 8-PSK, 16-QAM, AM, FM, CW)
    """
    if len(signal) < 256 or fs <= 0:
        return {
            "modulation_type": "Unknown / Insufficient Data",
            "confidence": 0.0,
            "decision_path": "Signal length < 256 samples",
            "cumulants": {}
        }

    env = np.abs(signal)
    env_var_ratio = float(np.std(env) / (np.mean(env) + 1e-12))

    # Extract spectral bandwidth for gates
    f_s, psd_db, psd_lin = compute_welch_psd(signal, fs, nperseg=1024)
    cum_pwr = np.cumsum(psd_lin)
    tot_pwr = cum_pwr[-1] + 1e-18
    low_idx = np.searchsorted(cum_pwr, 0.005 * tot_pwr)
    high_idx = np.searchsorted(cum_pwr, 0.995 * tot_pwr)
    bw_99 = float(abs(f_s[high_idx] - f_s[low_idx])) if high_idx > low_idx else 0.0

    # Autonomous Multi-Domain Zero False-Positive Triage Pre-Gate
    from .autonomous_detector import detect_signal_autonomously
    det = detect_signal_autonomously(signal, fs, pulse_info=pulse_info)
    if det["signal_class_id"] != "UNKNOWN" and not det["signal_class_id"].startswith("GENERIC_"):
        is_radar = (det["extraction_pipeline"] == "pulsed_radar")
        is_tdma = (det["extraction_pipeline"] == "tdma_burst")
        is_analog = (det["extraction_pipeline"] == "analog_voice")
        is_ale = (det["signal_class_id"] == "MIL_2G_ALE")

        return {
            "modulation_type": det["protocol_name"],
            "confidence": det["confidence"],
            "signal_class_id": det["signal_class_id"],
            "modulation_family": det["modulation_family"],
            "decision_path": " -> ".join(det["physical_evidence"][:3]),
            "cumulants": compute_higher_order_cumulants(signal),
            "envelope_variance_ratio": float(np.round(env_var_ratio, 4)),
            "frequency_std_hz": float(np.round(np.std(np.diff(np.unwrap(np.angle(signal)))) * (fs / (2.0 * np.pi)), 2)),
            "is_radar": is_radar,
            "is_tdma": is_tdma,
            "is_analog_audio": is_analog,
            "is_ale": is_ale,
            "baud_rate_hint": det.get("baud_rate_nominal"),
            "autonomous_detection": det
        }

    # -------------------------------------------------------------
    # PRE-GATE: 8-TONE MFSK (2G ALE / MIL-STD-188-141)
    # -------------------------------------------------------------
    ale_check = detect_8mfsk_ale(signal, fs)
    if ale_check is not None:
        return {
            "modulation_type": ale_check["modulation_type"],
            "confidence": ale_check["confidence"],
            "decision_path": ale_check["decision_path"],
            "cumulants": {},
            "envelope_variance_ratio": float(np.round(env_var_ratio, 4)),
            "frequency_std_hz": float(np.round(np.std(np.diff(np.unwrap(np.angle(signal)))) * (fs / (2.0 * np.pi)), 2)),
            "is_radar": False,
            "is_tdma": False,
            "is_analog_audio": False,
            "is_ale": True,
            "baud_rate_hint": 125.0
        }

    # -------------------------------------------------------------
    # GATE 1: PULSED RADAR & OTH INTERCEPT GATING
    # -------------------------------------------------------------
    if pulse_info is not None and pulse_info.get("is_pulsed", False) and pulse_info.get("is_radar", False):
        intra_mod = pulse_info.get("intra_pulse_modulation", {})
        radar_mod_type = intra_mod.get("intra_pulse_mod", "Pulsed Radar (FMOP / Chirp)")
        chirp_rate = intra_mod.get("chirp_rate_hz_per_sec", 0.0)
        r2 = intra_mod.get("r2_goodness_of_fit", 0.0)
        multi_prf = pulse_info.get("multi_rate_prf", {})

        decision_path = [
            f"Gate 1: Pulsed Radar Detected (Duty Cycle = {pulse_info.get('duty_cycle_pct', 0):.1f}%)",
            f"Linear Chirp Trajectory (R^2 = {r2:.2f}, Slope = {chirp_rate/1e6:+.2f} MHz/s)",
            f"PRF Mode: {multi_prf.get('prf_mode_label', 'N/A')}"
        ]

        return {
            "modulation_type": radar_mod_type,
            "confidence": 0.98,
            "decision_path": " -> ".join(decision_path),
            "cumulants": {},
            "envelope_variance_ratio": float(np.round(np.std(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-12), 4)),
            "frequency_std_hz": float(np.round(np.std(np.diff(np.unwrap(np.angle(signal)))) * (fs / (2.0 * np.pi)), 2)),
            "is_radar": True,
            "is_tdma": False,
            "is_analog_audio": False
        }

    # -------------------------------------------------------------
    # GATE 2: TDMA CELLULAR / DIGITAL BURST GATING (GSM / DMR / AIS)
    # -------------------------------------------------------------
    if pulse_info is not None and pulse_info.get("is_pulsed", False) and pulse_info.get("is_tdma", False):
        multi_prf = pulse_info.get("multi_rate_prf", {})
        frame_match = multi_prf.get("matched_standard_frame")

        if frame_match and "GSM" in frame_match:
            mod_title = "TDMA Cellular (GSM GMSK / Cellular Downlink)"
        elif frame_match and "DMR" in frame_match:
            mod_title = "TDMA Mobile Radio (DMR 4-FSK / TDMA)"
        elif frame_match and "TETRA" in frame_match:
            mod_title = "TDMA Trunked Radio (TETRA pi/4-DQPSK)"
        else:
            mod_title = "TDMA Digital Burst (GMSK / Continuous Phase FSK)"

        decision_path = [
            f"Gate 2: TDMA Digital Burst Detected (Duty Cycle = {pulse_info.get('duty_cycle_pct', 0):.1f}%)",
            f"Matched Frame: {frame_match or 'Packet Burst'}",
            "Intra-Burst Phase / Discrete Keying"
        ]

        return {
            "modulation_type": mod_title,
            "confidence": 0.96 if frame_match else 0.92,
            "decision_path": " -> ".join(decision_path),
            "cumulants": {},
            "envelope_variance_ratio": float(np.round(np.std(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-12), 4)),
            "frequency_std_hz": float(np.round(np.std(np.diff(np.unwrap(np.angle(signal)))) * (fs / (2.0 * np.pi)), 2)),
            "is_radar": False,
            "is_tdma": True,
            "is_analog_audio": False
        }

    # Extract spectral bandwidth for downstream gates
    f_s, psd_db, psd_lin = compute_welch_psd(signal, fs, nperseg=1024)
    cum_pwr = np.cumsum(psd_lin)
    tot_pwr = cum_pwr[-1] + 1e-18
    low_idx = np.searchsorted(cum_pwr, 0.005 * tot_pwr)
    high_idx = np.searchsorted(cum_pwr, 0.995 * tot_pwr)
    bw_99 = float(abs(f_s[high_idx] - f_s[low_idx])) if high_idx > low_idx else 0.0

    # -------------------------------------------------------------
    # GATE 3: ANALOG AUDIO / VOICE & NFM PRE-CLASSIFIER
    # -------------------------------------------------------------
    envelope = np.abs(signal)
    mean_env = float(np.mean(envelope)) + 1e-12
    std_env = float(np.std(envelope))
    env_var_ratio = std_env / mean_env

    sfm, num_formant_peaks = compute_spectral_flatness(signal, fs)

    # Speech physically requires audio telephony bandwidth (>= 1200 Hz),
    # formant frequency dynamics (num_formant_peaks >= 2), and envelope cadences (env_var_ratio > 0.35).
    # Narrowband digital modes (PSK31, FT8, CW) and pulsed radar signals are strictly excluded!
    is_radar_guard = bool(pulse_info is not None and pulse_info.get("is_radar", False))
    pulse_duty = pulse_info.get("duty_cycle_pct", 100.0) if pulse_info else 100.0
    is_pulsed_rf = bool(pulse_info and pulse_info.get("is_pulsed", False) and pulse_duty < 65.0)
    if not is_radar_guard and not is_pulsed_rf and fs <= 96_000.0 and sfm < 0.15 and env_var_ratio > 0.35 and bw_99 >= 1200.0 and num_formant_peaks >= 2:
        decision_path = [
            f"Gate 3: Analog Audio Baseband Detected (Fs={fs/1e3:.1f} kHz <= 96 kHz, BW={bw_99:,.1f} Hz)",
            f"Speech/Tone Formant Dynamics (SFM={sfm:.4f} < 0.15, Peaks={num_formant_peaks})",
            "Analog NFM / Voice & Variometer Tones"
        ]

        return {
            "modulation_type": "Analog (NFM / Voice & Variometer Tones)",
            "confidence": 0.96,
            "decision_path": " -> ".join(decision_path),
            "cumulants": {},
            "envelope_variance_ratio": float(np.round(env_var_ratio, 4)),
            "frequency_std_hz": float(np.round(np.std(np.diff(np.unwrap(np.angle(signal)))) * (fs / (2.0 * np.pi)), 2)),
            "is_radar": False,
            "is_tdma": False,
            "is_analog_audio": True
        }

    # -------------------------------------------------------------
    # GATE 3.5: FREQUENCY SHIFT KEYING (FSK) DISCRIMINATOR
    # -------------------------------------------------------------
    fsk_res = detect_frequency_shift_keying(signal, fs, bw_hz=bw_99, fc_est=fc_est)
    if fsk_res is not None:
        return {
            "modulation_type": fsk_res["modulation_type"],
            "confidence": fsk_res["confidence"],
            "decision_path": fsk_res["decision_path"],
            "cumulants": {},
            "envelope_variance_ratio": float(np.round(env_var_ratio, 4)),
            "frequency_std_hz": float(np.round(np.std(np.diff(np.unwrap(np.angle(signal)))) * (fs / (2.0 * np.pi)), 2)),
            "is_radar": False,
            "is_tdma": False,
            "is_analog_audio": False
        }

    # -------------------------------------------------------------
    # GATE 4: CONTINUOUS DIGITAL / ANALOG COMMUNICATIONS (HOC)
    # -------------------------------------------------------------
    t = np.arange(len(signal)) / fs
    sig_sub = signal[:min(len(signal), 65536)]

    # Non-linear squaring line recovery (for BPSK: peak at 2*fc)
    n_fft = 131072
    spec2 = np.abs(np.fft.fft(sig_sub ** 2, n=n_fft))
    freqs2 = np.fft.fftfreq(n_fft, d=1.0 / fs)
    mask2 = (freqs2 >= 2.0 * (fc_est - 120.0)) & (freqs2 <= 2.0 * (fc_est + 120.0))
    peak2_prom = 0.0
    exact_fc2 = fc_est

    if np.any(mask2):
        best_f2 = freqs2[mask2][np.argmax(spec2[mask2])]
        exact_fc2 = float(best_f2 / 2.0)
        median2 = float(np.median(spec2[mask2])) + 1e-12
        peak2_prom = float(np.max(spec2[mask2]) / median2)

    if peak2_prom > 8.0:
        bb_locked = signal * np.exp(-1j * 2.0 * np.pi * exact_fc2 * t)
        fine_offset = exact_fc2 - fc_est
    else:
        if abs(fc_est) > 5.0:
            bb_coarse = signal * np.exp(-1j * 2.0 * np.pi * fc_est * t)
        else:
            bb_coarse = signal

        fine_offset = estimate_fine_carrier_offset(bb_coarse, fs)
        if abs(fine_offset) > 0.05:
            bb_locked = bb_coarse * np.exp(-1j * 2.0 * np.pi * fine_offset * t)
        else:
            bb_locked = bb_coarse

    hoc_bb = compute_higher_order_cumulants(bb_locked)
    hoc_raw = compute_higher_order_cumulants(signal)
    c40_mag = float(max(hoc_bb["c40_mag"], hoc_raw["c40_mag"]))
    c42_real = float(min(hoc_bb["c42_real"], hoc_raw["c42_real"]))
    c20_mag = float(max(hoc_bb["c20_mag"], hoc_raw["c20_mag"]))
    hoc = {
        "c20_mag": c20_mag,
        "c21": hoc_bb["c21"],
        "c40_mag": c40_mag,
        "c40_real": hoc_bb["c40_real"],
        "c40_imag": hoc_bb["c40_imag"],
        "c42_real": c42_real,
        "c42_mag": hoc_bb["c42_mag"]
    }

    unwrapped_phase = np.unwrap(np.angle(bb_locked))
    inst_freq = np.diff(unwrapped_phase) * (fs / (2.0 * np.pi))
    freq_std = float(np.std(inst_freq))

    decision_path = []
    mod_type = "Unknown"
    confidence = 0.85

    # Guard against pure noise reaching Gate 4
    if c40_mag < 0.25 and abs(c42_real) < 0.25 and c20_mag < 0.25 and peak2_prom < 6.0:
        return {
            "modulation_type": "Unknown / Noise Floor",
            "confidence": 0.0,
            "decision_path": "Gate 4: Near-zero cumulants (|c40| < 0.25, |c42| < 0.25) -> Noise",
            "cumulants": hoc,
            "envelope_variance_ratio": float(np.round(env_var_ratio, 4)),
            "frequency_std_hz": float(np.round(freq_std, 2)),
            "fine_frequency_offset_hz": float(np.round(fine_offset, 2)),
            "is_radar": False,
            "is_tdma": False,
            "is_analog_audio": False
        }

    # Case A: Strict Constant Envelope (CW, FM, FSK, PSK)
    if env_var_ratio < 0.12:
        decision_path.append(f"Constant envelope (var={env_var_ratio:.3f} < 0.12)")

        if c20_mag >= 0.45 or c40_mag >= 1.30 or peak2_prom >= 10.0:
            mod_type = "BPSK"
            decision_path.append(f"High |c20|={c20_mag:.2f} / |c40|={c40_mag:.2f} -> BPSK")
            confidence = 0.95
        elif (c20_mag < 0.40) and (c42_real <= -0.50) and (c40_mag >= 0.45):
            mod_type = "QPSK"
            decision_path.append(f"QPSK signature (|c40|={c40_mag:.2f}, c42={c42_real:.2f}) -> QPSK")
            confidence = 0.94
        elif (c20_mag < 0.40) and (c42_real <= -0.50) and (c40_mag < 0.45):
            mod_type = "8-PSK / QPSK (Digital PSK)"
            decision_path.append(f"8-PSK signature (|c40|={c40_mag:.2f}, c42={c42_real:.2f}) -> 8-PSK/QPSK")
            confidence = 0.95
        elif freq_std > 0.010 * fs:
            mod_type = "FM (Frequency Modulated)"
            decision_path.append(f"Continuous frequency spread (std={freq_std/1e3:.1f}kHz) -> FM")
            confidence = 0.88
        else:
            mod_type = "Continuous Wave (CW)"
            decision_path.append("Near-zero frequency deviation -> CW")
            confidence = 0.95

    # Case B: Moderate Envelope Variation (QPSK, 8-PSK, BPSK, 16-QAM, AM)
    elif env_var_ratio < 0.28:
        decision_path.append(f"Moderate envelope variance ({env_var_ratio:.3f} in [0.12, 0.28))")

        if c20_mag >= 0.45 or c40_mag >= 1.30 or peak2_prom >= 10.0:
            mod_type = "BPSK"
            decision_path.append(f"BPSK signature (|c40|={c40_mag:.2f}, |c20|={c20_mag:.2f}) -> BPSK")
            confidence = 0.93
        elif (c20_mag < 0.40) and (c42_real <= -0.50) and (c40_mag >= 0.45):
            mod_type = "QPSK"
            decision_path.append(f"QPSK signature (|c40|={c40_mag:.2f}, c42={c42_real:.2f}) -> QPSK")
            confidence = 0.94
        elif (c20_mag < 0.40) and (c42_real <= -0.50) and (c40_mag < 0.45):
            mod_type = "8-PSK / QPSK (Digital PSK)"
            decision_path.append(f"8-PSK signature (|c40|={c40_mag:.2f}, c42={c42_real:.2f}) -> 8-PSK")
            confidence = 0.95
        elif (c20_mag < 0.40) and (-1.2 <= c42_real <= -0.28) and (c40_mag < 0.95):
            mod_type = "16-QAM"
            decision_path.append(f"16-QAM signature (|c40|={c40_mag:.2f}, c42={c42_real:.2f}) -> 16-QAM")
            confidence = 0.92
        else:
            mod_type = "Unknown / Novel Digital Signal"
            decision_path.append("Unmatched PSK/QAM boundary")
            confidence = 0.0

    # Case C: High Envelope Fluctuation (BPSK Filtered, 8-PSK, 16-QAM, AM)
    else:
        decision_path.append(f"High envelope variance ({env_var_ratio:.3f} >= 0.28)")

        if c20_mag >= 0.45 or c40_mag >= 1.30 or peak2_prom >= 10.0:
            mod_type = "BPSK (Filtered / PSK31)"
            decision_path.append(f"BPSK symmetry (|c40|={c40_mag:.2f}, |c20|={c20_mag:.2f}) -> BPSK")
            confidence = 0.94
        elif (c20_mag < 0.40) and (c42_real <= -0.50) and (c40_mag >= 0.45):
            mod_type = "QPSK (Filtered)"
            decision_path.append(f"QPSK signature (|c40|={c40_mag:.2f}, c42={c42_real:.2f}) -> QPSK")
            confidence = 0.93
        elif (c20_mag < 0.40) and (c42_real <= -0.50) and (c40_mag < 0.45):
            mod_type = "8-PSK / QPSK (STANAG 4285 / Digital PSK)"
            decision_path.append(f"8-PSK cumulants (|c40|={c40_mag:.2f}, c42={c42_real:.2f}) -> 8-PSK")
            confidence = 0.95
        elif (-1.2 <= c42_real <= -0.28) and c40_mag < 0.95 and c20_mag < 0.40:
            mod_type = "16-QAM"
            decision_path.append(f"16-QAM cumulants (|c40|={c40_mag:.2f}, c42={c42_real:.2f}) -> 16-QAM")
            confidence = 0.92
        elif c42_real >= -0.30 and env_var_ratio > 0.28:
            mod_type = "AM (Amplitude Modulated)"
            decision_path.append(f"Positive/near-zero c42={c42_real:.2f} -> AM")
            confidence = 0.90
        else:
            mod_type = "Unknown / Novel Digital Signal"
            decision_path.append("Unmatched multi-level amplitude distribution")
            confidence = 0.0

    return {
        "modulation_type": mod_type,
        "confidence": float(np.round(confidence, 3)),
        "decision_path": " -> ".join(decision_path),
        "cumulants": hoc,
        "envelope_variance_ratio": float(np.round(env_var_ratio, 4)),
        "frequency_std_hz": float(np.round(freq_std, 2)),
        "fine_frequency_offset_hz": float(np.round(fine_offset, 2)),
        "is_radar": False,
        "is_tdma": False,
        "is_analog_audio": False
    }
