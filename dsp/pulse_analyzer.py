"""
Pulsed Signal & Radar Parameter Extraction Engine
=================================================
Analyzes pulsed radio frequency (RF) signals, Over-The-Horizon (OTH) radars,
and TDMA burst communications using physics-based descriptors:
- Intra-Burst Frequency Modulation On Pulse (FMOP / Linear FM Chirp) via Linear Regression
- Intra-Burst TDMA Comms Discrimination (GMSK / FSK / GSM / DMR)
- Multi-Scale Envelope Autocorrelation (EAC) with Standard Frame Matching & GCD Sieve
- Segmented In-Pulse vs Inter-Pulse SNR Estimation
- Continuous Transmission Gating (Duty Cycle >= 70% Guardrail)
- Geometric PW <= PRI Invariant Enforcement
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def check_is_8mfsk_ale(signal: np.ndarray, fs: float) -> bool:
    """
    Detects 8-Tone MFSK Automatic Link Establishment (2G ALE / MIL-STD-188-141).
    Verifies 8 discrete tones spaced by ~250 Hz in the voice band [600, 2700] Hz.
    """
    if len(signal) < 2048 or fs <= 0:
        return False
    from scipy.signal import welch
    n_pts = min(len(signal), 131072)
    nperseg = 4096 if fs <= 48000 else 8192
    f_w, psd_w = welch(signal.real[:n_pts], fs=fs, nperseg=nperseg)
    voice_mask = (f_w >= 600.0) & (f_w <= 2700.0)
    if not np.any(voice_mask):
        return False
    f_v = f_w[voice_mask]
    psd_v = psd_w[voice_mask]
    pk_thresh = 0.08 * np.max(psd_v)
    bin_dist = max(3, int(180.0 / (f_w[1] - f_w[0])))
    pks_v, _ = find_peaks(psd_v, height=pk_thresh, distance=bin_dist)
    pk_freqs = f_v[pks_v]
    if len(pk_freqs) in [7, 8, 9]:
        base_f = pk_freqs[0]
        harm_errs = [abs((f - base_f) - round((f - base_f) / 250.0) * 250.0) for f in pk_freqs]
        return bool(np.mean(harm_errs) < 22.0 and np.max(harm_errs) < 55.0)
    return False


def match_standard_frame(
    pri_us: float,
    prf_hz: float,
    secondary_prf: Optional[float] = None,
    bw_hz: float = 0.0
) -> Optional[str]:
    """
    Matches extracted PRI / PRF against known standard radar and TDMA communication frames.
    Checks both primary and secondary candidate PRFs / PRIs.
    Guards OTH Radars against audio-band communications (bw_hz < 3500 Hz).
    """
    candidates = [(pri_us, prf_hz)]
    if secondary_prf and secondary_prf > 0:
        candidates.append((1e6 / secondary_prf, secondary_prf))

    for pri, prf in candidates:
        if pri <= 0 or prf <= 0:
            continue

        # 1. GSM 2G TDMA Frame (4.615 ms / 216.67 Hz), Timeslot (576.9 us / 1733.3 Hz)
        if (bw_hz == 0.0 or bw_hz >= 12000.0) and ((abs(pri - 4615.4) < 75.0) or (abs(pri - 576.9) < 20.0) or (215.5 <= prf <= 217.5)):
            return "GSM 2G TDMA Frame (4.615 ms / 216.7 Hz)"

        # 2. DMR / MOTOTRBO Digital TDMA Frame (60 ms / 16.67 Hz, Timeslot 30 ms / 33.33 Hz)
        if (bw_hz == 0.0 or bw_hz >= 5500.0) and (any(abs(pri - target) < 400.0 for target in [60000.0, 30000.0]) or (16.2 <= prf <= 17.2) or (32.8 <= prf <= 33.9)):
            return "DMR / MOTOTRBO Digital TDMA Frame (30 ms / 60 ms)"

        # 3. TETRA Frame (56.67 ms / 17.65 Hz, Timeslot 14.17 ms / 70.59 Hz)
        if (bw_hz == 0.0 or bw_hz >= 7500.0) and (any(abs(pri - target) < 350.0 for target in [56670.0, 14170.0]) or (17.0 <= prf <= 18.2) or (69.5 <= prf <= 71.5)):
            return "TETRA Digital Trunked Frame (14.17 ms / 56.67 ms)"

        # OTH Radars strictly require wideband RF channels (BW >= 3.5 kHz)
        if bw_hz == 0.0 or bw_hz >= 3500.0:
            # 4. Chinese OTH-SW Radar (23.26 ms / 43 Hz, 11.63 ms / 86 Hz)
            is_oth_43 = (abs(pri - 23260.0) < 650.0) or (42.0 <= prf <= 44.5 and abs(pri - 23260.0) < 1500.0)
            is_oth_86 = (abs(pri - 11630.0) < 250.0 and (85.0 <= prf <= 87.5))
            if is_oth_43 or is_oth_86:
                return "OTH-SW Radar Frame (23.26 ms / 43 Hz Mode)"

            # 5. Iranian Ghadir OTH Radar (3.25 ms / 307 Hz, 1.15 ms / 870 Hz)
            if (abs(pri - 3257.0) < 85.0 and (295.0 <= prf <= 318.0)) or (abs(pri - 1149.0) < 80.0 and (850.0 <= prf <= 900.0)):
                return "Ghadir OTH Radar Frame (307 Hz / 870 Hz Dual-Mode)"

            # 6. Russian Woodpecker / Duga OTH Radar (100 ms / 10 Hz Mode)
            if (abs(pri - 100000.0) < 5000.0) or (abs(pri - 50000.0) < 2500.0) or (9.7 <= prf <= 10.3) or (19.7 <= prf <= 20.3):
                return "Russian Woodpecker / Duga OTH Radar (10 Hz Mode)"

    return None


def analyze_intra_pulse_modulation(
    pulse_slices: List[np.ndarray],
    fs: float
) -> Dict[str, Any]:
    """
    Analyzes intra-pulse phase and frequency trajectory across isolated pulse segments
    to distinguish Linear FM Chirp (Radar FMOP) from TDMA Digital Communications.
    """
    if len(pulse_slices) == 0 or fs <= 0:
        return {
            "intra_pulse_mod": "Pulsed CW (Unmodulated Carrier)",
            "chirp_rate_hz_per_sec": 0.0,
            "chirp_bandwidth_hz": 0.0,
            "r2_goodness_of_fit": 0.0,
            "is_fmop_chirp": False,
            "is_tdma_comms": False
        }

    chirp_slopes: List[float] = []
    chirp_bws: List[float] = []
    r2_scores: List[float] = []
    freq_stds: List[float] = []

    for pulse in pulse_slices[:60]:
        if len(pulse) < 4:
            continue

        inst_phase = np.unwrap(np.angle(pulse))
        inst_freq = np.diff(inst_phase) * (fs / (2.0 * np.pi))

        if len(inst_freq) < 2:
            continue

        t_axis = np.arange(len(inst_freq)) / fs

        try:
            poly = np.polyfit(t_axis, inst_freq, 1)
            slope = float(poly[0])
            fit_curve = np.polyval(poly, t_axis)
            pulse_duration = len(pulse) / fs
            sweep_bw = abs(slope * pulse_duration)

            ss_res = np.sum((inst_freq - fit_curve) ** 2)
            ss_tot = np.sum((inst_freq - np.mean(inst_freq)) ** 2)
            r2 = float(1.0 - (ss_res / (ss_tot + 1e-12))) if ss_tot > 1e-6 else 0.0

            chirp_slopes.append(slope)
            chirp_bws.append(sweep_bw)
            r2_scores.append(r2)
            freq_stds.append(float(np.std(inst_freq)))
        except Exception:
            continue

    if len(chirp_slopes) == 0:
        return {
            "intra_pulse_mod": "Pulsed CW (Unmodulated Carrier)",
            "chirp_rate_hz_per_sec": 0.0,
            "chirp_bandwidth_hz": 0.0,
            "r2_goodness_of_fit": 0.0,
            "is_fmop_chirp": False,
            "is_tdma_comms": False
        }

    med_slope = float(np.median(chirp_slopes))
    med_bw = float(np.median(chirp_bws))
    med_r2 = float(np.median(r2_scores))
    med_fstd = float(np.median(freq_stds))
    avg_dur = float(np.mean([len(p) for p in pulse_slices])) / fs if pulse_slices else 0.0

    # Linear Chirp Detection (FMOP):
    # Significant frequency sweep bandwidth (>= 75 Hz), steep slope (|slope| >= 1e4 Hz/s),
    # with genuine linear trajectory goodness-of-fit (R^2 >= 0.35)
    is_chirp = (med_r2 >= 0.35 and med_bw >= 75.0 and abs(med_slope) >= 1e4)

    # TDMA Comms Detection:
    # Requires frequency variance across symbol transitions without linear chirp
    # and pulse duration >= 300 us (GSM is 577 us, DMR 30 ms, AIS 26 ms; microsecond pulses are radar)
    is_tdma = (not is_chirp) and (avg_dur >= 300e-6) and (med_fstd >= 350.0 and med_bw >= 200.0)

    if is_chirp:
        direction = "Up-Chirp" if med_slope > 0 else "Down-Chirp"
        mod_name = f"Pulsed FMOP (Linear FM {direction} / Radar Sweep)"
    elif is_tdma:
        mod_name = "TDMA Digital Burst (GMSK / Continuous Phase FSK)"
    else:
        mod_name = "Pulsed CW (Unmodulated Carrier)"

    return {
        "intra_pulse_mod": mod_name,
        "chirp_rate_hz_per_sec": med_slope,
        "chirp_bandwidth_hz": med_bw,
        "r2_goodness_of_fit": float(np.round(med_r2, 3)),
        "is_fmop_chirp": is_chirp,
        "is_tdma_comms": is_tdma
    }


def compute_segmented_pulsed_snr(
    signal: np.ndarray,
    rising_indices: np.ndarray,
    falling_indices: np.ndarray,
    eps: float = 1e-12
) -> Dict[str, float]:
    """
    Computes true In-Pulse vs Inter-Pulse SNR by segmenting active pulse intervals
    and quiet noise-floor intervals.
    """
    n_samples = len(signal)
    if n_samples == 0 or len(rising_indices) == 0:
        return {"pulsed_snr_db": 0.0, "in_pulse_power": 0.0, "noise_floor_power": 1.0}

    pulse_mask = np.zeros(n_samples, dtype=bool)

    for r, f in zip(rising_indices, falling_indices):
        if r < f and r < n_samples:
            pulse_mask[r:min(f, n_samples)] = True

    quiet_mask = ~pulse_mask
    inst_power = np.abs(signal) ** 2

    p_pulse = float(np.mean(inst_power[pulse_mask])) if np.any(pulse_mask) else float(np.mean(inst_power))
    p_noise = float(np.mean(inst_power[quiet_mask])) if np.any(quiet_mask) else eps

    p_signal_only = max(p_pulse - p_noise, eps)
    snr_lin = max(p_signal_only / (p_noise + eps), 1e-4)
    snr_db = float(10.0 * np.log10(snr_lin))

    return {
        "pulsed_snr_db": snr_db,
        "in_pulse_power": p_pulse,
        "noise_floor_power": p_noise
    }


def extract_autocorr_pri_prf(
    signal: np.ndarray,
    fs: float,
    min_prf: float = 0.5,
    max_prf: Optional[float] = None,
    bw_hz: float = 0.0
) -> Dict[str, Any]:
    """
    Extracts true radar & TDMA fundamental PRI & PRF using Envelope Autocorrelation (EAC),
    GCD harmonic sieve, candidate scoring, and standard frame matching.
    Supports low-PRF radars down to 0.5 Hz.
    """
    if bw_hz <= 0.0 and len(signal) >= 256:
        from scipy.signal import welch
        f_w, psd_w = welch(signal.real[:min(len(signal), 16384)], fs=fs, nperseg=min(1024, len(signal)))
        cum_p = np.cumsum(psd_w)
        tot_p = cum_p[-1] + 1e-12
        l_idx = np.searchsorted(cum_p, 0.005 * tot_p)
        h_idx = np.searchsorted(cum_p, 0.995 * tot_p)
        bw_hz = float(abs(f_w[h_idx] - f_w[l_idx])) if h_idx > l_idx else 0.0

    raw_env = np.abs(signal)
    
    sigma_samples = max(1, min(15, int(0.00035 * fs)))
    smooth_env = gaussian_filter1d(raw_env, sigma=sigma_samples)
    
    env_ac = smooth_env - np.mean(smooth_env)
    n = len(env_ac)
    fft_e = np.fft.rfft(env_ac, n=2 * n)
    r_xx = np.fft.irfft(fft_e * np.conj(fft_e))[:n]
    r_xx = r_xx / (r_xx[0] + 1e-12)

    if max_prf is None:
        max_prf = min(25000.0, 0.05 * fs) if fs > 100_000.0 else 1500.0

    min_lag = max(2, int(fs / max_prf))
    max_lag = min(len(r_xx) - 1, int(fs / max_prf_min_bound if (max_prf_min_bound := max(min_prf, 0.1)) else 0.5))

    search_r = r_xx[min_lag:max_lag]
    lags = np.arange(min_lag, max_lag)

    peaks, _ = find_peaks(search_r, height=0.18, distance=max(4, int(fs / (max_prf * 1.5))))
    if len(peaks) == 0:
        return {
            "primary_prf_hz": 0.0,
            "primary_pri_us": 0.0,
            "secondary_prf_hz": None,
            "is_dual_rate": False,
            "prf_mode_label": "N/A",
            "matched_standard_frame": None
        }

    peak_lags = lags[peaks]
    peak_heights = search_r[peaks]
    max_h = float(np.max(peak_heights))

    if max_h < 0.25:
        return {
            "primary_prf_hz": 0.0,
            "primary_pri_us": 0.0,
            "secondary_prf_hz": None,
            "is_dual_rate": False,
            "prf_mode_label": "N/A",
            "matched_standard_frame": None
        }

    sig_mask = peak_heights >= 0.45 * max_h
    sig_lags = peak_lags[sig_mask]
    sig_heights = peak_heights[sig_mask]

    # Sort candidates by lag ascending so fundamental period is tested first
    sort_order = np.argsort(sig_lags)
    sig_lags = sig_lags[sort_order]
    sig_heights = sig_heights[sort_order]

    # Fundamental vs Harmonic Candidate Scoring
    best_cand_idx = 0
    best_cand_score = -1.0
    for idx, (lag, h) in enumerate(zip(sig_lags, sig_heights)):
        cand_pri_s = lag / float(fs)
        cand_prf_hz = 1.0 / cand_pri_s
        frame = match_standard_frame(cand_pri_s * 1e6, cand_prf_hz, bw_hz=bw_hz)

        # Multiples support count (checking if 2*lag, 3*lag have autocorrelation peaks)
        mult_cnt = sum(1 for k in range(2, 6) if (k * lag) < len(r_xx) and r_xx[k * lag] >= 0.25 * max_h)

        # Subharmonic penalty: if lag is an integer multiple of a smaller candidate peak with strong height
        is_subharmonic_of_earlier = any(
            abs(lag / float(earlier_lag) - round(lag / float(earlier_lag))) < 0.06
            and round(lag / float(earlier_lag)) >= 2
            for earlier_lag in sig_lags[:idx]
        )

        cand_score = (h / max_h) + 0.35 * mult_cnt
        if is_subharmonic_of_earlier:
            cand_score -= 2.0  # Strongly penalize harmonic repetitions of the true fundamental
        elif frame:
            cand_score += 1.5  # Strong boost for matched standard radar/TDMA frame
        if cand_score > best_cand_score:
            best_cand_score = cand_score
            best_cand_idx = idx

    tau_0 = sig_lags[best_cand_idx]
    pri_0_s = tau_0 / float(fs)
    prf_0_hz = 1.0 / pri_0_s

    secondary_prf: Optional[float] = None
    is_dual = False

    for lag, h in zip(sig_lags, sig_heights):
        if lag == tau_0:
            continue
        cand_pri_s = lag / float(fs)
        cand_prf_hz = 1.0 / cand_pri_s
        ratio = lag / float(tau_0) if tau_0 > 0 else 1.0
        is_harmonic = any(abs(ratio - k) < 0.08 or abs(ratio - 1.0/k) < 0.08 for k in range(2, 25))
        
        if not is_harmonic and abs(cand_prf_hz - prf_0_hz) > 25.0 and cand_prf_hz >= 30.0 and h >= 0.45 * max_h:
            secondary_prf = float(cand_prf_hz)
            is_dual = True
            break

    if not is_dual and 38.0 <= prf_0_hz <= 48.0:
        half_lag = int(round(tau_0 / 2.0))
        if half_lag >= min_lag and r_xx[half_lag] >= 0.35:
            secondary_prf = float(fs / half_lag)
            is_dual = True

    matched_frame = match_standard_frame(pri_0_s * 1e6, prf_0_hz, secondary_prf=secondary_prf, bw_hz=bw_hz)

    if is_dual and secondary_prf is not None:
        low_p = min(prf_0_hz, secondary_prf)
        high_p = max(prf_0_hz, secondary_prf)
        mode_label = f"Dual-Rate / Multi-Mode ({low_p:.1f} Hz / {high_p:.1f} Hz)"
    else:
        mode_label = f"Single-Rate PRF ({prf_0_hz:.1f} Hz)"

    if matched_frame:
        mode_label += f" [{matched_frame}]"

    return {
        "primary_prf_hz": float(np.round(prf_0_hz, 2)),
        "primary_pri_us": float(np.round(pri_0_s * 1e6, 2)),
        "secondary_prf_hz": float(np.round(secondary_prf, 2)) if secondary_prf else None,
        "is_dual_rate": is_dual,
        "prf_mode_label": mode_label,
        "matched_standard_frame": matched_frame
    }


def analyze_pulse_train(
    signal: np.ndarray,
    fs: float,
    threshold_ratio: float = 0.40,
    smoothing_sigma: float = 2.0
) -> Dict[str, Any]:
    """
    Comprehensive physics-based radar pulse train and TDMA burst analysis engine.
    Guards against continuous analog speech / FM transmissions (Duty Cycle >= 70%).
    Enforces geometric invariant: PW <= PRI.
    """
    if len(signal) < 100 or fs <= 0:
        return {
            "is_pulsed": False,
            "is_radar": False,
            "is_tdma": False,
            "num_pulses": 0,
            "mean_pulse_width_us": 0.0,
            "mean_pri_us": 0.0,
            "mean_prf_hz": 0.0,
            "duty_cycle_pct": 100.0,
            "pulsed_snr_db": 0.0,
            "signal_mode": "Continuous Wave / Noise"
        }

    # 2G ALE Pre-Check: 8-Tone MFSK Tactical HF Data (MIL-STD-188-141)
    if check_is_8mfsk_ale(signal, fs):
        return {
            "is_pulsed": False,
            "is_radar": False,
            "is_tdma": False,
            "is_ale": True,
            "num_pulses": 0,
            "mean_pulse_width_us": 0.0,
            "mean_pri_us": 8000.0,
            "mean_prf_hz": 125.0,
            "duty_cycle_pct": 100.0,
            "pulsed_snr_db": 0.0,
            "signal_mode": "Tactical HF Data / Handshake Protocol (2G ALE)"
        }

    raw_envelope = np.abs(signal)
    envelope = gaussian_filter1d(raw_envelope, sigma=smoothing_sigma)

    noise_floor = float(np.percentile(envelope, 15))
    noise_median = float(np.median(envelope))
    noise_mad = float(np.median(np.abs(envelope - noise_median)))
    sigma_noise = float(1.4826 * noise_mad)

    p99 = float(np.percentile(envelope, 99))
    p99_5 = float(np.percentile(envelope, 99.5))
    p99_9 = float(np.percentile(envelope, 99.9))
    p_max = float(np.max(envelope))

    # Adaptive sparse-event detection (duty cycle < 1% down to 0.01%)
    # For sparse pulses, p99 can remain in the noise floor, but p99.9 or p_max spikes high.
    p99_in_noise = (p99 - noise_floor) <= 3.0 * max(sigma_noise, 1e-6)
    p_ratio_sparse = float(noise_median / (p99 + 1e-12))
    has_sparse_spikes = (p_max > noise_floor + 5.0 * max(sigma_noise, 1e-6)) and (p99_9 > noise_floor + 3.0 * max(sigma_noise, 1e-6))
    is_sparse = (p_ratio_sparse < 0.25) and p99_in_noise and has_sparse_spikes

    if is_sparse:
        peak_level = max(p99_9, 0.50 * p_max)
        dynamic_range = max(0.0, peak_level - noise_floor)
        v_thresh = noise_floor + max(0.30 * dynamic_range, 3.0 * max(sigma_noise, 1e-6))
    else:
        peak_level = p99
        dynamic_range = max(0.0, peak_level - noise_floor)
        v_thresh = noise_floor + threshold_ratio * dynamic_range

    if dynamic_range < 0.15 * (peak_level + 1e-12) or (not is_sparse and dynamic_range < 2.5 * max(sigma_noise, 1e-6)):
        return {
            "is_pulsed": False,
            "is_radar": False,
            "is_tdma": False,
            "num_pulses": 0,
            "mean_pulse_width_us": 0.0,
            "mean_pri_us": 0.0,
            "mean_prf_hz": 0.0,
            "duty_cycle_pct": 100.0,
            "pulsed_snr_db": 0.0,
            "signal_mode": "Continuous Transmission (Non-Pulsed)"
        }

    is_high = envelope >= v_thresh
    is_high_padded = np.pad(is_high.astype(np.int32), (1, 1), mode='constant', constant_values=0)
    diff = np.diff(is_high_padded)
    rising_indices = np.where(diff == 1)[0]
    falling_indices = np.where(diff == -1)[0]

    if len(rising_indices) == 0 or len(falling_indices) == 0:
        return {
            "is_pulsed": False,
            "is_radar": False,
            "is_tdma": False,
            "num_pulses": 0,
            "mean_pulse_width_us": 0.0,
            "mean_pri_us": 0.0,
            "mean_prf_hz": 0.0,
            "duty_cycle_pct": 100.0,
            "pulsed_snr_db": 0.0,
            "signal_mode": "Continuous Transmission (Non-Pulsed)"
        }

    pulse_widths: List[float] = []
    pulse_starts: List[int] = []
    pulse_slices: List[np.ndarray] = []
    valid_rising: List[int] = []
    valid_falling: List[int] = []

    min_pw_samples = max(2, int(0.0000005 * fs))
    r_idx = 0
    f_idx = 0

    while r_idx < len(rising_indices) and f_idx < len(falling_indices):
        rise = rising_indices[r_idx]
        while f_idx < len(falling_indices) and falling_indices[f_idx] <= rise:
            f_idx += 1

        if f_idx < len(falling_indices):
            fall = falling_indices[f_idx]
            pw_samples = fall - rise
            pw_seconds = pw_samples / float(fs)
            if pw_samples >= min_pw_samples:
                pulse_widths.append(pw_seconds)
                pulse_starts.append(rise)
                pulse_slices.append(signal[rise:fall])
                valid_rising.append(rise)
                valid_falling.append(fall)
            r_idx += 1
            f_idx += 1
        else:
            break

    if len(pulse_widths) == 0:
        return {
            "is_pulsed": False,
            "is_radar": False,
            "is_tdma": False,
            "num_pulses": 0,
            "mean_pulse_width_us": 0.0,
            "mean_pri_us": 0.0,
            "mean_prf_hz": 0.0,
            "duty_cycle_pct": 100.0,
            "pulsed_snr_db": 0.0,
            "signal_mode": "Continuous Transmission (Non-Pulsed)"
        }

    # Segmented In-Pulse SNR
    snr_data = compute_segmented_pulsed_snr(
        signal,
        np.array(valid_rising),
        np.array(valid_falling)
    )

    # Intra-Pulse Modulation Analysis (FMOP vs TDMA)
    intra_mod = analyze_intra_pulse_modulation(pulse_slices, fs)

    # Fundamental PRF / PRI via Envelope Autocorrelation (EAC) & Frame Matching
    eac_prf_info = extract_autocorr_pri_prf(signal, fs)
    matched_frame = eac_prf_info.get("matched_standard_frame")

    pris_seconds = [
        (pulse_starts[k + 1] - pulse_starts[k]) / float(fs)
        for k in range(len(pulse_starts) - 1)
        if pulse_starts[k + 1] > pulse_starts[k]
    ]

    if eac_prf_info["primary_prf_hz"] > 0:
        mean_prf_hz = eac_prf_info["primary_prf_hz"]
        mean_pri_us = eac_prf_info["primary_pri_us"]
        mean_pri_s = mean_pri_us / 1e6
    else:
        mean_pri_s = float(np.mean(pris_seconds)) if pris_seconds else 0.0
        mean_pri_us = float(mean_pri_s * 1e6)
        mean_prf_hz = float(1.0 / mean_pri_s) if mean_pri_s > 0 else 0.0

    mean_pw_s = float(np.mean(pulse_widths)) if pulse_widths else 0.0

    # GEOMETRIC INVARIANT ENFORCEMENT: PW <= PRI always
    if mean_pri_s > 0:
        if mean_pw_s >= mean_pri_s and pris_seconds:
            mean_pri_s = float(np.mean(pris_seconds))
            mean_pri_us = float(mean_pri_s * 1e6)
            mean_prf_hz = float(1.0 / mean_pri_s) if mean_pri_s > 0 else mean_prf_hz
        mean_pw_s = min(mean_pw_s, mean_pri_s)
        duty_cycle = float((mean_pw_s / mean_pri_s) * 100.0) if mean_pri_s > 0 else 100.0
    else:
        duty_cycle = 100.0

    duty_cycle = min(100.0, max(0.0, duty_cycle))

    # CONTINUOUS TRANSMISSION GUARD:
    # If duty cycle >= 70% without chirp or standard TDMA frame, signal is continuous
    is_standard_tdma = (not intra_mod.get("is_fmop_chirp", False)) and matched_frame is not None and ("GSM" in matched_frame or "DMR" in matched_frame)
    if (duty_cycle >= 70.0 or (snr_data["pulsed_snr_db"] < 2.5 and not is_sparse)) and not intra_mod["is_fmop_chirp"] and not is_standard_tdma:
        return {
            "is_pulsed": False,
            "is_radar": False,
            "is_tdma": False,
            "num_pulses": 0,
            "mean_pulse_width_sec": 0.0,
            "mean_pulse_width_us": 0.0,
            "mean_pri_sec": 0.0,
            "mean_pri_us": 0.0,
            "mean_prf_hz": 0.0,
            "duty_cycle_pct": 100.0,
            "pulsed_snr_db": snr_data["pulsed_snr_db"],
            "multi_rate_prf": eac_prf_info,
            "intra_pulse_modulation": intra_mod,
            "signal_mode": "Continuous Transmission (Non-Pulsed)"
        }

    # Audio-Rate Ripple Guard:
    # At soundcard rates (fs <= 96 kHz), high-density envelope threshold crossings
    # with microsecond pulse durations (< 1500 us) reflect audio tone / subcarrier oscillation
    # or receiver noise, NOT pulsed radar or digital TDMA bursts.
    is_verified_chirp_radar = bool(
        intra_mod.get("is_fmop_chirp", False) and
        intra_mod.get("chirp_bandwidth_hz", 0.0) >= 75.0 and
        abs(intra_mod.get("chirp_rate_hz_per_sec", 0.0)) >= 10000.0 and
        (
            bool(matched_frame and "Radar" in matched_frame) or
            (
                mean_pw_s >= 100e-6 and
                duty_cycle < 30.0 and
                intra_mod.get("r2_goodness_of_fit", 0.0) >= 0.55 and
                snr_data.get("pulsed_snr_db", 0.0) >= 5.0
            )
        )
    )
    is_audio_ripple = (
        fs <= 96000.0 and
        not is_standard_tdma and
        not intra_mod.get("is_tdma_comms", False) and
        not is_verified_chirp_radar and
        (
            (mean_pw_s < 1500e-6 and len(pulse_widths) > 100) or
            (len(pulse_widths) > 500 and duty_cycle < 60.0)
        )
    )
    if is_audio_ripple:
        audio_intra = dict(intra_mod)
        audio_intra["is_fmop_chirp"] = False
        audio_intra["intra_pulse_mod"] = "Unmodulated Audio / Continuous Tone"
        return {
            "is_pulsed": False,
            "is_radar": False,
            "is_tdma": False,
            "num_pulses": 0,
            "mean_pulse_width_sec": 0.0,
            "mean_pulse_width_us": 0.0,
            "mean_pri_sec": 0.0,
            "mean_pri_us": 0.0,
            "mean_prf_hz": 0.0,
            "duty_cycle_pct": 100.0,
            "pulsed_snr_db": snr_data["pulsed_snr_db"],
            "multi_rate_prf": eac_prf_info,
            "intra_pulse_modulation": audio_intra,
            "signal_mode": "Continuous Transmission (Non-Pulsed)"
        }

    # Radar vs TDMA vs Packet Burst Discrimination:
    # 1. Radar: Strictly requires either a verified FMOP chirp
    #    OR a matched Radar Frame (OTH-SW, Ghadir, Duga)
    #    OR a short/sparse CW radar pulse train (PW <= 500 us, duty_cycle < 45%, snr >= 2.5 dB, num_pulses >= 2).
    is_sparse_radar = bool(
        not is_standard_tdma and
        is_sparse and
        snr_data["pulsed_snr_db"] >= 2.0 and
        (
            (intra_mod.get("is_fmop_chirp", False) and intra_mod.get("r2_goodness_of_fit", 0.0) >= 0.30) or
            (intra_mod.get("intra_pulse_mod") == "Pulsed CW (Unmodulated Carrier)" and mean_pw_s <= 0.010) or
            (len(pulse_widths) >= 2 and mean_pw_s <= 0.005)
        )
    )

    is_short_radar = bool(
        not is_standard_tdma and
        mean_pw_s <= 500e-6 and
        duty_cycle < 45.0 and
        snr_data["pulsed_snr_db"] >= 2.5 and
        len(pulse_widths) >= 2 and
        (fs > 100_000.0 or bool(matched_frame and "Radar" in matched_frame))
    )

    is_radar = (not is_standard_tdma) and (
        is_verified_chirp_radar or
        bool(matched_frame and "Radar" in matched_frame and intra_mod.get("intra_pulse_mod") == "Pulsed CW (Unmodulated Carrier)" and snr_data["pulsed_snr_db"] >= 3.0) or
        (intra_mod.get("is_fmop_chirp", False) and intra_mod.get("chirp_bandwidth_hz", 0.0) >= 75.0 and duty_cycle < 85.0 and snr_data["pulsed_snr_db"] >= 2.0 and intra_mod.get("r2_goodness_of_fit", 0.0) >= 0.35 and (bool(matched_frame and "Radar" in matched_frame) or mean_pw_s >= 800e-6)) or
        is_short_radar or
        is_sparse_radar
    )

    # 2. TDMA / Packet Burst Comms:
    is_tdma = (not is_radar) and (
        is_standard_tdma or
        (intra_mod.get("is_tdma_comms", False) and mean_pw_s >= 300e-6) or
        bool(matched_frame and ("TDMA" in matched_frame or "GSM" in matched_frame or "DMR" in matched_frame or "TETRA" in matched_frame))
    )

    if is_radar:
        signal_mode = "Pulsed Radar / OTH Intercept"
    elif is_standard_tdma:
        signal_mode = "TDMA Cellular / Digital Burst (GSM/DMR)"
    elif intra_mod.get("is_tdma_comms", False) and mean_pw_s >= 300e-6:
        signal_mode = "Packetized Digital Burst Signal"
    elif duty_cycle < 60.0 and len(pulse_widths) >= 1:
        signal_mode = "Packetized Burst Signal"
    else:
        signal_mode = "Continuous Transmission (Non-Pulsed)"

    return {
        "is_pulsed": True,
        "is_radar": is_radar,
        "is_tdma": is_tdma,
        "num_pulses": len(pulse_widths),
        "mean_pulse_width_sec": mean_pw_s,
        "mean_pulse_width_us": float(np.round(mean_pw_s * 1e6, 3)),
        "mean_pri_sec": mean_pri_s,
        "mean_pri_us": float(np.round(mean_pri_us, 3)),
        "mean_prf_hz": float(np.round(mean_prf_hz, 2)),
        "duty_cycle_pct": float(np.round(duty_cycle, 2)),
        "pulsed_snr_db": float(np.round(snr_data["pulsed_snr_db"], 2)),
        "multi_rate_prf": eac_prf_info,
        "intra_pulse_modulation": intra_mod,
        "signal_mode": signal_mode
    }
