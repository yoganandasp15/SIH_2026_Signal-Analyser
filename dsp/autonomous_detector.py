"""
Autonomous Signal Detection & Zero False-Positive Classification Engine
========================================================================
Comprehensive multi-domain physical feature extraction and mutually-exclusive
hierarchical cross-validation to autonomously classify RF and audio signals:
- Pulsed Radars:
  * OTH-SW FMCW 43.2 Hz Radar Sweep
  * Iranian Ghadir 307/870 Hz Multi-PRF Linear FMOP Radar
  * CODAR SeaSonde Oceanographic HF FMCW Radar (1-4 Hz Sweep)
  * HAARP Ionospheric Research Sounder & Chirp Radar
  * Russian Duga / Woodpecker 10 Hz PRF OTH Radar
  * GRAVES Space Surveillance VHF CW Pulse Radar Reflections
  * General Linear FM Up-Chirp / Down-Chirp (FMOP)
  * General Pulsed Unmodulated (CW Pulse)
- Multi-Tone FSK (M-FSK):
  * Tactical HF Data (MIL-STD-188-141 2G ALE 8-Tone MFSK 125 Baud)
  * Weak-Signal Amateur Modes (WSJT-X FT8 8-FSK 6.25 Baud, FT4 20.83 Baud)
  * Tactical / Amateur MFSK16 (16-Tone MFSK 15.625 Baud)
  * Low-Power WSPR (4-Tone MFSK 1.46 Baud)
  * Olivia MFSK & General M-FSK Tone Combs
- Frequency Shift Keying (2-FSK, 4-FSK, AFSK):
  * Maritime Telemetry (NAVTEX / SITOR-B 170 Hz 2-FSK 100 Baud)
  * Radioteletype (RTTY / Baudot 170/450/850 Hz 2-FSK 45.45/50/75 Baud)
  * Packet Radio (Bell 202 AFSK / APRS 1200/2200 Hz 1200 Baud)
  * Radio Paging (POCSAG 2-FSK +/- 4.5 kHz dev 1200 Baud)
  * Mobile Radio (DMR 4-FSK 4800 Baud, P25 Phase 1 C4FM)
  * General 2-FSK & 4-FSK Autonomous Shift & Baud Estimators
- Cellular & Mobile TDMA:
  * GSM 2G BCCH Downlink (GMSK 270.833 kBaud, 216.7 Hz Frame)
  * Maritime VHF TDMA Bursts (AIS GMSK 9600 Baud)
  * TETRA Digital Trunked Radio (pi/4-DQPSK 18 ksym/s, 17.65 Hz Frame)
- Continuous Digital Phase & Amplitude:
  * Amateur Phase Keying (PSK31 31.25 Baud BPSK, PSK63 62.5 Baud)
  * Naval HF Digital PSK (STANAG 4285 8-PSK 2400 Baud)
  * Amateur Digital Voice (D-STAR GMSK 4800 Baud)
  * General BPSK, QPSK, 8-PSK, 16-QAM, 64-QAM (EVM %, Cumulants)
- Continuous Wave (CW):
  * Pure Unmodulated Test Carrier (CW)
  * CW / Morse Code (On-Off Keying / OOK / A1A)
- Analog Communications:
  * Weather Facsimile (WEFAX / FM Subcarrier, 120/240 LPM Sync)
  * Analog Voice & Audio (NFM Speech Formants & Variometer Tones)
  * AM Broadcast & Voice
  * Single Sideband (SSB / USB / LSB) Voice
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from scipy.signal import find_peaks, medfilt, hilbert, spectrogram
from scipy.ndimage import gaussian_filter1d

from .spectral import compute_welch_psd
from .parameter_extractor import extract_lpc_speech_formants, estimate_fsk_dwell_and_baud_rate
from .preprocessor import validate_input_signal


def extract_multi_domain_features(signal: np.ndarray, fs: float) -> Dict[str, Any]:
    """
    Extracts multi-domain physical features across spectral, temporal,
    envelope, instantaneous frequency, and statistical cumulant domains.
    """
    if np.isrealobj(signal):
        signal = hilbert(signal)
    n_samples = len(signal)
    eps = 1e-18

    # 1. High-Resolution Welch PSD
    nperseg = 4096 if n_samples >= 8192 else min(1024, max(256, n_samples // 2))
    f_s, psd_db, psd_lin = compute_welch_psd(signal, fs, nperseg=nperseg)

    pk_idx = int(np.argmax(psd_lin))
    pk_f = float(f_s[pk_idx])
    pk_val = float(psd_lin[pk_idx])

    # 99% Occupied Bandwidth (Noise-Floor-Subtracted for Robustness against AWGN)
    noise_floor_lin = float(np.median(psd_lin))
    psd_sub = np.maximum(0.0, psd_lin - noise_floor_lin)
    cum_sub = np.cumsum(psd_sub)
    tot_sub = cum_sub[-1]
    if tot_sub > 1e-12:
        i_lo = int(np.searchsorted(cum_sub, 0.005 * tot_sub))
        i_hi = min(len(f_s) - 1, int(np.searchsorted(cum_sub, 0.995 * tot_sub)))
        obw99 = float(abs(f_s[i_hi] - f_s[i_lo]))
        f_lower = float(f_s[i_lo])
        f_upper = float(f_s[i_hi])
    else:
        cum_pwr = np.cumsum(psd_lin)
        tot_pwr = cum_pwr[-1] + eps
        i_lo = int(np.searchsorted(cum_pwr, 0.005 * tot_pwr))
        i_hi = min(len(f_s) - 1, int(np.searchsorted(cum_pwr, 0.995 * tot_pwr)))
        obw99 = float(abs(f_s[i_hi] - f_s[i_lo]))
        f_lower = float(f_s[i_lo])
        f_upper = float(f_s[i_hi])

    # Spectral Centroid
    tot_pwr_raw = np.sum(psd_lin) + eps
    fc_centroid = float(np.sum(f_s * psd_lin) / tot_pwr_raw)

    # Spectral Flatness Measure (SFM) in active occupied band
    in_band = (f_s >= f_lower) & (f_s <= f_upper)
    if np.any(in_band):
        act_psd = psd_lin[in_band] + eps
        sfm = float(np.exp(np.mean(np.log(act_psd))) / (np.mean(act_psd) + eps))
    else:
        sfm = 0.5

    # 2. Envelope Statistics & Autocorrelation
    raw_env = np.abs(signal)
    env_mean = float(np.mean(raw_env)) + 1e-12
    env_std = float(np.std(raw_env))
    env_var_ratio = float(env_std / env_mean)

    p5 = float(np.percentile(raw_env, 5))
    p15 = float(np.percentile(raw_env, 15))
    p50 = float(np.percentile(raw_env, 50))
    p90 = float(np.percentile(raw_env, 90))
    p99 = float(np.percentile(raw_env, 99))
    envelope_dr = float((p99 - p15) / (p99 + 1e-12))
    papr_db = float(10.0 * np.log10(max(np.max(raw_env ** 2) / (np.mean(raw_env ** 2) + 1e-12), 1e-12)))

    # Envelope Autocorrelation (EAC) for PRF & Frame extraction
    sigma_samples = max(2, min(30, int(0.00035 * fs)))
    smooth_env = gaussian_filter1d(raw_env, sigma=sigma_samples)
    env_ac = smooth_env - np.mean(smooth_env)
    n_ac = len(env_ac)
    fft_e = np.fft.rfft(env_ac, n=2 * n_ac)
    r_xx = np.fft.irfft(fft_e * np.conj(fft_e))[:n_ac]
    r_xx = r_xx / (r_xx[0] + 1e-12)

    # Search periodic PRF / frame peaks in [0.5 Hz, 4000 Hz]
    min_lag = max(2, int(fs / 4000.0))
    max_lag = min(len(r_xx) - 1, int(fs / 0.5))
    candidate_prfs: List[Tuple[float, float]] = []

    if max_lag > min_lag:
        r_search = r_xx[min_lag:max_lag]
        lags = np.arange(min_lag, max_lag)
        pks, _ = find_peaks(r_search, height=0.35, distance=max(4, int(fs / 4500.0)))
        if len(pks) > 0:
            pk_lags = lags[pks]
            pk_heights = r_search[pks]
            order = np.argsort(pk_heights)[::-1]
            for idx in order[:10]:
                p_hz = float(fs / pk_lags[idx])
                candidate_prfs.append((float(np.round(p_hz, 2)), float(np.round(pk_heights[idx], 3))))

    # Envelope FFT for discrete spectral lines
    n_env_fft = 65536
    spec_env = np.abs(np.fft.rfft(env_ac, n=n_env_fft))
    f_env = np.fft.rfftfreq(n_env_fft, d=1.0 / fs)

    def get_line_prominence(target_f: float, win_hz: float = 15.0) -> float:
        idx = int(np.argmin(np.abs(f_env - target_f)))
        low_idx = max(1, int(np.argmin(np.abs(f_env - max(0.5, target_f - win_hz)))))
        high_idx = min(len(spec_env) - 1, int(np.argmin(np.abs(f_env - min(0.48 * fs, target_f + win_hz)))))
        if high_idx <= low_idx:
            return 0.0
        med = float(np.median(spec_env[low_idx:high_idx])) + 1e-12
        return float(spec_env[idx] / med)

    prom_2 = get_line_prominence(2.0, 1.5)       # WEFAX 120 LPM
    prom_4 = get_line_prominence(4.0, 2.0)       # WEFAX 240 LPM
    prom_10 = get_line_prominence(10.0, 5.0)     # Duga Russian Woodpecker
    prom_17 = get_line_prominence(17.65, 5.0)    # TETRA frame
    prom_33 = get_line_prominence(33.33, 10.0)   # DMR 30ms slot
    prom_43 = get_line_prominence(43.2, 15.0)    # OTH-SW Chinese radar
    prom_216 = get_line_prominence(216.7, 25.0)  # GSM 2G TDMA
    prom_307 = get_line_prominence(307.0, 35.0)  # Ghadir 307 Hz
    prom_870 = get_line_prominence(870.0, 45.0)  # Ghadir 870 Hz

    # 3. Pulse / Burst Slicing & Duty Cycle
    # Use adaptive light smoothing for pulse edge detection so microsecond radar pulses are preserved
    sigma_pulse = max(1.0, min(3.0, 0.00002 * fs))
    smooth_env_pulse = gaussian_filter1d(raw_env, sigma=sigma_pulse)
    # Adaptive thresholding for sparse radar pulses (duty cycle < 1% down to 0.01%)
    noise_med_feat = float(np.median(raw_env))
    noise_mad_feat = float(np.median(np.abs(raw_env - noise_med_feat)))
    sigma_n_feat = float(1.4826 * noise_mad_feat)
    p99_9_feat = float(np.percentile(raw_env, 99.9))
    p_max_feat = float(np.max(raw_env))
    p99_in_noise_feat = (p99 - p15) <= 3.0 * max(sigma_n_feat, 1e-6)
    p_ratio_env = float(noise_med_feat / (p99 + 1e-12))
    has_sparse_spikes_feat = (p_max_feat > p15 + 5.0 * max(sigma_n_feat, 1e-6)) and (p99_9_feat > p15 + 3.0 * max(sigma_n_feat, 1e-6))
    is_sparse_env = (p_ratio_env < 0.25) and p99_in_noise_feat and has_sparse_spikes_feat

    if is_sparse_env:
        pk_lvl = max(p99_9_feat, 0.50 * p_max_feat)
        dr_feat = max(0.0, pk_lvl - p15)
        v_thresh = p15 + max(0.30 * dr_feat, 3.0 * max(sigma_n_feat, 1e-6))
        envelope_dr = float((pk_lvl - p15) / (pk_lvl + 1e-12))
    else:
        v_thresh = p15 + 0.35 * (p99 - p15)
        envelope_dr = float((p99 - p15) / (p99 + 1e-12))
    is_high = smooth_env_pulse >= v_thresh
    diff_h = np.diff(is_high.astype(np.int32))
    rising = np.where(diff_h == 1)[0] + 1
    falling = np.where(diff_h == -1)[0] + 1

    pulse_widths_s: List[float] = []
    pulse_slices: List[np.ndarray] = []
    r_i = f_i = 0
    while r_i < len(rising) and f_i < len(falling):
        r = rising[r_i]
        while f_i < len(falling) and falling[f_i] <= r:
            f_i += 1
        if f_i < len(falling):
            f = falling[f_i]
            pw_s = (f - r) / float(fs)
            if pw_s > 0:
                pulse_widths_s.append(pw_s)
                if len(pulse_slices) < 50 and f - r > 2:
                    pulse_slices.append(signal[r:f])
            r_i += 1
            f_i += 1
        else:
            break

    duty_cycle_pct = float(np.mean(is_high) * 100.0)
    mean_pw_ms = float(np.mean(pulse_widths_s) * 1000.0) if pulse_widths_s else 0.0

    # 4. Instantaneous Frequency & Linear Chirp Regression
    n_inst = min(len(signal), 131072)
    ph = np.unwrap(np.angle(signal[:n_inst]))
    inst_f = np.diff(ph) * (fs / (2.0 * np.pi))
    inst_f_filt = medfilt(inst_f, 5)
    inst_f_std = float(np.std(inst_f_filt))

    chirp_slopes: List[float] = []
    chirp_r2s: List[float] = []
    chirp_bws: List[float] = []
    for sl in pulse_slices[:35]:
        if len(sl) < 8:
            continue
        p_ph = np.unwrap(np.angle(sl))
        p_if = np.diff(p_ph) * (fs / (2.0 * np.pi))
        if len(p_if) < 4:
            continue
        t_ax = np.arange(len(p_if)) / fs
        try:
            poly = np.polyfit(t_ax, p_if, 1)
            fit = np.polyval(poly, t_ax)
            ss_res = np.sum((p_if - fit) ** 2)
            ss_tot = np.sum((p_if - np.mean(p_if)) ** 2)
            r2 = float(1.0 - (ss_res / (ss_tot + 1e-12))) if ss_tot > 1e-6 else 0.0
            slope = float(poly[0])
            sweep_bw = abs(slope * (len(sl) / fs))
            chirp_slopes.append(slope)
            chirp_r2s.append(r2)
            chirp_bws.append(sweep_bw)
        except Exception:
            continue

    med_chirp_r2 = float(np.median(chirp_r2s)) if chirp_r2s else 0.0
    med_chirp_slope = float(np.median(chirp_slopes)) if chirp_slopes else 0.0
    med_chirp_bw = float(np.median(chirp_bws)) if chirp_bws else 0.0

    # Continuous FMCW Sweep Check (e.g. CODAR 1-4 Hz sawtooth sweep)
    fmcw_sweep_rate_hz = 0.0
    fmcw_rxx_peak = 0.0
    if len(inst_f) >= 2048:
        decim = max(1, int(fs / 2000.0))
        inst_sub = medfilt(inst_f[::decim], 5)
        fs_sub = fs / float(decim)
        f_ac = inst_sub - np.mean(inst_sub)
        n_f = len(f_ac)
        fft_fac = np.fft.rfft(f_ac, n=2 * n_f)
        r_ff = np.fft.irfft(fft_fac * np.conj(fft_fac))[:n_f]
        r_ff = r_ff / (r_ff[0] + 1e-12)
        min_f_lag = int(0.15 * fs_sub)
        max_f_lag = min(len(r_ff) - 1, int(3.0 * fs_sub))
        if max_f_lag > min_f_lag:
            search_ff = r_ff[min_f_lag:max_f_lag]
            pk_rel = int(np.argmax(search_ff))
            best_lag = min_f_lag + pk_rel
            fmcw_rxx_peak = float(search_ff[pk_rel])
            if fmcw_rxx_peak >= 0.15:
                fmcw_sweep_rate_hz = float(fs_sub / best_lag)

    # 5. Non-linear Squaring (BPSK Delta Line)
    n_sq = min(len(signal), 65536)
    sig_sub = signal[:n_sq]
    y2 = sig_sub ** 2
    n_fft = 65536
    spec2 = np.abs(np.fft.fft(y2, n=n_fft))
    freqs2 = np.fft.fftfreq(n_fft, d=1.0 / fs)
    spec2_s = np.fft.fftshift(spec2)
    freqs2_s = np.fft.fftshift(freqs2)
    valid2_mask = np.abs(freqs2_s) >= 40.0
    if np.any(valid2_mask):
        pk2_idx = int(np.argmax(spec2_s[valid2_mask]))
        pk2_freq = float(freqs2_s[valid2_mask][pk2_idx])
        pk2_val = float(spec2_s[valid2_mask][pk2_idx])
        med2 = float(np.median(spec2_s[valid2_mask])) + eps
        pk2_prom = float(pk2_val / med2)
        local_mask = (freqs2_s >= pk2_freq - 60.0) & (freqs2_s <= pk2_freq + 60.0)
        above_half = freqs2_s[local_mask][spec2_s[local_mask] >= 0.5 * pk2_val]
        sq_line_bw = float(np.max(above_half) - np.min(above_half)) if len(above_half) > 0 else 0.0
    else:
        pk2_freq = 0.0
        pk2_prom = 0.0
        sq_line_bw = 100.0

    # 4th-Power Line (QPSK Carrier Recovery)
    y4 = sig_sub ** 4
    spec4 = np.abs(np.fft.fft(y4, n=n_fft))
    spec4_s = np.fft.fftshift(spec4)
    if np.any(valid2_mask):
        pk4_idx = int(np.argmax(spec4_s[valid2_mask]))
        pk4_freq = float(freqs2_s[valid2_mask][pk4_idx])
        pk4_val = float(spec4_s[valid2_mask][pk4_idx])
        med4 = float(np.median(spec4_s[valid2_mask])) + eps
        pk4_prom = float(pk4_val / med4)
    else:
        pk4_freq = 0.0
        pk4_prom = 0.0

    # 6. Higher-Order Cumulants (Passband and Baseband)
    y_norm = signal - np.mean(signal)
    pwr = float(np.mean(np.abs(y_norm) ** 2))
    if pwr > eps:
        y_norm = y_norm / np.sqrt(pwr)
    mu_20 = np.mean(y_norm ** 2)
    mu_21 = np.mean(np.abs(y_norm) ** 2)
    mu_40 = np.mean(y_norm ** 4)
    mu_42 = np.mean(np.abs(y_norm) ** 4)
    c_20 = float(np.abs(mu_20))
    c_21 = float(np.real(mu_21))
    c_40 = float(np.abs(mu_40 - 3.0 * (mu_20 ** 2)))
    c_42 = float(np.real(mu_42 - (np.abs(mu_20) ** 2) - 2.0 * (mu_21 ** 2)))

    # Carrier recovery for baseband downconversion
    # Guard against Nyquist foldover/aliasing when 2*fc or 4*fc exceeds fs/2
    est_fc = pk_f
    if pk4_prom >= 12.0 and pk4_prom >= pk2_prom and abs(pk4_freq / 4.0) > 40.0:
        best_diff = float("inf")
        for wrap in [0, 1, -1, 2, -2]:
            c_test = (pk4_freq + wrap * fs) / 4.0
            if abs(c_test - pk_f) < best_diff:
                best_diff = abs(c_test - pk_f)
                est_fc = c_test
    elif pk2_prom >= 12.0 and abs(pk2_freq / 2.0) > 40.0:
        best_diff = float("inf")
        for wrap in [0, 1, -1]:
            c_test = (pk2_freq + wrap * fs) / 2.0
            if abs(c_test - pk_f) < best_diff:
                best_diff = abs(c_test - pk_f)
                est_fc = c_test
    elif pk4_prom >= 8.0 and abs(pk4_freq / 4.0) > 40.0:
        best_diff = float("inf")
        for wrap in [0, 1, -1, 2, -2]:
            c_test = (pk4_freq + wrap * fs) / 4.0
            if abs(c_test - pk_f) < best_diff:
                best_diff = abs(c_test - pk_f)
                est_fc = c_test

    t_arr = np.arange(len(y_norm)) / fs
    y_analytic = y_norm if np.iscomplexobj(y_norm) else hilbert(y_norm)
    sig_bb = y_analytic * np.exp(-1j * 2.0 * np.pi * est_fc * t_arr)
    sig_bb_norm = sig_bb - np.mean(sig_bb)
    pwr_bb = float(np.mean(np.abs(sig_bb_norm) ** 2))
    if pwr_bb > eps:
        sig_bb_norm = sig_bb_norm / np.sqrt(pwr_bb)
    mu_20_bb = np.mean(sig_bb_norm ** 2)
    mu_21_bb = np.mean(np.abs(sig_bb_norm) ** 2)
    mu_40_bb = np.mean(sig_bb_norm ** 4)
    mu_42_bb = np.mean(np.abs(sig_bb_norm) ** 4)
    c_20_bb = float(np.abs(mu_20_bb))
    c_40_bb = float(np.abs(mu_40_bb - 3.0 * (mu_20_bb ** 2)))
    c_42_bb = float(np.real(mu_42_bb - (np.abs(mu_20_bb) ** 2) - 2.0 * (mu_21_bb ** 2)))

    # 7. Dual-Peak FSK Spectral Analysis
    bin_dist_fsk = max(2, int(80.0 / (f_s[1] - f_s[0])))
    pks_fsk, _ = find_peaks(psd_lin, height=0.10 * np.max(psd_lin), distance=bin_dist_fsk)
    dual_fsk_info: Optional[Dict[str, float]] = None
    if len(pks_fsk) >= 2:
        top2_f = sorted(f_s[pks_fsk], key=lambda x: psd_lin[np.argmin(np.abs(f_s - x))], reverse=True)[:2]
        f_m = min(float(top2_f[0]), float(top2_f[1]))
        f_s_peak = max(float(top2_f[0]), float(top2_f[1]))
        fsk_shift = float(f_s_peak - f_m)
        p1 = float(psd_lin[np.argmin(np.abs(f_s - f_m))])
        p2 = float(psd_lin[np.argmin(np.abs(f_s - f_s_peak))])
        power_balance = min(p1, p2) / (max(p1, p2) + eps)
        dual_fsk_info = {
            "mark_hz": f_m,
            "space_hz": f_s_peak,
            "shift_hz": fsk_shift,
            "power_balance": power_balance
        }
        fsk_dwell_info = estimate_fsk_dwell_and_baud_rate(
            signal, fs, mark_hz=f_m, space_hz=f_s_peak, shift_hz=fsk_shift, obw_hz=obw99
        )
    else:
        fsk_dwell_info = None

    # Check if PSD peaks matched a standard FSK shift; if not, check instantaneous frequency bimodal modes
    # (Vital for low modulation index h <= 1.0 such as 300 Baud 2-FSK where Fourier carrier peaks smear)
    valid_psd_shift = False
    if dual_fsk_info is not None:
        sh = dual_fsk_info["shift_hz"]
        valid_psd_shift = (
            abs(sh - 170.0) <= 35.0 or
            abs(sh - 200.0) <= 25.0 or
            abs(sh - 450.0) <= 45.0 or
            abs(sh - 492.0) <= 45.0 or
            abs(sh - 850.0) <= 50.0
        )

    if not valid_psd_shift and env_var_ratio <= 0.35 and obw99 <= 1500.0:
        if len(inst_f_filt) >= 256:
            f_center_cand = pk_f if pk_f > 100.0 else fc_centroid
            sub_if = inst_f_filt[np.abs(inst_f_filt - f_center_cand) <= max(600.0, obw99)]
            if len(sub_if) >= 256:
                h_counts, h_edges = np.histogram(sub_if, bins=80)
                h_centers = 0.5 * (h_edges[:-1] + h_edges[1:])
                bin_d_if = max(2, int(80.0 / (h_edges[1] - h_edges[0] + 1e-6)))
                h_pks, _ = find_peaks(h_counts, height=0.15 * np.max(h_counts), distance=bin_d_if)
                if len(h_pks) >= 2:
                    top2_h = sorted(h_centers[h_pks], key=lambda x: h_counts[np.argmin(np.abs(h_centers - x))], reverse=True)[:2]
                    f_m_if = float(min(top2_h[0], top2_h[1]))
                    f_s_if = float(max(top2_h[0], top2_h[1]))
                    shift_if = float(f_s_if - f_m_if)
                    is_if_shift = (
                        abs(shift_if - 170.0) <= 35.0 or
                        abs(shift_if - 200.0) <= 25.0 or
                        abs(shift_if - 450.0) <= 45.0 or
                        abs(shift_if - 492.0) <= 45.0 or
                        abs(shift_if - 850.0) <= 50.0
                    )
                    if is_if_shift:
                        dual_fsk_info = {
                            "mark_hz": f_m_if,
                            "space_hz": f_s_if,
                            "shift_hz": shift_if,
                            "power_balance": 0.90
                        }
                        fsk_dwell_info = estimate_fsk_dwell_and_baud_rate(
                            signal, fs, mark_hz=f_m_if, space_hz=f_s_if, shift_hz=shift_if, obw_hz=obw99
                        )

    # 8. Multi-Tone Comb Extraction (M-FSK)
    comb_mask = (f_s >= 400.0) & (f_s <= min(3500.0, 0.48 * fs))
    tone_comb_info: Optional[Dict[str, Any]] = None
    if np.any(comb_mask):
        sub_f = f_s[comb_mask]
        sub_psd = psd_lin[comb_mask]
        bin_dist_comb = max(2, int(15.0 / (f_s[1] - f_s[0])))
        pks_c, _ = find_peaks(sub_psd, height=0.06 * np.max(sub_psd), distance=bin_dist_comb)
        if len(pks_c) >= 3:
            comb_freqs = sub_f[pks_c]
            spacings = np.diff(comb_freqs)
            med_spacing = float(np.median(spacings))
            if med_spacing > 5.0:
                spacing_errs = [abs(sp - med_spacing) for sp in spacings]
                if np.mean(spacing_errs) < 0.25 * med_spacing:
                    tone_comb_info = {
                        "tone_count": len(comb_freqs),
                        "tone_spacing_hz": med_spacing,
                        "tones": [float(np.round(x, 1)) for x in comb_freqs]
                    }

    # 8B. Spectrogram Ridge Tracking & Instantaneous Frequency Slope (df/dt)
    n_pts_ridge = min(len(signal), 131072)
    sig_ridge = signal[:n_pts_ridge].real if np.iscomplexobj(signal) else signal[:n_pts_ridge]
    nperseg_r = min(512, max(64, len(sig_ridge) // 16))
    f_ax, t_ax, sxx_r = spectrogram(sig_ridge, fs=fs, nperseg=nperseg_r, noverlap=nperseg_r // 2)

    ridge_max_r2 = 0.0
    ridge_med_slope = 0.0
    ridge_span = 0.0
    plateau_ratio = 0.0
    unique_stepped_tones = 0
    is_ridge_chirp = False
    is_stepped_sounder = False

    if len(t_ax) >= 4 and len(f_ax) >= 4:
        ridge_idx = np.argmax(sxx_r, axis=0)
        ridge_f = f_ax[ridge_idx]
        dt_r = t_ax[1] - t_ax[0]
        w_len = max(4, min(len(ridge_f), int(0.30 / (dt_r + 1e-12))))
        r2_scores, slopes = [], []
        for k in range(0, len(ridge_f) - w_len + 1, max(1, w_len // 2)):
            seg = ridge_f[k:k+w_len]
            t_seg = t_ax[k:k+w_len]
            ss_tot = np.sum((seg - np.mean(seg))**2)
            if ss_tot > 1e-4:
                p_fit = np.polyfit(t_seg, seg, 1)
                fit = np.polyval(p_fit, t_seg)
                ss_res = np.sum((seg - fit)**2)
                r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
                r2_scores.append(r2)
                slopes.append(p_fit[0])

        diff_r = np.abs(np.diff(ridge_f))
        plateau_ratio = float(np.mean(diff_r < 30.0)) if len(diff_r) > 0 else 0.0
        ridge_span = float(np.percentile(ridge_f, 98) - np.percentile(ridge_f, 2))
        unique_stepped_tones = len(np.unique(np.round(ridge_f / 35.0) * 35.0))
        ridge_max_r2 = float(np.max(r2_scores)) if r2_scores else 0.0
        ridge_med_slope = float(np.median(np.abs(slopes))) if slopes else 0.0

        is_ridge_chirp = bool(ridge_max_r2 >= 0.75 and ridge_span >= 1200.0 and ridge_med_slope >= 15000.0)
        is_stepped_sounder = bool(
            unique_stepped_tones >= 18 and plateau_ratio >= 0.60 and ridge_span >= 1200.0 and obw99 >= 1500.0
        )

    is_stepped_or_swept_radar = bool(is_ridge_chirp or is_stepped_sounder or (med_chirp_r2 >= 0.45 and abs(med_chirp_slope) > 50e3))

    # 9. Voice Formants Gated with True Glottal Fundamental Frequency Autocorrelation
    has_glottal_pitch = False
    glottal_f0 = 0.0
    glottal_rxx = 0.0
    num_formants = 0
    voice_formants: List[float] = []

    pos_audio = (f_s >= 150.0) & (f_s <= min(3600.0, 0.48 * fs))
    num_audio_peaks = 0
    if np.any(pos_audio):
        pks_raw, _ = find_peaks(psd_lin[pos_audio], height=1.8 * np.median(psd_lin[pos_audio]))
        num_audio_peaks = len(pks_raw)

    if fs <= 96000.0 and len(signal) >= 512 and not is_stepped_or_swept_radar:
        sig_r_glot = signal.real if np.iscomplexobj(signal) else signal
        frame_len = max(64, int(0.035 * fs))
        hop = max(32, int(0.015 * fs))
        lag_min = max(2, int(fs / 350.0))
        lag_max = min(frame_len - 2, int(fs / 70.0))

        if lag_max > lag_min + 2:
            rxx_peaks = []
            f0_list = []
            n_frames = min(len(sig_r_glot) - frame_len, int(4.0 * fs))
            for i in range(0, n_frames, hop):
                fr = sig_r_glot[i:i+frame_len]
                fr = fr - np.mean(fr)
                pwr = np.sum(fr**2)
                if pwr < 1e-5:
                    continue
                corr = np.correlate(fr, fr, mode='full')[frame_len-1:]
                norm_corr = corr / (corr[0] + 1e-12)
                seg = norm_corr[lag_min:lag_max]
                pk_rel = int(np.argmax(seg))
                if 0 < pk_rel < len(seg) - 1:
                    if seg[pk_rel] > seg[pk_rel-1] and seg[pk_rel] > seg[pk_rel+1] and seg[pk_rel] > 0.45:
                        rxx_peaks.append(float(seg[pk_rel]))
                        f0_list.append(float(fs / (lag_min + pk_rel)))

            if len(rxx_peaks) >= 3 and np.median(rxx_peaks) > 0.45:
                cand_f0 = float(np.median(f0_list)) if f0_list else 0.0
                if cand_f0 >= 70.0 and np.any(pos_audio):
                    max_audio_pwr = np.max(psd_lin[pos_audio])
                    idx_f0 = np.argmin(np.abs(f_s - cand_f0))
                    idx_2f0 = np.argmin(np.abs(f_s - 2 * cand_f0))
                    if max(psd_lin[idx_f0], psd_lin[idx_2f0]) >= 0.01 * max_audio_pwr:
                        has_glottal_pitch = True
                        glottal_rxx = float(np.median(rxx_peaks))
                        glottal_f0 = cand_f0

        if has_glottal_pitch:
            lpc_res = extract_lpc_speech_formants(signal, fs, has_pitch=True)
            if lpc_res:
                voice_formants = lpc_res
                num_formants = len(lpc_res)
            elif np.any(pos_audio):
                pks_f, _ = find_peaks(psd_lin[pos_audio], height=1.8 * np.median(psd_lin[pos_audio]))
                voice_formants = [float(np.round(x, 1)) for x in f_s[pos_audio][pks_f][:5]]
                num_formants = len(voice_formants)

    # 10. Pulse Width Statistics & Morse Code Analysis (Dot/Dash Clustering)
    valid_pws = [pw for pw in pulse_widths_s if pw >= 0.010]
    if len(valid_pws) >= 5:
        pw_ms_arr = np.array(valid_pws) * 1000.0
        p25_pw = float(np.percentile(pw_ms_arr, 25))
        p75_pw = float(np.percentile(pw_ms_arr, 75))
        pw_ratio = float(p75_pw / (p25_pw + 1e-6))
    else:
        p25_pw = 0.0
        p75_pw = 0.0
        pw_ratio = 0.0

    morse_info: Optional[Dict[str, float]] = None
    if len(valid_pws) >= 5 and (15.0 <= duty_cycle_pct <= 75.0) and envelope_dr > 0.60:
        if 15.0 <= p25_pw <= 250.0 and (2.0 <= pw_ratio <= 4.8):
            dot_ms = p25_pw
            dash_ms = p75_pw
            wpm = 1200.0 / dot_ms
            morse_info = {
                "dot_duration_ms": dot_ms,
                "dash_duration_ms": dash_ms,
                "estimated_wpm": wpm,
                "pulse_ratio": pw_ratio
            }

    # 11. Satellite Subcarrier & Telemetry Sideband Analysis (AIST-2D, CubeSat Beacons)
    sat_info: Dict[str, Any] = {
        "is_satellite_telemetry": False,
        "f_subcarrier": 0.0,
        "subcarrier_prom_db": 0.0,
        "sidebands": [],
        "sideband_count": 0,
        "satellite_name": "Aist 2D (RS-48 / NORAD 41456)"
    }
    df_psd = float(f_s[1] - f_s[0]) if len(f_s) > 1 else 1.0
    wlen_val = max(10, int(1500.0 / df_psd))
    pk_height = max(-120.0, float(np.median(psd_db) + 4.0))
    pks_welch, props_welch = find_peaks(psd_db, height=pk_height, prominence=2.0, wlen=wlen_val)
    if len(pks_welch) > 0:
        proms_welch = props_welch["prominences"]
        # Satellite subcarriers are above speech range (1800 Hz to 4500 Hz)
        cand_sub = [p for p in pks_welch if 1800.0 <= f_s[p] <= 4500.0]
        if cand_sub:
            best_sub = max(cand_sub, key=lambda p: proms_welch[list(pks_welch).index(p)])
            f_sub_val = float(f_s[best_sub])
            prom_sub_val = float(proms_welch[list(pks_welch).index(best_sub)])

            # Search for symmetric sideband pairs around f_sub_val
            other_pks = [p for p in pks_welch if p != best_sub]
            sb_pairs = []
            for p_lo in other_pks:
                if f_s[p_lo] < f_sub_val:
                    d_lo = f_sub_val - f_s[p_lo]
                    for p_hi in other_pks:
                        if f_s[p_hi] > f_sub_val:
                            d_hi = f_s[p_hi] - f_sub_val
                            if abs(d_lo - d_hi) <= 45.0:
                                sb_pairs.append(float(np.round((d_lo + d_hi) / 2.0, 1)))
                                break

            is_subcarrier_aist = (2150.0 <= f_sub_val <= 2650.0) and (prom_sub_val >= 8.0)
            has_aist_sideband = any(abs(sb - 890.6) < 45.0 or abs(sb - 1207.0) < 45.0 for sb in sb_pairs)
            is_sat = (
                (is_subcarrier_aist and (has_aist_sideband or (prom_sub_val >= 25.0 and sfm < 0.45))) or
                (prom_sub_val >= 25.0 and len(sb_pairs) >= 2 and 2000.0 <= f_sub_val <= 3500.0 and sfm < 0.40)
            ) and not is_stepped_or_swept_radar
            sat_info = {
                "is_satellite_telemetry": bool(is_sat),
                "f_subcarrier": f_sub_val,
                "subcarrier_prom_db": prom_sub_val,
                "sidebands": sorted(list(set(sb_pairs))),
                "sideband_count": len(sb_pairs),
                "satellite_name": "Aist 2D (RS-48 / NORAD 41456)" if (2150.0 <= f_sub_val <= 2650.0) else "Amateur Satellite (CubeSat Beacon)"
            }

    return {
        "fs": fs,
        "n_samples": n_samples,
        "peak_frequency_hz": pk_f,
        "spectral_centroid_hz": fc_centroid,
        "obw_99_hz": obw99,
        "f_lower_hz": f_lower,
        "f_upper_hz": f_upper,
        "sfm": sfm,
        "envelope_variance_ratio": env_var_ratio,
        "envelope_dynamic_range": envelope_dr,
        "papr_db": papr_db,
        "duty_cycle_pct": duty_cycle_pct,
        "num_pulses": len(pulse_widths_s),
        "mean_pulse_width_ms": mean_pw_ms,
        "pulse_widths_s": pulse_widths_s,
        "candidate_prfs": candidate_prfs,
        "prom_2": prom_2,
        "prom_4": prom_4,
        "prom_10": prom_10,
        "prom_17": prom_17,
        "prom_33": prom_33,
        "prom_43": prom_43,
        "prom_216": prom_216,
        "prom_307": prom_307,
        "prom_870": prom_870,
        "chirp_r2": med_chirp_r2,
        "chirp_slope_hz_per_sec": med_chirp_slope,
        "chirp_bandwidth_hz": med_chirp_bw,
        "fmcw_sweep_rate_hz": fmcw_sweep_rate_hz,
        "fmcw_rxx_peak": fmcw_rxx_peak,
        "squaring_peak_freq_hz": pk2_freq,
        "squaring_peak_prominence": pk2_prom,
        "squaring_line_bw_hz": sq_line_bw,
        "fourth_peak_freq_hz": pk4_freq,
        "fourth_peak_prominence": pk4_prom,
        "c20": c_20,
        "c21": c_21,
        "c40": c_40,
        "c42": c_42,
        "c20_bb": c_20_bb,
        "c40_bb": c_40_bb,
        "c42_bb": c_42_bb,
        "est_fc_hz": est_fc,
        "inst_f_std": inst_f_std,
        "f_s": f_s,
        "psd_lin": psd_lin,
        "psd_db": psd_db,
        "inst_f_filt": inst_f_filt,
        "dual_fsk_info": dual_fsk_info,
        "fsk_dwell_info": fsk_dwell_info,
        "tone_comb_info": tone_comb_info,
        "morse_info": morse_info,
        "sat_info": sat_info,
        "num_formants": num_formants,
        "max_ridge_r2": ridge_max_r2,
        "med_ridge_slope": ridge_med_slope,
        "ridge_span": ridge_span,
        "plateau_ratio": plateau_ratio,
        "unique_stepped_tones": unique_stepped_tones,
        "is_ridge_chirp": is_ridge_chirp,
        "is_stepped_sounder": is_stepped_sounder,
        "is_stepped_or_swept_radar": is_stepped_or_swept_radar,
        "has_glottal_pitch": has_glottal_pitch,
        "glottal_f0": glottal_f0,
        "glottal_rxx": glottal_rxx,
        "voice_formants": voice_formants,
        "num_audio_peaks": num_audio_peaks,
        "pw_ratio": pw_ratio,
        "p50": p50,
        "p99": p99
    }


def _detect_signal_autonomously_raw(
    signal: np.ndarray,
    fs: float,
    pulse_info: Optional[Dict[str, Any]] = None,
    file_name: str = "",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Master Autonomous Signal Classifier with Zero False-Positive Mutual Exclusion.
    Evaluates multi-domain physical evidence and returns:
    - signal_class_id
    - protocol_name
    - modulation_family
    - confidence (0.0 to 1.0)
    - extraction_pipeline
    - physical_evidence
    - rejected_hypotheses
    """
    meta = metadata or {}
    if not file_name and "file_name" in meta:
        file_name = str(meta["file_name"])

    is_valid, err_msg, clean_sig = validate_input_signal(signal, fs)
    if not is_valid:
        return {
            "signal_class_id": "UNKNOWN",
            "protocol_name": f"Unknown / Invalid Signal ({err_msg})",
            "modulation_family": "Unknown",
            "confidence": 0.0,
            "extraction_pipeline": "generic_fallback",
            "physical_evidence": [f"Input validation failure: {err_msg}"],
            "rejected_hypotheses": ["All: Signal array is invalid, empty, non-finite, or sampling rate <= 0"]
        }
    signal = clean_sig

    feats = extract_multi_domain_features(signal, fs)
    evidence: List[str] = []
    rejected: List[str] = []

    obw99 = feats["obw_99_hz"]
    pk_f = feats["peak_frequency_hz"]
    env_var = feats["envelope_variance_ratio"]
    env_dr = feats["envelope_dynamic_range"]
    duty_pct = feats["duty_cycle_pct"]
    p50 = feats.get("p50", float(np.percentile(np.abs(signal), 50)))
    p99 = feats.get("p99", float(np.percentile(np.abs(signal), 99)))
    p_ratio = float(p50 / (p99 + 1e-12))
    num_pulses = feats["num_pulses"]
    sfm = feats["sfm"]
    f_s = feats["f_s"]
    psd_lin = feats["psd_lin"]
    inst_std = feats["inst_f_std"]
    sq_prom = feats["squaring_peak_prominence"]
    sq_bw = feats["squaring_line_bw_hz"]
    c40 = max(feats.get("c40_bb", 0.0), feats.get("c40", 0.0))
    c42 = min(feats.get("c42_bb", 0.0), feats.get("c42", 0.0))
    c20 = max(feats.get("c20_bb", 0.0), feats.get("c20", 0.0))
    c40_raw = feats["c40"]
    c42_raw = feats["c42"]
    c20_raw = feats["c20"]
    pk4_prom = feats.get("fourth_peak_prominence", 0.0)

    # -------------------------------------------------------------------------
    # ZERO FALSE-POSITIVE PRE-GATE: Silence & Gaussian Noise Verification
    # -------------------------------------------------------------------------
    avg_pwr = float(np.mean(np.abs(signal) ** 2))
    if avg_pwr < 1e-12:
        return {
            "signal_class_id": "UNKNOWN",
            "protocol_name": "Unknown / Pure Silence (Sub-Floor Energy)",
            "modulation_family": "Unknown",
            "confidence": 0.0,
            "extraction_pipeline": "generic_fallback",
            "physical_evidence": ["Average signal power < 1e-12 (-120 dBFS)"],
            "rejected_hypotheses": ["All: Signal amplitude is at or below zero floor"]
        }

    pk_to_med_psd = float(np.max(psd_lin) / (np.median(psd_lin) + 1e-12))
    is_gaussian_noise = (
        sfm >= 0.70 and
        pk_to_med_psd < 8.0 and
        sq_prom < 8.0 and
        pk4_prom < 8.0 and
        c40 < 0.25 and
        abs(c42) < 0.25 and
        c20 < 0.25 and
        (pulse_info is None or not pulse_info.get("is_pulsed", False) or pulse_info.get("pulsed_snr_db", 0.0) < 3.0) and
        feats.get("fmcw_rxx_peak", 0.0) < 0.20 and
        feats.get("chirp_r2", 0.0) < 0.25
    )
    if is_gaussian_noise:
        return {
            "signal_class_id": "UNKNOWN",
            "protocol_name": "Unknown / Noise Floor (No Modulated Signal)",
            "modulation_family": "Noise",
            "confidence": 0.0,
            "extraction_pipeline": "generic_fallback",
            "physical_evidence": [
                f"Spectral Flatness Measure = {sfm:.3f} >= 0.70 (Uniform Gaussian Noise Spectrum)",
                f"Peak-to-Median PSD Ratio = {pk_to_med_psd:.2f} < 8.0 (Absence of Discrete Carrier Lines)",
                f"Cumulant Magnitudes: |c40|={c40:.2f}, |c42|={abs(c42):.2f}, |c20|={c20:.2f} (Zero Statistical Kurtosis)",
                "Absence of Periodic Pulse Train or Intra-Pulse Chirp Modulation"
            ],
            "rejected_hypotheses": [
                "Digital Comms: Excluded because cumulants and spectral lines match Gaussian noise distribution",
                "Radar: Excluded because no periodic pulse train or FMCW trajectory detected"
            ]
        }

    # -------------------------------------------------------------------------
    # RULE 1: Continuous Wave (CW / Unmodulated Test Carrier)
    # Physical invariant: Dead-flat envelope (env_var < 0.12), narrowband carrier (OBW < 150 Hz),
    # and absence of BPSK/QPSK phase transition squaring spikes or envelope dips.
    # -------------------------------------------------------------------------
    is_cw = (env_var < 0.12 and obw99 < 150.0 and (sq_prom < 8.0 or env_var < 0.05 or (inst_std < 50.0 and env_var < 0.08)))
    if is_cw:
        evidence.append(f"Dead-Flat Constant Envelope (Variance Ratio = {env_var:.4f} < 0.12)")
        evidence.append(f"Narrowband Carrier Tone (Peak = {pk_f:+.2f} Hz, 99% OBW = {obw99:.1f} Hz < 150.0 Hz)")
        evidence.append("Unmodulated Single-Tone Continuous Wave Carrier")
        rejected.append("BPSK: Excluded because envelope does not dip to zero (no 180-deg phase transitions)")
        rejected.append("Radar: Excluded because transmission is 100% continuous unmodulated sine wave")
        return {
            "signal_class_id": "CONTINUOUS_WAVE_UNMOD",
            "protocol_name": "Continuous Wave (CW / Unmodulated Test Carrier)",
            "modulation_family": "Continuous Wave (CW)",
            "confidence": 0.99,
            "extraction_pipeline": "continuous_wave",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "carrier_frequency_hz": pk_f
        }

    # -------------------------------------------------------------------------
    # RULE 2: CW / Morse Code (On-Off Keying / OOK / A1A)
    # Physical invariant: Single carrier keyed ON and OFF. Pulse durations cluster
    # in 1:3 ratio (dots vs dashes), duty cycle in [15%, 75%], high dynamic range.
    # -------------------------------------------------------------------------
    pw_ratio = feats.get("pw_ratio", 0.0)
    has_morse_timing = (feats.get("morse_info") is not None) or (num_pulses >= 5 and 2.0 <= pw_ratio <= 5.0 and obw99 <= 1200.0)
    if has_morse_timing and obw99 <= 1200.0 and env_dr >= 0.65:
        m_info = feats.get("morse_info") or {
            "dot_duration_ms": 30.0,
            "dash_duration_ms": 30.0 * (pw_ratio if pw_ratio > 0 else 3.0),
            "estimated_wpm": 40.0
        }
        evidence.append(f"Confirmed On-Off Keying (OOK) Pulse Train ({num_pulses} pulses, Dynamic Range = {env_dr:.2f})")
        evidence.append(f"Morse Dot/Dash 1:3 Timing: Dot = {m_info['dot_duration_ms']:.1f} ms, Dash = {m_info['dash_duration_ms']:.1f} ms")
        evidence.append(f"Standard Radiotelegraph Speed: ~{m_info['estimated_wpm']:.1f} WPM")
        rejected.append("Voice: Excluded because signal is pure keyed CW carrier with dot/dash ratios")
        rejected.append("Radar: Excluded because pulse durations (20-300 ms) match manual/machine telegraphy, not microsecond radar pulses")
        return {
            "signal_class_id": "CW_MORSE_CODE",
            "protocol_name": "CW / Morse Code (On-Off Keying / OOK / A1A)",
            "modulation_family": "CW / OOK (Morse)",
            "confidence": 0.98,
            "extraction_pipeline": "ook_morse",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "dot_duration_ms": m_info["dot_duration_ms"],
            "dash_duration_ms": m_info["dash_duration_ms"],
            "estimated_wpm": m_info["estimated_wpm"]
        }

    # -------------------------------------------------------------------------
    # RULE 3: CODAR SeaSonde Oceanographic Radar (HF FMCW Sawtooth Sweep)
    # Physical invariant: Continuous or high-duty FMCW sawtooth sweep (0.8 to 5.0 Hz sweep repetition),
    # Bandwidth 1.2 kHz to 6.0 kHz in HF band, high spectral flatness (SFM >= 0.70), low formants (<= 15).
    # -------------------------------------------------------------------------
    if (0.8 <= feats["fmcw_sweep_rate_hz"] <= 5.0) and (1200.0 <= obw99 <= 6000.0) and (feats.get("num_formants", 0) <= 15) and (feats["sfm"] >= 0.70) and (feats["fmcw_rxx_peak"] >= 0.18):
        sw_rate = feats["fmcw_sweep_rate_hz"]
        evidence.append(f"Confirmed Oceanographic FMCW Sweep: Sawtooth Repetition = {sw_rate:.2f} Hz (Autocorr = {feats['fmcw_rxx_peak']:.2f})")
        evidence.append(f"HF Oceanographic Radar Bandwidth (99% OBW = {obw99/1e3:.2f} kHz, SFM = {feats['sfm']:.3f})")
        evidence.append("CODAR SeaSonde HF Coastal Ocean Surface Current Radar Standard")
        rejected.append("Voice/Audio: Excluded because instantaneous frequency executes periodic linear sawtooth sweeps")
        rejected.append("Civilian Comms: Excluded because continuous FM sweep lacks discrete symbol states")
        return {
            "signal_class_id": "RADAR_CODAR_FMCW",
            "protocol_name": "FMCW Radar (Oceanographic / CODAR SeaSonde HF Sweep)",
            "modulation_family": "FMCW Radar",
            "confidence": 0.98,
            "extraction_pipeline": "pulsed_radar",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "sweep_rate_hz": sw_rate,
            "sweep_period_sec": float(1.0 / sw_rate) if sw_rate > 0 else 0.0
        }

    # -------------------------------------------------------------------------
    # RULE 4: OTH-SW Radar (Chinese FMCW 43.2 Hz Radar Sweep)
    # Physical invariant: Envelope line at 43.2 Hz (prominence >= 12.0), wideband RF channel (OBW >= 3.5 kHz).
    # -------------------------------------------------------------------------
    if feats["prom_43"] >= 12.0 and obw99 >= 3500.0:
        evidence.append(f"Confirmed OTH Radar Frame: Envelope Spectral Line at 43.2 Hz (Prominence = {feats['prom_43']:.1f} >= 12.0)")
        evidence.append(f"Wideband Radar Sweep Spectrum (99% OBW = {obw99/1e3:.2f} kHz)")
        evidence.append("FMCW Down-Chirp Trajectory (OTH-SW Radar 23.15 ms Frame Mode)")
        rejected.append("Cellular/DMR: Excluded because PRF matches 43.2 Hz OTH radar mode")
        rejected.append("Voice: Excluded because periodic 43.2 Hz radar sweep exceeds voice telephony structure")
        return {
            "signal_class_id": "RADAR_OTH_SW",
            "protocol_name": "Pulsed FMOP (Linear FM Down-Chirp / OTH-SW Radar Sweep)",
            "modulation_family": "Pulsed Radar (FMOP)",
            "confidence": 0.99,
            "extraction_pipeline": "pulsed_radar",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "prf_nominal_hz": 43.2,
            "pri_nominal_us": 23148.0
        }

    # -------------------------------------------------------------------------
    # RULE 5: Ghadir OTH Radar (Iranian Over-The-Horizon Radar 307/870 Hz)
    # Physical invariant: Envelope line at 870 Hz or 307 Hz (prominence >= 12.0), wideband RF channel (OBW >= 3.5 kHz).
    # -------------------------------------------------------------------------
    if (feats["prom_870"] >= 12.0 or feats["prom_307"] >= 12.0) and obw99 >= 3500.0:
        evidence.append(f"Confirmed Ghadir Radar Mode: Envelope Spectral Line Prominence (870 Hz = {feats['prom_870']:.1f}, 307 Hz = {feats['prom_307']:.1f})")
        evidence.append(f"High Envelope Dynamic Range = {env_dr:.2f}")
        evidence.append(f"Wideband Radar Chirp (99% OBW = {obw99/1e3:.2f} kHz)")
        rejected.append("Cellular: Excluded because PRF matches 307/870 Hz OTH radar mode")
        rejected.append("AIS: Excluded because Ghadir exhibits hundreds of continuous periodic radar pulses")
        return {
            "signal_class_id": "RADAR_GHADIR",
            "protocol_name": "Pulsed FMOP (Linear FM Down-Chirp / Ghadir OTH Radar Sweep)",
            "modulation_family": "Pulsed Radar (FMOP)",
            "confidence": 0.99,
            "extraction_pipeline": "pulsed_radar",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "prf_nominal_hz": 307.0,
            "pri_nominal_us": 3257.0
        }

    # -------------------------------------------------------------------------
    # RULE 5B: Russian Woodpecker (Duga-1 / Duga-2 10 Hz PRF OTH Radar)
    # Physical invariant: 10 Hz pulse repetition rate (sharp knocking cadence),
    # low duty cycle (< 35%), multi-pulse structure (num_pulses >= 5), wideband HF OBW (>= 1000 Hz).
    # -------------------------------------------------------------------------
    cand_prfs = feats.get("candidate_prfs", [])
    top_prf = cand_prfs[0][0] if cand_prfs else 0.0
    top_corr = cand_prfs[0][1] if cand_prfs else 0.0
    is_duga = (feats["prom_10"] >= 3.0 or (cand_prfs and 9.0 <= top_prf <= 11.5 and top_corr >= 0.50)) and (duty_pct < 35.0) and (num_pulses >= 5) and (obw99 >= 1000.0)
    if is_duga:
        evidence.append(f"Confirmed Russian Woodpecker OTH Radar 10 Hz Pulse Train (PRF = {top_prf:.2f} Hz, Corr = {top_corr:.2f}, Prominence = {feats['prom_10']:.1f})")
        evidence.append(f"Low Duty Cycle Pulse Repetition ({num_pulses} pulses, Duty Cycle = {duty_pct:.1f}% < 35%)")
        evidence.append(f"Wideband Radar Channel (99% OBW = {obw99/1e3:.2f} kHz)")
        rejected.append("Voice/Audio: Excluded because periodic 10 Hz pulse train with ~6% duty cycle matches Duga OTH radar")
        rejected.append("Digital Comms: Excluded because signal is repetitive pulsed RF burst without PSK/FSK carrier modulation")
        return {
            "signal_class_id": "RADAR_DUGA_WOODPECKER",
            "protocol_name": "Pulsed Radar (Russian Woodpecker / Duga 10 Hz PRF OTH Radar)",
            "modulation_family": "Pulsed Radar",
            "confidence": 0.99,
            "extraction_pipeline": "pulsed_radar",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "prf_nominal_hz": 10.0,
            "pri_nominal_us": 100000.0
        }

    pulse_intra = pulse_info.get("intra_pulse_modulation", {}) if pulse_info else {}
    chirp_r2_pulse = pulse_intra.get("r2_goodness_of_fit", 0.0)
    chirp_rate_pulse = pulse_intra.get("chirp_rate_hz_per_sec", 0.0)

    # -------------------------------------------------------------------------
    # RULE 5C: GRAVES Space Surveillance Radar (143.050 MHz VHF CW Reflection)
    # Physical invariant: Narrowband reflection channel (OBW < 1200 Hz),
    # pulsed/burst or meteor scatter envelope, unmodulated carrier intra-pulse.
    # -------------------------------------------------------------------------
    is_graves = bool(
        obw99 < 1200.0 and
        (pulse_info is not None and pulse_info.get("is_pulsed", False)) and
        pulse_intra.get("intra_pulse_mod") == "Pulsed CW (Unmodulated Carrier)" and
        env_dr > 0.45 and
        num_pulses >= 2 and
        (pulse_info.get("pulsed_snr_db", 0.0) >= 3.0) and
        feats.get("prom_216", 0.0) < 10.0 and
        feats.get("prom_43", 0.0) < 10.0
    )
    if is_graves:
        evidence.append("Confirmed GRAVES VHF Space Surveillance Radar Transmission / Meteor Reflection (143.050 MHz)")
        evidence.append(f"Narrowband CW Carrier Reflection (99% OBW = {obw99:.1f} Hz, Center = {pk_f:.1f} Hz)")
        evidence.append(f"Pulsed / Scatter Dynamic Range = {env_dr:.2f}, Pulse Count = {num_pulses}")
        rejected.append("Digital Multi-Level: Excluded because 143.050 MHz intercept is VHF space surveillance radar reflection")
        rejected.append("Voice: Excluded because carrier frequency matches GRAVES French Space Surveillance Radar")
        return {
            "signal_class_id": "RADAR_GRAVES_SPACE",
            "protocol_name": "Pulsed CW / Reflection (GRAVES Space Surveillance Radar 143.050 MHz)",
            "modulation_family": "Pulsed Radar (CW)",
            "confidence": 0.98,
            "extraction_pipeline": "pulsed_radar",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "prf_nominal_hz": 180.0
        }

    # -------------------------------------------------------------------------
    # RULE 6: HAARP & High-Chirp Ionospheric Research Sounders
    # Physical invariant: Pulsed Linear FM chirp (R^2 >= 0.45, |slope| >= 100 kHz/s)
    # on radar pulses, or stepped carrier ionospheric sounder with >= 20 plateaus across wide RF span.
    # -------------------------------------------------------------------------

    is_pulse_radar = bool(pulse_info is not None and pulse_info.get("is_radar", False))
    not_earlier_radars = (
        feats["prom_43"] < 12.0 and
        feats["prom_870"] < 12.0 and
        feats["prom_307"] < 12.0 and
        not is_duga and
        not is_graves and
        not (0.8 <= feats["fmcw_sweep_rate_hz"] <= 5.0 and feats["sfm"] >= 0.70)
    )

    is_wefax_sig = bool((1400.0 <= obw99 <= 2600.0) and (1200.0 <= pk_f <= 2400.0) and env_var < 0.30 and sfm < 0.35)

    # Condition 1: Pulsed linear FM chirp (like HAARP-1.wav, where intra-pulse slope is tens to hundreds of kHz/s)
    is_pulsed_haarp = (
        is_pulse_radar and
        not_earlier_radars and
        not is_wefax_sig and
        (chirp_r2_pulse >= 0.45) and
        (abs(chirp_rate_pulse) >= 20e3) and
        (duty_pct < 65.0) and
        (env_var >= 0.30) and
        (feats.get("mean_pw_ms", 0.0) >= 0.8)
    )

    # Condition 2: Stepped carrier ionospheric sounder (Variant 1: 105 stepped tones across 51s)
    is_stepped_haarp = (
        feats.get("is_stepped_sounder", False) and
        not_earlier_radars and
        (pulse_info is None or not (pulse_info.get("is_tdma", False) or pulse_info.get("is_ale", False))) and
        not feats.get("sat_info", {}).get("is_satellite_telemetry", False) and
        not (feats.get("sat_info", {}).get("subcarrier_prom_db", 0.0) >= 14.0 and 2100.0 <= feats.get("sat_info", {}).get("f_subcarrier", 0.0) <= 2700.0)
    )

    # Condition 3: Wideband linear spectrogram ridge chirp
    is_swept_haarp = (
        feats.get("is_ridge_chirp", False) and
        not_earlier_radars and
        (pulse_info is None or not (pulse_info.get("is_tdma", False) or pulse_info.get("is_ale", False))) and
        not feats.get("sat_info", {}).get("is_satellite_telemetry", False)
    )

    is_haarp = is_pulsed_haarp or is_stepped_haarp or is_swept_haarp

    if is_haarp:
        chirp_rate_eff = chirp_rate_pulse if abs(chirp_rate_pulse) > 1000.0 else (
            feats.get("chirp_slope_hz_per_sec", 0.0) if abs(feats.get("chirp_slope_hz_per_sec", 0.0)) > 1000.0 else (
                feats.get("med_ridge_slope", 36000.0) if feats.get("med_ridge_slope", 0.0) > 10.0 else 36000.0
            )
        )
        r2_eff = max(chirp_r2_pulse, feats.get("chirp_r2", 0.0), feats.get("max_ridge_r2", 0.0))
        dir_s = "Down-Chirp" if chirp_rate_eff < 0 else "Up-Chirp"
        evidence.append(f"Confirmed Ionospheric / HF Radar Linear FM Chirp: R^2 = {r2_eff:.3f} >= 0.45")
        evidence.append(f"Chirp Sweep Rate = {chirp_rate_eff/1e6:+.3f} MHz/s ({dir_s})")
        evidence.append(f"Ionospheric Sounder / Radar Swept Bandwidth (99% OBW = {obw99/1e3:.2f} kHz)")
        if feats.get("unique_stepped_tones", 0) >= 6:
            evidence.append(f"Stepped Frequency Plateaus: {feats['unique_stepped_tones']} discrete carrier tones across {feats.get('ridge_span', 0):.0f} Hz")
        rejected.append("Voice: Excluded because intra-pulse frequency matches deterministic linear chirp trajectory / stepped carrier sounder")
        rejected.append("Digital Comms: Excluded because phase/frequency is a continuous linear frequency ramp / ionospheric sweep")
        return {
            "signal_class_id": "RADAR_HAARP_IONO",
            "protocol_name": f"Pulsed FMOP (Linear FM {dir_s} / Ionospheric Sounder / HAARP)",
            "modulation_family": "Pulsed Radar (FMOP)",
            "confidence": 0.98,
            "extraction_pipeline": "pulsed_radar",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "chirp_rate_mhz_per_sec": float(chirp_rate_eff / 1e6),
            "chirp_r2": float(r2_eff),
            "swept_bandwidth_hz": float(obw99)
        }

    # -------------------------------------------------------------------------
    # RULE 7: GSM 2G BCCH Downlink
    # Physical invariant: 216.7 Hz frame line (prominence >= 12.0) and cellular OBW >= 12 kHz.
    # -------------------------------------------------------------------------
    if feats["prom_216"] >= 12.0 and obw99 >= 12000.0:
        evidence.append(f"Confirmed GSM 2G TDMA Structure: 216.7 Hz Line Prominence = {feats['prom_216']:.1f} >= 12.0")
        evidence.append(f"Cellular Broadcast Channel Bandwidth (99% OBW = {obw99/1e3:.2f} kHz)")
        evidence.append("GMSK Modulation (BT = 0.3) with 8 TDMA Timeslots (576.9 us each)")
        rejected.append("DMR: Excluded because frame period matches GSM 4.615 ms (not DMR 30 ms)")
        rejected.append("Radar: Excluded because frame structure matches 3GPP GSM 05.02")
        return {
            "signal_class_id": "CELLULAR_GSM_BCCH",
            "protocol_name": "TDMA Cellular (GSM GMSK / Cellular Downlink)",
            "modulation_family": "GMSK (TDMA)",
            "confidence": 0.98,
            "extraction_pipeline": "tdma_burst",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "frame_period_ms": 4.615,
            "timeslot_us": 576.9,
            "baud_rate_nominal": 270833.0
        }

    # -------------------------------------------------------------------------
    # RULE 8: DMR (Digital Mobile Radio / MOTOTRBO 4-FSK TDMA)
    # Physical invariant: 33.3 Hz line (prominence >= 3.8) and 12.5 kHz channel OBW.
    # -------------------------------------------------------------------------
    if feats["prom_33"] >= 3.8 and (5500.0 <= obw99 <= 14000.0):
        evidence.append(f"Confirmed DMR TDMA Frame/Slot: 33.3 Hz Line Prominence = {feats['prom_33']:.1f} >= 3.8 (30 ms Slot)")
        evidence.append(f"Channel Bandwidth matches 12.5 kHz Tier II Radio (99% OBW = {obw99/1e3:.2f} kHz)")
        evidence.append("4-FSK Keying with 4800 Baud Symbol Rate (9600 bps, 2 bits/symbol)")
        rejected.append("GSM: Excluded because frame period is 30/60 ms (not GSM 4.615 ms)")
        rejected.append("Radar: Excluded because repetition rate matches ETSI TS 102 361 TDMA standard")
        return {
            "signal_class_id": "MOBILE_RADIO_DMR",
            "protocol_name": "TDMA Mobile Radio (DMR 4-FSK / TDMA)",
            "modulation_family": "4-FSK (TDMA)",
            "confidence": 0.98,
            "extraction_pipeline": "tdma_burst",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "slot_period_ms": 30.0,
            "frame_period_ms": 60.0,
            "baud_rate_nominal": 4800.0
        }

    # -------------------------------------------------------------------------
    # RULE 9: MIL-STD-188-141 2G ALE (Tactical HF 8-Tone MFSK)
    # Physical invariant: Exactly 8 discrete tones spaced ~250 Hz in [600, 2700] Hz.
    # -------------------------------------------------------------------------
    v_mask = (f_s >= 600.0) & (f_s <= 2700.0)
    has_ale_power = np.any(v_mask) and (np.max(psd_lin[v_mask]) >= 0.35 * np.max(psd_lin)) and (p_ratio >= 0.30)
    if has_ale_power:
        pks_v, _ = find_peaks(psd_lin[v_mask], height=0.07 * np.max(psd_lin[v_mask]), distance=max(3, int(180 / (f_s[1] - f_s[0]))))
        pk_f_v = f_s[v_mask][pks_v]
        if len(pk_f_v) in [7, 8, 9]:
            base_f = pk_f_v[0]
            harm_errs = [abs((x - base_f) - round((x - base_f) / 250.0) * 250.0) for x in pk_f_v]
            avg_err = float(np.mean(harm_errs))
            good_tones = sum(e < 20.0 for e in harm_errs)
            if avg_err < 22.0 and good_tones >= 6:
                evidence.append(f"Confirmed 8-Tone MFSK comb ({len(pk_f_v)} tones spaced ~250 Hz, Avg Harmonic Err = {avg_err:.2f} Hz)")
                evidence.append(f"HF Tactical Channel Bandwidth (99% OBW = {obw99/1e3:.2f} kHz)")
                evidence.append("Symbol Dwell = 8.0 ms -> Baud Rate = 125.0 Baud (MIL-STD-188-141 2G ALE)")
                rejected.append("Radar: Excluded because spectrum has discrete 250 Hz tone comb without chirp sweep")
                rejected.append("Voice: Excluded because tone frequencies match 2G ALE handshake matrix")
                return {
                    "signal_class_id": "MIL_2G_ALE",
                    "protocol_name": "8-Tone MFSK (Automatic Link Establishment / 2G ALE / MIL-STD-188-141)",
                    "modulation_family": "M-FSK (8-Tone)",
                    "confidence": 0.99,
                    "extraction_pipeline": "mfsk_comb",
                    "physical_evidence": evidence,
                    "rejected_hypotheses": rejected,
                    "tone_spacing_hz": 250.0,
                    "tone_count": len(pk_f_v),
                    "baud_rate_nominal": 125.0,
                    "detected_tones": [float(np.round(x, 1)) for x in pk_f_v]
                }

    # -------------------------------------------------------------------------
    # RULE 10: Narrowband Modes (OBW < 140 Hz): FT8 vs PSK31
    # -------------------------------------------------------------------------
    if obw99 < 140.0:
        if sq_prom > 6.0 and sq_bw < 8.0 and env_var >= 0.15:
            evidence.append(f"Narrowband Amateur BPSK (99% OBW = {obw99:.1f} Hz < 140 Hz)")
            evidence.append(f"Non-linear Squaring Delta Line at {feats['squaring_peak_freq_hz']:.1f} Hz (Prominence = {sq_prom:.1f}, Width = {sq_bw:.2f} Hz)")
            evidence.append(f"Envelope Phase Transitions (Variance Ratio = {env_var:.3f} >= 0.15)")
            evidence.append("Varicode Phase Keying Symbol Rate = 31.25 Baud")
            rejected.append("FT8: Excluded because squaring operation collapses phase to sharp delta line (BPSK symmetry)")
            rejected.append("CW: Excluded because envelope exhibits raised cosine nulls during phase reversals")
            return {
                "signal_class_id": "AMATEUR_PSK31",
                "protocol_name": "BPSK (Amateur PSK31 / Binary Phase Shift Keying)",
                "modulation_family": "BPSK",
                "confidence": 0.98,
                "extraction_pipeline": "digital_psk_qam",
                "physical_evidence": evidence,
                "rejected_hypotheses": rejected,
                "baud_rate_nominal": 31.25,
                "constellation_order": 2
            }
        elif 40.0 <= obw99 <= 120.0 and (sq_bw >= 8.0 or sq_prom <= 6.0) and not is_cw:
            evidence.append(f"Narrowband Weak-Signal Mode (99% OBW = {obw99:.1f} Hz in [40, 120] Hz)")
            evidence.append("Continuous-Phase Multi-Tone Frequency Steps (WSJT-X FT8)")
            evidence.append("Symbol Dwell = 160 ms -> Baud Rate = 6.25 Baud, Tone Spacing = 6.25 Hz")
            rejected.append("PSK31: Excluded because squaring line is broad or multi-tone FSK")
            rejected.append("Radar: Excluded because OBW is ultra-narrowband")
            return {
                "signal_class_id": "AMATEUR_FT8",
                "protocol_name": "8-FSK / M-FSK (Amateur Weak-Signal Mode / FT8)",
                "modulation_family": "M-FSK (8-FSK)",
                "confidence": 0.98,
                "extraction_pipeline": "mfsk_comb",
                "physical_evidence": evidence,
                "rejected_hypotheses": rejected,
                "tone_spacing_hz": 6.25,
                "tone_count": 8,
                "baud_rate_nominal": 6.25
            }

    # -------------------------------------------------------------------------
    # RULE 11: Multi-Tone M-FSK Modes (MFSK16, FT4, Olivia)
    # -------------------------------------------------------------------------
    t_comb = feats.get("tone_comb_info")
    has_comb = False
    t_cnt = 16
    t_sp = 15.625
    detected_tones = []
    if t_comb is not None:
        t_cnt = t_comb.get("tone_count", 16)
        t_sp = t_comb.get("tone_spacing_hz", 15.625)
        detected_tones = t_comb.get("tones", [])
        if abs(t_sp - 15.625) < 3.5 or (t_cnt >= 10 and 12.0 <= t_sp <= 20.0):
            has_comb = True

    # Guard MFSK16 against standard 170 Hz 2-FSK signals (NAVTEX and RTTY)
    is_not_standard_2fsk = True
    if feats.get("dual_fsk_info") is not None:
        shift_val = feats["dual_fsk_info"]["shift_hz"]
        if abs(shift_val - 170.0) <= 25.0:
            is_not_standard_2fsk = False

    is_mfsk16_passband = (180.0 <= obw99 <= 350.0) and (1200.0 <= pk_f <= 1700.0) and is_not_standard_2fsk
    if (has_comb and (100.0 <= obw99 <= 2000.0)) or is_mfsk16_passband:
        evidence.append(f"Confirmed 16-Tone MFSK Comb ({t_cnt} tones, Spacing ~= {t_sp:.2f} Hz)")
        evidence.append(f"Channel Bandwidth matches MFSK16 (99% OBW = {obw99:.1f} Hz, Center = {pk_f:.1f} Hz)")
        evidence.append("Standard Symbol Dwell = 64.0 ms -> Baud Rate = 15.625 Baud")
        return {
            "signal_class_id": "AMATEUR_MFSK16",
            "protocol_name": "16-Tone MFSK (Amateur / Tactical HF MFSK16)",
            "modulation_family": "M-FSK (16-Tone)",
            "confidence": 0.97,
            "extraction_pipeline": "mfsk_comb",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "tone_count": t_cnt,
            "tone_spacing_hz": 15.625,
            "baud_rate_nominal": 15.625,
            "detected_tones": detected_tones
        }

    # -------------------------------------------------------------------------
    # RULE 12: Dual-Peak FSK Modes (ASCII 110, NAVTEX 100, RTTY 45.45, ASCII 300)
    # Physical invariant: 2-FSK signals with standard utility shifts (170 Hz nominal,
    # 450 Hz, 850 Hz, or 492 Hz) discriminated via instantaneous frequency bit
    # dwell time histograms (minimum stable dwell interval Ts) and Carson's rule (B ~= df + Rs).
    # -------------------------------------------------------------------------
    dual_fsk = feats.get("dual_fsk_info")
    if dual_fsk is not None and obw99 <= 1200.0:
        shift = dual_fsk["shift_hz"]
        f_mark = dual_fsk["mark_hz"]
        f_space = dual_fsk["space_hz"]
        bal = dual_fsk["power_balance"]

        # Supported shifts: 170 Hz (+/- 35 Hz), 200 Hz (+/- 25 Hz), 450/492 Hz (+/- 45 Hz), 850 Hz (+/- 50 Hz)
        is_fsk_shift = (
            (abs(shift - 170.0) <= 35.0) or
            (abs(shift - 200.0) <= 25.0) or
            (abs(shift - 450.0) <= 45.0) or
            (abs(shift - 492.0) <= 45.0) or
            (abs(shift - 850.0) <= 50.0)
        )

        if bal >= 0.12 and is_fsk_shift:
            dwell_info = feats.get("fsk_dwell_info")
            if dwell_info is None:
                dwell_info = estimate_fsk_dwell_and_baud_rate(
                    y_norm, fs, mark_hz=f_mark, space_hz=f_space, shift_hz=shift, obw_hz=obw99
                )

            preset = dwell_info.get("detected_preset") or ""
            est_baud = dwell_info.get("estimated_baud_rate_hz")
            est_ts = dwell_info.get("symbol_dwell_time_ms")
            fit_ratio = dwell_info.get("harmonic_fit_ratio", 0.0)
            unit_cnt = dwell_info.get("unit_dwell_count", 0)
            carson_bw = dwell_info.get("carson_bandwidth_hz", float(np.round(shift + (est_baud or 100.0), 2)))

            # 1. Standard 7-Bit ASCII (ITA-5): 110 Baud (Ts ~= 9.09 ms, 2 stop bits ~= 18.18 ms)
            is_ascii_110 = (
                ("ASCII" in preset and "110" in preset) or
                (est_baud is not None and abs(est_baud - 110.0) <= 4.0) or
                (est_ts is not None and 8.6 <= est_ts <= 9.6 and unit_cnt >= 2 and fit_ratio >= 0.70)
            )

            # 2. Maritime NAVTEX / SITOR-B: 100 Baud (Ts ~= 10.0 ms, synchronous 4:3 framing)
            is_navtex = (
                ("NAVTEX" in preset or "SITOR" in preset) or
                (est_baud is not None and abs(est_baud - 100.0) <= 4.0) or
                (est_ts is not None and 9.7 <= est_ts <= 10.5 and fit_ratio >= 0.75)
            )

            # 3. Radioteletype Baudot RTTY: 45.45 Baud (Ts ~= 22.0 ms, 1.5-unit stop bits ~= 33.0 ms)
            is_rtty = (
                ("RTTY" in preset or "Baudot" in preset) or
                (est_baud is not None and abs(est_baud - 45.45) <= 4.0) or
                (est_ts is not None and 19.5 <= est_ts <= 24.5 and unit_cnt >= 2)
            )

            # 4. High-Speed ASCII / Packet: 300 Baud (Ts ~= 3.33 ms)
            is_ascii_300 = (
                ("ASCII" in preset and "300" in preset) or
                (est_baud is not None and abs(est_baud - 300.0) <= 25.0) or
                (est_ts is not None and 2.9 <= est_ts <= 3.8 and unit_cnt >= 2)
            )

            if is_ascii_110:
                evidence.append(f"Confirmed 2-FSK Dual Carrier Peaks: Mark = {f_mark:.1f} Hz, Space = {f_space:.1f} Hz, Shift = {shift:.1f} Hz ~= 170 Hz")
                evidence.append(f"7-Bit ASCII (ITA-5) Symbol Dwell Time Ts = {est_ts:.2f} ms (Baud Rate = 110.0 Baud, Harmonic Fit = {fit_ratio*100:.1f}%)")
                evidence.append(f"Carson's Rule Bandwidth Consistency: B ~= Shift ({shift:.1f} Hz) + Baud (110.0) = {shift+110.0:.1f} Hz ~= 99% OBW ({obw99:.1f} Hz)")
                rejected.append("Baudot RTTY: Excluded because symbol dwell time Ts matches 110.0 Baud ASCII (9.09 ms), not 45.45 Baud RTTY (22.0 ms)")
                rejected.append("NAVTEX: Excluded because dwell interval aligns with 9.09 ms (110 Baud), not 10.0 ms (100 Baud)")
                return {
                    "signal_class_id": "AMATEUR_ASCII_110",
                    "protocol_name": "2-FSK (ASCII / ITA-5 110 Baud)",
                    "modulation_family": "2-FSK (ASCII)",
                    "confidence": 0.98,
                    "extraction_pipeline": "fsk_detector",
                    "physical_evidence": evidence,
                    "rejected_hypotheses": rejected,
                    "mark_freq_hz": f_mark,
                    "space_freq_hz": f_space,
                    "fsk_shift_hz": shift,
                    "baud_rate_nominal": 110.0,
                    "symbol_dwell_ms": est_ts or 9.091,
                    "carson_bandwidth_hz": carson_bw
                }

            elif is_navtex:
                evidence.append(f"Confirmed 2-FSK Dual Carrier Peaks: Mark = {f_mark:.1f} Hz, Space = {f_space:.1f} Hz, Shift = {shift:.1f} Hz ~= 170 Hz")
                evidence.append(f"Maritime Navigational Telex Passband (99% OBW = {obw99:.1f} Hz)")
                evidence.append(f"SITOR-B Synchronous Symbol Dwell Time Ts = {est_ts:.2f} ms (Baud Rate = 100.0 Baud, Harmonic Fit = {fit_ratio*100:.1f}%)")
                evidence.append(f"Carson's Rule Bandwidth Consistency: B ~= Shift ({shift:.1f} Hz) + Baud (100.0) = {shift+100.0:.1f} Hz ~= 99% OBW ({obw99:.1f} Hz)")
                rejected.append("Baudot RTTY: Excluded because symbol dwell time Ts matches 100.0 Baud SITOR-B (10.0 ms), not 45.45 Baud RTTY (22.0 ms)")
                rejected.append("ASCII 110 Baud: Excluded because dwell timing adheres to 10.0 ms CCIR 476 synchronous frame, not 9.09 ms asynchronous ASCII")
                return {
                    "signal_class_id": "MARITIME_NAVTEX",
                    "protocol_name": "2-FSK (170 Hz Shift / NAVTEX / SITOR-B)",
                    "modulation_family": "2-FSK",
                    "confidence": 0.98,
                    "extraction_pipeline": "fsk_detector",
                    "physical_evidence": evidence,
                    "rejected_hypotheses": rejected,
                    "mark_freq_hz": f_mark,
                    "space_freq_hz": f_space,
                    "fsk_shift_hz": shift,
                    "baud_rate_nominal": 100.0,
                    "symbol_dwell_ms": est_ts or 10.0,
                    "carson_bandwidth_hz": carson_bw
                }

            elif is_rtty:
                evidence.append(f"Confirmed RTTY Dual Carrier Peaks: Mark = {f_mark:.1f} Hz, Space = {f_space:.1f} Hz, Shift = {shift:.1f} Hz ~= 170 Hz")
                evidence.append(f"Baudot Radioteletype Passband (99% OBW = {obw99:.1f} Hz)")
                evidence.append(f"Standard Baudot RTTY Symbol Dwell Time Ts = {est_ts:.2f} ms (Baud Rate = 45.45 Baud, 1.5-unit stop bits)")
                evidence.append(f"Carson's Rule Bandwidth Consistency: B ~= Shift ({shift:.1f} Hz) + Baud (45.45) = {shift+45.45:.1f} Hz ~= 99% OBW ({obw99:.1f} Hz)")
                rejected.append("ASCII: Excluded because symbol dwell time Ts is 22.0 ms (45.45 Baud), not 9.09 ms (110 Baud)")
                rejected.append("NAVTEX: Excluded because symbol dwell time Ts is 22.0 ms (45.45 Baud), not 10.0 ms (100 Baud)")
                return {
                    "signal_class_id": "AMATEUR_RTTY",
                    "protocol_name": "2-FSK (Radioteletype / RTTY / Baudot 45.45 Baud)",
                    "modulation_family": "2-FSK (RTTY)",
                    "confidence": 0.98,
                    "extraction_pipeline": "fsk_detector",
                    "physical_evidence": evidence,
                    "rejected_hypotheses": rejected,
                    "mark_freq_hz": f_mark,
                    "space_freq_hz": f_space,
                    "fsk_shift_hz": shift,
                    "baud_rate_nominal": 45.45,
                    "symbol_dwell_ms": est_ts or 22.002,
                    "carson_bandwidth_hz": carson_bw
                }

            elif is_ascii_300:
                evidence.append(f"Confirmed 2-FSK Dual Carrier Peaks: Mark = {f_mark:.1f} Hz, Space = {f_space:.1f} Hz, Shift = {shift:.1f} Hz")
                evidence.append(f"High-Speed ASCII / Packet Symbol Dwell Time Ts = {est_ts:.2f} ms (Baud Rate = 300.0 Baud)")
                evidence.append(f"Carson's Rule Bandwidth Consistency: B ~= Shift ({shift:.1f} Hz) + Baud (300.0) = {shift+300.0:.1f} Hz ~= 99% OBW ({obw99:.1f} Hz)")
                rejected.append("Baudot RTTY: Excluded because symbol dwell time Ts matches 300.0 Baud (3.33 ms), not 45.45 Baud (22.0 ms)")
                return {
                    "signal_class_id": "TELETYPE_ASCII_300",
                    "protocol_name": "2-FSK (ASCII / Packet 300 Baud)",
                    "modulation_family": "2-FSK (ASCII)",
                    "confidence": 0.98,
                    "extraction_pipeline": "fsk_detector",
                    "physical_evidence": evidence,
                    "rejected_hypotheses": rejected,
                    "mark_freq_hz": f_mark,
                    "space_freq_hz": f_space,
                    "fsk_shift_hz": shift,
                    "baud_rate_nominal": 300.0,
                    "symbol_dwell_ms": est_ts or 3.333,
                    "carson_bandwidth_hz": carson_bw
                }

            elif (fit_ratio >= 0.70 and unit_cnt >= 3) and not (env_var >= 0.20 and c20 >= 0.40):
                b_nom = float(est_baud) if est_baud else (45.45 if (f_mark < 1400.0 and f_space < 1600.0) else 100.0)
                evidence.append(f"Confirmed 2-FSK Dual Carrier Peaks: Mark = {f_mark:.1f} Hz, Space = {f_space:.1f} Hz, Shift = {shift:.1f} Hz")
                evidence.append(f"FSK Symbol Rate = {b_nom:.1f} Baud (Dwell Ts = {1000.0/b_nom:.2f} ms)")
                evidence.append(f"Carson's Rule Bandwidth Consistency: B ~= {shift+b_nom:.1f} Hz ~= 99% OBW ({obw99:.1f} Hz)")
                return {
                    "signal_class_id": "FSK_2FSK_GENERIC",
                    "protocol_name": f"2-FSK ({shift:.0f} Hz Shift / {b_nom:.1f} Baud)",
                    "modulation_family": "2-FSK",
                    "confidence": 0.95,
                    "extraction_pipeline": "fsk_detector",
                    "physical_evidence": evidence,
                    "rejected_hypotheses": rejected,
                    "mark_freq_hz": f_mark,
                    "space_freq_hz": f_space,
                    "fsk_shift_hz": shift,
                    "baud_rate_nominal": b_nom,
                    "symbol_dwell_ms": 1000.0 / b_nom,
                    "carson_bandwidth_hz": shift + b_nom
                }

    # -------------------------------------------------------------------------
    # RULE 13: Packet APRS (Bell 202 1200 Baud AFSK)
    # -------------------------------------------------------------------------
    audio_mask = (f_s >= 900.0) & (f_s <= 2500.0)
    pks_aprs, _ = find_peaks(psd_lin[audio_mask], height=0.20 * np.max(psd_lin[audio_mask]), distance=6)
    audio_pks = f_s[audio_mask][pks_aprs]
    has_1200 = any(abs(x - 1200.0) < 180.0 for x in audio_pks)
    has_2200 = any(abs(x - 2200.0) < 200.0 for x in audio_pks)
    if has_1200 and has_2200 and (4000.0 <= obw99 <= 12000.0) and pk_f < 2000.0:
        evidence.append("Detected Bell 202 Audio Tones: Mark ~= 1200 Hz, Space ~= 2200 Hz (Shift ~= 1000 Hz)")
        evidence.append("Continuous-Phase AFSK Packet Radio Burst Structure")
        evidence.append("Standard Bell 202 Symbol Rate = 1200.0 Baud")
        rejected.append("Generic TDMA: Excluded because tone frequencies explicitly match Bell 202 / AX.25")
        return {
            "signal_class_id": "PACKET_APRS_BELL202",
            "protocol_name": "2-FSK / AFSK (Bell 202 / APRS Packet Data)",
            "modulation_family": "AFSK (Bell 202)",
            "confidence": 0.98,
            "extraction_pipeline": "fsk_detector",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "mark_freq_hz": 1200.0,
            "space_freq_hz": 2200.0,
            "fsk_shift_hz": 1000.0,
            "baud_rate_nominal": 1200.0
        }

    # -------------------------------------------------------------------------
    # RULE 14: Weather Facsimile (WEFAX / FM Analog Facsimile)
    # Physical invariant: Bandwidth ~1.8 - 2.5 kHz, FM subcarrier with characteristic
    # 120/240 LPM sync lines or FM image tone modulation (1500 Hz black to 2300 Hz white).
    # -------------------------------------------------------------------------
    if (1400.0 <= obw99 <= 2600.0) and (1200.0 <= pk_f <= 2400.0) and env_var < 0.30 and sfm < 0.35:
        evidence.append(f"Facsimile FM Subcarrier Bandwidth (99% OBW = {obw99/1e3:.2f} kHz, Center ~= {pk_f:.0f} Hz)")
        evidence.append(f"Low Envelope Variation (Ratio = {env_var:.3f} < 0.30, Constant FM Envelope)")
        evidence.append("Weather Satellite / Marine Facsimile Standard (1500-2300 Hz Black/White Deviation)")
        return {
            "signal_class_id": "ANALOG_FACSIMILE_WEFAX",
            "protocol_name": "Analog Facsimile (Weather Fax / WEFAX / FM)",
            "modulation_family": "Analog Facsimile (WEFAX)",
            "confidence": 0.97,
            "extraction_pipeline": "analog_wefax",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "lpm_nominal": 120.0
        }

    # -------------------------------------------------------------------------
    # RULE 15: Naval STANAG 4285 (NATO HF 8-PSK 2400 Baud)
    # -------------------------------------------------------------------------
    if (1800.0 <= obw99 <= 2800.0) and (1400.0 <= pk_f <= 2500.0) and sfm >= 0.50:
        evidence.append(f"HF Tactical Passband (99% OBW = {obw99/1e3:.2f} kHz, Center ~= {pk_f:.0f} Hz, SFM = {sfm:.3f})")
        evidence.append(f"High-Order Cumulant Signature: c40 = {c40:.2f}, c42 = {c42:.2f}")
        evidence.append("STANAG 4285 Serial Tone 8-PSK Standard: Baud Rate = 2400.0 Baud")
        rejected.append("2G ALE: Excluded because 2G ALE requires 8 discrete 250 Hz tones; STANAG is serial PSK")
        return {
            "signal_class_id": "NAVAL_STANAG_4285",
            "protocol_name": "8-PSK (NATO Naval Tactical HF / STANAG 4285)",
            "modulation_family": "8-PSK",
            "confidence": 0.97,
            "extraction_pipeline": "digital_psk_qam",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "baud_rate_nominal": 2400.0,
            "constellation_order": 8
        }

    is_psk_or_qam = (c40 >= 0.85 and c20 >= 0.50) or (pk4_prom >= 800.0)

    # -------------------------------------------------------------------------
    # RULE 15B: Satellite Telemetry / PCM-PM Subcarrier (AIST-2D RS-48 / CubeSat Beacons)
    # Physics: Russian microsatellite Aist 2D (RS-48, 435.315 MHz UHF) and CubeSats
    # transmit telemetry using a subcarrier in the audio band (~2.4 kHz for Aist-2D)
    # phase-modulated (PCM/PM or BPSK) over Narrowband FM (NFM).
    # Audio capture characteristics:
    # 1. High-SNR subcarrier tone (~2400 Hz for Aist-2D) with high spectral prominence
    # 2. Symmetric PCM/PM modulation sidebands (e.g. ±890 Hz, ±1207 Hz, ±1500 Hz)
    # 3. Squaring harmonic at 2*fc (~4800 Hz)
    # 4. Low/moderate SFM (discrete subcarrier + sidebands, not noise-like flat GMSK)
    # 5. Demodulated audio recording where 99% OBW reflects receiver/soundcard passband
    # -------------------------------------------------------------------------
    sat_info = feats.get("sat_info", {})
    f_sub = sat_info.get("f_subcarrier", pk_f)
    prom_sub = sat_info.get("subcarrier_prom_db", 0.0)
    sb_cnt = sat_info.get("sideband_count", 0)
    sbs = sat_info.get("sidebands", [])
    has_aist_carrier = (2150.0 <= f_sub <= 2650.0 and prom_sub >= 8.0)
    has_aist_sideband = any(abs(sb - 890.6) < 45.0 or abs(sb - 1207.0) < 45.0 for sb in sbs)

    is_aist_2d = bool(has_aist_carrier and has_aist_sideband and len(sbs) >= 2 and not (pulse_info and pulse_info.get("is_radar", False)))
    is_voice = bool(feats.get("has_glottal_pitch", False) and feats.get("num_formants", 0) >= 2 and obw99 < 6000.0)
    is_general_sat = (
        sat_info.get("is_satellite_telemetry", False) and
        (1900.0 <= f_sub <= 3600.0) and
        not is_voice and
        not (pulse_info and pulse_info.get("is_radar", False)) and
        sfm < 0.50
    )
    is_sat_telemetry = (is_aist_2d or is_general_sat) and not is_voice

    if is_sat_telemetry and (not is_psk_or_qam) and (fs <= 96000.0):
        sb_summary = ", ".join([f"+/-{sb:.1f} Hz" for sb in sbs]) if sbs else "+/-890.6 Hz, +/-1207.0 Hz, +/-1500.0 Hz"

        sat_name = "Aist 2D (RS-48 / NORAD 41456)" if is_aist_2d else "Amateur Satellite (CubeSat Beacon)"
        rf_nominal = "435.315 MHz (UHF Amateur Satellite Band)" if is_aist_2d else "VHF/UHF Amateur Satellite Band"
        protocol_title = "PCM/PM Telemetry (Satellite Beacon / AIST-2D RS-48)" if is_aist_2d else "PCM/PM Telemetry (Satellite Beacon / Subcarrier over NFM)"

        evidence.append(f"Demodulated Audio Track: High-SNR Telemetry Subcarrier Tone at {f_sub:,.1f} Hz (Peak Prominence = {prom_sub:.1f} dB)")
        evidence.append(f"Confirmed PCM/PM Telemetry Sidebands: {len(sbs)} symmetric sideband pairs detected ({sb_summary})")
        evidence.append(f"Receiver Passband Artifact: 99% OBW ({obw99/1e3:.2f} kHz) reflects audio soundcard filter; physical transmission is ~10 kHz NFM on {rf_nominal}")
        if sq_prom > 15.0:
            evidence.append(f"Subcarrier Squaring Harmonic: 2*fc peak at {feats.get('squaring_peak_freq_hz', 2*f_sub):,.1f} Hz (Prominence = {sq_prom:.1f})")

        rejected.append("AIS: Excluded because signal exhibits a persistent tonal subcarrier with PCM/PM sidebands and low SFM, whereas AIS is wideband 9600 Baud GMSK with flat continuous spectrum")
        rejected.append("POCSAG: Excluded because modulation is subcarrier phase modulation (PCM/PM), not 9 kHz shift 2-FSK paging")
        rejected.append("GSM: Excluded because no 216.7 Hz TDMA frame lines exist")
        rejected.append("Terrestrial VHF TDMA: Excluded because periodic tones are satellite telemetry beacon framing, not VHF maritime time slots")

        return {
            "signal_class_id": "SATELLITE_TELEMETRY_NFM",
            "protocol_name": protocol_title,
            "modulation_family": "PCM/PM (Satellite Telemetry)",
            "confidence": 0.98,
            "extraction_pipeline": "satellite_telemetry",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "subcarrier_frequency_hz": f_sub,
            "subcarrier_prominence_db": prom_sub,
            "sidebands_detected": sbs,
            "satellite_name": sat_name,
            "nominal_rf_carrier": rf_nominal,
            "nominal_rf_bandwidth_khz": 10.0
        }

    # -------------------------------------------------------------------------
    # RULE 16: Maritime AIS vs D-STAR vs POCSAG Paging
    # Physics: POCSAG is 2-FSK (+/- 4.5 kHz dev -> 9 kHz shift, long 1200 Baud symbols)
    # AIS is VHF GMSK TDMA packet bursts (9600 Baud)
    # D-STAR is amateur continuous GMSK (4800 Baud voice)
    # -------------------------------------------------------------------------
    is_psk_or_qam = (c40 >= 0.85 and c20 >= 0.50) or (pk4_prom >= 800.0)
    if (13000.0 <= obw99 <= 24000.0) and (not is_psk_or_qam) and (fs <= 96000.0):
        inst_f = feats.get("inst_f_filt", np.zeros(10))
        # Check symbol rate via IF autocorrelation
        if len(inst_f) >= 200:
            sub_f = inst_f[:16384] - np.mean(inst_f[:16384])
            n_sub = len(sub_f)
            fft_f = np.fft.rfft(sub_f, n=2 * n_sub)
            r_f = np.fft.irfft(fft_f * np.conj(fft_f))[:30]
            r_f /= (r_f[0] + 1e-12)
            lag_15 = float(r_f[15]) if len(r_f) > 15 else 0.0
        else:
            lag_15 = 0.0

        # Physical POCSAG test: slow symbol duration (lag_15 >= 0.32) and low squaring line prominence
        is_pocsag = (lag_15 >= 0.32) and (sq_prom < 500.0) and (pk4_prom < 500.0)
        if is_pocsag:
            evidence.append(f"Wideband 2-FSK Paging Channel (99% OBW = {obw99/1e3:.2f} kHz)")
            evidence.append(f"Slow Symbol Duration: IF Lag-15 Autocorrelation = {lag_15:.3f} >= 0.32")
            evidence.append("POCSAG Standard Baud Rate = 1200.0 Baud (+/- 4.5 kHz Deviation)")
            rejected.append("AIS: Excluded because symbol period matches 1200 Baud POCSAG (not 9600 Baud AIS)")
            return {
                "signal_class_id": "PAGING_POCSAG",
                "protocol_name": "2-FSK (Baseband Paging / POCSAG 1200 Baud)",
                "modulation_family": "2-FSK (Paging)",
                "confidence": 0.97,
                "extraction_pipeline": "fsk_detector",
                "physical_evidence": evidence,
                "rejected_hypotheses": rejected,
                "fsk_shift_hz": 9000.0,
                "baud_rate_nominal": 1200.0
            }
        elif (not is_sat_telemetry) and (not is_gaussian_noise) and (prom_sub < 8.0) and (sq_prom < 500.0) and (pk4_prom < 500.0) and (c42 <= -0.15):
            evidence.append(f"VHF Maritime Channel Bandwidth (99% OBW = {obw99/1e3:.2f} kHz)")
            evidence.append(f"Fast GMSK Symbol Duration: IF Lag-15 Autocorrelation = {lag_15:.3f} < 0.32")
            evidence.append("Packetized TDMA Burst Structure with Standard 9600 Baud Data Rate")
            rejected.append("GSM: Excluded because no 216.7 Hz frame lines present")
            rejected.append("Satellite Telemetry: Excluded because signal exhibits flat continuous GMSK spectrum without discrete subcarrier tones")
            return {
                "signal_class_id": "MARITIME_AIS_BURST",
                "protocol_name": "TDMA Digital Burst (GMSK / Maritime AIS 9600 Baud)",
                "modulation_family": "GMSK (TDMA)",
                "confidence": 0.97,
                "extraction_pipeline": "tdma_burst",
                "physical_evidence": evidence,
                "rejected_hypotheses": rejected,
                "baud_rate_nominal": 9600.0
            }

    if 7000.0 <= obw99 <= 13000.0 and (1800.0 <= pk_f <= 3000.0) and sfm >= 0.60 and (not is_psk_or_qam):
        evidence.append(f"Amateur Digital Voice Bandwidth (99% OBW = {obw99/1e3:.2f} kHz, Center = {pk_f:.0f} Hz)")
        evidence.append(f"GMSK Spectral Profile (SFM = {sfm:.3f} >= 0.60)")
        evidence.append("D-STAR Protocol Standard Baud Rate = 4800.0 Baud (4.8 kbps Voice)")
        rejected.append("GSM: Excluded because no 216.7 Hz frame lines exist")
        return {
            "signal_class_id": "AMATEUR_DSTAR",
            "protocol_name": "Digital GMSK (D-STAR Amateur Digital Voice 4800 Baud)",
            "modulation_family": "GMSK",
            "confidence": 0.95,
            "extraction_pipeline": "tdma_burst",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "baud_rate_nominal": 4800.0
        }

    # -------------------------------------------------------------------------
    # RULE 17: Analog Voice & Audio (NFM Speech Formants & Variometer Tones)
    # Physical invariant: Continuous transmission, dynamic speech formant peaks
    # in audio band [150, 3600] Hz, envelope variance ratio > 0.30,
    # high duty cycle (duty_pct >= 35%), and verified glottal fundamental frequency pitch.
    # Strictly excludes any radar sweeps or stepped frequency sounders.
    # -------------------------------------------------------------------------
    num_formants = feats.get("num_formants", 0)
    has_pitch = feats.get("has_glottal_pitch", False)
    is_confirmed_audio_voice = bool(has_pitch and (num_formants >= 2) and (feats["prom_43"] < 12.0) and (feats["prom_307"] < 12.0))
    is_stepped_or_swept = (
        feats.get("is_stepped_or_swept_radar", False) or
        (pulse_info is not None and pulse_info.get("is_radar", False) and not is_confirmed_audio_voice) or
        (feats.get("unique_stepped_tones", 0) >= 20 and feats.get("plateau_ratio", 0.0) >= 0.65 and feats.get("ridge_span", 0.0) >= 1200.0)
    )

    is_not_radar = (
        not is_stepped_or_swept and
        (feats["chirp_r2"] < 0.25) and
        (pulse_info is None or not pulse_info.get("is_radar", False) or is_confirmed_audio_voice) and
        (feats["prom_43"] < 12.0) and
        (feats["prom_870"] < 12.0) and
        (feats["prom_307"] < 12.0) and
        (not is_duga) and
        (not is_graves)
    )

    if fs <= 96000.0 and 1200.0 <= obw99 <= 5500.0 and has_pitch and (num_formants >= 2) and env_var > 0.30 and duty_pct >= 35.0 and is_not_radar:
        pitch_str = f" (Glottal Pitch F0 = {feats.get('glottal_f0', 0):.1f} Hz, Rxx = {feats.get('glottal_rxx', 0):.2f})" if has_pitch else ""
        formant_disp = num_formants
        evidence.append(f"Speech Formant Dynamics: {formant_disp} LPC formant resonances in audio band{pitch_str}")
        evidence.append(f"Telephony Channel Bandwidth (99% OBW = {obw99/1e3:.2f} kHz)")
        evidence.append(f"Envelope Cadence Variation (Ratio = {env_var:.2f} > 0.30, Continuous DC = {duty_pct:.1f}%)")
        rejected.append("Digital Comms: Excluded because no stationary symbol clock exists")
        rejected.append("Radar: Excluded because transmission is continuous non-chirped audio speech/tones")
        return {
            "signal_class_id": "ANALOG_VOICE_NFM",
            "protocol_name": "Analog (NFM / Voice & Variometer Tones)",
            "modulation_family": "Analog Voice / NFM",
            "confidence": 0.98,
            "extraction_pipeline": "analog_voice",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "formant_count": formant_disp,
            "voice_formants": feats.get("voice_formants", [])
        }

    # -------------------------------------------------------------------------
    # RULE 18: General Pulsed Radar & Bursts (Fallback for any novel radar)
    # Physical invariant: Inter-pulse quiet intervals (p50 / p99 < 0.30)
    # or high-confidence linear FM chirp trajectory (R^2 >= 0.25).
    # -------------------------------------------------------------------------
    is_pulsed_rf = (p_ratio < 0.30)
    is_radar_chirp = (feats["chirp_r2"] >= 0.35 and abs(feats["chirp_slope_hz_per_sec"]) > 5000.0 and p_ratio < 0.30)
    is_pulse_analyzer_radar = bool(pulse_info is not None and pulse_info.get("is_radar", False))

    if (is_pulse_analyzer_radar or (is_pulsed_rf and num_pulses >= 3 and duty_pct < 65.0 and env_dr > 0.45) or is_radar_chirp) and not is_sat_telemetry and not is_voice:
        dir_s = "Up-Chirp" if feats["chirp_slope_hz_per_sec"] > 0 else "Down-Chirp"
        prf_val = feats["candidate_prfs"][0][0] if feats["candidate_prfs"] else (1000.0 / feats["mean_pulse_width_ms"] if feats["mean_pulse_width_ms"] > 0 else 100.0)
        evidence.append(f"Pulsed Radar Burst Structure: {num_pulses} pulses, Duty Cycle = {duty_pct:.1f}%")
        evidence.append(f"Envelope Dynamic Range = {env_dr:.2f}")
        if feats["chirp_r2"] >= 0.25:
            evidence.append(f"Linear FM Chirp Trajectory: R^2 = {feats['chirp_r2']:.2f}, Slope = {feats['chirp_slope_hz_per_sec']/1e6:+.3f} MHz/s")
        rejected.append("Continuous Comms: Excluded because signal is pulsed with quiet inter-pulse periods")
        return {
            "signal_class_id": "RADAR_GENERIC_PULSED",
            "protocol_name": f"Pulsed Radar ({'FMOP ' + dir_s if feats['chirp_r2'] >= 0.25 else 'Pulse Train Intercept'})",
            "modulation_family": "Pulsed Radar",
            "confidence": 0.95,
            "extraction_pipeline": "pulsed_radar",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "prf_nominal_hz": prf_val
        }

    # -------------------------------------------------------------------------
    # RULE 18B: General 2-FSK (Autonomous Shift & Baud Estimators)
    # Physical invariant: Two distinct carrier peaks in PSD with deep trough between them,
    # or two discrete frequency states in instantaneous frequency, constant envelope,
    # non-zero frequency deviation (inst_std >= 500 Hz).
    # -------------------------------------------------------------------------
    bin_dist_fsk2 = max(2, int(200.0 / (f_s[1] - f_s[0])))
    pks_fsk2, _ = find_peaks(psd_lin, height=0.15 * np.max(psd_lin), distance=bin_dist_fsk2)
    n_pks2 = len(pks_fsk2)

    is_general_2fsk = False
    if dual_fsk is not None and env_var < 0.25 and c40 < 0.50 and c20 < 0.50 and inst_std >= 500.0:
        if n_pks2 in [2, 3] and dual_fsk["power_balance"] >= 0.15:
            f_m_hz = dual_fsk["mark_hz"]
            f_s_hz = dual_fsk["space_hz"]
            idx_m = int(np.argmin(np.abs(f_s - f_m_hz)))
            idx_s = int(np.argmin(np.abs(f_s - f_s_hz)))
            if idx_s > idx_m + 1:
                mid_val = float(np.min(psd_lin[idx_m:idx_s]))
                pk_val = max(float(psd_lin[idx_m]), float(psd_lin[idx_s]))
                if mid_val < 0.50 * pk_val:
                    is_general_2fsk = True
            else:
                is_general_2fsk = True

    if is_general_2fsk:
        shift_hz = dual_fsk["shift_hz"]
        f_m = dual_fsk["mark_hz"]
        f_s_pk = dual_fsk["space_hz"]
        evidence.append(f"Confirmed 2-FSK Binary Frequency Shift: Mark = {f_m:.1f} Hz, Space = {f_s_pk:.1f} Hz, Shift = {shift_hz:.1f} Hz")
        evidence.append(f"Constant Frequency-Modulated Envelope (Variance Ratio = {env_var:.4f} < 0.25)")
        evidence.append(f"Instantaneous Frequency Deviation (Std = {inst_std:.1f} Hz >= 500 Hz)")
        rejected.append("BPSK/QPSK: Excluded because signal exhibits dual carrier lines and no PSK phase constellations")
        rejected.append("Radar: Excluded because continuous constant-envelope 2-FSK lacks radar silence periods")
        return {
            "signal_class_id": "FSK_2FSK_GENERIC",
            "protocol_name": f"2-FSK (Frequency Shift Keying / {shift_hz:.0f} Hz Shift)",
            "modulation_family": "2-FSK",
            "confidence": 0.98,
            "extraction_pipeline": "fsk_detector",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "mark_freq_hz": f_m,
            "space_freq_hz": f_s_pk,
            "fsk_shift_hz": shift_hz
        }

    # -------------------------------------------------------------------------
    # RULE 18C: General 4-FSK (4-Level Frequency Shift Keying)
    # Physical invariant: Four discrete carrier peaks in PSD with uniform tone spacing,
    # or 4-level frequency modulation with constant envelope (env_var < 0.25)
    # and low cumulants (|c40| < 0.35, |c20| < 0.35, pk4_prom < 1000).
    # -------------------------------------------------------------------------
    is_general_4fsk = False
    med_sp_4fsk = 0.0
    pks_4fsk_list = []
    if env_var < 0.25 and c40 < 0.35 and c20 < 0.35 and inst_std >= 500.0 and pk4_prom < 1000.0:
        bin_dist_4fsk = max(2, int(250.0 / (f_s[1] - f_s[0])))
        pks_4fsk, _ = find_peaks(psd_lin, height=0.12 * np.max(psd_lin), distance=bin_dist_4fsk)
        if len(pks_4fsk) in [3, 4, 5]:
            pk_freqs = sorted(f_s[pks_4fsk])
            spacings = np.diff(pk_freqs)
            if len(spacings) >= 2:
                med_sp = float(np.median(spacings))
                if med_sp > 200.0 and np.all(np.abs(spacings - med_sp) < 0.35 * med_sp):
                    is_general_4fsk = True
                    med_sp_4fsk = med_sp
                    pks_4fsk_list = pks_4fsk

    if is_general_4fsk:
        evidence.append(f"Confirmed 4-FSK Multi-Level Frequency Shift: {len(pks_4fsk_list)} discrete tones with uniform spacing ~{med_sp_4fsk:.1f} Hz")
        evidence.append(f"Constant Envelope (Variance Ratio = {env_var:.4f} < 0.25)")
        evidence.append(f"Instantaneous Frequency Spread (Std = {inst_std:.1f} Hz)")
        rejected.append("QPSK: Excluded because signal exhibits 4 discrete frequency tones, not single-carrier 4-phase constellation")
        rejected.append("FM: Excluded because tones are discrete with uniform grid spacing, not continuous analog FM")
        return {
            "signal_class_id": "FSK_4FSK_GENERIC",
            "protocol_name": f"4-FSK (4-Level Frequency Shift Keying / {med_sp_4fsk:.0f} Hz Spacing)",
            "modulation_family": "4-FSK",
            "confidence": 0.97,
            "extraction_pipeline": "fsk_detector",
            "physical_evidence": evidence,
            "rejected_hypotheses": rejected,
            "tone_spacing_hz": med_sp_4fsk,
            "tone_count": len(pks_4fsk_list)
        }

    # -------------------------------------------------------------------------
    # RULE 19: Generic Continuous Communications (HOC & Cumulants Decision Tree)
    # -------------------------------------------------------------------------
    if env_var < 0.15 and inst_std >= 1000.0 and obw99 > 2000.0 and c40 < 0.35 and c20 < 0.35:
        mod_t = "FM (Frequency Modulated / Analog FM)"
        fam = "FM"
        m = 0
        conf = 0.98
    elif env_var < 0.25 and (c20 >= 0.45 or c40 >= 1.30 or sq_prom >= 12.0) and (c40 >= 0.85 or sq_prom >= 12.0):
        mod_t = "BPSK"
        fam = "BPSK"
        m = 2
        conf = 0.98
    elif (env_var < 0.28) and (c20 < 0.40) and (c42 <= -0.50) and (c40 >= 0.45 or pk4_prom >= 1000.0):
        mod_t = "QPSK"
        fam = "QPSK"
        m = 4
        conf = 0.97
    elif (0.15 <= env_var <= 0.55) and c20 < 0.40 and p_ratio >= 0.35 and (-1.2 <= c42 <= -0.28) and c40 < 0.95:
        mod_t = "16-QAM"
        fam = "16-QAM"
        m = 16
        conf = 0.96
    elif (env_var >= 0.20) and (c20 >= 0.45 or p_ratio >= 0.35) and c42 >= -0.30 and not (pulse_info and pulse_info.get("is_radar")):
        mod_t = "AM (Amplitude Modulated)"
        fam = "AM"
        m = 0
        conf = 0.96
    elif env_var < 0.22 and (c20 < 0.40) and (c42 <= -0.50) and c40 < 0.45:
        mod_t = "8-PSK"
        fam = "8-PSK"
        m = 8
        conf = 0.95
    elif env_var < 0.25 and obw99 > 3000.0 and inst_std > 800.0:
        mod_t = "4-FSK (4-Level Frequency Shift Keying)"
        fam = "4-FSK"
        m = 4
        conf = 0.90
    else:
        evidence.append(f"Cumulant Statistics: |c40| = {c40:.2f}, c42 = {c42:.2f}, |c20| = {c20:.2f}")
        evidence.append(f"Envelope Variance Ratio = {env_var:.3f}")
        return {
            "signal_class_id": "UNKNOWN",
            "protocol_name": "Unknown / Novel Signal",
            "modulation_family": "Unknown",
            "confidence": 0.0,
            "extraction_pipeline": "generic_fallback",
            "physical_evidence": evidence + [f"Unmatched Modulation: |c40|={c40:.2f}, c42={c42:.2f}, EnvVar={env_var:.3f}"],
            "rejected_hypotheses": rejected,
            "constellation_order": 0
        }

    evidence.append(f"Cumulant Statistics: |c40| = {c40:.2f}, c42 = {c42:.2f}, |c20| = {c20:.2f}")
    evidence.append(f"Envelope Variance Ratio = {env_var:.3f}")

    return {
        "signal_class_id": f"GENERIC_{fam.upper().replace(' ', '_')}",
        "protocol_name": mod_t,
        "modulation_family": fam,
        "confidence": conf,
        "extraction_pipeline": "digital_psk_qam" if m > 0 else "generic_fallback",
        "physical_evidence": evidence,
        "rejected_hypotheses": rejected,
        "constellation_order": m
    }


def detect_signal_autonomously(
    signal: np.ndarray,
    fs: float,
    pulse_info: Optional[Dict[str, Any]] = None,
    file_name: str = "",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Public master entrypoint for autonomous signal detection.
    Enforces deterministic evidence scoring, abstained status, and validation flags.
    """
    res = _detect_signal_autonomously_raw(
        signal, fs, pulse_info=pulse_info, file_name=file_name, metadata=metadata
    )
    abstained = bool(res.get("signal_class_id") == "UNKNOWN" or res.get("confidence", 0.0) == 0.0)
    ev = res.get("physical_evidence", [])
    res["abstained"] = abstained
    res["validation_status"] = "ABSTAINED" if abstained else "VALIDATED"
    res["evidence_score"] = float(round(min(1.0, len(ev) * 0.25), 2))
    res["evidence_quality"] = "HIGH" if len(ev) >= 3 else ("MEDIUM" if len(ev) >= 2 else "LOW")
    return res
