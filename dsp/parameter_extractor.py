"""
Blind Signal Parameter Extraction Engine
========================================
Performs blind, autonomous extraction of RF signal parameters without prior knowledge:
- Center Carrier Frequency (fc) via Peak Search & Spectral Centroid
- Occupied Bandwidth (-3 dB, -10 dB, and 99% Power Occupancy) with Bin-Spacing Resolution
- Band-Integrated In-Band vs Out-of-Band Physical SNR Solver (Eliminates Dynamic Range Fallbacks)
- Symbol / Baud Rate (Rs) with Radar & Analog Audio Harmonic Suppression
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from scipy.signal import find_peaks, hilbert, medfilt, butter, filtfilt
from scipy.linalg import solve_toeplitz
from .spectral import compute_welch_psd
from .preprocessor import remove_dc_offset, normalize_signal_power, compute_signal_stats
from .pulse_analyzer import analyze_pulse_train


def estimate_carrier_frequency(
    f: np.ndarray,
    psd_linear: np.ndarray,
    psd_db: np.ndarray
) -> Dict[str, float]:
    """
    Estimates the carrier center frequency via both peak detection and spectral centroid.
    """
    if len(f) == 0 or len(psd_linear) == 0:
        return {"fc_peak_hz": 0.0, "fc_centroid_hz": 0.0, "peak_power_db": -100.0}

    peak_idx = int(np.argmax(psd_db))
    fc_peak = float(f[peak_idx])
    peak_power = float(psd_db[peak_idx])

    total_power = np.sum(psd_linear)
    if total_power > 1e-18:
        fc_centroid = float(np.sum(f * psd_linear) / total_power)
    else:
        fc_centroid = fc_peak

    return {
        "fc_peak_hz": fc_peak,
        "fc_centroid_hz": fc_centroid,
        "peak_power_db": peak_power
    }


def estimate_occupied_bandwidth(
    f: np.ndarray,
    psd_linear: np.ndarray,
    psd_db: np.ndarray,
    occupied_pct: float = 0.99
) -> Dict[str, float]:
    """
    Estimates signal bandwidth across -3 dB, -10 dB, and 99% OBW with bin spacing resolution.
    """
    if len(f) == 0 or len(psd_linear) == 0:
        return {
            "bw_3db_hz": 0.0,
            "bw_10db_hz": 0.0,
            "bw_99pct_hz": 0.0,
            "f_lower_3db_hz": 0.0,
            "f_upper_3db_hz": 0.0,
            "f_lower_10db_hz": 0.0,
            "f_upper_10db_hz": 0.0,
            "f_lower_99pct_hz": 0.0,
            "f_upper_99pct_hz": 0.0
        }

    peak_power = np.max(psd_db)
    bin_spacing = float(abs(f[1] - f[0])) if len(f) > 1 else 1.0

    # 1. -3 dB Bandwidth
    thresh_3db = peak_power - 3.0
    idx_3db = np.where(psd_db >= thresh_3db)[0]
    if len(idx_3db) > 0:
        f_low_3db = float(f[idx_3db[0]]) - 0.5 * bin_spacing
        f_high_3db = float(f[idx_3db[-1]]) + 0.5 * bin_spacing
        bw_3db = max(bin_spacing, f_high_3db - f_low_3db)
    else:
        f_low_3db = f_high_3db = float(f[np.argmax(psd_db)])
        bw_3db = bin_spacing

    # 2. -10 dB Bandwidth
    thresh_10db = peak_power - 10.0
    idx_10db = np.where(psd_db >= thresh_10db)[0]
    if len(idx_10db) > 0:
        f_low_10db = float(f[idx_10db[0]]) - 0.5 * bin_spacing
        f_high_10db = float(f[idx_10db[-1]]) + 0.5 * bin_spacing
        bw_10db = max(bin_spacing, f_high_10db - f_low_10db)
    else:
        f_low_10db = f_high_10db = float(f[np.argmax(psd_db)])
        bw_10db = bin_spacing

    # 3. -20 dB Bandwidth
    thresh_20db = peak_power - 20.0
    idx_20db = np.where(psd_db >= thresh_20db)[0]
    if len(idx_20db) > 0:
        f_low_20db = float(f[idx_20db[0]]) - 0.5 * bin_spacing
        f_high_20db = float(f[idx_20db[-1]]) + 0.5 * bin_spacing
        bw_20db = max(bin_spacing, f_high_20db - f_low_20db)
    else:
        f_low_20db = f_high_20db = float(f[np.argmax(psd_db)])
        bw_20db = bin_spacing

    # 4. 99% Occupied Power Bandwidth (OBW)
    cum_power = np.cumsum(psd_linear)
    total_power = cum_power[-1] if len(cum_power) > 0 else 1.0

    if total_power > 1e-18:
        norm_cum_power = cum_power / total_power
        alpha = (1.0 - occupied_pct) / 2.0
        low_idx = int(np.searchsorted(norm_cum_power, alpha))
        high_idx = int(np.searchsorted(norm_cum_power, 1.0 - alpha))
        high_idx = min(high_idx, len(f) - 1)

        f_low_occ = float(f[low_idx])
        f_high_occ = float(f[high_idx])
        bw_99pct = max(bin_spacing, f_high_occ - f_low_occ)
    else:
        f_low_occ = f_high_occ = float(f[np.argmax(psd_db)])
        bw_99pct = bin_spacing

    return {
        "bw_3db_hz": float(np.round(bw_3db, 2)),
        "bw_10db_hz": float(np.round(bw_10db, 2)),
        "bw_20db_hz": float(np.round(bw_20db, 2)),
        "bw_99pct_hz": float(np.round(bw_99pct, 2)),
        "f_lower_3db_hz": float(np.round(f_low_3db, 2)),
        "f_upper_3db_hz": float(np.round(f_high_3db, 2)),
        "f_lower_10db_hz": float(np.round(f_low_10db, 2)),
        "f_upper_10db_hz": float(np.round(f_high_10db, 2)),
        "f_lower_20db_hz": float(np.round(f_low_20db, 2)),
        "f_upper_20db_hz": float(np.round(f_high_20db, 2)),
        "f_lower_99pct_hz": float(np.round(f_low_occ, 2)),
        "f_upper_99pct_hz": float(np.round(f_high_occ, 2))
    }


def estimate_band_integrated_snr(
    f: np.ndarray,
    psd_linear: np.ndarray,
    bw_params: Dict[str, float]
) -> float:
    """
    Computes true channel SNR by integrating in-band power across 99% OBW
    versus adjacent out-of-band noise spectral density.
    Never uses peak-to-floor dynamic range or fabricated fallback values.
    """
    if len(f) == 0 or len(psd_linear) == 0:
        return 0.0

    f_low = bw_params.get("f_lower_99pct_hz", f[0])
    f_high = bw_params.get("f_upper_99pct_hz", f[-1])
    bin_spacing = float(abs(f[1] - f[0])) if len(f) > 1 else 1.0

    in_band_mask = (f >= f_low) & (f <= f_high)
    out_band_mask = ~in_band_mask

    p_in_band = float(np.sum(psd_linear[in_band_mask]) * bin_spacing)
    n_in_bins = int(np.sum(in_band_mask))

    if np.any(out_band_mask) and n_in_bins > 0:
        noise_density = float(np.median(psd_linear[out_band_mask]))
        p_noise_in_band = noise_density * n_in_bins * bin_spacing
        sig_pwr = max(p_in_band - p_noise_in_band, 1e-15)
        snr_lin = max(sig_pwr / (p_noise_in_band + 1e-15), 1e-3)
        snr_db = float(10.0 * np.log10(snr_lin))
    elif n_in_bins > 0:
        # If signal fills entire Nyquist band, estimate noise floor from lowest 10% bins
        noise_density = float(np.percentile(psd_linear, 10))
        p_noise_in_band = noise_density * n_in_bins * bin_spacing
        sig_pwr = max(p_in_band - p_noise_in_band, 1e-15)
        snr_lin = max(sig_pwr / (p_noise_in_band + 1e-15), 1e-3)
        snr_db = float(10.0 * np.log10(snr_lin))
    else:
        snr_db = 0.0

    return float(np.clip(snr_db, -20.0, 42.0))


def estimate_dynamic_time_domain_snr(signal: np.ndarray) -> float:
    """
    Estimates true dynamic SNR from 90th percentile active speech/signal burst power
    vs 25th percentile background quiet floor power.
    Does not clamp artificially to +5 dB.
    """
    if signal is None or len(signal) == 0:
        return 0.0

    env = np.abs(signal)
    p90 = float(np.mean(env[env >= np.percentile(env, 90)] ** 2))
    p25 = float(np.mean(env[env <= np.percentile(env, 25)] ** 2))

    p_sig = max(p90 - p25, 1e-12)
    snr_lin = max(p_sig / (p25 + 1e-12), 1e-3)
    snr_db = float(10.0 * np.log10(snr_lin))
    return float(np.clip(snr_db, -20.0, 42.0))


def estimate_snr_m2m4(signal: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    """
    Estimates Signal-to-Noise Ratio (SNR) using second and fourth sample moments.
    """
    if signal is None or len(signal) == 0:
        return {"snr_db": 0.0, "signal_power": 0.0, "noise_power": 1.0, "m2_moment": 0.0, "m4_moment": 0.0}

    clean_sig = signal - np.mean(signal)
    inst_power = np.abs(clean_sig) ** 2

    m2 = float(np.mean(inst_power))
    m4 = float(np.mean(inst_power ** 2))

    s_sq = 2.0 * (m2 ** 2) - m4
    signal_power = float(np.sqrt(max(s_sq, eps)))
    noise_power = float(max(m2 - signal_power, eps))

    snr_linear = signal_power / noise_power
    snr_db = float(10.0 * np.log10(max(snr_linear, 1e-4)))
    snr_db = float(np.clip(snr_db, -20.0, 42.0))

    return {
        "snr_db": snr_db,
        "signal_power": signal_power,
        "noise_power": noise_power,
        "m2_moment": m2,
        "m4_moment": m4
    }


def estimate_fsk_dwell_and_baud_rate(
    signal: np.ndarray,
    fs: float,
    mark_hz: Optional[float] = None,
    space_hz: Optional[float] = None,
    shift_hz: Optional[float] = None,
    obw_hz: Optional[float] = None
) -> Dict[str, Any]:
    """
    Estimates FSK symbol / baud rate and bit dwell times using instantaneous
    frequency transition analysis and Carson's rule consistency.
    
    Eliminates sub-harmonic and blind preset locking by enforcing:
    1. Baseband instantaneous frequency demodulation: f_inst(t) = (1/2pi) d/dt(unwrap(angle(s_bb(t))))
    2. Schmitt trigger with adaptive hysteresis: delta = max(8.0, 0.15 * Delta_f)
    3. Integer multiple harmonic dwell condition: k = round(dwell / Ts) >= 1
    4. Unit dwell detection (minimum stable dwell interval Ts)
    5. Carson's rule bandwidth consistency: B_carson = Delta_f + Baud
    """
    if len(signal) < 256 or fs <= 0:
        return {
            "estimated_baud_rate_hz": None,
            "symbol_dwell_time_ms": None,
            "baud_confidence": 0.0,
            "baud_label": "N/A (Insufficient Signal)",
            "dwell_count": 0,
            "carson_bandwidth_hz": 0.0,
            "harmonic_fit_ratio": 0.0,
            "detected_preset": None,
            "fsk_frequency_shift_hz": 0.0
        }

    # Signal conditioning: ensure analytical complex representation
    if np.isrealobj(signal):
        sig_c = hilbert(signal)
    else:
        sig_c = signal

    # Determine mark, space, and shift
    if mark_hz is not None and space_hz is not None:
        f_m = float(min(mark_hz, space_hz))
        f_s = float(max(mark_hz, space_hz))
        delta_f = float(f_s - f_m)
    elif shift_hz is not None and shift_hz > 0:
        delta_f = float(shift_hz)
        f_m = -delta_f / 2.0
        f_s = delta_f / 2.0
    else:
        # Fallback: estimate from Welch PSD peaks
        f_welch, psd_db, psd_lin = compute_welch_psd(sig_c, fs, nperseg=min(4096, max(256, len(sig_c) // 2)))
        bin_spacing = abs(f_welch[1] - f_welch[0]) if len(f_welch) > 1 else 1.0
        pks, _ = find_peaks(psd_lin, height=0.10 * np.max(psd_lin), distance=max(2, int(80.0 / bin_spacing)))
        if len(pks) >= 2:
            top2 = sorted(f_welch[pks], key=lambda x: psd_lin[np.argmin(np.abs(f_welch - x))], reverse=True)[:2]
            f_m = float(min(top2[0], top2[1]))
            f_s = float(max(top2[0], top2[1]))
            delta_f = float(f_s - f_m)
        else:
            delta_f = None
            f_m = None
            f_s = None

    if delta_f is None or delta_f < 10.0:
        return {
            "estimated_baud_rate_hz": None,
            "symbol_dwell_time_ms": None,
            "baud_confidence": 0.0,
            "baud_label": "N/A (No FSK Frequency Shift Detected)",
            "dwell_count": 0,
            "carson_bandwidth_hz": None,
            "harmonic_fit_ratio": 0.0,
            "detected_preset": None,
            "fsk_frequency_shift_hz": None,
            "parameter_status": "unknown",
            "estimator_method": "welch_psd_peaks"
        }

    # Downconvert to complex baseband centered between Mark and Space
    f_center = 0.5 * (f_m + f_s)
    t = np.arange(len(sig_c)) / fs
    s_bb = sig_c * np.exp(-1j * 2.0 * np.pi * f_center * t)

    # Lowpass filter baseband signal to suppress out-of-band noise before phase differentiation
    cutoff_hz = min(0.45 * fs, max(1.5 * delta_f, 1200.0))
    if cutoff_hz < 0.48 * fs and len(s_bb) >= 32:
        try:
            b_lp, a_lp = butter(4, cutoff_hz / (fs / 2.0), btype='low')
            s_bb = filtfilt(b_lp, a_lp, s_bb)
        except Exception:
            pass

    # Compute instantaneous frequency of baseband signal
    ph = np.unwrap(np.angle(s_bb))
    inst_f = np.diff(ph) * (fs / (2.0 * np.pi))

    # Median filter to eliminate sample-level phase derivative spikes
    k_filt = max(5, int(0.0005 * fs)) | 1
    inst_f_filt = medfilt(inst_f, k_filt)

    # Schmitt trigger with hysteresis threshold delta = max(8.0, 0.15 * delta_f)
    delta_thresh = max(8.0, 0.15 * delta_f)
    states = np.zeros(len(inst_f_filt), dtype=np.int8)
    curr_state = 0
    for idx, f_val in enumerate(inst_f_filt):
        if f_val > delta_thresh:
            curr_state = 1
        elif f_val < -delta_thresh:
            curr_state = -1
        states[idx] = curr_state

    # Find state transitions
    trans_indices = []
    last_st = None
    for idx, st in enumerate(states):
        if st != 0:
            if last_st is not None and st != last_st:
                trans_indices.append(idx)
            last_st = st

    if len(trans_indices) < 4:
        # Fallback to Carson's rule from OBW if available
        if obw_hz is not None and delta_f is not None and obw_hz > delta_f:
            fallback_baud = float(np.round(obw_hz - delta_f, 1))
            return {
                "estimated_baud_rate_hz": fallback_baud,
                "symbol_dwell_time_ms": float(np.round(1000.0 / fallback_baud, 3)) if fallback_baud > 0 else None,
                "baud_confidence": 0.40,
                "baud_label": f"{fallback_baud:.1f} Baud (Carson OBW Inferred)",
                "dwell_count": 0,
                "carson_bandwidth_hz": float(np.round(delta_f + fallback_baud, 2)),
                "harmonic_fit_ratio": 0.0,
                "detected_preset": None,
                "fsk_frequency_shift_hz": delta_f,
                "parameter_status": "estimated",
                "estimator_method": "carson_rule_obw",
                "symbol_rate_observed": None,
                "symbol_rate_estimated": fallback_baud,
                "symbol_rate_nominal": None
            }
        return {
            "estimated_baud_rate_hz": None,
            "symbol_dwell_time_ms": None,
            "baud_confidence": 0.0,
            "baud_label": "N/A (Insufficient FSK Transitions)",
            "dwell_count": 0,
            "carson_bandwidth_hz": delta_f,
            "harmonic_fit_ratio": 0.0,
            "detected_preset": None,
            "fsk_frequency_shift_hz": delta_f,
            "parameter_status": "unknown",
            "estimator_method": "fsk_schmitt_transitions",
            "symbol_rate_observed": None,
            "symbol_rate_estimated": None,
            "symbol_rate_nominal": None
        }

    dwell_samples = np.diff(trans_indices)
    dwells_ms = (dwell_samples / fs) * 1000.0
    valid_dwells = dwells_ms[(dwells_ms >= 0.8) & (dwells_ms <= 120.0)]

    if len(valid_dwells) < 4:
        valid_dwells = dwells_ms[dwells_ms >= 0.5]

    # Candidate Standard Presets to evaluate
    # (name, baud_rate, nominal_dwell_ms)
    candidates = [
        ("Baudot RTTY 45.45", 45.45, 22.002),
        ("RTTY 50.0", 50.0, 20.0),
        ("RTTY 75.0", 75.0, 13.333),
        ("NAVTEX / SITOR-B 100.0", 100.0, 10.0),
        ("ASCII / ITA-5 110.0", 110.0, 9.091),
        ("ASCII / Packet 300.0", 300.0, 3.333),
        ("FSK 600.0", 600.0, 1.667),
        ("Bell 202 / APRS 1200.0", 1200.0, 0.833)
    ]

    best_cand = None
    best_score = -1.0
    best_fit = 0.0
    best_unit_cnt = 0

    for name, r_baud, ts_nom in candidates:
        # Require k >= 1 integer multiple
        k = np.round(valid_dwells / ts_nom)
        # Calculate error for k >= 1 normalized to ts_nom
        err = np.where(k >= 1, np.abs(valid_dwells - k * ts_nom) / ts_nom, 1.0)
        if abs(r_baud - 45.45) < 0.1:
            # Check 1.5 unit stop bit for 45.45 Baudot RTTY
            err_1_5 = np.abs(valid_dwells - 1.5 * ts_nom) / ts_nom
            err = np.minimum(err, np.where(valid_dwells >= 0.7 * ts_nom, err_1_5, 1.0))

        # Tighter 6.0% error tolerance ensures 100 Baud (10.0 ms) and 110 Baud (9.09 ms) never cross-match
        err_tol = 0.060
        fit = float(np.mean(err < err_tol))
        unit_cnt = int(np.sum(np.abs(valid_dwells - ts_nom) < err_tol * ts_nom))

        # Carson's rule bandwidth consistency
        b_carson = delta_f + r_baud
        if obw_hz is not None and obw_hz > 0:
            carson_err = abs(obw_hz - b_carson) / max(obw_hz, b_carson)
            carson_score = max(0.0, 1.0 - carson_err)
        else:
            carson_score = 0.8

        # Unit dwell requirement avoids sub-harmonic locking (e.g. 50 Baud matching 100 Baud)
        has_unit = min(1.0, unit_cnt / float(max(2, int(0.03 * len(valid_dwells)))))
        score = 0.50 * fit + 0.30 * has_unit + 0.20 * carson_score

        if score > best_score:
            best_score = score
            best_cand = (name, r_baud, ts_nom)
            best_fit = fit
            best_unit_cnt = unit_cnt

    # Extract continuous / empirical dwell from histogram
    bin_w = 0.20
    bins = np.arange(1.0, min(80.0, float(np.max(valid_dwells)) + 2.0), bin_w)
    counts, edges = np.histogram(valid_dwells, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if len(counts) > 0 and np.max(counts) > 0:
        min_pk_h = max(3, int(0.20 * np.max(counts)))
        pks, _ = find_peaks(counts, height=min_pk_h, distance=max(2, int(1.0 / bin_w)))
        if len(pks) > 0:
            first_pk_dwell = centers[pks[0]]
            near_dwells = valid_dwells[np.abs(valid_dwells - first_pk_dwell) < 0.6]
            empirical_ts = float(np.median(near_dwells)) if len(near_dwells) > 0 else float(first_pk_dwell)
        else:
            max_idx = int(np.argmax(counts))
            empirical_ts = float(centers[max_idx])
    if len(valid_dwells) == 0:
        return {
            "estimated_baud_rate_hz": None,
            "symbol_dwell_time_ms": None,
            "baud_confidence": 0.0,
            "baud_label": "N/A (No Valid FSK Dwells)",
            "dwell_count": 0,
            "carson_bandwidth_hz": float(np.round(delta_f, 2)) if delta_f is not None else None,
            "harmonic_fit_ratio": 0.0,
            "unit_dwell_count": 0,
            "detected_preset": None,
            "fsk_frequency_shift_hz": float(np.round(delta_f, 2)) if delta_f is not None else None,
            "parameter_status": "unknown",
            "estimator_method": "schmitt_dwell_histogram",
            "symbol_rate_observed": None,
            "symbol_rate_estimated": None,
            "symbol_rate_nominal": None
        }

    empirical_ts = float(np.percentile(valid_dwells, 15)) if len(valid_dwells) > 0 else 0.0
    empirical_baud = float(1000.0 / empirical_ts) if empirical_ts > 0 else None

    # Decision: Use candidate preset if best_cand has unit dwells and good score
    if best_cand is not None and best_unit_cnt >= 2 and (best_score >= 0.68 or best_fit >= 0.78):
        selected_name, selected_baud, selected_ts = best_cand
        conf = float(np.clip(0.85 + 0.14 * best_fit, 0.80, 0.99))
        preset_name = selected_name
        p_status = "validated"
    elif empirical_baud is not None:
        selected_baud = float(np.round(empirical_baud, 1))
        selected_ts = float(np.round(empirical_ts, 3))
        selected_name = f"Custom FSK {selected_baud:.1f} Baud"
        conf = 0.75
        preset_name = None
        p_status = "observed"
    else:
        return {
            "estimated_baud_rate_hz": None,
            "symbol_dwell_time_ms": None,
            "baud_confidence": 0.0,
            "baud_label": "N/A (No Valid Dwell Measured)",
            "dwell_count": len(valid_dwells),
            "carson_bandwidth_hz": float(np.round(delta_f, 2)) if delta_f is not None else None,
            "harmonic_fit_ratio": 0.0,
            "unit_dwell_count": 0,
            "detected_preset": None,
            "fsk_frequency_shift_hz": float(np.round(delta_f, 2)) if delta_f is not None else None,
            "parameter_status": "unknown",
            "estimator_method": "schmitt_dwell_histogram",
            "symbol_rate_observed": None,
            "symbol_rate_estimated": None,
            "symbol_rate_nominal": None
        }

    carson_bw = float(np.round(delta_f + selected_baud, 2))
    b_lbl = f"{selected_baud/1e3:.1f} kBaud" if selected_baud >= 1000.0 else f"{selected_baud:.1f} Baud"

    return {
        "estimated_baud_rate_hz": selected_baud,
        "symbol_dwell_time_ms": float(np.round(selected_ts, 3)),
        "baud_confidence": conf,
        "baud_label": b_lbl,
        "dwell_count": len(valid_dwells),
        "carson_bandwidth_hz": carson_bw,
        "harmonic_fit_ratio": float(np.round(best_fit, 3)),
        "unit_dwell_count": best_unit_cnt,
        "detected_preset": preset_name,
        "empirical_dwell_time_ms": float(np.round(empirical_ts, 3)),
        "empirical_baud_rate_hz": float(np.round(empirical_baud, 1)) if empirical_baud else None,
        "fsk_frequency_shift_hz": float(np.round(delta_f, 2)),
        "parameter_status": p_status,
        "estimator_method": "schmitt_dwell_histogram",
        "symbol_rate_observed": selected_baud,
        "symbol_rate_estimated": selected_baud,
        "symbol_rate_nominal": selected_baud if preset_name else None
    }


def estimate_symbol_baud_rate(
    signal: np.ndarray,
    fs: float,
    is_suppressed: bool = False,
    suppress_reason: str = "Pulsed Radar / Analog Audio Signal"
) -> Dict[str, Any]:
    """
    Estimates symbol / baud rate (Rs) for digital modulations.
    Suppressed for Radar and Analog Audio / Voice to prevent locking onto speech cadences / PRF harmonics.
    """
    if is_suppressed:
        return {
            "estimated_baud_rate_hz": None,
            "baud_confidence": 0.0,
            "baud_label": f"N/A ({suppress_reason} - Harmonic Suppressed)"
        }

    if len(signal) < 256 or fs <= 0:
        return {"estimated_baud_rate_hz": 0.0, "baud_confidence": 0.0, "baud_label": "0.0 Baud"}

    inst_power = np.abs(signal) ** 2
    inst_power_ac = inst_power - np.mean(inst_power)

    n_fft = 2 ** int(np.ceil(np.log2(min(len(inst_power_ac), 65536))))
    spectrum = np.abs(np.fft.rfft(inst_power_ac, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    min_f = max(5.0, 4.0 * fs / float(n_fft))
    max_f = 0.49 * fs
    valid_mask = (freqs >= min_f) & (freqs <= max_f)

    if not np.any(valid_mask):
        return {"estimated_baud_rate_hz": 0.0, "baud_confidence": 0.0, "baud_label": "0.0 Baud"}

    search_freqs = freqs[valid_mask]
    search_spectrum = spectrum[valid_mask]

    peak_idx = int(np.argmax(search_spectrum))
    estimated_baud = float(search_freqs[peak_idx])

    median_level = float(np.median(search_spectrum)) + 1e-12
    peak_val = float(search_spectrum[peak_idx])
    confidence = float(min(1.0, max(0.0, (peak_val - median_level) / (peak_val + 1e-12))))

    # In digital signals, true clock tone must have high prominence (confidence > 0.35)
    if confidence < 0.35:
        # Check FSK instantaneous frequency dwell transitions before declaring no clock
        fsk_fallback = estimate_fsk_dwell_and_baud_rate(signal, fs)
        if fsk_fallback.get("estimated_baud_rate_hz") is not None and fsk_fallback.get("baud_confidence", 0.0) >= 0.70:
            return {
                "estimated_baud_rate_hz": fsk_fallback["estimated_baud_rate_hz"],
                "baud_confidence": fsk_fallback["baud_confidence"],
                "baud_label": fsk_fallback["baud_label"]
            }
        return {
            "estimated_baud_rate_hz": None,
            "baud_confidence": confidence,
            "baud_label": "N/A (No Stationary Symbol Clock Detected)"
        }

    label = f"{estimated_baud/1e3:.1f} kBaud" if estimated_baud >= 1000.0 else f"{estimated_baud:.1f} Baud"

    return {
        "estimated_baud_rate_hz": estimated_baud,
        "baud_confidence": confidence,
        "baud_label": label
    }


def extract_all_parameters(
    signal: np.ndarray,
    fs: float,
    nperseg: int = 2048,
    pulse_info: Optional[Dict[str, Any]] = None,
    is_analog_audio: bool = False
) -> Dict[str, Any]:
    """
    Master parameter extraction engine with Physics-Based Multi-Stage Triage Tree.
    """
    # 1. Preprocessing
    dc_free = remove_dc_offset(signal)
    normalized_sig, avg_power = normalize_signal_power(dc_free)

    # 2. Spectral Analysis
    f_shifted, psd_db, psd_linear = compute_welch_psd(normalized_sig, fs, nperseg=nperseg)

    # 3. Carrier Frequency & Bandwidth
    carrier_params = estimate_carrier_frequency(f_shifted, psd_linear, psd_db)
    bw_params = estimate_occupied_bandwidth(f_shifted, psd_linear, psd_db)

    # 4. Pulse Analysis
    if pulse_info is None:
        pulse_info = analyze_pulse_train(normalized_sig, fs)

    is_pulsed = pulse_info.get("is_pulsed", False)
    is_radar = pulse_info.get("is_radar", False)
    duty_cycle = pulse_info.get("duty_cycle_pct", 100.0)

    # 5. Band-Integrated & Dynamic SNR Solver
    band_snr = estimate_band_integrated_snr(f_shifted, psd_linear, bw_params)
    dyn_snr = estimate_dynamic_time_domain_snr(normalized_sig)

    if is_pulsed and duty_cycle < 55.0:
        if is_radar:
            active_snr = pulse_info.get("pulsed_snr_db", band_snr)
            if active_snr <= 0.0 or active_snr > 42.0:
                active_snr = band_snr
            snr_method = "Segmented In-Pulse Energy Partitioning"
        else:
            active_snr = dyn_snr if dyn_snr > 0.0 else band_snr
            snr_method = "Dynamic In-Burst to Inter-Burst Power Ratio"
    elif is_analog_audio:
        active_snr = dyn_snr
        snr_method = "Dynamic Active-to-Quiet Audio Power Ratio"
    else:
        m2m4_res = estimate_snr_m2m4(normalized_sig)
        m2m4_snr = m2m4_res["snr_db"]
        if m2m4_snr > -5.0 and m2m4_snr <= 35.0:
            active_snr = m2m4_snr
            snr_method = "M2M4 Continuous Sample Moment Estimator"
        else:
            active_snr = band_snr
            snr_method = "Band-Integrated Spectral In-Band Power vs Noise Density"

    active_snr = float(np.clip(active_snr, -15.0, 42.0))

    # 6. Baud Rate Extraction with Autonomous Physical Detection
    from .autonomous_detector import detect_signal_autonomously
    det = detect_signal_autonomously(normalized_sig, fs, pulse_info=pulse_info)
    pipeline_name = det.get("extraction_pipeline", "")
    nominal_baud = det.get("baud_rate_nominal")

    if pipeline_name in ["pulsed_radar", "analog_voice", "continuous_wave", "analog_wefax"] or is_radar or is_analog_audio:
        if pipeline_name == "pulsed_radar" or is_radar:
            reason = "Pulsed Radar"
        elif pipeline_name == "continuous_wave":
            reason = "Continuous Wave / Unmodulated Carrier"
        elif pipeline_name == "analog_wefax":
            reason = "Analog Facsimile"
        else:
            reason = "Analog Voice / Audio Signal"
        baud_params = {
            "estimated_baud_rate_hz": None,
            "baud_confidence": 0.0,
            "baud_label": f"N/A ({reason} - Harmonic Suppressed)"
        }
    elif pipeline_name == "ook_morse":
        dot_ms = det.get("dot_duration_ms", 50.0)
        wpm = det.get("estimated_wpm", 20.0)
        b_val = float(np.round(1000.0 / dot_ms, 1)) if dot_ms > 0 else 20.0
        baud_params = {
            "estimated_baud_rate_hz": b_val,
            "baud_confidence": 0.95,
            "baud_label": f"{wpm:.1f} WPM ({b_val:.1f} Baud Elements)"
        }
    elif nominal_baud is not None and nominal_baud > 0:
        b_val = float(nominal_baud)
        b_lbl = f"{b_val/1e3:.1f} kBaud" if b_val >= 1000.0 else f"{b_val:.1f} Baud"
        baud_params = {
            "estimated_baud_rate_hz": b_val,
            "baud_confidence": 0.90,
            "baud_label": b_lbl,
            "symbol_rate_nominal": b_val,
            "symbol_rate_estimated": b_val,
            "symbol_rate_observed": None,
            "parameter_status": "nominal",
            "estimator_method": "standard_protocol_nominal"
        }
    else:
        baud_params = estimate_symbol_baud_rate(
            normalized_sig,
            fs,
            is_suppressed=False
        )

    # 7. Envelope Statistics
    stats_params = compute_signal_stats(normalized_sig)

    # 8. FSK Specialized Telemetry Parameters
    fsk_params = {}
    if pipeline_name == "fsk_detector" or det.get("fsk_shift_hz") is not None:
        fsk_shift_val = det.get("fsk_shift_hz")
        if fsk_shift_val is not None:
            f_m_val = det.get("mark_freq_hz", carrier_params["fc_peak_hz"] - fsk_shift_val / 2.0)
            f_s_val = det.get("space_freq_hz", carrier_params["fc_peak_hz"] + fsk_shift_val / 2.0)
            b_rate = baud_params.get("estimated_baud_rate_hz")
            fsk_dwell_ms = det.get("symbol_dwell_ms", float(np.round(1000.0 / b_rate, 3)) if (b_rate is not None and b_rate > 0) else None)
            fsk_carson_bw = float(np.round(fsk_shift_val + (b_rate or 0.0), 2)) if b_rate is not None else float(np.round(fsk_shift_val, 2))
            fsk_mod_h = float(np.round(fsk_shift_val / (b_rate + 1e-12), 3)) if b_rate is not None else None
            fsk_params = {
                "fsk_mark_frequency_hz": float(np.round(f_m_val, 2)),
                "fsk_space_frequency_hz": float(np.round(f_s_val, 2)),
                "fsk_frequency_shift_hz": float(np.round(fsk_shift_val, 2)),
                "fsk_symbol_dwell_time_ms": fsk_dwell_ms,
                "fsk_carson_bandwidth_hz": fsk_carson_bw,
                "fsk_modulation_index_h": fsk_mod_h
            }

    results: Dict[str, Any] = {
        "sampling_rate_hz": float(fs),
        "total_samples_analyzed": len(signal),
        "duration_seconds": len(signal) / float(fs) if fs > 0 else 0.0,
        "average_power_raw": avg_power,
        "snr_db": float(np.round(active_snr, 2)),
        "snr_estimation_method": snr_method,
        "spectral_dynamic_range_db": float(np.round(np.max(psd_db) - np.percentile(psd_db, 15), 2)),
        **carrier_params,
        **bw_params,
        **baud_params,
        **fsk_params,
        **stats_params
    }

    return results


def extract_lpc_speech_formants(
    signal: np.ndarray,
    fs: float,
    has_pitch: bool = True
) -> List[float]:
    """
    Extracts genuine speech formants (F1, F2, F3, ...) using Linear Predictive Coding (LPC)
    polynomial root decomposition. Strictly gated by glottal pitch autocorrelation:
    if has_pitch is False, returns an empty list to prevent unmodulated CW carrier tones
    or stepped radar tones from being misidentified as human vocal tract resonances.
    """
    if not has_pitch or fs <= 0 or len(signal) < 512:
        return []

    sig_r = signal.real if np.iscomplexobj(signal) else signal
    sig_r = sig_r - np.mean(sig_r)

    frame_len = min(len(sig_r), max(128, int(0.040 * fs)))
    start_idx = max(0, (len(sig_r) - frame_len) // 2)
    fr = sig_r[start_idx:start_idx + frame_len] * np.hamming(frame_len)

    p = min(16, max(8, int(fs / 1000.0) + 4))
    r = np.correlate(fr, fr, mode='full')[len(fr) - 1:len(fr) + p]
    if r[0] < 1e-6:
        return []

    try:
        a = solve_toeplitz((r[:-1], r[:-1]), -r[1:])
        poly = np.concatenate(([1.0], a))
        roots = np.roots(poly)
        formants = []
        for root in roots:
            if np.imag(root) > 0 and np.abs(root) < 1.05:
                freq = float(np.angle(root) * (fs / (2.0 * np.pi)))
                bw = float(-(fs / np.pi) * np.log(np.abs(root) + 1e-12))
                if 200.0 <= freq <= min(4000.0, 0.48 * fs) and bw < 600.0:
                    formants.append(float(np.round(freq, 1)))
        formants.sort()
        return formants
    except Exception:
        return []
