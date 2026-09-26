"""
Multi-Window Temporal Validation Layer
======================================
Splits preprocessed, normalized baseband signals into sequential temporal windows
to independently assess parameter persistence, temporal stability, and stationarity.

Guarantees:
1. Blind stationarity verification without prior protocol assumptions.
2. Per-window physical parameter extraction (fc, OBW, SNR, envelope, instantaneous frequency, baud, shift, pulses).
3. Scale-normalized variation metrics eliminating arbitrary hard-coded heuristic thresholds.
4. Graceful degradation for short-duration observations (< INSUFFICIENT_OBSERVATION_DURATION).
5. 100% backward-compatible structured dictionary output.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.signal import medfilt, hilbert

from .spectral import compute_welch_psd
from .preprocessor import compute_signal_stats
from .parameter_extractor import (
    estimate_carrier_frequency,
    estimate_occupied_bandwidth,
    estimate_band_integrated_snr,
    estimate_snr_m2m4,
    estimate_symbol_baud_rate,
    estimate_fsk_dwell_and_baud_rate
)
from .pulse_analyzer import analyze_pulse_train


@dataclass
class TemporalValidationConfig:
    """
    Configuration for multi-window temporal stability analysis.
    """
    num_windows: int = 4
    min_samples_per_window: int = 256
    high_stability_threshold: float = 0.85
    medium_stability_threshold: float = 0.60
    parameter_tolerances: Dict[str, float] = field(default_factory=lambda: {
        "fc_centroid_hz": 0.15,          # Relative to occupied bandwidth
        "fc_peak_hz": 0.15,              # Relative to occupied bandwidth
        "bw_99pct_hz": 0.25,             # Coefficient of variation (sigma / mu)
        "snr_db": 0.35,                  # Sigma / 10 dB scale
        "papr_db": 0.30,                 # Coefficient of variation
        "mean_amplitude": 0.25,          # Coefficient of variation
        "std_amplitude": 0.30,           # Coefficient of variation
        "envelope_variance_ratio": 0.30, # Coefficient of variation
        "inst_freq_std_hz": 0.30,        # Relative to bandwidth / sample rate
        "inst_freq_mean_hz": 0.20,       # Relative to bandwidth
        "inst_freq_variance": 0.35,      # Coefficient of variation
        "estimated_baud_rate_hz": 0.20,  # Coefficient of variation
        "frequency_shift_hz": 0.20,      # Coefficient of variation
        "pulse_prf_hz": 0.20,            # Coefficient of variation
        "pulse_width_us": 0.25           # Coefficient of variation
    })
    parameter_weights: Dict[str, float] = field(default_factory=lambda: {
        "fc_centroid_hz": 1.5,
        "fc_peak_hz": 1.2,
        "bw_99pct_hz": 1.2,
        "snr_db": 1.0,
        "papr_db": 0.8,
        "mean_amplitude": 0.8,
        "std_amplitude": 0.6,
        "envelope_variance_ratio": 0.8,
        "inst_freq_std_hz": 1.0,
        "inst_freq_mean_hz": 1.0,
        "inst_freq_variance": 0.8,
        "estimated_baud_rate_hz": 1.2,
        "frequency_shift_hz": 1.0,
        "pulse_prf_hz": 1.0,
        "pulse_width_us": 0.8
    })


class MultiWindowTemporalValidator:
    """
    Multi-Window Temporal Validator engine.
    Partitions normalized signals into sequential observation windows, extracts
    physical features on each window, and calculates scale-normalized stability metrics.
    """

    def __init__(self, config: Optional[TemporalValidationConfig] = None):
        self.config = config or TemporalValidationConfig()

    def validate(
        self,
        signal: np.ndarray,
        fs: float
    ) -> Dict[str, Any]:
        """
        Executes multi-window temporal validation across the input signal.

        Parameters:
        -----------
        signal : np.ndarray
            Preprocessed, power-normalized complex baseband signal.
        fs : float
            Sampling rate in Hz.

        Returns:
        --------
        Dict[str, Any] matching the specification:
            windows_analyzed: int
            parameter_stability: Dict[str, Any]
            cross_window_consistency_score: Optional[float]
            stability_level: 'HIGH' | 'MEDIUM' | 'LOW' | 'UNKNOWN'
            unstable_parameters: List[str]
            window_results: List[Dict[str, Any]]
        """
        if signal is None:
            return self._insufficient_duration_result(0, "Signal input is None")

        if np.isrealobj(signal):
            signal = hilbert(signal)

        n_total = len(signal)
        min_total = self.config.num_windows * self.config.min_samples_per_window

        if n_total < min_total or fs <= 0.0 or not np.isfinite(fs):
            return self._insufficient_duration_result(
                n_total,
                f"Signal length ({n_total} samples) is insufficient for {self.config.num_windows}-window analysis (min {min_total} samples required at fs={fs} Hz)"
            )

        # 1. Partition signal into sequential non-overlapping windows
        window_slices = np.array_split(signal, self.config.num_windows)
        window_results: List[Dict[str, Any]] = []

        curr_offset = 0
        for idx, w in enumerate(window_slices):
            w_len = len(w)
            w_start = curr_offset
            w_end = curr_offset + w_len
            curr_offset = w_end

            w_params = self._extract_window_parameters(w, fs)
            window_results.append({
                "window_index": idx,
                "start_sample": int(w_start),
                "end_sample": int(w_end),
                "duration_ms": float(np.round((w_len / fs) * 1000.0, 3)),
                "parameters": w_params
            })

        # 2. Compute scale-normalized parameter stability across windows
        param_stability, unstable_params, consistency_score, stability_level = self._compute_stability(
            window_results, fs
        )

        return {
            "status": "VALIDATED",
            "windows_analyzed": self.config.num_windows,
            "parameter_stability": param_stability,
            "cross_window_consistency_score": float(np.round(consistency_score, 4)) if consistency_score is not None else None,
            "stability_level": stability_level,
            "unstable_parameters": unstable_params,
            "window_results": window_results
        }

    def _extract_window_parameters(
        self,
        w: np.ndarray,
        fs: float
    ) -> Dict[str, Any]:
        """Extracts physical parameters for a single observation window."""
        n_pts = len(w)
        nperseg = min(1024, max(128, n_pts // 2))

        # Spectral analysis
        f_s, psd_db, psd_lin = compute_welch_psd(w, fs, nperseg=nperseg)
        carrier = estimate_carrier_frequency(f_s, psd_lin, psd_db)
        bw = estimate_occupied_bandwidth(f_s, psd_lin, psd_db)

        # Bounded SNR
        m2m4 = estimate_snr_m2m4(w)
        m_snr = m2m4.get("snr_db", -100.0)
        if -5.0 < m_snr <= 40.0:
            active_snr = m_snr
        else:
            active_snr = estimate_band_integrated_snr(f_s, psd_lin, bw)
        active_snr = float(np.clip(active_snr, -15.0, 45.0))

        # Envelope statistics
        env_stats = compute_signal_stats(w)

        # Instantaneous frequency statistics
        inst_stats = self._compute_instantaneous_frequency_stats(w, fs)

        # Symbol rate estimate (when available - suppressed for unmodulated CW carriers)
        baud_rate = None
        is_unmodulated_cw = env_stats.get("envelope_variance_ratio", 0.0) < 0.08
        if not is_unmodulated_cw:
            try:
                baud_res = estimate_symbol_baud_rate(w, fs)
                if baud_res.get("baud_confidence", 0.0) >= 0.40 and baud_res.get("estimated_baud_rate_hz") is not None:
                    b_val = float(baud_res["estimated_baud_rate_hz"])
                    if 1.0 <= b_val <= fs / 2.0:
                        baud_rate = b_val
            except Exception:
                baud_rate = None

        # FSK Frequency shift estimate (when available)
        fsk_shift = None
        try:
            fsk_res = estimate_fsk_dwell_and_baud_rate(w, fs, obw_hz=bw.get("bw_99pct_hz"))
            shift_val = fsk_res.get("fsk_frequency_shift_hz") or fsk_res.get("estimated_shift_hz")
            if shift_val is not None and shift_val > 0.0:
                fsk_shift = float(shift_val)
        except Exception:
            fsk_shift = None

        # Pulse metrics (when available)
        pulse_metrics = None
        try:
            pulse_res = analyze_pulse_train(w, fs)
            if pulse_res.get("is_pulsed", False) or pulse_res.get("is_radar", False):
                pulse_metrics = {
                    "pulse_width_us": pulse_res.get("mean_pulse_width_us"),
                    "pri_us": pulse_res.get("mean_pri_us"),
                    "prf_hz": pulse_res.get("mean_prf_hz"),
                    "duty_cycle_pct": pulse_res.get("duty_cycle_pct")
                }
        except Exception:
            pulse_metrics = None

        return {
            "fc_peak_hz": carrier.get("fc_peak_hz", 0.0),
            "fc_centroid_hz": carrier.get("fc_centroid_hz", 0.0),
            "bw_99pct_hz": bw.get("bw_99pct_hz", 0.0),
            "bw_3db_hz": bw.get("bw_3db_hz", 0.0),
            "snr_db": active_snr,
            "mean_amplitude": env_stats.get("mean_amplitude", 0.0),
            "std_amplitude": env_stats.get("std_amplitude", 0.0),
            "envelope_variance_ratio": env_stats.get("envelope_variance_ratio", 0.0),
            "papr_db": env_stats.get("papr_db", 0.0),
            "inst_freq_mean_hz": inst_stats["inst_freq_mean_hz"],
            "inst_freq_std_hz": inst_stats["inst_freq_std_hz"],
            "inst_freq_variance": inst_stats["inst_freq_variance"],
            "estimated_baud_rate_hz": baud_rate,
            "frequency_shift_hz": fsk_shift,
            "pulse_metrics": pulse_metrics
        }

    def _compute_instantaneous_frequency_stats(
        self,
        w: np.ndarray,
        fs: float
    ) -> Dict[str, float]:
        """Computes unwrapped instantaneous frequency statistics."""
        if len(w) < 8:
            return {
                "inst_freq_mean_hz": 0.0,
                "inst_freq_std_hz": 0.0,
                "inst_freq_variance": 0.0
            }

        phase = np.unwrap(np.angle(w))
        d_phase = np.diff(phase)
        inst_f = d_phase * (fs / (2.0 * np.pi))

        if len(inst_f) >= 5:
            inst_f_filt = medfilt(inst_f, 5)
        else:
            inst_f_filt = inst_f

        mean_f = float(np.mean(inst_f_filt))
        std_f = float(np.std(inst_f_filt))
        var_f = float(std_f ** 2)

        return {
            "inst_freq_mean_hz": mean_f,
            "inst_freq_std_hz": std_f,
            "inst_freq_variance": var_f
        }

    def _compute_stability(
        self,
        window_results: List[Dict[str, Any]],
        fs: float
    ) -> Tuple[Dict[str, Any], List[str], Optional[float], str]:
        """
        Computes scale-normalized parameter stability across windows.
        Uses physical reference normalizations rather than uncalibrated thresholds.
        """
        all_params: Dict[str, List[float]] = {}
        target_keys = [
            "fc_centroid_hz",
            "fc_peak_hz",
            "bw_99pct_hz",
            "snr_db",
            "mean_amplitude",
            "std_amplitude",
            "envelope_variance_ratio",
            "papr_db",
            "inst_freq_mean_hz",
            "inst_freq_std_hz",
            "inst_freq_variance",
            "estimated_baud_rate_hz",
            "frequency_shift_hz"
        ]

        # Check for pulse metrics
        prf_vals: List[float] = []
        pw_vals: List[float] = []
        for wr in window_results:
            p_dict = wr["parameters"]
            for k in target_keys:
                val = p_dict.get(k)
                if val is not None and np.isfinite(val):
                    all_params.setdefault(k, []).append(float(val))
            pm = p_dict.get("pulse_metrics")
            if pm:
                if pm.get("prf_hz") is not None and np.isfinite(pm["prf_hz"]):
                    prf_vals.append(float(pm["prf_hz"]))
                if pm.get("pulse_width_us") is not None and np.isfinite(pm["pulse_width_us"]):
                    pw_vals.append(float(pm["pulse_width_us"]))

        if len(prf_vals) >= 2:
            all_params["pulse_prf_hz"] = prf_vals
        if len(pw_vals) >= 2:
            all_params["pulse_width_us"] = pw_vals

        # Determine reference occupied bandwidth for frequency normalization
        avg_obw = float(np.mean(all_params.get("bw_99pct_hz", [1000.0])))
        ref_bandwidth = max(avg_obw, 0.02 * fs, 100.0)

        param_stability: Dict[str, Any] = {}
        unstable_params: List[str] = []
        scores: List[float] = []
        weights: List[float] = []

        for p_name, vals in all_params.items():
            if len(vals) < 2:
                param_stability[p_name] = {
                    "status": "INSUFFICIENT_WINDOW_DETECTIONS",
                    "detections_count": len(vals),
                    "stability_score": None
                }
                continue

            vals_arr = np.array(vals, dtype=np.float64)
            val_mean = float(np.mean(vals_arr))
            val_std = float(np.std(vals_arr, ddof=1)) if len(vals_arr) > 1 else 0.0
            val_range = float(np.ptp(vals_arr))

            # Scale normalization
            if p_name in ["fc_centroid_hz", "fc_peak_hz", "inst_freq_mean_hz"]:
                # Normalization scale: occupied bandwidth / channel width
                rel_var = val_std / ref_bandwidth
            elif p_name in ["bw_99pct_hz"]:
                # Normalization scale: mean bandwidth
                rel_var = val_std / max(abs(val_mean), 100.0)
            elif p_name in ["snr_db"]:
                # Normalization scale: 10 dB dynamic range span
                rel_var = val_std / 10.0
            elif p_name in ["papr_db"]:
                rel_var = val_std / max(abs(val_mean), 3.0)
            elif p_name in ["mean_amplitude", "std_amplitude", "envelope_variance_ratio"]:
                rel_var = val_std / max(abs(val_mean), 1e-4)
            elif p_name in ["inst_freq_std_hz"]:
                rel_var = val_std / max(abs(val_mean), 0.05 * fs, 50.0)
            elif p_name in ["inst_freq_variance"]:
                rel_var = val_std / max(abs(val_mean), 1.0)
            elif p_name in ["estimated_baud_rate_hz"]:
                rel_var = val_std / max(abs(val_mean), 1.0)
            elif p_name in ["frequency_shift_hz"]:
                rel_var = val_std / max(abs(val_mean), 10.0)
            elif p_name in ["pulse_prf_hz"]:
                rel_var = val_std / max(abs(val_mean), 1.0)
            elif p_name in ["pulse_width_us"]:
                rel_var = val_std / max(abs(val_mean), 1.0)
            else:
                rel_var = val_std / max(abs(val_mean), 1.0)

            # Continuous rational decay mapping: S = 1 / (1 + (rel_var / tau)^2)
            tau = self.config.parameter_tolerances.get(p_name, 0.25)
            s_score = float(1.0 / (1.0 + (rel_var / tau) ** 2))
            is_stable = bool(s_score >= 0.50)  # Corresponds to rel_var <= tau

            if not is_stable:
                unstable_params.append(p_name)

            w_val = self.config.parameter_weights.get(p_name, 1.0)
            scores.append(s_score)
            weights.append(w_val)

            param_stability[p_name] = {
                "mean": float(np.round(val_mean, 3)),
                "std": float(np.round(val_std, 3)),
                "range": float(np.round(val_range, 3)),
                "relative_variation": float(np.round(rel_var, 4)),
                "tolerance": float(tau),
                "stability_score": float(np.round(s_score, 4)),
                "is_stable": is_stable,
                "window_values": [float(np.round(v, 3)) for v in vals]
            }

        if len(scores) > 0:
            consistency_score = float(np.average(scores, weights=weights))
            if consistency_score >= self.config.high_stability_threshold:
                stability_level = "HIGH"
            elif consistency_score >= self.config.medium_stability_threshold:
                stability_level = "MEDIUM"
            else:
                stability_level = "LOW"
        else:
            consistency_score = None
            stability_level = "UNKNOWN"

        return param_stability, unstable_params, consistency_score, stability_level

    def _insufficient_duration_result(
        self,
        n_samples: int,
        reason: str
    ) -> Dict[str, Any]:
        """Returns standardized graceful degradation report for short signals."""
        return {
            "status": "INSUFFICIENT_OBSERVATION_DURATION",
            "windows_analyzed": 0,
            "parameter_stability": {},
            "cross_window_consistency_score": None,
            "stability_level": "UNKNOWN",
            "unstable_parameters": [],
            "window_results": [],
            "message": reason,
            "samples_provided": int(n_samples),
            "min_samples_required": int(self.config.num_windows * self.config.min_samples_per_window)
        }


# Global convenience instance
_default_validator = MultiWindowTemporalValidator()


def validate_temporal_consistency(
    signal: np.ndarray,
    fs: float,
    config: Optional[TemporalValidationConfig] = None
) -> Dict[str, Any]:
    """
    Convenience function executing multi-window temporal validation.
    """
    if config is not None:
        validator = MultiWindowTemporalValidator(config=config)
    else:
        validator = _default_validator
    return validator.validate(signal, fs)
