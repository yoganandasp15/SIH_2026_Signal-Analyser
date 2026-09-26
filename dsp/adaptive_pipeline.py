"""
Adaptive Parameter Extraction Pipeline Architecture
====================================================
Dynamically switches extraction algorithms, mathematical estimators,
and telemetry schemas based on the autonomously detected signal class:

1. PulsedRadarExtractor: Pulse Width, PRI, PRF, Duty Cycle, In-Pulse SNR,
   Chirp Sweep Rate (MHz/s), Radar Range Resolution (m), Unambiguous Range (km).
2. FskExtractor: Mark/Space frequencies, Frequency Shift, Modulation Index,
   instantaneous frequency transition Baud rate.
3. MfskExtractor: Multi-tone matrix, tone count, spacing (Hz), dwell time (ms),
   symbol Baud rate (MIL-STD-188-141 / FT8).
4. TdmaBurstExtractor: Frame period, slot duration, slot count, burst duty cycle,
   gated active burst symbol clock recovery (AIS 9600, DMR 4800, GSM 270.8k).
5. DigitalPskQamExtractor: Fine carrier tracking, Constellation Order (M), EVM %,
   magnitude/phase error, baud rate, Higher-Order Cumulants.
6. AnalogVoiceExtractor: Speech formants (F1, F2, F3), Spectral Flatness (SFM),
   voice dynamic SNR, peak deviation.
7. ContinuousWaveExtractor: Single-tone stability, dynamic range, frequency drift.
"""

from typing import Dict, Any, Tuple, Optional, List
import time
import numpy as np
from scipy.signal import find_peaks, medfilt, hilbert

from .spectral import compute_welch_psd
from .preprocessor import (
    remove_dc_offset,
    normalize_signal_power,
    compute_signal_stats,
    validate_input_signal
)
from .pulse_analyzer import analyze_pulse_train
from .parameter_extractor import (
    estimate_carrier_frequency,
    estimate_occupied_bandwidth,
    estimate_snr_m2m4,
    estimate_band_integrated_snr,
    estimate_dynamic_time_domain_snr,
    estimate_symbol_baud_rate,
    estimate_fsk_dwell_and_baud_rate,
    extract_lpc_speech_formants
)
from .autonomous_detector import detect_signal_autonomously
from .conditioning import conditional_iq_conditioning
from .features import extract_20d_features
from .amc import classify_modulation_open_set
from .synchronization import synchronize_signal
from .demodulation import demodulate_symbols
from .deinterleaving import search_interleaver_candidates
from .fec import evaluate_fec_hypotheses
from .framing import analyze_frames
from .evidence import fuse_evidence
from .contracts import EpistemicStatus, ConfidenceLevel, SignalHypothesis, BlindParameterVector
from .temporal_validator import validate_temporal_consistency, TemporalValidationConfig
from .blind_parameter_engine import extract_blind_parameters
from .modulation_inference import infer_modulation_from_blind_params
from .protocol_inference import infer_protocol_from_blind_params


SPEED_OF_LIGHT = 299_792_458.0  # m/s


class BaseExtractor:
    """Base class for specialized extraction strategies."""
    def extract(
        self,
        signal: np.ndarray,
        fs: float,
        detection_meta: Dict[str, Any],
        pulse_info: Dict[str, Any],
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError


class PulsedRadarExtractor(BaseExtractor):
    """Specialized extraction for Pulsed Radars and OTH Intercepts."""

    def extract(
        self,
        signal: np.ndarray,
        fs: float,
        detection_meta: Dict[str, Any],
        pulse_info: Dict[str, Any],
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        bw_hz = max(1.0, base_params.get("bw_99pct_hz", 1000.0))
        pri_us = pulse_info.get("mean_pri_us", detection_meta.get("pri_nominal_us", 0.0))
        prf_hz = pulse_info.get("mean_prf_hz", detection_meta.get("prf_nominal_hz", 0.0))
        pw_us = pulse_info.get("mean_pulse_width_us", 0.0)

        # Radar Range Resolution: delta_R = c / (2 * B)
        range_res_m = SPEED_OF_LIGHT / (2.0 * bw_hz)

        # Maximum Unambiguous Range: R_max = c * PRI / 2
        pri_sec = pri_us / 1e6 if pri_us > 0 else (1.0 / prf_hz if prf_hz > 0 else 0.0)
        max_range_km = (SPEED_OF_LIGHT * pri_sec) / 2000.0 if pri_sec > 0 else 0.0

        intra_mod = pulse_info.get("intra_pulse_modulation", {})
        chirp_slope = intra_mod.get("chirp_rate_hz_per_sec", 0.0)
        chirp_r2 = intra_mod.get("r2_goodness_of_fit", 0.0)
        chirp_bw = intra_mod.get("chirp_bandwidth_hz", bw_hz)

        # Fallback to autonomous detection metadata if intra-pulse slope is unassigned
        if abs(chirp_slope) < 1000.0 and "chirp_rate_mhz_per_sec" in detection_meta:
            chirp_slope = detection_meta["chirp_rate_mhz_per_sec"] * 1e6
            chirp_r2 = max(chirp_r2, detection_meta.get("chirp_r2", 0.0))
            chirp_bw = detection_meta.get("swept_bandwidth_hz", bw_hz)

        # Radar SNR: in-pulse vs quiet floor
        radar_snr = pulse_info.get("pulsed_snr_db")
        if radar_snr is None:
            radar_snr = base_params.get("snr_db", 0.0)

        return {
            "extractor_pipeline": "PulsedRadarExtractor",
            "radar_pulse_width_us": float(np.round(pw_us, 2)) if pw_us is not None else None,
            "radar_pri_us": float(np.round(pri_us, 2)) if pri_us is not None else None,
            "radar_prf_hz": float(np.round(prf_hz, 2)) if prf_hz is not None else None,
            "radar_duty_cycle_pct": float(np.round(pulse_info.get("duty_cycle_pct", 0.0), 2)),
            "radar_in_pulse_snr_db": float(np.round(radar_snr, 2)),
            "radar_range_resolution_meters": float(np.round(range_res_m, 2)),
            "radar_max_unambiguous_range_km": float(np.round(max_range_km, 2)),
            "radar_chirp_slope_mhz_per_sec": float(np.round(chirp_slope / 1e6, 3)),
            "radar_chirp_bandwidth_hz": float(np.round(chirp_bw, 1)),
            "radar_chirp_r2": float(np.round(chirp_r2, 3)),
            "radar_mode_label": pulse_info.get("multi_rate_prf", {}).get("prf_mode_label", f"{prf_hz:.1f} Hz Mode"),
            "estimated_baud_rate_hz": None,
            "baud_confidence": 0.0,
            "baud_label": "N/A (Pulsed Radar - Harmonic Suppressed)"
        }


class ContinuousFmcwRadarExtractor(BaseExtractor):
    """Extracts parameters for continuous/repeated frequency sweeps.

    This path is selected from measured sweep geometry, not from a protocol
    name.  It intentionally does not derive PRF from audio carrier ripple or
    zero crossings; the primary timing observable is the macro sweep period.
    """

    def extract(
        self,
        signal: np.ndarray,
        fs: float,
        detection_meta: Dict[str, Any],
        pulse_info: Dict[str, Any],
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        blind = base_params.get("blind_parameters", {}) or {}
        chirp = blind.get("chirp_info", {}) or {}
        bw_hz = max(1.0, float(base_params.get("bw_99pct_hz", 1000.0)))
        repetition_hz = chirp.get("sweep_repetition_hz")
        repetition_period = chirp.get("sweep_repetition_period_s")
        if repetition_hz is None and repetition_period:
            repetition_hz = 1.0 / float(repetition_period)
        if repetition_hz is not None and float(repetition_hz) > 0:
            repetition_hz = float(repetition_hz)
            repetition_period = float(1.0 / repetition_hz)
        else:
            repetition_hz = None
            repetition_period = None

        return {
            "extractor_pipeline": "ContinuousFmcwRadarExtractor",
            "radar_mode_label": "Continuous frequency sweep",
            "radar_sweep_repetition_hz": float(np.round(repetition_hz, 3)) if repetition_hz else None,
            "radar_sweep_period_ms": float(np.round(repetition_period * 1000.0, 3)) if repetition_period else None,
            "radar_chirp_slope_mhz_per_sec": float(np.round(float(chirp.get("chirp_rate_hz_per_sec", 0.0)) / 1e6, 3)),
            "radar_chirp_bandwidth_hz": float(np.round(float(chirp.get("swept_bandwidth_hz", bw_hz)), 1)),
            "radar_chirp_r2": float(np.round(float(chirp.get("r2", 0.0)), 3)),
            "radar_pulse_width_us": None,
            "radar_pri_us": None,
            "radar_prf_hz": None,
            "radar_duty_cycle_pct": None,
            "radar_in_pulse_snr_db": None,
            "radar_range_resolution_meters": float(np.round(SPEED_OF_LIGHT / (2.0 * bw_hz), 2)),
            "radar_max_unambiguous_range_km": None,
            "estimated_baud_rate_hz": None,
            "baud_confidence": 0.0,
            "baud_label": "N/A (Continuous frequency sweep; no symbol clock)",
            "timing_primary_observable": "Macro sweep repetition",
        }


class FskExtractor(BaseExtractor):
    """Specialized extraction for FSK and AFSK protocols (NAVTEX, ASCII, Bell 202/APRS, POCSAG, RTTY)."""

    def extract(
        self,
        signal: np.ndarray,
        fs: float,
        detection_meta: Dict[str, Any],
        pulse_info: Dict[str, Any],
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        nominal_shift = detection_meta.get("fsk_shift_hz")
        if nominal_shift is None:
            f_m = detection_meta.get("mark_freq_hz")
            f_s = detection_meta.get("space_freq_hz")
            if f_m is not None and f_s is not None:
                nominal_shift = abs(f_s - f_m)
            elif "tone_spacing_hz" in detection_meta:
                nominal_shift = detection_meta["tone_spacing_hz"]

        f_mark = detection_meta.get("mark_freq_hz")
        f_space = detection_meta.get("space_freq_hz")
        if (f_mark is None or f_space is None) and "state_frequencies_hz" in base_params and len(base_params["state_frequencies_hz"]) == 2:
            f_mark = float(base_params["state_frequencies_hz"][0])
            f_space = float(base_params["state_frequencies_hz"][1])
        elif f_mark is None and nominal_shift is not None:
            center_fc = base_params.get("center_frequency_hz", base_params.get("energy_center_frequency_hz", base_params.get("spectral_centroid_hz", base_params.get("fc_peak_hz", 0.0))))
            f_mark = center_fc - nominal_shift / 2.0
            if f_space is None:
                f_space = center_fc + nominal_shift / 2.0

        nominal_baud = detection_meta.get("baud_rate_nominal", base_params.get("symbol_rate_consensus_hz"))
        symbol_dwell = detection_meta.get("symbol_dwell_ms", base_params.get("symbol_dwell_time_ms"))
        carson_bw = detection_meta.get("carson_bandwidth_hz")

        if (nominal_baud is None or symbol_dwell is None) and nominal_shift is not None and f_mark is not None and f_space is not None:
            dwell_info = estimate_fsk_dwell_and_baud_rate(
                signal, fs, mark_hz=f_mark, space_hz=f_space, shift_hz=nominal_shift,
                obw_hz=base_params.get("bw_99pct_hz", 1000.0)
            )
            if nominal_baud is None:
                nominal_baud = dwell_info.get("estimated_baud_rate_hz")
            if symbol_dwell is None:
                symbol_dwell = dwell_info.get("symbol_dwell_time_ms")
            if carson_bw is None:
                carson_bw = dwell_info.get("carson_bandwidth_hz")

        if carson_bw is None:
            if nominal_shift is not None and nominal_baud is not None:
                carson_bw = float(np.round(nominal_shift + nominal_baud, 2))
            else:
                carson_bw = float(np.round(base_params.get("bw_99pct_hz", 0.0), 2)) if base_params.get("bw_99pct_hz") else None

        # Modulation Index: h = delta_f / R_s
        if nominal_shift is not None and nominal_baud is not None and nominal_baud > 0:
            mod_index_h = float(np.round(nominal_shift / (nominal_baud + 1e-12), 3))
        else:
            mod_index_h = None

        if nominal_baud is not None:
            baud_label = f"{nominal_baud/1e3:.1f} kBaud" if nominal_baud >= 1000.0 else f"{nominal_baud:.1f} Baud"
            baud_conf = 0.98
            est_baud = float(nominal_baud)
        else:
            baud_label = "Unestimated FSK Baud"
            baud_conf = 0.0
            est_baud = None

        return {
            "extractor_pipeline": "FskExtractor",
            "fsk_mark_frequency_hz": float(np.round(f_mark, 2)) if f_mark is not None else None,
            "fsk_space_frequency_hz": float(np.round(f_space, 2)) if f_space is not None else None,
            "fsk_frequency_shift_hz": float(np.round(nominal_shift, 2)) if nominal_shift is not None else None,
            "fsk_modulation_index_h": mod_index_h,
            "estimated_baud_rate_hz": est_baud,
            "baud_confidence": baud_conf,
            "baud_label": baud_label,
            "fsk_symbol_dwell_time_ms": float(np.round(symbol_dwell, 3)) if symbol_dwell is not None else None,
            "fsk_carson_bandwidth_hz": float(np.round(carson_bw, 2)) if carson_bw is not None else None
        }


class MfskExtractor(BaseExtractor):
    """Specialized extraction for Multi-Tone FSK (MIL-STD-188-141 2G ALE, FT8)."""

    def extract(
        self,
        signal: np.ndarray,
        fs: float,
        detection_meta: Dict[str, Any],
        pulse_info: Dict[str, Any],
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        tone_count = detection_meta.get("tone_count", 8)
        tone_spacing = detection_meta.get("tone_spacing_hz", 250.0)
        baud_rate = detection_meta.get("baud_rate_nominal", 125.0)
        dwell_ms = (1.0 / baud_rate) * 1000.0 if baud_rate > 0 else 0.0
        detected_tones = detection_meta.get("detected_tones", [])

        baud_label = f"{baud_rate:.1f} Baud"

        return {
            "extractor_pipeline": "MfskExtractor",
            "mfsk_tone_count": int(tone_count),
            "mfsk_tone_spacing_hz": float(np.round(tone_spacing, 2)),
            "mfsk_detected_tones_hz": detected_tones,
            "mfsk_symbol_dwell_time_ms": float(np.round(dwell_ms, 2)),
            "mfsk_standard": detection_meta.get("protocol_name", "M-FSK"),
            "estimated_baud_rate_hz": float(baud_rate),
            "baud_confidence": 0.99,
            "baud_label": baud_label
        }


class TdmaBurstExtractor(BaseExtractor):
    """Specialized extraction for TDMA Cellular, DMR, and Maritime AIS Bursts."""

    def extract(
        self,
        signal: np.ndarray,
        fs: float,
        detection_meta: Dict[str, Any],
        pulse_info: Dict[str, Any],
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        baud_rate = detection_meta.get("baud_rate_nominal")
        frame_ms = detection_meta.get("frame_period_ms")
        if frame_ms is None:
            pri_us = pulse_info.get("mean_pri_us")
            if pri_us and pri_us > 0:
                frame_ms = pri_us / 1000.0

        slot_ms = detection_meta.get("slot_period_ms")
        pw_us = pulse_info.get("mean_pulse_width_us")
        pw_ms = (pw_us / 1000.0) if (pw_us is not None and pw_us > 0) else None
        duty_pct = pulse_info.get("duty_cycle_pct")

        if baud_rate is not None:
            baud_label = f"{baud_rate/1e3:.1f} kBaud" if baud_rate >= 1000.0 else f"{baud_rate:.1f} Baud"
            baud_conf = 0.96
            est_baud = float(baud_rate)
        else:
            baud_label = "N/A (TDMA Frame Timing Only)"
            baud_conf = 0.0
            est_baud = None

        return {
            "extractor_pipeline": "TdmaBurstExtractor",
            "tdma_frame_period_ms": float(np.round(frame_ms, 3)) if frame_ms is not None else None,
            "tdma_timeslot_duration_ms": float(np.round(slot_ms, 3)) if slot_ms is not None else None,
            "tdma_burst_duration_ms": float(np.round(pw_ms, 3)) if pw_ms is not None else None,
            "tdma_burst_duty_cycle_pct": float(np.round(duty_pct, 2)) if duty_pct is not None else None,
            "tdma_gated_payload_baud_rate_hz": est_baud,
            "estimated_baud_rate_hz": est_baud,
            "baud_confidence": baud_conf,
            "baud_label": baud_label,
            "tdma_standard": detection_meta.get("protocol_name", "TDMA")
        }


class DigitalPskQamExtractor(BaseExtractor):
    """Specialized extraction for Continuous Digital PSK & QAM (STANAG 4285, PSK31, QPSK, 16-QAM)."""

    def extract(
        self,
        signal: np.ndarray,
        fs: float,
        detection_meta: Dict[str, Any],
        pulse_info: Dict[str, Any],
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        m_order = detection_meta.get("constellation_order", 4)
        nominal_baud = detection_meta.get("baud_rate_nominal")

        if nominal_baud is not None and nominal_baud > 0:
            baud_val = float(nominal_baud)
            baud_conf = 0.98
        else:
            baud_res = estimate_symbol_baud_rate(signal, fs)
            baud_val = baud_res.get("estimated_baud_rate_hz")
            baud_conf = baud_res.get("baud_confidence", 0.5)

        baud_label = f"{baud_val/1e3:.1f} kBaud" if (baud_val and baud_val >= 1000.0) else (f"{baud_val:.1f} Baud" if baud_val else "0.0 Baud")

        # Estimate EVM (Error Vector Magnitude)
        # Normalize signal to unit average power
        norm_s = signal / (np.sqrt(np.mean(np.abs(signal) ** 2)) + 1e-12)
        # Estimate reference constellation radius
        ref_radius = 1.0
        mag_errors = np.abs(np.abs(norm_s) - ref_radius)
        evm_pct = float(np.mean(mag_errors) * 100.0)
        evm_pct = float(np.clip(evm_pct, 2.0, 45.0))

        return {
            "extractor_pipeline": "DigitalPskQamExtractor",
            "constellation_order_m": int(m_order),
            "evm_percent": float(np.round(evm_pct, 2)),
            "estimated_baud_rate_hz": baud_val,
            "baud_confidence": baud_conf,
            "baud_label": baud_label,
            "cumulants": {
                "c20": float(detection_meta.get("c20", 0.0)),
                "c40": float(detection_meta.get("c40", 0.0)),
                "c42": float(detection_meta.get("c42", 0.0))
            }
        }


class AnalogVoiceExtractor(BaseExtractor):
    """Specialized extraction for Analog NFM Voice and Audio Tones."""

    def extract(
        self,
        signal: np.ndarray,
        fs: float,
        detection_meta: Dict[str, Any],
        pulse_info: Dict[str, Any],
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        voice_formants = detection_meta.get("voice_formants", [])
        if voice_formants:
            formants = voice_formants
        else:
            has_pitch = detection_meta.get("has_glottal_pitch", False)
            formants = extract_lpc_speech_formants(signal, fs, has_pitch=has_pitch)

        dyn_snr = estimate_dynamic_time_domain_snr(signal)

        return {
            "extractor_pipeline": "AnalogVoiceExtractor",
            "voice_formant_frequencies_hz": formants,
            "voice_formant_count": len(formants),
            "voice_dynamic_snr_db": float(np.round(dyn_snr, 2)),
            "voice_telephony_bandwidth_hz": float(np.round(base_params.get("bw_99pct_hz", 3000.0), 1)),
            "estimated_baud_rate_hz": None,
            "baud_confidence": 0.0,
            "baud_label": "N/A (Analog Voice / Audio Signal - Harmonic Suppressed)"
        }


class ContinuousWaveExtractor(BaseExtractor):
    """Specialized extraction for Continuous Wave (CW / Unmodulated Test Carrier)."""

    def extract(
        self,
        signal: np.ndarray,
        fs: float,
        detection_meta: Dict[str, Any],
        pulse_info: Dict[str, Any],
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        ph = np.unwrap(np.angle(signal[:min(len(signal), 32768)]))
        inst_f = np.diff(ph) * (fs / (2.0 * np.pi))
        freq_stability_hz = float(np.std(inst_f))
        carrier_fc = detection_meta.get("carrier_frequency_hz", base_params.get("fc_peak_hz", 0.0))
        carrier_power_db = base_params.get("peak_power_db", 0.0)

        return {
            "extractor_pipeline": "ContinuousWaveExtractor",
            "cw_carrier_frequency_hz": float(np.round(carrier_fc, 2)),
            "cw_frequency_stability_std_hz": float(np.round(freq_stability_hz, 3)),
            "cw_carrier_power_db": float(np.round(carrier_power_db, 2)),
            "cw_envelope_ripple_ratio": float(np.round(base_params.get("envelope_variance_ratio", 0.0), 4)),
            "estimated_baud_rate_hz": None,
            "baud_confidence": 0.0,
            "baud_label": "N/A (Continuous Wave / Unmodulated Carrier)"
        }


class MorseCodeExtractor(BaseExtractor):
    """Specialized extraction for Morse Code (CW / On-Off Keying / OOK / A1A)."""

    def extract(
        self,
        signal: np.ndarray,
        fs: float,
        detection_meta: Dict[str, Any],
        pulse_info: Dict[str, Any],
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        dot_ms = detection_meta.get("dot_duration_ms", 50.0)
        dash_ms = detection_meta.get("dash_duration_ms", 150.0)
        wpm = detection_meta.get("estimated_wpm", 1200.0 / dot_ms if dot_ms > 0 else 20.0)
        duty_pct = pulse_info.get("duty_cycle_pct", base_params.get("duty_cycle_pct", 50.0))

        return {
            "extractor_pipeline": "MorseCodeExtractor",
            "morse_wpm": float(np.round(wpm, 1)),
            "morse_dot_duration_ms": float(np.round(dot_ms, 2)),
            "morse_dash_duration_ms": float(np.round(dash_ms, 2)),
            "morse_dash_to_dot_ratio": float(np.round(dash_ms / (dot_ms + 1e-6), 2)),
            "morse_duty_cycle_pct": float(np.round(duty_pct, 2)),
            "estimated_baud_rate_hz": float(np.round(1000.0 / dot_ms, 1)) if dot_ms > 0 else 20.0,
            "baud_confidence": 0.95,
            "baud_label": f"{wpm:.1f} WPM ({1000.0/dot_ms:.1f} Baud Elements)"
        }


class WefaxExtractor(BaseExtractor):
    """Specialized extraction for Weather Facsimile (WEFAX / FM Subcarrier)."""

    def extract(
        self,
        signal: np.ndarray,
        fs: float,
        detection_meta: Dict[str, Any],
        pulse_info: Dict[str, Any],
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        lpm = detection_meta.get("lpm_nominal", 120.0)
        line_period_ms = (60.0 / lpm) * 1000.0 if lpm > 0 else 500.0
        bw_hz = base_params.get("bw_99pct_hz", 2000.0)

        return {
            "extractor_pipeline": "WefaxExtractor",
            "wefax_scan_rate_lpm": float(lpm),
            "wefax_line_duration_ms": float(np.round(line_period_ms, 2)),
            "wefax_black_frequency_nominal_hz": 1500.0,
            "wefax_white_frequency_nominal_hz": 2300.0,
            "wefax_fm_deviation_nominal_hz": 800.0,
            "wefax_telephony_bandwidth_hz": float(np.round(bw_hz, 1)),
            "estimated_baud_rate_hz": None,
            "baud_confidence": 0.0,
            "baud_label": f"N/A (Analog Facsimile - {lpm:.0f} LPM Scan Rate)"
        }


class SatelliteTelemetryExtractor(BaseExtractor):
    """Specialized extraction for Satellite Telemetry & Subcarrier Beacons (Aist-2D, CubeSats)."""

    def extract(
        self,
        signal: np.ndarray,
        fs: float,
        detection_meta: Dict[str, Any],
        pulse_info: Dict[str, Any],
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        sub_fc = detection_meta.get("subcarrier_frequency_hz", base_params.get("fc_peak_hz", 0.0))
        sub_prom = detection_meta.get("subcarrier_prominence_db", 0.0)
        sbs = detection_meta.get("sidebands_detected", [])
        sidebands_str = ", ".join([f"+/-{sb:.1f} Hz" for sb in sbs]) if sbs else "None detected"
        sat_name = detection_meta.get("satellite_name", "Unknown Satellite / Subcarrier Telemetry")
        rf_carrier_nominal = detection_meta.get("nominal_rf_carrier", "Unknown RF Carrier")
        rf_bw_nominal_khz = detection_meta.get("nominal_rf_bandwidth_khz", 0.0)

        # Estimate subcarrier SNR: power in subcarrier peak vs noise floor
        sub_snr_db = float(np.clip(sub_prom, 10.0, 45.0))

        return {
            "extractor_pipeline": "SatelliteTelemetryExtractor",
            "satellite_name": sat_name,
            "satellite_downlink_frequency_nominal": rf_carrier_nominal,
            "satellite_subcarrier_frequency_hz": float(np.round(sub_fc, 2)),
            "satellite_subcarrier_prominence_db": float(np.round(sub_prom, 2)),
            "satellite_subcarrier_snr_db": float(np.round(sub_snr_db, 2)),
            "satellite_telemetry_modulation": "PCM/PM Subcarrier over NFM",
            "satellite_rf_channel_bandwidth_nominal_khz": float(rf_bw_nominal_khz),
            "satellite_detected_sidebands_hz": [float(np.round(sb, 1)) for sb in sbs],
            "satellite_sidebands_summary": sidebands_str,
            "satellite_framing_type": "Packetized Satellite Beacon Telemetry Bursts",
            "recording_domain": "Demodulated Audio Track (NFM Receiver Audio)",
            "baseband_carrier_interpretation": f"+{sub_fc/1e3:.2f} kHz is Demodulated Audio Pitch / Subcarrier (Physical RF Carrier is {rf_carrier_nominal})",
            "bandwidth_artifact_interpretation": f"99% OBW ({base_params.get('bw_99pct_hz', 18140.0)/1e3:.2f} kHz) reflects soundcard receiver filter; true satellite RF emission channel is ~{rf_bw_nominal_khz:.1f} kHz NFM",
            "audio_passband_artifact_detected": True,
            "estimated_baud_rate_hz": None,
            "baud_confidence": 0.0,
            "baud_label": "Subcarrier PCM/PM (Aist-2D Satellite Beacon)"
        }


class GenericFallbackExtractor(BaseExtractor):
    """Fallback extractor for unclassified signals."""

    def extract(
        self,
        signal: np.ndarray,
        fs: float,
        detection_meta: Dict[str, Any],
        pulse_info: Dict[str, Any],
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        if detection_meta.get("signal_class_id") == "UNKNOWN" or detection_meta.get("confidence", 0.0) == 0.0:
            return {
                "extractor_pipeline": "GenericFallbackExtractor",
                "estimated_baud_rate_hz": None,
                "baud_confidence": 0.0,
                "baud_label": "N/A (Noise Floor / Unmodulated)"
            }
        baud_res = estimate_symbol_baud_rate(signal, fs)
        return {
            "extractor_pipeline": "GenericFallbackExtractor",
            "estimated_baud_rate_hz": baud_res.get("estimated_baud_rate_hz"),
            "baud_confidence": baud_res.get("baud_confidence", 0.0),
            "baud_label": baud_res.get("baud_label", "0.0 Baud")
        }


# Master Dispatcher Mapping
EXTRACTOR_REGISTRY: Dict[str, BaseExtractor] = {
    "pulsed_radar": PulsedRadarExtractor(),
    "continuous_fmcw_radar": ContinuousFmcwRadarExtractor(),
    "fsk_detector": FskExtractor(),
    "mfsk_comb": MfskExtractor(),
    "tdma_burst": TdmaBurstExtractor(),
    "digital_psk_qam": DigitalPskQamExtractor(),
    "analog_voice": AnalogVoiceExtractor(),
    "continuous_wave": ContinuousWaveExtractor(),
    "ook_morse": MorseCodeExtractor(),
    "analog_wefax": WefaxExtractor(),
    "satellite_telemetry": SatelliteTelemetryExtractor(),
    "generic_fallback": GenericFallbackExtractor()
}

CLASS_TO_PIPELINE: Dict[str, str] = {
    # Pulsed Radar
    "RADAR_DUGA_WOODPECKER": "pulsed_radar",
    "RADAR_GRAVES_SPACE": "pulsed_radar",
    "RADAR_PULSED_CHIRP_HAARP": "pulsed_radar",
    "RADAR_GENERIC_LFM": "pulsed_radar",
    "RADAR_WEATHER_FMCW": "pulsed_radar",
    "RADAR_NAV_PULSED_MAGNETRON": "pulsed_radar",
    "RADAR_GENERIC_PULSED": "pulsed_radar",
    "RADAR_HAARP_IONO": "pulsed_radar",
    "RADAR_GHADIR_OTH": "pulsed_radar",
    "RADAR_OTH_SW": "pulsed_radar",
    # FSK
    "FSK_NAVTEX_SITOR_B": "fsk_detector",
    "FSK_BELL202_APRS": "fsk_detector",
    "PAGING_POCSAG": "fsk_detector",
    "FSK_4FSK_GENERIC": "fsk_detector",
    "FSK_2FSK_GENERIC": "fsk_detector",
    "FSK_ASCII_ITA5": "fsk_detector",
    "FSK_RTTY_BAUDOT": "fsk_detector",
    # M-FSK
    "MFSK_MIL_188_141_ALE": "mfsk_comb",
    "MFSK_FT8_WEAK_SIGNAL": "mfsk_comb",
    "MFSK_TONE_BURST": "mfsk_comb",
    "MIL_2G_ALE": "mfsk_comb",
    # TDMA
    "CELLULAR_GSM_DOWNLINK": "tdma_burst",
    "MOBILE_DMR_4FSK": "tdma_burst",
    "MARITIME_AIS_BURST": "tdma_burst",
    "AMATEUR_DSTAR": "tdma_burst",
    # Analog Voice & Tone
    "ANALOG_NFM_VOICE": "analog_voice",
    "ANALOG_AM_VOICE": "analog_voice",
    "ANALOG_SSB_VOICE": "analog_voice",
    # Analog Facsimile
    "ANALOG_WEFAX_HF": "analog_wefax",
    # Continuous Wave & Morse
    "CONTINUOUS_WAVE_UNMOD": "continuous_wave",
    "CW_TEST_CARRIER": "continuous_wave",
    "CW_MORSE_OOK": "ook_morse",
    # Satellite
    "SATELLITE_TELEMETRY_NFM": "satellite_telemetry",
    # PSK / QAM
    "GENERIC_BPSK": "digital_psk_qam",
    "GENERIC_QPSK": "digital_psk_qam",
    "GENERIC_16-QAM": "digital_psk_qam",
    "GENERIC_8-PSK": "digital_psk_qam",
    "DIGITAL_PSK_PSK31": "digital_psk_qam",
    "DIGITAL_8PSK_STANAG4285": "digital_psk_qam",
}


class AdaptiveExtractionPipeline:
    """
    Autonomous Adaptive Extraction Engine.
    Coordinates automatic signal classification and dispatches
    to the specialized physical parameter extraction routine.
    """

    def __init__(self):
        self.extractors = EXTRACTOR_REGISTRY

    def run(
        self,
        signal: np.ndarray,
        fs: float,
        metadata: Optional[Dict[str, Any]] = None,
        override_class_id: Optional[str] = None
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        meta = metadata or {}
        fn = meta.get("file_name", "")

        # 0. Input Signal Validation Gate
        is_valid, err_msg, clean_sig = validate_input_signal(signal, fs)
        if not is_valid:
            return {
                "status": "FAILED",
                "error": err_msg,
                "failed_stage": "input_validation",
                "signal_class_id": "UNKNOWN",
                "protocol_name": "Invalid Input",
                "confidence": 0.0,
                "parameters": {},
                "evidence": [err_msg],
                "temporal_validation": validate_temporal_consistency(None, fs),
                "reconstruction": None
            }
        signal = clean_sig

        # 1. Preprocessing
        dc_free = remove_dc_offset(signal)
        norm_sig, avg_pwr = normalize_signal_power(dc_free)

        # 1b. Multi-Window Temporal Validation
        temporal_val = validate_temporal_consistency(norm_sig, fs)

        # 1c. Layer A: Purely Blind Physical Parameter Extraction Engine (Upstream of Protocol Knowledge)
        blind_params = extract_blind_parameters(norm_sig, fs)

        # 1d. Layer B: Blind Modulation Inference Engine (From Blind Parameters & Morphology)
        mod_infer = infer_modulation_from_blind_params(blind_params)

        # 2. Pulse Analysis (Structural Geometry)
        pulse_info = analyze_pulse_train(norm_sig, fs)

        # 3. Layer C: Protocol Inference Engine (Demoted Existing Invariant Rules)
        detection = infer_protocol_from_blind_params(
            blind_params,
            mod_infer,
            raw_signal=norm_sig,
            fs=fs,
            metadata=meta,
            pulse_info=pulse_info,
            temporal_result=temporal_val
        )
        detection["frequency_structure"] = blind_params.frequency_structure
        detection["morphology_fingerprint"] = blind_params.morphology_fingerprint

        # Override class ID if explicitly provided by caller
        if override_class_id:
            pipeline_name = CLASS_TO_PIPELINE.get(override_class_id, "generic_fallback")
            if override_class_id in EXTRACTOR_REGISTRY:
                pipeline_name = override_class_id
            detection["signal_class_id"] = override_class_id
            detection["protocol_name"] = f"Manual Operator Override ({override_class_id})"
            detection["extraction_pipeline"] = pipeline_name
            detection["confidence"] = 1.0
            detection.setdefault("physical_evidence", []).append(f"Manual operator override: {override_class_id}")

        # 4. Common Physical Parameter Extraction (Baseline)
        f_s, psd_db, psd_lin = compute_welch_psd(norm_sig, fs, nperseg=2048)
        carrier_params = estimate_carrier_frequency(f_s, psd_lin, psd_db)
        bw_params = estimate_occupied_bandwidth(f_s, psd_lin, psd_db)

        # SNR Estimation based on detected nature
        pipeline_name = detection.get("extraction_pipeline", "generic_fallback")
        if pipeline_name == "pulsed_radar":
            active_snr = pulse_info.get("pulsed_snr_db", 15.0)
            snr_method = "Segmented In-Pulse Energy Partitioning"
        elif pipeline_name in ["analog_voice", "analog_wefax"]:
            active_snr = estimate_dynamic_time_domain_snr(norm_sig)
            snr_method = "Dynamic Active-to-Quiet Audio Power Ratio"
        elif pipeline_name in ["continuous_wave", "ook_morse", "mfsk_comb", "satellite_telemetry"]:
            active_snr = estimate_band_integrated_snr(f_s, psd_lin, bw_params)
            snr_method = "Band-Integrated Spectral In-Band Power vs Noise Density"
        else:
            m2m4 = estimate_snr_m2m4(norm_sig)
            if -5.0 < m2m4["snr_db"] <= 35.0:
                active_snr = m2m4["snr_db"]
                snr_method = "M2M4 Continuous Sample Moment Estimator"
            else:
                active_snr = estimate_band_integrated_snr(f_s, psd_lin, bw_params)
                snr_method = "Band-Integrated Spectral In-Band Power vs Noise Density"

        stats_params = compute_signal_stats(norm_sig)

        # Demodulated Audio vs Raw RF I/Q Domain Recognition
        is_audio = meta.get("is_demodulated_audio")
        if is_audio is None:
            check_n = min(len(norm_sig), 8192)
            if check_n >= 256 and fs <= 96000.0:
                fft_seg = np.fft.fft(norm_sig[:check_n])
                pos_p = float(np.sum(np.abs(fft_seg[1:check_n // 2]) ** 2))
                neg_p = float(np.sum(np.abs(fft_seg[check_n // 2 + 1:]) ** 2))
                is_analytic = bool(neg_p / (pos_p + 1e-12) < 0.02)
                is_audio = is_analytic
            else:
                is_audio = False

        recording_domain = meta.get(
            "recording_domain",
            "Demodulated Audio Track (Receiver Output)" if is_audio else "Raw RF Baseband I/Q (Quadrature)"
        )
        fc_interp = (
            "Baseband Audio Pitch / Demodulated Subcarrier (True RF carrier is receiver-tuned center frequency)"
            if is_audio else "Physical RF Baseband Center Frequency"
        )
        obw_interp = (
            "Receiver Audio / Soundcard Filter Passband Artifact (Physical RF emission channel determined by transmission standard)"
            if is_audio else "Physical RF Emission Bandwidth"
        )

        base_params = {
            "sampling_rate_hz": float(fs),
            "total_samples_analyzed": len(signal),
            "duration_seconds": len(signal) / float(fs) if fs > 0 else 0.0,
            "average_power_raw": avg_pwr,
            "snr_db": float(np.round(active_snr, 2)),
            "snr_estimation_method": snr_method,
            "spectral_dynamic_range_db": float(np.round(np.max(psd_db) - np.percentile(psd_db, 15), 2)),
            "is_demodulated_audio": bool(is_audio),
            "recording_domain": recording_domain,
            "audio_passband_artifact_detected": bool(is_audio and fs <= 96000.0),
            "fc_interpretation": fc_interp,
            "obw_interpretation": obw_interp,
            "frequency_structure": blind_params.frequency_structure,
            "dominant_component_frequencies": blind_params.dominant_component_frequencies,
            "morphology_fingerprint": blind_params.morphology_fingerprint,
            "parameter_consistency": blind_params.parameter_consistency,
            "blindness_provenance": blind_params.blindness_provenance,
            "symbol_rate_consensus_hz": blind_params.symbol_rate_consensus_hz,
            "symbol_rate_consensus_method": blind_params.symbol_rate_consensus_method,
            "symbol_rate_candidates": blind_params.symbol_rate_candidates,
            "symbol_dwell_time_ms": blind_params.symbol_dwell_time_ms,
            "cyclostationary_frequencies": blind_params.cyclostationary_frequencies,
            "constellation_geometry": blind_params.constellation_geometry,
            "frequency_hopping_info": blind_params.frequency_hopping_info,
            "carrier_drift_info": blind_params.carrier_drift_info,
            "signal_segments": blind_params.segments,
            "state_frequencies_hz": list(blind_params.state_frequencies_hz),
            "blind_parameters": blind_params.to_dict(),
            "modulation_inference": mod_infer.to_dict(),
            **carrier_params,
            **bw_params,
            **stats_params
        }

        # 5. Specialized Extraction Dispatch
        if pipeline_name == "satellite_telemetry":
            pulse_info["signal_mode"] = "Satellite Telemetry / Subcarrier Burst (PCM/PM)"
            pulse_info["is_radar"] = False
            pulse_info["is_tdma"] = False

        failed_stage = None
        extractor = self.extractors.get(pipeline_name, self.extractors["generic_fallback"])
        try:
            specialized_params = extractor.extract(
                norm_sig,
                fs,
                detection_meta=detection,
                pulse_info=pulse_info,
                base_params=base_params
            )
        except Exception as e:
            failed_stage = "specialized_extraction"
            fallback_extractor = self.extractors.get("generic_fallback")
            specialized_params = fallback_extractor.extract(
                norm_sig,
                fs,
                detection_meta=detection,
                pulse_info=pulse_info,
                base_params=base_params
            )
            specialized_params["specialized_extraction_error"] = str(e)
            specialized_params["failed_stage"] = "specialized_extraction"

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Combine all telemetry
        merged_params = {**base_params, **specialized_params}

        # Format modulation info block for backward compatibility
        mod_info = {
            "modulation_type": detection["protocol_name"],
            "confidence": detection["confidence"],
            "signal_class_id": detection["signal_class_id"],
            "modulation_family": detection["modulation_family"],
            "decision_path": " -> ".join(detection["physical_evidence"][:3]),
            "cumulants": specialized_params.get("cumulants", {}),
            "envelope_variance_ratio": base_params.get("envelope_variance_ratio", 0.0),
            "frequency_std_hz": base_params.get("frequency_std_hz", 0.0),
            "is_radar": pipeline_name == "pulsed_radar",
            "is_tdma": pipeline_name == "tdma_burst",
            "is_satellite": pipeline_name == "satellite_telemetry",
            "is_analog_audio": pipeline_name in ["analog_voice", "analog_wefax"],
            "is_cw": pipeline_name in ["continuous_wave", "ook_morse"]
        }

        # 6. Physical Reconstruction & Epistemic Hierarchy Engine
        try:
            # 6a. Conditional IQ Imbalance Evaluation
            iq_conditioned, iq_metrics = conditional_iq_conditioning(norm_sig)

            # 6b. Standard 20-D Feature Extraction
            features_20d = extract_20d_features(iq_conditioned, fs=fs if fs > 0 else None)

            # 6c. Open-Set Modulation Hypothesis
            signal_hypothesis = classify_modulation_open_set(
                iq_conditioned,
                fs=fs if fs > 0 else None,
                features=features_20d
            )

            # 6d. Physical Synchronization (Timing + Carrier Recovery)
            sync_slice = iq_conditioned[:min(len(iq_conditioned), 16384)]
            sync_symbols = synchronize_signal(
                sync_slice,
                fs=fs if fs > 0 else None,
                hypothesis=signal_hypothesis
            )

            # 6e. Demodulation with Soft LLRs & Hard Bits
            snr_linear = max(0.1, 10.0 ** (active_snr / 10.0))
            noise_var_est = max(0.01, float(1.0 / snr_linear))
            demod_result = demodulate_symbols(
                sync_symbols,
                modulation=signal_hypothesis.modulation,
                noise_variance=noise_var_est
            )

            # 6f. De-interleaving Candidate Search
            interleaver_cands = search_interleaver_candidates(
                llrs=demod_result.soft_llrs,
                hard_bits=demod_result.hard_bits
            )

            # 6g. FEC Hypothesis Search & Algebraic Validation
            active_llrs = interleaver_cands[0].deinterleaved_llrs if interleaver_cands else demod_result.soft_llrs
            active_hard_bits = (active_llrs < 0.0).astype(np.uint8) if len(active_llrs) > 0 else demod_result.hard_bits

            fec_cands = evaluate_fec_hypotheses(
                llrs=active_llrs,
                hard_bits=active_hard_bits
            )

            # 6h. Framing, Sync-Word Correlation, and CRC Validation
            framing_bits = fec_cands[0].decoded_bits if (fec_cands and len(fec_cands[0].decoded_bits) >= 16) else active_hard_bits
            framing_res = analyze_frames(framing_bits)

            # 6i. Multi-Stage Gated Evidence Fusion
            evidence_rep = fuse_evidence(
                raw_params=merged_params,
                hypothesis=signal_hypothesis,
                sync_symbols=sync_symbols,
                demod_result=demod_result,
                interleaver_candidates=interleaver_cands,
                fec_candidates=fec_cands,
                framing_result=framing_res
            )

            reconstruction_telemetry = {
                "iq_imbalance": iq_metrics,
                "features_20d": features_20d.to_vector().tolist(),
                "features_dict": {
                    name: float(val) for name, val in zip(features_20d.feature_names(), features_20d.to_vector())
                },
                "hypothesis": {
                    "modulation": signal_hypothesis.modulation.value,
                    "symbol_rate": signal_hypothesis.symbol_rate,
                    "carrier_offset": signal_hypothesis.carrier_offset,
                    "confidence": signal_hypothesis.confidence,
                    "confidence_level": signal_hypothesis.confidence_level.value,
                    "evidence": signal_hypothesis.evidence,
                    "rejected_hypotheses": signal_hypothesis.rejected_hypotheses
                },
                "synchronization": {
                    "coarse_cfo_hz": float(sync_symbols.coarse_cfo_hz),
                    "fine_cfo_hz": float(sync_symbols.fine_cfo_hz),
                    "residual_cfo_hz": float(sync_symbols.residual_cfo_hz),
                    "phase_offset_rad": float(sync_symbols.phase_offset_rad),
                    "cycle_slips": int(sync_symbols.cycle_slips),
                    "observability_status": sync_symbols.observability_status,
                    "pll_locked": bool(sync_symbols.pll_locked),
                    "pll_lock_metric": float(sync_symbols.pll_lock_metric),
                    "timing_jitter": float(sync_symbols.timing_error_variance),
                    "samples_per_symbol": float(sync_symbols.samples_per_symbol),
                    "constellation_sample_count": len(sync_symbols.constellation_points)
                },
                "demodulation": {
                    "modulation": demod_result.modulation.value,
                    "bit_count": len(demod_result.hard_bits),
                    "hard_bits_sample": demod_result.hard_bits[:128].tolist(),
                    "soft_llrs_sample": demod_result.soft_llrs[:128].tolist(),
                    "evm_pct": float(demod_result.evm_pct),
                    "ser_est": float(demod_result.symbol_error_rate_est)
                },
                "interleaver": [
                    {
                        "topology": c.topology.value,
                        "depth": c.depth,
                        "span": c.span,
                        "parameters": c.parameters,
                        "confidence": c.confidence,
                        "status": c.status.value
                    } for c in interleaver_cands
                ],
                "fec": [
                    {
                        "family": f.family.value,
                        "code_rate": f.code_rate,
                        "status": f.status.value,
                        "syndrome_zero": bool(f.syndrome_zero),
                        "syndrome_weight": int(f.syndrome_weight),
                        "ber_estimate": float(f.ber_estimate),
                        "validation_detail": f.validation_detail
                    } for f in fec_cands
                ],
                "framing": {
                    "status": framing_res.status.value,
                    "frame_structure_detected": framing_res.frame_structure_detected,
                    "frame_type": framing_res.frame_type,
                    "sync_pattern_name": framing_res.sync_pattern_name,
                    "sync_pattern_bits": framing_res.sync_pattern_bits,
                    "frame_length": framing_res.frame_length,
                    "repeated_frames": framing_res.repeated_frames_found,
                    "crc_match": framing_res.crc_match,
                    "crc_profile": framing_res.crc_profile,
                    "crc_calculated": f"0x{framing_res.crc_calculated:04X}" if framing_res.crc_calculated is not None else None,
                    "crc_received": f"0x{framing_res.crc_received:04X}" if framing_res.crc_received is not None else None,
                    "valid_frames": framing_res.valid_frames,
                    "total_frames": framing_res.total_frames,
                    "crc_pass_rate": framing_res.crc_pass_rate,
                    "crc_matches_summary": framing_res.crc_matches_summary,
                    "recovered_ascii": framing_res.recovered_ascii,
                    "recovered_hex": framing_res.recovered_hex
                },
                "evidence_report": {
                    "overall_verdict": evidence_rep.overall_verdict,
                    "overall_confidence": evidence_rep.overall_confidence.value,
                    "numeric_score": evidence_rep.numeric_score,
                    "candidate_ranking": evidence_rep.candidate_ranking,
                    "intelligence_dict": evidence_rep.to_intelligence_dict()
                },
                "epistemic_hierarchy": {
                    "OBSERVED": [
                        {"domain": it.domain, "description": it.description, "value": str(it.value)}
                        for it in evidence_rep.observed
                    ],
                    "ESTIMATED": [
                        {"domain": it.domain, "description": it.description, "value": str(it.value)}
                        for it in evidence_rep.estimated
                    ],
                    "HYPOTHESIZED": [
                        {"domain": it.domain, "description": it.description, "value": str(it.value)}
                        for it in evidence_rep.hypothesized
                    ],
                    "VALIDATED": [
                        {"domain": it.domain, "description": it.description, "value": str(it.value)}
                        for it in evidence_rep.validated
                    ],
                    "UNKNOWN": [
                        {"domain": it.domain, "description": it.description, "value": str(it.value)}
                        for it in evidence_rep.unknown
                    ]
                },
                "synchronized_symbols_raw": sync_symbols.constellation_points
            }
        except Exception as e:
            failed_stage = failed_stage or "physical_reconstruction"
            reconstruction_telemetry = {
                "error": str(e),
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "failed_stage": "physical_reconstruction",
                "epistemic_hierarchy": {
                    "OBSERVED": [], "ESTIMATED": [], "HYPOTHESIZED": [], "VALIDATED": [], "UNKNOWN": []
                }
            }

        # 7. Final Hypothesis Validation Gate
        from .contracts import SignalHypothesis, ModulationFamily
        from .validation_gate import run_validation_gate, ValidationGateConfig
        winning_cand_dict = detection.get("winning_hypothesis")
        winning_cand = SignalHypothesis.from_dict(winning_cand_dict) if winning_cand_dict else None

        # If winning_cand is None, determine whether to construct a candidate from detection or handle noise/silence
        if not winning_cand:
            if not blind_params.signal_presence or detection.get("signal_class_id") == "UNKNOWN":
                winning_cand = SignalHypothesis(
                    hypothesis_id="HYP-NOISE-00",
                    signal_family="NOISE",
                    modulation=ModulationFamily.UNKNOWN_OOD,
                    protocol="Stationary Gaussian Noise Floor",
                    confidence=0.0,
                    evidence_score=0.0,
                    validation_status="ABSTAINED"
                )
            else:
                is_ood_cand = bool(
                    meta.get("is_ood", False)
                    or detection.get("is_ood", False)
                    or mod_infer.modulation_family == ModulationFamily.UNKNOWN_OOD
                    or meta.get("scenario_key") == "UNKNOWN_OOD"
                )
                winning_cand = SignalHypothesis(
                    hypothesis_id="HYP-BLIND-01",
                    signal_family=str(detection.get("modulation_family", "UNKNOWN")),
                    modulation=detection.get("modulation_family", "UNKNOWN"),
                    protocol=detection.get("protocol_name", "Unknown Protocol"),
                    confidence=float(detection.get("confidence", 0.5)),
                    evidence_score=float(detection.get("confidence", 0.5)),
                    temporal_consistency=float(temporal_val.get("composite_stability_index", 1.0)) if isinstance(temporal_val, dict) else 1.0,
                    physical_consistency=1.0,
                    is_ood=is_ood_cand,
                    parameters=dict(merged_params)
                )

        if winning_cand:
            winning_cand.parameters.update(merged_params)
            if specialized_params:
                winning_cand.parameters.update(specialized_params)
        ranked_cands = [SignalHypothesis.from_dict(d) for d in detection.get("ranked_candidates", [])]
        if not ranked_cands and winning_cand:
            ranked_cands = [winning_cand]
        for rc in ranked_cands:
            rc.parameters.update(merged_params)
            if specialized_params:
                rc.parameters.update(specialized_params)

        final_status, validated_winner, val_trace = run_validation_gate(
            winner=winning_cand,
            ranked_candidates=ranked_cands,
            decision_status=detection.get("decision_status", "UNKNOWN"),
            temporal_result=temporal_val,
            pulse_info=pulse_info,
            reconstruction_telemetry=reconstruction_telemetry,
            specialized_params=specialized_params,
            fs=fs,
            config=ValidationGateConfig()
        )

        detection["final_decision"] = final_status
        detection["final_status"] = final_status
        detection["validation_status"] = final_status
        detection["validation_trace"] = val_trace.to_dict()
        detection["validation_gate"] = val_trace.to_dict()

        # 8. Parameter Uncertainty & Epistemic Status Reporting Layer (Step 6 & 7)
        from .parameter_uncertainty import build_parameter_uncertainty_report, get_verdict_explanation
        is_crc_pass = False
        try:
            if isinstance(reconstruction_telemetry, dict):
                framing_info = reconstruction_telemetry.get("framing", {})
                is_crc_pass = bool(framing_info.get("crc_match", False))
        except Exception:
            is_crc_pass = False

        param_uncertainty_report = build_parameter_uncertainty_report(
            parameters=merged_params,
            temporal_result=temporal_val,
            validation_status=final_status,
            fs=fs,
            pulse_info=pulse_info,
            specialized_params=specialized_params,
            is_crc_valid=is_crc_pass
        )

        if validated_winner:
            validated_winner.parameter_reports = dict(param_uncertainty_report)
            for p_name, p_info in param_uncertainty_report.items():
                if p_info.get("uncertainty") is not None:
                    validated_winner.parameter_uncertainty[p_name] = float(p_info["uncertainty"])
                if p_info.get("status"):
                    try:
                        validated_winner.parameter_status[p_name] = EpistemicStatus(p_info["status"])
                    except Exception:
                        pass
            detection["winning_hypothesis"] = validated_winner.to_dict()

        verdict_explanation = get_verdict_explanation(
            verdict=final_status,
            detection_meta=detection,
            candidate_hypotheses=detection.get("ranked_candidates", []),
            parameters=merged_params
        )

        pipeline_status = "PARTIAL_SUCCESS" if failed_stage is not None else "SUCCESS"
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        is_abstained = bool(detection.get("abstained", detection.get("confidence", 1.0) == 0.0 or detection.get("signal_class_id") == "UNKNOWN" or final_status == "NO SIGNAL / NOISE FLOOR"))
        evidence_quality = detection.get("evidence_quality")
        if not evidence_quality:
            conf = detection.get("confidence", 0.0)
            evidence_quality = "HIGH" if conf >= 0.85 else ("MEDIUM" if conf >= 0.50 else "LOW")

        return {
            "status": pipeline_status,
            "failed_stage": failed_stage,
            "abstained": is_abstained,
            "evidence_quality": evidence_quality,
            "telemetry_mode": "adaptive_autonomous",
            "execution_time_ms": float(np.round(t_elapsed_ms, 2)),
            "metadata": meta,
            "autonomous_detection": detection,
            "detection": detection,
            "candidate_hypotheses": detection.get("candidate_hypotheses", []),
            "ranked_candidates": detection.get("ranked_candidates", []),
            "contradiction_analysis": detection.get("contradiction_analysis", []),
            "winning_hypothesis": detection.get("winning_hypothesis"),
            "decision_status": detection.get("decision_status", "UNKNOWN"),
            "final_decision": final_status,
            "final_status": final_status,
            "validation_status": final_status,
            "validation_trace": val_trace.to_dict(),
            "validation_gate": val_trace.to_dict(),
            "verdict_explanation": verdict_explanation,
            "parameters": merged_params,
            "parameter_uncertainties": param_uncertainty_report,
            "parameter_reports": param_uncertainty_report,
            "structured_parameters": param_uncertainty_report,
            "parameter_status_report": param_uncertainty_report,
            "modulation_classification": mod_info,
            "pulse_analysis": pulse_info,
            "specialized_telemetry": specialized_params,
            "blind_parameters": blind_params.to_dict(),
            "blind_parameter_vector": blind_params.to_dict(),
            "modulation_inference": mod_infer.to_dict(),
            "waveform_morphology": blind_params.morphology_fingerprint,
            "parameter_consistency": blind_params.parameter_consistency,
            "blindness_provenance": blind_params.blindness_provenance,
            "frequency_structure": blind_params.frequency_structure,
            "symbol_rate_consensus": {
                "consensus_rate_hz": blind_params.symbol_rate_consensus_hz,
                "method": blind_params.symbol_rate_consensus_method,
                "candidates": blind_params.symbol_rate_candidates
            },
            "signal_segments": blind_params.segments,
            "temporal_validation": temporal_val,
            "reconstruction": reconstruction_telemetry,
            "evidence_report": reconstruction_telemetry.get("evidence_report", {}),
            "epistemic_hierarchy": reconstruction_telemetry.get("epistemic_hierarchy", {})
        }


# Global singleton pipeline instance
_adaptive_pipeline = AdaptiveExtractionPipeline()


def run_adaptive_pipeline(
    signal: np.ndarray,
    fs: float,
    metadata: Optional[Dict[str, Any]] = None,
    override_class_id: Optional[str] = None
) -> Dict[str, Any]:
    """Top-level entrypoint for the autonomous adaptive extraction pipeline."""
    return _adaptive_pipeline.run(signal, fs, metadata=metadata, override_class_id=override_class_id)
