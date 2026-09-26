"""
Parameter Uncertainty and Epistemic Status Reporting Layer
==========================================================
Provides scientifically honest parameter uncertainty, stability,
and epistemic tier reporting without modifying underlying mathematical estimators.

Epistemic Tiers:
- OBSERVED: Direct physical and spectral measurements (FFT peaks, OBW, envelope).
- ESTIMATED: Deterministic continuous estimators (CFO, SNR, Baud, FSK shift, PRF).
- HYPOTHESIZED: Catalog / standard nominal candidate parameters.
- VALIDATED: Closed-loop mathematically proven parameters (CRC pass, syndrome == 0).
- UNKNOWN: Undetermined or unmeasurable parameters; prevents forced false values.

Scientific Honesty Invariants:
1. Do not invent uncertainty values.
2. If uncertainty cannot be reliably estimated from empirical multi-window variance
   or physical instrument resolution, explicitly set uncertainty = None and provide
   an explicit reason.
3. Preserves backward compatibility by maintaining existing numerical dictionaries.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum
import numpy as np

from .contracts import EpistemicStatus, ConfidenceLevel


def _map_stability_level(score: Optional[float], is_stable: Optional[bool]) -> str:
    """Maps continuous stability score and boolean flag to discrete stability level."""
    if score is None:
        return "UNKNOWN"
    if score >= 0.85 and (is_stable is None or is_stable):
        return "HIGH"
    elif score >= 0.60:
        return "MEDIUM"
    else:
        return "LOW"


def build_parameter_uncertainty_report(
    parameters: Dict[str, Any],
    temporal_result: Optional[Dict[str, Any]] = None,
    validation_status: Optional[str] = None,
    fs: Optional[float] = None,
    pulse_info: Optional[Dict[str, Any]] = None,
    specialized_params: Optional[Dict[str, Any]] = None,
    is_crc_valid: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Constructs a scientifically honest parameter uncertainty and epistemic status report.

    Parameters:
    -----------
    parameters : Dict[str, Any]
        Flat dictionary of extracted parameters from baseline and specialized extractors.
    temporal_result : Optional[Dict[str, Any]]
        Multi-window temporal validation output containing per-parameter variance and stability.
    validation_status : Optional[str]
        Overall validation status (e.g. VALIDATED, ESTIMATED, AMBIGUOUS, UNKNOWN).
    fs : Optional[float]
        Physical sampling rate in Hz.
    pulse_info : Optional[Dict[str, Any]]
        Pulse analysis metadata if pulsed signal.
    specialized_params : Optional[Dict[str, Any]]
        Specialized extractor telemetry.
    is_crc_valid : bool
        True if closed-loop CRC or FEC syndrome validation passed.

    Returns:
    --------
    Dict[str, Dict[str, Any]]:
        Mapping of parameter name to structured status dictionary:
        {
            "parameter_name": {
                "value": float | None,
                "status": "OBSERVED" | "ESTIMATED" | "HYPOTHESIZED" | "VALIDATED" | "UNKNOWN",
                "stability": "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN",
                "uncertainty": float | None,
                "uncertainty_type": str | None,
                "uncertainty_reason": str | None,
                "reason": str | None,
                "range": [min, max] | None,
                "unit": str
            }
        }
    """
    merged: Dict[str, Any] = dict(parameters)
    if specialized_params:
        merged.update(specialized_params)

    param_stability: Dict[str, Any] = {}
    if temporal_result and isinstance(temporal_result, dict):
        param_stability = temporal_result.get("parameter_stability", {})

    report: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------------------------------
    # 1. Carrier Frequency (fc)
    # -------------------------------------------------------------------------
    fc_val = (
        merged.get("carrier_frequency")
        if merged.get("carrier_frequency") is not None
        else merged.get("fc_peak_hz", merged.get("carrier_frequency_hz"))
    )
    fc_stab = param_stability.get("fc_peak_hz", param_stability.get("fc_centroid_hz", {}))
    fc_std = fc_stab.get("std")
    fc_is_stable = fc_stab.get("is_stable")
    fc_score = fc_stab.get("stability_score")

    if fc_val is not None and np.isfinite(fc_val):
        fc_val_float = float(np.round(fc_val, 2))
        if fc_std is not None and np.isfinite(fc_std) and len(fc_stab.get("window_values", [])) >= 2:
            fc_unc = float(np.round(fc_std, 2))
            fc_unc_type = "cross_window_std"
            fc_unc_reason = None
            fc_range = [
                float(np.round(fc_val_float - 1.96 * fc_unc, 2)),
                float(np.round(fc_val_float + 1.96 * fc_unc, 2))
            ]
        elif fs is not None and fs > 0:
            # FFT bin resolution bound (N=2048)
            bin_res = fs / 2048.0
            fc_unc = float(np.round(bin_res / 2.0, 2))
            fc_unc_type = "fft_bin_resolution"
            fc_unc_reason = None
            fc_range = [
                float(np.round(fc_val_float - fc_unc, 2)),
                float(np.round(fc_val_float + fc_unc, 2))
            ]
        else:
            fc_unc = None
            fc_unc_type = None
            fc_unc_reason = "Single observation window; cross-window carrier variance not observed"
            fc_range = None

        fc_status = EpistemicStatus.OBSERVED.value
        fc_stability = _map_stability_level(fc_score, fc_is_stable)
    else:
        fc_val_float = None
        fc_unc = None
        fc_unc_type = None
        fc_unc_reason = "Carrier peak not resolved above spectral noise floor"
        fc_range = None
        fc_status = EpistemicStatus.UNKNOWN.value
        fc_stability = "UNKNOWN"

    report["carrier_frequency"] = {
        "value": fc_val_float,
        "status": fc_status,
        "stability": fc_stability,
        "uncertainty": fc_unc,
        "uncertainty_type": fc_unc_type,
        "uncertainty_reason": fc_unc_reason,
        "reason": fc_unc_reason,
        "range": fc_range,
        "unit": "Hz"
    }

    # -------------------------------------------------------------------------
    # 2. 99% Occupied Bandwidth (OBW)
    # -------------------------------------------------------------------------
    obw_val = (
        merged.get("occupied_bandwidth_99")
        if merged.get("occupied_bandwidth_99") is not None
        else merged.get("bw_99pct_hz", merged.get("obw_99_hz"))
    )
    obw_stab = param_stability.get("bw_99pct_hz", {})
    obw_std = obw_stab.get("std")
    obw_is_stable = obw_stab.get("is_stable")
    obw_score = obw_stab.get("stability_score")

    if obw_val is not None and np.isfinite(obw_val) and obw_val > 0.0:
        obw_val_float = float(np.round(obw_val, 2))
        if obw_std is not None and np.isfinite(obw_std) and len(obw_stab.get("window_values", [])) >= 2:
            obw_unc = float(np.round(obw_std, 2))
            obw_unc_type = "cross_window_std"
            obw_unc_reason = None
            obw_range = [
                float(np.round(max(0.0, obw_val_float - 1.96 * obw_unc), 2)),
                float(np.round(obw_val_float + 1.96 * obw_unc, 2))
            ]
        elif fs is not None and fs > 0:
            bin_res = fs / 2048.0
            obw_unc = float(np.round(bin_res, 2))
            obw_unc_type = "spectral_bin_resolution"
            obw_unc_reason = None
            obw_range = [
                float(np.round(max(0.0, obw_val_float - obw_unc), 2)),
                float(np.round(obw_val_float + obw_unc, 2))
            ]
        else:
            obw_unc = None
            obw_unc_type = None
            obw_unc_reason = "Single observation window; bandwidth variation across temporal slices not observed"
            obw_range = None

        obw_status = EpistemicStatus.OBSERVED.value
        obw_stability = _map_stability_level(obw_score, obw_is_stable)
    else:
        obw_val_float = None
        obw_unc = None
        obw_unc_type = None
        obw_unc_reason = "Occupied bandwidth integral could not be converged"
        obw_range = None
        obw_status = EpistemicStatus.UNKNOWN.value
        obw_stability = "UNKNOWN"

    report["occupied_bandwidth_99"] = {
        "value": obw_val_float,
        "status": obw_status,
        "stability": obw_stability,
        "uncertainty": obw_unc,
        "uncertainty_type": obw_unc_type,
        "uncertainty_reason": obw_unc_reason,
        "reason": obw_unc_reason,
        "range": obw_range,
        "unit": "Hz"
    }

    # -------------------------------------------------------------------------
    # 3. Signal-to-Noise Ratio (SNR)
    # -------------------------------------------------------------------------
    snr_val = merged.get("snr_db", merged.get("snr", merged.get("snr_m2m4_db")))
    snr_stab = param_stability.get("snr_db", {})
    snr_std = snr_stab.get("std")
    snr_is_stable = snr_stab.get("is_stable")
    snr_score = snr_stab.get("stability_score")

    if snr_val is not None and np.isfinite(snr_val):
        snr_val_float = float(np.round(snr_val, 2))
        if snr_std is not None and np.isfinite(snr_std) and len(snr_stab.get("window_values", [])) >= 2:
            snr_unc = float(np.round(snr_std, 2))
            snr_unc_type = "cross_window_std"
            snr_unc_reason = None
            snr_range = [
                float(np.round(snr_val_float - 1.96 * snr_unc, 2)),
                float(np.round(snr_val_float + 1.96 * snr_unc, 2))
            ]
        else:
            snr_unc = None
            snr_unc_type = None
            snr_unc_reason = "Single observation window; cross-window SNR variance unavailable"
            snr_range = None

        snr_status = EpistemicStatus.ESTIMATED.value
        snr_stability = _map_stability_level(snr_score, snr_is_stable)
    else:
        snr_val_float = None
        snr_unc = None
        snr_unc_type = None
        snr_unc_reason = "SNR estimation unmeasurable"
        snr_range = None
        snr_status = EpistemicStatus.UNKNOWN.value
        snr_stability = "UNKNOWN"

    report["snr_db"] = {
        "value": snr_val_float,
        "status": snr_status,
        "stability": snr_stability,
        "uncertainty": snr_unc,
        "uncertainty_type": snr_unc_type,
        "uncertainty_reason": snr_unc_reason,
        "reason": snr_unc_reason,
        "range": snr_range,
        "unit": "dB"
    }

    # -------------------------------------------------------------------------
    # 4. Symbol Rate (Baud)
    # -------------------------------------------------------------------------
    baud_val = (
        merged.get("symbol_rate")
        if merged.get("symbol_rate") is not None
        else merged.get("estimated_baud_rate_hz", merged.get("baud_rate_nominal"))
    )
    baud_stab = param_stability.get("estimated_baud_rate_hz", {})
    baud_std = baud_stab.get("std")
    baud_is_stable = baud_stab.get("is_stable")
    baud_score = baud_stab.get("stability_score")

    if baud_val is not None and np.isfinite(baud_val) and baud_val > 0.0:
        baud_val_float = float(np.round(baud_val, 2))
        if baud_std is not None and np.isfinite(baud_std) and len(baud_stab.get("window_values", [])) >= 2:
            baud_unc = float(np.round(baud_std, 2))
            baud_unc_type = "cross_window_std"
            baud_unc_reason = None
            baud_range = [
                float(np.round(max(0.0, baud_val_float - 1.96 * baud_unc), 2)),
                float(np.round(baud_val_float + 1.96 * baud_unc, 2))
            ]
        else:
            baud_unc = None
            baud_unc_type = None
            baud_unc_reason = "Single-point symbol clock recovery; multi-window clock variance not observed"
            baud_range = None

        if is_crc_valid:
            baud_status = EpistemicStatus.VALIDATED.value
        elif merged.get("baud_rate_nominal") is not None and merged.get("estimated_baud_rate_hz") is None:
            baud_status = EpistemicStatus.HYPOTHESIZED.value
        else:
            baud_status = EpistemicStatus.ESTIMATED.value

        baud_stability = _map_stability_level(baud_score, baud_is_stable)
    else:
        baud_val_float = None
        baud_unc = None
        baud_unc_type = None
        baud_unc_reason = "Not applicable (Continuous Wave, Radar, or Analog Audio signal class)"
        baud_range = None
        baud_status = EpistemicStatus.NOT_APPLICABLE.value if merged.get("envelope_variance_ratio", 1.0) < 0.08 else EpistemicStatus.UNKNOWN.value
        baud_stability = "UNKNOWN"

    report["symbol_rate"] = {
        "value": baud_val_float,
        "status": baud_status,
        "stability": baud_stability,
        "uncertainty": baud_unc,
        "uncertainty_type": baud_unc_type,
        "uncertainty_reason": baud_unc_reason,
        "reason": baud_unc_reason,
        "range": baud_range,
        "unit": "Baud"
    }

    # -------------------------------------------------------------------------
    # 5. Frequency Shift (FSK Shift)
    # -------------------------------------------------------------------------
    shift_val = (
        merged.get("frequency_shift")
        if merged.get("frequency_shift") is not None
        else merged.get("fsk_shift_hz", merged.get("frequency_shift_hz", merged.get("fsk_shift")))
    )
    shift_stab = param_stability.get("frequency_shift_hz", {})
    shift_std = shift_stab.get("std")
    shift_is_stable = shift_stab.get("is_stable")
    shift_score = shift_stab.get("stability_score")

    if shift_val is not None and np.isfinite(shift_val) and shift_val > 0.0:
        shift_val_float = float(np.round(shift_val, 2))
        if shift_std is not None and np.isfinite(shift_std) and len(shift_stab.get("window_values", [])) >= 2:
            shift_unc = float(np.round(shift_std, 2))
            shift_unc_type = "cross_window_std"
            shift_unc_reason = None
            shift_range = [
                float(np.round(max(0.0, shift_val_float - 1.96 * shift_unc), 2)),
                float(np.round(shift_val_float + 1.96 * shift_unc, 2))
            ]
        else:
            shift_unc = None
            shift_unc_type = None
            shift_unc_reason = "Single observation window; FSK shift variance across temporal windows unavailable"
            shift_range = None

        shift_status = EpistemicStatus.ESTIMATED.value
        shift_stability = _map_stability_level(shift_score, shift_is_stable)
    else:
        shift_val_float = None
        shift_unc = None
        shift_unc_type = None
        shift_unc_reason = "Not applicable (Non-FSK waveform)"
        shift_range = None
        shift_status = EpistemicStatus.NOT_APPLICABLE.value
        shift_stability = "UNKNOWN"

    report["frequency_shift"] = {
        "value": shift_val_float,
        "status": shift_status,
        "stability": shift_stability,
        "uncertainty": shift_unc,
        "uncertainty_type": shift_unc_type,
        "uncertainty_reason": shift_unc_reason,
        "reason": shift_unc_reason,
        "range": shift_range,
        "unit": "Hz"
    }

    # -------------------------------------------------------------------------
    # 6. Pulse Repetition Frequency (PRF)
    # -------------------------------------------------------------------------
    prf_val = (
        merged.get("pulse_repetition_frequency")
        if merged.get("pulse_repetition_frequency") is not None
        else merged.get("radar_prf_hz", merged.get("pulse_prf_hz", merged.get("prf_nominal_hz")))
    )
    prf_stab = param_stability.get("pulse_prf_hz", {})
    prf_std = prf_stab.get("std")
    prf_is_stable = prf_stab.get("is_stable")
    prf_score = prf_stab.get("stability_score")

    if prf_val is not None and np.isfinite(prf_val) and prf_val > 0.0:
        prf_val_float = float(np.round(prf_val, 2))
        if prf_std is not None and np.isfinite(prf_std) and len(prf_stab.get("window_values", [])) >= 2:
            prf_unc = float(np.round(prf_std, 2))
            prf_unc_type = "cross_window_std"
            prf_unc_reason = None
            prf_range = [
                float(np.round(max(0.0, prf_val_float - 1.96 * prf_unc), 2)),
                float(np.round(prf_val_float + 1.96 * prf_unc, 2))
            ]
        else:
            prf_unc = None
            prf_unc_type = None
            prf_unc_reason = "Single observation window; pulse jitter across temporal windows unavailable"
            prf_range = None

        prf_status = EpistemicStatus.ESTIMATED.value
        prf_stability = _map_stability_level(prf_score, prf_is_stable)
    else:
        prf_val_float = None
        prf_unc = None
        prf_unc_type = None
        prf_unc_reason = "Not applicable (Continuous non-pulsed transmission)"
        prf_range = None
        prf_status = EpistemicStatus.NOT_APPLICABLE.value
        prf_stability = "UNKNOWN"

    report["pulse_repetition_frequency"] = {
        "value": prf_val_float,
        "status": prf_status,
        "stability": prf_stability,
        "uncertainty": prf_unc,
        "uncertainty_type": prf_unc_type,
        "uncertainty_reason": prf_unc_reason,
        "reason": prf_unc_reason,
        "range": prf_range,
        "unit": "Hz"
    }

    # -------------------------------------------------------------------------
    # 7. Pulse Width (PW)
    # -------------------------------------------------------------------------
    pw_val = (
        merged.get("pulse_width")
        if merged.get("pulse_width") is not None
        else merged.get("radar_pulse_width_us", merged.get("pulse_width_us"))
    )
    pw_stab = param_stability.get("pulse_width_us", {})
    pw_std = pw_stab.get("std")
    pw_is_stable = pw_stab.get("is_stable")
    pw_score = pw_stab.get("stability_score")

    if pw_val is not None and np.isfinite(pw_val) and pw_val > 0.0:
        pw_val_float = float(np.round(pw_val, 2))
        if pw_std is not None and np.isfinite(pw_std) and len(pw_stab.get("window_values", [])) >= 2:
            pw_unc = float(np.round(pw_std, 2))
            pw_unc_type = "cross_window_std"
            pw_unc_reason = None
            pw_range = [
                float(np.round(max(0.0, pw_val_float - 1.96 * pw_unc), 2)),
                float(np.round(pw_val_float + 1.96 * pw_unc, 2))
            ]
        else:
            pw_unc = None
            pw_unc_type = None
            pw_unc_reason = "Single observation window; pulse width variance across temporal windows unavailable"
            pw_range = None

        pw_status = EpistemicStatus.ESTIMATED.value
        pw_stability = _map_stability_level(pw_score, pw_is_stable)
    else:
        pw_val_float = None
        pw_unc = None
        pw_unc_type = None
        pw_unc_reason = "Not applicable (Continuous non-pulsed transmission)"
        pw_range = None
        pw_status = EpistemicStatus.NOT_APPLICABLE.value
        pw_stability = "UNKNOWN"

    report["pulse_width"] = {
        "value": pw_val_float,
        "status": pw_status,
        "stability": pw_stability,
        "uncertainty": pw_unc,
        "uncertainty_type": pw_unc_type,
        "uncertainty_reason": pw_unc_reason,
        "reason": pw_unc_reason,
        "range": pw_range,
        "unit": "us"
    }

    # -------------------------------------------------------------------------
    # 8. PAPR and Envelope Variance
    # -------------------------------------------------------------------------
    papr_val = merged.get("papr_db")
    papr_stab = param_stability.get("papr_db", {})
    if papr_val is not None and np.isfinite(papr_val):
        papr_std = papr_stab.get("std")
        papr_unc = float(np.round(papr_std, 2)) if papr_std is not None and np.isfinite(papr_std) and len(papr_stab.get("window_values", [])) >= 2 else None
        report["papr_db"] = {
            "value": float(np.round(papr_val, 2)),
            "status": EpistemicStatus.OBSERVED.value,
            "stability": _map_stability_level(papr_stab.get("stability_score"), papr_stab.get("is_stable")),
            "uncertainty": papr_unc,
            "uncertainty_type": "cross_window_std" if papr_unc is not None else None,
            "uncertainty_reason": None if papr_unc is not None else "Single observation window; PAPR variance unavailable",
            "reason": None if papr_unc is not None else "Single observation window; PAPR variance unavailable",
            "range": [float(np.round(papr_val - 1.96 * papr_unc, 2)), float(np.round(papr_val + 1.96 * papr_unc, 2))] if papr_unc is not None else None,
            "unit": "dB"
        }

    return report


def get_verdict_explanation(
    verdict: str,
    detection_meta: Optional[Dict[str, Any]] = None,
    candidate_hypotheses: Optional[List[Any]] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Returns structured, human-readable explanations and metadata for each
    of the 6 first-class application verdict states:
    1. VALIDATED
    2. ESTIMATED
    3. AMBIGUOUS
    4. UNKNOWN
    5. UNKNOWN_OOD
    6. NO SIGNAL / NOISE FLOOR

    Requirements:
    - Never collapse states into one generic error.
    - AMBIGUOUS must show the top competing hypotheses.
    - UNKNOWN must show what physical observations were successfully measured.
    - NO SIGNAL / NOISE FLOOR must clearly indicate that the system rejected the input rather than failing.
    """
    v_upper = (verdict or "UNKNOWN").upper()
    det = detection_meta or {}
    params = parameters or {}

    if "VALIDATED" in v_upper and "UNVALIDATED" not in v_upper:
        return {
            "verdict": "VALIDATED",
            "status_badge": "VALIDATED",
            "color": "#10b981",  # emerald-500
            "title": "Mathematical & Physical Proof Confirmed",
            "explanation": (
                "Waveform mathematically validated against physical invariants with high temporal stationarity, "
                "conforming modulation properties, and zero unresolved contradictions."
            ),
            "is_rejected": False,
            "top_hypotheses": [],
            "measured_observations": []
        }

    elif "ESTIMATED" in v_upper:
        return {
            "verdict": "ESTIMATED",
            "status_badge": "ESTIMATED",
            "color": "#f59e0b",  # amber-500
            "title": "Continuous Physical Parameters Estimated",
            "explanation": (
                "Signal parameters and modulation structure estimated with high confidence, but unvalidated by "
                "closed-loop algebraic proof (syndrome/CRC) or exhibiting temporal parameter variation."
            ),
            "is_rejected": False,
            "top_hypotheses": [],
            "measured_observations": []
        }

    elif "AMBIGUOUS" in v_upper:
        top_candidates = []
        if candidate_hypotheses:
            for c in candidate_hypotheses[:3]:
                if isinstance(c, dict):
                    top_candidates.append({
                        "protocol": c.get("protocol") or c.get("signal_family", "Candidate"),
                        "score": c.get("evidence_score", 0.0),
                        "evidence": c.get("supporting_evidence", []),
                        "contradictions": c.get("contradictions", [])
                    })
                elif hasattr(c, "to_dict"):
                    cd = c.to_dict()
                    top_candidates.append({
                        "protocol": cd.get("protocol") or cd.get("signal_family", "Candidate"),
                        "score": cd.get("evidence_score", 0.0),
                        "evidence": cd.get("supporting_evidence", []),
                        "contradictions": cd.get("contradictions", [])
                    })

        return {
            "verdict": "AMBIGUOUS",
            "status_badge": "AMBIGUOUS",
            "color": "#a855f7",  # purple-500
            "title": "Multiple Competing Hypotheses Detected",
            "explanation": (
                "Multiple candidate modulation hypotheses exhibit competing physical evidence within the ambiguity threshold. "
                "The classifier refuses to force a single arbitrary classification."
            ),
            "is_rejected": False,
            "top_hypotheses": top_candidates,
            "measured_observations": []
        }

    elif "UNKNOWN_OOD" in v_upper or "OUT_OF_DISTRIBUTION" in v_upper:
        obs = []
        if params.get("fc_peak_hz") is not None and np.isfinite(params.get("fc_peak_hz")):
            obs.append(f"Carrier Peak: {params['fc_peak_hz']:+,.1f} Hz")
        if params.get("bw_99pct_hz") is not None and np.isfinite(params.get("bw_99pct_hz")):
            obs.append(f"99% Occupied Bandwidth: {params['bw_99pct_hz']:,.1f} Hz")
        if params.get("snr_db") is not None and np.isfinite(params.get("snr_db")):
            obs.append(f"Signal-to-Noise Ratio: {params['snr_db']:+.1f} dB")
        if params.get("papr_db") is not None and np.isfinite(params.get("papr_db")):
            obs.append(f"PAPR: {params['papr_db']:.1f} dB")
        if params.get("envelope_variance_ratio") is not None and np.isfinite(params.get("envelope_variance_ratio")):
            obs.append(f"Envelope Variance Ratio: {params['envelope_variance_ratio']:.4f}")

        return {
            "verdict": "UNKNOWN_OOD",
            "status_badge": "UNKNOWN_OOD",
            "color": "#ef4444",  # red-500
            "title": "Out-of-Distribution Waveform Intercepted",
            "explanation": (
                "Waveform exhibits anomalous physical feature distributions statistically deviating from all recognized "
                "signal classes in the knowledge base (Out-of-Distribution / Mahalanobis distance threshold exceeded)."
            ),
            "is_rejected": False,
            "top_hypotheses": [],
            "measured_observations": obs
        }

    elif "NO SIGNAL" in v_upper or "NOISE FLOOR" in v_upper or "NOISE" in v_upper or det.get("signal_class_id") == "NOISE":
        return {
            "verdict": "NO SIGNAL / NOISE FLOOR",
            "status_badge": "NO SIGNAL / NOISE FLOOR",
            "color": "#64748b",  # slate-500
            "title": "Input Rejected by Gaussian Noise Pre-Gate",
            "explanation": (
                "System rejected input: Signal energy and spectral flatness conform to stationary Gaussian noise / quiet channel floor. "
                "The engine deliberately rejected processing rather than producing a false-positive classification."
            ),
            "is_rejected": True,
            "top_hypotheses": [],
            "measured_observations": []
        }

    else:  # UNKNOWN
        # Extract successfully measured physical observations
        obs = []
        if params.get("fc_peak_hz") is not None and np.isfinite(params.get("fc_peak_hz")):
            obs.append(f"Carrier Peak: {params['fc_peak_hz']:+,.1f} Hz")
        if params.get("bw_99pct_hz") is not None and np.isfinite(params.get("bw_99pct_hz")):
            obs.append(f"99% Occupied Bandwidth: {params['bw_99pct_hz']:,.1f} Hz")
        if params.get("snr_db") is not None and np.isfinite(params.get("snr_db")):
            obs.append(f"Signal-to-Noise Ratio: {params['snr_db']:+.1f} dB")
        if params.get("papr_db") is not None and np.isfinite(params.get("papr_db")):
            obs.append(f"PAPR: {params['papr_db']:.1f} dB")
        if params.get("envelope_variance_ratio") is not None and np.isfinite(params.get("envelope_variance_ratio")):
            obs.append(f"Envelope Variance Ratio: {params['envelope_variance_ratio']:.4f}")

        return {
            "verdict": "UNKNOWN",
            "status_badge": "UNKNOWN",
            "color": "#38bdf8",  # sky-400
            "title": "Uncataloged Modulation Profile",
            "explanation": (
                "Physical RF energy and spectral features were successfully measured, but the waveform does not match any "
                "known modulation standard or protocol profile in the catalog."
            ),
            "is_rejected": False,
            "top_hypotheses": [],
            "measured_observations": obs
        }
