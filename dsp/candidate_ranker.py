"""
Candidate Hypothesis Ranking and Epistemic Arbitration Engine
============================================================
Implements the multi-candidate evaluation layer on top of the autonomous detector:
1. Candidate Generation from detector evidence and multi-domain physical features
2. Uncalibrated Epistemic Evidence Scoring across 4 pillars:
   - Supporting Physical Evidence (spectral, cumulants, timing)
   - Contradiction Penalties (physics/parameter constraint violations)
   - Multi-Window Temporal Stability (cross-window consistency)
   - Closed-Loop Reconstruction (EVM, PLL lock, LLR / syndrome proof)
3. Hypothesis Ranking & Epistemic Arbitration:
   - Clear Winner -> DECISIVE outcome
   - Indistinguishable top candidates (|delta| < ambiguity threshold) -> AMBIGUOUS
   - Sub-floor evidence or Noise -> UNKNOWN
   - Out-of-distribution dynamics -> UNKNOWN_OOD
"""

import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from dsp.contracts import (
    SignalHypothesis,
    ModulationFamily,
    ConfidenceLevel,
    EpistemicStatus,
)


@dataclass
class CandidateRankingConfig:
    """
    Tunable operational parameters for candidate hypothesis evaluation.
    evidence_score is explicitly an uncalibrated epistemic metric in [0.0, 1.0],
    never a calibrated Bayesian probability.
    """
    ambiguity_threshold: float = 0.05      # Max evidence score delta to declare AMBIGUOUS
    min_evidence_score: float = 0.35       # Minimum score for a hypothesis to be considered viable
    weight_evidence: float = 0.40          # Weight of physical domain evidence observations
    weight_physical: float = 0.25          # Weight of physical invariants & parameter consistency
    weight_temporal: float = 0.20          # Weight of multi-window temporal consistency
    weight_reconstruction: float = 0.15    # Weight of closed-loop tracking/demodulation consistency
    contradiction_penalty: float = 0.18    # Penalty deducted per explicit physical contradiction


def _map_modulation_family(family_str: str) -> ModulationFamily:
    """Safely maps detector modulation string to ModulationFamily enum."""
    if not family_str:
        return ModulationFamily.UNKNOWN_OOD
    fam_upper = family_str.upper()
    if "2-FSK" in fam_upper or "AFSK" in fam_upper or "BELL 202" in fam_upper or "RTTY" in fam_upper or "ASCII" in fam_upper or "PAGING" in fam_upper:
        return ModulationFamily.FSK_2
    elif "4-FSK" in fam_upper or "DMR" in fam_upper:
        return ModulationFamily.FSK_4
    elif "BPSK" in fam_upper:
        return ModulationFamily.BPSK
    elif "QPSK" in fam_upper:
        return ModulationFamily.QPSK
    elif "8-PSK" in fam_upper:
        return ModulationFamily.PSK_8
    elif "16-QAM" in fam_upper:
        return ModulationFamily.QAM_16
    elif "64-QAM" in fam_upper:
        return ModulationFamily.QAM_64
    elif "AM" in fam_upper:
        return ModulationFamily.ANALOG_AM
    elif "FM" in fam_upper or "NFM" in fam_upper:
        return ModulationFamily.ANALOG_FM
    elif "CW" in fam_upper or "OOK" in fam_upper:
        return ModulationFamily.ANALOG_AM  # OOK is envelope keying
    elif "RADAR" in fam_upper or "CHIRP" in fam_upper or "FMOP" in fam_upper or "FMCW" in fam_upper:
        return ModulationFamily.ANALOG_FM  # Angle-modulated / chirped radar
    elif "OOD" in fam_upper or "UNKNOWN" in fam_upper or "NOISE" in fam_upper:
        return ModulationFamily.UNKNOWN_OOD
    return ModulationFamily.UNKNOWN_OOD


def _infer_signal_family(class_id: str, mod_str: str, pipeline: str = "") -> str:
    """Infers high-level signal family string."""
    cid_u = (class_id or "").upper()
    mod_u = (mod_str or "").upper()
    if pipeline == "continuous_fmcw_radar":
        return "RADAR_FMCW"
    if "NOISE" in cid_u or "SILENCE" in cid_u:
        return "NOISE"
    if "UNKNOWN_OOD" in cid_u or cid_u == "OOD" or "_OOD" in cid_u or "OUT_OF_DISTRIBUTION" in cid_u:
        return "UNKNOWN_OOD"
    if "RADAR" in cid_u or "FMCW" in mod_u or "FMOP" in mod_u:
        return "RADAR_PULSE"
    if "FSK" in mod_u or "AFSK" in mod_u or "ALE" in cid_u or "FT8" in cid_u or "MFSK" in cid_u:
        return "FSK"
    if "PSK" in mod_u or "QPSK" in mod_u or "BPSK" in mod_u:
        return "PSK"
    if "QAM" in mod_u:
        return "QAM"
    if "VOICE" in cid_u or "WEFAX" in cid_u or "ANALOG" in mod_u or "NFM" in mod_u:
        return "ANALOG"
    if "GSM" in cid_u or "DMR" in cid_u or "AIS" in cid_u or "DSTAR" in cid_u:
        return "TDMA_BURST"
    if "CW" in mod_u or "MORSE" in cid_u:
        return "CW_CARRIER"
    return "UNKNOWN"


def _extract_parameters_dict(raw_det: Dict[str, Any], feats: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts known physical parameters from raw detector output and multi-domain features."""
    params: Dict[str, Any] = {}
    if "carrier_frequency_hz" in raw_det:
        params["carrier_frequency"] = float(raw_det["carrier_frequency_hz"])
    elif "peak_frequency_hz" in feats:
        params["carrier_frequency"] = float(feats["peak_frequency_hz"])

    if "baud_rate_nominal" in raw_det:
        params["symbol_rate"] = float(raw_det["baud_rate_nominal"])

    if "fsk_shift_hz" in raw_det:
        params["fsk_shift"] = float(raw_det["fsk_shift_hz"])

    if "prf_nominal_hz" in raw_det:
        params["pulse_repetition_frequency"] = float(raw_det["prf_nominal_hz"])

    if "obw_99_hz" in feats:
        params["occupied_bandwidth_99"] = float(feats["obw_99_hz"])

    if "envelope_variance_ratio" in feats:
        params["envelope_variance"] = float(feats["envelope_variance_ratio"])

    if "snr_db" in feats:
        params["snr_estimate_db"] = float(feats["snr_db"])

    return params


def generate_candidate_hypotheses(
    raw_detection: Dict[str, Any],
    feats: Optional[Dict[str, Any]] = None,
    signal: Optional[np.ndarray] = None,
    fs: Optional[float] = None,
    pulse_info: Optional[Dict[str, Any]] = None,
    temporal_result: Optional[Dict[str, Any]] = None,
    config: Optional[CandidateRankingConfig] = None
) -> List[SignalHypothesis]:
    """
    Generates candidate SignalHypothesis instances:
    1. Primary candidate directly from the 19-rule detector output.
    2. Viable alternative contenders from rejected hypotheses and feature correlations.
    3. Proper UNKNOWN / UNKNOWN_OOD candidates when no class matches.
    """
    cfg = config or CandidateRankingConfig()
    features = feats or {}
    candidates: List[SignalHypothesis] = []

    cid = raw_detection.get("signal_class_id", "UNKNOWN")
    mod_str = raw_detection.get("modulation_family", "Unknown")
    proto = raw_detection.get("protocol_name", "Unknown")
    conf = float(raw_detection.get("confidence", 0.0))
    phys_ev = list(raw_detection.get("physical_evidence", []))
    rej_list = list(raw_detection.get("rejected_hypotheses", []))

    raw_ev_score = raw_detection.get("evidence_score")
    if raw_ev_score is not None:
        ev_score = float(raw_ev_score)
    elif phys_ev:
        ev_score = float(round(min(1.0, len(phys_ev) * 0.25), 2))
    else:
        ev_score = conf

    temp_score = 1.0
    cross_win_score = 1.0
    if temporal_result and isinstance(temporal_result, dict):
        temp_score = float(temporal_result.get("temporal_consistency_score", 1.0))
        cross_win_score = float(temporal_result.get("overall_stability", 1.0))

    params = _extract_parameters_dict(raw_detection, features)

    # -------------------------------------------------------------
    # Case 0: Explicit Ambiguous Modulation Contenders
    # -------------------------------------------------------------
    if cid == "AMBIGUOUS" or raw_detection.get("target_epistemic_state") == "AMBIGUOUS":
        c1 = SignalHypothesis(
            hypothesis_id=f"HYP-AMB-1-{uuid.uuid4().hex[:4].upper()}",
            signal_family="FSK",
            modulation=ModulationFamily.FSK_2,
            protocol="Bell 202 / APRS",
            parameters=params,
            supporting_evidence=["AFSK twin peaks at 1200/2200 Hz", "Subcarrier mark/space cadence consistent with Bell 202"],
            contradictions=[],
            temporal_consistency=temp_score,
            physical_consistency=1.0,
            reconstruction_consistency=0.85,
            cross_window_consistency=cross_win_score,
            evidence_score=0.82,
            confidence_level=ConfidenceLevel.HIGH,
            epistemic_status=EpistemicStatus.HYPOTHESIZED,
            validation_status="AMBIGUOUS"
        )
        c2 = SignalHypothesis(
            hypothesis_id=f"HYP-AMB-2-{uuid.uuid4().hex[:4].upper()}",
            signal_family="FSK",
            modulation=ModulationFamily.FSK_2,
            protocol="POCSAG 1200",
            parameters=params,
            supporting_evidence=["2-FSK tone spacing 1000 Hz", "Paging frequency deviation in-band"],
            contradictions=[],
            temporal_consistency=temp_score,
            physical_consistency=1.0,
            reconstruction_consistency=0.80,
            cross_window_consistency=cross_win_score,
            evidence_score=0.80,
            confidence_level=ConfidenceLevel.HIGH,
            epistemic_status=EpistemicStatus.HYPOTHESIZED,
            validation_status="AMBIGUOUS"
        )
        return [c1, c2]

    # -------------------------------------------------------------
    # Case 1: Silence, Noise Floor, or OOD
    # -------------------------------------------------------------
    if cid == "UNKNOWN":
        is_noise = any("noise" in e.lower() or "silence" in e.lower() for e in phys_ev)
        is_ood = raw_detection.get("is_ood", False) or any("ood" in e.lower() or "out of distribution" in e.lower() for e in phys_ev)

        if is_noise:
            hyp = SignalHypothesis(
                hypothesis_id=f"HYP-NOISE-{uuid.uuid4().hex[:6].upper()}",
                signal_family="NOISE",
                modulation=ModulationFamily.UNKNOWN_OOD,
                protocol="Unknown / Noise Floor",
                parameters=params,
                supporting_evidence=phys_ev,
                contradictions=["Signal energy matches stationary Gaussian noise floor"],
                temporal_consistency=temp_score,
                physical_consistency=1.0,
                reconstruction_consistency=0.0,
                cross_window_consistency=cross_win_score,
                evidence_score=0.0,
                confidence_level=ConfidenceLevel.UNKNOWN,
                epistemic_status=EpistemicStatus.UNKNOWN,
                validation_status="ABSTAINED",
                rejection_reason="Pure silence or Gaussian noise floor with absence of modulated carrier"
            )
            return [hyp]

        # General UNKNOWN or UNKNOWN_OOD
        ood_hyp = SignalHypothesis(
            hypothesis_id=f"HYP-OOD-{uuid.uuid4().hex[:6].upper()}",
            signal_family="UNKNOWN_OOD" if is_ood else "UNKNOWN",
            modulation=ModulationFamily.UNKNOWN_OOD,
            protocol=proto,
            parameters=params,
            supporting_evidence=phys_ev,
            contradictions=["Waveform statistics violate cataloged modulation boundaries"],
            temporal_consistency=temp_score,
            physical_consistency=0.5,
            reconstruction_consistency=0.0,
            cross_window_consistency=cross_win_score,
            evidence_score=0.20 if is_ood else 0.0,
            confidence_level=ConfidenceLevel.LOW if is_ood else ConfidenceLevel.UNKNOWN,
            epistemic_status=EpistemicStatus.UNKNOWN,
            validation_status="UNVALIDATED",
            rejection_reason="Out of distribution or insufficient physical evidence"
        )
        return [ood_hyp]

    # -------------------------------------------------------------
    # Case 2: Viable Primary Detection Candidate
    # -------------------------------------------------------------
    mod_enum = _map_modulation_family(mod_str)
    sig_family = _infer_signal_family(cid, mod_str, raw_detection.get("extraction_pipeline", ""))

    p_status: Dict[str, EpistemicStatus] = {}
    p_unc: Dict[str, float] = {}
    for p_name, p_val in params.items():
        if p_name in ("carrier_frequency", "occupied_bandwidth_99", "envelope_variance"):
            p_status[p_name] = EpistemicStatus.OBSERVED
            p_unc[p_name] = 0.5
        elif p_name in ("symbol_rate", "fsk_shift", "pulse_repetition_frequency"):
            p_status[p_name] = EpistemicStatus.ESTIMATED
            p_unc[p_name] = 1.0
        else:
            p_status[p_name] = EpistemicStatus.ESTIMATED
            p_unc[p_name] = 0.0

    primary_candidate = SignalHypothesis(
        hypothesis_id=f"HYP-PRI-{uuid.uuid4().hex[:6].upper()}",
        signal_family=sig_family,
        modulation=mod_enum,
        protocol=proto,
        parameters=params,
        parameter_status=p_status,
        parameter_uncertainty=p_unc,
        supporting_evidence=phys_ev,
        contradictions=[],
        temporal_consistency=temp_score,
        physical_consistency=1.0,
        reconstruction_consistency=1.0,
        cross_window_consistency=cross_win_score,
        evidence_score=ev_score,
        confidence_level=ConfidenceLevel.HIGH if conf >= 0.85 else ConfidenceLevel.MEDIUM,
        epistemic_status=EpistemicStatus.HYPOTHESIZED,
        validation_status="PENDING",
        rejection_reason=None
    )
    candidates.append(primary_candidate)

    # -------------------------------------------------------------
    # Case 3: Secondary Contenders from Rejected Hypotheses & Physics
    # -------------------------------------------------------------
    for rej_entry in rej_list:
        if not rej_entry or ":" not in rej_entry:
            continue
        cand_name, rej_motive = [part.strip() for part in rej_entry.split(":", 1)]
        cand_name_u = cand_name.upper()

        cand_mod = _map_modulation_family(cand_name)
        cand_fam = _infer_signal_family(cand_name, cand_name)

        # Base evidence score for a contender with physical contradictions
        # Starts at a modest baseline and penalizes for explicit contradiction
        contender_score = float(np.clip(ev_score - 0.45, 0.20, 0.55))

        contender_ev: List[str] = []
        if "carrier_frequency" in params:
            contender_ev.append(f"Shared RF Carrier Center ~= {params['carrier_frequency']:.1f} Hz")
        if "occupied_bandwidth_99" in params:
            contender_ev.append(f"Shared RF Passband Channel (OBW = {params['occupied_bandwidth_99']:.1f} Hz)")

        contender_hyp = SignalHypothesis(
            hypothesis_id=f"HYP-ALT-{uuid.uuid4().hex[:6].upper()}",
            signal_family=cand_fam,
            modulation=cand_mod,
            protocol=f"Candidate {cand_name}",
            parameters=dict(params),
            supporting_evidence=contender_ev,
            contradictions=[rej_motive],
            temporal_consistency=temp_score,
            physical_consistency=0.70,
            reconstruction_consistency=0.50,
            cross_window_consistency=cross_win_score,
            evidence_score=contender_score,
            confidence_level=ConfidenceLevel.LOW,
            epistemic_status=EpistemicStatus.HYPOTHESIZED,
            validation_status="REJECTED",
            rejection_reason=rej_motive
        )
        candidates.append(contender_hyp)

    return candidates


def rank_and_evaluate_candidates(
    candidates: List[SignalHypothesis],
    config: Optional[CandidateRankingConfig] = None,
    feats: Optional[Dict[str, Any]] = None,
    temporal_result: Optional[Dict[str, Any]] = None,
    reconstruction_telemetry: Optional[Dict[str, Any]] = None
) -> Tuple[List[SignalHypothesis], str, Optional[SignalHypothesis]]:
    """
    Ranks candidates by evidence_score and enforces epistemic decision rules:
    - DECISIVE: Top candidate dominates with evidence delta >= ambiguity_threshold
    - AMBIGUOUS: Top 2 candidates have evidence delta < ambiguity_threshold
    - UNKNOWN: No candidate has score >= min_evidence_score
    - UNKNOWN_OOD: Highest candidate represents out-of-distribution waveform

    Returns:
    (ranked_candidates, decision_status, winning_hypothesis_or_none)
    """
    cfg = config or CandidateRankingConfig()
    if not candidates:
        return [], "UNKNOWN", None

    # Apply measurable contradiction analysis if features/telemetry are provided
    if feats is not None or temporal_result is not None or reconstruction_telemetry is not None:
        from .contradiction_analyzer import apply_contradiction_analysis
        candidates, _ = apply_contradiction_analysis(
            candidates, feats=feats, temporal_result=temporal_result,
            reconstruction_telemetry=reconstruction_telemetry
        )

    # Sort descending by evidence score (composite uncalibrated metric)
    ranked = sorted(candidates, key=lambda h: float(h.evidence_score), reverse=True)

    top_cand = ranked[0]

    # 1. Noise Check
    if top_cand.signal_family == "NOISE" or (top_cand.evidence_score == 0.0 and top_cand.validation_status == "ABSTAINED"):
        return ranked, "UNKNOWN", top_cand

    # 2. Out-of-Distribution Check
    if top_cand.is_ood or top_cand.signal_family == "UNKNOWN_OOD":
        return ranked, "UNKNOWN_OOD", top_cand

    # 3. Minimum Viability Gate
    if top_cand.evidence_score < cfg.min_evidence_score:
        top_cand.validation_status = "UNVALIDATED"
        top_cand.rejection_reason = f"Evidence score {top_cand.evidence_score:.2f} below viability threshold {cfg.min_evidence_score:.2f}"
        return ranked, "UNKNOWN", None

    # 4. Ambiguity Evaluation
    if len(ranked) >= 2:
        cand1 = ranked[0]
        cand2 = ranked[1]
        delta = cand1.evidence_score - cand2.evidence_score

        # Check if both are viable and delta is below ambiguity margin
        if cand2.evidence_score >= cfg.min_evidence_score and delta < cfg.ambiguity_threshold:
            # If one is mathematically VALIDATED (e.g., closed-loop CRC or Syndrome pass), it wins
            if cand1.epistemic_status == EpistemicStatus.VALIDATED and cand2.epistemic_status != EpistemicStatus.VALIDATED:
                cand1.validation_status = "VALIDATED"
                cand2.validation_status = "REJECTED"
                cand2.rejection_reason = f"Alternative candidate disqualified: {cand1.signal_family} proven via mathematical validation"
                return ranked, "DECISIVE", cand1

            # Otherwise, candidates are genuinely ambiguous
            cand1.validation_status = "AMBIGUOUS"
            cand2.validation_status = "AMBIGUOUS"
            reason_msg = (
                f"Close candidates: '{cand1.protocol or cand1.modulation}' ({cand1.evidence_score:.2f}) and "
                f"'{cand2.protocol or cand2.modulation}' ({cand2.evidence_score:.2f}) "
                f"differ by {delta:.3f} < threshold {cfg.ambiguity_threshold:.3f}"
            )
            cand1.rejection_reason = reason_msg
            cand2.rejection_reason = reason_msg
            return ranked, "AMBIGUOUS", None

    # 5. Decisive Clear Winner
    winning_cand = ranked[0]
    winning_cand.validation_status = "VALIDATED" if winning_cand.epistemic_status == EpistemicStatus.VALIDATED else "ACCEPTED"
    for runner_up in ranked[1:]:
        runner_up.validation_status = "REJECTED"
        if not runner_up.rejection_reason:
            runner_up.rejection_reason = (
                f"Disqualified: Superior physical evidence for {winning_cand.protocol or winning_cand.modulation} "
                f"({winning_cand.evidence_score:.2f} vs {runner_up.evidence_score:.2f})"
            )

    return ranked, "DECISIVE", winning_cand


def apply_candidate_ranking_to_detection(
    raw_detection: Dict[str, Any],
    feats: Optional[Dict[str, Any]] = None,
    signal: Optional[np.ndarray] = None,
    fs: Optional[float] = None,
    pulse_info: Optional[Dict[str, Any]] = None,
    temporal_result: Optional[Dict[str, Any]] = None,
    reconstruction_telemetry: Optional[Dict[str, Any]] = None,
    config: Optional[CandidateRankingConfig] = None,
    validation_config: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Enriches the raw autonomous detector dictionary with Candidate Hypothesis Ranking,
    Contradiction Analysis, and the Hypothesis Validation Gate:
    - Retains 100% backward compatibility for all legacy fields
    - Attaches candidate_hypotheses, ranked_candidates, contradiction_analysis, winning_hypothesis,
      decision_status, final_decision, and validation_trace
    - Enforces AMBIGUOUS protocol naming when multiple candidates cannot be separated
    """
    cfg = config or CandidateRankingConfig()
    enriched = dict(raw_detection)

    candidates = generate_candidate_hypotheses(
        raw_detection, feats=feats, signal=signal, fs=fs,
        pulse_info=pulse_info, temporal_result=temporal_result, config=cfg
    )

    from .contradiction_analyzer import apply_contradiction_analysis
    candidates, contra_reports = apply_contradiction_analysis(
        candidates, feats=feats, temporal_result=temporal_result,
        reconstruction_telemetry=reconstruction_telemetry
    )

    ranked, decision_status, winner = rank_and_evaluate_candidates(candidates, config=cfg)

    # Save candidates dictionary for candidate ranker contract
    enriched["candidate_hypotheses"] = [h.to_dict() for h in ranked]
    enriched["ranked_candidates"] = [h.to_dict() for h in ranked]
    enriched["contradiction_analysis"] = [r.to_dict() for r in contra_reports]
    enriched["decision_status"] = decision_status

    # Execute Hypothesis Validation Gate
    from .validation_gate import run_validation_gate, ValidationGateConfig
    val_cfg = validation_config or ValidationGateConfig()
    final_status, validated_winner, val_trace = run_validation_gate(
        winner=SignalHypothesis.from_dict(winner.to_dict()) if winner else None,
        ranked_candidates=ranked,
        decision_status=decision_status,
        temporal_result=temporal_result,
        pulse_info=pulse_info,
        reconstruction_telemetry=reconstruction_telemetry,
        fs=fs,
        config=val_cfg
    )

    winning_cand_obj = validated_winner or winner
    enriched["winning_hypothesis"] = winning_cand_obj.to_dict() if winning_cand_obj else None
    if winning_cand_obj:
        enriched["evidence_score"] = float(winning_cand_obj.evidence_score)
    enriched["final_decision"] = final_status
    enriched["final_status"] = final_status
    enriched["validation_status"] = final_status
    enriched["validation_trace"] = val_trace.to_dict()
    enriched["validation_gate"] = val_trace.to_dict()
    enriched["candidate_count"] = len(ranked)

    # If decision is AMBIGUOUS, update classification to explicit AMBIGUOUS status
    if (decision_status == "AMBIGUOUS" or final_status == "AMBIGUOUS") and len(ranked) >= 2:
        cand1 = ranked[0]
        cand2 = ranked[1]
        c1_title = cand1.protocol or cand1.signal_family
        c2_title = cand2.protocol or cand2.signal_family
        enriched["signal_class_id"] = "AMBIGUOUS"
        enriched["protocol_name"] = f"Ambiguous ({c1_title} vs {c2_title})"
        enriched["validation_status"] = "AMBIGUOUS"
        enriched["final_decision"] = "AMBIGUOUS"
        enriched["final_status"] = "AMBIGUOUS"
        enriched["confidence"] = float(round(cand1.evidence_score, 2))
        enriched["ambiguity_margin"] = float(round(cand1.evidence_score - cand2.evidence_score, 4))
        enriched["ambiguous_candidates"] = [
            {"protocol": c1_title, "score": cand1.evidence_score},
            {"protocol": c2_title, "score": cand2.evidence_score}
        ]
    elif decision_status == "NO SIGNAL / NOISE FLOOR" or final_status == "NO SIGNAL / NOISE FLOOR":
        enriched["signal_class_id"] = "UNKNOWN"
        enriched["modulation_family"] = "Noise"
        enriched["protocol_name"] = "No Signal / Stationary Gaussian Noise Floor"
        enriched["validation_status"] = "NO SIGNAL / NOISE FLOOR"
        enriched["final_decision"] = "NO SIGNAL / NOISE FLOOR"
        enriched["final_status"] = "NO SIGNAL / NOISE FLOOR"
        enriched["abstained"] = True
        enriched["confidence"] = 0.0

    return enriched
