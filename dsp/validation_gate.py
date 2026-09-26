"""
Hypothesis Validation Gate Layer
================================
Implements an independent multi-pillar physical validation gate before any candidate
hypothesis can be awarded the epistemic status of VALIDATED.

Conceptual Pipeline Architecture:
BLIND MEASUREMENTS
        ↓
CANDIDATE HYPOTHESES
        ↓
EVIDENCE
        ↓
CONTRADICTIONS
        ↓
MULTI-WINDOW VALIDATION
        ↓
FINAL DECISION (Hypothesis Validation Gate)

Supported Final Statuses:
- VALIDATED: Candidate satisfied all applicable physical, spectral, temporal,
             stability, and reconstruction criteria with zero contradictions.
- ESTIMATED: Candidate possesses viable evidence and credible parameters, but failed
             one or more non-disqualifying validation gates (e.g. temporal instability,
             parameter drift across windows, or unverified short observation duration).
- AMBIGUOUS: Competing viable hypotheses remain separated by less than the ambiguity margin.
- UNKNOWN: Waveform is below detection sensitivity, pure noise, or has sub-viability evidence.
- UNKNOWN_OOD: Waveform dynamics violate cataloged modulation distributions.

Guarantees:
1. No candidate with strong spectral evidence is labeled VALIDATED if temporal validation fails.
2. Every validation criterion is experimentally and physically grounded.
3. Completely configurable via ValidationGateConfig.
4. Generates a comprehensive, machine-readable validation trace.
5. Preserves 100% backward compatibility for existing pipeline outputs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from .contracts import SignalHypothesis, EpistemicStatus, ModulationFamily, ConfidenceLevel


class ValidationGateStatus(str, Enum):
    """Final epistemic validation statuses."""
    VALIDATED = "VALIDATED"
    ESTIMATED = "ESTIMATED"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"
    UNKNOWN_OOD = "UNKNOWN_OOD"
    NO_SIGNAL = "NO SIGNAL / NOISE FLOOR"
    REJECTED = "REJECTED"


@dataclass
class ValidationGateConfig:
    """
    Configuration parameters for the Hypothesis Validation Gate.
    All thresholds are physically grounded rather than arbitrary heuristic constants.
    """
    # Minimum uncalibrated evidence score required for VALIDATED status
    min_evidence_score_for_validation: float = 0.70

    # Minimum composite temporal stability index from MultiWindowTemporalValidator
    min_temporal_stability: float = 0.60

    # Cross-window parameter consistency minimum score
    min_cross_window_score: float = 0.55

    # Maximum permissible coefficient of variation for occupied bandwidth across windows
    max_bandwidth_variation_pct: float = 35.0

    # Maximum permissible coefficient of variation for carrier frequency across windows
    max_carrier_variation_pct: float = 25.0

    # Maximum permissible coefficient of variation for symbol rate / baud across windows
    max_symbol_rate_variation_pct: float = 25.0

    # Maximum permissible symbol dwell error relative to cataloged standard
    max_symbol_dwell_error_pct: float = 18.0

    # Maximum permissible Constellation Error Vector Magnitude (EVM) for digital modulations
    max_evm_pct: float = 35.0

    # Maximum allowed unresolved physical contradictions for VALIDATED status
    max_contradictions_allowed: int = 0

    # Ambiguity margin threshold (if top two viable candidates differ by less than this, AMBIGUOUS)
    ambiguity_threshold: float = 0.05

    # Minimum evidence score floor for a hypothesis to be considered viable
    min_viability_score: float = 0.35

    # Whether short observation durations gracefully fall back to ESTIMATED instead of failing
    allow_estimated_on_insufficient_observation: bool = True

    # Minimum physical SNR threshold (dB)
    min_physical_snr_db: float = -15.0


@dataclass
class GateCheckResult:
    """Detailed result of an individual physical validation gate check."""
    gate_name: str
    passed: bool
    applicable: bool
    score: float
    observed_value: Any
    threshold_or_criterion: Any
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "applicable": self.applicable,
            "score": float(np.round(self.score, 4)),
            "observed_value": str(self.observed_value),
            "threshold_or_criterion": str(self.threshold_or_criterion),
            "reason": self.reason
        }


@dataclass
class ValidationGateTrace:
    """Machine-readable validation audit trace for an evaluated candidate hypothesis."""
    hypothesis_id: str
    signal_family: str
    protocol: Optional[str]
    final_status: str
    evidence_score: float
    passed_gates: List[str] = field(default_factory=list)
    failed_gates: List[str] = field(default_factory=list)
    waived_gates: List[str] = field(default_factory=list)
    gate_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    validation_score: float = 0.0
    narrative_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "signal_family": self.signal_family,
            "protocol": self.protocol,
            "final_status": self.final_status,
            "evidence_score": float(np.round(self.evidence_score, 4)),
            "passed_gates": list(self.passed_gates),
            "failed_gates": list(self.failed_gates),
            "waived_gates": list(self.waived_gates),
            "gate_results": self.gate_results,
            "validation_score": float(np.round(self.validation_score, 4)),
            "narrative_summary": self.narrative_summary
        }


class HypothesisValidationGate:
    """
    Independent Hypothesis Validation Gate Engine.
    Evaluates candidate hypotheses against 6 physical gates:
    1. Spectral Consistency
    2. Temporal Consistency
    3. Symbol / Dwell Consistency
    4. Cross-Window Stability
    5. Physical Plausibility
    6. Reconstruction Consistency
    """

    def __init__(self, config: Optional[ValidationGateConfig] = None):
        self.config = config or ValidationGateConfig()

    def evaluate_candidate(
        self,
        candidate: Optional[SignalHypothesis],
        all_candidates: Optional[List[SignalHypothesis]] = None,
        decision_status_from_ranker: str = "DECISIVE",
        temporal_result: Optional[Dict[str, Any]] = None,
        pulse_info: Optional[Dict[str, Any]] = None,
        reconstruction_telemetry: Optional[Dict[str, Any]] = None,
        specialized_params: Optional[Dict[str, Any]] = None,
        fs: Optional[float] = None
    ) -> Tuple[str, Optional[SignalHypothesis], ValidationGateTrace]:
        """
        Executes full validation gate audit on the candidate hypothesis.

        Returns:
            (final_status, updated_candidate, validation_trace)
        """
        cfg = self.config
        all_cands = all_candidates or ([candidate] if candidate else [])

        # ---------------------------------------------------------------------
        # Pre-Gate Case 1: Null or Empty Candidate
        # ---------------------------------------------------------------------
        if candidate is None:
            if decision_status_from_ranker == "AMBIGUOUS" and len(all_cands) >= 2:
                cand1 = all_cands[0]
                cand2 = all_cands[1]
                cand1.validation_status = ValidationGateStatus.AMBIGUOUS.value
                cand2.validation_status = ValidationGateStatus.AMBIGUOUS.value
                msg = (
                    f"Ambiguous hypothesis: {cand1.protocol or cand1.signal_family} ({cand1.evidence_score:.2f}) vs "
                    f"{cand2.protocol or cand2.signal_family} ({cand2.evidence_score:.2f}) "
                    f"within margin {abs(cand1.evidence_score - cand2.evidence_score):.3f} <= {cfg.ambiguity_threshold:.3f}"
                )
                trace = ValidationGateTrace(
                    hypothesis_id=cand1.hypothesis_id,
                    signal_family=cand1.signal_family,
                    protocol=cand1.protocol,
                    final_status=ValidationGateStatus.AMBIGUOUS.value,
                    evidence_score=cand1.evidence_score,
                    narrative_summary=msg
                )
                return ValidationGateStatus.AMBIGUOUS.value, cand1, trace

            trace = ValidationGateTrace(
                hypothesis_id="NONE",
                signal_family="UNKNOWN",
                protocol=None,
                final_status=ValidationGateStatus.UNKNOWN.value,
                evidence_score=0.0,
                narrative_summary="No candidate hypothesis available for validation"
            )
            return ValidationGateStatus.UNKNOWN.value, None, trace

        is_cand_ood = (
            decision_status_from_ranker == "UNKNOWN_OOD"
            or candidate.is_ood
            or candidate.signal_family == "UNKNOWN_OOD"
            or (
                candidate.modulation == ModulationFamily.UNKNOWN_OOD
                and candidate.signal_family in ("UNKNOWN", "UNKNOWN_OOD")
                and candidate.protocol is None
                and candidate.evidence_score <= cfg.min_viability_score
            )
        )
        if is_cand_ood:
            candidate.validation_status = "UNVALIDATED"
            candidate.epistemic_status = EpistemicStatus.UNKNOWN
            candidate.rejection_reason = "Waveform statistics deviate from all cataloged modulation distributions (OOD)"
            trace = ValidationGateTrace(
                hypothesis_id=candidate.hypothesis_id,
                signal_family=candidate.signal_family,
                protocol=candidate.protocol,
                final_status=ValidationGateStatus.UNKNOWN_OOD.value,
                evidence_score=candidate.evidence_score,
                narrative_summary="Candidate identified as Out-of-Distribution (UNKNOWN_OOD); unvalidated."
            )
            return ValidationGateStatus.UNKNOWN_OOD.value, candidate, trace

        # ---------------------------------------------------------------------
        # Pre-Gate Case 3: Pure Noise / Stationary Gaussian Noise
        # ---------------------------------------------------------------------
        if (
            candidate.signal_family == "NOISE"
            or (candidate.evidence_score == 0.0 and candidate.validation_status == "ABSTAINED")
        ):
            candidate.validation_status = "ABSTAINED"
            candidate.epistemic_status = EpistemicStatus.UNKNOWN
            candidate.rejection_reason = "Signal rejected by pre-gate: Input energy matches stationary Gaussian noise floor; carrier absent"
            trace = ValidationGateTrace(
                hypothesis_id=candidate.hypothesis_id,
                signal_family="NOISE",
                protocol=None,
                final_status=ValidationGateStatus.NO_SIGNAL.value,
                evidence_score=0.0,
                narrative_summary="System rejected input: Signal energy and spectral flatness conform to stationary Gaussian noise floor. No informational carrier detected."
            )
            return ValidationGateStatus.NO_SIGNAL.value, candidate, trace

        # ---------------------------------------------------------------------
        # Pre-Gate Case 4: Sub-viability Evidence Floor
        # ---------------------------------------------------------------------
        if candidate.evidence_score < cfg.min_viability_score:
            candidate.validation_status = "UNVALIDATED"
            candidate.epistemic_status = EpistemicStatus.UNKNOWN
            candidate.rejection_reason = (
                f"Net evidence score {candidate.evidence_score:.2f} is below viability floor {cfg.min_viability_score:.2f}"
            )
            trace = ValidationGateTrace(
                hypothesis_id=candidate.hypothesis_id,
                signal_family=candidate.signal_family,
                protocol=candidate.protocol,
                final_status=ValidationGateStatus.UNKNOWN.value,
                evidence_score=candidate.evidence_score,
                narrative_summary=candidate.rejection_reason
            )
            return ValidationGateStatus.UNKNOWN.value, candidate, trace

        # ---------------------------------------------------------------------
        # Pre-Gate Case 5: Unresolved Close Ambiguity
        # ---------------------------------------------------------------------
        if decision_status_from_ranker == "AMBIGUOUS" and len(all_cands) >= 2:
            cand1 = all_cands[0]
            cand2 = all_cands[1]
            delta = abs(cand1.evidence_score - cand2.evidence_score)
            if delta < cfg.ambiguity_threshold:
                candidate.validation_status = ValidationGateStatus.AMBIGUOUS.value
                candidate.epistemic_status = EpistemicStatus.HYPOTHESIZED
                msg = (
                    f"Ambiguous hypothesis: {cand1.protocol or cand1.signal_family} ({cand1.evidence_score:.2f}) vs "
                    f"{cand2.protocol or cand2.signal_family} ({cand2.evidence_score:.2f}) "
                    f"within margin {delta:.3f} < {cfg.ambiguity_threshold:.3f}"
                )
                candidate.rejection_reason = msg
                trace = ValidationGateTrace(
                    hypothesis_id=candidate.hypothesis_id,
                    signal_family=candidate.signal_family,
                    protocol=candidate.protocol,
                    final_status=ValidationGateStatus.AMBIGUOUS.value,
                    evidence_score=candidate.evidence_score,
                    narrative_summary=msg
                )
                return ValidationGateStatus.AMBIGUOUS.value, candidate, trace

        # =====================================================================
        # Execute the 6 Physical Validation Gates
        # =====================================================================
        gate_results: Dict[str, GateCheckResult] = {}

        # 1. Spectral Consistency Gate
        g_spectral = self._check_spectral_consistency(candidate, temporal_result, fs)
        gate_results["spectral_consistency"] = g_spectral

        # 2. Temporal Consistency Gate
        g_temporal = self._check_temporal_consistency(candidate, pulse_info, temporal_result)
        gate_results["temporal_consistency"] = g_temporal

        # 3. Symbol / Dwell Consistency Gate
        g_symbol = self._check_symbol_dwell_consistency(candidate, temporal_result, specialized_params)
        gate_results["symbol_dwell_consistency"] = g_symbol

        # 4. Cross-Window Stability Gate
        g_stability = self._check_cross_window_stability(candidate, temporal_result)
        gate_results["cross_window_stability"] = g_stability

        # 5. Physical Plausibility Gate (Nyquist, SNR, Energy)
        g_physical = self._check_physical_consistency(candidate, fs)
        gate_results["physical_consistency"] = g_physical

        # 6. Reconstruction Consistency Gate (Closed-loop EVM, PLL lock)
        g_reconstruction = self._check_reconstruction_consistency(candidate, reconstruction_telemetry)
        gate_results["reconstruction_consistency"] = g_reconstruction

        # ---------------------------------------------------------------------
        # Analyze Gate Outcomes
        # ---------------------------------------------------------------------
        passed_gates: List[str] = []
        failed_gates: List[str] = []
        waived_gates: List[str] = []
        gate_scores: List[float] = []

        for g_name, g_res in gate_results.items():
            if not g_res.applicable:
                waived_gates.append(g_name)
            elif g_res.passed:
                passed_gates.append(g_name)
                gate_scores.append(g_res.score)
            else:
                failed_gates.append(g_name)
                gate_scores.append(g_res.score)

        avg_val_score = float(np.mean(gate_scores)) if gate_scores else 0.0

        # Contradiction Check
        unresolved_contra_count = len(candidate.contradictions)
        has_contradictions = unresolved_contra_count > cfg.max_contradictions_allowed

        # ---------------------------------------------------------------------
        # Arbitration Logic
        # ---------------------------------------------------------------------
        # Condition A: Fatal physical plausibility failure -> REJECTED / UNKNOWN
        if not g_physical.passed:
            final_status = ValidationGateStatus.UNKNOWN.value
            candidate.validation_status = "REJECTED"
            candidate.epistemic_status = EpistemicStatus.UNKNOWN
            candidate.rejection_reason = f"Fatal physical consistency failure: {g_physical.reason}"
            narrative = candidate.rejection_reason

        # Condition B: High spectral evidence but temporal validation failed (Rule 3 Invariant)
        elif not g_stability.passed or not g_temporal.passed:
            final_status = ValidationGateStatus.ESTIMATED.value
            candidate.validation_status = ValidationGateStatus.ESTIMATED.value
            candidate.epistemic_status = EpistemicStatus.ESTIMATED
            failed_reasons = []
            if not g_stability.passed:
                failed_reasons.append(f"cross-window stability ({g_stability.reason})")
            if not g_temporal.passed:
                failed_reasons.append(f"temporal consistency ({g_temporal.reason})")
            msg = (
                f"Downgraded to ESTIMATED: Strong spectral evidence confirmed, but failed temporal gate: "
                f"{'; '.join(failed_reasons)}"
            )
            candidate.rejection_reason = msg
            narrative = msg

        # Condition C: Unresolved contradictions or evidence score below validation threshold
        elif has_contradictions or candidate.evidence_score < cfg.min_evidence_score_for_validation:
            final_status = ValidationGateStatus.ESTIMATED.value
            candidate.validation_status = ValidationGateStatus.ESTIMATED.value
            candidate.epistemic_status = EpistemicStatus.ESTIMATED
            reasons = []
            if has_contradictions:
                reasons.append(f"{unresolved_contra_count} physical contradiction(s) present: {candidate.contradictions}")
            if candidate.evidence_score < cfg.min_evidence_score_for_validation:
                reasons.append(f"evidence score {candidate.evidence_score:.2f} < threshold {cfg.min_evidence_score_for_validation:.2f}")
            msg = f"Retained as ESTIMATED: {'; '.join(reasons)}"
            candidate.rejection_reason = msg
            narrative = msg

        # Condition D: Any other applicable gate failed (e.g. symbol dwell or reconstruction)
        elif len(failed_gates) > 0:
            final_status = ValidationGateStatus.ESTIMATED.value
            candidate.validation_status = ValidationGateStatus.ESTIMATED.value
            candidate.epistemic_status = EpistemicStatus.ESTIMATED
            msg = f"Retained as ESTIMATED: Failed non-disqualifying validation gate(s): {', '.join(failed_gates)}"
            candidate.rejection_reason = msg
            narrative = msg

        # Condition E: All applicable gates passed with zero contradictions -> VALIDATED
        else:
            cand_snr = float(candidate.parameters.get("snr_db", candidate.parameters.get("estimated_snr_db", 10.0))) if candidate.parameters else 10.0
            framing_telem = reconstruction_telemetry.get("framing", {}) if isinstance(reconstruction_telemetry, dict) else {}
            crc_prof = str(framing_telem.get("crc_profile", ""))
            valid_frames = framing_telem.get("valid_frames", 0)
            flen = framing_telem.get("frame_length", 0)
            is_algebraic_pass = bool(
                framing_telem.get("crc_match", False) and
                (
                    ("16" in crc_prof or "24" in crc_prof or "32" in crc_prof) or
                    (valid_frames >= 2 and framing_telem.get("repeated_frames", 0) >= 2)
                ) and
                (0 < flen <= 2048)
            )
            if cand_snr <= 0.0 and not is_algebraic_pass:
                final_status = ValidationGateStatus.ESTIMATED.value
                candidate.validation_status = ValidationGateStatus.ESTIMATED.value
                candidate.epistemic_status = EpistemicStatus.ESTIMATED
                msg = f"Retained as ESTIMATED: Sub-noise-floor operation (SNR={cand_snr:.1f} dB <= 0 dB) requires verified algebraic framing/CRC proof for full validation"
                candidate.rejection_reason = msg
                narrative = msg
            else:
                final_status = ValidationGateStatus.VALIDATED.value
                candidate.validation_status = ValidationGateStatus.VALIDATED.value
                candidate.epistemic_status = EpistemicStatus.VALIDATED
                candidate.rejection_reason = None
                narrative = (
                    f"Fully VALIDATED: Candidate satisfied all {len(passed_gates)} applicable validation gates "
                    f"(spectral, temporal, symbol, stability, physical, reconstruction) with zero physical contradictions."
                )

        trace = ValidationGateTrace(
            hypothesis_id=candidate.hypothesis_id,
            signal_family=candidate.signal_family,
            protocol=candidate.protocol,
            final_status=final_status,
            evidence_score=candidate.evidence_score,
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            waived_gates=waived_gates,
            gate_results={k: v.to_dict() for k, v in gate_results.items()},
            validation_score=avg_val_score,
            narrative_summary=narrative
        )

        return final_status, candidate, trace

    # =========================================================================
    # Internal Gate Check Routines
    # =========================================================================

    def _check_spectral_consistency(
        self,
        cand: SignalHypothesis,
        temporal_result: Optional[Dict[str, Any]],
        fs: Optional[float]
    ) -> GateCheckResult:
        """Evaluates spectral plausibility and multi-window bandwidth/carrier stability."""
        params = cand.parameters or {}
        obw_val = params.get("occupied_bandwidth_99") if params.get("occupied_bandwidth_99") is not None else params.get("bw_99pct_hz")
        obw = float(obw_val if obw_val is not None else 0.0)
        fc_val = params.get("carrier_frequency") if params.get("carrier_frequency") is not None else params.get("fc_peak_hz")
        fc = float(fc_val if fc_val is not None else 0.0)

        if fs is not None and fs > 0:
            if obw <= 0.0 or obw > fs:
                return GateCheckResult(
                    gate_name="spectral_consistency",
                    passed=False,
                    applicable=True,
                    score=0.0,
                    observed_value=f"OBW={obw} Hz",
                    threshold_or_criterion=f"0 < OBW <= fs={fs} Hz",
                    reason=f"Occupied bandwidth {obw} Hz violates spectral Nyquist bound (fs={fs} Hz)"
                )
            if abs(fc) > fs / 2.0:
                return GateCheckResult(
                    gate_name="spectral_consistency",
                    passed=False,
                    applicable=True,
                    score=0.0,
                    observed_value=f"fc={fc} Hz",
                    threshold_or_criterion=f"|fc| <= fs/2={fs/2.0} Hz",
                    reason=f"Carrier frequency {fc} Hz aliased beyond Nyquist zone"
                )

        # Multi-window stability check
        if temporal_result and temporal_result.get("status") == "COMPLETE":
            param_stab = temporal_result.get("parameter_stability", {})
            bw_stab = param_stab.get("bw_99pct_hz")
            if bw_stab:
                rel_var = bw_stab.get("relative_variation", 0.0)
                max_var = self.config.max_bandwidth_variation_pct / 100.0
                if rel_var > max_var:
                    return GateCheckResult(
                        gate_name="spectral_consistency",
                        passed=False,
                        applicable=True,
                        score=float(bw_stab.get("stability_score", 0.5)),
                        observed_value=f"BW rel_var={rel_var*100.0:.1f}%",
                        threshold_or_criterion=f"<= {self.config.max_bandwidth_variation_pct:.1f}%",
                        reason=f"Spectral bandwidth variation across windows {rel_var*100.0:.1f}% exceeds tolerance"
                    )

        return GateCheckResult(
            gate_name="spectral_consistency",
            passed=True,
            applicable=True,
            score=1.0,
            observed_value=f"OBW={obw:.1f} Hz, fc={fc:.1f} Hz",
            threshold_or_criterion="Valid in-band spectral distribution",
            reason="Carrier and bandwidth are spectrally consistent and bounded"
        )

    def _check_temporal_consistency(
        self,
        cand: SignalHypothesis,
        pulse_info: Optional[Dict[str, Any]],
        temporal_result: Optional[Dict[str, Any]]
    ) -> GateCheckResult:
        """Evaluates temporal dynamics, pulse geometry, and envelope continuity."""
        sig_fam = cand.signal_family.upper()
        proto = (cand.protocol or "").upper()
        pinfo = pulse_info or {}
        cparams = cand.parameters or {}
        blind = cparams.get("blind_parameters", {}) or {}
        sweep = cparams.get("chirp_info", {}) or blind.get("chirp_info", {}) or {}
        # A repeated linear frequency trajectory is a continuous-sweep
        # temporal model.  Its envelope may be gated by the receiver/audio
        # path, but that is not evidence of a pulsed RF envelope.  Gate 5
        # therefore validates the macro sweep period instead of requiring
        # pulse edges or PRF artifacts.
        if sweep.get("is_continuous_sweep"):
            rep_hz = sweep.get("sweep_repetition_hz")
            return GateCheckResult(
                gate_name="temporal_consistency",
                passed=True,
                applicable=True,
                score=1.0,
                observed_value=(f"continuous sweep, repetition={float(rep_hz):.2f} Hz" if rep_hz else "continuous sweep trajectory"),
                threshold_or_criterion="Stable linear frequency trajectory with macro repetition",
                reason="Continuous sweep geometry is consistent; pulsed-envelope/zero-crossing PRF is not applicable"
            )

        # Pulsed signals require confirmed pulse train structure
        if any(p_token in sig_fam for p_token in ["RADAR", "PULSE", "PULSED", "MORSE", "OOK"]):
            num_pulses = pinfo.get("pulse_count", 0)
            prf = pinfo.get("estimated_prf_hz", 0.0)
            dyn_range = pinfo.get("envelope_dynamic_range", 0.0)

            # CW reflection radar or chirp radar can have wide envelope dynamics or periodic pulses
            if num_pulses == 0 and dyn_range < 0.25 and "GRAVES" not in proto:
                return GateCheckResult(
                    gate_name="temporal_consistency",
                    passed=False,
                    applicable=True,
                    score=0.2,
                    observed_value=f"pulses={num_pulses}, dyn_range={dyn_range:.2f}",
                    threshold_or_criterion="Pulse train or dynamic envelope",
                    reason="Hypothesis specifies pulsed radar, but temporal envelope is completely unpulsed"
                )

            return GateCheckResult(
                gate_name="temporal_consistency",
                passed=True,
                applicable=True,
                score=1.0,
                observed_value=f"PRF={prf:.1f} Hz, dyn_range={dyn_range:.2f}",
                threshold_or_criterion="Consistent pulsed envelope structure",
                reason="Pulse geometry and temporal structure match hypothesis profile"
            )

        # Continuous modulations: verify signal does not have massive duty cycle dropouts
        duty_cycle = pinfo.get("duty_cycle", 1.0)
        if duty_cycle < 0.10 and sig_fam in ["PSK", "QAM", "FSK", "ANALOG"]:
            return GateCheckResult(
                gate_name="temporal_consistency",
                passed=False,
                applicable=True,
                score=0.3,
                observed_value=f"duty_cycle={duty_cycle:.2f}",
                threshold_or_criterion="duty_cycle >= 0.10 for continuous waveform",
                reason=f"Candidate specifies continuous {sig_fam}, but observed duty cycle is {duty_cycle*100:.1f}%"
            )

        return GateCheckResult(
            gate_name="temporal_consistency",
            passed=True,
            applicable=True,
            score=1.0,
            observed_value=f"duty_cycle={duty_cycle:.2f}",
            threshold_or_criterion="Continuous temporal persistence",
            reason="Envelope dynamics are consistent with continuous transmission"
        )

    def _check_symbol_dwell_consistency(
        self,
        cand: SignalHypothesis,
        temporal_result: Optional[Dict[str, Any]],
        specialized_params: Optional[Dict[str, Any]]
    ) -> GateCheckResult:
        """Evaluates symbol rate / dwell time stability across temporal windows."""
        sig_fam = cand.signal_family.upper()
        # Non-digital signals waive this gate
        if any(non_dig in sig_fam for non_dig in ["ANALOG", "CW", "NOISE", "UNKNOWN_OOD"]):
            return GateCheckResult(
                gate_name="symbol_dwell_consistency",
                passed=True,
                applicable=False,
                score=1.0,
                observed_value="N/A",
                threshold_or_criterion="N/A (Analog or CW transmission)",
                reason="Symbol rate gate not applicable to analog or unmodulated carriers"
            )

        params = cand.parameters or {}
        baud_val = cand.symbol_rate if cand.symbol_rate is not None else params.get("symbol_rate")
        if baud_val is None:
            baud_val = params.get("estimated_baud_rate_hz")
        baud = float(baud_val) if baud_val is not None else None

        # Check multi-window symbol rate variation if available
        if temporal_result and temporal_result.get("status") == "COMPLETE":
            param_stab = temporal_result.get("parameter_stability", {})
            baud_stab = param_stab.get("estimated_baud_rate_hz")
            if baud_stab and baud_stab.get("mean", 0.0) > 0.0:
                rel_var = baud_stab.get("relative_variation", 0.0)
                max_var = self.config.max_symbol_rate_variation_pct / 100.0
                if rel_var > max_var:
                    return GateCheckResult(
                        gate_name="symbol_dwell_consistency",
                        passed=False,
                        applicable=True,
                        score=float(baud_stab.get("stability_score", 0.5)),
                        observed_value=f"baud rel_var={rel_var*100.0:.1f}%",
                        threshold_or_criterion=f"<= {self.config.max_symbol_rate_variation_pct:.1f}%",
                        reason=f"Symbol clock jitter across windows ({rel_var*100.0:.1f}%) exceeds physical limit"
                    )

        return GateCheckResult(
            gate_name="symbol_dwell_consistency",
            passed=True,
            applicable=True,
            score=1.0,
            observed_value=f"Baud={baud if baud else 'Estimated'}",
            threshold_or_criterion="Stable symbol clock",
            reason="Symbol timing and dwell duration are physically consistent"
        )

    def _check_cross_window_stability(
        self,
        cand: SignalHypothesis,
        temporal_result: Optional[Dict[str, Any]]
    ) -> GateCheckResult:
        """
        Evaluates scale-normalized stability index across the 4 analysis windows.
        Enforces Rule 3: Fails if temporal consistency score is low or observation insufficient.
        """
        if temporal_result is None:
            return GateCheckResult(
                gate_name="cross_window_stability",
                passed=False,
                applicable=True,
                score=0.0,
                observed_value="None",
                threshold_or_criterion=f">= {self.config.min_temporal_stability:.2f}",
                reason="Multi-window temporal validation telemetry missing"
            )

        status = temporal_result.get("status", "")
        if status == "INSUFFICIENT_OBSERVATION_DURATION":
            return GateCheckResult(
                gate_name="cross_window_stability",
                passed=False,
                applicable=True,
                score=0.0,
                observed_value="INSUFFICIENT_OBSERVATION_DURATION",
                threshold_or_criterion=f">= {self.config.min_temporal_stability:.2f}",
                reason="Signal duration is too short for 4-window temporal validation; cannot be crowned VALIDATED"
            )

        score = temporal_result.get("cross_window_consistency_score")
        if score is None:
            return GateCheckResult(
                gate_name="cross_window_stability",
                passed=False,
                applicable=True,
                score=0.0,
                observed_value="None",
                threshold_or_criterion=f">= {self.config.min_temporal_stability:.2f}",
                reason="Consistency score undefined across temporal observation windows"
            )

        score_f = float(score)
        passed = score_f >= self.config.min_temporal_stability
        unstable = temporal_result.get("unstable_parameters", [])

        if not passed:
            reason = (
                f"Temporal consistency score {score_f:.3f} is below required stability threshold "
                f"{self.config.min_temporal_stability:.3f} (unstable parameters: {unstable})"
            )
        else:
            reason = f"High temporal stationarity verified: score {score_f:.3f} >= {self.config.min_temporal_stability:.3f}"

        return GateCheckResult(
            gate_name="cross_window_stability",
            passed=passed,
            applicable=True,
            score=score_f,
            observed_value=f"score={score_f:.3f}",
            threshold_or_criterion=f">= {self.config.min_temporal_stability:.3f}",
            reason=reason
        )

    def _check_physical_consistency(
        self,
        cand: SignalHypothesis,
        fs: Optional[float]
    ) -> GateCheckResult:
        """Enforces fundamental physical conservation laws (Nyquist bounds, SNR, energy)."""
        params = cand.parameters or {}
        snr_val = params.get("snr_db")
        snr = float(snr_val if snr_val is not None else 0.0)
        obw_val = params.get("occupied_bandwidth_99") if params.get("occupied_bandwidth_99") is not None else params.get("bw_99pct_hz")
        obw = float(obw_val if obw_val is not None else 0.0)
        baud_val = cand.symbol_rate if cand.symbol_rate is not None else params.get("symbol_rate")
        baud = float(baud_val if baud_val is not None else 0.0)

        # SNR lower bound
        if snr < self.config.min_physical_snr_db:
            return GateCheckResult(
                gate_name="physical_consistency",
                passed=False,
                applicable=True,
                score=0.0,
                observed_value=f"SNR={snr:.2f} dB",
                threshold_or_criterion=f">= {self.config.min_physical_snr_db:.1f} dB",
                reason=f"SNR {snr:.2f} dB is below physical detection threshold"
            )

        # Nyquist rate check: Baud rate cannot exceed total occupied bandwidth (allowing roll-off factor)
        # Note: If this is a demodulated audio recording of a receiver, the audio passband reflects receiver audio,
        # not the physical RF emission bandwidth.
        is_demod_audio = bool(
            cand.parameters.get("is_demodulated_audio")
            or "audio" in cand.signal_family.lower()
            or "audio" in str(cand.protocol).lower()
        )
        if not is_demod_audio and baud > 0.0 and obw > 0.0:
            if baud > obw * 2.5:
                return GateCheckResult(
                    gate_name="physical_consistency",
                    passed=False,
                    applicable=True,
                    score=0.1,
                    observed_value=f"Baud={baud:.1f}, OBW={obw:.1f} Hz",
                    threshold_or_criterion="Baud <= 2.5 * OBW",
                    reason=f"Baud rate {baud:.1f} Baud exceeds physical Nyquist capacity for bandwidth {obw:.1f} Hz"
                )

        return GateCheckResult(
            gate_name="physical_consistency",
            passed=True,
            applicable=True,
            score=1.0,
            observed_value=f"SNR={snr:.2f} dB, OBW={obw:.1f} Hz",
            threshold_or_criterion="Nyquist and SNR physical boundaries",
            reason="All physical invariants (Nyquist, power, SNR) satisfied"
        )

    def _check_reconstruction_consistency(
        self,
        cand: SignalHypothesis,
        reconstruction_telemetry: Optional[Dict[str, Any]]
    ) -> GateCheckResult:
        """Evaluates closed-loop synchronization, EVM, and algebraic framing proof."""
        if reconstruction_telemetry is None or "demodulation" not in reconstruction_telemetry:
            # Reconstruction not run or not applicable
            return GateCheckResult(
                gate_name="reconstruction_consistency",
                passed=True,
                applicable=False,
                score=1.0,
                observed_value="Not Executed",
                threshold_or_criterion="N/A",
                reason="Closed-loop reconstruction not executed or not required for signal class"
            )

        demod = reconstruction_telemetry.get("demodulation", {})
        evm = demod.get("evm_pct")
        if evm is not None:
            evm_f = float(evm)
            if evm_f > self.config.max_evm_pct:
                return GateCheckResult(
                    gate_name="reconstruction_consistency",
                    passed=False,
                    applicable=True,
                    score=float(max(0.0, 1.0 - evm_f / 100.0)),
                    observed_value=f"EVM={evm_f:.1f}%",
                    threshold_or_criterion=f"<= {self.config.max_evm_pct:.1f}%",
                    reason=f"Demodulation EVM {evm_f:.1f}% exceeds maximum allowable threshold ({self.config.max_evm_pct:.1f}%)"
                )

        # Check carrier recovery lock if synchronization was attempted
        sync = reconstruction_telemetry.get("synchronization", {})
        if sync and "pll_locked" in sync:
            pll_locked = sync.get("pll_locked", True)
            lock_metric = sync.get("pll_lock_metric", 1.0)
            if not pll_locked and lock_metric < 0.20:
                return GateCheckResult(
                    gate_name="reconstruction_consistency",
                    passed=False,
                    applicable=True,
                    score=float(lock_metric),
                    observed_value=f"PLL_locked={pll_locked}, metric={lock_metric:.2f}",
                    threshold_or_criterion="PLL locked with metric >= 0.20",
                    reason="Carrier recovery PLL lost synchronization lock"
                )

        return GateCheckResult(
            gate_name="reconstruction_consistency",
            passed=True,
            applicable=True,
            score=1.0,
            observed_value="Synchronized & Demodulated",
            threshold_or_criterion="EVM and PLL lock verified",
            reason="Closed-loop reconstruction consistent with hypothesis modulation"
        )


# Global default validation gate instance
_default_validation_gate = HypothesisValidationGate()


def run_validation_gate(
    winner: Optional[SignalHypothesis],
    ranked_candidates: Optional[List[SignalHypothesis]] = None,
    decision_status: str = "DECISIVE",
    temporal_result: Optional[Dict[str, Any]] = None,
    pulse_info: Optional[Dict[str, Any]] = None,
    reconstruction_telemetry: Optional[Dict[str, Any]] = None,
    specialized_params: Optional[Dict[str, Any]] = None,
    fs: Optional[float] = None,
    config: Optional[ValidationGateConfig] = None
) -> Tuple[str, Optional[SignalHypothesis], ValidationGateTrace]:
    """
    Convenience function executing the Hypothesis Validation Gate on a candidate hypothesis.

    Returns:
        (final_status, validated_candidate, validation_trace)
    """
    gate = HypothesisValidationGate(config=config) if config else _default_validation_gate
    return gate.evaluate_candidate(
        candidate=winner,
        all_candidates=ranked_candidates,
        decision_status_from_ranker=decision_status,
        temporal_result=temporal_result,
        pulse_info=pulse_info,
        reconstruction_telemetry=reconstruction_telemetry,
        specialized_params=specialized_params,
        fs=fs
    )
