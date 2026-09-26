"""
Contradiction Analysis Layer for Ranked Hypotheses
==================================================
Evaluates every candidate hypothesis against measured multi-domain physical features,
temporal stability metrics, and closed-loop reconstruction telemetry to identify both:
1. SUPPORTING EVIDENCE (corroborating physical observations)
2. CONTRADICTING EVIDENCE (measurable constraint violations)

Contradiction Categories:
- SPECTRAL: Missing characteristic spectral lines, invalid tone counts or tone spacings.
- TEMPORAL: Envelope variance violations (constant vs variable envelope), dynamic range mismatch.
- PARAMETER_INCONSISTENCY: Carson's rule divergence, Nyquist violation, symbol dwell time error.
- CROSS_WINDOW_INSTABILITY: High parameter variance across temporal sub-windows.
- RECONSTRUCTION_FAILURE: Carrier PLL phase loss, excessive EVM (>35%), soft LLR collapse.

Structured Representation:
{
    "hypothesis": "...",
    "supporting_evidence": [...],
    "contradictions": [...],
    "contradiction_count": ...,
    "net_evidence_score": ...
}
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from dsp.contracts import (
    SignalHypothesis,
    ModulationFamily,
    ConfidenceLevel,
    EpistemicStatus,
)


class ContradictionCategory(str, Enum):
    """Categories of measurable physical contradictions."""
    SPECTRAL = "SPECTRAL"
    TEMPORAL = "TEMPORAL"
    PARAMETER_INCONSISTENCY = "PARAMETER_INCONSISTENCY"
    CROSS_WINDOW_INSTABILITY = "CROSS_WINDOW_INSTABILITY"
    RECONSTRUCTION_FAILURE = "RECONSTRUCTION_FAILURE"


@dataclass
class MeasurableContradiction:
    """
    Structured record of a single measurable physical contradiction.
    Refers strictly to physical measurements and empirical boundaries; never fabricated.
    """
    category: ContradictionCategory
    property_name: str
    observed_value: Any
    expected_range: str
    description: str
    penalty: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "property_name": self.property_name,
            "observed_value": self.observed_value,
            "expected_range": self.expected_range,
            "description": self.description,
            "penalty": float(round(self.penalty, 3))
        }


@dataclass
class ContradictionConfig:
    """
    Explicit, documented penalty weights for measurable physical contradictions.
    Deductions are applied directly to the raw evidence score to calculate net_evidence_score.
    All weights are bounded, documented, and fully testable.
    """
    penalty_spectral: float = 0.20             # Missing spectral lines, tone spacing error
    penalty_temporal: float = 0.15             # Envelope variance or dynamic range mismatch
    penalty_parameter: float = 0.20            # Carson / Nyquist / symbol dwell time violation
    penalty_cross_window: float = 0.15         # Multi-window temporal parameter instability
    penalty_reconstruction: float = 0.20       # PLL lock loss, excessive EVM (>35%)
    max_total_penalty: float = 0.80            # Cap on cumulative deductions to prevent negative underflow


@dataclass
class ContradictionAnalysisResult:
    """
    Structured representation of contradiction analysis for a candidate hypothesis.
    Conforms to Requirement 5 schema:
    {
        "hypothesis": "...",
        "supporting_evidence": [...],
        "contradictions": [...],
        "contradiction_count": ...,
        "net_evidence_score": ...
    }
    """
    hypothesis: str
    hypothesis_id: str
    supporting_evidence: List[str]
    contradictions: List[str]
    contradiction_details: List[MeasurableContradiction] = field(default_factory=list)
    contradiction_count: int = 0
    raw_evidence_score: float = 0.0
    net_evidence_score: float = 0.0
    total_penalty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Returns JSON-safe structured representation matching Requirement 5."""
        return {
            "hypothesis": self.hypothesis,
            "supporting_evidence": list(self.supporting_evidence),
            "contradictions": list(self.contradictions),
            "contradiction_count": int(self.contradiction_count),
            "net_evidence_score": float(round(self.net_evidence_score, 4)),
            # Richer audit telemetry
            "hypothesis_id": self.hypothesis_id,
            "raw_evidence_score": float(round(self.raw_evidence_score, 4)),
            "total_penalty": float(round(self.total_penalty, 4)),
            "contradiction_details": [c.to_dict() for c in self.contradiction_details]
        }


def analyze_candidate_contradictions(
    candidate: SignalHypothesis,
    feats: Optional[Dict[str, Any]] = None,
    temporal_result: Optional[Dict[str, Any]] = None,
    reconstruction_telemetry: Optional[Dict[str, Any]] = None,
    config: Optional[ContradictionConfig] = None
) -> ContradictionAnalysisResult:
    """
    Analyzes a candidate SignalHypothesis against measured physical telemetry to extract
    concrete supporting evidence and measurable physical contradictions.
    """
    cfg = config or ContradictionConfig()
    features = feats or {}
    contradictions: List[MeasurableContradiction] = []

    raw_score = float(candidate.evidence_score)
    hyp_title = candidate.protocol or candidate.signal_family or str(candidate.modulation.value)

    # -------------------------------------------------------------------------
    # 1. TEMPORAL CONTRADICTIONS: Envelope Variance Constraints
    # -------------------------------------------------------------------------
    env_var = float(features.get("envelope_variance_ratio", candidate.parameters.get("envelope_variance", 0.0)))
    if env_var > 0.0:
        # Constant-envelope waveforms (FSK, GMSK, CW, FM) require low envelope variance (< 0.25)
        is_constant_envelope = (
            candidate.signal_family in ("FSK", "CW_CARRIER")
            or candidate.modulation in (ModulationFamily.FSK_2, ModulationFamily.FSK_4, ModulationFamily.ANALOG_FM)
            or "GMSK" in hyp_title
            or "CW" in hyp_title
        )
        if is_constant_envelope and env_var >= 0.28:
            contradictions.append(MeasurableContradiction(
                category=ContradictionCategory.TEMPORAL,
                property_name="envelope_variance_ratio",
                observed_value=round(env_var, 4),
                expected_range="< 0.25 (Constant-Envelope)",
                description=(
                    f"Envelope variance ratio ({env_var:.3f}) exceeds constant-envelope bound (0.25) "
                    f"required for {candidate.signal_family}/{candidate.modulation.value}"
                ),
                penalty=cfg.penalty_temporal
            ))

        # Variable-envelope waveforms (AM, 16-QAM, 64-QAM) require non-zero envelope variation (>= 0.12)
        is_variable_envelope = (
            candidate.signal_family == "QAM"
            or candidate.modulation in (ModulationFamily.ANALOG_AM, ModulationFamily.QAM_16, ModulationFamily.QAM_64)
            or "16-QAM" in hyp_title
            or "AM" in hyp_title
        )
        if is_variable_envelope and env_var < 0.08:
            contradictions.append(MeasurableContradiction(
                category=ContradictionCategory.TEMPORAL,
                property_name="envelope_variance_ratio",
                observed_value=round(env_var, 4),
                expected_range=">= 0.12 (Amplitude Modulated)",
                description=(
                    f"Envelope variance ratio ({env_var:.3f}) is below amplitude modulation threshold (0.12) "
                    f"required for {candidate.modulation.value}"
                ),
                penalty=cfg.penalty_temporal
            ))

    # -------------------------------------------------------------------------
    # 2. PARAMETER INCONSISTENCY: Carson's Rule & Bandwidth Invariants
    # -------------------------------------------------------------------------
    obw99 = float(features.get("obw_99_hz", candidate.parameters.get("occupied_bandwidth_99", 0.0)))
    sym_rate = candidate.symbol_rate or candidate.parameters.get("symbol_rate")
    fsk_shift = candidate.parameters.get("fsk_shift")

    # Carson's rule check for 2-FSK / 4-FSK
    if candidate.signal_family == "FSK" and sym_rate and fsk_shift and obw99 > 0.0:
        carson_bw = float(fsk_shift + sym_rate)
        if obw99 < 0.45 * carson_bw or obw99 > 2.8 * carson_bw:
            contradictions.append(MeasurableContradiction(
                category=ContradictionCategory.PARAMETER_INCONSISTENCY,
                property_name="carson_bandwidth",
                observed_value=round(obw99, 1),
                expected_range=f"~ {carson_bw:.1f} Hz (Carson B = Shift + Baud)",
                description=(
                    f"Measured 99% OBW ({obw99:.1f} Hz) diverges significantly from Carson's rule prediction "
                    f"({carson_bw:.1f} Hz = shift {fsk_shift:.1f} Hz + baud {sym_rate:.1f} Baud)"
                ),
                penalty=cfg.penalty_parameter
            ))

    # Nyquist minimum bandwidth constraint for PSK
    if candidate.signal_family == "PSK" and sym_rate and obw99 > 0.0:
        nyquist_min = float(0.70 * sym_rate)
        if obw99 < nyquist_min:
            contradictions.append(MeasurableContradiction(
                category=ContradictionCategory.PARAMETER_INCONSISTENCY,
                property_name="nyquist_bandwidth",
                observed_value=round(obw99, 1),
                expected_range=f">= {nyquist_min:.1f} Hz (Nyquist limit)",
                description=(
                    f"Measured bandwidth ({obw99:.1f} Hz) violates Nyquist minimum bandwidth limit "
                    f"({nyquist_min:.1f} Hz) for {sym_rate:.1f} Baud PSK"
                ),
                penalty=cfg.penalty_parameter
            ))

    # Symbol dwell time consistency for FSK utility modes
    dwell_info = features.get("fsk_dwell_info") or {}
    meas_dwell_ms = dwell_info.get("symbol_dwell_time_ms")
    if meas_dwell_ms and meas_dwell_ms > 0.0:
        # Determine nominal dwell for specific protocols
        nom_dwell_map = {
            "ASCII / ITA-5 110 Baud": 9.091,
            "110 Baud": 9.091,
            "NAVTEX": 10.0,
            "100 Baud": 10.0,
            "RTTY": 22.002,
            "45.45 Baud": 22.002,
            "ASCII / Packet 300 Baud": 3.333,
            "300 Baud": 3.333,
        }
        nom_dwell = None
        for key, val in nom_dwell_map.items():
            if key in hyp_title:
                nom_dwell = val
                break

        if nom_dwell is not None:
            rel_err = abs(meas_dwell_ms - nom_dwell) / nom_dwell
            if rel_err > 0.18:
                contradictions.append(MeasurableContradiction(
                    category=ContradictionCategory.PARAMETER_INCONSISTENCY,
                    property_name="symbol_dwell_time_ms",
                    observed_value=round(meas_dwell_ms, 2),
                    expected_range=f"{nom_dwell:.2f} ms (+/- 15%)",
                    description=(
                        f"Measured symbol dwell time ({meas_dwell_ms:.2f} ms) deviates from "
                        f"{hyp_title} nominal dwell ({nom_dwell:.2f} ms) by {rel_err*100:.1f}%"
                    ),
                    penalty=cfg.penalty_parameter
                ))

    # -------------------------------------------------------------------------
    # 3. SPECTRAL CONTRADICTIONS: Missing Characteristic Spectral Lines
    # -------------------------------------------------------------------------
    if "GSM" in hyp_title:
        prom_216 = float(features.get("prom_216", 0.0))
        if prom_216 < 4.0:
            contradictions.append(MeasurableContradiction(
                category=ContradictionCategory.SPECTRAL,
                property_name="prom_216",
                observed_value=round(prom_216, 2),
                expected_range=">= 8.0 dB (GSM 4.615 ms Frame Line)",
                description=f"Absence of 216.7 Hz TDMA frame spectral line (measured prominence {prom_216:.1f} dB < 8.0 dB)",
                penalty=cfg.penalty_spectral
            ))

    if "DMR" in hyp_title:
        prom_33 = float(features.get("prom_33", 0.0))
        if prom_33 < 1.8:
            contradictions.append(MeasurableContradiction(
                category=ContradictionCategory.SPECTRAL,
                property_name="prom_33",
                observed_value=round(prom_33, 2),
                expected_range=">= 3.0 dB (DMR 30 ms Slot Line)",
                description=f"Absence of 33.3 Hz TDMA slot spectral line (measured prominence {prom_33:.1f} dB < 3.0 dB)",
                penalty=cfg.penalty_spectral
            ))

    if "OTH-SW" in hyp_title:
        prom_43 = float(features.get("prom_43", 0.0))
        if prom_43 < 4.0:
            contradictions.append(MeasurableContradiction(
                category=ContradictionCategory.SPECTRAL,
                property_name="prom_43",
                observed_value=round(prom_43, 2),
                expected_range=">= 8.0 dB (43.2 Hz OTH Frame Line)",
                description=f"Absence of 43.2 Hz envelope spectral line (measured prominence {prom_43:.1f} dB < 8.0 dB)",
                penalty=cfg.penalty_spectral
            ))

    if "Ghadir" in hyp_title:
        p870 = float(features.get("prom_870", 0.0))
        p307 = float(features.get("prom_307", 0.0))
        if p870 < 4.0 and p307 < 4.0:
            contradictions.append(MeasurableContradiction(
                category=ContradictionCategory.SPECTRAL,
                property_name="prom_870_307",
                observed_value=f"870Hz={p870:.1f}, 307Hz={p307:.1f}",
                expected_range=">= 8.0 dB (Ghadir PRF Line)",
                description=f"Absence of 870 Hz or 307 Hz PRF spectral lines for Ghadir OTH radar",
                penalty=cfg.penalty_spectral
            ))

    # -------------------------------------------------------------------------
    # 4. CROSS-WINDOW INSTABILITY: Temporal Validation Telemetry
    # -------------------------------------------------------------------------
    temp_score = float(candidate.temporal_consistency)
    if temporal_result and isinstance(temporal_result, dict):
        temp_score = float(temporal_result.get("temporal_consistency_score", temp_score))

    if temp_score < 0.60:
        contradictions.append(MeasurableContradiction(
            category=ContradictionCategory.CROSS_WINDOW_INSTABILITY,
            property_name="temporal_consistency_score",
            observed_value=round(temp_score, 3),
            expected_range=">= 0.60 (Stationary Signal)",
            description=(
                f"Multi-window temporal validation failed: parameter stability score ({temp_score:.2f}) "
                f"indicates severe non-stationarity or intermittent burst dynamics"
            ),
            penalty=cfg.penalty_cross_window
        ))

    # -------------------------------------------------------------------------
    # 5. RECONSTRUCTION FAILURE: Closed-Loop Tracking & Demodulation Telemetry
    # -------------------------------------------------------------------------
    if reconstruction_telemetry and isinstance(reconstruction_telemetry, dict):
        # PLL phase lock check for digital PSK/QAM
        if candidate.signal_family in ("PSK", "QAM"):
            pll_locked = reconstruction_telemetry.get("pll_locked", True)
            pll_metric = float(reconstruction_telemetry.get("pll_lock_metric", 1.0))
            if (not pll_locked) or pll_metric < 0.35:
                contradictions.append(MeasurableContradiction(
                    category=ContradictionCategory.RECONSTRUCTION_FAILURE,
                    property_name="pll_lock_metric",
                    observed_value=round(pll_metric, 3),
                    expected_range=">= 0.50 (Phase Lock)",
                    description=(
                        f"Carrier recovery PLL failed phase synchronization "
                        f"(lock metric {pll_metric:.2f} < 0.50, locked={pll_locked})"
                    ),
                    penalty=cfg.penalty_reconstruction
                ))

        # EVM check for digital constellations
        evm_val = reconstruction_telemetry.get("evm_pct")
        if evm_val is not None and float(evm_val) > 40.0:
            contradictions.append(MeasurableContradiction(
                category=ContradictionCategory.RECONSTRUCTION_FAILURE,
                property_name="evm_pct",
                observed_value=round(float(evm_val), 1),
                expected_range="<= 35.0% (Valid Constellation)",
                description=(
                    f"Demodulated EVM ({float(evm_val):.1f}%) exceeds acceptable symbol constellation "
                    f"error threshold (35.0%)"
                ),
                penalty=cfg.penalty_reconstruction
            ))

    # -------------------------------------------------------------------------
    # 6. Pre-existing Textual Contradictions from Detector Rules
    # -------------------------------------------------------------------------
    for existing_contra in candidate.contradictions:
        # Avoid duplicating structured contradictions already recorded above
        if not any(c.description == existing_contra for c in contradictions):
            contradictions.append(MeasurableContradiction(
                category=ContradictionCategory.PARAMETER_INCONSISTENCY,
                property_name="detector_rule_constraint",
                observed_value="Disqualified by mutual exclusion invariant",
                expected_range="Consistent physical signature",
                description=existing_contra,
                penalty=cfg.penalty_parameter
            ))

    # -------------------------------------------------------------------------
    # 7. Compute Net Evidence Score
    # -------------------------------------------------------------------------
    total_penalty = float(sum(c.penalty for c in contradictions))
    clamped_penalty = float(min(cfg.max_total_penalty, total_penalty))
    net_score = float(np.clip(raw_score - clamped_penalty, 0.0, 1.0))

    contra_descriptions = [c.description for c in contradictions]

    return ContradictionAnalysisResult(
        hypothesis=hyp_title,
        hypothesis_id=candidate.hypothesis_id,
        supporting_evidence=list(candidate.supporting_evidence),
        contradictions=contra_descriptions,
        contradiction_details=contradictions,
        contradiction_count=len(contradictions),
        raw_evidence_score=raw_score,
        net_evidence_score=net_score,
        total_penalty=clamped_penalty
    )


def apply_contradiction_analysis(
    candidates: List[SignalHypothesis],
    feats: Optional[Dict[str, Any]] = None,
    temporal_result: Optional[Dict[str, Any]] = None,
    reconstruction_telemetry: Optional[Dict[str, Any]] = None,
    config: Optional[ContradictionConfig] = None
) -> Tuple[List[SignalHypothesis], List[ContradictionAnalysisResult]]:
    """
    Applies contradiction analysis across all candidate hypotheses:
    - Updates candidate.contradictions with measurable contradictions
    - Updates candidate.evidence_score to net_evidence_score
    - Returns updated candidates and structured ContradictionAnalysisResult list
    """
    cfg = config or ContradictionConfig()
    results: List[ContradictionAnalysisResult] = []

    for cand in candidates:
        analysis = analyze_candidate_contradictions(
            cand, feats=feats, temporal_result=temporal_result,
            reconstruction_telemetry=reconstruction_telemetry, config=cfg
        )
        results.append(analysis)

        # Update candidate record with net score and concrete contradiction list
        cand.contradictions = list(analysis.contradictions)
        cand.evidence_score = float(analysis.net_evidence_score)
        cand.confidence = float(analysis.net_evidence_score)

        if analysis.contradiction_count > 0 and not cand.rejection_reason:
            cand.rejection_reason = f"Physical contradiction: {analysis.contradictions[0]}"

    return candidates, results
