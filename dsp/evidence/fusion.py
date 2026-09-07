"""
Multi-Stage Gated Evidence Fusion Engine
========================================
Implements the defense-grade 5-stage gated evidence hierarchy:
Stage 1: Modulation Plausibility Gate
Stage 2: Synchronization Convergence Gate
Stage 3: Demodulation Stability Gate
Stage 4: Candidate FEC Algebraic Validity Gate
Stage 5: Framing & CRC Verification Gate

Strict Epistemic Partitioning:
------------------------------
- OBSERVED: FFT spectral peaks, 99% OBW, power envelope, higher-order cumulants
- ESTIMATED: Carrier frequency offset (CFO), symbol baud rate, SNR, PLL lock metric, timing jitter
- HYPOTHESIZED: Modulation family, candidate interleaver topologies, candidate FEC generator polynomials
- VALIDATED: Closed-loop mathematical proofs: np.all(H * c^T == 0 mod 2), RS syndromes == 0 in GF(256), CRC match
- UNKNOWN: Undetermined or ambiguous parameters explicitly preserved without forcing guesses
"""

from typing import Dict, List, Any, Optional
import numpy as np

from dsp.contracts import (
    EpistemicStatus,
    ConfidenceLevel,
    EvidenceItem,
    EvidenceReport,
    SignalHypothesis,
    SynchronizedSymbols,
    DemodulationResult,
    InterleaverHypothesis,
    FECHypothesis,
    FrameAnalysisResult
)


def _val(item: Any) -> str:
    """Safely extracts value from enum or string."""
    if hasattr(item, "value"):
        return str(item.value)
    return str(item)


def fuse_evidence(

    raw_params: Dict[str, Any],
    hypothesis: SignalHypothesis,
    sync_symbols: SynchronizedSymbols,
    demod_result: DemodulationResult,
    interleaver_candidates: List[InterleaverHypothesis],
    fec_candidates: List[FECHypothesis],
    framing_result: FrameAnalysisResult
) -> EvidenceReport:
    """
    Executes gated multi-stage evidence fusion and structures the audit report.
    """
    observed: List[EvidenceItem] = []
    estimated: List[EvidenceItem] = []
    hypothesized: List[EvidenceItem] = []
    validated: List[EvidenceItem] = []
    unknown: List[EvidenceItem] = []

    # -------------------------------------------------------------
    # TIER 1: OBSERVED (Direct Physical Measurements)
    # -------------------------------------------------------------
    fc_peak = raw_params.get("fc_peak_hz", 0.0)
    observed.append(EvidenceItem(
        tier=EpistemicStatus.OBSERVED,
        domain="Spectral",
        description="Welch PSD Spectral Peak Frequency",
        value=f"{fc_peak:+,.1f} Hz",
        confidence=1.0
    ))

    obw = raw_params.get("bw_99pct_hz", 0.0)
    observed.append(EvidenceItem(
        tier=EpistemicStatus.OBSERVED,
        domain="Spectral",
        description="99% Occupied Bandwidth (OBW)",
        value=f"{obw:,.1f} Hz",
        confidence=1.0
    ))

    env_var = raw_params.get("envelope_variance_ratio", 0.0)
    env_desc = "Constant-Envelope Waveform" if env_var < 0.18 else "Variable-Envelope Waveform (Amplitude Modulated)"
    observed.append(EvidenceItem(
        tier=EpistemicStatus.OBSERVED,
        domain="Time-Domain Envelope",
        description=env_desc,
        value=f"Variance Ratio = {env_var:.4f}",
        confidence=0.98
    ))

    # -------------------------------------------------------------
    # TIER 2: ESTIMATED (Deterministic Continuous Estimators)
    # Note: PLL convergence & timing jitter belong here, NOT under validated!
    # -------------------------------------------------------------
    snr_val = raw_params.get("snr_db", 0.0)
    estimated.append(EvidenceItem(
        tier=EpistemicStatus.ESTIMATED,
        domain="Signal Quality",
        description="Estimated Signal-to-Noise Ratio (M2M4 / In-Band)",
        value=f"{snr_val:+.1f} dB",
        confidence=0.92
    ))

    baud_val = hypothesis.symbol_rate
    if baud_val is not None and baud_val > 0:
        estimated.append(EvidenceItem(
            tier=EpistemicStatus.ESTIMATED,
            domain="Symbol Clock",
            description="Estimated Baud Symbol Rate",
            value=f"{baud_val:,.1f} Baud",
            confidence=0.88
        ))
    else:
        unknown.append(EvidenceItem(
            tier=EpistemicStatus.UNKNOWN,
            domain="Symbol Clock",
            description="Baud Symbol Rate",
            value="UNKNOWN / Not Applicable",
            confidence=0.0
        ))

    estimated.append(EvidenceItem(
        tier=EpistemicStatus.ESTIMATED,
        domain="Carrier Tracking",
        description="Carrier Frequency Offset (CFO)",
        value=f"Coarse: {sync_symbols.coarse_cfo_hz:+.1f} Hz | Fine: {sync_symbols.fine_cfo_hz:+.1f} Hz",
        confidence=0.90
    ))

    pll_status_str = "LOCKED" if sync_symbols.pll_locked else "UNLOCKED / DRIFTING"
    estimated.append(EvidenceItem(
        tier=EpistemicStatus.ESTIMATED,
        domain="Carrier Tracking",
        description="Digital Costas / Decision-Directed PLL Lock Metric",
        value=f"{pll_status_str} (Convergence Metric = {sync_symbols.pll_lock_metric:.2f}, Residual CFO = {sync_symbols.residual_cfo_hz:+.2f} Hz)",
        confidence=sync_symbols.pll_lock_metric
    ))

    estimated.append(EvidenceItem(
        tier=EpistemicStatus.ESTIMATED,
        domain="Clock Synchronization",
        description="Symbol Timing Clock Jitter Variance",
        value=f"Timing Jitter = {sync_symbols.timing_error_variance:.4f}",
        confidence=float(np.clip(1.0 - 5.0 * sync_symbols.timing_error_variance, 0.2, 0.95))
    ))

    estimated.append(EvidenceItem(
        tier=EpistemicStatus.ESTIMATED,
        domain="Demodulation Quality",
        description="Error Vector Magnitude (EVM) %",
        value=f"{demod_result.evm_pct:.2f}%",
        confidence=float(np.clip(1.0 - demod_result.evm_pct / 50.0, 0.1, 0.95))
    ))

    # -------------------------------------------------------------
    # TIER 3: HYPOTHESIZED (Model Inferences & Candidate Choices)
    # -------------------------------------------------------------
    hypothesized.append(EvidenceItem(
        tier=EpistemicStatus.HYPOTHESIZED,
        domain="Modulation AMC",
        description=f"Modulation Classification: {_val(hypothesis.modulation)}",
        value=f"Confidence: {hypothesis.confidence*100:.1f}% ({_val(hypothesis.confidence_level)})",
        confidence=hypothesis.confidence
    ))

    # Top Interleaver Candidate
    if interleaver_candidates:
        top_int = interleaver_candidates[0]
        int_desc = f"{_val(top_int.topology)} De-interleaver"
        if top_int.parameters:
            int_desc += f" ({top_int.parameters})"
        hypothesized.append(EvidenceItem(
            tier=EpistemicStatus.HYPOTHESIZED,
            domain="Interleaver Search",
            description="Candidate De-interleaver Topology",
            value=int_desc,
            confidence=top_int.confidence
        ))

    # Top FEC Candidate (unless already validated)
    top_fec = fec_candidates[0] if fec_candidates else None
    if top_fec and top_fec.status != EpistemicStatus.VALIDATED:
        hypothesized.append(EvidenceItem(
            tier=EpistemicStatus.HYPOTHESIZED,
            domain="FEC Search",
            description=f"Candidate FEC Codebook: {_val(top_fec.family)} (Rate {top_fec.code_rate})",
            value=top_fec.validation_detail,
            confidence=0.50
        ))

    # -------------------------------------------------------------
    # TIER 4: VALIDATED (Closed-Loop Mathematical Proofs)
    # -------------------------------------------------------------
    fec_validated = False
    for fec in fec_candidates:
        if fec.status == EpistemicStatus.VALIDATED and fec.syndrome_zero:
            fec_validated = True
            validated.append(EvidenceItem(
                tier=EpistemicStatus.VALIDATED,
                domain="FEC Algebraic Parity",
                description=f"Zero-Syndrome Parity Satisfaction: {_val(fec.family)} ({fec.code_rate})",
                value=f"{fec.validation_detail} | Syndrome Parity Condition = 0",
                confidence=0.99
            ))
            break


    crc_validated = False
    if framing_result.crc_match:
        crc_validated = True
        validated.append(EvidenceItem(
            tier=EpistemicStatus.VALIDATED,
            domain="Data Integrity",
            description=f"Closed-Loop CRC Exact Match ({framing_result.crc_profile})",
            value=f"Calculated: 0x{framing_result.crc_calculated:04X} == Received: 0x{framing_result.crc_received:04X}",
            confidence=1.00
        ))

    if framing_result.repeated_frames_found >= 2:
        validated.append(EvidenceItem(
            tier=EpistemicStatus.VALIDATED,
            domain="Framing Structure",
            description="Periodic Frame Consistency Confirmed",
            value=f"{framing_result.repeated_frames_found} Consecutive Synchronized Frames (Length = {framing_result.frame_length} bits)",
            confidence=0.98
        ))

    # -------------------------------------------------------------
    # TIER 5: UNKNOWN / NOT DETERMINED (Scientific Honesty)
    # -------------------------------------------------------------
    if not fec_validated:
        unknown.append(EvidenceItem(
            tier=EpistemicStatus.UNKNOWN,
            domain="FEC Validation",
            description="Closed-Loop FEC Code",
            value="NOT VALIDATED (No zero-syndrome candidate confirmed)",
            confidence=0.0
        ))

    if not crc_validated:
        unknown.append(EvidenceItem(
            tier=EpistemicStatus.UNKNOWN,
            domain="CRC Integrity",
            description="CRC Verification",
            value="NOT DETERMINED (No candidate CRC matched or unpacketized stream)",
            confidence=0.0
        ))

    # -------------------------------------------------------------
    # Gated Multi-Stage Evidence Score Formulation
    # -------------------------------------------------------------
    # Gate 1: Physical / Numerical Observability & Modulation Plausibility (0.0 - 0.25)
    # Checks: finite samples, valid PSD, nonzero energy, reasonable bandwidth
    # Crucial Invariant: Low/negative SNR (SNR <= 0 dB) is NOT a hard rejection gate.
    is_physically_observable = bool(
        np.isfinite(fc_peak) and
        np.isfinite(obw) and
        obw > 0.0 and
        np.isfinite(env_var)
    )
    if is_physically_observable:
        g1_score = hypothesis.confidence * 0.25
    else:
        g1_score = 0.05 * hypothesis.confidence

    # Gate 2: Synchronization Convergence (0.0 - 0.20)
    g2_score = (sync_symbols.pll_lock_metric if sync_symbols.pll_locked else 0.4 * sync_symbols.pll_lock_metric) * 0.20

    # Gate 3: Demodulation Stability (0.0 - 0.20)
    demod_stability = float(np.clip(1.0 - (demod_result.evm_pct / 60.0), 0.0, 1.0))
    g3_score = demod_stability * 0.20

    # Gate 4: FEC Algebraic Verification (0.0 - 0.20)
    g4_score = 0.20 if fec_validated else 0.05

    # Gate 5: Framing & CRC Verification (0.0 - 0.15)
    g5_score = 0.15 if crc_validated else (0.08 if framing_result.frame_structure_detected else 0.0)

    total_evidence_score = float(np.clip(g1_score + g2_score + g3_score + g4_score + g5_score, 0.0, 1.0))

    mod_val = _val(hypothesis.modulation)
    if crc_validated and fec_validated:
        overall_conf = ConfidenceLevel.HIGH
        top_fam = _val(top_fec.family) if top_fec else ""
        verdict = f"FULLY VALIDATED INTELLIGENCE ({mod_val} + {top_fam} + CRC PASS)"
    elif crc_validated or fec_validated:
        overall_conf = ConfidenceLevel.HIGH
        verdict = f"ALGEBRAICALLY CONFIRMED ({mod_val} - Structural Parity Pass)"
    elif sync_symbols.pll_locked and demod_result.evm_pct < 25.0:
        overall_conf = ConfidenceLevel.MEDIUM
        verdict = f"SYNCHRONIZED DEMODULATION ({mod_val} - Stable Eye / Constellation)"
    elif hypothesis.confidence >= 0.70:
        overall_conf = ConfidenceLevel.LOW
        verdict = f"HYPOTHESIZED EMISSION ({mod_val} - Pending Independent FEC Parity)"
    else:
        overall_conf = ConfidenceLevel.UNKNOWN
        verdict = "ANOMALOUS / UNKNOWN EMISSION"

    # Assemble candidate rankings table
    candidate_ranking: List[Dict[str, Any]] = []
    for f in fec_candidates:
        candidate_ranking.append({
            "Candidate FEC": f"{_val(f.family)} ({f.code_rate})",
            "Status": _val(f.status),
            "Syndrome Valid": getattr(f, "syndrome_zero", getattr(f, "syndrome_pass", False)),
            "Details": getattr(f, "validation_detail", "")
        })


    return EvidenceReport(
        observed=observed,
        estimated=estimated,
        hypothesized=hypothesized,
        validated=validated,
        unknown=unknown,
        candidate_ranking=candidate_ranking,
        overall_verdict=verdict,
        overall_confidence=overall_conf,
        numeric_score=float(np.round(total_evidence_score, 3))
    )
