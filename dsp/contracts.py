"""
Global Type Contracts and Data Schemas for NTRO SIH26147 Signal Intelligence Engine
==================================================================================
Defines the strict epistemic evidence hierarchy and typed communication contracts
across all processing modules:
1. Epistemic Confidence Tiers (OBSERVED, ESTIMATED, HYPOTHESIZED, VALIDATED, UNKNOWN)
2. Ingestion & Sampling-Rate contracts (SignalRecord)
3. 20-Dimensional Physical Feature Contract (SignalFeatures)
4. Modulation Hypothesis (SignalHypothesis)
5. Physical Synchronization Telemetry (SynchronizedSymbols)
6. Demodulation Results with Soft LLRs (DemodulationResult)
7. De-interleaving and FEC Hypotheses (InterleaverHypothesis, FECHypothesis)
8. Framing and CRC Validation (FrameAnalysisResult, CRCProfile)
9. Multi-Stage Gated Evidence Fusion (EvidenceReport)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any, Union
import uuid
import numpy as np


class EpistemicStatus(str, Enum):
    """
    Epistemic confidence tiers governing every parameter and assertion.
    Guarantees scientific honesty: convergence != validation.
    """
    OBSERVED = "OBSERVED"             # Direct physical measurements (FFT peaks, OBW, envelope)
    ESTIMATED = "ESTIMATED"           # Deterministic continuous estimators (CFO, Baud, SNR, PLL metric)
    HYPOTHESIZED = "HYPOTHESIZED"     # Model / candidate inferences (Modulation, Code candidates)
    VALIDATED = "VALIDATED"           # Closed-loop mathematical proof (Syndrome == 0, CRC pass)
    UNKNOWN = "UNKNOWN"               # Undetermined parameter; prevents false forced guesses
    NOT_APPLICABLE = "NOT_APPLICABLE" # Not applicable to this waveform class


class SamplingRateStatus(str, Enum):
    """Authority status for the physical timebase."""
    VERIFIED_METADATA = "VERIFIED_METADATA"   # Read from RIFF WAV / SigMF header
    INFERRED_PROFILE = "INFERRED_PROFILE"     # Inferred from hardware profile or recognized standard
    NORMALIZED_DOMAIN = "NORMALIZED_DOMAIN"   # Unknown physical clock; normalized [-0.5, 0.5]


class ModulationFamily(str, Enum):
    """Supported modulation classifications."""
    FSK_2 = "2-FSK"
    FSK_4 = "4-FSK"
    BPSK = "BPSK"
    QPSK = "QPSK"
    PSK_8 = "8-PSK"
    QAM_16 = "16-QAM"
    QAM_64 = "64-QAM"
    ANALOG_AM = "AM"
    ANALOG_FM = "FM"
    UNKNOWN_OOD = "UNKNOWN_OOD"


class InterleaverType(str, Enum):
    """Interleaver topology classifications."""
    BLOCK = "BLOCK"
    CONVOLUTIONAL = "CONVOLUTIONAL"
    DIAGONAL = "DIAGONAL"
    PSEUDO_RANDOM = "PSEUDO_RANDOM"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class FECCodeFamily(str, Enum):
    """Forward Error Correction code families."""
    CONVOLUTIONAL = "CONVOLUTIONAL"
    REED_SOLOMON = "REED_SOLOMON"
    CONCATENATED = "CONCATENATED"
    LDPC = "LDPC"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class ConfidenceLevel(str, Enum):
    """Discrete confidence categories."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


@dataclass
class SignalRecord:
    """
    Universal input contract for intercepted waveforms.
    When physical sample rate is unavailable, sample_rate is None (never fake 1.0 Hz).
    """
    samples: np.ndarray
    sample_rate: Optional[float]
    rate_status: SamplingRateStatus = SamplingRateStatus.NORMALIZED_DOMAIN
    estimation_confidence: float = 1.0
    estimation_source: str = "Direct Header / Ingestion"
    center_frequency: float = 0.0
    complex_or_real: str = "complex"
    datatype: str = "complex64"
    source: str = "UNKNOWN"
    source_format: str = "RAW"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        samples: np.ndarray,
        sample_rate: Optional[float] = None,
        rate_status: Optional[Union[SamplingRateStatus, str]] = None,
        estimation_confidence: float = 1.0,
        estimation_source: str = "Direct Header / Ingestion",
        center_frequency: float = 0.0,
        complex_or_real: str = "complex",
        datatype: str = "complex64",
        source: str = "UNKNOWN",
        source_format: str = "RAW",
        is_normalized: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        sample_rate_status: Optional[Union[SamplingRateStatus, str]] = None,
        confidence: Optional[float] = None,
    ):
        self.samples = samples
        self.sample_rate = sample_rate

        status = sample_rate_status if sample_rate_status is not None else rate_status
        if status is None:
            status = SamplingRateStatus.NORMALIZED_DOMAIN
        elif isinstance(status, str):
            if status in SamplingRateStatus.__members__:
                status = SamplingRateStatus[status]
            else:
                try:
                    status = SamplingRateStatus(status)
                except ValueError:
                    status = SamplingRateStatus.NORMALIZED_DOMAIN
        self.rate_status = status

        self.estimation_confidence = float(confidence) if confidence is not None else float(estimation_confidence)
        self.estimation_source = str(estimation_source)
        self.center_frequency = float(center_frequency)
        self.complex_or_real = str(complex_or_real)
        self.datatype = str(datatype)
        self.source = str(source)
        self.source_format = str(source_format)
        self.is_normalized = bool(is_normalized)
        self.metadata = metadata if metadata is not None else {}

    @property
    def sample_rate_status(self) -> SamplingRateStatus:
        return self.rate_status

    @sample_rate_status.setter
    def sample_rate_status(self, val: SamplingRateStatus):
        self.rate_status = val

    @property
    def confidence(self) -> float:
        return self.estimation_confidence

    @confidence.setter
    def confidence(self, val: float):
        self.estimation_confidence = val

    @property
    def effective_fs(self) -> float:
        """Return 1.0 for normalized computation if physical rate is unavailable."""
        return self.sample_rate if self.sample_rate is not None else 1.0

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.sample_rate and self.sample_rate > 0:
            return len(self.samples) / self.sample_rate
        return None



@dataclass
class SignalFeatures:
    """
    Concrete 20-Dimensional Physical Feature Schema.
    Strictly defined scalar elements ensuring a deterministic ML/DSP feature contract.
    No feature is mathematically constant after unit-power normalization.
    """
    c20_mag: float
    c20_angle: float
    c40_mag: float
    c40_angle: float
    c42_real: float
    c63_norm: float
    spectral_centroid: float
    spectral_spread: float
    spectral_flatness: float
    spectral_rolloff: float
    obw_99: float
    papr_db: float
    envelope_variance: float
    amplitude_skewness: float
    amplitude_kurtosis: float
    instantaneous_freq_variance: float
    instantaneous_phase_variance: float
    cyclic_peak_prominence: float
    zero_crossing_rate: float
    snr_m2m4_db: float
    c21: float = 1.0  # Preserved for backward compatibility

    def to_vector(self) -> np.ndarray:
        """Serializes features into a 20-element float64 1D array."""
        return np.array([
            self.c20_mag,
            self.c20_angle,
            self.c40_mag,
            self.c40_angle,
            self.c42_real,
            self.c63_norm,
            self.spectral_centroid,
            self.spectral_spread,
            self.spectral_flatness,
            self.spectral_rolloff,
            self.obw_99,
            self.papr_db,
            self.envelope_variance,
            self.amplitude_skewness,
            self.amplitude_kurtosis,
            self.instantaneous_freq_variance,
            self.instantaneous_phase_variance,
            self.cyclic_peak_prominence,
            self.zero_crossing_rate,
            self.snr_m2m4_db
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> List[str]:
        """Returns ordered list of the 20 feature names."""
        return [
            "c20_mag", "c20_angle", "c40_mag", "c40_angle",
            "c42_real", "c63_norm", "spectral_centroid", "spectral_spread",
            "spectral_flatness", "spectral_rolloff", "obw_99", "papr_db",
            "envelope_variance", "amplitude_skewness", "amplitude_kurtosis",
            "instantaneous_freq_variance", "instantaneous_phase_variance",
            "cyclic_peak_prominence", "zero_crossing_rate", "snr_m2m4_db"
        ]


@dataclass
class SignalHypothesis:
    """
    Hypothesis governing downstream synchronization, demodulation, and evidence fusion.
    Implements the Round-2 Epistemic Hypothesis Model with 18 core fields:
    - hypothesis_id: Unique string identifier for the hypothesis instance
    - signal_family: General signal family (e.g. "PSK", "FSK", "QAM", "ANALOG", "RADAR_PULSE", "UNKNOWN_OOD")
    - modulation: Inferred modulation classification (ModulationFamily enum or string)
    - protocol: Candidate transmission protocol profile if identified (or None)
    - parameters: Map of physical parameter estimates and measurements
    - parameter_status: Epistemic status (OBSERVED/ESTIMATED/HYPOTHESIZED/VALIDATED/UNKNOWN) per parameter
    - parameter_uncertainty: Quantitative uncertainty / error margin / variance per parameter
    - supporting_evidence: Concrete physical observations and test results supporting this hypothesis
    - contradictions: Explicit contradictory evidence or failed constraint checks
    - temporal_consistency: Multi-window temporal stability score in [0.0, 1.0]
    - physical_consistency: Physical plausibility score in [0.0, 1.0] (Nyquist, bandwidth, power)
    - reconstruction_consistency: Closed-loop reconstruction score in [0.0, 1.0] (EVM, PLL lock, LLR)
    - cross_window_consistency: Parameter agreement across temporal sub-windows in [0.0, 1.0]
    - evidence_score: Composite epistemic evidence score in [0.0, 1.0]. Strictly an epistemic metric,
      NOT a Bayesian probability.
    - confidence_level: Discrete confidence category (HIGH, MEDIUM, LOW, UNKNOWN)
    - epistemic_status: Epistemic hierarchy tier (OBSERVED, ESTIMATED, HYPOTHESIZED, VALIDATED, UNKNOWN)
    - validation_status: Lifecycle status ("PENDING", "VALIDATED", "REJECTED", "UNVALIDATED")
    - rejection_reason: Explanation if hypothesis was rejected or invalidated

    Epistemic Contract & Backward Compatibility:
    - evidence_score represents an uncalibrated composite epistemic evidence metric in [0.0, 1.0].
      It is strictly NOT a Bayesian probability.
    - Preserves full backward compatibility with legacy fields: symbol_rate, carrier_offset,
      samples_per_symbol, pulse_model, confidence, is_ood, mahalanobis_distance, evidence,
      rejected_hypotheses.
    - Supports both legacy positional constructors and Round-2 constructors.
    """
    # Round-2 Architectural Fields
    hypothesis_id: str = ""
    signal_family: str = "UNKNOWN"
    modulation: ModulationFamily = ModulationFamily.UNKNOWN_OOD
    protocol: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    parameter_status: Dict[str, EpistemicStatus] = field(default_factory=dict)
    parameter_uncertainty: Dict[str, float] = field(default_factory=dict)
    parameter_reports: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    supporting_evidence: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    temporal_consistency: float = 1.0
    physical_consistency: float = 1.0
    reconstruction_consistency: float = 1.0
    cross_window_consistency: float = 1.0
    evidence_score: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    epistemic_status: EpistemicStatus = EpistemicStatus.HYPOTHESIZED
    validation_status: str = "PENDING"
    rejection_reason: Optional[str] = None

    # Legacy Telemetry Fields
    symbol_rate: Optional[float] = None
    carrier_offset: float = 0.0
    samples_per_symbol: float = 2.0
    pulse_model: Optional[str] = None
    is_ood: bool = False
    mahalanobis_distance: float = 0.0
    rejected_hypotheses: List[str] = field(default_factory=list)

    def __init__(
        self,
        *args,
        hypothesis_id: Optional[str] = None,
        signal_family: Optional[str] = None,
        modulation: Optional[Union[ModulationFamily, str]] = None,
        protocol: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        parameter_status: Optional[Dict[str, Union[EpistemicStatus, str]]] = None,
        parameter_uncertainty: Optional[Dict[str, float]] = None,
        parameter_reports: Optional[Dict[str, Dict[str, Any]]] = None,
        supporting_evidence: Optional[List[str]] = None,
        contradictions: Optional[List[str]] = None,
        temporal_consistency: Optional[float] = None,
        physical_consistency: Optional[float] = None,
        reconstruction_consistency: Optional[float] = None,
        cross_window_consistency: Optional[float] = None,
        evidence_score: Optional[float] = None,
        confidence_level: Optional[Union[ConfidenceLevel, str]] = None,
        epistemic_status: Optional[Union[EpistemicStatus, str]] = None,
        validation_status: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        # Legacy positional/keyword aliases
        symbol_rate: Optional[float] = None,
        carrier_offset: Optional[float] = None,
        samples_per_symbol: Optional[float] = None,
        pulse_model: Optional[str] = None,
        confidence: Optional[float] = None,
        is_ood: Optional[bool] = None,
        mahalanobis_distance: Optional[float] = None,
        evidence: Optional[List[str]] = None,
        rejected_hypotheses: Optional[List[str]] = None,
        **kwargs
    ):
        # 1. Handle positional arguments (*args)
        if len(args) > 0:
            first_arg = args[0]
            is_legacy = (
                isinstance(first_arg, ModulationFamily)
                or (isinstance(first_arg, str) and (
                    first_arg in [m.value for m in ModulationFamily]
                    or first_arg in [m.name for m in ModulationFamily]
                ))
            )
            if is_legacy:
                # Legacy positional signature:
                # (modulation, symbol_rate, carrier_offset, samples_per_symbol, pulse_model,
                #  confidence, confidence_level, is_ood, mahalanobis_distance, evidence, rejected_hypotheses)
                modulation = first_arg
                if len(args) > 1 and symbol_rate is None:
                    symbol_rate = args[1]
                if len(args) > 2 and carrier_offset is None:
                    carrier_offset = args[2]
                if len(args) > 3 and samples_per_symbol is None:
                    samples_per_symbol = args[3]
                if len(args) > 4 and pulse_model is None:
                    pulse_model = args[4]
                if len(args) > 5 and confidence is None:
                    confidence = args[5]
                if len(args) > 6 and confidence_level is None:
                    confidence_level = args[6]
                if len(args) > 7 and is_ood is None:
                    is_ood = args[7]
                if len(args) > 8 and mahalanobis_distance is None:
                    mahalanobis_distance = args[8]
                if len(args) > 9 and evidence is None:
                    evidence = args[9]
                if len(args) > 10 and rejected_hypotheses is None:
                    rejected_hypotheses = args[10]
            else:
                # Round-2 positional signature (ordered as specified in requirement 1):
                # (hypothesis_id, signal_family, modulation, protocol, parameters,
                #  parameter_status, parameter_uncertainty, supporting_evidence, contradictions,
                #  temporal_consistency, physical_consistency, reconstruction_consistency,
                #  cross_window_consistency, evidence_score, confidence_level, epistemic_status,
                #  validation_status, rejection_reason)
                if len(args) > 0 and hypothesis_id is None:
                    hypothesis_id = args[0]
                if len(args) > 1 and signal_family is None:
                    signal_family = args[1]
                if len(args) > 2 and modulation is None:
                    modulation = args[2]
                if len(args) > 3 and protocol is None:
                    protocol = args[3]
                if len(args) > 4 and parameters is None:
                    parameters = args[4]
                if len(args) > 5 and parameter_status is None:
                    parameter_status = args[5]
                if len(args) > 6 and parameter_uncertainty is None:
                    parameter_uncertainty = args[6]
                if len(args) > 7 and supporting_evidence is None:
                    supporting_evidence = args[7]
                if len(args) > 8 and contradictions is None:
                    contradictions = args[8]
                if len(args) > 9 and temporal_consistency is None:
                    temporal_consistency = args[9]
                if len(args) > 10 and physical_consistency is None:
                    physical_consistency = args[10]
                if len(args) > 11 and reconstruction_consistency is None:
                    reconstruction_consistency = args[11]
                if len(args) > 12 and cross_window_consistency is None:
                    cross_window_consistency = args[12]
                if len(args) > 13 and evidence_score is None:
                    evidence_score = args[13]
                if len(args) > 14 and confidence_level is None:
                    confidence_level = args[14]
                if len(args) > 15 and epistemic_status is None:
                    epistemic_status = args[15]
                if len(args) > 16 and validation_status is None:
                    validation_status = args[16]
                if len(args) > 17 and rejection_reason is None:
                    rejection_reason = args[17]

        # 2. Normalize modulation
        if isinstance(modulation, str):
            matched = False
            for fam in ModulationFamily:
                if fam.value == modulation or fam.name == modulation:
                    modulation = fam
                    matched = True
                    break
            if not matched:
                if modulation in ("UNKNOWN", "OOD", "UNKNOWN_OOD"):
                    modulation = ModulationFamily.UNKNOWN_OOD
        elif modulation is None:
            modulation = ModulationFamily.UNKNOWN_OOD

        # 3. Handle OOD status
        if is_ood is None:
            is_ood = bool(
                modulation == ModulationFamily.UNKNOWN_OOD
                and signal_family in ("UNKNOWN_OOD", "OOD")
            )
        else:
            is_ood = bool(is_ood)
        if is_ood:
            modulation = ModulationFamily.UNKNOWN_OOD

        # 4. Reconcile evidence_score and confidence (uncalibrated epistemic metric, never probability)
        if evidence_score is not None and confidence is None:
            resolved_score = float(evidence_score)
        elif confidence is not None and evidence_score is None:
            resolved_score = float(confidence)
        elif evidence_score is not None and confidence is not None:
            resolved_score = float(evidence_score)
        else:
            resolved_score = 0.0
        self.evidence_score = float(np.clip(resolved_score, 0.0, 1.0))

        # 5. Normalize confidence_level
        if isinstance(confidence_level, str):
            try:
                confidence_level = ConfidenceLevel[confidence_level]
            except KeyError:
                try:
                    confidence_level = ConfidenceLevel(confidence_level)
                except ValueError:
                    confidence_level = ConfidenceLevel.UNKNOWN
        elif confidence_level is None:
            if self.evidence_score >= 0.85:
                confidence_level = ConfidenceLevel.HIGH
            elif self.evidence_score >= 0.65:
                confidence_level = ConfidenceLevel.MEDIUM
            elif self.evidence_score > 0.0:
                confidence_level = ConfidenceLevel.LOW
            else:
                confidence_level = ConfidenceLevel.UNKNOWN
        self.confidence_level = confidence_level

        # 6. Normalize epistemic_status
        if isinstance(epistemic_status, str):
            try:
                epistemic_status = EpistemicStatus[epistemic_status]
            except KeyError:
                try:
                    epistemic_status = EpistemicStatus(epistemic_status)
                except ValueError:
                    epistemic_status = EpistemicStatus.UNKNOWN
        elif epistemic_status is None:
            if is_ood or modulation == ModulationFamily.UNKNOWN_OOD:
                epistemic_status = EpistemicStatus.UNKNOWN
            elif validation_status == "VALIDATED":
                epistemic_status = EpistemicStatus.VALIDATED
            else:
                epistemic_status = EpistemicStatus.HYPOTHESIZED
        self.epistemic_status = epistemic_status

        # 7. Normalize validation_status
        if validation_status is None:
            if self.epistemic_status == EpistemicStatus.VALIDATED:
                validation_status = "VALIDATED"
            elif is_ood or modulation == ModulationFamily.UNKNOWN_OOD:
                validation_status = "UNVALIDATED"
            else:
                validation_status = "PENDING"
        self.validation_status = str(validation_status)

        # 8. Infer signal_family if not provided
        if not signal_family:
            if is_ood or modulation == ModulationFamily.UNKNOWN_OOD:
                signal_family = "UNKNOWN_OOD"
            elif modulation in (ModulationFamily.FSK_2, ModulationFamily.FSK_4):
                signal_family = "FSK"
            elif modulation in (ModulationFamily.BPSK, ModulationFamily.QPSK, ModulationFamily.PSK_8):
                signal_family = "PSK"
            elif modulation in (ModulationFamily.QAM_16, ModulationFamily.QAM_64):
                signal_family = "QAM"
            elif modulation in (ModulationFamily.ANALOG_AM, ModulationFamily.ANALOG_FM):
                signal_family = "ANALOG"
            else:
                signal_family = "UNKNOWN"
        self.signal_family = str(signal_family)

        # 9. Auto-generate hypothesis_id if not given
        if not hypothesis_id:
            hypothesis_id = f"HYP-{uuid.uuid4().hex[:8].upper()}"
        self.hypothesis_id = str(hypothesis_id)

        # 10. Harmonize supporting_evidence and evidence
        if supporting_evidence is not None and evidence is None:
            self.supporting_evidence = list(supporting_evidence)
        elif evidence is not None and supporting_evidence is None:
            self.supporting_evidence = list(evidence)
        elif supporting_evidence is not None and evidence is not None:
            self.supporting_evidence = list(supporting_evidence)
        else:
            self.supporting_evidence = []

        # 11. Contradictions & Rejected Hypotheses
        self.contradictions = list(contradictions) if contradictions is not None else []
        self.rejected_hypotheses = list(rejected_hypotheses) if rejected_hypotheses is not None else []

        # 12. Harmonize parameters dict
        self.parameters = dict(parameters) if parameters is not None else {}
        if symbol_rate is None and "symbol_rate" in self.parameters:
            symbol_rate = self.parameters["symbol_rate"]
        if carrier_offset is None and "carrier_offset" in self.parameters:
            carrier_offset = self.parameters["carrier_offset"]
        if samples_per_symbol is None and "samples_per_symbol" in self.parameters:
            samples_per_symbol = self.parameters["samples_per_symbol"]
        if pulse_model is None and "pulse_model" in self.parameters:
            pulse_model = self.parameters["pulse_model"]

        if "symbol_rate" not in self.parameters:
            self.parameters["symbol_rate"] = symbol_rate
        if "carrier_offset" not in self.parameters:
            self.parameters["carrier_offset"] = carrier_offset if carrier_offset is not None else 0.0
        if "samples_per_symbol" not in self.parameters:
            self.parameters["samples_per_symbol"] = samples_per_symbol if samples_per_symbol is not None else 2.0
        if "pulse_model" not in self.parameters:
            self.parameters["pulse_model"] = pulse_model

        # 13. Harmonize parameter_status
        self.parameter_status = {}
        raw_p_status = parameter_status if parameter_status is not None else {}
        for k, v in raw_p_status.items():
            if isinstance(v, str):
                try:
                    self.parameter_status[k] = EpistemicStatus(v)
                except ValueError:
                    try:
                        self.parameter_status[k] = EpistemicStatus[v]
                    except KeyError:
                        self.parameter_status[k] = EpistemicStatus.UNKNOWN
            elif isinstance(v, EpistemicStatus):
                self.parameter_status[k] = v
            else:
                self.parameter_status[k] = EpistemicStatus.UNKNOWN

        for p_key, p_val in self.parameters.items():
            if p_key not in self.parameter_status:
                if p_val is None:
                    self.parameter_status[p_key] = EpistemicStatus.UNKNOWN
                elif p_key in ("carrier_offset", "fc_peak", "snr", "obw_99"):
                    self.parameter_status[p_key] = EpistemicStatus.OBSERVED
                elif p_key in ("symbol_rate", "samples_per_symbol", "freq_shift"):
                    self.parameter_status[p_key] = EpistemicStatus.ESTIMATED
                elif p_key in ("pulse_model", "protocol", "modulation"):
                    self.parameter_status[p_key] = EpistemicStatus.HYPOTHESIZED
                else:
                    self.parameter_status[p_key] = EpistemicStatus.ESTIMATED

        # 14. Harmonize parameter_uncertainty
        self.parameter_uncertainty = {
            k: float(v) for k, v in (parameter_uncertainty or {}).items()
        }
        for p_key in self.parameters:
            if p_key not in self.parameter_uncertainty:
                self.parameter_uncertainty[p_key] = 0.0

        self.parameter_reports = dict(parameter_reports) if parameter_reports is not None else {}

        # 15. Set consistency metrics & telemetry
        self.temporal_consistency = float(temporal_consistency) if temporal_consistency is not None else 1.0
        self.physical_consistency = float(physical_consistency) if physical_consistency is not None else 1.0
        self.reconstruction_consistency = float(reconstruction_consistency) if reconstruction_consistency is not None else 1.0
        self.cross_window_consistency = float(cross_window_consistency) if cross_window_consistency is not None else 1.0
        self.protocol = protocol
        self.rejection_reason = rejection_reason

        self.modulation = modulation
        self.symbol_rate = symbol_rate
        self.carrier_offset = float(carrier_offset) if carrier_offset is not None else 0.0
        self.samples_per_symbol = float(samples_per_symbol) if samples_per_symbol is not None else 2.0
        self.pulse_model = pulse_model
        self.is_ood = bool(is_ood)
        self.mahalanobis_distance = float(mahalanobis_distance) if mahalanobis_distance is not None else 0.0

    @property
    def confidence(self) -> float:
        """Legacy alias for evidence_score."""
        return self.evidence_score

    @confidence.setter
    def confidence(self, val: float):
        self.evidence_score = float(val)

    @property
    def evidence(self) -> List[str]:
        """Legacy alias for supporting_evidence."""
        return self.supporting_evidence

    @evidence.setter
    def evidence(self, val: List[str]):
        self.supporting_evidence = list(val) if val is not None else []

    def mark_validated(self, proof_evidence: str = "") -> None:
        """Promotes hypothesis to mathematically VALIDATED status (e.g., CRC/Syndrome pass)."""
        self.epistemic_status = EpistemicStatus.VALIDATED
        self.validation_status = "VALIDATED"
        self.reconstruction_consistency = 1.0
        if proof_evidence:
            self.supporting_evidence.append(proof_evidence)

    def mark_rejected(self, reason: str) -> None:
        """Marks hypothesis as REJECTED with an explicit contradiction reason."""
        self.validation_status = "REJECTED"
        self.rejection_reason = reason
        self.contradictions.append(reason)

    def mark_ood(self, reason: str = "Signal out-of-distribution") -> None:
        """Sets hypothesis to first-class UNKNOWN_OOD state."""
        self.is_ood = True
        self.modulation = ModulationFamily.UNKNOWN_OOD
        self.signal_family = "UNKNOWN_OOD"
        self.epistemic_status = EpistemicStatus.UNKNOWN
        self.validation_status = "UNVALIDATED"
        self.rejection_reason = reason
        self.contradictions.append(reason)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes SignalHypothesis to a complete dictionary.
        Strictly preserves epistemic honesty: evidence_score is NOT a probability.
        """
        return {
            "hypothesis_id": self.hypothesis_id,
            "signal_family": self.signal_family,
            "modulation": self.modulation.value if isinstance(self.modulation, Enum) else str(self.modulation),
            "protocol": self.protocol,
            "parameters": dict(self.parameters),
            "parameter_status": {
                k: (v.value if isinstance(v, Enum) else str(v))
                for k, v in self.parameter_status.items()
            },
            "parameter_uncertainty": {k: float(v) for k, v in self.parameter_uncertainty.items()},
            "parameter_reports": dict(self.parameter_reports),
            "parameter_uncertainties": dict(self.parameter_reports),
            "supporting_evidence": list(self.supporting_evidence),
            "contradictions": list(self.contradictions),
            "temporal_consistency": float(self.temporal_consistency),
            "physical_consistency": float(self.physical_consistency),
            "reconstruction_consistency": float(self.reconstruction_consistency),
            "cross_window_consistency": float(self.cross_window_consistency),
            "evidence_score": float(self.evidence_score),
            "confidence_level": self.confidence_level.value if isinstance(self.confidence_level, Enum) else str(self.confidence_level),
            "epistemic_status": self.epistemic_status.value if isinstance(self.epistemic_status, Enum) else str(self.epistemic_status),
            "validation_status": str(self.validation_status),
            "rejection_reason": self.rejection_reason,
            # Legacy compatibility fields
            "symbol_rate": self.symbol_rate,
            "carrier_offset": float(self.carrier_offset),
            "samples_per_symbol": float(self.samples_per_symbol),
            "pulse_model": self.pulse_model,
            "confidence": float(self.confidence),
            "is_ood": bool(self.is_ood),
            "mahalanobis_distance": float(self.mahalanobis_distance),
            "evidence": list(self.evidence),
            "rejected_hypotheses": list(self.rejected_hypotheses),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SignalHypothesis":
        """
        Deserializes SignalHypothesis from a dictionary, restoring enums
        and epistemic statuses.
        """
        data = dict(d)

        # Modulation Enum restoration
        mod_val = data.get("modulation", ModulationFamily.UNKNOWN_OOD)
        if isinstance(mod_val, str):
            mod_enum = None
            for m in ModulationFamily:
                if m.value == mod_val or m.name == mod_val:
                    mod_enum = m
                    break
            if mod_enum is None:
                mod_enum = ModulationFamily.UNKNOWN_OOD
            data["modulation"] = mod_enum

        # ConfidenceLevel Enum restoration
        conf_level = data.get("confidence_level", ConfidenceLevel.MEDIUM)
        if isinstance(conf_level, str):
            try:
                data["confidence_level"] = ConfidenceLevel(conf_level)
            except ValueError:
                try:
                    data["confidence_level"] = ConfidenceLevel[conf_level]
                except KeyError:
                    data["confidence_level"] = ConfidenceLevel.UNKNOWN

        # EpistemicStatus Enum restoration
        ep_status = data.get("epistemic_status", EpistemicStatus.HYPOTHESIZED)
        if isinstance(ep_status, str):
            try:
                data["epistemic_status"] = EpistemicStatus(ep_status)
            except ValueError:
                try:
                    data["epistemic_status"] = EpistemicStatus[ep_status]
                except KeyError:
                    data["epistemic_status"] = EpistemicStatus.UNKNOWN

        # Parameter Status map restoration
        raw_p_status = data.get("parameter_status", {})
        parsed_p_status = {}
        for k, v in raw_p_status.items():
            if isinstance(v, str):
                try:
                    parsed_p_status[k] = EpistemicStatus(v)
                except ValueError:
                    try:
                        parsed_p_status[k] = EpistemicStatus[v]
                    except KeyError:
                        parsed_p_status[k] = EpistemicStatus.UNKNOWN
            elif isinstance(v, EpistemicStatus):
                parsed_p_status[k] = v
            else:
                parsed_p_status[k] = EpistemicStatus.UNKNOWN
        data["parameter_status"] = parsed_p_status

        # Parameter Reports restoration
        raw_reports = data.get("parameter_reports") or data.get("parameter_uncertainties")
        if raw_reports:
            data["parameter_reports"] = dict(raw_reports)

        return cls(**data)



@dataclass
class SynchronizedSymbols:
    """Synchronized symbols and closed-loop tracking telemetry."""
    symbols: np.ndarray
    soft_llrs: np.ndarray
    coarse_cfo_hz: float
    fine_cfo_hz: float
    residual_cfo_hz: float
    phase_ambiguity_rad: float
    pll_lock_metric: float
    pll_locked: bool
    timing_error_variance: float
    constellation_points: np.ndarray
    eye_samples: np.ndarray
    samples_per_symbol: float
    phase_offset_rad: float = 0.0
    cycle_slips: int = 0
    observability_status: str = "OBSERVABLE"

    @property
    def coarse_cfo(self) -> float:
        return self.coarse_cfo_hz

    @property
    def fine_cfo(self) -> float:
        return self.fine_cfo_hz

    @property
    def residual_cfo(self) -> float:
        return self.residual_cfo_hz

    @property
    def phase_offset(self) -> float:
        return self.phase_offset_rad

    @property
    def phase_lock_status(self) -> bool:
        return self.pll_locked

    @property
    def cycle_slip_detected(self) -> bool:
        return self.cycle_slips > 0


@dataclass
class DemodulationResult:
    """Outputs of the digital demodulator."""
    modulation: ModulationFamily
    hard_bits: np.ndarray
    soft_llrs: np.ndarray
    symbol_error_rate_est: float
    evm_pct: float
    noise_var_est: float


@dataclass
class InterleaverHypothesis:
    """Candidate de-interleaver evaluation record."""
    topology: InterleaverType
    depth: int
    span: int
    parameters: Dict[str, Any]
    deinterleaved_llrs: np.ndarray
    confidence: float
    status: EpistemicStatus = EpistemicStatus.HYPOTHESIZED


@dataclass
class FECHypothesis:
    """Candidate FEC decoder evaluation record."""
    family: FECCodeFamily
    code_rate: str
    parameters: Dict[str, Any]
    decoded_bits: np.ndarray
    syndrome: np.ndarray
    syndrome_zero: bool
    syndrome_weight: int
    iterations: int
    ber_estimate: float
    status: EpistemicStatus = EpistemicStatus.HYPOTHESIZED
    validation_detail: str = ""


@dataclass
class CRCProfile:
    """Complete specification of a CRC algorithm."""
    name: str
    width: int
    poly: int
    init: int
    refin: bool
    refout: bool
    xorout: int
    check: int = 0
    residue: int = 0
    bit_order: str = "MSB"
    byte_order: str = "BIG"


@dataclass
class FrameAnalysisResult:
    """Framing, bit-stream correlation, and payload extraction record."""
    status: EpistemicStatus = EpistemicStatus.HYPOTHESIZED
    frame_structure_detected: bool = False
    frame_type: str = "GENERIC"
    sync_pattern_name: Optional[str] = None
    sync_pattern_bits: Optional[str] = None
    frame_length: int = 0
    repeated_frames_found: int = 0
    bit_offset: int = 0
    header_bits: Optional[np.ndarray] = None
    payload_bits: Optional[np.ndarray] = None
    crc_profile: Optional[str] = None
    crc_match: bool = False
    crc_calculated: Optional[int] = None
    crc_received: Optional[int] = None
    recovered_ascii: str = ""
    recovered_hex: str = ""
    valid_frames: int = 0
    total_frames: int = 0
    crc_pass_rate: float = 0.0
    crc_matches_summary: str = ""

    @property
    def preamble_detected(self) -> bool:
        return self.sync_pattern_name is not None

    @property
    def crc_name(self) -> Optional[str]:
        return self.crc_profile

    @property
    def sync_word_name(self) -> Optional[str]:
        return self.sync_pattern_name




@dataclass
class EvidenceItem:
    """Single item of evidence categorized in the epistemic hierarchy."""
    tier: EpistemicStatus
    domain: str
    description: str
    value: Any
    confidence: float


@dataclass
class EvidenceReport:
    """Consolidated multi-stage evidence report with candidate rankings."""
    observed: List[EvidenceItem] = field(default_factory=list)
    estimated: List[EvidenceItem] = field(default_factory=list)
    hypothesized: List[EvidenceItem] = field(default_factory=list)
    validated: List[EvidenceItem] = field(default_factory=list)
    unknown: List[EvidenceItem] = field(default_factory=list)
    candidate_ranking: List[Dict[str, Any]] = field(default_factory=list)
    overall_verdict: str = "UNKNOWN"
    overall_confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    numeric_score: float = 0.0

    def to_intelligence_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Exports all extracted intelligence parameters partitioned into:
        { "parameter_name": { "value": ..., "unit": ..., "status": ..., "confidence": ..., "evidence": [...] } }
        """
        res: Dict[str, Dict[str, Any]] = {}
        for item in self.observed:
            key = f"observed_{item.domain.lower().replace(' ', '_')}"
            res[key] = {
                "value": item.value,
                "unit": "measurement",
                "status": item.tier.value,
                "confidence": item.confidence,
                "evidence": [item.description]
            }
        for item in self.estimated:
            key = f"estimated_{item.domain.lower().replace(' ', '_')}"
            res[key] = {
                "value": item.value,
                "unit": "estimate",
                "status": item.tier.value,
                "confidence": item.confidence,
                "evidence": [item.description]
            }
        for item in self.hypothesized:
            key = f"hypothesized_{item.domain.lower().replace(' ', '_')}"
            res[key] = {
                "value": item.value,
                "unit": "candidate",
                "status": item.tier.value,
                "confidence": item.confidence,
                "evidence": [item.description]
            }
        for item in self.validated:
            key = f"validated_{item.domain.lower().replace(' ', '_')}"
            res[key] = {
                "value": item.value,
                "unit": "proof",
                "status": item.tier.value,
                "confidence": item.confidence,
                "evidence": [item.description]
            }
        return res


@dataclass
class BlindParameterVector:
    """
    Contract for purely blind physical parameter extraction without prior protocol knowledge.
    Computable without knowing whether the signal is NAVTEX, GSM, DMR, STANAG, etc.
    Every field is derived strictly from mathematics and physical wave properties.
    """
    signal_presence: bool = True
    signal_duration: float = 0.0
    center_frequency_hz: float = 0.0
    peak_frequency_hz: float = 0.0
    spectral_centroid_hz: float = 0.0
    spectral_median_hz: float = 0.0
    energy_center_frequency_hz: float = 0.0
    dominant_component_frequencies: List[float] = field(default_factory=list)
    frequency_structure: str = "UNKNOWN"
    occupied_bandwidth_hz: float = 0.0
    bandwidth_3db_hz: float = 0.0
    bandwidth_10db_hz: float = 0.0
    bandwidth_20db_hz: float = 0.0
    snr_db: float = 0.0
    local_noise_floor_db: float = -100.0
    signal_floor_db: float = -50.0
    peak_floor_db: float = 0.0
    papr_db: float = 0.0
    envelope_mean: float = 0.0
    envelope_std: float = 0.0
    envelope_variance_ratio: float = 0.0
    envelope_dynamic_range: float = 0.0
    instantaneous_freq_mean_hz: float = 0.0
    instantaneous_freq_std_hz: float = 0.0
    frequency_deviation_hz: Optional[float] = None
    num_frequency_states: Optional[int] = None
    state_frequencies_hz: List[float] = field(default_factory=list)
    tone_spacing_hz: Optional[float] = None
    symbol_rate_candidates: List[Dict[str, Any]] = field(default_factory=list)
    symbol_rate_consensus_hz: Optional[float] = None
    symbol_rate_consensus_method: str = "UNKNOWN"
    timing_periodicity_s: Optional[float] = None
    symbol_dwell_time_ms: Optional[float] = None
    burst_duration_ms: Optional[float] = None
    duty_cycle_pct: float = 100.0
    pri_us: Optional[float] = None
    prf_hz: Optional[float] = None
    pulse_width_us: Optional[float] = None
    cyclostationary_frequencies: List[float] = field(default_factory=list)
    spectral_flatness: float = 0.0
    spectral_kurtosis: float = 0.0
    phase_statistics: Dict[str, float] = field(default_factory=dict)
    amplitude_statistics: Dict[str, float] = field(default_factory=dict)
    constellation_geometry: Dict[str, Any] = field(default_factory=dict)
    frequency_hopping_info: Dict[str, Any] = field(default_factory=dict)
    carrier_drift_info: Dict[str, Any] = field(default_factory=dict)
    chirp_info: Dict[str, Any] = field(default_factory=dict)
    morphology_fingerprint: Dict[str, str] = field(default_factory=dict)
    parameter_consistency: Dict[str, Any] = field(default_factory=dict)
    blindness_provenance: Dict[str, Any] = field(default_factory=dict)
    segments: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def pw_us(self) -> Optional[float]:
        """Convenience alias for pulse_width_us."""
        return self.pulse_width_us

    def to_dict(self) -> Dict[str, Any]:
        """Convert blind parameter vector to standard dictionary."""
        return {
            "signal_presence": self.signal_presence,
            "signal_duration": self.signal_duration,
            "center_frequency_hz": self.center_frequency_hz,
            "peak_frequency_hz": self.peak_frequency_hz,
            "spectral_centroid_hz": self.spectral_centroid_hz,
            "spectral_median_hz": self.spectral_median_hz,
            "energy_center_frequency_hz": self.energy_center_frequency_hz,
            "dominant_component_frequencies": list(self.dominant_component_frequencies),
            "frequency_structure": self.frequency_structure,
            "occupied_bandwidth_hz": self.occupied_bandwidth_hz,
            "bandwidth_3db_hz": self.bandwidth_3db_hz,
            "bandwidth_10db_hz": self.bandwidth_10db_hz,
            "bandwidth_20db_hz": self.bandwidth_20db_hz,
            "snr_db": self.snr_db,
            "local_noise_floor_db": self.local_noise_floor_db,
            "signal_floor_db": self.signal_floor_db,
            "peak_floor_db": self.peak_floor_db,
            "papr_db": self.papr_db,
            "envelope_mean": self.envelope_mean,
            "envelope_std": self.envelope_std,
            "envelope_variance_ratio": self.envelope_variance_ratio,
            "envelope_dynamic_range": self.envelope_dynamic_range,
            "instantaneous_freq_mean_hz": self.instantaneous_freq_mean_hz,
            "instantaneous_freq_std_hz": self.instantaneous_freq_std_hz,
            "frequency_deviation_hz": self.frequency_deviation_hz,
            "num_frequency_states": self.num_frequency_states,
            "state_frequencies_hz": list(self.state_frequencies_hz),
            "tone_spacing_hz": self.tone_spacing_hz,
            "symbol_rate_candidates": list(self.symbol_rate_candidates),
            "symbol_rate_consensus_hz": self.symbol_rate_consensus_hz,
            "symbol_rate_consensus_method": self.symbol_rate_consensus_method,
            "timing_periodicity_s": self.timing_periodicity_s,
            "symbol_dwell_time_ms": self.symbol_dwell_time_ms,
            "burst_duration_ms": self.burst_duration_ms,
            "duty_cycle_pct": self.duty_cycle_pct,
            "pri_us": self.pri_us,
            "prf_hz": self.prf_hz,
            "pulse_width_us": self.pulse_width_us,
            "pw_us": self.pulse_width_us,
            "cyclostationary_frequencies": list(self.cyclostationary_frequencies),
            "spectral_flatness": self.spectral_flatness,
            "spectral_kurtosis": self.spectral_kurtosis,
            "phase_statistics": dict(self.phase_statistics),
            "amplitude_statistics": dict(self.amplitude_statistics),
            "constellation_geometry": dict(self.constellation_geometry),
            "frequency_hopping_info": dict(self.frequency_hopping_info),
            "carrier_drift_info": dict(self.carrier_drift_info),
            "chirp_info": dict(self.chirp_info),
            "morphology_fingerprint": dict(self.morphology_fingerprint),
            "parameter_consistency": dict(self.parameter_consistency),
            "blindness_provenance": dict(self.blindness_provenance),
            "segments": list(self.segments)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlindParameterVector":
        """Instantiate BlindParameterVector from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        d = dict(data)
        if "pw_us" in d and "pulse_width_us" not in d:
            d["pulse_width_us"] = d["pw_us"]
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
