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
    """Hypothesis governing downstream synchronization and demodulation."""
    modulation: ModulationFamily
    symbol_rate: Optional[float]
    carrier_offset: float
    samples_per_symbol: float
    pulse_model: Optional[str]
    confidence: float
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    is_ood: bool = False
    mahalanobis_distance: float = 0.0
    evidence: List[str] = field(default_factory=list)
    rejected_hypotheses: List[str] = field(default_factory=list)



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
