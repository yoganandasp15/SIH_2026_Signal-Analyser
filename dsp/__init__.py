"""
NTRO Automated Signal Analysis Suite - Digital Signal Processing (DSP) Core
===========================================================================
Modular, deterministic signal processing modules for:
- Binary I/Q & WAV file loading and auto-probing (loaders.py)
- DC offset removal & power normalization (preprocessor.py)
- Welch Power Spectral Density & Spectrogram (spectral.py)
- Blind parameter extraction: fc, BW, M2M4 SNR, Baud Rate (parameter_extractor.py)
- Higher-Order Cumulant (HOC) modulation classifier (modulation_classifier.py)
- Radar pulse parameter extraction: PW, PRI, Duty Cycle (pulse_analyzer.py)
- Autonomous Signal Detection & Zero False-Positive AMC (autonomous_detector.py)
- Dynamic Adaptive Extraction Pipeline (adaptive_pipeline.py)

V3 Hardened Epistemic Signal Intelligence Architecture:
- contracts.py: Epistemic tiers, typed schemas, data contracts
- forensics.py: Universal input forensics & 3-state sampling rate authority
- conditioning.py: DC correction, conditional IQ imbalance compensation, scale normalization
- features.py: 16-scalar feature vector, mathematically verified C63 cumulant
- amc.py: Open-set AMC with Mahalanobis outlier rejection
- synchronization.py: Costas carrier phase PLL, Gardner clock recovery, early-late FSK
- demodulator.py: Multi-modulation soft slicers emitting continuous LLRs
- deinterleaver.py: Four de-interleaver topologies with bounded searching
- fec.py: Multi-decoder FEC (Viterbi, Reed-Solomon GF(2^8), LDPC Min-Sum, Concatenated)
- framing.py: Parameterized CRC engines, sync marker cross-correlation, conditional ASCII
- evidence_fusion.py: Gated multi-stage evidence fusion producing epistemic audit reports
"""

# Legacy & Core DSP exports
from .loaders import (
    load_signal_file,
    parse_iq_binary,
    load_wav_signal,
    probe_binary_format,
    auto_detect_file_sample_rate
)
from .preprocessor import remove_dc_offset, normalize_signal_power, apply_bandpass_filter
from .spectral import compute_welch_psd, compute_spectrogram
from .parameter_extractor import (
    estimate_carrier_frequency,
    estimate_occupied_bandwidth,
    estimate_snr_m2m4,
    estimate_symbol_baud_rate,
    extract_all_parameters
)
from .modulation_classifier import (
    compute_higher_order_cumulants,
    classify_modulation_cumulants
)
from .pulse_analyzer import analyze_pulse_train
from .autonomous_detector import detect_signal_autonomously
from .adaptive_pipeline import run_adaptive_pipeline, AdaptiveExtractionPipeline
from .temporal_validator import (
    TemporalValidationConfig,
    MultiWindowTemporalValidator,
    validate_temporal_consistency
)
from .candidate_ranker import (
    CandidateRankingConfig,
    generate_candidate_hypotheses,
    rank_and_evaluate_candidates,
    apply_candidate_ranking_to_detection
)
from .contradiction_analyzer import (
    ContradictionCategory,
    MeasurableContradiction,
    ContradictionConfig,
    ContradictionAnalysisResult,
    analyze_candidate_contradictions,
    apply_contradiction_analysis
)
from .validation_gate import (
    ValidationGateStatus,
    ValidationGateConfig,
    GateCheckResult,
    ValidationGateTrace,
    HypothesisValidationGate,
    run_validation_gate
)
from .parameter_uncertainty import (
    build_parameter_uncertainty_report,
    get_verdict_explanation
)

# V3 Architecture Epistemic & Pipeline Exports
from .contracts import (
    EpistemicStatus,
    SamplingRateStatus,
    ModulationFamily,
    InterleaverType,
    FECCodeFamily,
    ConfidenceLevel,
    SignalRecord,
    SignalFeatures,
    SignalHypothesis,
    SynchronizedSymbols,
    DemodulationResult,
    InterleaverHypothesis,
    FECHypothesis,
    CRCProfile,
    FrameAnalysisResult,
    EvidenceItem,
    EvidenceReport
)
from .forensics import inspect_signal_file
from .conditioning import condition_signal, estimate_iq_imbalance, compensate_iq_imbalance
from .features import extract_signal_features, compute_c63_cumulant
from .amc import classify_modulation_open_set
from .synchronization import synchronize_signal
from .demodulation import demodulate_signal
from .deinterleaving import evaluate_interleaver_candidates
from .fec import (
    evaluate_fec_candidates,
    encode_convolutional,
    decode_viterbi_soft,
    decode_viterbi_hard,
    decode_reed_solomon,
    decode_ldpc_minsum
)
from .framing import analyze_frame_structure, compute_crc, CRC_PROFILES
from .evidence import fuse_evidence

# Layer A & Layer B Blind Parameter & Inference Exports
from .contracts import BlindParameterVector
from .blind_parameter_engine import (
    extract_blind_parameters,
    detect_signal_activity_and_segments,
    estimate_adaptive_noise_floor,
    compute_multi_resolution_spectrum,
    estimate_generic_frequency_structure,
    discover_blind_fsk_states,
    estimate_blind_symbol_rate_consensus,
    estimate_blind_cyclostationary_frequencies,
    discover_constellation_geometry,
    discover_blind_frequency_hopping,
    extract_waveform_morphology,
    validate_parameter_consistency,
    build_blindness_provenance
)
from .modulation_inference import infer_modulation_from_blind_params, ModulationInferenceResult
from .protocol_inference import infer_protocol_from_blind_params


__all__ = [
    # Legacy Core DSP
    "load_signal_file",
    "parse_iq_binary",
    "load_wav_signal",
    "probe_binary_format",
    "auto_detect_file_sample_rate",
    "remove_dc_offset",
    "normalize_signal_power",
    "apply_bandpass_filter",
    "compute_welch_psd",
    "compute_spectrogram",
    "estimate_carrier_frequency",
    "estimate_occupied_bandwidth",
    "estimate_snr_m2m4",
    "estimate_symbol_baud_rate",
    "extract_all_parameters",
    "compute_higher_order_cumulants",
    "classify_modulation_cumulants",
    "analyze_pulse_train",
    "detect_signal_autonomously",
    "run_adaptive_pipeline",
    "AdaptiveExtractionPipeline",
    "CandidateRankingConfig",
    "generate_candidate_hypotheses",
    "rank_and_evaluate_candidates",
    "apply_candidate_ranking_to_detection",
    "ContradictionCategory",
    "MeasurableContradiction",
    "ContradictionConfig",
    "ContradictionAnalysisResult",
    "analyze_candidate_contradictions",
    "apply_contradiction_analysis",
    "ValidationGateStatus",
    "ValidationGateConfig",
    "GateCheckResult",
    "ValidationGateTrace",
    "HypothesisValidationGate",
    "run_validation_gate",
    "build_parameter_uncertainty_report",
    "get_verdict_explanation",
    # V3 Epistemic Contracts
    "EpistemicStatus",
    "SamplingRateStatus",
    "ModulationFamily",
    "InterleaverType",
    "FECCodeFamily",
    "ConfidenceLevel",
    "SignalRecord",
    "SignalFeatures",
    "SignalHypothesis",
    "SynchronizedSymbols",
    "DemodulationResult",
    "InterleaverHypothesis",
    "FECHypothesis",
    "CRCProfile",
    "FrameAnalysisResult",
    "EvidenceItem",
    "EvidenceReport",
    # V3 Processing Pipeline
    "inspect_signal_file",
    "condition_signal",
    "estimate_iq_imbalance",
    "compensate_iq_imbalance",
    "extract_signal_features",
    "compute_c63_cumulant",
    "classify_modulation_open_set",
    "synchronize_signal",
    "demodulate_signal",
    "evaluate_interleaver_candidates",
    "evaluate_fec_candidates",
    "encode_convolutional",
    "decode_viterbi_soft",
    "decode_viterbi_hard",
    "decode_reed_solomon",
    "decode_ldpc_minsum",
    "analyze_frame_structure",
    "compute_crc",
    "CRC_PROFILES",
    "fuse_evidence",
    # Temporal Validation
    "TemporalValidationConfig",
    "MultiWindowTemporalValidator",
    "validate_temporal_consistency",
    # Blind Parameter Engine & Inference (Layer A & B)
    "BlindParameterVector",
    "extract_blind_parameters",
    "detect_signal_activity_and_segments",
    "estimate_adaptive_noise_floor",
    "compute_multi_resolution_spectrum",
    "estimate_generic_frequency_structure",
    "discover_blind_fsk_states",
    "estimate_blind_symbol_rate_consensus",
    "estimate_blind_cyclostationary_frequencies",
    "discover_constellation_geometry",
    "discover_blind_frequency_hopping",
    "extract_waveform_morphology",
    "validate_parameter_consistency",
    "build_blindness_provenance",
    "infer_modulation_from_blind_params",
    "ModulationInferenceResult",
    "infer_protocol_from_blind_params"
]

