"""
Modulation Inference Engine (Layer B)
=====================================
Infers modulation family and physical modulation geometry strictly from the
Layer A BlindParameterVector and waveform morphology fingerprint.

Strict Architectural Contract:
- Layer B operates without knowing whether the signal is NAVTEX, GSM, DMR,
  STANAG, ALE, FT8, RTTY, etc.
- Maps physical wave features (frequency states, phase fold symmetry, envelope
  statistics, instantaneous frequency variance) to canonical modulation types:
  2-FSK, 4-FSK, M-FSK, BPSK, QPSK, 8-PSK, 16-QAM, FMCW, Pulsed CW, Analog FM, AM, CW.
"""

from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field
import numpy as np

from .contracts import BlindParameterVector, ModulationFamily, ConfidenceLevel


@dataclass
class ModulationInferenceResult:
    """Outcome of blind modulation inference."""
    inferred_modulation: str
    modulation_family: ModulationFamily
    confidence: float
    confidence_level: ConfidenceLevel
    rationales: List[str]
    candidate_modulations: List[Tuple[str, float]]
    constellation_order: Optional[int] = None
    is_digital: bool = True
    is_pulsed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inferred_modulation": self.inferred_modulation,
            "modulation_family": self.modulation_family.value,
            "confidence": float(np.round(self.confidence, 3)),
            "confidence_level": self.confidence_level.value,
            "rationales": list(self.rationales),
            "candidate_modulations": [(m, float(np.round(c, 3))) for m, c in self.candidate_modulations],
            "constellation_order": self.constellation_order,
            "is_digital": self.is_digital,
            "is_pulsed": self.is_pulsed
        }


def infer_modulation_from_blind_params(
    params: BlindParameterVector
) -> ModulationInferenceResult:
    """
    Infers modulation class purely from blind physical observations in BlindParameterVector.
    """
    rationales: List[str] = []
    candidates: List[Tuple[str, float]] = []

    if not params.signal_presence:
        return ModulationInferenceResult(
            inferred_modulation="Noise Floor / Inactive",
            modulation_family=ModulationFamily.UNKNOWN_OOD,
            confidence=0.0,
            confidence_level=ConfidenceLevel.LOW,
            rationales=["Zero signal presence detected across temporal segmentation"],
            candidate_modulations=[("Noise Floor", 1.0)],
            is_digital=False,
            is_pulsed=False
        )

    morph = params.morphology_fingerprint
    geom = params.constellation_geometry
    chirp = params.chirp_info
    hop = params.frequency_hopping_info

    temporal = morph.get("temporal_pattern", "CONTINUOUS")
    tone_nat = morph.get("tone_nature", "SINGLE_TONE")
    env_nat = morph.get("envelope_nature", "CONSTANT_ENVELOPE")
    freq_nat = morph.get("frequency_nature", "CONSTANT_FREQUENCY")
    drift = params.carrier_drift_info or {}

    # Resolve keyed, drifting carriers before chirp/FSK branches.  A slow
    # monotonic drift is a carrier impairment/physical motion, not evidence
    # of either a deliberate FMCW sweep or discrete FSK states.
    if (
        drift.get("is_drifting")
        and drift.get("keyed_carrier")
        and temporal in ["PULSED", "BURSTY"]
        and env_nat == "AMPLITUDE_VARYING"
        and tone_nat in ["SINGLE_TONE", "WIDEBAND_CONTINUOUS"]
    ):
        rationales.append("Persistent carrier drift with keyed envelope; classified as OOK/CW rather than chirp or FSK")
        candidates.append(("On-Off Keying (OOK / CW Morse)", 0.94))
        return ModulationInferenceResult(
            inferred_modulation="On-Off Keying (OOK)",
            modulation_family=ModulationFamily.ANALOG_AM,
            confidence=0.94,
            confidence_level=ConfidenceLevel.HIGH,
            rationales=rationales,
            candidate_modulations=candidates,
            constellation_order=2,
            is_digital=False,
            is_pulsed=True
        )

    # -------------------------------------------------------------------------
    # 1. Pulsed Radar / FMCW Chirp Detection
    # -------------------------------------------------------------------------
    if chirp.get("is_chirp") or params.frequency_structure == "CHIRP":
        rationales.append("Frequency-time analysis reveals linear swept frequency trajectory (Chirp/FMOP)")
        candidates.append(("FMCW / Linear FM Chirp Radar", 0.95))
        return ModulationInferenceResult(
            inferred_modulation="FMCW / Linear FM Chirp",
            modulation_family=ModulationFamily.ANALOG_FM,
            confidence=0.95,
            confidence_level=ConfidenceLevel.HIGH,
            rationales=rationales,
            candidate_modulations=candidates,
            constellation_order=None,
            is_digital=False,
            is_pulsed=True
        )

    # -------------------------------------------------------------------------
    # 2. Frequency Hopping
    # -------------------------------------------------------------------------
    if hop.get("hop_detected") or params.frequency_structure == "HOPPING":
        h_cnt = hop.get("hop_count", 0)
        rationales.append(f"Discovered discrete step frequency transitions across {h_cnt} hop channels")
        candidates.append(("Frequency Hopping Spread Spectrum (FHSS)", 0.88))
        return ModulationInferenceResult(
            inferred_modulation="Frequency Hopping Spread Spectrum",
            modulation_family=ModulationFamily.UNKNOWN_OOD,
            confidence=0.88,
            confidence_level=ConfidenceLevel.HIGH,
            rationales=rationales,
            candidate_modulations=candidates,
            is_digital=True,
            is_pulsed=False
        )

    # -------------------------------------------------------------------------
    # 3. Frequency Shift Keying (FSK / M-FSK)
    # -------------------------------------------------------------------------
    m_states = params.num_frequency_states
    spacing = params.tone_spacing_hz

    if m_states == 2 and spacing is not None and spacing > 10.0:
        rationales.append(f"Discovered exactly 2 instantaneous frequency states with {spacing:.1f} Hz tone spacing")
        if params.symbol_rate_consensus_hz is not None:
            rationales.append(f"Blind symbol rate consensus: {params.symbol_rate_consensus_hz:.1f} Baud")
        candidates.append(("2-FSK", 0.96))
        candidates.append(("AFSK", 0.70))
        return ModulationInferenceResult(
            inferred_modulation="2-FSK",
            modulation_family=ModulationFamily.FSK_2,
            confidence=0.96,
            confidence_level=ConfidenceLevel.HIGH,
            rationales=rationales,
            candidate_modulations=candidates,
            constellation_order=2,
            is_digital=True,
            is_pulsed=False
        )

    if m_states == 4 and spacing is not None:
        rationales.append(f"Discovered 4 distinct frequency states with {spacing:.1f} Hz spacing")
        candidates.append(("4-FSK", 0.94))
        return ModulationInferenceResult(
            inferred_modulation="4-FSK",
            modulation_family=ModulationFamily.FSK_4,
            confidence=0.94,
            confidence_level=ConfidenceLevel.HIGH,
            rationales=rationales,
            candidate_modulations=candidates,
            constellation_order=4,
            is_digital=True,
            is_pulsed=False
        )

    if ((m_states is not None and m_states >= 5) or (len(params.dominant_component_frequencies) >= 6 and (m_states is None or m_states >= 4))) and spacing is not None and spacing > 1.0:
        n_tones = m_states if (m_states and m_states >= 4) else len(params.dominant_component_frequencies)
        rationales.append(f"Discovered multi-tone frequency comb with {n_tones} discrete states and {spacing:.1f} Hz spacing")
        candidates.append((f"{n_tones}-Tone M-FSK", 0.95))
        return ModulationInferenceResult(
            inferred_modulation=f"{n_tones}-Tone M-FSK",
            modulation_family=ModulationFamily.FSK_4,
            confidence=0.95,
            confidence_level=ConfidenceLevel.HIGH,
            rationales=rationales,
            candidate_modulations=candidates,
            constellation_order=n_tones,
            is_digital=True,
            is_pulsed=False
        )

    # -------------------------------------------------------------------------
    # 3b. Continuous Wave (CW / Unmodulated Carrier)
    # -------------------------------------------------------------------------
    if (params.occupied_bandwidth_hz < 200.0 or params.instantaneous_freq_std_hz < 25.0) and env_nat == "CONSTANT_ENVELOPE" and tone_nat == "SINGLE_TONE" and params.symbol_rate_consensus_hz is None and not geom:
        rationales.append("Ultra-narrow spectral emission and unmodulated constant envelope (no phase/symbol transitions)")
        candidates.append(("Continuous Wave (CW)", 0.98))
        return ModulationInferenceResult(
            inferred_modulation="Continuous Wave (CW)",
            modulation_family=ModulationFamily.UNKNOWN_OOD,
            confidence=0.98,
            confidence_level=ConfidenceLevel.HIGH,
            rationales=rationales,
            candidate_modulations=candidates,
            constellation_order=1,
            is_digital=False,
            is_pulsed=False
        )

    # -------------------------------------------------------------------------
    # 4. Digital Phase / Amplitude Modulation (PSK / QAM)
    # -------------------------------------------------------------------------
    phase_fold = geom.get("phase_fold_symmetry_m")
    amp_struct = geom.get("amplitude_structure", "")
    cluster_cnt = geom.get("cluster_count_estimate")

    if phase_fold == 2 and amp_struct == "CONSTANT_ENVELOPE" and (params.symbol_rate_consensus_hz is not None or params.occupied_bandwidth_hz >= 150.0 or params.instantaneous_freq_std_hz >= 25.0):
        rationales.append("Phase fold symmetry maximizes at M=2 (180-deg phase inversion); constant envelope")
        candidates.append(("BPSK", 0.93))
        return ModulationInferenceResult(
            inferred_modulation="BPSK",
            modulation_family=ModulationFamily.BPSK,
            confidence=0.93,
            confidence_level=ConfidenceLevel.HIGH,
            rationales=rationales,
            candidate_modulations=candidates,
            constellation_order=2,
            is_digital=True,
            is_pulsed=False
        )

    if phase_fold == 4 and amp_struct == "CONSTANT_ENVELOPE":
        rationales.append("Phase fold symmetry maximizes at M=4 (90-deg phase shifts); constant envelope")
        candidates.append(("QPSK", 0.92))
        return ModulationInferenceResult(
            inferred_modulation="QPSK",
            modulation_family=ModulationFamily.QPSK,
            confidence=0.92,
            confidence_level=ConfidenceLevel.HIGH,
            rationales=rationales,
            candidate_modulations=candidates,
            constellation_order=4,
            is_digital=True,
            is_pulsed=False
        )

    if phase_fold == 8 and amp_struct == "CONSTANT_ENVELOPE":
        rationales.append("Phase fold symmetry maximizes at M=8 (45-deg phase shifts); constant envelope")
        candidates.append(("8-PSK", 0.91))
        return ModulationInferenceResult(
            inferred_modulation="8-PSK",
            modulation_family=ModulationFamily.PSK_8,
            confidence=0.91,
            confidence_level=ConfidenceLevel.HIGH,
            rationales=rationales,
            candidate_modulations=candidates,
            constellation_order=8,
            is_digital=True,
            is_pulsed=False
        )

    if amp_struct == "MULTILEVEL_AMPLITUDE" and phase_fold == 4:
        rationales.append("Multi-ring amplitude distribution with 4-fold phase quadrant symmetry (QAM)")
        candidates.append(("16-QAM", 0.88))
        return ModulationInferenceResult(
            inferred_modulation="16-QAM",
            modulation_family=ModulationFamily.QAM_16,
            confidence=0.88,
            confidence_level=ConfidenceLevel.MEDIUM,
            rationales=rationales,
            candidate_modulations=candidates,
            constellation_order=16,
            is_digital=True,
            is_pulsed=False
        )

    # -------------------------------------------------------------------------
    # 5. Continuous Wave (CW) & On-Off Keying (OOK)
    # -------------------------------------------------------------------------
    # 4b. GMSK / Continuous Phase Modulation (CPM)
    # -------------------------------------------------------------------------
    if env_nat == "CONSTANT_ENVELOPE" and params.symbol_rate_consensus_hz is not None and params.symbol_rate_consensus_hz > 100.0 and (m_states is None or m_states <= 2) and phase_fold is None:
        rationales.append(f"Continuous phase frequency modulation with constant envelope and blind clock at {params.symbol_rate_consensus_hz:.1f} Baud (GMSK/CPM)")
        candidates.append(("GMSK / Continuous Phase Modulation", 0.92))
        return ModulationInferenceResult(
            inferred_modulation="GMSK",
            modulation_family=ModulationFamily.FSK_2,
            confidence=0.92,
            confidence_level=ConfidenceLevel.HIGH,
            rationales=rationales,
            candidate_modulations=candidates,
            constellation_order=2,
            is_digital=True,
            is_pulsed=(temporal in ["PULSED", "BURSTY"])
        )

    # -------------------------------------------------------------------------
    # 5. Continuous Wave (CW) & On-Off Keying (OOK)
    # -------------------------------------------------------------------------
    if params.occupied_bandwidth_hz < 150.0 and env_nat == "CONSTANT_ENVELOPE" and tone_nat == "SINGLE_TONE" and not geom:
        rationales.append("Ultra-narrow spectral emission (< 150 Hz) and unmodulated constant envelope")
        candidates.append(("Continuous Wave (CW)", 0.98))
        return ModulationInferenceResult(
            inferred_modulation="Continuous Wave (CW)",
            modulation_family=ModulationFamily.UNKNOWN_OOD,
            confidence=0.98,
            confidence_level=ConfidenceLevel.HIGH,
            rationales=rationales,
            candidate_modulations=candidates,
            constellation_order=1,
            is_digital=False,
            is_pulsed=False
        )

    if env_nat == "AMPLITUDE_VARYING" and tone_nat == "SINGLE_TONE" and temporal in ["BURSTY", "PULSED"] and params.instantaneous_freq_std_hz < 300.0:
        rationales.append("Carrier envelope keyed on and off with discrete timing intervals (OOK)")
        candidates.append(("On-Off Keying (OOK / CW Morse)", 0.90))
        return ModulationInferenceResult(
            inferred_modulation="On-Off Keying (OOK)",
            modulation_family=ModulationFamily.ANALOG_AM,
            confidence=0.90,
            confidence_level=ConfidenceLevel.HIGH,
            rationales=rationales,
            candidate_modulations=candidates,
            constellation_order=2,
            is_digital=True,
            is_pulsed=False
        )

    # -------------------------------------------------------------------------
    # 5b. Pulsed CW / Unmodulated Radar
    # -------------------------------------------------------------------------
    if (temporal == "PULSED" or (params.prf_hz is not None and params.prf_hz > 0)) and params.duty_cycle_pct < 45.0:
        rationales.append(f"Short duty cycle ({params.duty_cycle_pct:.1f}%) with discrete pulse train geometry")
        candidates.append(("Pulsed CW / Radar", 0.90))
        return ModulationInferenceResult(
            inferred_modulation="Pulsed CW / Radar",
            modulation_family=ModulationFamily.UNKNOWN_OOD,
            confidence=0.90,
            confidence_level=ConfidenceLevel.HIGH,
            rationales=rationales,
            candidate_modulations=candidates,
            constellation_order=None,
            is_digital=False,
            is_pulsed=True
        )

    # -------------------------------------------------------------------------
    # 6. Analog FM & Analog AM
    # -------------------------------------------------------------------------
    if params.instantaneous_freq_std_hz > 500.0 and params.spectral_flatness < 0.70:
        rationales.append(f"High continuous frequency deviation (std = {params.instantaneous_freq_std_hz:.1f} Hz) with smooth spectrum")
        candidates.append(("Analog FM / Voice", 0.85))
        return ModulationInferenceResult(
            inferred_modulation="Analog FM",
            modulation_family=ModulationFamily.ANALOG_FM,
            confidence=0.85,
            confidence_level=ConfidenceLevel.MEDIUM,
            rationales=rationales,
            candidate_modulations=candidates,
            is_digital=False,
            is_pulsed=False
        )

    if env_nat == "AMPLITUDE_VARYING" and params.papr_db > 6.0:
        rationales.append("Continuous amplitude variation with pronounced envelope PAPR (AM Speech/Audio)")
        candidates.append(("Analog AM", 0.80))
        return ModulationInferenceResult(
            inferred_modulation="Analog AM",
            modulation_family=ModulationFamily.ANALOG_AM,
            confidence=0.80,
            confidence_level=ConfidenceLevel.MEDIUM,
            rationales=rationales,
            candidate_modulations=candidates,
            is_digital=False,
            is_pulsed=False
        )

    # -------------------------------------------------------------------------
    # 7. Generic Fallback
    # -------------------------------------------------------------------------
    rationales.append("Waveform exhibits complex or uncataloged modulation geometry")
    candidates.append(("Generic RF Waveform", 0.50))
    return ModulationInferenceResult(
        inferred_modulation="Generic RF Waveform",
        modulation_family=ModulationFamily.UNKNOWN_OOD,
        confidence=0.50,
        confidence_level=ConfidenceLevel.LOW,
        rationales=rationales,
        candidate_modulations=candidates,
        is_digital=True,
        is_pulsed=False
    )
