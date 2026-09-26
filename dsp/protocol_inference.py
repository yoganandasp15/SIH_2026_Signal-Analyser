"""
Protocol Inference Engine (Layer C)
===================================
Matches objective parameters from Layer A (BlindParameterVector) and
Layer B (ModulationInferenceResult) against cataloged protocol invariants.

Strict Architectural Contract:
- Layer C is downstream of blind parameter extraction.
- Protocol knowledge (NAVTEX, DMR, GSM, STANAG, ALE, FT8, etc.) is utilized
  strictly as an epistemic classification/matching layer, NEVER upstream of
  the physical estimators.
- If a signal does not match any cataloged protocol standard, it remains
  an uncataloged physical waveform with 100% blind parameter recovery intact.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from .contracts import BlindParameterVector, ModulationFamily
from .modulation_inference import ModulationInferenceResult


# Catalog of Physical Invariant Profiles for Standard Signals
PROTOCOL_INVARIANT_CATALOG = [
    {
        "class_id": "FSK_NAVTEX_SITOR_B",
        "protocol_name": "NAVTEX / SITOR-B (Maritime Safety Telemetry)",
        "modulation_family": "2-FSK",
        "expected_spacing_hz": 170.0,
        "spacing_tolerance_hz": 30.0,
        "expected_baud": 100.0,
        "baud_tolerance": 12.0,
        "description": "IMO maritime safety 170 Hz shift 100 Baud 2-FSK broadcast"
    },
    {
        "class_id": "FSK_RTTY_BAUDOT",
        "protocol_name": "Baudot RTTY (Radioteletype)",
        "modulation_family": "2-FSK",
        "expected_spacing_hz": 170.0,
        "spacing_tolerance_hz": 35.0,
        "expected_baud": 45.45,
        "baud_tolerance": 8.0,
        "description": "Standard HF amateur/diplomatic radioteletype"
    },
    {
        "class_id": "FSK_ASCII_ITA5",
        "protocol_name": "ASCII / ITA-5 FSK (110 Baud)",
        "modulation_family": "2-FSK",
        "expected_spacing_hz": 170.0,
        "spacing_tolerance_hz": 35.0,
        "expected_baud": 110.0,
        "baud_tolerance": 10.0,
        "description": "ASCII 110 Baud 170 Hz shift telecommunications"
    },
    {
        "class_id": "FSK_BELL202_APRS",
        "protocol_name": "Bell 202 AFSK / APRS (Packet Data)",
        "modulation_family": "2-FSK",
        "expected_spacing_hz": 1000.0,
        "spacing_tolerance_hz": 200.0,
        "expected_baud": 1200.0,
        "baud_tolerance": 150.0,
        "description": "Bell 202 AFSK (1200/2200 Hz) 1200 Baud packet radio"
    },
    {
        "class_id": "PAGING_POCSAG",
        "protocol_name": "POCSAG (Radio Paging)",
        "modulation_family": "2-FSK",
        "expected_spacing_hz": 9000.0,
        "spacing_tolerance_hz": 1500.0,
        "expected_baud": 1200.0,
        "baud_tolerance": 200.0,
        "description": "POCSAG radio paging +/- 4.5 kHz dev 1200 Baud"
    },
    {
        "class_id": "MOBILE_DMR_4FSK",
        "protocol_name": "DMR Tier II (Digital Mobile Radio TDMA 4-FSK)",
        "modulation_family": "4-FSK",
        "expected_spacing_hz": None,
        "expected_baud": 4800.0,
        "baud_tolerance": 400.0,
        "description": "ETSI DMR Tier II 4-FSK 4800 Baud TDMA two-slot"
    },
    {
        "class_id": "MIL_STD_188_141_2G_ALE",
        "protocol_name": "MIL-STD-188-141 2G ALE (Tactical HF Handshake)",
        "modulation_family": "M-FSK",
        "expected_spacing_hz": 250.0,
        "spacing_tolerance_hz": 40.0,
        "expected_baud": 125.0,
        "baud_tolerance": 15.0,
        "description": "Military 8-tone MFSK 125 Baud automatic link establishment"
    },
    {
        "class_id": "MFSK_FT8_WEAK_SIGNAL",
        "protocol_name": "WSJT-X FT8 (Weak-Signal Amateur 8-FSK)",
        "modulation_family": "M-FSK",
        "expected_spacing_hz": 6.25,
        "spacing_tolerance_hz": 3.0,
        "expected_baud": 6.25,
        "baud_tolerance": 2.0,
        "description": "WSJT-X FT8 8-tone MFSK 6.25 Baud weak signal protocol"
    },
    {
        "class_id": "MFSK16_TACTICAL",
        "protocol_name": "MFSK16 (16-Tone MFSK Data)",
        "modulation_family": "M-FSK",
        "expected_spacing_hz": 15.625,
        "spacing_tolerance_hz": 5.0,
        "expected_baud": 15.625,
        "baud_tolerance": 4.0,
        "description": "16-tone MFSK 15.625 Baud digital mode"
    },
    {
        "class_id": "CELLULAR_GSM_DOWNLINK",
        "protocol_name": "GSM 2G BCCH Downlink (Cellular TDMA)",
        "modulation_family": "GMSK",
        "expected_spacing_hz": None,
        "expected_baud": 270833.0,
        "baud_tolerance": 25000.0,
        "description": "GSM 2G 270.833 kBaud GMSK 4.615 ms frame TDMA"
    },
    {
        "class_id": "MARITIME_AIS_BURST",
        "protocol_name": "AIS (Automatic Identification System GMSK TDMA)",
        "modulation_family": "GMSK",
        "expected_spacing_hz": None,
        "expected_baud": 9600.0,
        "baud_tolerance": 800.0,
        "description": "Maritime VHF transponder 9600 Baud GMSK TDMA"
    },
    {
        "class_id": "DIGITAL_8PSK_STANAG4285",
        "protocol_name": "STANAG 4285 (NATO Naval Tactical HF 8-PSK)",
        "modulation_family": "8-PSK",
        "expected_spacing_hz": None,
        "expected_baud": 2400.0,
        "baud_tolerance": 200.0,
        "description": "NATO military naval 2400 Baud 8-PSK digital HF modem"
    },
    {
        "class_id": "DIGITAL_PSK_PSK31",
        "protocol_name": "PSK31 (Amateur Varicode BPSK)",
        "modulation_family": "BPSK",
        "expected_spacing_hz": None,
        "expected_baud": 31.25,
        "baud_tolerance": 4.0,
        "description": "Amateur narrow HF 31.25 Baud BPSK Varicode"
    },
    {
        "class_id": "AMATEUR_DSTAR",
        "protocol_name": "D-STAR (Amateur Digital Voice GMSK)",
        "modulation_family": "GMSK",
        "expected_spacing_hz": None,
        "expected_baud": 4800.0,
        "baud_tolerance": 400.0,
        "description": "JARL D-STAR 4800 Baud GMSK digital voice protocol"
    },
    {
        "class_id": "CW_MORSE_OOK",
        "protocol_name": "CW / Morse Code (On-Off Keying)",
        "modulation_family": "OOK",
        "expected_spacing_hz": None,
        "expected_baud": None,
        "description": "Continuous wave carrier keyed with Morse elements"
    },
    {
        "class_id": "ANALOG_WEFAX_HF",
        "protocol_name": "WEFAX (Weather Facsimile FM Subcarrier)",
        "modulation_family": "FM Subcarrier",
        "expected_spacing_hz": None,
        "expected_baud": None,
        "description": "Maritime analog weather facsimile (120/240 LPM)"
    },
    {
        "class_id": "RADAR_DUGA_WOODPECKER",
        "protocol_name": "Russian Woodpecker / Duga 10 Hz PRF OTH Radar",
        "modulation_family": "Pulsed Radar",
        "expected_spacing_hz": None,
        "expected_baud": None,
        "description": "Soviet OTH-B radar 10 Hz PRF"
    },
    {
        "class_id": "RADAR_GENERIC_LFM",
        "protocol_name": "Linear FM Chirp Radar",
        "modulation_family": "FMCW Chirp",
        "expected_spacing_hz": None,
        "expected_baud": None,
        "description": "Linear frequency-modulated chirp radar sweep"
    },
    {
        "class_id": "SATELLITE_TELEMETRY_NFM",
        "protocol_name": "PCM/PM Telemetry (Satellite Beacon / Subcarrier over NFM)",
        "modulation_family": "FM Subcarrier",
        "expected_spacing_hz": None,
        "expected_baud": None,
        "description": "Amateur satellite beacon / PCM-PM telemetry subcarrier"
    }
]


PROTOCOL_TO_PIPELINE: Dict[str, str] = {
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
    "MIL_STD_188_141_2G_ALE": "mfsk_comb",
    "MFSK16_TACTICAL": "mfsk_comb",
    "MFSK_GENERIC": "mfsk_comb",
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


def is_detector_compatible_with_modulation(
    det: Dict[str, Any],
    mod_family: ModulationFamily,
    inferred_mod: str = ""
) -> bool:
    """Verifies that an autonomous detection candidate conforms to the blind modulation family."""
    class_id = det.get("signal_class_id", "")
    det_mod = det.get("modulation_family", "")
    pipe = det.get("extraction_pipeline", "")
    conf = det.get("confidence", 0.0)

    # 1. UNKNOWN_OOD: accept any valid autonomous detection
    if mod_family == ModulationFamily.UNKNOWN_OOD:
        return True

    # 2. Prevent fatal physical contradictions (e.g. pulsed radar vs 2-FSK)
    is_radar_det = ("Radar" in det_mod or "RADAR_" in class_id or "radar" in pipe or "FMCW" in det_mod)
    is_radar_inferred = ("Radar" in inferred_mod or "Chirp" in inferred_mod)
    is_narrow_fsk_det = ("FSK" in det_mod and "4-FSK" not in det_mod and not is_radar_det)
    if is_radar_inferred and is_narrow_fsk_det and "fsk" in pipe:
        return False
    if is_radar_det and ("BPSK" in inferred_mod or "QPSK" in inferred_mod or "M-FSK" in inferred_mod):
        if "GRAVES" not in class_id and "DUGA" not in class_id and "OTH" not in class_id and "GHADIR" not in class_id and "HAARP" not in class_id:
            return False

    # 3. 2-FSK & Continuous Phase Modulation (GMSK / AFSK / TDMA)
    if mod_family == ModulationFamily.FSK_2:
        if (
            "FSK" in det_mod or "fsk" in pipe or class_id.startswith("FSK_") or
            "PAGING_POCSAG" in class_id or "POCSAG" in class_id or
            "AFSK" in det_mod or "Bell 202" in det_mod or "APRS" in class_id or
            "GMSK" in det_mod or "tdma" in pipe or "GSM" in class_id or "AIS" in class_id or "DSTAR" in class_id or
            "DUGA" in class_id or "Woodpecker" in det_mod or "Woodpecker" in class_id
        ):
            return True
        return False

    # 4. 4-FSK and M-FSK Family
    if mod_family == ModulationFamily.FSK_4:
        if (
            # A short pulsed chirp can look like a 4-state frequency comb to
            # Layer B before enough dwell history is available. Preserve the
            # higher-level radar decision when Layer C has an independent,
            # high-confidence physical radar signature.
            (is_radar_det and conf >= 0.90 and ("HAARP" in class_id or "RADAR_" in class_id)) or
            any(m in det_mod for m in ("MFSK", "M-FSK", "8-FSK", "16-FSK", "4-FSK")) or
            any(k in class_id for k in ("ALE", "FT8", "MFSK", "DMR", "4FSK")) or
            pipe in ("mfsk_comb", "tdma_burst", "fsk_detector") or
            "POCSAG" in class_id or
            "STANAG" in class_id or "8-PSK" in det_mod or
            "WEFAX" in class_id or "Facsimile" in det_mod or "wefax" in pipe or
            "Voice" in det_mod or "ANALOG" in class_id or "analog_voice" in pipe
        ):
            return True
        return False

    # 5. Digital PSK & QAM Family
    if mod_family in (ModulationFamily.BPSK, ModulationFamily.QPSK, ModulationFamily.PSK_8, ModulationFamily.QAM_16):
        if any(k in det_mod for k in ("PSK", "QAM", "BPSK", "QPSK", "8-PSK", "16-QAM")) or pipe in ("digital_psk_qam", "psk"):
            return True
        if "PCM/PM" in det_mod or "satellite_telemetry" in pipe or "AIST" in class_id or "SATELLITE" in class_id:
            return True
        if "OTH" in class_id or "RADAR_OTH_SW" in class_id:
            return True
        if conf >= 0.85 and (any(k in class_id for k in ("POCSAG", "MFSK", "DMR", "APRS")) or "fsk" in pipe or "tdma" in pipe):
            return True
        return False

    # 6. Analog FM & FM Subcarrier Family
    if mod_family == ModulationFamily.ANALOG_FM:
        if (
            "FM" in det_mod or "Voice" in det_mod or "analog_voice" in pipe or
            "WEFAX" in class_id or "analog_wefax" in pipe or
            "CODAR" in class_id or "FMCW" in det_mod or
            "satellite_telemetry" in pipe or "PCM/PM" in det_mod or
            "APRS" in class_id or "AFSK" in det_mod or
            "radar" in pipe or "RADAR" in class_id or "HAARP" in class_id or
            "GMSK" in det_mod or "AIS" in class_id or "DSTAR" in class_id or "tdma" in pipe
        ):
            return True
        return False

    # 7. Analog AM & On-Off Keying Family
    if mod_family == ModulationFamily.ANALOG_AM:
        if (
            "AM" in det_mod or "Morse" in det_mod or "OOK" in det_mod or "CW" in det_mod or
            pipe in ("ook_morse", "continuous_wave", "analog_voice") or
            "GRAVES" in class_id or "radar" in pipe or "APRS" in class_id
        ):
            return True
        # A blind envelope can look keyed for packetized telemetry or
        # continuous-phase digital bursts.  Preserve an independent,
        # high-confidence downstream detector decision for those cases; the
        # drift-vs-FSK correction must not erase a stronger protocol result.
        if conf >= 0.85 and any(token in class_id.upper() for token in ("SATELLITE", "DSTAR", "GSM", "DMR", "AIS", "TELEMETRY")):
            return True
        if conf >= 0.85 and any(token in det_mod.upper() for token in ("PCM/PM", "GMSK", "TELEMETRY")):
            return True
        return False

    # 8. Radar & FMCW Chirp Inferences
    if is_radar_inferred:
        if is_radar_det:
            return True
        return False

    # 9. OOK & Morse Inferences
    if "OOK" in inferred_mod:
        if "OOK" in det_mod or "Morse" in det_mod or "CW_MORSE" in class_id or "morse" in pipe or "GRAVES" in class_id:
            return True
        if conf >= 0.85 and ("APRS" in class_id or "fsk" in pipe):
            return True
        return False

    # 10. Continuous Wave
    if "Continuous Wave" in inferred_mod or "CW" in inferred_mod:
        if "Continuous Wave" in det_mod or "CW_" in class_id or "continuous_wave" in pipe or "GRAVES" in class_id:
            return True
        return False

    return True


def infer_protocol_from_blind_params(
    params: BlindParameterVector,
    mod_res: ModulationInferenceResult,
    raw_signal: Optional[np.ndarray] = None,
    fs: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    pulse_info: Optional[Dict[str, Any]] = None,
    temporal_result: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Evaluates protocol standards against the physical BlindParameterVector.
    Demotes candidate standards to an objective post-estimation validation step.
    """
    meta = metadata or {}
    fn = meta.get("file_name", "").lower()
    mod_fam = mod_res.modulation_family.value

    # Check for inactive signal / silence / noise floor
    if not params.signal_presence:
        is_silence = (
            (raw_signal is not None and len(raw_signal) > 0 and float(np.max(np.abs(raw_signal))) < 1e-6) or
            (params.peak_floor_db is not None and params.peak_floor_db <= -140.0)
        )
        proto = "Silence / Zero Amplitude" if is_silence else "Noise Floor / Stationary Gaussian Noise"
        return {
            "signal_class_id": "UNKNOWN",
            "protocol_name": proto,
            "modulation_family": "UNKNOWN",
            "confidence": 0.0,
            "extraction_pipeline": "generic_fallback",
            "physical_evidence": list(mod_res.rationales),
            "parameters_origin": "Layer A BlindParameterVector",
            "blind_parameters": params.to_dict(),
            "modulation_inference": mod_res.to_dict()
        }

    # Check if raw signal or autonomous detector candidate is available
    if raw_signal is not None and fs is not None and fs > 0:
        try:
            from .autonomous_detector import detect_signal_autonomously
            det = detect_signal_autonomously(
                raw_signal, fs, pulse_info=pulse_info, file_name=meta.get("file_name", ""),
                metadata=meta, temporal_result=temporal_result
            )
            if det and det.get("signal_class_id") != "UNKNOWN" and is_detector_compatible_with_modulation(det, mod_res.modulation_family, mod_res.inferred_modulation):
                # Route by measured waveform geometry.  A continuous sweep
                # can be named by a downstream catalog (for example, FMCW),
                # but it must not enter a pulsed-envelope extractor merely
                # because the catalog family contains the word RADAR.
                sweep = params.chirp_info or {}
                if sweep.get("is_continuous_sweep"):
                    det["extraction_pipeline"] = "continuous_fmcw_radar"
                    det["timing_primary_observable"] = "Macro sweep repetition"
                    det["physical_evidence"] = list(det.get("physical_evidence", [])) + [
                        "Continuous sweep trajectory routed away from pulsed-envelope extraction"
                    ]
                    # The legacy detector may have already serialized its
                    # candidate as RADAR_PULSE.  Preserve the protocol label,
                    # but correct the physical family before validation so
                    # downstream gates reason over the measured waveform
                    # model rather than the catalog's historical family name.
                    for cand_key in ("candidate_hypotheses", "ranked_candidates"):
                        for cand in det.get(cand_key, []) or []:
                            if isinstance(cand, dict):
                                cand["signal_family"] = "RADAR_FMCW"
                    winning = det.get("winning_hypothesis")
                    if isinstance(winning, dict):
                        winning["signal_family"] = "RADAR_FMCW"
                det["parameters_origin"] = "Layer A BlindParameterVector"
                det["blind_parameters"] = params.to_dict()
                det["modulation_inference"] = mod_res.to_dict()
                return det
        except Exception:
            pass

    # Pure parameter matching against catalog
    meas_spacing = params.tone_spacing_hz
    meas_baud = params.symbol_rate_consensus_hz

    best_match = None
    best_score = 0.0

    for proto in PROTOCOL_INVARIANT_CATALOG:
        if proto.get("modulation_family") != mod_fam and not (mod_fam in ["2-FSK", "4-FSK"] and proto.get("modulation_family") == "2-FSK"):
            continue

        score = 0.0

        # Spacing match
        exp_sp = proto.get("expected_spacing_hz")
        if exp_sp is not None and meas_spacing is not None:
            tol_sp = proto.get("spacing_tolerance_hz", 20.0)
            if abs(meas_spacing - exp_sp) <= tol_sp:
                score += 0.50
            else:
                continue

        # Baud rate match
        exp_b = proto.get("expected_baud")
        if exp_b is not None and meas_baud is not None:
            tol_b = proto.get("baud_tolerance", 0.15 * exp_b)
            if abs(meas_baud - exp_b) <= tol_b:
                score += 0.50
            else:
                continue

        if score > best_score:
            best_score = score
            best_match = proto

    pipeline_map = {
        ModulationFamily.FSK_2: "fsk_detector",
        ModulationFamily.FSK_4: "fsk_detector",
        ModulationFamily.BPSK: "digital_psk_qam",
        ModulationFamily.QPSK: "digital_psk_qam",
        ModulationFamily.PSK_8: "digital_psk_qam",
        ModulationFamily.QAM_16: "digital_psk_qam",
        ModulationFamily.ANALOG_FM: "pulsed_radar" if mod_res.is_pulsed else "analog_voice",
        ModulationFamily.ANALOG_AM: "ook_morse" if "OOK" in mod_res.inferred_modulation else "analog_voice",
    }
    target_pipeline = pipeline_map.get(mod_res.modulation_family, "generic_fallback")
    if "M-FSK" in mod_res.inferred_modulation or "MFSK" in mod_res.inferred_modulation:
        target_pipeline = "mfsk_comb"
    elif "Radar" in mod_res.inferred_modulation or "Chirp" in mod_res.inferred_modulation:
        target_pipeline = "pulsed_radar"
    elif "Continuous Wave" in mod_res.inferred_modulation or "CW" in mod_res.inferred_modulation:
        target_pipeline = "continuous_wave"
    elif "OOK" in mod_res.inferred_modulation:
        target_pipeline = "ook_morse"

    # Structure-aware dispatch remains valid even when no catalog detector
    # candidate survives modulation compatibility checks.
    if (params.chirp_info or {}).get("is_continuous_sweep"):
        target_pipeline = "continuous_fmcw_radar"

    if best_match and best_score >= 0.50:
        proto_name = best_match["protocol_name"]
        class_id = best_match["class_id"]
        conf = float(np.clip(0.80 + 0.18 * best_score, 0.75, 0.98))
        ev = [
            f"Blind parameter vector matched {best_match['protocol_name']}",
            f"Observed spacing: {meas_spacing:.1f} Hz" if meas_spacing else "Continuous carrier",
            f"Observed baud consensus: {meas_baud:.1f} Baud" if meas_baud else "No discrete baud clock"
        ]
        extraction_pipe = PROTOCOL_TO_PIPELINE.get(class_id, target_pipeline)
    else:
        # Construct generic uncataloged protocol verdict
        if mod_res.modulation_family == ModulationFamily.FSK_2:
            shift = meas_spacing or (params.frequency_deviation_hz * 2.0 if params.frequency_deviation_hz else 170.0)
            proto_name = f"2-FSK (Frequency Shift Keying / {shift:.0f} Hz Shift)"
            class_id = "FSK_2FSK_GENERIC"
        elif mod_res.modulation_family == ModulationFamily.FSK_4:
            if "M-FSK" in mod_res.inferred_modulation:
                n_tones = params.num_frequency_states or (len(params.dominant_component_frequencies) if params.dominant_component_frequencies else 8)
                proto_name = f"{n_tones}-Tone MFSK (Multi-Frequency Shift Keying)"
                class_id = "MFSK_GENERIC"
            else:
                proto_name = "4-FSK (4-Level Frequency Shift Keying)"
                class_id = "FSK_4FSK_GENERIC"
        elif mod_res.modulation_family == ModulationFamily.BPSK:
            proto_name = "BPSK (Binary Phase Shift Keying)"
            class_id = "GENERIC_BPSK"
        elif mod_res.modulation_family == ModulationFamily.QPSK:
            proto_name = "QPSK (Quadrature Phase Shift Keying)"
            class_id = "GENERIC_QPSK"
        elif mod_res.modulation_family == ModulationFamily.PSK_8:
            proto_name = "8-PSK (8-Phase Shift Keying)"
            class_id = "GENERIC_8-PSK"
        elif mod_res.modulation_family == ModulationFamily.QAM_16:
            proto_name = "16-QAM (16-State Quadrature Amplitude Modulation)"
            class_id = "GENERIC_16-QAM"
        elif "Radar" in mod_res.inferred_modulation or "Chirp" in mod_res.inferred_modulation:
            proto_name = "Pulsed Radar / Chirp (Generic Intercept)"
            class_id = "RADAR_GENERIC_PULSED"
        elif "OOK" in mod_res.inferred_modulation:
            proto_name = "CW / Morse Code (On-Off Keying / OOK)"
            class_id = "CW_MORSE_OOK"
        elif "Continuous Wave" in mod_res.inferred_modulation:
            proto_name = "Continuous Wave (Unmodulated Carrier)"
            class_id = "CONTINUOUS_WAVE_UNMOD"
        elif mod_res.modulation_family == ModulationFamily.ANALOG_FM:
            proto_name = "Analog FM / Voice (Narrowband FM)"
            class_id = "ANALOG_NFM_VOICE"
            extraction_pipe = "analog_voice"
        elif mod_res.modulation_family == ModulationFamily.ANALOG_AM:
            proto_name = "Analog AM (Amplitude Modulation)"
            class_id = "ANALOG_AM_VOICE"
            extraction_pipe = "analog_voice"
        else:
            proto_name = f"Generic {mod_res.inferred_modulation}"
            class_id = f"GENERIC_{mod_fam.replace('-', '_')}"

        conf = mod_res.confidence
        ev = list(mod_res.rationales)
        extraction_pipe = target_pipeline

    ret = {
        "signal_class_id": class_id,
        "protocol_name": proto_name,
        "modulation_family": mod_fam,
        "confidence": conf,
        "extraction_pipeline": extraction_pipe,
        "physical_evidence": ev,
        "parameters_origin": "Layer A BlindParameterVector",
        "blind_parameters": params.to_dict(),
        "modulation_inference": mod_res.to_dict()
    }

    if mod_res.modulation_family == ModulationFamily.FSK_2:
        if len(params.state_frequencies_hz) == 2:
            ret["mark_freq_hz"] = float(params.state_frequencies_hz[0])
            ret["space_freq_hz"] = float(params.state_frequencies_hz[1])
            ret["fsk_shift_hz"] = abs(ret["space_freq_hz"] - ret["mark_freq_hz"])
        elif params.tone_spacing_hz:
            c_f = params.center_frequency_hz or params.peak_frequency_hz
            ret["mark_freq_hz"] = c_f - params.tone_spacing_hz / 2.0
            ret["space_freq_hz"] = c_f + params.tone_spacing_hz / 2.0
            ret["fsk_shift_hz"] = params.tone_spacing_hz

    return ret
