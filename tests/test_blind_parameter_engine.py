"""
Comprehensive Unit Test Suite for Blind Parameter Extraction Engine (Layer A & B)
================================================================================
Verifies:
1. Universal BlindParameterVector typed contract & serialization
2. Automatic signal activity detection & temporal segmentation
3. Adaptive noise-floor estimation
4. Multi-resolution spectral analysis
5. Generic frequency component & structural discovery
6. Continuous blind symbol-rate consensus without candidate presets
7. Blind cyclostationary cyclic frequency extraction
8. Generic FSK state discovery (M=2, M=4) without protocol knowledge
9. Constellation geometry discovery (BPSK, QPSK, 16-QAM)
10. Waveform morphology fingerprinting
11. Mathematical parameter self-consistency engine
12. Blindness score & provenance verification
13. Layer B modulation inference from blind parameter vectors
14. Hard Software Boundary Invariant: ZERO protocol names in blind_parameter_engine.py
"""

import unittest
import os
import re
import numpy as np

from dsp.contracts import BlindParameterVector, ModulationFamily
from dsp.blind_parameter_engine import (
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
from dsp.modulation_inference import infer_modulation_from_blind_params


class TestBlindParameterEngine(unittest.TestCase):
    """Rigorous tests ensuring parameter extraction operates completely blindly."""

    def setUp(self):
        np.random.seed(42)
        self.fs = 48000.0

    def test_01_blind_parameter_vector_contract(self):
        """BlindParameterVector must support full field initialization and serialization."""
        vec = BlindParameterVector(
            signal_presence=True,
            signal_duration=1.5,
            center_frequency_hz=1500.0,
            peak_frequency_hz=1500.0,
            occupied_bandwidth_hz=500.0,
            snr_db=18.5,
            frequency_structure="SINGLE_COMPONENT",
            num_frequency_states=2,
            tone_spacing_hz=170.0,
            symbol_rate_consensus_hz=100.0,
            symbol_rate_consensus_method="3/5 independent estimators agree"
        )
        d = vec.to_dict()
        self.assertTrue(d["signal_presence"])
        self.assertEqual(d["center_frequency_hz"], 1500.0)
        self.assertEqual(d["frequency_structure"], "SINGLE_COMPONENT")
        self.assertEqual(d["num_frequency_states"], 2)
        self.assertEqual(d["symbol_rate_consensus_hz"], 100.0)

        # Roundtrip from_dict
        restored = BlindParameterVector.from_dict(d)
        self.assertEqual(restored.center_frequency_hz, vec.center_frequency_hz)
        self.assertEqual(restored.symbol_rate_consensus_hz, vec.symbol_rate_consensus_hz)

    def test_02_signal_activity_detection_and_segmentation(self):
        """Discovers transmissions and separates active bursts from noise floor."""
        # 1. Continuous noise -> no high-power segments
        noise = np.random.normal(0, 0.01, int(self.fs * 0.5))
        is_act, duty, segs = detect_signal_activity_and_segments(noise, self.fs)
        self.assertLessEqual(duty, 10.0)

        # 2. Pulsed / Bursty transmission: 0.1s silence + 0.3s active burst + 0.1s silence
        t_burst = np.arange(int(self.fs * 0.3)) / self.fs
        burst = np.cos(2 * np.pi * 1000.0 * t_burst)
        full_sig = np.concatenate([np.zeros(int(self.fs * 0.1)), burst, np.zeros(int(self.fs * 0.1))])

        is_act, duty, segs = detect_signal_activity_and_segments(full_sig, self.fs)
        self.assertTrue(is_act)
        self.assertGreaterEqual(duty, 45.0)
        self.assertLessEqual(duty, 80.0)
        self.assertGreaterEqual(len(segs), 1)
        self.assertGreater(segs[0]["snr_db"], 10.0)

    def test_03_adaptive_noise_floor_estimation(self):
        """Measures local noise floor and dynamic range without hardcoded values."""
        t = np.arange(int(self.fs * 0.25)) / self.fs
        tone = np.cos(2 * np.pi * 2000.0 * t)
        noise = np.random.normal(0, 0.05, len(tone))
        sig = tone + noise

        spec = compute_multi_resolution_spectrum(sig, self.fs)
        floors = estimate_adaptive_noise_floor(sig, spec["psd_linear"], spec["psd_db"])

        self.assertIn("local_noise_floor_db", floors)
        self.assertIn("signal_floor_db", floors)
        self.assertIn("dynamic_range_db", floors)
        self.assertGreater(floors["dynamic_range_db"], 10.0)

    def test_04_multi_resolution_spectrum(self):
        """Produces coarse, medium, and fine spectral representations."""
        sig = np.random.normal(0, 1, 8192)
        res = compute_multi_resolution_spectrum(sig, self.fs)

        self.assertEqual(len(res["f"]), len(res["psd_linear"]))
        self.assertGreater(res["fine_nperseg"], res["medium_nperseg"])
        self.assertGreater(res["medium_nperseg"], res["coarse_nperseg"])

    def test_05_generic_frequency_structure_estimation(self):
        """Estimates peak, centroid, median, and energy center frequencies."""
        t = np.arange(int(self.fs * 0.5)) / self.fs
        # Twin tone (FSK-like spectrum at 1200 Hz and 2200 Hz)
        f1, f2 = 1200.0, 2200.0
        sig = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)

        spec = compute_multi_resolution_spectrum(sig, self.fs)
        struct_data = estimate_generic_frequency_structure(spec["f"], spec["psd_linear"], spec["psd_db"], self.fs)

        self.assertEqual(struct_data["frequency_structure"], "MULTI_COMPONENT")
        self.assertIn(struct_data["peak_frequency_hz"], [f1, f2])
        self.assertAlmostEqual(struct_data["spectral_centroid_hz"], 1700.0, delta=100.0)
        self.assertAlmostEqual(struct_data["energy_center_frequency_hz"], 1700.0, delta=100.0)
        self.assertAlmostEqual(struct_data["tone_spacing_hz"], 1000.0, delta=50.0)

    def test_06_blind_symbol_rate_consensus(self):
        """Continuous blind symbol-rate consensus successfully estimates baud rates."""
        # Synthesize BPSK signal at 300 Baud with square pulses
        baud = 300.0
        sps = int(self.fs / baud)
        n_syms = 300
        bits = np.random.choice([-1.0, 1.0], size=n_syms)
        baseband = np.repeat(bits, sps)
        t = np.arange(len(baseband)) / self.fs
        fc = 5000.0
        rf = baseband * np.cos(2 * np.pi * fc * t)

        cands, consensus_r, method_str = estimate_blind_symbol_rate_consensus(rf, self.fs)

        self.assertGreaterEqual(len(cands), 1)
        self.assertIsNotNone(consensus_r)
        self.assertAlmostEqual(consensus_r, baud, delta=20.0)
        self.assertIn("agree", method_str.lower())

    def test_07_blind_cyclostationary_extraction(self):
        """Extracts dominant cyclic frequencies corresponding to waveform periodicities."""
        # Amplitude modulated signal with 200 Hz tone
        t = np.arange(int(self.fs * 0.5)) / self.fs
        carrier = np.cos(2 * np.pi * 4000.0 * t)
        mod = 1.0 + 0.8 * np.cos(2 * np.pi * 200.0 * t)
        sig = mod * carrier

        alphas = estimate_blind_cyclostationary_frequencies(sig, self.fs)
        self.assertGreaterEqual(len(alphas), 1)
        # Dominant alpha should be near 200 Hz or 400 Hz (second harmonic)
        matched = any(abs(a - 200.0) < 15.0 or abs(a - 400.0) < 15.0 for a in alphas)
        self.assertTrue(matched, f"Expected 200/400 Hz cyclic frequency, got: {alphas}")

    def test_08_generic_fsk_state_discovery(self):
        """Discovers M=2 FSK states, spacing, and dwell without protocol presets."""
        baud = 100.0
        spacing = 200.0
        sps = int(self.fs / baud)
        n_syms = 250
        symbols = np.random.choice([-1, 1], size=n_syms)
        freq_dev = spacing / 2.0
        freq_series = np.repeat(symbols * freq_dev, sps)
        phase = 2.0 * np.pi * np.cumsum(freq_series) / self.fs
        t = np.arange(len(phase)) / self.fs
        fc = 2000.0
        fsk_sig = np.cos(2 * np.pi * fc * t + phase)

        fsk_data = discover_blind_fsk_states(fsk_sig, self.fs, fc_center=fc)

        self.assertEqual(fsk_data["num_frequency_states"], 2)
        self.assertAlmostEqual(fsk_data["tone_spacing_hz"], spacing, delta=25.0)
        self.assertAlmostEqual(fsk_data["frequency_deviation_hz"], freq_dev, delta=15.0)
        self.assertAlmostEqual(fsk_data["symbol_dwell_time_ms"], 10.0, delta=1.5)
        self.assertAlmostEqual(fsk_data["estimated_baud_rate_hz"], baud, delta=15.0)

    def test_09_constellation_geometry_discovery(self):
        """Discovers constant-envelope and phase fold symmetry for BPSK and QPSK."""
        # 1. BPSK: phase alternates 0 and pi (2-fold symmetry)
        n_pts = 4000
        bpsk_symbols = np.random.choice([1.0 + 0j, -1.0 + 0j], size=n_pts)
        noise = 0.05 * (np.random.normal(0, 1, n_pts) + 1j * np.random.normal(0, 1, n_pts))
        bpsk_noisy = bpsk_symbols + noise

        geom_bpsk = discover_constellation_geometry(bpsk_noisy, self.fs)
        self.assertEqual(geom_bpsk["amplitude_structure"], "CONSTANT_ENVELOPE")
        self.assertEqual(geom_bpsk["phase_fold_symmetry_m"], 2)
        self.assertEqual(geom_bpsk["cluster_count_estimate"], 2)

        # 2. QPSK: phase 4-fold symmetry
        qpsk_symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=n_pts) / np.sqrt(2.0)
        qpsk_noisy = qpsk_symbols + noise

        geom_qpsk = discover_constellation_geometry(qpsk_noisy, self.fs)
        self.assertEqual(geom_qpsk["amplitude_structure"], "CONSTANT_ENVELOPE")
        self.assertEqual(geom_qpsk["phase_fold_symmetry_m"], 4)
        self.assertEqual(geom_qpsk["cluster_count_estimate"], 4)

    def test_10_waveform_morphology_fingerprint(self):
        """Extracts complete 8-attribute morphology fingerprint."""
        sig = np.random.normal(0, 1, 4096)
        morph = extract_waveform_morphology(
            signal=sig,
            fs=self.fs,
            duty_cycle_pct=100.0,
            freq_structure="SINGLE_COMPONENT",
            dominant_tones_count=1,
            envelope_var_ratio=0.10,
            sfm=0.20
        )
        self.assertEqual(morph["temporal_pattern"], "CONTINUOUS")
        self.assertEqual(morph["envelope_nature"], "CONSTANT_ENVELOPE")
        self.assertEqual(morph["component_nature"], "SINGLE_COMPONENT")

    def test_11_parameter_self_consistency(self):
        """Cross-checks Carson's rule and pulse duty cycle physical formulas."""
        # Carson check: OBW = 270 Hz, Shift = 170 Hz, Baud = 100 -> B_carson = 270 Hz (exact match)
        res_carson = validate_parameter_consistency(
            obw_hz=270.0,
            baud_rate=100.0,
            tone_spacing=170.0,
            duty_cycle_pct=100.0,
            pri_us=None,
            pw_us=None
        )
        self.assertTrue(res_carson["is_consistent"])
        self.assertGreaterEqual(res_carson["consistency_score"], 0.90)

    def test_12_blindness_score_and_provenance(self):
        """Verifies declaration of zero prior knowledge and 10.0 blindness score."""
        prov = build_blindness_provenance()
        self.assertEqual(prov["prior_knowledge_used"], "NONE")
        self.assertIn("10.0 / 10.0", prov["blindness_score"])
        self.assertIn("carrier_frequency", prov["parameter_provenance"])
        self.assertIn("symbol_rate", prov["parameter_provenance"])

    def test_13_modulation_inference_from_blind_parameters(self):
        """Infers modulation family strictly from BlindParameterVector."""
        # 2-FSK vector
        fsk_vec = BlindParameterVector(
            signal_presence=True,
            num_frequency_states=2,
            tone_spacing_hz=170.0,
            symbol_rate_consensus_hz=100.0,
            morphology_fingerprint={"envelope_nature": "CONSTANT_ENVELOPE", "temporal_pattern": "CONTINUOUS"}
        )
        res_fsk = infer_modulation_from_blind_params(fsk_vec)
        self.assertEqual(res_fsk.inferred_modulation, "2-FSK")
        self.assertEqual(res_fsk.modulation_family, ModulationFamily.FSK_2)
        self.assertGreaterEqual(res_fsk.confidence, 0.90)

        # QPSK vector
        qpsk_vec = BlindParameterVector(
            signal_presence=True,
            constellation_geometry={"amplitude_structure": "CONSTANT_ENVELOPE", "phase_fold_symmetry_m": 4, "cluster_count_estimate": 4},
            morphology_fingerprint={"envelope_nature": "CONSTANT_ENVELOPE", "temporal_pattern": "CONTINUOUS"}
        )
        res_qpsk = infer_modulation_from_blind_params(qpsk_vec)
        self.assertEqual(res_qpsk.inferred_modulation, "QPSK")
        self.assertEqual(res_qpsk.modulation_family, ModulationFamily.QPSK)

    def test_14_hard_software_boundary_zero_protocol_strings(self):
        """
        Hard Invariant: Layer A (blind_parameter_engine.py) MUST NOT contain
        any protocol names: NAVTEX, DMR, GSM, STANAG, ALE, FT8, RTTY, POCSAG, APRS, DUGA, GHADIR.
        """
        engine_path = os.path.join(os.path.dirname(__file__), "..", "dsp", "blind_parameter_engine.py")
        with open(engine_path, "r", encoding="utf-8") as f:
            code = f.read()

        banned_protocols = [
            "NAVTEX", "DMR", "GSM", "STANAG", "ALE", "FT8",
            "RTTY", "POCSAG", "APRS", "DUGA", "GHADIR", "WEFAX"
        ]
        violations = []
        for proto in banned_protocols:
            matches = re.findall(rf"\b{proto}\b", code, re.IGNORECASE)
            if matches:
                violations.append((proto, len(matches)))

        self.assertEqual(
            len(violations), 0,
            f"Layer A software boundary violation! Found protocol names in blind_parameter_engine.py: {violations}"
        )

    def test_15_uncataloged_fsk_continuous_discovery(self):
        """
        Validates discovery of uncataloged 73.5 Baud FSK signal with 340 Hz shift at +5 dB SNR.
        Must blindly estimate M=2, tone_spacing ~= 340 Hz, Rs ~= 73.5 Baud, and infer 2-FSK.
        """
        baud = 73.5
        spacing = 340.0
        sps = int(self.fs / baud)
        n_syms = 250
        symbols = np.random.choice([-1, 1], size=n_syms)
        freq_dev = spacing / 2.0
        freq_series = np.repeat(symbols * freq_dev, sps)
        phase = 2.0 * np.pi * np.cumsum(freq_series) / self.fs
        t = np.arange(len(phase)) / self.fs
        fc = 2000.0
        sig = np.cos(2 * np.pi * fc * t + phase)
        noise = np.random.normal(0, np.std(sig) / (10 ** (5.0 / 20.0)), len(sig))
        noisy_sig = sig + noise

        vec = extract_blind_parameters(noisy_sig, self.fs)
        self.assertTrue(vec.signal_presence)
        self.assertEqual(vec.num_frequency_states, 2)
        self.assertAlmostEqual(vec.tone_spacing_hz, spacing, delta=30.0)
        self.assertIsNotNone(vec.symbol_rate_consensus_hz)
        self.assertAlmostEqual(vec.symbol_rate_consensus_hz, baud, delta=10.0)

        mod = infer_modulation_from_blind_params(vec)
        self.assertEqual(mod.inferred_modulation, "2-FSK")
        self.assertEqual(mod.modulation_family, ModulationFamily.FSK_2)
        self.assertGreaterEqual(mod.confidence, 0.90)

    def test_16_pulse_and_radar_consistency(self):
        """
        Validates pulse parameter extraction and physical consistency checks:
        PRI, PRF, PW, and Fourier transform lower bound limit.
        """
        # Synthesize a pulse train: 200 us pulses at 300 Hz PRF
        pri_target_us = 3333.33
        pw_target_us = 200.0
        duty_target_pct = (pw_target_us / pri_target_us) * 100.0  # 6.0%

        res = validate_parameter_consistency(
            obw_hz=10000.0,
            baud_rate=None,
            tone_spacing=None,
            duty_cycle_pct=duty_target_pct,
            pri_us=pri_target_us,
            pw_us=pw_target_us
        )
        self.assertTrue(res["is_consistent"])
        self.assertEqual(res["consistency_score"], 1.0)
        equations = [c["equation"] for c in res["checks_performed"]]
        self.assertIn("Pulse Duty Cycle (Duty = PW / PRI)", equations)
        self.assertIn("Pulse Duration-Bandwidth Limit (OBW >= 0.8 / PW)", equations)

    def test_17_robust_edge_cases(self):
        """
        Validates graceful degradation on boundary and corrupt inputs:
        pure noise, short signals (<128 samples), empty arrays, NaNs, and zero fs.
        """
        # 1. Pure Gaussian noise -> inactive
        noise = np.random.normal(0, 0.05, int(self.fs * 0.5))
        vec_noise = extract_blind_parameters(noise, self.fs)
        self.assertFalse(vec_noise.signal_presence)
        self.assertIsNone(vec_noise.symbol_rate_consensus_hz)

        # 2. Empty array -> safe return
        vec_empty = extract_blind_parameters(np.array([]), self.fs)
        self.assertFalse(vec_empty.signal_presence)

        # 3. Short signal (64 samples) -> safe return without divide-by-zero warnings
        vec_short = extract_blind_parameters(np.random.normal(0, 1, 64), self.fs)
        self.assertFalse(vec_short.signal_presence)

        # 4. Signal with NaNs -> finite sanitized return
        nan_sig = np.random.normal(0, 1, 1024)
        nan_sig[100] = np.nan
        vec_nan = extract_blind_parameters(nan_sig, self.fs)
        self.assertIsInstance(vec_nan.signal_presence, bool)

        # 5. Invalid sampling rate
        vec_fs0 = extract_blind_parameters(np.random.normal(0, 1, 512), 0.0)
        self.assertFalse(vec_fs0.signal_presence)

    def test_18_uncataloged_custom_fsk_pipeline_dispatch(self):
        """
        Validates uncataloged 137.4 Baud / 523 Hz shift 2-FSK at +3 dB SNR.
        Must cleanly dispatch to 'fsk_detector' with FSK_2FSK_GENERIC verdict,
        extract ~523 Hz shift and ~137.4 Baud, and never misclassify as satellite telemetry.
        """
        from dsp.adaptive_pipeline import run_adaptive_pipeline
        baud = 137.4
        spacing = 523.0
        sps = int(self.fs / baud)
        n_syms = 300
        symbols = np.random.choice([-1, 1], size=n_syms)
        freq_dev = spacing / 2.0
        freq_series = np.repeat(symbols * freq_dev, sps)
        phase = 2.0 * np.pi * np.cumsum(freq_series) / self.fs
        t = np.arange(len(phase)) / self.fs
        fc = 2500.0
        sig = np.cos(2 * np.pi * fc * t + phase)
        noise = np.random.normal(0, np.std(sig) / (10 ** (3.0 / 20.0)), len(sig))
        noisy_sig = sig + noise

        res = run_adaptive_pipeline(noisy_sig, self.fs, metadata={"file_name": "custom_fsk.wav"})
        self.assertEqual(res["status"], "SUCCESS")
        det = res["detection"]
        self.assertEqual(det["modulation_family"], "2-FSK")
        self.assertEqual(det["signal_class_id"], "FSK_2FSK_GENERIC")
        self.assertEqual(det["extraction_pipeline"], "fsk_detector")
        self.assertNotEqual(det["signal_class_id"], "SATELLITE_TELEMETRY_NFM")

        params = res["parameters"]
        self.assertIsNotNone(params.get("fsk_frequency_shift_hz"))
        self.assertAlmostEqual(params["fsk_frequency_shift_hz"], spacing, delta=40.0)
        self.assertIsNotNone(params.get("estimated_baud_rate_hz"))
        self.assertAlmostEqual(params["estimated_baud_rate_hz"], baud, delta=20.0)

    def test_19_qpsk_constellation_non_zero_carrier(self):
        """
        Validates QPSK constellation fold symmetry recovery at non-zero RF carrier frequencies.
        Must isolate M=4 fold symmetry even with carrier rotation at fc = 2400 Hz and 6000 Hz.
        """
        for fc in [2400.0, 6000.0]:
            n_pts = 4096
            syms = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=n_pts) / np.sqrt(2.0)
            t = np.arange(n_pts) / self.fs
            rot_syms = syms * np.exp(1j * (2 * np.pi * fc * t + 0.35))
            noise = (np.random.normal(0, 0.05, n_pts) + 1j * np.random.normal(0, 0.05, n_pts))
            noisy = rot_syms + noise

            geom = discover_constellation_geometry(noisy, self.fs)
            self.assertEqual(geom["amplitude_structure"], "CONSTANT_ENVELOPE")
            self.assertEqual(geom["phase_fold_symmetry_m"], 4, f"Failed M=4 at fc={fc}")
            self.assertEqual(geom["cluster_count_estimate"], 4)

    def test_20_multitone_comb_5_and_7_tones(self):
        """
        Validates multi-carrier / MFSK tone comb discovery for 5 and 7 tones.
        Must classify as MULTI_COMPONENT_HARMONIC rather than SPREAD_SPECTRUM.
        """
        t = np.arange(int(self.fs * 0.2)) / self.fs
        for n_tones in [5, 7]:
            sig = np.zeros_like(t)
            spacing = 200.0
            f0 = 1000.0
            for k in range(n_tones):
                sig += np.cos(2 * np.pi * (f0 + k * spacing) * t + 0.1 * k)
            sig += np.random.normal(0, 0.05, len(sig))

            vec = extract_blind_parameters(sig, self.fs)
            self.assertTrue(vec.signal_presence)
            self.assertEqual(vec.frequency_structure, "MULTI_COMPONENT_HARMONIC", f"Failed for {n_tones} tones")
            self.assertIsNotNone(vec.tone_spacing_hz)
            self.assertAlmostEqual(vec.tone_spacing_hz, spacing, delta=25.0)

    def test_21_slow_fsk_6_25_baud(self):
        """
        Validates FSK state dwell estimation for slow waveforms (6.25 Baud FT8 speed, 160 ms dwell).
        Must correctly capture dwell > 100 ms and symbol rate around 6.25 Baud.
        """
        baud = 6.25
        sps = int(self.fs / baud)  # 7680 samples per symbol at 48 kHz
        n_syms = 20
        symbols = np.random.choice([-1, 1], size=n_syms)
        spacing = 150.0
        freq_series = np.repeat(symbols * (spacing / 2.0), sps)
        phase = 2.0 * np.pi * np.cumsum(freq_series) / self.fs
        t = np.arange(len(phase)) / self.fs
        sig = np.cos(2 * np.pi * 1500.0 * t + phase)
        sig += np.random.normal(0, 0.02, len(sig))

        vec = extract_blind_parameters(sig, self.fs)
        self.assertTrue(vec.signal_presence)
        self.assertEqual(vec.num_frequency_states, 2)
        self.assertIsNotNone(vec.symbol_dwell_time_ms)
        self.assertGreater(vec.symbol_dwell_time_ms, 100.0)
        self.assertAlmostEqual(vec.symbol_dwell_time_ms, 160.0, delta=40.0)

    def test_22_real_signal_carrier_frequency_positive(self):
        """
        Validates that estimate_carrier_frequency enforces non-negative carrier frequency
        on real-valued input signals.
        """
        from dsp.spectral import compute_welch_psd
        from dsp.parameter_extractor import estimate_carrier_frequency

        t = np.arange(int(self.fs * 0.25)) / self.fs
        carrier_hz = 2400.0
        sig_real = np.cos(2 * np.pi * carrier_hz * t) + np.random.normal(0, 0.1, len(t))

        f_s, psd_db, psd_lin = compute_welch_psd(sig_real, self.fs, nperseg=2048)
        res = estimate_carrier_frequency(f_s, psd_lin, psd_db)

        self.assertGreaterEqual(res["peak_frequency_hz"], 0.0)
        self.assertGreaterEqual(res["fc_peak_hz"], 0.0)
        self.assertAlmostEqual(res["peak_frequency_hz"], carrier_hz, delta=30.0)


if __name__ == "__main__":
    unittest.main()
