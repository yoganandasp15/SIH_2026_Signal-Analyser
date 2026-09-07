"""
Automated Test Suite for NTRO Signal Analysis Engine
===================================================
Tests core DSP mathematical integrity, binary loader formats,
parameter estimators (fc, BW, SNR, Baud), Higher-Order Cumulants,
Segmented Pulsed SNR, Intra-Pulse FMOP Chirp Detection, and Real-World Intercepts.
"""

import unittest
import os
import tempfile
import numpy as np

from dsp.loaders import parse_iq_binary, load_wav_signal, load_signal_file
from dsp.preprocessor import remove_dc_offset, normalize_signal_power, compute_signal_stats
from dsp.spectral import compute_welch_psd, compute_spectrogram
from dsp.parameter_extractor import (
    estimate_carrier_frequency,
    estimate_occupied_bandwidth,
    estimate_snr_m2m4,
    estimate_band_integrated_snr,
    estimate_symbol_baud_rate,
    extract_all_parameters
)
from dsp.modulation_classifier import (
    compute_higher_order_cumulants,
    classify_modulation_cumulants
)
from dsp.pulse_analyzer import (
    analyze_pulse_train,
    compute_segmented_pulsed_snr,
    analyze_intra_pulse_modulation,
    extract_autocorr_pri_prf,
    match_standard_frame
)
from utils.exporter import export_results_to_json, export_results_to_csv
from utils.synthetic_generator import (
    generate_synthetic_signal,
    add_awgn_noise,
    save_synthetic_iq,
    save_synthetic_wav
)


class TestNTROSignalAnalysisEngine(unittest.TestCase):

    def setUp(self):
        self.fs = 1_000_000.0  # 1 MSPS
        self.num_samples = 100_000
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for file in files:
                try:
                    os.remove(os.path.join(root, file))
                except Exception:
                    pass
            for d in dirs:
                try:
                    os.rmdir(os.path.join(root, d))
                except Exception:
                    pass
        try:
            os.rmdir(self.test_dir)
        except Exception:
            pass

    # -------------------------------------------------------------
    # 1. LOADER TESTS (cf32, cs16, cu8, WAV)
    # -------------------------------------------------------------
    def test_binary_iq_loader_complex64(self):
        sig, _, _ = generate_synthetic_signal("QPSK", fs=self.fs, num_samples=10_000)
        file_path = os.path.join(self.test_dir, "test_cf32.iq")
        save_synthetic_iq(sig, file_path, format_type="complex64")

        loaded_sig, fs_eff, meta = load_signal_file(file_path, sample_rate=self.fs, format_type="complex64")
        self.assertEqual(len(loaded_sig), 10_000)
        self.assertEqual(fs_eff, self.fs)
        self.assertAlmostEqual(float(np.real(loaded_sig[0])), float(np.real(sig[0])), places=4)

    def test_binary_iq_loader_int16(self):
        sig, _, _ = generate_synthetic_signal("BPSK", fs=self.fs, num_samples=10_000)
        file_path = os.path.join(self.test_dir, "test_cs16.iq")
        save_synthetic_iq(sig, file_path, format_type="int16")

        loaded_sig, fs_eff, meta = load_signal_file(file_path, sample_rate=self.fs, format_type="int16")
        self.assertEqual(len(loaded_sig), 10_000)
        self.assertLessEqual(np.max(np.abs(loaded_sig)), 1.05)

    def test_binary_iq_loader_uint8(self):
        sig, _, _ = generate_synthetic_signal("CW", fs=self.fs, num_samples=5_000)
        file_path = os.path.join(self.test_dir, "test_cu8.bin")
        save_synthetic_iq(sig, file_path, format_type="uint8")

        loaded_sig, fs_eff, meta = load_signal_file(file_path, sample_rate=self.fs, format_type="uint8")
        self.assertEqual(len(loaded_sig), 5_000)

    def test_wav_loader_stereo_and_mono(self):
        sig, _, _ = generate_synthetic_signal("FM", fs=self.fs, num_samples=10_000)
        file_path = os.path.join(self.test_dir, "test_stereo.wav")
        save_synthetic_wav(sig, 48000.0, file_path)

        loaded_sig, fs_eff, meta = load_signal_file(file_path)
        self.assertEqual(fs_eff, 48000.0)
        self.assertEqual(len(loaded_sig), 10_000)
        self.assertTrue(np.iscomplexobj(loaded_sig))

    # -------------------------------------------------------------
    # 2. PREPROCESSING & SPECTRAL TESTS
    # -------------------------------------------------------------
    def test_dc_offset_removal_and_normalization(self):
        sig = np.ones(5000, dtype=np.complex64) * (2.0 + 3.0j) + np.random.normal(0, 0.1, 5000)
        dc_free = remove_dc_offset(sig)
        self.assertAlmostEqual(float(np.abs(np.mean(dc_free))), 0.0, places=5)

        norm_sig, pwr = normalize_signal_power(dc_free)
        self.assertAlmostEqual(float(np.mean(np.abs(norm_sig) ** 2)), 1.0, places=5)

    def test_carrier_frequency_estimation(self):
        target_fc = 120_000.0
        sig, _, _ = generate_synthetic_signal("CW", fs=self.fs, fc=target_fc, snr_db=20.0, num_samples=50_000)
        f_s, psd_db, psd_lin = compute_welch_psd(sig, self.fs, nperseg=2048)

        carrier_res = estimate_carrier_frequency(f_s, psd_lin, psd_db)
        est_fc = carrier_res["fc_peak_hz"]
        freq_resolution = self.fs / 2048.0
        self.assertLess(abs(est_fc - target_fc), freq_resolution * 2.0)

    # -------------------------------------------------------------
    # 3. SNR & RADAR PULSE TESTS
    # -------------------------------------------------------------
    def test_snr_m2m4_accuracy(self):
        target_snrs = [10.0, 15.0, 20.0]
        for target_snr in target_snrs:
            sig, _, _ = generate_synthetic_signal("QPSK", fs=self.fs, fc=0.0, snr_db=target_snr, num_samples=100_000)
            snr_res = estimate_snr_m2m4(sig)
            est_snr = snr_res["snr_db"]
            self.assertLess(abs(est_snr - target_snr), 1.5)

    def test_radar_pulse_and_segmented_snr(self):
        target_pw_us = 25.0
        target_pri_us = 120.0
        sig, _, _ = generate_synthetic_signal(
            "PULSED_RADAR",
            fs=self.fs,
            snr_db=25.0,
            num_samples=100_000,
            pulse_width_us=target_pw_us,
            pri_us=target_pri_us
        )
        pulse_res = analyze_pulse_train(sig, self.fs)

        self.assertTrue(pulse_res["is_pulsed"])
        self.assertTrue(pulse_res["is_radar"])
        self.assertGreater(pulse_res["num_pulses"], 10)
        self.assertAlmostEqual(pulse_res["mean_pulse_width_us"], target_pw_us, delta=2.0)
        self.assertAlmostEqual(pulse_res["mean_pri_us"], target_pri_us, delta=3.0)
        # Pulsed SNR should be positive and close to injected 25 dB (within 3 dB)
        self.assertGreater(pulse_res["pulsed_snr_db"], 20.0)

    def test_multi_rate_prf_clustering(self):
        sig, _, _ = generate_synthetic_signal(
            "PULSED_RADAR",
            fs=self.fs,
            snr_db=25.0,
            num_samples=100_000,
            pulse_width_us=25.0,
            pri_us=120.0
        )
        prf_info = extract_autocorr_pri_prf(sig, self.fs)
        self.assertGreater(prf_info["primary_prf_hz"], 100.0)
        self.assertAlmostEqual(prf_info["primary_pri_us"], 120.0, delta=10.0)

    # -------------------------------------------------------------
    # 4. REAL-WORLD GHADIR OTH RADAR BENCHMARK
    # -------------------------------------------------------------
    def test_ghadir_real_world_intercept(self):
        ghadir_path = "C:/Users/Yogi/Downloads/Ghadir.wav"
        if not os.path.exists(ghadir_path):
            ghadir_path = os.path.join(os.path.dirname(__file__), "..", "verified_samples", "Ghadir_Radar.wav")
        if not os.path.exists(ghadir_path):
            self.skipTest("Ghadir.wav not present")

        sig, fs, meta = load_signal_file(ghadir_path, max_samples=200_000)
        pulse_info = analyze_pulse_train(sig, fs)
        params = extract_all_parameters(sig, fs, pulse_info=pulse_info)
        mod_info = classify_modulation_cumulants(sig, fs, pulse_info=pulse_info)

        # 1. Must be recognized as Radar, NOT digital QAM
        self.assertTrue(pulse_info["is_radar"])
        self.assertTrue(mod_info.get("is_radar", False))
        self.assertIn("Pulsed", mod_info["modulation_type"])

        # 2. SNR must be positive (> 5 dB), NOT collapsed to -40 dB
        self.assertGreater(params["snr_db"], 5.0)

        # 3. Baud rate must be suppressed (None)
        self.assertIsNone(params["estimated_baud_rate_hz"])

        # 4. Multi-PRF / Dual-Rate detected
        self.assertGreater(pulse_info["num_pulses"], 100)

    def test_prc_oth_sw_real_world_intercept(self):
        oth_path = "C:/Users/Yogi/Downloads/PRC_OTH_SW.wav"
        if not os.path.exists(oth_path):
            oth_path = os.path.join(os.path.dirname(__file__), "..", "verified_samples", "OTH_SW_Radar.wav")
        if not os.path.exists(oth_path):
            self.skipTest("PRC_OTH_SW.wav not present")

        sig, fs, meta = load_signal_file(oth_path, max_samples=200_000)
        pulse_info = analyze_pulse_train(sig, fs)
        params = extract_all_parameters(sig, fs, pulse_info=pulse_info)
        mod_info = classify_modulation_cumulants(sig, fs, pulse_info=pulse_info)

        # 1. Must be recognized as Radar FMCW / Chirp
        self.assertTrue(pulse_info["is_radar"])
        self.assertIn("Pulsed FMOP", mod_info["modulation_type"])

        # 2. Fundamental PRF must match ~43 Hz (within 2 Hz)
        self.assertAlmostEqual(pulse_info["mean_prf_hz"], 43.2, delta=2.0)

        # 3. SNR must be positive (> 5 dB)
        self.assertGreater(params["snr_db"], 5.0)

        # 4. Baud rate must be suppressed (None)
        self.assertIsNone(params["estimated_baud_rate_hz"])

        # 5. Bandwidth must be non-zero
        self.assertGreater(params["bw_3db_hz"], 0.0)

    def test_vario_analog_voice_intercept(self):
        vario_path = "C:/Users/Yogi/Downloads/Sigid12-Jun-2019_15h54m18s_-_462.611.5_MHz,_NFM.wav"
        if not os.path.exists(vario_path):
            vario_path = os.path.join(os.path.dirname(__file__), "..", "verified_samples", "Vario_Voice_Tone.wav")
        if not os.path.exists(vario_path):
            self.skipTest("Vario NFM audio not present")

        sig, fs, meta = load_signal_file(vario_path, max_samples=200_000)
        pulse_info = analyze_pulse_train(sig, fs)
        mod_info = classify_modulation_cumulants(sig, fs, pulse_info=pulse_info)
        is_analog = mod_info.get("is_analog_audio", False)
        params = extract_all_parameters(sig, fs, pulse_info=pulse_info, is_analog_audio=is_analog)

        # 1. Must be recognized as Analog Voice / Tones
        self.assertTrue(is_analog)
        self.assertIn("Analog", mod_info["modulation_type"])

        # 2. Continuous transmission (not pulsed radar)
        self.assertFalse(pulse_info["is_pulsed"])

        # 3. SNR must be physical positive value (+15 to +35 dB), not 148 dB or -25 dB
        self.assertGreater(params["snr_db"], 10.0)
        self.assertLess(params["snr_db"], 40.0)

        # 4. Baud rate must be suppressed (None)
        self.assertIsNone(params["estimated_baud_rate_hz"])

    def test_tdma_burst_discrimination_vs_radar(self):
        # Synthetic TDMA burst (GSM-like burst: low R^2 non-linear chirp, discrete phase transitions)
        fs = 1_000_000.0
        # Deterministic GMSK-like bit transitions inside burst
        bit_pattern = np.array([1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1])
        bits = np.repeat(bit_pattern, 40)
        freq_dev = bits * 67700.0
        phase = 2 * np.pi * np.cumsum(freq_dev) / fs
        pulse_slice = np.exp(1j * phase)

        intra_mod = analyze_intra_pulse_modulation([pulse_slice], fs)
        # Must NOT be classified as FMOP Radar Chirp
        self.assertFalse(intra_mod["is_fmop_chirp"])
        self.assertTrue(intra_mod["is_tdma_comms"])
        self.assertIn("TDMA", intra_mod["intra_pulse_mod"])

    def test_standard_frame_matching(self):
        # 1. GSM Frame (~4615 us)
        gsm_match = match_standard_frame(4615.0, 216.7)
        self.assertIsNotNone(gsm_match)
        self.assertIn("GSM", gsm_match)

        # 2. Chinese OTH Radar Mode (~23260 us / 43 Hz)
        oth_match = match_standard_frame(23260.0, 43.2)
        self.assertIsNotNone(oth_match)
        self.assertIn("OTH-SW", oth_match)

        # 3. Ghadir Radar Mode (~3257 us / 307 Hz)
        ghadir_match = match_standard_frame(3257.0, 307.0)
        self.assertIsNotNone(ghadir_match)
        self.assertIn("Ghadir", ghadir_match)

    def test_band_integrated_snr_physics(self):
        # Verify that band-integrated SNR produces physically realistic numbers (+10 dB injected -> ~10 dB estimated)
        sig, _, _ = generate_synthetic_signal("QPSK", fs=self.fs, snr_db=12.0, num_samples=100_000)
        f_s, psd_db, psd_lin = compute_welch_psd(sig, self.fs, nperseg=2048)
        bw_params = estimate_occupied_bandwidth(f_s, psd_lin, psd_db)
        band_snr = estimate_band_integrated_snr(f_s, psd_lin, bw_params)

        self.assertGreater(band_snr, 5.0)
        self.assertLess(band_snr, 20.0)
        self.assertAlmostEqual(band_snr, 12.0, delta=4.0)

    def test_dmr_real_world_intercept(self):
        dmr_path = os.path.join(os.path.dirname(__file__), "..", "verified_samples", "DMR.wav")
        if not os.path.exists(dmr_path):
            self.skipTest("DMR sample not available")
        sig, fs, _ = load_signal_file(dmr_path, max_samples=500_000)
        pulse_info = analyze_pulse_train(sig, fs)
        mod_info = classify_modulation_cumulants(sig, fs, pulse_info=pulse_info)
        params = extract_all_parameters(sig, fs, pulse_info=pulse_info)
        self.assertIn("DMR", mod_info["modulation_type"])
        self.assertTrue(pulse_info["is_tdma"])
        self.assertGreater(params["snr_db"], 5.0)

    def test_ft8_real_world_intercept(self):
        ft8_path = os.path.join(os.path.dirname(__file__), "..", "verified_samples", "FT8.wav")
        if not os.path.exists(ft8_path):
            self.skipTest("FT8 sample not available")
        sig, fs, _ = load_signal_file(ft8_path, max_samples=200_000)
        pulse_info = analyze_pulse_train(sig, fs)
        mod_info = classify_modulation_cumulants(sig, fs, pulse_info=pulse_info)
        self.assertIn("FT8", mod_info["modulation_type"])
        self.assertFalse(pulse_info["is_pulsed"])

    def test_psk31_real_world_intercept(self):
        psk31_path = os.path.join(os.path.dirname(__file__), "..", "verified_samples", "PSK31.wav")
        if not os.path.exists(psk31_path):
            self.skipTest("PSK31 sample not available")
        sig, fs, _ = load_signal_file(psk31_path, max_samples=200_000)
        pulse_info = analyze_pulse_train(sig, fs)
        mod_info = classify_modulation_cumulants(sig, fs, pulse_info=pulse_info)
        params = extract_all_parameters(sig, fs, pulse_info=pulse_info)
        self.assertIn("PSK", mod_info["modulation_type"])
        self.assertAlmostEqual(params["estimated_baud_rate_hz"], 31.25, delta=1.5)

    def test_navtex_real_world_intercept(self):
        navtex_path = os.path.join(os.path.dirname(__file__), "..", "verified_samples", "NAVTEX.wav")
        if not os.path.exists(navtex_path):
            self.skipTest("NAVTEX sample not available")
        sig, fs, _ = load_signal_file(navtex_path, max_samples=200_000)
        pulse_info = analyze_pulse_train(sig, fs)
        mod_info = classify_modulation_cumulants(sig, fs, pulse_info=pulse_info)
        self.assertIn("2-FSK", mod_info["modulation_type"])

    def test_2g_ale_real_world_intercept(self):
        ale_path = os.path.join(os.path.dirname(__file__), "..", "verified_samples", "2G_ALE.wav")
        if not os.path.exists(ale_path):
            self.skipTest("2G ALE sample not available")
        sig, fs, _ = load_signal_file(ale_path, max_samples=250_000)
        pulse_info = analyze_pulse_train(sig, fs)
        mod_info = classify_modulation_cumulants(sig, fs, pulse_info=pulse_info)
        params = extract_all_parameters(sig, fs, pulse_info=pulse_info)

        # 1. Must be recognized as 8-Tone MFSK 2G ALE
        self.assertIn("8-Tone MFSK", mod_info["modulation_type"])
        self.assertIn("2G ALE", mod_info["modulation_type"])
        self.assertGreaterEqual(mod_info["confidence"], 0.95)

        # 2. Must not be classified as radar
        self.assertFalse(pulse_info.get("is_radar", False))
        self.assertEqual(pulse_info.get("signal_mode"), "Tactical HF Data / Handshake Protocol (2G ALE)")

        # 3. Baud rate must be recovered as 125.0 Baud
        self.assertAlmostEqual(params["estimated_baud_rate_hz"], 125.0, delta=1.0)

        # 4. Bandwidth and Centroid within HF audio band specs
        self.assertGreater(params["bw_99pct_hz"], 1800.0)
        self.assertLess(params["bw_99pct_hz"], 2500.0)
        self.assertGreater(params["fc_centroid_hz"], 1500.0)
        self.assertLess(params["fc_centroid_hz"], 1900.0)

    def test_autonomous_binary_probing(self):
        """Verify autonomous format prober identifies cf32, cs16, and cu8."""
        from dsp.loaders import probe_binary_format
        from utils.synthetic_generator import generate_synthetic_signal, save_synthetic_iq

        sig, _, _ = generate_synthetic_signal("QPSK", fs=1_000_000.0, num_samples=5000)
        f_cf32 = os.path.join(tempfile.gettempdir(), "test_probe_cf32.iq")
        f_cs16 = os.path.join(tempfile.gettempdir(), "test_probe_cs16.bin")
        f_cu8 = os.path.join(tempfile.gettempdir(), "test_probe_cu8.raw")

        try:
            save_synthetic_iq(sig, f_cf32, format_type="complex64")
            save_synthetic_iq(sig, f_cs16, format_type="int16")
            save_synthetic_iq(sig, f_cu8, format_type="uint8")

            fmt1, conf1, _ = probe_binary_format(f_cf32)
            fmt2, conf2, _ = probe_binary_format(f_cs16)
            fmt3, conf3, _ = probe_binary_format(f_cu8)

            self.assertEqual(fmt1, "complex64")
            self.assertEqual(fmt2, "int16")
            self.assertEqual(fmt3, "uint8")
            self.assertGreater(conf1, 0.80)
            self.assertGreater(conf2, 0.80)
            self.assertGreater(conf3, 0.80)
        finally:
            for p in [f_cf32, f_cs16, f_cu8]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    def test_autonomous_sample_rate_detection(self):
        """Verify autonomous sample rate detector from WAV headers, regex tokens, and SDR profiles."""
        from dsp.loaders import auto_detect_file_sample_rate

        # 1. Filename regex token: MSPS
        fs1, ok1, _ = auto_detect_file_sample_rate("capture_2.048MSPS_rf.iq")
        self.assertTrue(ok1)
        self.assertAlmostEqual(fs1, 2_048_000.0)

        # 2. Filename regex token: kHz
        fs2, ok2, _ = auto_detect_file_sample_rate("recording_250kHz_data.bin")
        self.assertTrue(ok2)
        self.assertAlmostEqual(fs2, 250_000.0)

        # 3. SDR Hardware Profile: HackRF
        fs3, ok3, _ = auto_detect_file_sample_rate("hackrf_band_scan.raw")
        self.assertTrue(ok3)
        self.assertAlmostEqual(fs3, 10_000_000.0)

        # 4. WAV Audio Header
        navtex_path = os.path.join(os.path.dirname(__file__), "..", "verified_samples", "NAVTEX.wav")
        if os.path.exists(navtex_path):
            fs4, ok4, _ = auto_detect_file_sample_rate(navtex_path)
            self.assertTrue(ok4)
            self.assertAlmostEqual(fs4, 44_100.0)

    def test_adaptive_pipeline_dispatch(self):
        """Verify adaptive pipeline switches extractor strategy based on detected signal nature."""
        from dsp.adaptive_pipeline import run_adaptive_pipeline

        # 1. 2G ALE -> MfskExtractor
        ale_path = os.path.join(os.path.dirname(__file__), "..", "verified_samples", "2G_ALE.wav")
        if os.path.exists(ale_path):
            sig, fs, meta = load_signal_file(ale_path, max_samples=100_000)
            res = run_adaptive_pipeline(sig, fs, metadata=meta)
            self.assertEqual(res["specialized_telemetry"]["extractor_pipeline"], "MfskExtractor")
            self.assertEqual(res["specialized_telemetry"]["mfsk_tone_count"], 8)
            self.assertAlmostEqual(res["parameters"]["estimated_baud_rate_hz"], 125.0, delta=1.0)

        # 2. OTH Radar -> PulsedRadarExtractor with Range Resolution & PRF
        oth_path = os.path.join(os.path.dirname(__file__), "..", "verified_samples", "OTH_SW_Radar.wav")
        if os.path.exists(oth_path):
            sig, fs, meta = load_signal_file(oth_path, max_samples=100_000)
            res = run_adaptive_pipeline(sig, fs, metadata=meta)
            self.assertEqual(res["specialized_telemetry"]["extractor_pipeline"], "PulsedRadarExtractor")
            self.assertGreater(res["specialized_telemetry"]["radar_range_resolution_meters"], 0.0)
            self.assertGreater(res["specialized_telemetry"]["radar_max_unambiguous_range_km"], 0.0)
            self.assertAlmostEqual(res["specialized_telemetry"]["radar_prf_hz"], 43.2, delta=2.0)
            self.assertIsNone(res["parameters"]["estimated_baud_rate_hz"])

    def test_satellite_telemetry_aist2d(self):
        """Verify AIST-2D satellite PCM/PM over NFM classification and audio domain metadata."""
        from dsp.adaptive_pipeline import run_adaptive_pipeline
        aist_path = os.path.join(os.path.dirname(__file__), "..", "verified_samples", "AIST-2D.wav")
        if os.path.exists(aist_path):
            sig, fs, meta = load_signal_file(aist_path, max_samples=250_000)
            self.assertTrue(meta.get("is_demodulated_audio"))
            self.assertIn("Demodulated Audio Track", meta.get("recording_domain", ""))
            res = run_adaptive_pipeline(sig, fs, metadata=meta)
            det = res.get("autonomous_detection", {})
            spec = res.get("specialized_telemetry", {})
            params = res.get("parameters", {})
            pulse_info = res.get("pulse_analysis", {})

            # Must NOT be classified as Maritime AIS Burst or Radar
            self.assertNotEqual(det.get("signal_class_id"), "MARITIME_AIS_BURST")
            self.assertEqual(det.get("signal_class_id"), "SATELLITE_TELEMETRY_NFM")
            self.assertIn("PCM/PM", det.get("protocol_name", ""))
            self.assertEqual(spec.get("extractor_pipeline"), "SatelliteTelemetryExtractor")
            self.assertAlmostEqual(spec.get("satellite_subcarrier_frequency_hz", 0), 2402.0, delta=20.0)
            self.assertTrue(params.get("audio_passband_artifact_detected"))
            self.assertEqual(pulse_info.get("signal_mode"), "Satellite Telemetry / Subcarrier Burst (PCM/PM)")
            self.assertFalse(pulse_info.get("is_radar"))
            self.assertFalse(pulse_info.get("is_tdma"))

    def test_satellite_telemetry_doppler_robustness(self):
        """Verify AIST-2D satellite classification under Doppler frequency shift without filename hints."""
        from dsp.autonomous_detector import detect_signal_autonomously
        from dsp.pulse_analyzer import analyze_pulse_train
        aist_path = os.path.join(os.path.dirname(__file__), "..", "verified_samples", "AIST-2D.wav")
        if os.path.exists(aist_path):
            sig, fs, _ = load_signal_file(aist_path, max_samples=200_000)
            t = np.arange(len(sig)) / fs

            # Test positive Doppler (+150 Hz)
            sig_pos = sig * np.exp(2j * np.pi * 150.0 * t)
            p_pos = analyze_pulse_train(sig_pos, fs)
            det_pos = detect_signal_autonomously(sig_pos, fs, pulse_info=p_pos, file_name="unlabeled_track.wav")
            self.assertEqual(det_pos["signal_class_id"], "SATELLITE_TELEMETRY_NFM")
            self.assertNotEqual(det_pos["signal_class_id"], "RADAR_GENERIC_PULSED")

            # Test negative Doppler (-180 Hz)
            sig_neg = sig * np.exp(-2j * np.pi * 180.0 * t)
            p_neg = analyze_pulse_train(sig_neg, fs)
            det_neg = detect_signal_autonomously(sig_neg, fs, pulse_info=p_neg, file_name="unlabeled_track.wav")
            self.assertEqual(det_neg["signal_class_id"], "SATELLITE_TELEMETRY_NFM")
            self.assertNotEqual(det_neg["signal_class_id"], "RADAR_HAARP_IONO")

    def test_satellite_telemetry_low_snr_robustness(self):
        """Verify AIST-2D does not fall back to Maritime AIS or Radar at low SNR."""
        from dsp.autonomous_detector import detect_signal_autonomously
        from dsp.pulse_analyzer import analyze_pulse_train
        aist_path = os.path.join(os.path.dirname(__file__), "..", "verified_samples", "AIST-2D.wav")
        if os.path.exists(aist_path):
            sig, fs, _ = load_signal_file(aist_path, max_samples=200_000)
            np.random.seed(42)
            noise = (np.random.randn(len(sig)) + 1j * np.random.randn(len(sig))) * 0.1
            noisy_sig = sig + noise
            p_info = analyze_pulse_train(noisy_sig, fs)
            det = detect_signal_autonomously(noisy_sig, fs, pulse_info=p_info, file_name="unlabeled_track.wav")
            self.assertEqual(det["signal_class_id"], "SATELLITE_TELEMETRY_NFM")
            self.assertNotEqual(det["signal_class_id"], "MARITIME_AIS_BURST")

    def test_satellite_telemetry_missing_metadata(self):
        """Verify adaptive pipeline correctly infers audio domain and framing when metadata is empty."""
        from dsp.adaptive_pipeline import run_adaptive_pipeline
        aist_path = os.path.join(os.path.dirname(__file__), "..", "verified_samples", "AIST-2D.wav")
        if os.path.exists(aist_path):
            sig, fs, _ = load_signal_file(aist_path, max_samples=200_000)
            res = run_adaptive_pipeline(sig, fs, metadata={})
            p = res["parameters"]
            det = res["autonomous_detection"]
            self.assertTrue(p["is_demodulated_audio"])
            self.assertEqual(det["signal_class_id"], "SATELLITE_TELEMETRY_NFM")
            self.assertEqual(res["pulse_analysis"]["signal_mode"], "Satellite Telemetry / Subcarrier Burst (PCM/PM)")

    def test_ascii_110_baud_real_world_intercept(self):
        """Verify real-world ASCII.wav is accurately recognized as 110 Baud ITA-5 with Carson consistency."""
        from dsp.adaptive_pipeline import run_adaptive_pipeline
        ascii_path = os.path.join(os.path.dirname(__file__), "..", "verified_samples", "ASCII.wav")
        if os.path.exists(ascii_path):
            sig, fs, _ = load_signal_file(ascii_path, max_samples=250_000)
            res = run_adaptive_pipeline(sig, fs, metadata={"file_name": "unlabeled_fsk.wav"})
            det = res["autonomous_detection"]
            p = res["parameters"]

            self.assertEqual(det["signal_class_id"], "AMATEUR_ASCII_110")
            self.assertEqual(det["baud_rate_nominal"], 110.0)
            self.assertIn("ASCII", det["protocol_name"])
            self.assertAlmostEqual(p["estimated_baud_rate_hz"], 110.0, delta=2.0)
            self.assertAlmostEqual(p["fsk_symbol_dwell_time_ms"], 9.09, delta=0.5)
            self.assertAlmostEqual(p["fsk_frequency_shift_hz"], 150.7, delta=25.0)
            self.assertAlmostEqual(p["fsk_carson_bandwidth_hz"], p["fsk_frequency_shift_hz"] + 110.0, delta=1.0)

    def test_fsk_rtty_navtex_ascii_discrimination(self):
        """Verify RTTY (45.45 Baud), NAVTEX (100 Baud), and ASCII (110 Baud) are cleanly discriminated."""
        from dsp.adaptive_pipeline import run_adaptive_pipeline
        samples_dir = os.path.join(os.path.dirname(__file__), "..", "verified_samples")

        fsk_targets = {
            "RTTY.wav": ("AMATEUR_RTTY", 45.45, 22.0),
            "NAVTEX.wav": ("MARITIME_NAVTEX", 100.0, 10.0),
            "ASCII.wav": ("AMATEUR_ASCII_110", 110.0, 9.09)
        }

        for filename, (expected_class, expected_baud, expected_dwell) in fsk_targets.items():
            path = os.path.join(samples_dir, filename)
            self.assertTrue(os.path.exists(path), f"Required verified sample missing: {path}")
            sig, fs, _ = load_signal_file(path, max_samples=250_000)
            res = run_adaptive_pipeline(sig, fs, metadata={"file_name": "unlabeled.wav"})
            det = res["autonomous_detection"]
            p = res["parameters"]

            self.assertEqual(det["signal_class_id"], expected_class,
                             f"Failed discrimination for {filename}: got {det['signal_class_id']}, expected {expected_class}")
            self.assertAlmostEqual(p["estimated_baud_rate_hz"], expected_baud, delta=4.0,
                                   msg=f"Baud rate mismatch for {filename}")
            self.assertAlmostEqual(p["fsk_symbol_dwell_time_ms"], expected_dwell, delta=1.0,
                                   msg=f"Dwell time mismatch for {filename}")

        # Also verify tests/ASCII.wav directly
        tests_ascii_path = os.path.join(os.path.dirname(__file__), "ASCII.wav")
        self.assertTrue(os.path.exists(tests_ascii_path), f"tests/ASCII.wav missing: {tests_ascii_path}")
        sig_t, fs_t, _ = load_signal_file(tests_ascii_path, max_samples=250_000)
        res_t = run_adaptive_pipeline(sig_t, fs_t, metadata={"file_name": "tests_ascii.wav"})
        self.assertEqual(res_t["autonomous_detection"]["signal_class_id"], "AMATEUR_ASCII_110")

    def test_synthetic_fsk_dwell_and_baud_rate(self):
        """Verify synthetic FSK signals across 45.45, 100, 110, and 300 Baud extract exact dwell and baud parameters."""
        from dsp.parameter_extractor import estimate_fsk_dwell_and_baud_rate
        fs = 48000.0
        duration = 1.5

        test_cases = [
            (45.45, 170.0, 22.002, "Baudot RTTY 45.45"),
            (100.0, 170.0, 10.0, "NAVTEX / SITOR-B 100.0"),
            (110.0, 170.0, 9.091, "ASCII / ITA-5 110.0"),
            (300.0, 170.0, 3.333, "ASCII / Packet 300.0")
        ]

        for target_baud, shift_hz, expected_dwell_ms, expected_preset in test_cases:
            bit_duration = 1.0 / target_baud
            num_bits = int(duration * target_baud)
            np.random.seed(123)
            bits = np.random.choice([0, 1], size=num_bits)
            samples_per_bit = int(fs * bit_duration)

            f_center = 1000.0
            f_mark = f_center - shift_hz / 2.0
            f_space = f_center + shift_hz / 2.0

            freq_series = np.repeat(np.where(bits == 1, f_mark, f_space), samples_per_bit)
            phase = 2.0 * np.pi * np.cumsum(freq_series) / fs
            synth_fsk = np.cos(phase).astype(np.float32)

            dwell_res = estimate_fsk_dwell_and_baud_rate(
                synth_fsk, fs, mark_hz=f_mark, space_hz=f_space, shift_hz=shift_hz
            )

            self.assertEqual(dwell_res["detected_preset"], expected_preset)
            self.assertAlmostEqual(dwell_res["estimated_baud_rate_hz"], target_baud, delta=2.0)
            self.assertAlmostEqual(dwell_res["symbol_dwell_time_ms"], expected_dwell_ms, delta=0.5)
            self.assertGreaterEqual(dwell_res["harmonic_fit_ratio"], 0.85)

    def test_all_four_fsk_protocols_end_to_end(self):
        """Verify 45.45 Baud RTTY, 100 Baud NAVTEX, 110 Baud ASCII, and 300 Baud ASCII end-to-end in pipeline."""
        from dsp.adaptive_pipeline import run_adaptive_pipeline
        fs = 48000.0
        duration = 1.5

        configs = [
            (45.45, 170.0, "AMATEUR_RTTY", "RTTY", 22.0),
            (100.0, 170.0, "MARITIME_NAVTEX", "NAVTEX", 10.0),
            (110.0, 170.0, "AMATEUR_ASCII_110", "ASCII", 9.09),
            (300.0, 170.0, "TELETYPE_ASCII_300", "ASCII", 3.33)
        ]

        for target_baud, shift_hz, expected_class, expected_proto_substr, expected_dwell in configs:
            bit_duration = 1.0 / target_baud
            num_bits = int(duration * target_baud)
            np.random.seed(42)
            bits = np.random.choice([0, 1], size=num_bits)
            samples_per_bit = int(fs * bit_duration)

            f_center = 1000.0
            f_mark = f_center - shift_hz / 2.0
            f_space = f_center + shift_hz / 2.0

            freq_series = np.repeat(np.where(bits == 1, f_mark, f_space), samples_per_bit)
            phase = 2.0 * np.pi * np.cumsum(freq_series) / fs
            synth = np.cos(phase).astype(np.float32)

            res = run_adaptive_pipeline(synth, fs, metadata={"file_name": f"synth_{int(target_baud)}.wav"})
            det = res["autonomous_detection"]
            p = res["parameters"]

            self.assertEqual(det["signal_class_id"], expected_class,
                             f"Failed end-to-end for {target_baud} Baud: got {det['signal_class_id']}")
            self.assertIn(expected_proto_substr, det["protocol_name"])
            self.assertAlmostEqual(p["estimated_baud_rate_hz"], target_baud, delta=4.0)
            self.assertAlmostEqual(p["fsk_symbol_dwell_time_ms"], expected_dwell, delta=0.5)


if __name__ == "__main__":
    unittest.main()
