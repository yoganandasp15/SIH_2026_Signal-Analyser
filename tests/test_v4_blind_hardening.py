"""
Unit Tests for V4 Blind Hardening & Physics-Based Discrimination
===============================================================
Verifies zero false-positive rejection on synthetic noise and silence,
filename invariance across GRAVES and satellite telemetry, input validation,
operator override, and QPSK vs FM cumulant/spectral discrimination.
"""

import unittest
import numpy as np

from dsp.adaptive_pipeline import AdaptiveExtractionPipeline
from dsp.autonomous_detector import detect_signal_autonomously
from dsp.pulse_analyzer import analyze_pulse_train
from dsp.preprocessor import validate_input_signal
from dsp.parameter_extractor import estimate_snr_m2m4


class TestV4BlindHardening(unittest.TestCase):
    """Rigorous blind tests against edge cases and physics-based invariants."""

    def setUp(self):
        self.pipeline = AdaptiveExtractionPipeline()
        np.random.seed(42)

    def test_pure_gaussian_noise_rejection(self):
        """Pure AWGN must yield UNKNOWN with 0.0 confidence, never AIS or Radar."""
        fs = 48000.0
        noise = np.random.normal(0, 1, 48000)
        
        pulse_info = analyze_pulse_train(noise, fs)
        res = self.pipeline.run(noise, fs)
        det = res.get("autonomous_detection", {})

        self.assertEqual(det.get("signal_class_id"), "UNKNOWN")
        self.assertEqual(det.get("confidence"), 0.0)
        self.assertIn("Noise", det.get("protocol_name", ""))

    def test_pure_silence_rejection(self):
        """Pure zero amplitude silence must be rejected at pre-gate."""
        fs = 48000.0
        silence = np.zeros(24000)

        res = self.pipeline.run(silence, fs)
        det = res.get("autonomous_detection", {})

        self.assertEqual(det.get("signal_class_id"), "UNKNOWN")
        self.assertEqual(det.get("confidence"), 0.0)
        self.assertIn("Silence", det.get("protocol_name", ""))

    def test_input_validation_empty_and_short(self):
        """Input with < 16 samples or non-finite values must fail gracefully."""
        fs = 48000.0

        # Empty array
        is_val, err, _ = validate_input_signal(np.array([]), fs)
        self.assertFalse(is_val)
        self.assertIn("minimum 16 samples", err.lower())

        # 8 samples (< 16 min_samples)
        is_val, err, _ = validate_input_signal(np.ones(8), fs)
        self.assertFalse(is_val)
        self.assertIn("minimum 16 samples", err.lower())

        # NaN / Inf values
        bad_sig = np.array([1.0, np.nan, 2.0] * 10)
        is_val, err, _ = validate_input_signal(bad_sig, fs)
        self.assertFalse(is_val)
        self.assertIn("nan", err.lower())

    def test_filename_invariance_graves_radar(self):
        """
        GRAVES space surveillance radar must be recognized from physics alone
        even when metadata has file_name='generic_capture_001.wav'.
        """
        import os
        from dsp.loaders import load_signal_file

        graves_path = os.path.join(os.path.dirname(__file__), "..", "verified_samples", "GRAVES_Radar.wav")
        if not os.path.exists(graves_path):
            self.skipTest("GRAVES_Radar.wav not present")

        sig, fs, _ = load_signal_file(graves_path, max_samples=250000)
        
        # Test with stripped / generic filename
        res = self.pipeline.run(sig, fs, metadata={"file_name": "anonymous_recording_143mhz.wav"})
        det = res.get("autonomous_detection", {})

        self.assertEqual(det.get("signal_class_id"), "RADAR_GRAVES_SPACE")
        self.assertGreaterEqual(det.get("confidence", 0.0), 0.95)
        self.assertIn("GRAVES", det.get("protocol_name", ""))

    def test_filename_invariance_satellite_aist2d(self):
        """
        AIST-2D satellite PCM/PM telemetry must be recognized from subcarrier physics
        even when metadata has an arbitrary non-satellite filename.
        """
        import os
        from dsp.loaders import load_signal_file

        aist_path = os.path.join(os.path.dirname(__file__), "..", "verified_samples", "AIST-2D.wav")
        if not os.path.exists(aist_path):
            self.skipTest("AIST-2D.wav not present")

        sig, fs, _ = load_signal_file(aist_path, max_samples=250000)

        # Test with stripped / generic filename
        res = self.pipeline.run(sig, fs, metadata={"file_name": "unknown_sensor_feed.wav"})
        det = res.get("autonomous_detection", {})

        self.assertEqual(det.get("signal_class_id"), "SATELLITE_TELEMETRY_NFM")
        self.assertGreaterEqual(det.get("confidence", 0.0), 0.95)
        self.assertIn("AIST-2D", det.get("protocol_name", ""))

    def test_operator_override_class_id(self):
        """Caller override_class_id must force pipeline dispatch without crashing."""
        fs = 48000.0
        t = np.arange(48000) / fs
        sig = np.cos(2 * np.pi * 1000.0 * t)

        res = self.pipeline.run(sig, fs, override_class_id="MIL_STD_188_141_2G_ALE")
        det = res.get("autonomous_detection", {})

        self.assertEqual(det.get("signal_class_id"), "MIL_STD_188_141_2G_ALE")
        self.assertEqual(det.get("confidence"), 1.0)
        self.assertIn("Manual Operator Override", det.get("protocol_name", ""))

    def test_qpsk_vs_fm_discrimination(self):
        """
        Synthesize pure QPSK (4th-order discrete spectral line, c40 ~ -1.0)
        and FM (continuous phase variance, inst_std >= 1000 Hz, c42 ~ -1.0).
        Ensure zero cross-contamination.
        """
        fs = 100000.0
        n_syms = 2000
        sps = 20
        n_samples = n_syms * sps
        t = np.arange(n_samples) / fs

        # 1. QPSK: constellation points (1+1j)/sqrt(2), etc.
        symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=n_syms) / np.sqrt(2.0)
        baseband = np.repeat(symbols, sps)
        # Carrier at 15 kHz
        fc = 15000.0
        qpsk_rf = np.real(baseband * np.exp(2j * np.pi * fc * t))

        det_qpsk = detect_signal_autonomously(qpsk_rf, fs)
        self.assertIn(det_qpsk.get("signal_class_id"), ["GENERIC_QPSK", "GENERIC_BPSK", "GENERIC_8-PSK"])
        self.assertNotEqual(det_qpsk.get("modulation_family"), "Analog FM")

        # 2. FM: Wideband voice-like modulation (message freq 1 kHz, dev 5 kHz)
        msg = np.sin(2 * np.pi * 1000.0 * t)
        phase = 2 * np.pi * 5000.0 * np.cumsum(msg) / fs
        fm_rf = np.cos(2 * np.pi * fc * t + phase)

        det_fm = detect_signal_autonomously(fm_rf, fs)
        self.assertIn(det_fm.get("signal_class_id"), ["GENERIC_FM", "ANALOG_FM_VOICE", "ANALOG_VOICE_NFM"])
        self.assertNotIn("QPSK", det_fm.get("protocol_name", ""))


if __name__ == "__main__":
    unittest.main()
