"""
Unit and Integration Tests for Multi-Window Temporal Validation Layer
=====================================================================
Validates:
1. Stable synthetic signal (high stability, score >= 0.85, zero unstable parameters).
2. Changing-frequency signal (carrier frequency instability detected, score drops).
3. Short signal (< 1024 samples gracefully degrades to INSUFFICIENT_OBSERVATION_DURATION).
4. Noisy signal (AWGN degradation handled gracefully without NaNs).
5. Pipeline integration (run_adaptive_pipeline emits valid temporal_validation key).
"""

import unittest
import numpy as np

from dsp.temporal_validator import (
    MultiWindowTemporalValidator,
    TemporalValidationConfig,
    validate_temporal_consistency
)
from dsp.adaptive_pipeline import run_adaptive_pipeline


class TestMultiWindowTemporalValidation(unittest.TestCase):
    """Test suite for Multi-Window Temporal Validation layer."""

    def setUp(self):
        self.fs = 48000.0

    def test_stable_synthetic_signal(self):
        """Stable stationary CW / tone signal should yield HIGH stability and 0 unstable parameters."""
        n_samples = 8192
        t = np.arange(n_samples) / self.fs
        fc = 3500.0
        # Clean complex tone with slight AWGN (SNR ~ 30 dB)
        clean_sig = np.exp(1j * 2.0 * np.pi * fc * t)
        noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) * 0.03
        sig = clean_sig + noise

        res = validate_temporal_consistency(sig, self.fs)

        self.assertEqual(res["status"], "VALIDATED")
        self.assertEqual(res["windows_analyzed"], 4)
        self.assertEqual(len(res["window_results"]), 4)
        self.assertEqual(res["stability_level"], "HIGH")
        self.assertIsNotNone(res["cross_window_consistency_score"])
        self.assertGreaterEqual(res["cross_window_consistency_score"], 0.85)
        self.assertEqual(len(res["unstable_parameters"]), 0)

        # Verify tracked parameters in window 0
        w0_params = res["window_results"][0]["parameters"]
        self.assertIn("fc_centroid_hz", w0_params)
        self.assertIn("bw_99pct_hz", w0_params)
        self.assertIn("snr_db", w0_params)
        self.assertIn("mean_amplitude", w0_params)
        self.assertIn("inst_freq_mean_hz", w0_params)

        # Carrier frequency should be ~3500 Hz
        self.assertAlmostEqual(res["parameter_stability"]["fc_centroid_hz"]["mean"], fc, delta=100.0)
        self.assertTrue(res["parameter_stability"]["fc_centroid_hz"]["is_stable"])

    def test_changing_frequency_signal(self):
        """Signal with jumping carrier frequency across windows must flag frequency instability."""
        n_per_window = 2048
        # 4 different frequencies in 4 sequential windows
        freqs = [1000.0, 4000.0, 9000.0, 16000.0]
        chunks = []
        for f in freqs:
            t_chunk = np.arange(n_per_window) / self.fs
            chunks.append(np.exp(1j * 2.0 * np.pi * f * t_chunk))
        sig = np.concatenate(chunks)

        res = validate_temporal_consistency(sig, self.fs)

        self.assertEqual(res["status"], "VALIDATED")
        self.assertEqual(res["windows_analyzed"], 4)

        # Carrier frequency must be detected as unstable
        unstable = res["unstable_parameters"]
        has_freq_unstable = any(p in unstable for p in ["fc_centroid_hz", "fc_peak_hz", "inst_freq_mean_hz"])
        self.assertTrue(
            has_freq_unstable,
            f"Expected carrier frequency in unstable parameters, got: {unstable}"
        )

        # Stability score must be substantially lower than stable signal
        self.assertIn(res["stability_level"], ["LOW", "MEDIUM"])
        self.assertLess(res["cross_window_consistency_score"], 0.85)

        # Check relative variation for fc_centroid_hz
        fc_stab = res["parameter_stability"]["fc_centroid_hz"]
        self.assertFalse(fc_stab["is_stable"])
        self.assertGreater(fc_stab["std"], 1000.0)

    def test_short_signal_degradation(self):
        """Signal shorter than minimum required duration must degrade gracefully."""
        # 400 samples < 1024 samples (4 * 256)
        short_sig = np.exp(1j * 2.0 * np.pi * 1000.0 * np.arange(400) / self.fs)

        res = validate_temporal_consistency(short_sig, self.fs)

        self.assertEqual(res["status"], "INSUFFICIENT_OBSERVATION_DURATION")
        self.assertEqual(res["windows_analyzed"], 0)
        self.assertIsNone(res["cross_window_consistency_score"])
        self.assertEqual(res["stability_level"], "UNKNOWN")
        self.assertEqual(len(res["unstable_parameters"]), 0)
        self.assertEqual(len(res["window_results"]), 0)
        self.assertIn("insufficient", res["message"].lower())
        self.assertEqual(res["samples_provided"], 400)
        self.assertGreater(res["min_samples_required"], 400)

        # None input signal test
        none_res = validate_temporal_consistency(None, self.fs)
        self.assertEqual(none_res["status"], "INSUFFICIENT_OBSERVATION_DURATION")

        # Invalid fs test
        inv_fs_res = validate_temporal_consistency(short_sig, 0.0)
        self.assertEqual(inv_fs_res["status"], "INSUFFICIENT_OBSERVATION_DURATION")

    def test_noisy_signal(self):
        """Very low SNR / noisy signal must process without crashing or generating NaNs."""
        n_samples = 8192
        # Pure AWGN noise (complex Gaussian)
        np.random.seed(42)
        noisy_sig = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)).astype(np.complex64)

        res = validate_temporal_consistency(noisy_sig, self.fs)

        self.assertEqual(res["status"], "VALIDATED")
        self.assertEqual(res["windows_analyzed"], 4)
        self.assertIsNotNone(res["cross_window_consistency_score"])
        self.assertGreaterEqual(res["cross_window_consistency_score"], 0.0)
        self.assertLessEqual(res["cross_window_consistency_score"], 1.0)

        # Check that no NaN or infinite values exist in parameter stability
        for p_name, p_data in res["parameter_stability"].items():
            if p_data.get("stability_score") is not None:
                self.assertTrue(np.isfinite(p_data["stability_score"]))
                self.assertTrue(np.isfinite(p_data["relative_variation"]))
                self.assertTrue(np.isfinite(p_data["mean"]))
                self.assertTrue(np.isfinite(p_data["std"]))

    def test_pipeline_integration_backward_compatibility(self):
        """Verify run_adaptive_pipeline includes temporal_validation while preserving all prior keys."""
        n_samples = 4096
        t = np.arange(n_samples) / self.fs
        sig = np.exp(1j * 2.0 * np.pi * 2000.0 * t)

        results = run_adaptive_pipeline(sig, self.fs)

        # Core keys must be present
        expected_keys = [
            "status",
            "autonomous_detection",
            "parameters",
            "modulation_classification",
            "pulse_analysis",
            "specialized_telemetry",
            "temporal_validation",
            "reconstruction"
        ]
        for k in expected_keys:
            self.assertIn(k, results, f"Missing expected key '{k}' in pipeline results")

        # Verify temporal validation sub-dict
        tv = results["temporal_validation"]
        self.assertEqual(tv["status"], "VALIDATED")
        self.assertEqual(tv["windows_analyzed"], 4)
        self.assertIn(tv["stability_level"], ["HIGH", "MEDIUM", "LOW"])

    def test_stable_fsk_modulated_signal(self):
        """Stable 2-FSK digital modulated signal should track consistent bandwidth, shift and SNR."""
        from utils.synthetic_generator import generate_synthetic_signal
        fs = 200_000.0
        fc = 20_000.0
        baud = 5_000.0
        noisy_sig, _, _ = generate_synthetic_signal(
            mod_type="2-FSK",
            fs=fs,
            fc=fc,
            baud_rate=baud,
            snr_db=25.0,
            num_samples=16384
        )
        res = validate_temporal_consistency(noisy_sig, fs)
        self.assertEqual(res["status"], "VALIDATED")
        self.assertEqual(res["windows_analyzed"], 4)
        self.assertGreaterEqual(res["cross_window_consistency_score"], 0.80)
        self.assertIn(res["stability_level"], ["HIGH", "MEDIUM"])

    def test_custom_configuration(self):
        """Verify custom TemporalValidationConfig options take effect."""
        custom_cfg = TemporalValidationConfig(
            num_windows=4,
            min_samples_per_window=128,
            high_stability_threshold=0.98,
            medium_stability_threshold=0.70
        )
        sig = np.exp(1j * 2.0 * np.pi * 1000.0 * np.arange(2048) / self.fs)
        res = validate_temporal_consistency(sig, self.fs, config=custom_cfg)
        self.assertEqual(res["status"], "VALIDATED")
        self.assertEqual(res["windows_analyzed"], 4)


if __name__ == "__main__":
    unittest.main()
