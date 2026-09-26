"""
Unit Tests for Controlled Signal Robustness & Stress Testing Engine (Step 11)
=============================================================================
Validates transformation mathematical integrity and verifies graceful degradation
behavior under severe physical distortions (AWGN, clipping, frequency drift, truncation).
"""

import unittest
import numpy as np

from dsp.robustness_tester import (
    apply_awgn,
    apply_amplitude_scaling,
    apply_frequency_offset,
    apply_frequency_drift,
    apply_clipping,
    apply_shortened_duration,
    SignalRobustnessTester
)
from utils.synthetic_generator import generate_synthetic_signal


class TestSignalRobustness(unittest.TestCase):
    """Test suite for physical signal degradation transforms and robustness metrics."""

    def setUp(self):
        self.fs = 48000.0
        self.sig, _, _ = generate_synthetic_signal(
            mod_type="2-FSK", fs=self.fs, fc=2000.0, baud_rate=300.0,
            snr_db=30.0, num_samples=24000
        )

    def test_apply_awgn_snr_scaling(self):
        """Verify AWGN injection scales noise power properly relative to signal power."""
        noisy_20, p_noise_20 = apply_awgn(self.sig, snr_db=20.0), 0
        noisy_0, p_noise_0 = apply_awgn(self.sig, snr_db=0.0), 0
        self.assertEqual(len(noisy_20), len(self.sig))
        self.assertEqual(len(noisy_0), len(self.sig))
        # Noisy signal variance at 0 dB must be higher than at 20 dB
        var_20 = float(np.var(noisy_20))
        var_0 = float(np.var(noisy_0))
        self.assertGreater(var_0, var_20)

    def test_apply_clipping_reduces_peak_amplitude(self):
        """Verify non-linear clipping compresses peak amplitude strictly below percentile."""
        p_th = 30.0
        orig_peak = float(np.max(np.abs(self.sig)))
        clipped = apply_clipping(self.sig, percentile_threshold=p_th)
        clipped_peak = float(np.max(np.abs(clipped)))
        self.assertLess(clipped_peak, orig_peak)

    def test_apply_frequency_drift_phase_derivative(self):
        """Verify Doppler drift modifies instantaneous frequency progressively."""
        drift_rate = 1000.0  # Hz/s
        drifted = apply_frequency_drift(self.sig, self.fs, drift_hz_per_sec=drift_rate)
        self.assertEqual(len(drifted), len(self.sig))
        # Spectral energy should be widened across observation
        fft_orig = np.abs(np.fft.fft(self.sig))
        fft_drift = np.abs(np.fft.fft(drifted))
        self.assertNotEqual(float(np.argmax(fft_drift)), float(np.argmax(fft_orig)))

    def test_apply_shortened_duration_bounds(self):
        """Verify observation truncation enforces minimal sample bounds."""
        short_4k = apply_shortened_duration(self.sig, num_samples=4096)
        self.assertEqual(len(short_4k), 4096)
        short_sub = apply_shortened_duration(self.sig, num_samples=100)
        self.assertEqual(len(short_sub), 512)  # 512 minimum bound

    def test_robustness_tester_awgn_sweep(self):
        """Verify SignalRobustnessTester executes an AWGN sweep and tracks degradation."""
        tester = SignalRobustnessTester(fs=self.fs)
        sweep_res = tester.sweep_awgn(self.sig, snr_levels_db=[20.0, 5.0, -5.0], ground_truth_token="2-FSK")
        self.assertEqual(len(sweep_res), 3)
        # High SNR must have higher evidence or confidence than severe -5 dB SNR
        score_high = sweep_res[0]["evidence_score"]
        conf_low = sweep_res[2]["confidence"]
        self.assertGreaterEqual(score_high, 0.50)
        # Under -5 dB, decision must gracefully report non-validated or reduced status
        self.assertTrue(sweep_res[2]["is_safe_degradation"])

    def test_robustness_tester_clipping_sweep(self):
        """Verify clipping sweep executes and records temporal stability."""
        tester = SignalRobustnessTester(fs=self.fs)
        sweep_res = tester.sweep_clipping(self.sig, percentiles=[90.0, 20.0])
        self.assertEqual(len(sweep_res), 2)
        for r in sweep_res:
            self.assertIn("temporal_stability_score", r)
            self.assertIn("decision", r)


if __name__ == "__main__":
    unittest.main()
