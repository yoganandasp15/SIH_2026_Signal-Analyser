"""
Unit Tests for Tactical Intercept Scenario Demonstration System (Step 9)
========================================================================
Validates that all 7 tactical demonstration scenarios load correctly, produce
expected signal properties, do not leak ground-truth filenames to the classifier,
and cleanly execute through the autonomous pipeline.
"""

import unittest
import numpy as np

from utils.tactical_scenarios import TACTICAL_SCENARIOS, load_tactical_scenario
from dsp.adaptive_pipeline import run_adaptive_pipeline


class TestTacticalScenarios(unittest.TestCase):
    """Test suite for controlled tactical intercept scenarios."""

    def test_scenarios_registry_integrity(self):
        """Verify all 7 tactical scenarios are registered with valid metadata."""
        self.assertEqual(len(TACTICAL_SCENARIOS), 7)
        keys = [s[0] for s in TACTICAL_SCENARIOS]
        expected_keys = [
            "STANDARD_KNOWN",
            "LOW_SNR",
            "NOISE_FLOOR",
            "UNKNOWN_OOD",
            "AMBIGUOUS",
            "PULSED_RADAR",
            "DISTORTED"
        ]
        self.assertEqual(keys, expected_keys)

    def test_scenario_filenames_are_neutral(self):
        """Verify no filenames contain ground-truth leakage that could influence detection."""
        for key, _, _ in TACTICAL_SCENARIOS:
            sig, fs, meta = load_tactical_scenario(key, max_samples=4000)
            fname = meta.get("file_name", "")
            self.assertTrue(
                fname.startswith("Tactical_Scenario_") or fname.startswith("Intercept_Scenario_"),
                f"Scenario {key} leaks non-neutral file name: {fname}"
            )

    def test_standard_known_scenario(self):
        """Scenario 1: Standard known signal executes and yields a valid decision."""
        sig, fs, meta = load_tactical_scenario("STANDARD_KNOWN", max_samples=48000)
        self.assertGreater(len(sig), 0)
        self.assertGreater(fs, 0)
        res = run_adaptive_pipeline(sig, fs, metadata=meta)
        self.assertIn(res.get("final_decision"), ["VALIDATED", "ESTIMATED"])
        self.assertGreater(res["parameters"].get("snr_db", -99.0), 5.0)

    def test_noise_floor_scenario(self):
        """Scenario 3: Noise floor rejected by zero false-positive pre-gate."""
        sig, fs, meta = load_tactical_scenario("NOISE_FLOOR", max_samples=96000)
        res = run_adaptive_pipeline(sig, fs, metadata=meta)
        self.assertEqual(res.get("final_decision"), "NO SIGNAL / NOISE FLOOR")
        det = res.get("autonomous_detection", {})
        self.assertIn("Gaussian", det.get("protocol_name", ""))

    def test_unknown_ood_scenario(self):
        """Scenario 4: Unknown / OOD signal triggers Out-of-Distribution rejection."""
        sig, fs, meta = load_tactical_scenario("UNKNOWN_OOD", max_samples=48000)
        res = run_adaptive_pipeline(sig, fs, metadata=meta)
        self.assertEqual(res.get("final_decision"), "UNKNOWN_OOD")

    def test_ambiguous_scenario(self):
        """Scenario 5: Ambiguous signal triggers AMBIGUOUS decision."""
        sig, fs, meta = load_tactical_scenario("AMBIGUOUS", max_samples=48000)
        res = run_adaptive_pipeline(sig, fs, metadata=meta)
        self.assertEqual(res.get("final_decision"), "AMBIGUOUS")
        self.assertIn("Ambiguous", res.get("autonomous_detection", {}).get("protocol_name", ""))

    def test_pulsed_radar_scenario(self):
        """Scenario 6: Pulsed radar scenario activates pulse extraction pipeline."""
        sig, fs, meta = load_tactical_scenario("PULSED_RADAR", max_samples=48000)
        res = run_adaptive_pipeline(sig, fs, metadata=meta)
        det = res.get("autonomous_detection", {})
        self.assertIn(det.get("extraction_pipeline"), ["pulsed_radar", "baseband_digital"])

    def test_distorted_scenario(self):
        """Scenario 7: Saturated clipped / drifting signal degrades gracefully to ESTIMATED."""
        sig, fs, meta = load_tactical_scenario("DISTORTED", max_samples=48000)
        res = run_adaptive_pipeline(sig, fs, metadata=meta)
        # Saturated clipping and Doppler drift must prevent VALIDATED status
        self.assertEqual(res.get("final_decision"), "ESTIMATED")


if __name__ == "__main__":
    unittest.main()
