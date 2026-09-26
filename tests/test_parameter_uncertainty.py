"""
Unit Tests for Parameter Uncertainty and Epistemic Status Reporting (Step 6)
=============================================================================
Validates scientifically honest parameter uncertainty, stability,
and epistemic tier reporting:
1. Verifies structured parameter dictionary schema (value, status, stability, uncertainty, reason, unit).
2. Verifies that when uncertainty cannot be reliably estimated, uncertainty is null (None) and an explicit reason is returned.
3. Verifies strict distinction among OBSERVED, ESTIMATED, HYPOTHESIZED, VALIDATED, and UNKNOWN tiers.
4. Verifies no invented or fake uncertainty values.
5. Verifies backward compatibility: raw numerical parameters are preserved in res["parameters"].
6. Verifies end-to-end integration with run_adaptive_pipeline.
"""

import unittest
import numpy as np

from dsp.contracts import EpistemicStatus
from dsp.parameter_uncertainty import (
    build_parameter_uncertainty_report,
    get_verdict_explanation
)
from dsp.adaptive_pipeline import run_adaptive_pipeline


class TestParameterUncertainty(unittest.TestCase):
    """Rigorous tests for Step 6 Parameter Uncertainty & Epistemic Reporting."""

    def test_parameter_uncertainty_structure(self):
        """Each parameter dictionary must return value, status, stability, uncertainty, and reason."""
        mock_params = {
            "carrier_frequency": 12000.0,
            "occupied_bandwidth_99": 2200.0,
            "snr_db": 18.5,
            "symbol_rate": 1200.0,
            "frequency_shift": 1000.0
        }
        mock_temporal = {
            "status": "VALIDATED",
            "parameter_stability": {
                "fc_peak_hz": {
                    "mean": 12000.0,
                    "std": 1.25,
                    "is_stable": True,
                    "stability_score": 0.98,
                    "window_values": [11999.0, 12000.0, 12001.0, 12000.0]
                },
                "bw_99pct_hz": {
                    "mean": 2200.0,
                    "std": 15.0,
                    "is_stable": True,
                    "stability_score": 0.95,
                    "window_values": [2190.0, 2210.0, 2205.0, 2195.0]
                },
                "snr_db": {
                    "mean": 18.5,
                    "std": 0.35,
                    "is_stable": True,
                    "stability_score": 0.96,
                    "window_values": [18.2, 18.8, 18.4, 18.6]
                },
                "estimated_baud_rate_hz": {
                    "mean": 1200.0,
                    "std": 2.5,
                    "is_stable": True,
                    "stability_score": 0.94,
                    "window_values": [1198.0, 1202.0, 1201.0, 1199.0]
                },
                "frequency_shift_hz": {
                    "mean": 1000.0,
                    "std": 3.0,
                    "is_stable": True,
                    "stability_score": 0.97,
                    "window_values": [998.0, 1002.0, 1001.0, 999.0]
                }
            }
        }

        report = build_parameter_uncertainty_report(
            parameters=mock_params,
            temporal_result=mock_temporal,
            validation_status="VALIDATED",
            fs=48000.0
        )

        for param_key in ["carrier_frequency", "occupied_bandwidth_99", "snr_db", "symbol_rate", "frequency_shift"]:
            self.assertIn(param_key, report)
            p_data = report[param_key]
            self.assertIn("value", p_data)
            self.assertIn("status", p_data)
            self.assertIn("stability", p_data)
            self.assertIn("uncertainty", p_data)
            self.assertIn("reason", p_data)
            self.assertIn("unit", p_data)

            # Values must be finite
            self.assertIsNotNone(p_data["value"])
            self.assertIn(p_data["status"], [s.value for s in EpistemicStatus])
            self.assertIn(p_data["stability"], ["HIGH", "MEDIUM", "LOW", "UNKNOWN"])
            self.assertIsNotNone(p_data["uncertainty"])
            self.assertGreaterEqual(p_data["uncertainty"], 0.0)

    def test_unreliable_uncertainty_returns_null_with_reason(self):
        """When uncertainty cannot be legitimately estimated, uncertainty must be None with an explicit reason."""
        mock_params = {
            "carrier_frequency": 5000.0,
            "occupied_bandwidth_99": 800.0,
            "snr_db": 10.0,
            "symbol_rate": 300.0
        }
        # Single window or missing temporal validation
        mock_temporal = {
            "status": "INSUFFICIENT_OBSERVATION_DURATION",
            "parameter_stability": {}
        }

        report = build_parameter_uncertainty_report(
            parameters=mock_params,
            temporal_result=mock_temporal,
            validation_status="ESTIMATED",
            fs=None  # No sampling rate available
        )

        # symbol_rate has no multi-window clock variance -> uncertainty must be None
        baud_rep = report["symbol_rate"]
        self.assertIsNone(baud_rep["uncertainty"])
        self.assertIsNotNone(baud_rep["reason"])
        self.assertIn("not observed", baud_rep["reason"].lower())

        # snr_db has no multi-window variance -> uncertainty must be None
        snr_rep = report["snr_db"]
        self.assertIsNone(snr_rep["uncertainty"])
        self.assertIsNotNone(snr_rep["reason"])
        self.assertIn("unavailable", snr_rep["reason"].lower())

    def test_epistemic_status_distinction(self):
        """Strictly distinguishes OBSERVED, ESTIMATED, HYPOTHESIZED, VALIDATED, and UNKNOWN."""
        # Case 1: Estimated continuous parameters
        mock_params = {
            "carrier_frequency": 14070000.0,
            "occupied_bandwidth_99": 50.0,
            "snr_db": -5.0,
            "symbol_rate": 6.25,
            "estimated_baud_rate_hz": 6.25
        }
        rep1 = build_parameter_uncertainty_report(
            parameters=mock_params,
            temporal_result=None,
            validation_status="ESTIMATED",
            is_crc_valid=False
        )
        self.assertEqual(rep1["carrier_frequency"]["status"], "OBSERVED")
        self.assertEqual(rep1["occupied_bandwidth_99"]["status"], "OBSERVED")
        self.assertEqual(rep1["snr_db"]["status"], "ESTIMATED")
        self.assertEqual(rep1["symbol_rate"]["status"], "ESTIMATED")

        # Case 1b: Nominal protocol hypothesis (not measured by continuous clock recovery)
        mock_nominal = {
            "carrier_frequency": 14070000.0,
            "symbol_rate": 100.0,
            "baud_rate_nominal": 100.0
        }
        rep_nom = build_parameter_uncertainty_report(parameters=mock_nominal)
        self.assertEqual(rep_nom["symbol_rate"]["status"], "HYPOTHESIZED")

        # Case 2: Closed-loop CRC proven -> symbol rate is VALIDATED
        rep2 = build_parameter_uncertainty_report(
            parameters=mock_params,
            temporal_result=None,
            validation_status="VALIDATED",
            is_crc_valid=True
        )
        self.assertEqual(rep2["symbol_rate"]["status"], "VALIDATED")

        # Case 3: Undetermined / unmeasurable parameter -> UNKNOWN
        rep3 = build_parameter_uncertainty_report(
            parameters={"carrier_frequency": None, "snr_db": float("nan")},
            temporal_result=None
        )
        self.assertEqual(rep3["carrier_frequency"]["status"], "UNKNOWN")
        self.assertIsNone(rep3["carrier_frequency"]["value"])
        self.assertEqual(rep3["snr_db"]["status"], "UNKNOWN")

    def test_do_not_invent_uncertainty(self):
        """Ensures that no arbitrary non-zero uncertainty is fabricated when data is absent."""
        mock_params = {"carrier_frequency": None, "snr_db": None, "symbol_rate": None}
        report = build_parameter_uncertainty_report(mock_params)
        for key in ["carrier_frequency", "snr_db", "symbol_rate"]:
            self.assertIsNone(report[key]["uncertainty"])
            self.assertIsNotNone(report[key]["reason"])

    def test_backward_compatibility_preserved(self):
        """Existing numerical output keys must remain unchanged in run_adaptive_pipeline."""
        fs = 48000.0
        t = np.arange(4800) / fs
        # 1.2 kHz BPSK tone
        sig = np.exp(1j * 2.0 * np.pi * 1200.0 * t).astype(np.complex64)

        res = run_adaptive_pipeline(sig, fs)

        # 1. Legacy numerical parameters dictionary must exist and contain scalar numbers
        self.assertIn("parameters", res)
        p = res["parameters"]
        self.assertIsInstance(p, dict)
        self.assertIn("fc_peak_hz", p)
        self.assertIsInstance(p["fc_peak_hz"], (int, float))
        self.assertIn("snr_db", p)
        self.assertIsInstance(p["snr_db"], (int, float))

        # 2. Step 6 structured parameter uncertainty dictionary must exist
        self.assertIn("parameter_uncertainties", res)
        p_unc = res["parameter_uncertainties"]
        self.assertIsInstance(p_unc, dict)
        self.assertIn("carrier_frequency", p_unc)
        self.assertIn("value", p_unc["carrier_frequency"])
        self.assertIn("status", p_unc["carrier_frequency"])
        self.assertIn("stability", p_unc["carrier_frequency"])
        self.assertIn("uncertainty", p_unc["carrier_frequency"])

        # 3. Winning hypothesis must include parameter_reports
        winning = res.get("winning_hypothesis")
        if winning:
            self.assertIn("parameter_reports", winning)
            self.assertIn("parameter_uncertainties", winning)


if __name__ == "__main__":
    unittest.main()
