"""
Unit Tests for Hypothesis Validation Gate Layer
==============================================
Validates the multi-pillar physical validation gate (dsp/validation_gate.py).

Test Matrix:
1. test_golden_candidate_is_validated: Strong evidence, zero contradictions, high temporal stability -> VALIDATED
2. test_high_spectral_evidence_fails_temporal_validation: Rule 3 Invariant -> ESTIMATED (never VALIDATED)
3. test_short_observation_duration_downgrades_to_estimated: Insufficient duration -> ESTIMATED
4. test_physical_nyquist_violation_rejects_candidate: Baud rate >> bandwidth -> UNKNOWN / REJECTED
5. test_unresolved_contradictions_prevent_validation: Presence of physical contradiction -> ESTIMATED
6. test_ambiguous_candidates_remain_ambiguous: Two candidates within margin -> AMBIGUOUS
7. test_sub_viability_evidence_returns_unknown: Evidence score < 0.35 -> UNKNOWN
8. test_ood_waveform_preserves_unknown_ood: OOD classification -> UNKNOWN_OOD
9. test_reconstruction_evm_failure_prevents_validation: EVM > 35% -> ESTIMATED
10. test_validation_trace_schema_and_completeness: Comprehensive machine-readable trace verification
11. test_configurable_thresholds_override: Custom ValidationGateConfig tuning
12. test_end_to_end_detector_validation_gate_integration: detect_signal_autonomously attaches validation gate
"""

import unittest
import numpy as np

from dsp.contracts import (
    SignalHypothesis,
    EpistemicStatus,
    ModulationFamily,
    ConfidenceLevel
)
from dsp.validation_gate import (
    ValidationGateStatus,
    ValidationGateConfig,
    GateCheckResult,
    ValidationGateTrace,
    HypothesisValidationGate,
    run_validation_gate
)
from dsp.autonomous_detector import detect_signal_autonomously
from dsp.temporal_validator import validate_temporal_consistency


class TestHypothesisValidationGate(unittest.TestCase):
    """Rigorous unit tests for HypothesisValidationGate and Rule 3 Invariants."""

    def setUp(self):
        self.config = ValidationGateConfig()
        self.gate = HypothesisValidationGate(config=self.config)

    def _create_mock_temporal_result(
        self,
        consistency_score: float = 0.88,
        status: str = "COMPLETE",
        bw_rel_var: float = 0.05,
        fc_rel_var: float = 0.02,
        unstable_params: list = None
    ) -> dict:
        """Helper generating synthetic multi-window temporal validation result."""
        unstable = unstable_params or []
        return {
            "status": status,
            "windows_analyzed": 4,
            "parameter_stability": {
                "bw_99pct_hz": {
                    "mean": 1000.0,
                    "std": 1000.0 * bw_rel_var,
                    "relative_variation": bw_rel_var,
                    "stability_score": 0.95,
                    "is_stable": bw_rel_var <= 0.25
                },
                "fc_peak_hz": {
                    "mean": 12000.0,
                    "std": 12000.0 * fc_rel_var,
                    "relative_variation": fc_rel_var,
                    "stability_score": 0.98,
                    "is_stable": fc_rel_var <= 0.15
                },
                "estimated_baud_rate_hz": {
                    "mean": 1200.0,
                    "std": 20.0,
                    "relative_variation": 0.016,
                    "stability_score": 0.96,
                    "is_stable": True
                }
            },
            "cross_window_consistency_score": consistency_score,
            "stability_level": "HIGH" if consistency_score >= 0.85 else ("MEDIUM" if consistency_score >= 0.60 else "LOW"),
            "unstable_parameters": unstable
        }

    def test_golden_candidate_is_validated(self):
        """A candidate with high evidence, high temporal stability, valid physics, and 0 contradictions -> VALIDATED."""
        cand = SignalHypothesis(
            hypothesis_id="HYP-GOLD-01",
            signal_family="FSK",
            modulation=ModulationFamily.FSK_2,
            protocol="Bell 202 / APRS",
            parameters={
                "carrier_frequency": 1200.0,
                "occupied_bandwidth_99": 2200.0,
                "snr_db": 18.5,
                "symbol_rate": 1200.0
            },
            supporting_evidence=["2-FSK spectral tones", "Symbol rate matches Bell 202"],
            contradictions=[],
            evidence_score=0.92,
            confidence_level=ConfidenceLevel.HIGH
        )
        temp_res = self._create_mock_temporal_result(consistency_score=0.88)
        pulse_info = {"signal_mode": "Continuous Transmission (Non-Pulsed)", "duty_cycle": 1.0}

        status, val_cand, trace = self.gate.evaluate_candidate(
            cand,
            all_candidates=[cand],
            decision_status_from_ranker="DECISIVE",
            temporal_result=temp_res,
            pulse_info=pulse_info,
            fs=48000.0
        )

        self.assertEqual(status, ValidationGateStatus.VALIDATED.value)
        self.assertEqual(val_cand.validation_status, "VALIDATED")
        self.assertEqual(val_cand.epistemic_status, EpistemicStatus.VALIDATED)
        self.assertIsNone(val_cand.rejection_reason)
        self.assertIn("spectral_consistency", trace.passed_gates)
        self.assertIn("temporal_consistency", trace.passed_gates)
        self.assertIn("cross_window_stability", trace.passed_gates)
        self.assertIn("physical_consistency", trace.passed_gates)
        self.assertEqual(len(trace.failed_gates), 0)

    def test_high_spectral_evidence_fails_temporal_validation(self):
        """
        RULE 3 INVARIANT:
        A candidate that has strong spectral evidence (0.95) but fails temporal validation (0.42)
        must NOT automatically be marked VALIDATED. It must be downgraded to ESTIMATED.
        """
        cand = SignalHypothesis(
            hypothesis_id="HYP-RULE3-01",
            signal_family="FSK",
            modulation=ModulationFamily.FSK_2,
            protocol="2-FSK Carrier",
            parameters={
                "carrier_frequency": 15000.0,
                "occupied_bandwidth_99": 3000.0,
                "snr_db": 22.0,
                "symbol_rate": 1200.0
            },
            supporting_evidence=[
                "Strong twin spectral peaks at 14.5 kHz and 15.5 kHz",
                "High in-band spectral SNR +22.0 dB",
                "Constant envelope variance ratio = 0.05"
            ],
            contradictions=[],
            evidence_score=0.95,  # Strong spectral evidence
            confidence_level=ConfidenceLevel.HIGH
        )
        # Multi-window stability fails (score 0.42 < min_temporal_stability 0.60)
        temp_res = self._create_mock_temporal_result(
            consistency_score=0.42,
            unstable_params=["bw_99pct_hz", "fc_peak_hz"]
        )

        status, val_cand, trace = self.gate.evaluate_candidate(
            cand,
            all_candidates=[cand],
            decision_status_from_ranker="DECISIVE",
            temporal_result=temp_res,
            pulse_info={"duty_cycle": 1.0},
            fs=48000.0
        )

        # Must NOT be marked VALIDATED
        self.assertNotEqual(status, ValidationGateStatus.VALIDATED.value)
        # Must be marked ESTIMATED
        self.assertEqual(status, ValidationGateStatus.ESTIMATED.value)
        self.assertEqual(val_cand.validation_status, "ESTIMATED")
        self.assertEqual(val_cand.epistemic_status, EpistemicStatus.ESTIMATED)
        self.assertIn("cross_window_stability", trace.failed_gates)
        self.assertIn("Downgraded to ESTIMATED", val_cand.rejection_reason)

    def test_short_observation_duration_downgrades_to_estimated(self):
        """Short observation bursts where multi-window validation cannot run -> ESTIMATED (not VALIDATED)."""
        cand = SignalHypothesis(
            hypothesis_id="HYP-SHORT-01",
            signal_family="PSK",
            modulation=ModulationFamily.BPSK,
            protocol="BPSK Telemetry",
            parameters={"occupied_bandwidth_99": 500.0, "snr_db": 15.0},
            supporting_evidence=["BPSK squared carrier tone detected"],
            contradictions=[],
            evidence_score=0.85
        )
        short_temp_res = {
            "status": "INSUFFICIENT_OBSERVATION_DURATION",
            "windows_analyzed": 0,
            "parameter_stability": {},
            "cross_window_consistency_score": None,
            "stability_level": "UNKNOWN",
            "unstable_parameters": [],
            "message": "Signal duration too short for 4 windows"
        }

        status, val_cand, trace = self.gate.evaluate_candidate(
            cand,
            all_candidates=[cand],
            decision_status_from_ranker="DECISIVE",
            temporal_result=short_temp_res,
            fs=48000.0
        )

        self.assertEqual(status, ValidationGateStatus.ESTIMATED.value)
        self.assertEqual(val_cand.validation_status, "ESTIMATED")
        self.assertIn("cross_window_stability", trace.failed_gates)

    def test_physical_nyquist_violation_rejects_candidate(self):
        """Baud rate physically exceeding bandwidth by 10x violates Nyquist capacity -> REJECTED / UNKNOWN."""
        cand = SignalHypothesis(
            hypothesis_id="HYP-PHYS-FAIL",
            signal_family="FSK",
            modulation=ModulationFamily.FSK_2,
            parameters={
                "carrier_frequency": 5000.0,
                "occupied_bandwidth_99": 200.0,  # Only 200 Hz
                "symbol_rate": 50000.0,           # Impossibly 50 kBaud in 200 Hz
                "snr_db": 10.0
            },
            supporting_evidence=["Spectral peak"],
            contradictions=[],
            evidence_score=0.75
        )
        temp_res = self._create_mock_temporal_result(consistency_score=0.85)

        status, val_cand, trace = self.gate.evaluate_candidate(
            cand,
            all_candidates=[cand],
            decision_status_from_ranker="DECISIVE",
            temporal_result=temp_res,
            fs=48000.0
        )

        self.assertEqual(status, ValidationGateStatus.UNKNOWN.value)
        self.assertEqual(val_cand.validation_status, "REJECTED")
        self.assertIn("physical_consistency", trace.failed_gates)

    def test_unresolved_contradictions_prevent_validation(self):
        """A candidate with high score (0.80) but 1 unresolved physical contradiction cannot be VALIDATED."""
        cand = SignalHypothesis(
            hypothesis_id="HYP-CONTRA-01",
            signal_family="FSK",
            modulation=ModulationFamily.FSK_2,
            protocol="Radioteletype / RTTY",
            parameters={"occupied_bandwidth_99": 300.0, "snr_db": 12.0},
            supporting_evidence=["Orthogonal 2-tone FSK spectrum"],
            contradictions=["Symbol dwell time 9.09 ms deviates 58.7% from nominal RTTY dwell 22.00 ms"],
            evidence_score=0.80
        )
        temp_res = self._create_mock_temporal_result(consistency_score=0.90)

        status, val_cand, trace = self.gate.evaluate_candidate(
            cand,
            all_candidates=[cand],
            decision_status_from_ranker="DECISIVE",
            temporal_result=temp_res,
            fs=48000.0
        )

        self.assertEqual(status, ValidationGateStatus.ESTIMATED.value)
        self.assertEqual(val_cand.validation_status, "ESTIMATED")
        self.assertIn("contradiction(s) present", val_cand.rejection_reason)

    def test_ambiguous_candidates_remain_ambiguous(self):
        """Competing hypotheses with overlapping evidence within ambiguity threshold -> AMBIGUOUS."""
        c1 = SignalHypothesis(
            hypothesis_id="HYP-C1",
            signal_family="FSK",
            protocol="Candidate 1",
            evidence_score=0.78
        )
        c2 = SignalHypothesis(
            hypothesis_id="HYP-C2",
            signal_family="FSK",
            protocol="Candidate 2",
            evidence_score=0.76
        )

        status, val_cand, trace = self.gate.evaluate_candidate(
            c1,
            all_candidates=[c1, c2],
            decision_status_from_ranker="AMBIGUOUS",
            temporal_result=self._create_mock_temporal_result(0.85)
        )

        self.assertEqual(status, ValidationGateStatus.AMBIGUOUS.value)
        self.assertEqual(val_cand.validation_status, "AMBIGUOUS")
        self.assertEqual(val_cand.epistemic_status, EpistemicStatus.HYPOTHESIZED)

    def test_sub_viability_evidence_returns_unknown(self):
        """Evidence score below viability floor (0.25 < 0.35) -> UNKNOWN."""
        cand = SignalHypothesis(
            hypothesis_id="HYP-WEAK",
            signal_family="FSK",
            evidence_score=0.25
        )

        status, val_cand, trace = self.gate.evaluate_candidate(
            cand,
            all_candidates=[cand],
            decision_status_from_ranker="UNKNOWN"
        )

        self.assertEqual(status, ValidationGateStatus.UNKNOWN.value)
        self.assertEqual(val_cand.validation_status, "UNVALIDATED")

    def test_ood_waveform_preserves_unknown_ood(self):
        """Out-of-distribution waveforms return UNKNOWN_OOD as a first-class outcome."""
        cand = SignalHypothesis(
            hypothesis_id="HYP-OOD",
            signal_family="UNKNOWN_OOD",
            modulation=ModulationFamily.UNKNOWN_OOD,
            is_ood=True,
            evidence_score=0.20
        )

        status, val_cand, trace = self.gate.evaluate_candidate(
            cand,
            all_candidates=[cand],
            decision_status_from_ranker="UNKNOWN_OOD"
        )

        self.assertEqual(status, ValidationGateStatus.UNKNOWN_OOD.value)
        self.assertEqual(val_cand.validation_status, "UNVALIDATED")

    def test_reconstruction_evm_failure_prevents_validation(self):
        """High demodulation EVM (45% > 35% max limit) prevents VALIDATED status -> ESTIMATED."""
        cand = SignalHypothesis(
            hypothesis_id="HYP-DEMOD-FAIL",
            signal_family="PSK",
            modulation=ModulationFamily.QPSK,
            protocol="QPSK Stream",
            parameters={"occupied_bandwidth_99": 20000.0, "snr_db": 15.0},
            supporting_evidence=["4-phase cyclic feature"],
            contradictions=[],
            evidence_score=0.88
        )
        temp_res = self._create_mock_temporal_result(consistency_score=0.88)
        reconstruction = {
            "demodulation": {
                "evm_pct": 45.2,  # > 35% limit
                "symbol_error_rate_est": 0.25
            },
            "synchronization": {"pll_locked": True, "pll_lock_metric": 0.85}
        }

        status, val_cand, trace = self.gate.evaluate_candidate(
            cand,
            all_candidates=[cand],
            decision_status_from_ranker="DECISIVE",
            temporal_result=temp_res,
            reconstruction_telemetry=reconstruction,
            fs=100000.0
        )

        self.assertEqual(status, ValidationGateStatus.ESTIMATED.value)
        self.assertEqual(val_cand.validation_status, "ESTIMATED")
        self.assertIn("reconstruction_consistency", trace.failed_gates)

    def test_validation_trace_schema_and_completeness(self):
        """Validates machine-readable trace serialization and dictionary integrity."""
        cand = SignalHypothesis(
            hypothesis_id="HYP-TRACE-TEST",
            signal_family="CW",
            modulation=ModulationFamily.ANALOG_AM,
            protocol="Continuous Wave",
            parameters={"carrier_frequency": 1000.0, "occupied_bandwidth_99": 50.0, "snr_db": 30.0},
            supporting_evidence=["Dead-flat envelope"],
            contradictions=[],
            evidence_score=0.95
        )
        temp_res = self._create_mock_temporal_result(consistency_score=0.92)

        status, val_cand, trace = self.gate.evaluate_candidate(
            cand,
            all_candidates=[cand],
            decision_status_from_ranker="DECISIVE",
            temporal_result=temp_res,
            fs=48000.0
        )

        trace_dict = trace.to_dict()
        self.assertIsInstance(trace_dict, dict)
        self.assertIn("hypothesis_id", trace_dict)
        self.assertIn("final_status", trace_dict)
        self.assertIn("passed_gates", trace_dict)
        self.assertIn("failed_gates", trace_dict)
        self.assertIn("waived_gates", trace_dict)
        self.assertIn("gate_results", trace_dict)
        self.assertIn("validation_score", trace_dict)
        self.assertIn("narrative_summary", trace_dict)
        self.assertGreater(trace_dict["validation_score"], 0.8)

    def test_configurable_thresholds_override(self):
        """Custom ValidationGateConfig correctly tightens validation constraints."""
        strict_cfg = ValidationGateConfig(min_temporal_stability=0.95)
        strict_gate = HypothesisValidationGate(config=strict_cfg)

        cand = SignalHypothesis(
            hypothesis_id="HYP-STRICT",
            signal_family="FSK",
            modulation=ModulationFamily.FSK_2,
            parameters={"occupied_bandwidth_99": 1000.0, "snr_db": 20.0},
            evidence_score=0.90
        )
        # 0.88 is normally high stability, but strict gate requires 0.95
        temp_res = self._create_mock_temporal_result(consistency_score=0.88)

        status, val_cand, trace = strict_gate.evaluate_candidate(
            cand,
            all_candidates=[cand],
            decision_status_from_ranker="DECISIVE",
            temporal_result=temp_res,
            fs=48000.0
        )

        self.assertEqual(status, ValidationGateStatus.ESTIMATED.value)
        self.assertIn("cross_window_stability", trace.failed_gates)

    def test_end_to_end_detector_validation_gate_integration(self):
        """detect_signal_autonomously executes candidate ranking and validation gate end-to-end."""
        fs = 48000.0
        t = np.arange(4800) / fs
        cw_sig = np.exp(1j * 2.0 * np.pi * 1000.0 * t).astype(np.complex64)

        det = detect_signal_autonomously(cw_sig, fs)

        # Legacy and Candidate Ranking fields
        self.assertIn("signal_class_id", det)
        self.assertIn("candidate_hypotheses", det)
        self.assertIn("winning_hypothesis", det)

        # New Validation Gate fields
        self.assertIn("final_decision", det)
        self.assertIn("final_status", det)
        self.assertIn("validation_status", det)
        self.assertIn("validation_trace", det)
        self.assertIn("validation_gate", det)

        # Status must be one of the supported final statuses
        self.assertIn(det["final_decision"], ["VALIDATED", "ESTIMATED", "AMBIGUOUS", "UNKNOWN", "UNKNOWN_OOD", "NO SIGNAL / NOISE FLOOR"])
        self.assertIsInstance(det["validation_trace"], dict)
        self.assertIn("gate_results", det["validation_trace"])


if __name__ == "__main__":
    unittest.main()
