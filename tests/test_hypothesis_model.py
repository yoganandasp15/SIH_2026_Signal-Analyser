"""
Unit Tests for Round-2 SignalHypothesis Model
=============================================
Verifies:
1. Default construction and field defaults
2. Legacy positional and keyword construction compatibility
3. Round-2 positional and keyword construction with all 18 fields
4. Parameter map auto-population and epistemic status tagging
5. UNKNOWN_OOD first-class outcome handling and transitions
6. Epistemic hierarchy transitions (HYPOTHESIZED -> VALIDATED)
7. Lossless dictionary serialization and deserialization (to_dict / from_dict)
8. Epistemic score invariant (evidence_score in [0.0, 1.0], not probability)
"""

import unittest
from typing import Dict, Any

from dsp.contracts import (
    SignalHypothesis,
    ModulationFamily,
    ConfidenceLevel,
    EpistemicStatus,
)


class TestSignalHypothesisModel(unittest.TestCase):
    """Test suite validating the Round-2 SignalHypothesis contract."""

    def test_default_construction(self):
        """Default instantiation yields a consistent, well-formed hypothesis."""
        hyp = SignalHypothesis()
        self.assertTrue(hyp.hypothesis_id.startswith("HYP-"))
        self.assertEqual(hyp.modulation, ModulationFamily.UNKNOWN_OOD)
        self.assertEqual(hyp.signal_family, "UNKNOWN_OOD")
        self.assertEqual(hyp.epistemic_status, EpistemicStatus.UNKNOWN)
        self.assertEqual(hyp.validation_status, "UNVALIDATED")
        self.assertEqual(hyp.evidence_score, 0.0)
        self.assertEqual(hyp.confidence, 0.0)
        self.assertEqual(hyp.confidence_level, ConfidenceLevel.UNKNOWN)
        self.assertIsNone(hyp.protocol)
        self.assertEqual(hyp.supporting_evidence, [])
        self.assertEqual(hyp.contradictions, [])
        self.assertIsNone(hyp.rejection_reason)
        self.assertIn("symbol_rate", hyp.parameters)
        self.assertEqual(hyp.parameter_status["symbol_rate"], EpistemicStatus.UNKNOWN)
        self.assertEqual(hyp.parameter_uncertainty["symbol_rate"], 0.0)

    def test_legacy_positional_construction(self):
        """Legacy 11-argument positional construction works without regression."""
        hyp = SignalHypothesis(
            ModulationFamily.QPSK,      # 1: modulation
            2400.0,                     # 2: symbol_rate
            150.0,                      # 3: carrier_offset
            4.0,                        # 4: samples_per_symbol
            "RRC",                      # 5: pulse_model
            0.88,                       # 6: confidence
            ConfidenceLevel.HIGH,       # 7: confidence_level
            False,                      # 8: is_ood
            3.2,                        # 9: mahalanobis_distance
            ["Spectral peak clear"],    # 10: evidence
            ["BPSK (d=8.1)"],           # 11: rejected_hypotheses
        )
        self.assertEqual(hyp.modulation, ModulationFamily.QPSK)
        self.assertEqual(hyp.symbol_rate, 2400.0)
        self.assertEqual(hyp.carrier_offset, 150.0)
        self.assertEqual(hyp.samples_per_symbol, 4.0)
        self.assertEqual(hyp.pulse_model, "RRC")
        self.assertEqual(hyp.evidence_score, 0.88)
        self.assertEqual(hyp.confidence, 0.88)
        self.assertEqual(hyp.confidence_level, ConfidenceLevel.HIGH)
        self.assertFalse(hyp.is_ood)
        self.assertEqual(hyp.mahalanobis_distance, 3.2)
        self.assertEqual(hyp.evidence, ["Spectral peak clear"])
        self.assertEqual(hyp.supporting_evidence, ["Spectral peak clear"])
        self.assertEqual(hyp.rejected_hypotheses, ["BPSK (d=8.1)"])
        self.assertEqual(hyp.signal_family, "PSK")
        self.assertEqual(hyp.epistemic_status, EpistemicStatus.HYPOTHESIZED)
        self.assertEqual(hyp.validation_status, "PENDING")

    def test_legacy_keyword_construction(self):
        """Legacy keyword construction works with partial fields."""
        hyp = SignalHypothesis(
            modulation=ModulationFamily.FSK_2,
            symbol_rate=1200.0,
            carrier_offset=0.0,
            samples_per_symbol=8.0,
            pulse_model="Gaussian",
            confidence=0.92,
        )
        self.assertEqual(hyp.modulation, ModulationFamily.FSK_2)
        self.assertEqual(hyp.signal_family, "FSK")
        self.assertEqual(hyp.evidence_score, 0.92)
        self.assertEqual(hyp.confidence, 0.92)
        self.assertEqual(hyp.confidence_level, ConfidenceLevel.HIGH)
        self.assertEqual(hyp.parameters["symbol_rate"], 1200.0)
        self.assertEqual(hyp.parameter_status["symbol_rate"], EpistemicStatus.ESTIMATED)

    def test_round2_positional_construction(self):
        """Round-2 18-parameter positional signature initializes all fields."""
        params = {"symbol_rate": 4800.0, "freq_shift": 1200.0}
        p_status = {
            "symbol_rate": EpistemicStatus.ESTIMATED,
            "freq_shift": EpistemicStatus.ESTIMATED,
        }
        p_unc = {"symbol_rate": 15.0, "freq_shift": 5.0}

        hyp = SignalHypothesis(
            "HYP-TEST-001",                 # 1: hypothesis_id
            "FSK",                          # 2: signal_family
            ModulationFamily.FSK_4,         # 3: modulation
            "CUSTOM_TDMA",                  # 4: protocol
            params,                         # 5: parameters
            p_status,                       # 6: parameter_status
            p_unc,                          # 7: parameter_uncertainty
            ["4 spectral tones observed"],  # 8: supporting_evidence
            ["No AM envelope variance"],    # 9: contradictions
            0.96,                           # 10: temporal_consistency
            0.94,                           # 11: physical_consistency
            0.91,                           # 12: reconstruction_consistency
            0.95,                           # 13: cross_window_consistency
            0.93,                           # 14: evidence_score
            ConfidenceLevel.HIGH,           # 15: confidence_level
            EpistemicStatus.HYPOTHESIZED,   # 16: epistemic_status
            "PENDING",                      # 17: validation_status
            None,                           # 18: rejection_reason
        )

        self.assertEqual(hyp.hypothesis_id, "HYP-TEST-001")
        self.assertEqual(hyp.signal_family, "FSK")
        self.assertEqual(hyp.modulation, ModulationFamily.FSK_4)
        self.assertEqual(hyp.protocol, "CUSTOM_TDMA")
        self.assertEqual(hyp.parameters["symbol_rate"], 4800.0)
        self.assertEqual(hyp.parameter_status["symbol_rate"], EpistemicStatus.ESTIMATED)
        self.assertEqual(hyp.parameter_uncertainty["symbol_rate"], 15.0)
        self.assertEqual(hyp.supporting_evidence, ["4 spectral tones observed"])
        self.assertEqual(hyp.contradictions, ["No AM envelope variance"])
        self.assertEqual(hyp.temporal_consistency, 0.96)
        self.assertEqual(hyp.physical_consistency, 0.94)
        self.assertEqual(hyp.reconstruction_consistency, 0.91)
        self.assertEqual(hyp.cross_window_consistency, 0.95)
        self.assertEqual(hyp.evidence_score, 0.93)
        self.assertEqual(hyp.confidence, 0.93)
        self.assertEqual(hyp.confidence_level, ConfidenceLevel.HIGH)
        self.assertEqual(hyp.epistemic_status, EpistemicStatus.HYPOTHESIZED)
        self.assertEqual(hyp.validation_status, "PENDING")
        self.assertIsNone(hyp.rejection_reason)

    def test_missing_parameters_and_auto_population(self):
        """Missing parameters are gracefully populated with defaults and status."""
        hyp = SignalHypothesis(
            modulation=ModulationFamily.QAM_16,
            symbol_rate=None,
        )
        self.assertIn("symbol_rate", hyp.parameters)
        self.assertIsNone(hyp.parameters["symbol_rate"])
        self.assertEqual(hyp.parameter_status["symbol_rate"], EpistemicStatus.UNKNOWN)
        self.assertEqual(hyp.parameter_uncertainty["symbol_rate"], 0.0)
        self.assertEqual(hyp.parameters["carrier_offset"], 0.0)
        self.assertEqual(hyp.parameters["samples_per_symbol"], 2.0)

    def test_unknown_ood_first_class_outcome(self):
        """UNKNOWN_OOD is treated as a first-class outcome with UNKNOWN epistemic tier."""
        hyp = SignalHypothesis(
            modulation=ModulationFamily.UNKNOWN_OOD,
            is_ood=True,
            mahalanobis_distance=14.5,
            evidence=["Anomalous chirped chirp rate outside digital dictionary"],
        )
        self.assertTrue(hyp.is_ood)
        self.assertEqual(hyp.modulation, ModulationFamily.UNKNOWN_OOD)
        self.assertEqual(hyp.signal_family, "UNKNOWN_OOD")
        self.assertEqual(hyp.epistemic_status, EpistemicStatus.UNKNOWN)
        self.assertEqual(hyp.validation_status, "UNVALIDATED")

        # Verify mark_ood transition on existing hypothesis
        hyp2 = SignalHypothesis(
            modulation=ModulationFamily.BPSK,
            evidence_score=0.6,
        )
        self.assertEqual(hyp2.signal_family, "PSK")
        hyp2.mark_ood(reason="Novel radar pulse pattern identified")
        self.assertTrue(hyp2.is_ood)
        self.assertEqual(hyp2.modulation, ModulationFamily.UNKNOWN_OOD)
        self.assertEqual(hyp2.signal_family, "UNKNOWN_OOD")
        self.assertEqual(hyp2.epistemic_status, EpistemicStatus.UNKNOWN)
        self.assertEqual(hyp2.validation_status, "UNVALIDATED")
        self.assertIn("Novel radar pulse pattern identified", hyp2.contradictions)

    def test_validated_state_transition(self):
        """Hypothesis promotes to VALIDATED only through mathematical proof."""
        hyp = SignalHypothesis(
            hypothesis_id="HYP-VALID-01",
            modulation=ModulationFamily.QPSK,
            symbol_rate=12000.0,
            evidence_score=0.82,
        )
        self.assertEqual(hyp.epistemic_status, EpistemicStatus.HYPOTHESIZED)
        self.assertEqual(hyp.validation_status, "PENDING")

        # Promote via mathematical proof (e.g., CRC match)
        hyp.mark_validated("CRC-16-CCITT matched with 0 residual errors")
        self.assertEqual(hyp.epistemic_status, EpistemicStatus.VALIDATED)
        self.assertEqual(hyp.validation_status, "VALIDATED")
        self.assertEqual(hyp.reconstruction_consistency, 1.0)
        self.assertIn("CRC-16-CCITT matched with 0 residual errors", hyp.supporting_evidence)

    def test_rejection_transition(self):
        """Hypothesis marks rejected with recorded contradiction reason."""
        hyp = SignalHypothesis(
            modulation=ModulationFamily.PSK_8,
            evidence_score=0.45,
        )
        hyp.mark_rejected("Costas 8-phase loop failed convergence; metric 0.12 < 0.60")
        self.assertEqual(hyp.validation_status, "REJECTED")
        self.assertEqual(hyp.rejection_reason, "Costas 8-phase loop failed convergence; metric 0.12 < 0.60")
        self.assertIn("Costas 8-phase loop failed convergence; metric 0.12 < 0.60", hyp.contradictions)

    def test_serialization_and_deserialization_fidelity(self):
        """to_dict and from_dict perform lossless round-trip serialization."""
        original = SignalHypothesis(
            hypothesis_id="HYP-SER-99",
            signal_family="QAM",
            modulation=ModulationFamily.QAM_64,
            protocol="STANAG-4285",
            parameters={"symbol_rate": 2400.0, "carrier_offset": 50.0, "samples_per_symbol": 4.0},
            parameter_status={
                "symbol_rate": EpistemicStatus.ESTIMATED,
                "carrier_offset": EpistemicStatus.OBSERVED,
                "samples_per_symbol": EpistemicStatus.ESTIMATED,
            },
            parameter_uncertainty={"symbol_rate": 1.2, "carrier_offset": 0.5, "samples_per_symbol": 0.01},
            supporting_evidence=["64-QAM constellation grid aligned", "EVM < 4.5%"],
            contradictions=["High peak-to-average power ratio contradicts FSK"],
            temporal_consistency=0.98,
            physical_consistency=0.95,
            reconstruction_consistency=0.92,
            cross_window_consistency=0.97,
            evidence_score=0.94,
            confidence_level=ConfidenceLevel.HIGH,
            epistemic_status=EpistemicStatus.VALIDATED,
            validation_status="VALIDATED",
            rejection_reason=None,
        )

        d = original.to_dict()

        # Verify dictionary types are JSON-safe primitives
        self.assertIsInstance(d["hypothesis_id"], str)
        self.assertIsInstance(d["modulation"], str)
        self.assertEqual(d["modulation"], "64-QAM")
        self.assertIsInstance(d["confidence_level"], str)
        self.assertEqual(d["confidence_level"], "HIGH")
        self.assertIsInstance(d["epistemic_status"], str)
        self.assertEqual(d["epistemic_status"], "VALIDATED")
        self.assertIsInstance(d["parameter_status"]["carrier_offset"], str)
        self.assertEqual(d["parameter_status"]["carrier_offset"], "OBSERVED")
        self.assertEqual(d["confidence"], 0.94)
        self.assertEqual(d["evidence"], d["supporting_evidence"])

        # Reconstruct from dict
        reconstructed = SignalHypothesis.from_dict(d)

        self.assertEqual(reconstructed.hypothesis_id, original.hypothesis_id)
        self.assertEqual(reconstructed.signal_family, original.signal_family)
        self.assertEqual(reconstructed.modulation, original.modulation)
        self.assertEqual(reconstructed.protocol, original.protocol)
        self.assertEqual(reconstructed.parameters, original.parameters)
        self.assertEqual(reconstructed.parameter_status, original.parameter_status)
        self.assertEqual(reconstructed.parameter_uncertainty, original.parameter_uncertainty)
        self.assertEqual(reconstructed.supporting_evidence, original.supporting_evidence)
        self.assertEqual(reconstructed.contradictions, original.contradictions)
        self.assertAlmostEqual(reconstructed.temporal_consistency, original.temporal_consistency, places=5)
        self.assertAlmostEqual(reconstructed.physical_consistency, original.physical_consistency, places=5)
        self.assertAlmostEqual(reconstructed.reconstruction_consistency, original.reconstruction_consistency, places=5)
        self.assertAlmostEqual(reconstructed.cross_window_consistency, original.cross_window_consistency, places=5)
        self.assertAlmostEqual(reconstructed.evidence_score, original.evidence_score, places=5)
        self.assertEqual(reconstructed.confidence_level, original.confidence_level)
        self.assertEqual(reconstructed.epistemic_status, original.epistemic_status)
        self.assertEqual(reconstructed.validation_status, original.validation_status)
        self.assertEqual(reconstructed.rejection_reason, original.rejection_reason)

    def test_epistemic_score_not_probability(self):
        """evidence_score is bounded in [0.0, 1.0] and acts as epistemic evidence score."""
        hyp = SignalHypothesis(evidence_score=1.5)
        self.assertLessEqual(hyp.evidence_score, 1.0)
        self.assertGreaterEqual(hyp.evidence_score, 0.0)

        # Mutating confidence property synchronizes evidence_score
        hyp.confidence = 0.77
        self.assertEqual(hyp.evidence_score, 0.77)

        # Mutating evidence property synchronizes supporting_evidence
        hyp.evidence = ["New test observation"]
        self.assertEqual(hyp.supporting_evidence, ["New test observation"])


if __name__ == "__main__":
    unittest.main()
