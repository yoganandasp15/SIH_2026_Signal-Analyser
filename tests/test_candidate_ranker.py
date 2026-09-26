"""
Unit Tests for Candidate Hypothesis Ranking and Epistemic Arbitration Engine
============================================================================
Verifies:
1. Clear Winner: Top candidate dominates with evidence delta >= ambiguity_threshold.
2. Close Candidates: Multiple viable candidates with delta < ambiguity_threshold return AMBIGUOUS.
3. Closed-Loop Validation Override: Mathematically validated candidate wins even if scores are close.
4. No Valid Candidate: Sub-floor evidence scores return UNKNOWN.
5. Noise Floor: Pure Gaussian noise or silence returns UNKNOWN / ABSTAINED.
6. UNKNOWN_OOD: Out-of-distribution dynamics return UNKNOWN_OOD as first-class outcome.
7. Integration: End-to-end integration with detect_signal_autonomously and backward compatibility.
"""

import unittest
import numpy as np

from dsp.contracts import (
    SignalHypothesis,
    ModulationFamily,
    ConfidenceLevel,
    EpistemicStatus,
)
from dsp.candidate_ranker import (
    CandidateRankingConfig,
    generate_candidate_hypotheses,
    rank_and_evaluate_candidates,
    apply_candidate_ranking_to_detection,
)
from dsp.autonomous_detector import detect_signal_autonomously


class TestCandidateHypothesisRanker(unittest.TestCase):
    """Test suite validating Candidate Hypothesis Ranking and Epistemic Decision Layer."""

    def setUp(self):
        self.config = CandidateRankingConfig(
            ambiguity_threshold=0.05,
            min_evidence_score=0.35
        )

    def test_clear_winner(self):
        """Top candidate dominates by a wide margin -> DECISIVE winner."""
        c1 = SignalHypothesis(
            hypothesis_id="HYP-1",
            signal_family="FSK",
            modulation=ModulationFamily.FSK_2,
            protocol="2-FSK (ASCII / 110 Baud)",
            evidence_score=0.96,
            supporting_evidence=["Matched 110 Baud symbol dwell", "Carson bandwidth exact match"],
            contradictions=[]
        )
        c2 = SignalHypothesis(
            hypothesis_id="HYP-2",
            signal_family="FSK",
            modulation=ModulationFamily.FSK_2,
            protocol="2-FSK (NAVTEX / 100 Baud)",
            evidence_score=0.45,
            supporting_evidence=["Dual peak shift 170 Hz"],
            contradictions=["Dwell time 9.09 ms deviates from 10.0 ms NAVTEX spec by 9.1%"]
        )
        c3 = SignalHypothesis(
            hypothesis_id="HYP-3",
            signal_family="FSK",
            modulation=ModulationFamily.FSK_2,
            protocol="2-FSK (RTTY / 45.45 Baud)",
            evidence_score=0.25,
            supporting_evidence=["Dual peak shift 170 Hz"],
            contradictions=["Dwell time 9.09 ms deviates from 22.0 ms RTTY spec by 58.7%"]
        )

        ranked, status, winner = rank_and_evaluate_candidates([c3, c1, c2], config=self.config)

        self.assertEqual(status, "DECISIVE")
        self.assertIsNotNone(winner)
        self.assertEqual(winner.hypothesis_id, "HYP-1")
        self.assertEqual(ranked[0].hypothesis_id, "HYP-1")
        self.assertEqual(ranked[1].hypothesis_id, "HYP-2")
        self.assertEqual(ranked[2].hypothesis_id, "HYP-3")

        # Winner is accepted/validated; runner-ups are rejected
        self.assertIn(winner.validation_status, ["VALIDATED", "ACCEPTED"])
        self.assertEqual(ranked[1].validation_status, "REJECTED")
        self.assertEqual(ranked[2].validation_status, "REJECTED")
        self.assertTrue(len(ranked[1].rejection_reason) > 0)

    def test_close_candidates_ambiguous(self):
        """Competing candidates with delta < ambiguity_threshold return AMBIGUOUS."""
        c1 = SignalHypothesis(
            hypothesis_id="HYP-PSK",
            signal_family="PSK",
            modulation=ModulationFamily.QPSK,
            protocol="Digital QPSK",
            evidence_score=0.82,
            supporting_evidence=["c42 cumulant aligns with 4-phase constellation"],
            contradictions=[]
        )
        c2 = SignalHypothesis(
            hypothesis_id="HYP-QAM",
            signal_family="QAM",
            modulation=ModulationFamily.QAM_16,
            protocol="Digital 16-QAM",
            evidence_score=0.80,  # Delta = 0.02 < ambiguity_threshold (0.05)
            supporting_evidence=["Envelope dynamic range compatible with multi-ring QAM"],
            contradictions=[]
        )

        ranked, status, winner = rank_and_evaluate_candidates([c1, c2], config=self.config)

        self.assertEqual(status, "AMBIGUOUS")
        self.assertIsNone(winner)
        self.assertEqual(c1.validation_status, "AMBIGUOUS")
        self.assertEqual(c2.validation_status, "AMBIGUOUS")
        self.assertIn("Close candidates", c1.rejection_reason)
        self.assertIn("Close candidates", c2.rejection_reason)

    def test_close_candidates_with_validation_proof(self):
        """Closed-loop mathematical proof (VALIDATED) breaks ambiguity tie."""
        c1 = SignalHypothesis(
            hypothesis_id="HYP-AIS",
            signal_family="TDMA_BURST",
            modulation=ModulationFamily.FSK_2,
            protocol="Maritime AIS 9600 Baud",
            evidence_score=0.81,
            epistemic_status=EpistemicStatus.VALIDATED,  # CRC / Syndrome verified
            validation_status="VALIDATED",
            supporting_evidence=["CRC-16 pass with 0 bit errors", "9600 Baud burst timing"]
        )
        c2 = SignalHypothesis(
            hypothesis_id="HYP-DSTAR",
            signal_family="TDMA_BURST",
            modulation=ModulationFamily.FSK_2,
            protocol="D-STAR Digital Voice",
            evidence_score=0.80,  # Delta = 0.01, but unvalidated
            epistemic_status=EpistemicStatus.HYPOTHESIZED,
            validation_status="PENDING",
            supporting_evidence=["GMSK spectrum shape"]
        )

        ranked, status, winner = rank_and_evaluate_candidates([c1, c2], config=self.config)

        self.assertEqual(status, "DECISIVE")
        self.assertIsNotNone(winner)
        self.assertEqual(winner.hypothesis_id, "HYP-AIS")
        self.assertEqual(c2.validation_status, "REJECTED")
        self.assertIn("mathematical validation", c2.rejection_reason)

    def test_no_valid_candidate(self):
        """Hypotheses below min_evidence_score return UNKNOWN."""
        c1 = SignalHypothesis(
            hypothesis_id="HYP-LOW1",
            signal_family="PSK",
            modulation=ModulationFamily.BPSK,
            evidence_score=0.20,
            supporting_evidence=["Weak spectral bump"]
        )
        c2 = SignalHypothesis(
            hypothesis_id="HYP-LOW2",
            signal_family="FSK",
            modulation=ModulationFamily.FSK_2,
            evidence_score=0.15,
            supporting_evidence=["Indeterminate frequency shift"]
        )

        ranked, status, winner = rank_and_evaluate_candidates([c1, c2], config=self.config)

        self.assertEqual(status, "UNKNOWN")
        self.assertIsNone(winner)

    def test_noise_floor(self):
        """Silence or stationary Gaussian noise floor returns UNKNOWN / ABSTAINED."""
        raw_noise_det = {
            "signal_class_id": "UNKNOWN",
            "protocol_name": "Unknown / Noise Floor (No Modulated Signal)",
            "modulation_family": "Noise",
            "confidence": 0.0,
            "extraction_pipeline": "generic_fallback",
            "physical_evidence": [
                "Spectral Flatness Measure = 0.85 >= 0.70 (Uniform Gaussian Noise Spectrum)",
                "Peak-to-Median PSD Ratio = 3.2 < 8.0"
            ],
            "rejected_hypotheses": ["Digital Comms: Excluded because spectrum matches Gaussian noise"]
        }

        candidates = generate_candidate_hypotheses(raw_noise_det)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].signal_family, "NOISE")
        self.assertEqual(candidates[0].evidence_score, 0.0)

        ranked, status, winner = rank_and_evaluate_candidates(candidates, config=self.config)
        self.assertEqual(status, "UNKNOWN")
        self.assertEqual(winner.validation_status, "ABSTAINED")

    def test_unknown_ood(self):
        """Out-of-distribution signal returns UNKNOWN_OOD as first-class outcome."""
        raw_ood_det = {
            "signal_class_id": "UNKNOWN",
            "protocol_name": "Unknown / Novel Out-of-Distribution Waveform",
            "modulation_family": "Unknown",
            "confidence": 0.20,
            "is_ood": True,
            "extraction_pipeline": "generic_fallback",
            "physical_evidence": [
                "Mahalanobis OOD distance 18.4 exceeds rejection threshold 8.5",
                "Uncataloged non-linear phase trajectory"
            ],
            "rejected_hypotheses": ["Known dictionary modulations: Distance exceeds 3-sigma bounds"]
        }

        candidates = generate_candidate_hypotheses(raw_ood_det)
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].is_ood)
        self.assertEqual(candidates[0].signal_family, "UNKNOWN_OOD")
        self.assertEqual(candidates[0].epistemic_status, EpistemicStatus.UNKNOWN)

        ranked, status, winner = rank_and_evaluate_candidates(candidates, config=self.config)
        self.assertEqual(status, "UNKNOWN_OOD")
        self.assertEqual(winner.signal_family, "UNKNOWN_OOD")

    def test_end_to_end_detector_enrichment(self):
        """detect_signal_autonomously generates candidate hypotheses while preserving legacy API."""
        fs = 48000.0
        t = np.arange(4800) / fs
        # Synthetic CW carrier
        cw_sig = np.exp(1j * 2.0 * np.pi * 1000.0 * t).astype(np.complex64)

        det = detect_signal_autonomously(cw_sig, fs)

        # Legacy API fields must be present and valid
        self.assertEqual(det["signal_class_id"], "CONTINUOUS_WAVE_UNMOD")
        self.assertIn("Continuous Wave", det["protocol_name"])
        self.assertGreaterEqual(det["confidence"], 0.95)
        self.assertIn("physical_evidence", det)
        self.assertIn("rejected_hypotheses", det)
        self.assertFalse(det["abstained"])

        # Richer candidate ranking fields must be attached
        self.assertIn("candidate_hypotheses", det)
        self.assertIn("ranked_candidates", det)
        self.assertIn("winning_hypothesis", det)
        self.assertIn("decision_status", det)
        self.assertEqual(det["decision_status"], "DECISIVE")

        candidates = det["candidate_hypotheses"]
        self.assertGreaterEqual(len(candidates), 1)
        top = candidates[0]
        self.assertEqual(top["modulation"], "AM")  # CW mapped under unmodulated continuous carrier
        self.assertGreaterEqual(top["evidence_score"], 0.70)
        self.assertEqual(top["validation_status"], "ACCEPTED")


if __name__ == "__main__":
    unittest.main()
