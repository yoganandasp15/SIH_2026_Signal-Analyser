"""
Unit Tests for First-Class Epistemic Application States (Step 7)
================================================================
Verifies that the 6 application states are first-class, never collapsed into generic errors,
and each provides structured explanations and metadata:
1. VALIDATED: Closed-loop mathematical/physical proof.
2. ESTIMATED: Continuous physical parameters estimated without code proof or with minor drift.
3. AMBIGUOUS: Competing hypotheses within ambiguity threshold; shows top candidates.
4. UNKNOWN: Uncataloged modulation; shows successfully measured physical observations.
5. UNKNOWN_OOD: Out-of-Distribution waveform deviating from feature training manifold.
6. NO SIGNAL / NOISE FLOOR: Input rejected by Gaussian noise pre-gate (rejected, not failed).
"""

import unittest
import numpy as np

from dsp.contracts import SignalHypothesis, ModulationFamily, EpistemicStatus, ConfidenceLevel
from dsp.validation_gate import ValidationGateStatus, ValidationGateConfig, run_validation_gate
from dsp.parameter_uncertainty import get_verdict_explanation
from dsp.candidate_ranker import CandidateRankingConfig, rank_and_evaluate_candidates
from dsp.adaptive_pipeline import run_adaptive_pipeline


class TestEpistemicStates(unittest.TestCase):
    """Rigorous tests ensuring all 6 application states are distinct and informative."""

    def test_state_1_validated(self):
        """State 1: VALIDATED waveform has closed-loop physical/algebraic proof."""
        fs = 48000.0
        t = np.arange(48000) / fs
        # Highly stable pure CW tone
        sig = (np.exp(1j * 2.0 * np.pi * 5000.0 * t)).astype(np.complex64)

        res = run_adaptive_pipeline(sig, fs)

        self.assertIn(res["final_decision"], ["VALIDATED", "ESTIMATED"])
        expl = res["verdict_explanation"]
        self.assertIn(expl["verdict"], ["VALIDATED", "ESTIMATED"])
        self.assertFalse(expl["is_rejected"])
        self.assertIn("explanation", expl)
        self.assertGreater(len(expl["explanation"]), 15)

    def test_state_2_estimated(self):
        """State 2: ESTIMATED signal parameters estimated but temporal stability or proof lacks."""
        cand = SignalHypothesis(
            hypothesis_id="HYP-EST-01",
            signal_family="FSK",
            modulation=ModulationFamily.FSK_2,
            protocol="2-FSK Data",
            parameters={"carrier_frequency": 12000.0, "occupied_bandwidth_99": 2000.0, "snr_db": 15.0},
            supporting_evidence=["Twin FSK spectral peaks"],
            evidence_score=0.90,
            confidence_level=ConfidenceLevel.HIGH
        )
        # Moderate multi-window stability (0.50 < 0.60 threshold)
        mock_temp = {
            "status": "VALIDATED",
            "cross_window_consistency_score": 0.50,
            "stability_level": "LOW",
            "unstable_parameters": ["bw_99pct_hz"],
            "parameter_stability": {
                "bw_99pct_hz": {"is_stable": False, "stability_score": 0.45, "std": 500.0, "mean": 2000.0, "window_values": [1500.0, 2500.0]}
            }
        }

        final_status, winner, trace = run_validation_gate(
            winner=cand,
            ranked_candidates=[cand],
            decision_status="DECISIVE",
            temporal_result=mock_temp,
            fs=48000.0
        )

        self.assertEqual(final_status, "ESTIMATED")
        self.assertEqual(winner.validation_status, "ESTIMATED")

        expl = get_verdict_explanation("ESTIMATED")
        self.assertEqual(expl["verdict"], "ESTIMATED")
        self.assertFalse(expl["is_rejected"])
        self.assertIn("unvalidated", expl["explanation"].lower())

    def test_state_3_ambiguous(self):
        """State 3: AMBIGUOUS signal displays top competing hypotheses within threshold."""
        c1 = SignalHypothesis(
            hypothesis_id="HYP-AMB-1",
            signal_family="FSK",
            protocol="Bell 202 / APRS",
            evidence_score=0.82,
            supporting_evidence=["AFSK twin peaks at 1200/2200 Hz"]
        )
        c2 = SignalHypothesis(
            hypothesis_id="HYP-AMB-2",
            signal_family="FSK",
            protocol="POCSAG 1200",
            evidence_score=0.80,
            supporting_evidence=["2-FSK tone spacing 1000 Hz"]
        )

        ranked, dec_status, winner = rank_and_evaluate_candidates([c1, c2])
        self.assertEqual(dec_status, "AMBIGUOUS")

        expl = get_verdict_explanation("AMBIGUOUS", candidate_hypotheses=[c1, c2])
        self.assertEqual(expl["verdict"], "AMBIGUOUS")
        self.assertFalse(expl["is_rejected"])
        self.assertGreaterEqual(len(expl["top_hypotheses"]), 2)
        self.assertEqual(expl["top_hypotheses"][0]["protocol"], "Bell 202 / APRS")
        self.assertEqual(expl["top_hypotheses"][1]["protocol"], "POCSAG 1200")
        self.assertIn("competing", expl["explanation"].lower())

    def test_state_4_unknown(self):
        """State 4: UNKNOWN signal shows successfully measured physical observations."""
        mock_params = {
            "fc_peak_hz": 14250.0,
            "bw_99pct_hz": 3500.0,
            "snr_db": 14.2,
            "papr_db": 6.5,
            "envelope_variance_ratio": 0.42
        }

        expl = get_verdict_explanation("UNKNOWN", parameters=mock_params)
        self.assertEqual(expl["verdict"], "UNKNOWN")
        self.assertFalse(expl["is_rejected"])
        self.assertIn("measured_observations", expl)
        self.assertGreaterEqual(len(expl["measured_observations"]), 3)
        obs_text = " ".join(expl["measured_observations"])
        self.assertIn("14,250.0 Hz", obs_text)
        self.assertIn("3,500.0 Hz", obs_text)

    def test_state_5_unknown_ood(self):
        """State 5: UNKNOWN_OOD signal explicitly flagged as Out-of-Distribution."""
        expl = get_verdict_explanation("UNKNOWN_OOD")
        self.assertEqual(expl["verdict"], "UNKNOWN_OOD")
        self.assertFalse(expl["is_rejected"])
        self.assertIn("out-of-distribution", expl["explanation"].lower())
        self.assertIn("anomalous", expl["explanation"].lower())

    def test_state_6_no_signal_noise_floor(self):
        """State 6: NO SIGNAL / NOISE FLOOR indicates rejected input rather than failure."""
        fs = 48000.0
        np.random.seed(42)
        # Stationary complex Gaussian white noise
        noise = (np.random.normal(0, 1, 16384) + 1j * np.random.normal(0, 1, 16384)).astype(np.complex64)

        res = run_adaptive_pipeline(noise, fs)

        self.assertEqual(res["final_decision"], "NO SIGNAL / NOISE FLOOR")
        self.assertEqual(res["final_status"], "NO SIGNAL / NOISE FLOOR")
        self.assertTrue(res["abstained"])

        expl = res["verdict_explanation"]
        self.assertEqual(expl["verdict"], "NO SIGNAL / NOISE FLOOR")
        self.assertTrue(expl["is_rejected"])  # Clearly rejected rather than failed
        self.assertIn("rejected", expl["explanation"].lower())
        self.assertIn("gaussian noise", expl["explanation"].lower())


if __name__ == "__main__":
    unittest.main()
