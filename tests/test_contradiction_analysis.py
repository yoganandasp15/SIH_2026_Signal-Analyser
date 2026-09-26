"""
Unit Tests for Contradiction Analysis Layer
===========================================
Verifies:
1. Correct hypothesis has strong positive evidence and ZERO contradictions.
2. Incorrect alternative has measurable contradictions that reduce net_evidence_score.
3. Ambiguous hypotheses without distinguishing contradictions remain AMBIGUOUS.
4. Structured representation schema matches:
   {
       "hypothesis": "...",
       "supporting_evidence": [...],
       "contradictions": [...],
       "contradiction_count": ...,
       "net_evidence_score": ...
   }
5. Contradiction categories:
   - TEMPORAL: envelope variance violations (e.g. constant envelope vs AM)
   - PARAMETER_INCONSISTENCY: Carson's rule divergence, symbol dwell mismatch
   - SPECTRAL: missing TDMA / radar frame spectral lines
   - CROSS_WINDOW_INSTABILITY: multi-window temporal validation instability
   - RECONSTRUCTION_FAILURE: excessive EVM or carrier PLL lock failure
6. Penalty deductions are mathematically exact, bounded, and testable.
"""

import unittest
from typing import Dict, Any

from dsp.contracts import (
    SignalHypothesis,
    ModulationFamily,
    ConfidenceLevel,
    EpistemicStatus,
)
from dsp.contradiction_analyzer import (
    ContradictionCategory,
    MeasurableContradiction,
    ContradictionConfig,
    ContradictionAnalysisResult,
    analyze_candidate_contradictions,
    apply_contradiction_analysis,
)
from dsp.candidate_ranker import (
    CandidateRankingConfig,
    rank_and_evaluate_candidates,
)


class TestContradictionAnalysis(unittest.TestCase):
    """Test suite validating the Contradiction Analysis layer."""

    def setUp(self):
        self.config = ContradictionConfig(
            penalty_spectral=0.20,
            penalty_temporal=0.15,
            penalty_parameter=0.20,
            penalty_cross_window=0.15,
            penalty_reconstruction=0.20,
            max_total_penalty=0.80
        )

    def test_correct_hypothesis_strong_evidence_no_contradiction(self):
        """A correct hypothesis matching physical telemetry produces zero contradictions."""
        hyp = SignalHypothesis(
            hypothesis_id="HYP-CORRECT-110",
            signal_family="FSK",
            modulation=ModulationFamily.FSK_2,
            protocol="2-FSK (ASCII / ITA-5 110 Baud)",
            symbol_rate=110.0,
            evidence_score=0.98,
            parameters={
                "carrier_frequency": 1000.0,
                "symbol_rate": 110.0,
                "fsk_shift": 170.0,
                "occupied_bandwidth_99": 285.0,  # Matches Carson: 170 + 110 = 280 Hz
                "envelope_variance": 0.04        # Constant envelope (< 0.25)
            },
            supporting_evidence=[
                "Dual FSK carrier peaks (170 Hz shift)",
                "Matched 9.09 ms symbol dwell (110 Baud)",
                "Carson's rule bandwidth consistency"
            ],
            contradictions=[],
            temporal_consistency=0.97
        )

        feats = {
            "envelope_variance_ratio": 0.04,
            "obw_99_hz": 285.0,
            "fsk_dwell_info": {
                "symbol_dwell_time_ms": 9.09,
                "detected_preset": "ASCII 110"
            }
        }

        result = analyze_candidate_contradictions(hyp, feats=feats, config=self.config)

        # Structured schema assertions (Requirement 5)
        d = result.to_dict()
        self.assertIn("hypothesis", d)
        self.assertIn("supporting_evidence", d)
        self.assertIn("contradictions", d)
        self.assertIn("contradiction_count", d)
        self.assertIn("net_evidence_score", d)

        # Correct hypothesis verification
        self.assertEqual(result.contradiction_count, 0)
        self.assertEqual(len(result.contradictions), 0)
        self.assertEqual(result.total_penalty, 0.0)
        self.assertEqual(result.net_evidence_score, 0.98)
        self.assertEqual(len(result.supporting_evidence), 3)

    def test_incorrect_alternative_has_measurable_contradictions(self):
        """An incorrect competing candidate exhibits measurable parameter contradictions."""
        # Consider a candidate claiming RTTY (45.45 Baud, 22.0 ms dwell) on an ASCII 110 signal
        hyp_rtty = SignalHypothesis(
            hypothesis_id="HYP-WRONG-RTTY",
            signal_family="FSK",
            modulation=ModulationFamily.FSK_2,
            protocol="2-FSK (Radioteletype / RTTY / Baudot 45.45 Baud)",
            symbol_rate=45.45,
            evidence_score=0.60,
            parameters={
                "fsk_shift": 170.0,
                "symbol_rate": 45.45,
                "occupied_bandwidth_99": 285.0
            },
            supporting_evidence=["Dual FSK carrier peaks (170 Hz shift)"],
            contradictions=[],
            temporal_consistency=0.95
        )

        # Measured telemetry belongs to 110 Baud (9.09 ms dwell)
        feats = {
            "envelope_variance_ratio": 0.04,
            "obw_99_hz": 285.0,
            "fsk_dwell_info": {
                "symbol_dwell_time_ms": 9.09,  # 9.09 ms vs 22.0 ms -> > 50% error!
                "detected_preset": "ASCII 110"
            }
        }

        result = analyze_candidate_contradictions(hyp_rtty, feats=feats, config=self.config)

        self.assertGreater(result.contradiction_count, 0)
        self.assertGreater(result.total_penalty, 0.0)
        # Net evidence score must be strictly reduced by the penalty
        self.assertLess(result.net_evidence_score, result.raw_evidence_score)
        self.assertAlmostEqual(
            result.net_evidence_score,
            result.raw_evidence_score - self.config.penalty_parameter,
            places=3
        )

        # Ensure contradiction specifically identifies the measured discrepancy
        found_dwell_error = any("symbol dwell time" in c.lower() for c in result.contradictions)
        self.assertTrue(found_dwell_error)

    def test_ambiguous_hypotheses_remain_ambiguous(self):
        """Competing candidates with equal physical support and no contradiction remain AMBIGUOUS."""
        cand_a = SignalHypothesis(
            hypothesis_id="HYP-AMBIG-A",
            signal_family="PSK",
            modulation=ModulationFamily.QPSK,
            protocol="Candidate Profile Alpha",
            evidence_score=0.82,
            supporting_evidence=["4-phase constellation compatible"],
            contradictions=[],
            temporal_consistency=0.95
        )
        cand_b = SignalHypothesis(
            hypothesis_id="HYP-AMBIG-B",
            signal_family="PSK",
            modulation=ModulationFamily.QPSK,
            protocol="Candidate Profile Beta",
            evidence_score=0.81,  # Delta = 0.01 < ambiguity_threshold (0.05)
            supporting_evidence=["4-phase constellation compatible"],
            contradictions=[],
            temporal_consistency=0.95
        )

        feats = {"envelope_variance_ratio": 0.18, "obw_99_hz": 5000.0}

        ranked, status, winner = rank_and_evaluate_candidates(
            [cand_a, cand_b],
            config=CandidateRankingConfig(ambiguity_threshold=0.05),
            feats=feats
        )

        self.assertEqual(status, "AMBIGUOUS")
        self.assertIsNone(winner)
        self.assertEqual(ranked[0].validation_status, "AMBIGUOUS")
        self.assertEqual(ranked[1].validation_status, "AMBIGUOUS")

    def test_temporal_envelope_contradiction(self):
        """A constant-envelope candidate on a high-variance signal triggers TEMPORAL contradiction."""
        hyp_fsk = SignalHypothesis(
            hypothesis_id="HYP-FSK-AM-FAIL",
            signal_family="FSK",
            modulation=ModulationFamily.FSK_2,
            evidence_score=0.75,
            parameters={"envelope_variance": 0.38}
        )

        # Signal has amplitude modulation (envelope variance = 0.38 >= 0.28)
        feats = {"envelope_variance_ratio": 0.38}
        result = analyze_candidate_contradictions(hyp_fsk, feats=feats, config=self.config)

        self.assertGreaterEqual(result.contradiction_count, 1)
        self.assertEqual(result.contradiction_details[0].category, ContradictionCategory.TEMPORAL)
        self.assertEqual(result.contradiction_details[0].property_name, "envelope_variance_ratio")
        self.assertAlmostEqual(result.total_penalty, self.config.penalty_temporal, places=3)

    def test_spectral_missing_frame_line_contradiction(self):
        """GSM candidate missing 216.7 Hz frame line triggers SPECTRAL contradiction."""
        hyp_gsm = SignalHypothesis(
            hypothesis_id="HYP-GSM-FRAME-FAIL",
            signal_family="TDMA_BURST",
            modulation=ModulationFamily.FSK_2,
            protocol="TDMA Cellular (GSM GMSK / Cellular Downlink)",
            evidence_score=0.85
        )

        feats = {
            "prom_216": 1.2,  # Sub-floor line prominence (< 4.0 dB)
            "obw_99_hz": 18000.0
        }
        result = analyze_candidate_contradictions(hyp_gsm, feats=feats, config=self.config)

        found_prom_contra = any(c.property_name == "prom_216" for c in result.contradiction_details)
        self.assertTrue(found_prom_contra)
        self.assertIn(ContradictionCategory.SPECTRAL, [c.category for c in result.contradiction_details])

    def test_cross_window_instability_contradiction(self):
        """Low multi-window temporal consistency triggers CROSS_WINDOW_INSTABILITY contradiction."""
        hyp_burst = SignalHypothesis(
            hypothesis_id="HYP-UNSTABLE",
            signal_family="FSK",
            modulation=ModulationFamily.FSK_2,
            evidence_score=0.70,
            temporal_consistency=0.42  # Unstable (< 0.60)
        )

        temporal_result = {
            "temporal_consistency_score": 0.42,
            "overall_stability": 0.42
        }

        result = analyze_candidate_contradictions(
            hyp_burst, temporal_result=temporal_result, config=self.config
        )

        found_cw = any(c.category == ContradictionCategory.CROSS_WINDOW_INSTABILITY for c in result.contradiction_details)
        self.assertTrue(found_cw)
        self.assertAlmostEqual(result.total_penalty, self.config.penalty_cross_window, places=3)

    def test_reconstruction_failure_contradiction(self):
        """Demodulation EVM > 35% triggers RECONSTRUCTION_FAILURE contradiction."""
        hyp_qpsk = SignalHypothesis(
            hypothesis_id="HYP-EVM-FAIL",
            signal_family="PSK",
            modulation=ModulationFamily.QPSK,
            evidence_score=0.80
        )

        recon_telem = {
            "pll_locked": False,
            "pll_lock_metric": 0.22,
            "evm_pct": 48.5  # Excessive EVM (> 40%)
        }

        result = analyze_candidate_contradictions(
            hyp_qpsk, reconstruction_telemetry=recon_telem, config=self.config
        )

        categories = [c.category for c in result.contradiction_details]
        self.assertIn(ContradictionCategory.RECONSTRUCTION_FAILURE, categories)
        self.assertGreaterEqual(result.contradiction_count, 2)  # PLL lock + EVM

    def test_apply_contradiction_analysis_batch(self):
        """apply_contradiction_analysis updates candidates in-place with net scores and structured results."""
        c_good = SignalHypothesis(
            hypothesis_id="H-GOOD",
            signal_family="FSK",
            modulation=ModulationFamily.FSK_2,
            protocol="2-FSK (ASCII / ITA-5 110 Baud)",
            evidence_score=0.95
        )
        c_bad = SignalHypothesis(
            hypothesis_id="H-BAD",
            signal_family="TDMA_BURST",
            modulation=ModulationFamily.FSK_2,
            protocol="TDMA Cellular (GSM GMSK)",
            evidence_score=0.70
        )

        feats = {
            "envelope_variance_ratio": 0.05,
            "prom_216": 0.5,  # Missing GSM line
            "fsk_dwell_info": {"symbol_dwell_time_ms": 9.09}
        }

        updated_cands, reports = apply_contradiction_analysis([c_good, c_bad], feats=feats, config=self.config)

        self.assertEqual(len(updated_cands), 2)
        self.assertEqual(len(reports), 2)

        # Good candidate retained score and 0 contradictions
        self.assertEqual(updated_cands[0].evidence_score, 0.95)
        self.assertEqual(reports[0].contradiction_count, 0)

        # Bad candidate received penalty and updated contradictions
        self.assertLess(updated_cands[1].evidence_score, 0.70)
        self.assertGreater(reports[1].contradiction_count, 0)
        self.assertTrue(len(updated_cands[1].contradictions) > 0)


if __name__ == "__main__":
    unittest.main()
