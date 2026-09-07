"""
Comprehensive Test Suite for NTRO SIH26147 V3 Hardened Signal Intelligence Pipeline
===================================================================================
Verifies all 10 core architectural stages:
1. Input Forensics & 3-State Sampling Rate Authority
2. Signal Conditioning (DC, Conditional IQ, Scaling)
3. Feature Extraction & Mathematically Verified C63 Cumulants
4. Open-Set AMC with Mahalanobis OOD Rejection
5. Digital Carrier & Symbol Timing Synchronization
6. Multi-Modulation Soft Demodulation & LLRs
7. De-interleaving Inversion (Block, Convolutional, Diagonal, Pseudo-Random)
8. Multi-Decoder FEC (Viterbi Soft, Reed-Solomon GF(2^8), LDPC Min-Sum)
9. Bitstream Framing, Preamble Correlation & Parameterized CRC
10. Gated Epistemic Evidence Fusion (Observed, Estimated, Hypothesized, Validated, Unknown)
"""

import unittest
import numpy as np
import tempfile
import os

import dsp
from dsp.contracts import (
    EpistemicStatus,
    SamplingRateStatus,
    ModulationFamily,
    InterleaverType,
    FECCodeFamily,
    ConfidenceLevel
)
from dsp.deinterleaving import (
    interleave_block, deinterleave_block,
    interleave_convolutional, deinterleave_convolutional,
    interleave_diagonal, deinterleave_diagonal,
    interleave_pseudorandom, deinterleave_pseudorandom
)


class TestV3Pipeline(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    # -------------------------------------------------------------------------
    # 1. Input Forensics & Sampling Rate Authority
    # -------------------------------------------------------------------------
    def test_input_forensics_wav(self):
        """Authoritative sampling rate from WAV header."""
        wav_path = "tests/ASCII.wav"
        if os.path.exists(wav_path):
            record = dsp.inspect_signal_file(wav_path)
            self.assertEqual(record.sample_rate_status, SamplingRateStatus.VERIFIED_METADATA)
            self.assertIsNotNone(record.sample_rate)
            self.assertFalse(record.is_normalized)
            self.assertGreater(len(record.samples), 0)

    def test_input_forensics_raw_iq_normalized(self):
        """Unknown raw binary without header enforces NORMALIZED_DOMAIN and sample_rate is None."""
        with tempfile.NamedTemporaryFile(suffix=".iq", delete=False) as tmp:
            raw_data = (np.random.randn(500) + 1j * np.random.randn(500)).astype(np.complex64)
            tmp.write(raw_data.tobytes())
            tmp_path = tmp.name

        try:
            record = dsp.inspect_signal_file(tmp_path)
            self.assertEqual(record.sample_rate_status, SamplingRateStatus.NORMALIZED_DOMAIN)
            self.assertIsNone(record.sample_rate)
            self.assertTrue(record.is_normalized)
            self.assertEqual(len(record.samples), 500)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # -------------------------------------------------------------------------
    # 2. Signal Conditioning
    # -------------------------------------------------------------------------
    def test_conditioning_dc_and_power(self):
        """DC cancellation and unit power normalization."""
        raw = (np.random.randn(2000) + 1j * np.random.randn(2000)).astype(np.complex64) + (5.0 + 3.0j)
        conditioned, report = dsp.condition_signal(raw, apply_iq=False, apply_agc=False)
        self.assertAlmostEqual(float(np.abs(np.mean(conditioned))), 0.0, places=2)
        power = float(np.mean(np.abs(conditioned) ** 2))
        self.assertAlmostEqual(power, 1.0, places=2)

    def test_conditioning_iq_imbalance(self):
        """Conditional IQ compensation detects and corrects amplitude & phase skew."""
        t = np.arange(4000)
        i = np.cos(2 * np.pi * 0.05 * t)
        # 1.5 amplitude mismatch, 15 degree phase skew
        phi = np.radians(15.0)
        q = 1.5 * np.sin(2 * np.pi * 0.05 * t + phi)
        imbalanced = (i + 1j * q).astype(np.complex64)

        compensated, report = dsp.condition_signal(imbalanced, apply_iq=True)
        self.assertTrue(report["imbalance_detected"])
        self.assertTrue(report["compensation_applied"])
        self.assertLess(report["estimated_irr_db"], 30.0)

    # -------------------------------------------------------------------------
    # 3. Features & Cumulant C63 Calculation
    # -------------------------------------------------------------------------
    def test_features_and_c63_formula(self):
        """Verified C63 formulation: M63 - 6*M20*M40 - 9*M42*M21 + 18*(M20^2)*M21 + 12*(M21^3)."""
        # BPSK has theoretical C40 = -2.0, C42 = -2.0, C63 = +16.0
        bits = np.random.randint(0, 2, 5000)
        bpsk = 2.0 * bits - 1.0 + 0j
        c63_unnorm, c63_norm = dsp.compute_c63_cumulant(bpsk)
        # BPSK normalized C63 theoretical value is ~16.0
        self.assertAlmostEqual(c63_norm, 16.0, delta=1.5)

        # QPSK under verified C63 equation evaluates to 12.0
        qpsk_bits = np.random.randint(0, 4, 5000)
        const = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2.0)
        qpsk = const[qpsk_bits]
        _, qpsk_c63 = dsp.compute_c63_cumulant(qpsk)
        self.assertAlmostEqual(qpsk_c63, 12.0, delta=1.5)


        # 20-dimensional feature schema
        features = dsp.extract_signal_features(bpsk, fs=100_000.0)
        vec = features.to_vector()
        self.assertEqual(len(vec), 20)
        self.assertEqual(len(features.feature_names()), 20)

    # -------------------------------------------------------------------------
    # 4. Open-Set AMC with OOD Rejection
    # -------------------------------------------------------------------------
    def test_amc_bpsk_classification(self):
        """BPSK classification from physical invariant cumulants."""
        bits = np.random.randint(0, 2, 2000)
        bpsk = np.repeat(2.0 * bits - 1.0, 4).astype(np.complex64)
        hyp = dsp.classify_modulation_open_set(bpsk, fs=100_000.0)
        self.assertEqual(hyp.modulation, ModulationFamily.BPSK)
        self.assertGreater(hyp.confidence, 0.70)
        self.assertFalse(hyp.is_ood)

    def test_amc_ood_noise_rejection(self):
        """Pure noise or anomalous signal rejected as UNKNOWN_OOD."""
        noise = (np.random.randn(1000) + 1j * np.random.randn(1000)).astype(np.complex64)
        # Envelope variance and cumulants for Gaussian noise trigger OOD
        hyp = dsp.classify_modulation_open_set(noise, fs=100_000.0)
        # Must either be labeled UNKNOWN_OOD or have low confidence
        self.assertTrue(hyp.is_ood or hyp.confidence < 0.60 or hyp.modulation == ModulationFamily.UNKNOWN_OOD)

    # -------------------------------------------------------------------------
    # 5. Digital Synchronization
    # -------------------------------------------------------------------------
    def test_synchronization_cfo_and_gardner(self):
        """Coarse CFO estimation, Costas PLL tracking, and Gardner timing recovery."""
        bits = np.random.randint(0, 2, 400)
        syms = 2.0 * bits - 1.0
        sps = 4
        tx = np.repeat(syms, sps).astype(np.complex64)
        # Apply 120 Hz CFO
        fs = 50_000.0
        t = np.arange(len(tx)) / fs
        cfo_applied = 120.0
        rx = tx * np.exp(1j * 2 * np.pi * cfo_applied * t)

        hyp = dsp.classify_modulation_open_set(rx, fs=fs)
        sync = dsp.synchronize_signal(rx, fs=fs, hypothesis=hyp)
        self.assertIsNotNone(sync.symbols)
        self.assertGreater(len(sync.symbols), 0)
        self.assertAlmostEqual(sync.coarse_cfo_hz, cfo_applied, delta=30.0)

    # -------------------------------------------------------------------------
    # 6. Demodulation & Soft Information (LLRs)
    # -------------------------------------------------------------------------
    def test_demodulation_soft_llrs(self):
        """Continuous LLR extraction and hard bit slicing for BPSK and QPSK."""
        bits = np.random.randint(0, 2, 100, dtype=np.uint8)
        syms = (2.0 * bits - 1.0).astype(np.complex64)
        # Create dummy SynchronizedSymbols
        sync = dsp.SynchronizedSymbols(
            symbols=syms,
            soft_llrs=np.zeros(0, dtype=np.float32),
            coarse_cfo_hz=0.0,
            fine_cfo_hz=0.0,
            residual_cfo_hz=0.0,
            phase_ambiguity_rad=0.0,
            pll_lock_metric=1.0,
            pll_locked=True,
            timing_error_variance=0.01,
            constellation_points=syms,
            eye_samples=syms,
            samples_per_symbol=1.0
        )

        demod = dsp.demodulate_signal(sync, modulation=ModulationFamily.BPSK)
        self.assertEqual(len(demod.hard_bits), len(bits))
        # Standard LLR convention: log(P(b=0)/P(b=1))
        # Bit 0 -> LLR > 0, Bit 1 -> LLR < 0
        for i in range(len(bits)):
            if demod.hard_bits[i] == 0:
                self.assertGreater(demod.soft_llrs[i], 0.0)
            else:
                self.assertLess(demod.soft_llrs[i], 0.0)


    # -------------------------------------------------------------------------
    # 7. De-interleaving Topologies
    # -------------------------------------------------------------------------
    def test_block_deinterleaver_inversion(self):
        """Block interleaver/de-interleaver matrix inversion roundtrip."""
        data = np.random.randint(0, 2, 64, dtype=np.uint8)
        interleaved = interleave_block(data, rows=8, cols=8)
        deinterleaved = deinterleave_block(interleaved, rows=8, cols=8)
        self.assertTrue(np.array_equal(data, deinterleaved))

    def test_convolutional_deinterleaver_inversion(self):
        """Forney convolutional shift register de-interleaving with B * (B - 1) * M delay."""
        B, M = 3, 1
        data = np.random.randint(0, 2, 100, dtype=np.uint8)
        interleaved = interleave_convolutional(data, branches=B, delay_step=M)
        deinterleaved = deinterleave_convolutional(interleaved, branches=B, delay_step=M)
        delay = B * (B - 1) * M
        self.assertTrue(np.array_equal(data[:len(data) - delay], deinterleaved[delay:]))

    def test_diagonal_deinterleaver_inversion(self):
        """Diagonal skew matrix de-interleaver roundtrip."""
        data = np.random.randint(0, 2, 64, dtype=np.uint8)
        interleaved = interleave_diagonal(data, rows=8, cols=8)
        deinterleaved = deinterleave_diagonal(interleaved, rows=8, cols=8)
        self.assertTrue(np.array_equal(data, deinterleaved))

    def test_pseudorandom_deinterleaver_inversion(self):
        """Seeded Galois LFSR pseudo-random de-interleaver roundtrip."""
        data = np.random.randint(0, 2, 128, dtype=np.uint8)
        interleaved = interleave_pseudorandom(data, block_size=128, seed=1)
        deinterleaved = deinterleave_pseudorandom(interleaved, block_size=128, seed=1)
        self.assertTrue(np.array_equal(data, deinterleaved))

    # -------------------------------------------------------------------------
    # 8. FEC Decoders
    # -------------------------------------------------------------------------
    def test_viterbi_soft_decoding(self):
        """Viterbi soft decoder recovers corrupted convolutional codewords (NASA K=7 R=1/2)."""
        msg = np.random.randint(0, 2, 48, dtype=np.uint8)
        poly = (0o171, 0o133)
        coded = dsp.encode_convolutional(msg, poly=poly, K=7)

        # Soft LLRs with noise and 2 flipped bits
        llrs = np.where(coded == 0, 4.0, -4.0).astype(np.float32)
        llrs[5] = -llrs[5]   # bit error
        llrs[12] = -llrs[12] # bit error

        decoded, metric, ber = dsp.decode_viterbi_soft(llrs, poly=poly, K=7)
        self.assertTrue(np.array_equal(msg, decoded[:len(msg)]))
        self.assertLess(ber, 0.05)

    def test_reed_solomon_gf256(self):
        """Reed-Solomon GF(2^8) Berlekamp-Massey decoder corrects corrupted symbols."""
        msg = [ord(c) for c in "NTRO_DEFENSE_RS"]
        two_t = 6
        encoded = dsp.fec.reed_solomon.encode_reed_solomon(msg, two_t=two_t, fcr=0)

        # Corrupt 2 bytes
        corrupted = list(encoded)
        corrupted[1] ^= 0xAA
        corrupted[4] ^= 0x55

        decoded, valid, err_count = dsp.decode_reed_solomon(corrupted, two_t=two_t, fcr=0)
        self.assertTrue(valid)
        self.assertEqual(err_count, 2)
        self.assertEqual(list(decoded[:len(msg)]), msg)

    def test_ldpc_min_sum(self):
        """Normalized Min-Sum LDPC decoder verifies parity condition H * c^T == 0."""
        H = dsp.fec.ldpc.get_standard_ldpc_matrix()
        c = np.zeros(H.shape[1], dtype=np.uint8)
        llrs = np.where(c == 0, 5.0, -5.0).astype(np.float32)
        decoded, valid, iters, weights = dsp.decode_ldpc_minsum(llrs, H=H, max_iter=10)
        self.assertTrue(valid)
        self.assertEqual(weights[-1], 0)
        self.assertTrue(np.all(decoded == 0))

    # -------------------------------------------------------------------------
    # 9. Framing, Preamble & Parameterized CRC
    # -------------------------------------------------------------------------
    def test_crc_known_vectors(self):
        """Verifies CRC algorithms against standard check values for b'123456789'."""
        test_data = b"123456789"
        # CRC-16-CCITT standard check value = 0x29B1
        crc_ccitt = dsp.compute_crc(test_data, dsp.CRC_PROFILES["CRC-16-CCITT"])
        self.assertEqual(crc_ccitt, 0x29B1)

        # CRC-16-IBM standard check value = 0xBB3D
        crc_ibm = dsp.compute_crc(test_data, dsp.CRC_PROFILES["CRC-16-IBM"])
        self.assertEqual(crc_ibm, 0xBB3D)

        # CRC-32-IEEE standard check value = 0xCBF43926
        crc_32 = dsp.compute_crc(test_data, dsp.CRC_PROFILES["CRC-32-IEEE"])
        self.assertEqual(crc_32, 0xCBF43926)

    def test_frame_preamble_and_crc_discovery(self):
        """Cross-correlates Barker sync marker and validates CRC trailer."""
        # Barker-13: 1111100110101
        barker13 = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
        payload = b"NTRO_RADAR_DATA"
        crc_val = dsp.compute_crc(payload, dsp.CRC_PROFILES["CRC-16-CCITT"])
        frame_bytes = payload + crc_val.to_bytes(2, byteorder="big")
        payload_bits = list(np.unpackbits(np.frombuffer(frame_bytes, dtype=np.uint8)))

        full_stream = np.array(barker13 + payload_bits, dtype=np.uint8)
        result = dsp.analyze_frame_structure(full_stream)
        self.assertTrue(result.preamble_detected)
        self.assertTrue(result.crc_match)
        self.assertEqual(result.crc_profile, "CRC-16-CCITT")
        self.assertIn("NTRO_RADAR_DATA", result.recovered_ascii)

    # -------------------------------------------------------------------------
    # 10. Gated Epistemic Evidence Fusion
    # -------------------------------------------------------------------------
    def test_evidence_fusion_epistemic_partitioning(self):
        """Evidence hierarchy separates PLL convergence (ESTIMATED) from CRC/Syndrome proof (VALIDATED)."""
        # Case A: PLL locked but no FEC/CRC pass -> ESTIMATED / HYPOTHESIZED only
        hyp = dsp.SignalHypothesis(
            modulation=ModulationFamily.BPSK,
            symbol_rate=1000.0,
            carrier_offset=0.0,
            samples_per_symbol=4.0,
            pulse_model=None,
            confidence=0.90,
            confidence_level=ConfidenceLevel.HIGH
        )
        sync = dsp.SynchronizedSymbols(
            symbols=np.ones(100, dtype=np.complex64),
            soft_llrs=np.ones(100, dtype=np.float32),
            coarse_cfo_hz=0.0,
            fine_cfo_hz=0.0,
            residual_cfo_hz=0.0,
            phase_ambiguity_rad=0.0,
            pll_lock_metric=0.95,
            pll_locked=True,
            timing_error_variance=0.02,
            constellation_points=np.ones(100, dtype=np.complex64),
            eye_samples=np.ones(100, dtype=np.complex64),
            samples_per_symbol=4.0
        )
        demod = dsp.DemodulationResult(
            modulation=ModulationFamily.BPSK,
            hard_bits=np.ones(100, dtype=np.uint8),
            soft_llrs=np.ones(100, dtype=np.float32),
            symbol_error_rate_est=0.01,
            evm_pct=5.0,
            noise_var_est=0.05
        )
        fec_unval = [dsp.FECHypothesis(
            family=FECCodeFamily.CONVOLUTIONAL,
            code_rate="1/2",
            parameters={},
            decoded_bits=np.ones(50, dtype=np.uint8),
            syndrome=np.ones(1, dtype=np.uint8),
            syndrome_zero=False,
            syndrome_weight=5,
            iterations=1,
            ber_estimate=0.1,
            status=EpistemicStatus.HYPOTHESIZED,
            validation_detail="Residual errors"
        )]
        framing_unval = dsp.FrameAnalysisResult(
            status=EpistemicStatus.UNKNOWN,
            frame_structure_detected=False,
            frame_type="GENERIC",
            sync_pattern_name=None,
            sync_pattern_bits=None,
            frame_length=0,
            repeated_frames_found=0,
            bit_offset=0,
            header_bits=None,
            payload_bits=None,
            crc_profile=None,
            crc_match=False,
            crc_calculated=0,
            crc_received=0,
            recovered_ascii="",
            recovered_hex=""
        )

        report_unval = dsp.fuse_evidence(
            raw_params={"fc_peak_hz": 1000.0, "bw_99pct_hz": 2000.0, "envelope_variance_ratio": 0.05, "snr_db": 15.0},
            hypothesis=hyp,
            sync_symbols=sync,
            demod_result=demod,
            interleaver_candidates=[],
            fec_candidates=fec_unval,
            framing_result=framing_unval
        )

        # PLL lock convergence is in ESTIMATED, NOT VALIDATED
        pll_items = [e for e in report_unval.estimated if "PLL" in e.description]
        self.assertEqual(len(pll_items), 1)
        self.assertEqual(len(report_unval.validated), 0)
        self.assertNotIn("FULLY VALIDATED", report_unval.overall_verdict)

        # Case B: Closed-Loop CRC and FEC Validation confirmed
        fec_val = [dsp.FECHypothesis(
            family=FECCodeFamily.CONVOLUTIONAL,
            code_rate="1/2",
            parameters={},
            decoded_bits=np.ones(50, dtype=np.uint8),
            syndrome=np.zeros(1, dtype=np.uint8),
            syndrome_zero=True,
            syndrome_weight=0,
            iterations=1,
            ber_estimate=0.0,
            status=EpistemicStatus.VALIDATED,
            validation_detail="Zero parity syndrome"
        )]
        framing_val = dsp.FrameAnalysisResult(
            status=EpistemicStatus.VALIDATED,
            frame_structure_detected=True,
            frame_type="CCSDS",
            sync_pattern_name="Barker-13",
            sync_pattern_bits="1111100110101",
            frame_length=64,
            repeated_frames_found=3,
            bit_offset=0,
            header_bits=None,
            payload_bits=None,
            crc_profile="CRC-16-CCITT",
            crc_match=True,
            crc_calculated=0x29B1,
            crc_received=0x29B1,
            recovered_ascii="TEST_VALIDATED",
            recovered_hex="54 45 53 54"
        )

        report_val = dsp.fuse_evidence(
            raw_params={"fc_peak_hz": 1000.0, "bw_99pct_hz": 2000.0, "envelope_variance_ratio": 0.05, "snr_db": 25.0},
            hypothesis=hyp,
            sync_symbols=sync,
            demod_result=demod,
            interleaver_candidates=[],
            fec_candidates=fec_val,
            framing_result=framing_val
        )

        self.assertGreater(len(report_val.validated), 0)
        self.assertEqual(report_val.overall_confidence, ConfidenceLevel.HIGH)
        self.assertIn("FULLY VALIDATED INTELLIGENCE", report_val.overall_verdict)
        self.assertGreater(report_val.numeric_score, 0.75)


if __name__ == "__main__":
    unittest.main()
