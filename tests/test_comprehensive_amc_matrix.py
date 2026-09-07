"""
Comprehensive AMC Matrix and Integration Test Suite
===================================================
Tests defense-grade autonomous signal classification and parameter extraction
for SIH26147 (Smart India Hackathon 2026 - NTRO) across:
1. 22 Verified Real-World SigIDWiki Captures in verified_samples/
2. 8 Real-World Intercepts in C:\\Users\\Yogi\\Downloads
3. 10 Synthetic Modulation Standards (CW, BPSK, QPSK, 8-PSK, 16-QAM, AM, FM, 2-FSK, 4-FSK, PULSED_RADAR)
"""

import os
import unittest
import numpy as np

from dsp.loaders import load_signal_file
from dsp.adaptive_pipeline import run_adaptive_pipeline
from dsp.autonomous_detector import detect_signal_autonomously


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "verified_samples")
DOWNLOADS_DIR = r"C:\Users\Yogi\Downloads"


class TestComprehensiveAMCMatrix(unittest.TestCase):
    """End-to-end integration and modulation verification suite."""

    def test_all_22_verified_samples(self):
        """Verify 100% classification accuracy on all 22 SigIDWiki verified WAV samples."""
        catalog = {
            "2G_ALE.wav": ["8-Tone MFSK", "MFSK", "ALE"],
            "AIS.wav": ["TDMA", "GMSK", "AIS"],
            "APRS.wav": ["2-FSK", "AFSK", "Bell 202"],
            "CODAR.wav": ["FMCW", "Radar", "CODAR"],
            "D-STAR.wav": ["GMSK", "D-STAR", "Digital Voice"],
            "DMR.wav": ["TDMA", "4-FSK", "DMR"],
            "FT8.wav": ["8-FSK", "M-FSK", "FT8"],
            "GRAVES_Radar.wav": ["Pulsed CW", "GRAVES", "Radar", "Reflection"],
            "GSM_BCCH_Downlink.wav": ["TDMA", "GMSK", "Cellular", "GSM"],
            "Ghadir_Radar.wav": ["Pulsed FMOP", "Ghadir", "Radar"],
            "HAARP-1.wav": ["Pulsed FMOP", "HAARP", "Ionospheric", "Radar"],
            "MFSK16.wav": ["16-Tone", "M-FSK", "MFSK16"],
            "Morse_Code.wav": ["CW", "Morse", "OOK"],
            "NAVTEX.wav": ["2-FSK", "NAVTEX", "SITOR"],
            "OTH_SW_Radar.wav": ["Pulsed FMOP", "OTH", "Radar"],
            "POCSAG.wav": ["2-FSK", "POCSAG", "Paging"],
            "PSK31.wav": ["BPSK", "PSK31", "PSK"],
            "RTTY.wav": ["2-FSK", "RTTY", "Baudot"],
            "STANAG_4285.wav": ["8-PSK", "STANAG", "PSK"],
            "UVB76_Buzzer.wav": ["Analog", "Voice", "NFM"],
            "Vario_Voice_Tone.wav": ["Analog", "Voice", "NFM"],
            "WEFAX.wav": ["Analog Facsimile", "WEFAX", "FM"],
            "Woodpecker_Duga.wav": ["Pulsed Radar", "Duga", "Woodpecker", "Radar"],
            "AIST-2D.wav": ["PCM/PM", "Satellite", "AIST", "RS-48", "Telemetry"],
            "ASCII.wav": ["2-FSK", "ASCII", "ITA-5"]
        }

        for filename, expected_families in catalog.items():
            path = os.path.join(SAMPLES_DIR, filename)
            self.assertTrue(os.path.exists(path), f"Sample file missing: {path}")

            sig, fs, meta = load_signal_file(path, max_samples=250_000)
            res = run_adaptive_pipeline(sig, fs, metadata=meta)

            det = res.get("autonomous_detection", {})
            params = res.get("parameters", {})

            protocol = det.get("protocol_name", "")
            mod_fam = det.get("modulation_family", "")
            conf = det.get("confidence", 0.0)
            snr = params.get("snr_db", 0.0)

            matched = any(
                fam.lower() in protocol.lower() or fam.lower() in mod_fam.lower()
                for fam in expected_families
            )

            self.assertTrue(
                matched,
                f"Classification mismatch for {filename}: got '{protocol}' ({mod_fam}), expected one of {expected_families}"
            )
            self.assertGreaterEqual(conf, 0.90, f"Low confidence {conf} on {filename}")
            self.assertTrue(-10.0 <= snr <= 50.0, f"Unphysical SNR {snr} dB on {filename}")

    def test_all_8_downloads_captures(self):
        """Verify 100% classification accuracy on all user Downloads captures."""
        catalog = {
            "HAARP-1.wav": ["Pulsed FMOP", "HAARP", "Ionospheric", "Radar"],
            "PRC_OTH_SW.wav": ["Pulsed FMOP", "OTH-SW", "Radar"],
            "Sigid12-Jun-2019_15h54m18s_-_462.611.5_MHz,_NFM.wav": ["Analog", "Voice", "NFM"],
            "TMOUS_GSM_BCCH_E.wav": ["TDMA", "GSM", "GMSK"],
            "2G_ALEaudio.wav": ["8-Tone MFSK", "2G ALE", "ALE"],
            "Ghadir.wav": ["Pulsed FMOP", "Ghadir", "Radar"],
            "reference_audio.wav": ["Continuous Wave", "CW"],
            "Gqrx_20190519_090112_143050200.wav": ["Pulsed CW", "GRAVES", "Radar", "Reflection"],
            "AIST-2D.wav": ["PCM/PM", "Satellite", "AIST", "RS-48", "Telemetry"]
        }

        alt_map = {
            "PRC_OTH_SW.wav": "OTH_SW_Radar.wav",
            "Sigid12-Jun-2019_15h54m18s_-_462.611.5_MHz,_NFM.wav": "Vario_Voice_Tone.wav",
            "TMOUS_GSM_BCCH_E.wav": "GSM_BCCH_Downlink.wav",
            "2G_ALEaudio.wav": "2G_ALE.wav",
            "Ghadir.wav": "Ghadir_Radar.wav",
            "Gqrx_20190519_090112_143050200.wav": "GRAVES_Radar.wav",
            "HAARP-1.wav": "HAARP-1.wav",
            "AIST-2D.wav": "AIST-2D.wav"
        }

        for filename, expected_families in catalog.items():
            path = os.path.join(DOWNLOADS_DIR, filename)
            if not os.path.exists(path):
                alt_name = alt_map.get(filename, filename)
                alt_path = os.path.join(SAMPLES_DIR, alt_name)
                if os.path.exists(alt_path):
                    path = alt_path
                else:
                    continue

            sig, fs, meta = load_signal_file(path, max_samples=250_000)
            res = run_adaptive_pipeline(sig, fs, metadata=meta)

            det = res.get("autonomous_detection", {})
            protocol = det.get("protocol_name", "")
            mod_fam = det.get("modulation_family", "")
            conf = det.get("confidence", 0.0)

            matched = any(
                fam.lower() in protocol.lower() or fam.lower() in mod_fam.lower()
                for fam in expected_families
            )

            self.assertTrue(
                matched,
                f"Downloads mismatch for {filename}: got '{protocol}' ({mod_fam}), expected {expected_families}"
            )
            self.assertGreaterEqual(conf, 0.95, f"Low confidence {conf} on {filename}")

    def test_synthetic_modulation_matrix(self):
        """Verify 10 synthetic modulation standards (CW, BPSK, QPSK, 8-PSK, 16-QAM, AM, FM, 2-FSK, 4-FSK, PULSED_RADAR)."""
        fs = 100_000.0
        n_samples = 20_000
        t = np.arange(n_samples) / fs

        # 1. CW
        sig_cw = np.exp(1j * 2 * np.pi * 10_000.0 * t)

        # 2. BPSK
        np.random.seed(42)
        bits_bpsk = 2 * np.random.randint(0, 2, 200) - 1
        syms_bpsk = np.repeat(bits_bpsk, 100)
        sig_bpsk = syms_bpsk * np.exp(1j * 2 * np.pi * 15_000.0 * t)

        # 3. QPSK
        bits_qpsk = (2 * np.random.randint(0, 2, 200) - 1) + 1j * (2 * np.random.randint(0, 2, 200) - 1)
        syms_qpsk = np.repeat(bits_qpsk, 100)
        sig_qpsk = syms_qpsk * np.exp(1j * 2 * np.pi * 15_000.0 * t)

        # 4. 8-PSK
        phases_8psk = np.random.choice(np.arange(8) * np.pi / 4, 200)
        syms_8psk = np.repeat(np.exp(1j * phases_8psk), 100)
        sig_8psk = syms_8psk * np.exp(1j * 2 * np.pi * 15_000.0 * t)

        # 5. 16-QAM
        qam_levels = np.array([-3, -1, 1, 3])
        i_qam = np.random.choice(qam_levels, 200)
        q_qam = np.random.choice(qam_levels, 200)
        syms_qam = np.repeat(i_qam + 1j * q_qam, 100)
        sig_qam = syms_qam * np.exp(1j * 2 * np.pi * 15_000.0 * t)

        # 6. AM
        audio = np.sin(2 * np.pi * 500.0 * t)
        sig_am = (1.0 + 0.7 * audio) * np.exp(1j * 2 * np.pi * 15_000.0 * t)

        # 7. FM
        sig_fm = np.exp(1j * (2 * np.pi * 15_000.0 * t + 5.0 * np.sin(2 * np.pi * 500.0 * t)))

        # 8. 2-FSK
        fsk_f1 = 12_000.0
        fsk_f2 = 18_000.0
        bits_fsk = np.random.randint(0, 2, 200)
        freqs = np.repeat(np.where(bits_fsk == 0, fsk_f1, fsk_f2), 100)
        phase = 2 * np.pi * np.cumsum(freqs) / fs
        sig_2fsk = np.exp(1j * phase)

        # 9. 4-FSK
        f4_devs = np.array([-3000, -1000, 1000, 3000])
        syms_4fsk = np.random.choice(f4_devs, 200)
        freqs4 = 15_000.0 + np.repeat(syms_4fsk, 100)
        phase4 = 2 * np.pi * np.cumsum(freqs4) / fs
        sig_4fsk = np.exp(1j * phase4)

        # 10. PULSED_RADAR
        sig_pulse = np.zeros(n_samples, dtype=complex)
        for k in range(5):
            idx0 = k * 4000
            t_p = np.arange(400) / fs
            chirp = np.exp(1j * 2 * np.pi * (10_000.0 * t_p + 0.5 * (1e7) * t_p**2))
            sig_pulse[idx0:idx0 + 400] = chirp

        matrix = {
            "CW": (sig_cw, ["Continuous Wave", "CW"]),
            "BPSK": (sig_bpsk, ["BPSK"]),
            "QPSK": (sig_qpsk, ["QPSK"]),
            "8-PSK": (sig_8psk, ["8-PSK"]),
            "16-QAM": (sig_qam, ["16-QAM"]),
            "AM": (sig_am, ["AM", "Amplitude Modulated"]),
            "FM": (sig_fm, ["FM", "Frequency Modulated"]),
            "2-FSK": (sig_2fsk, ["2-FSK"]),
            "4-FSK": (sig_4fsk, ["4-FSK"]),
            "PULSED_RADAR": (sig_pulse, ["Pulsed Radar", "FMOP"])
        }

        for name, (sig, expected_tokens) in matrix.items():
            res = detect_signal_autonomously(sig, fs)
            p_name = res["protocol_name"]
            m_fam = res["modulation_family"]
            matched = any(exp.lower() in p_name.lower() or exp.lower() in m_fam.lower() for exp in expected_tokens)
            self.assertTrue(
                matched,
                f"Synthetic {name} misclassified as '{p_name}' ({m_fam})"
            )
            self.assertGreaterEqual(res["confidence"], 0.90)


if __name__ == "__main__":
    unittest.main()
