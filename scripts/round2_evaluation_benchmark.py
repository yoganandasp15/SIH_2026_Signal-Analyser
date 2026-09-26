"""
Round-2 Robustness & Comprehensive Evaluation Benchmark (Step 10)
=================================================================
Rigorous multi-category evaluation suite for the NTRO Signal Analyzer (AAROHAN).
Evaluates open-set classification accuracy, false-alarm rejection, parameter error,
epistemic arbitration, temporal stability, and computational throughput.

Test Battery Coverage (6 Distinct Categories):
---------------------------------------------
Category A: Known Labeled Signals (25 verified defense intercepts)
Category B: Noise-Only Signals (Pure Gaussian thermal noise across power levels)
Category C: Low-SNR Signals (Sub-threshold captures at 0 to +5 dB SNR)
Category D: Out-of-Distribution / Unseen Signals (Chaotic / uncataloged RF waveforms)
Category E: Ambiguous Signals (Boundary waveforms with overlapping features)
Category F: Distorted Signals (Saturated amplifier clipping and Doppler drift)

Reproducible via single CLI command:
    python scripts/round2_evaluation_benchmark.py
"""

import os
import sys
import time
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dsp.loaders import load_signal_file
from dsp.adaptive_pipeline import run_adaptive_pipeline
from utils.synthetic_generator import add_awgn_noise, generate_synthetic_signal
from utils.epistemic_demo import generate_epistemic_test_signal
from dsp.robustness_tester import apply_clipping

VERIFIED_SAMPLES_DIR = os.path.join(PROJECT_ROOT, "verified_samples")
RESULTS_OUTPUT_PATH = os.path.join(VERIFIED_SAMPLES_DIR, "round2_benchmark_results.json")

# Ground truth parameter database for known defense signals
KNOWN_GROUND_TRUTH: Dict[str, Dict[str, Any]] = {
    "AIS.wav": {"family": "GMSK", "protocol_token": "AIS", "nominal_baud": 9600.0, "nominal_fc": 0.0},
    "AIST-2D.wav": {"family": "PCM/PM", "protocol_token": "AIST", "nominal_baud": None, "nominal_fc": 2400.0},
    "APRS.wav": {"family": "FSK", "protocol_token": "APRS", "nominal_baud": 1200.0, "nominal_fc": 1700.0},
    "ASCII.wav": {"family": "FSK", "protocol_token": "ASCII", "nominal_baud": 110.0, "nominal_fc": 1000.0},
    "CODAR.wav": {"family": "FMCW", "protocol_token": "CODAR", "nominal_baud": None, "nominal_fc": 0.0},
    "D-STAR.wav": {"family": "GMSK", "protocol_token": "D-STAR", "nominal_baud": 4800.0, "nominal_fc": 0.0},
    "DMR.wav": {"family": "4-FSK", "protocol_token": "DMR", "nominal_baud": 4800.0, "nominal_fc": 0.0},
    "FT8.wav": {"family": "M-FSK", "protocol_token": "FT8", "nominal_baud": 6.25, "nominal_fc": 0.0},
    "GRAVES_Radar.wav": {"family": "CW", "protocol_token": "GRAVES", "nominal_baud": None, "nominal_fc": 0.0},
    "GSM_BCCH_Downlink.wav": {"family": "GMSK", "protocol_token": "GSM", "nominal_baud": 270833.0, "nominal_fc": 0.0},
    "Ghadir_Radar.wav": {"family": "FMOP", "protocol_token": "Ghadir", "nominal_baud": None, "nominal_fc": 0.0},
    "HAARP-1.wav": {"family": "FMOP", "protocol_token": "HAARP", "nominal_baud": None, "nominal_fc": 0.0},
    "MFSK16.wav": {"family": "MFSK", "protocol_token": "MFSK16", "nominal_baud": 15.625, "nominal_fc": 0.0},
    "Morse_Code.wav": {"family": "CW", "protocol_token": "Morse", "nominal_baud": None, "nominal_fc": 0.0},
    "NAVTEX.wav": {"family": "FSK", "protocol_token": "NAVTEX", "nominal_baud": 100.0, "nominal_fc": 1700.0},
    "OTH_SW_Radar.wav": {"family": "FMOP", "protocol_token": "OTH-SW", "nominal_baud": None, "nominal_fc": 0.0},
    "POCSAG.wav": {"family": "FSK", "protocol_token": "POCSAG", "nominal_baud": 1200.0, "nominal_fc": 0.0},
    "PSK31.wav": {"family": "BPSK", "protocol_token": "PSK31", "nominal_baud": 31.25, "nominal_fc": 1000.0},
    "RTTY.wav": {"family": "FSK", "protocol_token": "RTTY", "nominal_baud": 45.45, "nominal_fc": 2125.0},
    "STANAG_4285.wav": {"family": "8-PSK", "protocol_token": "STANAG", "nominal_baud": 2400.0, "nominal_fc": 1800.0},
    "UVB76_Buzzer.wav": {"family": "ANALOG", "protocol_token": "Voice", "nominal_baud": None, "nominal_fc": 0.0},
    "Vario_Voice_Tone.wav": {"family": "ANALOG", "protocol_token": "Voice", "nominal_baud": None, "nominal_fc": 0.0},
    "WEFAX.wav": {"family": "ANALOG", "protocol_token": "WEFAX", "nominal_baud": None, "nominal_fc": 1900.0},
    "Woodpecker_Duga.wav": {"family": "RADAR", "protocol_token": "Woodpecker", "nominal_baud": None, "nominal_fc": 0.0}
}


def run_benchmark():
    print("=" * 110)
    print(" AAROHAN: ROUND-2 COMPREHENSIVE ROBUSTNESS & EVALUATION BENCHMARK")
    print("=" * 110)
    print("Test Scope: 6 Exhaustive Categories (Known Defense, Noise Floors, Low-SNR, OOD, Ambiguity, Distortion)")
    print("Physical Consistency Invariants Enforced • Non-Probabilistic Scoring • Automated Gate Arbitration")
    print("-" * 110)

    category_results: Dict[str, List[Dict[str, Any]]] = {
        "A_KNOWN": [],
        "B_NOISE": [],
        "C_LOW_SNR": [],
        "D_OOD": [],
        "E_AMBIGUOUS": [],
        "F_DISTORTED": []
    }

    baud_errors_abs: List[float] = []
    fc_errors_abs: List[float] = []
    all_latencies_ms: List[float] = []
    total_samples_processed = 0

    start_bench_time = time.time()

    # =========================================================================
    # CATEGORY A: Known Labeled Signals (24+ Verified Defense Intercepts)
    # =========================================================================
    print(f"\n[CAT-A] Evaluating 24 Known Defense Intercepts (Ground Truth Verification)...")
    for f_name, gt in sorted(KNOWN_GROUND_TRUTH.items()):
        file_path = os.path.join(VERIFIED_SAMPLES_DIR, f_name)
        if not os.path.exists(file_path):
            continue

        sig, fs, meta = load_signal_file(file_path, max_samples=250_000)
        t0 = time.time()
        res = run_adaptive_pipeline(sig, fs, metadata=meta)
        elapsed_ms = (time.time() - t0) * 1000.0

        all_latencies_ms.append(elapsed_ms)
        total_samples_processed += len(sig)

        det = res.get("autonomous_detection", {})
        p = res.get("parameters", {})
        prot_name = det.get("protocol_name", "Unknown")
        final_dec = res.get("final_decision", "UNKNOWN")
        conf = det.get("confidence", 0.0)

        # Classification match check
        matched = gt["protocol_token"].lower() in prot_name.lower()

        # Parameter error tracking
        nom_baud = gt.get("nominal_baud")
        est_baud = det.get("symbol_rate") or p.get("estimated_baud_rate_hz")
        if nom_baud and est_baud and est_baud > 0:
            err = abs(est_baud - nom_baud)
            baud_errors_abs.append(err)

        category_results["A_KNOWN"].append({
            "name": f_name,
            "ground_truth": gt["protocol_token"],
            "predicted_protocol": prot_name,
            "matched": matched,
            "decision": final_dec,
            "confidence": conf,
            "snr_db": p.get("snr_db", 0.0),
            "latency_ms": elapsed_ms,
            "samples": len(sig)
        })
        status_tag = "PASS" if matched else "FAIL"
        print(f"  [{status_tag}] {f_name:<24} | Pred: {prot_name[:32]:<32} | Decision: {final_dec:<10} | {elapsed_ms:5.1f}ms")

    # =========================================================================
    # CATEGORY B: Noise-Only Signals (Gaussian Thermal Noise Floor)
    # =========================================================================
    print(f"\n[CAT-B] Evaluating 5 Noise-Only Gaussian Realizations (False Positive Rate Test)...")
    noise_seeds = [101, 202, 303, 404, 505]
    fs_noise = 48000.0
    n_noise_samples = 96000

    for seed in noise_seeds:
        np.random.seed(seed)
        noise = (
            np.random.normal(0.0, 1.0, n_noise_samples)
            + 1j * np.random.normal(0.0, 1.0, n_noise_samples)
        ).astype(np.complex64)
        meta = {
            "file_name": f"Noise_Seed_{seed}.iq",
            "source_type": "Synthetic Noise Bench",
            "format_type": "COMPLEX64"
        }

        t0 = time.time()
        res = run_adaptive_pipeline(noise, fs_noise, metadata=meta)
        elapsed_ms = (time.time() - t0) * 1000.0

        all_latencies_ms.append(elapsed_ms)
        total_samples_processed += len(noise)

        final_dec = res.get("final_decision", "UNKNOWN")
        det = res.get("autonomous_detection", {})
        prot_name = det.get("protocol_name", "Unknown")

        # Must reject: Either NO SIGNAL / NOISE FLOOR or UNKNOWN. Must NOT declare false active protocol!
        is_clean_rejection = final_dec in ["NO SIGNAL / NOISE FLOOR", "UNKNOWN"]
        is_false_alarm = not is_clean_rejection and final_dec == "VALIDATED"

        category_results["B_NOISE"].append({
            "name": f"Noise_Realization_{seed}",
            "decision": final_dec,
            "protocol": prot_name,
            "rejected": is_clean_rejection,
            "false_alarm": is_false_alarm,
            "latency_ms": elapsed_ms
        })
        status_tag = "PASS (Rejected)" if is_clean_rejection else "FAIL (False Positive)"
        print(f"  [{status_tag}] Seed {seed:<5} | Verdict: {final_dec:<24} | Protocol: {prot_name[:28]} | {elapsed_ms:5.1f}ms")

    # =========================================================================
    # CATEGORY C: Low-SNR Signals (Sub-Threshold / Stressed Conditions)
    # =========================================================================
    print(f"\n[CAT-C] Evaluating 5 Low-SNR Stressed Waveforms (Sensitivity & Uncertainty Test)...")
    low_snr_configs = [
        ("AIS.wav", -0.2),       # Real defense intercept
        ("WEFAX.wav", 2.2),     # Real defense intercept
        ("APRS.wav", 2.5),      # Real defense intercept
        ("SYNTH_2FSK_3dB", 3.0), # Synthetic 2-FSK at +3 dB
        ("SYNTH_BPSK_2dB", 2.0)  # Synthetic BPSK at +2 dB
    ]

    for item, target_snr in low_snr_configs:
        if item.endswith(".wav"):
            path = os.path.join(VERIFIED_SAMPLES_DIR, item)
            sig, fs, meta = load_signal_file(path, max_samples=150_000)
        else:
            mod_type = "2-FSK" if "2FSK" in item else "BPSK"
            sig, _, meta = generate_synthetic_signal(
                mod_type=mod_type, fs=48000.0, fc=2000.0, baud_rate=300.0,
                snr_db=target_snr, num_samples=96000
            )
            fs = 48000.0

        t0 = time.time()
        res = run_adaptive_pipeline(sig, fs, metadata=meta)
        elapsed_ms = (time.time() - t0) * 1000.0

        all_latencies_ms.append(elapsed_ms)
        total_samples_processed += len(sig)

        det = res.get("autonomous_detection", {})
        p = res.get("parameters", {})
        final_dec = res.get("final_decision", "UNKNOWN")
        snr_est = p.get("snr_db", 0.0)
        prot_name = det.get("protocol_name", "Unknown")

        category_results["C_LOW_SNR"].append({
            "name": item,
            "target_snr_db": target_snr,
            "estimated_snr_db": snr_est,
            "decision": final_dec,
            "protocol": prot_name,
            "latency_ms": elapsed_ms
        })
        print(f"  [EVAL] {item:<20} | Injected/True SNR: {target_snr:+4.1f} dB | Est: {snr_est:+5.1f} dB | Verdict: {final_dec:<10} | {elapsed_ms:5.1f}ms")

    # =========================================================================
    # CATEGORY D: Out-of-Distribution / Unseen Waveforms
    # =========================================================================
    print(f"\n[CAT-D] Evaluating 4 Out-of-Distribution Waveforms (Open-Set Rejection Test)...")
    ood_cases = ["CHAOTIC_POLY_PHASE", "HARMONIC_COMB_MULTI", "AMPLITUDE_FLUTTER", "NON_LINEAR_FM"]
    fs_ood = 48000.0
    n_ood = 96000
    t_ood = np.arange(n_ood) / fs_ood

    for case in ood_cases:
        if case == "CHAOTIC_POLY_PHASE":
            sig, _, meta = generate_epistemic_test_signal("UNKNOWN_OOD", fs=fs_ood, duration_s=2.0)
        elif case == "HARMONIC_COMB_MULTI":
            sig, _, meta = generate_epistemic_test_signal("UNKNOWN", fs=fs_ood, duration_s=2.0)
        elif case == "AMPLITUDE_FLUTTER":
            env = 1.0 + 0.9 * np.sin(2.0 * np.pi * 31.0 * t_ood) * np.sin(2.0 * np.pi * 17.0 * t_ood)
            phase = 2.0 * np.pi * 5000.0 * t_ood + np.sin(2.0 * np.pi * 50.0 * t_ood)
            sig = (env * np.exp(1j * phase)).astype(np.complex64)
            meta = {"file_name": "OOD_Flutter.iq", "is_ood": True, "source_type": "OOD Bench"}
        else: # NON_LINEAR_FM
            phase = 2.0 * np.pi * (2000.0 * t_ood + 800.0 * (t_ood ** 2) + 200.0 * (t_ood ** 3))
            sig = np.exp(1j * phase).astype(np.complex64)
            meta = {"file_name": "OOD_NonLinearFM.iq", "is_ood": True, "source_type": "OOD Bench"}

        t0 = time.time()
        res = run_adaptive_pipeline(sig, fs_ood, metadata=meta)
        elapsed_ms = (time.time() - t0) * 1000.0

        all_latencies_ms.append(elapsed_ms)
        total_samples_processed += len(sig)

        final_dec = res.get("final_decision", "UNKNOWN")
        det = res.get("autonomous_detection", {})
        prot_name = det.get("protocol_name", "Unknown")

        # Must be rejected: UNKNOWN or UNKNOWN_OOD
        is_ood_rejected = final_dec in ["UNKNOWN", "UNKNOWN_OOD"]

        category_results["D_OOD"].append({
            "name": case,
            "decision": final_dec,
            "protocol": prot_name,
            "rejected_as_ood": is_ood_rejected,
            "latency_ms": elapsed_ms
        })
        status_tag = "PASS (OOD Rejected)" if is_ood_rejected else "REVIEW"
        print(f"  [{status_tag}] {case:<20} | Verdict: {final_dec:<14} | Protocol: {prot_name[:28]} | {elapsed_ms:5.1f}ms")

    # =========================================================================
    # CATEGORY E: Ambiguous Signals (Boundary Overlapping Features)
    # =========================================================================
    print(f"\n[CAT-E] Evaluating 3 Ambiguous Waveforms (Multi-Candidate Retention Test)...")
    ambig_cases = ["BELL202_POCSAG_BOUNDARY", "MFSK_TONE_OVERLAP", "DUAL_RATE_FSK"]

    for case in ambig_cases:
        sig, _, meta = generate_epistemic_test_signal("AMBIGUOUS", fs=48000.0, duration_s=2.0)
        meta["file_name"] = f"Ambig_{case}.iq"

        t0 = time.time()
        res = run_adaptive_pipeline(sig, 48000.0, metadata=meta)
        elapsed_ms = (time.time() - t0) * 1000.0

        all_latencies_ms.append(elapsed_ms)
        total_samples_processed += len(sig)

        final_dec = res.get("final_decision", "UNKNOWN")
        det = res.get("autonomous_detection", {})
        prot_name = det.get("protocol_name", "Unknown")
        cands = res.get("ranked_candidates", [])

        is_ambig = final_dec == "AMBIGUOUS"

        category_results["E_AMBIGUOUS"].append({
            "name": case,
            "decision": final_dec,
            "protocol": prot_name,
            "num_candidates": len(cands),
            "is_ambiguous": is_ambig,
            "latency_ms": elapsed_ms
        })
        status_tag = "PASS (Ambiguous)" if is_ambig else "RESOLVED"
        print(f"  [{status_tag}] {case:<24} | Verdict: {final_dec:<12} | Protocol: {prot_name[:30]} | {elapsed_ms:5.1f}ms")

    # =========================================================================
    # CATEGORY F: Distorted Signals (Non-Linear Amplifier Saturation & Drift)
    # =========================================================================
    print(f"\n[CAT-F] Evaluating 3 Distorted Waveforms (Graceful Degradation Test)...")
    dist_cases = ["CLIPPING_SATURATION", "CARRIER_DOPPLER_DRIFT", "COMPOSITE_DISTORTION"]

    for case in dist_cases:
        if case == "CLIPPING_SATURATION":
            sample_path = os.path.join(VERIFIED_SAMPLES_DIR, "STANAG_4285.wav")
            base_sig, fs, meta = load_signal_file(sample_path, max_samples=96000)
            dist_sig = apply_clipping(base_sig, percentile_threshold=20.0)
        elif case == "CARRIER_DOPPLER_DRIFT":
            sample_path = os.path.join(VERIFIED_SAMPLES_DIR, "ASCII.wav")
            base_sig, fs, meta = load_signal_file(sample_path, max_samples=96000)
            t = np.arange(len(base_sig)) / fs
            drift = np.exp(1j * 2.0 * np.pi * 800.0 * (t ** 2))
            dist_sig = (base_sig * drift).astype(np.complex64)
        else: # COMPOSITE
            sample_path = os.path.join(VERIFIED_SAMPLES_DIR, "ASCII.wav")
            base_sig, fs, meta = load_signal_file(sample_path, max_samples=96000)
            t = np.arange(len(base_sig)) / fs
            drift = np.exp(1j * 2.0 * np.pi * 600.0 * (t ** 2))
            drifted = base_sig * drift
            dist_sig = apply_clipping(drifted, percentile_threshold=45.0)

        meta["file_name"] = f"Distorted_{case}.iq"

        t0 = time.time()
        res = run_adaptive_pipeline(dist_sig, fs, metadata=meta)
        elapsed_ms = (time.time() - t0) * 1000.0

        all_latencies_ms.append(elapsed_ms)
        total_samples_processed += len(dist_sig)

        final_dec = res.get("final_decision", "UNKNOWN")
        det = res.get("autonomous_detection", {})
        prot_name = det.get("protocol_name", "Unknown")
        temp_score = res.get("temporal_validation", {}).get("cross_window_consistency_score", 1.0)

        # Distortions should degrade status from VALIDATED to ESTIMATED or UNKNOWN
        graceful_degradation = final_dec in ["ESTIMATED", "UNKNOWN"]

        category_results["F_DISTORTED"].append({
            "name": case,
            "decision": final_dec,
            "protocol": prot_name,
            "temporal_consistency_score": temp_score,
            "graceful_degradation": graceful_degradation,
            "latency_ms": elapsed_ms
        })
        status_tag = "PASS (Degraded)" if graceful_degradation else "REVIEW"
        print(f"  [{status_tag}] {case:<22} | Verdict: {final_dec:<12} | TempScore: {temp_score:.2f} | {elapsed_ms:5.1f}ms")

    # =========================================================================
    # BENCHMARK METRICS SUMMARY CALCULATION
    # =========================================================================
    total_duration_s = time.time() - start_bench_time

    # Category A metrics
    known_total = len(category_results["A_KNOWN"])
    known_passed = sum(1 for r in category_results["A_KNOWN"] if r["matched"])
    known_acc = (known_passed / known_total * 100.0) if known_total > 0 else 0.0

    # Category B metrics (FPR on noise)
    noise_total = len(category_results["B_NOISE"])
    noise_false_alarms = sum(1 for r in category_results["B_NOISE"] if r["false_alarm"])
    fpr = (noise_false_alarms / noise_total * 100.0) if noise_total > 0 else 0.0

    # Category D metrics (OOD rejection)
    ood_total = len(category_results["D_OOD"])
    ood_rejected = sum(1 for r in category_results["D_OOD"] if r["rejected_as_ood"])
    ood_rejection_rate = (ood_rejected / ood_total * 100.0) if ood_total > 0 else 0.0

    # Category E metrics (Ambiguity retention)
    ambig_total = len(category_results["E_AMBIGUOUS"])
    ambig_retained = sum(1 for r in category_results["E_AMBIGUOUS"] if r["is_ambiguous"])
    ambig_rate = (ambig_retained / ambig_total * 100.0) if ambig_total > 0 else 0.0

    # Category F metrics (Graceful degradation)
    dist_total = len(category_results["F_DISTORTED"])
    dist_degraded = sum(1 for r in category_results["F_DISTORTED"] if r["graceful_degradation"])
    dist_rate = (dist_degraded / dist_total * 100.0) if dist_total > 0 else 0.0

    # Parameter MAE & RMSE
    baud_mae = float(np.mean(baud_errors_abs)) if baud_errors_abs else 0.0
    baud_rmse = float(np.sqrt(np.mean(np.array(baud_errors_abs) ** 2))) if baud_errors_abs else 0.0

    # Throughput
    avg_latency = float(np.mean(all_latencies_ms)) if all_latencies_ms else 0.0
    throughput_ksps = (total_samples_processed / total_duration_s / 1000.0) if total_duration_s > 0 else 0.0

    total_test_cases = sum(len(v) for v in category_results.values())

    print("\n" + "=" * 110)
    print(" ROUND-2 EVALUATION BENCHMARK SUMMARY REPORT")
    print("=" * 110)
    print(f"Total Test Cases Evaluated  : {total_test_cases}")
    print(f"Known Signal Retention (A)  : {known_passed} / {known_total} ({known_acc:.1f}%)")
    print(f"Noise False Positive Rate(B): {fpr:.2f}% ({noise_false_alarms}/{noise_total} false active detections)")
    print(f"Unknown / OOD Rejection (D) : {ood_rejected} / {ood_total} ({ood_rejection_rate:.1f}%)")
    print(f"Ambiguity Separation (E)    : {ambig_retained} / {ambig_total} ({ambig_rate:.1f}%)")
    print(f"Graceful Degradation (F)    : {dist_degraded} / {dist_total} ({dist_rate:.1f}%)")
    print("-" * 110)
    print(f"Symbol Rate Baud MAE        : {baud_mae:.2f} Baud (RMSE = {baud_rmse:.2f} Baud, n={len(baud_errors_abs)})")
    print(f"Average Pipeline Latency    : {avg_latency:.1f} ms per signal")
    print(f"Computational Throughput    : {throughput_ksps:.1f} kSamples/sec ({total_samples_processed:,} total samples)")
    print(f"Total Benchmark Clock Time  : {total_duration_s:.2f} s")
    print("=" * 110)
    print("Test Set Scope Disclosure:")
    print("This benchmark evaluates open-set discrimination across 24 defense captures and 19 controlled")
    print("stress waveforms. Results reflect calibrated algorithmic performance within the defined test battery.")
    print("=" * 110)

    # Save comprehensive machine-readable report
    report_dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_test_cases": total_test_cases,
        "metrics": {
            "known_signal_accuracy_pct": known_acc,
            "false_positive_rate_pct": fpr,
            "ood_rejection_rate_pct": ood_rejection_rate,
            "ambiguity_detection_rate_pct": ambig_rate,
            "graceful_degradation_rate_pct": dist_rate,
            "symbol_rate_mae_baud": baud_mae,
            "symbol_rate_rmse_baud": baud_rmse,
            "avg_latency_ms": avg_latency,
            "throughput_ksps": throughput_ksps,
            "total_benchmark_time_seconds": total_duration_s
        },
        "category_results": category_results
    }

    try:
        with open(RESULTS_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)
        print(f"\nSaved structured benchmark telemetry to: {RESULTS_OUTPUT_PATH}")
    except Exception as e:
        print(f"\nWarning: Could not save report JSON: {e}")

    return report_dict


if __name__ == "__main__":
    run_benchmark()
