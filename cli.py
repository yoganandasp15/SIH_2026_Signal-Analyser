"""
Command-Line Interface (CLI) for NTRO Automated Signal Analysis
===============================================================
Autonomous defense intelligence CLI for batch processing, headless server
analysis, and zero-configuration RF intercept inspection.
"""

import argparse
import sys
import time
from typing import Dict, Any, Optional
import numpy as np

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from dsp.loaders import load_signal_file
from dsp.adaptive_pipeline import run_adaptive_pipeline
from utils.exporter import export_results_to_json, export_results_to_csv
from utils.synthetic_generator import generate_synthetic_signal, save_synthetic_iq, save_synthetic_wav


def print_cli_summary(results: Dict[str, Any]) -> None:
    """
    Prints an elite, terminal-native defense intelligence report to stdout.
    """
    p = results["parameters"]
    m = results["modulation_classification"]
    pulse = results["pulse_analysis"]
    meta = results["metadata"]
    det = results.get("autonomous_detection", {})
    spec = results.get("specialized_telemetry", {})

    print("\n" + "=" * 78)
    print(" [NTRO] AUTONOMOUS RF/SIGINT INTELLIGENCE ENGINE -- TELEMETRY REPORT")
    print("=" * 78)
    print(f" Source File       : {meta.get('file_name', 'Synthetic In-Memory')}")
    print(f" Container Format  : {meta.get('source_type', 'N/A')} [Auto-Probed: {meta.get('format_auto_detected', False)}]")
    if meta.get("format_detection_reason"):
        print(f" Format Evidence   : {meta.get('format_detection_reason')}")
    print(f" Sampling Rate     : {p.get('sampling_rate_hz', 0):,.1f} Hz ({p.get('sampling_rate_hz', 0)/1e6:.3f} MSPS)")
    if meta.get("fs_detection_reason"):
        print(f" Sample Rate Source: {meta.get('fs_detection_reason')}")
    print(f" Analyzed Duration : {p.get('duration_seconds', 0)*1e3:.2f} ms ({p.get('total_samples_analyzed', 0):,} samples)")
    print(f" DSP Pipeline Lat  : {results.get('execution_time_ms', 0):.2f} ms")
    print("-" * 78)

    print(" AUTONOMOUS IDENTIFICATION & ZERO FALSE-POSITIVE AMC:")
    print(f"   * Protocol Identity   : {det.get('protocol_name', m.get('modulation_type', 'Unknown'))}")
    print(f"   * AMC Confidence      : {det.get('confidence', m.get('confidence', 0.0))*100:.1f}%")
    print(f"   * Modulation Family   : {det.get('modulation_family', m.get('modulation_family', 'N/A'))}")
    print(f"   * Adaptive Pipeline   : {spec.get('extractor_pipeline', 'Base')}")
    
    if det.get("physical_evidence"):
        print("   * Verified Invariants :")
        for ev in det["physical_evidence"]:
            print(f"       [+] {ev}")
    if det.get("rejected_hypotheses"):
        print("   * Rejected Hypotheses :")
        for rej in det["rejected_hypotheses"]:
            print(f"       [-] {rej}")
    print("-" * 78)

    print(" COMMON PHYSICAL RF PARAMETERS:")
    fc = p.get('fc_peak_hz', 0)
    print(f"   * Carrier Peak (fc)   : {fc:+,.2f} Hz ({fc/1e3:+,.2f} kHz)")
    print(f"   * Spectral Centroid   : {p.get('fc_centroid_hz', 0):+,.2f} Hz")
    print(f"   * Bandwidth (-3 dB)   : {p.get('bw_3db_hz', 0):,.2f} Hz ({p.get('bw_3db_hz', 0)/1e3:.2f} kHz)")
    print(f"   * Occupied BW (99%)   : {p.get('bw_99pct_hz', 0):,.2f} Hz ({p.get('bw_99pct_hz', 0)/1e3:.2f} kHz)")
    print(f"   * Estimated SNR       : {p.get('snr_db', 0):+.2f} dB [{p.get('snr_estimation_method', 'N/A')}]")
    print(f"   * Spectral Dynamic Rng: {p.get('spectral_dynamic_range_db', 0):.2f} dB")
    print(f"   * Envelope PAPR       : {p.get('papr_db', 0):.2f} dB")
    print("-" * 78)

    # Specialized Protocol Telemetry Card
    pipeline_name = det.get("extraction_pipeline", "")
    if pipeline_name == "pulsed_radar":
        print(" SPECIALIZED RADAR TELEMETRY:")
        print(f"   * Range Resolution (dR): {spec.get('radar_range_resolution_meters', 0):.2f} m")
        print(f"   * Max Range (R_max)    : {spec.get('radar_max_unambiguous_range_km', 0):,.1f} km")
        print(f"   * Pulse Rep Freq (PRF) : {spec.get('radar_prf_hz', 0):.2f} Hz (PRI: {spec.get('radar_pri_us', 0):,.1f} us)")
        print(f"   * Pulse Width (PW)     : {spec.get('radar_pulse_width_us', 0):,.2f} us (Duty: {spec.get('radar_duty_cycle_pct', 0):.2f}%)")
        print(f"   * In-Pulse Segment SNR : {spec.get('radar_in_pulse_snr_db', 0):+.2f} dB")
        if spec.get("radar_chirp_slope_mhz_per_sec"):
            print(f"   * FMOP Chirp Slope     : {spec.get('radar_chirp_slope_mhz_per_sec', 0):+.3f} MHz/s (R^2 = {spec.get('radar_chirp_r2', 0):.2f})")
    elif pipeline_name == "fsk_detector":
        print(" SPECIALIZED FSK / AFSK TELEMETRY:")
        print(f"   * Mark Frequency (f1)  : {spec.get('fsk_mark_frequency_hz', 0):,.1f} Hz")
        print(f"   * Space Frequency (f2) : {spec.get('fsk_space_frequency_hz', 0):,.1f} Hz")
        print(f"   * Frequency Shift (df) : {spec.get('fsk_frequency_shift_hz', 0):,.1f} Hz")
        print(f"   * Modulation Index (h) : {spec.get('fsk_modulation_index_h', 0):.3f}")
        print(f"   * Symbol Baud Rate     : {p.get('baud_label', 'N/A')}")
    elif pipeline_name == "mfsk_comb":
        print(" SPECIALIZED M-FSK TONE COMB TELEMETRY:")
        print(f"   * Tone Matrix Count    : {spec.get('mfsk_tone_count', 0)} Discrete Tones")
        print(f"   * Tone Spacing (df)    : {spec.get('mfsk_tone_spacing_hz', 0):.2f} Hz")
        print(f"   * Symbol Dwell Time    : {spec.get('mfsk_symbol_dwell_time_ms', 0):.2f} ms")
        print(f"   * Symbol Baud Rate     : {p.get('baud_label', 'N/A')}")
        if spec.get("mfsk_detected_tones_hz"):
            tones_str = ", ".join([f"{t:.1f}" for t in spec['mfsk_detected_tones_hz']])
            print(f"   * Detected Tones (Hz)  : [{tones_str}]")
    elif pipeline_name == "tdma_burst":
        print(" SPECIALIZED TDMA BURST TELEMETRY:")
        print(f"   * Frame Period (T_frame): {spec.get('tdma_frame_period_ms', 0):.3f} ms")
        print(f"   * Timeslot Duration    : {spec.get('tdma_timeslot_duration_ms', 0):.3f} ms")
        print(f"   * Burst Active Duration: {spec.get('tdma_burst_duration_ms', 0):.3f} ms")
        print(f"   * Burst Duty Cycle     : {spec.get('tdma_burst_duty_cycle_pct', 0):.2f}%")
        print(f"   * Gated Payload Baud   : {p.get('baud_label', 'N/A')}")
    elif pipeline_name == "digital_psk_qam":
        print(" SPECIALIZED DIGITAL PSK / QAM TELEMETRY:")
        print(f"   * Constellation Order M: {spec.get('constellation_order_m', 0)}")
        print(f"   * Error Vector Mag(EVM): {spec.get('evm_percent', 0):.2f}%")
        print(f"   * Symbol Clock / Baud  : {p.get('baud_label', 'N/A')}")
    elif pipeline_name == "analog_voice":
        print(" SPECIALIZED ANALOG VOICE / AUDIO TELEMETRY:")
        print(f"   * Formant Frequencies  : {spec.get('voice_formant_frequencies_hz', [])} Hz")
        print(f"   * Dynamic Speech SNR   : {spec.get('voice_dynamic_snr_db', 0):+.2f} dB")
        print(f"   * Telephony Passband   : {spec.get('voice_telephony_bandwidth_hz', 0):,.1f} Hz")
        print("   * Symbol Baud Rate     : Suppressed (Continuous Analog Voice)")
    elif pipeline_name == "satellite_telemetry":
        print(" SPECIALIZED SATELLITE TELEMETRY (PCM/PM over NFM):")
        print(f"   * Satellite Target     : {spec.get('satellite_name', 'Aist 2D (RS-48)')}")
        print(f"   * Subcarrier Frequency : {spec.get('satellite_subcarrier_frequency_hz', 0):,.1f} Hz (Prominence: {spec.get('satellite_subcarrier_prominence_db', 0):.1f} dB)")
        print(f"   * Subcarrier SNR       : {spec.get('satellite_subcarrier_snr_db', 0):+.2f} dB")
        print(f"   * Physical Downlink RF : {spec.get('satellite_downlink_frequency_nominal', '435.315 MHz (UHF)')}")
        print(f"   * Channel Bandwidth    : ~{spec.get('satellite_rf_channel_bandwidth_nominal_khz', 10.0):.1f} kHz NFM")
        print(f"   * Detected Sidebands   : [{spec.get('satellite_sidebands_summary', 'N/A')}]")
        print(f"   * Framing Structure    : {spec.get('satellite_framing_type', 'Packetized Satellite Telemetry')}")
        if spec.get("audio_passband_artifact_detected"):
            print(f"   * Baseband Artifact    : Demodulated audio passband recording; carrier & OBW reflect soundcard filter")
    print("=" * 78 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NTRO Autonomous Signal Intelligence Engine -- SIH26147",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 100% Autonomous Zero-Config Analysis:
  python cli.py -i verified_samples/AIS.wav
  python cli.py -i verified_samples/2G_ALE.wav
  python cli.py -i verified_samples/OTH_SW_Radar.wav

  # Generate & inspect synthetic test signals:
  python cli.py -g --mod QPSK --snr 15 --fs 2000000 --save-synth test_qpsk.iq
        """
    )

    parser.add_argument("--input", "-i", type=str, help="Path to .iq / .dat / .bin / .wav file")
    parser.add_argument("--generate", "-g", action="store_true", help="Generate synthetic test signal")

    parser.add_argument("--fs", type=float, default=None, help="Sampling frequency in Hz (Default: Auto-detected)")
    parser.add_argument("--format", "-f", type=str, default="auto",
                        choices=["auto", "complex64", "cf32", "int16", "cs16", "uint8", "cu8"],
                        help="Raw binary IQ format (Default: Auto-probed)")
    parser.add_argument("--max-samples", "-n", type=int, default=1_000_000, help="Max samples to analyze (default: 1,000,000)")

    parser.add_argument("--mod", type=str, default="QPSK",
                        choices=["BPSK", "QPSK", "16-QAM", "2-FSK", "4-FSK", "AM", "FM", "CW", "PULSED_RADAR"],
                        help="Modulation for synthetic generation (default: QPSK)")
    parser.add_argument("--fc", type=float, default=100_000.0, help="Carrier frequency offset in Hz for synthetic mode")
    parser.add_argument("--baud", type=float, default=25_000.0, help="Baud rate for synthetic digital signals")
    parser.add_argument("--snr", type=float, default=15.0, help="Injected AWGN SNR in dB (default: 15.0 dB)")
    parser.add_argument("--save-synth", type=str, default=None, help="Save generated synthetic signal to path (.iq or .wav)")

    parser.add_argument("--output-json", "-j", type=str, default=None, help="Path to export JSON telemetry report")
    parser.add_argument("--output-csv", "-c", type=str, default=None, help="Path to export CSV summary report")

    args = parser.parse_args()

    if not args.input and not args.generate:
        parser.print_help()
        print("\n[!] Error: Please provide an input file (--input <file>) or use synthetic mode (--generate).")
        return 1

    try:
        if args.generate:
            eff_fs = args.fs if args.fs is not None else 1_000_000.0
            print(f"[*] Generating synthetic {args.mod} signal (SNR={args.snr} dB, Fs={eff_fs/1e6:.2f} MSPS)...")
            signal, clean_sig, meta = generate_synthetic_signal(
                mod_type=args.mod,
                fs=eff_fs,
                fc=args.fc,
                baud_rate=args.baud,
                snr_db=args.snr,
                num_samples=min(args.max_samples, 200_000)
            )
            meta["file_name"] = f"Synthetic_{args.mod}_{args.snr}dB.iq"
            meta["source_type"] = f"Synthetic Generator ({args.mod})"

            if args.save_synth:
                if args.save_synth.lower().endswith(".wav"):
                    save_synthetic_wav(signal, eff_fs, args.save_synth)
                else:
                    save_synthetic_iq(signal, args.save_synth, format_type=args.format if args.format != 'auto' else 'complex64')
                print(f"[+] Saved synthetic capture to: {args.save_synth}")
            fs = eff_fs
        else:
            print(f"[*] Autonomous Ingestion Probing: {args.input}...")
            signal, fs, meta = load_signal_file(
                file_path=args.input,
                sample_rate=args.fs,
                format_type=args.format,
                max_samples=args.max_samples
            )

        print("[*] Executing Adaptive Autonomous Signal Analysis Pipeline...")
        results = run_adaptive_pipeline(signal, fs, metadata=meta)
        print_cli_summary(results)

        if args.output_json:
            export_results_to_json(results, args.output_json)
            print(f"[+] Telemetry report exported to JSON: {args.output_json}")

        if args.output_csv:
            export_results_to_csv(results, args.output_csv)
            print(f"[+] Telemetry summary exported to CSV: {args.output_csv}")

        return 0

    except Exception as e:
        print(f"\n[!] Pipeline Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
