import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from dsp.loaders import load_signal_file
from dsp.adaptive_pipeline import run_adaptive_pipeline

samples_dir = os.path.join(PROJECT_ROOT, "verified_samples")

print(f"{'FILE':<24} | {'PROTOCOL':<48} | {'CONF':<5} | {'SNR':<8} | {'EXTRA'}")
print("-" * 115)

for f in sorted(os.listdir(samples_dir)):
    if not f.endswith(".wav"):
        continue
    path = os.path.join(samples_dir, f)
    sig, fs, meta = load_signal_file(path, max_samples=250000)
    res = run_adaptive_pipeline(sig, fs, metadata=meta)
    det = res["autonomous_detection"]
    p = res["parameters"]
    pipe = det.get("extraction_pipeline", "")
    
    if pipe == "pulsed_radar":
        extra = f"PRF={res['specialized_telemetry'].get('radar_prf_hz', 0):.1f} Hz"
    elif pipe == "ook_morse":
        extra = f"{res['specialized_telemetry'].get('morse_wpm', 0):.1f} WPM"
    elif pipe == "analog_wefax":
        extra = f"{res['specialized_telemetry'].get('wefax_scan_rate_lpm', 0):.0f} LPM"
    else:
        extra = f"Baud={p.get('baud_label', 'N/A')}"
        
    print(f"{f:<24} | {det.get('protocol_name', 'Unknown'):<48} | {det.get('confidence', 0)*100:3.0f}% | {p.get('snr_db', 0):+6.1f}dB | {extra}")
