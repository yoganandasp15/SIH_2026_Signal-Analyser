import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from dsp.loaders import load_signal_file
from dsp.adaptive_pipeline import run_adaptive_pipeline

downloads = r"C:\Users\Yogi\Downloads"
samples_dir = os.path.join(PROJECT_ROOT, "verified_samples")
files = [
    "HAARP-1.wav",
    "PRC_OTH_SW.wav",
    "Sigid12-Jun-2019_15h54m18s_-_462.611.5_MHz,_NFM.wav",
    "TMOUS_GSM_BCCH_E.wav",
    "2G_ALEaudio.wav",
    "Ghadir.wav",
    "reference_audio.wav",
    "Gqrx_20190519_090112_143050200.wav",
    "AIST-2D.wav"
]

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

print(f"{'FILE':<40} | {'MODULATION':<35} | {'CONF':<6} | {'SNR':<8} | {'BAUD/PRF':<25}")
print("-" * 120)

for f in files:
    path = os.path.join(downloads, f)
    if not os.path.exists(path):
        alt_path = os.path.join(samples_dir, alt_map.get(f, f))
        if os.path.exists(alt_path):
            path = alt_path
        else:
            print(f"{f:<40} | MISSING")
            continue
    try:
        sig, fs, meta = load_signal_file(path, max_samples=250000)
        res = run_adaptive_pipeline(sig, fs, metadata=meta)
        p = res["parameters"]
        m = res["modulation_classification"]
        det = res.get("autonomous_detection", {})
        spec = res.get("specialized_telemetry", {})
        
        mod_t = det.get("protocol_name", m.get("modulation_type", "Unknown"))
        conf = f"{det.get('confidence', m.get('confidence', 0.0))*100:.0f}%"
        snr = f"{p.get('snr_db', 0):+.1f} dB"
        
        pipe = det.get("extraction_pipeline", "")
        if pipe == "pulsed_radar":
            extra = f"PRF: {spec.get('radar_prf_hz', 0):.1f} Hz"
        elif pipe == "satellite_telemetry":
            extra = f"Sub: {spec.get('satellite_subcarrier_frequency_hz', 0):.1f} Hz"
        else:
            extra = f"Baud: {p.get('baud_label', 'N/A')}"
            
        print(f"{f:<40} | {mod_t:<35} | {conf:<6} | {snr:<8} | {extra:<25}")
        if det.get("physical_evidence"):
            print("   Evidence: " + "; ".join(det["physical_evidence"][:2]))
    except Exception as e:
        print(f"{f:<40} | ERROR: {e}")
