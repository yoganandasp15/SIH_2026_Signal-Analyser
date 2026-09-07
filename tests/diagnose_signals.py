import os
import sys
import numpy as np
from scipy.signal import find_peaks, medfilt
from scipy.ndimage import gaussian_filter1d

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from dsp.loaders import load_signal_file
from dsp.spectral import compute_welch_psd
from dsp.pulse_analyzer import analyze_pulse_train

samples_dir = os.path.join(PROJECT_ROOT, "verified_samples")
test_files = [
    ("HAARP-1", os.path.join(samples_dir, "HAARP-1.wav")),
    ("reference_audio", os.path.join(samples_dir, "Morse_Code.wav")),
    ("Gqrx_GRAVES", os.path.join(samples_dir, "GRAVES_Radar.wav")),
    ("CODAR", os.path.join(samples_dir, "CODAR.wav")),
    ("RTTY", os.path.join(samples_dir, "RTTY.wav")),
    ("MFSK16", os.path.join(samples_dir, "MFSK16.wav")),
    ("WEFAX", os.path.join(samples_dir, "WEFAX.wav")),
    ("Morse_Code", os.path.join(samples_dir, "Morse_Code.wav")),
    ("OTH_SW", os.path.join(samples_dir, "OTH_SW_Radar.wav")),
    ("Ghadir", os.path.join(samples_dir, "Ghadir_Radar.wav")),
    ("2G_ALE", os.path.join(samples_dir, "2G_ALE.wav")),
    ("PSK31", os.path.join(samples_dir, "PSK31.wav"))
]

print(f"{'NAME':<16} | {'FS':<6} | {'OBW99':<8} | {'ENV_VAR':<7} | {'PULSES':<6} | {'DUTY%':<6} | {'CHIRP_R2':<8} | {'CHIRP_SLOPE':<12} | {'SQ_PROM':<8} | {'C42':<6}")
print("-" * 115)

for name, path in test_files:
    if not os.path.exists(path):
        print(f"{name:<16} | MISSING: {path}")
        continue
    sig, fs, meta = load_signal_file(path, max_samples=150000)
    
    # Envelope
    env = np.abs(sig)
    env_var = float(np.std(env) / (np.mean(env) + 1e-12))
    
    # PSD
    f_s, psd_db, psd_lin = compute_welch_psd(sig, fs, nperseg=2048)
    cum_pwr = np.cumsum(psd_lin)
    tot = cum_pwr[-1] + 1e-18
    lo = int(np.searchsorted(cum_pwr, 0.005 * tot))
    hi = min(len(f_s) - 1, int(np.searchsorted(cum_pwr, 0.995 * tot)))
    obw = float(abs(f_s[hi] - f_s[lo]))
    
    # Pulse
    pulse = analyze_pulse_train(sig, fs)
    npulses = pulse.get("num_pulses", 0)
    duty = pulse.get("duty_cycle_pct", 100.0)
    
    # Chirp / IF
    ph = np.unwrap(np.angle(sig[:32768]))
    inst_f = np.diff(ph) * (fs / (2.0 * np.pi))
    inst_f_f = medfilt(inst_f, 5)
    t_ax = np.arange(len(inst_f_f)) / fs
    p = np.polyfit(t_ax, inst_f_f, 1)
    fit = np.polyval(p, t_ax)
    ss_res = np.sum((inst_f_f - fit)**2)
    ss_tot = np.sum((inst_f_f - np.mean(inst_f_f))**2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12)) if ss_tot > 1e-6 else 0.0
    slope = p[0]
    
    # Squaring
    sig_sub = sig[:32768]
    spec2 = np.abs(np.fft.fft(sig_sub ** 2, n=65536))
    f2 = np.fft.fftfreq(65536, d=1.0/fs)
    mask = f2 > 50.0
    pk2_prom = float(np.max(spec2[mask]) / (np.median(spec2[mask]) + 1e-12)) if np.any(mask) else 0.0
    
    # C42
    y_norm = sig - np.mean(sig)
    pwr = np.mean(np.abs(y_norm)**2)
    if pwr > 1e-12: y_norm = y_norm / np.sqrt(pwr)
    mu_20 = np.mean(y_norm**2)
    mu_21 = np.mean(np.abs(y_norm)**2)
    mu_42 = np.mean((np.abs(y_norm)**2) * (y_norm**2))
    c42 = float(np.real(mu_42 - (np.abs(mu_20)**2) - 2.0*(mu_21**2)))
    
    print(f"{name:<16} | {fs:<6.0f} | {obw:<8.1f} | {env_var:<7.3f} | {npulses:<6} | {duty:<6.1f} | {r2:<8.3f} | {slope/1e3:<12.2f} | {pk2_prom:<8.1f} | {c42:<6.2f}")
