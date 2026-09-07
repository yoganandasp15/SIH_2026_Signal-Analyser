import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from dsp.loaders import load_signal_file
from dsp.autonomous_detector import extract_multi_domain_features

names = ['Morse_Code.wav', 'CODAR.wav', 'D-STAR.wav', 'MFSK16.wav', 'STANAG_4285.wav', 'WEFAX.wav', 'Vario_Voice_Tone.wav']

for name in names:
    p = os.path.join(PROJECT_ROOT, "verified_samples", name)
    sig, fs, _ = load_signal_file(p, max_samples=100000)
    feats = extract_multi_domain_features(sig, fs)
    pw = np.array(feats['pulse_widths_s']) * 1000.0
    p25 = float(np.percentile(pw, 25)) if len(pw) else 0.0
    p75 = float(np.percentile(pw, 75)) if len(pw) else 0.0
    ratio = p75 / (p25 + 1e-6)
    obw = feats['obw_99_hz']
    pk_f = feats['peak_frequency_hz']
    morse_info = feats.get('morse_info')
    
    # Morse check:
    is_morse = (morse_info is not None) or (len(pw) >= 5 and 2.0 <= ratio <= 5.0 and obw <= 1200.0)
    # D-STAR check:
    is_dstar = (7000.0 <= obw <= 13000.0) and (1800.0 <= pk_f <= 3000.0) and feats['sfm'] >= 0.60
    # STANAG check:
    is_stanag = (1800.0 <= obw <= 2800.0) and (1400.0 <= pk_f <= 2500.0) and feats['sfm'] >= 0.50
    # WEFAX check:
    is_wefax = (1500.0 <= obw <= 2600.0) and (1200.0 <= pk_f <= 2200.0) and feats['sfm'] < 0.35
    # MFSK16 check:
    is_mfsk16 = (180.0 <= obw <= 350.0) and (1200.0 <= pk_f <= 1700.0)
    
    print(f"{name:<18}: Morse={is_morse}, DSTAR={is_dstar}, STANAG={is_stanag}, WEFAX={is_wefax}, MFSK16={is_mfsk16}, morse_info={morse_info}")
