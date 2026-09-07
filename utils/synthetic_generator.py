"""
Synthetic Signal Generation & Test Bench Module
===============================================
Generates mathematically pure, calibrated RF test signals across various
digital and analog modulations with Additive White Gaussian Noise (AWGN)
for automated test verification and benchmarking:
- Digital: BPSK, QPSK, 16-QAM, 2-FSK, 4-FSK
- Analog: AM, FM, CW
- Pulsed: Radar Pulse Trains
"""

from typing import Tuple, Optional
import os
import numpy as np
from scipy.io import wavfile


def add_awgn_noise(signal: np.ndarray, snr_db: float) -> Tuple[np.ndarray, float]:
    """
    Injects complex Additive White Gaussian Noise (AWGN) at the exact target SNR.

    Formula:
    --------
    P_sig = mean(|s|^2)
    SNR_lin = 10^(snr_db / 10)
    P_noise = P_sig / SNR_lin
    noise = sqrt(P_noise / 2) * (randn + j*randn)

    Returns:
    --------
    Tuple[np.ndarray, float]
        (noisy_signal, measured_noise_power)
    """
    sig_power = float(np.mean(np.abs(signal) ** 2))
    if sig_power < 1e-12:
        sig_power = 1.0

    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_lin

    # Complex Gaussian noise (divide variance by 2 across real & imag)
    noise_std = np.sqrt(noise_power / 2.0)
    noise_real = np.random.normal(0.0, noise_std, size=len(signal))
    noise_imag = np.random.normal(0.0, noise_std, size=len(signal))
    noise = noise_real + 1j * noise_imag

    noisy_signal = signal + noise
    return noisy_signal, noise_power


def generate_synthetic_signal(
    mod_type: str = "QPSK",
    fs: float = 1_000_000.0,
    fc: float = 100_000.0,
    baud_rate: float = 25_000.0,
    snr_db: Optional[float] = 20.0,
    num_samples: int = 100_000,
    pulse_width_us: float = 20.0,
    pri_us: float = 100.0
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generates a synthetic complex baseband RF signal.

    Parameters:
    -----------
    mod_type : str
        'BPSK', 'QPSK', '16-QAM', '2-FSK', '4-FSK', 'AM', 'FM', 'CW', 'PULSED_RADAR'
    fs : float
        Sampling rate in Hz (default: 1.0 MHz).
    fc : float
        Carrier frequency offset in Hz (default: 100 kHz).
    baud_rate : float
        Symbol rate in Baud (default: 25 kBaud).
    snr_db : Optional[float]
        Injected AWGN SNR in dB. If None, clean signal is returned.
    num_samples : int
        Number of output samples.
    pulse_width_us : float
        Pulse width for PULSED_RADAR mode (microseconds).
    pri_us : float
        Pulse repetition interval for PULSED_RADAR mode (microseconds).

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, dict]
        (noisy_signal, clean_signal, ground_truth_meta)
    """
    mod = mod_type.upper().strip()
    t = np.arange(num_samples) / fs

    samples_per_symbol = max(2, int(fs / baud_rate)) if baud_rate > 0 else 10
    num_symbols = int(np.ceil(num_samples / samples_per_symbol))

    clean_baseband: np.ndarray

    if mod == "BPSK":
        bits = np.random.randint(0, 2, size=num_symbols)
        symbols = 2 * bits - 1  # [-1, +1]
        raw_sig = np.repeat(symbols, samples_per_symbol)[:num_samples].astype(np.complex64)
        clean_baseband = raw_sig

    elif mod == "QPSK":
        bits_i = np.random.randint(0, 2, size=num_symbols)
        bits_q = np.random.randint(0, 2, size=num_symbols)
        sym_i = (2 * bits_i - 1) / np.sqrt(2.0)
        sym_q = (2 * bits_q - 1) / np.sqrt(2.0)
        symbols = sym_i + 1j * sym_q
        raw_sig = np.repeat(symbols, samples_per_symbol)[:num_samples].astype(np.complex64)
        clean_baseband = raw_sig

    elif mod in ["16-QAM", "16QAM", "QAM"]:
        levels = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)  # Normalized power = 1
        sym_i = np.random.choice(levels, size=num_symbols)
        sym_q = np.random.choice(levels, size=num_symbols)
        symbols = sym_i + 1j * sym_q
        raw_sig = np.repeat(symbols, samples_per_symbol)[:num_samples].astype(np.complex64)
        clean_baseband = raw_sig

    elif mod in ["2-FSK", "2FSK", "FSK"]:
        freq_dev = baud_rate / 2.0
        bits = np.random.randint(0, 2, size=num_symbols)
        freqs = (2 * bits - 1) * freq_dev
        freq_series = np.repeat(freqs, samples_per_symbol)[:num_samples]
        phase = 2.0 * np.pi * np.cumsum(freq_series) / fs
        clean_baseband = np.exp(1j * phase).astype(np.complex64)

    elif mod in ["4-FSK", "4FSK"]:
        freq_dev = baud_rate / 4.0
        syms = np.random.choice([-3, -1, 1, 3], size=num_symbols)
        freq_series = np.repeat(syms * freq_dev, samples_per_symbol)[:num_samples]
        phase = 2.0 * np.pi * np.cumsum(freq_series) / fs
        clean_baseband = np.exp(1j * phase).astype(np.complex64)

    elif mod == "AM":
        # Multi-tone audio message
        fm1, fm2 = 1000.0, 3000.0
        msg = 0.5 * np.sin(2 * np.pi * fm1 * t) + 0.3 * np.cos(2 * np.pi * fm2 * t)
        mod_index = 0.8
        am_envelope = 1.0 + mod_index * msg
        clean_baseband = am_envelope.astype(np.complex64)

    elif mod == "FM":
        fm = 2000.0
        msg = np.sin(2 * np.pi * fm * t)
        freq_dev = 25000.0
        phase = 2.0 * np.pi * (freq_dev / fm) * (1.0 - np.cos(2 * np.pi * fm * t))
        clean_baseband = np.exp(1j * phase).astype(np.complex64)

    elif mod in ["CW", "CONTINUOUS WAVE"]:
        clean_baseband = np.ones(num_samples, dtype=np.complex64)

    elif mod in ["PULSED_RADAR", "RADAR", "PULSE"]:
        pw_samples = max(2, int((pulse_width_us * 1e-6) * fs))
        pri_samples = max(pw_samples + 2, int((pri_us * 1e-6) * fs))
        pulse_train = np.zeros(num_samples, dtype=np.complex64)
        for start_idx in range(0, num_samples, pri_samples):
            end_idx = min(start_idx + pw_samples, num_samples)
            pulse_train[start_idx:end_idx] = 1.0
        clean_baseband = pulse_train

    else:
        raise ValueError(f"Unknown modulation type '{mod_type}'")

    # Modulate onto carrier frequency fc: s_rf[n] = s_bb[n] * exp(j * 2*pi*fc*t)
    if fc != 0.0:
        carrier = np.exp(1j * 2.0 * np.pi * fc * t).astype(np.complex64)
        clean_signal = clean_baseband * carrier
    else:
        clean_signal = clean_baseband

    # Add AWGN Noise if requested
    if snr_db is not None:
        noisy_signal, _ = add_awgn_noise(clean_signal, snr_db)
    else:
        noisy_signal = clean_signal.copy()

    ground_truth = {
        "ground_truth_modulation": mod,
        "ground_truth_fs_hz": float(fs),
        "ground_truth_fc_hz": float(fc),
        "ground_truth_baud_rate": float(baud_rate) if "FSK" in mod or "PSK" in mod or "QAM" in mod else None,
        "ground_truth_snr_db": float(snr_db) if snr_db is not None else "Clean (Inf)",
        "ground_truth_num_samples": num_samples,
        "ground_truth_pulse_width_us": pulse_width_us if "RADAR" in mod or "PULSE" in mod else None,
        "ground_truth_pri_us": pri_us if "RADAR" in mod or "PULSE" in mod else None
    }

    return noisy_signal, clean_signal, ground_truth


def save_synthetic_iq(
    signal: np.ndarray,
    file_path: str,
    format_type: str = "complex64"
) -> str:
    """
    Saves complex signal to raw binary interleaved IQ file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    fmt = format_type.lower()

    if fmt in ["complex64", "cf32", "float32"]:
        interleaved = np.empty(len(signal) * 2, dtype=np.float32)
        interleaved[0::2] = np.real(signal).astype(np.float32)
        interleaved[1::2] = np.imag(signal).astype(np.float32)
        interleaved.tofile(file_path)

    elif fmt in ["int16", "cs16"]:
        # Normalize and scale to signed 16-bit integer range [-32767, 32767]
        max_val = np.max(np.abs(signal)) + 1e-12
        norm_sig = signal / max_val
        interleaved = np.empty(len(signal) * 2, dtype=np.int16)
        interleaved[0::2] = (np.real(norm_sig) * 32767.0).astype(np.int16)
        interleaved[1::2] = (np.imag(norm_sig) * 32767.0).astype(np.int16)
        interleaved.tofile(file_path)

    elif fmt in ["uint8", "cu8"]:
        max_val = np.max(np.abs(signal)) + 1e-12
        norm_sig = signal / max_val
        interleaved = np.empty(len(signal) * 2, dtype=np.uint8)
        interleaved[0::2] = np.clip(np.real(norm_sig) * 127.5 + 127.5, 0, 255).astype(np.uint8)
        interleaved[1::2] = np.clip(np.imag(norm_sig) * 127.5 + 127.5, 0, 255).astype(np.uint8)
        interleaved.tofile(file_path)
    else:
        raise ValueError(f"Unsupported save format '{format_type}'")

    return file_path


def save_synthetic_wav(
    signal: np.ndarray,
    fs: float,
    file_path: str
) -> str:
    """
    Saves complex signal to standard stereo 16-bit PCM WAV (Ch0=I, Ch1=Q).
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    max_val = np.max(np.abs(signal)) + 1e-12
    norm_sig = signal / max_val

    stereo_int16 = np.empty((len(signal), 2), dtype=np.int16)
    stereo_int16[:, 0] = (np.real(norm_sig) * 32767.0).astype(np.int16)
    stereo_int16[:, 1] = (np.imag(norm_sig) * 32767.0).astype(np.int16)

    wavfile.write(file_path, int(fs), stereo_int16)
    return file_path
