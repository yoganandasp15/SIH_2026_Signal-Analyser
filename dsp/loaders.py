"""
Binary I/Q and Audio WAV Signal Ingestion Module
================================================
Handles streaming, memory-bounded parsing of raw interleaved binary I/Q streams
(Float32, Int16, Uint8) and audio WAV captures (Stereo I/Q and Mono analytic).
Features autonomous format probing and sampling rate detection to completely
eliminate manual user selection.
"""

from typing import Tuple, Dict, Any, Optional
import os
import re
import json
import subprocess
import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert


def probe_binary_format(file_path: str, chunk_bytes: int = 65536) -> Tuple[str, float, str]:
    """
    Autonomously probes raw binary SDR files to determine whether the sample encoding is:
    - 'complex64' (float32 I/Q pairs)
    - 'int16' (signed 16-bit integers, cs16)
    - 'uint8' (unsigned 8-bit integers, RTL-SDR cu8)

    Uses companion SigMF metadata, filename hardware/format tokens,
    and baseband smoothness / difference-variance ratio analysis.

    Returns:
    --------
    Tuple[format_name, confidence, reason]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Signal file not found: {file_path}")

    # 1. Check SigMF companion file
    base_no_ext = os.path.splitext(file_path)[0]
    for meta_candidate in [f"{file_path}.sigmf-meta", f"{base_no_ext}.sigmf-meta"]:
        if os.path.exists(meta_candidate):
            try:
                with open(meta_candidate, "r", encoding="utf-8") as f:
                    sigmf = json.load(f)
                dtype = sigmf.get("global", {}).get("core:datatype", "").lower()
                if "cf32" in dtype or "float32" in dtype:
                    return "complex64", 1.0, f"SigMF metadata specification ('{dtype}')"
                elif "ci16" in dtype or "cs16" in dtype or "int16" in dtype:
                    return "int16", 1.0, f"SigMF metadata specification ('{dtype}')"
                elif "cu8" in dtype or "uint8" in dtype:
                    return "uint8", 1.0, f"SigMF metadata specification ('{dtype}')"
            except Exception:
                pass

    # 2. Check filename tokens
    fn = os.path.basename(file_path).lower()
    if any(tag in fn for tag in ["cf32", "float32", "complex64", "_f32"]):
        return "complex64", 0.98, "Filename explicit format tag (cf32/complex64)"
    if any(tag in fn for tag in ["cs16", "int16", "signed16", "_s16", "ci16"]):
        return "int16", 0.98, "Filename explicit format tag (cs16/int16)"
    if any(tag in fn for tag in ["cu8", "uint8", "unsigned8", "_u8", "rtlsdr", "rtl_sdr"]):
        return "uint8", 0.98, "Filename explicit format tag (cu8/RTL-SDR)"

    # 3. Statistical Baseband Smoothness & Smooth Difference-Variance Ratio
    # Real baseband SDR signals have high oversampling -> Var(Delta I) / Var(I) < 1.0
    # Wrong format interpretation yields chaotic byte entropy -> Var(Delta I) / Var(I) ~ 2.0 or NaN
    file_size = os.path.getsize(file_path)
    bytes_to_read = min(chunk_bytes, file_size)
    if bytes_to_read < 32:
        return "complex64", 0.60, "File too small for statistical probe; default complex64"

    scores = {}
    candidates = ["complex64", "int16", "uint8"]

    for candidate in candidates:
        try:
            sample_sig = parse_iq_binary(file_path, format_type=candidate, max_samples=2048)
            real_part = np.real(sample_sig)
            if len(real_part) < 32:
                scores[candidate] = 0.0
                continue

            # Check for NaNs or Infinities
            if np.any(np.isnan(real_part)) or np.any(np.isinf(real_part)):
                scores[candidate] = 0.0
                continue

            # Check maximum magnitude
            max_mag = float(np.max(np.abs(sample_sig)))
            if max_mag > 20.0 or max_mag < 1e-6:
                scores[candidate] = 0.0
                continue

            var_sig = float(np.var(real_part))
            if var_sig < 1e-12:
                scores[candidate] = 0.1
                continue

            diff_var = float(np.var(np.diff(real_part)))
            diff_ratio = diff_var / var_sig

            # Baseband smoothness score: lower diff_ratio is better
            # Ideal physical signal has diff_ratio in [0.01, 1.20]
            if diff_ratio < 1.0:
                score = 0.95 - (diff_ratio * 0.25)
            elif diff_ratio < 1.6:
                score = 0.60 - (diff_ratio - 1.0) * 0.40
            else:
                score = 0.10

            scores[candidate] = max(0.0, score)
        except Exception:
            scores[candidate] = 0.0

    best_cand = max(scores, key=scores.get)
    best_score = scores[best_cand]

    if best_score > 0.40:
        return best_cand, float(np.round(best_score, 2)), f"Statistical baseband continuity & variance ratio ({best_cand})"
    return "complex64", 0.70, "Default fallback to complex64 standard"


def auto_detect_file_sample_rate(file_path: str, default_fs: float = 1_000_000.0) -> Tuple[float, bool, str]:
    """
    Autonomously extracts or deduces sampling rate from:
    1. SigMF companion JSON metadata
    2. RIFF WAV audio header
    3. Filename SDR regex patterns (_2MSPS_, _2.048M_, _1000k_, _rtlsdr_, etc.)
    4. Safe defense SDR fallback
    """
    ext = os.path.splitext(file_path)[1].lower()

    # 1. WAV Header (Exact)
    if ext in [".wav", ".wave"]:
        try:
            fs, _ = wavfile.read(file_path, mmap=True)
            return float(fs), True, f"RIFF WAV Header Sample Rate ({float(fs):,.0f} Hz)"
        except Exception:
            pass

    # 1b. Compressed Audio (MP3, OGG, FLAC, M4A, AAC)
    if ext in [".mp3", ".ogg", ".flac", ".m4a", ".aac", ".wma", ".opus"]:
        try:
            p_probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate", "-of", "default=noprint_wrappers=1:nokey=1", file_path],
                capture_output=True, text=True, timeout=5
            )
            if p_probe.returncode == 0 and p_probe.stdout.strip().isdigit():
                fs = float(p_probe.stdout.strip())
                return fs, True, f"Audio Stream Probed Sample Rate ({fs:,.0f} Hz)"
        except Exception:
            pass

    # 2. SigMF metadata
    base_no_ext = os.path.splitext(file_path)[0]
    for meta_candidate in [f"{file_path}.sigmf-meta", f"{base_no_ext}.sigmf-meta"]:
        if os.path.exists(meta_candidate):
            try:
                with open(meta_candidate, "r", encoding="utf-8") as f:
                    sigmf = json.load(f)
                sigmf_fs = sigmf.get("global", {}).get("core:sample_rate")
                if sigmf_fs and float(sigmf_fs) > 0:
                    return float(sigmf_fs), True, f"SigMF specification ('core:sample_rate': {float(sigmf_fs):,.0f} Hz)"
            except Exception:
                pass

    # 3. Filename heuristics
    fn = os.path.basename(file_path)
    # Match patterns like _2.048MSPS_, _2MSPS_, _2.048M_, _2M_
    m_msps = re.search(r'[_.-](\d+(?:\.\d+)?)\s*(?:msps|mhz|m)[_.-]', fn, re.IGNORECASE)
    if m_msps:
        val = float(m_msps.group(1)) * 1e6
        return val, True, f"Filename SDR sample rate token ({val/1e6:.3f} MSPS)"

    # Match patterns like _250kSPS_, _250kHz_, _250k_
    m_ksps = re.search(r'[_.-](\d+(?:\.\d+)?)\s*(?:ksps|khz|k)[_.-]', fn, re.IGNORECASE)
    if m_ksps:
        val = float(m_ksps.group(1)) * 1e3
        return val, True, f"Filename SDR sample rate token ({val/1e3:.1f} kSPS)"

    # Match patterns like _1000000Hz_
    m_hz = re.search(r'[_.-](\d+)\s*hz[_.-]', fn, re.IGNORECASE)
    if m_hz:
        val = float(m_hz.group(1))
        return val, True, f"Filename explicit sample rate token ({val:,.0f} Hz)"

    # SDR Hardware Preset Tags
    fn_lower = fn.lower()
    if "rtlsdr" in fn_lower or "rtl_sdr" in fn_lower or "rtl" in fn_lower:
        return 2_048_000.0, True, "Inferred from RTL-SDR standard hardware profile (2.048 MSPS)"
    if "hackrf" in fn_lower:
        return 10_000_000.0, True, "Inferred from HackRF standard hardware profile (10.0 MSPS)"
    if "airspy" in fn_lower:
        return 2_500_000.0, True, "Inferred from Airspy standard hardware profile (2.5 MSPS)"
    if "usrp" in fn_lower or "b210" in fn_lower:
        return 20_000_000.0, True, "Inferred from USRP B210 standard profile (20.0 MSPS)"

    # 4. Fallback
    return default_fs, False, f"Default standard SDR baseline ({default_fs/1e6:.1f} MSPS)"


def parse_iq_binary(
    file_path: str,
    format_type: str = "complex64",
    max_samples: int = 1_000_000,
    offset_samples: int = 0
) -> np.ndarray:
    """
    Parses raw binary interleaved I/Q data with bounded memory usage.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Signal file not found: {file_path}")

    fmt = format_type.lower().strip()
    bytes_per_sample_element: int
    raw_dtype: Any

    if fmt in ["complex64", "cf32", "float32"]:
        raw_dtype = np.float32
        bytes_per_sample_element = 4
    elif fmt in ["int16", "cs16", "signed16"]:
        raw_dtype = np.int16
        bytes_per_sample_element = 2
    elif fmt in ["uint8", "cu8", "unsigned8"]:
        raw_dtype = np.uint8
        bytes_per_sample_element = 1
    else:
        raise ValueError(
            f"Unsupported format '{format_type}'. Supported: 'complex64', 'int16', 'uint8'"
        )

    bytes_per_complex_sample = 2 * bytes_per_sample_element
    file_size_bytes = os.path.getsize(file_path)
    total_complex_samples = file_size_bytes // bytes_per_complex_sample

    if total_complex_samples == 0:
        raise ValueError(f"File '{file_path}' has insufficient bytes for even 1 complex sample.")

    if offset_samples >= total_complex_samples:
        raise ValueError(
            f"Offset {offset_samples} exceeds total complex samples ({total_complex_samples})."
        )

    if max_samples is not None:
        samples_to_read = min(max_samples, total_complex_samples - offset_samples)
    else:
        samples_to_read = total_complex_samples - offset_samples

    byte_offset = offset_samples * bytes_per_complex_sample
    elements_to_read = samples_to_read * 2

    with open(file_path, "rb") as f:
        f.seek(byte_offset)
        raw_data = np.fromfile(f, dtype=raw_dtype, count=elements_to_read)

    if len(raw_data) < 2:
        raise ValueError(f"Failed to read valid interleaved IQ data from {file_path}")

    if len(raw_data) % 2 != 0:
        raw_data = raw_data[:-1]

    i_raw = raw_data[0::2]
    q_raw = raw_data[1::2]

    # Normalize to continuous float range [-1.0, 1.0]
    if fmt in ["complex64", "cf32", "float32"]:
        i_comp = i_raw.astype(np.float32)
        q_comp = q_raw.astype(np.float32)
    elif fmt in ["int16", "cs16", "signed16"]:
        i_comp = (i_raw.astype(np.float32)) / 32768.0
        q_comp = (q_raw.astype(np.float32)) / 32768.0
    elif fmt in ["uint8", "cu8", "unsigned8"]:
        i_comp = (i_raw.astype(np.float32) - 127.5) / 127.5
        q_comp = (q_raw.astype(np.float32) - 127.5) / 127.5

    complex_signal = i_comp + 1j * q_comp
    return complex_signal


def load_wav_signal(
    file_path: str,
    max_samples: int = 1_000_000,
    offset_samples: int = 0,
    return_meta: bool = False
) -> Any:
    """
    Loads an audio WAV file.
    - If stereo (2 channels): Maps Channel 0 -> I, Channel 1 -> Q.
    - If mono (1 channel): Generates analytic complex baseband via Hilbert transform.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"WAV file not found: {file_path}")

    fs, audio_data = wavfile.read(file_path)

    if np.issubdtype(audio_data.dtype, np.integer):
        max_int = np.iinfo(audio_data.dtype).max
        audio_data = audio_data.astype(np.float32) / float(max_int)
    else:
        audio_data = audio_data.astype(np.float32)

    total_samples = audio_data.shape[0]
    if offset_samples >= total_samples:
        raise ValueError(
            f"Offset {offset_samples} exceeds total WAV samples ({total_samples})."
        )

    end_idx = min(offset_samples + max_samples, total_samples) if max_samples is not None else total_samples
    audio_slice = audio_data[offset_samples:end_idx]

    is_demod = True
    if audio_slice.ndim == 2 and audio_slice.shape[1] >= 2:
        c0 = audio_slice[:, 0]
        c1 = audio_slice[:, 1]
        std0 = float(np.std(c0))
        std1 = float(np.std(c1))

        if std0 > 1e-6 and std1 > 1e-6:
            corr = float(np.corrcoef(c0, c1)[0, 1])
        else:
            corr = 1.0

        if corr > 0.75:
            # Dual-mono audio: merge channels and compute complex analytic signal via Hilbert
            mono_signal = (c0 + c1) / 2.0
            complex_signal = hilbert(mono_signal)
            is_demod = True
        else:
            # Quadrature I/Q channels
            complex_signal = c0 + 1j * c1
            is_demod = False
    elif audio_slice.ndim == 1 or (audio_slice.ndim == 2 and audio_slice.shape[1] == 1):
        mono_signal = audio_slice.flatten()
        complex_signal = hilbert(mono_signal)
        is_demod = True
    else:
        raise ValueError(f"Unsupported audio channel shape: {audio_slice.shape}")

    if return_meta:
        return complex_signal, int(fs), is_demod
    return complex_signal, int(fs)


def load_compressed_audio(
    file_path: str,
    max_samples: int = 1_000_000,
    offset_samples: int = 0
) -> Tuple[np.ndarray, int]:
    """
    Autonomously decodes compressed audio (.mp3, .ogg, .flac, .m4a, .aac, .wma, .opus)
    directly to 16-bit PCM using ffmpeg streaming to stdout.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # 1. Probe native sample rate with ffprobe
    fs = 44100
    try:
        p_probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate", "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            capture_output=True, text=True, timeout=5
        )
        if p_probe.returncode == 0 and p_probe.stdout.strip().isdigit():
            fs = int(p_probe.stdout.strip())
    except Exception:
        pass

    # 2. Decode to raw signed 16-bit PCM, 2 channels
    cmd = ["ffmpeg", "-y", "-i", file_path, "-f", "s16le", "-ac", "2", "-ar", str(fs), "-"]
    p = subprocess.run(cmd, capture_output=True, timeout=30)
    if p.returncode != 0 or len(p.stdout) == 0:
        raise RuntimeError(f"FFmpeg failed to decode audio file '{file_path}': {p.stderr.decode('utf-8', errors='ignore')[:200]}")

    raw_pcm = np.frombuffer(p.stdout, dtype=np.int16).reshape(-1, 2)
    audio_data = raw_pcm.astype(np.float32) / 32768.0

    total_samples = audio_data.shape[0]
    if offset_samples >= total_samples:
        raise ValueError(f"Offset {offset_samples} exceeds total audio samples ({total_samples}).")

    end_idx = min(offset_samples + max_samples, total_samples)
    audio_slice = audio_data[offset_samples:end_idx]

    c0 = audio_slice[:, 0]
    c1 = audio_slice[:, 1]
    std0 = float(np.std(c0))
    std1 = float(np.std(c1))

    if std0 > 1e-6 and std1 > 1e-6:
        corr = float(np.corrcoef(c0, c1)[0, 1])
    else:
        corr = 1.0

    if corr > 0.75:
        mono_signal = (c0 + c1) / 2.0
        complex_signal = hilbert(mono_signal)
    else:
        complex_signal = c0 + 1j * c1

    return complex_signal, int(fs)


def load_signal_file(
    file_path: str,
    sample_rate: Optional[float] = None,
    format_type: str = "auto",
    max_samples: int = 1_000_000,
    offset_samples: int = 0
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Unified, autonomous signal loader.
    Completely eliminates manual format and sample rate selection when defaults
    or 'auto' are used.
    """
    ext = os.path.splitext(file_path)[1].lower()
    file_size = os.path.getsize(file_path)

    # 1. Resolve Sample Rate
    if sample_rate is None or sample_rate <= 0:
        eff_fs, fs_auto, fs_reason = auto_detect_file_sample_rate(file_path)
    else:
        eff_fs = float(sample_rate)
        fs_auto = False
        fs_reason = f"User/Caller explicitly specified {eff_fs:,.0f} Hz"

    # 2. Resolve Format
    fmt_norm = format_type.lower().strip()
    compressed_exts = [".mp3", ".ogg", ".flac", ".m4a", ".aac", ".wma", ".opus"]
    if fmt_norm in ["auto", "none", ""]:
        if ext in [".wav", ".wave"]:
            effective_fmt = "wav"
            fmt_auto = True
            fmt_reason = "RIFF WAV container detected from extension and header"
        elif ext in compressed_exts:
            effective_fmt = ext.lstrip(".")
            fmt_auto = True
            fmt_reason = f"Compressed audio container ({ext}) probed"
        else:
            effective_fmt, fmt_conf, fmt_reason = probe_binary_format(file_path)
            fmt_auto = True
    else:
        effective_fmt = fmt_norm
        fmt_auto = False
        fmt_reason = f"Explicitly configured format '{format_type}'"

    metadata: Dict[str, Any] = {
        "file_name": os.path.basename(file_path),
        "file_path": file_path,
        "file_size_bytes": file_size,
        "extension": ext,
        "format_type": effective_fmt,
        "format_auto_detected": fmt_auto,
        "format_detection_reason": fmt_reason,
        "effective_fs": eff_fs,
        "fs_auto_detected": fs_auto,
        "fs_detection_reason": fs_reason,
        "requested_max_samples": max_samples
    }

    if ext in [".wav", ".wave"]:
        signal, fs, is_audio = load_wav_signal(file_path, max_samples=max_samples, offset_samples=offset_samples, return_meta=True)
        metadata["effective_fs"] = float(fs)
        metadata["is_demodulated_audio"] = is_audio
        metadata["recording_domain"] = "Demodulated Audio Track (Receiver Output)" if is_audio else "Raw RF Baseband I/Q (Quadrature)"
        metadata["source_type"] = "WAV Audio / Baseband"
        metadata["audio_passband_detected"] = bool(is_audio and float(fs) <= 96000.0)
        metadata["receiver_audio_filter_note"] = (
            "Signal is post-demodulation audio; center frequency represents audio pitch/subcarrier, "
            "and 99% OBW reflects receiver/soundcard anti-aliasing filter passband." if is_audio
            else "Signal is direct raw quadrature I/Q baseband."
        )
        metadata["num_samples"] = len(signal)
        metadata["duration_seconds"] = len(signal) / float(fs)
        return signal, float(fs), metadata
    elif ext in compressed_exts:
        signal, fs = load_compressed_audio(file_path, max_samples=max_samples, offset_samples=offset_samples)
        metadata["effective_fs"] = float(fs)
        metadata["is_demodulated_audio"] = True
        metadata["recording_domain"] = f"Demodulated Audio Track ({ext.upper().lstrip('.')})"
        metadata["source_type"] = f"Compressed Audio ({ext.upper().lstrip('.')})"
        metadata["audio_passband_detected"] = bool(float(fs) <= 96000.0)
        metadata["receiver_audio_filter_note"] = (
            "Signal is post-demodulation audio; center frequency represents audio pitch/subcarrier, "
            "and 99% OBW reflects receiver/soundcard filter passband."
        )
        metadata["num_samples"] = len(signal)
        metadata["duration_seconds"] = len(signal) / float(fs)
        return signal, float(fs), metadata
    else:
        signal = parse_iq_binary(
            file_path,
            format_type=effective_fmt,
            max_samples=max_samples,
            offset_samples=offset_samples
        )
        metadata["effective_fs"] = eff_fs
        metadata["is_demodulated_audio"] = False
        metadata["recording_domain"] = f"Raw RF Baseband I/Q ({effective_fmt})"
        metadata["source_type"] = f"Raw IQ Binary ({effective_fmt})"
        metadata["audio_passband_detected"] = False
        metadata["receiver_audio_filter_note"] = "Signal is direct raw quadrature I/Q baseband."
        metadata["num_samples"] = len(signal)
        metadata["duration_seconds"] = len(signal) / float(eff_fs)
        return signal, eff_fs, metadata
