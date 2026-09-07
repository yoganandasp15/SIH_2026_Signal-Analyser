"""
NTRO Autonomous Signal Intelligence Engine - Input Forensics & Sampling Rate Authority
========================================================================================
Phase 1: Universal Input Layer adhering to NTRO SIH26147 V3 Execution Specification.
Strictly eliminates false claims of blind physical sample rate recovery.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
import numpy as np

from .contracts import SamplingRateStatus, SignalRecord
from .loaders import (
    load_signal_file,
    load_wav_signal,
    parse_iq_binary,
    probe_binary_format
)


def inspect_signal_file(
    file_path: Union[str, Path],
    override_fs: Optional[float] = None,
    max_samples: Optional[int] = None
) -> SignalRecord:
    """
    Universal ingestion with strict epistemic sampling rate authority.
    
    1. Probes WAV RIFF metadata or paired SigMF (.sigmf-meta).
    2. Probes raw binary container types.
    3. Emits SignalRecord with explicit SamplingRateStatus.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Signal file not found: {path}")

    suffix = path.suffix.lower()
    metadata: Dict[str, Any] = {}

    # Check for paired SigMF metadata file (.sigmf-meta)
    sigmf_meta_path = path.with_suffix(".sigmf-meta")
    if sigmf_meta_path.exists():
        try:
            with open(sigmf_meta_path, "r", encoding="utf-8") as f:
                sigmf_data = json.load(f)
                global_meta = sigmf_data.get("global", {})
                sample_rate = global_meta.get("core:sample_rate")
                center_freq = 0.0
                captures = sigmf_data.get("captures", [])
                if captures and isinstance(captures, list):
                    center_freq = captures[0].get("core:frequency", 0.0)
                datatype = global_meta.get("core:datatype", "cf32_le")
                
                # Load binary samples based on SigMF format
                raw_bytes = path.read_bytes()
                if datatype in ("cf32_le", "complex64"):
                    samples = np.frombuffer(raw_bytes, dtype=np.complex64)
                elif datatype in ("ci16_le", "int16"):
                    i16 = np.frombuffer(raw_bytes, dtype=np.int16)
                    samples = (i16[0::2] + 1j * i16[1::2]).astype(np.complex64) / 32768.0
                elif datatype in ("cu8", "uint8"):
                    u8 = np.frombuffer(raw_bytes, dtype=np.uint8)
                    samples = ((u8[0::2].astype(np.float32) - 127.5) + 1j * (u8[1::2].astype(np.float32) - 127.5)) / 128.0
                else:
                    samples = np.frombuffer(raw_bytes, dtype=np.complex64)

                if max_samples:
                    samples = samples[:max_samples]

                return SignalRecord(
                    samples=samples,
                    sample_rate=float(sample_rate) if sample_rate else None,
                    is_normalized=sample_rate is None,
                    sample_rate_status=SamplingRateStatus.VERIFIED_METADATA if sample_rate else SamplingRateStatus.NORMALIZED_DOMAIN,
                    estimation_source="SigMF Metadata JSON",
                    confidence=1.0 if sample_rate else 0.5,
                    center_frequency=float(center_freq),
                    datatype=datatype,
                    source_format="SIGMF",
                    metadata=sigmf_data
                )
        except Exception as e:
            metadata["sigmf_read_error"] = str(e)

    # Check for WAV file
    if suffix in (".wav", ".wave"):
        samples, fs = load_wav_signal(str(path), max_samples=max_samples)
        metadata["format"] = "RIFF_WAV"
        metadata["wav_fs"] = fs
        
        # In WAV, sample rate is an authoritative physical header parameter
        return SignalRecord(
            samples=samples,
            sample_rate=float(fs),
            is_normalized=False,
            rate_status=SamplingRateStatus.VERIFIED_METADATA,
            estimation_source="WAV RIFF Header",
            estimation_confidence=1.0,
            datatype="float32",
            source_format="WAV",
            metadata=metadata
        )

    # Raw binary / IQ / DAT
    # Probe binary format (complex64, int16, uint8)
    detected_dtype = probe_binary_format(str(path))
    metadata["probed_dtype"] = detected_dtype

    if override_fs is not None and override_fs > 0:
        samples, fs_loaded, _ = load_signal_file(str(path), sample_rate=override_fs, max_samples=max_samples)
        return SignalRecord(
            samples=samples,
            sample_rate=float(fs_loaded),
            is_normalized=False,
            rate_status=SamplingRateStatus.INFERRED_PROFILE,
            estimation_source="Operator / Profile Override",
            estimation_confidence=0.90,
            datatype=detected_dtype,
            source_format="RAW_IQ",
            metadata=metadata
        )

    # No metadata available -> Strictly normalized domain
    samples, _, _ = load_signal_file(str(path), sample_rate=None, max_samples=max_samples)
    return SignalRecord(
        samples=samples,
        sample_rate=None,
        is_normalized=True,
        rate_status=SamplingRateStatus.NORMALIZED_DOMAIN,
        estimation_source="None (Normalized Domain ω ∈ [-0.5, +0.5])",
        estimation_confidence=0.50,
        datatype=detected_dtype,
        source_format="RAW_IQ",
        metadata=metadata
    )

