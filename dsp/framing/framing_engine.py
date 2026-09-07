"""
Unified Framing and Payload Analysis Engine
===========================================
Performs generic frame structure discovery:
- Bit periodicity & byte alignment analysis
- Repeated frame length candidates
- Sync-word preamble cross-correlation
- Multi-frame CRC verification across candidate profiles
- Conditional header / payload segmentation and ASCII/Hex recovery
Marks frame structure as DETECTED, NOT_DETECTED, or UNCERTAIN.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from dsp.contracts import (
    FrameAnalysisResult,
    EpistemicStatus
)
from .crc import CRC_PROFILES, verify_crc_stream, compute_crc
from .correlator import correlate_sync_words, PREAMBLE_CATALOG


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Packs bit array into bytes."""
    num_bytes = len(bits) // 8
    if num_bytes == 0:
        return b""
    packed = np.packbits(bits[:num_bytes * 8])
    return bytes(packed)


def format_hex_dump(data_bytes: bytes, max_bytes: int = 64) -> str:
    """Formats byte array into clean hex representation."""
    show_bytes = data_bytes[:max_bytes]
    hex_str = " ".join(f"{b:02X}" for b in show_bytes)
    if len(data_bytes) > max_bytes:
        hex_str += f" ... (+{len(data_bytes) - max_bytes} bytes)"
    return hex_str


def format_ascii_repr(data_bytes: bytes, max_len: int = 128) -> str:
    """Extracts printable ASCII characters or replaces non-printables with dot."""
    chars = []
    for b in data_bytes[:max_len]:
        if 32 <= b <= 126:
            chars.append(chr(b))
        elif b in (10, 13):
            chars.append(" ")
        else:
            chars.append(".")
    return "".join(chars)


def detect_repeated_frames(
    sync_matches: List[Dict[str, Any]],
    tolerance: int = 4
) -> Tuple[int, int]:
    """
    Checks if identical sync words repeat at regular periodic intervals.
    Requires at least 2 consistent intervals (3 occurrences) to confirm periodicity.
    """
    if len(sync_matches) < 3:
        return 0, 0

    by_name: Dict[str, List[int]] = {}
    for m in sync_matches:
        by_name.setdefault(m["name"], []).append(m["offset"])

    for name, offsets in by_name.items():
        if len(offsets) >= 3:
            diffs = np.diff(offsets)
            median_diff = int(np.median(diffs))
            if median_diff >= 32:
                close_count = sum(1 for d in diffs if abs(d - median_diff) <= tolerance)
                if close_count >= 2:
                    return median_diff, close_count + 1

    return 0, 0


def analyze_frames(
    bits: np.ndarray,
    candidate_frame_len: Optional[int] = None
) -> FrameAnalysisResult:
    """
    Analyzes recovered bitstream for framing boundaries, repeated frames, and CRC matches.
    """
    n = len(bits)
    if n < 16:
        return FrameAnalysisResult(
            status=EpistemicStatus.NOT_APPLICABLE,
            frame_structure_detected=False,
            frame_type="Stream Too Short",
            sync_pattern_name=None,
            sync_pattern_bits=None,
            frame_length=0,
            repeated_frames_found=0,
            bit_offset=0,
            header_bits=np.zeros(0, dtype=np.uint8),
            payload_bits=np.zeros(0, dtype=np.uint8),
            crc_profile=None,
            crc_match=False,
            crc_calculated=None,
            crc_received=None,
            recovered_ascii="",
            recovered_hex=""
        )

    # 1. Preamble Correlation
    sync_matches = correlate_sync_words(bits)
    frame_len, repeats = detect_repeated_frames(sync_matches)

    # Candidate offsets to evaluate: offset 0 (direct stream) + sync word offsets
    offsets_to_test = [0]
    for m in sync_matches:
        if m["offset"] not in offsets_to_test:
            offsets_to_test.append(m["offset"])

    best_result: Optional[FrameAnalysisResult] = None

    for start_offset in offsets_to_test:
        active_bits = bits
        matched_sync_name = None
        sync_len = 0

        # Check if this offset corresponds to a sync match
        matching_sync = next((m for m in sync_matches if m["offset"] == start_offset), None)
        if matching_sync:
            matched_sync_name = matching_sync["name"]
            sync_len = matching_sync["length"]
            if matching_sync["inverted"]:
                active_bits = 1 - bits

        cand_lens = [n - start_offset]
        if frame_len > 0 and frame_len not in cand_lens:
            cand_lens.insert(0, frame_len)
        if candidate_frame_len and candidate_frame_len >= 32 and candidate_frame_len not in cand_lens:
            cand_lens.insert(0, candidate_frame_len)

        for cur_frame_len in cand_lens:
            end_offset = min(n, start_offset + cur_frame_len)
            frame_bits = active_bits[start_offset:end_offset]

            if sync_len > 0:
                header_bits = frame_bits[:sync_len]
                payload_bits = frame_bits[sync_len:]
            else:
                header_bits = np.zeros(0, dtype=np.uint8)
                payload_bits = frame_bits

            frame_bytes = bits_to_bytes(frame_bits)
            payload_bytes = bits_to_bytes(payload_bits)

            # Test CRC
            crc_match_found = False
            matched_profile_name = None
            matched_calc_crc = None
            matched_rx_crc = None

            for test_bytes in [payload_bytes, frame_bytes]:
                if len(test_bytes) >= 4 and not crc_match_found:
                    for p_name, profile in CRC_PROFILES.items():
                        match, calc, rx = verify_crc_stream(test_bytes, profile)
                        if match:
                            # Statistical safeguard: short CRCs (< 16 bits) require repeated frames or candidate_frame_len
                            if profile.width < 16 and repeats < 2 and candidate_frame_len is None:
                                continue
                            crc_match_found = True
                            matched_profile_name = p_name
                            matched_calc_crc = calc
                            matched_rx_crc = rx
                            break

            if crc_match_found:
                break

        # Printable representations
        ascii_repr = format_ascii_repr(payload_bytes if sync_len > 0 else frame_bytes)
        hex_repr = format_hex_dump(payload_bytes if sync_len > 0 else frame_bytes)

        if crc_match_found and repeats >= 2:
            status = EpistemicStatus.VALIDATED
            structure_detected = True
            frame_type = f"Validated Periodic Frame ({matched_profile_name})"
        elif crc_match_found:
            status = EpistemicStatus.VALIDATED
            structure_detected = True
            frame_type = f"Validated Frame ({matched_profile_name})"
        elif matched_sync_name and (repeats >= 2 or candidate_frame_len is not None):
            status = EpistemicStatus.ESTIMATED
            structure_detected = True
            frame_type = f"Preamble Synchronized ({matched_sync_name})"
        else:
            status = EpistemicStatus.UNKNOWN
            structure_detected = False
            frame_type = "Continuous Stream"

        # Multi-frame CRC tracking across all available frames in bitstream
        valid_frames_count = 0
        total_frames_count = 1
        if crc_match_found and matched_profile_name:
            prof = CRC_PROFILES[matched_profile_name]
            frame_len_bits = len(frame_bits)
            if frame_len_bits > 0 and n >= 2 * frame_len_bits:
                total_frames_count = min(32, n // frame_len_bits)
                for f_idx in range(total_frames_count):
                    f_start = start_offset + f_idx * frame_len_bits
                    f_end = f_start + frame_len_bits
                    if f_end <= n:
                        f_chunk = active_bits[f_start:f_end]
                        f_p_bytes = bits_to_bytes(f_chunk[sync_len:] if sync_len > 0 else f_chunk)
                        m_ok, _, _ = verify_crc_stream(f_p_bytes, prof)
                        if m_ok:
                            valid_frames_count += 1
            else:
                valid_frames_count = 1
        
        crc_pass_rate = float(valid_frames_count / max(1, total_frames_count))
        crc_matches_summary = f"CRC matches: {valid_frames_count} / {total_frames_count} frames"

        cand_result = FrameAnalysisResult(
            status=status,
            frame_structure_detected=structure_detected,
            frame_type=frame_type,
            sync_pattern_name=matched_sync_name if structure_detected else None,
            sync_pattern_bits=PREAMBLE_CATALOG.get(matched_sync_name, {}).get("hex") if (matched_sync_name and structure_detected) else None,
            frame_length=int(len(frame_bits)),
            repeated_frames_found=repeats,
            bit_offset=start_offset,
            header_bits=header_bits,
            payload_bits=payload_bits,
            crc_profile=matched_profile_name,
            crc_match=crc_match_found,
            crc_calculated=matched_calc_crc,
            crc_received=matched_rx_crc,
            recovered_ascii=ascii_repr,
            recovered_hex=hex_repr,
            valid_frames=valid_frames_count,
            total_frames=total_frames_count,
            crc_pass_rate=crc_pass_rate,
            crc_matches_summary=crc_matches_summary
        )

        if crc_match_found:
            return cand_result

        if best_result is None or (structure_detected and not best_result.frame_structure_detected):
            best_result = cand_result

    return best_result if best_result is not None else cand_result
