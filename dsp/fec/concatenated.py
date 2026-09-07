"""
Concatenated FEC Decoding Module
================================
Implements configurable concatenated decoding chains:
Candidate Chain: Inner FEC (Soft Viterbi) -> De-interleaver -> Outer FEC (Reed-Solomon).
Supports arbitrary chain order and profiles.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

from dsp.deinterleaving.block import deinterleave_block
from dsp.deinterleaving.convolutional import deinterleave_convolutional
from .viterbi import decode_viterbi_soft
from .reed_solomon import decode_reed_solomon


def decode_concatenated_chain(
    llrs: np.ndarray,
    inner_code: str = "NASA_K7_R12",
    interleaver_type: str = "convolutional",
    interleaver_params: Optional[Dict[str, Any]] = None,
    outer_code: str = "CCSDS_RS_255_223"
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    """
    Executes a concatenated FEC decoding chain:
    1. Inner Soft Viterbi Decoding
    2. De-interleaving of decoded bitstream
    3. Outer Reed-Solomon algebraic decoding

    Returns:
    --------
    Tuple[decoded_bytes_or_bits, chain_valid, telemetry]
    """
    if len(llrs) < 32:
        return np.zeros(0, dtype=np.uint8), False, {"error": "Insufficient LLRs"}

    # 1. Inner Code: Soft Viterbi
    poly = (0o171, 0o133)
    k_len = 7
    viterbi_bits, viterbi_metric, viterbi_ber = decode_viterbi_soft(llrs, poly=poly, K=k_len)

    # 2. De-interleaving
    params = interleaver_params or {"branches": 4, "delay_step": 2}
    if interleaver_type == "convolutional":
        deint_bits = deinterleave_convolutional(
            viterbi_bits,
            branches=params.get("branches", 4),
            delay_step=params.get("delay_step", 2)
        )
    elif interleaver_type == "block":
        deint_bits = deinterleave_block(
            viterbi_bits,
            rows=params.get("rows", 16),
            cols=params.get("cols", 16)
        )
    else:
        deint_bits = viterbi_bits

    # 3. Outer Code: Reed-Solomon (group bits into 8-bit bytes)
    num_bytes = len(deint_bits) // 8
    if num_bytes < 16:
        return deint_bits, False, {
            "viterbi_ber": viterbi_ber,
            "rs_valid": False,
            "chain_valid": False
        }

    rx_bytes = []
    for i in range(num_bytes):
        byte_bits = deint_bits[i * 8 : (i + 1) * 8]
        byte_val = int(np.packbits(byte_bits)[0])
        rx_bytes.append(byte_val)

    two_t = 32 if "255_223" in outer_code else 16
    fcr = 1 if "CCSDS" in outer_code else 0

    corrected_bytes, rs_valid, err_count = decode_reed_solomon(
        rx_bytes,
        two_t=two_t,
        fcr=fcr
    )

    telemetry = {
        "viterbi_ber": viterbi_ber,
        "viterbi_metric": viterbi_metric,
        "rs_valid": rs_valid,
        "rs_corrected_errors": err_count,
        "chain_valid": rs_valid
    }

    # Convert corrected bytes back to bit array
    out_bits = np.unpackbits(np.array(corrected_bytes, dtype=np.uint8))
    return out_bits, rs_valid, telemetry
