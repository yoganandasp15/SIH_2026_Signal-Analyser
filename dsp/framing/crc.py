"""
Complete Parametric Cyclic Redundancy Check (CRC) Engine
========================================================
Implements the full standard CRC parameterization:
- polynomial
- initial value
- input reflection (refin)
- output reflection (refout)
- final XOR value (xorout)
- bit width
Prevents conflicting CRC conventions from causing validation failures.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from dsp.contracts import CRCProfile


def reflect_bits(val: int, width: int) -> int:
    """Reflects / reverses the lowest width bits of val."""
    res = 0
    for i in range(width):
        if (val >> i) & 1:
            res |= (1 << (width - 1 - i))
    return res


CRC_PROFILES: Dict[str, CRCProfile] = {
    "CRC-16-CCITT": CRCProfile(
        name="CRC-16-CCITT",
        width=16,
        poly=0x1021,
        init=0xFFFF,
        refin=False,
        refout=False,
        xorout=0x0000,
        check=0x29B1,
        residue=0x0000,
        bit_order="MSB",
        byte_order="big"
    ),
    "CRC-16-IBM": CRCProfile(
        name="CRC-16-IBM",
        width=16,
        poly=0x8005,
        init=0x0000,
        refin=True,
        refout=True,
        xorout=0x0000,
        check=0xBB3D,
        residue=0x0000,
        bit_order="LSB",
        byte_order="little"
    ),
    "CRC-16-MODBUS": CRCProfile(
        name="CRC-16-MODBUS",
        width=16,
        poly=0x8005,
        init=0xFFFF,
        refin=True,
        refout=True,
        xorout=0x0000,
        check=0x4B37,
        residue=0x0000,
        bit_order="LSB",
        byte_order="little"
    ),
    "CRC-32-IEEE": CRCProfile(
        name="CRC-32-IEEE",
        width=32,
        poly=0x04C11DB7,
        init=0xFFFFFFFF,
        refin=True,
        refout=True,
        xorout=0xFFFFFFFF,
        check=0xCBF43926,
        residue=0xDEBB20E3,
        bit_order="LSB",
        byte_order="little"
    ),
    "CRC-8-ATM": CRCProfile(
        name="CRC-8-ATM",
        width=8,
        poly=0x07,
        init=0x00,
        refin=False,
        refout=False,
        xorout=0x55,
        check=0xA1,
        residue=0x00,
        bit_order="MSB",
        byte_order="big"
    ),
    "CRC-8-DARC": CRCProfile(
        name="CRC-8-DARC",
        width=8,
        poly=0x39,
        init=0x00,
        refin=True,
        refout=True,
        xorout=0x00,
        check=0x15,
        residue=0x00,
        bit_order="LSB",
        byte_order="little"
    )
}


def compute_crc(data_bytes: bytes, profile: CRCProfile) -> int:
    """
    Computes parametric CRC over byte data according to standard catalog conventions.
    """
    width = profile.width
    poly = profile.poly
    reg = profile.init
    top_bit = 1 << (width - 1)
    mask = (1 << width) - 1

    for b in data_bytes:
        if profile.refin:
            b = reflect_bits(b, 8)

        reg ^= (b << (width - 8))
        for _ in range(8):
            if reg & top_bit:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask

    if profile.refout:
        reg = reflect_bits(reg, width)

    return (reg ^ profile.xorout) & mask


def verify_crc_stream(
    frame_bytes: bytes,
    profile: CRCProfile
) -> Tuple[bool, int, int]:
    """
    Splits frame into payload and CRC field, computes CRC over payload, and tests equality.
    Returns: (match_bool, calculated_crc, received_crc).
    """
    crc_bytes_len = (profile.width + 7) // 8
    if len(frame_bytes) <= crc_bytes_len:
        return False, 0, 0

    payload = frame_bytes[:-crc_bytes_len]
    rx_crc_bytes = frame_bytes[-crc_bytes_len:]

    calc_crc = compute_crc(payload, profile)

    # Decode received CRC according to big-endian or profile reflection
    rx_crc = int.from_bytes(rx_crc_bytes, byteorder='big')
    if profile.refout and profile.width == 16:
        # Check little-endian alternative if big-endian mismatch
        rx_crc_le = int.from_bytes(rx_crc_bytes, byteorder='little')
        if calc_crc == rx_crc_le:
            return True, calc_crc, rx_crc_le

    return (calc_crc == rx_crc), calc_crc, rx_crc
