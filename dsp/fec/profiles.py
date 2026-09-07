"""
Standard Forward Error Correction (FEC) Profiles Module
======================================================
Defines verified standard profiles for:
1. Reed-Solomon algebraic codes (CCSDS, DVB-T shortened, Generic, ATSC)
2. Convolutional trellis codes (NASA K=7, GSM K=5, DVB punctured rates)
3. LDPC parity-check matrix specifications (IEEE 802.16 WiMAX, DVB-S2)
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional


@dataclass
class ReedSolomonProfile:
    name: str
    field_size: int
    prim_poly: int
    n: int
    k: int
    t: int
    two_t: int
    fcr: int
    shortening: int
    symbol_ordering: str
    standard_ref: str


@dataclass
class ConvolutionalProfile:
    name: str
    k: int
    rate: str
    poly: Tuple[int, ...]
    punctured: bool
    standard_ref: str


@dataclass
class LDPCProfile:
    name: str
    block_length: int
    info_length: int
    rate: str
    standard_ref: str


STANDARD_RS_PROFILES: Dict[str, ReedSolomonProfile] = {
    "CCSDS_RS_255_223": ReedSolomonProfile(
        name="CCSDS_RS_255_223",
        field_size=256,
        prim_poly=0x11D,
        n=255,
        k=223,
        t=16,
        two_t=32,
        fcr=1,
        shortening=0,
        symbol_ordering="MSB_FIRST",
        standard_ref="CCSDS 131.0-B-3 TM Synchronization and Channel Coding"
    ),
    "DVB_T_RS_204_188": ReedSolomonProfile(
        name="DVB_T_RS_204_188",
        field_size=256,
        prim_poly=0x11D,
        n=204,
        k=188,
        t=8,
        two_t=16,
        fcr=0,
        shortening=51,
        symbol_ordering="MSB_FIRST",
        standard_ref="ETSI EN 300 744 V1.6.1 DVB-T Framing Structure and Channel Coding"
    ),
    "GENERIC_RS_255_239": ReedSolomonProfile(
        name="GENERIC_RS_255_239",
        field_size=256,
        prim_poly=0x11D,
        n=255,
        k=239,
        t=8,
        two_t=16,
        fcr=0,
        shortening=0,
        symbol_ordering="MSB_FIRST",
        standard_ref="ITU-T G.975 Forward Error Correction for Submarine Systems"
    ),
    "ATSC_RS_207_187": ReedSolomonProfile(
        name="ATSC_RS_207_187",
        field_size=256,
        prim_poly=0x11D,
        n=207,
        k=187,
        t=10,
        two_t=20,
        fcr=0,
        shortening=48,
        symbol_ordering="MSB_FIRST",
        standard_ref="ATSC A/53 Digital Television Standard Part 2"
    )
}

STANDARD_CONV_PROFILES: Dict[str, ConvolutionalProfile] = {
    "NASA_K7_R12": ConvolutionalProfile(
        name="NASA_K7_R12",
        k=7,
        rate="1/2",
        poly=(0o171, 0o133),
        punctured=False,
        standard_ref="NASA Deep Space Network / Voyager / CCSDS 131.0-B-3"
    ),
    "GSM_K5_R12": ConvolutionalProfile(
        name="GSM_K5_R12",
        k=5,
        rate="1/2",
        poly=(0o23, 0o33),
        punctured=False,
        standard_ref="3GPP TS 05.03 GSM Channel Coding"
    ),
    "DVB_K7_R23": ConvolutionalProfile(
        name="DVB_K7_R23",
        k=7,
        rate="2/3",
        poly=(0o171, 0o133),
        punctured=True,
        standard_ref="ETSI EN 300 744 DVB Inner Convolutional Code (Rate 2/3)"
    ),
    "DVB_K7_R34": ConvolutionalProfile(
        name="DVB_K7_R34",
        k=7,
        rate="3/4",
        poly=(0o171, 0o133),
        punctured=True,
        standard_ref="ETSI EN 300 744 DVB Inner Convolutional Code (Rate 3/4)"
    )
}

STANDARD_LDPC_PROFILES: Dict[str, LDPCProfile] = {
    "WIMAX_R12_128": LDPCProfile(
        name="WIMAX_R12_128",
        block_length=128,
        info_length=64,
        rate="1/2",
        standard_ref="IEEE 802.16e / WiMAX QC-LDPC"
    ),
    "DVB_S2_SHORT_R12": LDPCProfile(
        name="DVB_S2_SHORT_R12",
        block_length=16200,
        info_length=8100,
        rate="1/2",
        standard_ref="ETSI EN 302 307 DVB-S2 Short Frame"
    )
}
