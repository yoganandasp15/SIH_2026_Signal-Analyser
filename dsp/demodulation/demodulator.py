"""
Unified Demodulation Dispatcher
===============================
Dispatches synchronized symbols to PSK, QAM, or FSK demodulation engines,
generating hard bits, soft LLRs, and error metrics in a typed DemodulationResult.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

from dsp.contracts import (
    ModulationFamily,
    SynchronizedSymbols,
    DemodulationResult
)
from .psk import demodulate_psk
from .qam import demodulate_qam
from .fsk import demodulate_fsk


def demodulate_symbols(
    sync_symbols: SynchronizedSymbols,
    modulation: ModulationFamily,
    noise_variance: float = 0.1
) -> DemodulationResult:
    """
    Demodulates synchronized baseband symbols into hard bits and soft LLRs.
    """
    syms = sync_symbols.symbols
    if len(syms) == 0:
        return DemodulationResult(
            modulation=modulation,
            hard_bits=np.zeros(0, dtype=np.uint8),
            soft_llrs=np.zeros(0, dtype=np.float32),
            symbol_error_rate_est=0.5,
            evm_pct=100.0,
            noise_var_est=noise_variance
        )

    if modulation == ModulationFamily.BPSK:
        bits, llrs, evm = demodulate_psk(syms, order=2, noise_variance=noise_variance)
        ser = float(0.5 * np.exp(-1.0 / (2.0 * max(1e-4, noise_variance))))

    elif modulation == ModulationFamily.QPSK:
        bits, llrs, evm = demodulate_psk(syms, order=4, noise_variance=noise_variance)
        ser = float(0.5 * np.exp(-0.5 / (max(1e-4, noise_variance))))

    elif modulation == ModulationFamily.PSK_8:
        bits, llrs, evm = demodulate_psk(syms, order=8, noise_variance=noise_variance)
        ser = float(0.5 * np.exp(-0.25 / (max(1e-4, noise_variance))))

    elif modulation == ModulationFamily.QAM_16:
        bits, llrs, evm = demodulate_qam(syms, order=16, noise_variance=noise_variance)
        ser = float(0.75 * np.exp(-0.2 / (max(1e-4, noise_variance))))

    elif modulation == ModulationFamily.QAM_64:
        bits, llrs, evm = demodulate_qam(syms, order=64, noise_variance=noise_variance)
        ser = float(0.85 * np.exp(-0.05 / (max(1e-4, noise_variance))))

    elif modulation == ModulationFamily.FSK_2:
        bits, llrs, ser = demodulate_fsk(syms, order=2, sps=sync_symbols.samples_per_symbol, noise_variance=noise_variance)
        evm = 15.0

    elif modulation == ModulationFamily.FSK_4:
        bits, llrs, ser = demodulate_fsk(syms, order=4, sps=sync_symbols.samples_per_symbol, noise_variance=noise_variance)
        evm = 20.0

    else:
        # Fallback binary slicing on real part
        real_part = np.real(syms)
        bits = (real_part < 0.0).astype(np.uint8)
        llrs = (2.0 * real_part / max(1e-4, noise_variance)).astype(np.float32)
        llrs = np.clip(llrs, -20.0, 20.0)
        evm = 30.0
        ser = 0.25

    return DemodulationResult(
        modulation=modulation,
        hard_bits=bits,
        soft_llrs=llrs,
        symbol_error_rate_est=ser,
        evm_pct=evm,
        noise_var_est=noise_variance
    )
