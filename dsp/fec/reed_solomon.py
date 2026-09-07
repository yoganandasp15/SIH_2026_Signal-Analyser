"""
Reed-Solomon Algebraic Decoder over GF(2^8) Module
==================================================
Implements Galois Field GF(256) arithmetic, systematic polynomial encoding,
and complete Berlekamp-Massey / Chien Search / Forney algorithm decoding.
Validates codewords via GF(256) syndrome checks (all syndromes == 0).

Standard Profiles:
------------------
- CCSDS RS(255, 223), 2t=32
- DVB RS(204, 188), 2t=16 (Shortened RS(255, 239))
- RS(255, 239), 2t=16
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class GF256:
    """
    Galois Field GF(2^8) representation using lookup tables.
    Primitive polynomial: p(x) = x^8 + x^4 + x^3 + x^2 + 1 (0x11D = 285).
    """
    def __init__(self, prim_poly: int = 0x11D):
        self.prim = prim_poly
        self.exp = [0] * 512
        self.log = [0] * 256

        x = 1
        for i in range(255):
            self.exp[i] = x
            self.exp[i + 255] = x
            self.log[x] = i
            x <<= 1
            if x & 0x100:
                x ^= self.prim
        self.log[0] = -1

    def add(self, a: int, b: int) -> int:
        """Addition in GF(256) is bitwise XOR."""
        return a ^ b

    def mul(self, a: int, b: int) -> int:
        """Multiplication in GF(256) using exp and log tables."""
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def div(self, a: int, b: int) -> int:
        """Division in GF(256)."""
        if b == 0:
            raise ZeroDivisionError("GF(256) division by zero")
        if a == 0:
            return 0
        return self.exp[(self.log[a] - self.log[b] + 255) % 255]

    def inv(self, a: int) -> int:
        """Multiplicative inverse in GF(256)."""
        if a == 0:
            raise ZeroDivisionError("Zero has no multiplicative inverse")
        return self.exp[255 - self.log[a]]

    def poly_mul(self, p: List[int], q: List[int]) -> List[int]:
        """Multiplies two polynomials over GF(256)."""
        res = [0] * (len(p) + len(q) - 1)
        for i, a in enumerate(p):
            for j, b in enumerate(q):
                res[i + j] ^= self.mul(a, b)
        return res


_GF = GF256()


RS_PROFILES = {
    "CCSDS_RS_255_223": {
        "n": 255,
        "k": 223,
        "two_t": 32,
        "fcr": 1,
        "name": "CCSDS Telemetry RS(255, 223)"
    },
    "DVB_RS_204_188": {
        "n": 204,
        "k": 188,
        "two_t": 16,
        "fcr": 0,
        "name": "DVB Broadcast RS(204, 188)"
    },
    "RS_255_239": {
        "n": 255,
        "k": 239,
        "two_t": 16,
        "fcr": 0,
        "name": "Generic Standard RS(255, 239)"
    }
}


def get_generator_poly(two_t: int, fcr: int = 0) -> List[int]:
    """Computes generator polynomial g(x) = prod_{i=0}^{two_t-1} (x - alpha^(fcr + i))."""
    g = [1]
    for i in range(two_t):
        root = _GF.exp[fcr + i]
        g = _GF.poly_mul(g, [1, root])
    return g


def encode_reed_solomon(
    msg: List[int],
    two_t: int = 16,
    fcr: int = 0
) -> List[int]:
    """Systematic Reed-Solomon encoder: appends two_t parity symbols to message."""
    gen = get_generator_poly(two_t, fcr=fcr)
    # Pad message with two_t zeros
    padded = list(msg) + [0] * two_t
    # Polynomial long division
    for i in range(len(msg)):
        coeff = padded[i]
        if coeff != 0:
            for j in range(1, len(gen)):
                padded[i + j] ^= _GF.mul(coeff, gen[j])

    # Parity symbols are in the padded tail
    parity = padded[len(msg):]
    return list(msg) + parity


def calculate_syndromes(
    codeword: List[int],
    two_t: int,
    fcr: int = 0
) -> List[int]:
    """
    Computes RS syndromes S_i = r(alpha^(fcr + i)) for i = 0 .. two_t-1.
    Codeword is valid if and only if all syndromes == 0.
    """
    n = len(codeword)
    syndromes = []
    for i in range(two_t):
        alpha_i = _GF.exp[fcr + i]
        # Evaluate polynomial using Horner's method
        val = 0
        for coeff in codeword:
            val = _GF.mul(val, alpha_i) ^ coeff
        syndromes.append(val)
    return syndromes


def berlekamp_massey(syndromes: List[int], two_t: int) -> Tuple[List[int], int]:
    """
    Berlekamp-Massey algorithm for finding the error locator polynomial Lambda(x).
    """
    c = [1]  # Lambda(x)
    b = [1]  # Correction polynomial
    l = 0
    m = 1
    d_b = 1

    for n in range(two_t):
        # Discrepancy delta
        delta = syndromes[n]
        for i in range(1, l + 1):
            if i < len(c):
                delta ^= _GF.mul(c[i], syndromes[n - i])

        if delta == 0:
            m += 1
        else:
            t = list(c)
            scale = _GF.div(delta, d_b)
            # c(x) = c(x) - (delta / d_b) * x^m * b(x)
            shift_b = [0] * m + b
            while len(c) < len(shift_b):
                c.append(0)
            for j in range(len(shift_b)):
                c[j] ^= _GF.mul(scale, shift_b[j])

            if 2 * l <= n:
                l = n + 1 - l
                b = t
                d_b = delta
                m = 1
            else:
                m += 1

    return c, l


def chien_search(lambda_poly: List[int], n: int) -> List[int]:
    """
    Chien search: finds roots of Lambda(x) to determine error positions.
    Returns list of error indices (0-indexed from high power).
    """
    error_positions = []
    deg = len(lambda_poly) - 1

    for i in range(n):
        # Evaluate Lambda(alpha^-i) = Lambda(alpha^(255 - i))
        x = _GF.exp[(255 - i) % 255]
        val = 0
        x_pwr = 1
        for coeff in lambda_poly:
            val ^= _GF.mul(coeff, x_pwr)
            x_pwr = _GF.mul(x_pwr, x)
        if val == 0:
            error_positions.append(n - 1 - i)

    return error_positions


def forney_algorithm(
    syndromes: List[int],
    lambda_poly: List[int],
    error_positions: List[int],
    n: int,
    fcr: int = 0
) -> Dict[int, int]:
    """
    Forney algorithm for error value evaluation:
    e_j = X_j^(1 - fcr) * Omega(X_j^-1) / Lambda'(X_j^-1)
    """
    two_t = len(syndromes)
    # Omega(x) = S(x) * Lambda(x) mod x^two_t
    omega = _GF.poly_mul(syndromes, lambda_poly)[:two_t]

    # Formal derivative of Lambda(x) (only odd powers survive in characteristic 2)
    lambda_prime = [0] * len(lambda_poly)
    for i in range(1, len(lambda_poly), 2):
        lambda_prime[i] = lambda_poly[i]

    error_magnitudes = {}
    for pos in error_positions:
        i = n - 1 - pos
        x_inv = _GF.exp[(255 - i) % 255]
        x_val = _GF.exp[i % 255]

        # Evaluate Omega(x_inv)
        num = 0
        pwr = 1
        for coeff in omega:
            num ^= _GF.mul(coeff, pwr)
            pwr = _GF.mul(pwr, x_inv)

        # Evaluate Lambda'(x_inv)
        denom = 0
        pwr = 1
        for coeff in lambda_prime:
            denom ^= _GF.mul(coeff, pwr)
            pwr = _GF.mul(pwr, x_inv)

        if denom == 0:
            continue

        e_val = _GF.div(num, denom)
        if fcr != 0:
            e_val = _GF.mul(e_val, _GF.exp[(fcr * i) % 255])
        error_magnitudes[pos] = e_val

    return error_magnitudes


def decode_reed_solomon(
    received_bytes: List[int],
    two_t: int = 16,
    fcr: int = 0
) -> Tuple[List[int], bool, int]:
    """
    Decodes received bytes via RS algebraic decoding.

    Returns:
    --------
    Tuple[corrected_bytes, success_boolean, corrected_symbol_count]
    """
    n = len(received_bytes)
    syndromes = calculate_syndromes(received_bytes, two_t=two_t, fcr=fcr)

    # 1. Zero Syndrome Check
    if all(s == 0 for s in syndromes):
        return list(received_bytes), True, 0

    # 2. Berlekamp-Massey
    lambda_poly, num_errors = berlekamp_massey(syndromes, two_t=two_t)
    if num_errors == 0 or num_errors > two_t // 2:
        return list(received_bytes), False, 0

    # 3. Chien Search
    error_positions = chien_search(lambda_poly, n)
    if len(error_positions) != num_errors:
        return list(received_bytes), False, 0

    # 4. Forney Algorithm
    error_values = forney_algorithm(syndromes, lambda_poly, error_positions, n, fcr=fcr)

    corrected = list(received_bytes)
    for pos, val in error_values.items():
        corrected[pos] ^= val

    # 5. Closed-loop validation: post-correction syndromes MUST all be zero!
    post_syndromes = calculate_syndromes(corrected, two_t=two_t, fcr=fcr)
    is_valid = all(s == 0 for s in post_syndromes)

    return corrected, is_valid, len(error_positions)
