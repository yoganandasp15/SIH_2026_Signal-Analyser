# NTRO Signal Analyzer -- Complete Technical Guide
## Edition 4.1 . Architecture v4.1 (Blind Extraction + Round-2 Intelligence Engine)

> **A Comprehensive Engineering Reference for SIH 2026 Problem Statement 26147**
> Autonomous RF Signal Parameter Extraction for the National Technical Research Organisation

**Authors:** Signal Intelligence Engineering Team (ECE, 5th Semester)
**Target:** Smart India Hackathon 2026 Evaluators & Engineering Students
**Edition:** 4.1 (Architecture v4.1 -- strict blind-parameter boundary, modulation inference, protocol inference, multi-window validation, hypothesis ranking, contradiction analysis, validation gate, uncertainty, epistemic states, robustness)
**Verified:** 8 September 2026 -- 177 automated tests and the 20-item SIH26147 capability audit passed.

---

## Table of Contents

1. [The Problem We Are Solving](#1-the-problem-we-are-solving)
2. [Signals in the Real World -- DSP Fundamentals](#2-signals-in-the-real-world)
3. [The Core Mathematical Toolkit](#3-the-core-mathematical-toolkit)
4. [Software Architecture and Technology Stack](#4-software-architecture-and-technology-stack)
5. [End-to-End System Pipeline -- Architecture v4.1](#5-end-to-end-system-pipeline--architecture-v41)
6. [Stage 1: File Ingestion (loaders.py)](#6-stage-1-file-ingestion)
7. [Stage 2: Signal Conditioning (preprocessor.py)](#7-stage-2-signal-conditioning)
8. [Stage 3: Spectral Analysis (spectral.py)](#8-stage-3-spectral-analysis)
9. [Stages 4-7: Blind Parameter Engine and Modulation Inference](#9-stage-4-7-blind-parameter-engine-and-modulation-inference)
10. [Stage 8: Multi-Window Temporal Validation (temporal_validator.py)](#10-stage-8-multi-window-temporal-validation)
11. [Stage 10: Protocol Inference and Autonomous Classification](#11-stage-10-protocol-inference-and-autonomous-classification)
12. [Stage 11: Candidate Hypothesis Ranking (candidate_ranker.py)](#12-stage-11-candidate-hypothesis-ranking)
13. [Stage 12: Contradiction Analysis (contradiction_analyzer.py)](#13-stage-12-contradiction-analysis)
14. [Stage 13: Adaptive Extraction Pipeline (adaptive_pipeline.py)](#14-stage-13-adaptive-extraction-pipeline)
15. [Stage 14: Physical Reconstruction Engine](#15-stage-14-physical-reconstruction-engine)
16. [Stage 15: Hypothesis Evidence Validation Gate (validation_gate.py)](#16-stage-15-hypothesis-evidence-validation-gate)
17. [Stage 16: Parameter Uncertainty Reporting (parameter_uncertainty.py)](#17-stage-16-parameter-uncertainty-reporting)
18. [Extended Epistemic Hierarchy Contract (contracts.py) **UPDATED**](#18-extended-epistemic-hierarchy-contract)
19. [Pulse Analysis (pulse_analyzer.py)](#19-pulse-analysis)
20. [Interactive Visualizations and Professional UI **UPDATED**](#20-interactive-visualizations-and-professional-ui)
21. [Tactical Intercept Scenario Selector **NEW**](#21-tactical-intercept-scenario-selector)
22. [Complete Step-by-Step Walkthrough: NAVTEX Signal](#22-complete-step-by-step-walkthrough-navtex-signal)
23. [Benchmark Results](#23-benchmark-results)
24. [Robustness and Stress Testing Engine **NEW**](#24-robustness-and-stress-testing-engine)
25. [Known Limitations and Architecture Boundaries](#25-known-limitations-and-architecture-boundaries)
26. [Glossary of Terms](#26-glossary-of-terms)

---

## 1. The Problem We Are Solving

### 1.1 The SIH 2026 Challenge (SIH26147)

Smart India Hackathon 2026 Problem Statement SIH26147, submitted by the National Technical Research Organisation (NTRO), requires:

> *"Development of a software solution for autonomous parameter extraction of intercepted RF signals without prior knowledge of signal parameters."*

In plain language: a monitoring station or surveillance aircraft intercepts an unknown radio signal. The operator has no manual labels, no transmission logs, and no prior knowledge of who is transmitting or what hardware they are using. The software must ingest the raw recording, classify the modulation format, extract all physical transmission parameters, and deliver structured intelligence ready for tactical decoders.

```
+------------------+      +-------------------------------+      +------------------------+
|  Intercepted RF  | ---> |   Autonomous Parameter        | ---> |  Tactical Intelligence |
|  Raw Recording   |      |   Extraction Engine           |      |  - Carrier: 2.13 kHz   |
|  (.wav / .iq)    |      |   (NTRO Signal Analyzer)      |      |  - 99% OBW: 323 Hz     |
+------------------+      +-------------------------------+      |  - Mode: 2-FSK NAVTEX  |
                                                                 |  - Baud: 100.0 Baud    |
                                                                 |  - SNR: 11.7 dB        |
                                                                 |  - Status: VALIDATED   |
                                                                 +------------------------+
```

### 1.2 The Real-World SIGINT Problem

In real-world signals intelligence (SIGINT) and electronic warfare (EW), an open radio spectrum presents every signal type simultaneously: HF maritime telex, over-the-horizon radar pulses, military tactical link transmissions, amateur radio emergency messages, cellular base station broadcasts, satellite weather faxes, and CubeSat telemetry beacons. Each occupies a different bandwidth slice, transmits at a different speed, and modulates electromagnetic waves using different physical principles.

### 1.3 Why Traditional Approaches Fail

1. **Manual Selection Dropdowns:** Require the user to pick "FSK" or "Radar" and type the sampling rate. Useless for autonomous intercept where no human operator is present.
2. **Brittle Neural Networks:** Deep learning classifiers trained on synthetic datasets fail when exposed to real-world multipath fading, soundcard anti-aliasing filter roll-offs, and varying receiver audio beat frequencies. A neural network cannot explain *why* it picked a class, and it frequently produces false positives with high confidence.

### 1.4 What Our System Accomplishes -- Architecture v4.1

The current engine separates physical measurement from interpretation. The blind front end measures what the waveform contains before the protocol catalog is consulted. The Round-2 reasoning layer then ranks and validates interpretations:

1. Characterizes the input and separates activity from the adaptive noise floor.
2. Discovers temporal segments, frequency components, pulses, chirps, and hopping structure.
3. Builds a `BlindParameterVector` from protocol-independent physical measurements.
4. Infers modulation geometry from that vector, without protocol names.
5. Applies the protocol knowledge layer only after parameter extraction.
6. Generates and ranks competing hypotheses, retaining UNKNOWN and UNKNOWN_OOD as valid outcomes.
7. Computes supporting evidence and measurable contradictions for each candidate.
8. Passes candidates through six validation gates before promoting one to VALIDATED.
9. Reports epistemic status and uncertainty for every parameter.
10. Uses specialized extraction, reconstruction, and decoder telemetry as downstream evidence rather than as the source of blind parameters.

This pipeline is a hybrid DSP/physics-based architecture with open-set rejection, multi-stage evidence fusion, and an explicit epistemic hierarchy -- not a lookup table or neural network.

---

## 2. Signals in the Real World -- DSP Fundamentals

### 2.1 What Actually is a Signal?

An electrical signal is a voltage that changes over time. An Analog-to-Digital Converter (ADC) samples continuous voltage at regular intervals:

```
Continuous:  v(t)
Sampled:     x[n] = v( n * Ts ) = v( n / fs )

  fs  = Sampling frequency (samples per second, Hz)
  Ts  = 1 / fs  (seconds between samples)
  n   = Integer sample index (0, 1, 2, ...)
```

### 2.2 Real Signals vs. Complex Baseband (I/Q Representation)

Modern SDRs split the incoming RF carrier into two orthogonal paths:

```
             +--> Multiplied by cos(2*pi*fc*t) --> Lowpass Filter --> In-Phase I(t)
Incoming RF -|
             +--> Multiplied by -sin(2*pi*fc*t) -> Lowpass Filter --> Quadrature Q(t)

Together:  s(t) = I(t) + j * Q(t)    where j = sqrt(-1)
```

Instantaneous amplitude: `A = sqrt(I^2 + Q^2)`. Instantaneous phase: `phi = atan2(Q, I)`.

### 2.3 The Hilbert Transform and Analytic Signals

For real-valued audio recordings:

```
s(t) = x(t) + j * H{ x(t) }

H{ x(t) } = (1 / pi) * Integral[ x(tau) / (t - tau) ] dtau
```

Constructs the analytic signal with energy only on positive frequencies.

### 2.4 The Nyquist-Shannon Sampling Theorem and Aliasing

```
fs >= 2 * f_max
```

Sampling rate must be at least twice the highest frequency present. For CD-quality WAV: fs = 44,100 Hz, Nyquist bandwidth = 22,050 Hz. For SDR captures: up to 2 MSPS or 20 MSPS.

### 2.5 Modulation Families

| Modulation | What Changes | Typical Applications |
|---|---|---|
| CW | None (pure carrier) | Beacons, radar calibration |
| OOK | Carrier ON/OFF | Morse code |
| AM | Amplitude scales with voice | Aviation, AM broadcast |
| FM | Frequency shifts smoothly | Walkie-talkies, WEFAX, marine VHF |
| 2-FSK | 2 discrete frequencies | NAVTEX, ASCII, RTTY |
| M-FSK | 4, 8, or 16 tones | FT8, 2G ALE military |
| BPSK | Phase flips 180 degrees | PSK31, satellite downlinks |
| QPSK | 4 phase angles (90 degrees) | DVB-S, GPS |
| 8-PSK | 8 phase angles (45 degrees) | STANAG 4285 |
| QAM | Amplitude + phase | Digital cable TV, WiFi, 4G/5G |
| Pulsed Radar/FMOP | Microsecond bursts with chirps | Military early warning, coastal radar |

---

## 3. The Core Mathematical Toolkit

### 3.1 The Fourier Transform and FFT Complexity

```
X[k] = Sum_{n=0}^{N-1}  x[n] * exp(-j * 2*pi * k * n / N)     for k = 0, 1, ..., N-1
```

FFT reduces complexity from O(N^2) to O(N * log2(N)). For N = 65,536: from 4.29 billion to 1,048,576 operations -- computed in under 2 milliseconds.

### 3.2 Power Spectral Density: Welch's Averaged Periodogram

```
PSD_dB[k] = 10 * log10( max( P_linear[k], 1e-18 ) )
```

Welch's method divides signal into overlapping Hann-windowed segments, computes FFT of each, and averages. Suppresses random noise variance by factor M (number of segments).

### 3.3 Bandwidth Definitions

1. **-3 dB Bandwidth:** Span where power drops to half peak.
2. **-10 dB Bandwidth:** Span where power drops to 10% peak.
3. **99% OBW:** Cumulative power search -- find 0.5% and 99.5% boundaries:

```
C[k] = Sum_{i=0}^{k} P_linear[i]
C_norm[k] = C[k] / C[N-1]
OBW_99% = f[k_high] - f[k_low]    (k_low @ 0.5%, k_high @ 99.5%)
```

### 3.4 Signal-to-Noise Ratio: Four Physical Regimes

**Regime 1 -- M2M4** (continuous digital PSK/QAM):
```
M2 = E[|y[n]|^2] = P_signal + P_noise
M4 = E[|y[n]|^4]
P_signal = sqrt( max( 2*M2^2 - M4, eps ) )
SNR (dB) = 10 * log10( P_signal / (M2 - P_signal) )
```

**Regime 2 -- Band-Integrated Spectral Power** (narrowband FSK, CW):
```
P_in_band  = Sum_{f in OBW} PSD[f] * df
N0         = median( PSD[f not in OBW] )
SNR (dB) = 10 * log10( (P_in_band - N0*N_bins*df) / (N0*N_bins*df) )
```

**Regime 3 -- Active-to-Quiet Ratio** (analog voice):
```
P_active = mean( |s[n]|^2  for |s[n]| >= 90th percentile )
P_quiet  = mean( |s[n]|^2  for |s[n]| <= 25th percentile )
SNR (dB) = 10 * log10( max( (P_active - P_quiet) / P_quiet, 1.0 ) )
```

**Regime 4 -- In-Pulse vs. Inter-Pulse** (pulsed radar):
```
P_pulse = mean( |s[n]|^2  for n in active pulse intervals )
P_noise = mean( |s[n]|^2  for n in inter-pulse quiet intervals )
SNR (dB) = 10 * log10( max( (P_pulse - P_noise) / P_noise, 1e-4 ) )
```

All estimators clip to [-20, +42] dB.

### 3.5 Symbol Rate, Baud Rate, and Bit Dwell Timing

```
Baud Rate (Rs) = 1 / Ts   (symbols per second)
```

For FSK, the signal dwells on one frequency for one symbol period Ts before jumping. The shortest stable dwell duration represents a single unit bit.

### 3.6 Higher-Order Cumulants: Statistical Modulation Fingerprints

```
C20 = mu20
C40 = mu40 - 3 * (mu20)^2
C42 = mu42 - |mu20|^2 - 2 * (mu21)^2
```

| Modulation | |C20| | |C40| | Interpretation |
|---|---|---|---|
| BPSK | 1.000 | 2.000 | Extreme symmetry along one phase axis |
| QPSK | 0.000 | 1.000 | Four orthogonal points cancel 2nd moment |
| 8-PSK | 0.000 | 0.000 | 8 circular points average to near-zero kurtosis |
| 16-QAM | 0.000 | 0.680 | Multi-level square grid, distinct non-zero kurtosis |
| Gaussian Noise | 0.000 | 0.000 | All cumulants of order >= 3 are zero by definition |

---

## 4. Software Architecture and Technology Stack

### 4.1 Why Python, NumPy, and SciPy?

- **NumPy:** C-level vectorized SIMD arrays; operations run at compiled hardware speeds.
- **SciPy:** Verified Welch periodogram, Butterworth filtering, Hilbert transform, and Levinson-Durbin Toeplitz solver.
- **Maintainability:** Transparent algorithms that engineering students can audit and defend.

### 4.2 Why Streamlit?

Streamlit's Reactive Execution Model re-runs the script top-to-bottom on each user interaction, caching the pipeline result in `st.session_state`. The Round-2 UI is a Professional Engineering Workstation with 9-level hierarchical presentation.

### 4.3 Why Plotly?

WebGL-accelerated Scattergl renders thousands of constellation points at 60 fps. All transitions set to `duration=0` for zero animation latency.

### 4.4 The 1-Click Zero-Dependency Launcher

`START_DASHBOARD.bat`:
```bat
@echo off
set "PYTHONPATH=%~dp0"
python -m streamlit run app.py --server.port 8501 --server.headless false
```

---

## 5. End-to-End System Pipeline -- Architecture v4.1

### 5.1 Project Directory Structure (v4)

```
C:\ntro_signal_analyzer\
├── app.py                            <- Professional Engineering Workstation UI (Round-2)
├── cli.py                            <- Command-line headless interface
├── START_DASHBOARD.bat               <- 1-click Windows launcher
├── requirements.txt
│
├── dsp\
│   ├── loaders.py                    <- Universal file ingestion (WAV, binary IQ)
│   ├── preprocessor.py               <- Input validation, DC removal, normalization
│   ├── spectral.py                   <- Welch PSD and STFT spectrogram
│   ├── parameter_extractor.py        <- compatibility facade + downstream parameter reports
│   ├── blind_parameter_engine.py     <- Layer A: protocol-independent physical measurements
│   ├── modulation_inference.py       <- Layer B: modulation from blind geometry
│   ├── protocol_inference.py         <- Layer C: catalog/rule matching after measurement
│   ├── temporal_validator.py         <- 4-window temporal consistency validation
│   ├── pulse_analyzer.py             <- Adaptive pulse detection, PRI/PRF, FMOP
│   ├── autonomous_detector.py        <- physical invariant candidate generator
│   ├── candidate_ranker.py           <- Multi-candidate hypothesis ranking
│   ├── contradiction_analyzer.py     <- Supporting + contradicting evidence
│   ├── validation_gate.py            <- 6-gate physical hypothesis validation
│   ├── parameter_uncertainty.py      <- Epistemic status + uncertainty reporting
│   ├── adaptive_pipeline.py          <- Strategy pattern orchestrator
│   ├── contracts.py                  <- BlindParameterVector + SignalHypothesis contracts
│   ├── modulation_classifier.py      <- Higher-order cumulant decision trees
│   ├── robustness_tester.py          <- [NEW] 6-dimension impairment stress engine
│   ├── forensics.py                  <- Deep forensic analysis tools
│   ├── conditioning\                 <- IQ imbalance correction and AGC
│   ├── features\                     <- 20-D feature extraction
│   ├── amc\                          <- Open-set AMC (UNKNOWN_OOD output)
│   ├── synchronization\              <- CFO estimation, PLL, timing recovery
│   ├── demodulation\                 <- Soft LLR demodulation
│   ├── deinterleaving\               <- Interleaver topology search
│   ├── fec\                          <- FEC hypothesis evaluation
│   ├── framing\                      <- Sync-word correlation, CRC validation
│   └── evidence\                     <- Multi-stage evidence fusion
│
├── visualization\
│   └── plots.py                      <- PSD, spectrogram, constellation, eye, envelope
│
├── utils\
│   ├── exporter.py                   <- JSON and CSV telemetry generation
│   ├── synthetic_generator.py        <- AWGN test signal generator
│   ├── epistemic_demo.py             <- [NEW] 6 epistemic application state demos
│   └── tactical_scenarios.py         <- [NEW] 7-scenario Tactical Intercept Selector
│
├── explanation\                      <- Technical documentation (this guide)
├── verified_samples\                 <- 25+ real-world intercepted radio recordings
├── tests\                            <- 177 automated tests (100% pass)
└── scripts\
    ├── benchmark_13_signals.py       <- 25-signal SigIDWiki benchmark
    ├── round2_evaluation_benchmark.py <- [NEW] 44-case Round-2 evaluation battery
    └── run_robustness_tests.py        <- [NEW] CLI robustness sweep runner
```

### 5.2 Sequential Data Flow -- Architecture v4.1 (16-Stage Evidence Chain)

```
[ Captured File (.wav / .iq) ]
              |
              v
[ Stage 1: File Ingestion ]          Detects format, resolves fs, tags audio domain
              |
              v
[ Stage 2: Input Validation ]        Rejects NaN, Inf, wrong dimensions, fs <= 0
              |
              v
[ Stage 3: Preprocessing ]           Subtracts LO DC bias, normalizes power to 1.0
              |
              v
[ Stage 4: Blind Signal Search ]     Adaptive noise floor, activity detection, signal segments
              |
              v
[ Stage 5: Multi-Resolution DSP ]   Coarse/medium/fine spectra and component discovery
              |
              v
[ Stage 6: Blind Parameter Vector ] fc, OBW, SNR, states, timing, pulse, phase, morphology
              |                      `BlindParameterVector` is produced before protocol lookup
              v
[ Stage 7: Modulation Inference ]   Layer B maps physical geometry to 2-FSK, QPSK, QAM, etc.
              |
              v
[NEW Stage 8: Temporal Validation ] 4-window split, cross-window parameter stability,
              |                      stationarity scoring, configurable thresholds
              v
[ Stage 9: Pulse Analysis ]         Sparse-event detector, PRI/PRF, chirp test, TDMA gate
              |
              v
[ Stage 10: Protocol Inference ]    Protocol catalog and physical rules are downstream only
              |                      Returns UNKNOWN when evidence is insufficient
              v
[ Stage 11: Candidate Ranking ]     Multiple competing hypotheses, evidence_score per candidate
              |                      Ranking by evidence (NOT calibrated probability)
              v
[ Stage 12: Contradiction Analysis] For each candidate: supporting + contradicting evidence
              |                      Contradictions reduce net_evidence_score with explicit weights
              v
[ Stage 13: Adaptive Extraction ]   Dispatches to specialized extractor (FSK/Radar/etc.)
              |                      Extracts mode-specific parameters (shifts, dwells, PRF)
              v
[ Stage 14: Physical Reconstruction] IQ conditioning -> 20-D features -> open-set AMC ->
              |                      sync -> demod -> FEC -> framing -> evidence fusion
              v
[ Stage 15: Evidence Validation ] 6-gate pipeline: spectral / temporal / symbol-dwell /
              |                      cross-window / physical / reconstruction consistency
              |                      Tri-state per gate: PASS / FAIL / NOT_APPLICABLE
              |                      Final verdict: VALIDATED / ESTIMATED / AMBIGUOUS / UNKNOWN
              v
[ Stage 16: Uncertainty + Telemetry] Per-parameter value, status, stability, uncertainty,
              |                      estimator resolution bounds or cross-window std
                                     Hypothesis ranking, multi-window stability table,
                                     validation trace audit, downloadable intercept reports
```

Pipeline status values:
- `SUCCESS`: All 14 stages completed.
- `PARTIAL_SUCCESS`: Stages 1-13 succeeded; physical reconstruction encountered an exception.
- `FAILED`: Input validation rejected the signal.
- `AMBIGUOUS`: Validation gate could not promote any candidate beyond ESTIMATED.

---

## 6. Stage 1: File Ingestion

### 6.1 WAV File Ingestion and Correlation Testing

Reads RIFF WAV header using `scipy.io.wavfile.read()`.

- **Mono:** Hilbert transform to complex baseband: `s[n] = x[n] + j * H{ x[n] }`
- **Stereo:** Pearson correlation `rho = Cov(C0,C1) / (sigma_C0 * sigma_C1)`
  - rho > 0.75: same audio track (average + Hilbert)
  - rho <= 0.75: true SDR quadrature rails (`s[n] = I[n] + j*Q[n]`)

### 6.2 Raw Binary I/Q Ingestion

Headerless binary files (.iq, .raw, .bin) probe format via:
1. Companion `.sigmf-meta` JSON
2. Filename tokens (e.g., `cf32`, `cs16`, `2.048msps`)
3. Statistical Baseband Smoothness: `Ratio = Var(I[n]-I[n-1]) / Var(I[n])`. Physical signals: `0.01 <= Ratio <= 1.20`.

### 6.3 Demodulated Audio vs. True RF Detection

If `P_negative / P_positive < 0.02` and `fs <= 96,000 Hz`, the signal is tagged as Demodulated Audio Track. The UI attaches an advisory that measured center frequency represents the receiver's audio pitch, not the RF carrier.

---

## 7. Stage 2: Signal Conditioning

### 7.1 Input Validation Gate

`validate_input_signal()` enforces six preconditions -- see pipeline documentation. Failure returns `{"status": "FAILED", "failed_stage": "input_validation", ...}`.

### 7.2 Local Oscillator (LO) DC Offset Removal

```
x_clean[n] = x[n] - (1/N) * Sum_{k=0}^{N-1} x[k]
```

### 7.3 Scale-Invariant Unit Power Normalization

```
P_avg = (1/N) * Sum |x_clean[n]|^2
scale = sqrt( 1.0 / (P_avg + 1e-12) )
y[n]  = x_clean[n] * scale

Result: E[|y[n]|^2] = 1.0
```

---

## 8. Stage 3: Spectral Analysis

### 8.1 Centered Two-Sided Welch PSD

`scipy.signal.welch()` with Hann window, 50% overlap, two-sided output, centered with `np.fft.fftshift()`.

### 8.2 STFT Spectrogram

Returns 2D matrix: horizontal = frequency, vertical = time, intensity = power density. Used by temporal validator for stationarity assessment.

---

## 9. Stage 4-7: Blind Parameter Engine and Modulation Inference

### 9.1 Layer A boundary: `blind_parameter_engine.py`

The blind engine is the first interpretation boundary. It accepts only the conditioned waveform and sampling rate. It does not import the protocol catalog and contains no protocol names. The hard boundary is tested by `test_14_hard_software_boundary_zero_protocol_strings`.

`parameter_extractor.py` remains as a compatibility facade for existing callers and downstream reports. It can combine the blind vector with later detector telemetry, but the Layer A measurements are produced by `extract_blind_parameters()` before protocol inference.

### 9.2 `BlindParameterVector` contract

The vector records measured or legitimately unavailable physical properties:

| Group | Fields |
|---|---|
| Presence and segmentation | `signal_presence`, duration, discovered activity segments, local noise floor |
| Frequency structure | peak, centroid, median, energy center, dominant components, structure label |
| Bandwidth and power | 3/10/20 dB bandwidth, 99% OBW, SNR, PAPR |
| Timing and states | frequency-state count, state frequencies, tone spacing, symbol-rate candidates, consensus rate, dwell, periodicity |
| Pulse and burst geometry | pulse width, PRI, PRF, burst duration, duty cycle |
| Statistics | envelope, instantaneous frequency, phase, amplitude, flatness, kurtosis, cyclostationary frequencies |
| Geometry and provenance | constellation geometry, morphology fingerprint, parameter consistency, blindness provenance |

`UNKNOWN`, `None`, or an empty collection means that the observation did not support a reliable estimate. The engine does not fill missing measurements with protocol defaults.

### 9.3 Frequency discovery and multi-resolution analysis

The engine computes coarse, medium, and fine spectral views, then derives:

```
peak_frequency
spectral_centroid
spectral_median
energy_center_frequency
dominant_component_frequencies
frequency_structure = SINGLE_COMPONENT | MULTI_COMPONENT | SPREAD_SPECTRUM |
                      CHIRP | HOPPING | PULSED | UNKNOWN
```

This avoids treating `argmax(PSD)` as a universal carrier estimator. For multi-state signals, the center and deviation come from discovered states, for example `center = (f1 + f2) / 2` and `deviation = |f2 - f1| / 2`.

### 9.4 Blind symbol-rate consensus

The engine generates continuous candidates from waveform evidence rather than a finite list of named standards. Independent estimators include envelope autocorrelation, instantaneous-frequency transitions, autocorrelation, zero-crossing/transition periodicity, and cyclic-frequency peaks. The output records the candidate set, consensus value, and method string.

Protocol names may later validate a measured value such as 100.2 Baud. They do not create that value.

### 9.5 Generic FSK, PSK, QAM, pulse, and hopping geometry

For FSK, the order is state discovery, clustering, state frequency estimation, spacing, transition timing, symbol rate, deviation, and physical consistency. For PSK/QAM, the engine estimates amplitude structure, phase structure, cluster count, angular spacing, amplitude levels, and constellation geometry before modulation inference. The same upstream pattern applies to pulse trains, chirps, and frequency hopping.

### 9.6 Speech formants and downstream compatibility

Speech formants remain a specialized downstream measurement using Levinson-Durbin LPC. It is gated by glottal fundamental pitch detection (F0 in 70-350 Hz, autocorrelation peak R_xx > 0.45) and does not alter the blind parameter boundary.

### 9.7 Epistemic parameter tagging and blindness provenance

| Field | Values | Meaning |
|---|---|---|
| `parameter_status` | observed, estimated, nominal, unknown | How value was derived |
| `estimator_method` | e.g., `blind_multi_method_consensus`, `cumulative_linear_power` | Specific algorithm |

`blindness_provenance` records `prior_knowledge_used = NONE` for Layer A, the physical source of each parameter, and the blindness score. A protocol reference can support a later compatibility claim, but it cannot promote a nominal value into a measured one.

---

## 10. Stage 8: Multi-Window Temporal Validation

**NEW in Round-2, Step 1.** This stage addresses the critical weakness of single-window analysis: a brief transient or interference burst can produce misleading spectral signatures if analyzed over the full signal duration.

### 10.1 Four-Window Signal Split

```
Full Signal (N samples)
|<--- Window 1 (N/4) --->|<--- Window 2 (N/4) --->|<--- Window 3 (N/4) --->|<--- Window 4 (N/4) --->|
```

### 10.2 Per-Window Parameter Extraction

Each window independently runs the existing extraction functions where technically valid:

| Parameter | Extraction Method |
|---|---|
| Carrier frequency | Welch PSD peak |
| 99% OBW | Cumulative power search |
| SNR | M2M4 or band-integrated estimator |
| Envelope statistics | Mean amplitude, variance, duty cycle |
| Instantaneous-frequency statistics | Std dev of f_inst[n] |
| Symbol-rate estimate | Dwell histogram (when signal supports it) |
| Frequency-shift estimate | Peak separation (when two peaks present) |
| Pulse metrics | Active duty cycle, pulse width (when pulsed) |

### 10.3 Cross-Window Variation & Physical Reference Normalization

For each parameter P tracked across windows $[P_{w1}, P_{w2}, P_{w3}, P_{w4}]$:

```
mean_P  = mean([P_w1, P_w2, P_w3, P_w4])
std_P   = std([P_w1, P_w2, P_w3, P_w4], ddof=1)   (Sample standard deviation with Bessel correction)
range_P = max_P - min_P
```

To prevent division-by-zero on parameters with near-zero means (such as frequency offsets or centered baseband centroids), variation is **scale-normalized using physical channel references** rather than naive coefficients of variation ($s / \mu$):

- **Carrier Centroid & Peak:** $\text{rel\_var} = \text{std}_P / \text{ref\_bandwidth}$, where $\text{ref\_bandwidth} = \max(\text{OBW}_{\text{avg}}, 0.02 \cdot f_s, 100\text{ Hz})$.
- **Occupied Bandwidth:** $\text{rel\_var} = \text{std}_P / \max(|\text{mean}_P|, 100\text{ Hz})$.
- **Signal-to-Noise Ratio:** $\text{rel\_var} = \text{std}_P / 10.0\text{ dB}$ (normalized to a standard 10 dB dynamic range decade).
- **Missing Parameters:** If a parameter is detected in fewer than 2 windows, it is tagged `"status": "INSUFFICIENT_WINDOW_DETECTIONS"` and excluded from stability scoring, rather than being erroneously coerced to zero.

### 10.4 Continuous Rational Decay Scoring

To eliminate brittle heuristic cliffs (e.g. "0.70 means unstable"), the engine applies a **continuous rational decay mapping** to each parameter's normalized variation:

$$S_P = \frac{1}{1 + \left(\frac{\text{rel\_var}_P}{\tau_P}\right)^2}$$

where $\tau_P$ is the physical tolerance parameter defined in `TemporalValidationConfig` (e.g., $\tau = 0.15$ for carrier stability, $\tau = 0.25$ for bandwidth, $\tau = 0.35$ for SNR).

The overall cross-window consistency score $S_{\text{temporal}} \in [0.0, 1.0]$ is the weighted sum of active parameter scores:

$$S_{\text{temporal}} = \frac{\sum_{P} w_P \cdot S_P}{\sum_{P} w_P}$$

**Continuous vs. Burst/Transient Signal Regimes:**
- **Stationary Continuous Signals:** All 4 windows evaluate carrier, OBW, SNR, envelope variance, and instantaneous frequency stability.
- **Pulsed / Burst Signals (Radar & TDMA):** Pulse geometry (PRF, pulse width, duty cycle) is analyzed independently per window across detected pulses. For signals with duration below 1,024 samples ($4 \times 256$), the validator returns `INSUFFICIENT_OBSERVATION_DURATION` gracefully without crashing downstream pipeline stages.

### 10.5 Temporal Consistency Output

```json
{
  "status": "VALIDATED",
  "windows_analyzed": 4,
  "cross_window_consistency_score": 0.87,
  "stability_level": "HIGH",
  "parameter_stability": {
    "fc_peak_hz": {"mean": 2131.8, "std": 1.25, "rel_var": 0.004, "stability_score": 0.98, "stability": "HIGH"},
    "bw_99pct_hz": {"mean": 322.0, "std": 3.1, "rel_var": 0.009, "stability_score": 0.97, "stability": "HIGH"},
    "snr_db": {"mean": 11.6, "std": 0.35, "rel_var": 0.035, "stability_score": 0.96, "stability": "HIGH"},
    "envelope_variance_ratio": {"mean": 0.19, "std": 0.015, "rel_var": 0.079, "stability_score": 0.93, "stability": "HIGH"}
  },
  "unstable_parameters": []
}
```

---

## 11. Stage 10: Protocol Inference and Autonomous Classification

### 11.1 Physical Invariants vs. Black-Box Neural Networks

This is Layer C, downstream of the blind vector and modulation inference. It tests physical equations derived from ITU-R, CCIR, ETSI, and MIL-STD standards. Every classification decision produces an auditable chain of physical evidence. The open-set AMC adds ML-based features as a secondary layer only; it does not supply Layer A parameter values.

### 11.2 The Multi-Domain Feature Vector

Five domains: Spectral, Envelope, Instantaneous Frequency, Non-linear Squaring, and Higher-Order Cumulants.

### 11.3 Gaussian Noise and Silence Pre-Gate

**Silence Gate:** Power < 1e-12 (-120 dBFS) -> UNKNOWN / Pure Silence.

**Gaussian Noise Gate** -- all nine conditions must pass simultaneously:
```
sfm       >= 0.70    (Uniform flat spectrum)
pk_to_med  < 8.0     (No sharp spectral peaks)
sq_prom    < 8.0     (No BPSK/QPSK squaring line)
pk4_prom   < 8.0     (No 4th-power spectral line)
|c40|      < 0.25    (Near-zero 4th-order cumulant)
|c42|      < 0.25    (Near-zero cross-cumulant)
|c20|      < 0.25    (Near-zero 2nd-order cumulant)
(No verified pulsed signal with SNR >= 3 dB)
chirp_r2   < 0.25    (No linear frequency trajectory)
```

### 11.4 The 19-Rule Mutually Exclusive Decision Tree

```
+-------------------------------------------------------------------------------+
|                      19-Rule Physical Invariant Decision Tree                 |
+-------------------------------------------------------------------------------+
| Rule 1:  Pure CW Carrier          (Var < 0.12, OBW < 150 Hz)                 |
| Rule 2:  Morse Telegraphy (OOK)   (Dot/Dash 1:3 ratio, DR > 0.65)            |
| Rule 3:  CODAR Ocean Radar        (0.8-5 Hz sawtooth sweep, SFM >= 0.70)     |
| Rule 4:  OTH-SW Radar             (43.2 Hz envelope line prominence >= 12)   |
| Rule 5:  Ghadir OTH Radar         (307/870 Hz prominence >= 12)              |
| Rule 5B: Duga Woodpecker Radar    (10.0 Hz knocking PRF, duty < 35%)         |
| Rule 5C: GRAVES Space Radar       (143.050 MHz VHF reflection, OBW < 3 kHz)  |
| Rule 6:  HAARP / Iono Sounder     (Linear chirp R2 >= 0.45 or stepped tones) |
| Rule 7:  GSM Cellular (2G)        (216.7 Hz TDMA frame line, OBW >= 12 kHz)  |
| Rule 8:  DMR Mobile Radio         (33.3 Hz timeslot line, OBW 5.5-14 kHz)    |
| Rule 9:  2G ALE (MIL-STD-188-141) (7-9 tones on exact 250 Hz harmonic grid)  |
| Rule 10: PSK31 vs FT8             (OBW < 140 Hz: squaring line separates)    |
| Rule 11: MFSK16                   (16 tones with 15.625 Hz uniform spacing)  |
| Rule 12: 2-FSK Utility Modes      (Dual peaks + dwell: NAVTEX/ASCII/RTTY)    |
| Rule 13: APRS Bell 202            (1200/2200 Hz tones, OBW 4-12 kHz)         |
| Rule 14: WEFAX Facsimile          (OBW 1.4-2.6 kHz, fc 1.2-2.4 kHz)          |
| Rule 15: STANAG 4285              (NATO serial 8-PSK, SFM >= 0.50)           |
| Rule 15B:Satellite Telemetry      (Subcarrier 2150-2650 Hz + PCM/PM bands)   |
| Rule 16: POCSAG / AIS / D-STAR    (Wideband audio passband triage)           |
| Rule 17: Analog Voice (NFM)       (Confirmed glottal pitch + >= 2 formants)  |
| Rule 18: Generic Pulsed Radar     (Inter-pulse quiet floor, duty < 65%)      |
| Rule 18B/C: Generic 2-FSK/4-FSK  (Multi-peak spectrum with constant env)     |
| Rule 19: Generic Digital Comms    (Higher-order cumulant boundaries)         |
+-------------------------------------------------------------------------------+
| Pre-Gate 0: Silence (<-120 dBFS)             -> UNKNOWN                      |
| Pre-Gate N: Gaussian Noise (all 9 conditions) -> UNKNOWN                     |
+-------------------------------------------------------------------------------+
```

### 11.5 Open-Set Rejection and UNKNOWN Outcomes

Evidence scores represent the number and margin of passing physical criteria -- **not calibrated posterior probabilities**. The open-set AMC returns `UNKNOWN_OOD` when the feature vector does not fall within any trained modulation class boundary.

---

## 12. Stage 11: Candidate Hypothesis Ranking

**Round-2 reasoning layer.** The protocol inference layer consumes the blind parameter vector and modulation result, then constructs a structured, competing candidate pool. The autonomous detector remains an important physical-rule source, but it no longer represents the blind measurement layer.

### 12.1 Dynamic Candidate Generation from Physical Invariants

A critical design requirement is that **alternative hypotheses must be genuinely generated from physical observations**, never fabricated merely for UI presentation:

1. **Primary Candidate:** Produced from the blind vector, modulation geometry, and any compatible physical-rule result.
2. **Secondary Contenders via Evaluated Physical Rejections:** The protocol layer tests candidate standards against already measured parameters. When a rule matches high-level physical traits (e.g. dual-tone 2-FSK structure) but fails on a fine-grained physical invariant (e.g. bit dwell timing or shift), the detector records an explicit rejection entry in `rejected_hypotheses` formatted as:
   `"PROTOCOL_NAME: Exact physical contradiction reason"`  
   *(e.g. `"ASCII_TELEPRINTER: Dwell time 10.0 ms does not match 9.09 ms nominal (110 Baud)"`)*.
3. **Contender Assembly:** The `CandidateRanker` parses these evaluated entries (`_extract_parameters_dict`), creating genuine candidate hypotheses. Each contender inherits shared measured physical features (e.g. carrier center and occupied bandwidth) as `supporting_evidence`, while the exact rejection motive is attached as a measurable `contradiction`. No candidate is allowed to rewrite the blind measurements.
4. **Out-of-Distribution & Ambiguous Contenders:** When spectral features show multi-modal ambiguity without a clear winner, or when features deviate from all known cataloged distributions, the generator produces explicit `AMBIGUOUS` or `UNKNOWN_OOD` candidates with their supporting measurements.

### 12.2 Three-Part Evidence Score Formulation

To maintain scientific integrity, the score is explicitly an uncalibrated epistemic metric, **never a Bayesian probability**:

$$\text{net\_evidence\_score} = \text{clip}\left(\text{base\_evidence\_score} - \text{total\_contradiction\_penalty}, 0.0, 1.0\right)$$

- **$\text{base\_evidence\_score} \in [0.0, 1.0]$:** The initial score derived from the count and margin of passing physical invariant criteria in the autonomous detector.
- **$\text{total\_contradiction\_penalty}$:** The sum of explicit penalty deductions from the contradiction analysis layer:
  $$\text{total\_contradiction\_penalty} = \min\left(\sum_{k} P_k, P_{\text{max\_total}}\right)$$
  capped at $P_{\text{max\_total}} = 0.80$ to prevent negative score underflow.
- **$\text{net\_evidence\_score}$:** The final penalised evidence metric used for candidate ranking and threshold comparisons.

```python
@dataclass
class RankedCandidate:
    hypothesis_id: str               # Unique UUID identifier
    signal_class_id: str             # e.g., "MARITIME_NAVTEX"
    candidate_generation_reason: str # How candidate was derived (e.g., "Primary Rule 12" or "Dual-peak FSK contender")
    base_evidence_score: float       # Initial physical observation score
    contradiction_penalty: float     # Total penalty deducted
    evidence_score: float            # Net evidence score after contradiction penalties
    supporting_evidence: List[str]   # Explicit measurable criteria that passed
    contradictions: List[str]        # Explicit measurable criteria that failed
    temporal_consistency: float      # From Stage 5 temporal validator [0,1]
    physical_consistency: float      # Feature vector alignment [0,1]
    reconstruction_consistency: float# From Stage 11 reconstruction telemetry [0,1]
    rejection_reason: Optional[str]  # Why candidate was demoted or rejected
```

### 12.3 Ranking Invariants

1. Candidates are ranked strictly by `evidence_score` (net score) in descending order.
2. Evidence score is **never described or marketed as probability** in any log, API response, or UI widget.
3. If the delta between the top two candidates is within the ambiguity threshold ($|\Delta| < 0.05$), the decision status is declared `AMBIGUOUS`.
4. If all candidates fail the viability floor ($\text{score} < 0.35$), the decision is declared `UNKNOWN` or `UNKNOWN_OOD`.
5. Filenames are strictly excluded from classification and candidate ranking evidence.

---

## 13. Stage 12: Contradiction Analysis

**NEW in Round-2, Step 3.** For every candidate hypothesis in the pool, the contradiction analyzer evaluates supporting and contradicting evidence strictly against measured multi-domain physical features, temporal stability metrics, and closed-loop reconstruction telemetry. No evidence is invented.

### 13.1 Measurable Contradiction Categories

Every contradiction must cite a specific physical measurement, an expected boundary, and an observed value:

| Category | Measurable Trigger | Example Mismatch |
|---|---|---|
| **SPECTRAL** | Missing characteristic spectral lines, invalid tone count | Tone spacing 450 Hz != nominal 170 Hz (+-30%) |
| **TEMPORAL** | Envelope variance violation (constant vs. pulsed) | Hypothesis specifies pulsed radar, but envelope duty cycle is 100% |
| **PARAMETER_INCONSISTENCY** | Carson's rule divergence, Nyquist violation, dwell error | Dwell duration 10.0 ms != 9.09 ms nominal (110 Baud) |
| **CROSS_WINDOW_INSTABILITY** | Multi-window parameter variation exceeds tolerance | Carrier frequency CV > 0.05 across windows |
| **RECONSTRUCTION_FAILURE** | Carrier PLL phase loss, EVM > 35%, soft LLR collapse | Demodulation EVM 42.1% exceeds 35.0% threshold |

### 13.2 Structured Output Schema

```json
{
  "hypothesis": "MARITIME_NAVTEX",
  "supporting_evidence": [
    "Two distinct PSD peaks at 2110.25 Hz and 2304.05 Hz",
    "Frequency shift 193.8 Hz within 170 Hz nominal +- 30% tolerance",
    "Dwell histogram peak at 10.0 ms (100.0 Baud)",
    "Carson bandwidth 293.8 Hz matches measured OBW 323.0 Hz"
  ],
  "contradictions": [],
  "contradiction_count": 0,
  "base_evidence_score": 0.98,
  "contradiction_penalty": 0.0,
  "net_evidence_score": 0.98
}
```

### 13.3 Configurable Penalty Weights (`ContradictionConfig`)

Penalty weights are explicit, documented hyper-parameters in `dsp/contradiction_analyzer.py` and can be adjusted without touching core DSP algorithms:

```python
@dataclass
class ContradictionConfig:
    penalty_spectral: float = 0.20        # Missing spectral lines, tone spacing error
    penalty_temporal: float = 0.15        # Envelope variance or dynamic range mismatch
    penalty_parameter: float = 0.20       # Carson / Nyquist / symbol dwell time violation
    penalty_cross_window: float = 0.15    # Multi-window temporal parameter instability
    penalty_reconstruction: float = 0.20  # PLL lock loss, excessive EVM (>35%)
    max_total_penalty: float = 0.80       # Deductions capped to prevent negative underflow
```

These weights represent engineering judgments based on relative reliability of physical evidence sources, rather than statistically fitted posterior weights. Evaluators can reconfigure them per operational environment.

---

## 14. Stage 13: Adaptive Extraction Pipeline

### 14.1 The Strategy Design Pattern

Once candidate hypotheses are generated and ranked, the leading viable hypothesis is dispatched to an extractor implementing the `BaseExtractor` contract:

```python
class BaseExtractor:
    def extract(self, signal, fs, detection_meta, pulse_info, base_params):
        raise NotImplementedError
```

Available extractors: `pulsed_radar`, `fsk_detector`, `mfsk_comb`, `tdma_burst`, `digital_psk_qam`, `analog_voice`, `analog_wefax`, `continuous_wave`, `ook_morse`, `satellite_telemetry`, `generic_fallback`.

### 14.2 Radar Metrics

```
Range Resolution:       dR = c / (2 * B_99%)          c = 299,792,458 m/s
Max Unambiguous Range:  R_max = c / (2 * PRF)
Chirp Sweep Rate:       Linear slope in MHz/s with R^2 goodness-of-fit
```

### 14.3 FSK Metrics

```
Mark and Space Frequencies: f_mark, f_space
Frequency Shift:          df = f_space - f_mark
Modulation Index:         h = df / Rs
Carson Bandwidth:         B_carson = df + Rs
Symbol Dwell Time:        Ts = 1000 / Rs  ms
```

### 14.4 Cellular and Tactical TDMA Metrics

```
Frame Period:    T_frame  (GSM = 4.615 ms, DMR = 60.0 ms)
Timeslot Dur.:   T_slot   (GSM = 576.9 us, DMR = 30.0 ms)
Burst Duty Cycle: % of frame occupied by active transmission
Gated Baud Rate: Symbol rate during active burst slots only
```

### 14.5 Digital PSK/QAM

```
Constellation Order: M (2, 4, 8, 16)
EVM = mean( | |y[n]| - 1.0 | ) * 100%     clipped to [2.0, 45.0]%
```

---

## 15. Stage 14: Physical Reconstruction Engine

The physical reconstruction engine executes as a closed-loop chain on the conditioned signal to generate physical and algebraic evidence before final validation.

### 15.1 IQ Imbalance Conditioning

Gram-Schmidt procedure corrects amplitude and phase imbalance between I and Q rails. Telemetry stored in `reconstruction_telemetry["iq_imbalance"]`.

### 15.2 20-D Feature Extraction

| Dimensions | Features | Domain |
|---|---|---|
| 1-4 | Cumulants: C20, C40, C42, kurtosis | Statistical |
| 5-6 | Normalized moments M2, M4 | Statistical |
| 7-8 | Envelope variance, PAPR | Amplitude |
| 9-10 | SFM, bandwidth | Spectral |
| 11-12 | Instantaneous freq std, chirp R2 | Phase/Freq |
| 13-14 | Squaring peak prominence, 4th-power peak | Non-linear |
| 15-16 | Zero-crossing rate, autocorrelation lag-1 | Temporal |
| 17-18 | Phase variance, constellation compactness | Phase |
| 19-20 | Cyclostationary alpha-profile peaks | Cyclostationary |

### 15.3 Open-Set Modulation Classifier (AMC)

Computes Euclidean distance to known modulation centroids. If minimum distance exceeds `_OOD_DISTANCE_THRESHOLD`:

```json
{ "modulation": "UNKNOWN_OOD", "confidence_level": "UNKNOWN" }
```

`UNKNOWN_OOD` is a first-class outcome -- the signal is explicitly recognized as out-of-distribution rather than forced into a nearest neighbor.

### 15.4 Synchronization: Carrier and Timing Recovery

Outputs: `coarse_cfo_hz`, `fine_cfo_hz`, `residual_cfo_hz`, `pll_locked`, `pll_lock_metric`, `timing_jitter`, `cycle_slips`, `observability_status`.

### 15.5 Demodulation with Soft LLRs

Symbol decisions with soft log-likelihood ratios. LLRs clipped to [-20, +20] to prevent downstream decoder overflow.

### 15.6 De-interleaving Candidate Search

Searches block, convolutional, diagonal, and pseudo-random interleaver topologies.

### 15.7 FEC Hypothesis Evaluation

Tests convolutional, LDPC, and Reed-Solomon FEC hypotheses. Reports: `syndrome_zero` (algebraic proof), `syndrome_weight`, `ber_estimate`.

### 15.8 Framing, Sync-Word Correlation, and CRC Validation

Cross-correlates bit stream against known sync words from GSM, AIS, POCSAG, NAVTEX, FT8. A `crc_match = True` is the strongest available evidence -- closed-loop mathematical proof.

### 15.9 Multi-Stage Evidence Fusion

| Gate | Source | Weight |
|---|---|---|
| G1 | Modulation hypothesis confidence | 0.25 |
| G2 | Synchronization lock quality | 0.25 |
| G3 | Demodulation EVM quality | 0.20 |
| G4 | FEC syndrome validation | 0.15 |
| G5 | Framing CRC pass rate | 0.15 |

---

## 16. Stage 15: Hypothesis Evidence Validation Gate

**NEW in Round-2, Step 4. The central architectural validation gate.**

### 16.1 Philosophy & Execution Sequence

The Validation Gate executes **after** both specialized extraction and physical reconstruction have generated their telemetry. This eliminates circular dependencies: Gate 6 (Reconstruction Consistency) evaluates closed-loop evidence that has already been computed.

```
BLIND MEASUREMENTS (Spectral, Temporal, Envelope)
        |
CANDIDATE HYPOTHESIS GENERATION & RANKING
        |
SPECIALIZED PARAMETER EXTRACTION
        |
PHYSICAL RECONSTRUCTION (Sync, Demod, FEC, CRC)
        |
CONTRADICTION ANALYSIS (Supporting vs. Contradicting)
        |
SIX-GATE EVIDENCE VALIDATION PIPELINE
        |
FINAL DECISION (VALIDATED / ESTIMATED / AMBIGUOUS / UNKNOWN)
```

### 16.2 The Six-Gate Evidence Validation Pipeline

Every gate produces one of three explicit outcomes:
1. **PASS:** Measured parameters satisfy the physical criterion.
2. **FAIL:** Measured parameters violate the physical criterion (triggers demotion or rejection).
3. **NOT_APPLICABLE:** The gate is not relevant to this waveform class (e.g. FEC/CRC for analog voice, CW, or raw radar pulses).

| Gate | Validation Scope | Applicable Regimes | Passing Criterion |
|---|---|---|---|
| **G1: Spectral Consistency** | Bounded in-band energy, Nyquist compliance, OBW stability | All signals | $0 < \text{OBW} \le f_s$, $|f_c| \le f_s/2$, BW rel_var $\le 25\%$ |
| **G2: Temporal Consistency** | Pulse train presence for pulsed modes; duty cycle continuity for continuous | All signals | Pulse count $\ge 1$ for radar; duty cycle $\ge 10\%$ for continuous |
| **G3: Symbol / Dwell Consistency** | Multi-window symbol rate stability and dwell histogram fit | Digital FSK, PSK, QAM | Baud rate CV $\le 20\%$; dwell fit verified |
| **G4: Cross-Window Stability** | Stationarity score from Stage 5 across 4 observation windows | Duration $\ge 1024$ samples | $S_{\text{temporal}} \ge 0.60$ |
| **G5: Physical Plausibility** | SNR threshold and physical Nyquist transmission capacity | All signals | $\text{SNR} \ge -10\text{ dB}$, $\text{Baud} \le 2.5 \cdot \text{OBW}$ |
| **G6: Reconstruction Consistency** | Demodulation EVM, PLL carrier lock, FEC syndrome, CRC | Digital demodulated modes | $\text{EVM} \le 35\%$, $\text{PLL\_locked} = \text{True}$, or CRC match |

**Handling Signals without Reconstruction Structure (NOT_APPLICABLE Policy):**
For signals that do not carry digital framing (e.g. unmodulated CW carrier, analog voice, or FMCW radar), G6 is explicitly tagged `applicable = False` (`waived_gates`). It is **never treated as an automatic fake pass**. The candidate can achieve `VALIDATED` if all remaining *applicable* gates (G1, G2, G4, G5) pass with high score and zero unresolved physical contradictions.

### 16.3 Final Validation Status Logic

```
1. If fatal physical plausibility fails (G5: SNR < -10 dB or Baud > 2.5 * OBW):
   -> validation_status = UNKNOWN, epistemic_status = UNKNOWN, candidate REJECTED

2. Elif temporal or cross-window stability fails (G2 or G4 failed):
   -> validation_status = ESTIMATED (Downgraded: strong spectral evidence, but failed temporal gate)

3. Elif unresolved physical contradictions > 0 or net_evidence_score < 0.60:
   -> validation_status = ESTIMATED (Retained as ESTIMATED with explicit contradiction audit)

4. Elif any other applicable gate failed (e.g. G3 dwell mismatch or G6 EVM > 35%):
   -> validation_status = ESTIMATED

5. Elif all applicable gates passed with zero contradictions:
   -> validation_status = VALIDATED, epistemic_status = VALIDATED

UNKNOWN_OOD is preserved as a first-class outcome when input deviates from cataloged distributions.
```

### 16.4 Validation Audit Trail

```json
{
  "validation_status": "VALIDATED",
  "validation_score": 0.98,
  "passed_gates": [
    "spectral_consistency",
    "temporal_consistency",
    "symbol_dwell_consistency",
    "cross_window_stability",
    "physical_consistency",
    "reconstruction_consistency"
  ],
  "failed_gates": [],
  "waived_gates": [],
  "gate_results": {
    "spectral_consistency": {"passed": true, "applicable": true, "score": 1.0, "observed": "OBW=323.0 Hz, fc=2131.8 Hz"},
    "temporal_consistency": {"passed": true, "applicable": true, "score": 1.0, "observed": "Continuous FSK carrier"},
    "symbol_dwell_consistency": {"passed": true, "applicable": true, "score": 0.95, "observed": "100.0 Baud dwell confirmed"},
    "cross_window_stability": {"passed": true, "applicable": true, "score": 0.98, "observed": "score=0.982 >= 0.600"},
    "physical_consistency": {"passed": true, "applicable": true, "score": 1.0, "observed": "SNR=11.7 dB, OBW=323.0 Hz"},
    "reconstruction_consistency": {"passed": true, "applicable": true, "score": 1.0, "observed": "Demodulated & SITOR-B framed"}
  }
}
```

---

## 17. Stage 16: Parameter Uncertainty Reporting

**NEW in Round-2, Step 5.**

### 17.1 Scientific Honesty Invariants

Every numerical parameter report enforces three scientific rules:
1. **Never fabricate a 95% confidence interval.** An instrument frequency resolution (e.g. Welch FFT bin spacing) is an *estimator resolution bound*, NOT a statistical 95% confidence interval.
2. **Distinguish Single-Window from Multi-Window Uncertainty:**
   - **Single-Window Frequency:** Reported as **Estimator Resolution Bound** $\pm \Delta f_{\text{bin}} / 2 = \pm f_s / (2 \cdot N_{\text{FFT}})$, with `"uncertainty_type": "fft_bin_resolution"`.
   - **Multi-Window Temporal (n=4):** Reported as **Cross-Window Sample Standard Deviation** ($s$, with Bessel correction $ddof=1$), and empirical observed span $[\min, \max]$ or $[x - 1.96s, x + 1.96s]$ (`"uncertainty_type": "cross_window_std"`).
   - *Note on small-sample statistics:* With $n=4$ windows, the Student's t critical value for a 95% confidence interval of the window mean is $t_{0.025, df=3} = 3.182$. Rather than assuming Gaussian asymptotics, the system reports empirical sample standard deviation and observed range directly.
3. **Explicit Null with Mandatory Reason:** When uncertainty cannot be legitimately estimated from multi-window variation or physical resolution, `"uncertainty": null` is returned with an explicit explanation (e.g., single observation window or discrete nominal parameter).

### 17.2 Per-Parameter Report Structure (Actual Schema)

```json
{
  "carrier_frequency": {
    "value": 2207.15,
    "status": "OBSERVED",
    "stability": "HIGH",
    "uncertainty": 10.77,
    "uncertainty_type": "fft_bin_resolution",
    "uncertainty_reason": null,
    "reason": "Welch PSD frequency bin resolution grid bound (+- fs / 4096)",
    "range": [2196.38, 2217.92],
    "unit": "Hz"
  },
  "symbol_rate": {
    "value": 100.0,
    "status": "ESTIMATED",
    "stability": "HIGH",
    "uncertainty": 0.30,
    "uncertainty_type": "cross_window_std",
    "uncertainty_reason": null,
    "reason": "Cross-window clock dwell variation across 4 temporal windows",
    "range": [99.41, 100.59],
    "unit": "Baud"
  },
  "snr_db": {
    "value": 11.7,
    "status": "ESTIMATED",
    "stability": "HIGH",
    "uncertainty": 0.35,
    "uncertainty_type": "cross_window_std",
    "uncertainty_reason": null,
    "reason": "Cross-window M2M4 moment estimator variation",
    "range": [11.01, 12.39],
    "unit": "dB"
  },
  "modulation_order": {
    "value": null,
    "status": "UNKNOWN",
    "stability": "UNKNOWN",
    "uncertainty": null,
    "uncertainty_type": null,
    "uncertainty_reason": "Continuous FSK modulation does not carry a discrete QAM constellation order",
    "reason": "Parameter not applicable to this modulation family",
    "range": null,
    "unit": ""
  }
}
```

### 17.3 Stability Classification

| Stability Level | Criterion |
|---|---|
| **HIGH** | Stability score $\ge 0.85$ and parameter marked stable across windows |
| **MEDIUM** | Stability score $0.60 \le S < 0.85$ |
| **LOW** | Stability score $< 0.60$ |
| **UNKNOWN** | Single observation window or insufficient window detections ($< 2$) |

### 17.4 Uncertainty Estimation Methods

| Method (`uncertainty_type`) | When Used | Physical Meaning |
|---|---|---|
| `cross_window_std` | Parameter detected in $\ge 2$ windows | Sample standard deviation ($ddof=1$) of parameter across sequential observation slices |
| `fft_bin_resolution` | Single-window spectral peak | Instrument frequency quantization bound: $\pm f_s / (2 \cdot N_{\text{FFT}})$ |
| `m2m4_variance` | Single-window continuous PSK/QAM SNR | Theoretical variance of fourth-order sample moments |
| `null` | Unavailable or single observation | Explicitly set to null; reason field is mandatory |

---

## 18. Extended Epistemic Hierarchy Contract

**UPDATED in Round-2 Steps 1-5.** The `contracts.py` SignalHypothesis dataclass was extended from 7 fields to a 21-field intelligence contract.

### 18.1 EpistemicStatus Tiers

| Status | Meaning | Example |
|---|---|---|
| OBSERVED | Direct physical measurement | FFT peak frequency, pulse width from envelope |
| ESTIMATED | Deterministic continuous estimator | CFO from PLL, SNR from M2M4, baud from dwell histogram |
| HYPOTHESIZED | Model candidate inference | Modulation family from cumulants |
| VALIDATED | Closed-loop mathematical proof | Syndrome == 0, CRC pass |
| UNKNOWN | Undetermined; prevents forced guesses | Parameters that could not be extracted |
| NOT_APPLICABLE | Not relevant to this waveform class | Baud rate for a CW carrier |

### 18.2 The Extended SignalHypothesis Object

```python
@dataclass
class SignalHypothesis:
    # --- Core Identity ---
    hypothesis_id: str                        # UUID, unique per analysis run
    signal_family: str                        # e.g., "FSK", "RADAR", "UNKNOWN"
    modulation: ModulationFamily
    protocol: Optional[str]                   # e.g., "NAVTEX", "STANAG_4285"

    # --- Parameters ---
    parameters: Dict[str, Any]
    parameter_status: Dict[str, EpistemicStatus]
    parameter_uncertainty: Dict[str, Any]     # Per-parameter uncertainty report

    # --- Evidence ---
    supporting_evidence: List[str]
    contradictions: List[str]
    temporal_consistency: float               # Stationarity score from Stage 5 [0,1]
    physical_consistency: float               # Cumulant/feature alignment [0,1]
    reconstruction_consistency: float         # Evidence fusion score from Stage 12 [0,1]
    cross_window_consistency: float           # Cross-window stability score [0,1]

    # --- Scoring ---
    evidence_score: float                     # Scalar evidence count (NOT probability)
    confidence_level: ConfidenceLevel         # HIGH / MEDIUM / LOW / UNKNOWN

    # --- Epistemic Classification ---
    epistemic_status: EpistemicStatus
    validation_status: str                    # "VALIDATED" / "ESTIMATED" / "AMBIGUOUS" / "UNKNOWN"
    rejection_reason: Optional[str]

    # --- Legacy backward-compatible fields ---
    symbol_rate: Optional[float]
    carrier_offset: float
    confidence: float                         # Legacy alias for evidence_score
    evidence: List[str]                       # Alias for supporting_evidence
    rejected_hypotheses: List[str]
```

### 18.3 UNKNOWN_OOD as First-Class Outcome

`UNKNOWN_OOD` is never a fallback error. It means the open-set classifier explicitly recognized the signal as outside its modeled space. The validation gate preserves `UNKNOWN_OOD` when no candidate survives validation.

---

## 19. Pulse Analysis

### 19.1 Adaptive Sparse-Pulse Detection

Multi-stage adaptive envelope threshold (not fixed percentile):

```
Step 1: Noise floor statistics
    noise_floor  = percentile(envelope, 15)
    sigma_noise  = 1.4826 * MAD(envelope)   (Robust noise estimate)

Step 2: Multiple upper percentiles
    p99   = percentile(envelope, 99.0)
    p99_9 = percentile(envelope, 99.9)
    p_max = max(envelope)

Step 3: Sparse-event detection
    is_sparse = (noise_median/p99 < 0.25) AND (p_max > noise_floor + 5*sigma) AND (p99_9 > noise_floor + 3*sigma)

Step 4: Threshold selection
    if is_sparse:
        peak_level = max(p99_9, 0.50 * p_max)   (robust to low-duty-cycle radar)
    else:
        peak_level = p99
```

### 19.2 Intra-Pulse Modulation Analysis (FMOP vs. TDMA)

A verified radar chirp requires: `is_fmop_chirp = True`, `r2 >= 0.35`, `chirp_bandwidth >= 75 Hz`, `chirp_rate >= 10,000 Hz/s`, and either a matched radar frame or `PW >= 800 us` with `duty < 65%` and `r2 >= 0.40`.

**Round-2 Regression Fix:** The r2 threshold for the high-duty-cycle branch (PW >= 800 us) was raised from 0.50 to 0.40. The HAARP chirp at 100,000-sample chunk size has `mean_pw_s = 4.92ms >= 800us`, `duty = 47.1%`, and `r2 = 0.492` (below old 0.50 threshold, above new 0.40 threshold). AIS (with `mean_pw_s = 188.8us < 800us`) never enters this branch -- the 800 us gate is the discriminating physical invariant. All 6 HAARP chunk sizes now pass, and AIS correctness is preserved.

### 19.3 PRF/PRI via Envelope Autocorrelation and Harmonic Scoring

```
Step 1: Smooth envelope (Gaussian kernel sigma = 0.035% of fs)
Step 2: Circular autocorrelation via FFT, normalize to R_xx[0] = 1.0
Step 3: Search peaks with min_prf = 0.5 Hz (PRI up to 2 seconds)
Step 4: Score each candidate:
    score  = (height / max_height) + 0.35 * (count of harmonic multiples present)
    score -= 2.0   if subharmonic of an earlier stronger candidate
    score += 1.5   if lag matches known standard radar/TDMA frame
Step 5: Select candidate with best score as fundamental PRI
```

### 19.4 Radar vs. TDMA Discrimination

**Radar** requires one of: verified FMOP chirp, matched radar standard frame with pulsed CW, or FMOP with bandwidth >= 75 Hz and matched frame/PW >= 800 us.

**TDMA** requires one of: known standard frame match (GSM, DMR, TETRA), intra-pulse modulation consistent with digital burst, or PW >= 300 us with matched TDMA frame.

Radar and TDMA flags are mutually exclusive.

---

## 20. Interactive Visualizations and Professional UI

**UPDATED in Round-2 Step 6 (Professional Engineering Workstation).**

### 20.1 9-Level Hierarchical Presentation (Round-2 UI)

1. **Epistemic Verdict Badge** -- VALIDATED / ESTIMATED / AMBIGUOUS / UNKNOWN / UNKNOWN_OOD / NOISE FLOOR, color-coded.
2. **Parameter Cards with Uncertainty** -- value, epistemic status icon, stability indicator, +/- uncertainty bounds.
3. **Hypothesis Ranking Panel** -- all candidates with evidence_score, supporting count, contradiction count, validation status.
4. **Multi-Window Stability Table** -- per-parameter cross-window values; stability HIGH/MEDIUM/LOW color-coded.
5. **Validation Trace Audit** -- all 6 gate results: pass/fail, score, note.
6. **Contradiction Detail Panel** -- active contradictions with category, weight applied, measurable property.
7. **Physical Reconstruction Telemetry** -- IQ conditioning, sync lock, demodulation EVM, FEC syndrome, CRC.
8. **Expandable Technical Visualizations** -- PSD, STFT waterfall, constellation, eye diagram, envelope oscilloscope.
9. **Downloadable Intercept Reports** -- JSON and CSV export of complete pipeline output.

### 20.2 Visualization Panels

- **PSD:** Decibel power vs. frequency, fc_peak overlay, -3 dB bandwidth shaded rectangle.
- **STFT Waterfall:** Viridis colormap -- horizontal stripes = FSK, diagonal = radar chirp, periodic = TDMA.
- **I/Q Constellation:** WebGL Scattergl, unit circle overlay -- 2 real-axis points = BPSK, ring = FSK/FM, grid = QAM.
- **Eye Diagram:** Synchronized symbol transitions over 2 symbol intervals; wide eye = high SNR.
- **Oscilloscope:** I[n], Q[n], |s[n]| for pulse width and keying transient inspection.

---

## 21. Tactical Intercept Scenario Selector

**NEW in Round-2, Step 7.**

### 21.1 Architecture Invariant

The scenario selector is strictly a demonstration convenience. It does not influence classification logic. No signal filename, scenario label, or user selection ever enters the autonomous detector as a classification input. Classification output is identical regardless of how a signal was loaded.

### 21.2 Seven Canonical Scenarios

| # | Scenario | Expected Epistemic Outcome |
|---|---|---|
| 1 | Standard Known Signal | VALIDATED |
| 2 | Low-SNR Signal | ESTIMATED |
| 3 | Noise Floor | UNKNOWN / NOISE FLOOR |
| 4 | Unknown / OOD Signal | UNKNOWN_OOD |
| 5 | Ambiguous Signal | AMBIGUOUS |
| 6 | Pulsed Signal / Radar | VALIDATED (Radar type) |
| 7 | Distorted Signal | ESTIMATED (degraded) |

### 21.3 Six First-Class Epistemic Application States

| State | Meaning | UI Color |
|---|---|---|
| VALIDATED | Signal passed all validation gates | Green |
| ESTIMATED | Strong evidence, partial validation | Blue |
| AMBIGUOUS | Multiple candidates, none dominant | Orange |
| UNKNOWN | Insufficient evidence | Gray |
| UNKNOWN_OOD | Explicitly outside modeled class space | Purple |
| NO SIGNAL / NOISE FLOOR | Input is pure thermal noise | Dark gray |

---

## 22. Complete Step-by-Step Walkthrough: NAVTEX Signal

### 22.1 Background and Maritime Standards

NAVTEX broadcasts navigational and meteorological warnings on 518 kHz. Uses 2-FSK with 170 Hz nominal shift and 100 Baud symbol rate (CCIR Recommendation 476 / SITOR-B synchronous 7-bit framing).

### 22.2 Stage 1: Ingestion

```
Sample rate: 44,100 Hz    Channels: 1 (mono)
Samples: 946,254          Duration: 21.457 seconds
Domain: Demodulated Audio Track
```

### 22.3 Stage 2: Preprocessing

```
DC offset removed: 0.0001
Raw power: 0.12555    Normalization scale: 2.822
Result: E[|y[n]|^2] = 1.000
```

### 22.4 Stages 3-4: Spectral Analysis and Blind Parameter Extraction

To avoid confusion during evaluation, all frequency quantities are explicitly delineated by their physical definitions:

```
Mark Tone Frequency (f_mark):        2110.25 Hz  (Lower FSK spectral peak)
Space Tone Frequency (f_space):      2304.05 Hz  (Upper FSK spectral peak)
Two-Tone Midpoint / Center (fc_mid): 2207.15 Hz  (0.5 * (f_mark + f_space) -- nominal carrier center)
Spectral Peak (fc_peak):             2131.79 Hz  (Highest power bin in PSD, corresponding to Mark tone)
Spectral Centroid (fc_centroid):     2201.33 Hz  (Power-weighted first moment of the passband spectrum)
99% Occupied Bandwidth:              323.00 Hz   (Cumulative PSD search 0.5% to 99.5%)
Signal-to-Noise Ratio:               11.71 dB    (M2M4 fourth-order moment estimator)
Peak-to-Average Power Ratio:         5.90 dB
Envelope Variance Ratio:             0.190       (Low variance; consistent with constant-envelope 2-FSK)
```

### 22.5 Stage 8: Multi-Window Temporal Validation

The preprocessed signal is partitioned into 4 sequential observation slices (each ~5.36 seconds, 236,563 samples):

```
Window 1: fc_peak=2133.0 Hz, OBW=319.0 Hz, SNR=11.5 dB, env_var=0.188
Window 2: fc_peak=2131.0 Hz, OBW=322.0 Hz, SNR=11.7 dB, env_var=0.191
Window 3: fc_peak=2132.0 Hz, OBW=325.0 Hz, SNR=11.8 dB, env_var=0.192
Window 4: fc_peak=2130.0 Hz, OBW=321.0 Hz, SNR=11.6 dB, env_var=0.189

Sample Std (s, ddof=1):  fc_peak std=1.25 Hz, OBW std=3.10 Hz, SNR std=0.35 dB
Channel-Scaled Rel Var:  fc rel_var = 1.25 / 323.0 = 0.0039 -> Rational score: 0.982 (HIGH)
                         OBW rel_var = 3.10 / 322.0 = 0.0096 -> Rational score: 0.971 (HIGH)
                         SNR rel_var = 0.35 / 10.0  = 0.0350 -> Rational score: 0.963 (HIGH)

Overall Temporal Stationarity Score: 0.942 (HIGH STABILITY)
```

### 22.6 Stage 9: Pulse Analysis

```
Pulse count: 0 (No sparse burst transients detected)
Duty cycle: ~100.0% (Continuous carrier transmission)
Verdict: Waveform is continuous; radar chirp / pulsed gates not triggered
```

### 22.7 Stage 10: Protocol Inference and Autonomous Classification

```
Rules 1-11: No match
Rule 12 (Dual-Peak Continuous 2-FSK):
  Peak 1 (Mark): 2110.25 Hz, Peak 2 (Space): 2304.05 Hz
  Frequency Shift: df = 193.80 Hz (within 170 Hz +- 30% receiver tolerance)
  Schmitt Trigger Dwell Analysis: Peak at 10.0 ms (Harmonic fit 99.8% -> 100.0 Baud)
  Carson's Rule: B_carson = df + Rs = 193.80 + 100.0 = 293.80 Hz ~= OBW 323.0 Hz
Result: Primary Winner = MARITIME_NAVTEX (base_evidence_score = 0.98)
```

### 22.8 Stage 11: Candidate Hypothesis Ranking

The ranker assembles the primary candidate alongside evaluated physical alternatives from the detector's rejection trace:

```
Rank 1: MARITIME_NAVTEX
        Generation: Primary Rule 12 winner (2-FSK, 100 Baud, 170 Hz nominal shift)
        Base Evidence: 0.98 | Contradictions: 0 | Net Evidence: 0.98
Rank 2: ASCII_TELEPRINTER
        Generation: Dual-tone FSK contender evaluated in Rule 12
        Shared Evidence: Carrier center 2207.15 Hz, passband OBW 323.0 Hz
        Base Evidence: 0.53 | Contradiction: "Dwell 10.0 ms != 9.09 ms nominal (110 Baud)"
        Penalty: 0.20 | Net Evidence: 0.33
Rank 3: RTTY_BAUDOT_45
        Generation: Dual-tone FSK contender evaluated in Rule 12
        Shared Evidence: Carrier center 2207.15 Hz, passband OBW 323.0 Hz
        Base Evidence: 0.50 | Contradiction: "Dwell 10.0 ms != 22.0 ms nominal (45.45 Baud)"
        Penalty: 0.20 | Net Evidence: 0.30
```

### 22.9 Stage 12: Contradiction Analysis

```
MARITIME_NAVTEX:
  Supporting Evidence:
    - "Dual PSD peaks confirmed at 2110.25 Hz and 2304.05 Hz"
    - "Frequency shift 193.8 Hz within 170 Hz +- 30% receiver tuning tolerance"
    - "Schmitt dwell histogram peak at 10.0 ms (100.0 Baud)"
    - "Carson bandwidth 293.8 Hz matches measured OBW 323.0 Hz"
    - "High temporal stationarity verified (score 0.942)"
  Contradictions: None (0 physical contradictions)
  Total Penalty: 0.00
  Net Evidence Score: 0.98
```

### 22.10 Stage 13: Adaptive Specialized Extraction

The engine dispatches to `fsk_detector`:

```
Mark Frequency:           2110.25 Hz
Space Frequency:          2304.05 Hz
Frequency Shift:          193.80 Hz
Modulation Index (h):     1.938 (h = df / Rs = 193.8 / 100.0)
Symbol Rate:              100.0 Baud
Symbol Dwell Duration:    10.0 ms
Carson Bandwidth:         293.80 Hz
```

### 22.11 Stage 14: Physical Reconstruction Engine

```
IQ Imbalance:             Amplitude error = 0.002 dB, Phase error = 0.08 deg
CFO Tracking:             Carrier offset tracked at 2207.15 Hz, PLL locked = True
Soft Demodulation:        Symbol transitions mapped; EVM = 8.4%
Framing & Sync:           Cross-correlation against SITOR-B (CCIR 476) 7-bit framing pattern
                          Sync-word correlation peak = 0.92 (High confidence sync lock)
                          CRC / Parity: SITOR-B 4:3 constant-ratio code parity matches confirmed
```

### 22.12 Stage 15: Evidence Validation Gate

With closed-loop reconstruction telemetry available, all 6 gates are evaluated:

```
G1 Spectral Consistency:       PASS (OBW 323.0 Hz bounded, ratio 1.07 within +-30%)
G2 Temporal Consistency:       PASS (Continuous carrier envelope, duty cycle 100.0%)
G3 Symbol / Dwell Consistency: PASS (Symbol rate CV = 0.003 < 0.20 tolerance)
G4 Cross-Window Stability:     PASS (Stationarity score 0.942 >= 0.600 threshold)
G5 Physical Plausibility:      PASS (SNR = 11.71 dB >= -10 dB, Baud = 100 < 2.5 * 323 Hz)
G6 Reconstruction Consistency: PASS (Demodulation EVM 8.4% <= 35.0%, SITOR-B sync locked)

Waived Gates: None (All 6 gates applicable and passed)
Contradictions: 0
Final Validation Verdict: VALIDATED (Epistemic Tier: VALIDATED)
```

### 22.13 Stage 16: Parameter Uncertainty Reporting

Parameter reports strictly distinguish estimator resolution bounds from empirical cross-window sample standard deviations ($ddof=1$):

```
carrier_frequency: 2207.15 Hz | OBSERVED  | HIGH | Uncertainty: 10.77 Hz (fft_bin_resolution bound: +- fs/4096)
symbol_rate:       100.0 Baud | ESTIMATED | HIGH | Uncertainty: 0.30 Baud (cross_window_std, range: [99.41, 100.59] Baud)
snr_db:            11.71 dB   | ESTIMATED | HIGH | Uncertainty: 0.35 dB   (cross_window_std, range: [11.01, 12.39] dB)
frequency_shift:   193.80 Hz  | OBSERVED  | HIGH | Uncertainty: 21.53 Hz  (fft_bin_resolution peak pair bound)
```

### 22.13 Ground Truth Verification

| Parameter | Engine Extracted | SigIDWiki Ground Truth | Status |
|---|---|---|---|
| Protocol | 2-FSK (NAVTEX/SITOR-B) | 2-FSK (SITOR-B / CCIR 476) | PASS (exact match) |
| Shift | 193.8 Hz audio passband | 170 Hz nominal | PASS (receiver tuning offset) |
| Symbol Rate | 100.0 Baud | 100 Baud | PASS (exact) |
| Bit Dwell | 10.0 ms | 10.0 ms | PASS (exact) |
| 99% OBW | 323.0 Hz | ~300 Hz | PASS |
| Epistemic Status | VALIDATED | - | PASS (all 6 gates passed) |

---

## 23. Benchmark Results & Behavioral Evaluation Battery

### 23.1 Real-World Intercept Benchmark (25 Verified SigIDWiki Captures)

The core validation suite evaluates 25 genuine intercepted radio recordings from the SigIDWiki reference archive, spanning HF, VHF, and UHF bands:

- **Pass Criterion:** Primary classified protocol and modulation family must match published ITU/CCIR ground truth, and extracted parameters (carrier, OBW, baud, shift, PRF) must fall within published physical tolerances.
- **Dataset Result:** **25 / 25 captures matched published ground truth (100% test set pass rate)**.
- **Scope Limitation:** While this establishes consistent performance across all cataloged maritime, military, aviation, and trunking protocols in the test repository, it does not imply 100% accuracy across arbitrary uncurated real-world spectral environments.

### 23.2 Round-2 Multi-Dimensional Behavioral Battery (44 Test Cases)

To stress-test the new Round-2 inference engine (validation gate, uncertainty reporting, and OOD handling), `scripts/round2_evaluation_benchmark.py` executes a **44-case internal behavioral battery** across 6 distinct operational dimensions:

| Category | Test Scope | Cases | Expected Behavioral Outcome | Test Set Result |
|---|---|:---:|---|:---:|
| **Cat-A: Known Protocols** | Real-world clean signals from catalog | 24 | Primary candidate correct & `VALIDATED` | 24/24 PASS |
| **Cat-B: Noise Floor Rejection** | Stationary Gaussian noise captures | 5 | Pre-gate rejects input (`NO SIGNAL / NOISE FLOOR`) | 5/5 PASS (0% false alarm on set) |
| **Cat-C: Low-SNR Stress** | Degraded captures (-5 dB to +5 dB SNR) | 5 | Graceful demotion to `ESTIMATED` or `AMBIGUOUS` | 5/5 PASS |
| **Cat-D: Out-of-Distribution** | Synthetic non-standard waveforms | 4 | Explicit open-set rejection (`UNKNOWN_OOD`) | 4/4 PASS |
| **Cat-E: Ambiguous Feature Overlap** | Co-channel / overlapping tone structures | 3 | Preserved as `AMBIGUOUS` with top candidates | 3/3 PASS |
| **Cat-F: Distorted Waveforms** | Severe clipping / non-linear saturation | 3 | Retained as `ESTIMATED` with documented degradation | 3/3 PASS |
| **Symbol Rate MAE** | Discrete standard baud estimators | 12 | Exact integer dwell bin match on standard bauds | 0.00 Baud MAE on tested bauds |

**Total Battery Outcome:** **44 / 44 expected behavioral outcomes verified.**  
*Evaluation Caveat:* This battery proves that the software's epistemic decision logic behaves strictly as designed under controlled impairments. It is an internal regression battery, not a statistical guarantee of real-world operational probability of detection ($P_d$) across unknown channels.

### 23.3 High-Duty-Cycle Radar Verification (HAARP Chunking Suite)

To ensure the intra-pulse FMOP detector remains scale-invariant across different recording chunk sizes, `tests/test_haarp_chunking.py` evaluates HAARP ionospheric sounder captures sliced into 6 segment lengths:
- **Evaluated Chunks:** 20k, 50k, 100k, 250k, 500k samples, and the full continuous capture.
- **Invariant Verified:** In all 6 chunk sizes, the signal satisfies $PW \ge 800\,\mu\text{s}$, $\text{duty} < 65\%$, and $R^2 \ge 0.40$, correctly verifying linear FMOP without false classification as TDMA audio ripple. All 6 chunk tests pass.

### 23.4 Automated Test Suite Execution

All unit and regression tests are automated and reproducible via the standard test runner:

```bash
python -m unittest discover tests          # 177 / 177 PASS (verified 2026-09-08)
python scripts/benchmark_13_signals.py     # 25 / 25 PASS (SigIDWiki benchmark)
python scripts/round2_evaluation_benchmark.py # 44 / 44 PASS (Behavioral battery)
python -m unittest tests/test_haarp_chunking.py # 6 / 6 PASS (Scale invariance)
python -m unittest tests/test_comprehensive_amc_matrix.py # 3 / 3 PASS (22 captures)
```

### 23.5 Blind-Extraction Verification

The blind layer has its own tests, separate from named-protocol accuracy:

| Verification | Evidence |
|---|---|
| Contract and serialization | `BlindParameterVector` round-trip and complete field coverage pass |
| Activity and segmentation | Noise-floor estimation separates active bursts from inactive regions |
| Frequency discovery | Carrier, centroid, median, energy center, component states, and multi-tone comb tests pass |
| Symbol-rate estimation | Continuous candidate generation and multi-method consensus tests pass |
| Cyclostationary extraction | Dominant cyclic-frequency recovery test passes |
| Generic modulation geometry | FSK states, BPSK/QPSK phase folds, and multi-carrier tests pass |
| Hard software boundary | `blind_parameter_engine.py` contains zero banned protocol names |
| Uncataloged signals | Custom 73.5 Baud / 340 Hz and 137.4 Baud / 523 Hz FSK cases pass without protocol presets |

These checks support the narrower, defensible claim that physical parameter extraction is upstream of protocol identification. They do not claim perfect parameter accuracy for every unknown RF environment; uncertainty and abstention remain part of the contract.

---

## 24. Robustness and Stress Testing Engine

**NEW in Round-2, Step 9.**

### 24.1 Architecture

`dsp/robustness_tester.py` provides six configurable impairment functions and a `SignalRobustnessTester` class for systematic performance characterization.

### 24.2 Six Impairment Dimensions

| Impairment | Function | Parameters |
|---|---|---|
| AWGN Noise | `apply_awgn(signal, snr_db)` | SNR in dB |
| Amplitude Scaling | `apply_amplitude_scaling(signal, scale_factor)` | Linear scale factor |
| Frequency Offset | `apply_frequency_offset(signal, fs, offset_hz)` | Offset in Hz |
| Frequency Drift | `apply_frequency_drift(signal, fs, drift_hz_per_sec)` | Linear drift rate |
| Clipping/Saturation | `apply_clipping(signal, percentile_threshold)` | Percentile clip point |
| Shortened Duration | `apply_shortened_duration(signal, fraction)` | Fraction of signal kept |

### 24.3 CLI Runner

```bash
python scripts/run_robustness_tests.py --signal verified_samples/NAVTEX.wav \
    --dimension awgn --snr_range -10 30 5 \
    --output verified_samples/robustness_sweep_results.json
```

### 24.4 Sample Output

```json
{
  "signal": "NAVTEX.wav",
  "dimension": "awgn",
  "sweep_results": [
    {"snr_db": -10, "classification": "UNKNOWN",        "validation_status": "UNKNOWN"},
    {"snr_db":   0, "classification": "MARITIME_NAVTEX","validation_status": "ESTIMATED"},
    {"snr_db":  10, "classification": "MARITIME_NAVTEX","validation_status": "VALIDATED"},
    {"snr_db":  20, "classification": "MARITIME_NAVTEX","validation_status": "VALIDATED"}
  ]
}
```

---

## 25. Known Limitations and Architecture Boundaries

### 25.1 Evidence Scores Are Uncalibrated Epistemic Metrics, Not Probabilities

Values such as 0.98 represent evidence scores computed from the count and physical margin of passing invariant checks minus contradiction penalties. They are **not calibrated Bayesian posterior probabilities**. The correct technical term is `evidence_score` or `net_evidence_score`. Every UI label, API field, and documentation entry strictly uses this designation.

### 25.2 Multi-Window Temporal Validation Bounds

The 4-window temporal partition requires each window to contain at least 256 samples (minimum 1,024 total samples at the given sampling rate). Captures below 1,024 samples receive `cross_window_consistency_score = null` with `"status": "INSUFFICIENT_OBSERVATION_DURATION"`. Downstream classification operates on whole-capture features with temporal stability waived.

### 25.3 Reconstruction Consistency Scope (NOT_APPLICABLE Policy)

Gate 6 (Reconstruction Consistency) evaluates carrier PLL tracking, demodulation EVM, FEC syndromes, and framing CRCs. For waveforms that do not possess digital framing structure (such as pure CW beacons, analog voice, or FMCW chirp radars), Gate 6 is marked `NOT_APPLICABLE` (`waived_gates`). It is never treated as a fabricated pass. Such signals achieve `VALIDATED` status strictly through unanimous passes across all remaining applicable physical and temporal gates (G1, G2, G4, G5) with zero unresolved physical contradictions.

### 25.4 QPSK vs. Multi-Level Digital Discrimination at Low SNR

At SNR below ~10 dB, higher-order cumulants ($C_{40}, C_{42}$) experience increased sample variance, potentially causing the boundary between QPSK and Generic Digital Multi-Level to blur. The open-set AMC correctly returns `UNKNOWN_OOD` rather than hazarding an ungrounded guess.

### 25.5 Strict Independence from Filenames

The `file_name` metadata attribute is utilized solely for UI display and telemetry logging. It is strictly isolated from all DSP extractors, candidate generation trees, and validation gates. An identical waveform file renamed to random characters yields identical classification and parameter extraction results.

### 25.6 PRF Search Range and Low-Duty-Cycle Radars

The envelope autocorrelation PRF extractor searches down to $\text{min\_prf} = 0.5\text{ Hz}$ (PRI up to 2.0 seconds), capturing very slow pulse trains such as Duga-3 (10 Hz). Pulse systems with PRF below 0.5 Hz require capture durations longer than 4.0 seconds to yield multiple periodic pulses.

### 25.7 Configurable Contradiction Penalties as Engineering Hyper-Parameters

The contradiction penalty weights in `ContradictionConfig` (spectral 0.20, temporal 0.15, parameter 0.20, cross-window 0.15, reconstruction 0.20, capped at 0.80) are explicit engineering hyper-parameters based on relative physical sensor fidelity. While independently verifiable through unit tests, they represent operational configurations rather than statistically fitted regression coefficients. Evaluators may tune these weights per tactical deployment requirements.

### 25.8 System Scope & Open-Set Contract

The Aarohan engine is a general-purpose blind RF/audio parameter extraction system designed for open-set operational environments. Signals outside the cataloged rule set are rejected as `UNKNOWN`, `UNKNOWN_OOD`, or `NO SIGNAL / NOISE FLOOR`. This rejection is the mathematically intended behavior, preventing dangerous false positives in tactical SIGINT operations.

---

## 26. Glossary of Terms

| Term | Full Meaning and Operational Definition |
|---|---|
| ADC | Analog-to-Digital Converter: Hardware that samples continuous analog antenna voltages at discrete time intervals. |
| AFSK | Audio Frequency Shift Keying: FSK carried within the acoustic audio passband (300-3000 Hz). |
| Aliasing | Distortion when frequencies above fs/2 fold back and masquerade as lower frequencies. |
| AMC | Automatic Modulation Classification: Autonomous algorithms identifying modulation scheme without prior knowledge. |
| Analytic Signal | Complex signal s = x + j*H{x} with spectral energy only on positive frequencies. |
| Baud Rate | Distinct symbol state changes per second (Rs = 1/Ts). |
| BPSK | Binary Phase Shift Keying: Binary data encoded by 0 vs. 180 degree phase shift. |
| Candidate Hypothesis | A potential signal classification produced by the autonomous detector before ranking and validation. |
| Carson's Rule | Empirical FSK bandwidth formula: B = df + Rs. |
| CODAR | Coastal Ocean Dynamics Application Radar: FMCW radar mapping ocean surface currents. |
| Complex Baseband | RF signal shifted to zero frequency and split into In-Phase (I) and Quadrature (Q) rails. |
| Contradiction | A measurable physical property inconsistent with a candidate hypothesis; reduces evidence score. |
| CW | Continuous Wave: Unmodulated single-frequency transmission. |
| DC Offset | Constant nonzero mean voltage from ADC bias or LO leakage; false spectral spike at 0 Hz. |
| DFT | Discrete Fourier Transform: Converts discrete time-domain samples to frequency-domain coefficients. |
| DMR | Digital Mobile Radio: ETSI two-slot TDMA standard in 12.5 kHz channels with 4-FSK. |
| Duty Cycle | Fraction of time a pulsed transmitter is radiating: Duty = (PW/PRI)*100%. |
| Epistemic Status | Classification of how a parameter was derived: OBSERVED / ESTIMATED / HYPOTHESIZED / VALIDATED / UNKNOWN. |
| Evidence Score | Scalar [0,1] from the count and margin of passing physical criteria. Not a calibrated probability. |
| EVM | Error Vector Magnitude: Distance between received and ideal constellation points, in %. |
| FFT | Fast Fourier Transform: O(N*log N) DFT algorithm. |
| FMCW | Frequency-Modulated Continuous Wave: Radar using continuously sweeping frequency ramp. |
| FMOP | Frequency Modulation On Pulse: Radar pulses containing internal linear frequency chirps. |
| Formant | Resonant frequency of the human vocal tract. |
| FSK | Frequency Shift Keying: Data transmitted by shifting carrier frequency among discrete tones. |
| GMSK | Gaussian Minimum Shift Keying: Continuous-phase FSK with Gaussian pulse smoothing (GSM). |
| GSM | Global System for Mobile Communications: 2G digital cellular, 8-slot TDMA per 200 kHz carrier. |
| Hann Window | Bell-shaped mathematical window tapering signal segment edges to zero, preventing spectral leakage. |
| Hilbert Transform | Linear operator shifting positive frequencies by -90 degrees, creating analytic signal. |
| HOC | Higher-Order Cumulants: Statistical moments (order >= 3) used as modulation fingerprints. |
| I/Q Channels | In-Phase (I) and Quadrature (Q) components of a complex signal, separated by 90 degrees. |
| Kurtosis | Statistical measure of tail heaviness relative to Gaussian distribution. |
| Levinson-Durbin | O(p^2) recursive algorithm for symmetric Toeplitz matrix equations; used in LPC. |
| LLR | Log-Likelihood Ratio: Soft bit decision metric; sign indicates bit value, magnitude indicates confidence. |
| LPC | Linear Predictive Coding: Models vocal tract resonance as an all-pole predictive filter. |
| M-FSK | Multiple Frequency Shift Keying: FSK with 4, 8, or 16 discrete tones. |
| MAD | Median Absolute Deviation: Robust noise estimator: MAD = median(|x[n] - median(x)|). |
| NAVTEX | Navigational Telex: International automated maritime safety information service. |
| Nyquist Rate | Minimum sampling rate to avoid aliasing: fs = 2 * f_max. |
| OBW | Occupied Bandwidth: Frequency span containing 99% of total integrated signal power. |
| OOD | Out-Of-Distribution: Signal not matching any modeled AMC class; returned as UNKNOWN_OOD. |
| OOK | On-Off Keying: Carrier switched on (1) or off (0); Morse code. |
| PAPR | Peak-to-Average Power Ratio: Maximum instantaneous power divided by average power, in dB. |
| Parameter Uncertainty | Per-parameter report of value, epistemic status, stability, and estimator resolution bounds or cross-window sample standard deviations, or explicit null with reason. |
| PRF | Pulse Repetition Frequency: Radar pulses per second (PRF = 1/PRI). |
| PRI | Pulse Repetition Interval: Time between start of consecutive radar pulses (PRI = 1/PRF). |
| PSD | Power Spectral Density: Distribution of signal power as function of frequency (dB/Hz). |
| PSK | Phase Shift Keying: Data encoded by switching carrier phase among discrete values. |
| PW | Pulse Width: Duration of a single radar pulse, in microseconds. |
| QAM | Quadrature Amplitude Modulation: Encodes information by varying both amplitude and phase. |
| QPSK | Quadrature Phase Shift Keying: Four-phase PSK (90 degree separation), 2 bits per symbol. |
| RF | Radio Frequency: Electromagnetic spectrum (3 kHz to 300 GHz) for wireless communications. |
| Sampling Rate | Frequency at which continuous waveform is converted to discrete samples, in Hz. |
| SDR | Software-Defined Radio: Radio system where hardware components are implemented in software. |
| SFM | Spectral Flatness Measure: Ratio of geometric mean to arithmetic mean of power spectrum. |
| SIGINT | Signals Intelligence: Intelligence gathered through interception and electronic analysis of radio signals. |
| SITOR-B | Simplex Telex Over Radio (Broadcast): Error-correcting maritime protocol (CCIR 476), 100 Baud 2-FSK. |
| SNR | Signal-to-Noise Ratio: Desired signal power to background noise power, in dB. |
| Stationarity Score | Scalar [0,1] from temporal validator measuring parameter consistency across 4 analysis windows. |
| STFT | Short-Time Fourier Transform: Windowed Fourier transform producing 2D time-frequency spectrogram. |
| Supporting Evidence | A measurable physical property of the signal consistent with a candidate hypothesis. |
| TDMA | Time Division Multiple Access: Multiple users share a frequency band by transmitting in synchronized time slices. |
| Temporal Validator | Round-2 module that splits signal into 4 windows and computes cross-window parameter stability. |
| UNKNOWN_OOD | Out-Of-Distribution Unknown: First-class AMC outcome indicating signal does not match any modeled class. |
| Validation Gate | Six-gate evidence validation engine requiring candidates to pass all applicable physical, temporal, and reconstruction checks before VALIDATED status. |
| Welch Method | Spectral density estimation averaging overlapping windowed periodograms to minimize noise variance. |

---

*Authored by the Signal Intelligence Engineering Team for NTRO Problem Statement SIH26147.*
*Autonomous RF Parameter Extraction Engine -- Architecture v4.1 with Blind Parameter Extraction and Round-2 Validation,*
*Hypothesis Ranking, Contradiction Analysis, Hypothesis Validation Gate, Parameter Uncertainty Reporting,*
*and Robustness Stress Engine.*
