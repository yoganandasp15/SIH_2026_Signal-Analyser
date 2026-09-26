# NTRO Signal Analyzer
## Autonomous RF Parameter Extraction Architecture

**Architecture v4 | End-to-End Signal Intelligence Pipeline**

> **How an unknown intercepted signal is transformed from raw samples into validated, explainable RF intelligence without requiring prior signal parameters.**

---

* **System Target:** Smart India Hackathon 2026 (Problem Statement SIH26147)
* **Agency:** National Technical Research Organisation (NTRO)
* **Document Objective:** Technical Architecture Walkthrough for Evaluators, Technical Observers, and SIGINT System Architects
* **Core Philosophy:** Measure first. Hypothesize second. Challenge the hypothesis. Validate before claiming certainty.

<div style="page-break-after: always;"></div>

---

# Page 1: What Problem Does the Architecture Solve?

## The Blind Intercept Challenge

In real-world signals intelligence (SIGINT) and electronic surveillance, radio receivers capture electromagnetic energy across crowded frequency bands. When an unknown transmission arrives at an intercept station, the receiving operator or automated collection pod possesses **zero prior knowledge** about the transmitter.

```text
               UNKNOWN INTERCEPTED SIGNAL
                           │
                           ▼
          ┌──────────────────────────────────┐
          │ "What is this transmission?"     │
          │ "What are its physical values?"  │
          │ "Can we trust this extraction?"  │
          └──────────────────────────────────┘
```

The incoming intercept file may arrive in several formats:
* An uncalibrated audio recording (`.wav`) captured from an analog receiver audio output.
* A raw high-speed digital In-Phase / Quadrature (`.iq` / `.raw` / `.bin`) stream from a Software Defined Radio (SDR).
* An arbitrary transmission with **unknown modulation** (continuous wave, frequency shift, phase shift, pulsed radar, spread spectrum).
* An arbitrary channel with **unknown center frequency**, **unknown occupied bandwidth**, and **unknown symbol rate**.

Traditional communications receivers require an operator to manually dial in the exact carrier frequency, channel filter, and baud rate before any data can be recovered. If the parameters are wrong, the receiver outputs noise.

## Traditional Machine Learning vs. Autonomous Physical Evidence Pipeline

Most automated attempts rely on direct deep neural network classifiers. When fed an unknown signal, a black-box model produces a statistical label without verifying whether the underlying physics holds true.

```text
Traditional Black-Box Approach

Unknown Signal ───► [ Deep Neural Network ] ───► "Probably NAVTEX (99% confidence)"
                          │
                          └── Cannot explain why.
                          └── Hallucinates on unfamiliar noise.
                          └── Provides no physical parameter extraction.


Our Physical Evidence Architecture

Unknown Signal
      │
      ▼
[ Layer 1: Signal Conditioning ]       (DC removal, power normalization)
      │
      ▼
[ Layer 2: Blind Measurement ]         (Measures physical values without protocol names)
      │
      ▼
[ Layer 3: Competing Hypotheses ]      (Generates multiple plausible explanations)
      │
      ▼
[ Layer 4: Contradiction Analysis ]    (Actively searches for physical mismatches)
      │
      ▼
[ Layer 5: Six Validation Gates ]      (Enforces mathematical proof and stability)
      │
      ▼
Validated Signal Intelligence Record   (Fully explainable, auditable, and bounded)
```

## The Fundamental System Mandate

> **The system must first measure what is physically present in the waveform, and only then use those empirical measurements to construct, test, and challenge competing hypotheses.**

Parameters are never assigned because a catalog says they should exist. They are extracted directly from the raw waveform data.

<div style="page-break-after: always;"></div>

---

# Page 2: The Complete Architecture at a Glance

The architecture organizes the processing pipeline into **14 sequential stages** grouped across **5 operational layers**. Every stage performs a distinct mathematical transformation, producing verifiable evidence before passing data downstream.

```text
                                RAW INTERCEPTED STREAM (.wav / .iq / .bin)
                                                   │
===================================================│===================================================
LAYER 1: SIGNAL PREPARATION                        ▼
  01. FILE INGESTION                 Detect file container, resolve sampling rate, format complex I/Q
  02. INPUT VALIDATION               Reject NaNs, infinities, empty arrays, and impossible sample rates
  03. SIGNAL CONDITIONING            Remove receiver DC bias, normalize signal power to unit variance
===================================================│===================================================
LAYER 2: BLIND OBSERVATION                         ▼
  04. SPECTRAL & BLIND EXTRACTION    Extract carrier frequency, occupied bandwidth, and noise floor
  05. TEMPORAL VALIDATION            Split into 4 windows to verify parameter stability over time
  06. PULSE & MORPHOLOGY ANALYSIS    Detect pulsed bursts, measure pulse repetition interval and duty
===================================================│===================================================
LAYER 3: INTELLIGENCE & HYPOTHESIS                 ▼
  07. AUTONOMOUS CLASSIFICATION      Evaluate physical invariants; reject pure noise and silence
  08. CANDIDATE RANKING              Build pool of competing explanations with raw evidence scores
  09. CONTRADICTION ANALYSIS         Penalize hypotheses that contradict measured physical parameters
  10. VALIDATION GATES               Pass candidates through 6 independent mathematical and physical tests
===================================================│===================================================
LAYER 4: DEEP EXTRACTION                           ▼
  11. ADAPTIVE EXTRACTION            Dispatch specialized extractors (FSK, Radar, TDMA, PSK, Audio)
  12. PHYSICAL RECONSTRUCTION        Attempt carrier lock, symbol timing sync, soft demod, and CRC
===================================================│===================================================
LAYER 5: TRUST & OUTPUT                            ▼
  13. PARAMETER UNCERTAINTY          Quantify estimator resolution bounds and cross-window variance
  14. INTELLIGENCE OUTPUT            Deliver auditable tactical record with epistemic trust labels
=======================================================================================================
                                                   │
                                                   ▼
                               VALIDATED TACTICAL SIGNAL INTELLIGENCE
```

### Concise Stage Reference

* **01 Ingestion:** "What kind of recording has arrived?"
* **02 Validation:** "Is this numerical data safe and physically meaningful?"
* **03 Conditioning:** "Clean the signal of hardware artifacts."
* **04 Spectral Analysis:** "What frequencies and bandwidths exist in the energy spectrum?"
* **05 Temporal Validation:** "Are these measurements stable over time, or just transient bursts?"
* **06 Pulse Analysis:** "Is this a continuous transmission or a pulsing radar/TDMA signal?"
* **07 Classification:** "What physical modulation invariants are active?"
* **08 Candidate Ranking:** "What are all the plausible candidate explanations?"
* **09 Contradiction Analysis:** "What empirical evidence says each candidate might be wrong?"
* **10 Validation Gate:** "Has the leading hypothesis survived six independent physical tests?"
* **11 Adaptive Extraction:** "Run deep parameter extractors specific to the confirmed modulation."
* **12 Reconstruction:** "Can the receiver lock onto the carrier, track clock, and recover frames?"
* **13 Uncertainty:** "What is the physical error bound on every extracted parameter?"
* **14 Intelligence Output:** "Package the verified telemetry into an auditable intelligence record."

<div style="page-break-after: always;"></div>

---

# Page 3: The Core Intelligence Principle

Understanding this architecture requires understanding one foundational operational concept:

```text
      ┌─────────────┐       ┌──────────────┐       ┌─────────────┐       ┌────────────┐
      │   MEASURE   │  ──►  │ HYPOTHESIZE  │  ──►  │  CHALLENGE  │  ──►  │  VALIDATE  │
      └─────────────┘       └──────────────┘       └─────────────┘       └────────────┘
```

The system never jumps directly from an input to an identification label. It strictly executes four epistemic steps:

---

### 1. Measurement (What is physically present)

The system observes physical properties directly from the waveform data using deterministic mathematical transforms. These measurements are protocol-agnostic.

* **Carrier Center Frequency ($f_c$):** The center of energy in the passband.
* **Occupied Bandwidth (99% OBW):** The span containing 99% of total integrated power.
* **Signal-to-Noise Ratio (SNR):** The ratio of signal power to background noise power.
* **Frequency Separation ($\Delta f$):** The physical spacing between spectral peaks.
* **Clock Dwell Duration ($T_s$):** The time the carrier remains in a single modulation state.

---

### 2. Hypothesis (What could explain these measurements)

A hypothesis is a candidate interpretation capable of explaining the observed physical measurements. The engine generates multiple competing hypotheses simultaneously:

```text
Observed Measurements:
  * Two spectral peaks detected at 2110.25 Hz and 2304.05 Hz
  * Peak separation: 193.80 Hz
  * Constant envelope (zero amplitude variations)
  * Clock dwell peak: 10.0 ms (100.0 Baud)

Generated Competing Hypotheses:
  [Candidate 1]: Maritime NAVTEX Utility Transmission
  [Candidate 2]: Industrial ASCII Teleprinter Link
  [Candidate 3]: Amateur Radio RTTY (Baudot 45.45)
```

---

### 3. Evidence (What supports the candidate)

The system cross-checks empirical measurements against known communication physics:

* $\checkmark$ Dual frequency peaks match 2-FSK binary keying.
* $\checkmark$ Dwell histogram peaks at 10.0 ms, indicating a steady 100 Baud clock.
* $\checkmark$ Carson's bandwidth theorem matches the measured 99% occupied bandwidth:
  $$B_{\text{Carson}} = \Delta f + R_s = 193.8\text{ Hz} + 100.0\text{ Hz} = 293.8\text{ Hz} \approx 323.0\text{ Hz}$$

---

### 4. Contradiction (What refutes the candidate)

Rather than merely accumulating positive evidence, the system actively searches for contradictions:

* $\times$ **For ASCII Teleprinter:** Nominal baud rate is 110 Baud ($T_s = 9.09\text{ ms}$). Measured dwell of 10.0 ms contradicts the hypothesis. Penalty applied: $-0.20$.
* $\times$ **For RTTY Baudot:** Nominal baud rate is 45.45 Baud ($T_s = 22.0\text{ ms}$). Measured dwell of 10.0 ms contradicts the hypothesis. Penalty applied: $-0.20$.
* $\checkmark$ **For Maritime NAVTEX:** Nominal baud rate is exactly 100.0 Baud ($T_s = 10.0\text{ ms}$). Zero contradictions found.

The candidate surviving all physical challenges with zero unresolved contradictions is promoted to the validation gate.

<div style="page-break-after: always;"></div>

---

# Page 4: Layer 1 — Make the Signal Usable (Stages 1–3)

Before any digital signal processing (DSP) or feature extraction can take place, the raw intercepted recording must be safely loaded, validated against corruption, and conditioned into a standardized mathematical form.

```text
RAW CAPTURE FILE (.wav / .iq / .bin)
                 │
                 ▼
┌──────────────────────────────────┐
│ STAGE 1: FILE INGESTION          │ ──► Identifies format, sampling rate, channels
└──────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│ STAGE 2: INPUT VALIDATION        │ ──► Rejects corrupt samples, NaNs, infinities
└──────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│ STAGE 3: SIGNAL CONDITIONING     │ ──► Removes DC hardware bias, normalizes power
└──────────────────────────────────┘
                 │
                 ▼
STANDARDIZED ANALYTIC BASEBAND SIGNAL: y[n]
```

---

### Stage 1: File Ingestion (`dsp/loaders.py`)

* **What It Does:** Ingests raw intercept files across heterogeneous formats (standard RIFF `.wav`, headerless binary `.iq`, raw interleaved 16-bit integers, or 32-bit floating point).
* **Why It Exists:** Intercept hardware varies widely. Surveillance pods record complex I/Q data, while auxiliary receiver ports output demodulated baseband audio.
* **What Goes In:** Raw file path on disk.
* **What Comes Out:** Complex baseband array $s[n]$, sampling frequency $f_s$, and domain metadata tag.

> **Key Concept — Hilbert Transform & Analytic Signal:**
> Real-world audio recordings contain only real numbers $x[n]$. To measure instantaneous phase and frequency without negative-frequency mathematical ambiguities, the engine computes the **Hilbert Transform** $\mathcal{H}\{x[n]\}$. This creates a complex analytic signal:
> $$s[n] = x[n] + j \cdot \mathcal{H}\{x[n]\}$$
> For stereo recordings, the engine measures channel correlation $\rho$. If $\rho > 0.75$, the file is treated as duplicate mono audio; if $\rho \le 0.75$, the channels are treated as physical orthogonal In-Phase ($I$) and Quadrature ($Q$) rails.

---

### Stage 2: Input Validation (`dsp/preprocessor.py`)

* **What It Does:** Enforces strict numeric entry preconditions before passing data to vector processors.
* **Why It Exists:** Corrupted files, hardware glitches, or zero-byte recordings cause divide-by-zero crashes or NaN propagation in downstream filters.
* **What Goes In:** Complex signal array $s[n]$ and reported sampling rate $f_s$.
* **What Comes Out:** Verified pass token or immediate structured failure notification (`FAILED: input_validation`).

```text
Validation Checklist:
  [x] Sampling rate valid?        (fs > 0 Hz and fs <= 100 GHz)
  [x] Buffer dimension valid?     (1D vector of length N >= 64 samples)
  [x] Contains Not-a-Number?      (Rejects if any NaN present)
  [x] Contains Infinities?        (Rejects if any +/- Inf present)
  [x] Zero signal power?          (Rejects if RMS amplitude == 0.0)
```

---

### Stage 3: Signal Conditioning (`dsp/preprocessor.py`)

* **What It Does:** Eliminates receiver hardware artifacts and normalizes energy.
* **Why It Exists:** Radio hardware mixers produce local-oscillator (LO) leakage that appears as a false DC spike at 0 Hz. Varied transmitter distances also cause arbitrary input volumes.
* **What Goes In:** Validated signal array $s[n]$.
* **What Comes Out:** Zero-mean, unit-power normalized complex signal $y[n]$ where $E[|y[n]|^2] = 1.0$.

1. **DC Bias Subtraction:** Removes false carrier spikes at 0 Hz:
   $$x_{\text{clean}}[n] = s[n] - \frac{1}{N}\sum_{k=0}^{N-1} s[k]$$
2. **Scale-Invariant Power Normalization:** Sets average signal power to $1.0$ (0 dBFS normalized):
   $$y[n] = \frac{x_{\text{clean}}[n]}{\sqrt{\frac{1}{N}\sum_{k=0}^{N-1} |x_{\text{clean}}[k]|^2 + 10^{-12}}}$$

* **NAVTEX Running Trace:** Ingested 946,254 samples at $f_s = 44,100\text{ Hz}$ (21.46 s duration). Tagged as Demodulated Audio. DC offset of $0.0001$ subtracted; unit power scaled by $2.822\times$.

<div style="page-break-after: always;"></div>

---

# Page 5: Stage 4 — Blind Parameter Extraction

The core requirement of autonomous signal analysis is **blindness**: the system must extract physical transmission parameters directly from the observed waveform **without** knowing the protocol, standard, or transmitter beforehand.

```text
                                UNKNOWN SIGNAL y[n]
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 ▼                       ▼                       ▼
          CARRIER & PEAKS            BANDWIDTH               SNR & NOISE
                 │                       │                       │
                 ▼                       ▼                       ▼
          Carrier Center fc          99% OBW               SNR Estimate
           (Midpoint & Peak)      (Power Integral)           (M2M4 Method)
                 │
                 ├──────────────────────► FSK Peak Separation (Δf)
                 │
                 ├──────────────────────► Instantaneous Frequency Derivation
                 │
                 └──────────────────────► Clock Dwell Time & Symbol Rate (Rs)
```

### Parameter Definitions

* **Carrier Center Frequency ($f_c$):** The frequency around which the transmission is centered. For symmetric multi-tone signals, the engine calculates the exact arithmetic midpoint between detected outer tones:
  $$f_{c,\text{mid}} = \frac{f_{\text{mark}} + f_{\text{space}}}{2}$$
* **Occupied Bandwidth (99% OBW):** The frequency span containing 99% of total integrated power, bounded between the 0.5% and 99.5% cumulative energy percentiles.
* **Signal-to-Noise Ratio (SNR):** The ratio of true signal power to noise power, calculated via the fourth-order moment ratio ($M_2 M_4$) method:
  $$\text{SNR} = 10 \log_{10}\left(\frac{\sqrt{2 M_2^2 - M_4}}{M_2 - \sqrt{2 M_2^2 - M_4}}\right)$$
* **Frequency Shift ($\Delta f$):** The physical spacing between the mark and space frequency peaks in a multi-frequency transmission.
* **Symbol Dwell Time ($T_s$) & Symbol Rate ($R_s$):** The time duration of a single transmitted digital state. The symbol rate is its inverse: $R_s = 1 / T_s$ (measured in Baud).

```text
                       BLIND OBSERVATION OCCURS FIRST
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │ fc (Midpoint) = 2207.15 Hz   │
                      │ 99% OBW       = 323.00 Hz    │
                      │ SNR           = 11.71 dB     │
                      │ Shift (Δf)    = 193.80 Hz    │
                      │ Dwell (Ts)    = 10.00 ms     │
                      │ Symbol Rate   = 100.00 Baud  │
                      └──────────────────────────────┘
                                     │
                         ONLY AFTER MEASUREMENT
                                     │
                                     ▼
                        PROTOCOL INTERPRETATION
               Candidate: Maritime NAVTEX (CCIR 476 standard)
```

* **NAVTEX Running Trace:** Extracted Mark tone at $2110.25\text{ Hz}$, Space tone at $2304.05\text{ Hz}$, center midpoint at $2207.15\text{ Hz}$, shift of $193.80\text{ Hz}$, and dwell time of $10.0\text{ ms}$ ($100.0\text{ Baud}$).

<div style="page-break-after: always;"></div>

---

# Page 6: How the System Sees a Signal (DSP Fundamentals)

The analyzer converts raw time-series voltage samples into frequency, phase, and statistical representations using four classic DSP tools.

```text
1. TIME DOMAIN                        2. FREQUENCY DOMAIN (PSD)
   Signal Voltage over Time              Power Distribution across Frequencies
   y[n]                                  P(f)
    1.0 ┤  /\    /\                       ▲       Mark       Space
    0.0 ┼─/──\──/──\──                     │       peak       peak
   -1.0 ┤     \/    \/                     │        /\         /\
        └───────────────► Time             └───────/──\───────/──\──────► Freq
                                                 2110 Hz     2304 Hz

3. TIME-FREQUENCY (SPECTOGRAM)        4. PHASE PLANE (CONSTELLATION)
   Energy Evolution over Time            In-Phase vs. Quadrature Polar Plot
   Freq ▲                                    Q ▲
        │ ════  ════  ════ Space (2304 Hz)     │       *
        │   ════  ════     Mark  (2110 Hz)     ┼──*─────────*──► I
        └─────────────────► Time               │       *
```

---

### 1. The Fast Fourier Transform (FFT)
* **Plain English:** A mathematical algorithm that breaks down a complex waveform into its constituent sine waves.
* **Analogy:** Like a prism splitting sunlight into individual colors of the rainbow, the FFT splits a complex radio signal into individual frequencies.

### 2. Power Spectral Density (PSD) via Welch's Method
* **Plain English:** Measures how much electromagnetic power is present at each specific frequency bin.
* **Why Welch's Method?** A single raw FFT produces noisy, jagged peaks. Welch's method splits the signal into overlapping segments, applies a Hann window to prevent edge leakage, computes individual periodograms, and averages them together. This smooths out noise spikes and reveals true carriers.

### 3. Short-Time Fourier Transform (STFT / Spectrogram)
* **Plain English:** Computes spectra across sliding time windows to show how frequencies shift over time.
* **Diagnostic Power:** Continuous wave displays a flat line; FSK displays alternating horizontal tone steps; radar chirps display diagonal ramps; TDMA bursts display periodic on/off energy blocks.

### 4. Analytic Phase & Complex Constellation
* **Plain English:** Plots the instantaneous In-Phase ($I$) and Quadrature ($Q$) components on a 2D Cartesian plane.
* **Diagnostic Power:** Pure carriers form an origin point; constant-amplitude signals (FSK/FM) form an open circle; phase-shift keyed signals (BPSK/QPSK) collapse into discrete clusters of points.

<div style="page-break-after: always;"></div>

---

# Page 7: Stage 5 — Multi-Window Temporal Validation

A major vulnerability in automated signal processing is relying on a single snapshot of data. A momentary burst of interference, motor noise, or lightning static can produce spectral peaks that masquerade as a radio transmission. 

The architecture introduces **Multi-Window Temporal Validation** (`dsp/temporal_validator.py`).

```text
FULL INTERCEPT RECORDING (946,254 samples / 21.46 seconds)
│◄─────────────────────────────────────────────────────────────────────────────►│
┌───────────────────┬───────────────────┬───────────────────┬───────────────────┐
│ Window 1 (25%)    │ Window 2 (25%)    │ Window 3 (25%)    │ Window 4 (25%)    │
│ Samples 0-236k    │ Samples 236k-473k │ Samples 473k-709k │ Samples 709k-946k │
└───────────────────┴───────────────────┴───────────────────┴───────────────────┘
          │                   │                   │                   │
          ▼                   ▼                   ▼                   ▼
    Extract Params      Extract Params      Extract Params      Extract Params
      fc, OBW, SNR        fc, OBW, SNR        fc, OBW, SNR        fc, OBW, SNR
          │                   │                   │                   │
          └───────────────────┼───────────────────┘                   │
                              ▼                                       ▼
                 CROSS-WINDOW STABILITY MATRIX ◄──────────────────────┘
                 Sample Mean (μ), Standard Deviation (s), Rational Decay Score (S)
```

### The Rational Decay Scoring Function

To prevent arbitrary heuristic cliffs (e.g. declaring a signal "invalid" if variation crosses an arbitrary threshold), cross-window variation is normalized against physical channel references and evaluated with a **continuous rational decay function**:

$$S_P = \frac{1}{1 + \left(\frac{\text{rel\_var}_P}{\tau_P}\right)^2}$$

Where $\text{rel\_var}_P = s_P / \text{Ref}_P$ is the parameter's standard deviation scaled by its channel reference, and $\tau_P$ is the physical tolerance constant (e.g. $\tau = 0.15$ for carrier frequency).

### Stable Signal vs. Transient Interference Comparison

```text
GENUINE SIGNAL (Stationary NAVTEX)          TRANSIENT BURST / INTERFERENCE
Window 1: fc = 2133.0 Hz, Baud = 100.0      Window 1: fc = 1420.0 Hz, Baud =  45.0
Window 2: fc = 2131.0 Hz, Baud = 100.0      Window 2: fc = 2850.0 Hz, Baud = 120.0
Window 3: fc = 2132.0 Hz, Baud = 100.0      Window 3: fc =   890.0 Hz, Baud =  75.0
Window 4: fc = 2130.0 Hz, Baud = 100.0      Window 4: fc = 2110.0 Hz, Baud = 110.0
──────────────────────────────────────      ──────────────────────────────────────
Std Dev:  s(fc) = 1.25 Hz                   Std Dev:  s(fc) = 835.4 Hz
Score:    S = 0.982 (HIGH STABILITY)        Score:    S = 0.114 (UNSTABLE / REJECT)
```

* **NAVTEX Running Trace:** Carrier variation across 4 windows was $s = 1.25\text{ Hz}$ ($\text{rel\_var} = 0.0039$, score $= 0.982$). Occupied bandwidth variation was $s = 3.10\text{ Hz}$ ($\text{rel\_var} = 0.0096$, score $= 0.971$). Overall temporal stationarity score: **$0.942$ (HIGH STABILITY)**.

<div style="page-break-after: always;"></div>

---

# Page 8: Stages 6–7 — From Measurements to Hypotheses

Once blind physical measurements are confirmed stable, the system moves from observation to hypothesis generation.

```text
CLEAN, STABILIZED SIGNAL
           │
           ▼
┌──────────────────────────────────────┐
│ PRE-GATE 0: SILENCE CHECK            │ ──► Average power < -120 dBFS? ──► Terminate (NOISE)
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ PRE-GATE N: GAUSSIAN NOISE CHECK     │ ──► Spectrum flat & cumulants ≈ 0? ──► Terminate (NOISE)
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ 19-RULE PHYSICAL INVARIANT TREE      │ ──► Evaluates deterministic physical criteria
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ CANDIDATE POOL GENERATION            │ ──► Produces ranked list of candidate explanations
└──────────────────────────────────────┘
```

---

### The Strict Noise Floor Pre-Gates

Before consulting any signal catalog, the system enforces two mandatory rejection gates:
1. **Silence Pre-Gate:** If total normalized power is below $10^{-12}$ ($-120\text{ dBFS}$), processing stops immediately with outcome `UNKNOWN / SILENCE`.
2. **Gaussian Noise Pre-Gate:** Pure thermal noise has a uniform flat spectrum (spectral flatness measure $\text{SFM} \ge 0.70$), no sharp peaks ($\text{prominence} < 8.0$), and near-zero higher-order cumulants ($|C_{40}| < 0.25$, $|C_{42}| < 0.25$). If all nine noise invariants pass simultaneously, processing halts with outcome `UNKNOWN / NOISE FLOOR`. No modulation is guessed.

---

### Physical Invariant Decision Rules

The engine tests deterministic physical equations derived from ITU-R, ETSI, and military standards:

* **Rule 1 (Pure CW):** Envelope variance $< 0.12$ and occupied bandwidth $< 150\text{ Hz}$.
* **Rule 2 (Morse OOK):** Dynamic range $> 0.65$ with $1:3$ dot-to-dash duration ratio.
* **Rule 3–6 (Radar Systems):** Repetitive pulse envelopes, periodic PRF, or linear frequency chirps.
* **Rule 7–8 (Cellular & Mobile TDMA):** Sharp periodic time-slot lines (GSM $216.7\text{ Hz}$, DMR $33.3\text{ Hz}$).
* **Rule 12 (Multi-Tone FSK Utility):** Dual PSD peaks with constant envelope and harmonic dwell distribution.

---

### Crucial Principle: Evidence Score $\neq$ Probability

The candidate ranker outputs an **evidence score**, not a statistical probability:

$$\text{Evidence Score} = \text{Count and margin of satisfied physical invariants}$$

> **Important Invariant:** In an open SIGINT environment where the true total number of global transmitters is unknown, computing a Bayesian posterior probability $P(\text{class}|\text{signal})$ is mathematically impossible without making false assumptions. The system reports **epistemic evidence**, never misleading percentage probabilities.

* **NAVTEX Running Trace:** Passed silence and noise gates. Matched Rule 12 criteria (dual peaks, constant envelope, stable dwell). Primary candidate generated: `MARITIME_NAVTEX` with base evidence score $= 0.98$.

<div style="page-break-after: always;"></div>

---

# Page 9: Stage 8 — Contradiction Analysis

Most classifiers search only for evidence that confirms their preferred choice. The NTRO Signal Analyzer actively searches for **contradictions** (`dsp/contradiction_analyzer.py`).

```text
                              CANDIDATE HYPOTHESIS
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
        SUPPORTING EVIDENCE                           CONTRADICTING EVIDENCE
  Measurable criteria that agree                Measurable criteria that conflict
  with physical observations                    with physical observations
                 │                                             │
                 ▼                                             ▼
        BASE EVIDENCE SCORE                        EXPLICIT PENALTY DEDUCTION
                 │                                             │
                 └──────────────────────┬──────────────────────┘
                                        ▼
                               NET EVIDENCE SCORE
                 Score = max(0.0, Base Score - Contradiction Penalties)
```

### The Five Measurable Contradiction Categories

Every contradiction must cite a specific physical measurement, an expected standard boundary, and the observed value:

| Category | Physical Condition Checked | Penalty Weight |
|---|---|:---:|
| **SPECTRAL** | Missing expected tones, invalid tone count, or bandwidth mismatch | $0.20$ |
| **TEMPORAL** | Envelope duty-cycle mismatch (e.g. constant carrier vs. pulsed radar) | $0.15$ |
| **PARAMETER** | Carson bandwidth violation, baud rate mismatch, or dwell deviation | $0.20$ |
| **CROSS_WINDOW** | Multi-window parameter variation exceeds stability limits | $0.15$ |
| **RECONSTRUCTION** | Carrier PLL loss of lock, or demodulation EVM $> 35\%$ | $0.20$ |

### Competing Candidate Contradiction Audit

```text
CANDIDATE 1: MARITIME_NAVTEX (CCIR 476)
  Supporting Evidence:
    ✓ Dual spectral peaks (2110.25 Hz, 2304.05 Hz)
    ✓ Shift: 193.8 Hz matches nominal 170 Hz (+-30% receiver tolerance)
    ✓ Clock dwell: 10.0 ms matches nominal 100.0 Baud
    ✓ Carson bandwidth: 293.8 Hz consistent with measured 323.0 Hz
  Contradictions Found: 0
  Net Evidence Score:   0.98 - 0.00 = 0.98  ──► RETAINED AS LEADING CANDIDATE

CANDIDATE 2: ASCII_TELEPRINTER (ITA-5)
  Supporting Evidence:
    ✓ Dual spectral peaks in audio passband
    ✓ Frequency shift within allowable utility spacing
  Contradictions Found: 1
    ✗ PARAMETER: Dwell time 10.0 ms does not match nominal 9.09 ms (110 Baud)
  Penalty Deducted:     -0.20
  Net Evidence Score:   0.53 - 0.20 = 0.33  ──► DEMOTED

CANDIDATE 3: RTTY_BAUDOT_45 (ITA-2)
  Supporting Evidence:
    ✓ Dual spectral peaks in audio passband
  Contradictions Found: 1
    ✗ PARAMETER: Dwell time 10.0 ms does not match nominal 22.0 ms (45.45 Baud)
  Penalty Deducted:     -0.20
  Net Evidence Score:   0.50 - 0.20 = 0.30  ──► DEMOTED
```

<div style="page-break-after: always;"></div>

---

# Page 10: Stage 9 — The Six Validation Gates

Before any candidate hypothesis is officially designated as `VALIDATED`, it must pass through **six independent physical and mathematical gates** (`dsp/validation_gate.py`).

```text
                            LEADING CANDIDATE HYPOTHESIS
                                         │
    ┌────────────────┬───────────────────┼───────────────────┬────────────────┐
    ▼                ▼                   ▼                   ▼                ▼
[ G1: Spectral ] [ G2: Temporal ] [ G3: Symbol ]      [ G4: Stability ] [ G5: Physical ]
Nyquist & BW      Pulse / Duty      Clock Dwell &       Cross-Window      SNR & Capacity
Boundaries        Continuity        Histogram Fit       Stationarity      Consistency
    │                │                   │                   │                │
    └────────────────┴───────────────────┼───────────────────┴────────────────┘
                                         ▼
                             [ G6: Reconstruction ]
                              Carrier Lock / EVM /
                              FEC / CRC Frame Proof
                                         │
                                         ▼
                            FINAL VALIDATION VERDICT
```

### The Six Gates Defined

| Gate | Validation Scope | Passing Physical Criterion |
|---|---|---|
| **G1: Spectral Consistency** | Passband energy & Nyquist bounds | $0 < \text{OBW} \le f_s$, $|f_c| \le f_s/2$, bandwidth variation $\le 25\%$ |
| **G2: Temporal Consistency** | Energy continuity or pulse count | Pulse count $\ge 1$ for radar; duty cycle $\ge 10\%$ for continuous |
| **G3: Symbol / Dwell Consistency** | Clock timing and state histogram | Baud rate variation $\le 20\%$; dwell matches detected state duration |
| **G4: Cross-Window Stability** | Stationarity across 4 time windows | Cross-window consistency score $S_{\text{temporal}} \ge 0.60$ |
| **G5: Physical Plausibility** | Transmission capacity limits | $\text{SNR} \ge -10\text{ dB}$; symbol rate $R_s \le 2.5 \times \text{OBW}$ |
| **G6: Reconstruction Consistency** | Demodulation and framing proofs | Demodulation $\text{EVM} \le 35\%$, PLL locked, or CRC frame match |

### Final Epistemic Verdicts

* **VALIDATED:** Passed all applicable gates with zero unresolved physical contradictions.
* **ESTIMATED:** Strong spectral evidence present, but minor temporal instability or missing digital framing prevented full validation.
* **AMBIGUOUS:** Two or more competing hypotheses achieved near-identical net evidence scores ($|\Delta \text{Score}| < 0.05$).
* **UNKNOWN:** Signal energy is present, but physical measurements fail plausibility gates.
* **UNKNOWN_OOD (Out-of-Distribution):** Waveform structure is confirmed present, but features deviate from all modeled standard classes.

* **NAVTEX Running Trace:** G1 passed (OBW $= 323\text{ Hz}$); G2 passed (continuous duty cycle); G3 passed ($10.0\text{ ms}$ dwell confirmed); G4 passed ($S_{\text{temporal}} = 0.982 \ge 0.60$); G5 passed ($\text{SNR} = 11.7\text{ dB}$); G6 passed (SITOR-B framing aligned). **Verdict: VALIDATED**.

<div style="page-break-after: always;"></div>

---

# Page 11: Stage 11 — Adaptive Parameter Extraction

Once the general signal family is determined, generic measurements are no longer sufficient. The system executes the **Strategy Design Pattern** (`dsp/adaptive_pipeline.py`) to dispatch specialized, mode-specific parameter extraction engines.

```text
                        VALIDATED SIGNAL FAMILY IDENTIFIED
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼
    FAMILY = FSK                    FAMILY = RADAR                  FAMILY = TDMA
         │                               │                               │
         ▼                               ▼                               ▼
┌──────────────────┐            ┌──────────────────┐            ┌──────────────────┐
│ FSK EXTRACTOR    │            │ RADAR EXTRACTOR  │            │ TDMA EXTRACTOR   │
│ Mark / Space fc  │            │ Range Resolution │            │ Frame Period     │
│ Frequency Shift  │            │ Max Unambig. R   │            │ Timeslot Width   │
│ Modulation Index │            │ Chirp Sweep Rate │            │ Active Duty %    │
│ Baud / Dwell     │            │ PRI / PRF Jitter │            │ Burst Baud Rate  │
└──────────────────┘            └──────────────────┘            └──────────────────┘
```

### Specialized Extraction Formulas

#### 1. Frequency Shift Keying (FSK) Engine
* **Modulation Index ($h$):** Ratio of frequency shift to symbol rate:
  $$h = \frac{\Delta f}{R_s} = \frac{|f_{\text{space}} - f_{\text{mark}}|}{R_s}$$
* **Carson Bandwidth Bound:** Theoretical minimum occupied channel width:
  $$B_{\text{Carson}} = \Delta f + R_s$$

#### 2. Pulsed Radar & Chirp Engine
* **Range Resolution ($\Delta R$):** The minimum distance between two separable targets:
  $$\Delta R = \frac{c}{2 \cdot B_{99\%}} \quad (c = 2.9979 \times 10^8\text{ m/s})$$
* **Maximum Unambiguous Range ($R_{\text{max}}$):** Distance radar wave travels before next pulse:
  $$R_{\text{max}} = \frac{c}{2 \cdot \text{PRF}}$$
* **Chirp Sweep Rate:** Frequency slope over time ($\text{MHz}/\mu\text{s}$) with linear regression goodness-of-fit ($R^2$).

#### 3. TDMA Cellular & Tactical Radio Engine
* **Frame Period ($T_{\text{frame}}$) & Timeslot Duration ($T_{\text{slot}}$):** Derived from cyclostationary envelope line harmonics (GSM: $4.615\text{ ms}$ frame, $576.9\ \mu\text{s}$ slot).
* **Gated Burst Baud Rate:** Extracts symbol timing strictly during active timeslot bursts.

* **NAVTEX Running Trace:** Dispatched to FSK Extractor. Derived mark frequency $= 2110.25\text{ Hz}$, space frequency $= 2304.05\text{ Hz}$, frequency shift $\Delta f = 193.80\text{ Hz}$, modulation index $h = 1.938$, and symbol rate $R_s = 100.0\text{ Baud}$.

<div style="page-break-after: always;"></div>

---

# Page 12: Stage 12 — Physical Reconstruction

Can the system recover the underlying transmitted data bits cleanly enough to verify the transmission?

The **Physical Reconstruction Engine** executes a closed-loop receiver chain on the conditioned signal array. This provides verifiable mathematical proof of the signal's identity.

```text
CONDITIONED SIGNAL y[n]
         │
         ▼
[ 1. IQ Conditioning ]          Corrects hardware amplitude/phase imbalances
         │
         ▼
[ 2. 20-D Feature Vector ]      Computes statistical cumulants, moments, and spectral form
         │
         ▼
[ 3. Carrier Synchronization ]  Estimates Carrier Frequency Offset (CFO) and locks PLL
         │
         ▼
[ 4. Timing Recovery ]          Aligns symbol clock at maximum eye opening
         │
         ▼
[ 5. Soft Demodulation ]        Computes Log-Likelihood Ratios (LLRs) for each bit
         │
         ▼
[ 6. Deinterleaving ]           Searches block and convolutional interleaver matrices
         │
         ▼
[ 7. FEC Decoding ]             Evaluates Forward Error Correction parity equations
         │
         ▼
[ 8. Framing & CRC Match ]      Correlates sync words and evaluates Cyclic Redundancy Check
         │
         ▼
CLOSED-LOOP MATHEMATICAL PROOF (CRC Match = True)
```

### The Seven Steps Explained

1. **IQ Conditioning:** Corrects gain and phase orthogonality mismatch between I and Q rails via Gram-Schmidt orthogonalization.
2. **Carrier Synchronization:** A Costas phase-locked loop (PLL) tracks residual carrier frequency offset (CFO) and phase jitter.
3. **Timing Recovery:** Gardner / early-late zero-crossing detectors recover the symbol clock without knowing transmitter clock phase.
4. **Soft Demodulation:** Computes soft Log-Likelihood Ratios (LLR), clipped to $[-20, +20]$ to prevent numerical overflow:
   $$\text{LLR}(b_k) = \ln\left(\frac{P(b_k = 1 | y)}{P(b_k = 0 | y)}\right)$$
5. **Deinterleaving:** Reverses deliberate bit-shuffling used by transmitters to disperse burst errors.
6. **FEC Decoding:** Checks whether the decoded bit stream satisfies algebraic parity-check matrix equations ($H \cdot c^T = 0$).
7. **CRC Validation:** Checks the Cyclic Redundancy Check checksum. A valid CRC match represents definitive mathematical verification.

<div style="page-break-after: always;"></div>

---

# Page 13: Where Does Machine Learning Fit?

A common question during technical evaluations is: **"Where is the machine learning in this architecture?"**

In the NTRO Signal Analyzer, **Machine Learning is an evidence source, never the sole decision-maker.**

```text
                                PHYSICAL WAVEFORM
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
               DETERMINISTIC DSP              STATISTICAL EXTRACTION
               Welch PSD, FFT, STFT,          Higher-Order Cumulants:
               Hilbert Transform, Envelopes   C20, C40, C42, Kurtosis, SFM
                         │                             │
                         └──────────────┬──────────────┘
                                        ▼
                           20-DIMENSIONAL FEATURE VECTOR
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │ OPEN-SET MODULATION         │
                         │ CLASSIFIER (AMC)            │
                         └─────────────────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                 KNOWN MODULATION                 OUT-OF-DISTRIBUTION
                 Centroid Distance d <= T         Centroid Distance d > T
                         │                             │
                         ▼                             ▼
               SECONDARY EVIDENCE FOR           UNKNOWN_OOD VERDICT
               VALIDATION GATE (Weight = 0.25)  (Rejects Forced Guess)
```

### The 20-Dimensional Physical Feature Vector

Instead of feeding raw, uncalibrated audio samples into an opaque neural network, the engine extracts a calibrated **20-dimensional physical feature vector**:

* **Dimensions 1–4 (Cumulants):** Second- and fourth-order cumulants ($C_{20}, C_{40}, C_{42}$, kurtosis) that separate constant-envelope signals from phase-keyed signals.
* **Dimensions 5–8 (Power & Envelope):** Normalized moments ($M_2, M_4$), envelope variance, and Peak-to-Average Power Ratio (PAPR).
* **Dimensions 9–12 (Spectral & Frequency):** Spectral Flatness Measure (SFM), occupied bandwidth, instantaneous frequency deviation, and linear chirp correlation ($R^2$).
* **Dimensions 13–16 (Non-Linear & Temporal):** Squaring-spectrum peak prominence, fourth-power line prominence, zero-crossing rate, and lag-1 autocorrelation.
* **Dimensions 17–20 (Phase & Cyclostationary):** Phase variance, constellation compactness, and cyclostationary $\alpha$-profile spectral correlation peaks.

### Open-Set Rejection (`UNKNOWN_OOD`)

Standard neural networks force inputs into the nearest trained category, causing noise or unfamiliar transmissions to be classified as military radars or voice signals with false 99% confidence.

The Open-Set AMC measures Euclidean distance to trained modulation centroids. If the minimum distance exceeds threshold $\mathcal{T}_{\text{OOD}}$, the system outputs `UNKNOWN_OOD`. The engine acknowledges when a signal lies outside its modeled universe.

<div style="page-break-after: always;"></div>

---

# Page 14: How the System Knows What It Knows (Epistemic Hierarchy)

In professional intelligence applications, **how** a number was derived is as important as the number itself.

The architecture enforces an **Epistemic Hierarchy Contract** (`dsp/contracts.py`). Every parameter carries an epistemic tag describing its scientific pedigree.

```text
        ▲
        │  VALIDATED     Passed closed-loop proof (CRC match, zero syndrome)
        │  ═════════
        │  OBSERVED      Directly measured from physical waveform (FFT peak, bandwidth)
   T    │  ═════════
   R    │  ESTIMATED     Derived via mathematical estimator (M2M4 SNR, clock dwell)
   U    │  ═════════
   S    │  HYPOTHESIZED  Model inference or catalog standard (candidate protocol)
   T    │  ═════════
        │  UNKNOWN       Insufficient evidence (refuses to guess)
        ▼
```

### The Five Epistemic Tiers Defined

1. **VALIDATED:** The parameter has been confirmed by closed-loop algebraic verification (e.g. CRC checksum passed, or all six validation gates passed).
2. **OBSERVED:** Directly measured from the physical data without behavioral models (e.g. highest peak in the Welch periodogram is at $2131.79\text{ Hz}$).
3. **ESTIMATED:** Derived using continuous mathematical estimation algorithms (e.g. fourth-order moment SNR estimation, instantaneous frequency dwell fitting).
4. **HYPOTHESIZED:** Inferred from a candidate model or standard (e.g. candidate protocol is NAVTEX based on physical alignment).
5. **UNKNOWN:** Parameter cannot be legitimately measured due to noise, short duration, or waveform type (e.g. constellation order for continuous FSK).

### The Invariant Regarding Catalog Values

> **Core Invariant:** A reference value from a protocol catalog (e.g. nominal NAVTEX shift is $170\text{ Hz}$) is **never** substituted for an empirical measurement. 
> 
> The system records what it observed ($193.80\text{ Hz}$), tags it as `ESTIMATED`, and notes that while the transmitter deviates by $+23.8\text{ Hz}$ from nominal, it falls within acceptable maritime receiver tolerance ($170\text{ Hz} \pm 30\%$). Nominal catalog values never rewrite physical reality.

<div style="page-break-after: always;"></div>

---

# Page 15: Parameter Uncertainty — A Number Is Not Enough

In engineering and signals intelligence, reporting a bare scalar number (e.g. `Carrier = 2207 Hz`) without error bounds is scientifically incomplete. 

The engine implements **Parameter Uncertainty Reporting** (`dsp/parameter_uncertainty.py`).

Every extracted parameter is delivered with four mandatory scientific fields:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ PARAMETER UNCERTAINTY RECORD                                           │
├────────────────────────────────────────────────────────────────────────┤
│ VALUE:       100.00 Baud                                               │
│ STATUS:      ESTIMATED                                                 │
│ STABILITY:   HIGH (Stationarity Score = 0.98)                          │
│ UNCERTAINTY: ±0.30 Baud [Range: 99.41 - 100.59 Baud]                   │
│ TYPE:        cross_window_std                                          │
│ REASON:      Sample standard deviation across 4 observation windows    │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Estimator Resolution Bounds vs. Cross-Window Standard Deviation

The system enforces strict mathematical rules regarding what uncertainty represents:

#### 1. Estimator Resolution Bound (`fft_bin_resolution`)
For a single-window spectral peak, the uncertainty is governed by the Fourier transform's frequency bin width:
$$\Delta f_{\text{bin}} = \frac{f_s}{N_{\text{FFT}}} = \frac{44,100\text{ Hz}}{4,096} = 10.77\text{ Hz}$$
The carrier peak is reported with an instrument resolution bound of $\pm 5.38\text{ Hz}$. 
The system **never** fabricates a Gaussian 95% confidence interval when observing an instrument quantization grid.

#### 2. Cross-Window Sample Standard Deviation (`cross_window_std`)
When a parameter is tracked across four temporal windows, the engine calculates sample standard deviation ($s$) with Bessel's correction ($ddof=1$):
$$s = \sqrt{\frac{1}{n-1}\sum_{w=1}^4 (P_w - \bar{P})^2}$$
The parameter uncertainty is reported as $\pm s$ alongside the empirical observed range $[\min(P), \max(P)]$.

#### 3. Explicit Null with Mandatory Explanation
When a parameter cannot be legitimately bounded (e.g. constellation order for an analog FM signal), uncertainty is explicitly set to `null` with a mandatory reason string:
```json
{
  "modulation_order": {
    "value": null,
    "status": "UNKNOWN",
    "uncertainty": null,
    "uncertainty_reason": "Continuous FSK modulation does not possess a discrete QAM constellation order"
  }
}
```

<div style="page-break-after: always;"></div>

---

# Page 16: Complete Walkthrough — Unknown Signal to NAVTEX

Here is how an intercepted recording travels through the 14 stages of the architecture, using empirical data from the test corpus (`verified_samples/navtex.wav`).

```text
STEP 1: INGESTION           Reads 946,254 samples (21.46 s) at fs = 44,100 Hz (Mono WAV)
        │
STEP 2: VALIDATION          Passes numeric checks (no NaNs, no Infs, valid length)
        │
STEP 3: CONDITIONING        Removes 0.0001 DC bias; scales power to unit variance E[|y|^2] = 1.0
        │
STEP 4: BLIND EXTRACTION    Measures physical spectrum without protocol hints:
                            * Mark tone peak:   2110.25 Hz
                            * Space tone peak:  2304.05 Hz
                            * Carrier center:   2207.15 Hz
                            * 99% Occupied BW:  323.00 Hz
                            * SNR (M2M4):       11.71 dB
                            * Shift (Δf):       193.80 Hz
                            * Clock dwell (Ts): 10.00 ms -> Symbol Rate = 100.00 Baud
        │
STEP 5: TEMPORAL SPLIT      Evaluates 4 sequential windows (each 5.36 s):
                            * fc variation:  s = 1.25 Hz (Stability Score = 0.982)
                            * OBW variation: s = 3.10 Hz (Stability Score = 0.971)
                            * SNR variation: s = 0.35 dB (Stability Score = 0.963)
                            * Overall stationarity score: 0.942 (HIGH STABILITY)
        │
STEP 6: PULSE CHECK         Pulse count = 0, active duty = 100% (Continuous carrier)
        │
STEP 7: CLASSIFICATION      Passes noise pre-gates; fires Physical Invariant Rule 12:
                            * Primary winner: MARITIME_NAVTEX (Base evidence score = 0.98)
        │
STEP 8: CANDIDATE RANKING   Assembles competing candidate pool:
                            * Rank 1: MARITIME_NAVTEX   (Net Score: 0.98)
                            * Rank 2: ASCII_TELEPRINTER (Base: 0.53, Penalty: -0.20 -> 0.33)
                            * Rank 3: RTTY_BAUDOT_45    (Base: 0.50, Penalty: -0.20 -> 0.30)
        │
STEP 9: CONTRADICTIONS      * NAVTEX: Zero physical contradictions
                            * ASCII:  Contradiction (10.0 ms dwell != 9.09 ms nominal)
                            * RTTY:   Contradiction (10.0 ms dwell != 22.0 ms nominal)
        │
STEP 10: VALIDATION GATES   Evaluates 6 independent gates on MARITIME_NAVTEX:
                            * G1 Spectral: PASS | G2 Temporal: PASS | G3 Symbol: PASS
                            * G4 Window:   PASS | G5 Physical: PASS | G6 Reconstruct: PASS
        │
STEP 11: ADAPTIVE PIPELINE  Dispatches FSK Extractor:
                            * Modulation index h = 1.938, Carson BW = 293.80 Hz
        │
STEP 12: RECONSTRUCTION     PLL locks; soft LLRs demodulate; SITOR-B frame pattern confirmed
        │
STEP 13: UNCERTAINTY        Quantifies error bounds:
                            * fc: 2207.15 ± 5.38 Hz | Rs: 100.00 ± 0.30 Baud | SNR: 11.71 ± 0.35 dB
        │
STEP 14: OUTPUT             Emits structured tactical intelligence record (VALIDATED)
```

<div style="page-break-after: always;"></div>

---

# Page 17: Final Intelligence Output

The operator does not receive an ambiguous script printout or an unverified label. The system produces a **structured, traceable intelligence record**.

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      TACTICAL RF SIGNAL INTELLIGENCE RECORD                            │
│                  National Technical Research Organisation (NTRO)                       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TARGET CLASSIFICATION:  MARITIME_NAVTEX (CCIR 476 / SITOR-B)                           │
│ EPISTEMIC STATUS:       VALIDATED                                                      │
│ MODULATION FAMILY:      2-FSK (Binary Frequency Shift Keying)                          │
│ CONFIDENCE LEVEL:       HIGH (Net Evidence Score: 0.98 / 1.00)                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ EXTRACTED PHYSICAL TRANSMISSION PARAMETERS                                             │
│                                                                                        │
│ Parameter               Value         Status     Stability   Uncertainty   Unit        │
│ ──────────────────────────────────────────────────────────────────────────────         │
│ Carrier Frequency (fc)  2207.15       OBSERVED   HIGH        ±5.38         Hz          │
│ Occupied Bandwidth(OBW)  323.00       OBSERVED   HIGH        ±3.10         Hz          │
│ Frequency Shift (Δf)     193.80       ESTIMATED  HIGH        ±2.15         Hz          │
│ Symbol Clock Rate (Rs)   100.00       ESTIMATED  HIGH        ±0.30         Baud        │
│ Signal-to-Noise (SNR)     11.71       ESTIMATED  HIGH        ±0.35         dB          │
│ Peak-to-Average (PAPR)     5.90       OBSERVED   HIGH        ±0.20         dB          │
│ Envelope Variance Ratio    0.19       OBSERVED   HIGH        ±0.015        ratio       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ MULTI-WINDOW STABILITY AUDIT (4 Slices @ 5.36s)                                        │
│ Stationarity Score: 0.942 / 1.000 (HIGH) | Unstable Parameters: 0                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ SIX-GATE VALIDATION AUDIT TRACE                                                        │
│ [PASS] G1 Spectral:      Passband energy bounded, OBW 323 Hz within Nyquist            │
│ [PASS] G2 Temporal:      Continuous carrier envelope confirmed (duty cycle ~100%)       │
│ [PASS] G3 Symbol Dwell:  10.0 ms dwell matches 100 Baud transmission rate              │
│ [PASS] G4 Cross-Window:  Stationarity score 0.982 >= 0.600 threshold                   │
│ [PASS] G5 Plausibility:  SNR 11.71 dB > -10 dB, Baud rate obeys Nyquist channel capacity│
│ [PASS] G6 Reconstruction:Carrier PLL locked, SITOR-B sync framing confirmed            │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ COMPETING HYPOTHESIS RANKING POOL                                                      │
│ 1. MARITIME_NAVTEX:     Score 0.98 | Contradictions: 0 | Status: VALIDATED             │
│ 2. ASCII_TELEPRINTER:   Score 0.33 | Contradictions: 1 (Dwell 10.0ms != 9.09ms)        │
│ 3. RTTY_BAUDOT_45:      Score 0.30 | Contradictions: 1 (Dwell 10.0ms != 22.0ms)        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

The output gives downstream systems complete clarity on **what the signal is**, **what its physical values are**, and **exactly why the answer is trusted**.

<div style="page-break-after: always;"></div>

---

# Page 18: Why the Architecture Is Built This Way

The architecture's structure directly addresses the failure modes of previous SIGINT and automatic modulation classification systems.

```text
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │    BLIND     │     │ MULTI-WINDOW │     │   OPEN-SET   │
   │  EXTRACTION  │     │  VALIDATION  │     │  REJECTION   │
   └──────────────┘     └──────────────┘     └──────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
   No prior manual      Eliminates false      Acknowledges unknown
   parameter entry      transient spikes      signals cleanly

   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │    MULTI-    │     │CONTRADICTION-│     │  AUDITABLE   │
   │  HYPOTHESIS  │     │    AWARE     │     │  GOVERNANCE  │
   └──────────────┘     └──────────────┘     └──────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
   Retains competing    Challenges every      Every decision links
   interpretations      candidate actively    to physical evidence
```

### The Six Robustness Pillars

1. **Blind Extraction:**
   Never requires an operator to enter center frequency, baud rate, or bandwidth. Measures the physical properties directly from raw samples.
2. **Multi-Window Temporal Validation:**
   Splits signals into four observation windows to verify temporal stability. Rejects transient interference, ignition noise, and lightning bursts that appear as false signals in single-window tools.
3. **Open-Set Rejection (`UNKNOWN_OOD`):**
   Maintains explicit `UNKNOWN` and `UNKNOWN_OOD` states. When an unmodeled military transmission or exotic waveform appears, the system flags it as out-of-distribution instead of forcing a false classification.
4. **Multi-Hypothesis Tracking:**
   Preserves multiple candidate explanations concurrently. Avoids premature lock-in on the first plausible match.
5. **Contradiction-Aware Analysis:**
   Actively searches for reasons a hypothesis might be false. Deducts explicit mathematical penalties when observed values deviate from candidate standards.
6. **Auditable Governance & Epistemic Traceability:**
   Every parameter links to an epistemic pedigree (`OBSERVED`, `ESTIMATED`, `VALIDATED`). Evaluators and commanders can review the exact physical evidence behind every decision.

<div style="page-break-after: always;"></div>

---

# Page 19: Architecture Limitations and Boundaries

A scientifically credible architecture must define its operational boundaries. The NTRO Signal Analyzer does not claim to magically decode every transmission in the electromagnetic spectrum.

```text
               SIGNAL COMPLEXITY & CHANNEL NOISE
   ┌──────────────────────────────────────────────────────────┐
   │ OPERATIONAL REGIME (SNR >= 3 dB, Duration >= 0.5s)       │
   │ * Full blind extraction, multi-window validation,         │
   │   and closed-loop reconstruction operational.            │
   └──────────────────────────────────────────────────────────┘
                                │
                                ▼ Decreasing SNR / Duration
   ┌──────────────────────────────────────────────────────────┐
   │ DEGRADED REGIME (-6 dB <= SNR < 3 dB)                    │
   │ * Carrier and bandwidth extracted; baud rate estimation  │
   │   degrades. Promoted to ESTIMATED, not VALIDATED.        │
   └──────────────────────────────────────────────────────────┘
                                │
                                ▼ Decreasing SNR / Duration
   ┌──────────────────────────────────────────────────────────┐
   │ BOUNDARY REGIME (SNR < -6 dB or Duration < 1024 samples) │
   │ * Temporal validator flags INSUFFICIENT_DURATION.        │
   │   System declares UNKNOWN rather than guessing.          │
   └──────────────────────────────────────────────────────────┘
```

### Documented Technical Boundaries

1. **Low Signal-to-Noise Ratio (SNR $< 3\text{ dB}$):**
   In severe noise, spectral peak estimation experiences jitter and clock dwell histograms lose prominence. The system automatically downgrades its epistemic verdict to `ESTIMATED` or `AMBIGUOUS`.
2. **Short Observation Duration ($N < 1,024\text{ samples}$):**
   Multi-window validation requires sufficient observation time across all four windows. If duration is insufficient, Stage 5 returns `INSUFFICIENT_OBSERVATION_DURATION` and halts promotion to `VALIDATED`.
3. **Severe Multipath Rayleigh Fading:**
   In high-frequency (HF) skywave channels, ionospheric fading can cause selective carrier cancellations, mimicking amplitude keying on continuous FSK signals. Contradiction analysis catches envelope variance shifts and flags stability as `LOW`.
4. **Unmodeled / Proprietary Modulation Schemes:**
   If a transmission uses an uncataloged military spread-spectrum scheme, the Open-Set AMC returns `UNKNOWN_OOD`. The system outputs blind physical measurements (carrier, bandwidth, power) but refuses to guess a protocol name.
5. **Nominal Standards Are Not Measurements:**
   Catalog entries provide bounds for hypothesis testing, not measurement values. If a transmitter drifts from standard specification, the system reports the actual observed drift.

<div style="page-break-after: always;"></div>

---

# Page 20: One-Page Architecture Cheat Sheet

### The Complete End-to-End Processing Chain

```text
                           RAW INTERCEPTED RF SIGNAL (.wav / .iq / .bin)
                                                 │
[ 01. INGESTION ]         Detects audio vs true I/Q; Hilbert transform for mono
        │
[ 02. VALIDATION ]        Rejects NaNs, infinities, zero-power, and invalid sample rates
        │
[ 03. CONDITIONING ]      Removes receiver DC bias; normalizes signal power to unit variance
        │
[ 04. BLIND MEASURE ]     Extracts fc, 99% OBW, SNR, and dwell timing without protocol names
        │
[ 05. TEMPORAL CHECK ]    4-window temporal split; validates cross-window parameter stability
        │
[ 06. PULSE ANALYSIS ]    Detects pulse trains, duty cycle, PRI/PRF, and chirp slope
        │
[ 07. CLASSIFICATION ]    Rejects silence/noise; evaluates 19 physical invariant rules
        │
[ 08. HYPOTHESIZE ]       Generates pool of competing candidate explanations
        │
[ 09. CONTRADICTIONS ]    Searches for physical mismatches; applies explicit score penalties
        │
[ 10. VALIDATION GATE ]   Enforces six independent physical tests (G1 through G6)
        │
[ 11. DEEP EXTRACTION ]   Dispatches specialized strategy extractor (FSK, Radar, TDMA, PSK)
        │
[ 12. RECONSTRUCTION ]    Carrier PLL lock, Gardner clock recovery, soft LLR demod, CRC
        │
[ 13. UNCERTAINTY ]       Calculates estimator resolution bounds and cross-window variance
        │
[ 14. INTELLIGENCE ]      Delivers auditable, structured tactical intelligence record
                                                 │
                                                 ▼
                             VALIDATED TACTICAL RF INTELLIGENCE
```

---

### Core Operational Takeaway

```text
      ┌────────────────────────────────────────────────────────┐
      │                     MEASURE FIRST.                     │
      │                  HYPOTHESIZE SECOND.                   │
      │                CHALLENGE THE HYPOTHESIS.               │
      │            VALIDATE BEFORE CLAIMING CERTAINTY.         │
      └────────────────────────────────────────────────────────┘
```

* **No Black Boxes:** Every classification is backed by an auditable chain of physical measurements.
* **No Forced Guesses:** Noise and out-of-distribution signals return `UNKNOWN` and `UNKNOWN_OOD`.
* **Complete Trust:** Every reported parameter includes its value, epistemic status, temporal stability level, and physical uncertainty bound.
