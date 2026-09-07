# 📡 NTRO Autonomous Signal Intelligence Workstation (SIH26147)
## Quick Start & Teammate Execution Guide

This package is completely self-contained. Anyone on the team or an evaluator can extract the zip file and run the application with a single double-click.

---

### Step 1: Extract the ZIP
Right-click the `.zip` file and select **Extract All...** to any location on your PC (e.g. `Downloads`, `Documents`, or `C:\`).

---

### Step 2: Launch the Application (1-Click)
Inside the extracted folder, double-click:
👉 **`START_DASHBOARD.bat`** (or **`LAUNCH_APP.bat`**)

#### What happens automatically:
1. **Python Detection**: It checks for Python (compatible with Python 3.10 to 3.14). If Python is missing, it displays a direct download link.
2. **Auto-Dependency Check**: If `numpy`, `scipy`, `plotly`, or `streamlit` are not installed yet on your PC, it automatically runs `pip install -r requirements.txt` for you.
3. **Browser Auto-Launch**: It starts the local server and automatically opens your default browser (Chrome, Canary, Edge) to:
   ```
   http://localhost:8501
   ```

---

### Verifying the System
Inside the folder, you also have two automated verification tools:

- **`RUN_BENCHMARK.bat`**: Double-click to run the defense benchmark across all 25 SigIDWiki signals. Outputs modulation classification, confidence score, SNR, Baud, and PRF scorecard in the terminal.
- **`RUN_ALL_TESTS.bat`**: Double-click to run the full automated unit and regression test suite (35 tests).

---

### How to Use the Dashboard
1. **Preset Catalog (Left Sidebar)**: Select from 25 real-world SigIDWiki signals (Satellite beacons, FMCW/OTH Radars, Cellular TDMA, Tactical MFSK, Morse Code, RTTY, etc.).
2. **Custom File Upload**: Drag and drop any `.wav` or raw binary `.iq` file. The engine autonomously probes the byte container, sample rate, modulation scheme, and parameters.
3. **Spectral Analyzers**: In the **Full Numbers View**, click any of the 5 quick-launch analyzer buttons (**Spectrogram Waterfall**, **Power Spectrum (PSD)**, **I/Q Constellation**, **Eye Diagram**, **Time Envelope**) to enter split-screen mode.
4. **Hardware Acceleration**: Constellation plots use WebGL for instant rendering with zero animation lag.
5. **Export**: Use the **Download JSON** and **Download CSV** buttons to export the extracted telemetry for external SIGINT reporting.
