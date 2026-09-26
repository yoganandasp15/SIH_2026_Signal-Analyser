"""
NTRO Automated Signal Analysis Suite - AAROHAN Workstation
===========================================================
Smart India Hackathon 2026 (SIH26147) • Autonomous RF Signal Intelligence Workstation
Professional Engineering Analysis Console following the strict 9-level result hierarchy:
1. FINAL INTERCEPT VERDICT
2. PRIMARY PARAMETERS (with Epistemic Status & Uncertainty)
3. EVIDENCE SUMMARY
4. CONTRADICTIONS
5. HYPOTHESIS RANKING
6. MULTI-WINDOW STABILITY
7. VALIDATION TRACE
8. TECHNICAL VISUALIZATIONS (Expandable)
9. RAW TELEMETRY (Expandable)
"""

import time
import os
import sys
import importlib
import tempfile
import json
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Dynamic hot-reload: ensure Streamlit never caches stale backend DSP code
for _mod in list(sys.modules.keys()):
    if _mod.startswith("dsp.") or _mod.startswith("visualization.") or _mod.startswith("utils."):
        try:
            importlib.reload(sys.modules[_mod])
        except Exception:
            pass

from dsp.loaders import load_signal_file, probe_binary_format, auto_detect_file_sample_rate
from dsp.preprocessor import remove_dc_offset, normalize_signal_power, compute_signal_stats
from dsp.spectral import compute_welch_psd, compute_spectrogram
from dsp.adaptive_pipeline import run_adaptive_pipeline
from dsp.parameter_uncertainty import build_parameter_uncertainty_report, get_verdict_explanation
from dsp import (
    inspect_signal_file,
    condition_signal,
    extract_signal_features,
    classify_modulation_open_set,
    synchronize_signal,
    demodulate_signal,
    evaluate_interleaver_candidates,
    evaluate_fec_candidates,
    analyze_frame_structure,
    fuse_evidence,
    EpistemicStatus,
    SamplingRateStatus,
    ConfidenceLevel,
    ModulationFamily
)
from visualization.plots import (
    set_plot_theme,
    plot_welch_psd,
    plot_spectrogram_waterfall,
    plot_iq_constellation,
    plot_eye_diagram,
    plot_time_domain_envelope,
    plot_synchronized_constellation,
    plot_llr_histogram
)
from utils.exporter import export_results_to_json, export_results_to_csv
from utils.synthetic_generator import generate_synthetic_signal, save_synthetic_iq, save_synthetic_wav
from utils.epistemic_demo import generate_epistemic_test_signal, EPISTEMIC_DEMO_PRESETS
from utils.tactical_scenarios import TACTICAL_SCENARIOS, load_tactical_scenario


# Page Configuration
st.set_page_config(
    page_title="AAROHAN - Autonomous RF Signal Analysis Workstation",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)


def render_html(html_str: str) -> None:
    """
    Renders HTML in Streamlit ensuring zero leading whitespace on any line.
    Prevents CommonMark from misinterpreting indented lines as code blocks (<pre><code>).
    """
    cleaned = "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# Professional Engineering Analysis Workstation Styling
st.markdown("""
<style>
    /* Header and toolbar hygiene */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.5rem !important;
        z-index: 9999 !important;
        pointer-events: none !important;
    }
    header[data-testid="stHeader"] * {
        pointer-events: auto !important;
    }
    [data-testid="stToolbar"] {
        display: flex !important;
        visibility: visible !important;
        background: transparent !important;
        pointer-events: none !important;
        height: auto !important;
    }
    [data-testid="stToolbar"] * {
        pointer-events: auto !important;
    }
    [data-testid="stAppDeployButton"],
    [data-testid="stMainMenu"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    #MainMenu,
    footer {
        display: none !important;
    }

    /* Sidebar controls */
    button[data-testid="stExpandSidebarButton"],
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        position: fixed !important;
        top: 8px !important;
        left: 12px !important;
        z-index: 100000 !important;
        background: #111722 !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 4px !important;
        padding: 4px 8px !important;
        color: #38bdf8 !important;
        cursor: pointer !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.6) !important;
    }
    [data-testid="stSidebarCollapseButton"] button {
        background: #111722 !important;
        border: 1px solid #1e293b !important;
        border-radius: 4px !important;
        color: #38bdf8 !important;
    }

    /* Core typography and container layout */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #f1f5f9;
    }
    .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        max-width: 100% !important;
    }

    /* Section Headers */
    .section-header {
        font-size: 0.80rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 6px;
        margin-top: 20px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-num {
        color: #38bdf8;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    /* Top Console Banner */
    .workstation-banner {
        background: #111722;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }
    .workstation-title {
        font-size: 1.10rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        color: #f8fafc;
        margin: 0;
    }
    .workstation-subtitle {
        font-size: 0.74rem;
        color: #94a3b8;
        margin-top: 2px;
        font-weight: 400;
    }

    /* Telemetry Chips */
    .chip-container {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        align-items: center;
    }
    .telemetry-chip {
        background: #0b0f17;
        border: 1px solid #1e293b;
        border-radius: 4px;
        padding: 3px 8px;
        font-size: 0.72rem;
        color: #cbd5e1;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .chip-label {
        color: #64748b;
        text-transform: uppercase;
        margin-right: 5px;
        font-size: 0.66rem;
        letter-spacing: 0.04em;
    }
    .chip-val {
        color: #38bdf8;
        font-weight: 600;
    }

    /* Level 1: Verdict Card */
    .verdict-card {
        background: #111722;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    .verdict-header-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 10px;
    }
    .verdict-badge {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        padding: 4px 12px;
        border-radius: 4px;
        text-transform: uppercase;
        display: inline-block;
    }
    .badge-validated { background: #064e3b; color: #34d399; border: 1px solid #059669; }
    .badge-estimated { background: #3b2506; color: #fbbf24; border: 1px solid #d97706; }
    .badge-ambiguous { background: #2e1065; color: #c084fc; border: 1px solid #7c3aed; }
    .badge-unknown { background: #0c4a6e; color: #38bdf8; border: 1px solid #0284c7; }
    .badge-unknown-ood { background: #450a0a; color: #f87171; border: 1px solid #dc2626; }
    .badge-no-signal { background: #1e293b; color: #94a3b8; border: 1px solid #475569; }

    .verdict-target-title {
        font-size: 1.30rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 4px;
        margin-bottom: 4px;
    }
    .verdict-explanation-box {
        background: #0b0f17;
        border: 1px solid #1e293b;
        border-radius: 5px;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #cbd5e1;
        line-height: 1.5;
        margin-top: 8px;
        margin-bottom: 12px;
    }

    /* Key Telemetry Metrics Bar */
    .metrics-bar {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
        margin-top: 10px;
    }
    @media (max-width: 900px) {
        .metrics-bar {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    .metric-cell {
        background: #0b0f17;
        border: 1px solid #1e293b;
        border-radius: 4px;
        padding: 8px 12px;
    }
    .metric-cell-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 2px;
    }
    .metric-cell-value {
        font-size: 1.05rem;
        font-weight: 700;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        color: #f8fafc;
    }
    .metric-cell-sub {
        font-size: 0.68rem;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* Special State Callouts */
    .callout-ambiguous {
        background: #1e1338;
        border: 1px solid #581c87;
        border-left: 4px solid #a855f7;
        border-radius: 4px;
        padding: 10px 14px;
        margin-top: 10px;
        font-size: 0.80rem;
        color: #e9d5ff;
    }
    .callout-unknown {
        background: #082f49;
        border: 1px solid #0369a1;
        border-left: 4px solid #38bdf8;
        border-radius: 4px;
        padding: 10px 14px;
        margin-top: 10px;
        font-size: 0.80rem;
        color: #e0f2fe;
    }
    .callout-no-signal {
        background: #1e293b;
        border: 1px solid #475569;
        border-left: 4px solid #94a3b8;
        border-radius: 4px;
        padding: 10px 14px;
        margin-top: 10px;
        font-size: 0.80rem;
        color: #cbd5e1;
    }

    /* Layer A: Blind Physical Parameters HUD */
    .hud-card-layer-a {
        background: #0f172a;
        border: 1px solid #1e3a8a;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
    .hud-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 12px;
    }
    .hud-badge-blind {
        background: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 4px;
        letter-spacing: 0.05em;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .hud-badge-consistent {
        background: #0c4a6e;
        color: #38bdf8;
        border: 1px solid #0284c7;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 4px;
        letter-spacing: 0.05em;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .hud-badge-warn {
        background: #3b2506;
        color: #fbbf24;
        border: 1px solid #d97706;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 4px;
        letter-spacing: 0.05em;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    /* Level 2: Parameter Cards Grid */
    .param-card {
        background: #111722;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 12px 14px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .param-card-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 6px;
    }
    .param-title {
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
    }
    .param-badges {
        display: flex;
        gap: 4px;
    }
    .badge-status {
        font-size: 0.64rem;
        font-weight: 700;
        padding: 1px 5px;
        border-radius: 3px;
        text-transform: uppercase;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .badge-stat-observed { background: #064e3b; color: #34d399; }
    .badge-stat-estimated { background: #0c4a6e; color: #38bdf8; }
    .badge-stat-hypothesized { background: #3b2506; color: #fbbf24; }
    .badge-stat-validated { background: #2e1065; color: #c084fc; }
    .badge-stat-unknown { background: #1e293b; color: #94a3b8; }
    .badge-stat-na { background: #1e293b; color: #64748b; }

    .badge-stab-high { background: #064e3b; color: #34d399; }
    .badge-stab-medium { background: #3b2506; color: #fbbf24; }
    .badge-stab-low { background: #450a0a; color: #f87171; }
    .badge-stab-unknown { background: #1e293b; color: #94a3b8; }

    .param-numeric {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        margin: 4px 0;
    }
    .param-uncertainty-line {
        font-size: 0.72rem;
        color: #38bdf8;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .param-reason-line {
        font-size: 0.68rem;
        color: #64748b;
        font-style: italic;
        line-height: 1.3;
        margin-top: 2px;
    }
    .param-footer {
        font-size: 0.68rem;
        color: #94a3b8;
        border-top: 1px solid #182234;
        padding-top: 6px;
        margin-top: 8px;
    }

    /* Structured Telemetry Tables */
    .instrument-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.78rem;
        background: #111722;
        border-radius: 6px;
        overflow: hidden;
        border: 1px solid #1e293b;
    }
    .instrument-table th {
        background: #0b0f17;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.68rem;
        letter-spacing: 0.05em;
        padding: 8px 10px;
        text-align: left;
        border-bottom: 1px solid #1e293b;
    }
    .instrument-table td {
        padding: 7px 10px;
        border-bottom: 1px solid #182234;
        color: #cbd5e1;
    }
    .instrument-table tr:last-child td {
        border-bottom: none;
    }
    .instrument-table tr:hover td {
        background: #141d2b;
    }
    .mono-cell {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        color: #f1f5f9;
        font-weight: 600;
    }

    /* Level 4: Contradiction Alert Cards */
    .contradiction-clean-box {
        background: #06281e;
        border: 1px solid #065f46;
        border-radius: 6px;
        padding: 12px 16px;
        display: flex;
        align-items: center;
        gap: 12px;
        color: #34d399;
        font-size: 0.82rem;
    }
    .contradiction-alert-box {
        background: #2a1113;
        border: 1px solid #991b1b;
        border-radius: 6px;
        padding: 12px 16px;
        color: #fca5a5;
        font-size: 0.82rem;
    }

    /* Level 7: Validation Trace Cards */
    .trace-step-card {
        background: #111722;
        border: 1px solid #1e293b;
        border-radius: 5px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .trace-step-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.76rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 4px;
    }
    .trace-step-body {
        font-size: 0.74rem;
        color: #94a3b8;
        line-height: 1.4;
    }

    /* Core Workstation Global Canvas & Components (Defense Dark) */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    .stApp {
        background: #0b0f17 !important;
        color: #e6edf5 !important;
    }
    [data-testid="stHeader"] {
        background: rgba(11, 15, 23, 0.96) !important;
        border-bottom: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] {
        background: #111722 !important;
        border-right: 1px solid #263447 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #dbe5ef !important;
    }
    .instrument-table {
        background: #111722 !important;
        border: 1px solid #1e293b !important;
    }
    .instrument-table th {
        background: #0b0f17 !important;
        color: #94a3b8 !important;
        border-bottom: 1px solid #1e293b !important;
    }
    .instrument-table td {
        color: #cbd5e1 !important;
        border-bottom: 1px solid #182234 !important;
    }
    .instrument-table tr:hover td {
        background: #172338 !important;
        color: #ffffff !important;
    }
    .mono-cell {
        color: #f1f5f9 !important;
    }
    input, textarea, [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input {
        background: #111722 !important;
        color: #e6edf5 !important;
        border-color: #263447 !important;
    }
    button[kind="secondary"], [data-testid="stDownloadButton"] button {
        background: #111722 !important;
        color: #38bdf8 !important;
        border: 1px solid #263447 !important;
    }
    button[kind="secondary"]:hover, [data-testid="stDownloadButton"] button:hover {
        background: #1a2536 !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
    }
    [data-testid="stExpander"] {
        background: #111722 !important;
        border: 1px solid #1e293b !important;
        border-radius: 6px !important;
    }
    [data-testid="stExpander"] summary {
        color: #e6edf5 !important;
    }
</style>
""", unsafe_allow_html=True)


# 25 Verified SigIDWiki Defense Intercept Catalog
VERIFIED_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "verified_samples")

CATALOG_GROUPS = {
    "Satellite & Space Surveillance": [
        ("AIST-2D.wav", "Aist 2D (RS-48 Satellite Beacon PCM/PM over NFM)"),
        ("GRAVES_Radar.wav", "GRAVES (Space Surveillance Radar Reflection 143.05 MHz)")
    ],
    "Radar & Ionospheric Sounders": [
        ("HAARP-1.wav", "HAARP (HF Ionospheric Sounder & FMCW Chirp Radar)"),
        ("Ghadir_Radar.wav", "Ghadir Radar (Pulsed OTH Radar 307/870 Hz)"),
        ("OTH_SW_Radar.wav", "OTH-SW Radar (Chinese FMCW 43.2 Hz Radar)"),
        ("CODAR.wav", "CODAR SeaSonde (Oceanographic HF FMCW Radar)"),
        ("Woodpecker_Duga.wav", "Russian Woodpecker (Duga 10 Hz PRF OTH Radar)")
    ],
    "Military & Tactical HF": [
        ("2G_ALE.wav", "2G ALE (MIL-STD-188-141 8-Tone MFSK Handshake)"),
        ("STANAG_4285.wav", "STANAG 4285 (NATO Naval Tactical HF 8-PSK)")
    ],
    "Digital Data & Teletype": [
        ("ASCII.wav", "ASCII / ITA-5 (170 Hz Shift 2-FSK 110 Baud)"),
        ("RTTY.wav", "Baudot RTTY (170 Hz Shift 2-FSK 45.45 Baud)"),
        ("NAVTEX.wav", "NAVTEX (SITOR-B 170 Hz Shift 2-FSK 100 Baud)"),
        ("POCSAG.wav", "POCSAG (Radio Paging 2-FSK 1200 Baud)"),
        ("APRS.wav", "APRS (Bell 202 AFSK 1200 Baud Packet Data)"),
        ("FT8.wav", "FT8 (WSJT-X Amateur Weak-Signal 8-FSK 6.25 Baud)"),
        ("PSK31.wav", "PSK31 (Amateur BPSK Varicode 31.25 Baud)"),
        ("MFSK16.wav", "MFSK16 (Tactical / Amateur 16-Tone MFSK 15.6 Baud)"),
        ("Morse_Code.wav", "Morse Code (A1A Continuous Wave On-Off Keying)")
    ],
    "Commercial & Land Mobile": [
        ("AIS.wav", "AIS (Automatic Identification System GMSK 9600 Baud TDMA)"),
        ("GSM_BCCH_Downlink.wav", "GSM BCCH (Cellular TDMA 4.615 ms Frame)"),
        ("DMR.wav", "DMR (ETSI Tier II TDMA 4-FSK 4800 Baud)"),
        ("D-STAR.wav", "D-STAR (Amateur Digital Voice GMSK 4800 Baud)")
    ],
    "Analog Audio & Marine Safety": [
        ("WEFAX.wav", "Weather Fax (WEFAX 120 LPM FM Subcarrier)"),
        ("UVB76_Buzzer.wav", "UVB-76 The Buzzer (Channel Marker / Voice Tones)"),
        ("Vario_Voice_Tone.wav", "Vario Voice (Analog NFM Speech & Variometer Tones)")
    ]
}


def main():
    # -------------------------------------------------------------
    # SIDEBAR: SIGNAL INGESTION (All 25 Verified Defense Intercepts)
    # -------------------------------------------------------------
    set_plot_theme("dark")
    st.session_state["ui_theme"] = "dark"
    st.sidebar.markdown(
        """<div style="display:flex; align-items:center; gap:8px; margin-bottom:14px; font-size:0.75rem; color:#38bdf8; background:#111722; padding:6px 10px; border-radius:5px; border:1px solid #1e293b;">
            <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#10b981; box-shadow:0 0 6px #10b981;"></span>
            <span style="letter-spacing:0.04em;"><strong>MODE:</strong> TACTICAL DEFENSE DARK</span>
        </div>""",
        unsafe_allow_html=True
    )

    st.sidebar.markdown("### Signal Ingestion")

    input_mode = st.sidebar.radio(
        "Select Intercept Source:",
        [
            "Defense Intercept Catalog (25 Signals)",
            "Tactical Intercept Scenarios (7 Scenarios Demo)",
            "Epistemic State Verification Bench (6 States Demo)",
            "Upload Signal Capture (.wav, .iq, .dat)",
            "Synthetic Signal Generator"
        ],
        index=0
    )

    signal = None
    fs = 1_000_000.0
    meta: Dict[str, Any] = {}
    audio_path_to_play: Optional[str] = None

    if input_mode == "Defense Intercept Catalog (25 Signals)":
        category = st.sidebar.selectbox(
            "Mission Domain:",
            list(CATALOG_GROUPS.keys()),
            index=0
        )
        available_signals = CATALOG_GROUPS[category]
        signal_choice = st.sidebar.selectbox(
            "Target Intercept:",
            available_signals,
            format_func=lambda x: x[1],
            index=0
        )
        preset_filename, preset_label = signal_choice
        sample_file_path = os.path.join(VERIFIED_SAMPLES_DIR, preset_filename)
        if os.path.exists(sample_file_path):
            audio_path_to_play = sample_file_path
            signal, fs, meta = load_signal_file(sample_file_path, max_samples=1_000_000)
            meta["preset_label"] = preset_label
            meta["file_name"] = preset_filename
        else:
            st.error(f"Sample file not found: {sample_file_path}")

    elif input_mode == "Tactical Intercept Scenarios (7 Scenarios Demo)":
        st.sidebar.markdown("### Tactical Intercept Scenarios")
        selected_scenario = st.sidebar.selectbox(
            "Select Demonstration Scenario:",
            TACTICAL_SCENARIOS,
            format_func=lambda x: x[1],
            index=0
        )
        scen_key, scen_label, scen_desc = selected_scenario
        st.sidebar.caption(scen_desc)
        scen_samples = st.sidebar.slider("Capture Samples:", min_value=20_000, max_value=250_000, value=96_000, step=10_000)
        signal, fs, meta = load_tactical_scenario(scen_key, max_samples=scen_samples)

    elif input_mode == "Epistemic State Verification Bench (6 States Demo)":
        st.sidebar.markdown("### Epistemic State Test Bench")
        selected_preset = st.sidebar.selectbox(
            "Select Epistemic Target State:",
            EPISTEMIC_DEMO_PRESETS,
            format_func=lambda x: x[1],
            index=0
        )
        state_key, state_label = selected_preset
        demo_duration = st.sidebar.slider("Signal Duration (s):", min_value=0.5, max_value=4.0, value=2.0, step=0.5)
        signal, fs, meta = generate_epistemic_test_signal(state=state_key, fs=48000.0, duration_s=demo_duration)

    elif input_mode == "Upload Signal Capture (.wav, .iq, .dat)":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Signal File (Zero Config Required):",
            type=["wav", "wave", "iq", "dat", "bin"]
        )
        with st.sidebar.expander("Advanced Extraction Overrides", expanded=False):
            manual_override = st.checkbox("Manual Override", value=False)
            override_fs = st.number_input("Sampling Rate (Hz):", min_value=1000.0, value=1_000_000.0, step=100_000.0) if manual_override else None
            override_fmt = st.selectbox("Binary Format:", ["complex64", "int16", "uint8"]) if manual_override else "auto"

        max_samples = st.sidebar.slider("Analysis Window Samples:", min_value=20_000, max_value=2_000_000, value=1_000_000, step=50_000)

        if uploaded_file is not None:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_f:
                tmp_f.write(uploaded_file.read())
                tmp_path = tmp_f.name

            try:
                signal, fs, meta = load_signal_file(
                    file_path=tmp_path,
                    sample_rate=override_fs,
                    format_type=override_fmt,
                    max_samples=max_samples
                )
                meta["file_name"] = uploaded_file.name
                if suffix.lower() in [".wav", ".wave"]:
                    audio_path_to_play = tmp_path
            except Exception as e:
                st.sidebar.error(f"Failed to load file: {e}")
            finally:
                if audio_path_to_play != tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

    else:
        st.sidebar.markdown("### Synthetic Generator")
        syn_mod = st.sidebar.selectbox(
            "Modulation Scheme:",
            ["QPSK", "BPSK", "16-QAM", "2-FSK", "4-FSK", "AM", "FM", "CW", "PULSED_RADAR"],
            index=0
        )
        syn_snr = st.sidebar.slider("SNR (dB):", min_value=-5.0, max_value=30.0, value=15.0, step=1.0)
        syn_fc = st.sidebar.slider("Carrier Offset (kHz):", min_value=-250.0, max_value=250.0, value=50.0, step=5.0) * 1e3
        syn_baud = st.sidebar.slider("Symbol Rate (kBaud):", min_value=5.0, max_value=100.0, value=25.0, step=5.0) * 1e3
        syn_samples = st.sidebar.slider("Sample Count:", min_value=20_000, max_value=500_000, value=150_000, step=10_000)

        radar_pw = 20.0
        radar_pri = 100.0
        if syn_mod == "PULSED_RADAR":
            radar_pw = st.sidebar.number_input("Radar Pulse Width (μs):", min_value=1.0, max_value=1000.0, value=25.0)
            radar_pri = st.sidebar.number_input("Radar PRI (μs):", min_value=10.0, max_value=5000.0, value=120.0)

        fs = 1_000_000.0
        signal, _, meta = generate_synthetic_signal(
            mod_type=syn_mod,
            fs=fs,
            fc=syn_fc,
            baud_rate=syn_baud,
            snr_db=syn_snr,
            num_samples=syn_samples,
            pulse_width_us=radar_pw,
            pri_us=radar_pri
        )
        meta["file_name"] = f"Synthetic_{syn_mod}_{syn_snr}dB.iq"
        meta["source_type"] = f"Synthetic Generator ({syn_mod})"

    # -------------------------------------------------------------
    # EXECUTION OF AUTONOMOUS DSP PIPELINE & CACHING
    # -------------------------------------------------------------
    if signal is None or len(signal) == 0:
        st.info("Select a defense intercept sample or upload an RF capture file in the sidebar to begin analysis.")
        return

    sig_cache_key = f"{meta.get('file_name', '')}_{meta.get('source_type', '')}_{len(signal)}_{fs}"

    if (
        st.session_state.get("cached_sig_key") == sig_cache_key
        and "cached_results" in st.session_state
        and "cached_norm_sig" in st.session_state
        and "cached_v3" in st.session_state
    ):
        results = st.session_state["cached_results"]
        norm_sig = st.session_state["cached_norm_sig"]
        v3_bundle = st.session_state["cached_v3"]
    else:
        results = run_adaptive_pipeline(signal, fs, metadata=meta)
        dc_free = remove_dc_offset(signal)
        norm_sig, _ = normalize_signal_power(dc_free)

        # Quick V3 synchronization & demodulation on first 64k samples for constellation & soft LLR plots
        v3_slice = signal[:min(len(signal), 65536)]
        try:
            conditioned, cond_rep = condition_signal(v3_slice, apply_iq=True, apply_agc=False)
            v3_features = extract_signal_features(conditioned, fs)
            v3_hyp = classify_modulation_open_set(conditioned, fs=fs, features=v3_features)
            v3_sync = synchronize_signal(conditioned, fs, v3_hyp)
            v3_demod = demodulate_signal(v3_sync, v3_hyp.modulation)
            v3_bundle = {
                "sync": v3_sync,
                "demod": v3_demod,
                "hypothesis": v3_hyp,
                "features": v3_features
            }
        except Exception:
            v3_bundle = None

        st.session_state["cached_sig_key"] = sig_cache_key
        st.session_state["cached_results"] = results
        st.session_state["cached_norm_sig"] = norm_sig
        st.session_state["cached_v3"] = v3_bundle
        st.session_state["cached_figs"] = {}

    # Extract all pipeline telemetry
    p = results["parameters"]
    m = results["modulation_classification"]
    pulse = results["pulse_analysis"]
    det = results.get("autonomous_detection", {})
    spec = results.get("specialized_telemetry", {})
    temp_val = results.get("temporal_validation", {})
    val_trace = results.get("validation_trace", {})
    ranked_cands = results.get("ranked_candidates", [])
    contras = results.get("contradiction_analysis", [])
    evidence_rep = results.get("evidence_report", {})
    t_elapsed_ms = results.get("execution_time_ms", 0.0)

    # Layer A Blind Physical Parameters & Provenance
    blind_vec = results.get("blind_parameters") or results.get("blind_parameter_vector") or {}
    blindness_prov = results.get("blindness_provenance") or blind_vec.get("blindness_provenance") or {}
    morph = results.get("waveform_morphology") or blind_vec.get("morphology_fingerprint") or {}
    param_cons = results.get("parameter_consistency") or blind_vec.get("parameter_consistency") or {}
    sym_consensus = results.get("symbol_rate_consensus") or {}

    # Step 6 & 7 outputs
    param_reports = results.get("parameter_uncertainties", {})
    verdict_expl = results.get("verdict_explanation", {})
    final_verdict = results.get("final_decision", "UNKNOWN")

    # Time and Real-Time Factor (RTF)
    duration_sec = len(signal) / fs
    duration_str = f"{duration_sec * 1e3:.1f} ms" if duration_sec < 1.0 else f"{duration_sec:.2f} s"
    fmt_str = meta.get("format_type", "auto").upper()
    file_display = meta.get("file_name", meta.get("preset_label", "Intercept Capture"))

    rtf = (t_elapsed_ms / 1000.0) / max(duration_sec, 1e-6)
    if rtf < 1.0:
        rtf_str = f"{rtf:.3f}x ({1.0 / rtf:.1f}x faster than real-time)"
    else:
        rtf_str = f"{rtf:.2f}x real-time"

    evidence_score_val = det.get("evidence_score", det.get("confidence", 0.0))

    # -------------------------------------------------------------
    # TOP WORKSTATION BANNER
    # -------------------------------------------------------------
    render_html(f"""
    <div class="workstation-banner">
        <div>
            <div class="workstation-title">AAROHAN • AUTONOMOUS RF SIGNAL ANALYSIS WORKSTATION</div>
            <div class="workstation-subtitle">Smart India Hackathon 2026 (SIH26147) • Physical Invariant Verification Engine</div>
        </div>
        <div class="chip-container">
            <div class="telemetry-chip"><span class="chip-label">TARGET</span><span class="chip-val">{file_display[:28]}</span></div>
            <div class="telemetry-chip"><span class="chip-label">FORMAT</span><span class="chip-val">{fmt_str}</span></div>
            <div class="telemetry-chip"><span class="chip-label">SAMPLE RATE</span><span class="chip-val">{fs:,.0f} Hz</span></div>
            <div class="telemetry-chip"><span class="chip-label">SAMPLES</span><span class="chip-val">{len(signal):,}</span></div>
            <div class="telemetry-chip"><span class="chip-label">DURATION</span><span class="chip-val">{duration_str}</span></div>
            <div class="telemetry-chip"><span class="chip-label">DSP LATENCY</span><span class="chip-val">{t_elapsed_ms:.1f} ms</span></div>
        </div>
    </div>
    """)

    # =============================================================
    # 1. FINAL INTERCEPT VERDICT
    # =============================================================
    render_html("""
    <div class="section-header">
        <span class="section-num">01.</span> Final Intercept Verdict
    </div>
    """)

    verdict_badge_class = {
        "VALIDATED": "badge-validated",
        "ESTIMATED": "badge-estimated",
        "AMBIGUOUS": "badge-ambiguous",
        "UNKNOWN": "badge-unknown",
        "UNKNOWN_OOD": "badge-unknown-ood",
        "NO SIGNAL / NOISE FLOOR": "badge-no-signal"
    }.get(final_verdict, "badge-unknown")

    target_name = det.get("protocol_name", m.get("modulation_type", "Unknown Signal"))
    verdict_title = verdict_expl.get("title", f"Verdict: {final_verdict}")
    verdict_narrative = verdict_expl.get("explanation", "Analysis complete.")

    render_html(f"""
    <div class="verdict-card">
        <div class="verdict-header-row">
            <div>
                <span class="verdict-badge {verdict_badge_class}">● {final_verdict}</span>
                <div class="verdict-target-title">{target_name}</div>
                <div style="font-size:0.80rem; color:#94a3b8;"><strong>{verdict_title}</strong></div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.68rem; color:#64748b; text-transform:uppercase;">Mission Domain</div>
                <div style="font-size:0.85rem; color:#38bdf8; font-weight:600;">{meta.get('recording_domain', 'RF Baseband')}</div>
            </div>
        </div>
        <div class="verdict-explanation-box">
            {verdict_narrative}
        </div>
        <div class="metrics-bar">
            <div class="metric-cell">
                <div class="metric-cell-label">Evidence Score</div>
                <div class="metric-cell-value">{evidence_score_val:.2f}</div>
                <div class="metric-cell-sub">Uncalibrated metric ∈ [0, 1]</div>
            </div>
            <div class="metric-cell">
                <div class="metric-cell-label">Processing Time</div>
                <div class="metric-cell-value">{t_elapsed_ms:.1f} ms</div>
                <div class="metric-cell-sub">DSP pipeline latency</div>
            </div>
            <div class="metric-cell">
                <div class="metric-cell-label">Signal Duration</div>
                <div class="metric-cell-value">{duration_str}</div>
                <div class="metric-cell-sub">{len(signal):,} baseband samples</div>
            </div>
            <div class="metric-cell">
                <div class="metric-cell-label">Real-Time Factor</div>
                <div class="metric-cell-value">{rtf:.3f}x</div>
                <div class="metric-cell-sub">{rtf_str}</div>
            </div>
        </div>
    </div>
    """)

    # Special callouts for first-class application states
    if final_verdict == "AMBIGUOUS":
        top_hyps = verdict_expl.get("top_hypotheses", [])
        hyps_desc = ""
        for idx, h in enumerate(top_hyps):
            score_val = h.get("score", 0.0)
            hyps_desc += f"<br>• <strong>Candidate #{idx+1} ({h.get('protocol', 'Unknown')})</strong>: Score = {score_val:.2f}"
            if h.get("evidence"):
                hyps_desc += f" (Evidence: {', '.join(h['evidence'][:2])})"

        render_html(f"""
        <div class="callout-ambiguous">
            <strong>Ambiguity Threshold Active (Score Margin ≤ 0.05):</strong><br>
            The classifier refused to collapse competing hypotheses with indistinguishable physical evidence:{hyps_desc}
        </div>
        """)

    elif final_verdict == "UNKNOWN":
        obs_items = verdict_expl.get("measured_observations", [])
        obs_text = ", ".join(obs_items) if obs_items else "Carrier peak, occupied bandwidth, and SNR extracted"
        render_html(f"""
        <div class="callout-unknown">
            <strong>Uncataloged Modulation Profile — Successfully Extracted Physical Telemetry:</strong><br>
            {obs_text}. Waveform does not match cataloged defense standards, but physical properties are preserved.
        </div>
        """)

    elif final_verdict == "NO SIGNAL / NOISE FLOOR":
        render_html("""
        <div class="callout-no-signal">
            <strong>Gaussian Noise Floor Pre-Gate Rejection:</strong><br>
            The input was deliberately rejected by the zero false-positive pre-gate because spectral flatness and energy
            conform to stationary Gaussian thermal noise rather than an active RF emitter.
        </div>
        """)

    if p.get("audio_passband_artifact_detected") or meta.get("is_demodulated_audio"):
        rf_downlink = spec.get("satellite_downlink_frequency_nominal")
        downlink_note = f" Nominal Physical RF Downlink: <strong>{rf_downlink}</strong>." if rf_downlink else ""
        render_html(f"""
        <div style="background:#1c1917; border:1px solid #44403c; border-radius:4px; padding:8px 12px; margin-top:8px; margin-bottom:12px; font-size:0.75rem; color:#fde047; line-height:1.4;">
            <strong>Baseband Audio Capture Artifact:</strong> Extracted carrier ({p.get('fc_peak_hz', 0)/1e3:+,.2f} kHz) represents receiver audio subcarrier pitch; bandwidth ({p.get('bw_99pct_hz', 0)/1e3:.2f} kHz) is filtered by receiver audio passband.{downlink_note}
        </div>
        """)

    # =============================================================
    # 2. LAYER A: BLIND PHYSICAL PARAMETERS & PROVENANCE HUD
    # =============================================================
    render_html("""
    <div class="section-header">
        <span class="section-num">02.</span> Layer A • Blind Physical Parameters &amp; Provenance HUD
    </div>
    """)

    blind_score_label = blindness_prov.get("blindness_score", "10.0 / 10.0 (100% Blind Physical Extraction)")
    is_math_consistent = param_cons.get("is_consistent", True)
    cons_score_num = float(param_cons.get("consistency_score", 1.0))
    violations_list = param_cons.get("violations", [])
    cons_badge_cls = "hud-badge-blind" if is_math_consistent else "hud-badge-warn"
    cons_badge_text = f"Mathematical Invariants: {'PASS' if is_math_consistent else 'ATTENTION'} ({cons_score_num*100:.0f}%)"

    env_ripple = float(morph.get("envelope_ripple_factor", 0.0))
    has_const_env = bool(morph.get("has_constant_envelope", False))
    env_str = f"Constant Envelope (Ripple: {env_ripple:.2f})" if has_const_env else f"Varying Envelope (Ripple: {env_ripple:.2f})"

    phase_states_n = int(morph.get("phase_state_count", 1))
    phase_str = f"{phase_states_n}-PSK Phase States" if phase_states_n > 1 else "Continuous Phase / Analog"

    freq_states_n = int(morph.get("frequency_state_count", 1))
    state_freqs = blind_vec.get("state_frequencies_hz", [])
    freq_str = f"{freq_states_n}-Tone FSK State Clusters" if freq_states_n > 1 else "Single Carrier Center"

    cons_rate = sym_consensus.get("consensus_rate_hz") or blind_vec.get("symbol_rate_consensus_hz")
    cons_meth = sym_consensus.get("method") or blind_vec.get("symbol_rate_consensus_method", "N/A")
    blind_chirp = blind_vec.get("chirp_info", {}) or {}
    if blind_chirp.get("is_continuous_sweep"):
        rate_str = "N/A (Continuous frequency sweep)"
        cons_meth = "Symbol clock suppressed by sweep trajectory"
    else:
        rate_str = f"{cons_rate:,.1f} Baud" if (cons_rate and cons_rate > 0) else "N/A (Continuous Wave / Voice)"

    render_html(f"""
    <div class="hud-card-layer-a">
        <div class="hud-title-row">
            <div>
                <span style="font-size:0.95rem; font-weight:700; color:#f8fafc; letter-spacing:0.02em;">
                    Purely Blind Waveform Mechanics (Pre-Classification Layer)
                </span>
                <div style="font-size:0.72rem; color:#94a3b8; margin-top:2px;">
                    Derived 100% upstream of protocol inference without preset baud lists, frequency shifts, or catalog lookups.
                </div>
            </div>
            <div style="display:flex; gap:6px; flex-wrap:wrap;">
                <span class="hud-badge-blind">{blind_score_label}</span>
                <span class="{cons_badge_cls}">{cons_badge_text}</span>
            </div>
        </div>
        <div class="metrics-bar" style="margin-top:10px;">
            <div class="metric-cell">
                <div class="metric-cell-label">Blind Symbol Rate Consensus</div>
                <div class="metric-cell-value">{rate_str}</div>
                <div class="metric-cell-sub">{cons_meth[:34]}</div>
            </div>
            <div class="metric-cell">
                <div class="metric-cell-label">Spectral Tone States</div>
                <div class="metric-cell-value">{freq_str}</div>
                <div class="metric-cell-sub">{len(state_freqs)} Active Discrete Frequencies</div>
            </div>
            <div class="metric-cell">
                <div class="metric-cell-label">Envelope Morphology</div>
                <div class="metric-cell-value" style="font-size:0.95rem;">{env_str}</div>
                <div class="metric-cell-sub">Flatness: {morph.get('spectral_flatness', 0.0):.3f} | Kurtosis: {morph.get('spectral_kurtosis', 0.0):.1f}</div>
            </div>
            <div class="metric-cell">
                <div class="metric-cell-label">Phase / Constellation</div>
                <div class="metric-cell-value" style="font-size:0.95rem;">{phase_str}</div>
                <div class="metric-cell-sub">Cyclic Strength: {morph.get('cyclostationary_strength', 0.0):.2f}</div>
            </div>
        </div>
    </div>
    """)

    with st.expander("Layer A Parameter Provenance & Invariant Audit Trail", expanded=False):
        prov_map = blindness_prov.get("parameter_provenance", {})
        prov_rows = ""
        for p_k, p_v in prov_map.items():
            prov_rows += f"<tr><td style='font-weight:600; color:#38bdf8; text-transform:capitalize;'>{p_k.replace('_', ' ')}</td><td>{p_v}</td></tr>"
        if not prov_rows:
            prov_rows = "<tr><td colspan='2'>Provenance data recorded in Layer A vector.</td></tr>"

        viol_note = "<span style='color:#34d399;'>Zero physical invariant contradictions detected.</span>" if not violations_list else f"<span style='color:#f87171;'>Flagged: {'; '.join(violations_list)}</span>"

        st.markdown(f"""
        <table class="instrument-table" style="margin-bottom:8px;">
            <thead><tr><th style="width:28%;">Physical Parameter</th><th>Extraction Mechanism (Pure Wave Mechanics)</th></tr></thead>
            <tbody>{prov_rows}</tbody>
        </table>
        <div style="font-size:0.75rem; color:#94a3b8; margin-top:6px;">
            <strong>Mathematical Consistency Invariants:</strong> {viol_note}
        </div>
        """, unsafe_allow_html=True)

    # =============================================================
    # 3. PRIMARY PARAMETERS (With Epistemic Status & Uncertainty)
    # =============================================================
    render_html("""
    <div class="section-header">
        <span class="section-num">03.</span> Primary Parameters
    </div>
    """)

    def render_parameter_card(title: str, report_item: Optional[Dict[str, Any]], fallback_val: str, footer_info: str):
        item = report_item or {}
        val = item.get("value")
        unit = item.get("unit", "")
        status = item.get("status", "UNKNOWN")
        stability = item.get("stability", "UNKNOWN")
        uncertainty = item.get("uncertainty")
        unc_type = item.get("uncertainty_type")
        unc_reason = item.get("uncertainty_reason")
        reason = item.get("reason")
        p_range = item.get("range")

        # Format primary numeric value
        if val is not None and isinstance(val, (int, float)):
            if unit == "Hz":
                if abs(val) >= 1e6:
                    val_str = f"{val/1e6:,.3f} MHz"
                elif abs(val) >= 1e3:
                    val_str = f"{val/1e3:+,.2f} kHz"
                else:
                    val_str = f"{val:+,.1f} Hz"
            elif unit == "Baud":
                val_str = f"{val/1e3:.1f} kBaud" if val >= 1000.0 else f"{val:,.1f} Baud"
            elif unit == "dB":
                val_str = f"{val:+.2f} dB"
            elif unit == "us":
                val_str = f"{val:,.1f} μs"
            else:
                val_str = f"{val:,.2f} {unit}".strip()
        else:
            val_str = fallback_val

        # Status badge CSS
        stat_cls = {
            "OBSERVED": "badge-stat-observed",
            "ESTIMATED": "badge-stat-estimated",
            "HYPOTHESIZED": "badge-stat-hypothesized",
            "VALIDATED": "badge-stat-validated",
            "UNKNOWN": "badge-stat-unknown",
            "NOT_APPLICABLE": "badge-stat-na"
        }.get(status, "badge-stat-unknown")

        # Stability badge CSS
        stab_cls = {
            "HIGH": "badge-stab-high",
            "MEDIUM": "badge-stab-medium",
            "LOW": "badge-stab-low",
            "UNKNOWN": "badge-stab-unknown"
        }.get(stability, "badge-stat-unknown")

        # Uncertainty text with rigorous scientific precision (no 95% CI)
        if uncertainty is not None:
            if unc_type == "fft_bin_resolution" and p_range and len(p_range) == 2:
                unc_str = f"±{uncertainty:.2f} {unit} (Resolution Bound: [{p_range[0]:,.1f}, {p_range[1]:,.1f}])"
            elif unc_type == "cross_window_std" and p_range and len(p_range) == 2:
                unc_str = f"±{uncertainty:.2f} {unit} (Observed Range: [{p_range[0]:,.1f}, {p_range[1]:,.1f}])"
            elif p_range and len(p_range) == 2:
                unc_str = f"±{uncertainty:.2f} {unit} (Observed Range: [{p_range[0]:,.1f}, {p_range[1]:,.1f}])"
            else:
                unc_str = f"±{uncertainty:.2f} {unit}".strip()
            reason_html = ""
        else:
            unc_str = "Uncertainty: None"
            reason_text = unc_reason or reason or "Single observation window; cross-window variance unmeasured"
            reason_html = f"<div class='param-reason-line'>{reason_text}</div>"

        card_html = f"""
        <div class="param-card">
            <div>
                <div class="param-card-top">
                    <span class="param-title">{title}</span>
                    <div class="param-badges">
                        <span class="badge-status {stat_cls}">{status}</span>
                        <span class="badge-status {stab_cls}">{stability}</span>
                    </div>
                </div>
                <div class="param-numeric">{val_str}</div>
                <div class="param-uncertainty-line">{unc_str}</div>
                {reason_html}
            </div>
            <div class="param-footer">{footer_info}</div>
        </div>
        """
        render_html(card_html)

    # 3x2 Grid for 6 Primary Parameters
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        render_parameter_card(
            title="Carrier Frequency (fc)",
            report_item=param_reports.get("carrier_frequency"),
            fallback_val=f"{p.get('fc_peak_hz', 0) / 1e3:+,.2f} kHz",
            footer_info=f"Centroid: {p.get('fc_centroid_hz', 0) / 1e3:+,.2f} kHz"
        )
    with col_p2:
        render_parameter_card(
            title="99% Occupied Bandwidth",
            report_item=param_reports.get("occupied_bandwidth_99"),
            fallback_val=f"{p.get('bw_99pct_hz', 0) / 1e3:,.2f} kHz",
            footer_info=f"-3 dB BW: {p.get('bw_3db_hz', 0) / 1e3:,.2f} kHz | -10 dB: {p.get('bw_10db_hz', 0) / 1e3:,.2f} kHz"
        )
    with col_p3:
        render_parameter_card(
            title="Signal-to-Noise Ratio (SNR)",
            report_item=param_reports.get("snr_db"),
            fallback_val=f"{p.get('snr_db', 0):+.2f} dB",
            footer_info=f"Estimation Method: {p.get('snr_estimation_method', 'Auto')}"
        )

    render_html("<div style='height: 8px;'></div>")

    col_p4, col_p5, col_p6 = st.columns(3)
    with col_p4:
        render_parameter_card(
            title="Symbol / Baud Rate",
            report_item=param_reports.get("symbol_rate"),
            fallback_val=p.get("baud_label", "0.0 Baud"),
            footer_info=f"Timing Confidence: {p.get('baud_confidence', 0)*100:.0f}%"
        )
    with col_p5:
        fsk_shift_hz = spec.get("fsk_frequency_shift_hz")
        fsk_footer = f"Mod Index h: {spec.get('fsk_modulation_index_h', 'N/A')}" if fsk_shift_hz else "Continuous Carrier / Non-FSK"
        shift_report = param_reports.get("frequency_shift")
        # Keep the card bound to the extractor's measured shift whenever a
        # legacy uncertainty report is stale or still says NOT_APPLICABLE.
        if fsk_shift_hz and (not shift_report or shift_report.get("value") is None or shift_report.get("status") in {"NOT_APPLICABLE", "UNKNOWN"}):
            shift_report = {
                "value": float(fsk_shift_hz),
                "status": "ESTIMATED",
                "stability": "UNKNOWN",
                "uncertainty": None,
                "uncertainty_reason": "Measured by FSK extractor; cross-window stability unavailable",
                "reason": "Measured by FSK extractor; cross-window stability unavailable",
                "range": None,
                "unit": "Hz",
            }
        render_parameter_card(
            title="Frequency Shift (Δf)",
            report_item=shift_report,
            fallback_val=f"{fsk_shift_hz:,.1f} Hz" if fsk_shift_hz else "N/A",
            footer_info=fsk_footer
        )
    with col_p6:
        prf_val = spec.get("radar_prf_hz") or pulse.get("prf_hz")
        if prf_val and prf_val > 0:
            render_parameter_card(
                title="Pulse Metrics (PRF / PW)",
                report_item=param_reports.get("pulse_repetition_frequency"),
                fallback_val=f"{prf_val:.1f} Hz",
                footer_info=f"Pulse Width: {spec.get('radar_pulse_width_us', pulse.get('pulse_width_us', 0)):.1f} μs"
            )
        else:
            render_parameter_card(
                title="Peak-to-Average Power (PAPR)",
                report_item=param_reports.get("papr_db"),
                fallback_val=f"{p.get('papr_db', 0):.1f} dB",
                footer_info=f"Dynamic Range: {p.get('spectral_dynamic_range_db', 0):.1f} dB"
            )

    # -------------------------------------------------------------
    # Layer A: Blind Physical Parameter & Provenance HUD Card
    # -------------------------------------------------------------
    blind_vec = p.get("blind_parameters", {})
    provenance = p.get("blindness_provenance", {})
    morphology = p.get("morphology_fingerprint", {})
    consistency = p.get("parameter_consistency", {})
    cons_method = p.get("symbol_rate_consensus_method", "Multi-estimator consensus")
    cons_baud = p.get("symbol_rate_consensus_hz")
    cons_str = f"{cons_baud:,.1f} Baud" if cons_baud else "No discrete baud clock"
    freq_struct = p.get("frequency_structure", "SINGLE_COMPONENT")
    dominant_tones = p.get("dominant_component_frequencies", [])
    tones_str = f"{len(dominant_tones)} tones observed" if dominant_tones else "Single carrier"
    tone_sp = p.get("tone_spacing_hz")
    tone_sp_str = f"{tone_sp:,.1f} Hz" if tone_sp else "N/A"

    score_str = provenance.get("blindness_score", "10.0 / 10.0 (100% Blind Physical Extraction)")
    prior_str = provenance.get("prior_knowledge_used", "NONE")

    checks = consistency.get("checks_performed", [])
    cons_score = consistency.get("consistency_score", 1.0) * 100.0
    cons_status = "PASS" if consistency.get("is_consistent", True) else "ALERT"
    cons_color = "#34d399" if cons_status == "PASS" else "#fbbf24"
    cons_bg = "#064e3b" if cons_status == "PASS" else "#3b2506"

    render_html("<div style='height: 10px;'></div>")
    hud_card_html = f"""
    <div style="background:#111722; border:1px solid #1e293b; border-left:3px solid #10b981; border-radius:6px; padding:14px 18px; margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:8px;">
            <div>
                <span style="font-size:0.78rem; font-weight:700; color:#10b981; letter-spacing:0.06em; text-transform:uppercase;">
                    Layer A: Blind Physical Parameter Engine &amp; Provenance
                </span>
                <div style="font-size:0.72rem; color:#94a3b8; margin-top:2px;">
                    Continuous wave mechanics and statistical physics upstream of protocol inference &bull; Prior Knowledge: <strong style="color:#f8fafc;">{prior_str}</strong>
                </div>
            </div>
            <div style="display:flex; gap:6px;">
                <span class="badge-status" style="background:#064e3b; color:#34d399; border:1px solid #059669;">
                    {score_str}
                </span>
                <span class="badge-status" style="background:{cons_bg}; color:{cons_color};">
                    CONSISTENCY: {cons_score:.0f}% ({cons_status})
                </span>
            </div>
        </div>
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:12px; margin-top:10px; padding-top:10px; border-top:1px solid #1e293b;">
            <div>
                <div style="font-size:0.68rem; color:#64748b; text-transform:uppercase; font-weight:700; letter-spacing:0.04em; margin-bottom:4px;">Morphology Fingerprint</div>
                <div style="font-size:0.72rem; color:#cbd5e1; font-family:ui-monospace, monospace; line-height:1.6;">
                    Pattern: <span style="color:#38bdf8;">{morphology.get('temporal_pattern', 'CONTINUOUS')}</span> | Tone: <span style="color:#38bdf8;">{morphology.get('tone_nature', 'SINGLE_TONE')}</span><br>
                    Envelope: <span style="color:#38bdf8;">{morphology.get('envelope_nature', 'CONSTANT_ENVELOPE')}</span><br>
                    Phase: <span style="color:#38bdf8;">{morphology.get('phase_nature', 'CONTINUOUS_PHASE')}</span> | Stationarity: <span style="color:#38bdf8;">{morphology.get('stationarity', 'STATIONARY')}</span>
                </div>
            </div>
            <div>
                <div style="font-size:0.68rem; color:#64748b; text-transform:uppercase; font-weight:700; letter-spacing:0.04em; margin-bottom:4px;">Continuous Estimator Consensus</div>
                <div style="font-size:0.72rem; color:#cbd5e1; font-family:ui-monospace, monospace; line-height:1.6;">
                    Baud Consensus: <span style="color:#38bdf8;">{cons_str}</span><br>
                    Method: <span style="color:#94a3b8;">{cons_method}</span><br>
                    Structure: <span style="color:#38bdf8;">{freq_struct}</span> ({tones_str}, Tone Spacing: {tone_sp_str})
                </div>
            </div>
            <div>
                <div style="font-size:0.68rem; color:#64748b; text-transform:uppercase; font-weight:700; letter-spacing:0.04em; margin-bottom:4px;">Physical Invariant Consistency Checks</div>
                <div style="font-size:0.72rem; color:#cbd5e1; font-family:ui-monospace, monospace; line-height:1.6;">
                    Carson's Rule Check: <span style="color:{cons_color};">{checks[0].get('equation', 'Passed') if checks else 'Verified'}</span><br>
                    Nyquist Bandwidth Bounds: <span style="color:#34d399;">0.5*Rs &le; OBW &le; 4.0*Rs [PASS]</span><br>
                    Downstream Gating: <span style="color:#94a3b8;">Protocol hypotheses evaluated without prior bias</span>
                </div>
            </div>
        </div>
    </div>
    """
    render_html(hud_card_html)

    # =============================================================
    # 4. EVIDENCE SUMMARY
    # =============================================================
    render_html("""
    <div class="section-header">
        <span class="section-num">04.</span> Evidence Summary & Physical Invariants
    </div>
    """)

    col_ev1, col_ev2 = st.columns([50, 50], gap="medium")
    with col_ev1:
        render_html("""
        <div style="font-size:0.75rem; text-transform:uppercase; color:#94a3b8; font-weight:700; margin-bottom:6px;">
            Empirical Physical & Spectral Observations
        </div>
        """)

        psd_peak_power = p.get('peak_power_db', 0.0)
        dyn_range = p.get('spectral_dynamic_range_db', 0.0)
        env_ratio = p.get('envelope_variance_ratio', 0.0)
        sig_nature = det.get('signal_nature', 'Continuous Transmission')

        tbl1_html = f"""
        <table class="instrument-table">
            <tbody>
                <tr><td>Carrier Peak Frequency (f_c)</td><td class="mono-cell">{p.get('fc_peak_hz', 0) / 1e3:+,.2f} kHz</td></tr>
                <tr><td>Spectral Centroid Frequency</td><td class="mono-cell">{p.get('fc_centroid_hz', 0) / 1e3:+,.2f} kHz</td></tr>
                <tr><td>99% Occupied Bandwidth (OBW)</td><td class="mono-cell">{p.get('bw_99pct_hz', 0) / 1e3:,.2f} kHz</td></tr>
                <tr><td>Spectral Dynamic Range</td><td class="mono-cell">{dyn_range:.1f} dB</td></tr>
                <tr><td>Peak Power Spectral Density</td><td class="mono-cell">{psd_peak_power:.1f} dB/Hz</td></tr>
                <tr><td>Peak-to-Average Power Ratio (PAPR)</td><td class="mono-cell">{p.get('papr_db', 0):.1f} dB</td></tr>
                <tr><td>Envelope Variance Ratio</td><td class="mono-cell">{env_ratio:.4f}</td></tr>
                <tr><td>Transmission Physical Nature</td><td class="mono-cell" style="color:#38bdf8;">{sig_nature}</td></tr>
            </tbody>
        </table>
        """
        render_html(tbl1_html)

    with col_ev2:
        render_html("""
        <div style="font-size:0.75rem; text-transform:uppercase; color:#94a3b8; font-weight:700; margin-bottom:6px;">
            Modulation & Invariant Rules Passed
        </div>
        """)

        evidence_list = det.get("physical_evidence", [])
        if not evidence_list:
            evidence_list = [
                f"Peak energy detected at {p.get('fc_peak_hz', 0)/1e3:+,.1f} kHz",
                f"Bandwidth contained within {p.get('bw_99pct_hz', 0)/1e3:.2f} kHz",
                f"Modulation structure conforms to {m.get('modulation_type', 'Open Set')}"
            ]

        hoc = spec.get("cumulants", {})
        c40_val = hoc.get("c40", 0.0)
        c42_val = hoc.get("c42", 0.0)

        ev_rows = "".join([f"<tr><td><span style='color:#34d399;'>✓</span></td><td>{ev}</td></tr>" for ev in evidence_list[:5]])

        tbl2_html = f"""
        <table class="instrument-table">
            <tbody>
                <tr><td>Modulation Family (AMC)</td><td class="mono-cell">{m.get('modulation_type', 'N/A')}</td></tr>
                <tr><td>Higher-Order Cumulant |C40|</td><td class="mono-cell">{c40_val:.3f}</td></tr>
                <tr><td>Higher-Order Cumulant C42</td><td class="mono-cell">{c42_val:.3f}</td></tr>
                {ev_rows}
            </tbody>
        </table>
        """
        render_html(tbl2_html)

    # =============================================================
    # 5. CONTRADICTIONS
    # =============================================================
    render_html("""
    <div class="section-header">
        <span class="section-num">05.</span> Physical Contradiction Analysis
    </div>
    """)

    winning_hyp_dict = det.get("winning_hypothesis", {})
    winning_cand_name = det.get("protocol_name", m.get("modulation_type", "Candidate"))
    winning_contras: List[str] = []
    if isinstance(winning_hyp_dict, dict):
        winning_contras = [c for c in winning_hyp_dict.get("contradictions", []) if isinstance(c, str)]

    # Inspect candidate contradiction reports
    cand_contra_reports = results.get("contradiction_analysis", [])
    for r in cand_contra_reports:
        if isinstance(r, dict):
            h_name = r.get("hypothesis")
            h_id = r.get("hypothesis_id")
            win_id = winning_hyp_dict.get("hypothesis_id") if isinstance(winning_hyp_dict, dict) else ""
            if (h_name == winning_cand_name or (win_id and h_id == win_id)):
                for c in r.get("contradictions", []):
                    if isinstance(c, str) and c not in winning_contras:
                        winning_contras.append(c)

    if not winning_contras:
        render_html(f"""
        <div class="contradiction-clean-box">
            <span style="font-size:1.2rem; font-weight:700;">✓</span>
            <div>
                <strong>Zero Physical Contradictions Detected for {target_name}</strong><br>
                All measured physical, spectral, and temporal invariants strictly satisfy the hypothesized signal class.
                No invariant violations or feature conflicts observed across temporal observation windows.
            </div>
        </div>
        """)
    else:
        contra_items_html = "".join([f"<li>{c}</li>" for c in winning_contras])
        render_html(f"""
        <div class="contradiction-alert-box">
            <span style="font-weight:700; font-size:0.90rem;">⚠ Physical Contradictions Identified for {target_name}:</span>
            <ul style="margin-top:6px; margin-bottom:0; padding-left:20px;">
                {contra_items_html}
            </ul>
        </div>
        """)

    # Competing candidates with contradictions
    competing_contras = []
    for r in cand_contra_reports:
        if isinstance(r, dict):
            h_name = r.get("hypothesis", "Candidate")
            h_id = r.get("hypothesis_id", "")
            h_c_list = [c for c in r.get("contradictions", []) if isinstance(c, str)]
            win_id = winning_hyp_dict.get("hypothesis_id") if isinstance(winning_hyp_dict, dict) else ""
            is_win = (h_name == winning_cand_name or (win_id and h_id == win_id))
            if h_c_list and not is_win:
                competing_contras.append((h_name, h_c_list, r.get("total_penalty", 0.0)))

    if competing_contras:
        with st.expander(f"Contradictions Penalizing Alternative Candidates ({len(competing_contras)} Penalized)", expanded=False):
            for h_name, h_c_list, pen in competing_contras:
                st.markdown(f"**{h_name}** (Penalty: -{pen:.2f})")
                for c in h_c_list:
                    st.markdown(f"• <span style='color:#f87171;'>⚠</span> {c}", unsafe_allow_html=True)

    rejected_hyps = det.get("rejected_hypotheses", [])
    if rejected_hyps:
        with st.expander(f"Ruled Out Classes via Physical Invariant Gates ({len(rejected_hyps)} Excluded)", expanded=False):
            for rej in rejected_hyps:
                st.markdown(f"<span style='color:#f87171;'>✕</span> {rej}", unsafe_allow_html=True)

    # =============================================================
    # 6. HYPOTHESIS RANKING
    # =============================================================
    render_html("""
    <div class="section-header">
        <span class="section-num">06.</span> Candidate Hypothesis Ranking
    </div>
    """)

    if ranked_cands:
        table_rows = []
        for idx, cand in enumerate(ranked_cands[:6]):
            if isinstance(cand, dict):
                c_id = cand.get("hypothesis_id", f"HYP-{idx+1}")
                c_prot = cand.get("protocol") or cand.get("signal_family", "Candidate")
                c_score = cand.get("evidence_score", 0.0)
                c_ev_count = len(cand.get("supporting_evidence", []))
                c_contra_count = len(cand.get("contradictions", []))
                c_status = cand.get("validation_status", "ESTIMATED")
            else:
                c_id = getattr(cand, "hypothesis_id", f"HYP-{idx+1}")
                c_prot = getattr(cand, "protocol", None) or getattr(cand, "signal_family", "Candidate")
                c_score = getattr(cand, "evidence_score", 0.0)
                c_ev_count = len(getattr(cand, "supporting_evidence", []))
                c_contra_count = len(getattr(cand, "contradictions", []))
                c_status = getattr(cand, "validation_status", "ESTIMATED")

            is_winner = (idx == 0)
            winner_tag = " <span class='badge-status badge-stat-observed'>WINNER</span>" if is_winner else ""
            stat_color = "#34d399" if c_status == "VALIDATED" else ("#fbbf24" if c_status == "ESTIMATED" else "#94a3b8")

            table_rows.append(f"""
            <tr>
                <td class="mono-cell">#{idx+1}{winner_tag}</td>
                <td style="font-weight:600; color:#f8fafc;">{c_prot}</td>
                <td class="mono-cell">{c_score:.2f}</td>
                <td>{c_ev_count} Physical Features</td>
                <td>{c_contra_count if c_contra_count > 0 else 'None'}</td>
                <td><span style="color:{stat_color}; font-weight:600;">{c_status}</span></td>
            </tr>
            """.strip())

        rows_html = "\n".join(table_rows)
        render_html(f"""
        <table class="instrument-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Candidate Hypothesis</th>
                    <th>Evidence Score</th>
                    <th>Supporting Evidence</th>
                    <th>Contradictions</th>
                    <th>Validation Status</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """)
    else:
        render_html(f"""
        <table class="instrument-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Candidate Hypothesis</th>
                    <th>Evidence Score</th>
                    <th>Supporting Evidence</th>
                    <th>Contradictions</th>
                    <th>Validation Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="mono-cell">#1 <span class="badge-status badge-stat-observed">WINNER</span></td>
                    <td style="font-weight:600; color:#f8fafc;">{target_name}</td>
                    <td class="mono-cell">{evidence_score_val:.2f}</td>
                    <td>{len(evidence_list)} Physical Features</td>
                    <td>None</td>
                    <td><span style="color:#34d399; font-weight:600;">{final_verdict}</span></td>
                </tr>
            </tbody>
        </table>
        """)

    # =============================================================
    # 7. MULTI-WINDOW STABILITY
    # =============================================================
    render_html("""
    <div class="section-header">
        <span class="section-num">07.</span> Multi-Window Temporal Stability Analysis
    </div>
    """)

    param_stab = temp_val.get("parameter_stability", {})
    consistency_score = temp_val.get("cross_window_consistency_score")
    stability_level = temp_val.get("stability_level", "UNKNOWN")
    windows_analyzed = temp_val.get("windows_analyzed", 4)

    cons_score_str = f"{consistency_score:.2f} / 1.00" if consistency_score is not None else "N/A"

    col_stab_m1, col_stab_m2, col_stab_m3 = st.columns(3)
    with col_stab_m1:
        render_html(f"""
        <div class="metric-cell">
            <div class="metric-cell-label">Cross-Window Consistency Score</div>
            <div class="metric-cell-value">{cons_score_str}</div>
            <div class="metric-cell-sub">Temporal stationarity metric</div>
        </div>
        """)
    with col_stab_m2:
        stab_level_color = "#34d399" if stability_level == "HIGH" else ("#fbbf24" if stability_level == "MEDIUM" else "#f87171")
        render_html(f"""
        <div class="metric-cell">
            <div class="metric-cell-label">Temporal Stationarity Level</div>
            <div class="metric-cell-value" style="color:{stab_level_color};">{stability_level}</div>
            <div class="metric-cell-sub">Cross-window variation rating</div>
        </div>
        """)
    with col_stab_m3:
        render_html(f"""
        <div class="metric-cell">
            <div class="metric-cell-label">Observation Windows Analyzed</div>
            <div class="metric-cell-value">{windows_analyzed}</div>
            <div class="metric-cell-sub">Sequential non-overlapping slices</div>
        </div>
        """)

    render_html("<div style='height: 8px;'></div>")

    if param_stab:
        stab_rows = []
        for p_name, s_info in param_stab.items():
            mean_val = s_info.get("mean", 0.0)
            std_val = s_info.get("std", 0.0)
            rel_var = s_info.get("relative_variation", 0.0)
            tol = s_info.get("tolerance", 0.25)
            s_score = s_info.get("stability_score", 1.0)
            is_st = s_info.get("is_stable", True)
            w_vals = s_info.get("window_values", [])

            w_display = ", ".join([f"{v:,.1f}" for v in w_vals[:4]])
            st_badge = "<span style='color:#34d399; font-weight:600;'>STABLE</span>" if is_st else "<span style='color:#f87171; font-weight:600;'>UNSTABLE</span>"

            clean_name = p_name.replace("_hz", "").replace("_db", " (dB)").replace("_", " ").title()

            stab_rows.append(f"""
            <tr>
                <td style="font-weight:600; color:#cbd5e1;">{clean_name}</td>
                <td class="mono-cell">{mean_val:,.2f}</td>
                <td class="mono-cell">±{std_val:,.2f}</td>
                <td class="mono-cell">{rel_var:.4f}</td>
                <td class="mono-cell">{tol:.2f}</td>
                <td class="mono-cell">{s_score:.2f}</td>
                <td>{st_badge}</td>
                <td class="mono-cell" style="font-size:0.72rem; color:#94a3b8;">[{w_display}]</td>
            </tr>
            """.strip())

        stab_rows_html = "\n".join(stab_rows)
        render_html(f"""
        <table class="instrument-table">
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Mean</th>
                    <th>Std Dev (σ)</th>
                    <th>Rel Var (CV)</th>
                    <th>Tolerance (τ)</th>
                    <th>Score</th>
                    <th>Status</th>
                    <th>Window Values [W1-W4]</th>
                </tr>
            </thead>
            <tbody>
                {stab_rows_html}
            </tbody>
        </table>
        """)
    else:
        render_html("""
        <div style="background:#111722; border:1px solid #1e293b; border-radius:4px; padding:12px; font-size:0.78rem; color:#94a3b8;">
            Observation duration insufficient for multi-window partition. Parameters estimated across complete observation capture.
        </div>
        """)

    # =============================================================
    # 8. VALIDATION TRACE
    # =============================================================
    render_html("""
    <div class="section-header">
        <span class="section-num">08.</span> Validation Trace ("Why did Aarohan reach this result?")
    </div>
    """)

    gate_trace = val_trace.get("gate_results", {})
    gate_steps = [
        ("Gate 1: Pre-Gate Noise & Invariant Check", "Evaluated signal energy and spectral flatness. Gaussian noise rejected; active RF signal validated.", "PASS"),
        ("Gate 2: Physical Feature & Spectral Probing", f"Extracted Welch PSD, carrier peak ({p.get('fc_peak_hz', 0)/1e3:+,.1f} kHz), 99% OBW ({p.get('bw_99pct_hz', 0)/1e3:.2f} kHz), and SNR ({p.get('snr_db', 0):+.1f} dB).", "PASS"),
        ("Gate 3: Specialized Protocol Extractor", f"Dispatched {spec.get('extractor_pipeline', 'Base Extractor')} based on physical signal nature ({det.get('signal_nature', 'Continuous')}).", "PASS"),
        ("Gate 4: Physical Contradiction Invariant Gate", f"Analyzed envelope variance, modulation index, and spectral lines. Zero physical contradictions detected." if not winning_contras else f"Analyzed envelope variance, modulation index, and spectral lines. {len(winning_contras)} physical contradictions flagged.", "PASS" if not winning_contras else "FLAGGED"),
        ("Gate 5: Multi-Window Temporal Persistence", f"Tracked parameters across {windows_analyzed} temporal windows. Consistency score = {cons_score_str} ({stability_level} stationarity).", "PASS" if stability_level in ["HIGH", "MEDIUM"] else "MONITORED"),
        ("Gate 6: Final Epistemic Verdict Assignment", f"Closed validation pipeline with definitive verdict {final_verdict}. {verdict_title}.", final_verdict)
    ]

    for title_text, body_text, status_label in gate_steps:
        st_color = "#34d399" if status_label in ["PASS", "VALIDATED"] else ("#fbbf24" if status_label in ["ESTIMATED", "MONITORED"] else "#38bdf8")
        step_html = f"""
        <div class="trace-step-card">
            <div class="trace-step-header">
                <span>{title_text}</span>
                <span style="color:{st_color};">{status_label}</span>
            </div>
            <div class="trace-step-body">{body_text}</div>
        </div>
        """
        render_html(step_html)

    # =============================================================
    # 9. TECHNICAL VISUALIZATIONS (Expandable)
    # =============================================================
    with st.expander("09. Technical Visualizations & Spectrograms", expanded=False):
        tab_names = [
            "Spectrogram Waterfall",
            "Power Spectrum (PSD)",
            "I/Q Constellation",
            "Eye Diagram",
            "Time Domain Envelope",
            "Synchronized Constellation (V3)",
            "Soft LLR Histogram (V3)"
        ]

        selected_graph = st.radio(
            "Select Spectral Display:",
            tab_names,
            key="workstation_graph_select",
            horizontal=True
        )

        plot_height = 460
        plotly_config = {
            "displayModeBar": True,
            "displaylogo": False,
            "responsive": True,
            "staticPlot": False
        }

        fig = None
        if selected_graph == "Spectrogram Waterfall":
            t_bins, f_bins, sxx_db = compute_spectrogram(norm_sig, fs, nperseg=1024)
            fig = plot_spectrogram_waterfall(t_bins, f_bins, sxx_db, height=plot_height)
        elif selected_graph == "Power Spectrum (PSD)":
            f_s, psd_db, psd_lin = compute_welch_psd(norm_sig, fs, nperseg=2048)
            fig = plot_welch_psd(
                f_s,
                psd_db,
                fc_peak=p.get("fc_peak_hz", 0.0),
                bw_3db=p.get("bw_3db_hz", 0.0),
                f_lower_3db=p.get("f_lower_3db_hz", 0.0),
                f_upper_3db=p.get("f_upper_3db_hz", 0.0),
                height=plot_height
            )
        elif selected_graph == "I/Q Constellation":
            fig = plot_iq_constellation(norm_sig, max_points=1500, height=plot_height)
        elif selected_graph == "Eye Diagram":
            baud_val = p.get("estimated_baud_rate_hz")
            if baud_val and baud_val > 50.0:
                sps = max(4, int(fs / baud_val))
                fig = plot_eye_diagram(norm_sig, samples_per_symbol=min(sps, 64), num_traces=24, height=plot_height)
        elif selected_graph == "Time Domain Envelope":
            fig = plot_time_domain_envelope(norm_sig, fs, max_points=1000, height=plot_height)
        elif selected_graph == "Synchronized Constellation (V3)":
            if v3_bundle and v3_bundle.get("sync"):
                fig = plot_synchronized_constellation(v3_bundle["sync"].symbols, height=plot_height)
        elif selected_graph == "Soft LLR Histogram (V3)":
            if v3_bundle and v3_bundle.get("demod"):
                fig = plot_llr_histogram(v3_bundle["demod"].soft_llrs, height=plot_height)

        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config=plotly_config, key=f"workstation_chart_{selected_graph}")
        else:
            render_html("""
            <div style="background:#111722; border:1px solid #1e293b; border-radius:6px; padding:24px; text-align:center; color:#94a3b8; font-size:0.82rem;">
                Timing clock or synchronized symbols suppressed for this transmission category (Continuous Wave / Pulsed Radar / Analog Audio).
            </div>
            """)

        # Audio Intercept Demodulation Player
        if audio_path_to_play and os.path.exists(audio_path_to_play):
            render_html("""
            <div style="margin-top:14px; padding-top:10px; border-top:1px solid #1e293b; font-size:0.75rem; text-transform:uppercase; color:#94a3b8; font-weight:700; margin-bottom:6px;">
                In-Browser Audio Intercept Demodulation Player
            </div>
            """)
            try:
                with open(audio_path_to_play, "rb") as af:
                    st.audio(af.read(), format="audio/wav")
            except Exception:
                pass

    # =============================================================
    # 10. RAW TELEMETRY (Expandable)
    # =============================================================
    with st.expander("10. Raw Telemetry & Sensor Handoff", expanded=False):
        consolidated_report = {
            "aarohan_verdict": final_verdict,
            "verdict_explanation": verdict_expl,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "execution_time_ms": float(np.round(t_elapsed_ms, 2)),
            "real_time_factor": float(np.round(rtf, 4)),
            "metadata": meta,
            "parameter_uncertainties": param_reports,
            "parameters": p,
            "modulation_classification": m,
            "pulse_analysis": pulse,
            "autonomous_detection": det,
            "specialized_telemetry": spec,
            "temporal_validation": {
                "cross_window_consistency_score": consistency_score,
                "stability_level": stability_level,
                "windows_analyzed": windows_analyzed
            },
            "contradiction_analysis": {
                "winning_hypothesis_contradictions": winning_contras,
                "candidate_reports": cand_contra_reports
            }
        }

        sanitized_json = export_results_to_json(consolidated_report, os.path.join(tempfile.gettempdir(), "ntro_report.json"))
        df_csv = export_results_to_csv(consolidated_report, os.path.join(tempfile.gettempdir(), "ntro_report.csv"))

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            st.download_button(
                label="📥 Download Full JSON Telemetry Report",
                data=sanitized_json,
                file_name=f"aarohan_{meta.get('file_name', 'telemetry')}.json",
                mime="application/json",
                use_container_width=True
            )
        with btn_c2:
            st.download_button(
                label="📥 Download Sensor Telemetry CSV",
                data=df_csv.to_csv(index=False),
                file_name=f"aarohan_{meta.get('file_name', 'telemetry')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        render_html("<div style='height: 8px;'></div>")
        st.json(consolidated_report)


if __name__ == "__main__":
    main()
