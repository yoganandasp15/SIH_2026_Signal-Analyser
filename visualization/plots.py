"""
Plotly Interactive Visualization Engine for RF Signals
======================================================
Professional, instrument-grade interactive plots for Streamlit and web dashboards:
- Calibrated PSD Spectrum with Peak & -3dB Bandwidth Overlays
- 2D Time-Frequency Waterfall Spectrogram (Perceptually Linear Colormap)
- In-Phase vs Quadrature Constellation Diagram with Unit Circle
- Multi-trace Eye Diagram for Timing Jitter & ISI Analysis
- Time-Domain Baseband (I/Q) & Magnitude Envelope Oscilloscope
"""

from typing import Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Instrument UI theme constants
PLOT_BG = "#0b0f17"
PAPER_BG = "rgba(0,0,0,0)"
GRID_COLOR = "#1e293b"
ZERO_LINE_COLOR = "#334155"
FONT_COLOR = "#94a3b8"
FONT_FAMILY = "Inter, -apple-system, system-ui, sans-serif"


def plot_welch_psd(
    f: np.ndarray,
    psd_db: np.ndarray,
    fc_peak: Optional[float] = None,
    bw_3db: Optional[float] = None,
    f_lower_3db: Optional[float] = None,
    f_upper_3db: Optional[float] = None,
    height: int = 440
) -> go.Figure:
    """
    Creates an instrument-grade Power Spectral Density (PSD) line plot with
    center frequency and -3 dB bandwidth annotations.
    """
    fig = go.Figure()

    if len(f) > 1024:
        step = len(f) // 1024
        f_plot = f[::step]
        psd_plot = psd_db[::step]
    else:
        f_plot = f
        psd_plot = psd_db

    max_freq = np.max(np.abs(f)) if len(f) > 0 else 1.0
    if max_freq >= 1e6:
        scale = 1e6
        unit = "MHz"
    elif max_freq >= 1e3:
        scale = 1e3
        unit = "kHz"
    else:
        scale = 1.0
        unit = "Hz"

    fig.add_trace(go.Scatter(
        x=f_plot / scale,
        y=psd_plot,
        mode='lines',
        name='Welch PSD',
        line=dict(color='#38bdf8', width=1.6),
        hovertemplate=f"Freq: %{{x:.3f}} {unit}<br>Power: %{{y:.2f}} dB/Hz<extra></extra>"
    ))

    if fc_peak is not None:
        fig.add_vline(
            x=fc_peak / scale,
            line_dash="dash",
            line_color="#f43f5e",
            annotation_text=f"fc = {fc_peak / scale:.2f} {unit}",
            annotation_position="top left",
            annotation_font_color="#f43f5e",
            annotation_font_size=11
        )

    if f_lower_3db is not None and f_upper_3db is not None:
        fig.add_vrect(
            x0=f_lower_3db / scale,
            x1=f_upper_3db / scale,
            fillcolor="#10b981",
            opacity=0.14,
            line_width=0,
            annotation_text=f"BW (-3dB) = {bw_3db / scale:.2f} {unit}" if bw_3db else "",
            annotation_position="bottom right",
            annotation_font_color="#10b981",
            annotation_font_size=11
        )

    fig.update_layout(
        title=dict(text="<b>Power Spectral Density (Welch Periodogram)</b>", font=dict(family=FONT_FAMILY, size=14, color="#e2e8f0")),
        xaxis=dict(title=f"Frequency ({unit})", showgrid=True, gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR),
        yaxis=dict(title="Power Spectral Density (dB/Hz)", showgrid=True, gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR),
        font=dict(family=FONT_FAMILY, size=11, color=FONT_COLOR),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=45, r=25, t=40, b=40),
        hovermode="x unified",
        height=height,
        transition=dict(duration=0)
    )
    return fig


def plot_spectrogram_waterfall(
    t: np.ndarray,
    f: np.ndarray,
    sxx_db: np.ndarray,
    height: int = 440
) -> go.Figure:
    """
    Creates an interactive 2D Waterfall / Spectrogram heatmap.
    """
    max_freq = np.max(np.abs(f)) if len(f) > 0 else 1.0
    if max_freq >= 1e6:
        scale = 1e6
        unit = "MHz"
    elif max_freq >= 1e3:
        scale = 1e3
        unit = "kHz"
    else:
        scale = 1.0
        unit = "Hz"

    if len(f) > 300:
        step = len(f) // 300
        f_plot = f[::step]
        sxx_plot = sxx_db[:, ::step]
    else:
        f_plot = f
        sxx_plot = sxx_db

    fig = go.Figure(data=go.Heatmap(
        z=sxx_plot,
        x=f_plot / scale,
        y=t,
        colorscale="Viridis",
        colorbar=dict(title="dB/Hz", thickness=14, len=0.9, tickfont=dict(family=FONT_FAMILY, size=10, color=FONT_COLOR)),
        hovertemplate=f"Freq: %{{x:.3f}} {unit}<br>Time: %{{y:.4f}} s<br>Power: %{{z:.1f}} dB<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="<b>Time-Frequency Spectrogram (STFT Waterfall)</b>", font=dict(family=FONT_FAMILY, size=14, color="#e2e8f0")),
        xaxis=dict(title=f"Frequency ({unit})", showgrid=False),
        yaxis=dict(title="Time (seconds)", showgrid=False),
        font=dict(family=FONT_FAMILY, size=11, color=FONT_COLOR),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=45, r=25, t=40, b=40),
        height=height,
        transition=dict(duration=0)
    )
    return fig


def plot_iq_constellation(
    signal: np.ndarray,
    max_points: int = 1500,
    height: int = 440
) -> go.Figure:
    """
    Creates a high-performance WebGL scatter plot of In-Phase (I) vs Quadrature (Q) complex baseband samples.
    """
    if len(signal) > max_points:
        step = len(signal) // max_points
        sig_sample = signal[::step]
    else:
        sig_sample = signal

    i_vals = np.real(sig_sample)
    q_vals = np.imag(sig_sample)
    mag_vals = np.abs(sig_sample)

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=i_vals,
        y=q_vals,
        mode='markers',
        marker=dict(
            size=3,
            color=mag_vals,
            colorscale='Viridis',
            opacity=0.8,
            showscale=False
        ),
        name='IQ Samples',
        hovertemplate="I: %{x:.3f}<br>Q: %{y:.3f}<extra></extra>"
    ))

    theta = np.linspace(0, 2 * np.pi, 120)
    fig.add_trace(go.Scattergl(
        x=np.cos(theta),
        y=np.sin(theta),
        mode='lines',
        line=dict(color='rgba(148, 163, 184, 0.35)', dash='dot', width=1.0),
        name='Unit Circle',
        hoverinfo='skip'
    ))

    max_lim = max(1.5, float(np.percentile(np.abs(sig_sample), 99) * 1.3))

    fig.update_layout(
        title=dict(text="<b>I/Q Baseband Constellation Diagram</b>", font=dict(family=FONT_FAMILY, size=14, color="#e2e8f0")),
        xaxis=dict(
            title="In-Phase (I)",
            range=[-max_lim, max_lim],
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridcolor=GRID_COLOR,
            zerolinecolor=ZERO_LINE_COLOR
        ),
        yaxis=dict(
            title="Quadrature (Q)",
            range=[-max_lim, max_lim],
            showgrid=True,
            gridcolor=GRID_COLOR,
            zerolinecolor=ZERO_LINE_COLOR
        ),
        font=dict(family=FONT_FAMILY, size=11, color=FONT_COLOR),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=45, r=25, t=40, b=40),
        height=height,
        showlegend=False,
        transition=dict(duration=0)
    )
    return fig


def plot_eye_diagram(
    signal: np.ndarray,
    samples_per_symbol: int = 16,
    num_traces: int = 24,
    height: int = 440
) -> go.Figure:
    """
    Creates an Eye Diagram across 2 symbol periods for timing jitter and ISI analysis.
    """
    trace_len = 2 * samples_per_symbol
    total_traces_available = len(signal) // samples_per_symbol - 2

    if total_traces_available <= 0:
        return go.Figure()

    traces_to_plot = min(num_traces, total_traces_available)
    step = max(1, total_traces_available // traces_to_plot)
    time_axis = np.linspace(-1.0, 1.0, trace_len)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("<b>In-Phase (I) Eye</b>", "<b>Quadrature (Q) Eye</b>")
    )

    for i in range(0, traces_to_plot * step, step):
        idx = i * samples_per_symbol
        if idx + trace_len <= len(signal):
            chunk = signal[idx : idx + trace_len]
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=np.real(chunk),
                    mode='lines',
                    line=dict(color='rgba(56, 189, 248, 0.40)', width=1.2),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=np.imag(chunk),
                    mode='lines',
                    line=dict(color='rgba(244, 63, 94, 0.40)', width=1.2),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=2
            )

    fig.update_layout(
        font=dict(family=FONT_FAMILY, size=11, color=FONT_COLOR),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=45, r=25, t=40, b=40),
        height=height,
        transition=dict(duration=0)
    )
    fig.update_xaxes(title_text="Symbol Period (Ts)", showgrid=True, gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR)
    fig.update_yaxes(title_text="Amplitude", showgrid=True, gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR)
    return fig


def plot_time_domain_envelope(
    signal: np.ndarray,
    fs: float,
    max_points: int = 1000,
    height: int = 440
) -> go.Figure:
    """
    Plots real, imaginary, and envelope magnitude time series.
    """
    if len(signal) > max_points:
        step = len(signal) // max_points
        sig_sample = signal[: max_points * step : step]
    else:
        sig_sample = signal

    t = np.arange(len(sig_sample)) / (fs / (len(signal) / len(sig_sample)))
    if np.max(t) < 1e-3:
        t_scale = 1e6
        t_unit = "μs"
    elif np.max(t) < 1.0:
        t_scale = 1e3
        t_unit = "ms"
    else:
        t_scale = 1.0
        t_unit = "s"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t * t_scale,
        y=np.real(sig_sample),
        mode='lines',
        name='In-Phase (I)',
        line=dict(color='#38bdf8', width=1.2)
    ))

    fig.add_trace(go.Scatter(
        x=t * t_scale,
        y=np.imag(sig_sample),
        mode='lines',
        name='Quadrature (Q)',
        line=dict(color='#f43f5e', width=1.2)
    ))

    fig.add_trace(go.Scatter(
        x=t * t_scale,
        y=np.abs(sig_sample),
        mode='lines',
        name='Envelope |s(t)|',
        line=dict(color='#10b981', width=1.6, dash='dash')
    ))

    fig.update_layout(
        title=dict(text="<b>Time-Domain Baseband Oscilloscope Trace</b>", font=dict(family=FONT_FAMILY, size=14, color="#e2e8f0")),
        xaxis=dict(title=f"Time ({t_unit})", showgrid=True, gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR),
        yaxis=dict(title="Normalized Amplitude", showgrid=True, gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR),
        font=dict(family=FONT_FAMILY, size=11, color=FONT_COLOR),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=45, r=25, t=40, b=40),
        hovermode="x unified",
        height=height,
        transition=dict(duration=0)
    )
    return fig


def plot_synchronized_constellation(
    symbols: np.ndarray,
    max_points: int = 1500,
    height: int = 400
) -> go.Figure:
    """
    Plots the digital constellation scatter diagram of 1 sample/symbol synchronized symbols
    after Costas loop phase lock and Gardner/Early-Late timing recovery.
    """
    if len(symbols) > max_points:
        subsample_idx = np.random.choice(len(symbols), max_points, replace=False)
        pts = symbols[subsample_idx]
    else:
        pts = symbols

    fig = go.Figure()

    # Unit circle reference
    theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(
        x=np.cos(theta),
        y=np.sin(theta),
        mode='lines',
        line=dict(color='rgba(148, 163, 184, 0.25)', width=1, dash='dash'),
        hoverinfo='skip',
        name='Unit Circle'
    ))

    # Constellation points
    fig.add_trace(go.Scatter(
        x=np.real(pts),
        y=np.imag(pts),
        mode='markers',
        marker=dict(
            size=5,
            color='#38bdf8',
            opacity=0.75,
            line=dict(width=0.5, color='#0284c7')
        ),
        name='Synchronized Symbols (1 sps)',
        hovertemplate='I: %{x:.3f}<br>Q: %{y:.3f}<extra></extra>'
    ))

    # Axis limits
    max_val = max(1.5, float(np.percentile(np.abs(pts), 98) * 1.3)) if len(pts) > 0 else 1.5

    fig.update_layout(
        title=dict(
            text="<b>Synchronized Symbol Constellation (1 sps)</b>",
            font=dict(family=FONT_FAMILY, size=14, color="#e2e8f0")
        ),
        xaxis=dict(
            title="In-Phase (I)",
            range=[-max_val, max_val],
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=True,
            zerolinecolor=ZERO_LINE_COLOR,
            constrain='domain'
        ),
        yaxis=dict(
            title="Quadrature (Q)",
            range=[-max_val, max_val],
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=True,
            zerolinecolor=ZERO_LINE_COLOR,
            scaleanchor="x",
            scaleratio=1
        ),
        font=dict(family=FONT_FAMILY, size=11, color=FONT_COLOR),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=40, r=25, t=40, b=40),
        showlegend=False,
        height=height,
        transition=dict(duration=0)
    )
    return fig


def plot_llr_histogram(
    llrs: np.ndarray,
    max_points: int = 10000,
    height: int = 400
) -> go.Figure:
    """
    Plots the probability density histogram of soft Log-Likelihood Ratios (LLRs),
    showing the decision margin separation across the zero boundary:
    LLR > 0 -> Bit 0
    LLR < 0 -> Bit 1
    """
    if len(llrs) > max_points:
        subsample = np.random.choice(llrs, max_points, replace=False)
    else:
        subsample = llrs

    fig = go.Figure()

    # Histogram of LLR values
    fig.add_trace(go.Histogram(
        x=subsample,
        nbinsx=60,
        marker=dict(
            color='#38bdf8',
            line=dict(color='#0284c7', width=0.5)
        ),
        opacity=0.85,
        name='Soft LLRs'
    ))

    # Vertical line at zero decision boundary
    fig.add_vline(
        x=0.0,
        line_width=2,
        line_dash="dash",
        line_color="#f43f5e",
        annotation_text="Decision Boundary (LLR=0)",
        annotation_position="top",
        annotation_font=dict(color="#f43f5e", size=10, family=FONT_FAMILY)
    )

    # Annotations for Bit 1 and Bit 0 regions
    fig.add_annotation(
        x=-0.5,
        y=0.9,
        yref="paper",
        text="<b>Bit 1 Region</b> (LLR &lt; 0)",
        showarrow=False,
        font=dict(color="#f87171", size=11, family=FONT_FAMILY),
        xanchor="right"
    )
    fig.add_annotation(
        x=0.5,
        y=0.9,
        yref="paper",
        text="<b>Bit 0 Region</b> (LLR &gt; 0)",
        showarrow=False,
        font=dict(color="#34d399", size=11, family=FONT_FAMILY),
        xanchor="left"
    )

    fig.update_layout(
        title=dict(
            text="<b>Soft Decision Log-Likelihood Ratio (LLR) Margin</b>",
            font=dict(family=FONT_FAMILY, size=14, color="#e2e8f0")
        ),
        xaxis=dict(
            title="Log-Likelihood Ratio (LLR) = ln[P(b=0) / P(b=1)]",
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False
        ),
        yaxis=dict(
            title="Observation Count",
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=True,
            zerolinecolor=ZERO_LINE_COLOR
        ),
        font=dict(family=FONT_FAMILY, size=11, color=FONT_COLOR),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=45, r=25, t=40, b=40),
        showlegend=False,
        height=height,
        transition=dict(duration=0)
    )
    return fig

