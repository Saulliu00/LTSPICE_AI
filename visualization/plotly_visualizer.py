"""
visualization/plotly_visualizer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Interactive Plotly figures for the LTspice AI optimisation pipeline.

:class:`PipelineVisualizer` generates publication-quality HTML figures
for convergence tracking, parameter exploration, waveform inspection,
and frequency-response analysis.

All ``plot_*`` methods return a ``plotly.graph_objects.Figure`` and
optionally write an HTML file and/or open the browser via
:meth:`_save_or_show`.

Usage example::

    viz = PipelineVisualizer(output_dir="results/plots")
    fig = viz.plot_convergence(optimizer.history)
    fig = viz.plot_frequency_response(best_result, signal_names=["V(out)"])
    viz.create_dashboard(optimizer.history, best_result)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from core import SimulationResult, TrialRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plotly import guard
# ---------------------------------------------------------------------------

try:
    import plotly.graph_objects as go  # type: ignore
    import plotly.subplots as sp  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore

    _PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLOTLY_AVAILABLE = False
    logger.warning(
        "plotly not installed – visualisation disabled.  "
        "Install with: pip install plotly"
    )


def _require_plotly() -> None:
    if not _PLOTLY_AVAILABLE:
        raise ImportError(
            "plotly is required for visualisation.  "
            "Install with: pip install plotly"
        )


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_PALETTE = {
    "primary":   "#636EFA",
    "secondary": "#EF553B",
    "success":   "#00CC96",
    "warning":   "#FFA15A",
    "neutral":   "#AB63FA",
    "grid":      "rgba(200,200,200,0.3)",
    "best":      "#FF6692",
}

_FONT = dict(family="Inter, Arial, sans-serif", size=13)


# ---------------------------------------------------------------------------
# PipelineVisualizer
# ---------------------------------------------------------------------------


class PipelineVisualizer:
    """Generate interactive Plotly figures for the optimisation pipeline.

    Parameters
    ----------
    output_dir:
        Directory where HTML files are saved.
    show_browser:
        If ``True``, open each figure in the default web browser.
    save_html:
        If ``True``, write each figure as an ``.html`` file in
        *output_dir*.
    """

    # Plot names that are auto-opened in the browser when show_browser=True.
    # Keep this small: open the most informative plots first.
    # For filter designs: frequency_response first, then dashboard.
    _AUTO_OPEN_PLOTS = {"frequency_response", "waveform", "dashboard"}

    def __init__(
        self,
        output_dir: str = "results/plots",
        show_browser: bool = True,
        save_html: bool = True,
    ) -> None:
        self.output_dir = str(output_dir)
        self.show_browser = show_browser
        self.save_html = save_html
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Convergence plot
    # ------------------------------------------------------------------

    def plot_convergence(
        self, history: List[TrialRecord]
    ) -> "go.Figure":
        """Plot objective score vs trial number with running-best overlay.

        Parameters
        ----------
        history:
            List of :class:`~core.TrialRecord` objects.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        _require_plotly()

        if not history:
            logger.warning("plot_convergence: empty history, returning blank figure")
            return go.Figure()

        trial_ids = [t.trial_id for t in history]
        scores = [t.score if np.isfinite(t.score) else np.nan for t in history]
        success_mask = [t.success for t in history]

        # Running best
        running_best: List[float] = []
        current_best = float("inf")
        for s, ok in zip(scores, success_mask):
            if ok and np.isfinite(s):
                current_best = min(current_best, s)
            running_best.append(current_best if np.isfinite(current_best) else np.nan)

        # Best trial marker
        finite_scores = [(i, s) for i, s in zip(trial_ids, scores) if np.isfinite(s)]
        best_id, best_score = min(finite_scores, key=lambda x: x[1]) if finite_scores else (None, None)

        fig = go.Figure()

        # All trial scores (scatter)
        fig.add_trace(
            go.Scatter(
                x=trial_ids,
                y=scores,
                mode="markers",
                name="Trial score",
                marker=dict(
                    color=[
                        _PALETTE["primary"] if ok else _PALETTE["secondary"]
                        for ok in success_mask
                    ],
                    size=7,
                    opacity=0.7,
                    symbol=["circle" if ok else "x" for ok in success_mask],
                ),
                hovertemplate=(
                    "Trial %{x}<br>Score: %{y:.4g}<extra></extra>"
                ),
            )
        )

        # Running best line
        fig.add_trace(
            go.Scatter(
                x=trial_ids,
                y=running_best,
                mode="lines",
                name="Running best",
                line=dict(color=_PALETTE["success"], width=2, dash="dash"),
                hovertemplate="Trial %{x}<br>Best so far: %{y:.4g}<extra></extra>",
            )
        )

        # Star on best trial
        if best_id is not None:
            fig.add_trace(
                go.Scatter(
                    x=[best_id],
                    y=[best_score],
                    mode="markers",
                    name=f"Best (trial {best_id})",
                    marker=dict(
                        color=_PALETTE["best"],
                        size=14,
                        symbol="star",
                        line=dict(color="white", width=1),
                    ),
                    hovertemplate=(
                        f"Best trial: {best_id}<br>Score: {best_score:.4g}"
                        "<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=dict(
                text="Optimisation Convergence",
                font=dict(size=18),
                x=0.5,
            ),
            xaxis=dict(
                title="Trial number",
                gridcolor=_PALETTE["grid"],
                zeroline=False,
            ),
            yaxis=dict(
                title="Objective score",
                gridcolor=_PALETTE["grid"],
                zeroline=False,
            ),
            legend=dict(orientation="h", y=-0.15),
            font=_FONT,
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="x unified",
        )

        self._save_or_show(fig, "convergence")
        return fig

    # ------------------------------------------------------------------
    # Parameter scatter
    # ------------------------------------------------------------------

    def plot_parameter_scatter(
        self,
        history: List[TrialRecord],
        param_x: str,
        param_y: Optional[str] = None,
    ) -> "go.Figure":
        """Scatter plot of parameter value(s) coloured by objective score.

        Parameters
        ----------
        history:
            Trial history.
        param_x:
            Name of the parameter for the x-axis.
        param_y:
            Optional name of a second parameter for the y-axis.  If
            ``None``, a 1-D scatter (param vs score) is produced.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        _require_plotly()

        if not history:
            return go.Figure()

        xs = [t.params.get(param_x, np.nan) for t in history]
        scores = [t.score if np.isfinite(t.score) else np.nan for t in history]

        if param_y is None:
            # 1-D: param_x vs score
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=scores,
                    mode="markers",
                    marker=dict(
                        color=scores,
                        colorscale="Viridis",
                        size=8,
                        colorbar=dict(title="Score"),
                        showscale=True,
                    ),
                    hovertemplate=(
                        f"{param_x}: %{{x:.4g}}<br>Score: %{{y:.4g}}"
                        "<extra></extra>"
                    ),
                )
            )
            title = f"{param_x} vs Objective Score"
            x_title = param_x
            y_title = "Objective score"
            x_log = _should_log_scale(xs)
            y_log = False

        else:
            # 2-D: param_x vs param_y coloured by score
            ys = [t.params.get(param_y, np.nan) for t in history]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker=dict(
                        color=scores,
                        colorscale="Viridis",
                        size=9,
                        colorbar=dict(title="Score"),
                        showscale=True,
                        line=dict(color="white", width=0.5),
                    ),
                    hovertemplate=(
                        f"{param_x}: %{{x:.4g}}<br>"
                        f"{param_y}: %{{y:.4g}}<br>"
                        "Score: %{marker.color:.4g}"
                        "<extra></extra>"
                    ),
                )
            )
            title = f"{param_x} vs {param_y}  (colour = score)"
            x_title = param_x
            y_title = param_y
            x_log = _should_log_scale(xs)
            y_log = _should_log_scale(ys)

        fig.update_layout(
            title=dict(text=title, font=dict(size=18), x=0.5),
            xaxis=dict(
                title=x_title,
                type="log" if x_log else "linear",
                gridcolor=_PALETTE["grid"],
                zeroline=False,
            ),
            yaxis=dict(
                title=y_title,
                type="log" if y_log else "linear",
                gridcolor=_PALETTE["grid"],
                zeroline=False,
            ),
            font=_FONT,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        name = f"scatter_{param_x}" + (f"_vs_{param_y}" if param_y else "_vs_score")
        self._save_or_show(fig, name)
        return fig

    # ------------------------------------------------------------------
    # Waveform (time-domain)
    # ------------------------------------------------------------------

    def plot_waveform(
        self,
        result: SimulationResult,
        signal_names: List[str],
    ) -> "go.Figure":
        """Plot one or more signals against time (transient simulation).

        Parameters
        ----------
        result:
            Parsed simulation output (expected ``sim_type="transient"``).
        signal_names:
            List of signal names to plot (e.g. ``["V(out)", "V(in)"]``).

        Returns
        -------
        plotly.graph_objects.Figure
        """
        _require_plotly()

        t = result.time_or_freq
        fig = go.Figure()

        colors = [
            _PALETTE["primary"],
            _PALETTE["secondary"],
            _PALETTE["success"],
            _PALETTE["warning"],
            _PALETTE["neutral"],
        ]

        for i, name in enumerate(signal_names):
            if name not in result.signals:
                logger.warning("plot_waveform: signal %s not found", name)
                continue
            sig = np.real(result.signals[name])
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=sig,
                    mode="lines",
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=(
                        f"{name}<br>t: %{{x:.3g}} s<br>V: %{{y:.4g}}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=dict(text="Transient Waveform", font=dict(size=18), x=0.5),
            xaxis=dict(
                title="Time (s)",
                gridcolor=_PALETTE["grid"],
                zeroline=False,
            ),
            yaxis=dict(
                title="Voltage (V)",
                gridcolor=_PALETTE["grid"],
                zeroline=False,
            ),
            legend=dict(orientation="h", y=-0.15),
            font=_FONT,
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="x unified",
        )

        self._save_or_show(fig, "waveform")
        return fig

    # ------------------------------------------------------------------
    # Frequency response
    # ------------------------------------------------------------------

    def plot_frequency_response(
        self,
        result: SimulationResult,
        signal_names: List[str],
    ) -> "go.Figure":
        """Plot magnitude (dB) and phase (deg) vs frequency.

        Parameters
        ----------
        result:
            Parsed AC simulation output.
        signal_names:
            Signals to include (e.g. ``["V(out)"]``).

        Returns
        -------
        plotly.graph_objects.Figure
            Two-row subplot: magnitude (top) and phase (bottom).
        """
        _require_plotly()

        freq = result.time_or_freq
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Magnitude (dB)", "Phase (deg)"),
        )

        colors = [
            _PALETTE["primary"],
            _PALETTE["secondary"],
            _PALETTE["success"],
            _PALETTE["warning"],
            _PALETTE["neutral"],
        ]

        for i, name in enumerate(signal_names):
            if name not in result.signals:
                logger.warning("plot_frequency_response: signal %s not found", name)
                continue

            sig = result.signals[name]
            mag_db = 20.0 * np.log10(np.abs(sig) + np.finfo(float).tiny)

            if np.iscomplexobj(sig):
                phase_deg = np.degrees(np.angle(sig))
            else:
                phase_deg = np.zeros_like(mag_db)

            color = colors[i % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=freq,
                    y=mag_db,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=2),
                    hovertemplate=(
                        f"{name}<br>f: %{{x:.4g}} Hz<br>"
                        "|H|: %{y:.2f} dB<extra></extra>"
                    ),
                    legendgroup=name,
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=freq,
                    y=phase_deg,
                    mode="lines",
                    name=f"{name} (phase)",
                    line=dict(color=color, width=2, dash="dot"),
                    hovertemplate=(
                        f"{name}<br>f: %{{x:.4g}} Hz<br>"
                        "Phase: %{y:.1f}°<extra></extra>"
                    ),
                    legendgroup=f"{name}_phase",
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

        # -3 dB reference line
        fig.add_hline(
            y=-3,
            line=dict(color=_PALETTE["secondary"], dash="dash", width=1),
            annotation_text="-3 dB",
            annotation_position="right",
            row=1,
            col=1,
        )

        fig.update_xaxes(
            type="log",
            title_text="Frequency (Hz)",
            gridcolor=_PALETTE["grid"],
            zeroline=False,
            row=2,
            col=1,
        )
        fig.update_xaxes(
            type="log",
            gridcolor=_PALETTE["grid"],
            zeroline=False,
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text="Magnitude (dB)",
            gridcolor=_PALETTE["grid"],
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text="Phase (deg)",
            gridcolor=_PALETTE["grid"],
            row=2,
            col=1,
        )

        fig.update_layout(
            title=dict(
                text="Frequency Response",
                font=dict(size=18),
                x=0.5,
            ),
            font=_FONT,
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.12),
        )

        self._save_or_show(fig, "frequency_response")
        return fig

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def create_dashboard(
        self,
        history: List[TrialRecord],
        best_result: SimulationResult,
    ) -> "go.Figure":
        """Combine convergence, parameter scatter, and best response into one dashboard.

        Layout (2 × 2 grid):
        +---------------------------------+----------------------------+
        | Convergence (full width, row 1) |                            |
        +---------------------------------+----------------------------+
        | Parameter scatter (row 2, col 1)| Freq response (row 2, col2)|
        +---------------------------------+----------------------------+

        Parameters
        ----------
        history:
            Full optimisation history.
        best_result:
            Parsed simulation result from the best trial.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        _require_plotly()

        # Build individual figures first
        conv_fig = self.plot_convergence(history)
        signal_names = list(best_result.signals.keys())[:3]

        if best_result.sim_type == "ac":
            wave_fig = self.plot_frequency_response(best_result, signal_names)
            wave_title = "Best Frequency Response"
        else:
            wave_fig = self.plot_waveform(best_result, signal_names)
            wave_title = "Best Waveform"

        # Determine which parameters to scatter
        if history and history[0].params:
            all_param_names = list(history[0].params.keys())
            px = all_param_names[0]
            py = all_param_names[1] if len(all_param_names) > 1 else None
        else:
            px, py = "param", None

        scatter_fig = self.plot_parameter_scatter(history, px, py)

        # Build combined dashboard
        n_scatter_rows = 1
        n_wave_rows = 2 if best_result.sim_type == "ac" else 1

        fig = make_subplots(
            rows=3,
            cols=2,
            row_heights=[0.35, 0.35, 0.30],
            column_widths=[0.5, 0.5],
            subplot_titles=(
                "Convergence",
                "",
                "Parameter Exploration",
                wave_title,
                "",
                "",
            ),
            vertical_spacing=0.10,
            horizontal_spacing=0.08,
            specs=[
                [{"colspan": 2}, None],
                [{}, {}],
                [{}, {}],
            ],
        )

        # --- Row 1: convergence (spans both columns) ---
        for trace in conv_fig.data:
            fig.add_trace(trace, row=1, col=1)

        # --- Row 2 col 1: parameter scatter ---
        for trace in scatter_fig.data:
            # Disable separate legend entries for dashboard clarity
            trace2 = trace
            fig.add_trace(trace2, row=2, col=1)

        # --- Rows 2-3 col 2: waveform / frequency response ---
        for trace in wave_fig.data:
            row = 2 if "magnitude" in str(trace.name).lower() or not hasattr(trace, "yaxis") else 2
            fig.add_trace(trace, row=2, col=2)

        fig.update_layout(
            title=dict(
                text="LTspice AI Optimisation Dashboard",
                font=dict(size=20),
                x=0.5,
            ),
            font=_FONT,
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=True,
            height=900,
        )

        # Log-scale x-axes for param scatter if needed
        xs = [t.params.get(px, np.nan) for t in history if t.params]
        if _should_log_scale(xs):
            fig.update_xaxes(type="log", row=2, col=1)

        self._save_or_show(fig, "dashboard")
        return fig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_or_show(self, fig: "go.Figure", name: str) -> None:
        """Write the figure to HTML and/or open the browser.

        Parameters
        ----------
        fig:
            The Plotly figure to persist.
        name:
            Base name used for the HTML file (e.g. ``"convergence"``
            → ``results/plots/convergence.html``).
        """
        if self.save_html:
            html_path = str(Path(self.output_dir) / f"{name}.html")
            try:
                fig.write_html(html_path, include_plotlyjs="cdn")
                logger.info("Saved figure → %s", html_path)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to save figure %s: %s", html_path, exc)

        if self.show_browser and name in self._AUTO_OPEN_PLOTS:
            try:
                fig.show()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not open browser for figure %s: %s", name, exc)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _should_log_scale(values: List[float]) -> bool:
    """Heuristic: use log scale if the range spans more than 2 decades."""
    finite = [v for v in values if v is not None and np.isfinite(v) and v > 0]
    if len(finite) < 2:
        return False
    ratio = max(finite) / min(finite)
    return ratio > 100
