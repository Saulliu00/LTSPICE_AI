"""
main.py
~~~~~~~
Entry point for the LTspice AI optimisation pipeline.

The :class:`Pipeline` class orchestrates all modules:

    config → search_space → optimizer → (cache check →) netlist edit
    → LTspice run → parse result → objective evaluation → report → repeat

A ``--demo`` flag activates :func:`_synthetic_rc_simulation`, which
generates analytic RC-filter data without LTspice installed, so you can
test the optimisation and visualisation code immediately.

CLI usage::

    # Run with default config (demo mode – no LTspice required)
    python main.py --demo

    # Full run against LTspice
    python main.py --config config/config.yaml --trials 50

    # Override engine
    python main.py --engine random --trials 20 --demo
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when running as a script
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------

from core import ParameterSet, SimulationResult, TrialRecord
from core.ltspice_runner import LTSpiceRunner, SimulationError
from core.netlist_editor import NetlistEditor
from core.result_parser import ResultParser
from core.objective import create_objective, ObjectiveFunction
from optimization.search_space import SearchSpace
from optimization.optimizer import BaseOptimizer, create_optimizer
from utils.logger import setup_logger
from utils.cache import SimulationCache
from visualization.plotly_visualizer import PipelineVisualizer

logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Synthetic simulation (demo mode)
# ---------------------------------------------------------------------------


def _synthetic_rc_simulation(params: ParameterSet) -> SimulationResult:
    """Generate analytic AC sweep data for an RC low-pass filter.

    Transfer function:  H(jω) = 1 / (1 + jω·R·C)

    Parameters
    ----------
    params:
        Must contain ``"R1"`` (Ω) and ``"C1"`` (F).

    Returns
    -------
    SimulationResult
        AC sweep from 10 Hz to 100 MHz, 100 points per decade.
    """
    R = params.get("R1", 10_000.0)
    C = params.get("C1", 100e-9)

    # Frequency vector: 100 points/decade from 10 Hz to 100 MHz
    freq = np.logspace(np.log10(10), np.log10(100e6), num=701)
    omega = 2 * np.pi * freq
    H = 1.0 / (1.0 + 1j * omega * R * C)

    return SimulationResult(
        signals={"V(out)": H},
        time_or_freq=freq,
        sim_type="ac",
        raw_file="(synthetic)",
        metadata={"R1": R, "C1": C},
    )


def _synthetic_rlc_simulation(params: ParameterSet) -> SimulationResult:
    """Generate analytic AC sweep data for a series RLC band-pass filter.

    Transfer function (output across R):

        H(jω) = (jω·R/L) / (1/LC + jω·R/L − ω²)

    Parameters
    ----------
    params:
        Must contain ``"R1"`` (Ω), ``"L1"`` (H), ``"C1"`` (F).
    """
    R = params.get("R1", 10.0)
    L = params.get("L1", 10e-6)
    C = params.get("C1", 2.533e-6)

    freq = np.logspace(np.log10(10), np.log10(100e6), num=701)
    omega = 2 * np.pi * freq
    # H(jω) = jω(R/L) / ((jω)² + jω(R/L) + 1/(LC))
    jw = 1j * omega
    H = (jw * R / L) / (jw**2 + jw * R / L + 1.0 / (L * C))

    return SimulationResult(
        signals={"V(out)": H},
        time_or_freq=freq,
        sim_type="ac",
        raw_file="(synthetic-rlc)",
        metadata={"R1": R, "L1": L, "C1": C},
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Orchestrate the LTspice AI optimisation loop.

    Parameters
    ----------
    config_path:
        Path to ``config/config.yaml``.
    demo_mode:
        If ``True``, use :func:`_synthetic_rc_simulation` instead of
        launching LTspice.  No LTspice installation required.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        demo_mode: bool = False,
    ) -> None:
        self.demo_mode = demo_mode
        self.config = self._load_config(config_path)

        # ------------------------------------------------------------------
        # Logger
        # ------------------------------------------------------------------
        log_cfg = self.config.get("logging", {})
        self._logger = setup_logger(
            name="pipeline",
            level=log_cfg.get("level", "INFO"),
            log_file=log_cfg.get("log_file"),
        )
        logger.info("Pipeline initialising  config=%s  demo=%s", config_path, demo_mode)

        # ------------------------------------------------------------------
        # Search space
        # ------------------------------------------------------------------
        param_cfg = self.config.get("parameters", {})
        self.search_space = SearchSpace.from_config(param_cfg)
        logger.info("Search space: %s", self.search_space)

        # ------------------------------------------------------------------
        # Optimizer
        # ------------------------------------------------------------------
        opt_cfg = self.config.get("optimization", {})
        engine = opt_cfg.get("engine", "optuna")
        self.n_trials: int = int(opt_cfg.get("n_trials", 50))
        self.optimizer: BaseOptimizer = create_optimizer(engine, self.search_space, opt_cfg)

        # ------------------------------------------------------------------
        # LTspice runner (skipped in demo mode)
        # ------------------------------------------------------------------
        if not demo_mode:
            ltspice_cfg = self.config.get("ltspice", {})
            exe = ltspice_cfg.get("executable", "")
            timeout = int(ltspice_cfg.get("timeout", 60))
            retries = int(ltspice_cfg.get("retry_count", 3))
            self.runner = LTSpiceRunner(exe, timeout=timeout, retry_count=retries)
        else:
            self.runner = None  # type: ignore[assignment]

        # ------------------------------------------------------------------
        # Netlist editor (skipped in demo mode)
        # ------------------------------------------------------------------
        if not demo_mode:
            circuit_cfg = self.config.get("circuit", {})
            schematic = circuit_cfg.get(
                "schematic_path", "circuits/example_rc_filter.asc"
            )
            if not Path(schematic).is_absolute():
                schematic = str(_PROJECT_ROOT / schematic)
            self.editor = NetlistEditor(schematic)
            self._schematic_path = schematic
        else:
            self.editor = None  # type: ignore[assignment]
            self._schematic_path = ""

        # ------------------------------------------------------------------
        # Result parser & objective
        # ------------------------------------------------------------------
        self.parser = ResultParser(sim_type="ac")
        # Support both new 'targets:' section and legacy 'objective:' section
        if "targets" in self.config:
            targets_cfg = self.config["targets"]
            self.objective: ObjectiveFunction = create_objective(targets_cfg)
            logger.info(
                "Objective: %s  (from targets: %s)",
                self.objective.name,
                list(targets_cfg.keys()),
            )
        else:
            obj_cfg = self.config.get("objective", {})
            self.objective: ObjectiveFunction = create_objective(obj_cfg)
            logger.info("Objective: %s", self.objective.name)

        # ------------------------------------------------------------------
        # Cache
        # ------------------------------------------------------------------
        cache_cfg = self.config.get("cache", {})
        cache_dir = cache_cfg.get("cache_dir", "results/cache")
        if not Path(cache_dir).is_absolute():
            cache_dir = str(_PROJECT_ROOT / cache_dir)
        self.cache = SimulationCache(
            cache_dir=cache_dir,
            enabled=bool(cache_cfg.get("enabled", True)),
        )

        # ------------------------------------------------------------------
        # Visualizer
        # ------------------------------------------------------------------
        viz_cfg = self.config.get("visualization", {})
        output_dir = viz_cfg.get("output_dir", "results/plots")
        if not Path(output_dir).is_absolute():
            output_dir = str(_PROJECT_ROOT / output_dir)
        self.visualizer = PipelineVisualizer(
            output_dir=output_dir,
            show_browser=bool(viz_cfg.get("show_browser", True)),
            save_html=bool(viz_cfg.get("save_html", True)),
        )

        logger.info("Pipeline ready")

    # ------------------------------------------------------------------
    # Single trial
    # ------------------------------------------------------------------

    def run_trial(
        self, trial_id: int, params: ParameterSet
    ) -> Tuple[float, Optional[SimulationResult]]:
        """Execute one simulation trial and return ``(score, result)``.

        Steps
        -----
        1. Check the on-disk cache.
        2. If demo mode, call :func:`_synthetic_rc_simulation`.
        3. Otherwise: apply parameters → save temp schematic →
           run LTspice → parse ``.raw`` → clean up temp file.
        4. Evaluate objective function.
        5. Store result in cache.

        Parameters
        ----------
        trial_id:
            Zero-based trial index (used for logging).
        params:
            Parameter values suggested by the optimiser.

        Returns
        -------
        tuple
            ``(score, result)`` where *result* is ``None`` on failure.
        """
        logger.debug("run_trial %d  params=%s", trial_id, params)

        # 1. Cache lookup
        cached = self.cache.get(params)
        if cached is not None:
            logger.debug("Trial %d: cache hit", trial_id)
            score = self.objective.evaluate(cached, params)
            return score, cached

        # 2. Simulate
        try:
            if self.demo_mode:
                # Auto-select synthetic model based on which parameters are present
                if "L1" in params:
                    result = _synthetic_rlc_simulation(params)
                else:
                    result = _synthetic_rc_simulation(params)
            else:
                result = self._run_ltspice(params)
        except Exception as exc:  # noqa: BLE001
            logger.error("Trial %d simulation failed: %s", trial_id, exc)
            return float("inf"), None

        # 3. Objective
        try:
            score = self.objective.evaluate(result, params)
        except Exception as exc:  # noqa: BLE001
            logger.error("Trial %d objective failed: %s", trial_id, exc)
            return float("inf"), result

        # 4. Cache store
        self.cache.put(params, result)

        return score, result

    # ------------------------------------------------------------------
    # LTspice execution (real mode)
    # ------------------------------------------------------------------

    def _run_ltspice(self, params: ParameterSet) -> SimulationResult:
        """Edit netlist, run LTspice, parse result, clean up.

        Parameters
        ----------
        params:
            Parameter values to inject.

        Returns
        -------
        SimulationResult
        """
        # Write a temporary schematic with the new parameter values
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=Path(self._schematic_path).suffix,
            prefix="ltspice_trial_",
        )
        os.close(tmp_fd)

        try:
            self.editor.restore()
            self.editor.apply_parameters(params)
            self.editor.save(tmp_path)

            raw_path = self.runner.run(tmp_path)
            result = self.parser.parse(raw_path)

            # Clean up .raw file alongside the temp schematic
            raw_p = Path(raw_path)
            if raw_p.exists():
                raw_p.unlink(missing_ok=True)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return result

    # ------------------------------------------------------------------
    # Optimisation loop
    # ------------------------------------------------------------------

    def run_optimization(self) -> TrialRecord:
        """Run the full optimisation loop.

        Iterates :attr:`n_trials` times, calling
        :meth:`run_trial` and reporting results back to the optimiser.

        Returns
        -------
        TrialRecord
            The best trial found.
        """
        logger.info(
            "Starting optimisation: engine=%s  n_trials=%d  demo=%s",
            self.config.get("optimization", {}).get("engine", "?"),
            self.n_trials,
            self.demo_mode,
        )

        for trial_id in range(self.n_trials):
            params = self.optimizer.suggest(trial_id)

            score, result = self.run_trial(trial_id, params)
            success = score < float("inf")

            self.optimizer.report(trial_id, params, score, success=success)

            # Attach result to history record for later visualisation
            if self.optimizer.history:
                self.optimizer.history[-1].result = result

            # Log progress every 10 trials
            if trial_id % 10 == 0 or trial_id == self.n_trials - 1:
                best = self.optimizer.best_score
                logger.info(
                    "Progress: %d/%d  current=%.4g  best=%.4g",
                    trial_id + 1,
                    self.n_trials,
                    score,
                    best if best is not None else float("nan"),
                )

        best = self.optimizer.best_trial
        if best is None:
            logger.error("No successful trials – cannot determine best result")
            raise RuntimeError("All trials failed; check LTspice configuration")

        logger.info(
            "Optimisation complete.  Best trial=%d  score=%.6g  params=%s",
            best.trial_id,
            best.score,
            best.params,
        )
        logger.info("Cache stats: %s", self.cache.stats())
        return best

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize_results(self, best_result: SimulationResult) -> None:
        """Generate all output plots.

        Parameters
        ----------
        best_result:
            :class:`~core.SimulationResult` from the best trial.
        """
        logger.info("Generating visualisations …")
        history = self.optimizer.history

        # 1. Convergence
        self.visualizer.plot_convergence(history)

        # 2. Parameter scatters
        param_names = list(self.search_space._name_to_spec.keys())
        if param_names:
            px = param_names[0]
            py = param_names[1] if len(param_names) > 1 else None
            self.visualizer.plot_parameter_scatter(history, px, py)

        # 3. Frequency / waveform
        signal_names = list(best_result.signals.keys())[:3]
        if best_result.sim_type == "ac":
            self.visualizer.plot_frequency_response(best_result, signal_names)
        else:
            self.visualizer.plot_waveform(best_result, signal_names)

        # 4. Dashboard
        self.visualizer.create_dashboard(history, best_result)

        logger.info("Visualisation complete")

    # ------------------------------------------------------------------
    # Config loader
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load and return the YAML config as a dict.

        Parameters
        ----------
        config_path:
            Path to the YAML file.

        Returns
        -------
        dict
        """
        path = Path(config_path)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        if not isinstance(cfg, dict):
            raise ValueError(f"Config file {path} did not parse as a YAML mapping")

        return cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ltspice_ai",
        description=(
            "LTspice AI Optimisation Pipeline.  "
            "Use --demo to run without LTspice installed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        metavar="PATH",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        metavar="N",
        help="Override the number of optimisation trials from config.",
    )
    parser.add_argument(
        "--engine",
        default=None,
        choices=["optuna", "random", "ml"],
        help="Override the optimisation engine from config.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help=(
            "Run in demo mode: generate synthetic RC-filter data "
            "instead of calling LTspice.  No installation required."
        ),
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        default=False,
        help="Disable automatic browser opening for plots.",
    )
    parser.add_argument(
        "--inspect",
        metavar="ASC_PATH",
        default=None,
        help=(
            "Inspect an LTspice .asc file and print discovered parameters. "
            "Does not run any simulation. "
            "Example: python main.py --inspect circuits/bandpass_filter.asc"
        ),
    )
    return parser


def _run_inspect(asc_path: str) -> None:
    """Scan a .asc schematic and print discovered parameters + suggested config.

    Parameters
    ----------
    asc_path:
        Path to the LTspice ``.asc`` file to inspect.
    """
    from core.netlist_editor import inspect_netlist

    path = Path(asc_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    print(f"\nInspecting: {path}\n{'='*60}")
    info = inspect_netlist(str(path))

    # --- Parameter references ---
    refs = info["param_refs"]
    defaults = info["param_defaults"]
    print(f"\nTunable parameters found ({len(refs)}):")
    if refs:
        for name in refs:
            default = defaults.get(name, defaults.get(name.upper(), "<no default>"))
            print(f"  {{{name}}}   default = {default}")
    else:
        print("  (none — use {{PARAM}} syntax in component values to make them tunable)")

    # --- Components ---
    comps = info["components"]
    print(f"\nComponent instances ({len(comps)}):")
    for inst, val in sorted(comps.items()):
        print(f"  {inst:<8s}  value = {val}")

    # --- Simulation directives ---
    sim = info["sim_directives"]
    print(f"\nSimulation directives:")
    if sim:
        for d in sim:
            print(f"  {d}")
    else:
        print("  (none found — add .ac/.tran/.dc to your schematic)")

    # --- Suggested config snippet ---
    print(f"\n{'='*60}")
    print("Suggested config/parameters section:\n")
    print("  parameters:")
    for name in refs:
        default_str = defaults.get(name, defaults.get(name.upper(), ""))
        # Try to guess magnitude for reasonable bounds
        print(f"    {name}:")
        print(f"      min: ???          # set your lower bound")
        print(f"      max: ???          # set your upper bound")
        print(f"      log_scale: true   # recommended for R, C, L")
        print(f"      type: float")
        if default_str:
            print(f"      # default in schematic: {default_str}")

    print("\nSuggested config/targets section:\n")
    print("  targets:")
    print("    # Choose the targets relevant to your circuit type:")
    print("    #")
    print("    # Band-pass filter:")
    print("    # bandwidth:")
    print("    #   signal: V(out)")
    print("    #   value: 200.0       # desired BW = f_high - f_low (Hz)")
    print("    #   weight: 1.0")
    print("    # center_frequency:")
    print("    #   signal: V(out)")
    print("    #   value: 1000.0      # desired peak frequency (Hz)")
    print("    #   weight: 1.0")
    print("    #")
    print("    # Low-pass / High-pass filter:")
    print("    # cutoff:")
    print("    #   signal: V(out)")
    print("    #   value: 1000.0      # desired -3 dB cutoff (Hz)")
    print("    #   filter_type: lowpass")
    print("    #   weight: 1.0")
    print("    #")
    print("    # Amplifier / specific frequency gain:")
    print("    # gain_at_frequency:")
    print("    #   signal: V(out)")
    print("    #   value: 0.0         # desired gain in dB")
    print("    #   freq: 1000.0       # at this frequency (Hz)")
    print("    #   weight: 1.0")
    print()


def main() -> None:
    """CLI entry point."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Bootstrap a basic console logger before the Pipeline logger is set up
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )

    # ------------------------------------------------------------------
    # --inspect mode: scan .asc and print parameters, then exit
    # ------------------------------------------------------------------
    if args.inspect:
        _run_inspect(args.inspect)
        return

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------
    pipeline = Pipeline(config_path=args.config, demo_mode=args.demo)

    # Apply CLI overrides
    if args.trials is not None:
        pipeline.n_trials = args.trials
        logger.info("n_trials overridden to %d", args.trials)

    if args.engine is not None:
        opt_cfg = pipeline.config.get("optimization", {}).copy()
        opt_cfg["engine"] = args.engine
        pipeline.optimizer = create_optimizer(
            args.engine, pipeline.search_space, opt_cfg
        )
        logger.info("Engine overridden to %s", args.engine)

    if args.no_browser:
        pipeline.visualizer.show_browser = False

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    best_trial = pipeline.run_optimization()

    best_result = best_trial.result
    if best_result is None and pipeline.demo_mode:
        # Re-run synthetic simulation with best params for plotting
        logger.info("Re-running synthetic simulation with best params for plots")
        best_result = _synthetic_rc_simulation(best_trial.params)

    if best_result is not None:
        pipeline.visualize_results(best_result)
    else:
        logger.warning("No best result available – skipping visualisation")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  OPTIMISATION COMPLETE")
    print("=" * 60)
    print(f"  Best trial : #{best_trial.trial_id}")
    print(f"  Best score : {best_trial.score:.6g}")
    print("  Best params:")
    for k, v in best_trial.params.items():
        print(f"    {k:>12s} = {v:.4g}")
    print("=" * 60)


if __name__ == "__main__":
    main()
