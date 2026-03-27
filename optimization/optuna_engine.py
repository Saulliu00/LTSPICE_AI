"""
optimization/optuna_engine.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Optuna-backed and pure-random optimisation engines.

Classes
-------
- :class:`OptunaOptimizer`  – wraps ``optuna.Study``; supports TPE, CMA-ES,
  and Random samplers selected via ``config["sampler"]``.
- :class:`RandomOptimizer`  – no feedback, pure random sampling; useful as
  a baseline.

Both implement the :class:`~optimization.optimizer.BaseOptimizer` interface
so the main pipeline loop requires no changes when switching engines.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core import ParameterSet, TrialRecord
from optimization.optimizer import BaseOptimizer
from optimization.search_space import SearchSpace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optuna optimizer
# ---------------------------------------------------------------------------


class OptunaOptimizer(BaseOptimizer):
    """Optimiser backed by an ``optuna.Study``.

    Parameters
    ----------
    search_space:
        Search-space definition.
    config:
        Optimisation config dict.  Recognised keys:

        - ``direction`` (``"minimize"`` | ``"maximize"``)
        - ``seed``      (int, for sampler reproducibility)
        - ``sampler``   (``"tpe"`` | ``"cmaes"`` | ``"random"``)
    """

    def __init__(self, search_space: SearchSpace, config: dict) -> None:
        super().__init__(search_space, config)

        try:
            import optuna  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "optuna is required for OptunaOptimizer. "
                "Install with: pip install optuna"
            ) from exc

        import optuna  # noqa: PLC0415  (re-import for narrower scope)

        direction = config.get("direction", "minimize")
        seed = config.get("seed", None)
        sampler_name = config.get("sampler", "tpe").lower()

        sampler = self._build_sampler(optuna, sampler_name, seed)

        # Suppress Optuna's default verbose logging – the pipeline has
        # its own logger.
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._study = optuna.create_study(direction=direction, sampler=sampler)
        self._optuna_trials: Dict[int, Any] = {}  # trial_id → optuna.Trial

        logger.info(
            "OptunaOptimizer ready  direction=%s  sampler=%s  seed=%s",
            direction,
            sampler_name,
            seed,
        )

    # ------------------------------------------------------------------
    # BaseOptimizer interface
    # ------------------------------------------------------------------

    def suggest(self, trial_id: int) -> ParameterSet:
        """Ask the Optuna study for the next parameter suggestion.

        Creates an internal Optuna trial and stores it for later
        reporting.

        Parameters
        ----------
        trial_id:
            Zero-based index used to correlate :meth:`suggest` and
            :meth:`report` calls.

        Returns
        -------
        ParameterSet
        """
        optuna_trial = self._study.ask()
        self._optuna_trials[trial_id] = optuna_trial
        params = self.search_space.suggest_optuna(optuna_trial)
        logger.debug("OptunaOptimizer.suggest [%d]: %s", trial_id, params)
        return params

    def report(
        self,
        trial_id: int,
        params: ParameterSet,
        score: float,
        success: bool = True,
    ) -> None:
        """Report the trial outcome back to the Optuna study.

        Parameters
        ----------
        trial_id:
            Must match a prior :meth:`suggest` call.
        params:
            Evaluated parameter set.
        score:
            Objective value.
        success:
            If ``False``, the trial is marked as failed in Optuna.
        """
        import optuna  # noqa: PLC0415

        optuna_trial = self._optuna_trials.get(trial_id)
        if optuna_trial is None:
            logger.warning(
                "OptunaOptimizer.report: no matching optuna trial for id=%d",
                trial_id,
            )
            self._record(trial_id, params, score, success)
            return

        if success and score < float("inf"):
            self._study.tell(optuna_trial, score)
        else:
            self._study.tell(
                optuna_trial,
                state=optuna.trial.TrialState.FAIL,
            )

        record = self._record(trial_id, params, score, success)

        best = self.best_score
        logger.info(
            "Trial %3d  score=%-14.4g  best=%-14.4g  params=%s",
            trial_id,
            score,
            best if best is not None else float("nan"),
            {k: f"{v:.4g}" for k, v in params.items()},
        )

    # ------------------------------------------------------------------
    # Extra Optuna-specific accessors
    # ------------------------------------------------------------------

    @property
    def study(self) -> Any:
        """The underlying ``optuna.Study`` object."""
        return self._study

    def get_optuna_best_params(self) -> Optional[ParameterSet]:
        """Return the best params according to the Optuna study object."""
        try:
            return dict(self._study.best_params)
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sampler(optuna: Any, name: str, seed: Optional[int]) -> Any:
        """Instantiate the requested Optuna sampler.

        Parameters
        ----------
        optuna:
            The imported ``optuna`` module.
        name:
            Sampler name: ``"tpe"``, ``"cmaes"``, or ``"random"``.
        seed:
            Optional random seed.

        Returns
        -------
        optuna.samplers.BaseSampler
        """
        if name == "tpe":
            return optuna.samplers.TPESampler(seed=seed)
        elif name in {"cmaes", "cma-es", "cma_es"}:
            return optuna.samplers.CmaEsSampler(seed=seed)
        elif name == "random":
            return optuna.samplers.RandomSampler(seed=seed)
        else:
            logger.warning(
                "Unknown sampler %r – falling back to TPE", name
            )
            return optuna.samplers.TPESampler(seed=seed)


# ---------------------------------------------------------------------------
# Random optimizer (no feedback)
# ---------------------------------------------------------------------------


class RandomOptimizer(BaseOptimizer):
    """Pure random sampling optimiser.

    No feedback from trial outcomes is used to guide subsequent
    suggestions.  Useful as a baseline to compare against TPE.

    Parameters
    ----------
    search_space:
        Search-space definition.
    config:
        Config dict.  Only ``seed`` is used.
    """

    def __init__(self, search_space: SearchSpace, config: dict) -> None:
        super().__init__(search_space, config)
        self._seed = config.get("seed", None)
        self._counter = 0
        logger.info("RandomOptimizer ready  seed=%s", self._seed)

    # ------------------------------------------------------------------
    # BaseOptimizer interface
    # ------------------------------------------------------------------

    def suggest(self, trial_id: int) -> ParameterSet:
        """Return a random sample from the search space.

        Parameters
        ----------
        trial_id:
            Trial index (used to derive a per-trial seed when a global
            seed is configured, ensuring reproducibility).

        Returns
        -------
        ParameterSet
        """
        per_trial_seed = (
            (self._seed + trial_id) if self._seed is not None else None
        )
        params = self.search_space.sample_random(seed=per_trial_seed)
        logger.debug("RandomOptimizer.suggest [%d]: %s", trial_id, params)
        self._counter += 1
        return params

    def report(
        self,
        trial_id: int,
        params: ParameterSet,
        score: float,
        success: bool = True,
    ) -> None:
        """Record the trial; no feedback is used for future suggestions.

        Parameters
        ----------
        trial_id:
            Trial index.
        params:
            Evaluated parameter set.
        score:
            Objective value.
        success:
            Whether the simulation succeeded.
        """
        self._record(trial_id, params, score, success)
        best = self.best_score
        logger.info(
            "Trial %3d  score=%-14.4g  best=%-14.4g  params=%s",
            trial_id,
            score,
            best if best is not None else float("nan"),
            {k: f"{v:.4g}" for k, v in params.items()},
        )
