"""
optimization/optimizer.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Abstract base class and factory for all optimisation engines.

The :class:`BaseOptimizer` interface is the **single plug-point** for
both Optuna-backed optimisers and the future ML surrogate.  The main
pipeline loop in :mod:`main` never needs to know which engine is active.

Design contract
---------------
1. ``suggest(trial_id)``  → :data:`~core.ParameterSet`
2. ``report(trial_id, params, score)``  → update internal state
3. ``history``  → list of all :class:`~core.TrialRecord` objects

Factory
-------
Use :func:`create_optimizer` to construct the right backend from a
config dict::

    optimizer = create_optimizer("optuna", search_space, config)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from core import ParameterSet, TrialRecord
from optimization.search_space import SearchSpace

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseOptimizer(ABC):
    """Abstract optimiser interface.

    All concrete optimisers (Optuna, Random, ML) must inherit from this
    class and implement :meth:`suggest` and :meth:`report`.

    Parameters
    ----------
    search_space:
        Defines the parameter bounds and types.
    config:
        Full optimisation config dict (loaded from ``config.yaml``).
    """

    def __init__(self, search_space: SearchSpace, config: dict) -> None:
        self.search_space = search_space
        self.config = config
        self.history: List[TrialRecord] = []

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def suggest(self, trial_id: int) -> ParameterSet:
        """Suggest the next set of parameters to evaluate.

        Parameters
        ----------
        trial_id:
            Zero-based index of the current trial.

        Returns
        -------
        ParameterSet
            Suggested values for all parameters.
        """
        ...

    @abstractmethod
    def report(
        self,
        trial_id: int,
        params: ParameterSet,
        score: float,
        success: bool = True,
    ) -> None:
        """Report the outcome of a trial back to the optimiser.

        Parameters
        ----------
        trial_id:
            Index of the trial being reported (must match a prior call
            to :meth:`suggest`).
        params:
            Parameter set that was evaluated.
        score:
            Objective value (lower is better).
        success:
            ``False`` if the simulation failed; the score is typically
            ``float("inf")`` in that case.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def best_trial(self) -> Optional[TrialRecord]:
        """The :class:`~core.TrialRecord` with the lowest score, or ``None``."""
        successful = [t for t in self.history if t.success and np.isfinite(t.score)]
        if not successful:
            return None
        return min(successful, key=lambda t: t.score)

    @property
    def best_score(self) -> Optional[float]:
        """Lowest score seen so far, or ``None`` if no successful trials."""
        trial = self.best_trial
        return trial.score if trial is not None else None

    @property
    def best_params(self) -> Optional[ParameterSet]:
        """Parameter set that produced the best score, or ``None``."""
        trial = self.best_trial
        return trial.params if trial is not None else None

    def get_history_df(self) -> "pd.DataFrame":
        """Return all trials as a :class:`pandas.DataFrame`.

        Columns: ``trial_id``, ``score``, ``success``, and one column per
        parameter.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ImportError
            If ``pandas`` is not installed.
        """
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pandas is required for get_history_df(). "
                "Install it with: pip install pandas"
            ) from exc

        rows = []
        for trial in self.history:
            row: dict = {
                "trial_id": trial.trial_id,
                "score": trial.score,
                "success": trial.success,
            }
            row.update(trial.params)
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internals shared by subclasses
    # ------------------------------------------------------------------

    def _record(
        self,
        trial_id: int,
        params: ParameterSet,
        score: float,
        success: bool = True,
        error_msg: str = "",
    ) -> TrialRecord:
        """Create a :class:`~core.TrialRecord` and append it to :attr:`history`."""
        record = TrialRecord(
            trial_id=trial_id,
            params=params,
            score=score,
            success=success,
            error_msg=error_msg,
        )
        self.history.append(record)
        return record


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_optimizer(
    engine: str, search_space: SearchSpace, config: dict
) -> BaseOptimizer:
    """Construct an optimiser backend from a string identifier.

    Parameters
    ----------
    engine:
        One of ``"optuna"``, ``"random"``, or ``"ml"``.
    search_space:
        :class:`~optimization.search_space.SearchSpace` instance.
    config:
        Optimisation config dict (the ``optimization`` section of
        ``config.yaml``).

    Returns
    -------
    BaseOptimizer

    Raises
    ------
    ValueError
        If *engine* is not recognised.
    """
    engine_lower = engine.lower()

    if engine_lower == "optuna":
        from optimization.optuna_engine import OptunaOptimizer  # noqa: PLC0415
        logger.info("Creating OptunaOptimizer")
        return OptunaOptimizer(search_space, config)

    elif engine_lower == "random":
        from optimization.optuna_engine import RandomOptimizer  # noqa: PLC0415
        logger.info("Creating RandomOptimizer")
        return RandomOptimizer(search_space, config)

    elif engine_lower == "ml":
        from ml.trainer import MLOptimizer  # noqa: PLC0415
        logger.info("Creating MLOptimizer (stub)")
        return MLOptimizer(search_space, config)

    else:
        raise ValueError(
            f"Unknown optimisation engine: {engine!r}. "
            f"Valid choices: 'optuna', 'random', 'ml'."
        )
