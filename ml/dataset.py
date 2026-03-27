"""
ml/dataset.py
~~~~~~~~~~~~~
Dataset container for simulation trials.

:class:`SimulationDataset` stores ``(ParameterSet, SimulationResult, score)``
tuples collected during real LTspice optimisation runs.  The stored data
can later be used to train a :class:`~ml.surrogate_model.SurrogateModel`
that approximates the expensive LTspice simulation.

What is implemented
-------------------
- :meth:`add`  – append a trial record.
- :meth:`save` – pickle the dataset to disk.
- :meth:`load` – restore from a pickle file.
- :meth:`__len__` and :meth:`__getitem__` – Python sequence protocol.

What is a stub (requires PyTorch)
-----------------------------------
- :meth:`to_tensors` – raises :exc:`NotImplementedError` with an
  instructive message.

Usage example::

    dataset = SimulationDataset()
    dataset.add(params={"R1": 10e3, "C1": 1e-8}, result=sim_result, score=42.1)
    dataset.save("results/cache/dataset.pkl")

    # Later …
    ds = SimulationDataset.load("results/cache/dataset.pkl")
    print(len(ds))  # → 1
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core import ParameterSet, SimulationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class SimulationDataset:
    """Dataset of ``(ParameterSet, SimulationResult, score)`` triples.

    Designed to be PyTorch DataLoader-compatible once :meth:`to_tensors`
    is implemented, but core storage requires no ML framework.

    Parameters
    ----------
    parameter_names:
        Optional ordered list of parameter names.  If provided, the
        order is used when converting to tensors.  If ``None``, the
        order is inferred from the first call to :meth:`add`.
    """

    def __init__(
        self, parameter_names: Optional[List[str]] = None
    ) -> None:
        self._param_names: Optional[List[str]] = parameter_names
        self._records: List[Tuple[ParameterSet, Optional[SimulationResult], float]] = []

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------

    def add(
        self,
        params: ParameterSet,
        result: Optional[SimulationResult],
        score: float,
    ) -> None:
        """Append a trial to the dataset.

        Parameters
        ----------
        params:
            The :data:`~core.ParameterSet` used for the trial.
        result:
            Full :class:`~core.SimulationResult` (may be ``None`` if the
            simulation failed).
        score:
            Objective value for this trial.
        """
        if self._param_names is None:
            self._param_names = sorted(params.keys())
            logger.debug("SimulationDataset: inferred param_names=%s", self._param_names)

        self._records.append((params, result, score))
        logger.debug(
            "SimulationDataset.add: total=%d  score=%.6f  params=%s",
            len(self._records),
            score,
            params,
        )

    def save(self, path: str) -> None:
        """Serialise the dataset to a pickle file.

        Parameters
        ----------
        path:
            Destination file path (parent directories are created if
            needed).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "param_names": self._param_names,
                    "records": self._records,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("SimulationDataset saved to %s (%d records)", path, len(self))

    @classmethod
    def load(cls, path: str) -> "SimulationDataset":
        """Load a dataset previously saved with :meth:`save`.

        Parameters
        ----------
        path:
            Path to the pickle file.

        Returns
        -------
        SimulationDataset

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        path = str(Path(path).resolve())
        if not Path(path).exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(path, "rb") as fh:
            data = pickle.load(fh)

        ds = cls(parameter_names=data.get("param_names"))
        ds._records = data.get("records", [])
        logger.info(
            "SimulationDataset loaded from %s (%d records)", path, len(ds)
        )
        return ds

    # ------------------------------------------------------------------
    # Sequence protocol (for future DataLoader compatibility)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of trials stored."""
        return len(self._records)

    def __getitem__(self, idx: int) -> Tuple[ParameterSet, Optional[SimulationResult], float]:
        """Return the trial at index *idx*.

        Returns
        -------
        tuple
            ``(params, result, score)``

        Note
        ----
        A future PyTorch implementation should override this to return
        ``(feature_tensor, target_tensor)`` for DataLoader compatibility.
        """
        return self._records[idx]

    # ------------------------------------------------------------------
    # ML conversion (stub)
    # ------------------------------------------------------------------

    def to_tensors(self) -> Any:
        """Convert stored data to PyTorch tensors.

        .. note::
            **This method is a stub.**  It requires PyTorch and a mapping
            from :class:`~core.SimulationResult` to a fixed-length feature
            vector.

        Returns
        -------
        tuple
            ``(X_tensor, y_tensor)`` where ``X_tensor`` has shape
            ``(N, n_params)`` and ``y_tensor`` has shape ``(N, 1)``.

        Raises
        ------
        NotImplementedError
            Always, until implemented.
        """
        raise NotImplementedError(
            "to_tensors() is not yet implemented.  "
            "Steps to implement:\n"
            "  1. Install PyTorch: pip install torch\n"
            "  2. Build a feature vector from each ParameterSet "
            "     (e.g. log-normalised parameter values).\n"
            "  3. Stack into a torch.Tensor of shape (N, n_params).\n"
            "  4. Stack scores into a torch.Tensor of shape (N, 1).\n"
            "  5. Return (X_tensor, y_tensor)."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def parameter_names(self) -> Optional[List[str]]:
        """Ordered list of parameter names, or ``None`` if the dataset is empty."""
        return self._param_names

    def scores(self) -> List[float]:
        """Return all scores as a plain Python list."""
        return [r[2] for r in self._records]

    def params_list(self) -> List[ParameterSet]:
        """Return all parameter sets as a list."""
        return [r[0] for r in self._records]

    def __repr__(self) -> str:
        return (
            f"SimulationDataset(n={len(self)}, "
            f"params={self._param_names})"
        )
