"""
ml/trainer.py
~~~~~~~~~~~~~
PyTorch training utilities and the :class:`MLOptimizer` surrogate-guided
optimiser.

.. note::
    **This module is a stub.**

    :class:`SurrogateTrainer` and :class:`MLOptimizer` are defined with
    correct interfaces but all meaningful logic raises
    :exc:`NotImplementedError` until PyTorch is installed and the
    training loop is implemented.

    :class:`MLOptimizer` **does** store trial data in a
    :class:`~ml.dataset.SimulationDataset` via :meth:`report`, so
    data collection is functional even before the ML model is trained.

Intended workflow (once implemented)
-------------------------------------
1. Run ~50–200 real LTspice trials with ``engine: optuna``.
2. Export the trial history to a :class:`~ml.dataset.SimulationDataset`.
3. Train the surrogate with ``SurrogateTrainer.train()``.
4. Switch to ``engine: ml`` – subsequent ``suggest()`` calls use the
   surrogate for cheap approximate optimisation with occasional real
   LTspice validation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from core import ParameterSet, TrialRecord
from ml.dataset import SimulationDataset
from ml.surrogate_model import SurrogateModel
from optimization.optimizer import BaseOptimizer
from optimization.search_space import SearchSpace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SurrogateTrainer (STUB)
# ---------------------------------------------------------------------------


class SurrogateTrainer:
    """Training harness for :class:`~ml.surrogate_model.SurrogateModel`.

    .. note::
        **Stub** – all methods raise :exc:`NotImplementedError`.

    Parameters
    ----------
    model:
        The surrogate model to train.
    dataset:
        Dataset of collected simulation trials.
    config:
        Training config dict.  Expected keys:

        - ``learning_rate`` (float, default ``1e-3``)
        - ``batch_size``    (int,   default ``32``)
        - ``validation_split`` (float, default ``0.1``)
    """

    def __init__(
        self,
        model: SurrogateModel,
        dataset: SimulationDataset,
        config: dict,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.config = config

        # TODO: Once PyTorch is available:
        #   import torch.optim as optim
        #   self._optimizer = optim.Adam(
        #       model._net.parameters(),
        #       lr=config.get("learning_rate", 1e-3)
        #   )
        #   self._loss_fn = torch.nn.MSELoss()

        logger.debug("SurrogateTrainer (stub) initialised")

    # ------------------------------------------------------------------
    # Training (STUB)
    # ------------------------------------------------------------------

    def train(self, epochs: int = 100) -> Dict[str, List[float]]:
        """Train the surrogate model on the collected dataset.

        .. note::
            **Stub** – raises :exc:`NotImplementedError`.

        Parameters
        ----------
        epochs:
            Number of full passes over the training data.

        Returns
        -------
        dict
            Training history: ``{"train_loss": [...], "val_loss": [...]}``.

        Raises
        ------
        NotImplementedError
            Always, until implemented.
        """
        # TODO: Implement training loop:
        #
        #   X_tensor, y_tensor = self.dataset.to_tensors()
        #
        #   # Train/validation split
        #   n = len(X_tensor)
        #   val_n = int(n * self.config.get("validation_split", 0.1))
        #   idx = torch.randperm(n)
        #   X_train, y_train = X_tensor[idx[val_n:]], y_tensor[idx[val_n:]]
        #   X_val,   y_val   = X_tensor[idx[:val_n]], y_tensor[idx[:val_n]]
        #
        #   train_losses, val_losses = [], []
        #   for epoch in range(epochs):
        #       # Mini-batch gradient descent
        #       perm = torch.randperm(len(X_train))
        #       batch_size = self.config.get("batch_size", 32)
        #       epoch_loss = 0.0
        #       for start in range(0, len(X_train), batch_size):
        #           batch_X = X_train[perm[start:start+batch_size]]
        #           batch_y = y_train[perm[start:start+batch_size]]
        #           pred = self.model._net(batch_X)
        #           loss = self._loss_fn(pred, batch_y)
        #           self._optimizer.zero_grad()
        #           loss.backward()
        #           self._optimizer.step()
        #           epoch_loss += loss.item() * len(batch_X)
        #       train_losses.append(epoch_loss / len(X_train))
        #
        #       with torch.no_grad():
        #           val_loss = self._loss_fn(
        #               self.model._net(X_val), y_val
        #           ).item()
        #       val_losses.append(val_loss)
        #
        #       if epoch % 10 == 0:
        #           logger.info("Epoch %d/%d  train=%.4f  val=%.4f",
        #                       epoch, epochs, train_losses[-1], val_losses[-1])
        #
        #   return {"train_loss": train_losses, "val_loss": val_losses}

        raise NotImplementedError(
            "SurrogateTrainer.train() is not implemented.  "
            "Install PyTorch and implement the training loop (see TODO comments)."
        )

    # ------------------------------------------------------------------
    # Evaluation (STUB)
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the trained surrogate on the full dataset.

        .. note::
            **Stub** – raises :exc:`NotImplementedError`.

        Returns
        -------
        dict
            Metrics: ``{"mse": float, "mae": float, "r2": float}``.

        Raises
        ------
        NotImplementedError
            Always, until implemented.
        """
        # TODO: Implement evaluation metrics:
        #
        #   X_tensor, y_tensor = self.dataset.to_tensors()
        #   with torch.no_grad():
        #       preds = self.model._net(X_tensor).squeeze()
        #   targets = y_tensor.squeeze()
        #
        #   mse = float(((preds - targets) ** 2).mean())
        #   mae = float((preds - targets).abs().mean())
        #   ss_tot = float(((targets - targets.mean()) ** 2).sum())
        #   ss_res = float(((preds - targets) ** 2).sum())
        #   r2 = 1 - ss_res / (ss_tot + 1e-10)
        #
        #   return {"mse": mse, "mae": mae, "r2": r2}

        raise NotImplementedError(
            "SurrogateTrainer.evaluate() is not implemented.  "
            "Install PyTorch and implement evaluation metrics (see TODO comments)."
        )


# ---------------------------------------------------------------------------
# MLOptimizer (STUB)
# ---------------------------------------------------------------------------


class MLOptimizer(BaseOptimizer):
    """Surrogate-guided optimiser using :class:`~ml.surrogate_model.SurrogateModel`.

    .. note::
        :meth:`suggest` is a **stub** and raises :exc:`NotImplementedError`.
        :meth:`report` is **fully functional** and stores each trial in a
        :class:`~ml.dataset.SimulationDataset` for future training.

    Parameters
    ----------
    search_space:
        Search-space definition.
    config:
        Optimisation config dict.
    """

    def __init__(self, search_space: SearchSpace, config: dict) -> None:
        super().__init__(search_space, config)
        self.dataset = SimulationDataset(
            parameter_names=sorted([s.name for s in search_space.specs])
        )
        # TODO: Instantiate and load (or train) a SurrogateModel here.
        #   input_dim = len(search_space.specs)
        #   self._model = SurrogateModel(input_dim=input_dim)
        self._model: Optional[SurrogateModel] = None
        logger.info(
            "MLOptimizer (stub) ready – suggest() will raise NotImplementedError"
        )

    # ------------------------------------------------------------------
    # BaseOptimizer interface
    # ------------------------------------------------------------------

    def suggest(self, trial_id: int) -> ParameterSet:
        """Suggest parameters using the surrogate model.

        .. note::
            **Stub** – raises :exc:`NotImplementedError`.

        Raises
        ------
        NotImplementedError
            Always.  Set ``engine: optuna`` in ``config/config.yaml`` to
            use a working optimiser.
        """
        # TODO: Once the surrogate is trained:
        #
        #   1. Generate a large number of candidate parameter sets
        #      (e.g. using random sampling or a grid).
        #   2. Score each candidate cheaply with self._model.predict().
        #   3. Return the candidate with the lowest predicted score.
        #
        #   Alternatively, use Optuna's TPE to optimise over the surrogate
        #   (Bayesian optimisation with a neural surrogate).

        raise NotImplementedError(
            "MLOptimizer.suggest() is not yet implemented.  "
            "Set engine=optuna in config/config.yaml to use the Optuna engine, "
            "or implement the surrogate-guided suggest() method."
        )

    def report(
        self,
        trial_id: int,
        params: ParameterSet,
        score: float,
        success: bool = True,
    ) -> None:
        """Record the trial for future surrogate training.

        This method is **fully functional**: it appends the trial to the
        internal :class:`~ml.dataset.SimulationDataset`.

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
        # Store in the ML dataset regardless of success so that failed
        # regions are also represented in the training data.
        self.dataset.add(params=params, result=None, score=score)
        logger.info(
            "MLOptimizer.report [%d]: score=%.4g (dataset size=%d)",
            trial_id,
            score,
            len(self.dataset),
        )
