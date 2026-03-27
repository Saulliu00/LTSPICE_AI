"""
ml/surrogate_model.py
~~~~~~~~~~~~~~~~~~~~~
Feedforward neural network surrogate for the LTspice objective function.

.. note::
    **This entire module is a stub.**

    Once implemented, the surrogate maps a :data:`~core.ParameterSet`
    to a predicted objective score, enabling fast approximate optimisation
    without invoking LTspice.

    Planned architecture
    --------------------
    - Input layer:  ``n_params`` neurons (log-normalised parameter values)
    - Hidden layers: configurable widths (default ``[64, 64]``)
    - Output layer: 1 neuron (predicted score)
    - Activation:   ReLU throughout, no activation on output
    - Training:     MSE loss, Adam optimiser

    Prerequisites
    -------------
    ``pip install torch``

Usage (once implemented)::

    model = SurrogateModel(input_dim=2, hidden_dims=[64, 64])
    score = model.predict({"R1": 10e3, "C1": 1e-8})
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np

from core import ParameterSet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Surrogate model (STUB)
# ---------------------------------------------------------------------------


class SurrogateModel:
    """Feedforward neural network mapping ``ParameterSet → predicted score``.

    This class is a **stub**.  All methods raise :exc:`NotImplementedError`
    until PyTorch is installed and the network is implemented.

    Parameters
    ----------
    input_dim:
        Number of input features (= number of tuneable parameters).
    hidden_dims:
        Widths of the hidden layers.  Default ``[64, 64]``.
    output_dim:
        Number of output neurons.  Default ``1`` (scalar score prediction).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 1,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [64, 64]
        self.output_dim = output_dim

        # TODO: Once PyTorch is available, construct the network here.
        #   import torch.nn as nn
        #   layers = []
        #   prev_dim = input_dim
        #   for h in self.hidden_dims:
        #       layers += [nn.Linear(prev_dim, h), nn.ReLU()]
        #       prev_dim = h
        #   layers.append(nn.Linear(prev_dim, output_dim))
        #   self._net = nn.Sequential(*layers)

        self._net: Optional[Any] = None  # placeholder
        logger.debug(
            "SurrogateModel (stub) created: input_dim=%d  hidden=%s  output_dim=%d",
            input_dim,
            self.hidden_dims,
            output_dim,
        )

    # ------------------------------------------------------------------
    # Prediction (STUB)
    # ------------------------------------------------------------------

    def predict(self, params: ParameterSet) -> float:
        """Predict the objective score for a single :data:`~core.ParameterSet`.

        .. note::
            **Stub** – raises :exc:`NotImplementedError`.

        Parameters
        ----------
        params:
            Parameter values to evaluate.

        Returns
        -------
        float
            Predicted objective score.

        Raises
        ------
        NotImplementedError
            Always, until implemented.
        """
        # TODO: implement once PyTorch is available.
        #   x = self._params_to_tensor(params)
        #   with torch.no_grad():
        #       return float(self._net(x).item())
        raise NotImplementedError(
            "SurrogateModel.predict() is not implemented.  "
            "Install PyTorch and implement the forward pass."
        )

    def predict_batch(self, params_list: List[ParameterSet]) -> np.ndarray:
        """Predict scores for a list of parameter sets.

        .. note::
            **Stub** – raises :exc:`NotImplementedError`.

        Parameters
        ----------
        params_list:
            List of :data:`~core.ParameterSet` dicts to evaluate.

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` array of predicted scores.

        Raises
        ------
        NotImplementedError
            Always, until implemented.
        """
        # TODO: implement using batched forward pass.
        #   X = torch.stack([self._params_to_tensor(p) for p in params_list])
        #   with torch.no_grad():
        #       return self._net(X).squeeze(-1).numpy()
        raise NotImplementedError(
            "SurrogateModel.predict_batch() is not implemented.  "
            "Install PyTorch and implement the batched forward pass."
        )

    # ------------------------------------------------------------------
    # Persistence (STUB)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise model weights to *path*.

        .. note::
            **Stub** – raises :exc:`NotImplementedError`.

        Parameters
        ----------
        path:
            Destination file (e.g. ``"results/surrogate.pt"``).

        Raises
        ------
        NotImplementedError
            Always, until implemented.
        """
        # TODO:
        #   import torch
        #   torch.save(self._net.state_dict(), path)
        raise NotImplementedError(
            "SurrogateModel.save() is not implemented.  "
            "Use torch.save(model.state_dict(), path) once PyTorch is installed."
        )

    @classmethod
    def load(cls, path: str) -> "SurrogateModel":
        """Load a previously saved model from *path*.

        .. note::
            **Stub** – raises :exc:`NotImplementedError`.

        Parameters
        ----------
        path:
            Source file.

        Returns
        -------
        SurrogateModel

        Raises
        ------
        NotImplementedError
            Always, until implemented.
        """
        # TODO:
        #   import torch
        #   # Infer input_dim / hidden_dims from saved state dict
        #   model = cls(input_dim=..., hidden_dims=...)
        #   model._net.load_state_dict(torch.load(path))
        #   return model
        raise NotImplementedError(
            "SurrogateModel.load() is not implemented.  "
            "Use model.load_state_dict(torch.load(path)) once PyTorch is installed."
        )

    # ------------------------------------------------------------------
    # Internal helpers (STUB)
    # ------------------------------------------------------------------

    def _params_to_tensor(self, params: ParameterSet) -> Any:
        """Convert a :data:`~core.ParameterSet` to a PyTorch tensor.

        .. note::
            **Stub** – not callable without PyTorch.

        Expected implementation:
            1. Extract values in sorted-key order.
            2. Apply log10 scaling for log-scale parameters.
            3. Normalise to [0, 1] using known min/max bounds.
            4. Return a ``torch.Tensor`` of shape ``(input_dim,)``.
        """
        # TODO: implement
        raise NotImplementedError(
            "_params_to_tensor() requires PyTorch and parameter normalisation bounds."
        )

    def __repr__(self) -> str:
        return (
            f"SurrogateModel(input_dim={self.input_dim}, "
            f"hidden_dims={self.hidden_dims}, "
            f"output_dim={self.output_dim}) [STUB]"
        )
