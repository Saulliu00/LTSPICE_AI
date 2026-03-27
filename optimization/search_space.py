"""
optimization/search_space.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Defines the parameter search space for the optimisation pipeline.

Classes
-------
- :class:`ParameterSpec`  – specification for a single tuneable parameter
- :class:`SearchSpace`    – collection of :class:`ParameterSpec` objects with
  sampling, validation, and Optuna-integration helpers

Usage example::

    space = SearchSpace.from_config({
        "R1": {"min": 1e3, "max": 1e6, "log_scale": True, "type": "float"},
        "C1": {"min": 1e-9, "max": 1e-6, "log_scale": True, "type": "float"},
    })
    params = space.sample_random(seed=42)
    # {'R1': 31622.776..., 'C1': 3.162e-08}
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from core import ParameterSet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ParameterSpec
# ---------------------------------------------------------------------------


@dataclass
class ParameterSpec:
    """Specification for a single tuneable parameter.

    Attributes
    ----------
    name:
        Parameter name as it appears in the netlist (e.g. ``"R1"``).
    min_val:
        Minimum allowed value.
    max_val:
        Maximum allowed value.
    log_scale:
        If ``True``, sampling and Optuna suggestions are done in
        log-space (suitable for parameters spanning several decades).
    param_type:
        One of ``"float"``, ``"int"``, or ``"categorical"``.
    choices:
        For ``param_type="categorical"``, the list of valid options.
        Ignored for ``"float"`` and ``"int"``.
    """

    name: str
    min_val: float
    max_val: float
    log_scale: bool = False
    param_type: str = "float"  # "float" | "int" | "categorical"
    choices: Optional[List[Any]] = None  # for categorical

    def __post_init__(self) -> None:
        if self.param_type not in {"float", "int", "categorical"}:
            raise ValueError(
                f"param_type must be 'float', 'int', or 'categorical', "
                f"got {self.param_type!r}"
            )
        if self.param_type == "categorical" and not self.choices:
            raise ValueError(
                "ParameterSpec with param_type='categorical' must supply choices"
            )
        if self.param_type != "categorical" and self.min_val >= self.max_val:
            raise ValueError(
                f"min_val ({self.min_val}) must be < max_val ({self.max_val}) "
                f"for parameter {self.name!r}"
            )
        if self.log_scale and self.min_val <= 0:
            raise ValueError(
                f"log_scale=True requires min_val > 0 (got {self.min_val}) "
                f"for parameter {self.name!r}"
            )


# ---------------------------------------------------------------------------
# SearchSpace
# ---------------------------------------------------------------------------


class SearchSpace:
    """Collection of :class:`ParameterSpec` objects.

    Provides helpers for random sampling, Optuna integration,
    validation, and clipping.

    Parameters
    ----------
    specs:
        List of :class:`ParameterSpec` objects defining each dimension.
    """

    def __init__(self, specs: List[ParameterSpec]) -> None:
        if not specs:
            raise ValueError("SearchSpace must have at least one ParameterSpec")
        self.specs: List[ParameterSpec] = specs
        self._name_to_spec: Dict[str, ParameterSpec] = {s.name: s for s in specs}

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, param_config: dict) -> "SearchSpace":
        """Build a :class:`SearchSpace` from the ``parameters`` section of
        ``config.yaml``.

        Expected format::

            parameters:
              R1:
                min: 1000.0
                max: 1000000.0
                log_scale: true
                type: float       # optional, default "float"
              C1:
                min: 1.0e-9
                max: 1.0e-6
                log_scale: true

        Parameters
        ----------
        param_config:
            Dictionary mapping parameter names to their spec dicts.

        Returns
        -------
        SearchSpace
        """
        specs: List[ParameterSpec] = []
        for name, spec_dict in param_config.items():
            param_type = spec_dict.get("type", "float")
            choices = spec_dict.get("choices")

            if param_type == "categorical":
                spec = ParameterSpec(
                    name=name,
                    min_val=0.0,
                    max_val=1.0,  # dummy, unused for categorical
                    log_scale=False,
                    param_type="categorical",
                    choices=choices,
                )
            else:
                spec = ParameterSpec(
                    name=name,
                    min_val=float(spec_dict["min"]),
                    max_val=float(spec_dict["max"]),
                    log_scale=bool(spec_dict.get("log_scale", False)),
                    param_type=param_type,
                    choices=choices,
                )
            specs.append(spec)
            logger.debug(
                "SearchSpace: added %s  [%g, %g]  log=%s  type=%s",
                name,
                spec.min_val,
                spec.max_val,
                spec.log_scale,
                param_type,
            )

        return cls(specs)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_random(self, seed: Optional[int] = None) -> ParameterSet:
        """Draw a uniformly random (or log-uniform) sample from the search space.

        Parameters
        ----------
        seed:
            Random seed for reproducibility.

        Returns
        -------
        ParameterSet
            Dictionary mapping each parameter name to a sampled value.
        """
        rng = np.random.default_rng(seed)
        params: ParameterSet = {}

        for spec in self.specs:
            if spec.param_type == "categorical":
                assert spec.choices is not None
                params[spec.name] = rng.choice(len(spec.choices))  # type: ignore[arg-type]
            elif spec.param_type == "int":
                if spec.log_scale:
                    log_lo = math.log10(spec.min_val)
                    log_hi = math.log10(spec.max_val)
                    val = 10 ** rng.uniform(log_lo, log_hi)
                    params[spec.name] = float(int(round(val)))
                else:
                    params[spec.name] = float(
                        int(rng.integers(int(spec.min_val), int(spec.max_val) + 1))
                    )
            else:  # float
                if spec.log_scale:
                    log_lo = math.log10(spec.min_val)
                    log_hi = math.log10(spec.max_val)
                    params[spec.name] = float(10 ** rng.uniform(log_lo, log_hi))
                else:
                    params[spec.name] = float(
                        rng.uniform(spec.min_val, spec.max_val)
                    )

        logger.debug("sample_random: %s", params)
        return params

    # ------------------------------------------------------------------
    # Optuna integration
    # ------------------------------------------------------------------

    def suggest_optuna(self, trial: Any) -> ParameterSet:
        """Ask an Optuna :class:`~optuna.trial.Trial` to suggest values.

        Parameters
        ----------
        trial:
            An ``optuna.trial.Trial`` instance.

        Returns
        -------
        ParameterSet
            Dictionary of suggested parameter values.
        """
        params: ParameterSet = {}
        for spec in self.specs:
            if spec.param_type == "categorical":
                assert spec.choices is not None
                val = trial.suggest_categorical(spec.name, spec.choices)
                params[spec.name] = float(val)
            elif spec.param_type == "int":
                val = trial.suggest_int(
                    spec.name,
                    int(spec.min_val),
                    int(spec.max_val),
                    log=spec.log_scale,
                )
                params[spec.name] = float(val)
            else:  # float
                val = trial.suggest_float(
                    spec.name,
                    spec.min_val,
                    spec.max_val,
                    log=spec.log_scale,
                )
                params[spec.name] = float(val)
        return params

    # ------------------------------------------------------------------
    # Validation & clipping
    # ------------------------------------------------------------------

    def validate(self, params: ParameterSet) -> bool:
        """Return ``True`` iff every parameter in *params* is within bounds.

        Parameters
        ----------
        params:
            Parameter set to check.

        Returns
        -------
        bool
        """
        for spec in self.specs:
            val = params.get(spec.name)
            if val is None:
                logger.debug("validate: missing parameter %s", spec.name)
                return False
            if spec.param_type == "categorical":
                if spec.choices and val not in spec.choices:
                    logger.debug(
                        "validate: %s = %s not in choices %s",
                        spec.name,
                        val,
                        spec.choices,
                    )
                    return False
            else:
                if not (spec.min_val <= val <= spec.max_val):
                    logger.debug(
                        "validate: %s = %g out of [%g, %g]",
                        spec.name,
                        val,
                        spec.min_val,
                        spec.max_val,
                    )
                    return False
        return True

    def clip(self, params: ParameterSet) -> ParameterSet:
        """Clip all numeric values in *params* to their spec bounds.

        Categorical parameters are passed through unchanged.

        Parameters
        ----------
        params:
            Parameter set to clip.

        Returns
        -------
        ParameterSet
            New dictionary with clipped values.
        """
        clipped: ParameterSet = {}
        for spec in self.specs:
            val = params.get(spec.name, spec.min_val)
            if spec.param_type == "categorical":
                clipped[spec.name] = val
            else:
                clipped[spec.name] = float(
                    np.clip(float(val), spec.min_val, spec.max_val)
                )
        return clipped

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.specs)

    def __repr__(self) -> str:
        names = [s.name for s in self.specs]
        return f"SearchSpace(params={names})"
