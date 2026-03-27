"""
core/__init__.py
~~~~~~~~~~~~~~~~
Shared data structures used across all pipeline modules.

Every other module should import ``ParameterSet``, ``SimulationResult``,
and ``TrialRecord`` from here to avoid circular dependencies and to
ensure a single source of truth for the data model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: A mapping of component / parameter name → numeric value.
#: Example: ``{"R1": 10_000.0, "C1": 1e-7}``
ParameterSet = Dict[str, float]


# ---------------------------------------------------------------------------
# Core data-classes
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    """Parsed output from a single LTspice simulation run.

    Attributes
    ----------
    signals:
        Dictionary mapping signal names (e.g. ``"V(out)"``) to NumPy arrays
        of complex (AC) or real (transient/DC) values.
    time_or_freq:
        One-dimensional array representing the simulation x-axis:
        - AC  → frequency vector (Hz)
        - Transient → time vector (seconds)
        - DC  → swept variable values
    sim_type:
        One of ``"ac"``, ``"transient"``, or ``"dc"``.
    raw_file:
        Absolute path to the ``.raw`` file that was parsed (informational).
    metadata:
        Arbitrary key/value pairs for extensibility (e.g. run timestamp,
        parameter set used, LTspice version string).
    """

    signals: Dict[str, np.ndarray]
    time_or_freq: np.ndarray
    sim_type: str  # "transient" | "ac" | "dc"
    raw_file: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialRecord:
    """Record of a single optimisation trial.

    Attributes
    ----------
    trial_id:
        Zero-based sequential trial index within the current run.
    params:
        The :data:`ParameterSet` that was evaluated.
    score:
        Scalar objective value (lower is better by convention).
    result:
        Full parsed simulation result, if available.
    success:
        ``False`` if the simulation failed or the objective could not be
        computed; the ``score`` may be ``float("inf")`` in that case.
    error_msg:
        Human-readable description of the failure reason, if any.
    """

    trial_id: int
    params: ParameterSet
    score: float
    result: Optional[SimulationResult] = None
    success: bool = True
    error_msg: str = ""


__all__ = [
    "ParameterSet",
    "SimulationResult",
    "TrialRecord",
]
