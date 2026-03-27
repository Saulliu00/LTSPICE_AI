"""
optimization/__init__.py
~~~~~~~~~~~~~~~~~~~~~~~~
Optimisation engines for the LTspice AI pipeline.

Public API
----------
- :class:`~optimization.optimizer.BaseOptimizer`  – abstract interface
- :class:`~optimization.optuna_engine.OptunaOptimizer`
- :class:`~optimization.optuna_engine.RandomOptimizer`
- :class:`~optimization.search_space.SearchSpace`
- :class:`~optimization.search_space.ParameterSpec`
- :func:`~optimization.optimizer.create_optimizer` – factory function
"""

from optimization.search_space import ParameterSpec, SearchSpace
from optimization.optimizer import BaseOptimizer, create_optimizer

__all__ = [
    "ParameterSpec",
    "SearchSpace",
    "BaseOptimizer",
    "create_optimizer",
]
