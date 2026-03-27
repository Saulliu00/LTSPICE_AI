"""
ML Module (Not Yet Implemented)
================================

This package is a **stub** for future PyTorch-based surrogate model
integration.

Planned architecture
--------------------
1. :mod:`ml.dataset`         – stores ``(ParameterSet, SimulationResult, score)``
   tuples gathered during real LTspice runs.
2. :mod:`ml.surrogate_model` – feedforward neural network mapping
   ``ParameterSet → predicted_score``.
3. :mod:`ml.trainer`         – PyTorch training loop + :class:`MLOptimizer`
   that uses the surrogate for fast approximate optimisation.

What is already implemented
----------------------------
- :class:`~ml.dataset.SimulationDataset` – ``add``, ``save``, ``load``
  (uses plain Python lists / pickle; no PyTorch required).

What raises ``NotImplementedError``
-------------------------------------
- :meth:`~ml.dataset.SimulationDataset.to_tensors`
- All of :class:`~ml.surrogate_model.SurrogateModel`
- All of :class:`~ml.trainer.SurrogateTrainer`
- :meth:`~ml.trainer.MLOptimizer.suggest`

To enable the ML module
-----------------------
1. Install PyTorch: ``pip install torch``
2. Implement the stubbed methods (see TODO comments in each file).
3. Change ``engine: ml`` in ``config/config.yaml``.
"""
