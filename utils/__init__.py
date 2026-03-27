"""
utils/__init__.py
~~~~~~~~~~~~~~~~~
Shared utility modules for the LTspice AI pipeline.

Public API
----------
- :func:`~utils.logger.setup_logger`  – configure a named logger
- :class:`~utils.cache.SimulationCache`  – content-addressable result cache
"""

from utils.logger import setup_logger
from utils.cache import SimulationCache

__all__ = ["setup_logger", "SimulationCache"]
