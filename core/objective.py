"""
core/objective.py
~~~~~~~~~~~~~~~~~
Objective functions that map a :class:`~core.SimulationResult` to a
scalar score.

Convention: **lower is better** for all objective implementations.

Objective types
---------------

``CutoffObjective``  (``type: cutoff``)
    For **low-pass** and **high-pass** filters.

    The cutoff frequency f_c is the -3 dB point where the output *power*
    drops to half of the passband power:

        |H(f_c)| = |H_passband| / √2   ⟺   |H(f_c)|_dB = |H_passband|_dB − 3 dB

    Score = |f_c_actual − f_c_target|  (Hz, lower is better)

``BandwidthObjective``  (``type: bandwidth``)
    For **band-pass** filters.

    Bandwidth is the total width of the passband, defined as the
    difference between the upper and lower -3 dB points:

        BW = f_high − f_low

    where f_high and f_low are the frequencies at which the response
    falls 3 dB below the peak passband level on the upper and lower
    skirts respectively.

    Score = |BW_actual − BW_target|  (Hz, lower is better)

``MSEObjective``  (``type: mse``)
    Mean-squared error between the simulated magnitude (dB) and a
    target curve.

``PeakObjective``  (``type: peak``)
    Deviation of the peak magnitude and frequency from desired values.

Factory
-------
Use :func:`create_objective` to build an objective from a config dict.

Usage example::

    obj = create_objective({
        "type": "cutoff",
        "target_signal": "V(out)",
        "filter_type": "lowpass",
        "target_cutoff_hz": 1000.0,
    })
    score = obj.evaluate(result, params)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

from core import ParameterSet, SimulationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class ObjectiveFunction(ABC):
    """Abstract base for all objective functions.

    All concrete implementations must satisfy:
    - :meth:`evaluate` returns a finite ``float`` (lower is better).
    - :attr:`name` returns a short human-readable identifier.
    """

    @abstractmethod
    def evaluate(
        self, result: SimulationResult, params: ParameterSet
    ) -> float:
        """Compute the objective score.

        Parameters
        ----------
        result:
            Parsed simulation output.
        params:
            The parameter set used to generate *result*.

        Returns
        -------
        float
            Scalar score; **lower is better**.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this objective (e.g. ``"cutoff"``)."""
        ...


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _magnitude_db(sig: np.ndarray) -> np.ndarray:
    """Return 20·log10(|sig|) with a small epsilon to avoid -inf."""
    return 20.0 * np.log10(np.abs(sig) + np.finfo(float).tiny)


def _log_interp_freq(
    f0: float, f1: float, m0: float, m1: float, threshold: float
) -> float:
    """Linearly interpolate crossing on a log-frequency axis.

    Returns the frequency where the magnitude crosses *threshold*
    between (f0, m0) and (f1, m1), using log10(f) for accuracy across
    wide frequency spans.
    """
    if m1 == m0:
        return f0
    t = (threshold - m0) / (m1 - m0)
    if f0 > 0 and f1 > 0:
        return float(10 ** (np.log10(f0) + t * (np.log10(f1) - np.log10(f0))))
    return float(f0 + t * (f1 - f0))


# ---------------------------------------------------------------------------
# CutoffObjective  –  Low-pass / High-pass filters
# ---------------------------------------------------------------------------


class CutoffObjective(ObjectiveFunction):
    """Penalise deviation of the -3 dB cutoff frequency from a target.

    **Definition (LPF and HPF)**:
    The cutoff frequency f_c is the -3 dB point where the output power
    drops to half of the passband power::

        |H(f_c)| = |H_passband| / √2
        |H(f_c)|_dB = |H_passband|_dB − 3 dB

    Parameters
    ----------
    target_signal:
        Signal to analyse, e.g. ``"V(out)"``.
    target_cutoff_hz:
        Desired -3 dB cutoff frequency in Hz.
    filter_type:
        ``"lowpass"`` (default) or ``"highpass"``.

        * *lowpass*  – passband is the low-frequency region; the cutoff
          is where the response first falls 3 dB below the passband.
        * *highpass* – passband is the high-frequency region; the cutoff
          is the lowest frequency where the response rises to within 3 dB
          of the passband.
    """

    def __init__(
        self,
        target_signal: str,
        target_cutoff_hz: float,
        filter_type: str = "lowpass",
    ) -> None:
        self.target_signal = target_signal
        self.target_cutoff_hz = float(target_cutoff_hz)
        filter_type = filter_type.lower().strip()
        if filter_type not in ("lowpass", "highpass"):
            raise ValueError(
                f"filter_type must be 'lowpass' or 'highpass', got {filter_type!r}"
            )
        self.filter_type = filter_type

    @property
    def name(self) -> str:
        return f"cutoff_{self.filter_type}"

    def evaluate(
        self, result: SimulationResult, params: ParameterSet
    ) -> float:
        """Return ``|f_c_actual − f_c_target|`` in Hz (lower is better).

        Returns a large penalty (1e12) if no -3 dB crossing is found.
        """
        if self.target_signal not in result.signals:
            logger.error(
                "CutoffObjective: signal %s not found. Available: %s",
                self.target_signal,
                list(result.signals.keys()),
            )
            return float("inf")

        freq = result.time_or_freq
        mag_db = _magnitude_db(result.signals[self.target_signal])

        if self.filter_type == "lowpass":
            cutoff = self._find_lowpass_cutoff(freq, mag_db)
        else:
            cutoff = self._find_highpass_cutoff(freq, mag_db)

        if cutoff is None:
            logger.warning(
                "CutoffObjective (%s): no -3 dB crossing found – penalty applied",
                self.filter_type,
            )
            return 1e12

        score = abs(cutoff - self.target_cutoff_hz)
        logger.debug(
            "CutoffObjective (%s): f_c_actual=%.3g Hz  f_c_target=%.3g Hz  score=%.4g",
            self.filter_type,
            cutoff,
            self.target_cutoff_hz,
            score,
        )
        return score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_lowpass_cutoff(
        freq: np.ndarray, mag_db: np.ndarray
    ) -> Optional[float]:
        """Find the first -3 dB frequency on a low-pass response.

        Passband level is estimated as the median of the lowest-frequency
        10 % of samples.  The cutoff is the first point where the
        magnitude drops to passband − 3 dB.
        """
        if len(freq) < 2:
            return None

        n_pb = max(1, len(mag_db) // 10)
        passband_db = float(np.median(mag_db[:n_pb]))
        threshold_db = passband_db - 3.0

        below = mag_db <= threshold_db
        if not np.any(below):
            return None

        idx = int(np.argmax(below))  # first True
        if idx == 0:
            return float(freq[0])

        return _log_interp_freq(
            float(freq[idx - 1]), float(freq[idx]),
            float(mag_db[idx - 1]), float(mag_db[idx]),
            threshold_db,
        )

    @staticmethod
    def _find_highpass_cutoff(
        freq: np.ndarray, mag_db: np.ndarray
    ) -> Optional[float]:
        """Find the -3 dB frequency on a high-pass response.

        Passband level is estimated as the median of the highest-frequency
        10 % of samples.  The cutoff is the lowest frequency where the
        magnitude rises to within 3 dB of the passband (scanning from
        low to high frequency).
        """
        if len(freq) < 2:
            return None

        n_pb = max(1, len(mag_db) // 10)
        passband_db = float(np.median(mag_db[-n_pb:]))
        threshold_db = passband_db - 3.0

        # Find the last index still below threshold (scanning low → high)
        below = mag_db <= threshold_db
        if not np.any(below):
            return None  # entire response is above threshold

        # Last True index = right edge of the stop-band
        idx = int(np.where(below)[0][-1])
        if idx >= len(freq) - 1:
            return float(freq[-1])

        return _log_interp_freq(
            float(freq[idx]), float(freq[idx + 1]),
            float(mag_db[idx]), float(mag_db[idx + 1]),
            threshold_db,
        )


# ---------------------------------------------------------------------------
# BandwidthObjective  –  Band-pass filters
# ---------------------------------------------------------------------------


class BandwidthObjective(ObjectiveFunction):
    """Penalise deviation of the -3 dB bandwidth from a target value.

    **Definition (Band-pass filter)**:
    Bandwidth is the total width of the passband, defined as the
    difference between the upper and lower -3 dB points::

        BW = f_high − f_low

    where f_high and f_low are the frequencies where the response falls
    3 dB below the peak passband level on the upper and lower skirts
    respectively.  In other words, at both f_low and f_high::

        |H(f)|_dB = |H_peak|_dB − 3 dB

    Parameters
    ----------
    target_signal:
        Signal to analyse, e.g. ``"V(out)"``.
    target_bw_hz:
        Desired bandwidth (f_high − f_low) in Hz.
    """

    def __init__(
        self,
        target_signal: str,
        target_bw_hz: float,
    ) -> None:
        self.target_signal = target_signal
        self.target_bw_hz = float(target_bw_hz)

    @property
    def name(self) -> str:
        return "bandwidth_bpf"

    def evaluate(
        self, result: SimulationResult, params: ParameterSet
    ) -> float:
        """Return ``|BW_actual − BW_target|`` in Hz (lower is better).

        Returns a large penalty if the -3 dB bandwidth cannot be
        determined (e.g. response never falls 3 dB below peak).
        """
        if self.target_signal not in result.signals:
            logger.error(
                "BandwidthObjective: signal %s not found. Available: %s",
                self.target_signal,
                list(result.signals.keys()),
            )
            return float("inf")

        freq = result.time_or_freq
        mag_db = _magnitude_db(result.signals[self.target_signal])

        bw, f_low, f_high = self._find_bandwidth(freq, mag_db)
        if bw is None:
            logger.warning(
                "BandwidthObjective: could not find both -3 dB points – penalty applied"
            )
            return 1e12

        score = abs(bw - self.target_bw_hz)
        logger.debug(
            "BandwidthObjective: f_low=%.3g Hz  f_high=%.3g Hz  "
            "BW=%.3g Hz  target=%.3g Hz  score=%.4g",
            f_low,
            f_high,
            bw,
            self.target_bw_hz,
            score,
        )
        return score

    @staticmethod
    def _find_bandwidth(
        freq: np.ndarray, mag_db: np.ndarray
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Return (bandwidth_hz, f_low_3dB, f_high_3dB) for a band-pass response.

        Algorithm
        ---------
        1. Locate the passband peak (maximum magnitude).
        2. Set threshold = peak_db − 3 dB.
        3. f_low  = highest frequency *below* the peak where mag crosses
                    the threshold (lower skirt).
        4. f_high = lowest  frequency *above* the peak where mag crosses
                    the threshold (upper skirt).
        5. BW = f_high − f_low.

        Returns ``(None, None, None)`` if either crossing cannot be found.
        """
        if len(freq) < 3:
            return None, None, None

        peak_idx = int(np.argmax(mag_db))
        peak_db = float(mag_db[peak_idx])
        threshold_db = peak_db - 3.0

        # --- Lower -3 dB crossing (left of peak) ---
        f_low: Optional[float] = None
        left_seg = mag_db[:peak_idx + 1]  # inclusive of peak
        below_left = left_seg <= threshold_db
        if np.any(below_left):
            # Last index still below threshold on the left side
            last_below = int(np.where(below_left)[0][-1])
            if last_below < peak_idx:
                f_low = _log_interp_freq(
                    float(freq[last_below]), float(freq[last_below + 1]),
                    float(mag_db[last_below]), float(mag_db[last_below + 1]),
                    threshold_db,
                )
            else:
                f_low = float(freq[0])
        else:
            # Entire left side is above threshold – edge of sweep is f_low
            f_low = float(freq[0])

        # --- Upper -3 dB crossing (right of peak) ---
        f_high: Optional[float] = None
        right_seg = mag_db[peak_idx:]  # from peak onward
        below_right = right_seg <= threshold_db
        if np.any(below_right):
            # First index below threshold on the right side
            first_below = int(np.argmax(below_right))  # index within right_seg
            abs_idx = peak_idx + first_below
            if abs_idx > 0:
                f_high = _log_interp_freq(
                    float(freq[abs_idx - 1]), float(freq[abs_idx]),
                    float(mag_db[abs_idx - 1]), float(mag_db[abs_idx]),
                    threshold_db,
                )
            else:
                f_high = float(freq[abs_idx])
        else:
            # Entire right side is above threshold – edge of sweep is f_high
            f_high = float(freq[-1])

        if f_low is None or f_high is None:
            return None, None, None

        bw = f_high - f_low
        return bw, f_low, f_high


# ---------------------------------------------------------------------------
# MSEObjective
# ---------------------------------------------------------------------------


class MSEObjective(ObjectiveFunction):
    """Mean-squared error between a simulation signal and a target curve.

    Parameters
    ----------
    target_signal:
        Name of the signal to compare, e.g. ``"V(out)"``.
    target_values:
        Array of target magnitudes (dB).  Must be the same length as the
        simulation frequency / time vector, or will be resampled.
    freq_weights:
        Optional per-frequency weighting array (same length as
        *target_values*).  If ``None``, all frequencies are weighted
        equally.
    """

    def __init__(
        self,
        target_signal: str,
        target_values: np.ndarray,
        freq_weights: Optional[np.ndarray] = None,
    ) -> None:
        self.target_signal = target_signal
        self.target_values = np.asarray(target_values, dtype=float)
        self.freq_weights = (
            np.asarray(freq_weights, dtype=float)
            if freq_weights is not None
            else None
        )

    @property
    def name(self) -> str:
        return "mse"

    def evaluate(
        self, result: SimulationResult, params: ParameterSet
    ) -> float:
        """Return weighted MSE between simulated and target curves (dB scale)."""
        if self.target_signal not in result.signals:
            logger.error(
                "MSEObjective: signal %s not in result. Available: %s",
                self.target_signal,
                list(result.signals.keys()),
            )
            return float("inf")

        sim_mag_db = _magnitude_db(result.signals[self.target_signal])

        target = self.target_values
        if len(target) != len(sim_mag_db):
            logger.debug(
                "MSEObjective: resampling target from %d to %d points",
                len(target),
                len(sim_mag_db),
            )
            x_target = np.linspace(0, 1, len(target))
            x_sim = np.linspace(0, 1, len(sim_mag_db))
            target = np.interp(x_sim, x_target, target)

        diff = sim_mag_db - target
        if self.freq_weights is not None:
            weights = self.freq_weights
            if len(weights) != len(diff):
                weights = np.interp(
                    np.linspace(0, 1, len(diff)),
                    np.linspace(0, 1, len(self.freq_weights)),
                    self.freq_weights,
                )
            mse = float(np.sum(weights * diff ** 2) / np.sum(weights))
        else:
            mse = float(np.mean(diff ** 2))

        logger.debug("MSEObjective: score = %.6f", mse)
        return mse


# ---------------------------------------------------------------------------
# PeakObjective
# ---------------------------------------------------------------------------


class PeakObjective(ObjectiveFunction):
    """Penalise deviation from a desired peak magnitude at a target frequency.

    Parameters
    ----------
    target_signal:
        Signal to analyse.
    target_peak_db:
        Desired peak magnitude in dB.
    target_freq_hz:
        Frequency (Hz) at which the peak should occur.
    freq_tolerance_decades:
        Search window (in decades) around *target_freq_hz* (default 0.5).
    """

    def __init__(
        self,
        target_signal: str,
        target_peak_db: float,
        target_freq_hz: float,
        freq_tolerance_decades: float = 0.5,
    ) -> None:
        self.target_signal = target_signal
        self.target_peak_db = float(target_peak_db)
        self.target_freq_hz = float(target_freq_hz)
        self.freq_tolerance_decades = freq_tolerance_decades

    @property
    def name(self) -> str:
        return "peak"

    def evaluate(
        self, result: SimulationResult, params: ParameterSet
    ) -> float:
        """Return combined error: |peak_db_error| + |peak_freq_error| (decades)."""
        if self.target_signal not in result.signals:
            logger.error(
                "PeakObjective: signal %s not found.", self.target_signal
            )
            return float("inf")

        freq = result.time_or_freq
        mag_db = _magnitude_db(result.signals[self.target_signal])

        peak_idx = int(np.argmax(mag_db))
        actual_peak_db = float(mag_db[peak_idx])
        actual_peak_freq = float(freq[peak_idx])

        db_error = abs(actual_peak_db - self.target_peak_db)

        if actual_peak_freq > 0 and self.target_freq_hz > 0:
            freq_error = abs(
                np.log10(actual_peak_freq) - np.log10(self.target_freq_hz)
            )
        else:
            freq_error = abs(actual_peak_freq - self.target_freq_hz) / max(
                1.0, self.target_freq_hz
            )

        score = db_error + freq_error
        logger.debug(
            "PeakObjective: actual=%.2f dB @ %.3g Hz  target=%.2f dB @ %.3g Hz  score=%.4f",
            actual_peak_db,
            actual_peak_freq,
            self.target_peak_db,
            self.target_freq_hz,
            score,
        )
        return score


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_objective(config: dict) -> ObjectiveFunction:
    """Create an :class:`ObjectiveFunction` from a config dictionary.

    Config keys by type
    -------------------

    ``type: cutoff``  (low-pass / high-pass)
        - ``target_signal``   – e.g. ``"V(out)"``
        - ``target_cutoff_hz`` – desired -3 dB cutoff in Hz
        - ``filter_type``     – ``"lowpass"`` (default) or ``"highpass"``

    ``type: bandwidth``  (band-pass)
        - ``target_signal``   – e.g. ``"V(out)"``
        - ``target_bw_hz``    – desired bandwidth (f_high − f_low) in Hz

    ``type: mse``
        - ``target_signal``
        - ``target_values``   – list of target dB values
        - ``freq_weights``    – optional list of per-frequency weights

    ``type: peak``
        - ``target_signal``
        - ``target_peak_db``  – desired peak magnitude in dB
        - ``target_freq_hz``  – desired peak frequency in Hz
        - ``freq_tolerance_decades`` – search window (default 0.5)

    Returns
    -------
    ObjectiveFunction

    Raises
    ------
    ValueError
        For unknown type or missing required keys.
    """
    obj_type = config.get("type", "").lower()
    signal = config.get("target_signal", "V(out)")

    if obj_type == "cutoff":
        cutoff = config.get("target_cutoff_hz")
        if cutoff is None:
            raise ValueError(
                "CutoffObjective requires 'target_cutoff_hz' in config"
            )
        filter_type = config.get("filter_type", "lowpass")
        logger.debug(
            "Creating CutoffObjective: signal=%s  filter=%s  f_c=%.3g Hz",
            signal,
            filter_type,
            float(cutoff),
        )
        return CutoffObjective(
            target_signal=signal,
            target_cutoff_hz=float(cutoff),
            filter_type=filter_type,
        )

    elif obj_type == "bandwidth":
        bw = config.get("target_bw_hz")
        if bw is None:
            raise ValueError(
                "BandwidthObjective requires 'target_bw_hz' in config"
            )
        logger.debug(
            "Creating BandwidthObjective (BPF): signal=%s  target_bw=%.3g Hz",
            signal,
            float(bw),
        )
        return BandwidthObjective(
            target_signal=signal,
            target_bw_hz=float(bw),
        )

    elif obj_type == "mse":
        target_values = config.get("target_values")
        if target_values is None:
            raise ValueError("MSEObjective requires 'target_values' in config")
        freq_weights = config.get("freq_weights")
        logger.debug("Creating MSEObjective: signal=%s", signal)
        return MSEObjective(
            target_signal=signal,
            target_values=np.asarray(target_values, dtype=float),
            freq_weights=(
                np.asarray(freq_weights, dtype=float)
                if freq_weights is not None
                else None
            ),
        )

    elif obj_type == "peak":
        target_peak_db = config.get("target_peak_db")
        target_freq_hz = config.get("target_freq_hz")
        if target_peak_db is None or target_freq_hz is None:
            raise ValueError(
                "PeakObjective requires 'target_peak_db' and 'target_freq_hz' in config"
            )
        logger.debug("Creating PeakObjective: signal=%s", signal)
        return PeakObjective(
            target_signal=signal,
            target_peak_db=float(target_peak_db),
            target_freq_hz=float(target_freq_hz),
            freq_tolerance_decades=float(
                config.get("freq_tolerance_decades", 0.5)
            ),
        )

    else:
        raise ValueError(
            f"Unknown objective type: {obj_type!r}. "
            f"Valid choices: 'cutoff', 'bandwidth', 'mse', 'peak'."
        )
