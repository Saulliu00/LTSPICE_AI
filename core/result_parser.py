"""
core/result_parser.py
~~~~~~~~~~~~~~~~~~~~~
Parse LTspice ``.raw`` output files into :class:`~core.SimulationResult`
objects.

Parsing strategy (in priority order)
-------------------------------------
1. ``spicelib.RawRead``  – the preferred, well-tested library.
2. ``ltspice`` package   – alternative pure-Python reader.
3. Raw binary / ASCII    – minimal fallback that handles basic LTspice
   format without external dependencies (AC sweep only).

Usage example::

    parser = ResultParser(sim_type="ac")
    result = parser.parse("circuits/example_rc_filter.raw")
    mag_db = parser.get_magnitude_db(result, "V(out)")
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from core import SimulationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

_SPICELIB_RAWREAD: Optional[type] = None
_LTSPICE_RAWREAD: Optional[type] = None

try:
    from spicelib import RawRead as _SpicerRawRead  # type: ignore
    _SPICELIB_RAWREAD = _SpicerRawRead
    logger.debug("spicelib.RawRead available")
except ImportError:
    pass

if _SPICELIB_RAWREAD is None:
    try:
        from ltspice import Ltspice as _LtspiceRaw  # type: ignore
        _LTSPICE_RAWREAD = _LtspiceRaw
        logger.debug("ltspice.Ltspice available (fallback)")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Result parser
# ---------------------------------------------------------------------------


class ResultParser:
    """Parse LTspice ``.raw`` files into :class:`~core.SimulationResult`.

    Parameters
    ----------
    sim_type:
        Expected simulation type: ``"ac"``, ``"transient"``, or ``"dc"``.
        Used as a hint when auto-detection fails.
    """

    def __init__(self, sim_type: str = "ac") -> None:
        if sim_type not in {"ac", "transient", "dc"}:
            raise ValueError(
                f"sim_type must be 'ac', 'transient', or 'dc', got {sim_type!r}"
            )
        self.sim_type = sim_type

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def parse(self, raw_file_path: str) -> SimulationResult:
        """Parse a ``.raw`` file and return a :class:`~core.SimulationResult`.

        Parameters
        ----------
        raw_file_path:
            Absolute (or relative) path to the LTspice ``.raw`` file.

        Returns
        -------
        SimulationResult
            Populated result object.

        Raises
        ------
        FileNotFoundError
            If *raw_file_path* does not exist.
        ValueError
            If the file cannot be parsed by any available backend.
        """
        path = str(Path(raw_file_path).resolve())
        if not Path(path).exists():
            raise FileNotFoundError(f"Raw file not found: {path}")

        logger.info("Parsing raw file: %s", path)

        # Try each backend in order
        if _SPICELIB_RAWREAD is not None:
            try:
                return self._parse_spicelib(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("spicelib parse failed: %s – trying fallback", exc)

        if _LTSPICE_RAWREAD is not None:
            try:
                return self._parse_ltspice_pkg(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ltspice pkg parse failed: %s – trying fallback", exc)

        # Last resort: minimal built-in parser
        return self._parse_minimal(path)

    def get_signal(
        self, result: SimulationResult, signal_name: str
    ) -> np.ndarray:
        """Return the raw signal array for *signal_name*.

        Parameters
        ----------
        result:
            Parsed simulation result.
        signal_name:
            Exact signal name, e.g. ``"V(out)"``.

        Returns
        -------
        np.ndarray
            Complex array (AC) or real array (transient/DC).

        Raises
        ------
        KeyError
            If *signal_name* is not found in the result.
        """
        if signal_name not in result.signals:
            available = list(result.signals.keys())
            raise KeyError(
                f"Signal {signal_name!r} not found.  "
                f"Available signals: {available}"
            )
        return result.signals[signal_name]

    def get_magnitude_db(
        self, result: SimulationResult, signal_name: str
    ) -> np.ndarray:
        """Return the magnitude of *signal_name* in dB.

        Computes ``20 * log10(|signal|)``.  Works for both complex (AC)
        and real (transient) signals.

        Parameters
        ----------
        result:
            Parsed simulation result.
        signal_name:
            Signal name, e.g. ``"V(out)"``.

        Returns
        -------
        np.ndarray
            Magnitude in dB, same length as the x-axis.
        """
        sig = self.get_signal(result, signal_name)
        magnitude = np.abs(sig)
        # Avoid log(0)
        magnitude = np.where(magnitude == 0.0, np.finfo(float).tiny, magnitude)
        return 20.0 * np.log10(magnitude)

    def get_phase_deg(
        self, result: SimulationResult, signal_name: str
    ) -> np.ndarray:
        """Return the phase of *signal_name* in degrees.

        Parameters
        ----------
        result:
            Parsed simulation result.
        signal_name:
            Signal name.

        Returns
        -------
        np.ndarray
            Phase in degrees (0 for real-valued signals).
        """
        sig = self.get_signal(result, signal_name)
        if np.iscomplexobj(sig):
            return np.degrees(np.angle(sig))
        return np.zeros_like(sig, dtype=float)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _parse_spicelib(self, path: str) -> SimulationResult:
        """Parse using spicelib.RawRead."""
        raw = _SPICELIB_RAWREAD(path)  # type: ignore[operator]

        # Detect simulation type from the raw file header
        sim_type = self._detect_sim_type_spicelib(raw)

        # Extract x-axis
        x_trace = raw.get_trace("frequency") or raw.get_trace("time") or raw.get_trace("v-sweep")
        if x_trace is None:
            raise ValueError("Could not find x-axis trace in spicelib RawRead")
        x_data = np.asarray(x_trace.get_wave(0))

        # For AC, frequency is stored as complex in some versions
        if np.iscomplexobj(x_data):
            x_data = np.real(x_data)

        signals: Dict[str, np.ndarray] = {}
        for name in raw.get_trace_names():
            # Skip the axis traces themselves
            if name.lower() in {"frequency", "time", "v-sweep"}:
                continue
            try:
                wave = np.asarray(raw.get_trace(name).get_wave(0))
                signals[name] = wave
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not read trace %s: %s", name, exc)

        return SimulationResult(
            signals=signals,
            time_or_freq=x_data,
            sim_type=sim_type,
            raw_file=path,
        )

    def _detect_sim_type_spicelib(self, raw: object) -> str:
        """Infer sim type from spicelib raw object."""
        try:
            header = str(raw.get_raw_property("Plotname")).lower()
            if "ac" in header:
                return "ac"
            if "tran" in header:
                return "transient"
            if "dc" in header:
                return "dc"
        except Exception:  # noqa: BLE001
            pass
        return self.sim_type

    def _parse_ltspice_pkg(self, path: str) -> SimulationResult:
        """Parse using the ``ltspice`` package."""
        l = _LTSPICE_RAWREAD(path)  # type: ignore[operator]
        l.parse()

        # ltspice package exposes .t (time) or .f (frequency) and signal dicts
        if hasattr(l, "f") and l.f is not None:
            x_data = np.asarray(l.f)
            sim_type = "ac"
        elif hasattr(l, "t") and l.t is not None:
            x_data = np.asarray(l.t)
            sim_type = "transient"
        else:
            raise ValueError("ltspice package: cannot determine x-axis")

        signals: Dict[str, np.ndarray] = {}
        for name in l.variables:
            try:
                signals[name] = np.asarray(l.get_data(name))
            except Exception as exc:  # noqa: BLE001
                logger.debug("ltspice pkg: cannot read variable %s: %s", name, exc)

        return SimulationResult(
            signals=signals,
            time_or_freq=x_data,
            sim_type=sim_type,
            raw_file=path,
        )

    def _parse_minimal(self, path: str) -> SimulationResult:
        """Minimal built-in LTspice raw-file parser.

        Handles ASCII raw files (``Flag: ASCII``) and the common binary
        format for AC sweeps produced by LTspice XVII.

        This is a best-effort fallback; complex binary formats (multiple
        simulation steps, op-amp models, etc.) may not parse correctly.
        """
        with open(path, "rb") as fh:
            header_bytes = fh.read(4096)

        # Try ASCII parsing first
        try:
            header_text = header_bytes.decode("utf-8", errors="replace")
        except Exception:
            header_text = ""

        if "FLAGS: real" in header_text.upper() or "FLAGS: complex" in header_text.upper():
            return self._parse_ascii_raw(path)

        return self._parse_binary_raw(path)

    def _parse_ascii_raw(self, path: str) -> SimulationResult:
        """Parse a LTspice ASCII .raw file."""
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        variables: List[str] = []
        n_vars = 0
        n_points = 0
        is_complex = False
        in_data = False
        rows: List[List[float]] = []
        sim_type = self.sim_type

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("Plotname:"):
                plotname = stripped.split(":", 1)[1].strip().lower()
                if "ac" in plotname:
                    sim_type = "ac"
                elif "tran" in plotname:
                    sim_type = "transient"
                elif "dc" in plotname:
                    sim_type = "dc"

            elif stripped.startswith("Flags:") and "complex" in stripped.lower():
                is_complex = True

            elif stripped.startswith("No. Variables:"):
                n_vars = int(stripped.split(":")[1].strip())

            elif stripped.startswith("No. Points:"):
                n_points = int(stripped.split(":")[1].strip())

            elif stripped.startswith("Variables:"):
                in_data = False  # variables block follows

            elif stripped.startswith("Values:"):
                in_data = True

            elif in_data and stripped:
                # Each row starts with index (tab-separated values per variable)
                parts = stripped.split()
                try:
                    row = [float(p) for p in parts if _is_number(p)]
                    if row:
                        rows.append(row)
                except ValueError:
                    pass

        if not rows:
            raise ValueError("ASCII raw parser: no data rows found")

        data = np.array(rows)
        # First column is x-axis (freq or time)
        x_data = data[:, 0] if data.shape[1] > 0 else np.arange(len(data))

        signals: Dict[str, np.ndarray] = {}
        # We don't have variable names in this minimal parser, use generic names
        for col in range(1, data.shape[1]):
            signals[f"signal_{col}"] = data[:, col]

        return SimulationResult(
            signals=signals,
            time_or_freq=x_data,
            sim_type=sim_type,
            raw_file=path,
        )

    def _parse_binary_raw(self, path: str) -> SimulationResult:
        """Minimal binary LTspice raw parser for AC sweeps.

        Reads the ASCII header section, then interprets the binary data
        block as little-endian double (real) or complex128 values.
        """
        with open(path, "rb") as fh:
            raw_bytes = fh.read()

        # Find the boundary between header and data
        # LTspice uses the literal string "Binary:\n" or "Values:\n"
        binary_marker = b"Binary:\n"
        marker_pos = raw_bytes.find(binary_marker)
        if marker_pos == -1:
            raise ValueError("Binary raw parser: 'Binary:' marker not found")

        header_bytes = raw_bytes[:marker_pos]
        data_bytes = raw_bytes[marker_pos + len(binary_marker):]

        try:
            header = header_bytes.decode("utf-8", errors="replace")
        except Exception:
            header = ""

        # Parse header for n_vars, n_points, variable names, flags
        n_vars = 0
        n_points = 0
        is_complex = False
        sim_type = self.sim_type
        variable_names: List[str] = []
        in_variables_block = False

        for line in header.splitlines():
            stripped = line.strip()

            if stripped.lower().startswith("plotname:"):
                plotname = stripped.split(":", 1)[1].strip().lower()
                if "ac" in plotname:
                    sim_type = "ac"
                    is_complex = True
                elif "tran" in plotname:
                    sim_type = "transient"
                elif "dc" in plotname:
                    sim_type = "dc"

            elif stripped.lower().startswith("flags:"):
                flags = stripped.split(":", 1)[1].strip().lower()
                if "complex" in flags:
                    is_complex = True

            elif stripped.lower().startswith("no. variables:"):
                try:
                    n_vars = int(stripped.split(":")[1].strip())
                except ValueError:
                    pass

            elif stripped.lower().startswith("no. points:"):
                try:
                    n_points = int(stripped.split(":")[1].strip())
                except ValueError:
                    pass

            elif stripped.lower().startswith("variables:"):
                in_variables_block = True

            elif in_variables_block and stripped:
                parts = stripped.split()
                if len(parts) >= 2:
                    # Format: <index> <name> <type>
                    variable_names.append(parts[1])

        if n_vars == 0 or n_points == 0:
            raise ValueError(
                f"Binary raw parser: could not parse header "
                f"(n_vars={n_vars}, n_points={n_points})"
            )

        # Decode binary data
        if is_complex:
            # AC: x-axis is real double (8 bytes), remaining vars are complex128 (16 bytes)
            row_size = 8 + (n_vars - 1) * 16
            dtype_x = np.float64
            dtype_sig = np.complex128
        else:
            # Transient / DC: all values are real double (8 bytes)
            row_size = n_vars * 8
            dtype_x = np.float64
            dtype_sig = np.float64

        expected_bytes = row_size * n_points
        if len(data_bytes) < expected_bytes:
            logger.warning(
                "Binary raw: expected %d bytes, got %d – truncating n_points",
                expected_bytes,
                len(data_bytes),
            )
            n_points = len(data_bytes) // row_size

        x_data = np.empty(n_points, dtype=np.float64)
        raw_signals: Dict[int, List] = {i: [] for i in range(n_vars - 1)}

        offset = 0
        for _ in range(n_points):
            x_val = struct.unpack_from("<d", data_bytes, offset)[0]
            x_data[_] = x_val
            offset += 8

            for var_idx in range(n_vars - 1):
                if is_complex:
                    real, imag = struct.unpack_from("<dd", data_bytes, offset)
                    raw_signals[var_idx].append(complex(real, imag))
                    offset += 16
                else:
                    val = struct.unpack_from("<d", data_bytes, offset)[0]
                    raw_signals[var_idx].append(val)
                    offset += 8

        signals: Dict[str, np.ndarray] = {}
        for var_idx, values in raw_signals.items():
            # Match to name if we parsed variable names
            if var_idx + 1 < len(variable_names):
                name = variable_names[var_idx + 1]
            else:
                name = f"signal_{var_idx + 1}"
            signals[name] = np.array(values, dtype=dtype_sig)  # type: ignore[arg-type]

        return SimulationResult(
            signals=signals,
            time_or_freq=x_data,
            sim_type=sim_type,
            raw_file=path,
        )


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _is_number(s: str) -> bool:
    """Return True if *s* looks like a numeric string."""
    try:
        float(s)
        return True
    except ValueError:
        return False
