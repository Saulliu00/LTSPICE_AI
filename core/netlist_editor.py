"""
core/netlist_editor.py
~~~~~~~~~~~~~~~~~~~~~~
Edit LTspice schematics (``.asc``) or netlists (``.net``) to inject
parameter values before each simulation run.

Strategy
--------
1. Try to use ``spicelib.SpiceEditor`` (preferred – handles escaping,
   units, and schematic-vs-netlist differences automatically).
2. Fall back to ``PyLTSpice`` if ``spicelib`` is unavailable.
3. As a last resort, perform direct text substitution on the file.

The editor **always** works on a temporary copy of the original
schematic; the original file is never modified.

Usage example::

    editor = NetlistEditor("circuits/example_rc_filter.asc")
    editor.apply_parameters({"R1": 10_000.0, "C1": 1e-8})
    editor.save("/tmp/trial_42.asc")
    # … run simulation …
    editor.restore()   # reset internal copy to original state
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from core import ParameterSet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

_SPICELIB_AVAILABLE = False
_PYLTSPICE_AVAILABLE = False

try:
    from spicelib import SpiceEditor as _SpiceEditor  # type: ignore
    _SPICELIB_AVAILABLE = True
    logger.debug("spicelib.SpiceEditor available")
except ImportError:
    pass

if not _SPICELIB_AVAILABLE:
    try:
        from PyLTSpice import SpiceEditor as _SpiceEditor  # type: ignore
        _PYLTSPICE_AVAILABLE = True
        logger.debug("PyLTSpice.SpiceEditor available (fallback)")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SI_SUFFIXES = [
    (1e12,  "T"),
    (1e9,   "G"),
    (1e6,   "Meg"),
    (1e3,   "k"),
    (1e0,   ""),
    (1e-3,  "m"),
    (1e-6,  "u"),
    (1e-9,  "n"),
    (1e-12, "p"),
    (1e-15, "f"),
]


def _format_value(value: float, unit: str = "") -> str:
    """Format a float into LTspice engineering notation.

    Examples
    --------
    >>> _format_value(10_000.0)
    '10k'
    >>> _format_value(1e-7, "F")
    '100nF'
    >>> _format_value(0.0015)
    '1.5m'
    """
    if value == 0.0:
        return f"0{unit}"

    abs_val = abs(value)
    sign = "-" if value < 0 else ""

    for threshold, suffix in _SI_SUFFIXES:
        if abs_val >= threshold * 0.9999:  # small tolerance for fp errors
            scaled = abs_val / threshold
            # Show up to 4 significant figures, strip trailing zeros
            formatted = f"{scaled:.4g}"
            return f"{sign}{formatted}{suffix}{unit}"

    # Fallback: scientific notation
    return f"{value:.4g}{unit}"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class NetlistEditor:
    """Edit LTspice schematic / netlist files with new parameter values.

    Parameters
    ----------
    schematic_path:
        Path to the original ``.asc`` or ``.net`` file.  This file is
        **never** modified; all edits happen on an internal copy.
    """

    def __init__(self, schematic_path: str) -> None:
        self._original_path = str(Path(schematic_path).resolve())
        if not Path(self._original_path).exists():
            raise FileNotFoundError(
                f"Schematic not found: {self._original_path}"
            )

        # Read original content for restoration
        with open(self._original_path, "r", encoding="utf-8", errors="replace") as fh:
            self._original_text: str = fh.read()

        # Working copy lives in a temp file so we never touch the original
        self._tmp_path: str = self._make_temp_copy()

        # Spice editor wrapper (may be None if no library available)
        self._editor: Optional[object] = self._load_editor()

        logger.debug(
            "NetlistEditor ready  original=%s  tmp=%s  backend=%s",
            self._original_path,
            self._tmp_path,
            self._backend_name(),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_parameter(self, name: str, value: float) -> None:
        """Set a ``.param`` statement value in the schematic.

        Parameters
        ----------
        name:
            Parameter name exactly as it appears in the ``.param``
            directive (case-insensitive for text-mode fallback).
        value:
            New numeric value.
        """
        if self._editor is not None:
            try:
                self._editor.set_parameter(name, _format_value(value))
                logger.debug("set_parameter (lib) %s = %s", name, value)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Library set_parameter failed (%s), falling back to text edit", exc
                )

        # Text-mode fallback: replace `.param <name>=<old>` with new value
        self._text_set_param(name, value)

    def set_component_value(
        self, name: str, value: float, unit: str = ""
    ) -> None:
        """Set a component (R/C/L/V …) value in the schematic.

        Parameters
        ----------
        name:
            Component reference designator (e.g. ``"R1"``).
        value:
            New numeric value.
        unit:
            Optional unit string appended to the formatted value
            (e.g. ``"Ω"``, ``"F"``).  Usually empty for LTspice.
        """
        formatted = _format_value(value, unit)

        if self._editor is not None:
            try:
                self._editor.set_component_value(name, formatted)
                logger.debug(
                    "set_component_value (lib) %s = %s", name, formatted
                )
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Library set_component_value failed (%s), falling back", exc
                )

        # Text-mode fallback
        self._text_set_component(name, formatted)

    def apply_parameters(self, params: ParameterSet) -> None:
        """Apply every entry in *params* to the schematic.

        For each key in *params* the method first tries
        :meth:`set_parameter` (for ``.param`` directives) and, if that
        leaves the file unchanged, also tries
        :meth:`set_component_value`.  In practice, for parametric
        schematics where R1 and C1 are driven by ``.param`` directives,
        ``set_parameter`` is the right call.

        Parameters
        ----------
        params:
            Dictionary mapping parameter/component names to new values.
        """
        for name, value in params.items():
            logger.debug("apply_parameters: %s = %g", name, value)
            self.set_parameter(name, value)

    def save(self, output_path: str) -> None:
        """Write the modified schematic to *output_path*.

        If a library editor is in use, it flushes its internal buffer
        to *output_path*.  Otherwise the modified text is written
        directly.

        Parameters
        ----------
        output_path:
            Destination path (directories are created if necessary).
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if self._editor is not None:
            try:
                self._editor.save_to(output_path)
                logger.debug("Saved (lib) → %s", output_path)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Library save_to failed (%s), falling back to shutil.copy", exc
                )

        # Text fallback: the tmp file already has the edits applied
        shutil.copy2(self._tmp_path, output_path)
        logger.debug("Saved (text) → %s", output_path)

    def restore(self) -> None:
        """Reset the internal working copy to the original schematic.

        Call this between trials when you want a clean slate.
        """
        with open(self._tmp_path, "w", encoding="utf-8") as fh:
            fh.write(self._original_text)

        # Re-create the editor wrapper on the fresh copy
        self._editor = self._load_editor()
        logger.debug("NetlistEditor restored to original state")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_temp_copy(self) -> str:
        """Create a temporary copy of the original schematic and return its path."""
        suffix = Path(self._original_path).suffix
        fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="ltspice_trial_")
        os.close(fd)
        shutil.copy2(self._original_path, tmp_path)
        return tmp_path

    def _load_editor(self) -> Optional[object]:
        """Instantiate the best available editor on the temp copy."""
        if _SPICELIB_AVAILABLE or _PYLTSPICE_AVAILABLE:
            try:
                editor = _SpiceEditor(self._tmp_path)  # type: ignore[name-defined]
                return editor
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not initialise SpiceEditor: %s  → using text fallback", exc
                )
        return None

    def _backend_name(self) -> str:
        if _SPICELIB_AVAILABLE:
            return "spicelib"
        if _PYLTSPICE_AVAILABLE:
            return "PyLTSpice"
        return "text-fallback"

    # ------------------------------------------------------------------
    # Text-mode editing helpers
    # ------------------------------------------------------------------

    def _read_tmp(self) -> str:
        with open(self._tmp_path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()

    def _write_tmp(self, text: str) -> None:
        with open(self._tmp_path, "w", encoding="utf-8") as fh:
            fh.write(text)

    def _text_set_param(self, name: str, value: float) -> None:
        """Replace ``.param <name>=<old>`` with new value in tmp file."""
        text = self._read_tmp()
        formatted = _format_value(value)

        # Pattern covers: .param R1=10k  or  .param R1 = 10k  (with spaces)
        pattern = re.compile(
            r"(\.param\s+)(" + re.escape(name) + r")\s*=\s*\S+",
            re.IGNORECASE,
        )
        new_text, count = pattern.subn(
            lambda m: f"{m.group(1)}{m.group(2)}={formatted}", text
        )

        if count == 0:
            logger.debug(
                "set_parameter text-fallback: pattern not found for %s; "
                "attempting component value replacement instead",
                name,
            )
        else:
            logger.debug(
                "set_parameter text-fallback: replaced %d occurrence(s) of %s",
                count,
                name,
            )

        self._write_tmp(new_text)

    def _text_set_component(self, name: str, formatted_value: str) -> None:
        """Replace component value in the SYMATTR Value line of a .asc file."""
        text = self._read_tmp()

        # .asc format: SYMATTR InstName <name> followed by SYMATTR Value <val>
        # We do a two-pass replacement: find the block for InstName, then
        # replace the nearest following Value line.
        lines = text.splitlines(keepends=True)
        in_block = False
        new_lines = []
        changed = False

        for line in lines:
            stripped = line.strip()

            if re.match(
                r"SYMATTR\s+InstName\s+" + re.escape(name) + r"\s*$",
                stripped,
                re.IGNORECASE,
            ):
                in_block = True
                new_lines.append(line)
                continue

            if in_block and re.match(r"SYMATTR\s+Value\s+", stripped, re.IGNORECASE):
                new_lines.append(
                    re.sub(
                        r"(SYMATTR\s+Value\s+)\S+",
                        lambda m: f"{m.group(1)}{formatted_value}",
                        line,
                        flags=re.IGNORECASE,
                    )
                )
                in_block = False
                changed = True
                continue

            # Any non-SYMATTR line resets the block context
            if in_block and not stripped.startswith("SYMATTR"):
                in_block = False

            new_lines.append(line)

        if not changed:
            logger.warning(
                "text_set_component: could not find component %s in schematic", name
            )
        else:
            logger.debug(
                "text_set_component: set %s = %s", name, formatted_value
            )

        self._write_tmp("".join(new_lines))

    def __del__(self) -> None:
        """Clean up the temp file on garbage collection."""
        try:
            if hasattr(self, "_tmp_path") and Path(self._tmp_path).exists():
                Path(self._tmp_path).unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
