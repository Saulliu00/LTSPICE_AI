"""
core/ltspice_runner.py
~~~~~~~~~~~~~~~~~~~~~~
Thin wrapper around the LTspice executable.

Responsibilities
----------------
- Launch LTspice in batch (headless) mode via ``subprocess``.
- Locate the ``.raw`` output file produced by the run.
- Implement configurable retry logic.
- Raise :class:`SimulationError` on persistent failure.

Usage example::

    runner = LTSpiceRunner("/Applications/LTspice.app/Contents/MacOS/LTspice")
    raw_path = runner.run("circuits/example_rc_filter.asc")
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class SimulationError(RuntimeError):
    """Raised when LTspice fails to produce a valid ``.raw`` file.

    Parameters
    ----------
    message:
        Human-readable description of the failure.
    schematic_path:
        Path to the schematic that triggered the failure.
    returncode:
        Process return code from the last attempt (``None`` if the process
        could not be launched at all).
    """

    def __init__(
        self,
        message: str,
        schematic_path: str = "",
        returncode: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.schematic_path = schematic_path
        self.returncode = returncode


# ---------------------------------------------------------------------------
# Main runner class
# ---------------------------------------------------------------------------


class LTSpiceRunner:
    """Run LTspice simulations in batch mode.

    Parameters
    ----------
    executable_path:
        Full path to the LTspice binary.  Examples:

        - macOS: ``/Applications/LTspice.app/Contents/MacOS/LTspice``
        - Windows: ``C:/Program Files/LTC/LTspiceXVII/XVIIx64.exe``
        - Linux (Wine): ``wine ~/.wine/drive_c/.../XVIIx64.exe``
    timeout:
        Maximum number of seconds to wait for a single simulation attempt
        before killing the subprocess.  Default: ``60``.
    retry_count:
        How many times to retry a failed simulation before raising
        :class:`SimulationError`.  Default: ``3``.
    """

    # CLI flags used by LTspice XVII on each platform.
    #
    # macOS / Linux: -b alone runs headlessly and exits when done.
    #   Adding -Run overrides -b and opens the GUI, causing a timeout.
    # Windows: -Run is required to trigger execution; -b suppresses dialogs.
    _BATCH_FLAGS: dict[str, List[str]] = {
        "darwin": ["-b"],
        "win32":  ["-Run", "-b"],
        "linux":  ["-b"],
    }

    def __init__(
        self,
        executable_path: str,
        timeout: int = 60,
        retry_count: int = 3,
    ) -> None:
        self.executable_path = str(executable_path)
        self.timeout = timeout
        self.retry_count = max(1, retry_count)
        logger.debug(
            "LTSpiceRunner initialised: exe=%s  timeout=%ds  retries=%d",
            self.executable_path,
            self.timeout,
            self.retry_count,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, schematic_path: str) -> str:
        """Run LTspice on *schematic_path* and return the ``.raw`` file path.

        The method retries up to :attr:`retry_count` times with a short
        back-off delay between attempts.

        Parameters
        ----------
        schematic_path:
            Path to the ``.asc`` or ``.net`` file.

        Returns
        -------
        str
            Absolute path to the generated ``.raw`` output file.

        Raises
        ------
        FileNotFoundError
            If *schematic_path* does not exist.
        SimulationError
            If LTspice fails on every retry attempt.
        """
        schematic_path = str(Path(schematic_path).resolve())

        if not Path(schematic_path).exists():
            raise FileNotFoundError(
                f"Schematic not found: {schematic_path}"
            )

        last_error: Optional[Exception] = None

        for attempt in range(1, self.retry_count + 1):
            logger.info(
                "LTspice run attempt %d/%d  schematic=%s",
                attempt,
                self.retry_count,
                schematic_path,
            )
            try:
                self._execute(schematic_path)
                raw_path = self._find_raw_file(schematic_path)
                logger.info("Simulation succeeded  raw=%s", raw_path)
                return raw_path
            except (SimulationError, FileNotFoundError) as exc:
                last_error = exc
                logger.warning(
                    "Attempt %d failed: %s", attempt, exc
                )
                if attempt < self.retry_count:
                    delay = 2 ** (attempt - 1)  # 1s, 2s, 4s …
                    logger.debug("Retrying in %.1fs …", delay)
                    time.sleep(delay)

        raise SimulationError(
            f"LTspice failed after {self.retry_count} attempt(s): {last_error}",
            schematic_path=schematic_path,
        ) from last_error

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute(self, schematic_path: str) -> None:
        """Launch the LTspice process and wait for completion.

        Parameters
        ----------
        schematic_path:
            Absolute path to the schematic.

        Raises
        ------
        SimulationError
            If the process exits with a non-zero return code or times out.
        """
        platform_key = sys.platform if sys.platform in self._BATCH_FLAGS else "linux"
        flags = self._BATCH_FLAGS[platform_key]

        cmd: List[str] = [self.executable_path] + flags + [schematic_path]
        logger.info("LTspice command: %s", " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                check=False,  # we handle return code ourselves
            )
        except subprocess.TimeoutExpired as exc:
            raise SimulationError(
                f"LTspice timed out after {self.timeout}s",
                schematic_path=schematic_path,
            ) from exc
        except FileNotFoundError as exc:
            raise SimulationError(
                f"LTspice executable not found: {self.executable_path}",
                schematic_path=schematic_path,
            ) from exc
        except OSError as exc:
            raise SimulationError(
                f"OS error launching LTspice: {exc}",
                schematic_path=schematic_path,
            ) from exc

        stdout = proc.stdout.decode(errors="replace").strip()
        stderr = proc.stderr.decode(errors="replace").strip()

        if stdout:
            logger.info("LTspice stdout:\n%s", stdout)
        if stderr:
            logger.info("LTspice stderr:\n%s", stderr)

        if proc.returncode != 0:
            logger.error(
                "LTspice failed (code %d).\n  command : %s\n  stdout  : %s\n  stderr  : %s",
                proc.returncode,
                " ".join(cmd),
                stdout or "(empty)",
                stderr or "(empty)",
            )
            raise SimulationError(
                f"LTspice exited with code {proc.returncode}. "
                f"stdout: {stdout[:300]}  stderr: {stderr[:300]}",
                schematic_path=schematic_path,
                returncode=proc.returncode,
            )

    def _find_raw_file(self, schematic_path: str) -> str:
        """Locate the ``.raw`` file produced by the simulation.

        LTspice places the ``.raw`` file in the same directory as the
        schematic, with the same base name.

        Parameters
        ----------
        schematic_path:
            Absolute path to the schematic.

        Returns
        -------
        str
            Absolute path to the ``.raw`` file.

        Raises
        ------
        SimulationError
            If no ``.raw`` file can be found.
        """
        base = Path(schematic_path).with_suffix("")

        # LTspice may produce several raw-file variants; try common suffixes.
        candidates = [
            base.with_suffix(".raw"),
            base.with_suffix(".RAW"),
            # Some versions produce a .op.raw for operating-point sims
            Path(str(base) + ".op.raw"),
        ]

        for candidate in candidates:
            if candidate.exists():
                logger.debug("Found raw file: %s", candidate)
                return str(candidate)

        raise SimulationError(
            f"No .raw file found after simulation.  "
            f"Searched: {[str(c) for c in candidates]}",
            schematic_path=schematic_path,
        )
