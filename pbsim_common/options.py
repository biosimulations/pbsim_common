"""Optional settings of the uniform time course steps, named as viva-copasi and viva-tellurium name them.

pbsim-common's ``CopasiUTCStep`` and ``TelluriumUTCStep`` are the standard COPASI and Tellurium steps compose-api runs;
viva-copasi and viva-tellurium are alternatives with more settings. These are the ones a SED-ML or composite author
reaches for, given the same names and meanings so a document moves between the two families unchanged:

- ``output_times``: the times to report, in place of ``time`` and ``n_points`` (from 0, ascending);
- ``selections``: the ids to report (``time`` is always reported), in place of every one;
- ``absolute_tolerance``, ``relative_tolerance``: the integrator's tolerances, in place of its defaults;
- the ``model_in`` input port: SBML text that replaces the configured model when it carries a different one.

Each defaults to None, which leaves a step exactly as it was. bigraph-schema fills an unset ``maybe`` with its type's
empty value (``[]``, ``0.0``), so an empty list and a tolerance of 0 are unset too: neither is a usable setting.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from pandas import DataFrame


@dataclass(frozen=True)
class UTCOptions:
    output_times: list[float] | None = None
    selections: list[str] | None = None
    absolute_tolerance: float | None = None
    relative_tolerance: float | None = None

    @property
    def writes_own_csv(self) -> bool:
        """Whether the step writes ``results.csv`` itself (what it reports differs from the engine's full run)."""
        return self.output_times is not None or self.selections is not None

    @classmethod
    def from_config(cls, config: Mapping[str, Any], step: str) -> UTCOptions:
        times = config.get("output_times") or None
        if times is not None:
            times = [float(t) for t in times]
            if times[0] < 0 or any(b <= a for a, b in zip(times, times[1:], strict=False)):
                msg = f"{step}: output_times must be ascending from 0 or later, got {times}"
                raise ValueError(msg)
        selections = config.get("selections") or None
        if selections is not None:
            selections = [str(s) for s in selections]
        return cls(
            output_times=times,
            selections=selections,
            absolute_tolerance=_maybe_positive(config.get("absolute_tolerance"), "absolute_tolerance", step),
            relative_tolerance=_maybe_positive(config.get("relative_tolerance"), "relative_tolerance", step),
        )


def _maybe_positive(value: Any, name: str, step: str) -> float | None:
    if value is None or float(value) == 0:
        return None
    number = float(value)
    if number < 0:
        msg = f"{step}: {name} must be positive, got {number}"
        raise ValueError(msg)
    return number


def select_columns(tc: DataFrame, selections: list[str] | None, strip: str = "") -> DataFrame:
    """The time course's ``time`` column and the selected ids' columns, in the order selected. A column matches an id
    with ``strip``'s characters removed from its ends (Tellurium's ``[S1]`` is ``S1``). An id with no column is an
    error naming what there is."""
    if selections is None:
        return tc
    by_id = {str(c).strip(strip): c for c in tc.columns}
    missing = [s for s in selections if s not in by_id]
    if missing:
        msg = f"selections {missing} are not in the results; they have {sorted(i for i in by_id if i != 'time')}"
        raise ValueError(msg)
    keep = ([by_id["time"]] if "time" in by_id else []) + [by_id[s] for s in selections if s != "time"]
    return tc[keep]


def model_in_text(inputs: Mapping[str, Any], loaded: str | None) -> str | None:
    """The SBML text on the ``model_in`` port when it is one to load: not empty, and not the model already loaded."""
    text = inputs.get("model_in")
    if not isinstance(text, str) or not text.strip() or text == loaded:
        return None
    return text


@contextmanager
def model_in_file(text: str) -> Iterator[str]:
    """``text`` in a temporary ``.xml`` file, for engines (and libsbml) that read a model from a path."""
    handle, path = tempfile.mkstemp(suffix=".xml", prefix="model_in_")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            f.write(text)
        yield path
    finally:
        os.unlink(path)
