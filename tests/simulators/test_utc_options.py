"""The optional settings of CopasiUTCStep and TelluriumUTCStep (pbsim_common.options), named as viva-copasi and
viva-tellurium name them: each leaves the step as it was when unset."""

import math
from pathlib import Path
from typing import Any

import pytest
from bigraph_schema import Core

from pbsim_common.options import UTCOptions
from pbsim_common.simulators.copasi_process import CopasiUTCStep
from pbsim_common.simulators.tellurium_process import TelluriumUTCStep
from tests.fixtures.utils import root_dir_path

MODEL = str(Path(root_dir_path()) / "resources" / "phase_cycle.sbml")
STEPS = [CopasiUTCStep, TelluriumUTCStep]


def run(step_class: Any, core: Core, inputs: dict[str, Any] | None = None, **options: Any) -> dict[str, Any]:
    config = {"model_source": MODEL, "time": 10.0, "n_points": 11, "output_dir": "", **options}
    step = step_class(config, core=core)
    result: dict[str, Any] = step.update(inputs or {})["result"]
    return result


def column(result: dict[str, Any], name: str) -> list[float]:
    i = result["columns"].index(name)
    return [row[i] for row in result["values"]]


@pytest.mark.parametrize("step_class", STEPS)
def test_unset_options_leave_the_step_as_it_was(step_class: Any, fully_registered_core: Core) -> None:
    result = run(step_class, fully_registered_core)
    assert len(result["time"]) == 11 and math.isclose(result["time"][-1], 10.0)
    assert {"PX", "PY", "PZ", "X", "Y", "Z"} <= set(result["columns"])


@pytest.mark.parametrize("step_class", STEPS)
def test_output_times_replace_time_and_n_points(step_class: Any, fully_registered_core: Core) -> None:
    times = [0.0, 1.0, 2.5, 7.0]
    result = run(step_class, fully_registered_core, output_times=times, n_points=0)
    assert [round(t, 9) for t in result["time"]] == times
    # the same values as the uniform grid at the times both have
    uniform = run(step_class, fully_registered_core)
    assert math.isclose(column(result, "X")[3], column(uniform, "X")[7], rel_tol=1e-4)


@pytest.mark.parametrize("step_class", STEPS)
def test_selections_report_only_those_ids(step_class: Any, fully_registered_core: Core) -> None:
    result = run(step_class, fully_registered_core, selections=["Y", "PX"])
    assert result["columns"] == ["Y", "PX"]
    assert len(result["values"][0]) == 2


@pytest.mark.parametrize("step_class", STEPS)
def test_an_unknown_selection_names_what_there_is(step_class: Any, fully_registered_core: Core) -> None:
    with pytest.raises(ValueError, match=r"\['nope'\] are not in the results"):
        run(step_class, fully_registered_core, selections=["nope"])


@pytest.mark.parametrize("step_class", STEPS)
def test_tolerances_are_used(step_class: Any, fully_registered_core: Core) -> None:
    tight = run(step_class, fully_registered_core, absolute_tolerance=1e-12, relative_tolerance=1e-10)
    loose = run(step_class, fully_registered_core, absolute_tolerance=1e-2, relative_tolerance=1e-2)
    # same grid and columns; the values agree to the loose tolerance but are not identical
    assert tight["columns"] == loose["columns"]
    x_tight, x_loose = column(tight, "X"), column(loose, "X")
    assert all(math.isclose(a, b, rel_tol=0.1, abs_tol=0.1) for a, b in zip(x_tight, x_loose, strict=True))
    assert x_tight != x_loose


@pytest.mark.parametrize("step_class", STEPS)
def test_model_in_replaces_the_model(step_class: Any, fully_registered_core: Core) -> None:
    text = Path(MODEL).read_text(encoding="utf-8")
    # the same model with X starting at 5 (an amount): the run starts from the model on the port
    changed = text.replace('id="X" initialAmount="0"', 'id="X" initialAmount="5"', 1)
    assert changed != text
    assert column(run(step_class, fully_registered_core), "X")[0] == 0
    assert math.isclose(column(run(step_class, fully_registered_core, inputs={"model_in": changed}), "X")[0], 5.0)
    # an empty port keeps the configured model
    assert column(run(step_class, fully_registered_core, inputs={"model_in": ""}), "X")[0] == 0


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"output_times": [0.0, 2.0, 1.0]}, "ascending"),
        ({"output_times": [-1.0, 1.0]}, "ascending"),
        ({"absolute_tolerance": -1e-6}, "absolute_tolerance must be positive"),
    ],
)
def test_invalid_options_are_refused(options: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        UTCOptions.from_config(options, "Step")


def test_the_filled_defaults_of_maybe_are_unset() -> None:
    """bigraph-schema fills an unset maybe[list[...]] with [] and maybe[float] with 0.0."""
    filled = {"output_times": [], "selections": [], "absolute_tolerance": 0.0, "relative_tolerance": 0.0}
    assert UTCOptions.from_config(filled, "Step") == UTCOptions()
