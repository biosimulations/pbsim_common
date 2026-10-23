"""Adapter bugs found by compose-api's 300-model BioModels suite (2026-10-22). Each step is fed its own initial state
as input, as a process-bigraph document wiring `species_concentrations` does."""

import math
from pathlib import Path

import numpy
import pytest
import tellurium as te
from bigraph_schema import Core

from pbsim_common.simulators import CopasiUTCStep, TelluriumUTCStep

MODELS = Path(__file__).resolve().parents[1] / "resources" / "biomodels"


def _config(model: str, time: float = 10.0, n_points: int = 11) -> dict[str, object]:
    return {"model_source": str(MODELS / f"{model}.xml"), "time": time, "n_points": n_points, "output_dir": ""}


@pytest.mark.parametrize(
    ("model", "species", "reactions"),
    [
        ("BIOMD0000000020", 0, 0),  # only rate rules on parameters
        ("BIOMD0000000241", 5, 0),  # species, no reactions
        ("BIOMD0000000143", 20, 20),  # display names repeat across compartments
    ],
)
def test_copasi_runs_models_without_species_reactions_or_unique_names(
    fully_registered_core: Core, model: str, species: int, reactions: int
) -> None:
    step = CopasiUTCStep(_config(model), core=fully_registered_core)
    assert (len(step.species_ids), len(step.reaction_ids)) == (species, reactions)
    assert len(set(step.species_ids)) == species  # one entry per species, keyed by SBML id
    state = step.initial_state()
    assert sorted(state["species_concentrations"]) == sorted(step.species_ids)
    result = step.update(state)["result"]
    assert len(result["time"]) == 11
    assert all(math.isfinite(v) for row in result["values"] for v in row)


def test_tellurium_skips_species_an_assignment_rule_defines(fully_registered_core: Core) -> None:
    step = TelluriumUTCStep(_config("BIOMD0000000007"), core=fully_registered_core)
    assert "Cig2Total" in step.species_ids  # defined by an assignment rule: it must not be set
    result = step.update(step.initial_state())["result"]
    assert len(result["time"]) == 11


def test_tellurium_writes_concentrations_back_as_concentrations(fully_registered_core: Core) -> None:
    """In a compartment of volume 1e-16, setting a concentration as an amount moved the run by ~1e11."""
    step = TelluriumUTCStep(_config("BIOMD0000000001", time=100.0), core=fully_registered_core)
    result = step.update(step.initial_state())["result"]
    untouched = te.loadSBMLModel(str(MODELS / "BIOMD0000000001.xml")).simulate(0, 100.0, 11)
    columns = [c.strip("[]") for c in untouched.colnames[1:]]
    ours = numpy.array(result["values"])[:, [result["columns"].index(c) for c in columns]]
    numpy.testing.assert_allclose(ours, numpy.asarray(untouched)[:, 1:], rtol=1e-6, atol=1e-12)
