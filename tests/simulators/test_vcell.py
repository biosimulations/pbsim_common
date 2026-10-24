"""VCellUTCStep: VCell's ODE solver through pyvcell, checked against COPASI on the same models. Skipped without the
``vcell`` extra (pyvcell with libvcell and pyvcell-odesolver)."""

import importlib.util
import os
import tempfile
from pathlib import Path

import numpy
import pandas
import pytest
from bigraph_schema import Core

from pbsim_common.simulators import CopasiUTCStep, VCellUTCStep
from tests.fixtures.utils import root_dir_path, run_experiment

needs_vcell = pytest.mark.skipif(
    importlib.util.find_spec("libvcell") is None or importlib.util.find_spec("pyvcell_odesolver") is None,
    reason="needs the vcell extra (pyvcell[native,ode])",
)
RESOURCES = root_dir_path() / "resources"


def _refuses_systems_without_odes() -> bool:
    try:
        from pyvcell.vcml import vcml_ode_simulation
    except ModuleNotFoundError:
        return False
    return hasattr(vcml_ode_simulation, "_num_equations")


MODELS = RESOURCES / "biomodels"


def _config(model_source: Path, time: float = 10.0, n_points: int = 11) -> dict[str, object]:
    return {"model_source": str(model_source), "time": time, "n_points": n_points, "output_dir": ""}


def _assert_close(a: pandas.DataFrame, b: pandas.DataFrame, rel: float, columns: list[str]) -> None:
    for column in columns:
        scale = max(numpy.abs(b[column]).max(), 1e-12)
        worst = numpy.abs(a[column].to_numpy() - b[column].to_numpy()).max() / scale
        assert worst < rel, f"{column}: differs by {worst:.3g} of its range"


@needs_vcell
def test_vcell_matches_copasi_on_the_phase_cycle(fully_registered_core: Core) -> None:
    """The fixture's species are amounts (hasOnlySubstanceUnits), so this also checks amounts come back as amounts."""
    experiment = {
        "state": {
            "time_course": {
                "_type": "step",
                "address": "local:pbsim_common.simulators.vcell_process.VCellUTCStep",
                "config": {"model_source": str(RESOURCES / "phase_cycle.sbml"), "time": 10, "n_points": 51},
                "interval": 1.0,
                "inputs": {},
                "outputs": {},
            }
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        experiment["state"]["time_course"]["config"]["output_dir"] = tmpdir
        run_experiment(tmp_dir=tmpdir, pbg=experiment, core=fully_registered_core, interval=1)
        vcell = pandas.read_csv(os.path.join(tmpdir, "results.csv"))
    copasi = pandas.read_csv(RESOURCES / "copasi_report.csv")
    assert len(vcell) == len(copasi) == 51
    assert numpy.allclose(vcell["time"], numpy.linspace(0, 10, 51))
    species = ["PX", "PY", "PZ", "X", "Y", "Z"]
    assert sorted(c for c in vcell.columns if c != "time") == species
    _assert_close(vcell, copasi, rel=1e-4, columns=species)


@needs_vcell
@pytest.mark.parametrize("model", ["BIOMD0000000001", "BIOMD0000000143"])
def test_vcell_agrees_with_copasi_by_sbml_id(fully_registered_core: Core, model: str) -> None:
    """BIOMD0000000143's display names repeat across compartments of sizes 1 and 10: columns must be SBML ids, and
    concentrations must not be scaled by the volume."""
    vcell_step = VCellUTCStep(_config(MODELS / f"{model}.xml"), core=fully_registered_core)
    copasi_step = CopasiUTCStep(_config(MODELS / f"{model}.xml"), core=fully_registered_core)
    vcell_result = vcell_step.update(vcell_step.initial_state())["result"]
    copasi_result = copasi_step.update(copasi_step.initial_state())["result"]
    vcell = pandas.DataFrame(vcell_result["values"], columns=vcell_result["columns"])
    copasi = pandas.DataFrame(copasi_result["values"], columns=copasi_result["columns"])
    assert len(vcell_result["time"]) == 11
    assert set(vcell.columns) <= set(vcell_step.species_ids)
    assert set(vcell.columns) <= set(copasi.columns)
    _assert_close(vcell, copasi, rel=1e-3, columns=list(vcell.columns))


_UNITLESS_L3 = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="decay">
    <listOfCompartments><compartment id="cell" size="1" constant="true" spatialDimensions="3"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="cell" initialConcentration="10" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.5" constant="true"/></listOfParameters>
    <listOfReactions>
      <reaction id="R" reversible="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>A</ci><ci>cell</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


@needs_vcell
def test_a_model_without_units_is_read_in_sbml_level_2_units(fully_registered_core: Core, tmp_path: Path) -> None:
    """SBML Level 3 has no default units; pyvcell refuses such a model unless told what its numbers are."""
    model = tmp_path / "decay.xml"
    model.write_text(_UNITLESS_L3)
    step = VCellUTCStep(_config(model, time=4.0, n_points=5), core=fully_registered_core)
    result = step.update(step.initial_state())["result"]
    a = numpy.array(result["values"])[:, result["columns"].index("A")]
    assert numpy.allclose(a, 10 * numpy.exp(-0.5 * numpy.array(result["time"])), rtol=1e-5)


@needs_vcell
@pytest.mark.skipif(
    not _refuses_systems_without_odes(),
    reason="this pyvcell lets the solver crash the process on a system with no equations (fixed in virtualcell/pyvcell#66)",
)
def test_a_model_without_odes_fails_with_a_message(fully_registered_core: Core) -> None:
    """BIOMD0000000241's species never change (no reactions): VCell's math has no equations to solve."""
    step = VCellUTCStep(_config(MODELS / "BIOMD0000000241.xml"), core=fully_registered_core)
    with pytest.raises(RuntimeError, match="no ODEs to solve"):
        step.update(step.initial_state())
