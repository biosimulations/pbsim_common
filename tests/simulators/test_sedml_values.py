"""Each step reports what SED-ML reads (pbsim_common.sbml_values), found by compose-api's sbml-test-suite-sample:
a species with only substance units as its amount, and a parameter a rule changes, under its SBML id.

The model: A (only substance units, 4 in compartment C of size 2) -> B at rate k*A with k = 1, so A's amount is
4*exp(-t) and B's concentration (4 - A)/2; P has a rate rule dP/dt = 1, Q an assignment rule Q = 2P."""

import math
from pathlib import Path

import pytest
from bigraph_schema import Core

from pbsim_common.sbml_values import read_sedml_values
from pbsim_common.simulators import CopasiUTCStep, TelluriumUTCStep

MODEL = str(Path(__file__).resolve().parents[1] / "resources" / "sedml_values.xml")


def test_the_model_s_sedml_values_are_read() -> None:
    values = read_sedml_values(MODEL)
    assert values.amount_species == frozenset({"A"})
    assert values.compartment_of == {"A": "C", "B": "C"}
    assert (values.varying_parameters, values.varying_compartments) == (("P", "Q"), ())
    assert values.sizes == {"C": 2.0}


@pytest.mark.parametrize("step_class", [CopasiUTCStep, TelluriumUTCStep])
def test_a_step_reports_amounts_and_changing_parameters(fully_registered_core: Core, step_class: type) -> None:
    step = step_class({"model_source": MODEL, "time": 1.0, "n_points": 3, "output_dir": ""}, core=fully_registered_core)
    result = step.update(step.initial_state())["result"]
    last = dict(zip(result["columns"], result["values"][-1], strict=True))
    amount_a = 4 * math.exp(-1)
    assert last["A"] == pytest.approx(amount_a, rel=1e-4)
    assert last["B"] == pytest.approx((4 - amount_a) / 2, rel=1e-4)
    assert (last["P"], last["Q"]) == (pytest.approx(1.0, rel=1e-6), pytest.approx(2.0, rel=1e-6))
