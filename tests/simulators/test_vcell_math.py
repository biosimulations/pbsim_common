"""VCellUTCStep solves the model the SBML describes (pbsim_common.vcell_math): the rules, events and fast reactions
pyvcell's model does not keep are restored, and the values VCell's math defines but its solver does not output are
reported. Skipped without the ``vcell`` extra."""

import importlib.util
import math
from pathlib import Path
from typing import Any

import pytest
from bigraph_schema import Core

from pbsim_common.simulators import CopasiUTCStep, VCellUTCStep
from pbsim_common.vcell_math import derived_columns, fast_reactions, kept_elements, math_names, restore

needs_vcell = pytest.mark.skipif(
    importlib.util.find_spec("libvcell") is None or importlib.util.find_spec("pyvcell_odesolver") is None,
    reason="needs the vcell extra (pyvcell[native,ode])",
)

# S1 decays and is reset by an event; S2 is defined by an assignment rule, S3 by a rate rule, and the parameter x
# (a name VCell reserves for a coordinate, so its math calls it param_x) by an assignment rule
RULES_AND_EVENTS = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="rules_and_events">
    <listOfCompartments>
      <compartment id="C" spatialDimensions="3" size="2" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S1" compartment="C" initialConcentration="1" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="S2" compartment="C" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="S3" compartment="C" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.3" constant="true"/>
      <parameter id="x" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="S2">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><cn>2</cn><ci>S1</ci></apply></math>
      </assignmentRule>
      <rateRule variable="S3">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0.5</cn></math>
      </rateRule>
      <assignmentRule variable="x">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><apply><plus/><ci>S1</ci><cn>1</cn></apply></math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfReactants>
          <speciesReference species="S1" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>S1</ci><ci>C</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
    <listOfEvents>
      <event id="reset" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML"><apply><geq/><csymbol
            encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol><cn>4.5</cn></apply></math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="S1">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>3</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>
"""

# A <-> B is fast (kf = 1, kr = 2), so A and B start at its equilibrium, A = 2 B, of their total 2; B decays slowly
FAST_REACTION = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="fast_reaction">
    <listOfCompartments>
      <compartment id="C" size="1"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="2"/>
      <species id="B" compartment="C" initialConcentration="0"/>
    </listOfSpecies>
    <listOfReactions>
      <reaction id="fast" reversible="true" fast="true">
        <listOfReactants><speciesReference species="A"/></listOfReactants>
        <listOfProducts><speciesReference species="B"/></listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><minus/>
              <apply><times/><ci>C</ci><cn>1</cn><ci>A</ci></apply>
              <apply><times/><ci>C</ci><cn>2</cn><ci>B</ci></apply>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
      <reaction id="decay" reversible="false">
        <listOfReactants><speciesReference species="B"/></listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>C</ci><cn>0.1</cn><ci>B</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def _run(step_class: Any, core: Core, model: Path, time: float = 10.0, n_points: int = 21) -> dict[str, list[float]]:
    config = {"model_source": str(model), "time": time, "n_points": n_points, "output_dir": ""}
    result = step_class(config, core=core).update({})["result"]
    columns = {c: [row[i] for row in result["values"]] for i, c in enumerate(result["columns"])}
    return {"time": result["time"], **columns}


@needs_vcell
def test_rules_and_events_are_solved_and_reported(fully_registered_core: Core, tmp_path: Path) -> None:
    model = tmp_path / "rules_and_events.xml"
    model.write_text(RULES_AND_EVENTS, encoding="utf-8")
    vcell = _run(VCellUTCStep, fully_registered_core, model)
    copasi = _run(CopasiUTCStep, fully_registered_core, model)
    for i, t in enumerate(vcell["time"]):
        s1 = vcell["S1"][i]
        assert math.isclose(vcell["S2"][i], 2 * s1, rel_tol=1e-6), "the assignment rule"
        assert math.isclose(vcell["S3"][i], 0.5 * t, rel_tol=1e-6, abs_tol=1e-9), "the rate rule"
        assert math.isclose(vcell["x"][i], s1 + 1, rel_tol=1e-6), "a parameter VCell renames"
    for column in ("S1", "S2", "S3", "x"):
        for a, b in zip(vcell[column], copasi[column], strict=True):
            assert math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-6), column
    # the event: S1 is 3 again at t = 4.5, more than it had decayed to by t = 4
    assert vcell["time"][9] == 4.5
    assert math.isclose(vcell["S1"][9], 3.0, rel_tol=1e-6) and vcell["S1"][8] < 1


@needs_vcell
def test_a_fast_reaction_starts_at_its_equilibrium(fully_registered_core: Core, tmp_path: Path) -> None:
    model = tmp_path / "fast_reaction.xml"
    model.write_text(FAST_REACTION, encoding="utf-8")
    vcell = _run(VCellUTCStep, fully_registered_core, model, time=1.0, n_points=3)
    assert math.isclose(vcell["A"][0], 4 / 3, rel_tol=1e-4)
    assert math.isclose(vcell["B"][0], 2 / 3, rel_tol=1e-4)


VCML = """<vcml xmlns="http://sourceforge.net/projects/vcell/vcml"><BioModel Name="m"><Model Name="m">
<ModelParameters><Parameter Name="param_x" SbmlName="x">1</Parameter></ModelParameters><Feature Name="C"/></Model>
<SimulationSpec Name="a"><ReactionContext><ReactionSpec ReactionStepRef="r1" ReactionMapping="fast"/>
<ReactionSpec ReactionStepRef="r2" ReactionMapping="included"/></ReactionContext>
<MathDescription Name="math"><Constant Name="k">2.0</Constant><Constant Name="Size_C">3.0</Constant>
<Function Name="param_x" Domain="c">(k * S1)</Function><Function Name="S2" Domain="c">(param_x + t)</Function>
<Function Name="S3" Domain="c">factorial(3)</Function></MathDescription>
<AssignmentRules><AssignmentRule Name="r" AssignmentRuleVariable="S2">(k * S1)</AssignmentRule></AssignmentRules>
</SimulationSpec></BioModel></vcml>"""

# what pyvcell writes: no namespace, no rules, a fast reaction excluded
PYVCELL = """<vcml><BioModel Name="m"><SimulationSpec Name="a"><ReactionContext>
<ReactionSpec ReactionStepRef="r1" ReactionMapping="excluded"/><ReactionSpec ReactionStepRef="r2" ReactionMapping="included"/>
</ReactionContext><Simulation Name="s"/></SimulationSpec></BioModel></vcml>"""


@needs_vcell
def test_restore_puts_back_what_pyvcell_loses() -> None:
    from lxml import etree

    restored = etree.fromstring(restore(PYVCELL, kept_elements(VCML), fast_reactions(VCML)).encode())  # noqa: S320, RUF100
    spec = restored.find("BioModel/SimulationSpec")
    assert spec is not None
    # in the host's (empty) namespace, before the simulations
    assert [child.tag for child in spec] == ["ReactionContext", "AssignmentRules", "Simulation"]
    assert spec.find("AssignmentRules/AssignmentRule").get("AssignmentRuleVariable") == "S2"  # type: ignore[union-attr]
    mappings = {e.get("ReactionStepRef"): e.get("ReactionMapping") for e in spec.iter("ReactionSpec")}
    assert mappings == {"r1": "fast", "r2": "included"}


@needs_vcell
def test_derived_columns_evaluate_the_math_by_sbml_id() -> None:
    import numpy

    from pbsim_common.vcell_math import math_expressions

    names = math_names(VCML, ["x", "S2", "S3", "S1", "absent"], ["C"])
    assert names == {"x": "param_x", "S2": "S2", "S3": "S3", "C": "Size_C"}
    times = numpy.array([0.0, 1.0, 2.0])
    columns = derived_columns(math_expressions(VCML), times, {"S1": numpy.array([1.0, 2.0, 4.0])}, names)
    assert list(columns) == ["x", "S2", "S3", "C"]
    assert columns["x"].tolist() == [2.0, 4.0, 8.0]
    assert columns["S2"].tolist() == [2.0, 5.0, 10.0]
    assert columns["S3"].tolist() == [6.0, 6.0, 6.0], "factorial, which numexpr has not, one time point at a time"
    assert columns["C"].tolist() == [3.0, 3.0, 3.0]
