"""VCell's Sundials ODE solver as a uniform time course step, through pyvcell (the ``vcell`` extra).

VCell imports the SBML in the model's own units; units the model leaves undefined are read as SBML Level 2's
defaults (mol, litre, m², m, s), and nothing is converted, so results come back in the model's units, as COPASI's
and Tellurium's do. VCell's Combined solver runs CVODE, or IDA when the math has fast systems.

VCell outputs the variables it solves: a species that VCell's math makes a constant or a function (one an assignment
rule defines, or a boundary species) is not among the result's columns.
"""

import os
from typing import Any, ClassVar

import numpy
from pandas import DataFrame
from process_bigraph import Step

from pbsim_common.utils import model_path_resolution

SIMULATION_NAME = "pbsim_utc"


def _pyvcell_vcml() -> Any:
    try:
        import pyvcell.vcml as vc
    except ModuleNotFoundError as e:
        err_msg = "VCellUTCStep needs pyvcell with its native and ode extras: install pbsim-common[vcell]"
        raise ModuleNotFoundError(err_msg) from e
    return vc


def _number(value: Any) -> float | None:
    """An initial condition as a number, or None when it is an expression."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class VCellUTCStep(Step):
    config_schema: ClassVar[dict[str, str]] = {
        "model_source": "string",
        "time": "float",
        "n_points": "integer",
        "output_dir": "string",
    }

    def initialize(self, config: Any = None) -> None:
        vc = _pyvcell_vcml()
        model_source = self.config["model_source"]
        try:
            # read as UTF-8 here: COPASI and libroadrunner set the C locale when loaded, which makes Python's default
            # text encoding ASCII, and SBML often carries non-ASCII names and notes
            with open(model_path_resolution(model_source), encoding="utf-8") as f:
                sbml = f.read()
            self.biomodel = vc.load_sbml_str(sbml, assume_units="sbml_l2")
        except Exception as e:
            err_msg = f"VCell could not import the SBML model: {model_source}\n{e}"
            raise RuntimeError(err_msg) from e
        if len(self.biomodel.applications) != 1:
            err_msg = f"Expected one application from VCell's SBML import, got {len(self.biomodel.applications)}"
            raise RuntimeError(err_msg)
        self.application = self.biomodel.applications[0]

        self.time = float(self.config.get("time", 1.0))
        self.n_points = int(self.config.get("n_points", 2))
        if self.n_points < 2:
            err_msg = f"VCellUTCStep: n_points must be >= 2, got {self.n_points}"
            raise ValueError(err_msg)

        # explicit output times, not a step: exactly n_points rows, at the times COPASI and Tellurium report
        simulation = self.application.add_ode_sim(
            SIMULATION_NAME, duration=self.time, output_time_step=self.time / (self.n_points - 1)
        )
        simulation.output_time_step = None
        simulation.output_times = [float(t) for t in numpy.linspace(0.0, self.time, self.n_points)]

        # species by SBML id: VCell's SBML import keeps the ids as its species names
        self._mappings = {m.species_name: m for m in self.application.species_mappings}
        self.species_ids = list(self._mappings)

    def initial_state(self) -> dict[str, Any]:
        concentrations = {sid: _number(m.init_conc) for sid, m in self._mappings.items()}
        return {"species_concentrations": {sid: v for sid, v in concentrations.items() if v is not None}}

    def inputs(self) -> dict[str, str]:
        return {
            "species_concentrations": "map[float]",
            "species_counts": "map[float]",
        }

    def outputs(self) -> dict[str, str]:
        return {"result": "numeric_result"}

    def _set_initial_conditions(self, inputs: Any) -> None:
        """Counts set a species given as an amount, concentrations one given as a concentration; a species given
        the other way is left as the model defines it."""
        counts = inputs.get("species_counts") or {}
        concentrations = inputs.get("species_concentrations") or {}
        for sid, mapping in self._mappings.items():
            if mapping.init_count is not None and sid in counts:
                mapping.init_count = float(counts[sid])
            elif mapping.init_conc is not None and sid in concentrations:
                mapping.init_conc = float(concentrations[sid])

    def update(self, inputs: Any) -> dict[str, Any]:
        vc = _pyvcell_vcml()
        self._set_initial_conditions(inputs)
        try:
            ode_result = vc.simulate_ode(self.biomodel, SIMULATION_NAME)
        except Exception as e:
            err_msg = f"VCell's ODE solver failed: {e}"
            raise RuntimeError(err_msg) from e

        time = [float(t) for t in ode_result.times]
        columns = list(ode_result.variable_names)
        values = ode_result.values.tolist()
        if self.config.get("output_dir"):
            DataFrame(data=values, columns=columns).assign(time=time)[["time", *columns]].to_csv(
                os.path.join(self.config["output_dir"], "results.csv"), index=False
            )

        return {"result": {"time": time, "columns": columns, "values": values}}
