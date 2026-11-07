"""VCell's Sundials ODE solver as a uniform time course step, through pyvcell (the ``vcell`` extra).

VCell imports the SBML in the model's own units; units the model leaves undefined are read as SBML Level 2's
defaults (mol, litre, m², m, s), and nothing is converted, so results come back in the model's units, as COPASI's
and Tellurium's do. VCell's Combined solver runs CVODE, or IDA when the math has fast systems.

VCell's solver outputs the variables it solves. A species or a varying parameter or compartment that VCell's math makes
a constant or a function (one an assignment rule defines, or a boundary species) is evaluated from the math over the
solved time course (pbsim_common.vcell_math), so the step reports what COPASI's and Tellurium's do.

``absolute_tolerance`` and ``relative_tolerance`` set the solver's error tolerances, named as the COPASI and Tellurium
steps name them (pbsim_common.options); unset, VCell's defaults (1e-9 and 1e-7) apply.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, ClassVar

import numpy
from pandas import DataFrame
from process_bigraph import Step

from pbsim_common.options import UTCOptions
from pbsim_common.sbml_values import read_sedml_values
from pbsim_common.utils import model_path_resolution
from pbsim_common.vcell_math import (
    derived_columns,
    fast_reactions,
    kept_elements,
    math_expressions,
    math_names,
    restore,
)

SIMULATION_NAME = "pbsim_utc"
#: the variable VCell solves when the math has no ODEs (vcell-core's OdeFileWriter.NO_ODE_PLACEHOLDER)
NO_ODE_PLACEHOLDER = "__vcell_no_ode__"


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


def _import_sbml(sbml: str) -> str:
    """VCell's import of the SBML as VCML, as ``pyvcell.vcml.load_sbml_str(sbml, assume_units="sbml_l2")`` makes it
    before reading it into pyvcell's model."""
    import libvcell
    from pyvcell.vcml.utils import _resolve_undefined_units

    resolved, _ = _resolve_undefined_units(sbml, "sbml_l2")
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "model.vcml"
        ok, message = libvcell.sbml_to_vcml(sbml_content=resolved, vcml_file_path=path)
        if not ok:
            raise ValueError(message)
        return path.read_text(encoding="utf-8")


def _regenerated(vcml: str) -> str:
    """The VCML as VCell rewrites it, with the math generated from the model it describes."""
    import libvcell

    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "model.vcml"
        ok, message = libvcell.vcml_to_vcml(vcml_content=vcml, vcml_file_path=path)
        if not ok:
            raise ValueError(message)
        return path.read_text(encoding="utf-8")


class VCellUTCStep(Step):
    config_schema: ClassVar[dict[str, str]] = {
        "model_source": "string",
        "time": "float",
        "n_points": "integer",
        "output_dir": "string",
        # optional, as the COPASI and Tellurium steps name them (pbsim_common.options)
        "absolute_tolerance": "maybe[float]",
        "relative_tolerance": "maybe[float]",
    }

    def initialize(self, config: Any = None) -> None:
        vc = _pyvcell_vcml()
        self.options = UTCOptions.from_config(self.config, "VCellUTCStep")
        model_source = self.config["model_source"]
        try:
            # read as UTF-8 here: COPASI and libroadrunner set the C locale when loaded, which makes Python's default
            # text encoding ASCII, and SBML often carries non-ASCII names and notes
            path = model_path_resolution(model_source)
            with open(path, encoding="utf-8") as f:
                sbml = f.read()
            vcml = _import_sbml(sbml)
            # pyvcell's model has no rules or events and no fast reactions: keep VCell's to restore in each document
            # the solver reads
            self._kept = kept_elements(vcml)
            self._fast = fast_reactions(vcml)
            self.biomodel = vc.load_vcml_str(vcml)
        except Exception as e:
            err_msg = f"VCell could not import the SBML model: {model_source}\n{e}"
            raise RuntimeError(err_msg) from e
        if len(self.biomodel.applications) != 1:
            err_msg = f"Expected one application from VCell's SBML import, got {len(self.biomodel.applications)}"
            raise RuntimeError(err_msg)
        self.application = self.biomodel.applications[0]
        # SED-ML reads a species with only substance units as an amount (pbsim_common.sbml_values); VCell reports
        # concentrations and imports only compartments of constant size, so the amount is the concentration times
        # the size the model sets
        values = read_sedml_values(path)
        #: what COPASI and Tellurium report (pbsim_common.sbml_values): every species, and the varying parameters
        #: and compartments
        self._reported = [*values.compartment_of, *values.varying_parameters]
        self._reported_compartments = list(values.varying_compartments)
        self._amount_scale = {
            sid: values.sizes[c]
            for sid in values.amount_species
            if (c := values.compartment_of.get(sid, "")) in values.sizes
        }

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
        if self.options.absolute_tolerance is not None:
            simulation.absolute_tolerance = self.options.absolute_tolerance
        if self.options.relative_tolerance is not None:
            simulation.relative_tolerance = self.options.relative_tolerance
        self.simulation = simulation

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
        from pyvcell.vcml.utils import to_vcml_str

        self._set_initial_conditions(inputs)
        try:
            # one document for the solver and for the math the derived columns are evaluated from: pyvcell's model,
            # with VCell's rules and events restored and the math generated again
            vcml = _regenerated(restore(to_vcml_str(bio_model=self.biomodel, regenerate=False), self._kept, self._fast))
            ode_result = vc.simulate_ode_vcml(vcml, SIMULATION_NAME)
        except Exception as e:
            err_msg = f"VCell's ODE solver failed: {e}"
            raise RuntimeError(err_msg) from e

        time = [float(t) for t in ode_result.times]
        # VCell solves a placeholder in math with no ODEs (libvcell >= 0.0.25); it is not the model's
        solved = {
            name: ode_result.values[:, i]
            for i, name in enumerate(ode_result.variable_names)
            if name != NO_ODE_PLACEHOLDER
        }
        names = math_names(vcml, self._reported, self._reported_compartments)
        derived = derived_columns(math_expressions(vcml), ode_result.times, solved, names)
        all_columns = {**solved, **derived}
        columns = list(all_columns)
        scale = numpy.array([self._amount_scale.get(name, 1.0) for name in columns])
        # a model with nothing that varies (SBML Test Suite 01271: a function definition only) reports time alone
        stacked = [all_columns[name] for name in columns] or [numpy.empty((len(time), 0))]
        values = (numpy.column_stack(stacked) * scale).tolist()
        if self.config.get("output_dir"):
            DataFrame(data=values, columns=columns).assign(time=time)[["time", *columns]].to_csv(
                os.path.join(self.config["output_dir"], "results.csv"), index=False
            )

        return {"result": {"time": time, "columns": columns, "values": values}}
