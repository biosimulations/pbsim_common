import os
from typing import Any, ClassVar

import COPASI
from basico import (
    get_reactions,
    load_model,
    run_steadystate,
    run_time_course,
)
from bigraph_schema import Core, allocate_core
from pandas import DataFrame
from process_bigraph import Process, Step

from pbsim_common.utils import model_path_resolution


def _species_by_sbml_id(dm: Any) -> dict[str, Any]:
    """Every species of the model, by SBML id (its COPASI name when it has none).

    COPASI display names are not unique (the same name in two compartments), and basico's ``get_species`` returns
    None for a model with no species, so neither is a safe key.
    """
    model = dm.getModel()
    if not isinstance(model, COPASI.CModel):
        error_msg = f"Model {model!r} is not a COPASI model."
        raise TypeError(error_msg)
    species: dict[str, Any] = {}
    for i in range(model.getNumMetabs()):
        metab = model.getMetabolite(i)
        species[metab.getSBMLId() or metab.getObjectName()] = metab
    return species


def _set_initial_concentrations(changes: Any, dm: Any) -> None:
    """
    changes: iterable of (COPASI species, value) pairs
    dm: COPASI DataModel as returned by basico.load_model
    """
    model = dm.getModel()
    if not isinstance(model, COPASI.CModel):
        error_msg = f"Model {model!r} is not a COPASI model."
        raise TypeError(error_msg)

    references = COPASI.ObjectStdVector()

    for species, value in changes:
        if not isinstance(species, COPASI.CMetab):
            error_msg = f"{species!r} is not a COPASI species."
            raise TypeError(error_msg)
        species.setInitialConcentration(float(value))
        references.append(species.getInitialConcentrationReference())

    if len(references) > 0:
        model.updateInitialValues(references)


def _with_every_species(tc: DataFrame, species: dict[str, Any]) -> DataFrame:
    """The time course with a column for every species. COPASI's leaves out a species whose value cannot change (in no
    reaction, or only a modifier of one), which a SED-ML report may still ask for: its concentration, the same at
    every time point."""
    missing = {sbml_id: float(metab.getConcentration()) for sbml_id, metab in species.items() if sbml_id not in tc}
    return tc.assign(**missing) if missing else tc


class BaseCopasi:
    cmodel = None
    dm = None
    species: dict[str, Any]
    species_ids: Any
    reaction_ids: Any
    sbml_to_name: dict[Any, Any]
    config: dict[str, Any]

    def interpret_sbml(self) -> None:
        model_source = self.config["model_source"]

        # ---- Load COPASI model ----
        self.dm = load_model(model_path_resolution(model_source))
        if self.dm is None:
            error_msg = (
                f"load_model({model_source!r}) returned None. "
                f"Check that the file exists and is a valid COPASI/SBML model."
            )
            raise RuntimeError(error_msg)

        self.cmodel = self.dm.getModel()

        # External canonical IDs: SBML IDs. A model may have no species (only rate rules on parameters).
        self.species = _species_by_sbml_id(self.dm)
        self.species_ids = list(self.species)

        # Mapping: SBML ID -> COPASI display name (not unique across compartments: never look a species up by it)
        self.sbml_to_name = {sbml_id: metab.getObjectName() for sbml_id, metab in self.species.items()}

        # These are typically SBML reaction ids already; basico returns None for a model with no reactions
        rxn_df = get_reactions(model=self.dm)
        self.reaction_ids = [] if rxn_df is None else rxn_df.index.tolist()

    def get_concentrations_from_sbml(self) -> dict[str, Any]:
        return {
            "species_concentrations": {
                sbml_id: float(metab.getConcentration()) for sbml_id, metab in self.species.items()
            }
        }


class CopasiUTCStep(Step, BaseCopasi):
    config_schema: ClassVar[dict[str, str]] = {
        "model_source": "string",
        "time": "float",
        "n_points": "integer",
        "output_dir": "string",
    }

    def initialize(self, config: Any = None) -> None:
        self.interpret_sbml()

        # Simulation parameters
        self.interval = float(self.config.get("time", 1.0))
        self.n_points = int(self.config.get("n_points", 2))  # <-- NEW
        if self.n_points < 2:
            error_msg = f"n_points={self.n_points} must be >= 2."
            raise ValueError(error_msg)

        self.intervals = self.n_points - 1  # COPASI requires this

    def initial_state(self) -> dict[str, Any]:
        return self.get_concentrations_from_sbml()

    def inputs(self) -> dict[str, str]:
        return {
            "species_concentrations": "map[float]",
            "species_counts": "map[float]",
        }

    def outputs(self) -> dict[str, str]:
        return {
            "result": "numeric_result",
        }

    def update(self, inputs: Any) -> dict[str, Any]:
        # Apply incoming concentrations
        spec_data = inputs.get("counts", {}) or {}
        changes = [
            (self.species[sbml_id], float(value)) for sbml_id, value in spec_data.items() if sbml_id in self.species
        ]

        if changes:
            _set_initial_concentrations(changes, self.dm)

        # --- Run COPASI time course with intervals = n_points - 1 ---
        tc: DataFrame = run_time_course(
            start_time=0.0,
            duration=self.config["time"],
            intervals=self.intervals,
            update_model=True,
            use_sbml_id=True,
            model=self.dm,
        )
        tc = _with_every_species(tc, self.species)
        if "output_dir" in self.config and self.config["output_dir"] != "":
            tc.to_csv(os.path.join(self.config["output_dir"], "results.csv"), index=False)

        # Time series
        time_list = tc.index.to_list()

        result = {
            "time": time_list,
            "columns": [c for c in tc.columns],  # noqa: C416
            "values": tc.values.tolist(),
            # "n_spacial_dimensions": tc.shape,
        }

        return {"result": result}


class CopasiSteadyStateStep(Step, BaseCopasi):
    config_schema: ClassVar[dict[str, str]] = {
        "model_source": "string",
        "time": "float",  # kept for symmetry, not used
    }

    def initialize(self, config: Any = None) -> None:
        self.interpret_sbml()

    # ------------------------------------------------
    # initial state (SBML IDs externally)
    # ------------------------------------------------
    def initial_state(self) -> dict[str, Any]:
        return self.get_concentrations_from_sbml()

    # ------------------------------------------------
    # ports
    # ------------------------------------------------
    def inputs(self) -> dict[str, str]:
        # Externally everything uses SBML IDs
        return {
            "species_concentrations": "map[float]",  # SBML IDs
            "counts": "map[float]",  # SBML IDs
        }

    def outputs(self) -> dict[str, str]:
        # Match TelluriumSteadyStateStep: nested results
        return {
            "results": "any",
        }

    # ------------------------------------------------
    # steady-state update
    # ------------------------------------------------
    def update(self, inputs: Any) -> dict[str, Any]:
        # 1) Prefer counts, otherwise concentrations (keys are SBML IDs)
        spec_data = inputs.get("counts") or inputs.get("concentrations") or {}

        changes = [
            (self.species[sbml_id], float(value)) for sbml_id, value in spec_data.items() if sbml_id in self.species
        ]

        if changes:
            _set_initial_concentrations(changes, self.dm)

        # 2) Run COPASI steady-state task
        # (use_sbml_id affects task I/O naming; species are read back from the model by SBML id)
        run_steadystate(
            update_model=True,
            use_sbml_id=True,
            model=self.dm,
        )

        # 3) Read back steady-state species concentrations (SBML IDs externally)
        species_conc_ss = {sbml_id: float(metab.getConcentration()) for sbml_id, metab in self.species.items()}

        # 4) Steady-state reaction fluxes
        rxn_df = get_reactions(model=self.dm)
        reaction_fluxes_ss = (
            {}
            if rxn_df is None
            else {rid: float(rxn_df.loc[rid, "flux"]) for rid in self.reaction_ids if rid in rxn_df.index}
        )

        # 5) Package as one-point "time series" (t = 0.0) to match Tellurium
        time_list = [0.0]

        species_json = {sid: [val] for sid, val in species_conc_ss.items()}
        flux_json = {rid: [val] for rid, val in reaction_fluxes_ss.items()}

        results = {
            "time": time_list,
            "species_concentrations": species_json,  # SBML IDs as keys
            "fluxes": flux_json,
        }

        return {"results": results}


class CopasiUTCProcess(Process, BaseCopasi):
    config_schema: ClassVar[dict[str, str]] = {
        "model_source": "string",
        "time": "float",
        "intervals": "integer",
    }

    def initialize(self, config: Any = None) -> None:
        self.interpret_sbml()

        # ---- Sim parameters ----
        self.time = float(self.config.get("time", 1.0))
        self.intervals = int(self.config.get("intervals", 10))
        self.elapsed = 0.0  # the model time at the start of the next interval

    # -----------------------------------------------------------------
    # initial state
    # -----------------------------------------------------------------
    def initial_state(self) -> dict[str, Any]:
        return self.get_concentrations_from_sbml()

    # -----------------------------------------------------------------
    # I/O schema
    # -----------------------------------------------------------------
    def inputs(self) -> dict[str, str]:
        return {
            "species_concentrations": "map[float]",  # SBML IDs
            "species_counts": "map[float]",  # SBML IDs
        }

    def outputs(self) -> dict[str, str]:
        # Each update reports the state at the end of the interval, not a change, so every port overwrites:
        # bigraph-schema adds float updates and appends list updates.
        return {
            "species_concentrations": "map[overwrite[float]]",  # SBML IDs
            "fluxes": "map[overwrite[float]]",
            "time": "overwrite[list[float]]",  # this interval's time points
        }

    # -----------------------------------------------------------------
    # update
    # -----------------------------------------------------------------
    def update(self, inputs: dict[str, Any], interval: float) -> dict[str, Any]:
        # --- 1) Determine incoming species map (SBML IDs)
        incoming = inputs.get("species_counts") or inputs.get("species_concentrations") or {}

        changes = [
            (self.species[sbml_id], float(value)) for sbml_id, value in incoming.items() if sbml_id in self.species
        ]

        if changes:
            _set_initial_concentrations(changes, self.dm)

        # --- 2) Run time course with SBML-ID columns ----
        tc = run_time_course(
            start_time=0.0,
            duration=interval,
            intervals=self.intervals,
            update_model=True,
            use_sbml_id=True,  # <-- critical
            model=self.dm,
        )

        # The interval's time points (basico indexes the time course by time), from the model's start
        time = [self.elapsed + float(t) for t in tc.index.to_list()]
        self.elapsed += interval

        # --- 3) Read back final state: export SBML IDs ----
        species_concentrations = {sbml_id: float(metab.getConcentration()) for sbml_id, metab in self.species.items()}

        # --- 4) Reaction fluxes (COPASI reaction IDs already match SBML IDs) ----
        rxn_df = get_reactions(model=self.dm)
        reaction_fluxes = {} if rxn_df is None else {rid: float(rxn_df.loc[rid, "flux"]) for rid in self.reaction_ids}

        return {
            "species_concentrations": species_concentrations,
            "fluxes": reaction_fluxes,
            "time": time,
        }


def run_copasi_utc(core: Any) -> None:
    copasi_process = CopasiUTCStep(
        {
            "model_source": "models/BIOMD0000000012_url.xml",  # represillator model
            "time": 10.0,
            "n_points": 5,
        },
        core=core,
    )

    initial_state = copasi_process.initial_state()

    print(f"Initial state: {initial_state}")

    results = copasi_process.update(initial_state)

    print(f"Results: {results}")


def run_copasi_ss(core: Any) -> None:
    copasi_process = CopasiSteadyStateStep(
        {
            "model_source": "models/BIOMD0000000012_url.xml",  # represillator model
        },
        core=core,
    )

    initial_state = copasi_process.initial_state()

    print(f"Initial state: {initial_state}")

    results = copasi_process.update(initial_state)

    print(f"Results: {results}")


if __name__ == "__main__":
    core: Core = allocate_core()
    core.register_link("copasi_utc", CopasiUTCStep)
    core.register_link("copasi_ss", CopasiSteadyStateStep)
    core.register_link("copasi_process", CopasiUTCProcess)

    run_copasi_utc(core=core)
    run_copasi_ss(core=core)
