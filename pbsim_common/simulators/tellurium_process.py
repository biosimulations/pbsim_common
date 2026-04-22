import os
from typing import Any, ClassVar

import pandas
import tellurium as te
from bigraph_schema import allocate_core
from bigraph_schema.core import Core
from pandas import DataFrame
from process_bigraph import Step
from roadrunner import RoadRunner

from pbsim_common.utils import model_path_resolution


class TelluriumStep(Step):
    def _tellurium_initialize(self) -> None:
        model_source = self.config["model_source"]

        # ----- Minimal Tellurium load (SBML) -----
        try:
            self.rr: RoadRunner = te.loadSBMLModel(model_path_resolution(model_source))
        except Exception as e:
            err_msg = f"Could not load SBML model: {model_source}\n{e}"
            raise RuntimeError(err_msg) from e

        # ----- Cache IDs -----
        self.species_ids = list(self.rr.getFloatingSpeciesIds())
        self.reaction_ids = list(self.rr.getReactionIds())
        self._species_index = {sid: i for i, sid in enumerate(self.species_ids)}

    # ------------------------------------------------
    # process-bigraph API
    # ------------------------------------------------
    def initial_state(self) -> dict[str, Any]:
        conc = self.rr.getFloatingSpeciesConcentrations()
        return {"species_concentrations": {sid: float(conc[i]) for i, sid in enumerate(self.species_ids)}}

    def inputs(self) -> dict[str, str]:
        return {
            "species_concentrations": "map[float]",
            "species_counts": "map[float]",
        }

    def set_road_runner_incoming_values(self, inputs: Any) -> None:
        spec_data = inputs.get("species_counts") or inputs.get("species_concentrations") or {}

        # 2) Set incoming values on the model
        print(spec_data)
        for sid, value in spec_data.items():
            if sid in self._species_index:
                self.rr.setValue(sid, float(value))


class TelluriumUTCStep(TelluriumStep):
    config_schema: ClassVar[dict[str, str]] = {
        "model_source": "string",
        "time": "float",
        "n_points": "integer",
        "output_dir": "string",
    }

    def initialize(self, config: Any) -> None:
        self._tellurium_initialize()

        # ----- sim parameters -----
        self.time = float(self.config.get("time", 1.0))
        self.n_points = int(self.config.get("n_points", 2))
        if self.n_points < 2:
            err_msg = f"TelluriumUTCStep: n_points must be >= 2, got {self.n_points}"
            raise ValueError(err_msg)

    def outputs(self) -> dict[str, str]:
        return {"result": "numeric_result"}

    # ------------------------------------------------
    # update logic
    # ------------------------------------------------
    def update(self, state: dict[str, Any], interval: Any = None) -> dict[str, Any]:
        # 1) Choose source
        # 2) Update species concentrations using Tellurium's setValue
        self.set_road_runner_incoming_values(state)
        output_file = (
            None if self.config["output_dir"] == "" else os.path.join(self.config["output_dir"], "results.csv")
        )

        # 3) Run simulation: from 0 -> self.time, n_points samples
        tc = self.rr.simulate(0, self.time, self.n_points, output_file=output_file)
        tc = DataFrame(data=tc.tolist(), columns=tc.colnames) if output_file is None else pandas.read_csv(output_file)
        colnames: list[str] = tc.columns.tolist()

        # Build a mapping from *normalized* column names to indices.
        # This turns "[S1]" -> "S1", etc.
        norm_to_index: dict[str, int] = {name.strip("[]"): i for i, name in enumerate(colnames)}

        time_idx = norm_to_index["time"]
        time = tc.values[:, time_idx].tolist()

        # 4) Species trajectories
        species_cols: dict[str, int] = {}
        for sid in self.species_ids:
            idx = norm_to_index.get(sid)
            if idx is not None:
                species_cols[sid] = idx

        # 5) Reaction flux time series
        flux_json: dict[Any, Any] = {rid: [] for rid in self.reaction_ids}

        # For each time point, set state and query reaction rates
        for row in range(tc.shape[0]):
            for sid, idx in species_cols.items():
                self.rr.setValue(sid, float(tc.values[row, idx]))

            rates = self.rr.getReactionRates()
            for j, rid in enumerate(self.reaction_ids):
                flux_json[rid].append(float(rates[j]))

        # 6) Restore last state (final row of the timecourse)
        last_row = tc.shape[0] - 1
        for sid, idx in species_cols.items():
            self.rr.setValue(sid, float(tc.values[last_row, idx]))

        # 7) Send update — structured for easy comparison / aggregation
        result = {
            "time": time,
            "columns": [c.strip("[]") for c in colnames if c != "time"],
            "values": tc.values[:, 1:].tolist(),
            # "n_spacial_dimensions": (tc.shape[0], tc.shape[1] - 1),
            # "fluxes": flux_json,
        }

        return {"result": result}


class TelluriumSteadyStateStep(TelluriumStep):
    config_schema: ClassVar[dict[str, str]] = {
        "model_source": "string",
    }

    def initialize(self, config: Any = None) -> None:
        self._tellurium_initialize()

    def outputs(self) -> dict[str, str]:
        return {"result": "map[numeric_result]"}

    # ------------------------------------------------
    # steady-state computation
    # ------------------------------------------------
    def update(self, inputs: Any) -> dict[str, Any]:
        # 1) Prefer counts, fall back to concentrations
        # 2) Set incoming values on the model
        self.set_road_runner_incoming_values(inputs)

        # 3) Run steady-state computation
        #    RoadRunner steadyState() modifies the internal state to a (near-)steady state.
        try:
            self.rr.steadyState()
        except Exception as e:
            err_msg = f"Tellurium steadyState() failed: {e}"
            raise RuntimeError(err_msg) from e

        # 4) Read back steady-state species concentrations
        # conc_ss = self.rr.getFloatingSpeciesConcentrations()
        # species_ss = {sid: float(conc_ss[i]) for i, sid in enumerate(self.species_ids)}

        # 5) Read back steady-state reaction fluxes
        # rates_ss = self.rr.getReactionRates()
        # flux_ss = {rid: float(rates_ss[i]) for i, rid in enumerate(self.reaction_ids)}

        # 6) Package as a one-point "time series", time = 0.0
        # time_list = [0.0]
        # species_json = {sid: [val] for sid, val in species_ss.items()}
        # flux_json = {rid: [val] for rid, val in flux_ss.items()}

        steady_state = {
            "time": [0],
            "columns": self.rr.getFloatingSpeciesConcentrationsNamedArray().colnames,
            "values": self.rr.getFloatingSpeciesConcentrationsNamedArray().tolist(),
        }

        jacobian = {
            "time": [0],
            "columns": self.rr.getFullJacobian().colnames,
            "values": self.rr.getFullJacobian().tolist(),
        }

        result = {
            "_add": {"jacobian": jacobian, "steady_state": steady_state},
        }

        return {"result": result}


# Simple test like Copasi
def run_utc_test(core: Any) -> None:
    step = TelluriumUTCStep(
        {
            "model_source": "models/BIOMD0000000012_url.xml",
            "time": 10.0,
            "n_points": 5,
        },
        core=core,
    )

    init = step.initial_state()
    print("Initial:", init)

    result = step.update(init)
    print("Result:", result)


def run_ss_test(core: Any) -> None:
    step = TelluriumSteadyStateStep(
        {
            "model_source": "models/BIOMD0000000012_url.xml",
            "time": 0.0,  # unused
        },
        core=core,
    )

    init = step.initial_state()
    print("Initial:", init)

    out = step.update(init)
    print("Steady-state result:", out)


if __name__ == "__main__":
    core: Core = allocate_core()
    core.register_link("tellurium_utc", TelluriumUTCStep)
    run_utc_test(core)
    run_ss_test(core)
