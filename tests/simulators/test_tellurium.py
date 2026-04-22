import json
import os.path
import tempfile
from pathlib import Path

from bigraph_schema import Core

from tests.fixtures.utils import compare_csv, root_dir_path, run_experiment

experiment = {
    "state": {
        "time_course": {
            "_type": "step",
            "address": "local:pbsim_common.simulators.tellurium_process.TelluriumUTCStep",
            "config": {
                "model_source": os.path.join(root_dir_path(), "resources", "phase_cycle.sbml"),
                "time": 10,
                "n_points": 51,
            },
            "interval": 1.0,
            "inputs": {},
            "outputs": {},
        }
    }
}


def test_tellurium(fully_registered_core: Core) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        experiment["state"]["time_course"]["config"]["output_dir"] = tmpdir
        run_experiment(
            tmp_dir=tmpdir,
            core=fully_registered_core,
            pbg=experiment,
            interval=1
        )

        compare_csv(
            experiment_result=os.path.join(tmpdir, "results.csv"),
            expected_csv_path=os.path.join(root_dir_path(), "resources", "tellurium_report.csv"),
        )
