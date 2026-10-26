import os

from bigraph_schema import Core
from process_bigraph import Composite

from pbsim_common.simulators import CopasiUTCProcess
from tests.fixtures.utils import root_dir_path


def test_the_store_holds_the_model_state_after_each_interval(fully_registered_core: Core) -> None:
    """The process reports the state at the end of each interval, so the store must equal it, not accumulate it
    (bigraph-schema adds float updates unless the type overwrites), and time runs on from interval to interval."""
    fully_registered_core.register_link("pbsim_common.simulators.copasi_process.CopasiUTCProcess", CopasiUTCProcess)
    composite = Composite(
        {
            "state": {
                "copasi": {
                    "_type": "process",
                    "address": "local:pbsim_common.simulators.copasi_process.CopasiUTCProcess",
                    "config": {"model_source": os.path.join(root_dir_path(), "resources", "phase_cycle.sbml")},
                    "interval": 1.0,
                    "inputs": {},
                    "outputs": {"species_concentrations": ["species"], "fluxes": ["fluxes"], "time": ["time"]},
                }
            }
        },
        core=fully_registered_core,
    )
    process = composite.state["copasi"]["instance"]
    for interval in range(3):
        composite.run(1.0)
        model = {sid: float(species.getConcentration()) for sid, species in process.species.items()}
        assert composite.state["species"] == model
        assert composite.state["time"][0] == interval
        assert composite.state["time"][-1] == interval + 1
