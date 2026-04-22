import importlib.metadata
from pathlib import Path
from typing import Any

import pytest
from bigraph_schema import allocate_core
from bigraph_schema.core import Core

# from biocompose import standard_types
from pbsim_common import standard_types
from pbsim_common.comparison import MSEComparison
from pbsim_common.simulators import TelluriumUTCStep, CopasiUTCStep


@pytest.fixture(scope="function")
def fully_registered_core() -> Core:
    importlib.metadata.packages_distributions()
    core: Core = allocate_core()
    manual = {
        "pbsim_common.simulators.tellurium_process.TelluriumUTCStep": TelluriumUTCStep,
        "pbsim_common.simulators.copasi_process.CopasiUTCStep": CopasiUTCStep,
        "pbsim_common.comparison.MSEComparison": MSEComparison
    }
    for k, i in standard_types.items():
        core.register_type(k, i)
    for k, i in manual.items():
        core.register_link(k, i)
    return core
