import datetime
import json
import math
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import docker
import numpy
from bigraph_schema import Core
from process_bigraph import Composite, gather_emitter_results


def is_docker_present() -> bool:
    client = docker.from_env()
    try:
        client.ping()
        return True  # noqa: TRY300
    except Exception:
        return False


def root_dir_path() -> Path:
    return Path(__file__).parent.parent


def compare_csv(experiment_result: str, expected_csv_path: str, difference_tolerance: float = 5e-8):
    experiment_numpy = numpy.genfromtxt(experiment_result, delimiter=",", dtype=object)
    report_numpy = numpy.genfromtxt(expected_csv_path, delimiter=",", dtype=object)
    assert report_numpy.shape == experiment_numpy.shape
    r, c = report_numpy.shape
    for i in range(r):
        for j in range(c):
            report_val = report_numpy[i, j].decode("utf-8")
            experiment_val = experiment_numpy[i, j].decode("utf-8")
            try:
                f_report = float(report_val)
                f_exp = float(experiment_val)
                is_close = math.isclose(f_report, f_exp, rel_tol=0, abs_tol=difference_tolerance)
                if not is_close:
                    print(f_report, f_exp)
                    assert is_close
            except ValueError:
                assert report_val == experiment_val  # Must be string portion of report then (columns)

def replace_relative_pbif_paths(dic: dict[Any, Any], root_dir: str) -> None:
    for k, v in dic.items():
        if isinstance(v, dict):
            replace_relative_pbif_paths(v, root_dir)
        elif k == "model_source" or k == "output_dir":
            dic[k] = os.path.join(root_dir, v)


def get_pb_schema_from_omex(omex_path: str, working_dir: str) -> dict[Any, Any]:
    input_file: str | None = None
    with zipfile.ZipFile(omex_path, "r") as zf:
        zf.extractall(working_dir)
    for file_name in os.listdir(working_dir):
        if not file_name.endswith(".pbg"):
            continue
        input_file = os.path.join(working_dir, file_name)
        break

    if input_file is None:
        err = f"Could not find any PBG or JSON file in or at `{omex_path}`."
        raise FileNotFoundError(err)
    with open(input_file) as input_data:
        result: dict[Any, Any] = json.load(input_data)
        replace_relative_pbif_paths(result, working_dir)
        return result


def run_experiment(tmp_dir: str, core: Core, interval: int, omex_file: str=None, pbg: dict[str, Any] = None) -> None:
    schema = pbg if pbg is not None else get_pb_schema_from_omex(omex_file, tmp_dir)
    prepared_composite = Composite(core=core, config=schema)

    prepared_composite.run(interval)
    query_results = gather_emitter_results(prepared_composite)

    current_dt = datetime.datetime.now()
    date, tz, time = str(current_dt.date()), str(current_dt.tzinfo), str(current_dt.time()).replace(":", "-")

    try:
        if len(query_results) != 0:
            emitter_results_file_path = os.path.join(
                tmp_dir, f"results_{date}[{tz}#{time}].pber"
            )
            with open(emitter_results_file_path, "w") as emitter_results_file:
                json.dump(query_results, emitter_results_file)
    except TypeError as e:
        err_msg = f"Tried to save query results to {emitter_results_file_path}: {e}"
        print(err_msg)
    prepared_composite.save(filename=f"state_{date}#{time}.pbg", outdir=tmp_dir)




