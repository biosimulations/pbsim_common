from typing import Any

from bigraph_schema import Core

RESULT = {"time": [0.0, 1.0], "columns": ["a"], "values": [[1.0], [2.0]]}


def _apply(core: Core, schema: Any, state: Any, update: Any) -> Any:
    applied = core.apply(core.access(schema), state, update)
    return applied[0] if isinstance(applied, tuple) else applied


def test_a_second_result_replaces_the_first(fully_registered_core: Core) -> None:
    """A step that fires twice (its inputs changed) leaves its latest result, not both appended (bigraph-schema
    appends list updates unless the type overwrites)."""
    _, state = fully_registered_core.default(fully_registered_core.access("numeric_result"))
    once = _apply(fully_registered_core, "numeric_result", state, RESULT)
    twice = _apply(fully_registered_core, "numeric_result", once, RESULT)
    assert once == RESULT
    assert twice == RESULT


def test_a_second_result_replaces_the_first_in_a_map(fully_registered_core: Core) -> None:
    state = {"steady": RESULT}
    assert _apply(fully_registered_core, "numeric_results", state, {"steady": RESULT}) == state


def test_a_comparison_or_statistic_replaces_the_one_before(fully_registered_core: Core) -> None:
    """A step's result is not a change: fired twice, its port holds the second result, not their sum (bigraph-schema
    adds float and array updates unless the type overwrites)."""
    import numpy

    from pbsim_common.comparison import MSEComparison
    from pbsim_common.stats import SumOfSquaresTool

    comparison = MSEComparison.outputs(None)["comparison_result"]  # type: ignore[arg-type]
    state = {"a": {"b": {"mse": 1.0}}}
    once = _apply(fully_registered_core, comparison, state, {"a": {"b": {"mse": 2.0}}})
    assert _apply(fully_registered_core, comparison, once, {"a": {"b": {"mse": 2.0}}}) == {"a": {"b": {"mse": 2.0}}}

    statistic = SumOfSquaresTool.outputs(None)["stats_result"]  # type: ignore[arg-type]
    once = _apply(fully_registered_core, statistic, numpy.array([1.0, 2.0]), [3.0, 4.0])
    assert list(_apply(fully_registered_core, statistic, once, [3.0, 4.0])) == [3.0, 4.0]
