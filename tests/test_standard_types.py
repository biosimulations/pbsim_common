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
