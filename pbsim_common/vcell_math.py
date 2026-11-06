"""VCell's math for a pyvcell model: the parts pyvcell does not keep, and the values the ODE solver does not output.

pyvcell reads VCell's SBML import into its own model, which has no assignment rules, rate rules or events (its reader
skips them) and writes a fast reaction as excluded, and writes that model back for the solver. ``kept_elements`` and
``fast_reactions`` take what it loses from VCell's own import, and ``restore`` puts it back into the document pyvcell
writes, so VCell solves the model the SBML describes. The SBML Test Suite found it (compose-api's
sbml-test-suite-sample, run 5, 2026-11-06): 21 of 50 sampled cases lost a rule or an event.

VCell's math solves only its state variables. A species an assignment rule defines, a parameter a rule sets, and any
other value VCell turns into a function of the state is a ``Function`` of the math description, and a value that never
changes is a ``Constant``; the Sundials solver writes neither. VCell's own result sets add them by evaluating the
math over the solved time course, and so does ``derived_columns``: a SED-ML data generator names them by SBML id, and
VCell's SBML import keeps the ids as the math's names.

Expressions are evaluated over whole columns with numexpr (a pyvcell dependency), translated from VCell's infix by
libvcell. The few functions numexpr has no equivalent for (``factorial``) fall back to libvcell's evaluator one time
point at a time.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence

import numpy
from lxml import etree  # type: ignore[import-untyped]
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: VCell's name for time in its math
TIME = "t"
#: the children of an application (``SimulationSpec``) that pyvcell's model does not keep
KEPT = ("AssignmentRules", "RateRules", "BioEvents")
#: VCML is VCell's own output, read here and handed straight back to VCell; entities and the network are off anyway
_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True)


def _parse(vcml: str) -> etree._Element:
    return etree.fromstring(vcml.encode("utf-8"), _PARSER)  # noqa: S320, RUF100 -- see _PARSER


def _local(tag: object) -> str:
    return etree.QName(tag).localname if isinstance(tag, str) else ""


def _application(root: etree._Element) -> etree._Element | None:
    return next((e for e in root.iter() if _local(e.tag) == "SimulationSpec"), None)


def kept_elements(vcml: str) -> list[bytes]:
    """The assignment rules, rate rules and events of the (first) application of VCell's own VCML, serialized."""
    application = _application(_parse(vcml))
    if application is None:
        return []
    return [etree.tostring(child) for child in application if _local(child.tag) in KEPT]


def fast_reactions(vcml: str) -> list[str]:
    """The reactions the (first) application of VCell's own VCML maps as fast."""
    application = _application(_parse(vcml))
    if application is None:
        return []
    return [
        str(e.get("ReactionStepRef"))
        for e in application.iter()
        if _local(e.tag) == "ReactionSpec" and e.get("ReactionMapping") == "fast"
    ]


def restore(vcml: str, kept: Sequence[bytes], fast: Sequence[str] = ()) -> str:
    """The VCML pyvcell wrote with ``kept`` (from ``kept_elements``) in its application, before its simulations, in
    place of any it wrote itself, and the ``fast`` reactions (from ``fast_reactions``) mapped as fast again."""
    if not kept and not fast:
        return vcml
    root = _parse(vcml)
    application = _application(root)
    if application is None:
        return vcml
    for spec in application.iter():
        if _local(spec.tag) == "ReactionSpec" and spec.get("ReactionStepRef") in fast:
            spec.set("ReactionMapping", "fast")
    for child in [c for c in application if _local(c.tag) in KEPT]:
        application.remove(child)
    position = next((i for i, c in enumerate(application) if _local(c.tag) == "Simulation"), len(application))
    namespace = etree.QName(application).namespace
    for offset, serialized in enumerate(kept):
        element = etree.fromstring(serialized, _PARSER)  # noqa: S320, RUF100 -- see _PARSER
        # in the host document's namespace: pyvcell writes VCML with none, and VCell skips an element in another
        for node in element.iter(tag=etree.Element):
            node.tag = etree.QName(namespace, _local(node.tag)).text if namespace else _local(node.tag)
        etree.cleanup_namespaces(element)
        application.insert(position + offset, element)
    return str(etree.tostring(root, encoding="unicode"))


def math_expressions(vcml: str) -> dict[str, str]:
    """The ``Constant`` and ``Function`` expressions of the VCML document's math description, by name. A document
    with no math (or with several, which a single SBML import never has) gives the first one's or none."""
    for element in _parse(vcml).iter():
        if _local(element.tag) == "MathDescription":
            return {
                child.attrib["Name"]: (child.text or "").strip()
                for child in element
                if _local(child.tag) in ("Constant", "Function") and "Name" in child.attrib
            }
    return {}


def math_names(vcml: str, ids: Iterable[str], compartments: Iterable[str] = ()) -> dict[str, str]:
    """The name in VCell's math of each SBML id, where it is not the id itself: an element VCell renamed keeps the id
    as its ``SbmlName``, a name VCell reserves gets the ``param_`` prefix (``x`` is ``param_x``), and a compartment's
    size is ``Size_`` and its structure's name."""
    root = _parse(vcml)
    renamed = {e.get("SbmlName"): e.get("Name") for e in root.iter() if e.get("SbmlName") and e.get("Name")}
    expressions = math_expressions(vcml)
    names: dict[str, str] = {}
    for sid in ids:
        name = renamed.get(sid, sid)
        for candidate in (name, f"param_{sid}"):
            if candidate in expressions:
                names[sid] = candidate
                break
    for cid in compartments:
        if f"Size_{renamed.get(cid, cid)}" in expressions:
            names[cid] = f"Size_{renamed.get(cid, cid)}"
    return names


def derived_columns(
    expressions: Mapping[str, str],
    times: NDArray[numpy.float64],
    solved: Mapping[str, NDArray[numpy.float64]],
    wanted: Mapping[str, str],
) -> dict[str, NDArray[numpy.float64]]:
    """Each wanted SBML id (``wanted`` maps it to its name in the math, from ``math_names``) that the solver did not
    output, evaluated at every time point, in the order wanted. A name that cannot be evaluated is left out (and
    logged)."""
    import libvcell

    known: dict[str, NDArray[numpy.float64]] = {TIME: numpy.asarray(times, dtype=float), **solved}
    columns: dict[str, NDArray[numpy.float64]] = {}
    for sid, name in wanted.items():
        if sid in solved:
            continue
        try:
            columns[sid] = _evaluate(name, expressions, known, set())
        except (ValueError, KeyError, TypeError, ArithmeticError, libvcell.VCellExpressionError) as e:
            logger.warning("VCell: %s is not reported: %s", sid, e)
    return columns


def _evaluate(
    name: str,
    expressions: Mapping[str, str],
    known: dict[str, NDArray[numpy.float64]],
    pending: set[str],
) -> NDArray[numpy.float64]:
    if name in known:
        return known[name]
    if name in pending:
        msg = f"{name} is defined in terms of itself"
        raise ValueError(msg)
    if name not in expressions:
        msg = f"{name} is not in VCell's math"
        raise KeyError(msg)
    pending.add(name)
    expression = expressions[name]
    symbols = {s for s in _IDENTIFIER.findall(expression) if s in known or s in expressions}
    table = {s: _evaluate(s, expressions, known, pending) for s in symbols}
    pending.discard(name)
    n = len(known[TIME])
    known[name] = numpy.broadcast_to(numpy.asarray(_column(expression, table, n), dtype=float), (n,)).copy()
    return known[name]


def _column(expression: str, table: Mapping[str, NDArray[numpy.float64]], n: int) -> NDArray[numpy.float64]:
    import libvcell
    import numexpr  # type: ignore[import-untyped]

    ok, message, translated = libvcell.vcell_infix_to_num_expr_infix(expression)
    if ok:
        result: NDArray[numpy.float64] = numexpr.evaluate(_numexpr_syntax(translated), local_dict=dict(table))
        return result
    logger.info("VCell: evaluating %r one time point at a time (%s)", expression, message)
    return numpy.array([
        libvcell.evaluate_expression(expression, {s: float(v[i]) for s, v in table.items()}) for i in range(n)
    ])


def _numexpr_syntax(translated: str) -> str:
    """libvcell's numexpr translation in the syntax numexpr evaluates (as pyvcell's own reader adjusts it)."""
    return translated.strip().replace(" and ", " & ").replace(" or ", " | ").replace(" not ", " ~").replace("math.", "")
