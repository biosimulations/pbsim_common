"""What a time course reports, read the way SED-ML reads an SBML model.

SED-ML's value for a species is its amount when it has only substance units (``hasOnlySubstanceUnits``) and its
concentration otherwise; a parameter's or compartment's is its value or size. A step reports every species that way,
and every parameter and compartment that is not constant, each under its SBML id, so a SED-ML data generator finds
the column it names. A constant one keeps the model's number, which the translator reads itself.

The SBML Test Suite found the gap (compose-api's sbml-test-suite-sample, run 1, 2026-11-05): every step reported
concentrations only, and no parameter or compartment.
"""

from dataclasses import dataclass, field

import libsbml  # type: ignore[import-untyped]


@dataclass(frozen=True)
class SedmlValues:
    #: species whose SED-ML value is an amount (they have only substance units), by SBML id
    amount_species: frozenset[str] = frozenset()
    #: each species' compartment, by SBML id
    compartment_of: dict[str, str] = field(default_factory=dict)
    #: parameters not marked constant, or that a rule or an event changes (SBML Level 1 has no `constant`), in model
    #: order: what a SED-ML translator reads from the result rather than from the model (process-bigraph-lang does)
    varying_parameters: tuple[str, ...] = ()
    #: compartments the same way
    varying_compartments: tuple[str, ...] = ()
    #: each compartment's size, where it is a number for the whole time course: set, constant, and not set by an
    #: initial assignment or a rule
    sizes: dict[str, float] = field(default_factory=dict)


def read_sedml_values(path: str) -> SedmlValues:
    """What a step reports for the model at ``path``. A model libsbml cannot read reports as before (species as
    concentrations): the engine's own error is the better message."""
    document = libsbml.readSBMLFromFile(path)
    model = document.getModel() if document is not None else None
    if model is None:
        return SedmlValues()
    changed = {r.getVariable() for r in model.getListOfRules() if r.isSetVariable()} | {
        a.getVariable() for e in model.getListOfEvents() for a in e.getListOfEventAssignments()
    }
    return SedmlValues(
        amount_species=frozenset(s.getId() for s in model.getListOfSpecies() if s.getHasOnlySubstanceUnits()),
        compartment_of={s.getId(): s.getCompartment() for s in model.getListOfSpecies()},
        varying_parameters=tuple(
            p.getId() for p in model.getListOfParameters() if not p.getConstant() or p.getId() in changed
        ),
        varying_compartments=tuple(
            c.getId() for c in model.getListOfCompartments() if not c.getConstant() or c.getId() in changed
        ),
        sizes={
            c.getId(): c.getSize()
            for c in model.getListOfCompartments()
            if c.isSetSize()
            and c.getConstant()
            and model.getInitialAssignmentBySymbol(c.getId()) is None
            and model.getRuleByVariable(c.getId()) is None
        },
    )
