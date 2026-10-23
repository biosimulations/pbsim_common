# BioModels regression models

SBML files from the BioModels database (CC0; `https://www.ebi.ac.uk/biomodels/<id>`, the `_url.xml` file of each
entry's archive). Each one failed in compose-api's `biomodels-curated-300` suite run on 2026-10-22 because of an adapter
bug fixed here:

| Model | What it exercises |
|---|---|
| BIOMD0000000001 | one compartment of volume 1e-16: concentrations written back as amounts changed the run (Tellurium) |
| BIOMD0000000007 | species defined by assignment rules (`Cig2Total`), which RoadRunner refuses to set (Tellurium) |
| BIOMD0000000020 | no species and no reactions, only rate rules on parameters (COPASI) |
| BIOMD0000000143 | 20 species whose display names repeat across compartments (COPASI) |
| BIOMD0000000241 | species but no reactions (COPASI) |
