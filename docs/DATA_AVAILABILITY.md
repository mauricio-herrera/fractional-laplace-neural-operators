# Data availability and redistribution policy

## Chilean seismic application

The primary seismic cohort is reconstructed from the relocated Chilean catalog associated with Potin et al., *A Revised Chilean Seismic Catalog from 1982 to Mid-2020*.

- Dataset release: https://doi.org/10.5281/zenodo.13146436
- Expected event table: `CHILE_SEISMICITY_RELOCATED.csv`
- Expected SHA-256: `0ca7ca6b766c805fb110178c9b08fb438b8838442efcfdf32430de8f61e58362`

The event-level Potin table is not redistributed in this GitHub repository because the current Zenodo record does not display an explicit redistribution license. Run:

```bash
python scripts/fetch_potin_catalog.py
```

to retrieve and verify the exact public source before rebuilding the seismic analysis.

## Epidemic application

The Chile daily-case series is derived from the historical JHU CSSE series distributed through the OWID repository. The Italian regional series is from the official Dipartimento della Protezione Civile COVID-19 repository. Their source records and terms are listed in `reproducibility/epidemic/metadata/provenance.csv`.

The versioned Zenodo reproducibility archive will contain the frozen third-party epidemic snapshots used by the computations together with their provenance and original-source licensing notes.

## Derived outputs

Aggregate fitted parameters, frozen numerical benchmark outputs, sensitivity summaries, and non-event-level derived tables produced by the authors are included with the reproducibility materials.
