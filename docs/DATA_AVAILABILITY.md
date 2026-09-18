# Data availability and provenance

## Controlled benchmarks

The operator-learning benchmarks are generated synthetically by the frozen scripts under `code/operator_benchmark/`; aggregate and per-run outputs used for the manuscript are preserved in the v1.0.0 Zenodo archive.

## Chilean seismic application

The source is the relocated Chilean catalog associated with Potin et al., Zenodo DOI `10.5281/zenodo.13146436`. The event-level source table is not redistributed here because the source record used for this release does not display an explicit redistribution license. `scripts/fetch_potin_catalog.py` retrieves the exact file and verifies SHA-256 `0ca7ca6b766c805fb110178c9b08fb438b8838442efcfdf32430de8f61e58362`. Aggregate fitted outputs are included.

## Epidemic application

The Chile and Italy data/provenance freeze is documented in the archived v1.0.0 reproducibility package. The Zenodo release contains the exact analysis-ready files used by the frozen epidemic scripts together with their source and licensing records.

## Related foundation archive

DOI `10.5281/zenodo.21779954` is the reproducibility archive of the previously published CNSNS foundation. It is not the DOI of this fLNO repository.

## Present release DOI

The version-specific archive of this v1.0.0 release is Zenodo DOI `10.5281/zenodo.22837154`. The all-versions DOI is `10.5281/zenodo.22837153`.
