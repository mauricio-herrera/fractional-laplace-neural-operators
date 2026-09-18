# Third-party data and redistribution

The MIT license in `LICENSE` applies only to original analysis code in this repository. It does not relicense third-party datasets.

## Potin et al. Chilean seismic catalog

The seismic analysis uses the public Zenodo v2 publication release at https://doi.org/10.5281/zenodo.13146436. The current Zenodo record does not display an explicit license in its Rights field. Consequently, the event-level table and event-level extracted subsets are **not redistributed here**. `scripts/fetch_potin_catalog.py` retrieves the official archive and verifies the exact SHA-256 used by the analysis. Aggregate fitted outputs may be distributed.

## JHU CSSE / OWID historical Chile COVID-19 cases

The Chile daily-case series is based on JHU CSSE COVID-19 data distributed under CC BY 4.0. Attribute the COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University and cite Dong, Du & Gardner (2020), DOI 10.1016/S1473-3099(20)30120-1.

## Italian Protezione Civile COVID-19 data

The regional Italian series is from the Dipartimento della Protezione Civile COVID-19 repository and is distributed under CC BY 4.0.

## Italian population weights

The exact rounded population weights used computationally are derived from ISTAT resident-population statistics. Consult ISTAT for authoritative statistics and applicable terms.
