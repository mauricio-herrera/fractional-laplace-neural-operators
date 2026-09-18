# fLNO epidemic reproducibility freeze — Chile and Italy

This package supports the epidemic application of the manuscript **Fractional Laplace Neural Operators: Exact Architectures, an Expressivity Frontier at Criticality, and Certified Stability for Memory-Driven Network Dynamics**.

## Chile scalar renewal stage

- Analysis window: 2020-03-15 to 2022-03-01.
- 24 windows of 28 days; 21-day renewal-kernel support.
- Negative-binomial likelihood with a learned nonnegative kernel and day-of-week effects.
- Learned-vs-frozen-kernel ΔAIC = -62.727.
- Growth/decline classification agreement with the fixed-kernel Cori baseline = 22/24 = 91.7%.
- Median ratio of fLNO NB-profile interval width to fixed-kernel Cori interval width = 9.41.

The width comparison is descriptive and is not a coverage statement.

## Italy 21-region network stage

- Analysis window: 2020-09-01 to 2022-03-01.
- 21 regions / autonomous provinces.
- Full-sample AIC M0/M1/M2 = 132831.9 / 132701.2 / 132685.0.
- M2 coupling strength ε = 0.0374; gravity range ℓ = 403 km.
- Rolling origin, train windows 1..k and score k+1 for k=10..17:
  mean gain M2-M0 = +0.0390 nats/region-day; M2 wins 6/8 origins.

AIC is retained as a structural diagnostic and is not used as a substitute for out-of-sample validation.

## Data provenance

Chile uses an OWID-hosted historical JHU daily-case matrix. Italy uses the official Dipartimento della Protezione Civile regional series. Exact source URLs, analysis windows, and licenses are recorded in `metadata/provenance.csv`.

Large third-party input snapshots are archived with the versioned Zenodo release; they are not duplicated in the GitHub tree.
