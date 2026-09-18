# Reproducibility guide

The repository separates four computational blocks so that a reviewer can reproduce them independently.

## 1. Controlled operator benchmark

Install the benchmark environment and run:

```bash
python code/operator_benchmark/run_operator_benchmark.py --phase main
python code/operator_benchmark/run_operator_benchmark.py --phase near
python code/operator_benchmark/run_operator_benchmark.py --phase graph
python code/operator_benchmark/run_stress_unconstrained_v23.py
```

The primary generator integrates the causal memory equation directly. It does not use the startup-inconsistent impulse-convolution construction found in an early discarded development branch.

## 2. Chile seismic analysis

Retrieve the public source:

```bash
python scripts/fetch_potin_catalog.py
```

Then run:

```bash
python code/rebuild_seismic_analysis.py
```

The primary cohort uses Tocopilla 2007, Maule 2010, Iquique 2014 and Illapel 2015, a 350-km radius, a 365-day observation window, a 180-day memory window, sequence-specific completeness thresholds, official mainshock anchors, and deterministic PCA/tertile along-strike zoning.

## 3. Epidemic analyses

The versioned Zenodo archive contains the frozen input snapshots. Once they are placed under `reproducibility/epidemic/data/`, the analysis scripts reproduce the Chile scalar fit and Italy spatial/rolling validation.

Quick numerical target check:

```bash
python reproducibility/epidemic/scripts/verify_reproducibility.py
```

## 4. Integrity

Run:

```bash
python scripts/verify_repository.py
```

The verifier also fails if the event-level Potin catalog has accidentally been committed.

## Interpretation of numerical reproducibility

Optimization-dependent values can change in the last few digits across BLAS/Python/PyTorch builds. Claims in the manuscript are tied to the frozen outputs archived with release v1.0.0; reruns are expected to agree to scientific, rather than bitwise, precision unless the exact frozen environment is used.
