# Fractional Laplace Neural Operators

Reproducibility repository for the manuscript **Fractional Laplace Neural Operators: Exact Architectures, an Expressivity Frontier at Criticality, and Certified Stability for Memory-Driven Network Dynamics**.

This repository contains the audited code, frozen numerical outputs, data-provenance records, controlled operator-learning benchmarks, the Chilean seismic workflow, and the Chile/Italy epidemic reproducibility package supporting the study.

## Scientific scope

The repository supports five distinct claims:

1. exact block-triangular fractional Laplace neural-operator representation on the commuting linear Volterra class;
2. a rational expressivity frontier separating finite-window approximation from exact fractional critical asymptotics;
3. by-construction stability parametrizations tied to the Volterra branching threshold;
4. controlled common-data benchmarks separating finite-horizon prediction from structural critical-coordinate recovery and graph-size transfer;
5. real-data seismic and epidemic applications with explicit sensitivity and out-of-sample diagnostics.

A secondary frozen-protocol replication is retained as a robustness check and is not pooled with the primary benchmark.

## Potin seismic source

The seismic analysis uses Potin et al., *A Revised Chilean Seismic Catalog from 1982 to Mid-2020*, Zenodo record **10.5281/zenodo.13146436**.

The event-level catalog is intentionally not republished here because the current Zenodo record does not display an explicit redistribution license. Retrieve and verify the exact file with:

```bash
python scripts/fetch_potin_catalog.py
```

Expected SHA-256:

`0ca7ca6b766c805fb110178c9b08fb438b8838442efcfdf32430de8f61e58362`

Then rebuild the seismic analysis with:

```bash
python code/rebuild_seismic_analysis.py
```

## Epidemic reproducibility

The Chile/Italy epidemic package reproduces the Chile scalar fit, the Italy network models, the single temporal holdout, and the eight-origin rolling validation.

## Related published foundation

Herrera-Marín, M. (2026). *Cascade dynamics in memory-driven network systems: Fractional Volterra operators, spectral stability criteria, graphon limits, and Hawkes process connections*. **Communications in Nonlinear Science and Numerical Simulation**, 163, 110837. https://doi.org/10.1016/j.cnsns.2026.110837

## License

Original analysis code is released under the MIT License. Third-party datasets retain their original terms; see `LICENSE-DATA.md`.
