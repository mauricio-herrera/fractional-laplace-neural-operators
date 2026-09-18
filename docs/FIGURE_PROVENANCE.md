# Figure provenance

This file records the canonical computational source for the quantitative figures.

| Figure family | Canonical source |
|---|---|
| Controlled benchmark / finite-horizon extrapolation | `code/operator_benchmark/run_operator_benchmark.py` and frozen files under `results/operator_benchmark/` |
| Near-critical and stability stress diagnostics | primary benchmark code plus `code/operator_benchmark/run_stress_unconstrained_v23.py` |
| Graph-size transfer | graph phase of `run_operator_benchmark.py` |
| Seismic scalar and spatial summaries | `code/rebuild_seismic_analysis.py` and `results/seismic/` |
| Chile epidemic scalar analysis | scripts under `reproducibility/epidemic/scripts/` |
| Italy spatial and rolling validation | scripts under `reproducibility/epidemic/scripts/` |

A secondary frozen-protocol operator replication is retained separately in the Zenodo archive. Its results are a robustness check and are not pooled numerically with the primary benchmark.
