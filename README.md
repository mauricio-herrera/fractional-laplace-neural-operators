# Fractional Laplace Neural Operators

Reproducibility repository for the manuscript

**Fractional Laplace Neural Operators: Exact Architectures, an Expressivity Frontier at Criticality, and Certified Stability for Memory-Driven Network Dynamics**

**Release:** v1.0.0  
**Release date:** 2026-09-18  
**Version-specific Zenodo DOI:** [10.5281/zenodo.22837154](https://doi.org/10.5281/zenodo.22837154)  
**All-versions Zenodo DOI:** [10.5281/zenodo.22837153](https://doi.org/10.5281/zenodo.22837153)

This GitHub repository is the live code/reproducibility mirror. The complete immutable v1.0.0 archival snapshot—including the exact submission-release manuscript and supplement, publication figures, frozen outputs, provenance records, and analysis-ready epidemic files—is preserved on Zenodo under the version-specific DOI above.

The paper's claims are deliberately structural: it does **not** claim universal predictive superiority of fLNO over rational, state-space, FNO, or DeepONet models.

## Repository scope

The GitHub mirror contains the audited analysis code, frozen numerical summaries, provenance records, controlled operator-learning benchmarks, Chilean seismic reconstruction workflow, and Chile/Italy epidemic reproducibility workflows. The Zenodo archive is the canonical immutable snapshot for v1.0.0.

## Primary benchmark

The main benchmark uses the same causal fractional-Volterra data generator and train/validation/test split for fLNO, positive and unconstrained rational pole-residue operators, a diagonal linear state-space model, FNO-1D, and DeepONet. Finite-horizon prediction is reported separately from recovery of the structural distance to criticality.

## Potin seismic source

The seismic analysis uses the relocated Chilean catalog associated with Potin et al., Zenodo DOI `10.5281/zenodo.13146436`.

The event-level file is **not redistributed** in this repository because the source record used for the analysis does not display an explicit redistribution license. Retrieve and checksum-verify the exact file with:

```bash
python scripts/fetch_potin_catalog.py
```

Expected SHA-256:

`0ca7ca6b766c805fb110178c9b08fb438b8838442efcfdf32430de8f61e58362`

Then rebuild the aggregate seismic outputs with:

```bash
python code/rebuild_seismic_analysis.py
```

## Epidemic reproducibility

The Chile/Italy epidemic package reproduces the Chile scalar renewal fit and frozen-kernel control, the Italy network models, the single temporal holdout, and the eight-origin rolling validation.

## Related published foundation

Herrera-Marín, M. (2026). *Cascade dynamics in memory-driven network systems: Fractional Volterra operators, spectral stability criteria, graphon limits, and Hawkes process connections*. **Communications in Nonlinear Science and Numerical Simulation**, 163, 110837. DOI: `10.1016/j.cnsns.2026.110837`.

The reproducibility DOI `10.5281/zenodo.21779954` belongs to that earlier CNSNS foundation and is **not** the DOI of the present fLNO release.

## Archived v1.0.0 release

The immutable reproducibility snapshot corresponding to version `v1.0.0` is archived on Zenodo with DOI [`10.5281/zenodo.22837154`](https://doi.org/10.5281/zenodo.22837154).

## Licensing

Original analysis code is released under the MIT License. Third-party datasets retain their original terms; see `LICENSE-DATA.md`.
