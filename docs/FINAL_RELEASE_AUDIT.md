# Final release audit - v1.0.0

Date: 2026-09-18

## Scientific freeze

The scientific claims and frozen numerical results remain those of the final handoff. Final manuscript-content changes were limited to (i) a targeted related-work update covering Memory Neural Operator (ICLR 2025), Mamba Neural Operator (JCP 2026), and Physics-Informed Laplace Neural Operator (JCP 2026), and (ii) a concise abstract trim. No theorem, experiment, table value, or empirical conclusion was changed.

## Manuscript and supplement

- Main manuscript: 25 pages.
- Supplement: 3 pages.
- Clean final LaTeX build after the literature/DOI update.
- No undefined citations or references in the final build.
- PDF preflight: openable, unencrypted, text-based rather than scanned.
- Publication figures and supplement figure inspected without clipping/overlap.

## Numerical/reproducibility checks

Root release integrity:

```text
Repository verification passed (147 manifested files)
```

Epidemic frozen-package invariant checker:

```text
Reproducibility verification passed
Chile: Delta AIC -62.727; median width ratio 9.41
Italy: AIC M0/M2 132831.9/132685.0; rolling gain +0.0390; M2 wins 6/8 origins
```

The Potin event-level catalog remains absent from the public release and is retrieved/checksum-verified externally as documented.

## Packaging correction

The incoming reproducibility bundle contained a 24-page development manuscript snapshot while the submission package contained the later 25-page final manuscript. The canonical v1.0.0 Zenodo release contains the exact final manuscript and supplement used for submission.

## DOI status

The immutable v1.0.0 archive has been published on Zenodo.

- Version-specific DOI: `10.5281/zenodo.22837154`
- All-versions DOI: `10.5281/zenodo.22837153`

The version-specific DOI is propagated to the manuscript, README, and CITATION metadata.

## GitHub status

GitHub is the live code/reproducibility mirror. The Zenodo record is the immutable archival source of truth for v1.0.0.
