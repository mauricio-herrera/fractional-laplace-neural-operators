#!/usr/bin/env python3
from pathlib import Path

R=Path(__file__).resolve().parents[1]
required=[
    'README.md','LICENSE','LICENSE-DATA.md','CITATION.cff',
    'code/rebuild_seismic_analysis.py',
    'results/seismic/scalar_results.csv',
    'reproducibility/epidemic/scripts/verify_reproducibility.py',
    'data/raw/potin/SOURCE_RECORD.md',
    'data/raw/potin/EXPECTED_SHA256.txt',
]
missing=[p for p in required if not (R/p).exists()]
if missing:
    raise SystemExit('Missing required files: '+', '.join(missing))
if (R/'data/raw/potin/CHILE_SEISMICITY_RELOCATED.csv').exists():
    raise SystemExit('Event-level Potin catalog must not be committed.')
print('Repository verification passed')
