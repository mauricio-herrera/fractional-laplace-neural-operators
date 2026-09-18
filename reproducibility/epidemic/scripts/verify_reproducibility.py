from pathlib import Path
import json, sys

ROOT=Path(__file__).resolve().parent.parent
errors=[]
required=[
    ROOT/'metadata'/'provenance.csv',
    ROOT/'metadata'/'versions_validated.txt',
    ROOT/'results'/'epidemic_summary.json',
    ROOT/'results'/'italy_rolling_origin_summary.json',
]
for p in required:
    if not p.exists():
        errors.append(f'missing: {p.relative_to(ROOT)}')

if not errors:
    s=json.loads((ROOT/'results'/'epidemic_summary.json').read_text())
    def near(a,b,tol,label):
        if abs(a-b)>tol: errors.append(f'{label}: {a} != {b} within {tol}')
    near(s['chile']['delta_AIC_learned_minus_frozen'],-62.7266,0.02,'Chile delta AIC')
    near(s['chile']['width_ratio_median'],9.412,0.02,'Chile median width ratio')
    near(s['italy']['AIC_M0'],132831.9,0.2,'Italy M0 AIC')
    near(s['italy']['AIC_M2'],132685.0,0.2,'Italy M2 AIC')
    near(s['italy']['rolling_mean_gain'],0.03895,0.002,'Italy rolling mean gain')
    if s['italy']['rolling_positive_origins']!=6:
        errors.append('Italy rolling positive-origin count failed')

if errors:
    print('REPRODUCIBILITY VERIFICATION FAILED')
    for e in errors: print(' -',e)
    sys.exit(1)

print('Reproducibility verification passed')
