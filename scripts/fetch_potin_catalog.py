#!/usr/bin/env python3
from pathlib import Path
import hashlib, io, zipfile, urllib.request

URL='https://zenodo.org/records/13146436/files/chilean_seismic_catalogue-1982-2020-publication_release.zip?download=1'
EXPECTED='0ca7ca6b766c805fb110178c9b08fb438b8838442efcfdf32430de8f61e58362'
TARGET=Path(__file__).resolve().parents[1]/'data/raw/potin/CHILE_SEISMICITY_RELOCATED.csv'

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print('Downloading Potin et al. Zenodo release...')
    with urllib.request.urlopen(URL) as r:
        archive=r.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        names=[n for n in z.namelist() if n.endswith('CHILE_SEISMICITY_RELOCATED.csv')]
        if len(names)!=1:
            raise RuntimeError(f'Expected exactly one relocated catalog, found {names}')
        data=z.read(names[0])
    got=sha256_bytes(data)
    if got!=EXPECTED:
        raise RuntimeError(f'SHA-256 mismatch: expected {EXPECTED}, got {got}')
    TARGET.write_bytes(data)
    print(f'Wrote {TARGET}')
    print(f'SHA-256 {got}')

if __name__=='__main__':
    main()
