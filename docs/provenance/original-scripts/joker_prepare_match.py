"""Freeze the built artifact, install it additively, and prepare a paired runner."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACT = HERE / 'artifacts/Davi_3.0.0_Joker'
MATCH = HERE.parent / 'joker-vs-raven-200-2026-11-17'
MED = Path('C:/Users/rafae/PycharmProjects/santoriniMediator')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    exe = ARTIFACT / 'Davi_3.0.0_Joker.exe'
    assert exe.exists()
    manifest = json.loads((HERE / 'manifest.json').read_text())
    manifest['status'] = 'release executable built; protocol/legal-move preflight pending'
    manifest['files'] = {p.name: sha(p) for p in ARTIFACT.iterdir() if p.suffix in ('.exe', '.pt', '.pth')}
    manifest['build'] = {'toolchain': 'MSVC 2022 x64 /O2 /MD /std:c++20',
                         'build_command_sha256': sha(HERE / 'build.cmd'),
                         'source_sha256': {p.name: sha(p) for p in (HERE / 'source/src').iterdir()},
                         'log': str(HERE / 'build.log')}
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (ARTIFACT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    destination = MED / 'engines/Davi/Joker'
    if destination.exists(): raise RuntimeError('Joker engine folder already exists; refusing overwrite')
    destination.mkdir()
    for filename in ('Davi_3.0.0_Joker.exe', 'santorini_evaluator.pt', 'santorini_zero_weights.pth', 'manifest.json'):
        shutil.copy2(ARTIFACT / filename, destination / filename)
    original = (HERE.parent / 'cypher-vs-raven-200-2026-11-17/run.py').read_text()
    original = original.replace('Cypher', 'Joker').replace('cypher-3.0.0-2026-11-17', 'joker-3.0.0-2026-11-17')
    original = original.replace("    assert builds['current']['sha256'] == builds['new']['sha256']", "    assert builds['current']['sha256'] != builds['new']['sha256']  # Joker has a rebuilt 35-plane runtime")
    MATCH.mkdir()
    (MATCH / 'run.py').write_text(original)
    launch = (HERE.parent / 'cypher-vs-raven-200-2026-11-17/launch.ps1').read_text().replace('Cypher', 'Joker')
    (MATCH / 'launch.ps1').write_text(launch)
    print('Installed', destination)
    print('Prepared runner', MATCH)


if __name__ == '__main__': main()
