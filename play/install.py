"""Install a built Davi engine into santoriniMediator so it can play matches there.

    python play/install.py                      # Joker 3.0.0 -> <mediator>/engines/Davi/Joker
    python play/install.py --release joker-3.0.1 --exe-name Davi_3.0.1_Joker --folder Joker_3.0.1

Copies engine/build/<exe> plus the release's TorchScript/weights/manifest. Never overwrites an
existing engine folder. LibTorch/CUDA DLLs are expected in the mediator's shared runtime
directory (as for Raven/Joker); copy them from engine/build if a machine lacks them.
Matches are then run with the mediator (play/main.py, play/sprt.py there).
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from paths import REPO, MEDIATOR


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', default='joker-3.0.0')
    parser.add_argument('--exe-name', default='Davi_3.0.0_Joker')
    parser.add_argument('--folder', default='Joker', help='folder under <mediator>/engines/Davi')
    args = parser.parse_args()
    exe = REPO / 'engine/build' / f'{args.exe_name}.exe'
    release = REPO / 'weights' / args.release
    if not exe.exists():
        raise SystemExit(f'{exe} not found; run engine\\build.cmd first')
    destination = MEDIATOR / 'engines/Davi' / args.folder
    if destination.exists():
        raise SystemExit(f'{destination} already exists; refusing to overwrite')
    destination.mkdir(parents=True)
    shutil.copy2(exe, destination / exe.name)
    for name in ('santorini_evaluator.pt', 'santorini_zero_weights.pth', 'manifest.json'):
        shutil.copy2(release / name, destination / name)
    record = {p.name: sha(p) for p in destination.iterdir()}
    (destination / 'install.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Installed {args.exe_name} to {destination}')


if __name__ == '__main__':
    main()
