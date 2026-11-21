"""Freeze a chosen training epoch as a release: weights + TorchScript in weights/<name>/.

    python export_release.py --epoch 3 --name joker-3.0.0

Joker 3.0.0 was released from epoch 3 (user decision, Nov 17 2026), not the final epoch.
Adapted from experiments/joker-3.0.0-2026-11-17/package.py (see docs/provenance/original-scripts).
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_joker as training
from paths import REPO, ENGINE_SRC


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--name', required=True, help='release folder under weights/, e.g. joker-3.0.1')
    args = parser.parse_args()
    artifact = REPO / 'weights' / args.name
    if artifact.exists():
        raise RuntimeError(f'Refusing to overwrite {artifact}')
    run = training.HERE
    weights = run / f'checkpoints/epoch-{args.epoch:02d}-weights.pth'
    torch.set_num_threads(4)
    model = training.load_architecture()().cpu().eval()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    artifact.mkdir(parents=True)
    shutil.copy2(weights, artifact / 'santorini_zero_weights.pth')
    indices = np.load(training.DATASET / 'validation-indices.npy')[:32]
    example, _, _ = training.tensorize(training.ContextRows()[indices], 'cpu')
    with torch.inference_mode():
        traced = torch.jit.trace(model, example[:1])
        traced.save(str(artifact / 'santorini_evaluator.pt'))
        restored = torch.jit.load(str(artifact / 'santorini_evaluator.pt')).eval()
        for expected, actual in zip(model(example), restored(example)):
            torch.testing.assert_close(expected, actual)
    manifest = {'name': args.name, 'selected_epoch': args.epoch,
                'weights_sha256': sha(weights), 'training_run': str(run),
                'training_manifest_sha256': sha(run / 'run-manifest.json') if (run / 'run-manifest.json').exists() else None,
                'input_planes': 35, 'parameters': sum(p.numel() for p in model.parameters()),
                'architecture_sha256': sha(training.SOURCE / 'network.py'),
                'torchscript_sha256': sha(artifact / 'santorini_evaluator.pt'),
                'torchscript_reload_output_check': 'passed on 32 held-out sample rows',
                'engine_source_sha256': {p.name: sha(p) for p in sorted(ENGINE_SRC.iterdir())}}
    (artifact / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
