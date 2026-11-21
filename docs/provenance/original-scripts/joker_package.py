"""Freeze Joker 3.0.0 at the explicitly selected context35 epoch 3."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAINING = HERE.parent / 'raven-context-35-training-2026-11-17'
RUNTIME = HERE.parent / 'cypher-context-35-2026-11-17/runtime-src'
ARTIFACT = HERE / 'artifacts/Davi_3.0.0_Joker'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if ARTIFACT.exists(): raise RuntimeError('Refusing to overwrite Joker artifact')
    assert json.loads((TRAINING / 'status.json').read_text())['event'] == 'stopped'
    sys.path.insert(0, str(TRAINING))
    spec = importlib.util.spec_from_file_location('joker_training', TRAINING / 'run.py')
    training = importlib.util.module_from_spec(spec); spec.loader.exec_module(training)
    torch.set_num_threads(4)
    weights = TRAINING / 'checkpoints/epoch-03-weights.pth'
    model = training.load_architecture()().cpu().eval()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    ARTIFACT.mkdir(parents=True)
    shutil.copy2(weights, ARTIFACT / 'santorini_zero_weights.pth')
    indices = __import__('numpy').load(training.DATASET / 'validation-indices.npy')[:32]
    example, _, _ = training.tensorize(training.ContextRows()[indices], 'cpu')
    with torch.inference_mode():
        traced = torch.jit.trace(model, example[:1])
        traced.save(str(ARTIFACT / 'santorini_evaluator.pt'))
        restored = torch.jit.load(str(ARTIFACT / 'santorini_evaluator.pt')).eval()
        for expected, actual in zip(model(example), restored(example)):
            torch.testing.assert_close(expected, actual)
    runtime_manifest = json.loads((RUNTIME / 'manifest.json').read_text())
    for name, digest in runtime_manifest['prepared_sha256'].items():
        assert sha(RUNTIME / name) == digest
    shutil.copytree(RUNTIME / 'src', HERE / 'source/src')
    manifest = {'name': 'Davi 3.0.0 Joker', 'selected_epoch': 3,
                'weights_source': str(weights), 'weights_sha256': sha(weights),
                'training_run': str(TRAINING), 'training_manifest_sha256': sha(TRAINING / 'run-manifest.json'),
                'input_planes': 35, 'parameters': 4608646,
                'architecture_sha256': sha(TRAINING / 'source/network.py'),
                'torchscript_sha256': sha(ARTIFACT / 'santorini_evaluator.pt'),
                'torchscript_reload_output_check': 'passed on 32 actual held-out sample rows',
                'runtime_source_association': runtime_manifest,
                'changes': 'Four input context planes; existing phases, rules and search retained.',
                'limitations': 'Historical source association does not establish identity with archived Raven EXE. '
                               'Prometheus pre-build phase remains unreachable in inherited source.',
                'status': 'weights exported; executable build pending'}
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__': main()
