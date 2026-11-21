"""Compare checkpoints on the identical Joker validation split (value/policy loss and accuracy).

    python compare_holdout.py --model "Joker 3.0.0=../weights/joker-3.0.0/santorini_zero_weights.pth:35" \
                              --model "Raven 2.3.9=<corpus>/checkpoints/best-weights.pth:31"

Each model is NAME=PATH:PLANES, PLANES 35 (Joker context net) or 31 (Raven net, fed planes 0-30).
Policy accuracy is imitation of recorded actions, not best-move accuracy; this is not a strength test.
Adapted from experiments/raven-context-35-training-2026-11-17/compare_holdout.py.
"""
import argparse
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_joker as run
from context import CONTEXT_RECORD, baseline, tensorize


def parse(spec):
    name, rest = spec.split('=', 1)
    path, planes = rest.rsplit(':', 1)
    if planes not in ('31', '35'):
        raise argparse.ArgumentTypeError('PLANES must be 31 or 35')
    return name, Path(path), int(planes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='append', type=parse, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    stamp = run.now().replace(':', '').replace('+', '_')
    out = run.HERE / 'comparisons' / stamp
    out.mkdir(parents=True)

    models, planes, metadata = {}, {}, {}
    for name, path, n in args.model:
        contents = path.read_bytes()
        model = (run.load_architecture() if n == 35 else baseline.load_architecture())().to(device).eval()
        model.load_state_dict(torch.load(io.BytesIO(contents), map_location=device, weights_only=True))
        models[name], planes[name] = model, n
        metadata[name] = {'source': str(path), 'planes': n, 'sha256': run.hashlib.sha256(contents).hexdigest()}

    dataset = json.loads((run.HERE / 'dataset.json').read_text())
    rows = np.memmap(run.DATASET / 'samples.bin', dtype=run.RECORD, mode='r')
    contexts = np.memmap(run.HERE / 'context.bin', dtype=CONTEXT_RECORD, mode='r')
    assert len(rows) == len(contexts) == dataset['samples']
    indices = np.load(run.DATASET / 'validation-indices.npy')
    totals = {name: torch.zeros((4, 5), dtype=torch.float64, device=device) for name in models}
    began = time.monotonic()
    with torch.inference_mode():
        for begin in range(0, len(indices), 256):
            ids = indices[begin:begin + 256]
            batch = rows[ids]
            x, yv, yp = tensorize(batch, contexts[ids], device)
            phases = torch.from_numpy(batch['phase'].copy()).to(device)
            for name, model in models.items():
                policy, value = model(x if planes[name] == 35 else x[:, :31].contiguous())
                lv = F.binary_cross_entropy_with_logits(value, yv, reduction='none').flatten()
                lp = F.cross_entropy(policy, yp, reduction='none')
                cv = ((value > 0) == (yv > .5)).flatten()
                cp = policy.argmax(1) == yp
                metrics = torch.stack((lv, lp, cv.float(), cp.float(), torch.ones_like(lv)), dim=1).double()
                masks = torch.stack((torch.ones_like(phases), phases == 0, phases == 1, phases == 2)).double()
                totals[name] += masks @ metrics
            if begin % (256 * 1000) == 0:
                print(f'{begin}/{len(indices)} holdout examples; {time.monotonic()-began:.1f}s', flush=True)

    result = {'at': run.now(), 'samples': len(indices), 'seconds': time.monotonic() - began,
              'scope': 'Joker validation split with inherited D4 overlap purge. Validation, not an untouched test set.',
              'metrics': 'Policy accuracy is imitation of recorded actions, not best-move accuracy. No playing-strength test.',
              'models': {}}
    for name, total in totals.items():
        total = total.cpu().numpy()
        result['models'][name] = metadata[name] | {'groups': {}}
        for group, label in enumerate(('all', 'move', 'first_build', 'second_or_prebuild')):
            v = total[group]
            m = dict(zip(('value_loss', 'policy_loss', 'value_accuracy', 'policy_accuracy'), (v[:4] / v[4]).tolist()))
            m['combined_loss'] = m['value_loss'] + m['policy_loss']
            m['samples'] = int(v[4])
            result['models'][name]['groups'][label] = m
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
