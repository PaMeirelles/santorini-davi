"""Train the requested four-context-plane model on EXACT Raven 2.3.9 games/splits."""
import argparse
import ctypes
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
import numpy as np

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
from context import RECORD, CONTEXT_RECORD, DATASET, JOKER_RUN as HERE, context_planes, tensorize as encode
HERE.mkdir(parents=True, exist_ok=True)


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def save_json(name, value):
    p = HERE / name
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(p)


def event(kind, **kwargs):
    row = {'at': now(), 'event': kind, **kwargs}
    with (HERE / ('smoke-events.jsonl' if SMOKE else 'events.jsonl')).open('a') as f:
        f.write(json.dumps(row) + '\n')
    save_json('smoke-status.json' if SMOKE else 'status.json', row)
    print(json.dumps(row), flush=True)


def stop_requested():
    if (HERE / 'STOP').exists(): raise InterruptedError('STOP requested')


class ContextRows:
    def __init__(self):
        self.rows = np.memmap(DATASET / 'samples.bin', dtype=RECORD, mode='r')
        self.contexts = np.memmap(HERE / 'context.bin', dtype=CONTEXT_RECORD, mode='r')
        if len(self.rows) != len(self.contexts): raise ValueError('Context alignment mismatch')
        self.dtype = np.dtype(RECORD.descr + [('turn_context', CONTEXT_RECORD)])

    def __len__(self): return len(self.rows)

    def __getitem__(self, indices):
        base = self.rows[indices]
        enriched = np.empty(base.shape, dtype=self.dtype)
        for name in RECORD.names: enriched[name] = base[name]
        enriched['turn_context'] = self.contexts[indices]
        return enriched


def split_data(path, manifest, smoke=False):
    data = ContextRows()
    train = np.load(DATASET / 'train-indices.npy')
    validation = np.load(DATASET / 'validation-indices.npy')
    assert len(data) == manifest['samples']
    assert len(train) == manifest['split']['train_samples']
    assert len(validation) == manifest['split']['validation_samples']
    return data, train, validation


def tensorize(rows, device): return encode(rows, rows['turn_context'], device)


def load_architecture():
    spec = importlib.util.spec_from_file_location('context35_frozen_network', SOURCE / 'network.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SantoriniZeroNet


exec(compile((SOURCE / '_train_loop.py').read_text(), str(SOURCE / '_train_loop.py'), 'exec'))
SMOKE = False


def main():
    global SMOKE
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    SMOKE = args.smoke
    lock = HERE / 'running.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        event('preflight', pid=os.getpid(), smoke=SMOKE)
        dataset = json.loads((HERE / 'dataset.json').read_text())
        for name, expected in dataset['source_sha256'].items():
            if sha(DATASET / name) != expected: raise ValueError(f'Frozen source changed: {name}')
        if sha(HERE / 'context.bin') != dataset['context_sha256']: raise ValueError('Context changed')
        assert json.loads((DATASET / 'split-audit.json').read_text())['remaining_full_d4_overlap'] == 0
        import torch
        assert torch.cuda.is_available(), 'CUDA required'
        run = {'at': now(), 'pid': os.getpid(), 'python': sys.executable,
               'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
               'dataset': str(DATASET), 'dataset_manifest_sha256': sha(HERE / 'dataset.json'),
               'scope': 'Exact Raven 2.3.9 accepted games and split; four additional input planes only',
               'comparison_note': 'Compare with Raven 2.3.9; comparison with Cypher 3.0 also changes the corpus',
               'games': 150282, 'result_counts': {'N': 143792, 'T': 4872, 'I': 1618},
               'input_planes': 35, 'train_samples': 7662868, 'validation_samples': 1460783,
               'recipe': {'epochs': 15, 'batch_size': 256, 'optimizer': 'Adam', 'learning_rate': .001,
                          'seed': 42, 'initialization': 'from scratch', 'precision': 'float32',
                          'sample_weights': 'uniform', 'value_loss': 'BCEWithLogitsLoss',
                          'policy_loss': 'CrossEntropyLoss', 'loss_weights': [1, 1]},
               'sources': {str(p.relative_to(SOURCE.parent)): sha(p) for p in [Path(__file__), *sorted(SOURCE.glob('*.py'))]},
               'artifact': 'weights and TorchScript only; no executable or engine promotion'}
        save_json('smoke-run-manifest.json' if SMOKE else 'run-manifest.json', run)
        train(DATASET / 'samples.bin', dataset, SMOKE)
    except BaseException as exc:
        event('stopped' if isinstance(exc, InterruptedError) else 'failed', error=repr(exc), traceback=traceback.format_exc())
        if not isinstance(exc, InterruptedError): raise
    finally:
        lock.unlink(missing_ok=True)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__': main()
