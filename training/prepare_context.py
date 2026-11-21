"""Prepare context sidecar only. Does not train, export a model, or launch games."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from context import HERE, BASE, DATASET, JOKER_RUN, RECORD, CONTEXT_RECORD, PLANE_NAMES, context_records, canonical_key


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as inp:
        for chunk in iter(lambda: inp.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    JOKER_RUN.mkdir(parents=True, exist_ok=True)
    target = JOKER_RUN / 'context.bin'
    manifest = JOKER_RUN / 'dataset.json'
    if manifest.exists() or target.exists():
        raise RuntimeError('Refusing to overwrite prepared context data')
    # Joker 3.0.0 was trained on exactly this frozen Raven 2.3.9 corpus. A different corpus is
    # allowed only with DAVI_ALLOW_NEW_CORPUS=1 (and is then a different model, not Joker 3.0.0).
    frozen = json.loads((BASE.parent / 'docs/provenance/joker-context-dataset.json').read_text())['source_sha256']
    source = {'outputs': {} if os.environ.get('DAVI_ALLOW_NEW_CORPUS') == '1'
              else {k: {'sha256': v} for k, v in frozen.items()}}
    fingerprints = {}
    for name in ('samples.bin', 'accepted-games.jsonl', 'train-indices.npy',
                 'validation-indices.npy', 'dataset.json', 'split-audit.json'):
        digest = sha(DATASET / name)
        if name in source['outputs'] and digest != source['outputs'][name]['sha256']:
            raise ValueError(f'Frozen source changed: {name}')
        fingerprints[name] = digest
    audit = json.loads((DATASET / 'split-audit.json').read_text())
    if audit['remaining_full_d4_overlap'] != 0:
        raise ValueError('Inherited D4 leakage audit failed')
    rows = np.memmap(DATASET / 'samples.bin', dtype=RECORD, mode='r')
    counts = np.zeros(4, dtype=np.int64)
    cursor = games = 0
    began = time.monotonic()
    temporary = target.with_suffix('.tmp')
    with temporary.open('wb') as out, (DATASET / 'accepted-games.jsonl').open() as inp:
        for line in inp:
            game = json.loads(line)
            if canonical_key(game['position']) != game['starting_symmetry_key']:
                raise ValueError('Starting-position D4 identity mismatch')
            start, size = game['row_start'], game['row_count']
            if start != cursor:
                raise ValueError('Noncontiguous source rows')
            data = context_records(rows[start:start + size])
            for i, name in enumerate(PLANE_NAMES):
                counts[i] += np.count_nonzero(data[name] if i == 3 else data[name] >= 0)
            out.write(data.tobytes())
            cursor += size
            games += 1
            if games % 10000 == 0:
                print(f'{games} games, {cursor} sample records, {time.monotonic()-began:.1f}s', flush=True)
    if cursor != len(rows) or temporary.stat().st_size != len(rows) * 4:
        raise ValueError('Sidecar alignment/length mismatch')
    temporary.replace(target)
    result = {'status': 'prepared; no training started', 'input_planes': 35,
              'phase_planes': '28-30 unchanged', 'context_planes': dict(enumerate(PLANE_NAMES, 31)),
              'record_dtype': CONTEXT_RECORD.descr, 'absent_square': -1,
              'source_dataset': str(DATASET), 'source_sha256': fingerprints,
              'baseline_architecture_sha256': sha(BASE / 'network_31.py'),
              'samples': cursor, 'games': games, 'context_sha256': sha(target),
              'nonempty_context_sample_records': dict(zip(PLANE_NAMES, map(int, counts))),
              'split': {k: audit[k] for k in ('train_samples', 'validation_samples')},
              'symmetry': 'Same rows, original orientations, labels, repetitions and saved splits. '
                          'Inherited board-only D4 overlap purge is conservative for enriched contexts; '
                          'no new symmetry-distinct context count claimed.',
              'preparation_seconds': time.monotonic() - began}
    manifest.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
