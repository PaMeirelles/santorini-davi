"""Four additional context planes; no phase, action or target changes."""
from dataclasses import dataclass, replace
import importlib.util
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import CORPUS as DATASET, JOKER_RUN, ENGINE_SRC
BASE = HERE
from board_symmetry import TRANSFORMS, transform_position, canonical_key

import raven_pipeline as baseline  # 31-plane Raven encoding, RECORD and architecture
RECORD = baseline.RECORD
CONTEXT_RECORD = np.dtype([('active_worker', 'i1'), ('movement_origin', 'i1'),
                           ('previous_build', 'i1'), ('prometheus_prebuilt', 'u1')])
PLANE_NAMES = ('active_worker', 'movement_origin', 'previous_build', 'prometheus_prebuilt')


@dataclass(frozen=True)
class Context:
    active_worker: int = -1
    movement_origin: int = -1
    previous_build: int = -1
    prometheus_prebuilt: bool = False

    def __post_init__(self):
        for name in PLANE_NAMES[:3]:
            if not -1 <= getattr(self, name) < 25:
                raise ValueError(f'{name} must be -1 (absent) or a square 0..24')
        if self.prometheus_prebuilt not in (False, True):
            raise ValueError('prometheus_prebuilt must be boolean')

    def record(self):
        return (self.active_worker, self.movement_origin, self.previous_build,
                int(self.prometheus_prebuilt))


def after_action(context, phase, action, god):
    """Advance context AFTER emitting the current input, never using future labels."""
    if phase == 0:
        if not 0 <= action < 625:
            raise ValueError('Invalid movement action')
        return replace(context, active_worker=action % 25, movement_origin=action // 25)
    if phase == 1:
        if not 625 <= action < 650:
            raise ValueError('Invalid first-build action')
        return replace(context, previous_build=action - 625)
    if phase != 2:
        raise ValueError('Invalid phase')
    if god == 9:
        if action == 675:
            return context
        if not 650 <= action < 675:
            raise ValueError('Invalid Prometheus pre-build action')
        return replace(context, previous_build=action - 650, prometheus_prebuilt=True)
    if action == 675 or (god == 3 and action == 676):
        return context
    if not 650 <= action < 675:
        raise ValueError('Invalid additional-build action')
    return replace(context, previous_build=action - 650)


def next_phase(phase, god):
    if phase == 0:
        return 1
    if phase == 1:
        return 2 if god in (3, 4, 5) else None
    if phase == 2:
        return 0 if god == 9 else None
    raise ValueError('Invalid phase')


def context_records(rows):
    """Reconstruct history from complete, chronological frozen sample records.

    Call on a complete game (or a sequence of complete games), NOT shuffled or
    train-index-filtered rows. Every action influences only subsequent inputs.
    """
    result = np.empty(len(rows), dtype=CONTEXT_RECORD)
    previous = None
    expected = None
    context = Context()
    for i, row in enumerate(rows):
        side = int(row['position'][50])
        identity = (int(row['game']), side)
        god = int(row['position'][51 if side == 48 else 52]) - 48
        phase, action = int(row['phase']), int(row['policy'])
        if identity != previous:
            if previous is not None and expected is not None:
                raise ValueError(f'Turn ended before completing phase {expected} at row {i}')
            context = Context()
            expected = 2 if god == 9 else 0
        if expected != phase:
            raise ValueError(f'Unexpected phase {phase}, expected {expected}, at row {i}')
        result[i] = context.record()
        context = after_action(context, phase, action, god)
        expected = next_phase(phase, god)
        previous = identity
    if expected is not None:
        raise ValueError('Truncated game/turn')
    return result


def context_planes(records):
    result = np.zeros((len(records), 4, 25), dtype=np.float32)
    for plane, name in enumerate(PLANE_NAMES[:3]):
        squares = records[name].astype(np.int64)
        if np.any((squares < -1) | (squares >= 25)):
            raise ValueError(f'Invalid square in {name}')
        valid = squares >= 0
        result[np.flatnonzero(valid), plane, squares[valid]] = 1
    flags = records['prometheus_prebuilt']
    if np.any(flags > 1):
        raise ValueError('Invalid pre-build flag')
    result[:, 3, :] = flags[:, None]
    return result.reshape(-1, 4, 5, 5)


def tensorize(rows, contexts, device='cpu'):
    import torch
    if len(rows) != len(contexts):
        raise ValueError('Rows/context alignment mismatch')
    original, values, policies = baseline.tensorize(rows, device)
    extra = torch.from_numpy(context_planes(contexts)).to(device)
    return torch.cat((original, extra), dim=1), values, policies


def transform_context(context, transform):
    mapping = TRANSFORMS[transform]  # destination -> source
    def square(s):
        return -1 if s == -1 else mapping.index(s)
    return Context(*(square(getattr(context, name)) for name in PLANE_NAMES[:3]),
                   context.prometheus_prebuilt)


def canonical_context_key(position, phase, context):
    """Full D4 identity INCLUDING phase and context; never discard history."""
    canonical_key(position)  # owning module's full-position validation
    return min((transform_position(position, t), phase, *transform_context(context, t).record())
               for t in range(8))


def load_model():
    """Same network, only the first convolution grows from 31 to 35 inputs."""
    import torch.nn as nn
    model = baseline.load_architecture()()
    old = model.shared_conv[0]
    model.shared_conv[0] = nn.Conv2d(35, old.out_channels, old.kernel_size,
                                    stride=old.stride, padding=old.padding, bias=old.bias is not None)
    return model
