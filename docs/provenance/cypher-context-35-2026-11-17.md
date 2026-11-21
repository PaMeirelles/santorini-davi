# Cypher: four additional context planes

Prepared separately on November 17, 2026. This is input preparation, not a named
engine release or a training run. Cypher 3.0 files and weights remain preserved.
The original three phase planes and 677 policy actions are retained.

| Plane | Meaning | Encoding |
|---|---|---|
| 31 | Active worker | One square at its current location; zero plane when absent |
| 32 | Movement origin | One square at the start of the movement; zero when absent |
| 33 | Previous build | Most recently executed build square in this turn; zero before building |
| 34 | Prometheus pre-built | Constant 0/1 plane; true after an actual pre-build, false after passing |

Planes 0-30 are bit-identical to the baseline for the same row. Plane 33 includes
an actual Prometheus pre-build: it remains available during movement and is
replaced by the first post-move build when that occurs. Movement origin is
absent until movement happens. Context resets when the player or game changes.
Worker identity is a location, not the index/order in the workers array.

## Dataset and reproducibility

`prepare.py` reconstructs a four-byte context record for **every** frozen sample
row, before applying any train/validation selection. It processes complete games
in chronological phase order. Each input includes only actions that have already
happened; the current target updates the context for the next input. Truncated
or misordered streams are rejected rather than assigned zero/default context.

`context.bin` is an aligned, read-only-to-consumers sidecar for
`../davi-normal-results-2026-11-17/samples.bin`; it is not standalone training data.
Its fields are three signed int8 squares (`-1` absent, `0..24` present) and one
uint8 boolean. The existing sample data, action/value labels, repetitions,
original orientations, and exact saved split indices are reused without changes.

The frozen corpus has 143,792 normal-result games and 9,078,194 sample records:
7,299,967 training, 1,386,454 validation and 391,773 inherited unused records.
These are sample counts, not symmetry-distinct position counts. The inherited
full D4 board-overlap purge is conservative for richer contexts: if there is no
board overlap, there cannot be overlap of the same board plus context. Original
board identities and splits are retained; no new distinct-context count is claimed.

`context.canonical_context_key()` uses the project's full-position symmetry
module and transforms the three context squares along with the board under all
eight symmetries. Phase and the pre-build flag are part of this enriched identity.
Use it for any future context-aware selection or deduplication; board-only keys
do not distinguish different intermediate contexts.

`dataset.json` records source fingerprints, sidecar checksum, counts, split and
schema. Preparation verifies the base data against its frozen output manifest
and rechecks every game's starting-position D4 key. It refuses to overwrite
completed sidecar data. No live database is queried.

```powershell
python experiments/cypher-context-35-2026-11-17/prepare_runtime.py
python experiments/cypher-context-35-2026-11-17/test_context.py
python experiments/cypher-context-35-2026-11-17/prepare.py
```

The first and third commands are one-time preparation commands. To consume:

```python
from dataset import PreparedDataset
from context import load_model

data = PreparedDataset()
x, value_targets, action_targets = data.batch(data.train_indices[:256], 'cpu')
model = load_model()  # Fresh, untrained 35-input-channel network
policy_logits, value_logits = model(x)
```

`load_model()` reuses the frozen Cypher architecture and changes only the first
convolution's input channels from 31 to 35: **4,608,646** trainable parameters,
an increase of 2,304. This is a model factory and data adapter, not a training
launcher. No new loss, phase separation, batching, or automatic promotion is added.
Do not load the old checkpoint directly into this changed input shape.

## Runtime preparation and limits

`prepare_runtime.py` exports a separate Raven source snapshot at the recorded
historical source association `77a3a01db18efae98f69063ed7cf38091fcb2cd9`. It exposes
the four trackers, records an executed Prometheus pre-build square in the
previous-build tracker, and appends the four planes to the evaluator tensor.
`runtime-src/manifest.json` records original and prepared file hashes. The shared
`context_planes.h` encoder has executable Python/C++ parity tests. Board trackers
are also compiled and compared against dataset context on ordinary movement,
Demeter, Hephaestus and Atlas sequences.

**This is not a compiled/playable engine.** The archived EXE is not compatible
with a 35-plane model. Exact archived-EXE-to-source provenance is not established.
Before any release, build the adapted runtime with its LibTorch dependencies and
run an end-to-end inference/replay audit using trained 35-plane weights.

**Prometheus remains an existing state-machine limitation.** The frozen Python
training generator emits a pre-build decision (phase 2) before movement. At that
point no worker has been selected by an earlier action, so planes 31 and 32 are
zero; after pre-building, they remain zero until the move. We deliberately do not
fill them from the future recorded move. This leaves the association between a
pre-build and its worker unresolved by these four planes alone. The associated
Raven C++ source starts turns in phase 0 and does not enter the pre-build phase
through its normal phase transitions. The source adapter preserves those phase
transitions; it does not claim to repair or validate Prometheus pre-build search.

Four planes also do not constitute a proof of full Markov-state completeness:
for example, the stored movement-height delta used by Athena is not added as a
fifth flag. Only the four user-requested inputs are prepared in this experiment.

## Checks

Tests cover causal context reconstruction; active worker/origin; build history;
Prometheus pre-build and pass; game/turn reset; rejection of incomplete streams;
unchanged original features/labels; eight-way D4 transformations; native encoder
parity; actual C++ board tracker parity; parameter count; gradients through new
channels; and TorchScript reload/output equality on synthetic tensors. These
checks do not measure playing strength or start substantive training.
