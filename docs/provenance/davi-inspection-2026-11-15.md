# Davi inspection — November 15, 2026

Davi is recoverable as a substantial neural-engine experiment: C++ search and
rules source, Python training/export code, four weight checkpoints, deployed
TorchScript models, and a populated training archive all survive. Raven is the
most developed neural architecture found. The saved data and training pipeline
need an audit and repairs before any new training or strength conclusions.

This inspection changed no engine, model, training archive, or database.
Both deployed network architectures loaded and executed on CPU. No training,
engine rebuild, full replay validation, or new playing-strength test was run.

## Locations and provenance

- Python training material: [mediator nn directory](<C:/Users/rafae/PycharmProjects/santoriniMediator/nn>).
- Populated dataset: [data/santorini_train.npz](<C:/Users/rafae/PycharmProjects/santoriniMediator/data/santorini_train.npz>).
- Executables and deployed models: [engines/Davi](<C:/Users/rafae/PycharmProjects/santoriniMediator/engines/Davi>).
- C++ Git history: [CLion checkout](<C:/Users/rafae/CLionProjects/santorini2025>),
  branches `Phoenix`, `flux`, and `raven`. Inspection used `git show`, preserving
  the current branch and working-tree changes.
- Source snapshots: Phoenix `6d22f91`, Flux `9233f0a`, Raven
  `77a3a01db18efae98f69063ed7cf38091fcb2cd9`. Relevant files are
  `src/search.h`, `src/evaluation.h`, `src/board.h`, `src/board.cpp`, and
  `CMakeLists.txt` in those snapshots.

These are source/build-name associations, not a fresh proof that each archived
EXE was compiled from that exact commit. The Python `nn` files are present on
disk but were absent from the mediator's tracked-file listing and `git log --all -- nn`.

## Architecture

| Component | Phoenix / Flux | Raven |
| --- | --- | --- |
| Network input | 28 × 5 × 5 | 31 × 5 × 5 |
| Shared convolution stack | 3×3 convolutions: 28 → 64 → 128 → 256; BatchNorm and ReLU | 3×3 convolutions: 31 → 64 → 128 → 256; BatchNorm and ReLU |
| Value head | Flatten 6,400 → 512 → 64 → 1, with ReLU and dropout | 1×1 convolution 256 → 32, BatchNorm/ReLU, flatten 800 → 256 → 1, ReLU and dropout 0.3 |
| Policy head | None | 1×1 convolution 256 → 128, BatchNorm/ReLU, flatten 3,200 → 1,024 → 677 logits |
| Trainable parameters | **3,696,321** | **4,606,342** |
| Search | UCT MCTS over full turns, random rollouts capped at 50 plies, neural sign at cutoff | Policy/value PUCT MCTS over individual move/build actions |

The retained Phoenix commit is titled “implement batched MCTS”, but its actual
search, also present in the inspected Flux branch, performs individual random
rollouts and single-position inference at the cutoff. No multi-position neural
batching is implemented in those inspected search files. The network is not
used at every leaf in this version.

Raven evaluates one leaf at a time, converts its value logit using
`2 * sigmoid(logit) - 1`, normalizes policy probabilities over legal actions,
and uses PUCT with exploration constant 1.4. It chooses the most visited action
and follows visits through the remaining sub-actions to construct a full turn;
unexpanded continuations fall back to the first legal action. The root is rebuilt
on every call. Both search versions budget about one tenth of remaining time.
The inspected search has no self-play training feedback loop, root Dirichlet
noise, or export of visit distributions as training targets.

The input planes encode heights 1–4 (4 planes), own/opponent workers (2),
absolute side to move (1), Athena restriction (1), own god (10), and opponent
god (10). Raven adds a one-hot indicator for move / first build / second build
(3). Ownership and god planes use the current player's perspective; coordinates
are not rotated or reflected.

The 677 policy entries are 625 origin/destination pairs (`from * 25 + to`),
25 first-build squares, 25 second/pre-build squares, pass, and dome. Artemis
and Hermes movement is collapsed to origin/final destination rather than a
policy distribution over every route.

The C++ evaluator uses LibTorch and hardcodes CUDA. CPU inference succeeded
when explicitly loading the serialized models onto CPU during this inspection;
the C++ source itself does not implement that fallback.

## Models actually available

| Artifact | Finding |
| --- | --- |
| `nn/santorini_evaluator_weights.pth` | Exactly matches every deployed Phoenix/Flux state-dictionary tensor |
| `nn/santorini_evaluator_weights_30_epochs.pth` | Separate value-only checkpoint with the same layer shapes |
| `nn/santorini_evaluator_weights_30_epochs_0.5_dropout.pth` | Another separate value-only checkpoint with the same layer shapes |
| `nn/santorini_zero_weights.pth` | Exactly matches every deployed Raven state-dictionary tensor |
| `nn/santorini_evaluator.pt` | Byte-identical to the deployed Raven TorchScript file |

Phoenix and Flux carry byte-identical models (SHA-256
`ed596bc7a21ea0b12d07f5a265121eaf79fbefab3747bb02e559f63f529f9cc6`).
Raven's SHA-256 is
`0fbfb929a601cb36bfaddb9f14a1bea1dbdce565bd0d67f25e9826f7c24ab43d`.
The saved checkpoints are state dictionaries, without optimizer state, training
metrics, dataset hashes, or a run manifest. Filenames suggest training variants;
they do not independently establish the epochs or settings actually used.

The archive contains one Phoenix EXE, one Flux EXE, and eleven Raven EXEs
(2.0, 2.1, 2.2, 2.3, 2.3.1, 2.3.3 through 2.3.8). Raven versions share the
single model file currently in their directory; historical model assignments
cannot be inferred from EXE filenames alone.

## Prepared training data

The populated `data/santorini_train.npz` is **164,219,168 bytes** compressed.

| Array | Shape | Meaning |
| --- | --- | --- |
| `x` | (6,350,609, 31, 5, 5), float32 | Phase-specific input tensors |
| `y` | (6,350,609,), float32 | Current player's eventual win/loss, 0 or 1 |
| `y_policy` | (6,350,609,), int32 | Recorded chosen action, range 0–676 |
| `ids` | (6,350,609,), int32 | Source match ID |

This is **6,350,609 sample rows from 96,337 source games**, not 6.35 million
independent or symmetry-distinct board positions. No D4 uniqueness count or
symmetry-leakage audit was performed; no game boards were selected or evaluated
by this inspection. The `x` payload alone expands to 19,686,887,900 bytes
(about 18.3 GiB). The audit reads its NPY header without materializing it.

Rows comprise 2,721,468 move targets, 2,721,468 first-build targets, and
907,673 second/pre-build/pass/dome targets. There are 3,215,310 winning-side
labels and 3,135,299 losing-side labels (50.63% versus 49.37%).

Every source ID resolves in the local SQLite archive. Included IDs range from
33,905 to 154,337, with game dates April 3, 2025 through April 3, 2026. All ten
gods are represented. These are mixed historical engine games, not just Davi
self-play. Frequent participants include Lunar, Prince, Atium, Phoenix, Mystic,
Spirit, and Firefly; Raven and Flux also occur. See `artifacts.json` for full
participant/god counts. Included source games contain 2,721,952 raw turn strings,
484 more than the dataset's movement-target count; the difference has not been
individually explained or replay-audited.

Of the included games, 92,176 ended normally, 3,746 by timeout, and 415 by
illegal move. The current generator does not filter those result types, and
uses the eventual signed winner for the value label in all cases.

**The similarly named `nn/santorini_train.npz` is empty**: four empty arrays in
a 736-byte file. The training code's intended path is the populated file in
`data`, when launched from the `nn` working directory.

More source data survives beyond this prepared archive. Read-only SQLite
inspection found **189,406 games**, including **133,506 with nonempty move logs**,
through April 15, 2026. A separate read-only PostgreSQL transaction found
**209,201 games** through November 15, 2026, with binary move storage. This is
a live count, not an assertion that all those games are usable for training.
The stores overlap and must not be added together. New PostgreSQL games require
decoding and an adapter; the retained NN preparation script queries the older
SQLite-style schema.

## Retained training recipe and limitations

[network.py](<C:/Users/rafae/PycharmProjects/santoriniMediator/nn/network.py>)
currently trains Raven from scratch using Adam at learning rate 0.001,
batch size 256, 15 epochs, and equally weighted binary cross-entropy on value
plus categorical cross-entropy on the single recorded policy action. The split
is 85/15 with seed 42. This is supervised action imitation plus outcome
prediction from engine game logs; the presence of PUCT and the name `ZeroNet`
does not establish an AlphaZero-style iterative self-play training system.

Findings to address before reuse:

1. **Validation leakage is directly reproducible.** Recreating the current
   sample-level split gives 96,070 validation games, all of which also occur in
   training. The code ignores the stored match IDs. Adjacent turns and even
   phases from the same turn can cross the boundary. Use game/group splitting
   and then audit full D4-equivalent positions across splits with
   `experiments/starting-position-probe/board_symmetry.py`. Phase states also
   need their extra turn context preserved when checking identity.
2. **Raven's input omits relevant intermediate state.** C++ legality depends
   on active worker, turn origin, previous build square, and whether Prometheus
   built early. The tensor includes phase but none of these explicit fields.
   Different decision contexts can therefore receive identical observations.
   Legal-action filtering helps inference but does not supply that context to
   the network or to its supervised loss.
3. **There is a concrete Prometheus source mismatch.** Python preparation
   emits pre-build phase 2 before movement. The inspected Raven C++ constructor
   and `_end_turn()` both set phase 0 unconditionally; phase 2 is entered after
   movement only for Demeter, Hephaestus and Atlas. Thus Prometheus pre-build
   actions are unreachable through the normal inspected C++ turn transitions.
   This is source analysis, not an EXE regression test.
4. **Preparation is not runnable unchanged.** It imports the absent
   `analysis.database`. It also swallows per-move exceptions and retains any
   samples already emitted before an invalid replay is detected. There is no
   skip/error audit or per-game atomic acceptance.
5. **Apollo intermediate simulation needs repair.** `_apply_move_only` first
   moves the swapped opponent onto the mover's origin, then uses
   `workers.index(from_sq)` to locate the mover. For a Blue swap, a Gray worker
   earlier in the list can be selected instead. This is a static source finding;
   how many saved tensors it affects remains unmeasured.
6. **Memory use is large.** The training script loads all float32 features,
   copies them through the split, and constructs additional torch tensors.
   Preparation accumulates individual tensors in Python lists before packing
   them. Streaming/sharded storage would make rebuilding and training practical.
7. **No training-quality claim is recoverable yet.** No saved loss curves,
   clean held-out scores, optimizer state, or immutable data/run linkage were
   found in the inspected NN directory. Current source settings need not be
   the exact settings that produced the deployed checkpoint. The older
   value-only training class/recipe was not found in current Python source,
   although its architecture is recoverable from TorchScript and weights.

## Practical assessment

There is enough here to restart from real artifacts rather than recreate the
experiment. Keep the existing checkpoints as baselines. First repair and audit
the phase/rules representation and replay conversion, then regenerate a
versioned dataset with explicit result filters, game IDs, D4 identity, and a
separate holdout. Preserve the raw games and old model/data files. Only then
train or compare strength; this inspection does not establish whether the
architecture or saved models are competitive with current Paladini builds.

## Reproduction

Run `python experiments/davi-inspection-2026-11-15/audit.py` from this project
with NumPy and PyTorch available. It writes [artifacts.json](artifacts.json),
reads model weights and dataset metadata/labels, reproduces match-ID overlap,
and queries SQLite with `mode=ro`. It does not connect to PostgreSQL; the live
PostgreSQL count above was obtained separately with a transaction verified as
read-only. It performs tensor-shape smoke checks on zero-filled inputs, not
evaluations of game positions. The feature payload and full replays were not
exhaustively validated.
