# Davi — MCTS/neural Santorini engine (Joker line)

Davi is the policy/value-network + batched-MCTS family that branched from Paladini
Firefly 8.1.8 (Phoenix → Flux → Raven). **Davi 3.0.0 Joker** is the first Davi 3
release: Raven's network and search, with four extra input planes describing the
turn in progress.

The C++ history up to Raven lives in the engine repo
[`santoriniCpp2025`](https://github.com/PaMeirelles/santoriniCpp2025) (branches
`Phoenix`, `flux`, `raven`, `joker`; tags `davi-2.3.8-raven`, `davi-3.0.0-joker`).
This repo is the self-contained home for Joker: engine source, training pipeline,
released weights and the scripts to install it for play.

```
engine/      C++ engine source (35-plane runtime), CMakeLists.txt, build.cmd
training/    corpus builder, context planes, network, training, export, holdout eval
weights/     released networks (plain git; ~37 MB per release) — joker-3.0.0/
play/        install into santoriniMediator, LibTorch/CUDA diagnose tool
docs/        provenance: original ChatGPT experiment READMEs, manifests, scripts
```

## Joker 3.0.0 at a glance

| | |
|---|---|
| Input | 35 planes: Raven's 31 + active worker, movement origin, last build, Prometheus pre-built |
| Network | `SantoriniZeroNet`, 677 policy logits + value, 4,608,646 parameters |
| Corpus | Exactly Raven 2.3.9's: 150,282 games, 7,662,868 train / 1,460,783 validation samples (incl. timeouts/forfeits) |
| Recipe | 15 epochs from scratch, Adam 1e-3, batch 256, seed 42, float32, BCE value + CE policy, equal weights |
| Release | **Epoch 3** checkpoint (user decision, Nov 17 2026), not the last epoch |
| Validation | value loss 0.5121, policy loss 1.0753, value acc 73.68%, policy acc 60.70% |
| Weights | TorchScript `ec4994cf…0976`; state dict `fba4012b…ca24` |

Known limitation inherited from Raven's source: the Prometheus pre-build phase is
unreachable in the engine's state machine.

## Build the engine

Requires MSVC 2022 x64 and a CUDA LibTorch (default: `santorini2025/extern/libtorch`;
override with `LIBTORCH`).

```powershell
engine\build.cmd
```

Produces `engine\build\Davi_3.0.0_Joker.exe` with `santorini_evaluator.pt` beside it.
CMake works too (`-DLIBTORCH_DIR=...`). The forced `warp_size` link symbol is required
on Windows or the CUDA backend is dropped.

## Play

```powershell
python play\install.py
```

Copies the exe + weights to `santoriniMediator\engines\Davi\Joker` (refuses to
overwrite). Matches, SPRT and ratings are run from the mediator.

## Train

Data never goes in git. Paths come from `training/paths.py` (env overrides):
`SANTORINI_MEDIATOR`, `DAVI_DATA` (default `./data`), `DAVI_CORPUS`, `DAVI_JOKER_RUN`.

```powershell
cd training
python -m unittest discover -s tests -v        # rules, encodings, C++ parity
python raven_pipeline.py --prepare-only        # snapshot DBs read-only -> corpus + D4 split
python prepare_context.py                      # 4-plane context sidecar (context.bin)
python train_joker.py --smoke                  # quick GPU check
powershell -File launch.ps1                    # full background training; create STOP to stop
python compare_holdout.py --model "Joker 3.0.0=..\weights\joker-3.0.0\santorini_zero_weights.pth:35"
python export_release.py --epoch N --name joker-3.0.1
```

`prepare_context.py` checks the corpus against Joker 3.0.0's frozen hashes; set
`DAVI_ALLOW_NEW_CORPUS=1` to train on a new corpus (that is then a new model).
`raven_pipeline.py` without `--prepare-only` also trains the 31-plane Raven baseline.

### Verified when this repo was assembled (Nov 21 2026)

- 15/15 unit tests pass, including C++ ↔ Python context-plane parity on `engine/src`.
- Pointed at the frozen Raven 2.3.9 corpus, `prepare_context.py` reproduces
  `context.bin` bit-for-bit (`96fc6fb8…f226`).
- `train_joker.py --smoke` passes (checkpoint + TorchScript reload).
- `compare_holdout.py` on the released weights reproduces the recorded epoch-3
  validation metrics exactly.
- `engine\build.cmd` rebuilds an executable of identical size to the released one
  (8 bytes differ, i.e. PE timestamp/build id).

## Versioning

Davi 2.x = Raven method (31 planes). Davi 3 = substantive method change (Joker:
context planes). Tag releases `davi-<version>-<name>`, keep each release's
`weights/<name>/manifest.json`, and record executable and model hashes together.
Cypher 3.0.0 (31 planes, normal-result-only corpus) is a sibling experiment and is
not part of this repo.
