# Davi 3.0.0 Joker

User decision November 17, 2026: stop the context35 training and release its
**epoch 3** checkpoint as Joker 3.0.0; compare against Raven 2.3.9 on two workers.
Training stopped gracefully in epoch 10; all completed epoch checkpoints remain.
The selected weights are the explicit epoch-03 file, not the last training state.

Joker uses 35 input planes: the original Raven 31 plus active-worker square,
movement-origin square, most recent build square, and Prometheus pre-build flag.
It keeps Raven's three phase planes, policy/value heads and original recipe.
The training corpus/split is exactly Raven 2.3.9's: 150,282 accepted games,
7,662,868 training examples and 1,460,783 validation examples, including the same
timeouts/forfeits. This differs from Cypher's normal-result-only corpus.

`manifest.json` records exact checkpoint, TorchScript, executable, source and build
identities. Epoch-3 state-dictionary SHA-256:
`fba4012bac27ca99e94919aee756837ee27cbfbc5580edb28279dbf8bb6eca24`.
TorchScript was exported in eval mode and reloaded on 32 actual held-out rows;
policy/value output equality passed. The model has 4,608,646 parameters.

The isolated runtime source comes from the context35 adapter of the historical
Raven source association `77a3a01`. It adds the context getters and encoding;
existing search, phase transitions and rule decisions are retained. Exact identity
between that historical source and the archived Raven EXE is not established.
The inherited source's unreachable Prometheus pre-build phase remains a known
limitation; this release does not silently change its state machine.

Build with `build.cmd` from this directory using MSVC x64 and the existing LibTorch
CUDA installation. The forced `warp_size` CUDA symbol is required by the Windows
LibTorch link setup. The first build omitted it and failed before search; that
EXE, its manifest/log and failed match-preparation manifest are preserved under
`failed-build-cuda-link/`. No match was dispatched with that build. The corrected
build passed an initial search/protocol check before all-god match preflight.

The canonical artifact is `artifacts/Davi_3.0.0_Joker/`; an additive installation
is at `C:/Users/rafae/PycharmProjects/santoriniMediator/engines/Davi/Joker/`.
No Raven/Cypher files were replaced. LibTorch/CUDA DLLs remain in the existing
shared runtime directory and are fingerprinted by the comparison manifest.

Match run: `../joker-vs-raven-200-2026-11-17/`.
