# Davi 2.3.9 Raven — overnight retraining

Authorized November 16, 2026: prepare the match logs and train the existing Raven
method overnight. The user's naming decision reserves a substantive change in
training method for **Davi 3**. This run is **Davi 2.3.9 Raven**.

## Training recipe

- Unchanged `SantoriniZeroNet`: 31 input planes, 677 policy logits, one value logit;
  4,606,342 trainable parameters. Original architecture source is frozen in `source/`.
- Train from scratch, as in the retained Raven training script; 15 epochs,
  batch size 256, Adam learning rate 0.001, float32.
- Binary cross-entropy for eventual winner, cross-entropy for the recorded
  action, equal loss weights and uniform sample weighting.
- No Elo weighting, deep-search labels, new policy objective, new input planes,
  symmetry augmentation, or engine search/rules change in this version.
- Whole games grouped by full D4 starting-position identity for the holdout.
  Further remove every training sample whose full D4 board identity occurs in
  validation, conservatively ignoring phase when checking overlap.

The validation protocol repairs the old sample-level leakage. Its loss/accuracy
must not be compared directly with the historical contaminated validation split.

## Frozen raw data and conversion

`raw/matches.db` is a consistent backup obtained through a read-only SQLite
connection. `raw/postgres.jsonl` is a repeatable-read, read-only PostgreSQL export
of exact text replays and rows newer than the SQLite snapshot. No original
database is updated. `source-manifest.json` records snapshot hashes.

Every full replay is checked for legal moves, no moves after the game ends, and
agreement with the recorded winner for normal board results. A rejected game
contributes **zero** training samples; rejection reasons and source IDs are
written to `rejected-games.jsonl`. All historical raw material is retained.

Losslessly available text is preferred. Some older PostgreSQL binary records
omitted Artemis/Hermes routes or Prometheus pre-build information; unsupported
binary-only games are quarantined rather than reconstructed speculatively.
Exact duplicate engine/start/replay sequences across stores are audited and
included once. Original outcome labels for normal wins, timeouts and forfeits
are retained when their complete recorded moves pass validation.

The original phase/action layout is preserved, with the Blue Apollo swap
conversion error corrected by capturing the mover index before displacing the
opponent. Tests compare vectorized inputs bit-for-bit with the original tensor
encoder across all ten gods, both sides, three phases and eight symmetries.

Features are stored as compact original-orientation positions plus phase and
targets in `samples.bin`, then expanded to the original float32 planes per GPU
batch. This avoids loading/copying tens of GB of dense float32 training features.
Full D4 identity is computed using the project's `board_symmetry.py` before
export. `split-audit.json` reports physical and symmetry-distinct board counts
separately. Repeated samples from different games are retained as in the original
learning method; counts of examples are not counts of distinct boards.

## Logs, checkpoints and completion

- `run-manifest.json`: exact recipe, Python, PID and source hashes.
- `status.json`: most recent progress, failure, stop or completion event.
- `events.jsonl`: append-only progress history, including preparation counts.
- `stdout.log` / `stderr.log`: background process output and errors.
- `accepted-games.jsonl`: game IDs, engines, original starts, D4 identity and row ranges.
- `dataset.json` / `split-audit.json`: accepted/rejected data and leakage audit.
- `metrics.jsonl`: per-epoch training/validation losses and accuracies.
- `checkpoints/epoch-XX-weights.pth`: all epoch weights.
- `checkpoints/last.pth`: model, optimizer and RNG state for resuming at the next epoch.
- `checkpoints/best-weights.pth`: lowest combined validation-loss checkpoint.
- `artifacts/Davi_2.3.9_Raven/`: final weights, CPU-portable TorchScript export,
  copied executable and manifest, written only after training finishes and the
  export/reload comparison passes.

The candidate executable is byte-identical to archived Raven 2.3.8, renamed for
the new model; it loads the separately packaged new `santorini_evaluator.pt`.
The existing Raven LibTorch/CUDA runtime is still required. Historical engine
folders and models are not overwritten. No default-engine promotion or
playing-strength claim is made by training completion.

The retained Raven intermediate-context input limitations and engine rules/search
behavior remain. Addressing those is outside this same-method retraining run.

## Operation

Run `launch.ps1` to start the detached worker with a hidden window. The worker
temporarily prevents automatic system sleep while active, restoring normal
behavior on exit. It does not keep the display on or change persistent settings.

Create a file named `STOP` in this directory for a graceful stop. The worker
checks it during preparation and training. A resume uses the latest completed
epoch; unfinished preparation is regenerated from the frozen snapshots. Remove
`STOP` before resuming. `running.lock` prevents concurrent launches; a lock left
by a killed process must be checked against its recorded PID before removal.

Preflight commands:

```powershell
python experiments/raven-2.3.9-overnight-2026-11-16/test_pipeline.py
python experiments/raven-2.3.9-overnight-2026-11-16/run.py --smoke
```

The five pipeline tests passed. A separate short GPU training run passed gradient
updates, validation, checkpoint writing, TorchScript export and reload comparison.
Smoke artifacts have `smoke-` prefixes and cannot replace full-run checkpoints.
