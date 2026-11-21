# Four-context-plane training on the Raven 2.3.9 corpus

Authorized November 17, 2026: train the four prepared additional inputs on the
same games as Raven 2.3.9. This run is an experimental model, with no new engine
name/version assigned and no automatic engine promotion.

The source is the full frozen `../raven-2.3.9-overnight-2026-11-16/` corpus:
150,282 accepted games (143,792 normal, 4,872 timeout, 1,618 illegal-move results),
9,539,650 sample rows, 7,662,868 training samples and 1,460,783 validation samples.
The 415,999 rows previously purged for D4 overlap remain unused. No games are
added, no rejected games restored, and no new filtering, labels or weights applied.
The exact original train/validation index arrays are reused. Context is derived
from past actions for every source row before split selection.

This is the correct corpus-matched input comparison against **Raven 2.3.9**.
Comparing with **Cypher 3.0** would change both input representation and corpus,
because Cypher excluded all timeout/forfeit games.

The recipe is unchanged: 15 epochs from scratch, float32 CUDA, seed 42,
Adam learning rate 0.001, batch size 256, uniform sample weighting, equal
binary value cross-entropy and recorded-action cross-entropy. The architecture
adds only four input channels, for 35 total and 4,608,646 trainable parameters.
The existing three phase planes, 677-action head and value head are unchanged.

`setup.py` freezes the prepared encoding and adapts the original training loop.
`source/prepare.py` writes the aligned context sidecar, checks frozen source
hashes against the prior archival manifest, and verifies every starting D4 key.
The inherited board-only D4 overlap purge remains conservative for enriched
contexts; sample counts are not symmetry-distinct position counts.

Run `python run.py --smoke` for a short separate GPU smoke test, then
`powershell -File launch.ps1` for hidden background full training. `STOP` requests
graceful stopping at the next checkpoint check. The lock prevents two workers
in this run directory. Inspect its PID before removing a stale lock. Full resumes
use the last completed epoch's model, optimizer and torch/CUDA RNG state.
Shuffling is reproducible from epoch number. Smoke checkpoints are separate.

`status.json`, `events.jsonl`, `metrics.jsonl`, `run-manifest.json`, `stdout.log`
and `stderr.log` record the full run. All epoch weights and resumable state are
kept. The lowest combined-validation-loss checkpoint is exported as TorchScript
and state-dictionary weights under `artifacts/context-35/`. CPU reload/output
comparison must pass before completion. No old executable is packaged: a
35-plane model requires a compatible rebuilt runtime. The known Prometheus
phase/state-machine limitations documented in
`../cypher-context-35-2026-11-17/README.md` remain outside this training change.
