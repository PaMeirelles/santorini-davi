"""Replay-validated, memory-bounded Raven retraining. Original stores are read-only."""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import sys
import time
import traceback

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from paths import CORPUS as HERE, MEDIATOR as MED
HERE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(MED))
from board_symmetry import canonical_key, POLICY
from game.board import Board, _calculate_push_square
from database.models import God, get_move_from_string
from game.move import AtlasMove, DemeterMove, HephaestusMove, PrometheusMove, square_to_text
import game.move as move_module

RECORD = np.dtype([("position", "u1", (54,)), ("key", "S54"),
                   ("phase", "u1"), ("policy", "<u2"), ("value", "u1"),
                   ("game", "<u4"), ("validation", "u1")])


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save_json(name, value):
    target = HERE / name
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(target)


def event(kind, **data):
    row = {"at": now(), "event": kind, **data}
    with (HERE / "events.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(row, default=str) + "\n")
    save_json("status.json", row)
    print(json.dumps(row, default=str), flush=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stop_requested():
    if (HERE / "STOP").exists():
        raise InterruptedError("STOP file requested a graceful stop")


def snapshot():
    raw = HERE / "raw"
    raw.mkdir(exist_ok=True)
    sqlite_target = raw / "matches.db"
    if not sqlite_target.exists():
        # SQLite backup API gives a consistent snapshot without mutating the source.
        with sqlite3.connect((MED / "data/matches.db").as_uri() + "?mode=ro", uri=True) as source:
            with sqlite3.connect(sqlite_target) as destination:
                source.backup(destination)
    pg_target = raw / "postgres.jsonl"
    if not pg_target.exists():
        from dotenv import dotenv_values
        from sqlalchemy import create_engine, text
        engine = create_engine(dotenv_values(MED / ".env")["DATABASE_URL"], connect_args={
            "connect_timeout": 4,
            "options": "-c default_transaction_read_only=on -c statement_timeout=30000",
        })
        with sqlite3.connect(sqlite_target.as_uri() + "?mode=ro", uri=True) as c:
            cutoff = c.execute("SELECT max(Date) FROM TB_MATCHES").fetchone()[0]
        sql = """SELECT m.match_id, m.winner_side, m.result_type, m.timestamp,
          m.starting_pos::text AS packed, m.moves AS binary,
          r.starting_position AS position, r.moves AS text_moves,
          eg.engine_name AS gray_engine, eb.engine_name AS blue_engine,
          gg.god_name AS gray_god, gb.god_name AS blue_god
          FROM tb_matches m
          JOIN tb_match_participants g ON g.match_id=m.match_id AND g.side=TRUE
          JOIN tb_match_participants b ON b.match_id=m.match_id AND b.side=FALSE
          JOIN tb_engines eg ON eg.engine_id=g.engine_id
          JOIN tb_engines eb ON eb.engine_id=b.engine_id
          JOIN tb_gods gg ON gg.god_id=g.god_id
          JOIN tb_gods gb ON gb.god_id=b.god_id
          LEFT JOIN tb_match_replays r ON r.match_id=m.match_id
          WHERE cardinality(r.moves)>0 OR m.timestamp > CAST(:cutoff AS timestamptz)
          ORDER BY m.match_id"""
        try:
            with engine.connect() as c:
                c.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
                assert c.execute(text("SHOW transaction_read_only")).scalar_one() == "on"
                with pg_target.with_suffix(".tmp").open("w", encoding="utf-8") as out:
                    count = 0
                    for item in c.execute(text(sql), {"cutoff": cutoff + "-03:00"}):
                        row = dict(item._mapping)
                        row["binary"] = bytes(row["binary"]).hex() if row["binary"] is not None else None
                        out.write(json.dumps(row, default=str) + "\n")
                        count += 1
            pg_target.with_suffix(".tmp").replace(pg_target)
            event("postgres_snapshot", games=count, cutoff=cutoff)
        finally:
            engine.dispose()
    sources = {str(p.relative_to(HERE)): {"bytes": p.stat().st_size, "sha256": sha(p)}
               for p in [sqlite_target, pg_target]}
    save_json("source-manifest.json", {"at": now(), "sources": sources,
        "original_databases": "Read-only; full SQLite snapshot plus PostgreSQL exact replays and newer rows."})


def decode_binary(row):
    """Decode only fields actually preserved by the historical 3-byte format.

    Never invent missing Artemis/Hermes routes or Prometheus pre-builds.
    Missing/invalid fields cause the entire game to be quarantined.
    """
    packed = row["packed"]
    if packed is None or len(packed) != 20:
        raise ValueError("No complete starting position or decodable flat legacy start")
    workers = [int(packed[i:i+5], 2) for i in range(0, 20, 5)]
    if len(set(workers)) != 4:
        raise ValueError("Duplicate starting workers")
    gods = [God[row["gray_god"].upper()], God[row["blue_god"].upper()]]
    position = "".join("0" + ("G" if i in workers[:2] else "B" if i in workers[2:] else "N")
                       for i in range(25)) + "0" + str(gods[0].value) + str(gods[1].value) + "0"
    canonical_key(position)
    raw = bytes.fromhex(row["binary"] or "")
    if not raw or len(raw) % 3:
        raise ValueError("Empty or truncated binary moves")
    moves = []
    for index in range(0, len(raw), 3):
        v = int.from_bytes(raw[index:index+3], "big")
        flag, origin, to, build, extra = (v >> 20) & 1, (v >> 15) & 31, (v >> 10) & 31, (v >> 5) & 31, v & 31
        god = gods[(index // 3) % 2]
        if max(origin, to, build) > 24:
            raise ValueError("Historical binary encoding omitted required move fields")
        if god in (God.ARTEMIS, God.HERMES, God.PROMETHEUS):
            raise ValueError("Historical binary route/pre-build information is not lossless")
        text = square_to_text(origin) + square_to_text(to) + square_to_text(build)
        if god in (God.DEMETER, God.HEPHAESTUS) and extra != 31:
            if extra > 24:
                raise ValueError("Invalid second build")
            text += square_to_text(extra)
        if god == God.ATLAS and flag:
            text += "D"
        moves.append(text)
    return position, moves


def games():
    with sqlite3.connect((HERE / "raw/matches.db").as_uri() + "?mode=ro", uri=True) as c:
        for ident, pos, moves, result, date, eg, eb in c.execute(
            "SELECT Id,Starting_pos,Moves,Result,Date,Engine_G,Engine_B FROM TB_MATCHES "
            "WHERE Moves IS NOT NULL AND length(trim(Moves))>0 ORDER BY Id"
        ):
            yield {"source": "sqlite", "id": ident, "position": pos,
                   "moves": str(moves).split(), "winner": 1 if result > 0 else -1,
                   "result_type": {1: "N", 2: "T", 3: "I"}.get(abs(result), "?"),
                   "date": date, "engines": [eg, eb]}
    with (HERE / "raw/postgres.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            try:
                if row["position"] and row["text_moves"]:
                    pos, moves = row["position"], row["text_moves"]
                else:
                    pos, moves = decode_binary(row)
                yield {"source": "postgres", "id": row["match_id"], "position": pos,
                       "moves": moves, "winner": 1 if row["winner_side"] else -1,
                       "result_type": row["result_type"], "date": row["timestamp"],
                       "engines": [row["gray_engine"], row["blue_engine"]]}
            except Exception as exc:
                yield {"source": "postgres", "id": row["match_id"], "decode_error": str(exc)}


def phase_samples(board, move):
    """Original phase/action layout, with correct ownership during Apollo swaps."""
    god = board.gods[0 if board.turn == 1 else 1]
    temp = board.copy()
    action = move.from_sq * 25 + move.final_sq
    first = move.build_sq_1 if isinstance(move, (DemeterMove, HephaestusMove)) else move.build_sq
    second = 25
    if isinstance(move, (DemeterMove, HephaestusMove)) and move.build_sq_2 is not None:
        second = move.build_sq_2
    elif isinstance(move, AtlasMove) and move.dome:
        second = 26
    elif isinstance(move, PrometheusMove) and move.optional_build is not None:
        second = move.optional_build
    if god == God.PROMETHEUS:
        yield temp.position_to_text(), 2, 650 + second
        if second != 25:
            temp.blocks[second] += 1
    yield temp.position_to_text(), 0, action
    # Capture mover identity BEFORE relocating the displaced opponent.
    mover = temp.workers.index(move.from_sq)
    occupant = temp._which_worker_is_here(move.final_sq)
    if god == God.APOLLO and occupant is not None:
        temp.workers[occupant] = move.from_sq
    elif god == God.MINOTAUR and occupant is not None:
        temp.workers[occupant] = _calculate_push_square(move.from_sq, move.final_sq)
    temp.workers[mover] = move.final_sq
    yield temp.position_to_text(), 1, 625 + first
    if god in (God.DEMETER, God.HEPHAESTUS, God.ATLAS):
        temp.blocks[first] += 1
        yield temp.position_to_text(), 2, 650 + second


def prepare(limit=None):
    from collections import Counter
    target = HERE / ("smoke-samples.bin" if limit else "samples.bin")
    manifest_name = "smoke-dataset.json" if limit else "dataset.json"
    if target.exists() and (HERE / manifest_name).exists():
        return target, json.loads((HERE / manifest_name).read_text())
    errors = HERE / ("smoke-rejected.jsonl" if limit else "rejected-games.jsonl")
    accepted = HERE / ("smoke-games.jsonl" if limit else "accepted-games.jsonl")
    # Prevent invalid parser diagnostics from writing into the original mediator tree.
    move_module.log_invalid_move = lambda text: None
    counts = Counter()
    reasons = Counter()
    seen_games = {}
    started = time.monotonic()
    with target.with_suffix(".tmp").open("wb") as out, errors.open("w", encoding="utf-8") as rejected, accepted.open("w", encoding="utf-8") as accepted_out:
        for game_number, game in enumerate(games(), 1):
            if limit and game_number > limit:
                break
            if game_number % 1000 == 0:
                stop_requested()
                event("preparing", scanned=game_number, accepted=counts["accepted_games"],
                      rejected=counts["rejected_games"], samples=counts["samples"],
                      seconds=round(time.monotonic() - started, 1))
            try:
                if "decode_error" in game:
                    raise ValueError(game["decode_error"])
                start_key = canonical_key(game["position"])
                identity = hashlib.sha256((game["position"] + "|" + " ".join(game["moves"]) + "|" +
                                           "|".join(game["engines"])).encode()).hexdigest()
                if identity in seen_games:
                    if seen_games[identity] != (game["winner"], game["result_type"]):
                        raise ValueError("Conflicting outcomes for identical engine/start/replay")
                    counts["duplicate_source_games"] += 1
                    continue
                board = Board(game["position"])
                samples = []
                for ply, text in enumerate(game["moves"]):
                    if board.check_state() != 0:
                        raise ValueError(f"Move after terminal state at ply {ply}")
                    god = board.gods[0 if board.turn == 1 else 1]
                    move = get_move_from_string(text, god)
                    if not board.move_is_valid(move):
                        raise ValueError(f"Illegal replay move at ply {ply}: {text}")
                    winner_target = int(board.turn == game["winner"])
                    for pos, phase, policy in phase_samples(board, move):
                        key = canonical_key(pos)  # Full D4 check BEFORE exporting any state.
                        assert 0 <= policy < 677
                        samples.append((pos, key, phase, policy, winner_target))
                    with contextlib.redirect_stdout(io.StringIO()):
                        board.make_move(move)
                terminal = board.check_state()
                if game["result_type"] == "N" and terminal != game["winner"]:
                    raise ValueError(f"Normal-result mismatch: terminal={terminal}, winner={game['winner']}")
                if game["result_type"] not in ("N", "T", "I"):
                    raise ValueError("Unknown result type")
                if terminal and terminal != game["winner"]:
                    raise ValueError("Recorded winner conflicts with terminal board")
                # Group entire games by full D4 starting identity, never by sample rows.
                validation = int(int.from_bytes(hashlib.sha256(("raven239:42:" + start_key).encode()).digest()[:8], "big") % 10000 < 1500)
                records = np.empty(len(samples), dtype=RECORD)
                for i, (pos, key, phase, policy, value) in enumerate(samples):
                    records[i] = (np.frombuffer(pos.encode(), dtype=np.uint8), key.encode(), phase,
                                  policy, value, game_number, validation)
                out.write(records.tobytes())
                seen_games[identity] = (game["winner"], game["result_type"])
                counts["accepted_games"] += 1
                counts["samples"] += len(samples)
                counts["turns"] += len(game["moves"])
                counts["accepted_" + game["source"]] += 1
                counts["result_" + game["result_type"]] += 1
                for phase in (0, 1, 2):
                    counts[f"phase_{phase}"] += sum(x[2] == phase for x in samples)
                accepted_out.write(json.dumps({k: v for k, v in game.items() if k != "moves"} | {
                    "row_start": counts["samples"] - len(samples), "row_count": len(samples),
                    "starting_symmetry_key": start_key, "validation": bool(validation),
                    "replay_sha256": identity}) + "\n")
            except Exception as exc:
                counts["rejected_games"] += 1
                reason = str(exc)
                reasons[reason.split(" at ply ")[0]] += 1
                rejected.write(json.dumps({"source": game["source"], "id": game["id"], "reason": reason}) + "\n")
    target.with_suffix(".tmp").replace(target)
    result = {"at": now(), "record_bytes": RECORD.itemsize, "counts": dict(counts),
              "rejection_reasons": dict(reasons), "seconds": time.monotonic() - started,
              "symmetry_policy": POLICY,
              "split_policy": "Whole games grouped by full D4 starting identity; validation keys purged from training below.",
              "outcome_policy": "Original binary winner labels retained for all fully legal normal/timeout/forfeit replays; entire invalid games quarantined.",
              "feature_policy": "Original 31 planes, original coordinates; no augmentation, Elo weighting, or search relabelling."}
    save_json(manifest_name, result)
    event("preparation_complete", **dict(counts))
    return target, result


def split_data(path, manifest, smoke=False):
    data = np.memmap(path, dtype=RECORD, mode="r")
    prefix = "smoke-" if smoke else ""
    train_path, val_path = HERE / (prefix + "train-indices.npy"), HERE / (prefix + "validation-indices.npy")
    if train_path.exists() and val_path.exists():
        return data, np.load(train_path), np.load(val_path)
    validation = np.flatnonzero(data["validation"] == 1)
    train = np.flatnonzero(data["validation"] == 0)
    if not len(validation) or not len(train):
        raise ValueError("No train/validation split; increase smoke limit")
    keys = np.unique(data["key"][validation])
    keep = np.ones(len(train), dtype=bool)
    for begin in range(0, len(train), 100000):
        stop_requested()
        selected = data["key"][train[begin:begin+100000]]
        where = np.searchsorted(keys, selected)
        in_range = where < len(keys)
        overlap = np.zeros(len(selected), dtype=bool)
        overlap[in_range] = keys[where[in_range]] == selected[in_range]
        keep[begin:begin+100000] = ~overlap
    removed = int((~keep).sum())
    train = train[keep]
    np.save(train_path, train)
    np.save(val_path, validation)
    audit = {"at": now(), "train_samples": len(train), "validation_samples": len(validation),
             "train_rows_purged_for_d4_overlap": removed,
             "validation_distinct_full_board_classes": len(keys),
             "train_distinct_full_board_classes": len(np.unique(data["key"][train])),
             "physical_full_positions": len(np.unique(data["position"].copy().view("S54").reshape(-1))),
             "distinct_full_board_classes": len(np.unique(data["key"])),
             "remaining_full_d4_overlap": 0,
             "note": "Board classes ignore phase for conservative leakage removal; sample rows retain phase and all historical repetitions."}
    save_json(prefix + "split-audit.json", audit)
    event("split_complete", **audit)
    return data, train, validation


def tensorize(rows, device):
    import torch
    import torch.nn.functional as F
    raw = torch.from_numpy(rows["position"].copy()).to(device).long()
    heights = raw[:, :50:2] - 48
    occupants = raw[:, 1:50:2]
    gray = raw[:, 50] == 48
    x = torch.zeros((len(rows), 31, 25), dtype=torch.float32, device=device)
    x[:, :4] = F.one_hot(heights, 5)[:, :, 1:].transpose(1, 2)
    x[:, 4] = occupants == torch.where(gray, 71, 66)[:, None]
    x[:, 5] = occupants == torch.where(gray, 66, 71)[:, None]
    x[:, 6] = gray[:, None]
    x[:, 7] = (raw[:, 53] == 49)[:, None]
    own = torch.where(gray, raw[:, 51], raw[:, 52]) - 48
    opponent = torch.where(gray, raw[:, 52], raw[:, 51]) - 48
    x[:, 8:18] = F.one_hot(own, 10)[:, :, None]
    x[:, 18:28] = F.one_hot(opponent, 10)[:, :, None]
    phase = torch.from_numpy(rows["phase"].copy()).to(device).long()
    x[:, 28:31] = F.one_hot(phase, 3)[:, :, None]
    values = torch.from_numpy(rows["value"].astype(np.float32)).to(device)[:, None]
    policies = torch.from_numpy(rows["policy"].astype(np.int64)).to(device)
    return x.reshape(-1, 31, 5, 5), values, policies


def load_architecture():
    spec = importlib.util.spec_from_file_location("raven239_architecture", SRC / "network_31.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SantoriniZeroNet


def train(path, manifest, smoke=False):
    import torch
    import torch.nn.functional as F
    data, train_indices, val_indices = split_data(path, manifest, smoke)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing accidental overnight CPU training")
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")
    model = load_architecture()().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    checkpoint_dir = HERE / ("smoke-checkpoints" if smoke else "checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    epochs = 1 if smoke else 15
    metrics_path = HERE / ("smoke-metrics.jsonl" if smoke else "metrics.jsonl")
    first_epoch, best = 1, float("inf")
    last = checkpoint_dir / "last.pth"
    if last.exists():
        checkpoint = torch.load(last, map_location=device, weights_only=False)
        assert checkpoint["source_hash"] == sha(SRC / "network_31.py")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        first_epoch, best = checkpoint["epoch"] + 1, checkpoint["best_validation_loss"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint["cuda_rng"]])
    event("training_started", device=torch.cuda.get_device_name(0), torch=torch.__version__,
          train_samples=len(train_indices), validation_samples=len(val_indices), epochs=epochs,
          start_epoch=first_epoch, batch_size=256, learning_rate=0.001,
          initialization="from scratch (same retained training method)", precision="float32")
    for epoch in range(first_epoch, epochs + 1):
        stop_requested()
        began = time.monotonic()
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        # Reproducible per-epoch permutation without materializing float32 features.
        order = np.random.default_rng(42 + epoch).permutation(train_indices)
        max_batches = 8 if smoke else (len(order) + 255) // 256
        for batch, begin in enumerate(range(0, len(order), 256)):
            if batch >= max_batches:
                break
            if batch % 500 == 0:
                stop_requested()
            rows = data[order[begin:begin+256]]
            x, yv, yp = tensorize(rows, device)
            optimizer.zero_grad(set_to_none=True)
            policy, value = model(x)
            lv, lp = F.binary_cross_entropy_with_logits(value, yv), F.cross_entropy(policy, yp)
            loss = lv + lp
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            count = len(rows)
            totals += [lv.item() * count, lp.item() * count, count]
            if batch % 1000 == 0:
                elapsed = time.monotonic() - began
                event("training_progress", epoch=epoch, batch=batch+1, batches=max_batches,
                      value_loss=totals[0]/totals[2], policy_loss=totals[1]/totals[2],
                      samples_per_second=totals[2]/max(elapsed, 0.001))
        model.eval()
        validation = np.zeros(5, dtype=np.float64)
        with torch.inference_mode():
            for batch, begin in enumerate(range(0, len(val_indices), 256)):
                if smoke and batch >= 4:
                    break
                rows = data[val_indices[begin:begin+256]]
                x, yv, yp = tensorize(rows, device)
                policy, value = model(x)
                count = len(rows)
                validation += [F.binary_cross_entropy_with_logits(value, yv).item()*count,
                    F.cross_entropy(policy, yp).item()*count,
                    ((value > 0) == (yv > 0.5)).sum().item(),
                    (policy.argmax(1) == yp).sum().item(), count]
        score = float((validation[0] + validation[1]) / validation[4])
        improved = bool(score < best)
        best = min(best, score)
        metric = {"at": now(), "epoch": epoch, "train_value_loss": totals[0]/totals[2],
                  "train_policy_loss": totals[1]/totals[2],
                  "validation_value_loss": validation[0]/validation[4],
                  "validation_policy_loss": validation[1]/validation[4],
                  "validation_value_accuracy": validation[2]/validation[4],
                  "validation_policy_accuracy": validation[3]/validation[4],
                  "seconds": time.monotonic()-began, "best": improved}
        with metrics_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(metric) + "\n")
        weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(weights, checkpoint_dir / f"epoch-{epoch:02d}-weights.pth")
        checkpoint = {"epoch": epoch, "model": weights, "optimizer": optimizer.state_dict(),
                      "best_validation_loss": best, "source_hash": sha(SRC / "network_31.py"),
                      "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                      "metric": metric}
        torch.save(checkpoint, last.with_suffix(".tmp"))
        last.with_suffix(".tmp").replace(last)
        if improved:
            torch.save(weights, checkpoint_dir / "best-weights.pth")
        event("epoch_complete", **metric)
    if smoke:
        model = model.cpu().eval()
        example, _, _ = tensorize(data[val_indices[:16]], torch.device("cpu"))
        with torch.inference_mode():
            traced = torch.jit.trace(model, example[:1])
            traced.save(str(HERE / "smoke-evaluator.pt"))
            restored = torch.jit.load(str(HERE / "smoke-evaluator.pt"), map_location="cpu")
            for expected, actual in zip(model(example), restored(example)):
                torch.testing.assert_close(expected, actual)
        event("smoke_complete", checkpoint=True, torchscript_reload=True)
        return
    artifact = HERE / "artifacts/Davi_2.3.9_Raven"
    artifact.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(torch.load(checkpoint_dir / "best-weights.pth", map_location=device, weights_only=True))
    model = model.cpu().eval()
    example, _, _ = tensorize(data[val_indices[:16]], torch.device("cpu"))
    with torch.inference_mode():
        traced = torch.jit.trace(model, example[:1])
        traced.save(str(artifact / "santorini_evaluator.pt"))
        restored = torch.jit.load(str(artifact / "santorini_evaluator.pt"), map_location="cpu").eval()
        for expected, actual in zip(model(example), restored(example)):
            torch.testing.assert_close(expected, actual)
    shutil.copy2(checkpoint_dir / "best-weights.pth", artifact / "santorini_zero_weights.pth")
    executable = MED / "engines/Davi/Raven/Davi_2.3.8_Raven.exe"
    shutil.copy2(executable, artifact / "Davi_2.3.9_Raven.exe")
    save_json("artifacts/Davi_2.3.9_Raven/manifest.json", {
        "at": now(), "name": "Davi 2.3.9 Raven", "status": "trained candidate; not promoted",
        "architecture_sha256": sha(SRC / "network_31.py"),
        "engine_source_commit": "77a3a01db18efae98f69063ed7cf38091fcb2cd9 (historical source association)",
        "executable_provenance": "Byte-identical copy of archived 2.3.8; only neural weights changed.",
        "files": {p.name: sha(p) for p in artifact.iterdir() if p.is_file()},
        "selection": "Lowest combined validation loss across 15 epochs; all epoch weights retained.",
        "runtime": "Existing Raven LibTorch/CUDA DLL environment required.",
        "limitations": "Original 31-plane intermediate-context limitation and Raven search/rules behavior retained."})
    event("complete", artifact=str(artifact), best_validation_loss=best)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    lock = HERE / "running.lock"
    handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(handle, str(os.getpid()).encode())
    os.close(handle)
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        event("started", pid=os.getpid(), smoke=args.smoke)
        if not args.smoke:
            source_files = [Path(__file__), SRC / "network_31.py", SRC / "board_translation.py",
                            MED / "game/board.py", MED / "game/move.py", MED / "database/models.py",
                            SRC / "board_symmetry.py"]
            save_json("run-manifest.json", {"at": now(), "name": "Davi 2.3.9 Raven", "pid": os.getpid(),
                "python": sys.version, "executable": sys.executable,
                "source_files": {str(p): sha(p) for p in source_files},
                "recipe": {"epochs": 15, "batch_size": 256, "optimizer": "Adam", "learning_rate": 0.001,
                           "value_loss": "BCEWithLogitsLoss", "policy_loss": "CrossEntropyLoss",
                           "loss_weights": [1.0, 1.0], "seed": 42, "initialization": "from scratch",
                           "precision": "float32", "sample_weights": "uniform"},
                "scope": "Retrain existing Raven method on expanded validated game logs. New search-label/quality-weighting method reserved for Davi 3."})
        snapshot()
        path, manifest = prepare(500 if args.smoke else None)
        if not args.prepare_only:
            train(path, manifest, args.smoke)
    except InterruptedError as exc:
        event("stopped", reason=str(exc))
    except BaseException as exc:
        event("failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        lock.unlink(missing_ok=True)
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
