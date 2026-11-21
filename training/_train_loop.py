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
        assert checkpoint["source_hash"] == sha(SOURCE / "network.py")
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
                      "best_validation_loss": best, "source_hash": sha(SOURCE / "network.py"),
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
    artifact = HERE / "artifacts/context-35"
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
    save_json("artifacts/context-35/manifest.json", {
        "at": now(), "status": "trained weights only; no compatible executable packaged",
        "input_planes": 35, "architecture_sha256": sha(SOURCE / "network.py"),
        "selection": "Lowest combined validation loss across 15 epochs",
        "files": {p.name: sha(p) for p in artifact.iterdir() if p.is_file()},
        "runtime": "Requires a separately built 35-plane runtime; old Raven/Cypher EXE incompatible"})
    event("complete", artifact=str(artifact), best_validation_loss=best)
