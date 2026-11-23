"""Repeated runs, aggregation, metadata, and outputs."""
from datetime import datetime, timezone
from pathlib import Path
import json, platform
import torch
import pandas as pd
from .data import load_dataset
from .models import MLP
from .optimizers import get_optimizer
from .training import train_model
from .plotting import create_plots

OPTIMIZERS = ("sgd", "momentum", "adamw", "lbfgs")

def run_experiment(config, project_root=None):
    root = Path(project_root or Path.cwd())
    out = root / config.get("output_dir", "results")
    out.mkdir(parents=True, exist_ok=True)
    device = config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    rows, histories = [], {}
    for noise in config["noise_levels"]:
        for seed in config["seeds"]:
            xtr, xte, ytr, yte = load_dataset(
                config["n_samples"], noise, config["test_fraction"], seed)
            for name in OPTIMIZERS:
                torch.manual_seed(seed)
                model = MLP(hidden_dim=config["hidden_dim"])
                opt = get_optimizer(name, model.parameters(),
                    config["learning_rates"][name], config.get("weight_decay", 0.0))
                hist = train_model(model, opt, xtr, ytr, xte, yte,
                    config["epochs"], config["batch_size"], device)
                histories[f"{name}_noise{noise}_seed{seed}"] = hist
                final = hist[-1]
                rows.append({
                    "optimizer": name, "noise": noise, "seed": seed,
                    "final_train_loss": final["train_loss"],
                    "final_test_loss": final["test_loss"],
                    "final_train_accuracy": final["train_accuracy"],
                    "final_test_accuracy": final["test_accuracy"],
                    "final_gradient_norm": final["gradient_norm"],
                    "total_epoch_seconds": sum(r["epoch_seconds"] for r in hist),
                    "epochs": config["epochs"]
                })
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "run_level_metrics.csv", index=False)
    metrics = ["final_train_loss", "final_test_loss", "final_train_accuracy",
               "final_test_accuracy", "final_gradient_norm", "total_epoch_seconds"]
    summary = runs.groupby(["noise", "optimizer"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(out / "summary.csv", index=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__, "device": device,
        "config": config,
        "note": "L-BFGS is full-batch; other optimizers are mini-batch. Epochs are not equal gradient-evaluation budgets."
    }
    (out / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    create_plots(runs, histories, out)
    print(f"Completed {len(runs)} runs. Results saved in {out}")
    return runs, summary
