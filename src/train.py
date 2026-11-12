"""
Train the LSTM variants and pick one using the VALIDATION period only.

For every variant in config.VARIANTS and every seed in config.SEEDS:
  1. train on the training period, stopping early when validation loss stops
     improving (the best epoch's weights are kept);
  2. save the checkpoint to models/experiment/<variant>_seed<seed>.pt;
  3. record its validation MAE / RMSE / bias.
Then the variant with the lowest mean validation MAE is written to
models/selected_model.json. The test period is never touched here.
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn

from src import config
from src.dataset import decode_target, encode_target, make_windows, target_stats, to_loader
from src.model import AQILSTM, pick_device
from src.preprocessing import StandardScaler, feature_columns, load_processed, run_preprocessing


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_data(verbose: bool):
    if not config.PROCESSED_CSV.exists() or not config.SCALER_JSON.exists():
        run_preprocessing(verbose=verbose)
    clean = load_processed()
    scaler = StandardScaler.from_json(config.SCALER_JSON)
    return make_windows(clean, scaler), scaler


@torch.no_grad()
def _loss_on(model, loader, loss_fn, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total += loss_fn(model(xb), yb).item() * len(yb)
        n += len(yb)
    return total / n


@torch.no_grad()
def _forecast(model, X: np.ndarray, device) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(X).to(device)).cpu().numpy()


def checkpoint_path(variant: str, seed: int):
    return config.EXPERIMENT_DIR / f"{variant}_seed{seed}.pt"


def train_one(variant: str, seed: int, windows, scaler, device) -> dict:
    """Train one (variant, seed). Returns a row of validation results."""
    from src.evaluate import regression_metrics

    v = config.VARIANTS[variant]
    mode = v["target_mode"]
    set_seed(seed)

    w_train, w_val = windows.subset("train"), windows.subset("val")
    stats = target_stats(w_train, mode, scaler)
    y_train = encode_target(w_train.y_true, w_train.last_aqi, mode, stats)
    y_val = encode_target(w_val.y_true, w_val.last_aqi, mode, stats)
    train_loader = to_loader(w_train.X, y_train, config.BATCH_SIZE, shuffle=True, seed=seed)  # shuffled within train only
    val_loader = to_loader(w_val.X, y_val, config.BATCH_SIZE, shuffle=False, seed=seed)

    model = AQILSTM(n_features=len(feature_columns()), hidden_size=v["hidden_size"],
                    num_layers=v["num_layers"], dropout=config.DROPOUT).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    history, best = [], {"val_loss": float("inf"), "epoch": -1, "state": None}
    bad_epochs = 0
    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimiser.step()
            running += loss.item() * len(yb)
            n += len(yb)
        val_loss = _loss_on(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": running / n, "val_loss": val_loss})
        if val_loss < best["val_loss"] - 1e-6:
            best = {"val_loss": val_loss, "epoch": epoch,
                    "state": {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= config.PATIENCE:
            break

    # validation forecast with the best weights, in real AQI units
    model.load_state_dict(best["state"])
    val_pred = decode_target(_forecast(model, w_val.X, device), w_val.last_aqi, mode, stats)
    m = regression_metrics(w_val.y_true, val_pred)

    config.EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best["state"],
        "variant": variant,
        "target_mode": mode,
        "target_mean": stats[0],
        "target_std": stats[1],
        "feature_columns": feature_columns(),
        "hidden_size": v["hidden_size"],
        "num_layers": v["num_layers"],
        "dropout": config.DROPOUT,
        "lookback": config.LOOKBACK,
        "horizon": config.HORIZON,
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "epochs_run": len(history),
        "seed": seed,
        "device_used": str(device),
        "start_date": config.START_DATE,
        "train_end": config.TRAIN_END,
        "val_end": config.VAL_END,
        "scaler": {"columns": scaler.columns, "mean": scaler.mean, "std": scaler.std},
    }, checkpoint_path(variant, seed))
    pd.DataFrame(history).to_csv(checkpoint_path(variant, seed).with_suffix(".history.csv"), index=False)

    n_params = sum(p.numel() for p in model.parameters())
    return {"variant": variant, "seed": seed, "target_mode": mode,
            "hidden_size": v["hidden_size"], "num_layers": v["num_layers"], "n_weights": n_params,
            "best_epoch": best["epoch"], "epochs_run": len(history),
            "val_MAE": m["MAE"], "val_RMSE": m["RMSE"], "val_Bias": m["Bias"]}


def train_all(device_name: str = config.DEVICE, verbose: bool = True) -> dict:
    device = pick_device(device_name)
    windows, scaler = _load_data(verbose)
    if verbose:
        print(f"Device: {device} | train samples: {len(windows.subset('train'))} | "
              f"val samples: {len(windows.subset('val'))} | features: {len(feature_columns())} | "
              f"lookback: {config.LOOKBACK}")
        print(f"Training {len(config.VARIANTS)} variants x {len(config.SEEDS)} seeds "
              f"= {len(config.VARIANTS) * len(config.SEEDS)} models\n")

    rows, t0 = [], time.time()
    for variant in config.VARIANTS:
        for seed in config.SEEDS:
            r = train_one(variant, seed, windows, scaler, device)
            rows.append(r)
            if verbose:
                print(f"  {variant:<13} seed {seed}: best epoch {r['best_epoch']:3d} | "
                      f"val MAE {r['val_MAE']:6.1f} | val bias {r['val_Bias']:+6.1f}")

    runs = pd.DataFrame(rows)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.VALIDATION_RUNS_CSV, index=False)

    summary = (runs.groupby("variant", sort=False)
               .agg(n_weights=("n_weights", "first"),
                    val_MAE_mean=("val_MAE", "mean"), val_MAE_sd=("val_MAE", "std"),
                    val_RMSE_mean=("val_RMSE", "mean"), val_Bias_mean=("val_Bias", "mean")))
    selected = summary["val_MAE_mean"].idxmin()
    with open(config.SELECTED_JSON, "w") as f:
        json.dump({
            "selected_variant": selected,
            "rule": "lowest mean validation MAE over seeds (test period not used)",
            "seeds": config.SEEDS,
            "validation_MAE_mean_by_variant": summary["val_MAE_mean"].round(3).to_dict(),
            "decided_at": datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2)
    _plot_histories(selected)

    if verbose:
        print("\nValidation summary (mean over seeds; lower MAE is better):")
        print(summary.round(1).to_string())
        print(f"\nSelected on validation: {selected}  (time {time.time() - t0:.0f}s)")
        print(f"Saved: {config.EXPERIMENT_DIR}/, {config.VALIDATION_RUNS_CSV.name}, "
              f"{config.SELECTED_JSON.name}")
    return {"selected": selected, "runs": runs}


def _plot_histories(variant: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    from src.evaluate import BLUE, ORANGE, style_axes

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, seed in enumerate(config.SEEDS):
        h = pd.read_csv(checkpoint_path(variant, seed).with_suffix(".history.csv"))
        ax.plot(h["epoch"], h["train_loss"], color=BLUE, linewidth=1.2, alpha=0.8,
                label="Training loss" if i == 0 else None)
        ax.plot(h["epoch"], h["val_loss"], color=ORANGE, linewidth=1.2, alpha=0.8,
                label="Validation loss" if i == 0 else None)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared error (scaled target)")
    ax.set_title(f"Training curves, {variant} (one line per seed)", loc="left", fontsize=11)
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / "lstm_training_curve.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    train_all()
