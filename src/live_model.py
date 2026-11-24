"""
Live GRU: the model behind the website's daily "tomorrow" forecast.

The live CPCB feed (data.gov.in, "Real time Air Quality Index from various
locations") gives pollutant sub-indices for the 7 Kolkata stations, not the
raw concentrations the main models were trained on. So the live model uses
only inputs the daily job can really get:

  * city AQI of the last 14 days
  * the 13 Open-Meteo weather inputs of the same days
  * day of year (sine, cosine)

It is trained, chosen and tested exactly like the main experiments:
train 2018-10-01..2023-12-31, validation 2024, test 2025-01-01..2026-10-31,
choice of target mode on validation only, test scored once against
persistence with a 95% block-bootstrap interval.

The chosen seeds are exported as plain numbers (models/live/gru_live.npz), so
the daily job runs the GRU with NumPy and needs no PyTorch.

    python main.py live-train
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.evaluate import alert_metrics, block_bootstrap_mae_diff
from src.model import AQIGRU, pick_device
from src.train import set_seed

FEATURES = ["AQI", *config.WEATHER_FEATURES, "doy_sin", "doy_cos"]
HIDDEN = 32
MODES = ["level", "change"]
SEEDS = config.V2_SEEDS
LIVE_DIR = config.PROJECT_ROOT / "models" / "live"
WEIGHTS = LIVE_DIR / "gru_live.npz"
META = LIVE_DIR / "gru_live.json"
REPORT = config.PROJECT_ROOT / "reports" / "live_model.md"
SITE_JSON = config.PROJECT_ROOT / "web" / "data" / "live_model.json"


def load_table() -> pd.DataFrame:
    t = pd.read_csv(config.V2_TABLE_CSV, parse_dates=["Date"])
    return t[["Date", "AQI_target", *FEATURES, "split"]].reset_index(drop=True)


def make_windows(t: pd.DataFrame, mean: np.ndarray, std: np.ndarray):
    """Inputs: days d-13..d (scaled). Target: AQI on d+1, never filled in."""
    X = ((t[FEATURES].to_numpy(float) - mean) / std).astype(np.float32)
    y = t["AQI_target"].to_numpy(float)
    today = t["AQI"].to_numpy(float)
    split = t["split"].to_numpy()
    xs, ys, last, sp, dates = [], [], [], [], []
    for d in range(config.LOOKBACK - 1, len(t) - 1):
        if np.isnan(y[d + 1]):
            continue
        xs.append(X[d - config.LOOKBACK + 1: d + 1])
        ys.append(y[d + 1])
        last.append(today[d])
        sp.append(split[d + 1])                 # a sample belongs to its target's split
        dates.append(t["Date"].iloc[d + 1])
    return np.stack(xs), np.array(ys), np.array(last), np.array(sp), pd.to_datetime(dates)


def encode(y, last, mode, s):
    return (y - s["mean"]) / s["std"] if mode == "level" else (y - last) / s["dstd"]


def decode(z, last, mode, s):
    return z * s["std"] + s["mean"] if mode == "level" else last + z * s["dstd"]


def train_one(mode, seed, X, z, tr, va, device):
    set_seed(seed)
    model = AQIGRU(len(FEATURES), hidden_size=HIDDEN, num_layers=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.tensor(X[tr]), torch.tensor(z[tr], dtype=torch.float32)),
                        batch_size=config.BATCH_SIZE, shuffle=True, generator=g)
    Xv = torch.tensor(X[va], device=device)
    zv = torch.tensor(z[va], dtype=torch.float32, device=device)
    best, best_state, wait = float("inf"), None, 0
    for _ in range(config.MAX_EPOCHS):
        model.train()
        for xb, zb in loader:
            xb, zb = xb.to(device), zb.to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), zb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(model(Xv), zv).item()
        if v < best - 1e-6:
            best, wait = v, 0
            best_state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
        else:
            wait += 1
            if wait >= config.PATIENCE:
                break
    model.load_state_dict(best_state)
    return model.eval()


def predict(model, X, device) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(X, device=device)).cpu().numpy().astype(float)


def numpy_gru(w: dict, x: np.ndarray) -> np.ndarray:
    """Same forward pass as torch.nn.GRU (1 layer) + linear head. x: (batch, days, features)."""
    H = w["weight_hh"].shape[1]
    h = np.zeros((x.shape[0], H))
    sig = lambda a: 1.0 / (1.0 + np.exp(-a))  # noqa: E731
    for step in range(x.shape[1]):
        gi = x[:, step, :] @ w["weight_ih"].T + w["bias_ih"]
        gh = h @ w["weight_hh"].T + w["bias_hh"]
        r = sig(gi[:, :H] + gh[:, :H])
        z = sig(gi[:, H:2 * H] + gh[:, H:2 * H])
        n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        h = (1 - z) * n + z * h
    return h @ w["head_w"].T[:, 0] + w["head_b"][0]


def export_weights(model) -> dict:
    s = {k: v.detach().cpu().numpy().astype(float) for k, v in model.state_dict().items()}
    return {"weight_ih": s["gru.weight_ih_l0"], "weight_hh": s["gru.weight_hh_l0"],
            "bias_ih": s["gru.bias_ih_l0"], "bias_hh": s["gru.bias_hh_l0"],
            "head_w": s["head.1.weight"], "head_b": s["head.1.bias"]}


def run_live_training(device_name: str = config.DEVICE) -> dict:
    device = pick_device(device_name)
    t = load_table()
    train_rows = t["split"] == "train"
    mean = t.loc[train_rows, FEATURES].mean().to_numpy(float).copy()
    std = t.loc[train_rows, FEATURES].std().to_numpy(float).copy()
    std[std == 0] = 1.0

    X, y, last, split, dates = make_windows(t, mean, std)
    tr, va, te = split == "train", split == "val", split == "test"
    s = {"mean": float(y[tr].mean()), "std": float(y[tr].std()),
         "dstd": float((y[tr] - last[tr]).std())}
    print(f"Samples: train {tr.sum()}, validation {va.sum()}, test {te.sum()}  (device: {device})")

    results = {}
    for mode in MODES:
        z = encode(y, last, mode, s)
        models = [train_one(mode, seed, X, z, tr, va, device) for seed in SEEDS]
        val_maes = [float(np.mean(np.abs(decode(predict(m, X[va], device), last[va], mode, s) - y[va])))
                    for m in models]
        results[mode] = {"models": models, "val_mae": float(np.mean(val_maes)),
                         "val_sd": float(np.std(val_maes))}
        print(f"  {mode:6s} validation MAE {results[mode]['val_mae']:.2f} +/- {results[mode]['val_sd']:.2f}")

    mode = min(MODES, key=lambda m: results[m]["val_mae"])
    models = results[mode]["models"]
    print(f"Chosen on validation: {mode}")

    def avg_pred(mask):
        return np.mean([decode(predict(m, X[mask], device), last[mask], mode, s) for m in models], axis=0)

    # 80% range from validation errors only
    val_err = y[va] - avg_pred(va)
    q_lo, q_hi = (float(v) for v in np.percentile(val_err, [10, 90]))

    # test, scored once
    p, yt, pers = avg_pred(te), y[te], last[te]
    mae, pmae = float(np.mean(np.abs(p - yt))), float(np.mean(np.abs(pers - yt)))
    rmse = float(np.sqrt(np.mean((p - yt) ** 2)))
    r2 = float(1 - np.sum((p - yt) ** 2) / np.sum((yt - yt.mean()) ** 2))
    diff, lo, hi = block_bootstrap_mae_diff(yt, p, pers)
    cover = float(np.mean((yt >= p + q_lo) & (yt <= p + q_hi)))
    f1, pf1 = alert_metrics(yt, p)["F1"], alert_metrics(yt, pers)["F1"]

    # check the NumPy copy gives the same forecasts as PyTorch
    exported = [export_weights(m) for m in models]
    np_pred = np.mean([decode(numpy_gru(w, X[te].astype(float)), last[te], mode, s) for w in exported], axis=0)
    max_gap = float(np.max(np.abs(np_pred - p)))
    assert max_gap < 0.05, f"NumPy GRU differs from PyTorch by {max_gap:.3f} AQI"

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(WEIGHTS, **{f"s{i}_{k}": v for i, w in enumerate(exported) for k, v in w.items()})
    meta = {"created": datetime.now().isoformat(timespec="seconds"), "features": FEATURES,
            "lookback": config.LOOKBACK, "mode": mode, "seeds": len(models),
            "feature_mean": mean.tolist(), "feature_std": std.tolist(), "target": s,
            "range80": [q_lo, q_hi]}
    META.write_text(json.dumps(meta, indent=2))

    metrics = {"model": f"GRU (1 x {HIDDEN}), {mode} target, average of {len(models)} seeds",
               "inputs": "past 14 days of city AQI + 13 weather variables + day of year",
               "test_period": f"{dates[te][0].date()} to {dates[te][-1].date()}",
               "test_days": int(te.sum()), "mae": round(mae, 1), "rmse": round(rmse, 1),
               "r2": round(r2, 3), "persistence_mae": round(pmae, 1),
               "diff": round(diff, 1), "diff_low": round(lo, 1), "diff_high": round(hi, 1),
               "range_coverage": round(cover, 3), "alert_f1": round(f1, 3),
               "persistence_alert_f1": round(pf1, 3),
               "validation": {m: round(results[m]["val_mae"], 2) for m in MODES}}
    SITE_JSON.parent.mkdir(parents=True, exist_ok=True)
    SITE_JSON.write_text(json.dumps(metrics, indent=2))

    verdict = "better than persistence" if hi < 0 else ("worse than persistence" if lo > 0
                                                       else "no clear difference")
    REPORT.write_text(
        "# Live GRU (for the website)\n\n"
        f"Inputs: {metrics['inputs']} ({len(FEATURES)} per day). Chosen target mode on validation: "
        f"**{mode}** (level {results['level']['val_mae']:.2f}, change {results['change']['val_mae']:.2f}).\n\n"
        "| test period | days | live GRU MAE | persistence MAE | difference [95%] | verdict | RMSE | R2 | "
        "80% range coverage | alert F1 (GRU / persistence) |\n|---|---|---|---|---|---|---|---|---|---|\n"
        f"| {metrics['test_period']} | {metrics['test_days']} | {mae:.1f} | {pmae:.1f} | "
        f"{diff:+.1f} [{lo:+.1f}, {hi:+.1f}] | {verdict} | {rmse:.1f} | {r2:.3f} | {cover:.0%} | "
        f"{f1:.3f} / {pf1:.3f} |\n")
    print(f"Test MAE {mae:.1f} vs persistence {pmae:.1f}, difference {diff:+.1f} [{lo:+.1f}, {hi:+.1f}] - {verdict}")
    print(f"80% range coverage on test: {cover:.0%}.  Saved {WEIGHTS.name}, {META.name}, {REPORT.name}")
    return metrics
