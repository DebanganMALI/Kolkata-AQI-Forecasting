"""
Advanced deep-learning experiment:  python main.py advanced

Same task, data, split and rules as the rest of the project (pollution inputs
only, 14 days -> next-day AQI). It adds four things the main LSTM cannot show:

  1. Attention LSTM  - which of the 14 past days does the network rely on?
  2. TCN             - a convolutional sequence model (different family).
  3. Transformer     - self-attention, the modern architecture, on small data.
  4. Quantile attention LSTM - a forecast RANGE (10th-90th percentile) trained
                       with the pinball loss, and a cautious alert from its upper end.
  + Multi-horizon    - the chosen attention LSTM trained to forecast 1, 3 and 7
                       days ahead, to see how error grows with distance.

Rules:
  * Candidate settings are fixed in config.py (ADVANCED_CANDIDATES) before running.
  * Every candidate is trained with 5 seeds; the one with the best mean
    VALIDATION score is chosen (MAE; pinball loss for the quantile model).
  * Choices are saved to models/advanced/selected.json, and only then are the
    already-trained chosen models asked for TEST forecasts, scored once.
  * The 3- and 7-day models reuse the setting chosen for 1 day (no re-tuning).
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn

from src import config
from src.advanced_models import TCN, AttentionLSTM, TransformerForecaster, pinball_loss
from src.dataset import decode_target, encode_target, make_windows, target_stats, to_loader
from src.evaluate import (BLUE, INK, INK2, ORANGE, alert_metrics, block_bootstrap_mae_diff,
                          regression_metrics, style_axes)
from src.model import pick_device
from src.preprocessing import StandardScaler, load_processed, run_preprocessing
from src.train import set_seed

Q = np.asarray(config.QUANTILES)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def make_model(arch: str, cand: dict, n_features: int) -> nn.Module:
    if arch == "Attention LSTM":
        return AttentionLSTM(n_features, hidden_size=cand["hidden_size"])
    if arch == "Quantile attention LSTM":
        return AttentionLSTM(n_features, hidden_size=cand["hidden_size"], quantiles=True)
    if arch == "TCN":
        return TCN(n_features, channels=cand["channels"])
    if arch == "Transformer":
        return TransformerForecaster(n_features, d_model=cand["d_model"], num_layers=cand["num_layers"])
    raise ValueError(arch)


def train_net(factory, seed, X_tr, y_tr, X_va, y_va, loss_fn, device) -> nn.Module:
    """Adam + early stopping on validation loss, as in src/train.py."""
    set_seed(seed)
    model = factory().to(device)
    train_loader = to_loader(X_tr, y_tr, config.BATCH_SIZE, shuffle=True, seed=seed)
    val_loader = to_loader(X_va, y_va, config.BATCH_SIZE, shuffle=False, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    best_loss, best_state, bad = float("inf"), None, 0
    for _ in range(config.MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            opt.step()
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                total += loss_fn(model(xb), yb).item() * len(yb)
                n += len(yb)
        if total / n < best_loss - 1e-6:
            best_loss, bad = total / n, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if bad >= config.PATIENCE:
            break
    model.load_state_dict(best_state)
    return model.eval()


def run_net(model, X, device):
    """Network output (scaled) + attention weights when the model has them."""
    with torch.no_grad():
        out = model(torch.from_numpy(X).to(device)).cpu().numpy()
    att = getattr(model, "last_attention", None)
    return out, (att.cpu().numpy() if att is not None else None)


def decode(z, w, mode, stats, quantile: bool):
    last = w.last_aqi[:, None] if quantile else w.last_aqi
    return decode_target(z, last, mode, stats)


def pinball_np(y, P) -> float:
    e = np.asarray(y, float)[:, None] - P
    return float(np.maximum(Q * e, (Q - 1) * e).mean())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_advanced(device_name: str = config.DEVICE, verbose: bool = True) -> dict:
    t0 = time.time()
    device = pick_device(device_name)
    if not config.PROCESSED_CSV.exists():
        run_preprocessing(verbose=False)
    clean = load_processed()
    scaler = StandardScaler.from_json(config.SCALER_JSON)     # fitted on training rows only
    windows = {h: make_windows(clean, scaler, horizon=h) for h in config.HORIZONS}
    w = windows[1]
    tr, va, te = w.subset("train"), w.subset("val"), w.subset("test")
    nf = w.X.shape[2]
    n_models = sum(len(c) for c in config.ADVANCED_CANDIDATES.values()) * len(config.SEEDS)
    if verbose:
        print(f"Device: {device} | train {len(tr)} | val {len(va)} | test {len(te)} | features {nf}")
        print(f"Phase 1 - choosing settings on VALIDATION ({n_models} networks)...\n")

    # ---- Phase 1: train every candidate, score on validation --------------
    rows, trained = [], {}
    for arch, cands in config.ADVANCED_CANDIDATES.items():
        quantile = arch.startswith("Quantile")
        loss_fn = pinball_loss if quantile else nn.MSELoss()
        ta = time.time()
        for cand in cands:
            mode = cand["target_mode"]
            stats = target_stats(tr, mode, scaler)
            y_tr = encode_target(tr.y_true, tr.last_aqi, mode, stats)
            y_va = encode_target(va.y_true, va.last_aqi, mode, stats)
            models = []
            for seed in config.SEEDS:
                m = train_net(lambda: make_model(arch, cand, nf), seed, tr.X, y_tr, va.X, y_va,
                              loss_fn, device)
                p = decode(run_net(m, va.X, device)[0], va, mode, stats, quantile)
                point = p[:, 1] if quantile else p
                rows.append({"model": arch, "candidate": cand["name"], "seed": seed,
                             "val_MAE": regression_metrics(va.y_true, point)["MAE"],
                             "val_pinball": pinball_np(va.y_true, p) if quantile else np.nan})
                models.append(m)
            trained[(arch, cand["name"])] = {"models": models, "mode": mode, "stats": stats,
                                             "cand": cand, "quantile": quantile}
        if verbose:
            r = pd.DataFrame([x for x in rows if x["model"] == arch])
            metric = "val_pinball" if quantile else "val_MAE"
            best = r.groupby("candidate")[metric].mean()
            print(f"  {arch:<24} best {metric} {best.min():6.2f} ({best.idxmin()})  "
                  f"[{time.time() - ta:.0f}s]")

    runs = pd.DataFrame(rows)
    config.ADVANCED_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.ADVANCED_DIR / "validation_runs.csv", index=False)
    chosen = {}
    for arch in config.ADVANCED_CANDIDATES:
        r = runs[runs["model"] == arch]
        metric = "val_pinball" if arch.startswith("Quantile") else "val_MAE"
        g = r.groupby("candidate", sort=False)[metric].agg(["mean", "std"])
        chosen[arch] = {"candidate": g["mean"].idxmin(), "metric": metric,
                        "mean": float(g["mean"].min()), "sd": float(g.loc[g["mean"].idxmin(), "std"])}
    with open(config.ADVANCED_DIR / "selected.json", "w") as f:
        json.dump({"rule": "lowest mean validation score over seeds (MAE; pinball for quantiles); "
                           "test period not used",
                   "selected": chosen, "decided_at": datetime.now().isoformat(timespec="seconds")},
                  f, indent=2)

    # ---- Phase 2: test forecasts from the chosen, already-trained models ---
    if verbose:
        print("\nPhase 2 - settings fixed; scoring the chosen models on TEST once...")
    test, attention = {}, {}
    for arch, c in chosen.items():
        t = trained[(arch, c["candidate"])]
        outs = [run_net(m, te.X, device) for m in t["models"]]
        preds = np.mean([decode(o[0], te, t["mode"], t["stats"], t["quantile"]) for o in outs], axis=0)
        val_p = np.mean([decode(run_net(m, va.X, device)[0], va, t["mode"], t["stats"], t["quantile"])
                         for m in t["models"]], axis=0)
        test[arch] = {"pred": preds, "val_pred": val_p, **t}
        if outs[0][1] is not None:
            attention[arch] = np.mean([o[1] for o in outs], axis=0)       # (samples, days)

    # ---- Multi-horizon: chosen attention LSTM setting, 1 / 3 / 7 days ------
    horizon_rows = []
    base = trained[("Attention LSTM", chosen["Attention LSTM"]["candidate"])]
    for h in config.HORIZONS:
        wh = windows[h]
        trh, vah, teh = wh.subset("train"), wh.subset("val"), wh.subset("test")
        if h == 1:
            pred = test["Attention LSTM"]["pred"]
        else:
            mode = base["mode"]
            stats = target_stats(trh, mode, scaler)
            y_tr = encode_target(trh.y_true, trh.last_aqi, mode, stats)
            y_va = encode_target(vah.y_true, vah.last_aqi, mode, stats)
            ms = [train_net(lambda: make_model("Attention LSTM", base["cand"], nf), s, trh.X, y_tr,
                            vah.X, y_va, nn.MSELoss(), device) for s in config.SEEDS]
            pred = np.mean([decode(run_net(m, teh.X, device)[0], teh, mode, stats, False)
                            for m in ms], axis=0)
        d, lo, hi = block_bootstrap_mae_diff(teh.y_true, pred, teh.last_aqi)
        horizon_rows.append({"horizon": h, "n": len(teh),
                             "model_MAE": regression_metrics(teh.y_true, pred)["MAE"],
                             "persistence_MAE": regression_metrics(teh.y_true, teh.last_aqi)["MAE"],
                             "diff": d, "lo": lo, "hi": hi})
        if verbose:
            print(f"  horizon {h} day(s): attention LSTM MAE {horizon_rows[-1]['model_MAE']:.1f} vs "
                  f"persistence {horizon_rows[-1]['persistence_MAE']:.1f}")

    r = _score(te, va, chosen, test, attention, pd.DataFrame(horizon_rows), str(device),
               time.time() - t0)
    if verbose:
        _print(r)
    return r


# ---------------------------------------------------------------------------
# Scoring, report and figures
# ---------------------------------------------------------------------------
def _score(te, va, chosen, test, attention, horizons, device, seconds) -> dict:
    y, pers = te.y_true, te.last_aqi
    lock = pd.to_datetime(te.target_dates) >= pd.Timestamp(config.LOCKDOWN_START)

    def point_row(name, p):
        m = regression_metrics(y, p)
        d, lo, hi = block_bootstrap_mae_diff(y, p, pers)
        return {"model": name, **m, "alert_F1": alert_metrics(y, p)["F1"],
                "MAE_before": float(np.abs(p - y)[~lock].mean()),
                "MAE_lockdown": float(np.abs(p - y)[lock].mean()), "diff": d, "lo": lo, "hi": hi}

    table = [point_row("Persistence", pers)]
    for arch, t in test.items():
        table.append(point_row(arch + (" (median)" if t["quantile"] else ""),
                               t["pred"][:, 1] if t["quantile"] else t["pred"]))
    table = pd.DataFrame(table)

    qt = test["Quantile attention LSTM"]
    P, Pv = qt["pred"], qt["val_pred"]
    inside = (y >= P[:, 0]) & (y <= P[:, 2])
    intervals = {
        "nominal": int(round((Q[-1] - Q[0]) * 100)),
        "cover_val": float(((va.y_true >= Pv[:, 0]) & (va.y_true <= Pv[:, 2])).mean()),
        "cover_test": float(inside.mean()),
        "cover_before": float(inside[~lock].mean()), "cover_lockdown": float(inside[lock].mean()),
        "width_test": float((P[:, 2] - P[:, 0]).mean()),
        "width_before": float((P[:, 2] - P[:, 0])[~lock].mean()),
        "width_lockdown": float((P[:, 2] - P[:, 0])[lock].mean()),
        "pinball_test": pinball_np(y, P),
    }
    alerts = pd.DataFrame([
        {"rule": "Persistence > 200", **alert_metrics(y, pers)},
        {"rule": "Quantile median > 200", **alert_metrics(y, P[:, 1])},
        {"rule": f"Quantile upper ({int(Q[-1] * 100)}th pct) > 200", **alert_metrics(y, P[:, 2])},
    ])

    att = attention["Attention LSTM"]                       # (samples, days), oldest day first
    days_before = np.arange(att.shape[1], 0, -1)            # 14 ... 1 day(s) before the forecast day
    att_table = pd.DataFrame({"days_before": days_before, "all": att.mean(0),
                              "before": att[~lock].mean(0), "lockdown": att[lock].mean(0)})

    preds = pd.DataFrame({"date": pd.to_datetime(te.target_dates).strftime("%Y-%m-%d"),
                          "actual_AQI": y, "persistence": pers})
    for arch, t in test.items():
        if t["quantile"]:
            for i, q in enumerate(Q):
                preds[f"{arch} q{int(q * 100)}"] = np.round(t["pred"][:, i], 2)
        else:
            preds[arch] = np.round(t["pred"], 2)
    preds.to_csv(config.ADVANCED_PREDICTIONS_CSV, index=False)

    r = {"table": table, "chosen": chosen, "intervals": intervals, "alerts": alerts,
         "attention": att_table, "horizons": horizons, "device": device, "seconds": seconds,
         "n_test": len(y)}
    r["figures"] = _plots(te, P, att_table, horizons)
    _write_report(r)
    return r


def _md(rows, header) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def _verdict(lo, hi) -> str:
    return "better than persistence" if hi < 0 else "worse than persistence" if lo > 0 else "no clear difference"


def _write_report(r: dict) -> None:
    iv, att = r["intervals"], r["attention"]
    top = att.sort_values("all", ascending=False).head(3)
    flat = 100 / len(att)
    L = ["# Advanced Deep-Learning Experiment", "",
         f"*Generated by `python main.py advanced` on {datetime.now():%Y-%m-%d %H:%M} "
         f"({r['seconds'] / 60:.1f} min on `{r['device']}`). All numbers come from this run.*", "",
         "## Set-up", "",
         "- Same task, data and date split as the main LSTM: pollution inputs only, previous "
         f"{config.LOOKBACK} days -> next-day AQI; train to {config.TRAIN_END}, validation to "
         f"{config.VAL_END}, test after that ({r['n_test']} days).",
         "- **Attention LSTM:** an LSTM whose additive attention layer weights each of the "
         "14 days; the weighted summary and the latest day feed the output.",
         "- **TCN:** three residual blocks of dilated causal 1-D convolutions (dilation 1, 2, 4).",
         "- **Transformer:** 2 encoder layers, 4 attention heads, learned position vectors.",
         "- **Quantile attention LSTM:** outputs the 10th, 50th and 90th percentiles, trained "
         "with the pinball loss; the outputs are built so they never cross.",
         f"- Each model: 2 candidate settings (predict AQI directly or the change from today) x "
         f"{len(config.SEEDS)} seeds; the best mean validation score was chosen and saved to "
         "`models/advanced/selected.json` before the test period was scored once.",
         "",
         "## Settings chosen on validation", "",
         _md([[a, c["candidate"], c["metric"].replace("val_", ""), f"{c['mean']:.2f} ± {c['sd']:.2f}"]
              for a, c in r["chosen"].items()], ["model", "chosen setting", "metric", "validation (mean ± sd)"]),
         "",
         "## Test period (scored once)", "",
         _md([[t["model"], f"{t['MAE']:.1f}", f"{t['RMSE']:.1f}", f"{t['R2']:.3f}", f"{t['Bias']:+.1f}",
               f"{t['alert_F1']:.3f}",
               "-" if t["model"] == "Persistence" else f"{t['diff']:+.1f} [{t['lo']:+.1f}, {t['hi']:+.1f}]",
               "-" if t["model"] == "Persistence" else _verdict(t["lo"], t["hi"])]
              for _, t in r["table"].iterrows()],
             ["model", "MAE", "RMSE", "R2", "Bias", f"alert F1 (>{config.ALERT_THRESHOLD})",
              "MAE vs persistence [95% interval]", "verdict"]),
         "",
         "## What the attention LSTM looks at", "",
         f"Average attention weight on each of the {len(att)} input days, test period. "
         f"Equal weighting would give {flat:.1f}% per day.", "",
         _md([[int(a["days_before"]), f"{a['all'] * 100:.1f}%", f"{a['before'] * 100:.1f}%",
               f"{a['lockdown'] * 100:.1f}%"] for _, a in att.iterrows()],
             ["days before forecast", "all test days", f"before {config.LOCKDOWN_START}", "lockdown"]),
         "",
         "Most weight: " + ", ".join(f"{int(a['days_before'])} day(s) before ({a['all'] * 100:.1f}%)"
                                       for _, a in top.iterrows()) + ".",
         "",
         f"## Forecast intervals ({iv['nominal']}% range, quantile attention LSTM)", "",
         _md([["validation", f"{iv['cover_val'] * 100:.0f}%", "-"],
              ["test (all)", f"{iv['cover_test'] * 100:.0f}%", f"{iv['width_test']:.0f}"],
              [f"test before {config.LOCKDOWN_START}", f"{iv['cover_before'] * 100:.0f}%", f"{iv['width_before']:.0f}"],
              ["test, lockdown", f"{iv['cover_lockdown'] * 100:.0f}%", f"{iv['width_lockdown']:.0f}"]],
             ["period", f"actual AQI inside the range (target {iv['nominal']}%)", "mean range width (AQI)"]),
         "",
         f"Test pinball loss: {iv['pinball_test']:.2f}. A well-calibrated range contains the actual "
         f"AQI on about {iv['nominal']}% of days; much less means the model is over-confident.",
         "",
         "### Alerts from the range", "",
         _md([[a["rule"], a["alerts_raised"], a["correct_alerts_TP"], a["false_alarms_FP"], a["missed_FN"],
               f"{a['precision']:.3f}", f"{a['recall']:.3f}", f"{a['F1']:.3f}"]
              for _, a in r["alerts"].iterrows()],
             ["alert rule", "alerts", "correct", "false alarms", "missed", "precision", "recall", "F1"]),
         "",
         "## Forecasting further ahead (attention LSTM)", "",
         _md([[h["horizon"], h["n"], f"{h['model_MAE']:.1f}", f"{h['persistence_MAE']:.1f}",
               f"{h['diff']:+.1f} [{h['lo']:+.1f}, {h['hi']:+.1f}]", _verdict(h["lo"], h["hi"])]
              for _, h in r["horizons"].iterrows()],
             ["days ahead", "test days", "attention LSTM MAE", "persistence MAE",
              "difference [95% interval]", "verdict"]),
         "",
         "The 3- and 7-day models reuse the setting chosen for 1 day; nothing was re-tuned.",
         "",
         "## How to read these results honestly", "",
         "- About two years of data and 458 training samples: large networks can easily memorise "
         "them, so a more complex architecture is not expected to win.",
         "- Attention weights show what the network used, not what causes pollution.",
         "- The test period contains the COVID-19 lockdown, unlike anything in training.",
         "- One chronological split; small differences may not hold for other periods.",
         "",
         "## Files", "",
         "- `models/advanced/validation_runs.csv`, `models/advanced/selected.json`",
         f"- `{config.ADVANCED_PREDICTIONS_CSV.relative_to(config.PROJECT_ROOT).as_posix()}`"]
    L += [f"- `{p.relative_to(config.PROJECT_ROOT).as_posix()}`" for p in r["figures"]]
    config.ADVANCED_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


def _plots(te, P, att, horizons) -> list:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []

    fig, ax = plt.subplots(figsize=(8, 4))
    x = att["days_before"].astype(str)
    ax.bar(x, att["all"] * 100, color=BLUE)
    ax.axhline(100 / len(att), color=INK2, linestyle="--", linewidth=1)
    ax.text(len(att) - 0.5, 100 / len(att) + 0.3, "equal weighting", color=INK2, fontsize=9, ha="right")
    ax.set_xlabel("Days before the forecast day")
    ax.set_ylabel("Average attention weight (%)")
    ax.set_title("Which past days the attention LSTM relies on (test period)", loc="left", fontsize=11)
    style_axes(ax)
    fig.tight_layout()
    p = config.FIG_DIR / "attention_by_day.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    dates = pd.to_datetime(te.target_dates)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(dates, P[:, 0], P[:, 2], color=BLUE, alpha=0.18, linewidth=0,
                    label=f"{int(Q[0] * 100)}th to {int(Q[-1] * 100)}th percentile")
    ax.plot(dates, P[:, 1], color=BLUE, linewidth=1.6, label="median forecast")
    ax.plot(dates, te.y_true, color=INK, linewidth=1.6, label="actual AQI")
    ax.axhline(config.ALERT_THRESHOLD, color=ORANGE, linestyle=":", linewidth=1)
    ax.axvline(pd.Timestamp(config.LOCKDOWN_START), color=INK2, linestyle="--", linewidth=1)
    ax.set_ylabel("AQI")
    ax.set_title("Test period: forecast range vs actual AQI", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    p = config.FIG_DIR / "quantile_forecast_test.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(horizons["horizon"], horizons["model_MAE"], marker="o", color=BLUE, label="attention LSTM")
    ax.plot(horizons["horizon"], horizons["persistence_MAE"], marker="o", color=INK2,
            linestyle="--", label="persistence")
    ax.set_xticks(horizons["horizon"])
    ax.set_xlabel("Days ahead")
    ax.set_ylabel("Test MAE (AQI points)")
    ax.set_title("Error grows the further ahead we forecast", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    p = config.FIG_DIR / "horizon_mae.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)
    return out


def _print(r: dict) -> None:
    print("\nTEST period (scored once):")
    for _, t in r["table"].iterrows():
        extra = "" if t["model"] == "Persistence" else \
            f"   {t['diff']:+5.1f} [{t['lo']:+.1f}, {t['hi']:+.1f}]  {_verdict(t['lo'], t['hi'])}"
        print(f"  {t['model']:<33} MAE {t['MAE']:6.1f}  R2 {t['R2']:.3f}{extra}")
    iv = r["intervals"]
    print(f"\n{iv['nominal']}% forecast range: actual inside on {iv['cover_test'] * 100:.0f}% of test days "
          f"(mean width {iv['width_test']:.0f} AQI)")
    top = r["attention"].sort_values("all", ascending=False).head(3)
    print("Attention: most weight on " + ", ".join(
        f"{int(a['days_before'])}d before ({a['all'] * 100:.1f}%)" for _, a in top.iterrows()))
    print(f"\nTotal {r['seconds'] / 60:.1f} min. Report: {config.ADVANCED_MD}")


if __name__ == "__main__":
    run_advanced()
