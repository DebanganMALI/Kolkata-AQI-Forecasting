"""
Robustness and spike experiment:  python main.py robust

Two questions the single test period could not answer:

1. Does the model win in EVERY year, or only in 2025-2026?  -> walk-forward
   evaluation: the whole procedure repeated four times, each round training
   only on data before its validation year and testing on the year after it:
       round   train            validation   test
       1       2018-10 .. 2020  2021         2022
       2       2018-10 .. 2021  2022         2023
       3       2018-10 .. 2022  2023         2024
       4       2018-10 .. 2023  2024         2025-01 .. 2026-10

2. Can the weak point, sudden high-pollution days, be improved?
   * Spike-weighted GRU: in the loss, each training day counts in proportion
     to (its AQI / the average AQI) ** gamma, so polluted days matter more.
     gamma is chosen per round on that round's validation year, by the error
     on its most polluted 20% of days.
   * Ensemble: the average of GRU, LSTM, Transformer and XGBoost.
   * Alert probability: a GRU classifier that outputs the probability that
     tomorrow's AQI is above 200; its alert threshold is set on validation.

Model settings and feature sets are the ones already chosen on 2024 validation
(models/extended/selected.json); nothing here is tuned on any test year.
The 2025-2026 test period was already scored once by `extended`; round 4
scores it again, and the report says so.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import config, extended
from src.dataset import decode_target, encode_target, target_stats
from src.evaluate import (BLUE, INK2, ORANGE, alert_metrics, block_bootstrap_mae_diff,
                          regression_metrics, style_axes)
from src.extended_data import build_extended_table
from src.model import AQIGRU, pick_device
from src.train import set_seed


def _log(msg: str) -> None:
    config.ROBUST_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.ROBUST_DIR / "progress.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%H:%M:%S}  {msg}\n")


def load_choices() -> dict:
    """Settings chosen on 2024 validation by `python main.py extended`."""
    path = config.V2_DIR / "selected.json"
    if not path.exists():
        raise FileNotFoundError("Run `python main.py extended` first: its validation choices are reused here.")
    s = json.load(open(path))
    best = pd.DataFrame(s["per_feature_set_and_model"])
    per_family = best.loc[best.groupby("family")["mean"].idxmin()].set_index("family")
    return {"main": s["overall"],
            "ensemble": {f: per_family.loc[f, ["feature_set", "candidate"]].to_dict()
                         for f in config.ENSEMBLE_FAMILIES}}


def fold_table(table: pd.DataFrame, fold: dict) -> pd.DataFrame:
    """Same table, split relabelled for one round; days after its test period are dropped."""
    d = table["Date"]
    t = table[d <= pd.Timestamp(fold["test_end"])].copy()
    d = t["Date"]
    t["split"] = np.where(d <= pd.Timestamp(fold["train_end"]), "train",
                          np.where(d <= pd.Timestamp(fold["val_end"]), "val", "test"))
    return t


def candidate(family: str, name: str) -> dict:
    return next(c for c in extended.candidates(family) if c["name"] == name)


# ---------------------------------------------------------------------------
# GRU trained with sample weights (spike-weighted) or as a classifier
# ---------------------------------------------------------------------------
def train_gru(cand, seed, X_tr, t_tr, w_tr, X_va, t_va, w_va, device, classify=False):
    set_seed(seed)
    model = AQIGRU(X_tr.shape[2], hidden_size=cand["hidden_size"], num_layers=cand["num_layers"],
                   dropout=config.DROPOUT).to(device)
    gen = torch.Generator().manual_seed(seed)
    tensors = lambda X, t, w: TensorDataset(torch.from_numpy(X), torch.from_numpy(t.astype(np.float32)),  # noqa: E731
                                            torch.from_numpy(w.astype(np.float32)))
    tl = DataLoader(tensors(X_tr, t_tr, w_tr), batch_size=config.BATCH_SIZE, shuffle=True, generator=gen)
    vl = DataLoader(tensors(X_va, t_va, w_va), batch_size=config.BATCH_SIZE, shuffle=False)
    if classify:
        pos = max(float(t_tr.sum()), 1.0)
        bce = nn.BCEWithLogitsLoss(reduction="none",
                                   pos_weight=torch.tensor((len(t_tr) - pos) / pos, device=device))
        loss_fn = lambda p, t, w: (bce(p, t) * w).mean()        # noqa: E731
    else:
        loss_fn = lambda p, t, w: (w * (p - t) ** 2).mean()      # noqa: E731
    opt = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    best, state, bad = float("inf"), None, 0
    for _ in range(config.MAX_EPOCHS):
        model.train()
        for xb, tb, wb in tl:
            xb, tb, wb = xb.to(device), tb.to(device), wb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), tb, wb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            opt.step()
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, tb, wb in vl:
                xb, tb, wb = xb.to(device), tb.to(device), wb.to(device)
                total += loss_fn(model(xb), tb, wb).item() * len(tb)
                n += len(tb)
        if total / n < best - 1e-6:
            best, bad = total / n, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= config.PATIENCE:
            break
    model.load_state_dict(state)
    model.eval()

    def predict(X):
        with torch.no_grad():
            return model(torch.from_numpy(X).to(device)).cpu().numpy()
    return predict


def spike_weights(y, train_mean, gamma):
    w = (np.asarray(y, float) / train_mean) ** gamma
    return w / w.mean()


def high_days(y) -> np.ndarray:
    """The most polluted 20% of days in a period."""
    return y >= np.quantile(y, 1 - config.SPIKE_TOP_SHARE)


# ---------------------------------------------------------------------------
# One walk-forward round
# ---------------------------------------------------------------------------
def run_fold(table, fold, choices, device) -> dict:
    t = fold_table(table, fold)
    main = choices["main"]
    data = {fs: extended.build_data(t, fs) for fs in config.V2_FEATURE_SETS}
    w_main = data[main["feature_set"]][0]
    tr, va, te = (w_main.subset(s) for s in ("train", "val", "test"))
    out = {"persistence": te.last_aqi, "y": te.y_true, "dates": te.target_dates}

    def avg(family, fs, name, splits=("val", "test")):
        cand = candidate(family, name)
        runs = []
        for s in extended.seeds_for(family):
            runs.append(extended.fit(family, cand, s, data[fs], list(splits), device)[0])
            _log(f"{fold['name']} | {family} [{fs}] seed {s}")
        return {k: np.mean([r[k] for r in runs], axis=0) for k in splits}

    # main GRU and ensemble members
    gru = avg("GRU", main["feature_set"], main["candidate"])
    out["GRU"] = gru["test"]
    members = {"GRU": gru}
    for fam, c in choices["ensemble"].items():
        if fam != "GRU" or c["feature_set"] != main["feature_set"]:
            members[fam] = avg(fam, c["feature_set"], c["candidate"])
    out["Ensemble"] = np.mean([m["test"] for m in members.values()], axis=0)

    # spike-weighted GRU: gamma chosen on this round's validation high days
    cand = candidate("GRU", main["candidate"])
    mode = cand["target_mode"]
    stats = target_stats(tr, mode, data[main["feature_set"]][1])
    enc = lambda w: encode_target(w.y_true, w.last_aqi, mode, stats)    # noqa: E731
    mean_y = tr.y_true.mean()
    hv = high_days(va.y_true)
    best_g, best_err, best_pred = None, float("inf"), None
    for g in config.SPIKE_GAMMAS:
        vals, tests = [], []
        for s in config.V2_SEEDS:
            f = train_gru(cand, s, tr.X, enc(tr), spike_weights(tr.y_true, mean_y, g),
                          va.X, enc(va), spike_weights(va.y_true, mean_y, g), device)
            vals.append(decode_target(f(va.X), va.last_aqi, mode, stats))
            tests.append(decode_target(f(te.X), te.last_aqi, mode, stats))
            _log(f"{fold['name']} | spike GRU gamma {g} seed {s}")
        err = float(np.abs(np.mean(vals, axis=0) - va.y_true)[hv].mean())
        if err < best_err:
            best_g, best_err, best_pred = g, err, np.mean(tests, axis=0)
    out["Spike-weighted GRU"], out["gamma"] = best_pred, best_g

    # alert probability: P(tomorrow > 200), threshold chosen on validation F1
    lab = lambda w: (w.y_true > config.ALERT_THRESHOLD).astype(np.float32)  # noqa: E731
    ones = lambda w: np.ones(len(w), np.float32)                           # noqa: E731
    pv, pt = [], []
    for s in config.V2_SEEDS:
        f = train_gru(cand, s, tr.X, lab(tr), ones(tr), va.X, lab(va), ones(va), device, classify=True)
        pv.append(1 / (1 + np.exp(-f(va.X))))
        pt.append(1 / (1 + np.exp(-f(te.X))))
        _log(f"{fold['name']} | alert classifier seed {s}")
    pv, pt = np.mean(pv, axis=0), np.mean(pt, axis=0)
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    f1s = [np.nan_to_num(alert_metrics(va.y_true, pv * 1000, th * 1000)["F1"], nan=-1) for th in grid]
    out["prob"], out["prob_threshold"] = pt, float(grid[int(np.argmax(f1s))])
    return out


# ---------------------------------------------------------------------------
def run_robust(device_name: str = config.DEVICE, verbose: bool = True) -> dict:
    t0 = time.time()
    (config.ROBUST_DIR / "progress.log").unlink(missing_ok=True)
    device = pick_device(device_name)
    choices = load_choices()
    table, _ = build_extended_table(save=False)
    if verbose:
        print(f"Device: {device} | main model: {choices['main']['family']} [{choices['main']['feature_set']}] "
              f"| ensemble: {', '.join(choices['ensemble'])}")
    folds = []
    for fold in config.ROBUST_FOLDS:
        tf = time.time()
        r = run_fold(table, fold, choices, device)
        r["fold"] = fold
        folds.append(r)
        if verbose:
            y, p = r["y"], r["persistence"]
            print(f"  {fold['name']:<22} persistence {regression_metrics(y, p)['MAE']:5.1f} | "
                  f"GRU {regression_metrics(y, r['GRU'])['MAE']:5.1f} | "
                  f"spike GRU {regression_metrics(y, r['Spike-weighted GRU'])['MAE']:5.1f} (gamma {r['gamma']}) | "
                  f"ensemble {regression_metrics(y, r['Ensemble'])['MAE']:5.1f}  [{time.time() - tf:.0f}s]")
    res = _report(folds, choices, str(device), time.time() - t0)
    if verbose:
        print(f"\nTotal {res['seconds'] / 60:.1f} min. Report: {config.ROBUST_MD}")
    return res


MODELS = ["GRU", "Spike-weighted GRU", "Ensemble"]


def _md(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def _ci(y, a, b):
    d, lo, hi = block_bootstrap_mae_diff(y, a, b)
    mark = " (better)" if hi < 0 else " (worse)" if lo > 0 else ""
    return f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]{mark}", hi < 0


def _report(folds, choices, device, seconds) -> dict:
    acc_rows, spike_rows, alert_rows, wins = [], [], [], {m: 0 for m in MODELS}
    for r in folds:
        y, p, name = r["y"], r["persistence"], r["fold"]["name"]
        hi = high_days(y)
        row = [name, len(y), f"{regression_metrics(y, p)['MAE']:.1f}"]
        for m in MODELS:
            txt, win = _ci(y, r[m], p)
            wins[m] += int(win)
            row.append(f"{regression_metrics(y, r[m])['MAE']:.1f}, {txt}")
        acc_rows.append(row)
        spike_rows.append([name, int(hi.sum()), f"{np.abs(p - y)[hi].mean():.1f}"]
                          + [f"{np.abs(r[m] - y)[hi].mean():.1f}" for m in MODELS] + [r["gamma"]])
        a_p, a_g, a_e = (alert_metrics(y, x) for x in (p, r["GRU"], r["Ensemble"]))
        a_c = alert_metrics(y, r["prob"] * 1000, r["prob_threshold"] * 1000)
        brier = float(np.mean((r["prob"] - (y > config.ALERT_THRESHOLD)) ** 2))
        alert_rows.append([name, a_p["actual_high_days"]] +
                          [f"{a['F1']:.3f}" if not np.isnan(a["F1"]) else "n/a" for a in (a_p, a_g, a_e, a_c)] +
                          [f"{a_c['correct_alerts_TP']}/{a_c['actual_high_days']}", f"{brier:.3f}",
                           r["prob_threshold"]])

    y_all = np.concatenate([r["y"] for r in folds])
    pooled = {m: np.concatenate([r[m] for r in folds]) for m in MODELS + ["persistence"]}
    pooled_txt = {m: _ci(y_all, pooled[m], pooled["persistence"])[0] for m in MODELS}

    L = ["# Robustness and Spike Experiment", "",
         f"*Generated by `python main.py robust` on {datetime.now():%Y-%m-%d %H:%M} "
         f"({seconds / 60:.1f} min on `{device}`). All numbers come from this run.*", "",
         "## Set-up", "",
         "- **Walk-forward evaluation:** the full procedure repeated in 4 rounds; each round trains only "
         "on data before its validation year and is tested on the following period.",
         _md([[f["name"], f"{config.V2_START} to {f['train_end']}", f"to {f['val_end']}", f"to {f['test_end']}"]
              for f in config.ROBUST_FOLDS], ["round", "train", "validation", "test"]),
         "",
         f"- **Main model:** {choices['main']['family']} with `{choices['main']['feature_set']}` inputs "
         f"(`{choices['main']['candidate']}`), as chosen on 2024 validation by `python main.py extended`.",
         "- **Spike-weighted GRU:** each training day counts in proportion to (AQI / average AQI) ^ gamma, "
         f"gamma from {config.SPIKE_GAMMAS} chosen per round on that round's validation year by the error "
         f"on its most polluted {int(config.SPIKE_TOP_SHARE * 100)}% of days.",
         "- **Ensemble:** the average of " + ", ".join(f"{f} (`{c['feature_set']}`)" for f, c in choices["ensemble"].items()) + ".",
         f"- **Alert probability:** a GRU classifier for P(AQI > {config.ALERT_THRESHOLD}); the alert threshold "
         "is chosen on each round's validation year.",
         "- Settings come from 2024 validation, so rounds 1-3 test years the choices never saw "
         "(round 3 tests 2024, the year those settings were chosen on, so it slightly favours the models). "
         "Round 4 re-scores the 2025-2026 test period already reported by `extended`.",
         "",
         "## Does the model win every year? (test MAE, then difference vs persistence [95% interval])", "",
         _md(acc_rows, ["round", "test days", "persistence"] + MODELS), "",
         f"Rounds where each model is clearly better than persistence: "
         + ", ".join(f"{m} {wins[m]} of {len(folds)}" for m in MODELS) + ".", "",
         f"All {len(y_all)} test days pooled: " + "; ".join(f"{m} {pooled_txt[m]}" for m in MODELS) + ".", "",
         f"## The most polluted {int(config.SPIKE_TOP_SHARE * 100)}% of days (MAE on those days only)", "",
         _md(spike_rows, ["round", "days", "persistence"] + MODELS + ["gamma chosen"]), "",
         f"## High-pollution alerts (AQI > {config.ALERT_THRESHOLD}), F1 score", "",
         _md(alert_rows, ["round", "high days", "persistence", "GRU", "Ensemble", "alert probability",
                          "caught (probability)", "Brier score", "threshold (from validation)"]), "",
         "The Brier score measures how well the probabilities match reality (0 = perfect; lower is better). "
         "n/a = no high days, or no alerts raised, in that period.", "",
         "## How to read these results honestly", "",
         "- Each round has a different number of polluted days; early years were dirtier than recent ones.",
         "- Settings were chosen once, on 2024; they were not re-tuned per round.",
         "- Round 4 re-uses the 2025-2026 test period, which was already scored once.", "",
         "## Files", "", f"- `{config.ROBUST_PREDICTIONS_CSV.relative_to(config.PROJECT_ROOT).as_posix()}`"]

    rows = []
    for r in folds:
        for i in range(len(r["y"])):
            rows.append({"round": r["fold"]["name"], "date": str(pd.Timestamp(r["dates"][i]).date()),
                         "actual_AQI": r["y"][i], "persistence": r["persistence"][i],
                         **{m: round(float(r[m][i]), 2) for m in MODELS},
                         "P_above_200": round(float(r["prob"][i]), 3)})
    pd.DataFrame(rows).to_csv(config.ROBUST_PREDICTIONS_CSV, index=False)
    fig = _plot(folds)
    if fig:
        L.append(f"- `{fig.relative_to(config.PROJECT_ROOT).as_posix()}`")
    config.ROBUST_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    return {"folds": folds, "seconds": seconds}


def _plot(folds):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    names = [r["fold"]["name"] for r in folds]
    x = np.arange(len(folds))
    series = [("persistence", "Persistence", "#b9b7ae"), ("GRU", "GRU", BLUE),
              ("Spike-weighted GRU", "Spike-weighted GRU", ORANGE), ("Ensemble", "Ensemble", INK2)]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for k, (key, label, col) in enumerate(series):
        vals = [regression_metrics(r["y"], r[key])["MAE"] for r in folds]
        ax.bar(x + (k - 1.5) * 0.2, vals, width=0.19, color=col, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Test MAE (AQI points)")
    ax.set_title("Walk-forward: test error in every round", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style_axes(ax)
    fig.tight_layout()
    path = config.FIG_DIR / "robust_walk_forward.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    run_robust()
