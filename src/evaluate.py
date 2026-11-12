"""
Evaluate forecasts: LSTM variants vs the persistence baseline.

Metrics (all in AQI units unless stated):
  MAE   - mean absolute error: on average, how many AQI points off we are.
  RMSE  - root mean squared error: like MAE but punishes big misses more.
  MAPE  - mean absolute percentage error: average % off.
  R2    - share of the day-to-day variation explained (1 = perfect, 0 = no better
          than always guessing the average, negative = worse than that).
  Bias  - mean (prediction - actual): positive means we over-predict on average.

Alert metrics (alert = forecast above a threshold, compared with actual AQI > 200):
  Precision - of the alerts raised, the share that were real high-pollution days.
  Recall    - of the real high-pollution days, the share that got an alert.
  F1        - a single score balancing precision and recall.

Order of work (so the test period cannot influence any choice):
  1. The variant was already chosen in train.py from VALIDATION results.
  2. Alert thresholds are tuned on VALIDATION.
  3. Only then is everything scored on the TEST period, once.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from src import config
from src.baseline import persistence_predict
from src.dataset import Windows, decode_target, make_windows
from src.preprocessing import StandardScaler, load_processed, run_preprocessing

# Plot colours (validated palette: blue / orange series, neutral greys for text)
BLUE, ORANGE, INK, INK2, GRID, SURFACE, RED = (
    "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#d03b3b")
ACTUAL_HIGH = config.ALERT_THRESHOLD  # a "high-pollution day" is always actual AQI > 200


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = pred - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(np.mean(np.abs(err) / np.abs(y)) * 100) if np.all(y != 0) else float("nan"),
        "R2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "Bias": float(np.mean(err)),
    }


def alert_metrics(y: np.ndarray, pred: np.ndarray, threshold: float = config.ALERT_THRESHOLD) -> dict:
    """Alert when the forecast is above `threshold`; truth is always actual AQI > 200."""
    actual = np.asarray(y) > ACTUAL_HIGH
    raised = np.asarray(pred) > threshold
    tp = int(np.sum(actual & raised))
    fp = int(np.sum(~actual & raised))
    fn = int(np.sum(actual & ~raised))
    tn = int(np.sum(~actual & ~raised))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (tp + fp) and (tp + fn) and (precision + recall) > 0 else float("nan"))
    return {"actual_high_days": int(actual.sum()), "alerts_raised": int(raised.sum()),
            "correct_alerts_TP": tp, "false_alarms_FP": fp, "missed_FN": fn,
            "correct_quiet_TN": tn, "precision": precision, "recall": recall, "F1": f1}


def tune_threshold(y_val: np.ndarray, pred_val: np.ndarray) -> int:
    """Alert threshold with the best validation F1 (ties -> closest to 200)."""
    best_t, best_f1 = config.ALERT_THRESHOLD, -1.0
    for t in config.ALERT_THRESHOLD_GRID:
        f1 = alert_metrics(y_val, pred_val, t)["F1"]
        f1 = -1.0 if np.isnan(f1) else f1
        closer = abs(t - config.ALERT_THRESHOLD) < abs(best_t - config.ALERT_THRESHOLD)
        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and closer):
            best_t, best_f1 = t, f1
    return best_t


def block_bootstrap_mae_diff(y, p_a, p_b, resamples: int = config.BOOTSTRAP_RESAMPLES,
                             block: int = config.BOOTSTRAP_BLOCK_DAYS, seed: int = config.SEED):
    """
    95% interval for MAE(a) - MAE(b) on the same days. Days are resampled in
    blocks of consecutive days, because neighbouring days are not independent.
    If the interval contains 0, the data cannot tell the two forecasts apart.
    """
    y, p_a, p_b = (np.asarray(v, float) for v in (y, p_a, p_b))
    d = np.abs(p_a - y) - np.abs(p_b - y)   # per-day difference in absolute error
    n = len(d)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    stats = np.empty(resamples)
    for i in range(resamples):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        stats[i] = d[idx].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _windows(scaler: StandardScaler | None = None):
    if not config.PROCESSED_CSV.exists() or not config.SCALER_JSON.exists():
        run_preprocessing(verbose=False)
    clean = load_processed()
    scaler = scaler or StandardScaler.from_json(config.SCALER_JSON)
    return make_windows(clean, scaler), scaler


def _fmt_table(rows: list[dict], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                v = float(v)
                cells.append("n/a" if np.isnan(v) else
                             f"{v:.3f}" if c in ("R2", "precision", "recall", "F1") else f"{v:.1f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pm(values, digits: int = 1) -> str:
    v = np.asarray(values, float)
    f = f"{{:.{digits}f}}"
    return (f.format(v.mean()) + " ± " + f.format(v.std(ddof=1))) if len(v) > 1 else f.format(v.mean())


# ---------------------------------------------------------------------------
# Baseline-only evaluation (works without PyTorch)
# ---------------------------------------------------------------------------
def evaluate_baseline(verbose: bool = True) -> list[dict]:
    windows, _ = _windows()
    rows = []
    for split in ("val", "test"):
        w = windows.subset(split)
        p = persistence_predict(w)
        rows.append({"split": split, "model": "Persistence", "n": len(w),
                     **regression_metrics(w.y_true, p), **alert_metrics(w.y_true, p)})
    if verbose:
        print(_fmt_table(rows, ["split", "model", "n", "MAE", "RMSE", "MAPE_%", "R2", "Bias"]))
        print()
        print(_fmt_table(rows, ["split", "model", "actual_high_days", "alerts_raised",
                                "correct_alerts_TP", "false_alarms_FP", "missed_FN",
                                "precision", "recall", "F1"]))
    return rows


# ---------------------------------------------------------------------------
# LSTM ensembles
# ---------------------------------------------------------------------------
def load_variant(variant: str, device):
    """Load the trained model for every seed of one variant."""
    from src.model import load_checkpoint
    from src.train import checkpoint_path

    out = []
    for seed in config.SEEDS:
        path = checkpoint_path(variant, seed)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run `python main.py train` first.")
        out.append(load_checkpoint(path, device=str(device)))
    return out


def predict_variant(models, w: Windows, device) -> np.ndarray:
    """Real-AQI forecasts, one row per seed: shape (seeds, samples)."""
    import torch
    preds = []
    for model, ckpt in models:
        model.eval()
        with torch.no_grad():
            z = model(torch.from_numpy(w.X).to(device)).cpu().numpy()
        preds.append(decode_target(z, w.last_aqi, ckpt["target_mode"],
                                   (ckpt["target_mean"], ckpt["target_std"])))
    return np.vstack(preds)


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------
def evaluate(device_name: str = "cpu", verbose: bool = True) -> dict:
    from src.model import pick_device

    if not config.SELECTED_JSON.exists():
        raise FileNotFoundError("No trained models found. Run `python main.py train` first.")
    selection = json.loads(config.SELECTED_JSON.read_text())
    selected = selection["selected_variant"]
    device = pick_device(device_name)

    all_models = {v: load_variant(v, device) for v in config.VARIANTS}
    ckpt0 = all_models[selected][0][1]
    scaler = StandardScaler(**ckpt0["scaler"])  # exactly the scaler used in training
    windows, _ = _windows(scaler)
    w_val, w_test = windows.subset("val"), windows.subset("test")

    # Forecasts for every variant (per seed) and the 5-seed average ("ensemble")
    P = {v: {s: predict_variant(m, w, device) for s, w in (("val", w_val), ("test", w_test))}
         for v, m in all_models.items()}
    pers = {"val": persistence_predict(w_val), "test": persistence_predict(w_test)}
    y = {"val": w_val.y_true, "test": w_test.y_true}
    ens = {s: P[selected][s].mean(axis=0) for s in ("val", "test")}

    # --- Table A: validation, all variants (this is what selection used)
    tA = []
    for v in config.VARIANTS:
        per_seed = [regression_metrics(y["val"], p) for p in P[v]["val"]]
        tA.append({"variant": v + (" (selected)" if v == selected else ""),
                   "target": config.VARIANTS[v]["target_mode"],
                   "size": f"{config.VARIANTS[v]['num_layers']}x{config.VARIANTS[v]['hidden_size']}",
                   "MAE (mean ± sd over seeds)": _pm([m["MAE"] for m in per_seed]),
                   "RMSE": _pm([m["RMSE"] for m in per_seed]),
                   "Bias": _pm([m["Bias"] for m in per_seed])})
    pv = regression_metrics(y["val"], pers["val"])
    tA.append({"variant": "Persistence", "target": "-", "size": "-",
               "MAE (mean ± sd over seeds)": f"{pv['MAE']:.1f}", "RMSE": f"{pv['RMSE']:.1f}",
               "Bias": f"{pv['Bias']:+.1f}"})

    # --- Alert thresholds tuned on VALIDATION (before looking at test)
    thr_lstm = tune_threshold(y["val"], ens["val"])
    thr_pers = tune_threshold(y["val"], pers["val"])

    # ===================== TEST (scored once, after all choices) =====================
    single = [regression_metrics(y["test"], p) for p in P[selected]["test"]]
    m_ens = regression_metrics(y["test"], ens["test"])
    m_pers = regression_metrics(y["test"], pers["test"])
    tB = [
        {"forecast": f"LSTM {selected}: single runs (mean ± sd of 5)",
         **{k: _pm([m[k] for m in single], 3 if k == "R2" else 1)
            for k in ("MAE", "RMSE", "MAPE_%", "R2", "Bias")}},
        {"forecast": f"LSTM {selected}: average of 5 runs",
         **{k: f"{m_ens[k]:.3f}" if k == "R2" else f"{m_ens[k]:.1f}" for k in ("MAE", "RMSE", "MAPE_%", "R2", "Bias")}},
        {"forecast": "Persistence",
         **{k: f"{m_pers[k]:.3f}" if k == "R2" else f"{m_pers[k]:.1f}" for k in ("MAE", "RMSE", "MAPE_%", "R2", "Bias")}},
    ]
    diff, lo, hi = block_bootstrap_mae_diff(y["test"], ens["test"], pers["test"])

    tD = []  # transparency: test for every variant (NOT used for any choice)
    for v in config.VARIANTS:
        e = regression_metrics(y["test"], P[v]["test"].mean(axis=0))
        tD.append({"variant": v + (" (selected)" if v == selected else ""),
                   "test MAE, average of 5 runs": f"{e['MAE']:.1f}",
                   "test MAE, single runs": _pm([regression_metrics(y["test"], p)["MAE"] for p in P[v]["test"]]),
                   "test Bias": f"{e['Bias']:+.1f}"})

    tE = []  # alerts at the project threshold (200)
    for split in ("val", "test"):
        for name, p in ((f"LSTM {selected} (avg of 5)", ens[split]), ("Persistence", pers[split])):
            tE.append({"split": split, "model": name, **alert_metrics(y[split], p)})

    tF = []  # alerts with validation-tuned thresholds, scored on test
    for name, p_val, p_test, t in ((f"LSTM {selected} (avg of 5)", ens["val"], ens["test"], thr_lstm),
                                   ("Persistence", pers["val"], pers["test"], thr_pers)):
        a = alert_metrics(y["test"], p_test, t)
        tF.append({"model": name, "threshold (from validation)": t,
                   "validation F1": alert_metrics(y["val"], p_val, t)["F1"], **a})

    # --- Error by month / lockdown on test
    dates = pd.to_datetime(w_test.target_dates)
    df = pd.DataFrame({"date": dates, "actual": y["test"], "lstm": ens["test"], "pers": pers["test"]})
    df["month"] = df["date"].dt.strftime("%Y-%m")
    df["period"] = np.where(df["date"] < pd.Timestamp(config.LOCKDOWN_START),
                            "before 2020-03-25", "from 2020-03-25 (lockdown)")
    tG = []
    for key in ("month", "period"):
        for g, d in df.groupby(key, sort=True):
            tG.append({"group": g, "days": len(d), "mean actual AQI": d["actual"].mean(),
                       "LSTM MAE": (d["lstm"] - d["actual"]).abs().mean(),
                       "LSTM bias": (d["lstm"] - d["actual"]).mean(),
                       "Persistence MAE": (d["pers"] - d["actual"]).abs().mean(),
                       "Persistence bias": (d["pers"] - d["actual"]).mean()})

    # --- Save predictions and figures
    out = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "actual_AQI": y["test"],
        "lstm_avg_pred": np.round(ens["test"], 1),
        "lstm_min_of_5": np.round(P[selected]["test"].min(axis=0), 1),
        "lstm_max_of_5": np.round(P[selected]["test"].max(axis=0), 1),
        "persistence_pred": pers["test"],
        "actual_high": y["test"] > ACTUAL_HIGH,
        "lstm_alert_200": ens["test"] > config.ALERT_THRESHOLD,
        "persistence_alert_200": pers["test"] > config.ALERT_THRESHOLD,
        f"lstm_alert_tuned_{thr_lstm}": ens["test"] > thr_lstm,
        f"persistence_alert_tuned_{thr_pers}": pers["test"] > thr_pers,
    })
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.PREDICTIONS_CSV, index=False)
    figs = _plot_results(w_test, P[selected]["test"], ens["test"], pers["test"], selected)

    res = dict(selected=selected, selection=selection, tA=tA, tB=tB, tD=tD, tE=tE, tF=tF, tG=tG,
               m_ens=m_ens, m_pers=m_pers, single=single, diff=diff, lo=lo, hi=hi,
               thr_lstm=thr_lstm, thr_pers=thr_pers, ckpt0=ckpt0, figs=figs, n_test=len(w_test))
    _write_results(res)

    if verbose:
        print("Validation (used to choose the variant):")
        print(_fmt_table(tA, list(tA[0].keys())))
        print(f"\nTEST period ({len(w_test)} days) - scored once:")
        print(_fmt_table(tB, list(tB[0].keys())))
        print(f"\nTest MAE difference, LSTM avg - persistence: {diff:+.1f} "
              f"(95% interval {lo:+.1f} to {hi:+.1f})")
        print(f"\nAlerts at AQI > {config.ALERT_THRESHOLD}:")
        print(_fmt_table(tE, ["split", "model", "actual_high_days", "alerts_raised",
                              "correct_alerts_TP", "false_alarms_FP", "missed_FN",
                              "precision", "recall", "F1"]))
        print("\nAlerts with thresholds chosen on validation, scored on test:")
        print(_fmt_table(tF, ["model", "threshold (from validation)", "alerts_raised",
                              "correct_alerts_TP", "false_alarms_FP", "missed_FN",
                              "precision", "recall", "F1"]))
        print("\nTest error by month and period (LSTM = average of 5 runs):")
        print(_fmt_table(tG, list(tG[0].keys())))
        print(f"\nResults report  : {config.RESULTS_MD}")
        print(f"Test predictions: {config.PREDICTIONS_CSV}")
    return res


def _write_results(r: dict) -> None:
    ck, sel, m_ens, m_pers = r["ckpt0"], r["selected"], r["m_ens"], r["m_pers"]
    better = m_ens["MAE"] < m_pers["MAE"]
    ci_has_zero = r["lo"] <= 0 <= r["hi"]
    if ci_has_zero:
        verdict = ("The 95% interval includes 0, so on this test period the data **cannot "
                   "distinguish** the LSTM from persistence.")
    elif r["hi"] < 0:
        verdict = "The whole 95% interval is below 0: the LSTM's lower error is unlikely to be luck on this test period."
    else:
        verdict = "The whole 95% interval is above 0: persistence is more accurate on this test period, by more than day-to-day luck explains."

    tE_cols = ["split", "model", "actual_high_days", "alerts_raised", "correct_alerts_TP",
               "false_alarms_FP", "missed_FN", "precision", "recall", "F1"]
    tF_cols = ["model", "threshold (from validation)", "validation F1", "alerts_raised",
               "correct_alerts_TP", "false_alarms_FP", "missed_FN", "precision", "recall", "F1"]
    L = [
        "# Forecasting Results: LSTM vs Persistence",
        "",
        f"*Generated by `python main.py evaluate` on {datetime.now():%Y-%m-%d %H:%M}. "
        "All numbers come from this run.*",
        "",
        "## How the experiment was run",
        "",
        f"- Data: Kolkata, {ck['start_date']} to {config.END_DATE}; train to {ck['train_end']}, "
        f"validation to {ck['val_end']}, test after (see `reports/preprocessing_decisions.md`).",
        f"- Inputs: previous {ck['lookback']} days of {', '.join(ck['feature_columns'])}.",
        f"- {len(config.VARIANTS)} LSTM variants x {len(config.SEEDS)} random seeds "
        f"({', '.join(map(str, config.SEEDS))}). Variants: predict tomorrow's AQI directly "
        "(*level*) or predict the change from today (*change*); 2 layers x 64 units (*large*) "
        "or 1 layer x 32 units (*small*).",
        "- The variant was chosen by **lowest mean validation MAE**; alert thresholds were "
        "tuned on **validation**. The test period was scored once, after those choices.",
        "- Honesty note: test results of the very first model (level_large, seed 42) had "
        "already been seen before this experiment was designed. The design above was fixed "
        "before any of its own results were seen.",
        "",
        "## 1. Validation results (used for selection)",
        "",
        _fmt_table(r["tA"], list(r["tA"][0].keys())),
        "",
        f"**Selected: {sel}.**",
        "",
        "## 2. Test results (scored once)",
        "",
        _fmt_table(r["tB"], list(r["tB"][0].keys())),
        "",
        f"- Test MAE: LSTM (average of 5 runs) {m_ens['MAE']:.1f} vs persistence "
        f"{m_pers['MAE']:.1f} - the LSTM is {'lower (better)' if better else 'higher (worse)'} "
        f"by {abs(m_ens['MAE'] - m_pers['MAE']):.1f} AQI points.",
        f"- Difference in MAE (LSTM - persistence): {r['diff']:+.1f}, 95% block-bootstrap "
        f"interval {r['lo']:+.1f} to {r['hi']:+.1f}. {verdict}",
        "",
        "Test MAE for every variant (shown for transparency only - **not** used to choose):",
        "",
        _fmt_table(r["tD"], list(r["tD"][0].keys())),
        "",
        f"## 3. High-pollution alerts (actual high day = AQI > {ACTUAL_HIGH})",
        "",
        f"At the project threshold (alert if forecast > {config.ALERT_THRESHOLD}):",
        "",
        _fmt_table(r["tE"], tE_cols),
        "",
        "Fair comparison - each method's alert threshold chosen on validation, then scored on test:",
        "",
        _fmt_table(r["tF"], tF_cols),
        "",
        "## 4. Where the errors happen (test period)",
        "",
        _fmt_table(r["tG"], list(r["tG"][0].keys())),
        "",
        "## How to read these results honestly",
        "",
        f"- The test period is {r['n_test']} days from one half-year, including the COVID-19 "
        "lockdown (from 2020-03-25, external fact), when pollution patterns were unusual.",
        "- The 95% interval accounts for day-to-day luck in the test days, not for how the "
        "model would do in other years.",
        "- These are associations learned from past data; they do not show what *causes* "
        "pollution.",
        "",
        "## Files",
        "",
        "- `models/validation_runs.csv` - validation score of every (variant, seed)",
        "- `models/selected_model.json` - the selection and the rule used",
        f"- `{config.PREDICTIONS_CSV.relative_to(config.PROJECT_ROOT).as_posix()}` - day-by-day test forecasts",
        *[f"- `{p.relative_to(config.PROJECT_ROOT).as_posix()}`" for p in r["figs"]],
        "",
    ]
    config.RESULTS_MD.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _plot_results(w: Windows, per_seed: np.ndarray, p_lstm: np.ndarray,
                  p_pers: np.ndarray, variant: str) -> list:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("NOTE: matplotlib not installed - result figures skipped.")
        return []
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    dates = pd.to_datetime(w.target_dates)

    # 1. Actual vs predicted over the test period
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(dates, per_seed.min(axis=0), per_seed.max(axis=0), color=BLUE, alpha=0.15,
                    linewidth=0, label="LSTM range over 5 runs")
    ax.plot(dates, w.y_true, color=INK, linewidth=2, label="Actual AQI")
    ax.plot(dates, p_lstm, color=BLUE, linewidth=2, label=f"LSTM {variant} (avg of 5)")
    ax.plot(dates, p_pers, color=ORANGE, linewidth=1.2, linestyle="--", label="Persistence")
    ax.axhline(config.ALERT_THRESHOLD, color=RED, linewidth=1, linestyle=":")
    ax.text(dates[-1], config.ALERT_THRESHOLD + 6, f"Alert threshold ({config.ALERT_THRESHOLD})",
            color=INK2, fontsize=9, ha="right", va="bottom",
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.axvline(pd.Timestamp(config.LOCKDOWN_START), color=INK2, linewidth=1, linestyle="--")
    ax.text(pd.Timestamp(config.LOCKDOWN_START), 0.97, "  lockdown starts", color=INK2,
            fontsize=9, va="top", transform=ax.get_xaxis_transform())
    ax.set_ylabel("AQI")
    ax.set_title("Test period: next-day AQI forecasts vs actual", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    p = config.FIG_DIR / "test_forecast_vs_actual.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 2. Predicted vs actual scatter (two panels, same axes)
    lim = (0, max(w.y_true.max(), p_lstm.max(), p_pers.max()) * 1.05)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    for ax, pred, name, col in ((axes[0], p_lstm, f"LSTM {variant} (avg of 5)", BLUE),
                                (axes[1], p_pers, "Persistence", ORANGE)):
        ax.scatter(w.y_true, pred, s=16, color=col, alpha=0.75, edgecolors="none")
        ax.plot(lim, lim, color=INK2, linewidth=1, linestyle="--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Actual AQI")
        ax.set_title(name, loc="left", fontsize=11)
        style_axes(ax)
    axes[0].set_ylabel("Predicted AQI")
    fig.suptitle("Test period: predicted vs actual (dashed line = perfect forecast)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    p = config.FIG_DIR / "test_predicted_vs_actual.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    hist_fig = config.FIG_DIR / "lstm_training_curve.png"
    if hist_fig.exists():
        paths.append(hist_fig)
    return paths


if __name__ == "__main__":
    evaluate()
