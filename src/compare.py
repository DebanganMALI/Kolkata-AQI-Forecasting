"""
Model comparison: which forecaster works best, and does weather help?

Every model forecasts the same thing as the main LSTM (tomorrow's AQI from the
previous 14 days), for the same days, with the same date split.

  Feature sets:
    "pollution"          past AQI, 11 pollutants, day of year (as in the main LSTM)
    "pollution+weather"  the same plus 13 daily weather inputs (Open-Meteo)

  Forecasters:
    Persistence      tomorrow = today (the baseline every model must beat)
    Linear (ridge)   linear regression with a penalty on large weights
    Random forest    average of many decision trees
    XGBoost          gradient-boosted decision trees
    GRU, LSTM        recurrent neural networks reading the 14 days in order

Rules (the same as the rest of the project):
  1. Each forecaster has a small, fixed list of settings in config.py, written
     down before any comparison was run.
  2. For every forecaster and feature set, the setting with the lowest mean
     VALIDATION MAE is chosen. Models with randomness are trained with 5 seeds
     and their forecasts averaged. The test period is not used here.
  3. Only after every choice is saved are the chosen models trained again
     (same seeds, so the same models) and asked for TEST forecasts. The test
     period is scored once.
  4. Inputs only come from days BEFORE the forecast day. The weather of the
     forecast day itself is never used.

Run:  python main.py compare
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

from src import config
from src.dataset import Windows, decode_target, encode_target, make_windows, target_stats
from src.evaluate import (BLUE, INK2, ORANGE, alert_metrics, block_bootstrap_mae_diff,
                          regression_metrics, style_axes)
from src.preprocessing import StandardScaler, feature_columns, load_processed, run_preprocessing
from src.weather import add_weather, load_weather

FAMILIES = ["Linear (ridge)", "Random forest", "XGBoost", "GRU", "LSTM"]
NEURAL = ("GRU", "LSTM")
FEATURE_SETS = {"pollution": False, "pollution+weather": True}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def feature_set_columns(with_weather: bool) -> list[str]:
    """Input columns for a feature set, in a fixed order."""
    return feature_columns() + (list(config.WEATHER_FEATURES) if with_weather else [])


def build_windows(with_weather: bool) -> tuple[Windows, StandardScaler]:
    """14-day samples for one feature set; the scaler is fitted on training rows only."""
    if not config.PROCESSED_CSV.exists():
        run_preprocessing(verbose=False)
    clean = load_processed()
    if with_weather:
        clean = add_weather(clean, load_weather())
    scaler = StandardScaler.fit(clean[clean["split"] == "train"], feature_set_columns(with_weather))
    return make_windows(clean, scaler), scaler


def tabular_features(X: np.ndarray, recent: int = config.TABULAR_RECENT_DAYS) -> np.ndarray:
    """
    The non-neural models need one flat row per sample instead of a 14-day grid.
    Each row holds:
      * every input on the last day before the forecast ("today"),
      * AQI on each of the last `recent` days (how pollution has been moving),
      * the average of every input over the last `recent` days.
    All of it comes from the window, so it is still only past information.
    """
    aqi = feature_columns().index(config.TARGET)
    return np.hstack([X[:, -1, :], X[:, -recent:, aqi], X[:, -recent:, :].mean(axis=1)])


def tabular_feature_names(columns: list[str], recent: int = config.TABULAR_RECENT_DAYS) -> list[str]:
    """Names matching tabular_features(), used for the XGBoost importance table."""
    lags = [f"AQI t-{recent - 1 - i}" for i in range(recent)]
    return [f"today:{c}" for c in columns] + lags + [f"mean{recent}:{c}" for c in columns]


# ---------------------------------------------------------------------------
# Candidate settings (fixed in config.py before running)
# ---------------------------------------------------------------------------
def candidates(family: str) -> list[dict]:
    if family == "Linear (ridge)":
        return [{"name": f"{m}, alpha={a}", "target_mode": m, "alpha": a}
                for m in config.TABULAR_TARGET_MODES for a in config.RIDGE_ALPHAS]
    if family == "Random forest":
        return [{"name": f"{m}, depth={p['max_depth']}, leaf={p['min_samples_leaf']}",
                 "target_mode": m, **p}
                for m in config.TABULAR_TARGET_MODES for p in config.RF_GRID]
    if family == "XGBoost":
        return [{"name": f"{m}, depth={p['max_depth']}", "target_mode": m, **p}
                for m in config.TABULAR_TARGET_MODES for p in config.XGB_GRID]
    if family in NEURAL:
        return [{"name": name, **spec} for name, spec in config.VARIANTS.items()]
    raise ValueError(f"Unknown family {family!r}")


def seeds_for(family: str) -> list[int]:
    """Ridge regression has no randomness, so one run is enough."""
    return [config.SEED] if family == "Linear (ridge)" else list(config.SEEDS)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _train_sequence(family, cand, seed, X_tr, y_tr, X_va, y_va):
    """Same training loop as src/train.py (Adam, early stopping on validation loss)."""
    import torch
    from torch import nn

    from src.dataset import to_loader
    from src.model import AQIGRU, AQILSTM
    from src.train import set_seed

    set_seed(seed)
    cls = AQILSTM if family == "LSTM" else AQIGRU
    model = cls(n_features=X_tr.shape[2], hidden_size=cand["hidden_size"],
                num_layers=cand["num_layers"], dropout=config.DROPOUT)
    train_loader = to_loader(X_tr, y_tr, config.BATCH_SIZE, shuffle=True, seed=seed)
    val_loader = to_loader(X_va, y_va, config.BATCH_SIZE, shuffle=False, seed=seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_loss, best_state, bad_epochs = float("inf"), None, 0
    for _ in range(config.MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimiser.step()
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                total += loss_fn(model(xb), yb).item() * len(yb)
                n += len(yb)
        val_loss = total / n
        if val_loss < best_loss - 1e-6:
            best_loss, bad_epochs = val_loss, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad_epochs += 1
        if bad_epochs >= config.PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()

    def predict(X):
        with torch.no_grad():
            return model(torch.from_numpy(X)).numpy()
    return model, predict


def _fit_tabular(family, cand, seed, X_tr, y_tr, X_va, y_va):
    if family == "Linear (ridge)":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=cand["alpha"]).fit(X_tr, y_tr)
        return model, model.predict
    if family == "Random forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=config.RF_TREES, max_depth=cand["max_depth"],
                                      min_samples_leaf=cand["min_samples_leaf"],
                                      random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
        return model, model.predict
    if family == "XGBoost":
        from xgboost import XGBRegressor
        model = XGBRegressor(**config.XGB_PARAMS, max_depth=cand["max_depth"], random_state=seed,
                             n_jobs=1, early_stopping_rounds=config.XGB_EARLY_STOPPING)
        # validation is used to decide when to stop adding trees (like early stopping for the nets)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return model, lambda X: model.predict(X, iteration_range=(0, model.best_iteration + 1))
    raise ValueError(family)


def fit_predict(family: str, cand: dict, seed: int, data, splits: list[str]):
    """Train one model and return its real-AQI forecasts for the requested splits."""
    windows, scaler = data
    w_tr, w_va = windows.subset("train"), windows.subset("val")
    mode = cand["target_mode"]
    stats = target_stats(w_tr, mode, scaler)                # learned from training samples only
    y_tr = encode_target(w_tr.y_true, w_tr.last_aqi, mode, stats)
    y_va = encode_target(w_va.y_true, w_va.last_aqi, mode, stats)

    if family in NEURAL:
        inputs = lambda w: w.X                               # noqa: E731
        model, predict = _train_sequence(family, cand, seed, w_tr.X, y_tr, w_va.X, y_va)
    else:
        inputs = lambda w: tabular_features(w.X)             # noqa: E731
        model, predict = _fit_tabular(family, cand, seed, inputs(w_tr), y_tr, inputs(w_va), y_va)

    out = {}
    for s in splits:
        w = windows.subset(s)
        out[s] = decode_target(predict(inputs(w)), w.last_aqi, mode, stats)
    return out, model


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_comparison(verbose: bool = True) -> dict:
    t0 = time.time()
    data = {fs: build_windows(w) for fs, w in FEATURE_SETS.items()}
    ref = data["pollution"][0]
    if not np.array_equal(ref.target_dates, data["pollution+weather"][0].target_dates):
        raise RuntimeError("The two feature sets do not cover the same forecast days.")

    n_models = sum(len(candidates(f)) * len(seeds_for(f)) for f in FAMILIES) * len(FEATURE_SETS)
    if verbose:
        print(f"Forecast days: train {len(ref.subset('train'))} | val {len(ref.subset('val'))} | "
              f"test {len(ref.subset('test'))}")
        print(f"Phase 1 - choosing settings on VALIDATION ({n_models} models, CPU)...\n")

    # ---- Phase 1: every candidate, validation only ------------------------
    rows = []
    for fs, d in data.items():
        y_val = d[0].subset("val").y_true
        for family in FAMILIES:
            tf = time.time()
            for cand in candidates(family):
                for seed in seeds_for(family):
                    preds, _ = fit_predict(family, cand, seed, d, ["val"])
                    m = regression_metrics(y_val, preds["val"])
                    rows.append({"feature_set": fs, "family": family, "candidate": cand["name"],
                                 "seed": seed, "val_MAE": m["MAE"], "val_RMSE": m["RMSE"],
                                 "val_Bias": m["Bias"]})
            if verbose:
                fam_rows = pd.DataFrame([r for r in rows if r["feature_set"] == fs
                                         and r["family"] == family])
                best = fam_rows.groupby("candidate")["val_MAE"].mean()
                print(f"  {fs:<18} {family:<15} best val MAE {best.min():6.1f} "
                      f"({best.idxmin()})  [{time.time() - tf:.0f}s]")

    runs = pd.DataFrame(rows)
    config.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.COMPARISON_DIR / "validation_runs.csv", index=False)

    summary = (runs.groupby(["feature_set", "family", "candidate"], sort=False)
               .agg(val_MAE_mean=("val_MAE", "mean"), val_MAE_sd=("val_MAE", "std"),
                    val_RMSE_mean=("val_RMSE", "mean"), val_Bias_mean=("val_Bias", "mean"),
                    seeds=("seed", "count"))
               .reset_index())
    selected = summary.loc[summary.groupby(["feature_set", "family"], sort=False)["val_MAE_mean"]
                           .idxmin()].reset_index(drop=True)
    overall = selected.loc[selected["val_MAE_mean"].idxmin()]

    # Save every choice BEFORE any test forecast is made.
    with open(config.COMPARISON_DIR / "selected.json", "w") as f:
        json.dump({
            "rule": "per forecaster and feature set: lowest mean validation MAE over seeds; "
                    "test period not used",
            "selected": selected[["feature_set", "family", "candidate", "val_MAE_mean"]]
            .round(3).to_dict(orient="records"),
            "overall_choice_on_validation": {"feature_set": overall["feature_set"],
                                             "family": overall["family"],
                                             "candidate": overall["candidate"]},
            "decided_at": datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2)

    # ---- Phase 2: chosen settings only, test scored once -------------------
    if verbose:
        print("\nPhase 2 - settings fixed; re-training the chosen models for the TEST forecasts...")
    final = {}
    for _, r in selected.iterrows():
        fs, family = r["feature_set"], r["family"]
        cand = next(c for c in candidates(family) if c["name"] == r["candidate"])
        val_runs, test_runs, models = [], [], []
        for seed in seeds_for(family):
            preds, model = fit_predict(family, cand, seed, data[fs], ["val", "test"])
            val_runs.append(preds["val"])
            test_runs.append(preds["test"])
            models.append(model)
        final[(fs, family)] = {"candidate": cand["name"], "val": np.mean(val_runs, axis=0),
                               "test": np.mean(test_runs, axis=0), "models": models}

    result = _score_and_report(data, selected, overall, final, time.time() - t0)
    if verbose:
        _print_summary(result)
    return result


# ---------------------------------------------------------------------------
# Scoring and report
# ---------------------------------------------------------------------------
def _score_and_report(data, selected, overall, final, seconds: float) -> dict:
    w_val = data["pollution"][0].subset("val")
    w_te = data["pollution"][0].subset("test")
    y, persistence = w_te.y_true, w_te.last_aqi.copy()
    lockdown = pd.to_datetime(w_te.target_dates) >= pd.Timestamp(config.LOCKDOWN_START)

    def score(pred) -> dict:
        m = regression_metrics(y, pred)
        a = alert_metrics(y, pred, config.ALERT_THRESHOLD)
        return {**m, "alert_F1": a["F1"], "alert_recall": a["recall"],
                "alert_precision": a["precision"],
                "MAE_before_lockdown": float(np.mean(np.abs(pred - y)[~lockdown])),
                "MAE_lockdown": float(np.mean(np.abs(pred - y)[lockdown]))}

    test_rows = [{"feature_set": "-", "family": "Persistence", "candidate": "-",
                  "val_MAE": regression_metrics(w_val.y_true, w_val.last_aqi)["MAE"],
                  **score(persistence), "diff": 0.0, "lo": 0.0, "hi": 0.0}]
    for (fs, family), f in final.items():
        d, lo, hi = block_bootstrap_mae_diff(y, f["test"], persistence)
        row = selected[(selected["feature_set"] == fs) & (selected["family"] == family)].iloc[0]
        test_rows.append({"feature_set": fs, "family": family, "candidate": f["candidate"],
                          "val_MAE": row["val_MAE_mean"], **score(f["test"]),
                          "diff": d, "lo": lo, "hi": hi})
    test = pd.DataFrame(test_rows)

    weather_rows = []
    for family in FAMILIES:
        a, b = final[("pollution+weather", family)]["test"], final[("pollution", family)]["test"]
        d, lo, hi = block_bootstrap_mae_diff(y, a, b)
        weather_rows.append({"family": family,
                             "val_without": final[("pollution", family)]["val"],
                             "val_with": final[("pollution+weather", family)]["val"],
                             "test_without": regression_metrics(y, b)["MAE"],
                             "test_with": regression_metrics(y, a)["MAE"],
                             "diff": d, "lo": lo, "hi": hi})
    for r in weather_rows:
        r["val_without"] = regression_metrics(w_val.y_true, r["val_without"])["MAE"]
        r["val_with"] = regression_metrics(w_val.y_true, r["val_with"])["MAE"]
    weather = pd.DataFrame(weather_rows)

    importance = _xgb_importance(final)

    preds = pd.DataFrame({"date": pd.to_datetime(w_te.target_dates).strftime("%Y-%m-%d"),
                          "actual_AQI": y, "persistence": persistence})
    for (fs, family), f in final.items():
        preds[f"{family} [{fs}]"] = np.round(f["test"], 2)
    preds.to_csv(config.COMPARISON_PREDICTIONS_CSV, index=False)

    r = {"test": test, "weather": weather, "selected": selected, "overall": overall,
         "importance": importance, "seconds": seconds, "n_test": len(y)}
    fig = _plot(test)
    _write_report(r, fig)
    return r


def _xgb_importance(final) -> pd.DataFrame:
    """XGBoost (with weather): share of total gain per input variable, averaged over seeds."""
    models = final[("pollution+weather", "XGBoost")]["models"]
    names = tabular_feature_names(feature_set_columns(True))
    gain = np.mean([m.feature_importances_ for m in models], axis=0)
    base = [n.split(":", 1)[1] if ":" in n else config.TARGET for n in names]
    s = pd.Series(gain, index=base).groupby(level=0).sum()
    s = (s / s.sum() * 100).sort_values(ascending=False)
    kind = {c: ("weather" if c in config.WEATHER_FEATURES else
                "calendar" if c.startswith("doy") else
                "past AQI" if c == config.TARGET else "pollutant") for c in s.index}
    return pd.DataFrame({"variable": s.index, "type": [kind[c] for c in s.index],
                         "share_%": s.values})


def _verdict(lo: float, hi: float, better: str, worse: str) -> str:
    if hi < 0:
        return better
    if lo > 0:
        return worse
    return "no clear difference"


def _md(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _write_report(r: dict, fig) -> None:
    test, weather, selected, overall = r["test"], r["weather"], r["selected"], r["overall"]
    pers = test[test["family"] == "Persistence"].iloc[0]
    ov = test[(test["feature_set"] == overall["feature_set"]) &
              (test["family"] == overall["family"])].iloc[0]

    L = ["# Model Comparison: Which Forecaster Works Best, and Does Weather Help?", "",
         f"*Generated by `python main.py compare` on {datetime.now():%Y-%m-%d %H:%M} "
         f"({r['seconds'] / 60:.1f} min on CPU). All numbers come from this run.*", "",
         "## How the comparison was run", "",
         "- **Task:** forecast tomorrow's AQI for Kolkata from the previous "
         f"{config.LOOKBACK} days, the same task and the same days as the main LSTM.",
         f"- **Split by date:** train to {config.TRAIN_END}, validation to {config.VAL_END}, "
         "test after that.",
         "- **Feature sets:** `pollution` (past AQI, 11 pollutants, day of year) and "
         f"`pollution+weather` (the same plus {len(config.WEATHER_FEATURES)} daily weather "
         "inputs from Open-Meteo).",
         "- **Forecasters:** persistence (tomorrow = today), ridge (linear) regression, "
         "random forest, XGBoost, GRU and LSTM.",
         "- **Choosing settings:** each forecaster has a small fixed list of settings "
         "(`src/config.py`). For each forecaster and feature set, the one with the lowest mean "
         f"validation MAE was chosen. Models with randomness used {len(config.SEEDS)} seeds, "
         "averaged. Choices were saved to `models/comparison/selected.json` before any test "
         "forecast was made.",
         "- **Test scored once:** the chosen models were then re-trained with the same seeds and "
         "scored on the test period. Nothing was changed after seeing the test numbers.",
         "- **Significance:** the bracket after each difference is a 95% block-bootstrap "
         "interval (7-day blocks). If it contains 0, the data cannot tell the two forecasts apart.",
         "",
         "## Weather data decisions", "",
         _md([
             ["1", "Source: Open-Meteo Historical Weather API, daily, central Kolkata, "
                   "2018-04-01 to 2020-07-01, timezone Asia/Kolkata",
              "Free, no account, gap-free reanalysis covering the whole AQI period"],
             ["2", "File kept unchanged at `data/external/kolkata_weather_open_meteo.csv`",
              "Same rule as the Kaggle file; a test checks its bytes"],
             ["3", "Inputs: " + ", ".join(f"`{c}`" for c in config.WEATHER_FEATURES),
              "Temperature (inversions), rain (washout), wind (dispersion), humidity, "
              "dew point, pressure (stagnant air), sunlight"],
             ["4", "Wind direction as sine and cosine", "An angle: 359 and 1 degree are "
                   "nearly the same direction"],
             ["5", "Only weather up to the day before the forecast day is used",
              "The forecast day's weather is not known in advance; a test checks this"],
             ["6", "Weather never filled in", "0 missing days in the period; the code stops "
                   "if this changes"],
             ["7", "Scaler fitted on training rows only", "Same rule as the pollutant inputs"],
         ], ["#", "Decision", "Reason"]),
         "",
         "## Settings chosen on validation", "",
         _md([[s["feature_set"], s["family"], s["candidate"],
               f"{s['val_MAE_mean']:.1f}" + (f" ± {s['val_MAE_sd']:.1f}" if s["seeds"] > 1 else ""),
               int(s["seeds"])] for _, s in selected.iterrows()],
             ["feature set", "forecaster", "chosen setting", "val MAE (mean ± sd)", "seeds"]),
         "",
         f"Persistence validation MAE: {pers['val_MAE']:.1f}. **Overall choice on validation:** "
         f"{overall['family']} with `{overall['feature_set']}` (val MAE "
         f"{overall['val_MAE_mean']:.1f}).",
         "",
         "## Test period (scored once)", "",
         _md([[t["family"], t["feature_set"], f"{t['MAE']:.1f}", f"{t['RMSE']:.1f}",
               f"{t['R2']:.3f}", f"{t['Bias']:+.1f}", f"{t['alert_F1']:.3f}",
               "-" if t["family"] == "Persistence" else
               f"{t['diff']:+.1f} [{t['lo']:+.1f}, {t['hi']:+.1f}]"]
              for _, t in test.iterrows()],
             ["forecaster", "feature set", "MAE", "RMSE", "R2", "Bias",
              f"alert F1 (>{config.ALERT_THRESHOLD})", "MAE vs persistence [95% interval]"]),
         "",
         f"MAE vs persistence: negative = more accurate than persistence. Alert F1 uses the fixed "
         f"threshold of {config.ALERT_THRESHOLD}.",
         "",
         f"**The model chosen on validation** ({overall['family']}, `{overall['feature_set']}`) "
         f"scored test MAE **{ov['MAE']:.1f}** vs persistence {pers['MAE']:.1f}: difference "
         f"{ov['diff']:+.1f} [{ov['lo']:+.1f}, {ov['hi']:+.1f}], "
         f"{_verdict(ov['lo'], ov['hi'], 'clearly better than persistence', 'clearly worse than persistence')}.",
         "",
         "## Does weather help?", "",
         _md([[w["family"], f"{w['val_without']:.1f}", f"{w['val_with']:.1f}",
               f"{w['test_without']:.1f}", f"{w['test_with']:.1f}",
               f"{w['diff']:+.1f} [{w['lo']:+.1f}, {w['hi']:+.1f}]",
               _verdict(w["lo"], w["hi"], "weather helps", "weather hurts")]
              for _, w in weather.iterrows()],
             ["forecaster", "val MAE without", "val MAE with", "test MAE without",
              "test MAE with", "test difference [95% interval]", "verdict"]),
         "",
         "Difference = MAE with weather minus MAE without (negative = weather helps). The "
         "validation MAE here is for the averaged forecast of the seeds, so it can differ "
         "slightly from the mean over seeds in the table above.",
         "",
         "## Before and during the COVID-19 lockdown (test MAE)", "",
         _md([[t["family"], t["feature_set"], f"{t['MAE_before_lockdown']:.1f}",
               f"{t['MAE_lockdown']:.1f}"] for _, t in test.iterrows()],
             ["forecaster", "feature set", f"before {config.LOCKDOWN_START}",
              f"from {config.LOCKDOWN_START}"]),
         "",
         "## What XGBoost relies on (with weather)", "",
         "Share of the model's total gain (how much each input improved its splits), "
         "averaged over seeds. Top 10:", "",
         _md([[v["variable"], v["type"], f"{v['share_%']:.1f}%"]
              for _, v in r["importance"].head(10).iterrows()],
             ["input", "type", "share"]),
         "",
         f"All weather inputs together: "
         f"{r['importance'].loc[r['importance']['type'] == 'weather', 'share_%'].sum():.1f}% of the gain.",
         "",
         "## How to read these results honestly", "",
         "- One chronological split, about two years of data, and a test period of "
         f"{r['n_test']} days. "
         "Small differences may not hold for other periods.",
         "- The test period contains the COVID-19 lockdown, when pollution fell far below "
         "anything in the training data.",
         "- Open-Meteo weather is a reanalysis (a model-based reconstruction from observations), "
         "not readings from a single Kolkata station.",
         "- Only past weather is used. An operational system could also use the weather "
         "*forecast* for the next day, which this project does not test.",
         "- Feature importance shows what the model used, not what causes pollution.",
         "",
         "## Files", "",
         "- `models/comparison/validation_runs.csv`: validation score of every setting and seed",
         "- `models/comparison/selected.json`: the choices, saved before testing",
         f"- `{config.COMPARISON_PREDICTIONS_CSV.relative_to(config.PROJECT_ROOT).as_posix()}`: "
         "every test forecast, day by day",
         ]
    if fig:
        L.append(f"- `{fig.relative_to(config.PROJECT_ROOT).as_posix()}`")
    L.append("")
    config.COMPARISON_MD.write_text("\n".join(L), encoding="utf-8")


def _plot(test: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    pers = float(test.loc[test["family"] == "Persistence", "MAE"].iloc[0])
    fams = FAMILIES[::-1]
    yy = np.arange(len(fams))
    get = lambda fs: [float(test[(test["family"] == f) & (test["feature_set"] == fs)]["MAE"].iloc[0])  # noqa: E731
                      for f in fams]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(yy + 0.2, get("pollution"), height=0.38, color=BLUE, label="pollution only")
    ax.barh(yy - 0.2, get("pollution+weather"), height=0.38, color=ORANGE,
            label="pollution + weather")
    ax.axvline(pers, color=INK2, linestyle="--", linewidth=1.2)
    ax.text(pers, len(fams) - 0.45, f" persistence {pers:.1f}", color=INK2, fontsize=9, va="bottom")
    ax.set_yticks(yy)
    ax.set_yticklabels(fams)
    ax.set_xlabel("Test MAE (AQI points, lower is better)")
    ax.set_title("Test period: forecast error by model", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / "model_comparison_test_mae.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _print_summary(r: dict) -> None:
    t = r["test"]
    print("\nTEST period (scored once):")
    print(f"  {'forecaster':<15} {'feature set':<18} {'MAE':>6} {'R2':>6} {'alert F1':>9}   "
          "MAE vs persistence [95%]")
    for _, x in t.iterrows():
        diff = "" if x["family"] == "Persistence" else \
            f"{x['diff']:+5.1f} [{x['lo']:+.1f}, {x['hi']:+.1f}]"
        print(f"  {x['family']:<15} {x['feature_set']:<18} {x['MAE']:6.1f} {x['R2']:6.3f} "
              f"{x['alert_F1']:9.3f}   {diff}")
    print("\nDoes weather help? (test MAE with - without; negative = helps)")
    for _, w in r["weather"].iterrows():
        print(f"  {w['family']:<15} {w['test_without']:6.1f} -> {w['test_with']:6.1f}   "
              f"{w['diff']:+5.1f} [{w['lo']:+.1f}, {w['hi']:+.1f}]  "
              f"{_verdict(w['lo'], w['hi'], 'weather helps', 'weather hurts')}")
    o = r["overall"]
    print(f"\nChosen on validation: {o['family']} [{o['feature_set']}]  "
          f"(total {r['seconds'] / 60:.1f} min)")
    print(f"Report: {config.COMPARISON_MD}")


if __name__ == "__main__":
    run_comparison()
