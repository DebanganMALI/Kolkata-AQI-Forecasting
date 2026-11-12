"""
Extended experiment on 2018-2026 official data:  python main.py extended

Same task as before (previous 14 days -> next-day AQI), now with:
  * the official CPCB city AQI as the target, 2,900+ days instead of 747;
  * a full calendar year (2024) as validation and 22 months as test;
  * three feature sets: pollution | + weather | + weather + fires + festivals;
  * eight forecasters: ridge, random forest, XGBoost, GRU, LSTM,
    attention LSTM, TCN, Transformer (+ persistence);
  * forecast ranges (quantile attention LSTM), 1/3/7 days ahead, and a
    learning curve (does more training data help?).

Rules: candidate settings fixed in config.py; every choice (setting, feature
set, model) on validation only and saved to models/extended/selected.json
before the chosen models are re-trained (same seeds) and scored on test, once.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from torch import nn

from src import advanced, compare, config
from src.dataset import Windows, encode_target, make_windows, target_stats
from src.evaluate import (BLUE, INK2, ORANGE, alert_metrics, block_bootstrap_mae_diff,
                          regression_metrics, style_axes)
from src.extended_data import POLLUTANTS, build_extended_table
from src.model import pick_device
from src.preprocessing import StandardScaler

def _log(msg: str) -> None:
    """One line per step in models/extended/progress.log, written immediately (readable mid-run)."""
    config.V2_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.V2_DIR / "progress.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%H:%M:%S}  {msg}\n")


FAMILIES = ["Linear (ridge)", "Random forest", "XGBoost", "GRU", "LSTM",
            "Attention LSTM", "TCN", "Transformer"]
ADVANCED = ("Attention LSTM", "TCN", "Transformer")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def columns_for(feature_set: str) -> list[str]:
    return ["AQI", *POLLUTANTS, "doy_sin", "doy_cos", *config.V2_FEATURE_SETS[feature_set]]


def extended_windows(table: pd.DataFrame, scaler: StandardScaler, horizon: int = 1) -> Windows:
    """14-day windows; samples whose target day has no official AQI are dropped."""
    w = make_windows(table, scaler, horizon=horizon)
    target = table.set_index("Date")["AQI_target"]
    y = target.reindex(pd.to_datetime(w.target_dates)).to_numpy()
    keep = ~np.isnan(y)
    return Windows(w.X[keep], y[keep], w.last_aqi[keep], w.target_dates[keep], w.split[keep])


def build_data(table: pd.DataFrame, feature_set: str, horizon: int = 1):
    scaler = StandardScaler.fit(table[table["split"] == "train"], columns_for(feature_set))
    return extended_windows(table, scaler, horizon), scaler


def last_years(data, years: int):
    """Same windows, but only the last `years` of the training period stay 'train'."""
    w, scaler = data
    cut = np.datetime64(pd.Timestamp(config.V2_TRAIN_END) - pd.DateOffset(years=years))
    split = np.where((w.split == "train") & (w.target_dates <= cut), "unused", w.split)
    return Windows(w.X, w.y_true, w.last_aqi, w.target_dates, split), scaler


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def candidates(family: str) -> list[dict]:
    if family in ("GRU", "LSTM"):
        return config.V2_RNN_CANDIDATES
    if family in ADVANCED:
        return config.ADVANCED_CANDIDATES[family]
    return compare.candidates(family)


def seeds_for(family: str) -> list[int]:
    return [config.SEED] if family == "Linear (ridge)" else list(config.V2_SEEDS)


def fit(family, cand, seed, data, splits, device, quantile=False):
    if family not in ADVANCED and not quantile:
        return compare.fit_predict(family, cand, seed, data, splits, device)
    w, scaler = data
    tr, va = w.subset("train"), w.subset("val")
    mode = cand["target_mode"]
    stats = target_stats(tr, mode, scaler)
    arch = "Quantile attention LSTM" if quantile else family
    model = advanced.train_net(lambda: advanced.make_model(arch, cand, w.X.shape[2]), seed,
                               tr.X, encode_target(tr.y_true, tr.last_aqi, mode, stats),
                               va.X, encode_target(va.y_true, va.last_aqi, mode, stats),
                               advanced.pinball_loss if quantile else nn.MSELoss(), device)
    out = {}
    for s in splits:
        ws = w.subset(s)
        out[s] = advanced.decode(advanced.run_net(model, ws.X, device)[0], ws, mode, stats, quantile)
    return out, model


def seed_average(family, cand, data, splits, device, quantile=False, label=""):
    runs = []
    for s in seeds_for(family):
        runs.append(fit(family, cand, s, data, splits, device, quantile))
        _log(f"{label} {family} | {cand['name']} | seed {s} done")
    return {s: np.mean([r[0][s] for r in runs], axis=0) for s in splits}, [r[1] for r in runs]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_extended(device_name: str = config.DEVICE, verbose: bool = True) -> dict:
    t0 = time.time()
    device = pick_device(device_name)
    (config.V2_DIR / "progress.log").unlink(missing_ok=True)
    _log(f"start on {device}")
    table, log = build_extended_table()
    data = {fs: build_data(table, fs) for fs in config.V2_FEATURE_SETS}
    ref = data["pollution"][0]
    if verbose:
        print(f"Device: {device} | days {len(table)} | samples train {len(ref.subset('train'))} | "
              f"val {len(ref.subset('val'))} | test {len(ref.subset('test'))}")
        print("Phase 1 - choosing settings and feature sets on VALIDATION...\n")

    # ---- Phase 1: validation only -----------------------------------------
    rows = []
    for fs, d in data.items():
        y_val = d[0].subset("val").y_true
        for family in FAMILIES:
            tf = time.time()
            for cand in candidates(family):
                for seed in seeds_for(family):
                    p = fit(family, cand, seed, d, ["val"], device)[0]["val"]
                    rows.append({"feature_set": fs, "family": family, "candidate": cand["name"],
                                 "seed": seed, "val_MAE": regression_metrics(y_val, p)["MAE"]})
                    _log(f"phase 1 | {fs} | {family} | {cand['name']} | seed {seed} | "
                         f"val MAE {rows[-1]['val_MAE']:.2f}")
            if verbose:
                r = pd.DataFrame([x for x in rows if x["feature_set"] == fs and x["family"] == family])
                b = r.groupby("candidate")["val_MAE"].mean()
                print(f"  {fs:<18} {family:<15} val MAE {b.min():6.2f} ({b.idxmin()})  "
                      f"[{time.time() - tf:.0f}s]")
    runs = pd.DataFrame(rows)
    summary = (runs.groupby(["feature_set", "family", "candidate"], sort=False)["val_MAE"]
               .agg(["mean", "std"]).reset_index())
    best = summary.loc[summary.groupby(["feature_set", "family"], sort=False)["mean"].idxmin()]
    overall = best.loc[best["mean"].idxmin()]

    # quantile model on the feature set the attention LSTM did best with
    q_fs = best[best["family"] == "Attention LSTM"].sort_values("mean").iloc[0]["feature_set"]
    qd, q_rows = data[q_fs], []
    for cand in config.ADVANCED_CANDIDATES["Quantile attention LSTM"]:
        for seed in config.V2_SEEDS:
            p = fit("Attention LSTM", cand, seed, qd, ["val"], device, quantile=True)[0]["val"]
            q_rows.append({"candidate": cand["name"], "pinball": advanced.pinball_np(qd[0].subset("val").y_true, p)})
    q_best = pd.DataFrame(q_rows).groupby("candidate")["pinball"].mean().idxmin()

    config.V2_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.V2_DIR / "validation_runs.csv", index=False)
    with open(config.V2_DIR / "selected.json", "w") as f:
        json.dump({"rule": "lowest mean validation MAE over seeds; test period not used",
                   "per_feature_set_and_model": best.round(3).to_dict(orient="records"),
                   "overall": {"feature_set": overall["feature_set"], "family": overall["family"],
                               "candidate": overall["candidate"]},
                   "quantile_model": {"feature_set": q_fs, "candidate": q_best},
                   "decided_at": datetime.now().isoformat(timespec="seconds")}, f, indent=2)

    # ---- Phase 2: re-train chosen settings, score test once ---------------
    if verbose:
        print(f"\nChosen on validation: {overall['family']} [{overall['feature_set']}] "
              f"({overall['candidate']}). Phase 2 - scoring on TEST once...")
    te = ref.subset("test")
    y, pers = te.y_true, te.last_aqi
    preds, attention = {}, None
    for _, b in best.iterrows():
        cand = next(c for c in candidates(b["family"]) if c["name"] == b["candidate"])
        out, models = seed_average(b["family"], cand, data[b["feature_set"]], ["test"], device,
                                   label=f"phase 2 | {b['feature_set']} |")
        preds[(b["feature_set"], b["family"])] = out["test"]
        if b["family"] == "Attention LSTM" and b["feature_set"] == q_fs:
            Xt = data[q_fs][0].subset("test").X
            attention = np.mean([advanced.run_net(m, Xt, device)[1] for m in models], axis=0)
    qcand = next(c for c in config.ADVANCED_CANDIDATES["Quantile attention LSTM"] if c["name"] == q_best)
    qpred = seed_average("Attention LSTM", qcand, qd, ["test"], device, quantile=True,
                         label="quantile |")[0]["test"]

    # horizons and learning curve for the overall choice
    ocand = next(c for c in candidates(overall["family"]) if c["name"] == overall["candidate"])
    horizons = []
    for h in config.HORIZONS:
        dh = data[overall["feature_set"]] if h == 1 else build_data(table, overall["feature_set"], h)
        ph = preds[(overall["feature_set"], overall["family"])] if h == 1 else \
            seed_average(overall["family"], ocand, dh, ["test"], device, label=f"horizon {h} |")[0]["test"]
        th = dh[0].subset("test")
        d_, lo, hi = block_bootstrap_mae_diff(th.y_true, ph, th.last_aqi)
        horizons.append({"horizon": h, "model": regression_metrics(th.y_true, ph)["MAE"],
                         "persistence": regression_metrics(th.y_true, th.last_aqi)["MAE"],
                         "diff": d_, "lo": lo, "hi": hi})
        if verbose:
            print(f"  {h} day(s) ahead: {horizons[-1]['model']:.1f} vs persistence {horizons[-1]['persistence']:.1f}")
    curve = []
    for yrs in config.V2_LEARNING_YEARS:
        dl = last_years(data[overall["feature_set"]], yrs)
        p = seed_average(overall["family"], ocand, dl, ["test"], device, label=f"learning {yrs}y |")[0]["test"]
        curve.append({"years": yrs, "train_samples": len(dl[0].subset("train")),
                      "test_MAE": regression_metrics(y, p)["MAE"]})
    curve.append({"years": round(len(ref.subset("train")) / 365.25, 1),
                  "train_samples": len(ref.subset("train")),
                  "test_MAE": regression_metrics(y, preds[(overall["feature_set"], overall["family"])])["MAE"]})
    if verbose:
        print("  learning curve: " + ", ".join(f"{c['years']}y -> {c['test_MAE']:.1f}" for c in curve))

    _log("writing report")
    r = _score(table, log, te, best, overall, preds, qpred, q_fs, attention, horizons, curve,
               str(device), time.time() - t0)
    _log(f"finished in {r['seconds'] / 60:.1f} min")
    if verbose:
        _print(r)
    return r


# ---------------------------------------------------------------------------
# Scoring and report
# ---------------------------------------------------------------------------
def _verdict(lo, hi):
    return "better than persistence" if hi < 0 else "worse than persistence" if lo > 0 else "no clear difference"


def _md(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def _score(table, log, te, best, overall, preds, qpred, q_fs, attention, horizons, curve, device, seconds):
    y, pers = te.y_true, te.last_aqi
    rows = [{"feature_set": "-", "family": "Persistence", "candidate": "-", "val": np.nan,
             **regression_metrics(y, pers), "F1": alert_metrics(y, pers)["F1"], "diff": 0, "lo": 0, "hi": 0}]
    for _, b in best.iterrows():
        p = preds[(b["feature_set"], b["family"])]
        d, lo, hi = block_bootstrap_mae_diff(y, p, pers)
        rows.append({"feature_set": b["feature_set"], "family": b["family"], "candidate": b["candidate"],
                     "val": b["mean"], **regression_metrics(y, p), "F1": alert_metrics(y, p)["F1"],
                     "diff": d, "lo": lo, "hi": hi})
    test = pd.DataFrame(rows)

    effect = []
    for fam in FAMILIES:
        base = preds[("pollution", fam)]
        for fs in ("pollution+weather", "all"):
            d, lo, hi = block_bootstrap_mae_diff(y, preds[(fs, fam)], base)
            effect.append({"family": fam, "feature_set": fs, "diff": d, "lo": lo, "hi": hi})
    effect = pd.DataFrame(effect)

    inside = (y >= qpred[:, 0]) & (y <= qpred[:, 2])
    intervals = {"cover": float(inside.mean()), "width": float((qpred[:, 2] - qpred[:, 0]).mean()),
                 "alerts": [{"rule": "Persistence > 200", **alert_metrics(y, pers)},
                            {"rule": "median > 200", **alert_metrics(y, qpred[:, 1])},
                            {"rule": "upper (90th pct) > 200", **alert_metrics(y, qpred[:, 2])}]}
    att = None
    if attention is not None:
        att = pd.DataFrame({"days_before": np.arange(attention.shape[1], 0, -1), "weight": attention.mean(0)})

    out = pd.DataFrame({"date": pd.to_datetime(te.target_dates).strftime("%Y-%m-%d"),
                        "actual_AQI": y, "persistence": pers})
    for (fs, fam), p in preds.items():
        out[f"{fam} [{fs}]"] = np.round(p, 2)
    for i, q in enumerate(config.QUANTILES):
        out[f"quantile q{int(q * 100)} [{q_fs}]"] = np.round(qpred[:, i], 2)
    out.to_csv(config.V2_PREDICTIONS_CSV, index=False)

    r = {"test": test, "effect": effect, "overall": overall, "intervals": intervals, "attention": att,
         "horizons": pd.DataFrame(horizons), "curve": pd.DataFrame(curve), "q_fs": q_fs, "log": log,
         "table": table, "device": device, "seconds": seconds, "n_test": len(y)}
    r["figures"] = _plots(r)
    _write(r)
    return r


def _write(r):
    t, ov, iv, log = r["test"], r["overall"], r["intervals"], r["log"]
    tab = r["table"]
    ovr = t[(t["family"] == ov["family"]) & (t["feature_set"] == ov["feature_set"])].iloc[0]
    yearly = tab.assign(year=tab["Date"].dt.year).groupby("year")["AQI_target"].mean().round(0)
    L = ["# Extended Experiment: Official Kolkata Data, 2018-2026", "",
         f"*Generated by `python main.py extended` on {datetime.now():%Y-%m-%d %H:%M} "
         f"({r['seconds'] / 60:.1f} min on `{r['device']}`). All numbers come from this run.*", "",
         f"**Result:** the model chosen on validation, **{ov['family']}** with `{ov['feature_set']}` inputs, "
         f"scored test MAE **{ovr['MAE']:.1f}** vs persistence {t.iloc[0]['MAE']:.1f}: difference "
         f"{ovr['diff']:+.1f} [{ovr['lo']:+.1f}, {ovr['hi']:+.1f}], {_verdict(ovr['lo'], ovr['hi'])}.", "",
         "## Data", "",
         _md([["Target", "CPCB city AQI, daily 4 PM bulletin (`data/raw/cpcb/city_aqi/`)",
               f"{log['target_missing_days']} days without AQI are never used as targets"],
              ["Pollutants", "7 WBPCB stations, CPCB CAAQMS 24-hour files (`data/raw/cpcb/stations/`)",
               "daily mean over the stations reporting; gaps up to "
               f"{config.V2_FFILL_LIMIT} days carried forward and flagged"],
              ["Weather", "Open-Meteo daily, 12 variables (`data/external/kolkata_weather_2018_2026.csv`)",
               "complete; boundary layer height left out (missing Jan-Jun 2024)"],
              ["Fires", "NASA FIRMS VIIRS S-NPP (`data/interim/firms_daily_2018_2026.csv`)",
               "vegetation fires, low confidence dropped; log count within 300 km, log power in region"],
              ["Calendar", "day of year; Kali Puja, Durga Puja, Holi, New Year; 2020 lockdown",
               "flags describe the forecast day (known in advance)"]],
             ["source", "what", "handling"]), "",
         f"Days: {log['days']} ({config.V2_START} to {config.V2_END}). Split by date: train to "
         f"{config.V2_TRAIN_END} ({log['split_days']['train']} days), validation 2024 "
         f"({log['split_days']['val']}), test from 2025-01-01 ({log['split_days']['test']}).", "",
         "Yearly mean official AQI: " + ", ".join(f"{int(k)}: {int(v)}" for k, v in yearly.items()) + ".", "",
         "## Test period (scored once)", "",
         _md([[x["family"], x["feature_set"], "-" if x["family"] == "Persistence" else f"{x['val']:.1f}",
               f"{x['MAE']:.1f}", f"{x['RMSE']:.1f}", f"{x['R2']:.3f}", f"{x['F1']:.3f}",
               "-" if x["family"] == "Persistence" else f"{x['diff']:+.1f} [{x['lo']:+.1f}, {x['hi']:+.1f}]"]
              for _, x in t.iterrows()],
             ["forecaster", "feature set", "val MAE", "test MAE", "RMSE", "R2", "alert F1 (>200)",
              "test MAE vs persistence [95%]"]), "",
         "## Do weather, fires and festivals help?", "",
         _md([[e["family"], e["feature_set"], f"{e['diff']:+.1f} [{e['lo']:+.1f}, {e['hi']:+.1f}]",
               "helps" if e["hi"] < 0 else "hurts" if e["lo"] > 0 else "no clear difference"]
              for _, e in r["effect"].iterrows()],
             ["forecaster", "inputs added to pollution", "test MAE change [95%]", "verdict"]), "",
         "## Does more data help? (learning curve, chosen model)", "",
         _md([[c["years"], c["train_samples"], f"{c['test_MAE']:.1f}"] for _, c in r["curve"].iterrows()],
             ["years of training data (most recent)", "training samples", "test MAE"]), "",
         "## Forecasting further ahead (chosen model)", "",
         _md([[h["horizon"], f"{h['model']:.1f}", f"{h['persistence']:.1f}",
               f"{h['diff']:+.1f} [{h['lo']:+.1f}, {h['hi']:+.1f}]"] for _, h in r["horizons"].iterrows()],
             ["days ahead", "model MAE", "persistence MAE", "difference [95%]"]), "",
         f"## Forecast ranges (quantile attention LSTM, `{r['q_fs']}`)", "",
         f"The 80% range (10th to 90th percentile) contained the actual AQI on "
         f"**{iv['cover'] * 100:.0f}%** of test days; mean width {iv['width']:.0f} AQI points.", "",
         _md([[a["rule"], a["alerts_raised"], a["correct_alerts_TP"], a["false_alarms_FP"], a["missed_FN"],
               f"{a['precision']:.3f}", f"{a['recall']:.3f}", f"{a['F1']:.3f}"] for a in iv["alerts"]],
             ["alert rule", "alerts", "correct", "false alarms", "missed", "precision", "recall", "F1"]), ""]
    if r["attention"] is not None:
        a = r["attention"].sort_values("weight", ascending=False).head(3)
        L += ["## What the attention LSTM looks at", "",
              "Most weight: " + ", ".join(f"{int(x['days_before'])} day(s) before ({x['weight'] * 100:.1f}%)"
                                            for _, x in a.iterrows())
              + f"; equal weighting would be {100 / len(r['attention']):.1f}% per day.", ""]
    L += ["## How to read these results honestly", "",
          "- One chronological split. The test period (2025-2026) is cleaner air than most training years.",
          "- City pollutant inputs are the mean of the stations reporting each day; only 2 stations ran in 2018.",
          "- Weather is reanalysis, fires are satellite detections; associations, not causes.",
          "- Festival dates come from public calendars.", "",
          "## Files", "", "- `data/processed/kolkata_extended_daily.csv` (built table)",
          "- `models/extended/validation_runs.csv`, `models/extended/selected.json`",
          f"- `{config.V2_PREDICTIONS_CSV.relative_to(config.PROJECT_ROOT).as_posix()}`"]
    L += [f"- `{p.relative_to(config.PROJECT_ROOT).as_posix()}`" for p in r["figures"]]
    config.V2_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


def _plots(r) -> list:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out, t = [], r["test"]
    pers = float(t.iloc[0]["MAE"])
    fams = FAMILIES[::-1]
    yy = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for k, (fs, col, a) in enumerate((("pollution", BLUE, 1.0), ("pollution+weather", ORANGE, 1.0),
                                       ("all", INK2, 0.6))):
        vals = [float(t[(t["family"] == f) & (t["feature_set"] == fs)]["MAE"].iloc[0]) for f in fams]
        ax.barh(yy + 0.27 * (1 - k), vals, height=0.26, color=col, alpha=a, label=fs)
    ax.axvline(pers, color=INK2, linestyle="--", linewidth=1.2)
    ax.text(pers, len(fams) - 0.4, f" persistence {pers:.1f}", color=INK2, fontsize=9, va="bottom")
    ax.set_yticks(yy)
    ax.set_yticklabels(fams)
    ax.set_xlabel("Test MAE, 2025-2026 (AQI points, lower is better)")
    ax.set_title("Extended data: test error by model and inputs", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    style_axes(ax)
    fig.tight_layout()
    p = config.FIG_DIR / "extended_test_mae.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    c = r["curve"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(c["train_samples"], c["test_MAE"], marker="o", color=BLUE)
    for _, x in c.iterrows():
        ax.annotate(f"{x['years']}y", (x["train_samples"], x["test_MAE"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color=INK2)
    ax.set_xlabel("Training samples (most recent years)")
    ax.set_ylabel("Test MAE (AQI points)")
    ax.set_title(f"Learning curve: {r['overall']['family']}", loc="left", fontsize=11)
    style_axes(ax)
    fig.tight_layout()
    p = config.FIG_DIR / "extended_learning_curve.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)
    return out


def _print(r):
    print("\nTEST period 2025-2026 (scored once):")
    for _, x in r["test"].iterrows():
        extra = "" if x["family"] == "Persistence" else \
            f"  {x['diff']:+5.1f} [{x['lo']:+.1f}, {x['hi']:+.1f}]  {_verdict(x['lo'], x['hi'])}"
        print(f"  {x['family']:<15} {x['feature_set']:<18} MAE {x['MAE']:6.1f}  R2 {x['R2']:.3f}{extra}")
    print(f"\n80% range covered {r['intervals']['cover'] * 100:.0f}% of test days. "
          f"Total {r['seconds'] / 60:.1f} min. Report: {config.V2_MD}")


if __name__ == "__main__":
    run_extended()
