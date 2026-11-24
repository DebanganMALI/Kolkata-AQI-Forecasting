"""Central settings for the whole project: paths, dates, model sizes and experiment choices."""

from pathlib import Path

# Project folders and file paths.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "city_day.csv"
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "kolkata_daily_clean.csv"
SCALER_JSON = PROJECT_ROOT / "data" / "processed" / "scaler.json"
MODEL_DIR = PROJECT_ROOT / "models"
EXPERIMENT_DIR = MODEL_DIR / "experiment"
VALIDATION_RUNS_CSV = MODEL_DIR / "validation_runs.csv"
SELECTED_JSON = MODEL_DIR / "selected_model.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
DECISIONS_MD = REPORTS_DIR / "preprocessing_decisions.md"
RESULTS_MD = REPORTS_DIR / "results.md"
PREDICTIONS_CSV = REPORTS_DIR / "test_predictions.csv"

# City studied.
CITY = "Kolkata"

# Modelling period of the Kaggle data.
START_DATE = "2018-06-16"
END_DATE = "2020-07-01"

# Pollutant inputs and the forecast target.
POLLUTANT_FEATURES = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene",
]
TARGET = "AQI"

# Add day-of-year inputs.
USE_CALENDAR_FEATURES = True

# Chronological train, validation and test cut-off dates.
TRAIN_END = "2019-09-30"
VAL_END = "2019-12-31"

# Window length, forecast horizon and alert threshold.
LOOKBACK = 14
HORIZON = 1
ALERT_THRESHOLD = 200

# Network size, training settings, seed and device.
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 20
GRAD_CLIP = 1.0
SEED = 42
DEVICE = "auto"

# LSTM variants and seeds compared on validation.
VARIANTS = {
    "level_large":  {"target_mode": "level",  "hidden_size": 64, "num_layers": 2},
    "level_small":  {"target_mode": "level",  "hidden_size": 32, "num_layers": 1},
    "change_large": {"target_mode": "change", "hidden_size": 64, "num_layers": 2},
    "change_small": {"target_mode": "change", "hidden_size": 32, "num_layers": 1},
}
SEEDS = [42, 43, 44, 45, 46]

# Model selection rule.
SELECTION_METRIC = "MAE"

# Alert threshold candidates.
ALERT_THRESHOLD_GRID = list(range(150, 255, 5))

# Block bootstrap settings.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_BLOCK_DAYS = 7

# Lockdown start date, used only to label periods.
LOCKDOWN_START = "2020-03-25"


# Weather file and its time zone.
WEATHER_CSV = PROJECT_ROOT / "data" / "external" / "kolkata_weather_open_meteo.csv"
WEATHER_TIMEZONE = "Asia/Kolkata"

# Open-Meteo columns, their short names, and the weather inputs.
WEATHER_COLUMNS = {
    "temperature_2m_mean": "temp_mean",
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "precipitation_sum": "rain_mm",
    "precipitation_hours": "rain_hours",
    "wind_speed_10m_max": "wind_max",
    "wind_speed_10m_mean": "wind_mean",
    "wind_direction_10m_dominant": "wind_dir_deg",
    "shortwave_radiation_sum": "radiation",
    "dew_point_2m_mean": "dew_point",
    "relative_humidity_2m_mean": "humidity",
    "surface_pressure_mean": "pressure",
}
WEATHER_FEATURES = [
    "temp_mean", "temp_max", "temp_min", "rain_mm", "rain_hours",
    "wind_max", "wind_mean", "wind_dir_sin", "wind_dir_cos",
    "radiation", "dew_point", "humidity", "pressure",
]

# Flattened-feature settings for non-neural models.
TABULAR_RECENT_DAYS = 7
TABULAR_TARGET_MODES = ["level", "change"]

# Candidate settings for ridge, random forest and XGBoost.
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
RF_TREES = 300
RF_ROW_SAMPLE = 0.632
RF_GRID = [{"max_depth": d, "min_child_weight": leaf} for d in (4, 8, 12) for leaf in (1, 5)]
XGB_GRID = [{"max_depth": d} for d in (2, 3, 4)]
XGB_PARAMS = {"n_estimators": 2000, "learning_rate": 0.03, "subsample": 0.8,
              "colsample_bytree": 0.8, "min_child_weight": 3, "reg_lambda": 1.0}
XGB_EARLY_STOPPING = 50

# Output folders and files of the comparison.
COMPARISON_DIR = MODEL_DIR / "comparison"
COMPARISON_MD = REPORTS_DIR / "model_comparison.md"
COMPARISON_PREDICTIONS_CSV = REPORTS_DIR / "comparison_test_predictions.csv"

# Advanced experiment: quantiles, horizons, candidates and outputs.
QUANTILES = (0.1, 0.5, 0.9)
HORIZONS = [1, 3, 7]
ADVANCED_CANDIDATES = {
    "Attention LSTM": [{"name": f"{m}, 32 units", "target_mode": m, "hidden_size": 32}
                       for m in ("level", "change")],
    "TCN": [{"name": f"{m}, 32 channels", "target_mode": m, "channels": 32}
            for m in ("level", "change")],
    "Transformer": [{"name": f"{m}, d=32, 2 layers", "target_mode": m, "d_model": 32, "num_layers": 2}
                    for m in ("level", "change")],
    "Quantile attention LSTM": [{"name": f"{m}, 32 units", "target_mode": m, "hidden_size": 32}
                                for m in ("level", "change")],
}
ADVANCED_DIR = MODEL_DIR / "advanced"
ADVANCED_MD = REPORTS_DIR / "advanced_results.md"
ADVANCED_PREDICTIONS_CSV = REPORTS_DIR / "advanced_test_predictions.csv"

# Extended dataset files.
V2_CITY_AQI_XLSX_DIR = PROJECT_ROOT / "data" / "raw" / "cpcb" / "city_aqi"
V2_CITY_AQI_CSV = PROJECT_ROOT / "data" / "interim" / "cpcb_kolkata_city_aqi_daily.csv"
V2_STATIONS_DIR = PROJECT_ROOT / "data" / "raw" / "cpcb" / "stations"
V2_WEATHER_CSV = PROJECT_ROOT / "data" / "external" / "kolkata_weather_2018_2026.csv"
V2_FIRES_CSV = PROJECT_ROOT / "data" / "interim" / "firms_daily_2018_2026.csv"
V2_TABLE_CSV = PROJECT_ROOT / "data" / "processed" / "kolkata_extended_daily.csv"

# Extended dataset dates, splits, input filling limit and seeds.
V2_START, V2_END = "2018-10-01", "2026-10-31"
V2_TRAIN_END = "2023-12-31"
V2_VAL_END = "2024-12-31"
V2_FFILL_LIMIT = 7
V2_SEEDS = [42, 43, 44]

# Fire, festival and lockdown inputs.
V2_FIRE_FEATURES = ["log_fires_300km", "log_frp_region"]
V2_EVENT_DATES = {
    "kali_puja": ["2018-11-06", "2019-10-27", "2020-11-14", "2021-11-04", "2022-10-24",
                  "2023-11-12", "2024-10-31", "2025-10-20"],
    "durga_puja": ["2018-10-19", "2019-10-08", "2020-10-26", "2021-10-15", "2022-10-05",
                   "2023-10-24", "2024-10-12", "2025-10-02", "2026-10-20"],
    "holi": ["2019-03-21", "2020-03-10", "2021-03-29", "2022-03-18", "2023-03-08",
             "2024-03-25", "2025-03-14", "2026-03-04"],
    "new_year": [f"{y}-01-01" for y in range(2019, 2027)],
}
V2_EVENT_WINDOWS = {"kali_puja": (1, 2), "durga_puja": (3, 0), "holi": (0, 0), "new_year": (1, 0)}
V2_LOCKDOWN = ("2020-03-25", "2020-05-31")
V2_EVENT_FEATURES = [f"{k}_next" for k in V2_EVENT_WINDOWS] + ["lockdown_next"]

# Extended feature sets, candidates, learning curve and outputs.
V2_FEATURE_SETS = {
    "pollution": [],
    "pollution+weather": list(WEATHER_FEATURES),
    "all": list(WEATHER_FEATURES) + V2_FIRE_FEATURES + V2_EVENT_FEATURES,
}
V2_RNN_CANDIDATES = [
    {"name": "level_small", "target_mode": "level", "hidden_size": 32, "num_layers": 1},
    {"name": "change_small", "target_mode": "change", "hidden_size": 32, "num_layers": 1},
]
V2_LEARNING_YEARS = [1, 2, 3, 5]
V2_DIR = MODEL_DIR / "extended"
V2_MD = REPORTS_DIR / "extended_results.md"
V2_PREDICTIONS_CSV = REPORTS_DIR / "extended_test_predictions.csv"

# Walk-forward rounds, spike weighting, ensemble members and outputs.
ROBUST_FOLDS = [
    {"name": "2022", "train_end": "2020-12-31", "val_end": "2021-12-31", "test_end": "2022-12-31"},
    {"name": "2023", "train_end": "2021-12-31", "val_end": "2022-12-31", "test_end": "2023-12-31"},
    {"name": "2024", "train_end": "2022-12-31", "val_end": "2023-12-31", "test_end": "2024-12-31"},
    {"name": "2025-2026", "train_end": "2023-12-31", "val_end": "2024-12-31", "test_end": "2026-10-31"},
]
SPIKE_GAMMAS = [1.0, 2.0]
SPIKE_TOP_SHARE = 0.2
ENSEMBLE_FAMILIES = ["GRU", "LSTM", "Transformer", "XGBoost"]
ROBUST_DIR = MODEL_DIR / "robust"
ROBUST_MD = REPORTS_DIR / "robust_results.md"
ROBUST_PREDICTIONS_CSV = REPORTS_DIR / "robust_test_predictions.csv"
