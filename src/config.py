"""
Central settings for the Kolkata AQI forecasting project.

Every choice that affects results lives here, so it can be changed in one
place and is easy to report. The reason for each choice is written next to it
and in reports/preprocessing_decisions.md.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "city_day.csv"
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "kolkata_daily_clean.csv"
SCALER_JSON = PROJECT_ROOT / "data" / "processed" / "scaler.json"
MODEL_DIR = PROJECT_ROOT / "models"
EXPERIMENT_DIR = MODEL_DIR / "experiment"          # one checkpoint per (variant, seed)
VALIDATION_RUNS_CSV = MODEL_DIR / "validation_runs.csv"
SELECTED_JSON = MODEL_DIR / "selected_model.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
DECISIONS_MD = REPORTS_DIR / "preprocessing_decisions.md"
RESULTS_MD = REPORTS_DIR / "results.md"
PREDICTIONS_CSV = REPORTS_DIR / "test_predictions.csv"

# ---------------------------------------------------------------------------
# Data selection
# ---------------------------------------------------------------------------
CITY = "Kolkata"

# Start of the modelling period. Evidence (reports/dataset_audit.md):
#   - AQI is missing for 59 days in a row (2018-04-18 .. 2018-06-15);
#   - CO is recorded as exactly 0 for 47 days from 2018-04-26 and Toluene for
#     8 days from 2018-06-08 (likely "no reading" stored as 0).
# From 2018-06-16 onward AQI is unbroken for 747 days, so no AQI value needs
# to be invented.
START_DATE = "2018-06-16"
END_DATE = "2020-07-01"  # last date in the file

# Pollutant inputs. Xylene is excluded: 51.6 % missing for Kolkata and only
# recorded from mid-2019, so using it would mean inventing ~half its values.
POLLUTANT_FEATURES = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene",
]
TARGET = "AQI"

# Calendar features (sine/cosine of day-of-year). These are known in advance
# for any date, so they do not leak future information.
USE_CALENDAR_FEATURES = True

# ---------------------------------------------------------------------------
# Chronological split (by the date of the day being forecast)
# ---------------------------------------------------------------------------
# Chosen from the audit so each split contains high-pollution (AQI > 200)
# days; a plain 70/15/15 split left only 2 such days in the test set, which
# would make the alert impossible to evaluate. Decided BEFORE any model was
# trained (based only on the AQI distribution, never on model results).
TRAIN_END = "2019-09-30"   # train:      2018-06-16 .. 2019-09-30
VAL_END = "2019-12-31"     # validation: 2019-10-01 .. 2019-12-31
                           # test:       2020-01-01 .. 2020-07-01

# ---------------------------------------------------------------------------
# Forecasting set-up
# ---------------------------------------------------------------------------
LOOKBACK = 14              # days of history fed to the LSTM
HORIZON = 1                # forecast 1 day ahead
ALERT_THRESHOLD = 200      # alert if predicted AQI > 200 ("Poor" or worse)

# ---------------------------------------------------------------------------
# Model and training
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 20              # early stopping: stop after 20 epochs without improvement
GRAD_CLIP = 1.0
SEED = 42                  # used when a single seed is needed
# "cpu" gives repeatable numbers on every run. "mps" (Apple GPU) also works but
# can give slightly different numbers between runs. "auto" picks mps if available.
DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Experiment (fixed BEFORE seeing any of its results)
# ---------------------------------------------------------------------------
# The first run (2-layer, 64-unit LSTM predicting tomorrow's AQI directly) lost
# to persistence and over-predicted. Two changes are tested, in a 2 x 2 design
# so the effect of each can be seen separately:
#   target_mode "level"  : predict tomorrow's AQI directly (original)
#   target_mode "change" : predict tomorrow's AQI minus today's AQI; the forecast
#                          is today's AQI + predicted change, so the model starts
#                          from persistence and only learns corrections
#   size "large" (64 units x 2 layers, original) vs "small" (32 units x 1 layer,
#   fewer weights -> less room to memorise 458 training samples)
VARIANTS = {
    "level_large":  {"target_mode": "level",  "hidden_size": 64, "num_layers": 2},
    "level_small":  {"target_mode": "level",  "hidden_size": 32, "num_layers": 1},
    "change_large": {"target_mode": "change", "hidden_size": 64, "num_layers": 2},
    "change_small": {"target_mode": "change", "hidden_size": 32, "num_layers": 1},
}
# Each variant is trained 5 times from different random starting weights;
# results are reported as the average (and spread) over the 5 runs.
SEEDS = [42, 43, 44, 45, 46]

# Selection rule: the variant with the lowest MEAN validation MAE over the 5
# seeds is chosen. The test period is not used for any choice.
SELECTION_METRIC = "MAE"

# Fair alert comparison: for persistence AND the LSTM, the alert threshold is
# picked on the validation period (best F1; ties -> closest to 200), then
# applied unchanged to the test period.
ALERT_THRESHOLD_GRID = list(range(150, 255, 5))

# Block bootstrap for a confidence interval on (LSTM MAE - persistence MAE):
# test days are resampled in 7-day blocks because neighbouring days are similar.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_BLOCK_DAYS = 7

LOCKDOWN_START = "2020-03-25"  # external fact (India national lockdown), used only to label periods


# ---------------------------------------------------------------------------
# Model comparison with weather  (python main.py compare)
# Fixed BEFORE the comparison was run. See src/compare.py and
# reports/model_comparison.md.
# ---------------------------------------------------------------------------
# Daily weather for central Kolkata from the Open-Meteo Historical Weather API
# (https://open-meteo.com/en/docs/historical-weather-api), downloaded as CSV with
# timezone Asia/Kolkata for 2018-04-01 .. 2020-07-01. Only ever READ.
WEATHER_CSV = PROJECT_ROOT / "data" / "external" / "kolkata_weather_open_meteo.csv"
WEATHER_TIMEZONE = "Asia/Kolkata"

# Open-Meteo column (unit suffix removed) -> short name used in the code
WEATHER_COLUMNS = {
    "temperature_2m_mean": "temp_mean",
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",          # cold nights -> inversions trap smog
    "precipitation_sum": "rain_mm",            # rain washes particles out
    "precipitation_hours": "rain_hours",
    "wind_speed_10m_max": "wind_max",          # wind disperses pollution
    "wind_speed_10m_mean": "wind_mean",
    "wind_direction_10m_dominant": "wind_dir_deg",
    "shortwave_radiation_sum": "radiation",
    "dew_point_2m_mean": "dew_point",
    "relative_humidity_2m_mean": "humidity",
    "surface_pressure_mean": "pressure",       # high pressure -> stagnant air
}
# Model inputs. Wind direction (an angle) enters as sine and cosine.
WEATHER_FEATURES = [
    "temp_mean", "temp_max", "temp_min", "rain_mm", "rain_hours",
    "wind_max", "wind_mean", "wind_dir_sin", "wind_dir_cos",
    "radiation", "dew_point", "humidity", "pressure",
]

# Non-neural models get one flat row per sample: every input "today", AQI on
# each of the last 7 days, and the 7-day mean of every input.
TABULAR_RECENT_DAYS = 7
TABULAR_TARGET_MODES = ["level", "change"]     # same two options as the LSTM

# Candidate settings for each forecaster (the best is chosen on validation)
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
RF_TREES = 300
RF_GRID = [{"max_depth": d, "min_samples_leaf": leaf} for d in (4, 8, None) for leaf in (1, 5)]
XGB_GRID = [{"max_depth": d} for d in (2, 3, 4)]
XGB_PARAMS = {"n_estimators": 2000, "learning_rate": 0.03, "subsample": 0.8,
              "colsample_bytree": 0.8, "min_child_weight": 3, "reg_lambda": 1.0}
XGB_EARLY_STOPPING = 50      # stop adding trees after 50 rounds without validation gain
# The GRU and LSTM use the same 4 VARIANTS and SEEDS defined above.

COMPARISON_DIR = MODEL_DIR / "comparison"
COMPARISON_MD = REPORTS_DIR / "model_comparison.md"
COMPARISON_PREDICTIONS_CSV = REPORTS_DIR / "comparison_test_predictions.csv"
