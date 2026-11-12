# Kolkata Air Quality Forecasting System

B.Tech 3rd-year Deep Learning project (Neural Network & Deep Learning, PCCCS582).

Forecasts **tomorrow's Air Quality Index (AQI) for Kolkata** from the previous 14 days of
air-quality data using a PyTorch **LSTM**, compares it with a **persistence baseline**
(tomorrow = today), and raises a **high-pollution alert** when predicted AQI > 200.
A second experiment compares **five forecasters** (ridge regression, random forest, XGBoost,
GRU, LSTM), each **with and without daily weather data**.

## Data

- **Air quality:** Kaggle - *Air Quality Data in India* (Rohan Rao),
  https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india - file `city_day.csv`.
  Stored unchanged at `data/raw/city_day.csv`
  (SHA-256 `0d84b21c3e4878bbad8df362f2ab05f61ad959538dddf5918e714077ed3c1847` with LF line
  endings).
- Kolkata coverage: 2018-04-10 to 2020-07-01 (814 days). Modelling period:
  2018-06-16 to 2020-07-01 (747 days with no AQI gaps).
- **Weather** (comparison experiment only): Open-Meteo Historical Weather API,
  https://open-meteo.com/en/docs/historical-weather-api - daily values for central Kolkata
  (22.5726 N, 88.3639 E), timezone Asia/Kolkata, 2018-04-01 to 2020-07-01, downloaded as CSV.
  Stored unchanged at `data/external/kolkata_weather_open_meteo.csv`. Variables: mean, max and
  min temperature; rainfall total and rain hours; max and mean wind speed; dominant wind
  direction; solar radiation; dew point; humidity; surface pressure.
- Full audit: `reports/dataset_audit.md`. Every cleaning decision:
  `reports/preprocessing_decisions.md`. Weather decisions: `reports/model_comparison.md`.

## Setup (Windows, NVIDIA GPU)

Python 3.13 or newer. From the project root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# replace the CPU-only PyTorch with the CUDA build (needs an NVIDIA driver; no separate CUDA toolkit)
pip uninstall -y torch
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If PowerShell refuses to activate the environment, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first.
On macOS or Linux, activate with `source .venv/bin/activate` and skip the CUDA step.

## How to run

```powershell
python main.py all                 # full demo: audit -> preprocess -> baseline -> train -> evaluate -> predict
```

Or step by step:

```powershell
python main.py audit               # dataset audit           -> reports/dataset_audit.md
python main.py preprocess          # clean data + scaler      -> data/processed/
python main.py baseline            # persistence scores (no training needed)
python main.py train               # 4 LSTM variants x 5 seeds -> models/experiment/, pick on validation
python main.py evaluate            # selected LSTM vs persistence on test -> reports/results.md
python main.py predict             # forecast 2020-07-02 (day after the data ends)
python main.py predict --date 2020-01-15   # forecast a past day and compare with the real AQI
python main.py compare             # 5 forecasters x (with / without weather) -> reports/model_comparison.md
python -m unittest discover -s tests -v    # 26 research-rule checks (no leakage etc.)
```

**Device.** The default (`--device auto`) uses the NVIDIA GPU when the CUDA build of PyTorch
can see one, otherwise the CPU. In `compare`, every forecaster runs on the chosen device:
GRU, LSTM and ridge regression through PyTorch, XGBoost and the random forest through
XGBoost's CUDA mode. On the GPU, deterministic mode is switched on, so the same machine gives
the same numbers on every run; GPU and CPU numbers can differ very slightly. Use
`--device cpu` to force the CPU. The results in `reports/` were produced on an
NVIDIA RTX 5070 Ti Laptop GPU (CUDA 13.2).

## Results (test period 2020-01-01 to 2020-07-01, 183 days, scored once)

Main experiment (`reports/results.md`): the selected LSTM (`change_small`, average of 5 seeds)
reached **MAE 19.8** vs persistence **20.5**. The 95% interval for the difference is
-1.9 to +0.5, so the two cannot be told apart.

Model comparison (`reports/model_comparison.md`), pollution inputs only:

| Forecaster | Validation MAE | Test MAE | Test MAE vs persistence [95% interval] |
|---|---|---|---|
| Persistence | 32.4 | 20.5 | - |
| XGBoost (chosen on validation) | 26.3 | 20.7 | +0.2 [-4.7, +4.7] |
| Ridge regression | 28.6 | 16.8 | -3.7 [-6.9, -0.6] |
| Random forest | 26.9 | 18.2 | -2.3 [-5.4, +0.7] |
| GRU | 30.9 | 18.7 | -1.8 [-5.8, +1.5] |
| LSTM | 31.9 | 19.8 | -0.7 [-1.9, +0.5] |

What this shows, stated honestly:

- The forecaster **chosen on validation (XGBoost) did not beat persistence** on test.
- **Ridge regression** had the lowest test error and was clearly better than persistence,
  but validation ranked it fourth. It is not promoted to "best model" after seeing the test
  results, because that would be choosing on the test period. The gap shows that the 92-day
  validation period (October to December 2019) was a weak guide to a test period containing
  winter, spring and the COVID-19 lockdown.
- **Adding weather did not help:** it made ridge, GRU and LSTM clearly worse and made no
  clear difference to the random forest and XGBoost. Likely reasons: 13 extra inputs for
  only 458 training days, a weather-pollution relationship that changed during lockdown,
  and seasonal information already present in the day-of-year inputs.

## Project structure

```text
data/raw/city_day.csv            original Kaggle file (never modified)
data/external/                   Open-Meteo daily weather for Kolkata (never modified)
data/processed/                  cleaned Kolkata table (with *_imputed flags) + scaler.json
models/experiment/               one trained LSTM per (variant, seed) + training history
models/validation_runs.csv       validation score of every LSTM run
models/selected_model.json       which LSTM variant was chosen, and the rule used
models/comparison/               comparison: validation score of every setting + saved choices
reports/                         audit, decisions, results, model comparison, figures
src/config.py                    every setting in one place (dates, features, models, device)
src/inspect_data.py              dataset audit
src/data_loader.py               reads the raw CSV
src/weather.py                   reads and checks the weather CSV, joins it by date
src/preprocessing.py             cleaning, chronological split, train-only scaling
src/dataset.py                   14-day history -> next-day AQI samples
src/baseline.py                  persistence baseline
src/model.py                     LSTM and GRU models, device choice
src/train.py                     LSTM training with early stopping
src/evaluate.py                  MAE / RMSE / MAPE / R2 / bias + alert precision/recall/F1
src/predict.py                   single-day forecast + alert
src/compare.py                   5 forecasters x 2 feature sets, chosen on validation
tests/test_pipeline.py           checks for leakage, splits, imputation flags, metrics
tests/test_comparison.py         checks for the weather data and the comparison
main.py                          command-line entry point
```

## Research rules this code follows

- Raw data is never changed; cleaned data is saved separately. Tests check the bytes of both
  source files.
- No random train/test split: train (2018-06-16 to 2019-09-30), validation
  (2019-10-01 to 2019-12-31), test (2020-01-01 to 2020-07-01), all by date.
- Inputs use only days *before* the forecast day, including the weather; the scaler is fitted
  on training data only.
- AQI (the target) is never filled in. The single missing input value (O3 on 2018-06-17)
  is copied from the previous day and flagged as imputed. Weather values are never filled in.
- The main LSTM uses air-quality data only. Weather is added only in the comparison
  experiment, with its own decision log.
- Results are reported against the persistence baseline on the same days, including
  when a model does worse.
- Every choice uses the validation period only: the LSTM variant (4 variants x 5 seeds),
  each comparison forecaster's settings, and the alert thresholds. Choices are saved to disk
  before the test period is scored, once.
- Test comparisons include a 95% block-bootstrap interval for the difference in MAE, so a
  small difference is not mistaken for a real one.

## Known limitations

- Only about 2 years of Kolkata data (two winters).
- The test period includes the COVID-19 lockdown (from 2020-03-25).
- The dataset ends in July 2020, so `predict` is a demonstration on historical data,
  not a live forecast.
- Results come from one chronological split (5 seeds per random model), and a 92-day
  validation period that turned out to be a weak guide to the test period.
- The weather is reanalysis data (a model-based reconstruction), not readings from a single
  Kolkata station, and only past weather is used; a live system could also use tomorrow's
  weather forecast.
