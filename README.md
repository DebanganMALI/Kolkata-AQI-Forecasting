# Kolkata Air Quality Forecasting System

B.Tech 3rd-year Deep Learning project (Neural Network & Deep Learning, PCCCS582).

Forecasts **tomorrow's Air Quality Index (AQI) for Kolkata** from the previous 14 days of
air-quality data using a PyTorch **LSTM**, compares it with a **persistence baseline**
(tomorrow = today), and raises a **high-pollution alert** when predicted AQI > 200.

## Data

- Source: Kaggle - *Air Quality Data in India* (Rohan Rao),
  https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india - file `city_day.csv`.
- Stored unchanged at `data/raw/city_day.csv`
  (SHA-256 `0d84b21c3e4878bbad8df362f2ab05f61ad959538dddf5918e714077ed3c1847`).
- Kolkata coverage: 2018-04-10 to 2020-07-01 (814 days). Modelling period:
  2018-06-16 to 2020-07-01 (747 days with no AQI gaps).
- Full audit: `reports/dataset_audit.md`. Every cleaning decision:
  `reports/preprocessing_decisions.md`.

## How to run (from the project root, on macOS)

```bash
cd ~/Desktop/Kolkata-Air-Quality-Forecasting
source .venv/bin/activate

python main.py all                 # full demo: audit -> preprocess -> baseline -> train -> evaluate -> predict
```

Or step by step:

```bash
python main.py audit               # dataset audit           -> reports/dataset_audit.md
python main.py preprocess          # clean data + scaler      -> data/processed/
python main.py baseline            # persistence scores (no training needed)
python main.py train               # 4 LSTM variants x 5 seeds -> models/experiment/, pick on validation
python main.py evaluate            # selected LSTM vs persistence on test -> reports/results.md
python main.py predict             # forecast 2020-07-02 (day after the data ends)
python main.py predict --date 2020-01-15   # forecast a past day and compare with the real AQI
python -m unittest discover -s tests -v    # research-rule checks (no leakage etc.)
```

Add `--device mps` to `train` / `evaluate` / `predict` to use the Apple GPU. The default is
`cpu` because it gives exactly the same numbers on every run.

## Project structure

```text
data/raw/city_day.csv            original Kaggle file (never modified)
data/processed/                  cleaned Kolkata table (with *_imputed flags) + scaler.json
models/experiment/               one trained model per (variant, seed) + training history
models/validation_runs.csv       validation score of every run
models/selected_model.json       which variant was chosen, and the rule used
reports/                         audit, preprocessing decisions, results, figures
src/config.py                    every setting in one place (dates, features, model size)
src/inspect_data.py              dataset audit
src/data_loader.py               reads the raw CSV
src/preprocessing.py             cleaning, chronological split, train-only scaling
src/dataset.py                   14-day history -> next-day AQI samples
src/baseline.py                  persistence baseline
src/model.py                     LSTM model
src/train.py                     training with early stopping
src/evaluate.py                  MAE / RMSE / MAPE / R2 / bias + alert precision/recall/F1
src/predict.py                   single-day forecast + alert
tests/test_pipeline.py           checks for leakage, splits, imputation flags, metrics
main.py                          command-line entry point
```

## Research rules this code follows

- Raw data is never changed; cleaned data is saved separately.
- No random train/test split: train (2018-06-16 to 2019-09-30), validation
  (2019-10-01 to 2019-12-31), test (2020-01-01 to 2020-07-01), all by date.
- Inputs use only days *before* the forecast day; the scaler is fitted on training data only.
- AQI (the target) is never filled in. The single missing input value (O3 on 2018-06-17)
  is copied from the previous day and flagged as imputed.
- No weather or external data is used.
- Results are reported against the persistence baseline on the same days, including
  when the LSTM does worse.
- Model choice uses the validation period only: 4 variants (predict AQI directly vs
  predict the change from today; 2x64 vs 1x32 LSTM) x 5 random seeds, lowest mean
  validation MAE wins. Alert thresholds are also tuned on validation. The test period
  is scored once, after every choice is made.
- The test comparison includes a 95% block-bootstrap interval for the difference in
  MAE, so a small difference is not mistaken for a real one.

## Known limitations

- Only about 2 years of Kolkata data (two winters).
- The test period includes the COVID-19 lockdown (from 2020-03-25).
- The dataset ends in July 2020, so `predict` is a demonstration on historical data,
  not a live forecast.
- Results come from one chronological split (5 seeds per variant).
