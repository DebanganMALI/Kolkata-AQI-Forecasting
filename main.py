"""
Kolkata Air Quality Forecasting - command-line entry point.

Run from the project root (with the .venv activated):

    python main.py audit        # dataset audit -> reports/dataset_audit.md
    python main.py preprocess   # clean table + scaler + decision log
    python main.py baseline     # persistence baseline scores (no PyTorch needed)
    python main.py train        # train 4 LSTM variants x 5 seeds, pick one on validation
    python main.py evaluate     # selected LSTM vs persistence on test -> reports/results.md
    python main.py predict      # forecast the day after the dataset ends
    python main.py predict --date 2020-01-15   # forecast a past date and compare
    python main.py all          # everything above, in order (the full demo)
    python main.py compare      # 5 model families, with and without weather -> reports/model_comparison.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make "from src import ..." work when run as `python main.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Kolkata next-day AQI forecasting")
    parser.add_argument("command", choices=["audit", "preprocess", "baseline", "train",
                                            "evaluate", "predict", "all", "compare"])
    parser.add_argument("--date", help="for predict: date to forecast (YYYY-MM-DD)")
    parser.add_argument("--device", default=config.DEVICE, choices=["cpu", "cuda", "mps", "auto"],
                        help="auto (default: NVIDIA GPU if available, else CPU), cuda, cpu or mps")
    args = parser.parse_args()
    cmd = args.command

    if cmd in ("audit", "all"):
        from src.inspect_data import print_summary, run_audit, write_report
        print("\n### 1. Dataset audit")
        f = run_audit(config.RAW_CSV)
        write_report(f)
        print_summary(f)

    if cmd in ("preprocess", "all"):
        from src.preprocessing import run_preprocessing
        print("\n### 2. Preprocessing")
        run_preprocessing()

    if cmd in ("baseline", "all"):
        from src.evaluate import evaluate_baseline
        print("\n### 3. Persistence baseline (tomorrow = today)")
        evaluate_baseline()

    if cmd in ("train", "all"):
        from src.train import train_all
        print("\n### 4. Training the LSTM variants (choice made on validation only)")
        train_all(device_name=args.device)

    if cmd in ("evaluate", "all"):
        from src.evaluate import evaluate
        print("\n### 5. Evaluation: selected LSTM vs persistence")
        evaluate(device_name=args.device)

    if cmd in ("predict", "all"):
        from src.predict import forecast, print_forecast
        print("\n### 6. Forecast demonstration")
        if cmd == "all":
            print_forecast(forecast("2020-01-15", device_name=args.device))  # a winter test day
            print_forecast(forecast(None, device_name=args.device))           # day after data ends
        else:
            print_forecast(forecast(args.date, device_name=args.device))

    if cmd == "compare":
        from src.compare import run_comparison
        print("\n### 7. Model comparison: 5 forecasters, with and without weather")
        run_comparison(device_name=args.device)


if __name__ == "__main__":
    main()
