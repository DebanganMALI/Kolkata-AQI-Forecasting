"""Command-line entry point: runs each stage of the project (audit, preprocess, train, evaluate, experiments)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the src package importable when run from the project folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config


def main() -> None:
    """Parse the command and run the matching project stage."""
    parser = argparse.ArgumentParser(description="Kolkata next-day AQI forecasting")
    parser.add_argument("command", choices=["audit", "preprocess", "baseline", "train",
                                            "evaluate", "predict", "all", "compare", "advanced", "extended", "robust", "live-train"])
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
            print_forecast(forecast("2020-01-15", device_name=args.device))
            print_forecast(forecast(None, device_name=args.device))
        else:
            print_forecast(forecast(args.date, device_name=args.device))

    if cmd == "compare":
        from src.compare import run_comparison
        print("\n### 7. Model comparison: 5 forecasters, with and without weather")
        run_comparison(device_name=args.device)

    if cmd == "advanced":
        from src.advanced import run_advanced
        print("\n### 8. Advanced deep learning: attention, TCN, Transformer, forecast ranges")
        run_advanced(device_name=args.device)

    if cmd == "extended":
        from src.extended import run_extended
        print("\n### 9. Extended data 2018-2026: 8 forecasters x 3 feature sets")
        run_extended(device_name=args.device)

    if cmd == "robust":
        from src.robust import run_robust
        print("\n### 10. Walk-forward rounds and spike-focused models")
        run_robust(device_name=args.device)

    if cmd == "live-train":
        from src.live_model import run_live_training
        print("\n### 11. Live GRU for the website")
        run_live_training(device_name=args.device)


if __name__ == "__main__":
    main()
