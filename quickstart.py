"""Quick start: forecast the next candles of a price series with Kronos.

Usage:
    python quickstart.py                         # uses the bundled sample data
    python quickstart.py --csv my_prices.csv     # your own OHLCV data

The CSV needs a `timestamps` column plus `open`, `high`, `low`, `close`
(`volume` and `amount` are optional). Model weights are downloaded from the
Hugging Face Hub on first run.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from model import Kronos, KronosPredictor, KronosTokenizer

DEFAULT_CSV = Path(__file__).parent / "tests" / "data" / "regression_input.csv"
FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def main():
    parser = argparse.ArgumentParser(description="Forecast OHLCV candles with Kronos.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to a CSV of historical candles.")
    parser.add_argument("--model", default="NeoQuasar/Kronos-small", help="Hugging Face model id.")
    parser.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base", help="Hugging Face tokenizer id.")
    parser.add_argument("--lookback", type=int, default=400, help="Number of past candles fed to the model (max 512).")
    parser.add_argument("--pred-len", type=int, default=24, help="Number of future candles to forecast.")
    parser.add_argument("--samples", type=int, default=1, help="Forecast paths to generate and average.")
    parser.add_argument("--out", default="forecast.csv", help="Where to write the forecast.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["timestamps"])
    cols = [c for c in FEATURES if c in df.columns]
    if len(df) < args.lookback:
        raise SystemExit(f"Need at least {args.lookback} rows, got {len(df)}.")

    history = df.tail(args.lookback).reset_index(drop=True)
    x_timestamp = history["timestamps"]
    # Future timestamps continue at the same spacing as the last two candles.
    step = x_timestamp.iloc[-1] - x_timestamp.iloc[-2]
    y_timestamp = pd.Series([x_timestamp.iloc[-1] + step * (i + 1) for i in range(args.pred_len)])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    pred_df = predictor.predict(
        df=history[cols],
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=args.pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=args.samples,
        verbose=False,
    )

    pred_df.to_csv(args.out)
    print(f"Last close: {history['close'].iloc[-1]:.4f}")
    print(pred_df[["open", "high", "low", "close"]].head(10))
    print(f"Saved {len(pred_df)} forecast rows to {args.out}")


if __name__ == "__main__":
    main()
