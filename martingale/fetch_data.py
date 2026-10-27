"""Download ~3 years of BTCUSDT 5-minute candles from Binance public data.

Uses data.binance.vision monthly archives (works from US-hosted runners where
api.binance.com is geo-blocked), then daily archives for the current month.
Writes martingale/data/BTCUSDT_5m.csv.gz with columns time,open,high,low,close,volume.
"""
import io
import os
import zipfile
from datetime import date, timedelta

import pandas as pd
import requests

SYMBOL, INTERVAL = "BTCUSDT", "5m"
BASE = "https://data.binance.vision/data/spot"
OUT = os.path.join(os.path.dirname(__file__), "data", f"{SYMBOL}_{INTERVAL}.csv.gz")


def fetch_zip(url):
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        return pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=range(6))


def main():
    today = date.today()
    start = date(today.year - 3, today.month, 1)
    frames, d = [], start
    while (d.year, d.month) < (today.year, today.month):
        df = fetch_zip(f"{BASE}/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{d:%Y-%m}.zip")
        if df is None:  # month archive not published yet, fall back to daily files
            dd = d
            while dd.month == d.month and dd < today:
                x = fetch_zip(f"{BASE}/daily/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{dd:%Y-%m-%d}.zip")
                if x is not None:
                    frames.append(x)
                dd += timedelta(days=1)
        else:
            frames.append(df)
        print(d, "ok" if df is not None else "daily")
        d = date(d.year + d.month // 12, d.month % 12 + 1, 1)
    d = date(today.year, today.month, 1)
    while d < today:
        x = fetch_zip(f"{BASE}/daily/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{d:%Y-%m-%d}.zip")
        if x is not None:
            frames.append(x)
        d += timedelta(days=1)

    df = pd.concat(frames, ignore_index=True)
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    # Binance switched spot timestamps to microseconds in 2025
    df["time"] = df["time"].where(df["time"] < 1e14, df["time"] // 1000)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.drop_duplicates("time").sort_values("time")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.2f")
    print(len(df), "rows", df["time"].min(), "->", df["time"].max())


if __name__ == "__main__":
    main()
