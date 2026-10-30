"""Download BTCUSDT 15-minute candles from Binance's public data archive.

Writes martingale/data/btcusdt_15m.csv covering the last ~3 years.
"""
import io
import zipfile
from datetime import date
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

BASE = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-{y}-{m:02d}.zip"
OUT = Path(__file__).parent / "data" / "btcusdt_15m.csv"
START = (2023, 10)


def months(start, end):
    y, m = start
    while (y, m) <= end:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def main():
    today = date.today()
    end = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    frames = []
    for y, m in months(START, end):
        url = BASE.format(y=y, m=m)
        try:
            raw = urlopen(url, timeout=60).read()
        except Exception as exc:  # month not published yet
            print(f"skip {y}-{m:02d}: {exc}")
            continue
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            df = pd.read_csv(zf.open(zf.namelist()[0]), header=None, usecols=range(6))
        df.columns = ["open_time", "open", "high", "low", "close", "volume"]
        # Binance switched spot timestamps to microseconds in 2025.
        unit = "us" if df["open_time"].iloc[0] > 1e14 else "ms"
        df["timestamps"] = pd.to_datetime(df["open_time"], unit=unit)
        frames.append(df[["timestamps", "open", "high", "low", "close", "volume"]])
        print(f"got {y}-{m:02d}: {len(df)} rows")
    out = pd.concat(frames).drop_duplicates("timestamps").sort_values("timestamps")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)} rows to {OUT}")


if __name__ == "__main__":
    main()
