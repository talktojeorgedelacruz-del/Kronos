"""Backtest scalping off daily BTC buy/sell zones (5m BTCUSDT).

Every day at 00:00 UTC, using only data up to that moment, build the zones the way they were
given in the thread:
  * 4H swing highs / lows (fractal, 2 bars each side) from the last 30 days
  * volume-profile points of control for the last 7, 30 and 90 days
  * daily EMA200
Levels below the day's open are buy zones, above are sell zones.

Entry: a limit order at each level. Long when price trades down into a buy level, short when it
trades up into a sell level. Each level fires at most once per day, one trade open at a time.
Exit: fixed TP / SL in USD per BTC. If TP and SL are both inside the same 5m bar, count it as a loss.
Optional bias filter: only longs when the day opens above the 4H EMA200, only shorts below it.
Baseline: the same TP/SL entered at random times, to see whether the zones add anything.
"""
import argparse
import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "data", "BTCUSDT_5m.csv.gz")


def poc(sub, bins=80):
    tp = (sub.high + sub.low + sub.close) / 3
    h, e = np.histogram(tp, bins=bins, weights=sub.volume)
    k = h.argmax()
    return (e[k] + e[k + 1]) / 2


def daily_levels(df):
    h4 = df.resample("4h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    is_ph = h4.high == h4.high.rolling(5, center=True).max()
    is_pl = h4.low == h4.low.rolling(5, center=True).min()
    h4_ema200 = h4.close.ewm(span=200, adjust=False).mean()
    d_close = df.close.resample("1D").last()
    d_ema200 = d_close.ewm(span=200, adjust=False).mean()
    out = {}
    for day in pd.date_range(df.index[0].normalize() + pd.Timedelta(days=91), df.index[-1].normalize(), freq="D"):
        # a 4H pivot is only known 2 bars (8h) after it prints
        known = h4.index < day - pd.Timedelta(hours=8)
        recent = known & (h4.index >= day - pd.Timedelta(days=30))
        lv = list(h4.high[recent & is_ph]) + list(h4.low[recent & is_pl])
        hist = df[df.index < day]
        for n in (7, 30, 90):
            lv.append(poc(hist[hist.index >= day - pd.Timedelta(days=n)]))
        lv.append(d_ema200[d_ema200.index < day].iloc[-1])
        bias = 1 if hist.close.iloc[-1] > h4_ema200[h4_ema200.index < day].iloc[-1] else -1
        out[day] = (sorted(set(round(x, 2) for x in lv)), bias)
    return out


def run(df, levels, tp, sl, spread=15.0, bias_filter=False, random_entries=0, seed=0):
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    days = df.index.normalize()
    rng = np.random.default_rng(seed)
    trades = []
    pos = None  # (side, entry, tp_px, sl_px, time)
    cur_day, todays, used, bias = None, [], set(), 0
    for i in range(len(df)):
        if days[i] != cur_day:
            cur_day = days[i]
            todays, bias = levels.get(cur_day, ([], 0))
            day_open = o[i]
            used = set()
        if not todays:
            continue
        if pos is None:
            if random_entries:
                if rng.random() < random_entries:
                    side = 1 if rng.random() < 0.5 else -1
                    pos = (side, o[i], o[i] + side * tp, o[i] - side * sl, df.index[i])
            else:
                for lv in todays:
                    if lv in used:
                        continue
                    side = 1 if lv < day_open else -1
                    if bias_filter and side != bias:
                        continue
                    if (side == 1 and l[i] <= lv < o[i]) or (side == -1 and h[i] >= lv > o[i]):
                        used.add(lv)
                        pos = (side, lv, lv + side * tp, lv - side * sl, df.index[i])
                        break
            if pos is None:
                continue
        side, entry, tp_px, sl_px, t0 = pos
        hit_sl = (l[i] <= sl_px) if side == 1 else (h[i] >= sl_px)
        hit_tp = (h[i] >= tp_px) if side == 1 else (l[i] <= tp_px)
        # on the entry bar only count moves after the fill: require the close beyond TP
        if df.index[i] == t0 and hit_tp:
            hit_tp = (c[i] >= tp_px) if side == 1 else (c[i] <= tp_px)
        if hit_sl or hit_tp:
            pnl = (-sl if hit_sl else tp) - spread
            trades.append((t0, side, entry, pnl, (df.index[i] - t0).total_seconds() / 60))
            pos = None
    t = pd.DataFrame(trades, columns=["time", "side", "entry", "pnl_usd_per_btc", "minutes"])
    return t


def summarize(t, label):
    if t.empty:
        return {"setup": label, "trades": 0}
    w = t.pnl_usd_per_btc > 0
    gp, gl = t.pnl_usd_per_btc[w].sum(), -t.pnl_usd_per_btc[~w].sum()
    months = max((t.time.iloc[-1] - t.time.iloc[0]).days / 30.4, 1)
    return {
        "setup": label, "trades": len(t), "per_day": round(len(t) / (months * 30.4), 1),
        "win_rate": round(w.mean() * 100, 1), "profit_factor": round(gp / gl, 2) if gl else None,
        "net_usd_per_1lot": round(t.pnl_usd_per_btc.sum()),
        "net_usc_per_0.01lot": round(t.pnl_usd_per_btc.sum() * 0.01, 1),
        "avg_minutes": round(t.minutes.median()),
        "worst_losing_streak": int((~w).astype(int).groupby(w.cumsum()).sum().max()),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    a = ap.parse_args()
    df = pd.read_csv(DATA, parse_dates=["time"]).set_index("time")
    levels = daily_levels(df)
    start = df.index[-1].normalize() - pd.Timedelta(days=a.days)
    test = df[df.index >= start]
    rows = []
    for tp, sl in [(120, 120), (120, 240), (200, 200), (200, 400), (200, 100)]:
        for bf in (False, True):
            t = run(test, levels, tp, sl, bias_filter=bf)
            rows.append({"tp": tp, "sl": sl, **summarize(t, "zones + bias" if bf else "zones")})
        r = run(test, levels, tp, sl, random_entries=0.01)
        rows.append({"tp": tp, "sl": sl, **summarize(r, "random entry")})
    print(f"Test window: {test.index[0]} -> {test.index[-1]}, spread $15/BTC included")
    print(pd.DataFrame(rows).to_string(index=False))
