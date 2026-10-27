"""Bar-by-bar backtester for the BTC grid martingale EA (martingale/BTC_Martingale_Grid.mq5).

The logic mirrors the EA:
  * Open a first trade (direction from an EMA trend filter, or fixed long/short).
  * Each time price moves `step` against the basket, add a layer with lot * multiplier
    (step itself can widen by `step_mult` per layer). At most `max_layers` trades.
  * Close the whole basket when price reaches the volume-weighted average entry +/- `tp`.
  * Optional equity stop: close the basket when floating loss >= `equity_sl` x equity at basket start.
  * Broker stop-out: when margin level (equity / used margin) falls below `stop_out`, every
    position is closed at the current price. That is the "account broke" event.

Cent account: balance and P&L are in US cents (USC). On a cent account BTCUSDc 1 lot = 1 BTC and a
$1 move on 0.01 lot pays 0.01 USC, i.e. the account behaves like a USD account 100x the real money.

Intrabar path on 5m candles: bullish bar O->L->H->C, bearish bar O->H->L->C.
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "data", "BTCUSDT_5m.csv.gz")

DEFAULTS = dict(
    balance=100_000.0,     # USC (100k cents = $1,000 real)
    start_lot=0.01,        # first layer lots (1 lot = 1 BTC)
    auto_lot_per=0.0,      # if >0: start_lot = 0.01 per `auto_lot_per` USC of balance (compounding)
    lot_mult=1.5,          # martingale multiplier
    max_layers=14,
    step_pct=1.5,          # first grid distance, % of entry price
    step_mult=1.0,         # each further gap = previous gap * step_mult
    tp_pct=1.0,            # basket take profit, % above/below average entry
    equity_sl=0.0,         # 0 = off; 0.10 = close basket at -10% of equity
    direction="trend",     # trend | long | short
    ema_bars=2400,         # EMA length in 5m bars for the trend filter (2400 = ~200 hours)
    leverage=200.0,
    stop_out=0.5,          # margin level that triggers broker stop-out
    spread=15.0,           # USD per BTC, paid on every entry
    swap_pct_day=0.02,     # % of notional charged per day held (long and short)
    cooldown_bars=0,       # bars to wait after a basket closes
    vol_pause_pct=0.0,     # >0: no new basket while the last 24h high-low range exceeds this % of price
)


def load(path=DATA):
    df = pd.read_csv(path, parse_dates=["time"])
    return df


def run(df, **kw):
    p = {**DEFAULTS, **kw}
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = df["time"].to_numpy()
    n = len(c)
    alpha = 2 / (p["ema_bars"] + 1)
    ema = pd.Series(c).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    # last completed 24h range as % of price (shifted one bar so it never peeks at the current bar)
    rng24 = ((pd.Series(h).rolling(288, 1).max() - pd.Series(l).rolling(288, 1).min()) / pd.Series(c) * 100).shift(1).fillna(0).to_numpy()

    bal = p["balance"]
    lots, prices = [], []
    side = 0
    next_gap = 0.0
    basket_start_eq = bal
    cooldown = 0
    swap_per_bar = p["swap_pct_day"] / 100 / 288

    peak_eq, max_dd, max_dd_pct = bal, 0.0, 0.0
    baskets = wins = sl_hits = stopouts = 0
    max_layers_used = 0
    layer_hist = np.zeros(p["max_layers"] + 1, int)
    eq_curve = np.empty(n)
    events = []
    blown = False
    layer0 = 0.01

    V = VP = 0.0  # running sum of lots and of lots * entry price

    def floating(px):
        return side * (V * px - VP) if lots else 0.0

    def close_all(px, why, i):
        nonlocal bal, lots, prices, side, baskets, wins, sl_hits, stopouts, cooldown, V, VP
        pnl = floating(px)
        bal += pnl
        baskets += 1
        layer_hist[len(lots)] += 1
        if why == "tp":
            wins += 1
        elif why == "sl":
            sl_hits += 1
            events.append((str(t[i])[:16], "equity SL", round(pnl), len(lots)))
        elif why == "stopout":
            stopouts += 1
            events.append((str(t[i])[:16], "STOP-OUT", round(pnl), len(lots)))
        lots, prices, side = [], [], 0
        V = VP = 0.0
        cooldown = p["cooldown_bars"]

    def avg_price():
        return VP / V

    def lot_round(x):
        # broker volume step 0.01, half-up (0.015 -> 0.02), as MathRound does in the EA
        return max(0.01, math.floor(x * 100 + 0.5 + 1e-9) / 100)

    def base_lot():
        if p["auto_lot_per"] > 0:
            return max(0.01, math.floor(bal / p["auto_lot_per"]) * 0.01)
        return p["start_lot"]

    for i in range(n):
        if blown:
            eq_curve[i] = bal
            continue
        # open new basket at bar open
        if not lots:
            if cooldown > 0:
                cooldown -= 1
            elif p["vol_pause_pct"] > 0 and rng24[i] > p["vol_pause_pct"]:
                pass
            else:
                d = p["direction"]
                side = 1 if d == "long" else -1 if d == "short" else (1 if c[i - 1 if i else 0] >= ema[i - 1 if i else 0] else -1)
                lt = lot_round(base_lot())
                layer0 = lt
                if lt * o[i] / p["leverage"] > bal * 0.9:
                    blown = True
                    eq_curve[i] = bal
                    continue
                lots, prices = [lt], [o[i] + side * p["spread"] / 2]
                V, VP = lt, lt * prices[0]
                next_gap = o[i] * p["step_pct"] / 100
                basket_start_eq = bal

        # swap
        if lots:
            bal -= V * c[i] * swap_per_bar

        path = (o[i], l[i], h[i], c[i]) if c[i] >= o[i] else (o[i], h[i], l[i], c[i])
        for k, px in enumerate(path[1:]):
            prev = path[k]
            if not lots:
                break
            # adds triggered on the adverse move (can be several layers in one leg)
            while len(lots) < p["max_layers"]:
                last = prices[-1] - side * p["spread"] / 2
                trigger = last - side * next_gap
                if (side == 1 and px <= trigger) or (side == -1 and px >= trigger):
                    # layer k lot = first lot * mult^k (not chained, so rounding never stalls growth)
                    lt = lot_round(layer0 * p["lot_mult"] ** len(lots))
                    lots.append(lt)
                    prices.append(trigger + side * p["spread"] / 2)
                    V += lt
                    VP += lt * prices[-1]
                    next_gap *= p["step_mult"]
                else:
                    break
            max_layers_used = max(max_layers_used, len(lots))
            fl = floating(px)
            eq = bal + fl
            margin = V * px / p["leverage"]
            # stops fill at the price where they trigger (an EA watches every tick), or at the
            # start of the leg if price was already past it (gap)
            worse = min if side == 1 else max
            if eq / margin < p["stop_out"]:
                so_px = (side * VP - bal) / (side * V - p["stop_out"] * V / p["leverage"])
                close_all(worse(so_px, prev), "stopout", i)
                if bal < p["balance"] * 0.05:
                    blown = True
                break
            if p["equity_sl"] > 0 and -fl >= p["equity_sl"] * basket_start_eq:
                sl_px = (VP - p["equity_sl"] * basket_start_eq * side) / V
                close_all(worse(sl_px, prev), "sl", i)
                break
            tp_px = avg_price() + side * avg_price() * p["tp_pct"] / 100
            if (side == 1 and px >= tp_px) or (side == -1 and px <= tp_px):
                close_all(tp_px, "tp", i)
                break

        eq = bal + floating(c[i])
        # worst intrabar equity for drawdown
        worst = bal + (floating(l[i]) if side == 1 else floating(h[i]) if side == -1 else 0)
        peak_eq = max(peak_eq, eq)
        dd = peak_eq - min(eq, worst)
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd / peak_eq
        eq_curve[i] = eq

    final = eq_curve[-1]
    years = (pd.Timestamp(t[-1]) - pd.Timestamp(t[0])).days / 365.25
    cagr = (final / p["balance"]) ** (1 / years) - 1 if final > 0 else -1.0
    return dict(
        params=p,
        start=str(t[0])[:10], end=str(t[-1])[:10], years=round(years, 2),
        final_balance=round(final), net_profit=round(final - p["balance"]),
        return_pct=round((final / p["balance"] - 1) * 100, 1),
        cagr_pct=round(cagr * 100, 1),
        monthly_pct=round(((1 + cagr) ** (1 / 12) - 1) * 100, 2) if cagr > -1 else -100.0,
        max_dd_pct=round(max_dd_pct * 100, 1),
        baskets=baskets, wins=wins, equity_sl_hits=sl_hits, stop_outs=stopouts,
        blown=bool(blown or final < p["balance"] * 0.05),
        max_layers_used=max_layers_used,
        layer_hist={k: int(v) for k, v in enumerate(layer_hist) if v},
        events=events[:30],
    ), pd.Series(eq_curve, index=pd.to_datetime(t))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    a = ap.parse_args()
    res, _ = run(load(), **vars(a))
    print(json.dumps(res, indent=2, default=str))


def break_move(price=100_000.0, balance=100_000.0, **kw):
    """Adverse move (% from the first entry) that fills every layer, and the move that
    triggers broker stop-out with no SL. Ignores spread/swap; long side (short is symmetric)."""
    p = {**DEFAULTS, **kw}
    lot0 = max(0.01, math.floor(p["start_lot"] * 100 + 0.5 + 1e-9) / 100)
    lots, prices, gap, px = [], [], price * p["step_pct"] / 100, price
    for k in range(p["max_layers"]):
        lots.append(max(0.01, math.floor(lot0 * p["lot_mult"] ** k * 100 + 0.5 + 1e-9) / 100))
        prices.append(px)
        px -= gap
        gap *= p["step_mult"]
    V = sum(lots)
    VP = sum(a * b for a, b in zip(lots, prices))
    last = prices[-1]
    # stop-out: balance + V*x - VP = stop_out * V * x / leverage
    so_px = (VP - balance) / (V - p["stop_out"] * V / p["leverage"])
    return dict(total_lots=round(V, 2), last_layer_lot=lots[-1],
                all_layers_filled_at_pct=round((1 - last / price) * 100, 1),
                stop_out_at_pct=round((1 - so_px / price) * 100, 1) if so_px > 0 else 100.0,
                stop_out_before_last_layer=so_px > last)
