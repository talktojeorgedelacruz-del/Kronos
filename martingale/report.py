"""Run the named scenarios, save equity charts and a results table (martingale/results/)."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import backtest as b  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results")

SCENARIOS = {
    "A_14L_SL10": dict(equity_sl=0.10),
    "B_14L_noSL": dict(equity_sl=0.0),
}


def yearly(eq):
    y = eq.resample("YE").last()
    start = pd.concat([pd.Series([eq.iloc[0]], index=[eq.index[0]]), y.iloc[:-1]])
    return {str(k.year): round((v / s - 1) * 100, 1) if s > 0 else None for k, v, s in zip(y.index, y.values, start.values)}


def main(scenarios=SCENARIOS):
    os.makedirs(OUT, exist_ok=True)
    df = b.load()
    rows = []
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, kw in scenarios.items():
        res, eq = b.run(df, **kw)
        res["yearly_pct"] = yearly(eq)
        res["break_test"] = b.break_move(price=60_000, **kw)
        with open(os.path.join(OUT, f"{name}.json"), "w") as f:
            json.dump(res, f, indent=2, default=str)
        ax.plot(eq.resample("1D").last().index, eq.resample("1D").last().values / 100, label=name, lw=1.2)
        rows.append({"scenario": name, **{k: res[k] for k in (
            "final_balance", "return_pct", "monthly_pct", "max_dd_pct", "baskets",
            "equity_sl_hits", "stop_outs", "blown", "max_layers_used")}})
    ax.set_ylabel("Equity, USD (100,000 USC = $1,000)")
    ax.set_title("BTC grid martingale, 5m BTCUSDT, Oct 2023 to Oct 2026")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "equity_curves.png"), dpi=120)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
