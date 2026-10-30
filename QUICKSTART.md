# Kronos quick start

Kronos is an open-source foundation model that forecasts financial candlestick (OHLCV) data.
This fork adds a one-command forecast script and a GitHub Actions check that installs Kronos
and runs it on every push.

## 1. Install

You need Python 3.10 or newer.

```bash
git clone https://github.com/talktojeorgedelacruz-del/kronos.git
cd kronos
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No GPU? Install the smaller CPU build of PyTorch first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`

## 2. Run a forecast

```bash
python quickstart.py
```

This loads `Kronos-small` from Hugging Face (downloaded once, about 100 MB), reads the bundled
sample data in `tests/data/regression_input.csv`, forecasts the next 24 candles and writes them
to `forecast.csv`.

To use your own data, give it a CSV with a `timestamps` column and `open`, `high`, `low`,
`close` columns (`volume` and `amount` are optional):

```bash
python quickstart.py --csv my_prices.csv --pred-len 48 --samples 5
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--lookback` | 400 | Past candles the model sees (max 512) |
| `--pred-len` | 24 | Future candles to forecast |
| `--samples` | 1 | Forecast paths to average (higher is smoother, slower) |
| `--model` | `NeoQuasar/Kronos-small` | Also `NeoQuasar/Kronos-base`, or `NeoQuasar/Kronos-mini` with `--tokenizer NeoQuasar/Kronos-Tokenizer-2k` |
| `--out` | `forecast.csv` | Output file |

## 3. Check it on GitHub

The **Install and run Kronos** workflow (Actions tab) installs everything on a clean machine,
runs `quickstart.py`, runs the regression tests, and attaches `forecast.csv` to the run.
You can also start it by hand with **Run workflow**.

## Next steps

- `examples/` has more scripts, including batch prediction and a backtest.
- `webui/` has a browser interface (`cd webui && python run.py`).
- `finetune/` and `finetune_csv/` show how to fine-tune on your own markets.

Forecasts are probabilistic and are not financial advice. Backtest before trading on them.
