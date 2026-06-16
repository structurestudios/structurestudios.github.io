# Momentum Backtester — honest by construction

A small momentum-breakout backtester whose entire design goal is **not to fool
you**. Anyone can produce a green equity curve; the hard part is knowing whether
it's a real edge or an artifact of survivorship bias, overfitting, ignored
costs, or plain luck. This project bakes the checks for all four into the
pipeline.

## The strategy (deliberately simple)

Long-only momentum breakout, one position per symbol:

- **Regime filter** — only trade while the index closes above its 50-day MA.
- **Entry** — a close above the prior 50-day high.
- **Risk** — initial stop at 2×ATR below entry; that distance is **1R**.
- **Exit** — a ratcheting (never-loosening) 3×ATR trailing stop.
- **Result** — each trade recorded as an R-multiple: `(exit − entry) / risk`.

Simple on purpose: fewer knobs means less surface area to overfit.

## The four traps, and how each is handled

| Trap | Handling |
|------|----------|
| **Survivorship bias** | The Norgate adapter uses a *"Current & Past"* watchlist that **includes delisted stocks**. Testing only survivors is the classic beautiful lie. |
| **Transaction costs** | `find_trades` charges commission/spread (bps) + slippage (ATR) on entry *and* exit, as a per-trade R haircut. Tight stops are penalized more, as in reality. |
| **Luck** | `momentum_significance.py` runs a bootstrap CI, a sign-flip permutation p-value, and a Monte-Carlo equity/drawdown sim. |
| **Overfitting** | `momentum_lab.py` sweeps the strategy parameters across a grid. A real edge is a broad plateau; a lone positive peak is noise. Multiple-testing is corrected with a Bonferroni threshold. |

## Files

| File | Role |
|------|------|
| `momentum_backtester.py` | Core engine: `regime_ok`, `find_trades` (net of costs), `summarize`. |
| `momentum_backtester_norgate.py` | Survivorship-bias-free data adapter. Writes `momentum_trades_R.csv` and caches raw data to `universe_cache.pkl`. |
| `momentum_significance.py` | Real-or-luck testing on the R-multiples. |
| `momentum_lab.py` | Cost-robustness sweep + parameter-robustness grid (overfit detector). Runs on the cache, or on a synthetic demo universe if none exists. |
| `test_momentum.py` | Pytest suite. Locks the no-lookahead guarantee, R-multiple math, cost monotonicity, and the significance tools. |

## Usage

```bash
pip install -r requirements.txt

# 1) Run the survivorship-bias-free backtest (needs Norgate Data Updater).
python momentum_backtester_norgate.py
#    -> momentum_trades_R.csv, universe_cache.pkl

# 2) Is it real or luck?
python momentum_significance.py

# 3) Is it robust, or overfit / killed by costs?
python momentum_lab.py            # uses the cache from step 1

# Verify the engine itself:
pytest -q
```

No Norgate subscription? `python momentum_lab.py` runs immediately on a
synthetic universe (clearly labelled) so you can see the methodology work.

## The standing caveat

Every guard here checks that a result isn't an *obvious* artifact. None of them
can manufacture out-of-sample truth. A free-trial window of ~2 years is roughly
one market regime; a clean p-value and a broad parameter plateau over that
window are **necessary but not sufficient** for a durable edge. Treat a positive
result as a hypothesis that survived its first serious attacks — not as a
discovered law.
