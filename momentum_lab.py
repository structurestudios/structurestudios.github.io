"""
Momentum Backtester — the lab (overfit & cost robustness)
=========================================================
A single positive backtest is a story, not evidence. This lab runs the two
tests that actually separate a tradeable edge from a curve-fit mirage:

  A) COST ROBUSTNESS SWEEP
     Re-run the whole universe at rising cost levels. An edge that only exists
     at zero friction is not an edge you can trade. We want to see HOW FAST the
     expectancy and its significance decay as costs climb.

  B) PARAMETER ROBUSTNESS GRID (the overfit detector)
     Vary the strategy's knobs (breakout lookback, stop/trail ATR multiples)
     across a grid. If the edge is real it should be a broad PLATEAU -- positive
     across most reasonable settings. If only a lonely peak is positive, you've
     found noise you fit to, not a market inefficiency.

     Because scanning a grid is multiple hypothesis testing, a low p-value at the
     single best combo is expected BY CHANCE. We therefore report a Bonferroni
     threshold (0.05 / number_of_combos) so you judge the best cell honestly.

DATA:
  Prefers the cache written by momentum_backtester_norgate.py
  (universe_cache.pkl). If that's missing, it FALLS BACK to a synthetic universe
  so the lab runs anywhere -- clearly labelled as a demo, not a real result.

USAGE:
    python momentum_lab.py                  # cache if present, else synthetic
    python momentum_lab.py universe_cache.pkl
"""

import os
import pickle
import sys
import itertools

import numpy as np
import pandas as pd

from momentum_backtester import find_trades, regime_ok
from momentum_significance import signflip_pvalue, bootstrap_mean_ci

CACHE_FILE = "universe_cache.pkl"
SEED = 42
GRID_RESAMPLES = 5000   # fewer than the headline test; we run it many times


# ----------------------------------------------------------------------------
# Data: load the real cache, or synthesize a labelled demo universe
# ----------------------------------------------------------------------------
def _ohlc_from_close(close, dates, wiggle=0.01):
    """Build a plausible OHLC frame from a close path."""
    c = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c * (1 + wiggle), "low": c * (1 - wiggle),
         "close": c, "volume": 1.0},
        index=dates,
    )


def synthetic_universe(n_symbols=200, n_days=500, seed=SEED):
    """A demo universe with a MILD, real momentum edge embedded.

    Returns (data_dict, regime). Daily returns carry slight positive
    autocorrelation and a regime-dependent drift, so breakouts follow through
    often enough to be profitable -- but not so cleanly that costs can't bite.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")

    # index: trend up ~70% of the window, then down -> a real regime cycle
    up = int(n_days * 0.7)
    idx_close = np.concatenate(
        [np.linspace(100, 175, up), np.linspace(175, 130, n_days - up)]
    ) * (1 + rng.normal(0, 0.003, n_days)).cumprod()
    regime = regime_ok(_ohlc_from_close(idx_close, dates))
    risk_on = regime.to_numpy()

    data = {}
    for k in range(n_symbols):
        r = np.zeros(n_days)
        prev = 0.0
        for t in range(n_days):
            drift = 0.0008 if risk_on[t] else -0.0010      # regime drift
            shock = rng.normal(0, 0.02)
            r[t] = drift + 0.10 * prev + shock             # mild momentum
            prev = r[t]
        close = 50.0 * np.exp(np.cumsum(r))
        data[f"SYN{k:03d}"] = _ohlc_from_close(close, dates)
    return data, regime


def load_universe(path):
    """Load the cached universe, or fall back to synthetic (clearly labelled)."""
    if path and os.path.exists(path):
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        print(f"Loaded cache: {path}  ({len(blob['data'])} symbols)\n")
        return blob["data"], blob["regime"], False
    print("!! No cache found -- using a SYNTHETIC demo universe.")
    print("!! Numbers below are a methodology demo, NOT a real backtest.\n")
    data, regime = synthetic_universe()
    return data, regime, True


# ----------------------------------------------------------------------------
# Core: run the strategy across the whole universe for one parameter set
# ----------------------------------------------------------------------------
def run_universe(data, regime, **params):
    """Concatenate per-trade R-multiples across every symbol for given params."""
    all_R = []
    for df in data.values():
        all_R += find_trades(df, regime, **params)
    return np.asarray(all_R, dtype=float)


# ----------------------------------------------------------------------------
# A) Cost robustness sweep
# ----------------------------------------------------------------------------
def cost_sweep(data, regime, rng):
    levels = [(0.0, 0.0), (2.0, 0.02), (5.0, 0.05), (15.0, 0.10), (30.0, 0.20)]
    print("=" * 68)
    print("A) COST ROBUSTNESS SWEEP")
    print("-" * 68)
    print(f"{'bps/side':>9} {'slip(ATR)':>10} {'trades':>7} "
          f"{'mean R':>8} {'p-value':>8}  verdict")
    for bps, slip in levels:
        R = run_universe(data, regime, cost_bps_per_side=bps, slippage_atr=slip)
        if len(R) == 0:
            print(f"{bps:>9.1f} {slip:>10.2f} {'0':>7}  (no trades)")
            continue
        _, p = signflip_pvalue(R, rng, n=GRID_RESAMPLES)
        verdict = "edge holds" if (R.mean() > 0 and p < 0.05) else "edge gone"
        print(f"{bps:>9.1f} {slip:>10.2f} {len(R):>7} "
              f"{R.mean():>+8.3f} {p:>8.4f}  {verdict}")
    print("-> A real edge degrades gracefully; a fake one dies at the first bps.")


# ----------------------------------------------------------------------------
# B) Parameter robustness grid (overfit detector)
# ----------------------------------------------------------------------------
def param_grid(data, regime, rng):
    breakout_grid = [20, 50, 100]
    stop_grid     = [1.5, 2.0, 3.0]
    trail_grid    = [2.0, 3.0, 4.0]
    combos = list(itertools.product(breakout_grid, stop_grid, trail_grid))
    m = len(combos)
    bonferroni = 0.05 / m

    print("\n" + "=" * 68)
    print("B) PARAMETER ROBUSTNESS GRID  (overfit detector)")
    print("-" * 68)
    print(f"   {m} combos | Bonferroni-corrected threshold = 0.05/{m} "
          f"= {bonferroni:.4f}")

    rows = []
    for bo, sm, tm in combos:
        R = run_universe(data, regime, breakout_lookback=bo,
                         atr_stop_mult=sm, atr_trail_mult=tm)
        if len(R) == 0:
            continue
        _, p = signflip_pvalue(R, rng, n=GRID_RESAMPLES)
        rows.append((bo, sm, tm, len(R), R.mean(), p))

    if not rows:
        print("   No trades on any combo.")
        return

    res = pd.DataFrame(rows, columns=["breakout", "stop", "trail",
                                      "trades", "mean_R", "p"])
    positive = (res["mean_R"] > 0).mean()
    survive = int((res["p"] < bonferroni).sum())

    print(f"   Positive expectancy in {positive:.0%} of combos "
          f"({(res['mean_R'] > 0).sum()}/{len(res)})")
    print(f"   Median expectancy across grid: {res['mean_R'].median():+.3f}R")
    print(f"   Combos significant AFTER Bonferroni: {survive}/{len(res)}")

    best = res.loc[res["mean_R"].idxmax()]
    worst = res.loc[res["mean_R"].idxmin()]
    print(f"\n   Best  : breakout={int(best.breakout)} stop={best.stop} "
          f"trail={best.trail} -> {best.mean_R:+.3f}R (p={best.p:.4f})")
    print(f"   Worst : breakout={int(worst.breakout)} stop={worst.stop} "
          f"trail={worst.trail} -> {worst.mean_R:+.3f}R (p={worst.p:.4f})")

    print("\n   Reading it:")
    if positive >= 0.8 and survive >= 1:
        print("   Broad plateau of positive results -> robust, not curve-fit.")
    elif positive >= 0.5:
        print("   Mixed: positive on average but param-sensitive. Be skeptical.")
    else:
        print("   Mostly negative; the lone positive peak is likely noise.")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CACHE_FILE
    rng = np.random.default_rng(SEED)
    data, regime, is_synthetic = load_universe(path)

    cost_sweep(data, regime, rng)
    param_grid(data, regime, rng)

    print("\n" + "=" * 68)
    if is_synthetic:
        print("DEMO RUN on synthetic data. Run the Norgate adapter first to")
        print("produce universe_cache.pkl, then re-run for real results.")
    print("Robustness is necessary, not sufficient: it can't fix a short")
    print("window or unmodeled regime risk. But it kills the obvious overfits.")
    print("=" * 68)


if __name__ == "__main__":
    main()
