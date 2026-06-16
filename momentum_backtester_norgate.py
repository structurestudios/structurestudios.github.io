"""
Momentum Backtester — NORGATE adapter (survivorship-bias-free)
==============================================================
Runs the momentum breakout strategy from momentum_backtester.py against a
Norgate watchlist that INCLUDES DELISTED STOCKS, so the test is honest.

WHY THE WATCHLIST CHOICE MATTERS (this is the whole point):
  Norgate has watchlists like "Russell 3000 Current & Past" that include stocks
  that have since been DELISTED. Using a "Current & Past" list is what makes this
  survivorship-bias-free. If you used a "Current" only list, you'd be testing
  survivors only and the result would be a beautiful LIE -- the exact trap.

REQUIREMENTS (on your Windows machine):
  - Norgate Data Updater running in the background (you confirmed it connects)
  - pip install norgatedata
  - This file in the same folder as momentum_backtester.py

NOTE ON THE FREE TRIAL:
  The trial gives ~2 years of data. That's roughly ONE market regime (bull-leaning).
  Even a positive result here carries the same asterisk as everything else: two
  years isn't enough to know an edge is durable across bull AND bear markets.
"""

import pickle

import numpy as np
import pandas as pd
import norgatedata

# reuse the strategy logic you already built & validated
from momentum_backtester import find_trades, regime_ok, summarize

# ---- CONFIG ----
# Universe watchlist. "Current & Past" = includes delisted = survivorship-bias-free.
# Good choices (pick what your subscription exposes):
#   "Russell 3000 Current & Past"   (broad, includes small caps + delisted)
#   "S&P 500 Current & Past"        (large caps, includes delisted)
WATCHLIST = "Russell 3000 Current & Past"

# Index symbol for the market-regime filter. Norgate index symbols start with "$".
# $COMPX = Nasdaq Composite (the PDF's $COMPQX). $SPX = S&P 500.
INDEX_SYMBOL = "$COMPX"

START_DATE = "2023-01-01"   # trial gives ~2 yrs; adjust to your trial window
PRICE_ADJ = norgatedata.StockPriceAdjustmentType.CAPITAL  # split-adjusted, not total-return
PADDING = norgatedata.PaddingType.NONE
MAX_SYMBOLS = None          # set e.g. 300 to test a subset first; None = full universe

# Transaction costs charged per side (see momentum_backtester.py). These names
# include delisted small caps, where fills are worse -- so the 5 bps / 0.05 ATR
# defaults are a floor. Raise them to see how fast the "edge" survives friction;
# set both to 0 for a gross/frictionless comparison.
COST_BPS_PER_SIDE = 5.0
SLIPPAGE_ATR = 0.05

# Where to cache fetched OHLCV + the regime series, for offline sweeps.
CACHE_FILE = "universe_cache.pkl"


def fetch(symbol):
    """Pull one symbol's OHLCV as a clean lowercase-column DataFrame, or None."""
    try:
        df = norgatedata.price_timeseries(
            symbol,
            stock_price_adjustment_setting=PRICE_ADJ,
            padding_setting=PADDING,
            start_date=START_DATE,
            format="pandas-dataframe",
        )
    except Exception as e:
        print(f"  skip {symbol}: {e}")
        return None
    if df is None or len(df) == 0:
        return None
    df.columns = [c.lower() for c in df.columns]
    # norgate columns are typically Open/High/Low/Close/Volume
    need = ["open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in need):
        return None
    return df[need].dropna()


def main():
    print(f"Norgate status: {norgatedata.status()}")
    print(f"Universe watchlist: {WATCHLIST}")
    print(f"Regime index: {INDEX_SYMBOL}")
    print(f"Costs: {COST_BPS_PER_SIDE} bps/side + {SLIPPAGE_ATR} ATR slippage/side "
          f"(net R-multiples)\n")

    # 1) build the market-regime filter from the index
    idx = fetch(INDEX_SYMBOL)
    if idx is None:
        # index symbols sometimes need a different fetch; try without start trim
        idx = norgatedata.price_timeseries(INDEX_SYMBOL, start_date=START_DATE,
                                            format="pandas-dataframe")
        idx.columns = [c.lower() for c in idx.columns]
        idx = idx[["open", "high", "low", "close", "volume"]].dropna()
    regime = regime_ok(idx)
    print(f"Regime series built: {len(regime)} days, "
          f"{int(regime.sum())} in uptrend\n")

    # 2) get the survivorship-bias-free universe
    symbols = norgatedata.watchlist_symbols(WATCHLIST)
    if MAX_SYMBOLS:
        symbols = symbols[:MAX_SYMBOLS]
    print(f"Universe has {len(symbols)} symbols (incl. delisted). Scanning...\n")

    # 3) scan every symbol for the momentum setup, caching each frame as we go.
    #    The cache lets the param/cost sweeps (momentum_lab.py) re-run offline
    #    without re-hitting Norgate -- fetch once, analyze forever.
    all_R = []
    cache = {}
    done = 0
    for sym in symbols:
        df = fetch(sym)
        if df is not None and len(df) > 80:
            cache[sym] = df
            all_R += find_trades(df, regime,
                                 cost_bps_per_side=COST_BPS_PER_SIDE,
                                 slippage_atr=SLIPPAGE_ATR)
        done += 1
        if done % 250 == 0:
            print(f"  ...{done}/{len(symbols)} symbols scanned, "
                  f"{len(all_R)} trades so far")

    print(f"\nScan complete: {done} symbols, {len(all_R)} total trades.\n")

    # 4) honest stats (reuses your summarize, which prints the survivorship warning)
    summarize(all_R)

    # 5) save the per-trade R-multiples so we can run significance testing next
    if all_R:
        pd.DataFrame({"R": all_R}).to_csv("momentum_trades_R.csv", index=False)
        print("\nSaved per-trade R-multiples to momentum_trades_R.csv")

    # 6) cache the raw data + regime so the lab can sweep params/costs offline
    if cache:
        with open(CACHE_FILE, "wb") as fh:
            pickle.dump({"data": cache, "regime": regime}, fh)
        print(f"Cached {len(cache)} symbols + regime to {CACHE_FILE}")
        print("Next: python momentum_significance.py   (real or luck?)")
        print("      python momentum_lab.py            (overfit & cost sweeps)")


if __name__ == "__main__":
    main()
