"""
Momentum Backtester — core strategy logic
==========================================
A small, honest momentum-breakout backtester. The point of this module is NOT
to sell you an edge; it's to give you a reusable, transparent engine whose
numbers you can trust enough to then *attack* with significance testing.

The three public functions are the contract used by the data adapters
(e.g. momentum_backtester_norgate.py):

    regime_ok(index_df)          -> bool Series   (market "risk-on" filter)
    find_trades(df, regime)      -> list[float]   (per-trade R-multiples)
    summarize(list_of_R)         -> None          (prints honest stats)

THE STRATEGY (classic trend/momentum breakout, one position per symbol):
  - Only take longs while the *market* (an index) is in an uptrend. Trading
    breakouts against the market's grain is how breakout systems bleed out.
  - Enter when price closes above its prior N-day high (a breakout).
  - Risk a fixed multiple of ATR below entry. That distance is "1R".
  - Ride the move with a *ratcheting* (never-loosening) ATR trailing stop.
  - Exit when the trailing stop is hit. The result of each trade is recorded
    as an R-multiple = (exit - entry) / (entry - initial_stop), so a -1R is a
    full planned loss and +3R is three times the risk taken.

WHY R-MULTIPLES: they make trades comparable across symbols and price levels,
and they're exactly what you want to feed into a Monte-Carlo / bootstrap
significance test afterwards.
"""

import numpy as np
import pandas as pd

# ---- STRATEGY PARAMETERS ----
BREAKOUT_LOOKBACK = 50      # enter on a close above the prior N-day high
ATR_LENGTH        = 14      # ATR period used for stops
ATR_STOP_MULT     = 2.0     # initial stop = entry - 2*ATR  (defines 1R)
ATR_TRAIL_MULT    = 3.0     # trailing stop = highest-high-since-entry - 3*ATR
REGIME_MA         = 50      # index uptrend = close above its 50-day average
MIN_BARS          = 80      # need enough history for lookbacks to warm up

# ---- TRANSACTION COSTS (the honesty knob) ----
# Real fills are worse than the close you backtest on. We charge two things,
# PER SIDE (entry and exit), so a round trip pays each twice:
#   COST_BPS_PER_SIDE: commission + half the bid/ask spread, in basis points
#       of the trade price. 5 bps/side ~= 0.10% round trip -- modest for liquid
#       names, optimistic for the small/illiquid/delisted ones in a broad
#       "Current & Past" universe, so treat it as a floor and raise it to probe.
#   SLIPPAGE_ATR: extra adverse fill (you buy higher, sell lower) expressed in
#       units of the entry ATR. Scales cost with volatility, which is realistic.
# Costs are converted to an R haircut using each trade's OWN risk, so tight-stop
# trades are penalized more in R than wide-stop trades -- exactly as in reality.
COST_BPS_PER_SIDE = 5.0
SLIPPAGE_ATR      = 0.05


def _atr(df, length=ATR_LENGTH):
    """Wilder-style ATR via a simple rolling mean of True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(length).mean()


def regime_ok(index_df, ma=REGIME_MA):
    """Market-regime filter: True on days the index closes above its MA.

    Returns a boolean Series indexed by date. NaN warm-up days are False
    (we don't trade until the filter is actually defined).
    """
    close = index_df["close"]
    sma = close.rolling(ma).mean()
    return (close > sma).fillna(False)


def find_trades(df, regime, cost_bps_per_side=COST_BPS_PER_SIDE,
                slippage_atr=SLIPPAGE_ATR, breakout_lookback=BREAKOUT_LOOKBACK,
                atr_length=ATR_LENGTH, atr_stop_mult=ATR_STOP_MULT,
                atr_trail_mult=ATR_TRAIL_MULT):
    """Scan one symbol for breakout trades; return a list of NET R-multiples.

    One position at a time. Any position still open at the end of the data is
    closed at the final close (no peeking, no free ride).

    Strategy params (exposed so robustness sweeps can vary them):
      breakout_lookback  enter on a close above this prior-day high.
      atr_length         ATR period for stops.
      atr_stop_mult      initial stop = entry - mult*ATR (defines 1R).
      atr_trail_mult     trailing stop = highest-high-since-entry - mult*ATR.

    Costs (set both to 0 for a gross/frictionless run):
      cost_bps_per_side  commission + half-spread, basis points of fill price.
      slippage_atr       adverse fill per side, in units of the entry ATR.
    Each is charged on entry and on exit, then divided by the trade's risk to
    become an R haircut.
    """
    if df is None or len(df) < MIN_BARS:
        return []

    close = df["close"]
    high = df["high"]
    low = df["low"]
    atr = _atr(df, atr_length)
    # prior N-day high (shifted so today's bar can't see its own high)
    breakout_level = close.rolling(breakout_lookback).max().shift(1)

    # align the market-regime filter onto this symbol's calendar
    reg = regime.reindex(df.index).ffill().fillna(False)

    bps = cost_bps_per_side / 10000.0  # basis points -> fraction of price

    def net_R(entry, exit_price, risk, atr_at_entry):
        """Gross R minus round-trip costs, expressed in R of the trade's risk."""
        # commission/spread on each leg's notional + slippage on each leg
        cost = bps * (entry + exit_price) + 2.0 * slippage_atr * atr_at_entry
        return (exit_price - entry - cost) / risk

    R = []
    in_pos = False
    entry = stop = risk = trail_high = atr_at_entry = 0.0

    for i in range(len(df)):
        a = atr.iloc[i]
        bl = breakout_level.iloc[i]
        if np.isnan(a) or np.isnan(bl):
            continue

        c = close.iloc[i]

        if not in_pos:
            # ENTRY: market risk-on AND a fresh breakout close
            if bool(reg.iloc[i]) and c > bl:
                entry = c
                stop = entry - atr_stop_mult * a
                risk = entry - stop
                if risk <= 0:
                    continue
                atr_at_entry = a
                trail_high = high.iloc[i]
                in_pos = True
        else:
            # MANAGE: ratchet the trailing stop up, never down
            trail_high = max(trail_high, high.iloc[i])
            stop = max(stop, trail_high - atr_trail_mult * a)
            # EXIT: intrabar low takes out the stop -> fill at the stop
            if low.iloc[i] <= stop:
                R.append(net_R(entry, stop, risk, atr_at_entry))
                in_pos = False

    # force-close any open trade at the last available close
    if in_pos:
        R.append(net_R(entry, close.iloc[-1], risk, atr_at_entry))

    return R


def summarize(all_R):
    """Print honest per-trade R-multiple statistics for a list of trades."""
    if not all_R:
        print("No trades found.")
        return

    R = np.asarray(all_R, dtype=float)
    n = len(R)
    wins = R[R > 0]
    losses = R[R <= 0]
    win_rate = len(wins) / n
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    expectancy = R.mean()

    # profit factor = gross gains / gross losses (in R)
    gross_win = wins.sum()
    gross_loss = -losses.sum()
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    print("=" * 56)
    print("RESULTS  (per-trade R-multiples)")
    print("-" * 56)
    print(f"  Trades:          {n}")
    print(f"  Win rate:        {win_rate:6.1%}")
    print(f"  Avg win:         {avg_win:6.2f}R")
    print(f"  Avg loss:        {avg_loss:6.2f}R")
    print(f"  Expectancy:      {expectancy:6.3f}R  per trade")
    print(f"  Profit factor:   {profit_factor:6.2f}")
    print(f"  Total:           {R.sum():6.1f}R")
    print("=" * 56)
    print(
        "SURVIVORSHIP / SAMPLE WARNING:\n"
        "  These numbers are only as honest as the universe they came from.\n"
        "  If the symbol list excludes delisted stocks, the result is\n"
        "  inflated by survivorship bias -- a beautiful lie. And a short\n"
        "  sample (e.g. ~2 years) is one market regime, not proof of an\n"
        "  edge. Treat a positive expectancy as a hypothesis to be attacked\n"
        "  with significance testing, not as a discovered edge."
    )
    print("=" * 56)
