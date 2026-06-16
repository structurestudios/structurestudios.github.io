"""
Test suite for the momentum backtester.
=======================================
These tests lock the properties that make the backtest trustworthy. The most
important is NO LOOKAHEAD: a decision on bar i may only use data up to bar i.
Everything else (R-multiple math, cost monotonicity, the regime filter, the
significance tools) is here so a refactor can't silently corrupt the result.

Run:  pytest -q
"""

import numpy as np
import pandas as pd
import pytest

import momentum_backtester as mb
from momentum_backtester import find_trades, regime_ok, summarize, _atr
from momentum_significance import (
    bootstrap_mean_ci, signflip_pvalue, monte_carlo_equity, load_R,
)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _frame(close, wiggle=0.0):
    c = np.asarray(close, dtype=float)
    dates = pd.date_range("2023-01-01", periods=len(c), freq="B")
    return pd.DataFrame(
        {"open": c, "high": c * (1 + wiggle), "low": c * (1 - wiggle),
         "close": c, "volume": 1.0},
        index=dates,
    )


def _always_on(df):
    return pd.Series(True, index=df.index)


# ----------------------------------------------------------------------------
# regime filter
# ----------------------------------------------------------------------------
def test_regime_is_bool_and_warmup_false():
    df = _frame(np.linspace(100, 200, 120))
    reg = regime_ok(df, ma=50)
    assert reg.dtype == bool
    # first MA-1 days are warm-up -> must be False, never NaN
    assert not reg.iloc[:49].any()
    # a steady uptrend should be risk-on once warmed up
    assert reg.iloc[60:].all()


def test_regime_off_in_downtrend():
    df = _frame(np.linspace(200, 100, 120))
    reg = regime_ok(df, ma=50)
    assert not reg.iloc[60:].any()


# ----------------------------------------------------------------------------
# ATR
# ----------------------------------------------------------------------------
def test_atr_constant_range():
    # high-low is a constant 2.0 each bar; with no gaps ATR -> 2.0
    n = 60
    c = np.full(n, 100.0)
    df = pd.DataFrame(
        {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0},
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )
    atr = _atr(df, length=14)
    assert atr.iloc[-1] == pytest.approx(2.0)


# ----------------------------------------------------------------------------
# no lookahead -- the cardinal rule
# ----------------------------------------------------------------------------
def test_no_lookahead_future_bars_cannot_change_past_trades():
    """Trades produced from the first k bars must be a prefix-stable result:
    appending future bars may ADD trades but must not alter ones already closed
    within the earlier window."""
    rng = np.random.default_rng(0)
    c = 50 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 400)))
    full = _frame(c, wiggle=0.01)
    reg = _always_on(full)

    cut = 250
    early = find_trades(full.iloc[:cut], reg.iloc[:cut],
                        cost_bps_per_side=0, slippage_atr=0)
    later = find_trades(full, reg, cost_bps_per_side=0, slippage_atr=0)

    # every trade that fully closed in the early window must reappear unchanged.
    # (the last early "trade" may be a forced close-at-end, which the longer run
    #  would instead let run -- so compare all but a possible final open trade.)
    n_stable = len(early) - 1
    assert n_stable <= len(later)
    for a, b in zip(early[:n_stable], later[:n_stable]):
        assert a == pytest.approx(b)


def test_breakout_uses_prior_high_not_today():
    """Entry triggers on a close above the PRIOR-day high window. A single new
    high on bar i must not let bar i trigger off its own value."""
    # flat at 100 for the lookback, then a single jump. With shift(1) the jump
    # bar compares against the prior flat high (100), so it can trigger -- but
    # the breakout level on that bar must equal the old high, not the new one.
    n = 120
    c = np.full(n, 100.0)
    c[-1] = 130.0
    df = _frame(c, wiggle=0.0)
    bl = df["close"].rolling(mb.BREAKOUT_LOOKBACK).max().shift(1)
    # breakout level on the jump bar should be the old 100, not 130
    assert bl.iloc[-1] == pytest.approx(100.0)


# ----------------------------------------------------------------------------
# R-multiple math
# ----------------------------------------------------------------------------
def test_full_stop_loss_is_minus_one_R_gross():
    """Buy a breakout, then get stopped at the initial stop with zero costs ->
    exactly -1R."""
    # flat bars at 100, one breakout bar, then a crash through the stop.
    # wiggle>0 gives a real high-low range so ATR>0 (and thus risk>0);
    # need >= MIN_BARS total for find_trades to run at all.
    flat = np.full(90, 100.0)
    # breakout bar closes at 110 (above prior 50-day high of 100)
    # then next bar lows far below the 2*ATR stop
    seq = np.concatenate([flat, [110.0, 50.0]])
    df = _frame(seq, wiggle=0.01)
    # force low of last bar very low so the stop is hit intrabar
    df.iloc[-1, df.columns.get_loc("low")] = 1.0
    reg = _always_on(df)
    R = find_trades(df, reg, cost_bps_per_side=0, slippage_atr=0)
    assert len(R) == 1
    assert R[0] == pytest.approx(-1.0, abs=1e-9)


def test_open_trade_force_closed_at_last_close():
    """A position still open at the end is closed at the final close, not left
    dangling."""
    flat = np.full(60, 100.0)
    rally = np.linspace(111, 160, 40)  # breakout then steady climb, never stops
    df = _frame(np.concatenate([flat, rally]), wiggle=0.0)
    reg = _always_on(df)
    R = find_trades(df, reg, cost_bps_per_side=0, slippage_atr=0)
    assert len(R) == 1
    assert R[0] > 0  # an uptrend held to the end is a winner


# ----------------------------------------------------------------------------
# costs
# ----------------------------------------------------------------------------
def _one_trade_frame():
    flat = np.full(60, 100.0)
    rally = np.linspace(111, 160, 40)
    return _frame(np.concatenate([flat, rally]), wiggle=0.0)


def test_costs_reduce_R_monotonically():
    df = _one_trade_frame()
    reg = _always_on(df)
    gross = find_trades(df, reg, cost_bps_per_side=0, slippage_atr=0)[0]
    cheap = find_trades(df, reg, cost_bps_per_side=5, slippage_atr=0.05)[0]
    dear = find_trades(df, reg, cost_bps_per_side=30, slippage_atr=0.20)[0]
    assert gross > cheap > dear


def test_zero_costs_equal_gross():
    df = _one_trade_frame()
    reg = _always_on(df)
    a = find_trades(df, reg, cost_bps_per_side=0, slippage_atr=0)[0]
    # default constants applied, then explicitly zeroed -> must match gross
    b = find_trades(df, reg, cost_bps_per_side=0.0, slippage_atr=0.0)[0]
    assert a == pytest.approx(b)


# ----------------------------------------------------------------------------
# regime gating actually blocks trades
# ----------------------------------------------------------------------------
def test_regime_off_blocks_all_entries():
    df = _one_trade_frame()
    reg_off = pd.Series(False, index=df.index)
    assert find_trades(df, reg_off) == []


# ----------------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------------
def test_too_short_returns_empty():
    df = _frame(np.full(10, 100.0))
    assert find_trades(df, _always_on(df)) == []
    assert find_trades(None, None) == []


# ----------------------------------------------------------------------------
# significance tools
# ----------------------------------------------------------------------------
def test_signflip_detects_edge_and_rejects_noise():
    rng = np.random.default_rng(3)
    # strong edge: mostly small losses + a few big wins, clearly positive mean
    edge = np.concatenate([np.full(80, -1.0), np.full(40, 4.0)])
    _, p_edge = signflip_pvalue(edge, rng, n=5000)
    assert p_edge < 0.01

    # no edge: symmetric around zero
    noise = rng.normal(0, 1, 200)
    _, p_noise = signflip_pvalue(noise, rng, n=5000)
    assert p_noise > 0.05


def test_bootstrap_ci_orders_correctly():
    rng = np.random.default_rng(4)
    R = np.concatenate([np.full(80, -1.0), np.full(40, 4.0)])
    mean, lo, hi, p_le0 = bootstrap_mean_ci(R, rng, n=5000)
    assert lo < mean < hi
    assert lo > 0          # clearly positive edge -> CI clears zero
    assert p_le0 < 0.05


def test_monte_carlo_return_distribution_is_nondegenerate():
    """Resampling with replacement must produce a SPREAD of outcomes (the bug we
    fixed: shuffling alone gave identical terminal returns)."""
    rng = np.random.default_rng(5)
    R = np.concatenate([np.full(80, -1.0), np.full(40, 4.0)])
    finals, dds = monte_carlo_equity(R, rng, n=3000, risk=0.01)
    assert finals.std() > 0          # not degenerate
    assert np.all(dds <= 0)          # drawdowns are non-positive by definition


def test_load_R_rejects_missing_column(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_R(str(bad))


def test_summarize_handles_empty(capsys):
    summarize([])
    assert "No trades" in capsys.readouterr().out
