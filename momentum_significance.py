"""
Momentum Backtester — significance testing
==========================================
You have a list of per-trade R-multiples (momentum_trades_R.csv from the
Norgate run). The backtest says the average is positive. So what? A positive
average can easily be luck. This script attacks that result and tries to tell
you whether the edge is *real* or just a pretty draw from a coin-flip process.

THREE TESTS, FROM THREE ANGLES:

  1) BOOTSTRAP CONFIDENCE INTERVAL on the mean R.
     Resample the trades with replacement many times and look at the spread of
     the mean. If the 95% interval comfortably clears 0, the positive
     expectancy is robust to which trades happened to occur. If it straddles 0,
     you cannot rule out luck.

  2) PERMUTATION / SIGN-FLIP TEST (the null is "no edge").
     Under the null hypothesis the sign of each trade's deviation is random.
     We randomly flip signs many times to build the distribution of mean R you'd
     see by chance, then ask how often chance beats your observed mean. That
     fraction is a p-value. Small p = unlikely to be luck.

  3) MONTE-CARLO EQUITY / DRAWDOWN.
     Resample the trades WITH REPLACEMENT many times (each run is a plausible
     alternate track record of the same length) and compound a fixed-fraction
     bet to see the range of terminal returns and worst drawdowns. NOTE: merely
     *shuffling* the trade order is useless here -- fixed-fraction compounding
     is a commutative product, so order doesn't change the terminal return.
     Resampling does change it, and it exposes the pain: a strategy can have a
     positive mean and still routinely hit a drawdown that would have stopped
     you out of trading it.

USAGE:
    python momentum_significance.py                      # reads momentum_trades_R.csv
    python momentum_significance.py path/to/other.csv    # any CSV with an "R" column

HONEST FRAMING:
  Passing these tests does NOT prove a durable edge. They only check that the
  in-sample result is unlikely under "pure luck" GIVEN this sample. The deeper
  threats -- a 2-year window being one regime, overfit parameters, costs and
  slippage not modeled -- are not fixable by resampling the same trades. A clean
  p-value here is necessary, not sufficient.
"""

import sys
import numpy as np
import pandas as pd

# ---- CONFIG ----
DEFAULT_CSV   = "momentum_trades_R.csv"
N_RESAMPLES   = 20000     # bootstrap / permutation iterations
CI_LEVEL      = 0.95      # confidence level for the bootstrap interval
RISK_FRACTION = 0.01      # Monte-Carlo: fraction of equity risked per trade (1R)
SEED          = 42        # reproducible


def load_R(path):
    """Load the R column from a CSV; raise a clear error if it's not there."""
    df = pd.read_csv(path)
    if "R" not in df.columns:
        raise ValueError(f"{path} has no 'R' column (found: {list(df.columns)})")
    R = df["R"].to_numpy(dtype=float)
    R = R[np.isfinite(R)]
    if len(R) == 0:
        raise ValueError(f"{path} has no usable R values")
    return R


def bootstrap_mean_ci(R, rng, n=N_RESAMPLES, level=CI_LEVEL):
    """Percentile bootstrap CI for the mean R."""
    n_trades = len(R)
    idx = rng.integers(0, n_trades, size=(n, n_trades))
    means = R[idx].mean(axis=1)
    lo = np.percentile(means, (1 - level) / 2 * 100)
    hi = np.percentile(means, (1 + level) / 2 * 100)
    p_mean_le_0 = float(np.mean(means <= 0))  # how often resampled mean <= 0
    return means.mean(), lo, hi, p_mean_le_0


def signflip_pvalue(R, rng, n=N_RESAMPLES):
    """One-sided permutation p-value for H0: mean deviation is symmetric about 0.

    Flip the sign of each trade's deviation from 0 at random and count how often
    the resulting mean meets or exceeds the observed mean.
    """
    observed = R.mean()
    flips = rng.choice([-1.0, 1.0], size=(n, len(R)))
    null_means = (flips * R).mean(axis=1)
    # +1 in num & den = unbiased small-sample estimate (never reports p=0)
    p = (np.sum(null_means >= observed) + 1) / (n + 1)
    return observed, float(p)


def monte_carlo_equity(R, rng, n=N_RESAMPLES, risk=RISK_FRACTION):
    """Resample trades w/ replacement, compound a fixed-fraction bet; return & DD."""
    n_trades = len(R)
    # each trade changes equity by (risk * R): +R wins risk*R of equity
    samples = 1.0 + risk * R[rng.integers(0, n_trades, size=(n, n_trades))]
    equity = np.cumprod(samples, axis=1)
    finals = equity[:, -1] - 1.0                            # total return
    running_max = np.maximum.accumulate(equity, axis=1)
    max_dds = np.min(equity / running_max - 1.0, axis=1)    # worst drawdown (<=0)
    return finals, max_dds


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    rng = np.random.default_rng(SEED)

    R = load_R(path)
    n = len(R)
    mean_R = R.mean()
    std_R = R.std(ddof=1) if n > 1 else 0.0

    print("=" * 60)
    print(f"SIGNIFICANCE TESTING  ({path})")
    print("-" * 60)
    print(f"  Trades:            {n}")
    print(f"  Observed mean:     {mean_R:+.4f}R")
    print(f"  Std dev:           {std_R:.4f}R")
    print("=" * 60)

    # 1) bootstrap CI on the mean
    bmean, lo, hi, p_le0 = bootstrap_mean_ci(R, rng)
    print("\n[1] BOOTSTRAP  (mean R, %d resamples)" % N_RESAMPLES)
    print(f"    {int(CI_LEVEL*100)}% CI: [{lo:+.4f}R, {hi:+.4f}R]")
    print(f"    P(resampled mean <= 0): {p_le0:.4f}")
    print("    -> " + ("CI clears 0: positive expectancy is robust to sampling."
                       if lo > 0 else
                       "CI straddles 0: cannot rule out luck from sampling."))

    # 2) sign-flip permutation p-value
    obs, p = signflip_pvalue(R, rng)
    print("\n[2] SIGN-FLIP PERMUTATION TEST  (H0: no edge)")
    print(f"    p-value: {p:.4f}")
    verdict = ("REJECT H0 at 0.05: result is unlikely under pure luck."
               if p < 0.05 else
               "CANNOT reject H0 at 0.05: consistent with luck.")
    print("    -> " + verdict)

    # 3) Monte-Carlo equity / drawdown
    finals, dds = monte_carlo_equity(R, rng)
    print("\n[3] MONTE-CARLO  (resampled w/ replacement, %.0f%% risk/trade)"
          % (RISK_FRACTION * 100))
    print(f"    Median total return: {np.median(finals):+.1%}")
    print(f"    5th-95th pct return: [{np.percentile(finals,5):+.1%}, "
          f"{np.percentile(finals,95):+.1%}]")
    print(f"    P(total return < 0): {np.mean(finals < 0):.4f}")
    print(f"    Median max drawdown: {np.median(dds):.1%}")
    print(f"    Worst 5% drawdown:   {np.percentile(dds,5):.1%}")

    print("\n" + "=" * 60)
    print("REMEMBER: these tests check the result isn't luck GIVEN this sample.")
    print("They do NOT fix a short window, overfit params, or unmodeled costs.")
    print("A clean p-value is necessary, not sufficient, for a real edge.")
    print("=" * 60)


if __name__ == "__main__":
    main()
