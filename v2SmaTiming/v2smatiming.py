import yaml
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

def calculate_metrics(returns, cum_returns, rf=0.0, num_trading_days=252):
    """
    Calculate performance metrics for a strategy.
    """
    # Total Return (multiplicative factor - 1, but example seems to show factor)
    # In example, TotalReturn 6.2326 likely the factor (1 + total return)
    # But for SPY 2000-2024, SPY ~100 to 500, ~5x, close.
    # Wait, actually probably (final / initial) which is total factor.
    total_return = cum_returns.iloc[-1]

    # Period in years
    years = len(returns) / num_trading_days

    # CAGR (geometric annualized return)
    cagr = (cum_returns.iloc[-1]) ** (1 / years) - 1

    # Annualized Return (arithmetic mean annualized)
    ann_return = np.mean(returns) * num_trading_days

    # Annualized Volatility
    ann_vol = np.std(returns) * np.sqrt(num_trading_days)

    # Sharpe Ratio (using daily rf)
    daily_rf = rf / num_trading_days
    sharpe = (np.mean(returns) - daily_rf) / np.std(returns) * np.sqrt(num_trading_days)

    # Max Drawdown
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_dd = drawdown.min()

    # Ulcer Index
    dd_squared = drawdown ** 2
    ulcer_index = np.sqrt(np.mean(dd_squared)) * 100  # Often multiplied by 100 for percentage

    return {
        'CAGR': cagr,
        'AnnReturn': ann_return,
        'AnnVol': ann_vol,
        'Sharpe': sharpe,
        'MaxDD': max_dd,
        'UlcerIndex': ulcer_index,
        'TotalReturn': total_return
    }

def main():
    # Load config
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    ticker = config['ticker']
    cash_proxy = config.get('cash_proxy', 'NONE')
    rf = config.get('risk_free_rate', 0.0)
    start = config.get('start')
    end = config.get('end')
    years = config.get('years', 10)
    sma_windows = config['sma_windows']
    outdir = config.get('outdir', 'outputs')

    # Handle dates
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    if start is None:
        start_date = datetime.strptime(end, '%Y-%m-%d') - timedelta(days=365 * years + 1)  # +1 for leap
        start = start_date.strftime('%Y-%m-%d')

    # Fetch data
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    prices = df['Adj Close']
    if isinstance(prices, pd.DataFrame):
        prices = prices[ticker]
    returns = prices.pct_change().dropna()

    proxy_returns = None
    if cash_proxy != 'NONE':
        proxy_df = yf.download(cash_proxy, start=start, end=end, progress=False, auto_adjust=False)
        proxy_prices = proxy_df['Adj Close']
        if isinstance(proxy_prices, pd.DataFrame):
            proxy_prices = proxy_prices[cash_proxy]
        proxy_returns = proxy_prices.pct_change().dropna()

    # Align dates
    if proxy_returns is not None:
        common_dates = returns.index.intersection(proxy_returns.index)
        returns = returns.loc[common_dates]
        proxy_returns = proxy_returns.loc[common_dates]
        prices = prices.loc[common_dates]

    # Buy and Hold
    bh_cum = (1 + returns).cumprod()
    bh_metrics = calculate_metrics(returns, bh_cum, rf)

    # Results dict
    results = {'BuyAndHold': bh_metrics}

    # For each SMA window
    for w in sma_windows:
        sma = prices.rolling(window=w).mean()
        signal = (prices > sma).astype(int)  # 1 if above, 0 below

        # Shift signal for no look-ahead
        signal = signal.shift(1).fillna(0)  # Start out of market if no data

        # Strategy returns
        if cash_proxy == 'NONE':
            cash_ret = rf / 252  # Approximate daily rf
            strat_returns = signal * returns + (1 - signal) * cash_ret
        else:
            strat_returns = signal * returns + (1 - signal) * proxy_returns

        strat_cum = (1 + strat_returns).cumprod()
        strat_metrics = calculate_metrics(strat_returns, strat_cum, rf)

        results[f'SMA{w}'] = strat_metrics

    # Create DataFrame for output
    metrics_df = pd.DataFrame(results).T
    metrics_df = metrics_df[['CAGR', 'AnnReturn', 'AnnVol', 'Sharpe', 'MaxDD', 'UlcerIndex', 'TotalReturn']].T

    # Reorder columns as in example: SMA100, SMA150, SMA200, BuyAndHold
    cols = [f'SMA{w}' for w in sorted(sma_windows)] + ['BuyAndHold']
    metrics_df = metrics_df[cols]

    # Format to 4 decimals for most, 4 for Ulcer, etc.
    print("Overall stats")
    print(metrics_df.to_string(float_format=lambda x: f"{x:.4f}"))

    # Optionally save to outdir
    os.makedirs(outdir, exist_ok=True)
    metrics_df.to_csv(os.path.join(outdir, 'backtest_results.csv'))

if __name__ == '__main__':
    main()