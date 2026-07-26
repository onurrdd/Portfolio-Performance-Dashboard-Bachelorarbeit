import numpy as np
import pandas as pd


def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252
    downside = excess_returns[excess_returns < 0]
    if len(downside) == 0:
        return np.nan
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std == 0:
        return np.nan
    return np.sqrt(252) * excess_returns.mean() / downside_std


def calculate_calmar_ratio(returns, portfolio_values, annualized_return=None):
    if annualized_return is None:
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        years = len(returns) / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    max_drawdown = abs(((cumulative - running_max) / running_max).min())

    if max_drawdown == 0:
        return np.nan
    return annualized_return / max_drawdown


def calculate_rolling_sharpe(returns, window=120, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252
    rolling_mean = excess_returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    return np.sqrt(252) * rolling_mean / rolling_std


def calculate_twr_metrics(positions, price_df):
    if not positions:
        return None

    positions_sorted = sorted(positions, key=lambda x: pd.to_datetime(x['buy_date']))
    buy_dates = [pd.to_datetime(p['buy_date']) for p in positions_sorted]

    if not price_df.empty and price_df.index.tz is not None:
        tz = price_df.index.tz
        buy_dates = [
            d.tz_localize(tz) if d.tz is None else d.tz_convert(tz)
            for d in buy_dates
        ]
        buy_dates.append(pd.Timestamp.now(tz=tz))
    else:
        buy_dates.append(pd.Timestamp.now())

    all_daily_returns = []
    all_return_dates = []
    all_portfolio_values = []
    all_dates = []

    for i in range(len(buy_dates) - 1):
        period_start = buy_dates[i]
        period_end = buy_dates[i + 1]

        active_positions = []
        for p in positions_sorted:
            pos_buy_date = pd.to_datetime(p['buy_date'])
            if price_df.index.tz is not None and pos_buy_date.tz is None:
                pos_buy_date = pos_buy_date.tz_localize(price_df.index.tz)
            if pos_buy_date <= period_start:
                active_positions.append(p)

        if not active_positions:
            continue

        period_mask = (price_df.index >= period_start) & (price_df.index < period_end)
        period_dates = price_df.index[period_mask]
        if len(period_dates) == 0:
            continue

        period_values = []
        period_dates_valid = []
        for date in period_dates:
            value = sum(
                p['shares'] * price_df.loc[date, p['ticker']]
                for p in active_positions
                if p['ticker'] in price_df.columns
            )
            if value > 0:
                period_values.append(value)
                period_dates_valid.append(date)

        all_portfolio_values.extend(period_values)
        all_dates.extend(period_dates_valid)

        # Tagesrenditen NUR innerhalb der Periode (Seam/Kapitalzufluss wird übersprungen).
        # Jede Rendite wird mit ihrem END-Datum beschriftet, damit sich die Datumslabels
        # an Perioden-Grenzen nicht verschieben.
        for k in range(1, len(period_values)):
            if period_values[k - 1] > 0:
                all_daily_returns.append((period_values[k] / period_values[k - 1]) - 1)
                all_return_dates.append(period_dates_valid[k])

    if not all_daily_returns:
        return None

    returns_series = pd.Series(all_daily_returns, index=all_return_dates)
    portfolio_series = pd.Series(all_portfolio_values, index=all_dates)

    volatility = returns_series.std() * np.sqrt(252) * 100
    sortino = calculate_sortino_ratio(returns_series)
    rolling_sharpe = calculate_rolling_sharpe(returns_series)

    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdown.min()) * 100
    total_return = ((1 + returns_series).prod() - 1) * 100
    years = len(returns_series) / 252
    annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
    calmar = calculate_calmar_ratio(returns_series, portfolio_series, annualized_return=annualized_return / 100)

    return {
        'returns': returns_series,
        'portfolio_values': portfolio_series,
        'volatility': volatility,
        'sortino': sortino,
        'calmar': calmar,
        'rolling_sharpe': rolling_sharpe,
        'max_drawdown': max_drawdown,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'drawdown_series': drawdown,
        'num_days': len(returns_series)
    }
