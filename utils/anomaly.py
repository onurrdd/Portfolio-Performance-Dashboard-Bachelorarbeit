"""
Anomalie-Erkennung für den Portfolio-vs-Benchmark-Vergleich (Plan B).

Ansatz (Market-Model + robuste Ausreißererkennung):
  1. Erwartete Bewegung: rollierendes Beta/Alpha (Portfolio ~ Benchmark) über ein
     Fenster VOR dem Tag t. surprise[t] = tatsächlich[t] − erwartet[t].
  2. Skalierung: surprise wird an seiner EIGENEN rollierenden Median/MAD-Streuung
     gemessen (Tag selbst ausgeschlossen). Median/MAD statt Mittel/Std, um das
     Maskierungsproblem (ein Extremtag bläht σ auf) zu vermeiden.
  3. Feste Schwelle: |mad_z| > THRESHOLD (nicht perzentilbasiert).
  4. Klassifikation am selben Tag: war der Benchmark selbst extrem?
       Benchmark extrem  → "Kopma" (Decoupling)
       Benchmark normal  → "Portföye özgü" (portfolio-spezifisch)
  5. Billige Filter: Kaufdatum ±1 Tag, stale/ffill-Preis am Tag.
  6. Ticker-Zerlegung: welcher Titel trägt den Surprise (vorzeichentreu)?
     Konzentration → "Hisseye özgü" (Einzeltitel) vs. "Faktör/Sektör".
"""
import numpy as np
import pandas as pd

WINDOW = 60          # Rollierendes Fenster (Handelstage)
MIN_OBS = 40         # Mindestanzahl gültiger Beobachtungen im Fenster
MAD_Z_THRESHOLD = 3  # Feste Schwelle auf robustem z-Score
_MAD_SCALE = 1.4826  # MAD → σ-Äquivalent bei Normalverteilung
_CONCENTRATION_CUTOFF = 0.6  # Einzeltitel- vs. Faktor-Grenze


def _rolling_mad(series, window, min_obs):
    def _mad(x):
        med = np.median(x)
        return np.median(np.abs(x - med))
    return series.rolling(window=window, min_periods=min_obs).apply(_mad, raw=True)


def _robust_z(series, window, min_obs):
    """Median/MAD-basierter z-Score; Fenster schließt den Tag selbst aus (shift(1))."""
    roll_median = series.rolling(window=window, min_periods=min_obs).median().shift(1)
    roll_mad = _rolling_mad(series, window, min_obs).shift(1)
    scale = _MAD_SCALE * roll_mad
    z = (series - roll_median) / scale.replace(0, np.nan)
    return z


def _rolling_beta_alpha(portfolio, benchmark, window, min_obs):
    """Rollierendes Beta/Alpha aus Fenster VOR dem Tag (shift(1) am Ende)."""
    cov = portfolio.rolling(window=window, min_periods=min_obs).cov(benchmark)
    var_b = benchmark.rolling(window=window, min_periods=min_obs).var()
    beta_raw = cov / var_b.replace(0, np.nan)
    mean_p = portfolio.rolling(window=window, min_periods=min_obs).mean()
    mean_b = benchmark.rolling(window=window, min_periods=min_obs).mean()
    alpha_raw = mean_p - beta_raw * mean_b
    return beta_raw.shift(1), alpha_raw.shift(1)


def _active_positions_on(positions, day, price_df):
    active = []
    for p in positions:
        buy_date = pd.to_datetime(p['buy_date'])
        if day.tz is not None and buy_date.tz is None:
            buy_date = buy_date.tz_localize(day.tz)
        if p['ticker'] in price_df.columns and day >= buy_date:
            active.append(p)
    return active


def _cheap_filter_flags(day, positions, price_df):
    flags = []
    # (a) Kaufdatum innerhalb ±1 Handelstag
    if day in price_df.index:
        pos_idx = price_df.index.get_loc(day)
        neighbors = price_df.index[max(0, pos_idx - 1): pos_idx + 2]
        for p in positions:
            bd = pd.to_datetime(p['buy_date'])
            if day.tz is not None and bd.tz is None:
                bd = bd.tz_localize(day.tz)
            if any(bd.normalize() == n.normalize() for n in neighbors):
                flags.append('alım±1g')
                break
    # (b) Stale/ffill-Preis: irgendein aktiver Titel unverändert ggü. Vortag
    if day in price_df.index:
        pos_idx = price_df.index.get_loc(day)
        if pos_idx > 0:
            prev = price_df.index[pos_idx - 1]
            for p in _active_positions_on(positions, day, price_df):
                t = p['ticker']
                if price_df.loc[day, t] == price_df.loc[prev, t]:
                    flags.append('stale fiyat')
                    break
    return ', '.join(dict.fromkeys(flags))  # Reihenfolge erhalten, Duplikate entfernen


def _ticker_attribution(day, surprise_value, positions, price_df, ticker_returns_df):
    """Vorzeichentreue Ticker-Zuordnung + Konzentrationsmaß."""
    if day not in price_df.index:
        return None, 0.0, None, None
    active = _active_positions_on(positions, day, price_df)
    if not active:
        return None, 0.0, None, None
    values = {p['ticker']: p['shares'] * price_df.loc[day, p['ticker']] for p in active}
    total_value = sum(values.values())
    if total_value <= 0:
        return None, 0.0, None, None

    contributions = {}
    own_returns = {}
    for p in active:
        t = p['ticker']
        r = ticker_returns_df.loc[day, t] if (day in ticker_returns_df.index and t in ticker_returns_df.columns) else np.nan
        if pd.isna(r):
            continue
        contributions[t] = (values[t] / total_value) * r
        own_returns[t] = r
    if not contributions:
        return None, 0.0, None, None

    # Nur Beiträge in Richtung des Surprise-Vorzeichens sind mögliche "Verursacher".
    sign = 1 if surprise_value >= 0 else -1
    same_dir = {t: c for t, c in contributions.items() if (c >= 0) == (sign >= 0) and c != 0}
    if not same_dir:
        # Kein Titel bewegt sich in Surprise-Richtung → nicht einem Titel zuschreibbar.
        return None, 0.0, None, None

    top_ticker = max(same_dir, key=lambda t: abs(same_dir[t]))
    top_value = same_dir[top_ticker]
    denom = sum(abs(c) for c in same_dir.values())
    concentration_ratio = abs(top_value) / denom if denom > 0 else 0
    concentration = 'Hisseye özgü' if concentration_ratio > _CONCENTRATION_CUTOFF else 'Faktör/Sektör'
    # Eigene (ungewichtete) Tagesrendite des Tickers — wichtig für "warum ist X um Y% gestiegen"-Fragen,
    # da der gewichtete Portfolio-Beitrag (top_value) durch die Portfoliogewichtung verwässert ist.
    own_return_pct = float(own_returns[top_ticker] * 100)
    return top_ticker, float(top_value * 100), concentration, own_return_pct


def detect_anomalies(portfolio_returns, benchmark_returns, positions, price_df,
                     window=WINDOW, min_obs=MIN_OBS, threshold=MAD_Z_THRESHOLD):
    """Gibt eine Liste von Anomalie-Tagen zurück (chronologisch)."""
    # Gemeinsamer Kalender
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    p = portfolio_returns.loc[common].sort_index()
    b = benchmark_returns.loc[common].sort_index()
    if len(p) <= window:
        return []

    beta, alpha = _rolling_beta_alpha(p, b, window, min_obs)
    expected = alpha + beta * b
    surprise = p - expected

    surprise_z = _robust_z(surprise, window, min_obs)
    benchmark_z = _robust_z(b, window, min_obs)

    ticker_returns_df = price_df.pct_change()

    flagged = surprise_z.index[surprise_z.abs() > threshold]
    results = []
    for day in flagged:
        s_val = surprise.loc[day]
        if pd.isna(s_val):
            continue
        b_z = benchmark_z.loc[day]
        classification = 'Kopma' if (pd.notna(b_z) and abs(b_z) > threshold) else 'Portföye özgü'
        responsible_ticker, ticker_contribution_pct, concentration, ticker_own_return_pct = _ticker_attribution(
            day, s_val, positions, price_df, ticker_returns_df
        )
        results.append({
            'date': str(day.date()),
            'actual_return_pct': float(p.loc[day] * 100),
            'expected_return_pct': float(expected.loc[day] * 100) if pd.notna(expected.loc[day]) else None,
            'surprise_pct': float(s_val * 100),
            'surprise_mad_z': float(surprise_z.loc[day]),
            'benchmark_return_pct': float(b.loc[day] * 100),
            'benchmark_mad_z': float(b_z) if pd.notna(b_z) else None,
            'beta': float(beta.loc[day]) if pd.notna(beta.loc[day]) else None,
            'classification': classification,
            'responsible_ticker': responsible_ticker,
            'ticker_contribution_pct': ticker_contribution_pct,
            'ticker_own_return_pct': ticker_own_return_pct,
            'concentration': concentration,
            'flags': _cheap_filter_flags(day, positions, price_df),
        })
    return results
