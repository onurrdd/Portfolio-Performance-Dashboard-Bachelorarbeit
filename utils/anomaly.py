"""
Anomalie-Erkennung für den Portfolio-vs-Benchmark-Vergleich (Plan B).

Ansatz (Market-Model + robuste Ausreißererkennung):
  1. Erwartete Bewegung: rollierendes Beta/Alpha (Portfolio ~ Benchmark) über ein
     Fenster VOR dem Tag t. surprise[t] = tatsächlich[t] − erwartet[t].
  2. Skalierung: surprise wird an seiner EIGENEN rollierenden Median/MAD-Streuung
     gemessen (Tag selbst ausgeschlossen). Median/MAD statt Mittel/Std, um das
     Maskierungsproblem (ein Extremtag bläht σ auf) zu vermeiden.
  3. Feste Schwelle: |mad_z| > THRESHOLD (nicht perzentilbasiert).
  4. Zustand des Benchmarks am selben Tag wird als ZAHL mitgeführt (benchmark_mad_z),
     nicht als Kategorie. Methodischer Sinn: bei einem extremen Benchmark-Tag
     extrapoliert das lineare Markt-Modell (α + β·Markt) weit über den Bereich hinaus,
     in dem β geschätzt wurde — die erwartete Rendite und damit das Residuum sind an
     solchen Tagen weniger belastbar und vorsichtiger zu interpretieren.
     BEWUSST KEIN Filter: an solchen Tagen finden sehr wohl echte titelspezifische
     Ereignisse statt (Makro-Schock und Einzeltitel-Nachricht schließen sich nicht aus).
     Ein Ausschluss würde reale, belegbare Ereignisse verwerfen; die Zahl dient daher
     nur der Interpretation.
  5. Billige Filter: Kaufdatum ±1 Tag, stale/ffill-Preis am Tag.
  6. Ticker-Zerlegung: welcher Titel trägt den Surprise (vorzeichentreu)?
     WICHTIG: Der Beitrag wird auf dem RESIDUUM des Titels gemessen, nicht auf seiner
     Rohrendite — Gewicht × (eigene Rendite − eigene markterwartete Rendite). Sonst
     würde ein schwergewichtiger High-Beta-Titel, der exakt das tut, was sein Beta
     vorhersagt (also null Überraschung liefert), fälschlich als "verantwortlich"
     ausgewiesen, während der Titel mit der echten Überraschung übersehen wird.
     Die Gewichtung bleibt erhalten: ein winziger Titel mit großem Residuum kann den
     Portfolio-Surprise nicht erklären und wird korrekt nicht ausgewählt.
  7. Signifikanz auf Titelebene: der gewählte Titel muss auch an SEINER EIGENEN
     Residuen-Historie gemessen einen außergewöhnlichen Tag gehabt haben (|z| > Schwelle).
     Sonst gilt der Tag als nicht zuordenbar — ein diversifiziertes Portfolio schlägt
     früher aus als ein einzelner volatiler Titel, sodass ein Portfolio-Ausreißer für den
     Titel selbst ein ganz normaler Tag sein kann (zu dem es keine Nachricht gibt).
     Konzentration → "Hisseye özgü" (Einzeltitel) vs. "Dağılmış" (Verteilt).
     Hinweis: "Dağılmış" beschreibt nur, dass der Surprise nicht auf einen Titel
     konzentriert ist — es ist KEINE Aussage über Sektor- oder Faktorzugehörigkeit
     (diese werden hier nicht gemessen).
"""
import numpy as np
import pandas as pd

from utils.finance import adjust_shares_for_splits

WINDOW = 60          # Rollierendes Fenster (Handelstage)
MIN_OBS = 40         # Mindestanzahl gültiger Beobachtungen im Fenster
MAD_Z_THRESHOLD = 3  # Feste Schwelle auf robustem z-Score
_MAD_SCALE = 1.4826  # MAD → σ-Äquivalent bei Normalverteilung
_CONCENTRATION_CUTOFF = 0.6  # Einzeltitel- vs. verteilt-Grenze
# Schwelle für den TITEL selbst: der verantwortliche Titel muss auch gemessen an seiner
# EIGENEN Residuen-Historie einen außergewöhnlichen Tag gehabt haben. Grund: ein
# diversifiziertes Portfolio hat geringere Residuen-Streuung als ein Einzeltitel — ein
# Portfolio-3-MAD-z-Tag kann für einen volatilen Titel (z. B. TSLA) ein völlig normaler
# ~2-Sigma-Tag sein, zu dem es keine Nachricht gibt. Ohne diesen Test würde das RAG für
# solche Tage vergeblich (und quotenverbrauchend) nach Nachrichten suchen.
#
# BEWUSST NIEDRIGER als die Portfolio-Schwelle (MAD_Z_THRESHOLD = 3): Die Begründung des
# Tests ist ja gerade, dass die Residuen-Streuung eines Einzeltitels HÖHER ist. Dieselbe
# Schwelle auf beiden Ebenen wäre auf Titelebene daher eine deutlich härtere Hürde und hat
# in der Praxis auch zweistellige Tagesverluste verworfen — also Tage, zu denen es mit
# hoher Wahrscheinlichkeit sehr wohl eine Nachricht gibt. 2,5 erhält den Schutzzweck
# (offensichtlich unauffällige Titeltage fallen weiterhin heraus), ohne diese Klasse
# echter Ereignisse zu verlieren.
_TICKER_MAD_Z_THRESHOLD = 2.5

# 2.5 -> 26 çıktı - 5 -> 20 - 3 yapayım ->  



def _rolling_mad(series, window, min_obs):
    # nanmedian statt median: pandas' min_periods garantiert genügend gültige Werte im
    # Fenster, übergibt der Funktion aber das ROHE Array inkl. NaN. Mit np.median würde
    # ein einziges NaN das gesamte 60-Tage-Fenster vergiften (MAD = NaN) — bei den
    # Titel-Residuen (Lücken an Kapitalzufluss-Tagen) fielen so ~26 % der Tage aus.
    def _mad(x):
        med = np.nanmedian(x)
        return np.nanmedian(np.abs(x - med))
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


def _ticker_residuals(ticker_returns_df, benchmark_returns, window, min_obs):
    """Je Ticker das Residuum: eigene Rendite − eigene markterwartete Rendite.

    Verwendet dieselbe Logik wie auf Portfolioebene (rollierendes Beta/Alpha aus dem
    Fenster VOR dem Tag, via _rolling_beta_alpha inkl. shift(1)), nur eben pro Titel.
    So ist "Überraschung" auf beiden Ebenen dieselbe Größe und die Zuordnung passt
    zu dem, was die Anomalie überhaupt ausgelöst hat.
    """
    b = benchmark_returns.reindex(ticker_returns_df.index)
    residuals = {}
    for t in ticker_returns_df.columns:
        r = ticker_returns_df[t]
        beta_t, alpha_t = _rolling_beta_alpha(r, b, window, min_obs)
        residuals[t] = r - (alpha_t + beta_t * b)
    return pd.DataFrame(residuals, index=ticker_returns_df.index)


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


def _ticker_attribution(day, surprise_value, positions, price_df, ticker_returns_df,
                        ticker_residuals_df, ticker_resid_z_df, ticker_threshold):
    """Vorzeichentreue Ticker-Zuordnung auf Residuenbasis + Konzentrationsmaß.

    Der Auswahlmaßstab ist der RESIDUEN-Beitrag: Gewicht × (eigene Rendite − erwartete
    Rendite des Titels). Damit wird genau die Größe zerlegt, die die Anomalie ausgelöst
    hat (den Portfolio-Surprise) — und nicht die Rohrendite, die auch reine Marktbewegung
    enthält. Die Gewichtung bleibt Teil der Formel: ein Titel mit Mini-Gewicht kann den
    Portfolio-Surprise nicht erklären, egal wie groß seine eigene Überraschung ist.
    """
    if day not in price_df.index:
        return None, 0.0, None, None, None, None
    active = _active_positions_on(positions, day, price_df)
    if not active:
        return None, 0.0, None, None, None, None
    # Split-bereinigte Stückzahl: price_df-Kurse sind split-adjustiert (yfinance-Standard),
    # rohe p['shares'] wäre sonst um den Split-Faktor falsch skaliert — genau der Fehler,
    # der GOOGL/TSLA/NVDA (Split nach buy_date) künstlich untergewichtet und sie dadurch
    # bei der "sorumlu hisse"-Auswahl praktisch chancenlos macht.
    values = {
        p['ticker']: adjust_shares_for_splits(p['ticker'], p['shares'], p['buy_date'])
                      * price_df.loc[day, p['ticker']]
        for p in active
    }
    total_value = sum(values.values())
    if total_value <= 0:
        return None, 0.0, None, None, None, None

    contributions = {}
    own_returns = {}
    own_residuals = {}
    for p in active:
        t = p['ticker']
        has_ret = day in ticker_returns_df.index and t in ticker_returns_df.columns
        has_res = day in ticker_residuals_df.index and t in ticker_residuals_df.columns
        r = ticker_returns_df.loc[day, t] if has_ret else np.nan
        resid = ticker_residuals_df.loc[day, t] if has_res else np.nan
        # Residuum kann in der Aufwärmphase (zu wenige Beobachtungen) NaN sein.
        if pd.isna(r) or pd.isna(resid):
            continue
        contributions[t] = (values[t] / total_value) * resid
        own_returns[t] = r
        own_residuals[t] = resid
    if not contributions:
        return None, 0.0, None, None, None, None

    # Nur Beiträge in Richtung des Surprise-Vorzeichens sind mögliche "Verursacher".
    sign = 1 if surprise_value >= 0 else -1
    same_dir = {t: c for t, c in contributions.items() if (c >= 0) == (sign >= 0) and c != 0}
    if not same_dir:
        # Kein Titel bewegt sich in Surprise-Richtung → nicht einem Titel zuschreibbar.
        return None, 0.0, None, None, None, None

    top_ticker = max(same_dir, key=lambda t: abs(same_dir[t]))
    top_value = same_dir[top_ticker]

    # Signifikanztest auf TITELEBENE: War der Tag auch für diesen Titel selbst
    # außergewöhnlich? Gemessen am robusten z-Score seiner eigenen Residuen-Historie.
    # Ist er es nicht (oder nicht bestimmbar), gilt der Tag als nicht zuordenbar —
    # dann sucht das RAG auch keine Nachricht, die es mit hoher Wahrscheinlichkeit
    # gar nicht gibt.
    has_z = day in ticker_resid_z_df.index and top_ticker in ticker_resid_z_df.columns
    own_z = ticker_resid_z_df.loc[day, top_ticker] if has_z else np.nan
    if pd.isna(own_z) or abs(own_z) <= ticker_threshold:
        return None, 0.0, None, None, None, None

    denom = sum(abs(c) for c in same_dir.values())
    concentration_ratio = abs(top_value) / denom if denom > 0 else 0
    concentration = 'Hisseye özgü' if concentration_ratio > _CONCENTRATION_CUTOFF else 'Dağılmış'
    # Eigene (ungewichtete) ROH-Tagesrendite — das ist die Zahl, nach der man in Nachrichten
    # sucht ("Warum fiel X am 12.08. um 12,3 %?"). Bewusst roh, nicht residual: die Nachricht
    # berichtet über die tatsächliche Kursbewegung, nicht über den modellbereinigten Rest.
    own_return_pct = float(own_returns[top_ticker] * 100)
    # Zusätzlich das eigene Residuum (ungewichtet) — macht in der Debug-Tabelle sichtbar,
    # wie viel der Bewegung wirklich Überraschung war und wie viel bloß Marktbewegung.
    own_residual_pct = float(own_residuals[top_ticker] * 100)
    return (top_ticker, float(top_value * 100), concentration, own_return_pct,
            own_residual_pct, float(own_z))


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
    # Residuen pro Titel einmalig vorberechnen (nicht je Anomalietag) — Basis der Zuordnung.
    ticker_residuals_df = _ticker_residuals(ticker_returns_df, b, window, min_obs)
    # Robuster z-Score der Titel-Residuen: Maßstab dafür, ob der Tag für den TITEL
    # selbst außergewöhnlich war (Signifikanztest in _ticker_attribution).
    ticker_resid_z_df = pd.DataFrame(
        {t: _robust_z(ticker_residuals_df[t], window, min_obs)
         for t in ticker_residuals_df.columns},
        index=ticker_residuals_df.index,
    )

    flagged = surprise_z.index[surprise_z.abs() > threshold]
    results = []
    for day in flagged:
        s_val = surprise.loc[day]
        if pd.isna(s_val):
            continue
        b_z = benchmark_z.loc[day]
        (responsible_ticker, ticker_contribution_pct, concentration,
         ticker_own_return_pct, ticker_own_residual_pct,
         ticker_own_mad_z) = _ticker_attribution(
            day, s_val, positions, price_df, ticker_returns_df, ticker_residuals_df,
            ticker_resid_z_df, _TICKER_MAD_Z_THRESHOLD
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
            'responsible_ticker': responsible_ticker,
            'ticker_contribution_pct': ticker_contribution_pct,   # Residuen-Beitrag (Auswahlmaß)
            'ticker_own_return_pct': ticker_own_return_pct,        # rohe Tagesrendite (für News)
            'ticker_own_residual_pct': ticker_own_residual_pct,    # eigenes Residuum (ungewichtet)
            'ticker_own_mad_z': ticker_own_mad_z,                  # Signifikanz auf Titelebene
            'concentration': concentration,
            'flags': _cheap_filter_flags(day, positions, price_df),
        })
    return results
