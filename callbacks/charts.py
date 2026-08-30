import numpy as np
import pandas as pd
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from dash import Input, Output, html

from utils.finance import (
    calculate_portfolio_timeseries,
    adjust_shares_for_splits,
    build_synthetic_benchmark_positions,
    get_current_price,
    get_benchmark_history,
)
from utils.anomaly import detect_anomalies
from utils.metrics import (
    calculate_twr_metrics,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_rolling_sharpe,
)

DARK = '#0d1117'
BORDER = '#30363d'
TEXT = '#c9d1d9'
ACCENT = '#58a6ff'

_EMPTY_LAYOUT = dict(
    xaxis={'visible': False},
    yaxis={'visible': False},
    annotations=[{
        'text': 'Füge Positionen hinzu um die Analyse zu starten',
        'xref': 'paper', 'yref': 'paper',
        'showarrow': False,
        'font': {'size': 16, 'color': '#8b949e'}
    }],
    template='plotly_dark',
    paper_bgcolor=DARK,
    plot_bgcolor=DARK,
)

_COMMON = dict(template='plotly_dark', paper_bgcolor=DARK, plot_bgcolor=DARK, font=dict(color=TEXT))


def _empty_fig(title="Keine Daten verfügbar"):
    fig = go.Figure()
    fig.update_layout(title=title, **_EMPTY_LAYOUT)
    return fig


def register(app):
    @app.callback(
        [Output('portfolio-table', 'children'),
         Output('key-metrics', 'children'),
         Output('portfolio-value-chart', 'figure'),
         Output('rolling-sharpe-chart', 'figure'),
         Output('drawdown-chart', 'figure'),
         Output('allocation-chart', 'figure'),
         Output('correlation-chart', 'figure'),
         Output('benchmark-chart', 'figure'),
         Output('benchmark-metrics', 'children'),
         Output('analysis-data', 'data')],
        [Input('portfolio-store', 'data')]
    )
    def update_dashboard(stored_data):
        positions = stored_data.get('positions', [])
        ef = _empty_fig()

        if not positions:
            return (
                dbc.Alert("Portfolio ist leer. Füge Positionen hinzu", color="info"),
                "", ef, ef, ef, ef, ef, ef, "", {}
            )

        try:
            df, price_df = calculate_portfolio_timeseries(positions)
            if df.empty:
                return (
                    dbc.Alert("Fehler beim Laden der Daten", color="danger"),
                    "", ef, ef, ef, ef, ef, ef, "", {}
                )

            # --- Position table ---
            current_data, allocation_data = [], []
            total_current_value = total_invested = 0

            for pos in positions:
                try:
                    current_price = get_current_price(pos['ticker'])
                    current_shares = adjust_shares_for_splits(pos['ticker'], pos['shares'], pos['buy_date'])

                    current_value = current_shares * current_price
                    # Kaufpreis bewusst in HEUTIGER (split-bereinigter) Skala — dieselbe Skala
                    # wie current_shares und current_price. Vorher wurde hier der auf die
                    # historische Vor-Split-Skala hochgerechnete Preis angezeigt, während die
                    # Shares-Spalte bereits split-bereinigt war: "Shares x Buy Price" ergab in
                    # der Tabelle dann ein um den Split-Faktor zu hohes Produkt (z. B. 300 x
                    # $2.346 statt 300 x $117). Der investierte Betrag ist in beiden Skalen
                    # identisch; nur die Anzeige ist jetzt in sich konsistent.
                    invested = current_shares * pos['buy_price']
                    pnl = current_value - invested
                    pnl_pct = (pnl / invested) * 100 if invested > 0 else 0

                    total_current_value += current_value
                    total_invested += invested
                    allocation_data.append({'Ticker': pos['ticker'], 'Value': current_value})
                    current_data.append({
                        'Ticker': pos['ticker'],
                        'Shares': f"{current_shares}",
                        'Buy Date': pos['buy_date'],
                        'Buy Price': f"${pos['buy_price']:.2f}",
                        'Current': f"${current_price:.2f}",
                        'Value': f"${current_value:.2f}",
                        'P&L': f"${pnl:.2f} ({pnl_pct:+.1f}%)"
                    })
                except Exception:
                    continue

            table = dbc.Table.from_dataframe(
                pd.DataFrame(current_data),
                striped=True, bordered=True, hover=True, responsive=True, className="mb-0"
            )
            total_pnl = total_current_value - total_invested
            total_pnl_pct = ((total_current_value / total_invested - 1) * 100) if total_invested > 0 else 0

            portfolio_table = dbc.Card([
                dbc.CardHeader("Aktuelles Portfolio", className="card-header-custom"),
                dbc.CardBody([
                    table,
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([html.H6("Gesamt investiert", className="text-muted"),
                                 html.H4(f"${total_invested:,.2f}", style={'fontWeight': '700'})], width=4),
                        dbc.Col([html.H6("Aktueller Wert", className="text-muted"),
                                 html.H4(f"${total_current_value:,.2f}", style={'fontWeight': '700'})], width=4),
                        dbc.Col([html.H6("Gesamt P&L", className="text-muted"),
                                 html.H4(f"${total_pnl:,.2f} ({total_pnl_pct:+.1f}%)",
                                        id="pnl-value",
                                        style={'fontWeight': '700',
                                               'color': '#28a745' if total_pnl > 0 else '#dc3545'})], width=4),
                    ])
                ])
            ], className="card-custom")

            # --- Metrics ---
            twr = calculate_twr_metrics(positions, price_df)

            if twr is None:
                returns = df['Returns'].dropna()
                portfolio_values = df['Portfolio_Value']
                sortino = calculate_sortino_ratio(returns)
                calmar = calculate_calmar_ratio(returns, portfolio_values)
                rolling_sharpe = calculate_rolling_sharpe(returns)
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = abs(drawdown.min()) * 100
                total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
                years = len(returns) / 252
                annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
                volatility = returns.std() * np.sqrt(252) * 100
            else:
                returns = twr['returns']
                portfolio_values = twr['portfolio_values']
                sortino = twr['sortino']
                calmar = twr['calmar']
                rolling_sharpe = twr['rolling_sharpe']
                max_drawdown = twr['max_drawdown']
                total_return = twr['total_return']
                annualized_return = twr['annualized_return']
                volatility = twr['volatility']
                drawdown = twr['drawdown_series']

            metrics = dbc.Row([
                dbc.Col([html.Div([
                    html.Div("Sortino Ratio", className="metric-label"),
                    html.Div(f"{sortino:.3f}" if not np.isnan(sortino) else "N/A", className="metric-value"),
                    html.Div("Risikoadjustierte Performance", className="metric-description")
                ], className="metric-card")], width=3),
                dbc.Col([html.Div([
                    html.Div("Calmar Ratio", className="metric-label"),
                    html.Div(f"{calmar:.3f}" if not np.isnan(calmar) else "N/A", className="metric-value"),
                    html.Div("Return / Max Drawdown", className="metric-description")
                ], className="metric-card")], width=3),
                dbc.Col([html.Div([
                    html.Div("Max Drawdown", className="metric-label"),
                    html.Div(f"{max_drawdown:.2f}%", className="metric-value"),
                    html.Div("Maximaler Verlust vom Peak", className="metric-description")
                ], className="metric-card")], width=3),
                dbc.Col([html.Div([
                    html.Div("Volatilität (Ann.)", className="metric-label"),
                    html.Div(f"{volatility:.2f}%", className="metric-value"),
                    html.Div("Annualisierte Standardabweichung", className="metric-description")
                ], className="metric-card")], width=3),
            ], className="mb-3")

            # --- Charts ---
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df['Date'], y=df['Portfolio_Value'], mode='lines',
                name='Portfolio Value',
                line=dict(color=ACCENT, width=2),
                fill='tozeroy', fillcolor='rgba(88, 166, 255, 0.1)'
            ))
            fig1.update_layout(title={'text': "Portfolio-Wert Entwicklung", 'font': {'size': 18, 'color': '#f0f6fc'}},
                               xaxis_title="Datum", yaxis_title="Wert (USD)",
                               hovermode='x unified', height=500, **_COMMON)

            fig2 = go.Figure()
            rs_valid = rolling_sharpe.dropna()
            fig2.add_trace(go.Scatter(x=rs_valid.index, y=rs_valid.values, mode='lines',
                                      name='Rolling Sharpe (120d)', line=dict(color='#79c0ff', width=2)))
            fig2.add_hline(y=0, line_dash="dash", line_color=BORDER, opacity=0.5)
            fig2.add_hline(y=1, line_dash="dot", line_color="#8b949e", opacity=0.3, annotation_text="Sharpe = 1")
            fig2.update_layout(title={'text': "Rolling Sharpe Ratio (120 Tage)", 'font': {'size': 18, 'color': '#f0f6fc'}},
                               xaxis_title="Datum", yaxis_title="Sharpe Ratio",
                               hovermode='x unified', height=400, **_COMMON)

            fig3 = go.Figure()
            drawdown_dates = drawdown.index if hasattr(drawdown, 'index') else df['Date']
            fig3.add_trace(go.Scatter(x=drawdown_dates, y=drawdown * 100, mode='lines',
                                      name='Drawdown', fill='tozeroy',
                                      line=dict(color='#8b949e', width=2),
                                      fillcolor='rgba(139, 148, 158, 0.2)'))
            fig3.update_layout(title={'text': "Portfolio Drawdown", 'font': {'size': 18, 'color': '#f0f6fc'}},
                               xaxis_title="Datum", yaxis_title="Drawdown (%)",
                               hovermode='x unified', height=400, **_COMMON)

            fig4 = go.Figure(data=[go.Pie(
                labels=[a['Ticker'] for a in allocation_data],
                values=[a['Value'] for a in allocation_data],
                hole=0.4,
                marker=dict(colors=['#58a6ff', '#79c0ff', '#a5d6ff', '#c9d1d9', '#8b949e', '#6e7681', '#484f58', '#30363d']),
                textinfo='label+percent', textfont_size=12
            )])
            fig4.update_layout(title={'text': "Asset Allocation (nach Marktwert)", 'font': {'size': 18, 'color': '#f0f6fc'}},
                               height=400, showlegend=True, **_COMMON)

            # Correlation
            corr_matrix = pd.DataFrame()
            asset_volatilities = {}
            if not price_df.empty and len(price_df.columns) > 1:
                returns_df = price_df.pct_change().dropna()
                corr_matrix = returns_df.corr()
                for t in price_df.columns:
                    v = price_df[t].pct_change().dropna().std() * np.sqrt(252) * 100
                    asset_volatilities[t] = v
                fig5 = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                    colorscale='Blues', zmid=0,
                    text=corr_matrix.values, texttemplate='%{text:.2f}',
                    textfont={"size": 10, "color": TEXT},
                    colorbar=dict(title="Korrelation")
                ))
                fig5.update_layout(title={'text': "Korrelationsmatrix (Returns)", 'font': {'size': 18, 'color': '#f0f6fc'}},
                                   height=400, xaxis={'side': 'bottom'}, **_COMMON)
            else:
                fig5 = ef

            # Benchmark
            benchmark_stats = {'total_return': 0, 'sortino': 0, 'sharpe_current': 0}
            active_return_breaks = []
            try:
                spy_hist = get_benchmark_history('SPY', df['Date'].min(), df['Date'].max())
                if not spy_hist.empty:
                    # Zahlungsstromgleicher Benchmark: simuliert die gleichen
                    # Kapitalzuflüsse (buy_date, investierter Betrag) im S&P 500, damit
                    # Total Return/Sortino/Sharpe zeitgleich und fair vergleichbar sind.
                    benchmark_positions = build_synthetic_benchmark_positions(positions, spy_hist['Close'])
                    benchmark_price_df = spy_hist['Close'].to_frame(name='SPY')
                    twr_benchmark = calculate_twr_metrics(benchmark_positions, benchmark_price_df)

                    if twr_benchmark is not None:
                        spy_total_return = twr_benchmark['total_return']
                        spy_sortino = twr_benchmark['sortino']
                        spy_rolling_sharpe = twr_benchmark['rolling_sharpe']
                        spy_returns_for_chart = twr_benchmark['returns']
                    else:
                        spy_returns_for_chart = spy_hist['Close'].pct_change().dropna()
                        spy_total_return = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0] - 1) * 100
                        spy_sortino = calculate_sortino_ratio(spy_returns_for_chart)
                        spy_rolling_sharpe = calculate_rolling_sharpe(spy_returns_for_chart)

                    # Beide Linien als kapitalflussbereinigte kumulierte Rendite (%):
                    # zeigt reale Performance statt Kontostand-Wachstum durch Kapitalzuflüsse.
                    port_cum_return = ((1 + returns).cumprod() - 1) * 100
                    spy_cum_return = ((1 + spy_returns_for_chart).cumprod() - 1) * 100
                    port_dates = returns.index if twr is not None else df.loc[returns.index, 'Date']

                    # Anomalie-Erkennung (Plan B): Market-Model-Surprise + robuster
                    # Median/MAD-z-Score, siehe utils/anomaly.py.
                    active_return_breaks = detect_anomalies(
                        returns, spy_returns_for_chart, positions, price_df
                    )

                    fig6 = go.Figure()
                    fig6.add_trace(go.Scatter(x=port_dates, y=port_cum_return, mode='lines',
                                              name='Portfolio', line=dict(color=ACCENT, width=2)))
                    fig6.add_trace(go.Scatter(x=spy_cum_return.index, y=spy_cum_return, mode='lines',
                                              name='S&P 500', line=dict(color='#8b949e', width=2, dash='dash')))
                    fig6.update_layout(
                        title={'text': "Portfolio vs. S&P 500 Benchmark", 'font': {'size': 18, 'color': '#f0f6fc'}},
                        xaxis_title="Datum", yaxis_title="Kumulierte Rendite (%)",
                        hovermode='x unified', height=500,
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                        **_COMMON
                    )

                    portfolio_vs_spy = total_return - spy_total_return

                    spy_rs_valid = spy_rolling_sharpe.dropna()
                    spy_sharpe_current = spy_rs_valid.iloc[-1] if not spy_rs_valid.empty else np.nan

                    portfolio_rs_valid = rolling_sharpe.dropna()
                    portfolio_sharpe_current = portfolio_rs_valid.iloc[-1] if not portfolio_rs_valid.empty else np.nan

                    benchmark_stats = {
                        'total_return': float(spy_total_return),
                        'sortino': float(spy_sortino) if not np.isnan(spy_sortino) else 0,
                        'sharpe_current': float(spy_sharpe_current) if not np.isnan(spy_sharpe_current) else 0,
                    }

                    benchmark_metrics = html.Div([
                        dbc.Row([
                            dbc.Col([dbc.Card([dbc.CardBody([
                                html.H5("Portfolio Return (TWR)", className="text-muted"),
                                html.H3(f"{total_return:.2f}%", style={'color': '#f0f6fc', 'fontWeight': '700'})
                            ])], className="card-custom")], width=3),
                            dbc.Col([dbc.Card([dbc.CardBody([
                                html.H5("S&P 500 Return (angeglichen)", className="text-muted"),
                                html.H3(f"{spy_total_return:.2f}%", style={'color': '#8b949e', 'fontWeight': '700'})
                            ])], className="card-custom")], width=3),
                            dbc.Col([dbc.Card([dbc.CardBody([
                                html.H5("Outperformance (angeglichen)", className="text-muted"),
                                html.H3(f"{portfolio_vs_spy:+.2f}%",
                                       style={'color': '#28a745' if portfolio_vs_spy > 0 else '#dc3545', 'fontWeight': '700'})
                            ])], className="card-custom")], width=3),
                            dbc.Col([dbc.Card([dbc.CardBody([
                                html.H5("Portfolio Volatilität", className="text-muted"),
                                html.H3(f"{volatility:.2f}%", style={'color': '#8b949e', 'fontWeight': '700'})
                            ])], className="card-custom")], width=3),
                        ]),
                        dbc.Row([
                            dbc.Col([dbc.Card([dbc.CardBody([
                                html.H5("Portfolio Sortino Ratio", className="text-muted"),
                                html.H3(f"{sortino:.3f}" if not np.isnan(sortino) else "N/A",
                                       style={'color': '#f0f6fc', 'fontWeight': '700'})
                            ])], className="card-custom")], width=3),
                            dbc.Col([dbc.Card([dbc.CardBody([
                                html.H5("S&P 500 Sortino Ratio (angeglichen)", className="text-muted"),
                                html.H3(f"{spy_sortino:.3f}" if not np.isnan(spy_sortino) else "N/A",
                                       style={'color': '#8b949e', 'fontWeight': '700'})
                            ])], className="card-custom")], width=3),
                            dbc.Col([dbc.Card([dbc.CardBody([
                                html.H5("Portfolio Sharpe Ratio", className="text-muted"),
                                html.H3(f"{portfolio_sharpe_current:.3f}" if not np.isnan(portfolio_sharpe_current) else "N/A",
                                       style={'color': '#f0f6fc', 'fontWeight': '700'})
                            ])], className="card-custom")], width=3),
                            dbc.Col([dbc.Card([dbc.CardBody([
                                html.H5("S&P 500 Sharpe Ratio (angeglichen)", className="text-muted"),
                                html.H3(f"{spy_sharpe_current:.3f}" if not np.isnan(spy_sharpe_current) else "N/A",
                                       style={'color': '#8b949e', 'fontWeight': '700'})
                            ])], className="card-custom")], width=3),
                        ], className="mt-3"),
                    ])
                else:
                    fig6, benchmark_metrics = ef, dbc.Alert("Benchmark-Daten nicht verfügbar", color="warning")
            except Exception:
                fig6, benchmark_metrics = ef, dbc.Alert("Fehler beim Laden der Benchmark-Daten", color="danger")

            # --- Analysis data for AI ---
            if not corr_matrix.empty:
                upper = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                high_corrs = []
                for i in range(len(corr_matrix)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            high_corrs.append(f"{corr_matrix.index[i]}-{corr_matrix.columns[j]}: {corr_matrix.iloc[i,j]:.2f}")
                correlation_stats = {
                    'avg_correlation': float(upper.mean()),
                    'max_correlation': float(upper.max()),
                    'min_correlation': float(upper.min()),
                    'high_correlations': high_corrs
                }
            else:
                correlation_stats = {'avg_correlation': 0, 'max_correlation': 0, 'min_correlation': 0, 'high_correlations': []}

            def _safe(series, fn):
                try:
                    v = fn(series)
                    return float(v) if not pd.isna(v) else 0
                except Exception:
                    return 0

            analysis_data = {
                'positions': positions,
                'allocation': allocation_data,
                'metrics': {
                    'sortino': float(sortino) if not np.isnan(sortino) else 0,
                    'calmar': float(calmar) if not np.isnan(calmar) else 0,
                    'max_drawdown': float(max_drawdown),
                    'volatility': float(volatility),
                    'total_return': float(total_return),
                    'annualized_return': float(annualized_return)
                },
                'rolling_sharpe': {
                    'current': _safe(rolling_sharpe, lambda s: s.iloc[-1]),
                    'mean': _safe(rolling_sharpe, lambda s: s.mean()),
                    'min': _safe(rolling_sharpe, lambda s: s.min()),
                    'max': _safe(rolling_sharpe, lambda s: s.max()),
                    'std': _safe(rolling_sharpe, lambda s: s.std()),
                },
                'correlation_stats': correlation_stats,
                'asset_volatilities': asset_volatilities,
                'correlation_matrix': corr_matrix.to_dict() if not corr_matrix.empty else {},
                'benchmark': benchmark_stats,
                'active_return_breaks': active_return_breaks
            }

            return portfolio_table, metrics, fig1, fig2, fig3, fig4, fig5, fig6, benchmark_metrics, analysis_data

        except Exception as e:
            import traceback
            traceback.print_exc()
            return (
                dbc.Alert(f"Fehler beim Laden: {str(e)}", color="danger"),
                "", ef, ef, ef, ef, ef, ef, "", {}
            )
