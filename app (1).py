import warnings
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dash import Dash, Input, Output, State, dash_table, dcc, html

warnings.filterwarnings("ignore")

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_SPLIT_DATE = "2024-01-01"
DEFAULT_RISK_FREE_RATE = 0.04
TRADING_DAYS = 252
CURRENCY_SYMBOL = "₹"


def format_pct(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.2%}"


def format_num(value, digits=4):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.{digits}f}"


def format_currency(amount):
    if amount is None or pd.isna(amount):
        return "-"
    return f"{CURRENCY_SYMBOL}{amount:,.0f}"


def parse_tickers(ticker_text):
    if not ticker_text:
        return DEFAULT_TICKERS
    tickers = [t.strip().upper() for t in ticker_text.replace("\n", ",").split(",") if t.strip()]
    return list(dict.fromkeys(tickers))[:10]


def download_prices(tickers, start_date, end_date):
    raw_data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw_data is None or raw_data.empty:
        return pd.DataFrame()

    if len(tickers) == 1:
        if "Close" in raw_data.columns:
            prices = pd.DataFrame({tickers[0]: raw_data["Close"]})
        else:
            return pd.DataFrame()
    else:
        if isinstance(raw_data.columns, pd.MultiIndex) and "Close" in raw_data.columns.get_level_values(0):
            prices = raw_data["Close"]
        elif isinstance(raw_data.columns, pd.MultiIndex) and "Adj Close" in raw_data.columns.get_level_values(0):
            prices = raw_data["Adj Close"]
        else:
            return pd.DataFrame()

    prices = prices.dropna(axis=1, how="all").dropna()
    valid_tickers = [ticker for ticker in tickers if ticker in prices.columns]
    return prices[valid_tickers]


def calculate_core_metrics(prices, split_date, risk_free_rate):
    daily_returns = prices.pct_change().dropna()
    train_prices = prices[prices.index < pd.Timestamp(split_date)]
    test_prices = prices[prices.index >= pd.Timestamp(split_date)]
    train_returns = daily_returns[daily_returns.index < pd.Timestamp(split_date)]
    test_returns = daily_returns[daily_returns.index >= pd.Timestamp(split_date)]

    train_annual_returns = train_returns.mean() * TRADING_DAYS
    train_annual_volatility = train_returns.std() * np.sqrt(TRADING_DAYS)
    train_sharpe = (train_annual_returns - risk_free_rate) / train_annual_volatility
    train_cov_matrix = train_returns.cov() * TRADING_DAYS

    return {
        "daily_returns": daily_returns,
        "train_prices": train_prices,
        "test_prices": test_prices,
        "train_returns": train_returns,
        "test_returns": test_returns,
        "train_annual_returns": train_annual_returns,
        "train_annual_volatility": train_annual_volatility,
        "train_sharpe": train_sharpe,
        "train_cov_matrix": train_cov_matrix,
    }


def simulate_portfolios(mean_returns, cov_matrix, tickers, risk_free_rate, n_portfolios=20000, seed=42):
    rng = np.random.default_rng(seed)
    n_assets = len(tickers)
    results = []

    mean_arr = mean_returns.reindex(tickers).values
    cov_arr = cov_matrix.reindex(index=tickers, columns=tickers).values

    for _ in range(int(n_portfolios)):
        weights = rng.random(n_assets)
        weights = weights / weights.sum()
        portfolio_return = float(np.dot(weights, mean_arr))
        portfolio_volatility = float(np.sqrt(weights.T @ cov_arr @ weights))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility else np.nan
        row = {
            "Return": portfolio_return,
            "Volatility": portfolio_volatility,
            "Sharpe Ratio": sharpe_ratio,
        }
        for ticker, weight in zip(tickers, weights):
            row[f"{ticker}_Weight"] = weight
        results.append(row)

    df = pd.DataFrame(results).replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return df, None, None

    max_sharpe = df.loc[df["Sharpe Ratio"].idxmax()].copy()
    min_volatility = df.loc[df["Volatility"].idxmin()].copy()
    return df, max_sharpe, min_volatility


def create_summary_cards(prices, metrics, tickers, max_sharpe, min_volatility):
    train_returns = metrics["train_returns"]
    test_returns = metrics["test_returns"]

    best_stock = metrics["train_sharpe"].idxmax()
    best_sharpe = metrics["train_sharpe"].max()

    card_style = {
        "padding": "18px",
        "border": "1px solid #e5e7eb",
        "borderRadius": "12px",
        "backgroundColor": "white",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
    }

    cards = [
        ("Stocks Analyzed", str(len(tickers)), ", ".join(tickers)),
        ("Train Period", f"{len(train_returns):,} days", "Used for optimization"),
        ("Test Period", f"{len(test_returns):,} days", "Used for validation"),
        ("Best Stock Sharpe", best_stock, f"Sharpe: {best_sharpe:.4f}"),
        ("Max Sharpe Portfolio", format_pct(max_sharpe["Return"]), f"Volatility: {format_pct(max_sharpe['Volatility'])}"),
        ("Min Volatility Portfolio", format_pct(min_volatility["Return"]), f"Volatility: {format_pct(min_volatility['Volatility'])}"),
    ]

    return html.Div(
        [
            html.Div(
                [html.Div(title, style={"fontSize": "13px", "color": "#6b7280"}),
                 html.Div(value, style={"fontSize": "24px", "fontWeight": "700", "marginTop": "4px"}),
                 html.Div(subtitle, style={"fontSize": "12px", "color": "#6b7280", "marginTop": "6px"})],
                style=card_style,
            )
            for title, value, subtitle in cards
        ],
        style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "14px", "marginBottom": "24px"},
    )


def create_normalized_price_chart(prices, split_date, tickers):
    normalized = prices / prices.iloc[0] * 100
    fig = go.Figure()
    for ticker in tickers:
        if ticker in normalized.columns:
            fig.add_trace(go.Scatter(x=normalized.index, y=normalized[ticker], mode="lines", name=ticker))

    fig.add_vline(x=pd.Timestamp(split_date), line_dash="dash", line_color="red")
    fig.add_annotation(x=pd.Timestamp(split_date), y=1.05, yref="paper", text="Train/Test Split", showarrow=False)
    fig.update_layout(
        title="Normalized Price History, Train vs Test Split",
        xaxis_title="Date",
        yaxis_title="Normalized Price, Base = 100",
        height=520,
        template="plotly_white",
    )
    return dcc.Graph(figure=fig)


def create_correlation_shift_chart(metrics):
    train_corr = metrics["train_returns"].corr()
    test_corr = metrics["test_returns"].corr()
    corr_diff = test_corr - train_corr

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_diff.values,
            x=corr_diff.columns,
            y=corr_diff.index,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr_diff.values, 2),
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}<br>Shift: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Structural Change, Correlation Shift from Test minus Train",
        height=520,
        template="plotly_white",
    )
    return dcc.Graph(figure=fig)


def create_risk_return_chart(metrics, tickers):
    fig = go.Figure()
    for ticker in tickers:
        fig.add_trace(
            go.Scatter(
                x=[metrics["train_annual_volatility"][ticker]],
                y=[metrics["train_annual_returns"][ticker]],
                mode="markers+text",
                name=ticker,
                text=[ticker],
                textposition="top center",
                marker=dict(size=16, line=dict(width=1, color="black")),
            )
        )
    fig.update_layout(
        title="Risk vs Return, Individual Stocks on Train Data",
        xaxis_title="Annual Volatility",
        yaxis_title="Annual Return",
        height=520,
        template="plotly_white",
    )
    return dcc.Graph(figure=fig)


def create_sharpe_chart(metrics):
    sharpe = metrics["train_sharpe"].sort_values(ascending=False)
    fig = go.Figure(data=go.Bar(x=sharpe.index, y=sharpe.values, text=[f"{v:.2f}" for v in sharpe.values], textposition="outside"))
    fig.add_hline(y=1.0, line_dash="dash", annotation_text="Sharpe = 1.0")
    fig.update_layout(title="Sharpe Ratio by Stock, Train Period", yaxis_title="Sharpe Ratio", height=480, template="plotly_white")
    return dcc.Graph(figure=fig)


def create_efficient_frontier_chart(sim_df, max_sharpe, min_volatility):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sim_df["Volatility"],
            y=sim_df["Return"],
            mode="markers",
            marker=dict(size=5, color=sim_df["Sharpe Ratio"], colorscale="Viridis", colorbar=dict(title="Sharpe")),
            name="Simulated Portfolios",
            opacity=0.65,
        )
    )
    fig.add_trace(go.Scatter(x=[max_sharpe["Volatility"]], y=[max_sharpe["Return"]], mode="markers", marker=dict(size=20, symbol="star", color="red"), name="Max Sharpe"))
    fig.add_trace(go.Scatter(x=[min_volatility["Volatility"]], y=[min_volatility["Return"]], mode="markers", marker=dict(size=18, symbol="diamond", color="black"), name="Min Volatility"))
    fig.update_layout(title="Efficient Frontier Simulation", xaxis_title="Volatility", yaxis_title="Expected Return", height=560, template="plotly_white")
    return dcc.Graph(figure=fig)


def create_allocation_table(optimal, tickers, investment_amount):
    rows = []
    for ticker in tickers:
        weight = float(optimal.get(f"{ticker}_Weight", 0))
        if weight <= 0.001:
            continue
        allocation = weight * investment_amount
        rows.append({
            "Ticker": ticker,
            "Weight (%)": f"{weight * 100:.2f}%",
            f"Allocation ({CURRENCY_SYMBOL})": format_currency(allocation),
        })
    return rows


def create_allocation_pie(optimal, tickers):
    labels = []
    values = []
    for ticker in tickers:
        weight = float(optimal.get(f"{ticker}_Weight", 0))
        if weight > 0.001:
            labels.append(ticker)
            values.append(weight)
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hovertemplate="%{label}<br>%{percent}<extra></extra>")])
    fig.update_layout(title="Max Sharpe Portfolio Allocation", height=520, template="plotly_white")
    return dcc.Graph(figure=fig)


def create_backtest_chart(metrics, max_sharpe, tickers):
    test_returns = metrics["test_returns"]
    if test_returns.empty:
        return dcc.Graph(figure=go.Figure().update_layout(title="Backtest unavailable, no test data"))

    weights = np.array([float(max_sharpe.get(f"{ticker}_Weight", 0)) for ticker in tickers])
    weights = weights / weights.sum()
    equal_weights = np.repeat(1 / len(tickers), len(tickers))

    test_port_returns = (test_returns[tickers] * weights).sum(axis=1)
    benchmark_returns = (test_returns[tickers] * equal_weights).sum(axis=1)
    cum_port = (1 + test_port_returns).cumprod()
    cum_benchmark = (1 + benchmark_returns).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port.values, mode="lines", name="Optimized Portfolio"))
    fig.add_trace(go.Scatter(x=cum_benchmark.index, y=cum_benchmark.values, mode="lines", name="Equal Weight Benchmark", line=dict(dash="dash")))
    fig.update_layout(title="Out-of-Sample Backtest", xaxis_title="Date", yaxis_title="Growth of 1", height=520, template="plotly_white")
    return dcc.Graph(figure=fig)


def run_random_forest_prediction(prices, ticker, split_date, look_back=60):
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error
        from sklearn.preprocessing import MinMaxScaler
        import math
    except Exception:
        return None, "scikit-learn is not installed."

    if ticker not in prices.columns:
        return None, f"{ticker} is not available in price data."

    dataset = prices[ticker].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)
    train_size = len(prices[prices.index < pd.Timestamp(split_date)])

    def create_dataset(dataset_slice):
        data_x, data_y = [], []
        for i in range(len(dataset_slice) - look_back):
            data_x.append(dataset_slice[i:i + look_back, 0])
            data_y.append(dataset_slice[i + look_back, 0])
        return np.array(data_x), np.array(data_y)

    if train_size <= look_back + 5 or len(scaled) - train_size <= look_back + 5:
        return None, "Not enough train/test data for Random Forest prediction."

    train_x, train_y = create_dataset(scaled[:train_size])
    test_slice = scaled[train_size - look_back:]
    test_x, test_y = create_dataset(test_slice)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(train_x, train_y)
    pred_scaled = model.predict(test_x)
    predictions = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    actual = scaler.inverse_transform(test_y.reshape(-1, 1)).flatten()
    rmse = math.sqrt(mean_squared_error(actual, predictions))

    test_index = prices.index[train_size:train_size + len(actual)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index[:train_size], y=prices[ticker].iloc[:train_size], mode="lines", name="Training Data"))
    fig.add_trace(go.Scatter(x=test_index, y=actual, mode="lines", name="Actual Test Price"))
    fig.add_trace(go.Scatter(x=test_index, y=predictions, mode="lines", name="Predicted Test Price", line=dict(dash="dash")))
    fig.update_layout(title=f"{ticker} Random Forest Prediction, RMSE: {rmse:.2f}", xaxis_title="Date", yaxis_title="Price", height=540, template="plotly_white")
    return dcc.Graph(figure=fig), f"Random Forest RMSE for {ticker}: {rmse:.2f}"


app = Dash(__name__)
server = app.server

control_style = {
    "padding": "18px",
    "border": "1px solid #e5e7eb",
    "borderRadius": "12px",
    "backgroundColor": "#ffffff",
    "marginBottom": "20px",
}

app.layout = html.Div(
    style={"maxWidth": "1280px", "margin": "0 auto", "padding": "24px", "fontFamily": "Inter, Arial, sans-serif", "backgroundColor": "#f9fafb"},
    children=[
        html.H1("Stock Portfolio Optimization Dashboard", style={"textAlign": "center", "marginBottom": "6px"}),
        html.P("Train/test portfolio analysis, Markowitz-style optimization, structural correlation shifts, backtesting, and Random Forest price prediction.", style={"textAlign": "center", "color": "#4b5563"}),
        html.Div(
            style=control_style,
            children=[
                html.H3("Configuration"),
                html.Label("Tickers, comma separated"),
                dcc.Input(id="ticker-input", type="text", value=", ".join(DEFAULT_TICKERS), style={"width": "100%", "padding": "10px", "marginBottom": "12px"}),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px"},
                    children=[
                        html.Div([html.Label("Start Date"), dcc.Input(id="start-date", type="text", value=DEFAULT_START_DATE, style={"width": "100%", "padding": "10px"})]),
                        html.Div([html.Label("End Date"), dcc.Input(id="end-date", type="text", value=DEFAULT_END_DATE, style={"width": "100%", "padding": "10px"})]),
                        html.Div([html.Label("Split Date"), dcc.Input(id="split-date", type="text", value=DEFAULT_SPLIT_DATE, style={"width": "100%", "padding": "10px"})]),
                        html.Div([html.Label("Investment Amount"), dcc.Input(id="investment-amount", type="number", value=100000, min=1, style={"width": "100%", "padding": "10px"})]),
                    ],
                ),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "18px", "marginTop": "18px"},
                    children=[
                        html.Div([html.Label("Risk-free Rate"), dcc.Slider(id="risk-free-rate", min=0, max=0.10, step=0.005, value=DEFAULT_RISK_FREE_RATE, marks={0: "0%", 0.04: "4%", 0.1: "10%"})]),
                        html.Div([html.Label("Portfolio Simulations"), dcc.Slider(id="num-portfolios", min=2000, max=50000, step=1000, value=20000, marks={2000: "2k", 20000: "20k", 50000: "50k"})]),
                    ],
                ),
                html.Button("Run Analysis", id="run-button", n_clicks=0, style={"width": "100%", "padding": "12px", "marginTop": "18px", "backgroundColor": "#111827", "color": "white", "border": "none", "borderRadius": "8px", "fontWeight": "700"}),
                html.Div(id="status-output", style={"marginTop": "12px", "fontWeight": "700"}),
            ],
        ),
        dcc.Loading(
            type="default",
            children=html.Div(
                id="results-container",
                children=html.Div("Set your configuration and click Run Analysis.", style={"padding": "24px", "textAlign": "center", "color": "#6b7280"}),
            ),
        ),
        html.Hr(),
        html.P("Disclaimer: Educational use only. This dashboard is not financial advice. Past performance does not guarantee future returns.", style={"fontSize": "12px", "color": "#6b7280", "textAlign": "center"}),
    ],
)


@app.callback(
    Output("results-container", "children"),
    Output("status-output", "children"),
    Input("run-button", "n_clicks"),
    State("ticker-input", "value"),
    State("start-date", "value"),
    State("end-date", "value"),
    State("split-date", "value"),
    State("investment-amount", "value"),
    State("risk-free-rate", "value"),
    State("num-portfolios", "value"),
)
def run_analysis(n_clicks, ticker_text, start_date, end_date, split_date, investment_amount, risk_free_rate, num_portfolios):
    if n_clicks == 0:
        return html.Div("Set your configuration and click Run Analysis.", style={"padding": "24px", "textAlign": "center", "color": "#6b7280"}), "Status: Ready"

    tickers = parse_tickers(ticker_text)
    if len(tickers) < 2:
        return html.Div("Please enter at least two valid tickers."), "Error: Need at least two tickers."

    try:
        pd.Timestamp(start_date)
        pd.Timestamp(end_date)
        pd.Timestamp(split_date)
    except Exception:
        return html.Div("Please use valid dates in YYYY-MM-DD format."), "Error: Invalid date format."

    prices = download_prices(tickers, start_date, end_date)
    tickers = [ticker for ticker in tickers if ticker in prices.columns]
    if prices.empty or len(tickers) < 2:
        return html.Div("Unable to fetch enough market data. Check the tickers or date range."), "Error: Data fetch failed."

    metrics = calculate_core_metrics(prices, split_date, risk_free_rate)
    if len(metrics["train_returns"]) < 100 or len(metrics["test_returns"]) < 20:
        return html.Div("Not enough train/test data. Move the split date or extend the date range."), "Error: Insufficient data."

    sim_df, max_sharpe, min_volatility = simulate_portfolios(
        metrics["train_annual_returns"],
        metrics["train_cov_matrix"],
        tickers,
        risk_free_rate,
        n_portfolios=num_portfolios,
    )
    if sim_df.empty or max_sharpe is None:
        return html.Div("Optimization failed due to invalid return or covariance data."), "Error: Optimization failed."

    allocation_rows = create_allocation_table(max_sharpe, tickers, investment_amount or 100000)
    rf_graph, rf_message = run_random_forest_prediction(prices, tickers[0], split_date)

    results = html.Div(
        children=[
            create_summary_cards(prices, metrics, tickers, max_sharpe, min_volatility),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "18px"},
                children=[
                    html.Div([html.H3("Allocation Plan"), dash_table.DataTable(
                        columns=[{"name": col, "id": col} for col in ["Ticker", "Weight (%)", f"Allocation ({CURRENCY_SYMBOL})"]],
                        data=allocation_rows,
                        sort_action="native",
                        style_table={"overflowX": "auto"},
                        style_cell={"padding": "10px", "textAlign": "left"},
                        style_header={"backgroundColor": "#f3f4f6", "fontWeight": "700"},
                    )], style=control_style),
                    html.Div([create_allocation_pie(max_sharpe, tickers)], style=control_style),
                ],
            ),
            html.Div([create_normalized_price_chart(prices, split_date, tickers)], style=control_style),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "18px"},
                children=[
                    html.Div([create_risk_return_chart(metrics, tickers)], style=control_style),
                    html.Div([create_sharpe_chart(metrics)], style=control_style),
                ],
            ),
            html.Div([create_efficient_frontier_chart(sim_df, max_sharpe, min_volatility)], style=control_style),
            html.Div([create_correlation_shift_chart(metrics)], style=control_style),
            html.Div([create_backtest_chart(metrics, max_sharpe, tickers)], style=control_style),
            html.Div([rf_graph or html.Div(rf_message)], style=control_style),
        ]
    )

    return results, f"Status: Complete. Analyzed {len(tickers)} stocks with {int(num_portfolios):,} simulations. {rf_message}"


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=7860)
