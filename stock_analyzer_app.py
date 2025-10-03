# --- [1] Import necessary libraries ---
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta
import warnings

# --- [2] Initial Setup and Configuration ---
# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")
pio.templates.default = "plotly_dark"  # Use a dark theme for plots

# Initialize the Dash app with Bootstrap components for styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}])

# This line is crucial for Gunicorn (the production server) to find the app
server = app.server

# --- [3] Application Layout ---
app.layout = dbc.Container([
    # Header Section
    dbc.Row([
        dbc.Col(html.H1("Trading Analysis & Prediction Dashboard",
                        className="text-center text-primary mt-4 mb-2"), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.P("Analyze NSE/BSE stocks with technicals, fundamentals, and AI-powered forecasts.",
                        className="text-center text-light mb-4"), width=12)
    ]),

    # Controls Section
    dbc.Row([
        dbc.Col([
            dbc.Input(id='stock-ticker-input', type='text',
                      placeholder='Enter Stock Ticker (e.g., RELIANCE.NS)',
                      value='RELIANCE.NS', className="mb-2"),
        ], width=12, lg=4),
        dbc.Col([
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=datetime.now() - timedelta(days=365*2),
                end_date=datetime.now(),
                display_format='YYYY-MM-DD',
                className="mb-2 w-100"
            ),
        ], width=12, lg=4),
        dbc.Col([
            dbc.Button('Analyze Stock', id='submit-button-state',
                       n_clicks=0, color='primary', className="w-100"),
        ], width=12, lg=4),
    ], className="mb-4"),

    # Loading spinner and main content area
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-spinner",
                type="bars",
                color="#007BFF",
                children=[
                    html.Div(id='error-message-output', className="text-danger text-center mb-2"),
                    html.Div(id='analysis-output-container')
                ]
            )
        ], width=12)
    ])
], fluid=True, className="dbc")

# --- [4] Callbacks for Interactivity ---
@app.callback(
    [Output('analysis-output-container', 'children'),
     Output('error-message-output', 'children')],
    [Input('submit-button-state', 'n_clicks')],
    [State('stock-ticker-input', 'value'),
     State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date')]
)
def update_analysis_output(n_clicks, ticker, start_date, end_date):
    if n_clicks == 0 or not ticker:
        return html.Div(dbc.Alert("Please enter a stock ticker and click 'Analyze'.", color="info")), ""

    try:
        # Fetch stock data using the more robust yf.Ticker method
        stock_ticker = yf.Ticker(ticker)
        stock_data = stock_ticker.history(start=start_date, end=end_date)
        if stock_data.empty:
            raise ValueError(f"No data found for ticker '{ticker}'. It may be delisted or an invalid symbol.")

        # --- Technical Analysis Calculations ---
        stock_data.ta.sma(length=20, append=True)
        stock_data.ta.sma(length=50, append=True)
        stock_data.ta.rsi(length=14, append=True)
        stock_data.ta.macd(fast=12, slow=26, signal=9, append=True)
        bollinger = stock_data.ta.bbands(length=20)
        if bollinger is not None and not bollinger.empty:
            stock_data = pd.concat([stock_data, bollinger], axis=1)

        # --- Generate Buy/Sell Signals (MA Crossover) ---
        # *** FIX: Check if SMA columns exist before creating signals ***
        if 'SMA_20' in stock_data.columns and 'SMA_50' in stock_data.columns:
            stock_data['signal'] = 0
            stock_data.loc[stock_data['SMA_20'] > stock_data['SMA_50'], 'signal'] = 1  # Buy signal
            stock_data.loc[stock_data['SMA_20'] < stock_data['SMA_50'], 'signal'] = -1 # Sell signal
            buy_signals = stock_data[stock_data['signal'] == 1]
            sell_signals = stock_data[stock_data['signal'] == -1]
        else:
            # If SMAs don't exist, create empty dataframes for signals
            buy_signals = pd.DataFrame()
            sell_signals = pd.DataFrame()


        # --- Create Price Action Candlestick Chart ---
        candlestick_fig = go.Figure()
        candlestick_fig.add_trace(go.Candlestick(x=stock_data.index,
                                             open=stock_data['Open'], high=stock_data['High'],
                                             low=stock_data['Low'], close=stock_data['Close'],
                                             name='Price'))
        # *** FIX: Check if SMA columns exist before plotting them ***
        if 'SMA_20' in stock_data.columns:
            candlestick_fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_20'],
                                               mode='lines', name='20-Day SMA', line=dict(color='yellow', width=1)))
        if 'SMA_50' in stock_data.columns:
            candlestick_fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_50'],
                                               mode='lines', name='50-Day SMA', line=dict(color='orange', width=1)))
        if 'BBL_20_2.0' in stock_data.columns and 'BBU_20_2.0' in stock_data.columns:
            candlestick_fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['BBL_20_2.0'],
                                               mode='lines', name='Lower Bollinger Band', line=dict(color='cyan', width=1, dash='dash')))
            candlestick_fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['BBU_20_2.0'],
                                               mode='lines', name='Upper Bollinger Band', line=dict(color='cyan', width=1, dash='dash')))
        
        # Plot buy/sell signals if they exist
        if not buy_signals.empty:
            candlestick_fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                                               mode='markers', name='Buy Signal', marker=dict(color='green', symbol='triangle-up', size=10)))
        if not sell_signals.empty:
            candlestick_fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                                               mode='markers', name='Sell Signal', marker=dict(color='red', symbol='triangle-down', size=10)))

        candlestick_fig.update_layout(title=f'{ticker.upper()} - Price Action & Signals',
                                    xaxis_title='Date', yaxis_title='Price',
                                    xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        # --- Create Technical Indicators Chart ---
        indicators_fig = go.Figure()
        if 'RSI_14' in stock_data.columns:
            indicators_fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI_14'], name='RSI'))
        if 'MACD_12_26_9' in stock_data.columns:
            indicators_fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD_12_26_9'], name='MACD'))
        indicators_fig.update_layout(title='Technical Indicators (RSI & MACD)',
                                   xaxis_title='Date', yaxis_title='Value',
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        # --- Prophet Forecasting ---
        prophet_df = stock_data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None) # Remove timezone for Prophet
        model = Prophet(daily_seasonality=True)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=90)
        forecast = model.predict(future)
        forecast_fig = plot_plotly(model, forecast)
        forecast_fig.update_layout(title='Future Price Prediction (Next 90 Days)',
                                 xaxis_title='Date', yaxis_title='Predicted Price')

        # --- Fundamental Data ---
        info = stock_ticker.info
        fundamental_data = [
            html.Li(f"Company Name: {info.get('longName', 'N/A')}", className="list-group-item bg-dark text-light"),
            html.Li(f"Sector: {info.get('sector', 'N/A')}", className="list-group-item bg-dark text-light"),
            html.Li(f"Industry: {info.get('industry', 'N/A')}", className="list-group-item bg-dark text-light"),
            html.Li(f"Market Cap: {info.get('marketCap', 0):,}", className="list-group-item bg-dark text-light"),
            html.Li(f"52 Week High/Low: {info.get('fiftyTwoWeekHigh', 'N/A')} / {info.get('fiftyTwoWeekLow', 'N/A')}", className="list-group-item bg-dark text-light"),
            html.Li(f"Forward P/E Ratio: {info.get('forwardPE', 'N/A')}", className="list-group-item bg-dark text-light"),
        ]

        # --- Risk Management ---
        risk_calculator = dbc.Card([
            dbc.CardHeader("Position Size Calculator"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(dbc.Input(id='account-size', placeholder='Account Size (e.g., 100000)', type='number'), width=6),
                    dbc.Col(dbc.Input(id='risk-percent', placeholder='Risk % per trade (e.g., 2)', type='number', value=2), width=6),
                ]),
                dbc.Row([
                    dbc.Col(dbc.Input(id='entry-price', placeholder='Entry Price', type='number', className="mt-2"), width=6),
                    dbc.Col(dbc.Input(id='stop-loss', placeholder='Stop-Loss Price', type='number', className="mt-2"), width=6),
                ]),
                dbc.Button("Calculate", id="calc-risk-button", className="mt-3 w-100"),
                html.Div(id="risk-output", className="mt-3 fw-bold text-center")
            ])
        ], className="mt-4")

        # --- Assemble the Final Layout ---
        final_layout = html.Div([
            dbc.Tabs([
                dbc.Tab(label="Price Action Analysis", children=[dcc.Graph(figure=candlestick_fig), dcc.Graph(figure=indicators_fig)]),
                dbc.Tab(label="AI Price Prediction", children=[dcc.Graph(figure=forecast_fig)]),
                dbc.Tab(label="Fundamental Analytics", children=[html.Ul(fundamental_data, className="list-group mt-4")]),
                dbc.Tab(label="Risk Management", children=[risk_calculator])
            ])
        ])

        return final_layout, ""

    except Exception as e:
        error_message = f"An error occurred: {e}. Please check the ticker symbol and try again."
        return None, dbc.Alert(error_message, color="danger", dismissable=True)

@app.callback(
    Output("risk-output", "children"),
    Input("calc-risk-button", "n_clicks"),
    [State("account-size", "value"),
     State("risk-percent", "value"),
     State("entry-price", "value"),
     State("stop-loss", "value")]
)
def calculate_position_size(n_clicks, account_size, risk_percent, entry_price, stop_loss):
    if n_clicks is None or not all([account_size, risk_percent, entry_price, stop_loss]):
        return ""
    try:
        risk_amount = float(account_size) * (float(risk_percent) / 100)
        risk_per_share = float(entry_price) - float(stop_loss)
        if risk_per_share <= 0:
            return "Stop-loss must be below entry price."
        position_size = risk_amount / risk_per_share
        return f"Position Size: {position_size:.2f} shares"
    except (ValueError, ZeroDivisionError) as e:
        return f"Error in calculation: {e}"

# --- [5] Main execution block ---
if __name__ == '__main__':
    app.run_server(debug=True)
