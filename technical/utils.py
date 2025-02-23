# utils.py
import os
import time
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
import logging

def select_currency():
    """Let the user select a cryptocurrency."""
    currencies = ['BTC', 'ETH', 'ADA', 'XRP', 'SOL']
    print("List of available cryptocurrencies:")
    for idx, currency in enumerate(currencies):
        print(f"{idx + 1}. {currency}")
    try:
        choice = int(input("Choose a cryptocurrency (number): ")) - 1
        if choice not in range(len(currencies)):
            raise ValueError
    except ValueError:
        print("Invalid choice! Please enter a valid number.")
        logging.error("User selected an invalid cryptocurrency choice.")
        exit()
    selected_currency = currencies[choice]
    print(f"Selected cryptocurrency: {selected_currency}")
    logging.info(f"Selected cryptocurrency: {selected_currency}")
    return selected_currency

def select_analysis_type():
    """Let the user select analysis type and set corresponding parameters."""
    print("\nSelect the type of analysis:")
    print("1. Short-term (hourly data)")
    print("2. Medium-term (daily data)")
    print("3. Long-term (daily data)")
    try:
        analysis_choice = int(input("Enter your choice (1, 2, or 3): "))
        if analysis_choice not in [1, 2, 3]:
            raise ValueError
    except ValueError:
        print("Invalid choice! Please select 1, 2, or 3.")
        logging.error("User selected an invalid analysis type.")
        exit()
    if analysis_choice == 1:
        return "2y", "1h", 20, 12
    elif analysis_choice == 2:
        return "max", "1d", 40, 7
    elif analysis_choice == 3:
        return "max", "1d", 40, 30

def setup_directories(selected_currency):
    """Create directories for model checkpoints and tuner results."""
    model_folder = f'/content/drive/MyDrive/model_checkpoints/{selected_currency}'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    tuner_dir_gru = os.path.join('tuner_dir_gru_base', selected_currency)
    tuner_dir_lstm = os.path.join('tuner_dir_lstm_base', selected_currency)
    if not os.path.exists(tuner_dir_gru):
        os.makedirs(tuner_dir_gru)
    if not os.path.exists(tuner_dir_lstm):
        os.makedirs(tuner_dir_lstm)
    return model_folder, tuner_dir_gru, tuner_dir_lstm

def fetch_data_yfinance(ticker, period, interval, max_retries=5, wait_time=10):
    """
    Fetch historical data from Yahoo Finance with a retry mechanism.
    """
    for attempt in range(max_retries):
        try:
            data = yf.download(tickers=ticker, period=period, interval=interval)
            if not data.empty:
                data.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volumefrom'
                }, inplace=True)
                data['volumeto'] = data['volumefrom']
                logging.info(f"Data successfully downloaded for {ticker}.")
                return data
            else:
                raise Exception("Failed to fetch data from Yahoo Finance.")
        except Exception as e:
            print(f"Error fetching data: {e}")
            logging.error(f"Error fetching data: {e}")
            if "Rate limited" in str(e):
                wait = wait_time * (attempt + 1)
                print(f"Rate limit encountered! Retrying in {wait} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                break
    print("Max retries reached. Could not fetch data.")
    logging.error("Max retries reached. Could not fetch data.")
    return None

def plot_historical_forecast(df, forecast_steps, interval, selected_currency):
    """Plot historical prices with placeholders for forecasted values using Plotly."""
    last_date = df.index[-1]
    freq = 'H' if interval == "1h" else 'D'
    forecast_dates = pd.date_range(last_date, periods=forecast_steps+1, freq=freq)[1:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['close'],
                             mode='lines', name='Historical Close Price'))
    fig.add_trace(go.Scatter(x=forecast_dates, y=[None]*len(forecast_dates),
                             mode='lines+markers', name='Forecasted Future Prices'))
    fig.update_layout(
        title=f"{selected_currency} - Historical and Forecasted Prices",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis=dict(rangeslider=dict(visible=True), type="date")
    )
    fig.show()
