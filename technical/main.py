#!/usr/bin/env python
"""
Main entry point for the Crypto Prediction and Analysis product.
This script coordinates the execution of the product using modular components.
"""

from config import *
from utils import select_currency, select_analysis_type, setup_directories, fetch_data_yfinance, plot_historical_forecast
from data_processing import split_normalize_data, create_multivariate_sequences
from training import setup_checkpoints, train_base_models, train_meta_model, extract_features, forecast_prices
from online_training import online_training_step_for_meta, online_training_step_for_base
from metrics import mean_absolute_percentage_error_custom, mase
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Model

def main():
    # --- User Input and Setup ---
    selected_currency = select_currency()
    period, interval, sequence_size, forecast_steps = select_analysis_type()
    model_folder, tuner_dir_gru, tuner_dir_lstm = setup_directories(selected_currency)
    yahoo_tickers = {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD',
        'ADA': 'ADA-USD',
        'XRP': 'XRP-USD',
        'SOL': 'SOL-USD'
    }
    selected_ticker = yahoo_tickers[selected_currency]
    
    # --- Data Fetching ---
    df = fetch_data_yfinance(selected_ticker, period, interval)
    if df is None or df.empty:
        print("Failed to fetch data. Please try again later.")
        exit()
    print(f"Fetched data head:\n{df.head()}")
    print(f"Fetched data tail:\n{df.tail()}")
    
    # --- Plot Historical Data ---
    plot_historical_forecast(df, forecast_steps, interval, selected_currency)
    
    # --- Data Preprocessing ---
    features = ['open', 'high', 'low', 'close']
    train_df, test_df, df, scaler, train_scaled, test_scaled = split_normalize_data(df, features)
    X_train, y_train = create_multivariate_sequences(train_scaled, sequence_size, features)
    X_test, y_test = create_multivariate_sequences(test_scaled, sequence_size, features)
    
    # --- Checkpoint Setup ---
    cp_gru, cp_lstm, cp_meta, cp_path_gru, cp_path_lstm, cp_path_meta = setup_checkpoints(model_folder)
    
    # --- Train Base Models ---
    train_base_models(X_train, y_train, tuner_dir_gru, tuner_dir_lstm, cp_gru, cp_lstm, sequence_size, features)
    
    # --- Feature Extraction for Meta Model ---
    X_train_meta, X_test_meta = extract_features(X_train, X_test)
    
    # --- Train Meta Model ---
    meta_model = train_meta_model(X_train_meta, y_train, cp_meta)
    
    # --- Extract Feature Extractors for Online Training ---
    feature_extractor_gru = Model(inputs=gru_base_model.input, outputs=gru_base_model.get_layer('features').output)
    feature_extractor_lstm = Model(inputs=lstm_base_model.input, outputs=lstm_base_model.get_layer('features').output)
    
    # --- Evaluate Meta Model ---
    def inverse_scale(y_pred, scaler, features):
        y_pred_scaled = np.zeros((len(y_pred), len(features)))
        y_pred_scaled[:, features.index('close')] = y_pred
        return scaler.inverse_transform(y_pred_scaled)[:, features.index('close')]
    
    yhat_test_meta = meta_model.predict(X_test_meta, verbose=0).flatten()
    y_test_inverse = inverse_scale(y_test, scaler, features)
    yhat_test_meta_inverse = inverse_scale(yhat_test_meta, scaler, features)
    
    mae_meta = mean_absolute_error(y_test_inverse, yhat_test_meta_inverse)
    mse_meta = mean_squared_error(y_test_inverse, yhat_test_meta_inverse)
    rmse_meta = np.sqrt(mse_meta)
    mape_meta = mean_absolute_percentage_error_custom(y_test_inverse, yhat_test_meta_inverse)
    r2_meta = r2_score(y_test_inverse, yhat_test_meta_inverse)
    mase_meta = mase(y_test_inverse, yhat_test_meta_inverse)
    
    print("\n--- Stacking Ensemble (Meta Model) Evaluation ---")
    print("MAE:", mae_meta)
    print("MSE:", mse_meta)
    print("RMSE:", rmse_meta)
    print("MAPE:", mape_meta)
    print("MASE:", mase_meta)
    print("R^2:", r2_meta)
    
    # --- Forecast Future Prices ---
    forecast_prices(test_scaled, forecast_steps, sequence_size, scaler, test_df)
    
    # --- Online Training Examples ---
    import os
    online_checkpoint_path = os.path.join(model_folder, 'meta_online_checkpoint.weights.h5')
    # Fine tune meta model with new daily data
    online_training_step_for_meta(model=meta_model,
                                  ticker=selected_ticker,
                                  period='1d',
                                  interval=interval,
                                  scaler=scaler,
                                  sequence_size=sequence_size,
                                  features=features,
                                  feature_extractor_gru=feature_extractor_gru,
                                  feature_extractor_lstm=feature_extractor_lstm,
                                  epochs=5,
                                  batch_size=32,
                                  checkpoint_path=online_checkpoint_path)
    
    # Fine tune base models with new daily data
    online_training_step_for_base(model=gru_base_model,
                                  ticker=selected_ticker,
                                  period='1d',
                                  interval=interval,
                                  scaler=scaler,
                                  sequence_size=sequence_size,
                                  features=features,
                                  epochs=5,
                                  batch_size=32,
                                  checkpoint_path=cp_path_gru)
    
    online_training_step_for_base(model=lstm_base_model,
                                  ticker=selected_ticker,
                                  period='1d',
                                  interval=interval,
                                  scaler=scaler,
                                  sequence_size=sequence_size,
                                  features=features,
                                  epochs=5,
                                  batch_size=32,
                                  checkpoint_path=cp_path_lstm)

if __name__ == "__main__":
    main()
