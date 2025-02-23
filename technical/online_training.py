# online_training.py
import yfinance as yf
import numpy as np

def online_training_step_for_meta(model, ticker, period, interval, scaler, sequence_size, features,
                                  feature_extractor_gru, feature_extractor_lstm,
                                  epochs=1, batch_size=32, checkpoint_path=None):
    """
    Fetch new data from Yahoo Finance, preprocess it, extract meta features, and update the meta model.
    """
    new_data_df = yf.download(tickers=ticker, period=period, interval=interval)
    if new_data_df.empty:
        print("No new data fetched from yfinance.")
        return None

    new_data_df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volumefrom'
    }, inplace=True)
    new_data_df['volumeto'] = new_data_df['volumefrom']
    new_data_df = new_data_df.dropna()

    data = new_data_df[features].values
    data_scaled = scaler.transform(data)

    if len(data_scaled) <= sequence_size:
        print("Not enough new data for online training. Skipping training step.")
        return None

    X_new, y_new = [], []
    target_index = features.index('close')
    for i in range(len(data_scaled) - sequence_size):
        X_new.append(data_scaled[i:i+sequence_size])
        y_new.append(data_scaled[i+sequence_size, target_index])
    X_new, y_new = np.array(X_new), np.array(y_new)

    features_gru_new = feature_extractor_gru.predict(X_new, verbose=0)
    features_lstm_new = feature_extractor_lstm.predict(X_new, verbose=0)
    X_new_meta = np.hstack((features_gru_new, features_lstm_new))

    history = model.fit(X_new_meta, y_new, epochs=epochs, batch_size=batch_size, verbose=1)

    if checkpoint_path is not None:
        model.save_weights(checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    return history

def online_training_step_for_base(model, ticker, period, interval, scaler, sequence_size, features,
                                  epochs=1, batch_size=32, checkpoint_path=None):
    """
    Fetch new data from Yahoo Finance, preprocess it, and update the base model (GRU or LSTM) via fine tuning.
    """
    new_data_df = yf.download(tickers=ticker, period=period, interval=interval)
    if new_data_df.empty:
        print("No new data fetched from yfinance for base model training.")
        return None

    new_data_df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volumefrom'
    }, inplace=True)
    new_data_df['volumeto'] = new_data_df['volumefrom']
    new_data_df = new_data_df.dropna()

    data = new_data_df[features].values
    data_scaled = scaler.transform(data)

    if len(data_scaled) <= sequence_size:
        print("Not enough new data for online training of base model. Skipping training step.")
        return None

    X_new, y_new = [], []
    target_index = features.index('close')
    for i in range(len(data_scaled) - sequence_size):
        X_new.append(data_scaled[i:i+sequence_size])
        y_new.append(data_scaled[i+sequence_size, target_index])
    X_new, y_new = np.array(X_new), np.array(y_new)

    history = model.fit(X_new, y_new, epochs=epochs, batch_size=batch_size, verbose=1)
    if checkpoint_path is not None:
        model.save_weights(checkpoint_path)
        print(f"Base model checkpoint saved at {checkpoint_path}")
    return history
