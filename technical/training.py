# training.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
from sklearn.model_selection import TimeSeriesSplit
import logging

from model_building import build_gru_base, build_lstm_base, build_meta_model

# Custom Tuner class using Walk-Forward CV
class WalkForwardCVTuner(kt.BayesianOptimization):
    """Custom tuner using walk-forward cross-validation."""
    def run_trial(self, trial, X, y, batch_size=64, epochs=20, **fit_kwargs):
        tscv = TimeSeriesSplit(n_splits=5)
        val_losses = []
        for train_idx, val_idx in tscv.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            model = self.hypermodel.build(trial.hyperparameters)
            model.compile(optimizer=model.optimizer, loss=model.loss)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            history = model.fit(X_train_fold, y_train_fold,
                                validation_data=(X_val_fold, y_val_fold),
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=[early_stopping],
                                verbose=0)
            val_losses.append(min(history.history['val_loss']))
        mean_val_loss = float(np.mean(val_losses))
        self.oracle.update_trial(trial.trial_id, {'val_loss': mean_val_loss})

def setup_checkpoints(model_folder):
    """Set up checkpoint paths and callbacks for models."""
    checkpoint_path_gru = os.path.join(model_folder, 'gru_checkpoint.weights.h5')
    checkpoint_path_lstm = os.path.join(model_folder, 'lstm_checkpoint.weights.h5')
    checkpoint_path_meta = os.path.join(model_folder, 'meta_checkpoint.weights.h5')
    
    checkpoint_callback_gru = ModelCheckpoint(
        filepath=checkpoint_path_gru,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='min'
    )
    checkpoint_callback_lstm = ModelCheckpoint(
        filepath=checkpoint_path_lstm,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='min'
    )
    checkpoint_callback_meta = ModelCheckpoint(
        filepath=checkpoint_path_meta,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='min'
    )
    return checkpoint_callback_gru, checkpoint_callback_lstm, checkpoint_callback_meta, checkpoint_path_gru, checkpoint_path_lstm, checkpoint_path_meta

def train_base_models(X_train, y_train, tuner_dir_gru, tuner_dir_lstm, cp_gru, cp_lstm, sequence_size, features):
    """Train GRU and LSTM base models using the custom tuner."""
    global gru_base_model, lstm_base_model  # to be used later by feature extraction and online training
    
    # GRU model training
    tuner_gru = WalkForwardCVTuner(
        hypermodel=lambda hp: build_gru_base(hp, sequence_size, features),
        objective='val_loss',
        max_trials=20,
        directory=os.path.join(tuner_dir_gru),
        project_name='gru_base_enhanced'
    )
    tuner_gru.search(X_train, y_train, epochs=50, batch_size=64, verbose=1)
    best_hps_gru = tuner_gru.get_best_hyperparameters(num_trials=1)[0]
    gru_base_model = tuner_gru.hypermodel.build(best_hps_gru)
    from tensorflow.keras.optimizers import Adam
    gru_base_model.compile(optimizer=Adam(learning_rate=best_hps_gru.get('lr_gru')), loss='mse')
    gru_base_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        verbose=1,
        callbacks=[EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
                   cp_gru]
    )
    
    # LSTM model training
    tuner_lstm = WalkForwardCVTuner(
        hypermodel=lambda hp: build_lstm_base(hp, sequence_size, features),
        objective='val_loss',
        max_trials=20,
        directory=os.path.join(tuner_dir_lstm),
        project_name='lstm_base_enhanced'
    )
    tuner_lstm.search(X_train, y_train, epochs=50, batch_size=64, verbose=1)
    best_hps_lstm = tuner_lstm.get_best_hyperparameters(num_trials=1)[0]
    lstm_base_model = tuner_lstm.hypermodel.build(best_hps_lstm)
    lstm_base_model.compile(optimizer=Adam(learning_rate=best_hps_lstm.get('lr_lstm')), loss='mse')
    lstm_base_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        verbose=1,
        callbacks=[EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
                   cp_lstm]
    )

def train_meta_model(X_train_meta, y_train, cp_meta):
    """Train the meta model that stacks features from base models."""
    global meta_model  # to be used later for forecasting and online training
    meta_model = build_meta_model()
    meta_model.fit(
        X_train_meta, y_train,
        epochs=100,
        batch_size=64,
        verbose=1,
        callbacks=[EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
                   cp_meta]
    )

def extract_features(X_train, X_test):
    """Extract features from base models using the trained feature extractors."""
    from tensorflow.keras.models import Model
    global gru_base_model, lstm_base_model
    feature_extractor_gru = Model(inputs=gru_base_model.input, outputs=gru_base_model.get_layer('features').output)
    feature_extractor_lstm = Model(inputs=lstm_base_model.input, outputs=lstm_base_model.get_layer('features').output)
    
    features_gru_train = feature_extractor_gru.predict(X_train, verbose=0)
    features_lstm_train = feature_extractor_lstm.predict(X_train, verbose=0)
    features_gru_test = feature_extractor_gru.predict(X_test, verbose=0)
    features_lstm_test = feature_extractor_lstm.predict(X_test, verbose=0)
    
    X_train_meta = np.hstack((features_gru_train, features_lstm_train))
    X_test_meta = np.hstack((features_gru_test, features_lstm_test))
    return X_train_meta, X_test_meta

def forecast_prices(test_scaled, forecast_steps, sequence_size, scaler, test_df):
    """Forecast future prices using the meta model and extracted features."""
    global meta_model, gru_base_model, lstm_base_model
    from tensorflow.keras.models import Model
    feature_extractor_gru = Model(inputs=gru_base_model.input, outputs=gru_base_model.get_layer('features').output)
    feature_extractor_lstm = Model(inputs=lstm_base_model.input, outputs=lstm_base_model.get_layer('features').output)
    
    last_sequence = test_scaled[-sequence_size:]
    current_sequence = last_sequence.copy()
    forecast_list = []
    
    for i in range(forecast_steps):
        input_seq = np.expand_dims(current_sequence, axis=0)
        features_gru_pred = feature_extractor_gru.predict(input_seq, verbose=0)
        features_lstm_pred = feature_extractor_lstm.predict(input_seq, verbose=0)
        meta_input = np.hstack((features_gru_pred, features_lstm_pred))
        next_close_scaled = meta_model.predict(meta_input, verbose=0)[0, 0]
        forecast_list.append(next_close_scaled)
        new_row = np.array([next_close_scaled, next_close_scaled, next_close_scaled, next_close_scaled])
        current_sequence = np.vstack((current_sequence[1:], new_row))
    
    forecast_scaled_array = np.array([[val, val, val, val] for val in forecast_list])
    forecast_unscaled = scaler.inverse_transform(forecast_scaled_array)[:, 3]
    
    last_actual_close = float(test_df['close'].iloc[-1])
    threshold = 0.01
    final_forecast = float(forecast_unscaled[-1])
    if final_forecast > last_actual_close * (1 + threshold):
        trend = "Uptrend"
    elif final_forecast < last_actual_close * (1 - threshold):
        trend = "Downtrend"
    else:
        trend = "Sideways"
    
    print("\nForecasted Prices:")
    print(forecast_unscaled)
    print(f"\nLast Actual Close Price: {last_actual_close}")
    print(f"Final Forecast Price: {final_forecast}")
    print(f"Predicted Trend: {trend}")
    logging.info(f"Forecasted future prices: {forecast_unscaled}")
    logging.info(f"Predicted trend: {trend}")
    
    plt.figure(figsize=(10,5))
    plt.plot(forecast_unscaled, marker='o', linestyle='-', color='blue')
    plt.title("Forecasted Future Prices")
    plt.xlabel("Forecast Step")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()
