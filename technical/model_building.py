# model_building.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

def build_gru_base(hp, sequence_size, features):
    """Build GRU base model using hyperparameters from keras_tuner."""
    units = hp.Int('units_gru', min_value=64, max_value=512, step=64)
    dropout_rate = hp.Float('dropout_gru', 0.1, 0.5, step=0.1)
    batch_norm_choice = hp.Choice('batch_norm_gru', [True, False])
    learning_rate = hp.Choice('lr_gru', [1e-3, 1e-4, 1e-5])
    inputs = Input(shape=(sequence_size, len(features)))
    x = GRU(units, return_sequences=True)(inputs)
    if batch_norm_choice:
        x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = GRU(units, return_sequences=False)(x)
    if batch_norm_choice:
        x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    feature_output = Dense(50, activation='relu', name='features')(x)
    prediction = Dense(1, activation='linear', name='prediction')(feature_output)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def build_lstm_base(hp, sequence_size, features):
    """Build LSTM base model using hyperparameters from keras_tuner."""
    units = hp.Int('units_lstm', min_value=64, max_value=512, step=64)
    dropout_rate = hp.Float('dropout_lstm', 0.1, 0.5, step=0.1)
    batch_norm_choice = hp.Choice('batch_norm_lstm', [True, False])
    learning_rate = hp.Choice('lr_lstm', [1e-3, 1e-4, 1e-5])
    inputs = Input(shape=(sequence_size, len(features)))
    x = LSTM(units, return_sequences=True)(inputs)
    if batch_norm_choice:
        x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = LSTM(units, return_sequences=False)(x)
    if batch_norm_choice:
        x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    feature_output = Dense(50, activation='relu', name='features')(x)
    prediction = Dense(1, activation='linear', name='prediction')(feature_output)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def build_meta_model():
    """Build the meta model for stacking features from base models."""
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    inputs = Input(shape=(100,))
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    prediction = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model
