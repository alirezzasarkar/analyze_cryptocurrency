# data_processing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def split_normalize_data(df, features, split_ratio=0.8):
    """Split the dataframe into training and testing sets and normalize the features."""
    split_index = int(len(df) * split_ratio)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    df = pd.concat([train_df, test_df]).dropna()
    scaler = MinMaxScaler()
    scaler.fit(train_df[features])
    train_scaled = scaler.transform(train_df[features])
    test_scaled = scaler.transform(test_df[features])
    return train_df, test_df, df, scaler, train_scaled, test_scaled

def create_multivariate_sequences(dataset, seq_size, features, target_column='close'):
    """Create sequences from the scaled dataset for time series forecasting."""
    X, y = [], []
    target_index = features.index(target_column)
    for i in range(len(dataset) - seq_size):
        X.append(dataset[i:i+seq_size])
        y.append(dataset[i+seq_size, target_index])
    return np.array(X), np.array(y)
