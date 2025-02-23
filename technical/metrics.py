# metrics.py
import numpy as np
from sklearn.metrics import mean_absolute_error

def mean_absolute_percentage_error_custom(y_true, y_pred):
    """Custom MAPE calculation."""
    y_true = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """Symmetric MAPE calculation."""
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

def mean_absolute_scaled_error(y_true, y_pred):
    """Mean Absolute Scaled Error calculation."""
    mae = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    return mae / mae_naive

def mase(y_true, y_pred):
    """MASE calculation using sklearn's MAE."""
    mae = mean_absolute_error(y_true, y_pred)
    mae_naive = mean_absolute_error(y_true[1:], y_true[:-1])
    return mae / mae_naive
