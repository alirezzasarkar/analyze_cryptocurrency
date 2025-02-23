# Crypto Prediction and Analysis

This project provides a platform for performing technical and fundamental analysis on cryptocurrencies using deep learning models. The system leverages GRU, LSTM, and a stacking ensemble meta model to forecast future prices and analyze trends.

## Features

- **Data Collection:** Automatically fetch historical cryptocurrency data from Yahoo Finance with a robust retry mechanism.
- **Data Preprocessing:** Split, normalize, and create time-series sequences from raw data.
- **Model Training:** Train base models (GRU and LSTM) using hyperparameter tuning with walk-forward cross-validation.
- **Meta Model:** Build a stacking ensemble meta model that integrates features extracted from the base models.
- **Forecasting:** Forecast future prices and determine trends (Uptrend, Downtrend, or Sideways).
- **Online Fine Tuning:** Update both base models and the meta model with new data via online training.
- **Visualization:** Interactive plotting of historical and forecasted prices using Plotly.

## Project Structure

The project is modularized into separate files for maintainability:

- **config.py:** Contains global configurations such as random seeds, logging setup, and Plotly defaults.
- **metrics.py:** Defines custom evaluation metrics.
- **utils.py:** Provides utility functions for user input, directory setup, data fetching, and plotting.
- **data_processing.py:** Includes functions for data splitting, normalization, and sequence creation.
- **model_building.py:** Contains functions to build GRU, LSTM, and meta models.
- **training.py:** Manages the training process including hyperparameter tuning (using a custom walk-forward CV tuner) and checkpointing.
- **online_training.py:** Defines functions for online fine tuning of the base and meta models with new data.
- **main.py:** The main entry point that coordinates the overall workflow.

## Installation

Install the required libraries using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

````

## Usage

Run the main script to start the analysis and forecasting process:

```bash
python main.py
```

The application will prompt you to:

- **Select a Cryptocurrency:** Choose from BTC, ETH, ADA, XRP, or SOL.
- **Select Analysis Type:** Choose between short-term (hourly data), medium-term (daily data), or long-term (daily data).

The system will then fetch data, preprocess it, train the models, and finally output forecasts along with trend analysis.

## Requirements

The following libraries (and their dependencies) are required:

- numpy
- pandas
- scikit-learn
- tensorflow
- keras-tuner
- plotly
- yfinance
- matplotlib
- seaborn

For the complete list, please refer to the `requirements.txt` file.

## Notes

- This project is designed for interactive environments such as Google Colab, but can be adapted for production use.
- For production deployment, consider converting the code into a package and integrating it with a web framework.
- Ensure that your environment has sufficient computational resources (e.g., a GPU) for training deep learning models.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests for improvements or bug fixes.
````
