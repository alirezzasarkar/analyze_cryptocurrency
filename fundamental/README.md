# Crypto News Sentiment Analysis

This project is a module-based application that fetches cryptocurrency news using NewsAPI and performs sentiment analysis using Hugging Face Transformers. It processes news articles related to a selected cryptocurrency, calculates weighted sentiment scores based on source credibility and publication time, and provides a recommendation.

## Features

- **News Collection:** Retrieve news articles for a selected cryptocurrency using the NewsAPI.
- **Sentiment Analysis:** Analyze sentiment on news article titles and descriptions using a pre-trained sentiment analysis model from Hugging Face.
- **Weighted Sentiment Calculation:** Calculate sentiment weights based on the news source credibility and publication time.
- **Recommendation:** Provide a buy, sell, or neutral recommendation based on the weighted sentiment score.
- **Modular Structure:** The code is organized into separate modules (config, news_fetch, sentiment, utils, main) for maintainability and reusability.
- **CSV Export:** Save analyzed results to a CSV file.

## Project Structure

- **config.py:** Contains configuration settings such as the News API key and cryptocurrency tickers.
- **news_fetch.py:** Contains functions to fetch news articles using NewsAPI.
- **sentiment.py:** Contains functions to perform sentiment analysis and calculate weighted sentiment.
- **utils.py:** Provides utility functions for processing articles and calculating weights.
- **main.py:** The main script that coordinates user input, news fetching, sentiment analysis, and outputs recommendations.
- **README.md:** Project documentation.
- **requirements.txt:** List of required libraries.

## Installation

Install the required libraries using:

```bash
pip install -r requirements.txt
```

````

## Usage

Run the main script:

```bash
python main.py
```

Follow the prompts to select a cryptocurrency and analysis type. The results will be printed to the console and saved to a CSV file.

## Requirements

- Python 3.7+
- A valid NewsAPI key (set in the environment variable `NEWS_API_KEY` or directly in `config.py`)

## Notes

- For production use, it is recommended to store sensitive keys as environment variables.
- This project can be run in any Python environment, including Jupyter Notebooks or Google Colab.
- For running in Google Colab, remember to mount your Google Drive if you need persistent storage.

````
