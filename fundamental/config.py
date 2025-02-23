import os

# News API key: It is recommended to set this as an environment variable.
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '3e3849190f8f4e91ab89e13cc5bd0c8c')

# Cryptocurrency tickers dictionary
CRYPTO_TICKERS = {
    1: 'BTC',
    2: 'ETH',
    3: 'ADA',
    4: 'XRP',
    5: 'SOL'
}
