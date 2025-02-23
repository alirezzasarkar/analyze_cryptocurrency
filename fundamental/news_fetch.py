from newsapi import NewsApiClient
import logging
from config import NEWS_API_KEY

# Initialize NewsAPI client with the API key from config.py
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

def fetch_crypto_news(ticker, start_date, page_size=10, max_pages=5):
    """
    Fetch cryptocurrency news using NewsAPI with pagination support.
    
    Parameters:
      ticker (str): The cryptocurrency ticker (e.g., 'BTC').
      start_date (str): Start date in 'YYYY-MM-DD' format.
      page_size (int): Number of articles per page.
      max_pages (int): Maximum number of pages to fetch.
      
    Returns:
      list: A list of news articles.
    """
    all_articles = []
    for page in range(1, max_pages + 1):
        response = newsapi.get_everything(
            q=ticker,
            language='en',
            from_param=start_date,
            sort_by='relevancy',
            page_size=page_size,
            page=page
        )
        articles = response.get('articles', [])
        if not articles:
            break
        all_articles.extend(articles)
        if len(articles) < page_size:
            break
    return all_articles