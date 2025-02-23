from datetime import datetime, timedelta

def get_start_date(analysis_type):
    """
    Get the start date based on analysis type.
    
    Parameters:
      analysis_type (int): 1 for short-term, 2 for medium-term, 3 for long-term.
    
    Returns:
      str: Start date in 'YYYY-MM-DD' format.
    """
    if analysis_type == 1:
        # Short-term: last 3 days
        return (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    elif analysis_type == 2:
        # Medium-term: last 7 days
        return (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    elif analysis_type == 3:
        # Long-term: last 30 days
        return (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    else:
        raise ValueError("Invalid analysis type")

def process_articles(articles):
    """
    Process raw news articles by extracting key information.
    
    Parameters:
      articles (list): List of articles from NewsAPI.
      
    Returns:
      list: List of dictionaries with processed article details.
    """
    processed_articles = []
    for article in articles:
        description = article.get('description', "")
        title = article.get('title', "")
        source = article.get('source', {}).get('name', "Unknown")
        published_at = article.get('publishedAt', "")
        processed_articles.append({
            'title': title,
            'description': description,
            'source': source,
            'published_at': published_at,
            'url': article.get('url', "")
        })
    return processed_articles

def calculate_weight(article):
    """
    Calculate the weight of a news article based on source credibility and publication time.
    
    Parameters:
      article (dict): Processed article dictionary.
      
    Returns:
      float: Calculated weight.
    """
    source_weights = {
         'CoinDesk': 1.5,
         'CoinTelegraph': 1.5,
         'CryptoSlate': 1.3,
         'Decrypt': 1.3,
         'Bitcoin Magazine': 1.4,
         'Bloomberg': 1.4,
         'Reuters': 1.4,
         'CNN': 1.2,
         'Unknown': 1.0
    }
    source = article.get('source', 'Unknown')
    source_weight = source_weights.get(source, 1.0)
    
    try:
        published_time = datetime.strptime(article['published_at'], '%Y-%m-%dT%H:%M:%SZ')
    except Exception:
        published_time = datetime.now()
        
    time_diff_hours = (datetime.now() - published_time).total_seconds() / 3600
    time_weight = 0.98 ** time_diff_hours
    return source_weight * time_weight
