from transformers import pipeline

# Initialize sentiment analysis pipeline from Hugging Face Transformers
sentiment = pipeline('sentiment-analysis')

def get_overall_sentiment(article, threshold=0.6):
    """
    Determine the overall sentiment of a news article by comparing title and description sentiment.
    
    Parameters:
      article (dict): Dictionary containing sentiment labels and confidence scores.
      threshold (float): Confidence threshold below which sentiment is set to 'Neutral'.
    
    Returns:
      str: Overall sentiment ('POSITIVE', 'NEGATIVE', or 'Neutral').
    """
    title_sent = article.get('sentiment_title')
    title_conf = article.get('confidence_title', 0)
    desc_sent = article.get('sentiment_description')
    desc_conf = article.get('confidence_description', 0)

    if title_sent == desc_sent:
        return title_sent
    else:
        if title_conf >= desc_conf:
            chosen = title_sent
            chosen_conf = title_conf
        else:
            chosen = desc_sent
            chosen_conf = desc_conf
        if chosen_conf < threshold:
            return 'Neutral'
        return chosen

def analyze_sentiment_weighted(articles, calculate_weight_func):
    """
    Analyze news sentiment for a list of articles and compute a weighted sentiment score.
    
    Parameters:
      articles (list): List of processed news articles.
      calculate_weight_func (function): Function to calculate article weight.
      
    Returns:
      tuple: Total weighted sentiment score and the list of articles updated with sentiment info.
    """
    weighted_score = 0.0
    for article in articles:
        # Sentiment analysis on title
        sentiment_title = sentiment(article['title'])[0]
        # Sentiment analysis on description (if available)
        if article.get('description'):
            sentiment_description = sentiment(article['description'])[0]
        else:
            sentiment_description = {'label': 'Neutral', 'score': 0.0}
            
        overall_sent = get_overall_sentiment({
            'sentiment_title': sentiment_title['label'],
            'confidence_title': sentiment_title['score'],
            'sentiment_description': sentiment_description['label'],
            'confidence_description': sentiment_description['score']
        })
        
        weight = calculate_weight_func(article)
        
        # Assign score: Positive=+1, Negative=-1, Neutral=0
        if overall_sent == 'POSITIVE':
            score = 1
        elif overall_sent == 'NEGATIVE':
            score = -1
        else:
            score = 0
            
        weighted_score += score * weight
        
        # Store sentiment details in the article dictionary
        article['overall_sentiment'] = overall_sent
        article['weight'] = weight
        article['weighted_score'] = score * weight
    return weighted_score, articles