import sys
import pandas as pd
from config import CRYPTO_TICKERS
from news_fetch import fetch_crypto_news
from utils import get_start_date, process_articles, calculate_weight
from sentiment import analyze_sentiment_weighted

def main():
    # Display available cryptocurrencies
    print("List of available cryptocurrencies:")
    for key, value in CRYPTO_TICKERS.items():
        print(f"{key}: {value}")
    
    try:
        selected_crypto_index = int(input("Enter the number of the desired cryptocurrency: "))
        selected_crypto = CRYPTO_TICKERS.get(selected_crypto_index)
        if selected_crypto is None:
            raise ValueError
    except ValueError:
        print("Invalid selection! Please enter a valid number.")
        sys.exit()
    
    print("\nSelect the type of analysis:")
    print("1: Short-term (hourly data) - Last 3 days")
    print("2: Medium-term (daily data) - Last 7 days")
    print("3: Long-term (daily data) - Last 30 days")
    try:
        selected_analysis = int(input("Enter the analysis type number: "))
    except ValueError:
        print("Invalid analysis selection!")
        sys.exit()
    
    try:
        start_date = get_start_date(selected_analysis)
    except ValueError:
        print("Invalid analysis selection. Please try again.")
        sys.exit()
    
    print(f"\nFetching news related to {selected_crypto} from {start_date}...")
    articles = fetch_crypto_news(selected_crypto, start_date, page_size=10, max_pages=5)
    
    processed_articles = process_articles(articles)
    
    total_weighted_score, analyzed_articles = analyze_sentiment_weighted(processed_articles, calculate_weight)
    
    print(f"\nTotal Weighted Sentiment Score: {total_weighted_score}")
    
    if total_weighted_score > 0.5:
        recommendation = "Buy Recommendation"
    elif total_weighted_score < -0.5:
        recommendation = "Sell Recommendation"
    else:
        recommendation = "No Recommendation"
    
    print(f"\nAnalysis Result: {recommendation}")
    
    # Save analysis results to a CSV file
    df = pd.DataFrame(analyzed_articles)
    df.to_csv("news_sentiment_analysis.csv", index=False)
    print("\nAnalysis results saved to news_sentiment_analysis.csv")

if __name__ == "__main__":
    main()