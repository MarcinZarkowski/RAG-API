import yfinance as yf
import pandas as pd
import re
from newspaper import Article

def table_to_string(df):
    """Converts a DataFrame to string representation."""
    return df.to_string(index=True) if isinstance(df, pd.DataFrame) else ""

def extract_article_content(url, givenTitle):
    """Extracts and returns content from the article at the provided URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()

        # Extract title and body
        title = article.title

        if title != givenTitle:
            return None, "Error Title mismatch, not searched article."
      
        # Clean and prepare body text
        body_text = re.sub(r'[^\w\s\$%\.,!?]', '', article.text.strip())
        body_text = re.sub(r'\s+', ' ', body_text)

        return body_text, None  # No error
    except Exception as e:
        return None, f"Error fetching article content: {str(e)}"

async def get_stock_context(ticker):
    """Fetches stock data, including news, calendar, and financial estimates."""
    stock = yf.Ticker(ticker)
    
    # Initialize variables to handle failures
    historical_data = None
    news = []
    articles_data = {}

    try:
        historical_data = stock.history(period="max")
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
    
    try:
        news = stock.news
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")

    # Process each article in the news
    for article in news:
        title = article.get('title')
        url = article.get('link')
        content, error = extract_article_content(url, title)

        if error:
            # If there's an error with content extraction, skip this article
            print(f"Error extracting content for article: {title}. Error: {error}")
            continue
        elif content:
            # If the content is extracted successfully
            articles_data[title] = content

    stock_data = {
        'news': articles_data if articles_data else [],
        'calendar': stock.calendar.to_dict() if isinstance(stock.calendar, pd.DataFrame) else stock.calendar or [],  
        'analyst share price targets': stock.analyst_price_targets or [],
    }

    more_stock_data = {}

    # Wrap the table extraction with try-except blocks to handle potential failures
    try:
        more_stock_data['analyst recommendations'] = table_to_string(stock.recommendations)
    except Exception as e:
        print(f"Error processing analyst recommendations for {ticker}: {e}")

    try:
        more_stock_data['earnings estimate'] = table_to_string(stock.earnings_estimate)
    except Exception as e:
        print(f"Error processing earnings estimate for {ticker}: {e}")

    try:
        more_stock_data['revenue estimate'] = table_to_string(stock.revenue_estimate)
    except Exception as e:
        print(f"Error processing revenue estimate for {ticker}: {e}")

    try:
        more_stock_data['earnings history'] = table_to_string(stock.earnings_history)
    except Exception as e:
        print(f"Error processing earnings history for {ticker}: {e}")

    try:
        more_stock_data['earnings per share trend'] = table_to_string(stock.eps_trend)
    except Exception as e:
        print(f"Error processing earnings per share trend for {ticker}: {e}")

    try:
        more_stock_data['earnings per share revisions'] = table_to_string(stock.eps_revisions)
    except Exception as e:
        print(f"Error processing earnings per share revisions for {ticker}: {e}")

    try:
        more_stock_data['growth estimates'] = table_to_string(stock.growth_estimates)
    except Exception as e:
        print(f"Error processing growth estimates for {ticker}: {e}")

    try:
        more_stock_data['last 60 days data'] = table_to_string(historical_data.tail(60))
    except Exception as e:
        print(f"Error processing last 60 days data for {ticker}: {e}")
    
    # Combine the stock data with additional information, filtering out empty values
    combined_stock_data = {**stock_data, **{key: value for key, value in more_stock_data.items() if value}}

    return combined_stock_data
