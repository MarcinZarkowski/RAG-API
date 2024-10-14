from newspaper import Article
import re
import pandas as pd 
import yfinance as yf


# Function to fetch and extract the title and body of an article
def extract_article_content(url, givenTitle):
    try:
        
        article = Article(url)
        article.download()
        article.parse()

        # Extract title and body
        title = article.title

        if title!=givenTitle:
            return None, f"Error Title mismatch, not searched article."
        
      
        body_text= re.sub(r'[^\w\s\$%\.,!?]', '', article.text.strip())


        body_text= re.sub(r'\s+', ' ', body_text)

        return  body_text
    except Exception as e:
        return  f"Error fetching article content: {str(e)}"
    
    
def table_to_string(df):
    return df.to_string(index=True)


async def get_stock_context(ticker):
    stock = yf.Ticker(ticker)
    historical_data = stock.history(period="max")
    news = stock.news
    articles_data = {}

    for article in news:
        title = article.get('title')
        url = article.get('link')
        string = extract_article_content(url, title)

        if "Error Title mismatch, not searched article." not in string and \
           "Error fetching article content:" not in string and \
           'Upgrade to read this MT Newswires article and get so much more.\n\nA Silver or Gold subscription plan is required to access premium news articles.' not in string:
            articles_data[title] = string

    stock_data = {
        'news': articles_data if articles_data else [],
        'calendar': stock.calendar.to_dict() if isinstance(stock.calendar, pd.DataFrame) else stock.calendar if stock.calendar else [],  
        'analyst share price targets': stock.analyst_price_targets if stock.analyst_price_targets else [],
    }

    more_stock_data = {
        'analyst recommendations': table_to_string(stock.recommendations) if stock.recommendations is not None else [],
        'earnings estimate': table_to_string(stock.earnings_estimate) if stock.earnings_estimate is not None else [],
        'revenue estimate': table_to_string(stock.revenue_estimate) if stock.revenue_estimate is not None else [],
        'earnings history': table_to_string(stock.earnings_history) if stock.earnings_history is not None else [],
        'earnings per share trend': table_to_string(stock.eps_trend) if stock.eps_trend is not None else [],
        'earnings per share revisions': table_to_string(stock.eps_revisions) if stock.eps_revisions is not None else [],
        'growth estimates': table_to_string(stock.growth_estimates) if stock.growth_estimates is not None else [],
        'last 60 days data ': table_to_string(historical_data.tail(60)) if historical_data is not None else [],
    }

    combined_stock_data = stock_data
    combined_stock_data.update(more_stock_data)

    return combined_stock_data