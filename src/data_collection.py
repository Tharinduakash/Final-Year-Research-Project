"""
Data collection module for stock prices, news, and social media data.
Sources: Yahoo Finance, Alpha Vantage, World Bank, IMF,
         NewsAPI, Twitter (v2), Reddit (PRAW)
"""

import yfinance as yf
import pandas as pd
import praw
import tweepy
import requests
import os
import time
from alpha_vantage.timeseries import TimeSeries
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/cache").mkdir(parents=True, exist_ok=True)


class DataCollector:
    def __init__(self):
        self.alpha_vantage_key        = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.news_api_key             = os.getenv('NEWS_API_KEY')
        self.twitter_bearer_token     = os.getenv('TWITTER_BEARER_TOKEN')
        self.reddit_client_id         = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret     = os.getenv('REDDIT_CLIENT_SECRET')
        self.world_bank_base_url      = "https://api.worldbank.org/v2"
        self.imf_base_url             = "http://dataservices.imf.org/REST/SDMX_JSON.svc"
        self._alpha_vantage_last_call = 0

    def get_stock_data_yahoo(self, ticker, start_date, end_date):
        """Fetch OHLCV stock data from Yahoo Finance."""
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        data.index = pd.to_datetime(data.index)
        return data

    def get_stock_data_alpha_vantage(self, ticker):
        """
        Fetch daily stock data from Alpha Vantage.
        Auto-throttles to respect the free-tier 5 calls/min limit.
        """
        elapsed = time.time() - self._alpha_vantage_last_call
        if elapsed < 12:
            time.sleep(12 - elapsed)

        ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
        data, _ = ts.get_daily(symbol=ticker, outputsize='full')
        self._alpha_vantage_last_call = time.time()

        data.index = pd.to_datetime(data.index)
        data.sort_index(inplace=True)
        data.columns = [c.split('. ')[1].capitalize() for c in data.columns]
        return data

    def get_multiple_stocks(self, tickers, start_date, end_date):
        """
        Fetch OHLCV data for multiple tickers at once.

        Returns:
            dict mapping ticker -> DataFrame
        """
        return {
            ticker: self.get_stock_data_yahoo(ticker, start_date, end_date)
            for ticker in tickers
        }

    def get_economic_data_world_bank(self, country_code="USA", indicator="NY.GDP.MKTP.CD",
                                     start_year=2010, end_year=2023):
        """
        Fetch macroeconomic data from World Bank API.

        Common indicators:
            NY.GDP.MKTP.CD  — GDP (current US$)
            FP.CPI.TOTL.ZG  — CPI Inflation
            FR.INR.RINR     — Real interest rate
        """
        url    = f"{self.world_bank_base_url}/country/{country_code}/indicator/{indicator}"
        params = {"format": "json", "date": f"{start_year}:{end_year}", "per_page": 1000}
        data   = requests.get(url, params=params, timeout=30).json()

        if len(data) > 1 and data[1]:
            df = pd.DataFrame(data[1])
            df['date']  = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df[['date', 'value']].set_index('date').sort_index()
            df.rename(columns={'value': indicator}, inplace=True)
            return df.dropna()
        return pd.DataFrame()

    def get_economic_data_imf(self, country_code="US", indicator="NGDP_R",
                               start_year=2010, end_year=2023):
        """Fetch economic data from IMF SDMX API."""
        url  = f"{self.imf_base_url}/CompactData/IFS/M.{country_code}.{indicator}"
        data = requests.get(url, timeout=30).json()

        if 'CompactData' in data and 'DataSet' in data['CompactData']:
            series_data = data['CompactData']['DataSet'].get('Series', {})
            records = [
                {'date': obs['@TIME_PERIOD'], 'value': float(obs['@OBS_VALUE'])}
                for obs in series_data.get('Obs', [])
                if '@OBS_VALUE' in obs
            ]
            if records:
                df = pd.DataFrame(records)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                df.rename(columns={'value': indicator}, inplace=True)
                return df[
                    (df.index.year >= start_year) & (df.index.year <= end_year)
                ].dropna()
        return pd.DataFrame()

    def get_central_bank_data_sri_lanka(self, start_date, end_date):
        """
        Fetch economic data from Central Bank of Sri Lanka (CBSL).
        NOTE: CBSL has no public REST API. Replace with actual scraping
        from https://www.cbsl.gov.lk/en/statistics when implementing.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='ME')
        data = {
            'interest_rate': [4.5 + 0.1 * i for i in range(len(dates))],
            'inflation':     [2.0 + 0.05 * i for i in range(len(dates))]
        }
        return pd.DataFrame(data, index=dates)


    def get_news_data(self, query, from_date, to_date):
        """
        Fetch news articles from NewsAPI.

        Returns:
            List of article dicts: title, description, content,
                                   publishedAt, source, url
        """
        newsapi  = NewsApiClient(api_key=self.news_api_key)
        response = newsapi.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language='en',
            sort_by='relevancy',
            page_size=100
        )
        return [
            {
                'title':       a.get('title', ''),
                'description': a.get('description', ''),
                'content':     a.get('content', ''),
                'publishedAt': a.get('publishedAt', ''),
                'source':      a.get('source', {}).get('name', ''),
                'url':         a.get('url', '')
            }
            for a in response.get('articles', [])
        ]

    def get_news_dataframe(self, query, from_date, to_date):
        """Return news articles as a DataFrame with parsed dates."""
        df = pd.DataFrame(self.get_news_data(query, from_date, to_date))
        if not df.empty:
            df['publishedAt'] = pd.to_datetime(df['publishedAt'], utc=True)
            df['date']        = df['publishedAt'].dt.date
        return df


    def get_twitter_data(self, query, count=100):
        """
        Fetch recent tweets using Twitter API v2.

        Returns:
            List of tweet dicts: id, text, created_at, likes, retweets
        """
        client   = tweepy.Client(bearer_token=self.twitter_bearer_token)
        response = client.search_recent_tweets(
            query=query,
            max_results=min(count, 100),
            tweet_fields=['created_at', 'public_metrics']
        )
        if not response.data:
            return []
        return [
            {
                'id':         t.id,
                'text':       t.text,
                'created_at': t.created_at,
                'likes':      t.public_metrics.get('like_count', 0) if t.public_metrics else 0,
                'retweets':   t.public_metrics.get('retweet_count', 0) if t.public_metrics else 0
            }
            for t in response.data
        ]

    def get_reddit_data(self, subreddits=None, query="", limit=100, time_filter="month"):
        """
        Fetch posts from finance-related subreddits using PRAW.
        Reddit is more reliable than Twitter for financial sentiment research.

        Args:
            subreddits:  List of subreddit names. Defaults to major finance subs.
            query:       Search term e.g. 'AAPL earnings'
            limit:       Max posts per subreddit
            time_filter: 'day' | 'week' | 'month' | 'year' | 'all'

        Returns:
            List of post dicts: subreddit, title, text, score,
                                upvote_ratio, num_comments, created_at, url
        """
        if subreddits is None:
            subreddits = ['wallstreetbets', 'investing', 'stocks', 'SecurityAnalysis']

        reddit = praw.Reddit(
            client_id=self.reddit_client_id,
            client_secret=self.reddit_client_secret,
            user_agent="StockSentimentBot/1.0"
        )

        posts = []
        for sub_name in subreddits:
            subreddit = reddit.subreddit(sub_name)
            results   = (
                subreddit.search(query, time_filter=time_filter, limit=limit, sort='relevance')
                if query else subreddit.hot(limit=limit)
            )
            for post in results:
                posts.append({
                    'subreddit':    sub_name,
                    'title':        post.title,
                    'text':         post.selftext,
                    'score':        post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'created_at':   datetime.utcfromtimestamp(post.created_utc),
                    'url':          post.url
                })
        return posts

    def get_reddit_dataframe(self, subreddits=None, query="", limit=100):
        """Return Reddit posts as a DataFrame with parsed dates."""
        df = pd.DataFrame(self.get_reddit_data(subreddits=subreddits, query=query, limit=limit))
        if not df.empty:
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['date']       = df['created_at'].dt.date
        return df

    # ------------------------------------------------------------------ #
    #  DATA PERSISTENCE & CACHING                                        #
    # ------------------------------------------------------------------ #

    def save_to_csv(self, df, filename, folder="data/raw"):
        """Save DataFrame to CSV. Returns the saved file path."""
        Path(folder).mkdir(parents=True, exist_ok=True)
        path = f"{folder}/{filename}.csv"
        df.to_csv(path)
        return path

    def load_from_csv(self, filename, folder="data/raw"):
        """Load DataFrame from CSV with parsed date index."""
        return pd.read_csv(f"{folder}/{filename}.csv", index_col=0, parse_dates=True)

    def get_or_fetch_stock(self, ticker, start_date, end_date, use_cache=True):
        """
        Fetch stock data, using local cache if available.
        Avoids repeated API calls during development and training runs.
        """
        cache_name = f"{ticker}_{str(start_date)[:10]}_{str(end_date)[:10]}"
        cache_path = f"data/cache/{cache_name}.csv"

        if use_cache and Path(cache_path).exists():
            return self.load_from_csv(cache_name, folder="data/cache")

        data = self.get_stock_data_yahoo(ticker, start_date, end_date)
        if use_cache:
            self.save_to_csv(data, cache_name, folder="data/cache")
        return data



#  Example usage                                                       #

if __name__ == "__main__":
    collector  = DataCollector()
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=90)

    # Stock data (auto-cached)
    aapl = collector.get_or_fetch_stock('AAPL', start_date, end_date)
    print("AAPL (last 5 rows):")
    print(aapl.tail())

    # Multiple tickers
    stocks = collector.get_multiple_stocks(['AAPL', 'MSFT', 'GOOGL'], start_date, end_date)
    print(f"\nFetched {len(stocks)} tickers")

    # World Bank GDP (uncomment to use)
    # gdp = collector.get_economic_data_world_bank("USA", "NY.GDP.MKTP.CD", 2015, 2023)

    # News (requires API key in .env)
    # news_df = collector.get_news_dataframe("Apple stock", "2024-01-01", "2024-01-31")

    # Reddit (requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env)
    # reddit_df = collector.get_reddit_dataframe(query="AAPL", limit=50)

