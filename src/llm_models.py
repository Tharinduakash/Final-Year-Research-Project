"""
LLM integration for sentiment analysis and direct forecasting.
"""

import openai
from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()

class LLMModels:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        # Load FinBERT for financial sentiment
        self.finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    def analyze_sentiment_finbert(self, text):
        """Analyze sentiment using FinBERT."""
        result = self.finbert(text)
        return result[0]['label'], result[0]['score'] 

    def analyze_sentiment_gpt(self, text):
        """Analyze sentiment using GPT."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found")

        prompt = f"Analyze the sentiment of the following financial text and respond with only 'positive', 'negative', or 'neutral':\n\n{text}"

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        sentiment = response.choices[0].message.content.strip().lower()
        return sentiment

    def forecast_with_llm(self, historical_data, news_summary):
        """Use LLM for direct stock price forecasting."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found")

        # Prepare prompt with historical data and news
        prompt = f"""
        Based on the following historical stock data and recent news, predict the next day's closing price.

        Historical data (last 5 days):
        {historical_data}

        Recent news summary:
        {news_summary}

        Provide only the predicted price as a number.
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3
        )
        prediction = response.choices[0].message.content.strip()
        return float(prediction)

# Example usage
if __name__ == "__main__":
    llm = LLMModels()
    text = "Apple's stock surged after strong earnings report."
    sentiment, score = llm.analyze_sentiment_finbert(text)
    print(f"Sentiment: {sentiment}, Score: {score}")