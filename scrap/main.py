import os
import warnings

# 1. SILENCE TERMINAL NOISE (Must be at the very top)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import pandas_ta_classic as ta
import xgboost as xgb
import torch
import requests
from bs4 import BeautifulSoup
import random
from transformers import pipeline, logging as hf_logging

# Silence Hugging Face weight loading warnings
hf_logging.set_verbosity_error() 

def get_live_sentiment(ticker_symbol, sentiment_pipeline):
    print(f"[*] Scraping live news for {ticker_symbol} from Finviz...")
    url = f'https://finviz.com/quote.ashx?t={ticker_symbol}'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        news_table = soup.find(id='news-table')
        headlines = []
        
        if news_table:
            for row in news_table.findAll('tr')[:10]:
                a_tag = row.a
                if a_tag:
                    headlines.append(a_tag.text)
        
        if not headlines:
            print("[-] No recent news found. Defaulting to Neutral sentiment.")
            return 0.0

        print(f"[+] Successfully scraped {len(headlines)} headlines. Analyzing with FinBERT...")
        sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        total_score = 0
        
        for hl in headlines:
            result = sentiment_pipeline(hl)[0]
            total_score += sentiment_map[result['label']] * result['score']
            
        return total_score / len(headlines)
    except Exception as e:
        print(f"[-] Error scraping news: {e}")
        return 0.0

def main():
    print("="*50)
    print(" FINANCIAL MARKET ANALYSER ENGINE ")
    print("="*50)
    
    ticker = input("\nEnter a Stock Ticker (e.g., AAPL, MSFT, NVDA): ").strip().upper()
    print(f"\n[*] Initializing FinBERT NLP Model...")
    
    # Load FinBERT
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
    
    # 1. Get Live Sentiment
    live_sentiment_score = get_live_sentiment(ticker, sentiment_pipeline)
    print(f"[+] Current Sentiment Score: {live_sentiment_score:.4f}")
    
    # 2. Fetch Historical Data for Background Model
    print(f"\n[*] Fetching 2-year market data for {ticker} to train background model...")
    df = yf.download(ticker, start="2022-01-01", end="2024-01-01", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df[['Close', 'Volume']].copy()
    df.index = df.index.tz_localize(None).normalize()

    # 3. Feature Engineering
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df['Return'] = df['Close'].pct_change()
    
    # Synthetic historical sentiment for training weights
    df['FinBERT_Score'] = [random.uniform(-0.5, 0.5) for _ in range(len(df))]
    
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    # 4. Train XGBoost Model
    print("[*] Training XGBoost Prediction Model...")
    features = ['Return', 'Volume', 'RSI_14', 'MACDh_12_26_9', 'BBL_20_2.0', 'BBU_20_2.0', 'FinBERT_Score']
    
    # Safety Check for Training
    for f in features:
        if f not in df.columns:
            print(f"[-] FATAL ERROR: Feature {f} failed to calculate during training.")
            return

    model = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(df[features], df['Target'])
    
    # 5. Live Prediction
    print("[*] Fetching 1-year live market data for real-time inference...")
    live_df = yf.download(ticker, period="1y", progress=False) 
    
    if isinstance(live_df.columns, pd.MultiIndex):
         live_df.columns = live_df.columns.get_level_values(0)
    
    live_df = live_df[['Close', 'Volume']].copy()
    live_df.ta.rsi(length=14, append=True)
    live_df.ta.macd(fast=12, slow=26, signal=9, append=True)
    live_df.ta.bbands(length=20, std=2, append=True)
    live_df['Return'] = live_df['Close'].pct_change()
    
    live_df.dropna(inplace=True) 
    
    today_data = live_df.iloc[-1:].copy()
    today_data['FinBERT_Score'] = live_sentiment_score
    
    # Safety Check for Live Inference
    for f in features:
        if f not in today_data.columns:
            print(f"[-] FATAL ERROR: Live market data missing indicator: {f}. Cannot proceed.")
            return

    # Make Prediction
    prediction = model.predict(today_data[features])[0]
    probability = model.predict_proba(today_data[features])[0]

    # Output Results
    print("\n" + "="*50)
    print(f" REAL-TIME ANALYSIS RESULTS: {ticker} ")
    print("="*50)
    print(f"Live NLP Sentiment:    {live_sentiment_score:.4f} (-1 to 1)")
    print(f"Current RSI (14):      {today_data['RSI_14'].values[0]:.2f}")
    print(f"Current MACD Hist:     {today_data['MACDh_12_26_9'].values[0]:.4f}")
    print("-" * 50)
    
    if prediction == 1:
        print(f"🎯 ALGO PREDICTION: BULLISH (UP)")
        print(f"Confidence: {probability[1]*100:.2f}%")
    else:
        print(f"🎯 ALGO PREDICTION: BEARISH (DOWN)")
        print(f"Confidence: {probability[0]*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()