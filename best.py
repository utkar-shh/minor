import os
import warnings

# 1. SILENCE TERMINAL NOISE
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
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import pipeline, logging as hf_logging

hf_logging.set_verbosity_error() 

def get_live_sentiment(ticker_symbol, sentiment_pipeline):
    """Scrapes Finviz for live news and scores it using the loaded NLP model"""
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

        print(f"[+] Successfully scraped {len(headlines)} headlines. Analyzing with NLP Model...")
        sentiment_map = {
            "LABEL_0": -1.0, "negative": -1.0, 
            "LABEL_1": 0.0, "neutral": 0.0, 
            "LABEL_2": 1.0, "positive": 1.0
        }
        total_score = 0
        
        for hl in headlines:
            result = sentiment_pipeline(str(hl), truncation=True, max_length=512)[0]
            label = result['label']
            score = result['score']
            mapped_value = sentiment_map.get(label, 0.0) 
            total_score += mapped_value * score
            
        return total_score / len(headlines)
    except Exception as e:
        print(f"[-] Error scraping news: {e}")
        return 0.0

def fetch_market_data(ticker, period="1y"):
    """Fetches the target stock AND macro indicators for live inference."""
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = df.index.tz_localize(None).normalize()

    spy = yf.download("SPY", period=period, progress=False)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
    spy = spy[['Close']].rename(columns={'Close': 'SPY_Close'})
    spy.index = spy.index.tz_localize(None).normalize()

    vix = yf.download("^VIX", period=period, progress=False)
    if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)
    vix = vix[['Close']].rename(columns={'Close': 'VIX_Close'})
    vix.index = vix.index.tz_localize(None).normalize()

    df = df.join(spy).join(vix)

    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.obv(append=True)
    df.ta.atr(length=14, append=True)
    
    df['Return'] = df['Close'].pct_change()
    df['SPY_Return'] = df['SPY_Close'].pct_change()
    df['VIX_Change'] = df['VIX_Close'].diff()
    
    return df

def plot_dashboard(df, ticker, sentiment_score, prediction, confidence, predicted_price, current_price, projected_move):
    """Generates an upgraded 4-panel institutional web dashboard"""
    print("\n[*] Launching 4-Panel Interactive Web Dashboard...")
    plot_df = df.iloc[-120:].copy() # Zoomed out slightly to 120 days for better MACD context
    
    pred_text = "BULLISH (UP)" if prediction == 1 else "BEARISH (DOWN)"
    pred_color = "#00ff00" if prediction == 1 else "#ff0000"
    move_text = f"+{projected_move:.2f}%" if projected_move > 0 else f"{projected_move:.2f}%"
    
    title_text = (f"<b>{ticker} Real-Time Market Analysis</b><br>"
                  f"<span style='font-size: 14px; color: {pred_color};'>Target: ${predicted_price:.2f} ({move_text}) | "
                  f"Confidence: {confidence:.2f}% | Live NLP: {sentiment_score:.2f}</span>")

    # Upgraded to 4 rows for Volume integration
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.02, 
                        row_heights=[0.5, 0.15, 0.15, 0.2])

    # --- PANEL 1: Candlesticks & Bollinger Bands ---
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                                 low=plot_df['Low'], close=plot_df['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BBU_20_2.0'], line=dict(color='rgba(0, 191, 255, 0.5)', width=1, dash='dot'), name='Upper Margin'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BBL_20_2.0'], line=dict(color='rgba(0, 191, 255, 0.5)', width=1, dash='dot'), 
                             fill='tonexty', fillcolor='rgba(0, 191, 255, 0.08)', name='Lower Margin'), row=1, col=1)
    fig.add_hline(y=predicted_price, line_dash="dash", line_color=pred_color, annotation_text=f" AI Target: ${predicted_price:.2f} ", annotation_position="top left", row=1, col=1)

    # --- PANEL 2: Volume ---
    volume_colors = ['#00cc00' if row['Close'] >= row['Open'] else '#cc0000' for index, row in plot_df.iterrows()]
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], marker_color=volume_colors, name='Volume'), row=2, col=1)

    # --- PANEL 3: MACD ---
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACD_12_26_9'], line=dict(color='#00bfff', width=1.5), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACDs_12_26_9'], line=dict(color='#ff9900', width=1.5), name='Signal'), row=3, col=1)
    macd_colors = ['#00cc00' if val > 0 else '#cc0000' for val in plot_df['MACDh_12_26_9']]
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['MACDh_12_26_9'], marker_color=macd_colors, name='MACD Hist'), row=3, col=1)

    # --- PANEL 4: RSI ---
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI_14'], line=dict(color='#b000ff', width=1.5), name='RSI'), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=4, col=1)
    fig.update_yaxes(range=[0, 100], row=4, col=1)

    fig.update_layout(title=title_text, xaxis_rangeslider_visible=False, template='plotly_dark', height=950, hovermode='x unified', showlegend=False)                
    fig.show()

def main():
    print("="*50)
    print(" ADVANCED MACRO-ENABLED FINANCIAL ANALYSER ")
    print("="*50)
    
    ticker = input("\nEnter a Stock Ticker (e.g., AAPL, MSFT, NVDA): ").strip().upper()
    
    # 1. NLP MODEL INITIALIZATION
    custom_model_path = r"C:\Users\ROG\OneDrive\Desktop\minor\custom_finbert_model"
    if os.path.exists(custom_model_path):
        print(f"\n[*] Booting CUSTOM NLP Model from: {custom_model_path}")
        active_model = custom_model_path
    else:
        print(f"\n[-] Custom NLP model not found at {custom_model_path}")
        print("[*] Falling back to default internet model (ProsusAI/finbert)...")
        active_model = "ProsusAI/finbert"

    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline("sentiment-analysis", model=active_model, device=device)
    
    live_sentiment_score = get_live_sentiment(ticker, sentiment_pipeline)
    print(f"[+] Current NLP Sentiment Score: {live_sentiment_score:.4f}")
    
    # 2. INSTANT ML MODEL LOADING
    print(f"\n[*] Loading pre-trained ML models for {ticker}...")
    classifier_path = f"{ticker}_xgb_classifier.json"
    regressor_path = f"{ticker}_xgb_regressor.json"
    
    if not os.path.exists(classifier_path) or not os.path.exists(regressor_path):
        print(f"[-] ERROR: Could not find pre-trained ML models for {ticker}.")
        print(f"[-] Please run 'python train_ml.py' and enter {ticker} to build the models first!")
        return
        
    best_classifier = xgb.XGBClassifier()
    best_classifier.load_model(classifier_path)
    
    best_regressor = xgb.XGBRegressor()
    best_regressor.load_model(regressor_path)
    print("[+] Models loaded instantly!")

    # 3. LIVE MARKET DATA FETCHING
    print("[*] Fetching live market & macro data for real-time inference...")
    live_df = fetch_market_data(ticker, period="1y")
    live_df.dropna(inplace=True) 
    
    today_data = live_df.iloc[-1:].copy()
    today_data['FinBERT_Score'] = live_sentiment_score
    current_price = today_data['Close'].values[0]

    features = [
        'Return', 'Volume', 'RSI_14', 'MACDh_12_26_9', 'BBL_20_2.0', 'BBU_20_2.0', 
        'OBV', 'ATRr_14', 'SPY_Return', 'VIX_Change', 'FinBERT_Score'
    ]

    for f in features:
        if f not in today_data.columns:
            print(f"[-] FATAL ERROR: Live market data missing indicator: {f}. Cannot proceed.")
            return

    # 4. INSTANT PREDICTIONS
    prediction = best_classifier.predict(today_data[features])[0]
    probability = best_classifier.predict_proba(today_data[features])[0]
    confidence = probability[1]*100 if prediction == 1 else probability[0]*100
    
    predicted_return = best_regressor.predict(today_data[features])[0]
    predicted_price = current_price * (1 + predicted_return)
    projected_move = predicted_return * 100

    print("\n" + "="*50)
    print(f" REAL-TIME ANALYSIS RESULTS: {ticker} ")
    print("="*50)
    print(f"Current Price:         ${current_price:.2f}")
    print(f"Predicted Price (T+1): ${predicted_price:.2f} ({projected_move:+.2f}%)")
    print("-" * 50)
    print(f"Live NLP Sentiment:    {live_sentiment_score:.4f} (-1 to 1)")
    print(f"Current RSI (14):      {today_data['RSI_14'].values[0]:.2f}")
    print(f"S&P 500 Daily Move:    {today_data['SPY_Return'].values[0]*100:+.2f}%")
    print(f"VIX (Volatility) Move: {today_data['VIX_Change'].values[0]:+.2f}")
    print("-" * 50)
    
    if prediction == 1:
        print(f"🎯 ALGO DIRECTION: BULLISH (UP)")
        print(f"Confidence: {confidence:.2f}%")
    else:
        print(f"🎯 ALGO DIRECTION: BEARISH (DOWN)")
        print(f"Confidence: {confidence:.2f}%")
    print("="*50)

    plot_dashboard(live_df, ticker, live_sentiment_score, prediction, confidence, predicted_price, current_price, projected_move)

if __name__ == "__main__":
    main()