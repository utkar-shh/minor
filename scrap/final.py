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
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

hf_logging.set_verbosity_error() 

def get_live_sentiment(ticker_symbol, sentiment_pipeline):
    """Scrapes Finviz for live news and scores it using FinBERT"""
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

def load_historical_sentiment(df, ticker):
    """
    Attempts to load real historical sentiment data from a local CSV.
    If none exists, it defaults to a neutral baseline for training.
    """
    filename = f"{ticker}_sentiment.csv"
    if os.path.exists(filename):
        print(f"[+] Found real historical sentiment dataset: {filename}. Merging...")
        try:
            sentiment_df = pd.read_csv(filename, parse_dates=['Date'])
            sentiment_df.set_index('Date', inplace=True)
            sentiment_df.index = sentiment_df.index.tz_localize(None).normalize()
            df = df.merge(sentiment_df[['FinBERT_Score']], left_index=True, right_index=True, how='left')
            df['FinBERT_Score'].fillna(0.0, inplace=True)
            return df
        except Exception as e:
            print(f"[-] Error loading {filename}: {e}. Falling back to baseline.")
            
    print(f"[-] No real historical news found ({filename} missing). Using neutral training baseline.")
    df['FinBERT_Score'] = np.random.uniform(-0.05, 0.05, size=len(df))
    return df

def optimize_model(model_type, X_train, y_train):
    """Runs Time-Series Cross Validation to find the best hyperparameters"""
    param_distributions = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    tscv = TimeSeriesSplit(n_splits=3)
    
    if model_type == 'classifier':
        base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    else:
        base_model = xgb.XGBRegressor(random_state=42, eval_metric='rmse')

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=10, 
        cv=tscv, 
        random_state=42, 
        n_jobs=-1 
    )
    
    search.fit(X_train, y_train)
    return search.best_estimator_

def plot_dashboard(df, ticker, sentiment_score, prediction, confidence, predicted_price, current_price, projected_move):
    """Generates an interactive, 3-panel professional web dashboard"""
    print("\n[*] Launching Interactive Web Dashboard...")
    
    plot_df = df.iloc[-90:].copy()
    
    pred_text = "BULLISH (UP)" if prediction == 1 else "BEARISH (DOWN)"
    pred_color = "#00ff00" if prediction == 1 else "#ff0000"
    move_text = f"+{projected_move:.2f}%" if projected_move > 0 else f"{projected_move:.2f}%"
    
    title_text = (f"<b>{ticker} Real-Time Market Analysis</b><br>"
                  f"<span style='font-size: 14px; color: {pred_color};'>Target: ${predicted_price:.2f} ({move_text}) | "
                  f"Confidence: {confidence:.2f}% | Live NLP: {sentiment_score:.2f}</span>")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.6, 0.2, 0.2])

    # --- PANEL 1: Candlesticks & Bollinger Bands ---
    fig.add_trace(go.Candlestick(x=plot_df.index,
                                 open=plot_df['Open'],
                                 high=plot_df['High'],
                                 low=plot_df['Low'],
                                 close=plot_df['Close'],
                                 name='Price'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BBU_20_2.0'], 
                             line=dict(color='rgba(0, 191, 255, 0.5)', width=1, dash='dot'), 
                             name='Upper Margin'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BBL_20_2.0'], 
                             line=dict(color='rgba(0, 191, 255, 0.5)', width=1, dash='dot'), 
                             fill='tonexty', fillcolor='rgba(0, 191, 255, 0.08)', 
                             name='Lower Margin'), row=1, col=1)

    fig.add_hline(y=predicted_price, line_dash="dash", line_color=pred_color, 
                  annotation_text=f" AI Target: ${predicted_price:.2f} ", 
                  annotation_position="top left", row=1, col=1)

    # --- PANEL 2: MACD ---
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACD_12_26_9'], line=dict(color='#00bfff', width=1.5), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACDs_12_26_9'], line=dict(color='#ff9900', width=1.5), name='Signal'), row=2, col=1)
    
    macd_colors = ['#00cc00' if val > 0 else '#cc0000' for val in plot_df['MACDh_12_26_9']]
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['MACDh_12_26_9'], marker_color=macd_colors, name='Histogram'), row=2, col=1)

    # --- PANEL 3: RSI ---
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI_14'], line=dict(color='#b000ff', width=1.5), name='RSI'), row=3, col=1)
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    # --- Global Layout Updates ---
    fig.update_layout(title=title_text, 
                      xaxis_rangeslider_visible=False, 
                      template='plotly_dark',          
                      height=850,
                      hovermode='x unified',           
                      showlegend=False)                

    fig.show()

def main():
    print("="*50)
    print(" ADVANCED FINANCIAL MARKET ANALYSER ENGINE ")
    print("="*50)
    
    ticker = input("\nEnter a Stock Ticker (e.g., AAPL, MSFT, NVDA): ").strip().upper()
    print(f"\n[*] Initializing FinBERT NLP Model...")
    
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline("sentiment-analysis", model_path = "C:/Users/ROG/OneDrive/Desktop/New folder/custom_finbert_model", device=device)
    
    live_sentiment_score = get_live_sentiment(ticker, sentiment_pipeline)
    print(f"[+] Current NLP Sentiment Score: {live_sentiment_score:.4f}")
    
    print(f"\n[*] Fetching 4-year market data for {ticker} to train background models...")
    df = yf.download(ticker, period="4y", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Crucial: Must pull Open, High, Low for Candlestick charts
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = df.index.tz_localize(None).normalize()

    # Feature Engineering
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df['Return'] = df['Close'].pct_change()
    
    # Load Real Historical Sentiment (or fallback to baseline)
    df = load_historical_sentiment(df, ticker)
    
    # Target Variables (Targeting Returns, not exact Prices)
    df['Target_Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df['Target_Return'] = df['Close'].pct_change().shift(-1) 
    
    df.dropna(inplace=True)

    features = ['Return', 'Volume', 'RSI_14', 'MACDh_12_26_9', 'BBL_20_2.0', 'BBU_20_2.0', 'FinBERT_Score']
    for f in features:
        if f not in df.columns:
            print(f"[-] FATAL ERROR: Feature {f} failed to calculate.")
            return

    # Hyperparameter Tuning & Training
    print("[*] Running Hyperparameter Tuning via Time-Series Split (This may take a minute)...")
    
    print("    -> Optimizing Classifier (Direction)...")
    best_classifier = optimize_model('classifier', df[features], df['Target_Direction'])
    
    print("    -> Optimizing Regressor (Target Return)...")
    best_regressor = optimize_model('regressor', df[features], df['Target_Return'])
    
    print("[*] Models successfully optimized and trained.")
    
    # Live Inference
    print("[*] Fetching live market data for real-time inference...")
    live_df = yf.download(ticker, period="1y", progress=False) 
    
    if isinstance(live_df.columns, pd.MultiIndex):
         live_df.columns = live_df.columns.get_level_values(0)
    
    # Crucial: Must pull Open, High, Low for Candlestick charts
    live_df = live_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    live_df.ta.rsi(length=14, append=True)
    live_df.ta.macd(fast=12, slow=26, signal=9, append=True)
    live_df.ta.bbands(length=20, std=2, append=True)
    live_df['Return'] = live_df['Close'].pct_change()
    live_df.dropna(inplace=True) 
    
    today_data = live_df.iloc[-1:].copy()
    today_data['FinBERT_Score'] = live_sentiment_score
    current_price = today_data['Close'].values[0]

    for f in features:
        if f not in today_data.columns:
            print(f"[-] FATAL ERROR: Live market data missing indicator: {f}. Cannot proceed.")
            return

    # Predictions
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
    print(f"Current MACD Hist:     {today_data['MACDh_12_26_9'].values[0]:.4f}")
    print("-" * 50)
    
    if prediction == 1:
        print(f"🎯 ALGO DIRECTION: BULLISH (UP)")
        print(f"Confidence (Margin of Error: {100-confidence:.2f}%): {confidence:.2f}%")
    else:
        print(f"🎯 ALGO DIRECTION: BEARISH (DOWN)")
        print(f"Confidence (Margin of Error: {100-confidence:.2f}%): {confidence:.2f}%")
    print("="*50)

    plot_dashboard(live_df, ticker, live_sentiment_score, prediction, confidence, predicted_price, current_price, projected_move)

if __name__ == "__main__":
    main()