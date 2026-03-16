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
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
    
    df['Return'] = df['Close'].pct_change()
    df['SPY_Return'] = df['SPY_Close'].pct_change()
    df['VIX_Change'] = df['VIX_Close'].diff()
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['Return_Lag1'] = df['Return'].shift(1)
    df['Return_Lag2'] = df['Return'].shift(2)
    df['Return_Lag5'] = df['Return'].shift(5)

    return df

def plot_dashboard(df, ticker, sentiment_score, predictions, current_price):
    """Generates an upgraded 4-panel institutional web dashboard with multi-horizon targets"""
    print("\n[*] Launching 4-Panel Interactive Web Dashboard...")
    plot_df = df.iloc[-120:].copy()

    # Extract per-horizon data
    d = predictions['daily']
    w = predictions['weekly']
    m = predictions['monthly']

    daily_color = "#00ff00" if d['prediction'] == 1 else "#ff0000"
    weekly_color = "#00ccff" if w['prediction'] == 1 else "#ff6600"
    monthly_color = "#ffff00" if m['prediction'] == 1 else "#ff00ff"
    sent_color = "#00ff00" if sentiment_score > 0.05 else ("#ff0000" if sentiment_score < -0.05 else "#888888")

    title_text = (
        f"<b>{ticker} Real-Time Market Analysis</b><br>"
        f"<span style='font-size: 13px;'>"
        f"<span style='color:{daily_color}'>Daily: ${d['predicted_price']:.2f} ({d['projected_move']:+.2f}%)</span> | "
        f"<span style='color:{weekly_color}'>Weekly: ${w['predicted_price']:.2f} ({w['projected_move']:+.2f}%)</span> | "
        f"<span style='color:{monthly_color}'>Monthly: ${m['predicted_price']:.2f} ({m['projected_move']:+.2f}%)</span>"
        f"</span>"
    )

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02,
                        row_heights=[0.5, 0.15, 0.15, 0.2])

    # --- PANEL 1: Candlesticks, Bollinger Bands & Moving Averages ---
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                                 low=plot_df['Low'], close=plot_df['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BBU_20_2.0'], line=dict(color='rgba(0, 191, 255, 0.5)', width=1, dash='dot'), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BBL_20_2.0'], line=dict(color='rgba(0, 191, 255, 0.5)', width=1, dash='dot'),
                             fill='tonexty', fillcolor='rgba(0, 191, 255, 0.08)', name='BB Lower'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BBM_20_2.0'], line=dict(color='rgba(0, 191, 255, 0.9)', width=1), name='BB Mid (SMA 20)'), row=1, col=1)
    if 'SMA_50' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_50'], line=dict(color='#ffaa00', width=1.2, dash='dash'), name='SMA 50'), row=1, col=1)
    if 'SMA_200' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_200'], line=dict(color='#ff5555', width=1.2, dash='dash'), name='SMA 200'), row=1, col=1)

    # Multi-horizon target lines
    fig.add_hline(y=current_price, line_dash="dot", line_color="white",
                  annotation_text=f" Current: ${current_price:.2f} ", annotation_position="bottom left", row=1, col=1)
    fig.add_hline(y=d['predicted_price'], line_dash="dash", line_color=daily_color,
                  annotation_text=f" Daily: ${d['predicted_price']:.2f} ", annotation_position="top left", row=1, col=1)
    fig.add_hline(y=w['predicted_price'], line_dash="dashdot", line_color=weekly_color,
                  annotation_text=f" Weekly: ${w['predicted_price']:.2f} ", annotation_position="top right", row=1, col=1)
    fig.add_hline(y=m['predicted_price'], line_dash="longdash", line_color=monthly_color,
                  annotation_text=f" Monthly: ${m['predicted_price']:.2f} ", annotation_position="bottom right", row=1, col=1)

    # Sentiment annotation box
    sent_label = "Bullish" if sentiment_score > 0.05 else ("Bearish" if sentiment_score < -0.05 else "Neutral")
    fig.add_annotation(
        text=f"NLP: {sent_label} ({sentiment_score:+.2f})",
        xref="paper", yref="paper", x=0.99, y=0.98,
        showarrow=False, font=dict(size=12, color=sent_color),
        bgcolor="rgba(0,0,0,0.6)", bordercolor=sent_color, borderwidth=1,
        borderpad=4, xanchor="right", yanchor="top"
    )

    # --- PANEL 2: Volume with SMA ---
    volume_colors = np.where(plot_df['Close'].values >= plot_df['Open'].values, '#00cc00', '#cc0000')
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], marker_color=volume_colors, name='Volume'), row=2, col=1)
    if 'Vol_SMA20' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Vol_SMA20'], line=dict(color='#ffaa00', width=1.2), name='Vol SMA 20'), row=2, col=1)

    # --- PANEL 3: MACD ---
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACD_12_26_9'], line=dict(color='#00bfff', width=1.5), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACDs_12_26_9'], line=dict(color='#ff9900', width=1.5), name='Signal'), row=3, col=1)
    macd_colors = np.where(plot_df['MACDh_12_26_9'].values > 0, '#00cc00', '#cc0000')
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['MACDh_12_26_9'], marker_color=macd_colors, name='MACD Hist'), row=3, col=1)

    # --- PANEL 4: RSI ---
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI_14'], line=dict(color='#b000ff', width=1.5), name='RSI'), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=4, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.4, row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=4, col=1)
    fig.update_yaxes(range=[0, 100], row=4, col=1)

    # --- Y-axis labels ---
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1)

    fig.update_layout(title=title_text, xaxis_rangeslider_visible=False, template='plotly_dark', height=950, hovermode='x unified', showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)))
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

    # 2. LOAD ML MODELS FOR ALL HORIZONS
    horizons = {'daily': 'T+1', 'weekly': 'T+5', 'monthly': 'T+21'}
    models = {}

    print(f"\n[*] Loading pre-trained ML models for {ticker}...")
    for horizon in horizons:
        cls_path = f"{ticker}_xgb_classifier_{horizon}.json"
        reg_path = f"{ticker}_xgb_regressor_{horizon}.json"
        if not os.path.exists(cls_path) or not os.path.exists(reg_path):
            print(f"[-] ERROR: Could not find {horizon} ML models for {ticker}.")
            print(f"[-] Please run 'python trainml.py' and enter {ticker} to build the models first!")
            return
        cls = xgb.XGBClassifier()
        cls.load_model(cls_path)
        reg = xgb.XGBRegressor()
        reg.load_model(reg_path)
        models[horizon] = {'classifier': cls, 'regressor': reg}
    print("[+] All models (daily/weekly/monthly) loaded instantly!")

    # 3. LIVE MARKET DATA FETCHING
    print("[*] Fetching live market & macro data for real-time inference...")
    live_df = fetch_market_data(ticker, period="1y")
    live_df.dropna(inplace=True)

    today_data = live_df.iloc[-1:].copy()
    today_data['FinBERT_Score'] = live_sentiment_score
    current_price = today_data['Close'].values[0]

    features = [
        'Return', 'Volume', 'RSI_14', 'MACDh_12_26_9', 'BBL_20_2.0', 'BBU_20_2.0',
        'OBV', 'ATRr_14', 'SPY_Return', 'VIX_Change', 'FinBERT_Score',
        'Volume_Ratio', 'Return_Lag1', 'Return_Lag2', 'Return_Lag5'
    ]

    for f in features:
        if f not in today_data.columns:
            print(f"[-] FATAL ERROR: Live market data missing indicator: {f}. Cannot proceed.")
            return

    # 4. MULTI-HORIZON PREDICTIONS
    predictions = {}
    for horizon, label in horizons.items():
        cls = models[horizon]['classifier']
        reg = models[horizon]['regressor']
        pred = cls.predict(today_data[features])[0]
        prob = cls.predict_proba(today_data[features])[0]
        conf = prob[1] * 100 if pred == 1 else prob[0] * 100
        pred_ret = reg.predict(today_data[features])[0]
        pred_price = current_price * (1 + pred_ret)
        proj_move = pred_ret * 100
        predictions[horizon] = {
            'prediction': pred, 'confidence': conf,
            'predicted_price': pred_price, 'projected_move': proj_move
        }

    print("\n" + "="*60)
    print(f" REAL-TIME ANALYSIS RESULTS: {ticker} ")
    print("="*60)
    print(f"Current Price:  ${current_price:.2f}")
    print("-" * 60)
    print(f"{'Horizon':<12} {'Direction':<15} {'Target Price':<16} {'Move':<10} {'Score'}")
    print("-" * 60)
    for horizon, label in horizons.items():
        p = predictions[horizon]
        direction = "BULLISH (UP)" if p['prediction'] == 1 else "BEARISH (DOWN)"
        print(f"{label:<12} {direction:<15} ${p['predicted_price']:<14.2f} {p['projected_move']:+.2f}%     {p['confidence']:.1f}%")
    print("-" * 60)
    print(f"Live NLP Sentiment:    {live_sentiment_score:.4f} (-1 to 1)")
    print(f"Current RSI (14):      {today_data['RSI_14'].values[0]:.2f}")
    print(f"S&P 500 Daily Move:    {today_data['SPY_Return'].values[0]*100:+.2f}%")
    print(f"VIX (Volatility) Move: {today_data['VIX_Change'].values[0]:+.2f}")

    # --- FEATURE IMPORTANCE (from daily classifier) ---
    importance = models['daily']['classifier'].get_booster().get_score(importance_type='gain')
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("-" * 60)
    print("Top Feature Drivers (by gain):")
    for feat, gain in sorted_imp[:5]:
        print(f"  {feat:20s} : {gain:.2f}")
    print("="*60)

    plot_dashboard(live_df, ticker, live_sentiment_score, predictions, current_price)

if __name__ == "__main__":
    main()