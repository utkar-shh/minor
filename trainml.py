import os
import warnings
import yfinance as yf
import pandas as pd
import pandas_ta_classic as ta
import xgboost as xgb
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

warnings.filterwarnings('ignore')

def fetch_market_data(ticker, period="10y"):
    """Fetches the target stock AND macro indicators."""
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
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

def load_historical_sentiment(df, ticker):
    filename = f"{ticker}_sentiment.csv"
    if os.path.exists(filename):
        print(f"[+] Found real historical sentiment dataset: {filename}.")
        sentiment_df = pd.read_csv(filename, parse_dates=['Date'])
        sentiment_df.set_index('Date', inplace=True)
        sentiment_df.index = sentiment_df.index.tz_localize(None).normalize()
        df = df.merge(sentiment_df[['FinBERT_Score']], left_index=True, right_index=True, how='left')
        df['FinBERT_Score'].fillna(0.0, inplace=True)
        return df
            
    print(f"[-] No real historical news found ({filename} missing). Using neutral training baseline.")
    df['FinBERT_Score'] = np.random.uniform(-0.05, 0.05, size=len(df))
    return df

def optimize_model(model_type, X_train, y_train):
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

    search = RandomizedSearchCV(estimator=base_model, param_distributions=param_distributions,
                                n_iter=10, cv=tscv, random_state=42, n_jobs=-1)
    
    search.fit(X_train, y_train)
    return search.best_estimator_

def main():
    print("="*50)
    print(" XGBOOST ML TRAINING ENGINE ")
    print("="*50)
    
    ticker = input("Enter the Stock Ticker to train ML models for (e.g., AAPL): ").strip().upper()
    
    print(f"\n[*] Fetching 10-year market & macro data...")
    df = fetch_market_data(ticker, period="10y")
    df = load_historical_sentiment(df, ticker)
    
    df['Target_Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df['Target_Return'] = df['Close'].pct_change().shift(-1) 
    df.dropna(inplace=True)

    features = [
        'Return', 'Volume', 'RSI_14', 'MACDh_12_26_9', 'BBL_20_2.0', 'BBU_20_2.0', 
        'OBV', 'ATRr_14', 'SPY_Return', 'VIX_Change', 'FinBERT_Score'
    ]
    
    for f in features:
        if f not in df.columns:
            print(f"[-] FATAL ERROR: Feature {f} failed to calculate.")
            return

    print("\n[*] Running Hyperparameter Tuning (This takes a moment)...")
    print("    -> Optimizing Classifier (Direction)...")
    best_classifier = optimize_model('classifier', df[features], df['Target_Direction'])
    
    print("    -> Optimizing Regressor (Target Return)...")
    best_regressor = optimize_model('regressor', df[features], df['Target_Return'])
    
    # --- SAVE THE MODELS LOCALLY ---
    print(f"\n[*] Saving optimized ML models locally...")
    best_classifier.save_model(f"{ticker}_xgb_classifier.json")
    best_regressor.save_model(f"{ticker}_xgb_regressor.json")
    
    print(f"[+] SUCCESS: Saved {ticker}_xgb_classifier.json")
    print(f"[+] SUCCESS: Saved {ticker}_xgb_regressor.json")
    print("[+] Your ML models are now fully trained and ready for instant live inference!")

if __name__ == "__main__":
    main()