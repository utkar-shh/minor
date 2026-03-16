import os
import warnings
import yfinance as yf
import pandas as pd
import pandas_ta_classic as ta
import xgboost as xgb
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error

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
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['Return_Lag1'] = df['Return'].shift(1)
    df['Return_Lag2'] = df['Return'].shift(2)
    df['Return_Lag5'] = df['Return'].shift(5)

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
    df['FinBERT_Score'] = 0.0
    return df

def optimize_model(model_type, X_train, y_train):
    param_distributions = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 500],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.3],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    
    if model_type == 'classifier':
        base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    else:
        base_model = xgb.XGBRegressor(random_state=42, eval_metric='rmse')

    search = RandomizedSearchCV(estimator=base_model, param_distributions=param_distributions,
                                n_iter=30, cv=tscv, random_state=42, n_jobs=-1)
    
    search.fit(X_train, y_train)
    return search.best_estimator_

def train_horizon(ticker, df, features, horizon_name, horizon_days, train_df, test_df):
    """Trains classifier + regressor for a single time horizon and evaluates on holdout."""
    dir_col = f'Target_Direction_{horizon_name}'
    ret_col = f'Target_Return_{horizon_name}'

    print(f"\n    --- {horizon_name.upper()} (T+{horizon_days}) ---")
    print(f"    -> Optimizing {horizon_name} Classifier...")
    best_cls = optimize_model('classifier', train_df[features], train_df[dir_col])

    print(f"    -> Optimizing {horizon_name} Regressor...")
    best_reg = optimize_model('regressor', train_df[features], train_df[ret_col])

    # Holdout evaluation
    cls_preds = best_cls.predict(test_df[features])
    cls_proba = best_cls.predict_proba(test_df[features])[:, 1]
    acc = accuracy_score(test_df[dir_col], cls_preds)
    f1 = f1_score(test_df[dir_col], cls_preds)
    auc = roc_auc_score(test_df[dir_col], cls_proba)
    print(f"    Classifier  ->  Accuracy: {acc:.4f}  |  F1: {f1:.4f}  |  AUC: {auc:.4f}")

    reg_preds = best_reg.predict(test_df[features])
    rmse = np.sqrt(mean_squared_error(test_df[ret_col], reg_preds))
    mae = mean_absolute_error(test_df[ret_col], reg_preds)
    print(f"    Regressor   ->  RMSE: {rmse:.6f}  |  MAE: {mae:.6f}")

    # Retrain on full data for production
    print(f"    -> Retraining {horizon_name} models on full dataset...")
    best_cls = optimize_model('classifier', df[features], df[dir_col])
    best_reg = optimize_model('regressor', df[features], df[ret_col])

    # Save
    cls_path = f"{ticker}_xgb_classifier_{horizon_name}.json"
    reg_path = f"{ticker}_xgb_regressor_{horizon_name}.json"
    best_cls.save_model(cls_path)
    best_reg.save_model(reg_path)
    print(f"    [+] Saved {cls_path}")
    print(f"    [+] Saved {reg_path}")

def main():
    print("="*50)
    print(" XGBOOST ML TRAINING ENGINE ")
    print("="*50)

    ticker = input("Enter the Stock Ticker to train ML models for (e.g., AAPL): ").strip().upper()

    print(f"\n[*] Fetching 10-year market & macro data...")
    df = fetch_market_data(ticker, period="10y")
    df = load_historical_sentiment(df, ticker)

    # --- MULTI-HORIZON TARGETS ---
    horizons = {'daily': 1, 'weekly': 5, 'monthly': 21}
    for name, days in horizons.items():
        df[f'Target_Direction_{name}'] = (df['Close'].shift(-days) > df['Close']).astype(int)
        df[f'Target_Return_{name}'] = (df['Close'].shift(-days) / df['Close'] - 1)

    df.dropna(inplace=True)

    features = [
        'Return', 'Volume', 'RSI_14', 'MACDh_12_26_9', 'BBL_20_2.0', 'BBU_20_2.0',
        'OBV', 'ATRr_14', 'SPY_Return', 'VIX_Change', 'FinBERT_Score',
        'Volume_Ratio', 'Return_Lag1', 'Return_Lag2', 'Return_Lag5'
    ]

    for f in features:
        if f not in df.columns:
            print(f"[-] FATAL ERROR: Feature {f} failed to calculate.")
            return

    # --- CHRONOLOGICAL TRAIN / TEST SPLIT (80/20) ---
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print(f"\n[*] Train/Test split: {len(train_df)} train rows, {len(test_df)} test rows")

    print("\n[*] Running Hyperparameter Tuning (This takes a moment)...")
    print("="*50)
    print(" HOLDOUT TEST SET EVALUATION")
    print("="*50)

    for name, days in horizons.items():
        train_horizon(ticker, df, features, name, days, train_df, test_df)

    print("\n" + "="*50)
    print(f"[+] SUCCESS: All 6 models saved for {ticker} (daily / weekly / monthly)")
    print("[+] Your ML models are now fully trained and ready for instant live inference!")

if __name__ == "__main__":
    main()