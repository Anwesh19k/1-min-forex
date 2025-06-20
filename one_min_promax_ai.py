import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from stable_baselines3 import DQN
from rl_candle_env import CandleTradeEnv  # Custom gym environment

# === API Keys and Symbols ===
API_KEYS = [
    '54a7479bdf2040d3a35d6b3ae6457f9d',
    'd162b35754ca4c54a13ebe7abecab4e0',
    'a7266b2503fd497496d47527a7e63b5d',
    '54a7479bdf2040d3a35d6b3ae6457f9d',
    '09c09d58ed5e4cf4afd9a9cac8e09b5d',
    'df00920c02c54a59a426948a47095543'
]

SYMBOLS = ['EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF', 'AUD/USD',
           'USD/CAD', 'NZD/USD', 'EUR/GBP', 'XAU/USD', "BTC/USD"]
api_index = 0
dqn_model = DQN.load("dqn_candle_model")  # Load once

def get_next_api_key():
    global api_index
    key = API_KEYS[api_index % len(API_KEYS)]
    api_index += 1
    return key

def fetch_data(symbol):
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&outputsize=400&apikey={get_next_api_key()}"
        data = requests.get(url).json()
        if "values" not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data["values"])
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float})
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df.sort_values('datetime')
    except:
        return pd.DataFrame()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    return 100 - (100 / (1 + rs))

def add_features(df):
    df['ma5'] = df['close'].rolling(5).mean()
    df['ema10'] = df['close'].ewm(span=10).mean()
    df['rsi14'] = compute_rsi(df['close'])
    df['momentum'] = df['close'] - df['close'].shift(4)
    df['return'] = df['close'].pct_change().shift(-1)
    df['target'] = np.select(
        [df['return'] > 0.00005, df['return'] < -0.00005],
        [1, -1],
        default=0
    )
    df['future_label'] = df['target'].shift(-2)
    df = df.dropna()
    df = df[df['future_label'].isin([-1, 0, 1])]
    return df

def fallback_signal(df):
    rsi = df['rsi14'].iloc[-2] if 'rsi14' in df and len(df) >= 2 else 50
    if rsi > 55:
        return {"Signal": "BUY 📈", "Confidence": 0.52, "Correct": "⚠️"}
    elif rsi < 45:
        return {"Signal": "SELL 🔻", "Confidence": 0.51, "Correct": "⚠️"}
    else:
        return {"Signal": "HOLD ❌", "Confidence": 0.5, "Correct": "⚠️"}

def run_signal_engine():
    results = []
    for symbol in SYMBOLS:
        df = fetch_data(symbol)
        if df.empty or len(df) < 60:
            results.append({"Symbol": symbol, **fallback_signal(df)})
            continue
        df = add_features(df)
        if df.empty:
            results.append({"Symbol": symbol, **fallback_signal(df)})
            continue

        # === RL-based Signal Prediction ===
        try:
            env = CandleTradeEnv(df)
            obs, _ = env.reset()
            done = False
            last_action = 0
            while not done:
                action, _ = dqn_model.predict(obs)
                obs, _, done, _, _ = env.step(int(action))
                last_action = int(action)

            signal_map = {0: "HOLD ❌", 1: "BUY 📈", 2: "SELL 🔻"}
            results.append({
                "Symbol": symbol,
                "Signal": signal_map[last_action],
                "Confidence": "RL",
                "Correct": "✅"
            })
        except:
            results.append({"Symbol": symbol, **fallback_signal(df)})
    return pd.DataFrame(results)

