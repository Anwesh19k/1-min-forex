import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

API_KEYS = ['your_api_key1', 'your_api_key2', 'your_api_key3']
SYMBOLS = ['EUR/USD', 'USD/JPY', 'GBP/USD']
api_index = 0

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
    df['return'] = df['close'].shift(-2) / df['close'] - 1

    # 3-class target with balanced thresholds
    df['target'] = np.select(
        [df['return'] > 0.00015, df['return'] < -0.00015],
        [1, -1],
        default=0
    )

    df['future_label'] = df['target'].shift(-2)
    df = df.dropna()
    df = df[df['future_label'].isin([-1, 0, 1])]
    return df

def balance_classes(df):
    class_counts = df['future_label'].value_counts()
    if class_counts.min() < 5:
        return None
    min_count = class_counts.min()
    return pd.concat([
        df[df['future_label'] == -1].sample(min_count, replace=True),
        df[df['future_label'] == 0].sample(min_count, replace=True),
        df[df['future_label'] == 1].sample(min_count, replace=True)
    ]).sample(frac=1).reset_index(drop=True)

def train_model(df):
    features = ['ma5', 'ema10', 'rsi14', 'momentum']
    df_bal = balance_classes(df)
    if df_bal is None:
        return None, None
    X = df_bal[features]
    y = df_bal['future_label']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBClassifier(objective='multi:softprob', num_class=3,
                          n_estimators=80, max_depth=4, learning_rate=0.05, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_scaled, y)
    return model, scaler

def predict(df, model, scaler):
    features = ['ma5', 'ema10', 'rsi14', 'momentum']
    if model is None or scaler is None or df.shape[0] < 5:
        return None
    X_pred = df[features].iloc[[-2]]
    X_scaled = scaler.transform(X_pred)
    probs = model.predict_proba(X_scaled)[0]
    predicted_class = np.argmax(probs)
    class_map = {0: "HOLD ❌", 1: "BUY 📈", 2: "SELL 🔻"}
    true_class = {-1: 2, 0: 0, 1: 1}.get(df.iloc[-2]['future_label'], 0)
    correct = "✅" if predicted_class == true_class else "❌"
    return {
        "Signal": class_map[predicted_class],
        "Confidence": round(probs[predicted_class], 2),
        "Correct": correct
    }

def run_signal_engine():
    results = []
    for symbol in SYMBOLS:
        df = fetch_data(symbol)
        if df.empty or len(df) < 100: continue
        df = add_features(df)
        if df['future_label'].nunique() < 2: continue
        model, scaler = train_model(df)
        result = predict(df, model, scaler)
        if result:
            result['Symbol'] = symbol
            results.append(result)
    return pd.DataFrame(results)