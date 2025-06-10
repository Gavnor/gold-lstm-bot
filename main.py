import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import asyncio
import websockets
import json
import socket
from scipy.stats import pearsonr
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator, EMAIndicator

# ======================
# CONFIGURATION
# ======================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")

MODEL_PATH = 'model/enhanced_gold_lstm.h5'
LOG_FILE = 'data/trade_log.csv'
MAX_RETRIES = 3
RETRY_DELAY = 5
WEBSOCKET_TIMEOUT = 10

# Trading Parameters
TRADE_INTERVAL_SECONDS = 7200
STOP_LOSS_PCT = 0.03
TAKE_PROFIT_PCT = 0.06
MAX_STAKE_PERCENT = 0.2
MIN_BALANCE = 10
MAX_STAKE = 10000
VOLATILITY_THRESHOLD = 1.5
CORRELATION_THRESHOLD = 0.7

# ======================
# INITIALIZATION
# ======================
socket.getaddrinfo = lambda *args: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (args[0], args[1]))]

# ======================
# TELEGRAM
# ======================
def send_telegram_message(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': msg}, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")

# ======================
# FETCHING
# ======================
def fetch_data(interval='1h', size=100):
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval={interval}&outputsize={size}&apikey={TWELVE_API_KEY}"
    try:
        data = requests.get(url, timeout=10).json()
        if 'values' in data:
            df = pd.DataFrame(data['values'])[::-1]
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['price'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            return df
    except Exception as e:
        send_telegram_message(f"❌ Fetch error: {e}")
    return None

def fetch_correlation(symbol='DXY'):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1h&outputsize=100&apikey={TWELVE_API_KEY}"
    try:
        data = requests.get(url, timeout=10).json()
        if 'values' in data:
            df = pd.DataFrame(data['values'])[::-1]
            return df['close'].astype(float).values
    except Exception:
        return None
    return None

# ======================
# FEATURES
# ======================
def add_indicators(df):
    df['rsi'] = RSIIndicator(df['price']).rsi()
    macd = MACD(df['price'])
    df['macd'] = macd.macd()
    df['adx'] = ADXIndicator(df['high'], df['low'], df['price']).adx()
    df['ema20'] = EMAIndicator(df['price'], window=20).ema_indicator()
    df['atr'] = df['high'] - df['low']

    dxy = fetch_correlation('DXY')
    if dxy is not None and len(dxy) == len(df):
        corr, _ = pearsonr(df['price'], dxy)
        df['corr_dxy'] = corr
    else:
        df['corr_dxy'] = 0

    return df.dropna()

# ======================
# MODEL
# ======================
def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_data(df):
    df = add_indicators(df)
    features = ['price', 'rsi', 'macd', 'adx', 'ema20', 'atr']
    scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_raw = df[features].values
    y_raw = df[['price']].values

    scaled_X = scaler.fit_transform(X_raw)
    target_scaler.fit(y_raw)

    X, y = [], []
    for i in range(24, len(df)):
        X.append(scaled_X[i-24:i])
        y.append(y_raw[i])

    return np.array(X), np.array(y), scaler, target_scaler

# ======================
# DERIV
# ======================
async def place_trade(contract_type, amount):
    try:
        async with websockets.connect("wss://ws.deriv.com/websockets/v3?app_id=1089") as ws:
            await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
            await ws.recv()
            await ws.send(json.dumps({
                "buy": 1,
                "price": amount,
                "parameters": {
                    "amount": amount,
                    "basis": "stake",
                    "contract_type": contract_type,
                    "currency": "USD",
                    "duration": 4,
                    "duration_unit": "h",
                    "symbol": "frxXAUUSD"
                }}))
            await ws.recv()
            return True
    except Exception as e:
        send_telegram_message(f"❌ Trade failed: {str(e)}")
        return False

async def get_balance():
    try:
        async with websockets.connect("wss://ws.deriv.com/websockets/v3?app_id=1089") as ws:
            await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
            await ws.recv()
            await ws.send(json.dumps({"balance": 1, "subscribe": 0}))
            response = await ws.recv()
            return float(json.loads(response)['balance']['balance'])
    except Exception:
        return 0

# ======================
# TRADING
# ======================
def is_market_open():
    est = datetime.utcnow() - timedelta(hours=5)
    if est.weekday() == 5 or (est.weekday() == 4 and est.hour >= 17) or (est.weekday() == 6 and est.hour < 18):
        return False
    return True

def log_trade(current, predicted, stake, contract):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(f"{datetime.utcnow()},{current},{predicted},{stake},{contract}\n")

async def trade_logic(current_price, predicted_price, balance):
    stake = min(round(balance * MAX_STAKE_PERCENT, 2), MAX_STAKE)
    contract = "CALL" if predicted_price > current_price else "PUT"
    success = await place_trade(contract, stake)
    if success:
        log_trade(current_price, predicted_price, stake, contract)
        send_telegram_message(f"✅ Trade: {contract} @ {current_price:.2f} → {predicted_price:.2f} | ${stake}")

# ======================
# MAIN LOOP
# ======================
async def main_loop():
    send_telegram_message("🚀 Day Trading Bot Started")
    model = None

    while True:
        try:
            if not is_market_open():
                await asyncio.sleep(1800)
                continue

            df = fetch_data()
            if df is None:
                await asyncio.sleep(600)
                continue

            X, y, scaler, target_scaler = prepare_data(df)

            if model is None:
                try:
                    model = load_model(MODEL_PATH)
                except:
                    model = build_model((X.shape[1], X.shape[2]))
                    model.fit(X, y, epochs=10, batch_size=8, verbose=0)
                    model.save(MODEL_PATH)

            prediction = model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]))
            predicted_price = target_scaler.inverse_transform([[prediction[0][0]]])[0][0]
            current_price = df['price'].iloc[-1]

            balance = await get_balance()
            if balance >= MIN_BALANCE:
                await trade_logic(current_price, predicted_price, balance)

            await asyncio.sleep(TRADE_INTERVAL_SECONDS)

        except Exception as e:
            send_telegram_message(f"🚨 Bot Error: {str(e)}")
            await asyncio.sleep(600)

if __name__ == '__main__':
    asyncio.run(main_loop())
