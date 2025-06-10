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
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from scipy.stats import pearsonr

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

TRADE_DURATION = 4
STOP_LOSS_PCT = 0.03
TAKE_PROFIT_PCT = 0.06
MAX_STAKE_PERCENT = 0.2
MIN_BALANCE = 10
MAX_STAKE = 10000
VOLATILITY_THRESHOLD = 1.5
CORRELATION_THRESHOLD = 0.7

socket.getaddrinfo = lambda *args: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (args[0], args[1]))]

def send_telegram_message(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': msg}, timeout=10)
    except Exception as e:
        print(f"Telegram Error: {e}")

def fetch_data():
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=100&apikey={TWELVE_API_KEY}"
    try:
        response = requests.get(url, timeout=10).json()
        if 'values' in response:
            df = pd.DataFrame(response['values'])[::-1]
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['price'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            return df.reset_index(drop=True)
        send_telegram_message(f"API error: {response.get('message', 'Unknown error')}")
    except Exception as e:
        send_telegram_message(f"Fetch failed: {e}")
    return None

def add_indicators(df):
    df['ema20'] = EMAIndicator(close=df['price'], window=20).ema_indicator()
    df['rsi'] = RSIIndicator(close=df['price'], window=14).rsi()
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['price'], window=14).average_true_range()
    return df.dropna()

def prepare_data(df):
    features = ['price', 'ema20', 'rsi', 'atr']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    X, y = [], []
    for i in range(24, len(scaled)):
        X.append(scaled[i-24:i])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.4),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def log_trade(entry, current, predicted, stake, contract):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(f"{datetime.utcnow().isoformat()},{entry},{current},{predicted},{stake},{contract}\n")

async def get_balance():
    async with websockets.connect("wss://ws.deriv.com/websockets/v3?app_id=1089") as ws:
        await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
        await ws.recv()
        await ws.send(json.dumps({"balance": 1, "subscribe": 0}))
        resp = await ws.recv()
        return float(json.loads(resp)['balance']['balance'])

async def place_trade(contract_type, amount):
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
                "duration": TRADE_DURATION,
                "duration_unit": "h",
                "symbol": "frxXAUUSD"
            }}))
        await ws.recv()
        return True

async def execute_trade(current_price, predicted_price, model_name, df, balance):
    stake = round(min(balance * MAX_STAKE_PERCENT, MAX_STAKE), 2)
    contract = "CALL" if predicted_price > current_price else "PUT"
    result = await place_trade(contract, stake)
    if result:
        log_trade(model_name, current_price, predicted_price, stake, contract)
        send_telegram_message(
            f"✅ TRADE EXECUTED:\nType: {contract}\nCurrent: {current_price:.2f}\nPredicted: {predicted_price:.2f}\nStake: ${stake}\nModel: {model_name}")

async def main_loop():
    send_telegram_message("🚀 Day Trading Bot Started")
    model = None
    while True:
        try:
            df = fetch_data()
            if df is None:
                await asyncio.sleep(600)
                continue
            df = add_indicators(df)
            X, y, scaler = prepare_data(df)
            if model is None:
                model = build_model((X.shape[1], X.shape[2]))
                try:
                    model = load_model(MODEL_PATH)
                except:
                    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
                    model.save(MODEL_PATH)
            else:
                model.fit(X, y, epochs=1, batch_size=16, verbose=0)
                model.save(MODEL_PATH)
            predicted_price = scaler.inverse_transform(
                model.predict(X[-1].reshape(1, X.shape[1], X.shape[2])))[0][0]
            current_price = df['price'].iloc[-1]
            balance = await get_balance()
            if balance >= MIN_BALANCE:
                await execute_trade(current_price, predicted_price, "EnhancedLSTM", df, balance)
            await asyncio.sleep(7200)
        except Exception as e:
            send_telegram_message(f"❌ Loop error: {e}")
            await asyncio.sleep(600)

if __name__ == '__main__':
    asyncio.run(main_loop())
        
