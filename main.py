import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
import asyncio
import websockets
import json
import socket
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from scipy.stats import pearsonr

# Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")

MODEL_PATH = 'model/enhanced_gold_lstm.h5'
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

DEBUG_FORCE_TRADE = True

# DNS Resilience
socket.getaddrinfo = lambda *args: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (args[0], args[1]))]

# Multi-endpoint Deriv websocket failover:
async def safe_deriv_ws_connect():
    endpoints = [
        "wss://ws.deriv.com/websockets/v3?app_id=1089",
        "wss://ws.binaryws.com/websockets/v3?app_id=1089",
        "wss://ws.deriv.be/websockets/v3?app_id=1089"
    ]

    for url in endpoints:
        try:
            family = socket.AF_INET
            host = url.split("/")[2].split(":")[0]
            socket.getaddrinfo(host, None, family)
            conn = await websockets.connect(url)
            return conn
        except Exception as e:
            print(f"⚠️ Failed to connect to {url}: {str(e)}")
            continue

    raise ConnectionError("❌ All websocket endpoints failed DNS resolution")

# Telegram helper
def send_telegram_message(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': msg}, timeout=10)
    except Exception as e:
        print(f"Telegram Error: {e}")

# Data fetching & feature engineering
def fetch_data():
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=200&apikey={TWELVE_API_KEY}"
    resp = requests.get(url, timeout=10)
    data = resp.json()
    if 'values' not in data:
        send_telegram_message("⚠️ Data fetch failed")
        return None
    df = pd.DataFrame(data['values'])[::-1]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['price'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    return df.reset_index(drop=True)

def add_features(df):
    df['rsi'] = RSIIndicator(df['price']).rsi()
    df['ema20'] = EMAIndicator(df['price'], window=20).ema_indicator()
    df['adx'] = ADXIndicator(df['high'], df['low'], df['price']).adx()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['price']).average_true_range()
    df = df.dropna()
    return df

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_data(df):
    df = add_features(df)
    features = ['price', 'rsi', 'ema20', 'adx', 'atr']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    X, y = [], []
    window_size = 24
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler, df

# Websocket trading
async def get_balance():
    async with await safe_deriv_ws_connect() as ws:
        await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
        await ws.recv()
        await ws.send(json.dumps({"balance": 1}))
        response = await ws.recv()
        return float(json.loads(response)['balance']['balance'])

async def place_trade(contract_type, amount):
    async with await safe_deriv_ws_connect() as ws:
        await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
        await ws.recv()
        await ws.send(json.dumps({
            "buy": 1, "price": amount,
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

async def execute_trade(current_price, predicted_price, df, balance, force=False):
    if not force:
        if abs(predicted_price - current_price) < 10:
            send_telegram_message("⏭️ Skipping - price gap too small")
            return

    stake = round(min(MAX_STAKE_PERCENT * balance, MAX_STAKE), 2)
    contract_type = "CALL" if predicted_price > current_price else "PUT"
    await place_trade(contract_type, stake)
    send_telegram_message(f"✅ Trade executed: {contract_type} | Current: {current_price:.2f} | Predicted: {predicted_price:.2f} | Stake: ${stake}")

async def main_loop():
    send_telegram_message("🚀 Bot Started (Debug Mode)" if DEBUG_FORCE_TRADE else "🚀 Bot Started")
    model = None

    while True:
        try:
            df_raw = fetch_data()
            if df_raw is None:
                await asyncio.sleep(600)
                continue

            X, y, scaler, df = prepare_data(df_raw)

            if model is None:
                model = build_model((X.shape[1], X.shape[2]))
                try:
                    model = load_model(MODEL_PATH)
                except:
                    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
                    model.save(MODEL_PATH)

            current_price = df['price'].iloc[-1]
            prediction_scaled = model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]))
            predicted_price = scaler.inverse_transform(
                np.concatenate([prediction_scaled, np.zeros((1, X.shape[1]-1))], axis=1)
            )[0][0]

            balance = await get_balance()
            if balance >= MIN_BALANCE:
                await execute_trade(current_price, predicted_price, df, balance, force=DEBUG_FORCE_TRADE)

            await asyncio.sleep(7200)
        except Exception as e:
            send_telegram_message(f"❌ Loop error: {str(e)}")
            await asyncio.sleep(600)

if __name__ == '__main__':
    asyncio.run(main_loop())
