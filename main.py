import os
import requests
import numpy as np
import pandas as pd
import asyncio
import websockets
import json
from datetime import datetime, timedelta, timezone
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# ENV VARIABLES
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")

# FILE PATHS
MODEL_PATH = "model/gold_lstm_model.h5"
LOG_FILE = "data/trade_log.csv"

# CONFIG
WINDOW_SIZE = 24
MIN_GAP = 10  # minimum gap between prediction and price to enter trade
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5
MAX_DAILY_TRADES = 5
DAILY_COUNTER_FILE = "data/daily_counter.txt"

# Initialize tensorflow thread-safe mode
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# UTILS

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})

def get_today_key():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return today

def load_daily_counter():
    if not os.path.exists(DAILY_COUNTER_FILE):
        return {}
    with open(DAILY_COUNTER_FILE, "r") as f:
        data = f.read().strip()
        if not data:
            return {}
        return json.loads(data)

def save_daily_counter(counter):
    with open(DAILY_COUNTER_FILE, "w") as f:
        f.write(json.dumps(counter))

def increment_trade_count():
    counter = load_daily_counter()
    today = get_today_key()
    if today not in counter:
        counter[today] = 0
    counter[today] += 1
    save_daily_counter(counter)
    return counter[today]

def get_trade_count_today():
    counter = load_daily_counter()
    today = get_today_key()
    return counter.get(today, 0)

def fetch_data():
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=200&apikey={TWELVE_API_KEY}"
    resp = requests.get(url).json()
    if 'values' not in resp:
        send_telegram("❌ Failed to fetch data.")
        return None
    df = pd.DataFrame(resp['values'])[::-1]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['price'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    return df[['datetime', 'price', 'high', 'low']].reset_index(drop=True)

def compute_atr(df, period=ATR_PERIOD):
    high = df['high']
    low = df['low']
    close = df['price']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def prepare_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['price']])
    X, y = [], []
    for i in range(WINDOW_SIZE, len(scaled)):
        X.append(scaled[i-WINDOW_SIZE:i])
        y.append(scaled[i])
    return np.array(X), np.array(y), scaler

def predict(df):
    X, y, scaler = prepare_data(df)
    model = load_model(MODEL_PATH)
    pred_scaled = model.predict(X[-1].reshape(1, X.shape[1], 1))[0][0]
    pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
    return pred_price

async def get_balance():
    async with websockets.connect("wss://ws.deriv.com/websockets/v3?app_id=1089") as ws:
        await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
        await ws.recv()
        await ws.send(json.dumps({"balance": 1, "subscribe": 0}))
        resp = await ws.recv()
        balance = float(json.loads(resp)['balance']['balance'])
        return balance

async def place_trade(contract_type, amount):
    async with websockets.connect("wss://ws.deriv.com/websockets/v3?app_id=1089") as ws:
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
            }
        }))
        await ws.recv()

async def execute_trade(current_price, predicted_price, atr_value):
    if abs(predicted_price - current_price) < MIN_GAP:
        print("Gap too small, no trade.")
        return

    count_today = get_trade_count_today()
    if count_today >= MAX_DAILY_TRADES:
        print("Max daily trades reached.")
        return

    balance = await get_balance()
    stake = round(min(balance * 0.2, 10000), 2)
    contract_type = "CALL" if predicted_price > current_price else "PUT"

    await place_trade(contract_type, stake)
    increment_trade_count()

    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.utcnow()},{current_price},{predicted_price},{stake},{contract_type}\n")

    send_telegram(f"✅ Trade executed: {'BUY' if contract_type == 'CALL' else 'SELL'} | Current: {current_price} | Predicted: {predicted_price} | Stake: {stake}")

async def main_loop():
    send_telegram("🚀 Bot Started")
    while True:
        try:
            df = fetch_data()
            if df is None:
                await asyncio.sleep(300)
                continue

            df['ATR'] = compute_atr(df)
            atr_value = df['ATR'].iloc[-1]

            predicted_price = predict(df)
            current_price = df['price'].iloc[-1]

            await execute_trade(current_price, predicted_price, atr_value)
            await asyncio.sleep(7200)  # 2 hours between checks

        except Exception as e:
            send_telegram(f"❌ Loop error: {str(e)}")
            await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(main_loop())
    
