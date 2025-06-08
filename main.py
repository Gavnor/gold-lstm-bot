import os
import requests
import numpy as np
import pandas as pd
import time
import csv
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import asyncio
import websockets
import json

TELEGRAM_TOKEN = os.getenv("7815331489:AAEe0BYA1lKdsFP88-4ta1d9Szg95bQYXw0")
TELEGRAM_CHAT_ID = os.getenv("1035839883")
DERIV_TOKEN = os.getenv("DOyvRgW9c8kERfE")
TWELVE_API_KEY = os.getenv("fed6f938e3ec484e929c9a3e5053fe3a")

MODEL_PATH = 'model/gold_lstm_model.h5'
LOG_FILE = 'data/trade_log.csv'

# Telegram alert
def send_telegram_message(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': msg})

# Fetch gold prices
def fetch_hourly_gold_data():
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=48&apikey=fed6f938e3ec484e929c9a3e5053fe3a"
    resp = requests.get(url)

    try:
        data = resp.json()
        print("🔍 API response preview:", data)  # DEBUG LINE

        if 'values' in data:
            df = pd.DataFrame(data['values'])[::-1]
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['price'] = df['close'].astype(float)
            return df[['datetime', 'price']].reset_index(drop=True)
        else:
            send_telegram_message(f"⚠️ API error: {data.get('message', 'No values returned.')}")
            return None
    except Exception as e:
        send_telegram_message(f"❌ Exception while fetching data: {e}")
        return None

# Prepare data for LSTM
def prepare_data(data, window_size=12):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[['price']])
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i])
        y.append(scaled[i])
    return np.array(X), np.array(y), scaler

# Train or load model, then predict
def predict_price(data):
    X, y, scaler = prepare_data(data)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    if not os.path.exists(MODEL_PATH):
        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, batch_size=8, verbose=0)
        model.save(MODEL_PATH)
    else:
        model = load_model(MODEL_PATH)
    pred_scaled = model.predict(X[-1].reshape(1, X.shape[1], 1))[0][0]
    return scaler.inverse_transform([[pred_scaled]])[0][0]

# Log trades to CSV
def log_trade(entry, current, predicted, stake, contract):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([datetime.utcnow(), entry, current, predicted, stake, contract])

# Deriv trading via WebSocket
async def get_balance():
    async with websockets.connect("wss://ws.deriv.com/websockets/v3?app_id=1089") as ws:
        await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
        await ws.recv()
        await ws.send(json.dumps({"balance": 1, "subscribe": 0}))
        return float(json.loads(await ws.recv())['balance']['balance'])

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
                "duration": 5,
                "duration_unit": "t",
                "symbol": "frxXAUUSD"
            }}))
        print("Trade placed.")

async def trade_on_signal(current_price, predicted_price):
    gap = abs(predicted_price - current_price)
    if gap < 20:
        print("Gap too small, skipping")
        return
    balance = await get_balance()
    stake = round(0.2 * balance, 2)
    contract = "CALL" if predicted_price > current_price else "PUT"
    await place_trade(contract, stake)
    log_trade("AUTO", current_price, predicted_price, stake, contract)
    send_telegram_message(f"TRADE: {contract} | Curr: {current_price:.2f} | Pred: {predicted_price:.2f} | Stake: ${stake}")

async def main_loop():
    while True:
        df = fetch_hourly_gold_data()
        if df is None:
            print("⚠️ Skipping this round due to bad data")
            await asyncio.sleep(600)  # Wait 10 minutes before retrying
            continue
        
        # If data is good, process prediction and trading
        predicted_price = predict_price(df)
        current_price = df['price'].iloc[-1]
        await trade_on_signal(current_price, predicted_price)
        
        await asyncio.sleep(14400)  # Wait 4 hours before next fetch

if __name__ == '__main__':
    asyncio.run(main_loop())
