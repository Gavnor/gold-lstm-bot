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
import socket

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")

MODEL_PATH = 'model/gold_lstm_model.h5'
LOG_FILE = 'data/trade_log.csv'

# Force IPv4 to prevent DNS resolution issues
socket.getaddrinfo = lambda *args: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (args[0], args[1]))]

def send_telegram_message(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': msg})

def fetch_hourly_gold_data():
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=48&apikey={TWELVE_API_KEY}"
    resp = requests.get(url)
    try:
        data = resp.json()
        print("🔍 API response preview:", data)
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

def prepare_data(data, window_size=12):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[['price']])
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i])
        y.append(scaled[i])
    return np.array(X), np.array(y), scaler

def predict_price(data):
    X, y, scaler = prepare_data(data)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"⚠️ Model load failed: {e}")
        print("🔁 Re-training model on CPU...")
        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, batch_size=8, verbose=0)
        model.save(MODEL_PATH)
    pred_scaled = model.predict(X[-1].reshape(1, X.shape[1], 1))[0][0]
    return scaler.inverse_transform([[pred_scaled]])[0][0]

def log_trade(entry, current, predicted, stake, contract):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.utcnow(), entry, current, predicted, stake, contract])

async def get_balance():
    try:
        async with websockets.connect('wss://ws.binaryws.com/websockets/v3?app_id=1089', timeout=10) as ws:
            await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
            auth_response = await ws.recv()
            if 'error' in json.loads(auth_response):
                send_telegram_message(f"Auth failed: {auth_response}")
                return 0
            
            await ws.send(json.dumps({"balance": 1, "subscribe": 0}))
            response = await ws.recv()
            return float(json.loads(response)['balance']['balance'])
    except Exception as e:
        send_telegram_message(f"Balance check failed: {str(e)}")
        return 0

async def place_trade(contract_type, amount):
    try:
        async with websockets.connect('wss://ws.binaryws.com/websockets/v3?app_id=1089', timeout=10) as ws:
            await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
            await ws.recv()
            
            trade_params = {
                "buy": 1,
                "price": amount,
                "parameters": {
                    "amount": amount,
                    "basis": "stake",
                    "contract_type": contract_type,
                    "currency": "USD",
                    "duration": 5,
                    "duration_unit": "t",
                    "symbol": "frxXAUUSD"
                }
            }
            
            await ws.send(json.dumps(trade_params))
            trade_response = await ws.recv()
            print(f"Trade placed. Response: {trade_response}")
            return True
    except Exception as e:
        send_telegram_message(f"Trade failed: {str(e)}")
        return False

async def trade_on_signal(current_price, predicted_price):
    try:
        gap = abs(predicted_price - current_price)
        print(f"🔎 Price gap detected: {gap:.2f}")
        
        balance = await get_balance()
        if balance <= 0:
            send_telegram_message("Zero balance - skipping trade")
            return
            
        stake = round(0.2 * balance, 2)
        contract = "CALL" if predicted_price > current_price else "PUT"
        
        success = await place_trade(contract, stake)
        if success:
            log_trade("AUTO", current_price, predicted_price, stake, contract)
            send_telegram_message(
                f"Trade executed:\n"
                f"Type: {contract}\n"
                f"Current: {current_price:.2f}\n"
                f"Predicted: {predicted_price:.2f}\n"
                f"Stake: ${stake:.2f}"
            )
    except Exception as e:
        send_telegram_message(f"Trade signal failed: {str(e)}")

async def main_loop():
    while True:
        try:
            print("Fetching new data...")
            df = fetch_hourly_gold_data()
            if df is None:
                await asyncio.sleep(300)
                continue
                
            predicted_price = predict_price(df)
            current_price = df['price'].iloc[-1]
            
            print(f"Current: {current_price:.2f} | Predicted: {predicted_price:.2f}")
            await trade_on_signal(current_price, predicted_price)
            
            await asyncio.sleep(600)
        except Exception as e:
            send_telegram_message(f"Main loop crashed: {str(e)}")
            await asyncio.sleep(300)

if __name__ == '__main__':
    print("Starting trading bot...")
    asyncio.run(main_loop())
