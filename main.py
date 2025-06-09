import os
import requests
import numpy as np
import pandas as pd
import time
import csv
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import asyncio
import websockets
import json
import socket

# Environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")

# Configuration
MODEL_PATH = 'model/gold_lstm_model.h5'
LOG_FILE = 'data/trade_log.csv'
MAX_RETRIES = 3
RETRY_DELAY = 5
WEBSOCKET_TIMEOUT = 10
MAX_STAKE_PERCENT = 0.2  # Risk management: max 20% of balance per trade

# Force IPv4 to prevent DNS resolution issues
socket.getaddrinfo = lambda *args: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (args[0], args[1]))]

def send_telegram_message(msg):
    """Send message to Telegram with error handling"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': msg}, timeout=10)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

def is_market_open():
    """Check if gold market is open (Sunday 6PM to Friday 5PM EST)"""
    utc_now = datetime.utcnow()
    est_time = utc_now - timedelta(hours=5)  # Convert to EST
    weekday = est_time.weekday()
    hour = est_time.hour
    
    # Market closed: Friday after 5PM EST to Sunday before 6PM EST
    if (weekday == 4 and hour >= 17) or weekday == 5 or (weekday == 6 and hour < 18):
        return False
    return True

async def get_deriv_connection():
    """Establish robust WebSocket connection to Deriv with fallback endpoints"""
    endpoints = [
        'wss://ws.binaryws.com/websockets/v3?app_id=1089',  # Primary
        'wss://ws.deriv.com/websockets/v3?app_id=1089',     # Fallback 1
        'wss://ws.deriv.be/websockets/v3?app_id=1089'       # Fallback 2 (EU)
    ]
    
    for url in endpoints:
        try:
            conn = await websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=60,
                close_timeout=10,
                ssl=True
            )
            # Test connection
            await asyncio.wait_for(conn.ping(), timeout=5)
            return conn
        except Exception as e:
            print(f"Failed to connect to {url}: {str(e)}")
            continue
    
    raise ConnectionError("All Deriv endpoints failed")

async def with_retry_async(func, operation_name="operation"):
    """Retry decorator for async operations"""
    for attempt in range(MAX_RETRIES):
        try:
            return await func()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                send_telegram_message(f"❌ Failed {operation_name} after {MAX_RETRIES} attempts: {str(e)}")
                raise
            await asyncio.sleep(RETRY_DELAY)
            send_telegram_message(f"⚠️ Retrying {operation_name} (attempt {attempt + 1}): {str(e)}")

def with_retry_sync(func, operation_name="operation"):
    """Retry decorator for synchronous operations"""
    for attempt in range(MAX_RETRIES):
        try:
            return func()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                send_telegram_message(f"❌ Failed {operation_name} after {MAX_RETRIES} attempts: {str(e)}")
                raise
            time.sleep(RETRY_DELAY)
            send_telegram_message(f"⚠️ Retrying {operation_name} (attempt {attempt + 1}): {str(e)}")

def fetch_hourly_gold_data():
    """Fetch gold price data from Twelve Data API"""
    try:
        url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=48&apikey={TWELVE_API_KEY}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if 'values' in data:
            df = pd.DataFrame(data['values'])[::-1]
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['price'] = df['close'].astype(float)
            return df[['datetime', 'price']].reset_index(drop=True)
        send_telegram_message(f"⚠️ API error: {data.get('message', 'No values returned')}")
    except Exception as e:
        send_telegram_message(f"❌ Data fetch failed: {str(e)}")
    return None

def prepare_data(data, window_size=12):
    """Prepare data for LSTM model"""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[['price']])
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i])
        y.append(scaled[i])
    return np.array(X), np.array(y), scaler

def predict_price(data):
    """Make price prediction using LSTM model"""
    X, y, scaler = prepare_data(data)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    try:
        model = load_model(MODEL_PATH)
    except Exception:
        model = Sequential([
            LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, batch_size=8, verbose=0)
        model.save(MODEL_PATH)
    
    pred_scaled = model.predict(X[-1].reshape(1, X.shape[1], 1))[0][0]
    return scaler.inverse_transform([[pred_scaled]])[0][0]

def log_trade(entry, current, predicted, stake, contract):
    """Log trade details to CSV"""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            entry,
            current,
            predicted,
            stake,
            contract,
            "SUCCESS" if stake > 0 else "FAILED"
        ])

async def get_balance():
    """Get current account balance"""
    async def _get_balance():
        async with await get_deriv_connection() as ws:
            # Authorize
            await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
            auth_response = await asyncio.wait_for(ws.recv(), timeout=WEBSOCKET_TIMEOUT)
            
            if 'error' in json.loads(auth_response):
                raise ValueError(f"Auth error: {auth_response}")
            
            # Get balance
            await ws.send(json.dumps({"balance": 1, "subscribe": 0}))
            response = await asyncio.wait_for(ws.recv(), timeout=WEBSOCKET_TIMEOUT)
            return float(json.loads(response)['balance']['balance'])
    
    return await with_retry_async(_get_balance, "balance check")

async def place_trade(contract_type, amount):
    """Execute trade on Deriv with valid parameters"""
    async def _place_trade():
        async with await get_deriv_connection() as ws:
            # Authorize
            await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
            await asyncio.wait_for(ws.recv(), timeout=WEBSOCKET_TIMEOUT)
            
            # Try multiple duration options for gold
            duration_options = [
                {"duration": 5, "duration_unit": "m"},  # 5 minutes
                {"duration": 1, "duration_unit": "h"},  # 1 hour
                {"duration": 1, "duration_unit": "d"}   # 1 day
            ]
            
            for duration in duration_options:
                trade_params = {
                    "buy": 1,
                    "price": amount,
                    "parameters": {
                        "amount": amount,
                        "basis": "stake",
                        "contract_type": contract_type,
                        "currency": "USD",
                        "symbol": "frxXAUUSD",
                        **duration
                    }
                }
                
                await ws.send(json.dumps(trade_params))
                response = await asyncio.wait_for(ws.recv(), timeout=WEBSOCKET_TIMEOUT)
                response_data = json.loads(response)
                
                if 'error' not in response_data:
                    return True
                
                # If error contains "duration", try next option
                if "duration" in response_data.get('error', {}).get('message', '').lower():
                    continue
                    
                raise ValueError(f"Trade error: {response}")
            
            raise ValueError("All duration options failed")
    
    return await with_retry_async(_place_trade, "trade execution")

async def trade_on_signal(current_price, predicted_price):
    """Execute trading logic with market hours check"""
    if not is_market_open():
        send_telegram_message("⏸️ Market closed - skipping trade")
        return
    
    try:
        balance = await get_balance()
        if balance <= 10:  # Minimum $10 balance
            send_telegram_message(f"⚠️ Low balance: ${balance:.2f}")
            return
        
        stake = min(round(MAX_STAKE_PERCENT * balance, 2), 10000)  # Cap at $10k
        contract = "CALL" if predicted_price > current_price else "PUT"
        
        if await place_trade(contract, stake):
            log_trade("AUTO", current_price, predicted_price, stake, contract)
            send_telegram_message(
                f"✅ Trade executed:\n"
                f"Type: {contract}\n"
                f"Current: {current_price:.2f}\n"
                f"Predicted: {predicted_price:.2f}\n"
                f"Stake: ${stake:.2f}\n"
                f"Balance: ${balance:.2f}"
            )
    except Exception as e:
        log_trade("AUTO", current_price, predicted_price, 0, "FAILED")
        send_telegram_message(f"❌ Trade failed: {str(e)}")

async def main_loop():
    """Main trading loop"""
    send_telegram_message("🚀 Gold Trading Bot Started")
    while True:
        try:
            if not is_market_open():
                next_open = "Sunday 6PM EST" if datetime.utcnow().weekday() >= 5 else "Next trading day"
                print(f"⏸️ Market closed. Next open: {next_open}")
                await asyncio.sleep(3600)  # Check hourly when market is closed
                continue
            
            df = with_retry_sync(fetch_hourly_gold_data, "data fetch")
            if df is not None:
                current_price = df['price'].iloc[-1]
                predicted_price = predict_price(df)
                
                print(f"📊 Current: {current_price:.2f} | Predicted: {predicted_price:.2f}")
                await trade_on_signal(current_price, predicted_price)
            
            await asyncio.sleep(600)  # Wait 10 minutes between cycles
            
        except Exception as e:
            send_telegram_message(f"🚨 Critical error in main loop: {str(e)}")
            await asyncio.sleep(300)  # Wait 5 minutes before retry

if __name__ == '__main__':
    print("🚀 Starting Gold Trading Bot")
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        send_telegram_message("🛑 Bot manually stopped")
    except Exception as e:
        send_telegram_message(f"💥 Bot crashed: {str(e)}")
        raise
