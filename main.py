import os
import requests
import pandas as pd
import numpy as np
import csv
import asyncio
import websockets
import json
from datetime import datetime, timezone
import socket

# === ENVIRONMENT VARIABLES ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")

# === CONFIGURATION ===
LOG_FILE = 'data/trade_log.csv'
MAX_TRADES_PER_DAY = 5
TRADE_INTERVAL = 3600  # Every hour
MAX_STAKE_PERCENT = 0.2
MAX_STAKE = 10000
MIN_BALANCE = 10
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# DNS workaround
socket.getaddrinfo = lambda *args: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (args[0], args[1]))]

DERIV_ENDPOINTS = [
    "wss://ws.deriv.com/websockets/v3?app_id=1089",
    "wss://ws.binaryws.com/websockets/v3?app_id=1089",
    "wss://ws.deriv.be/websockets/v3?app_id=1089"
]

# === UTILITIES ===

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': msg}, timeout=10)
    except Exception as e:
        print("Telegram Error:", e)

def fetch_data():
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=50&apikey={TWELVE_API_KEY}"
    try:
        data = requests.get(url, timeout=10).json()
        if 'values' not in data:
            send_telegram(f"⚠️ Data error: {data}")
            return None
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['close'] = df['close'].astype(float)
        df = df.sort_values('datetime').reset_index(drop=True)
        return df
    except Exception as e:
        send_telegram(f"❌ Fetch error: {e}")
        return None

def compute_rsi(series, period=RSI_PERIOD):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_ema(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def strategy_rsi_ema(df):
    df['rsi'] = compute_rsi(df['close'])
    df['ema20'] = compute_ema(df['close'])
    latest = df.iloc[-1]

    rsi = latest['rsi']
    price = latest['close']
    ema20 = latest['ema20']

    send_telegram(f"🔍 RSI: {rsi:.2f} | Price: {price:.2f} | EMA20: {ema20:.2f}")

    if rsi < RSI_OVERSOLD and price < ema20:
        return "CALL"
    elif rsi > RSI_OVERBOUGHT and price > ema20:
        return "PUT"
    else:
        return None

def log_trade(direction, price, stake):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(timezone.utc).isoformat(), direction, price, stake])

def count_today_trades():
    if not os.path.exists(LOG_FILE):
        return 0
    today = datetime.now(timezone.utc).date()
    df = pd.read_csv(LOG_FILE, header=None)
    df[0] = pd.to_datetime(df[0])
    return df[df[0].dt.date == today].shape[0]

async def websocket_connect():
    for endpoint in DERIV_ENDPOINTS:
        try:
            ws = await websockets.connect(endpoint, ping_interval=20, ping_timeout=10)
            return ws
        except Exception as e:
            print(f"❌ Failed connection to {endpoint}: {e}")
    raise ConnectionError("All Deriv WebSocket connections failed.")

async def get_balance():
    ws = await websocket_connect()
    try:
        await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
        await ws.recv()
        await ws.send(json.dumps({"balance": 1}))
        response = await ws.recv()
        await ws.close()
        return float(json.loads(response)['balance']['balance'])
    except Exception as e:
        await ws.close()
        raise e

async def place_trade(contract_type, stake):
    ws = await websocket_connect()
    try:
        await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
        await ws.recv()

        trade = {
            "buy": 1,
            "price": stake,
            "parameters": {
                "amount": stake,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": 4,
                "duration_unit": "h",
                "symbol": "frxXAUUSD"
            }
        }
        await ws.send(json.dumps(trade))
        response = await ws.recv()
        await ws.close()

        result = json.loads(response)
        return "error" not in result
    except Exception as e:
        await ws.close()
        raise e

async def trade_cycle():
    try:
        if count_today_trades() >= MAX_TRADES_PER_DAY:
            print("🔁 Daily trade limit reached")
            return

        df = fetch_data()
        if df is None: return

        signal = strategy_rsi_ema(df)
        if signal is None:
            print("📉 No valid signal")
            return

        balance = await get_balance()
        stake = round(min(MAX_STAKE_PERCENT * balance, MAX_STAKE), 2)

        success = await place_trade(signal, stake)
        if success:
            log_trade(signal, df['close'].iloc[-1], stake)
            send_telegram(f"✅ Trade executed: {signal}\nPrice: {df['close'].iloc[-1]:.2f}\nStake: ${stake}")
        else:
            send_telegram(f"❌ Trade failed. Signal: {signal} | Stake: ${stake}")
    except Exception as e:
        send_telegram(f"❌ Loop error: {e}")

async def main_loop():
    send_telegram("🚀 Gold Predictor RSI + EMA20 Bot Running")
    while True:
        await trade_cycle()
        await asyncio.sleep(TRADE_INTERVAL)

if __name__ == '__main__':
    asyncio.run(main_loop())
