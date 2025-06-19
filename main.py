import os
import requests
import pandas as pd
import numpy as np
import csv
import asyncio
import websockets
import json
from datetime import datetime, timedelta, timezone
import socket

# === ENVIRONMENT VARIABLES ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")

# === CONFIGURATION ===
LOG_FILE = 'data/trade_log.csv'
MAX_TRADES_PER_DAY = 5
TRADE_INTERVAL = 3600  # 1 hour
MAX_STAKE_PERCENT = 0.2
MAX_STAKE = 10000
MIN_BALANCE = 10
EMA_PERIOD = 20
RSI_PERIOD = 14
RSI_OVERBOUGHT = 60
RSI_OVERSOLD = 40

# Harden DNS resolution to IPv4
socket.getaddrinfo = lambda *args: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (args[0], args[1]))]

# === Deriv WebSocket Endpoints ===
DERIV_ENDPOINTS = [
    "wss://ws.deriv.com/websockets/v3?app_id=1089",
    "wss://ws.binaryws.com/websockets/v3?app_id=1089",
    "wss://ws.deriv.be/websockets/v3?app_id=1089"
]

# === Utilities ===
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': msg}, timeout=10)
    except Exception as e:
        print("Telegram Error:", e)

def fetch_data():
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=100&apikey={TWELVE_API_KEY}"
    try:
        data = requests.get(url, timeout=10).json()
        if 'values' not in data:
            send_telegram(f"⚠ Data error: {data}")
            return None
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['close'] = df['close'].astype(float)
        df = df.sort_values('datetime').reset_index(drop=True)
        return df
    except Exception as e:
        send_telegram(f"❌ Fetch error: {e}")
        return None

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_ema(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def rsi_ema_signal(df):
    df['rsi'] = compute_rsi(df['close'], RSI_PERIOD)
    df['ema'] = compute_ema(df['close'], EMA_PERIOD)
    rsi = df['rsi'].iloc[-1]
    price = df['close'].iloc[-1]
    ema = df['ema'].iloc[-1]

    msg = f"🔍 RSI check: {rsi:.2f} | Price: {price:.2f}\nEMA20: {ema:.2f}"
    send_telegram(msg)

    if rsi < RSI_OVERSOLD and price < ema:
        return "CALL"
    elif rsi > RSI_OVERBOUGHT and price > ema:
        return "PUT"
    return None

def log_trade(direction, price, stake):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(timezone.utc).isoformat(), direction, price, stake])

def count_today_trades():
    if not os.path.exists(LOG_FILE):
        return 0
    df = pd.read_csv(LOG_FILE, header=None)
    df[0] = pd.to_datetime(df[0])
    return df[df[0].dt.date == datetime.now(timezone.utc).date()].shape[0]

async def websocket_connect():
    for endpoint in DERIV_ENDPOINTS:
        try:
            ws = await websockets.connect(endpoint, ping_interval=20, ping_timeout=10)
            return ws
        except Exception as e:
            print(f"Failed connection to {endpoint}: {e}")
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
        return "error" not in json.loads(response)
    except Exception as e:
        await ws.close()
        raise e

async def trade_cycle():
    try:
        if count_today_trades() >= MAX_TRADES_PER_DAY:
            print("⛔ Reached daily trade limit")
            return

        df = fetch_data()
        if df is None:
            return

        signal = rsi_ema_signal(df)
        if signal is None:
            send_telegram("📉 No signal detected this round.")
            return

        balance = await get_balance()
        if balance < MIN_BALANCE:
            send_telegram(f"⚠ Low balance: ${balance:.2f}")
            return

        stake = round(min(MAX_STAKE_PERCENT * balance, MAX_STAKE), 2)
        success = await place_trade(signal, stake)

        if success:
            log_trade(signal, df['close'].iloc[-1], stake)
            send_telegram(f"\n🚀 Trade executed: {signal}\nPrice: {df['close'].iloc[-1]}\nStake: ${stake}")
        else:
            send_telegram("❌ Trade failed.")

    except Exception as e:
        send_telegram(f"❌ Loop error: {e}")

async def main_loop():
    send_telegram("🚀 RSI+EMA Gold Bot Running")
    while True:
        await trade_cycle()
        await asyncio.sleep(TRADE_INTERVAL)

if __name__ == '__main__':
    asyncio.run(main_loop())
