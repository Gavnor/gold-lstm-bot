import os
import requests
import pandas as pd
import numpy as np
import asyncio
import websockets
import json
import csv
from datetime import datetime, timezone, date
import socket

# === ENVIRONMENT VARIABLES ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")

# === CONFIGURATION ===
LOG_FILE = "data/trade_log.csv"
RSI_PERIOD = 14
MAX_TRADES_PER_DAY = 5
TRADE_INTERVAL = 3600
MAX_STAKE_PERCENT = 0.2
MAX_STAKE = 10000
MIN_BALANCE = 10

# === DNS HARDENING ===
socket.getaddrinfo = lambda *args: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (args[0], args[1]))]

# === Fallback WebSocket Endpoints ===
DERIV_ENDPOINTS = [
    "wss://ws.deriv.com/websockets/v3?app_id=1089",
    "wss://ws.binaryws.com/websockets/v3?app_id=1089",
    "wss://ws.deriv.be/websockets/v3?app_id=1089"
]

# === Helper Functions ===

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        print("Telegram send error:", e)

def fetch_data():
    try:
        url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=50&apikey={TWELVE_API_KEY}"
        resp = requests.get(url, timeout=10).json()
        if "values" not in resp:
            send_telegram(f"Data fetch error: {resp}")
            return None
        df = pd.DataFrame(resp["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["close"] = df["close"].astype(float)
        df = df.sort_values("datetime").reset_index(drop=True)
        return df
    except Exception as e:
        send_telegram(f"Fetch error: {e}")
        return None

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def count_today_trades():
    if not os.path.exists(LOG_FILE):
        return 0
    try:
        df = pd.read_csv(LOG_FILE, header=None)
        df[0] = pd.to_datetime(df[0])
        today = datetime.now(timezone.utc).date()
        return df[df[0].dt.date == today].shape[0]
    except:
        return 0

def log_trade(direction, price, stake):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(timezone.utc).isoformat(), direction, price, stake])

async def websocket_connect():
    for ep in DERIV_ENDPOINTS:
        try:
            ws = await websockets.connect(ep, ping_interval=20, ping_timeout=10)
            return ws
        except Exception as e:
            print(f"Connection failed: {ep} => {e}")
    raise ConnectionError("All Deriv endpoints failed")

async def get_balance():
    ws = await websocket_connect()
    try:
        await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
        await ws.recv()
        await ws.send(json.dumps({"balance": 1}))
        response = await ws.recv()
        await ws.close()
        return float(json.loads(response)["balance"]["balance"])
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
                "contract_type": "CALL" if contract_type == "rise" else "PUT",
                "currency": "USD",
                "duration": 4,
                "duration_unit": "h",
                "symbol": "frxXAUUSD"
            }
        }
        await ws.send(json.dumps(trade))
        result = json.loads(await ws.recv())
        await ws.close()
        return "error" not in result
    except Exception as e:
        await ws.close()
        print("Trade error:", e)
        return False

async def trade_cycle():
    try:
        if count_today_trades() >= MAX_TRADES_PER_DAY:
            print("📉 Trade limit reached")
            return

        df = fetch_data()
        if df is None or len(df) < RSI_PERIOD + 1:
            return

        df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        price = df["close"].iloc[-1]
        rsi = df["rsi"].iloc[-1]
        ema = df["ema20"].iloc[-1]

        send_telegram(f"🔍 RSI: {rsi:.2f} | Price: {price:.2f} | EMA20: {ema:.2f}")

        if rsi < 50 and price < ema:
            signal = "rise"
        elif rsi > 50 and price > ema:
            signal = "fall"
        else:
            send_telegram("📊 No trade signal.")
            return

        balance = await get_balance()
        if balance < MIN_BALANCE:
            send_telegram(f"⚠ Balance too low: ${balance:.2f}")
            return

        stake = round(min(balance * MAX_STAKE_PERCENT, MAX_STAKE), 2)
        success = await place_trade(signal, stake)

        if success:
            log_trade(signal, price, stake)
            send_telegram(f"✅ Trade executed: {signal.upper()} | Price: {price:.2f} | Stake: ${stake}")
        else:
            send_telegram(f"❌ Trade failed. Signal: {signal} | Stake: ${stake}")

    except Exception as e:
        send_telegram(f"❌ Loop error: {e}")

async def main_loop():
    send_telegram("🚀 Gold RSI+EMA20 Bot Started")
    while True:
        await trade_cycle()
        await asyncio.sleep(TRADE_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main_loop())
