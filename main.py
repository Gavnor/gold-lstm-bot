# enhanced_gold_bot/main.py

import os
import requests
import pandas as pd
import numpy as np
import asyncio
import json
import websockets
import csv
import socket
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# === ENV ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")

# === CONFIG ===
MAX_TRADES_PER_DAY = 5
TRADE_INTERVAL = 3600
LOG_FILE = 'data/trade_log.csv'
MAX_STAKE_PERCENT = 0.2
MAX_STAKE = 1000
STAKE_CAP = 4999
MIN_BALANCE = 10
CONF_THRESHOLD = 0.6
ADX_PERIOD = 14

socket.getaddrinfo = lambda *args: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (args[0], args[1]))]
DERIV_ENDPOINTS = [
    "wss://ws.deriv.com/websockets/v3?app_id=1089",
    "wss://ws.binaryws.com/websockets/v3?app_id=1089",
    "wss://ws.deriv.be/websockets/v3?app_id=1089"
]

# === UTILS ===
def send_telegram(msg):
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except Exception as e:
        print("Telegram error:", e)

def fetch_data():
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=60&apikey={TWELVE_API_KEY}"
    try:
        data = requests.get(url).json()
        if 'values' not in data:
            return None
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['price'] = df['close'].astype(float)
        df = df.sort_values('datetime').reset_index(drop=True)
        return df[['datetime', 'price']]
    except:
        return None

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_ema(series, period=20):
    return series.ewm(span=period).mean()

def compute_adx(prices, period=14):
    high = prices * 1.002
    low = prices * 0.998
    close = prices
    plus_dm = high.diff()
    minus_dm = low.diff()
    tr = (high.combine(low, max) - low.combine(close.shift(), min)).rolling(period).mean()
    adx = (abs(plus_dm - minus_dm) / tr).rolling(period).mean()
    return adx

def prepare_features(df):
    df['return'] = df['price'].pct_change()
    df['rsi'] = compute_rsi(df['price'])
    df['ema20'] = compute_ema(df['price'])
    df['delta'] = df['price'] - df['ema20']
    df['volatility'] = df['return'].rolling(10).std()
    df['adx'] = compute_adx(df['price'])
    df = df.dropna().copy()
    df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
    return df

def train_model(df):
    features = ['rsi', 'delta', 'volatility']
    X = df[features]
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, scaler

def count_today_trades():
    if not os.path.exists(LOG_FILE): return 0
    df = pd.read_csv(LOG_FILE, header=None)
    df[0] = pd.to_datetime(df[0])
    return df[df[0].dt.date == datetime.now(timezone.utc).date()].shape[0]

def recent_losses_and_direction():
    if not os.path.exists(LOG_FILE): return 0, []
    df = pd.read_csv(LOG_FILE, header=None)
    df.columns = ['time', 'direction', 'price', 'stake', 'pnl'] if df.shape[1] == 5 else ['time', 'direction', 'price', 'stake']
    df = df.tail(5)
    loss_streak = (df['pnl'] < 0).sum() if 'pnl' in df.columns else 0
    directions = df['direction'].tolist()
    return loss_streak, directions

def log_trade(direction, price, stake):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(timezone.utc).isoformat(), direction, price, stake])

async def ws_connect():
    for endpoint in DERIV_ENDPOINTS:
        try:
            return await websockets.connect(endpoint)
        except: continue
    raise ConnectionError("❌ All Deriv endpoints failed")

async def get_balance():
    ws = await ws_connect()
    await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
    await ws.recv()
    await ws.send(json.dumps({"balance": 1}))
    resp = await ws.recv()
    await ws.close()
    return float(json.loads(resp)['balance']['balance'])

async def place_trade(direction, stake):
    ws = await ws_connect()
    await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
    await ws.recv()
    payload = {
        "buy": 1,
        "price": stake,
        "parameters": {
            "amount": stake,
            "basis": "stake",
            "contract_type": "CALL" if direction == "buy" else "PUT",
            "currency": "USD",
            "duration": 4,
            "duration_unit": "h",
            "symbol": "frxXAUUSD"
        }
    }
    await ws.send(json.dumps(payload))
    resp = await ws.recv()
    await ws.close()
    return "error" not in json.loads(resp)

async def trade_cycle():
    if count_today_trades() >= MAX_TRADES_PER_DAY:
        return

    df = fetch_data()
    if df is None: return

    df = prepare_features(df)
    model, scaler = train_model(df)
    X_latest = scaler.transform([df[['rsi', 'delta', 'volatility']].iloc[-1]])
    proba = model.predict_proba(X_latest)[0]
    pred_class = model.predict(X_latest)[0]
    confidence = proba[pred_class]
    current_price = df['price'].iloc[-1]
    ema = df['ema20'].iloc[-1]
    rsi = df['rsi'].iloc[-1]
    adx = df['adx'].iloc[-1]
    direction = "buy" if pred_class == 1 else "sell"

    if confidence < CONF_THRESHOLD:
        send_telegram(f"⚠ Low confidence: {confidence:.2f}, skipping trade")
        return

    if adx < 20:
        send_telegram(f"📉 ADX too low ({adx:.2f}), market likely range-bound. Skipping.")
        return

    losses, directions = recent_losses_and_direction()
    if directions[-2:] == [direction, direction]:
        send_telegram(f"🚫 Avoiding 3rd {direction} in a row. Skipping.")
        return

    if direction == "buy" and not (rsi < 50 and current_price < ema):
        return
    if direction == "sell" and not (rsi > 50 and current_price > ema):
        return

    try:
        balance = await get_balance()
        stake = round(min(balance * MAX_STAKE_PERCENT, MAX_STAKE), 2)
        if losses >= 2:
            stake = round(stake * 0.5, 2)

        if stake > STAKE_CAP:
            stake = STAKE_CAP

        success = await place_trade(direction, stake)
        if success:
            log_trade(direction, current_price, stake)
            send_telegram(f"✅ Trade executed: {direction.upper()} | Price: {current_price:.2f} | Stake: ${stake:.2f}")
        else:
            send_telegram(f"❌ Trade failed. Direction: {direction} | Stake: ${stake:.2f}")
    except Exception as e:
        send_telegram(f"❌ Trade error: {str(e)}")

async def main_loop():
    send_telegram("🚀 ML Gold Bot with Filters Running")
    while True:
        await trade_cycle()
        await asyncio.sleep(TRADE_INTERVAL)

if __name__ == '__main__':
    asyncio.run(main_loop())
