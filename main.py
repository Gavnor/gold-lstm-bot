import os
import requests
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import asyncio
import websockets
import json
import socket
from scipy.stats import pearsonr
import joblib

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")

MODEL_PATH = 'model/enhanced_gold_lstm.h5'
SCALER_PATH = 'model/scaler.save'
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

def fetch_multi_timeframe_data():
    timeframes = [
        ('1h', 48),
        ('4h', 24),
        ('1d', 30)
    ]
    all_data = []
    for tf, size in timeframes:
        url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval={tf}&outputsize={size}&apikey={TWELVE_API_KEY}"
        try:
            data = requests.get(url, timeout=10).json()
            if 'values' in data:
                df = pd.DataFrame(data['values'])[::-1]
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['price'] = df['close'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['timeframe'] = tf
                all_data.append(df)
        except Exception as e:
            send_telegram_message(f"⚠️ Failed {tf} data: {str(e)}")
    return pd.concat(all_data) if all_data else None

def fetch_correlation_data():
    symbols = {
        'DXY': 'US Dollar Index',
        'TNX': '10-Year Treasury Yield'
    }
    corr_data = {}
    for symbol in symbols:
        try:
            url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1h&outputsize=48&apikey={TWELVE_API_KEY}"
            data = requests.get(url, timeout=10).json()
            if 'values' in data:
                df = pd.DataFrame(data['values'])[::-1]
                df['close'] = df['close'].astype(float)
                corr_data[symbol] = df['close'].values
        except Exception as e:
            send_telegram_message(f"⚠️ Failed {symbol} data: {str(e)}")
    return corr_data

def add_features(df):
    df['rsi'] = ta.rsi(df['price'], length=14)
    macd = ta.macd(df['price'])
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['atr'] = ta.atr(df['high'], df['low'], df['price'], length=14)
    df['adx'] = ta.adx(df['high'], df['low'], df['price'], length=14)['ADX_14']
    df['ema20'] = ta.ema(df['price'], length=20)
    corr_data = fetch_correlation_data()
    for symbol, values in corr_data.items():
        if len(values) == len(df):
            rolling_corr = pd.Series(df['price']).rolling(window=20).apply(lambda x: pearsonr(x, values[-len(x):])[0])
            df[f'corr_{symbol.lower()}'] = rolling_corr
    for tf in ['4h', '1d']:
        tf_df = df[df['timeframe'] == tf]
        if not tf_df.empty:
            df[f'{tf}_rsi'] = tf_df['rsi'].iloc[-1]
            df[f'{tf}_atr'] = tf_df['atr'].iloc[-1]
    return df.dropna()

def build_advanced_model(input_shape):
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

def prepare_enhanced_data(df):
    df = add_features(df)
    scaler = MinMaxScaler()
    features = ['price', 'rsi', 'macd', 'atr', 'adx', 'corr_dxy', 'corr_tnx', '4h_rsi', '1d_rsi']
    scaled = scaler.fit_transform(df[features])
    X, y = [], []
    window_size = 24
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i])
        y.append(scaled[i, 0])
    joblib.dump(scaler, SCALER_PATH)
    return np.array(X), np.array(y), scaler

def is_strong_signal(df, current_price, predicted_price):
    last = df.iloc[-1]
    if abs(last['corr_dxy']) > CORRELATION_THRESHOLD and last['corr_dxy'] * (predicted_price - current_price) > 0:
        return False
    if (last['4h_rsi'] > 70 and predicted_price > current_price) or \
       (last['4h_rsi'] < 30 and predicted_price < current_price):
        return False
    if last['atr'] > (df['atr'].mean() * VOLATILITY_THRESHOLD):
        return False
    return True

async def execute_day_trade(current_price, predicted_price, df, balance):
    if not is_strong_signal(df, current_price, predicted_price):
        send_telegram_message("⏭️ Skipping - weak signal confirmation")
        return False
    volatility_ratio = min(df['atr'].iloc[-1] / df['atr'].mean(), 2.0)
    stake = round(min(MAX_STAKE_PERCENT * balance * volatility_ratio, MAX_STAKE), 2)
    contract_type = "CALL" if predicted_price > current_price else "PUT"
    trade_result = await place_trade(contract_type, stake)
    if trade_result:
        send_telegram_message(
            f"✅ DAY TRADE ENTRY\n"
            f"Type: {contract_type}\n"
            f"Current: {round(current_price, 2)}\n"
            f"Predicted: {round(predicted_price, 2)}\n"
            f"Stake: ${stake}\n"
            f"Stop: -{STOP_LOSS_PCT*100}% | Take: +{TAKE_PROFIT_PCT*100}%\n"
            f"Correlation: DXY={df['corr_dxy'].iloc[-1]:.2f}\n"
            f"4H RSI: {df['4h_rsi'].iloc[-1]:.1f}"
        )
        return True
    return False

async def main_loop():
    send_telegram_message("🚀 Advanced Gold Trader Started")
    model = None
    while True:
        try:
            df = fetch_multi_timeframe_data()
            if df is None:
                await asyncio.sleep(600)
                continue
            X, y, scaler = prepare_enhanced_data(df)
            if model is None:
                try:
                    model = load_model(MODEL_PATH)
                except:
                    model = build_advanced_model((X.shape[1], X.shape[2]))
                    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
                    model.save(MODEL_PATH)
            else:
                model.fit(X, y, epochs=1, batch_size=8, verbose=0)
                model.save(MODEL_PATH)
            current_price = df['price'].iloc[-1]
            predicted_price = scaler.inverse_transform(
                model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]))
            )[0][0]
            balance = await get_balance()
            if balance >= MIN_BALANCE:
                await execute_day_trade(current_price, predicted_price, df, balance)
            await asyncio.sleep(7200)
        except Exception as e:
            send_telegram_message(f"🚨 Critical error: {str(e)}")
            await asyncio.sleep(600)

if __name__ == '__main__':
    asyncio.run(main_loop())
