import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px 
from plotly.subplots import make_subplots 
import warnings
import numpy as np
import os
import toml
import sqlite3
import math
import time
import requests 
from datetime import datetime, timedelta
from scipy.stats import norm 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import google.generativeai as genai
from fpdf import FPDF
import base64

# --- CONFIGURACIÓN ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sistema Quant V80 (The Sentiment)", layout="wide", page_icon="📰")

st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;}
    .news-card {background-color: #1e202e; border-left: 3px solid #FFD700; padding: 15px; margin-bottom: 10px; border-radius: 5px;}
    .sentiment-pos {color: #00ff00; font-weight: bold;}
    .sentiment-neg {color: #ff4b4b; font-weight: bold;}
    .stTabs [data-baseweb="tab-list"] {gap: 5px;}
    .stTabs [data-baseweb="tab"] {height: 40px; padding: 5px 15px; font-size: 14px;}
</style>""", unsafe_allow_html=True)

try:
    secrets = toml.load(".streamlit/secrets.toml") if os.path.exists(".streamlit/secrets.toml") else st.secrets
    genai.configure(api_key=secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'COIN', 'KO', 'DIS', 'SPY', 'QQQ', 'GLD', 'USO']
DB_NAME = "quant_database.db"

# --- MOTOR NLP DE SENTIMIENTO (NUEVO V80) ---
def analizar_sentimiento_noticias(ticker):
    """Descarga noticias y calcula un score de sentimiento básico"""
    if "USD" in ticker: return None # Las noticias de crypto en yfinance a veces fallan
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news: return None
        
        # Diccionarios de Sentimiento (Simple pero efectivo para velocidad)
        bullish_words = ['up', 'rise', 'jump', 'gain', 'buy', 'bull', 'profit', 'beat', 'growth', 'surge', 'record', 'high', 'strong']
        bearish_words = ['down', 'drop', 'fall', 'loss', 'sell', 'bear', 'miss', 'risk', 'crash', 'weak', 'lower', 'cut', 'lawsuit']
        
        total_score = 0
        news_processed = []
        
        for n in news[:5]: # Analizar las últimas 5 noticias
            title = n['title']
            link = n['link']
            publisher = n['publisher']
            # Convertir timestamp
            try: pub_time = datetime.fromtimestamp(n['providerPublishTime']).strftime('%Y-%m-%d %H:%M')
            except: pub_time = "Reciente"
            
            # Análisis NLP Básico
            title_lower = title.lower()
            score = 0
            for w in bullish_words: 
                if w in title_lower: score += 1
            for w in bearish_words: 
                if w in title_lower: score -= 1
            
            total_score += score
            
            # Etiqueta
            sentiment_label = "Positivo 🟢" if score > 0 else "Negativo 🔴" if score < 0 else "Neutral ⚪"
            
            news_processed.append({
                "Titulo": title,
                "Fuente": publisher,
                "Hora": pub_time,
                "Link": link,
                "Sentimiento": sentiment_label,
                "Score": score
            })
            
        # Normalizar Score Final (-10 a +10 aprox)
        final_sentiment = "NEUTRAL"
        if total_score >= 2: final_sentiment = "ALCISTA (Bullish) 🐂"
        elif total_score <= -2: final_sentiment = "BAJISTA (Bearish) 🐻"
        
        return {
            "Noticias": news_processed,
            "Score_Total": total_score,
            "Sentimiento_Global": final_sentiment
        }
            
    except Exception as e: return None

# --- MOTORES EXISTENTES (V79) ---
def entrenar_modelo_ml(ticker):
    try:
        df = yf.Ticker(ticker).history(period="2y")
        if len(df) < 200: return None
        df['RSI'] = ta.rsi(df['Close'], 14); df['SMA_Diff'] = (df['Close'] - ta.sma(df['Close'], 50)) / ta.sma(df['Close'], 50)
        df['Vol_Change'] = df['Volume'].pct_change(); df['Return'] = df['Close'].pct_change()
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        X = df[['RSI', 'SMA_Diff', 'Vol_Change', 'Return']]; y = df['Target']
        split = int(len(df) * 0.8)
        clf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        clf.fit(X.iloc[:split], y.iloc[:split])
        acc = accuracy_score(y.iloc[split:], clf.predict(X.iloc[split:]))
        pred = clf.predict(X.iloc[[-1]])[0]
        prob = clf.predict_proba(X.iloc[[-1]])[0][pred]
        return {"Prediccion": "SUBE 🟢" if pred == 1 else "BAJA 🔴", "Accuracy": acc * 100, "Probabilidad": prob * 100}
    except: return None

@st.cache_data(ttl=1800)
def obtener_datos_snapshot(ticker):
    try:
        stock = yf.Ticker(ticker); hist = stock.history(period="5d")
        if hist.empty: return None
        try: info = stock.info
        except: info = {}
        return {"Precio": hist['Close'].iloc[-1], "Previo": hist['Close'].iloc[-2], "RSI": ta.rsi(hist['Close'], 14).iloc[-1] if len(hist)>14 else 50, "Volumen": info.get('volume', 0), "Beta": info.get('beta', 1.0), "Target": info.get('targetMeanPrice', 0)}
    except: return None

def detectar_patrones_avanzados(df):
    if df.empty: return df
    df['Cuerpo'] = abs(df['Close'] - df['Open'])
    df['Mecha_Sup'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['Mecha_Inf'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    df['Patron_Martillo'] = (df['Mecha_Inf'] > 2 * df['Cuerpo']) & (df['Mecha_Sup'] < 0.5 * df['Cuerpo'])
    df['Patron_BullEng'] = (df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)) & (df['Close'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1))
    return df

def graficar_pro_v78(ticker):
    try:
        df = yf.Ticker(ticker).history(period="6mo", auto_adjust=True)
        if df.empty: return None
        df = detectar_patrones_avanzados(df)
        df['SMA50'] = ta.sma(df['Close'], 50); df['SMA200'] = ta.sma(df['Close'], 200)
        martillos = df[df['Patron_Martillo']]
        bull_eng = df[df['Patron_BullEng']]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Precio'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='cyan', width=1), name='SMA 50'), row=1, col=1)
        fig.add_trace(go.Scatter(x=martillos.index, y=martillos['Low']*0.98, mode='markers', marker=dict(symbol='diamond', size=10, color='#00ff00'), name='Martillo'), row=1, col=1)
        fig.add_trace(go.Scatter(x=bull_eng.index, y=bull_eng['Low']*0.97, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00cc96'), name='Bullish Engulfing'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='rgba(100, 100, 100, 0.5)', name='Volumen'), row=2, col=1)
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0)); return fig
    except: return None

@st.cache_data(ttl=3600)
def obtener_datos_insider(ticker):
    if "USD" in ticker: return None
    try:
        info = yf.Ticker(ticker).info
        sh_short = info.get('sharesShort', 0); float_sh = info.get('floatShares', 1)
        short_pct = (sh_short/float_sh)*100 if float_sh else 0
        inst_pct = info.get('heldPercentInstitutions', 0)*100
        return {"Institucional": inst_pct, "Short_Float": short_pct}
    except: return None

# --- MOTOR SQL ---
def init_db():
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, ticker TEXT, tipo TEXT, cantidad INTEGER, precio REAL, total REAL)''')
    conn.commit(); conn.close()
def registrar_operacion_sql(t, tipo, q, p):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    total = q * p; fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO trades (fecha, ticker, tipo, cantidad, precio, total) VALUES (?, ?, ?, ?, ?, ?)", (fecha, t, tipo, q, p, total))
    conn.commit(); conn.close()
def auditar_posiciones_sql():
    conn = sqlite3.connect(DB_NAME); df = pd.read_sql_query("SELECT * FROM trades", conn); conn.close()
    if df.empty: return pd.DataFrame()
    pos = {}
    for i, r in df.iterrows():
        t = r['ticker']
        if t not in pos: pos[t] = {"Qty": 0, "Cost": 0}
        if r['tipo'] == "COMPRA": pos[t]["Qty"] += r['cantidad']; pos[t]["Cost"] += r['total']
        elif r['tipo'] == "VENTA": 
            pos[t]["Qty"] -= r['cantidad']
            if pos[t]["Qty"] > 0: unit = pos[t]["Cost"]/(pos[t]["Qty"]+r['cantidad']); pos[t]["Cost"] -= (unit*r['cantidad'])
            else: pos[t]["Cost"] = 0
    res = []
    activos = [t for t, d in pos.items() if d['Qty'] > 0]
    if not activos: return pd.DataFrame()
    try: curr = yf.download(" ".join(activos), period="1d", progress=False, auto_adjust=True)['Close']
    except: return pd.DataFrame()
    for t in activos:
        d = pos[t]
        try:
