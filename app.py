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
import feedparser 
from datetime import datetime
from scipy.signal import argrelextrema 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import minimize 
from scipy.stats import norm 
import google.generativeai as genai
from fpdf import FPDF 

# --- CONFIGURACIÓN E INICIOS ---
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    HAVE_PYPFOPT = True
except ImportError:
    HAVE_PYPFOPT = False

warnings.filterwarnings('ignore')

try:
    secrets_path = ".streamlit/secrets.toml"
    if os.path.exists(secrets_path):
        secrets = toml.load(secrets_path)
        GOOGLE_API_KEY = secrets["GOOGLE_API_KEY"]
    else:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

st.set_page_config(page_title="Sistema Quant V51 (The Judge)", layout="wide", page_icon="⚖️")
st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 15px; text-align: center;}
    .score-card {background-color: #1c1c1c; border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; text-align: center;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #0e1117; border-radius: 5px;}
    .stTabs [aria-selected="true"] {background-color: #262730;}
</style>""", unsafe_allow_html=True)

WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'COIN', 'KO', 'DIS', 'SPY', 'QQQ', 'DIA', 'GLD', 'USO']
DB_NAME = "quant_database.db"

# --- MOTOR SQL ---
def init_db():
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, ticker TEXT, tipo TEXT, cantidad INTEGER, precio REAL, total REAL)''')
    conn.commit(); conn.close()

def registrar_operacion_sql(ticker, tipo, cantidad, precio):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S"); total = cantidad * precio
    c.execute("INSERT INTO trades (fecha, ticker, tipo, cantidad, precio, total) VALUES (?, ?, ?, ?, ?, ?)", (fecha, ticker, tipo, cantidad, precio, total))
    conn.commit(); conn.close(); return True

def auditar_posiciones_sql():
    empty_df = pd.DataFrame(columns=["Ticker", "Cantidad", "Precio Prom.", "Precio Actual", "Valor Mercado", "P&L ($)", "P&L (%)"])
    conn = sqlite3.connect(DB_NAME)
    try: df = pd.read_sql_query("SELECT * FROM trades", conn)
    except: return empty_df
    conn.close()
    if df.empty: return empty_df
    pos = {}
    for idx, row in df.iterrows():
        t = row['ticker']
        if t not in pos: pos[t] = {"Cantidad": 0, "Costo_Total": 0}
        if row['tipo'] == "COMPRA": pos[t]["Cantidad"] += row['cantidad']; pos[t]["Costo_Total"] += row['total']
        elif row['tipo'] == "VENTA":
            pos[t]["Cantidad"] -= row['cantidad']
            if pos[t]["Cantidad"] > 0: unit = pos[t]["Costo_Total"]/(pos[t]["Cantidad"]+row['cantidad']); pos[t]["Costo_Total"] -= (unit*row['cantidad'])
            else: pos[t]["Costo_Total"] = 0
    act = [t for t, d in pos.items() if d['Cantidad'] > 0]
    if not act: return empty_df
    try: curr = yf.download(" ".join(act), period="1d", progress=False, auto_adjust=True)['Close']
    except: return empty_df
    res = []
    for t, d in pos.items():
        if d['Cantidad'] > 0:
            try:
                if len(act) == 1: price = float(curr.iloc[-1])
                else: price = float(curr.iloc[-1][t])
                val = d['Cantidad']*price; pnl = val - d['Costo_Total']
                res.append({"Ticker": t, "Cantidad": d['Cantidad'], "Precio Prom.": d['Costo_Total']/d['Cantidad'], "Precio Actual": price, "Valor Mercado": val, "P&L ($)": pnl, "P&L (%)": (pnl/d['Costo_Total'])*100})
            except: pass
    return pd.DataFrame(res) if res else empty_df

init_db()

# --- MOTOR DE SCORING (THE JUDGE V51) ---
@st.cache_data(ttl=600)
def calcular_score_quant(ticker):
    """Calcula una nota del 0 al 100 para el activo"""
    score = 0
    breakdown = {"Técnico": 0, "Fundamental": 0, "Riesgo": 0}
    
    try:
        # 1. DATOS TÉCNICOS (40 pts)
        df = yf.Ticker(ticker).history(period="6mo", interval="1d", auto_adjust=True)
        if not df.empty:
            df['RSI'] = ta.rsi(df['Close'], 14)
            df['SMA50'] = ta.sma(df['Close'], 50)
            
            last_rsi = df['RSI'].iloc[-1]
            last_price = df['Close'].iloc[-1]
            last_sma = df['SMA50'].iloc[-1]
            
            # RSI Score (0-20 pts)
            if 30 <= last_rsi <= 65: breakdown['Técnico'] += 20 # Zona saludable
            elif last_rsi < 30: breakdown['Técnico'] += 15 # Rebote probable (pero riesgoso)
            elif last_rsi > 70: breakdown['Técnico'] += 5 # Sobrecompra (peligro)
            
            # Tendencia Score (0-20 pts)
            if last_price > last_sma: breakdown['Técnico'] += 20 # Tendencia alcista
            else: breakdown['Técnico'] += 0 # Tendencia bajista
            
        # 2. DATOS FUNDAMENTALES (40 pts)
        info = yf.Ticker(ticker).info
        eps = info.get('trailingEps', 0)
        pe = info.get('trailingPE', 0)
        book = info.get('bookValue', 0)
        price = info.get('currentPrice', df['Close'].iloc[-1])
        
        # Graham Score (0-20 pts)
        if eps and book and eps > 0 and book > 0:
            graham = math.sqrt(22.5 * eps * book)
            if price < graham: breakdown['Fundamental'] += 20 # Subvaluada
            elif price < graham * 1.2: breakdown['Fundamental'] += 10 # Precio justo
            else: breakdown['Fundamental'] += 0 # Cara
            
        # Rentabilidad Score (0-20 pts)
        if eps > 0: breakdown['Fundamental'] += 20 # Empresa rentable
        
        # Caso Crypto/ETF (Sin fundamentales clásicos)
        if not eps and "USD" in ticker: # Ajuste para Crypto
             breakdown['Fundamental'] = 20 # Asumimos neutral
             
        # 3. RIESGO (20 pts)
        beta = info.get('beta', 1.0)
        if beta is None: beta = 1.0
        
        # Preferimos Betas cercanos a 1 o menores. Betas muy altos restan.
        if 0.8 <= beta <= 1.2: breakdown['Riesgo'] += 20 # Riesgo Mercado
        elif beta < 0.8: breakdown['Riesgo'] += 15 # Defensiva
        elif beta > 1.5: breakdown['Riesgo'] += 5 # Muy volátil
        else: breakdown['Riesgo'] += 10
        
        score = breakdown['Técnico'] + breakdown['Fundamental'] + breakdown['Riesgo']
        return score, breakdown
        
    except Exception as e:
        return 0, breakdown

def dibujar_velocimetro(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Quant Score (0-100)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "white"},
            'steps': [
                {'range': [0, 40], 'color': "#ff4b4b"}, # Rojo
                {'range': [40, 70], 'color': "#ffa500"}, # Naranja
                {'range': [70, 100], 'color': "#00cc96"}], # Verde
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="#