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
from sklearn.ensemble import RandomForestClassifier # <--- CEREBRO ML
from sklearn.metrics import accuracy_score
import google.generativeai as genai
from fpdf import FPDF
import base64

# --- CONFIGURACIÓN ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sistema Quant V79 (The Oracle)", layout="wide", page_icon="🤖")

st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;}
    .ml-card {background-color: #1a1a2e; border: 1px solid #9c27b0; padding: 15px; border-radius: 8px; text-align: center;}
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

# --- MOTOR MACHINE LEARNING (NUEVO V79) ---
def entrenar_modelo_ml(ticker):
    """Entrena un Random Forest para predecir si el precio sube o baja mañana"""
    try:
        # 1. Preparar Datos
        df = yf.Ticker(ticker).history(period="2y")
        if len(df) < 200: return None
        
        # 2. Feature Engineering (Variables para que la IA aprenda)
        df['RSI'] = ta.rsi(df['Close'], 14)
        df['SMA_Diff'] = (df['Close'] - ta.sma(df['Close'], 50)) / ta.sma(df['Close'], 50)
        df['Vol_Change'] = df['Volume'].pct_change()
        df['Return'] = df['Close'].pct_change()
        
        # Variable Objetivo (Target): 1 si mañana sube, 0 si baja
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        df = df.dropna()
        
        # 3. Features (X) y Target (y)
        features = ['RSI', 'SMA_Diff', 'Vol_Change', 'Return']
        X = df[features]
        y = df['Target']
        
        # Split Train/Test (Últimos 100 días para testear)
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # 4. Entrenar Modelo
        clf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        clf.fit(X_train, y_train)
        
        # 5. Evaluar Precisión
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # 6. Predicción para MAÑANA
        last_data = X.iloc[[-1]]
        prediction = clf.predict(last_data)[0]
        prob = clf.predict_proba(last_data)[0][prediction] # Certeza del modelo
        
        return {
            "Prediccion": "SUBE 🟢" if prediction == 1 else "BAJA 🔴",
            "Accuracy": acc * 100,
            "Probabilidad": prob * 100,
            "Features": features
        }
        
    except Exception as e: return None

# --- MOTORES EXISTENTES ---
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
            if len(activos) == 1: px = float(curr.iloc[-1])
            else: px = float(curr.iloc[-1][t])
            val = d['Qty'] * px; pnl = val - d['Cost']
            res.append({"Ticker": t, "Cantidad": d['Qty'], "Valor": val, "P&L": pnl})
        except: pass
    return pd.DataFrame(res)
init_db()

# --- INTERFAZ V79 ---
c1, c2 = st.columns([3, 1])
with c1: st.title("🤖 Quant Terminal V79: The Oracle")
with c2: sel_ticker = st.selectbox("ACTIVO PRINCIPAL", WATCHLIST)

snap = obtener_datos_snapshot(sel_ticker)
if snap:
    delta = ((snap['Precio'] - snap['Previo'])/snap['Previo'])*100
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Precio", f"${snap['Precio']:.2f}", f"{delta:+.2f}%")
    k2.metric("RSI", f"{snap['RSI']:.0f}")
    k3.metric("Vol", f"{snap['Volumen']/1e6:.1f}M")
    k4.metric("Beta", f"{snap['Beta']:.2f}")
    k5.metric("Target", f"${snap['Target']:.2f}")

st.divider()

col_main, col_side = st.columns([2, 1])

with col_main:
    st.subheader("📉 Acción del Precio")
    fig_chart = graficar_pro_v78(sel_ticker)
    if fig_chart: st.plotly_chart(fig_chart, use_container_width=True)
    
    tabs_detail = st.tabs(["🤖 ORÁCULO ML", "🦈 Institucionales", "📝 IA"])
    
    # --- TAB 1: MACHINE LEARNING (NUEVO V79) ---
    with tabs_detail[0]:
        st.subheader("🤖 Predicción por Inteligencia Artificial (Random Forest)")
        st.caption("El modelo analiza patrones complejos (RSI, Volumen, Tendencia) de los últimos 2 años.")
        
        if st.button("🧠 ENTRENAR MODELO & PREDECIR"):
            with st.spinner(f"Entrenando red neuronal para {sel_ticker}..."):
                ml_result = entrenar_modelo_ml(sel_ticker)
                
                if ml_result:
                    c_ml1, c_ml2 = st.columns(2)
                    
                    # Tarjeta de Predicción
                    color_pred = "#00ff00" if "SUBE" in ml_result['Prediccion'] else "#ff4b4b"
                    with c_ml1:
                        st.markdown(f"""
                        <div class='ml-card' style='border-color: {color_pred}'>
                            <h4 style='color: #fff'>El Oráculo Predice:</h4>
                            <h1 style='color: {color_pred}'>{ml_result['Prediccion']}</h1>
                            <p>Certeza del Modelo: {ml_result['Probabilidad']:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Tarjeta de Calidad del Modelo
                    with c_ml2:
                        acc = ml_result['Accuracy']
                        color_acc = "green" if acc > 55 else "orange" if acc > 50 else "red"
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h4>Precisión Histórica (Backtest)</h4>
                            <h2 style='color: {color_acc}'>{acc:.1f}%</h2>
                            <small>¿Qué tan bueno es este modelo con {sel_ticker}?</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.info(f"🧠 Variables analizadas por la IA: {', '.join(ml_result['Features'])}")
                    if ml_result['Accuracy'] < 50:
                        st.warning("⚠️ Precaución: La precisión histórica es baja (<50%). El mercado está errático.")
                else:
                    st.error("Datos insuficientes para entrenar ML.")

    with tabs_detail[1]:
        insider = obtener_datos_insider(sel_ticker)
        if insider:
            c1, c2 = st.columns(2)
            c1.metric("Institucional", f"{insider['Institucional']:.1f}%")
            c2.metric("Short Float", f"{insider['Short_Float']:.2f}%", delta_color="inverse")
        else: st.info("Datos no disponibles.")

    with tabs_detail[2]:
        if st.button("Generar Informe IA"):
            try: st.write(model.generate_content(f"Analisis {sel_ticker}").text)
            except: st.error("Error IA")

with col_side:
    st.subheader("⚡ Quick Trade")
    with st.form("quick"):
        q = st.number_input("Qty", 1, 1000, 10); s = st.selectbox("Side", ["COMPRA", "VENTA"])
        if st.form_submit_button("EJECUTAR"): 
            if snap: registrar_operacion_sql(sel_ticker, s, q, snap['Precio']); st.success("Orden OK")
    
    st.subheader("💼 Cartera")
    df_p = auditar_posiciones_sql()
    if not df_p.empty: st.dataframe(df_p[['Ticker', 'P&L']], use_container_width=True)
