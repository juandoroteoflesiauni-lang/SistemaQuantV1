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
import google.generativeai as genai
from fpdf import FPDF
import base64

# --- CONFIGURACIÓN ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sistema Quant V78 (The Technician)", layout="wide", page_icon="🕯️")

st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;}
    .pattern-card {background-color: #1a2634; border: 1px solid #FFD700; padding: 10px; border-radius: 5px; margin-bottom: 5px;}
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

# --- MOTOR DE PATRONES DE VELAS (NUEVO V78) ---
def detectar_patrones_avanzados(df):
    """Detecta Martillos, Estrellas Fugaces, Dojis y Engulfing"""
    if df.empty: return df
    
    # Cálculos Geométricos
    df['Cuerpo'] = abs(df['Close'] - df['Open'])
    df['Mecha_Sup'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['Mecha_Inf'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    df['Cuerpo_Prom'] = df['Cuerpo'].rolling(10).mean()
    
    # 1. DOJI (Cuerpo minúsculo)
    df['Patron_Doji'] = df['Cuerpo'] <= (df['High'] - df['Low']) * 0.1
    
    # 2. MARTILLO (HAMMER) - Alcista
    # Mecha inferior > 2x Cuerpo | Mecha superior pequeña
    df['Patron_Martillo'] = (df['Mecha_Inf'] > 2 * df['Cuerpo']) & \
                            (df['Mecha_Sup'] < 0.5 * df['Cuerpo']) & \
                            (df['Close'] < df['Close'].shift(3)) # Contexto: Viene bajando
                            
    # 3. ESTRELLA FUGAZ (SHOOTING STAR) - Bajista
    # Mecha superior > 2x Cuerpo | Mecha inferior pequeña
    df['Patron_ShootingStar'] = (df['Mecha_Sup'] > 2 * df['Cuerpo']) & \
                                (df['Mecha_Inf'] < 0.5 * df['Cuerpo']) & \
                                (df['Close'] > df['Close'].shift(3)) # Contexto: Viene subiendo

    # 4. BULLISH ENGULFING (Envolvente Alcista)
    df['Patron_BullEng'] = (df['Close'] > df['Open']) & \
                           (df['Close'].shift(1) < df['Open'].shift(1)) & \
                           (df['Close'] > df['Open'].shift(1)) & \
                           (df['Open'] < df['Close'].shift(1))

    return df

@st.cache_data(ttl=1800)
def escanear_patrones_hoy(tickers):
    """Escanea toda la lista buscando patrones activos HOY"""
    alertas = []
    try:
        data = yf.download(" ".join(tickers), period="30d", group_by='ticker', progress=False, auto_adjust=True)
    except: return []
    
    for t in tickers:
        try:
            df = data[t].dropna() if len(tickers)>1 else data.dropna()
            if df.empty: continue
            
            df = detectar_patrones_avanzados(df)
            last = df.iloc[-1]
            
            if last['Patron_Martillo']: alertas.append({"Ticker": t, "Patron": "🔨 Martillo (Rebote)", "Tipo": "Alcista"})
            if last['Patron_BullEng']: alertas.append({"Ticker": t, "Patron": "🔼 Envolvente Alcista", "Tipo": "Alcista"})
            if last['Patron_ShootingStar']: alertas.append({"Ticker": t, "Patron": "💫 Estrella Fugaz (Caída)", "Tipo": "Bajista"})
            if last['Patron_Doji']: alertas.append({"Ticker": t, "Patron": "➖ Doji (Indecisión)", "Tipo": "Neutral"})
            
        except: pass
    return alertas

# --- GRÁFICO TÉCNICO V78 ---
def graficar_pro_v78(ticker):
    try:
        df = yf.Ticker(ticker).history(period="6mo", auto_adjust=True)
        if df.empty: return None
        
        # Procesar patrones
        df = detectar_patrones_avanzados(df)
        df['SMA50'] = ta.sma(df['Close'], 50)
        df['SMA200'] = ta.sma(df['Close'], 200)
        
        # Filtrar puntos para graficar (solo True)
        martillos = df[df['Patron_Martillo']]
        estrellas = df[df['Patron_ShootingStar']]
        dojis = df[df['Patron_Doji']]
        bull_eng = df[df['Patron_BullEng']]
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        
        # Velas
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Precio'), row=1, col=1)
        
        # Medias
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='cyan', width=1), name='SMA 50'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], line=dict(color='yellow', width=1), name='SMA 200'), row=1, col=1)
        
        # --- MARCADORES DE PATRONES ---
        # Martillo (Verde Diamante Debajo)
        fig.add_trace(go.Scatter(x=martillos.index, y=martillos['Low']*0.98, mode='markers', marker=dict(symbol='diamond', size=10, color='#00ff00'), name='Martillo'), row=1, col=1)
        
        # Estrella Fugaz (Rojo Diamante Arriba)
        fig.add_trace(go.Scatter(x=estrellas.index, y=estrellas['High']*1.02, mode='markers', marker=dict(symbol='diamond', size=10, color='#ff0000'), name='Estrella Fugaz'), row=1, col=1)
        
        # Bullish Engulfing (Triángulo Arriba)
        fig.add_trace(go.Scatter(x=bull_eng.index, y=bull_eng['Low']*0.97, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00cc96'), name='Bullish Engulfing'), row=1, col=1)
        
        # Doji (Cruz Amarilla)
        fig.add_trace(go.Scatter(x=dojis.index, y=dojis['High']*1.01, mode='markers', marker=dict(symbol='cross', size=8, color='yellow'), name='Doji'), row=1, col=1)
        
        # Volumen
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='rgba(100, 100, 100, 0.5)', name='Volumen'), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=550, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        return fig
    except: return None

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

# --- INTERFAZ V78 ---
c1, c2 = st.columns([3, 1])
with c1: st.title("🕯️ Quant Terminal V78: The Technician")
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

# --- SIDEBAR: SCANNER DE PATRONES (NUEVO V78) ---
with st.sidebar:
    st.header("🔍 Scanner de Patrones")
    st.caption("Detectando Martillos, Dojis y Estrellas Fugaces hoy...")
    if st.button("🔄 ESCANEAR VELAS"):
        with st.spinner("Analizando geometría de velas..."):
            patrones_hoy = escanear_patrones_hoy(WATCHLIST)
            if patrones_hoy:
                for p in patrones_hoy:
                    color = "bull-pattern" if "Alcista" in p['Tipo'] else "bear-pattern" if "Bajista" in p['Tipo'] else "white"
                    st.markdown(f"""
                    <div class='pattern-card'>
                        <b>{p['Ticker']}</b>
                        <br><span class='{color}'>{p['Patron']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Ningún patrón relevante detectado al cierre de hoy.")

col_main, col_side = st.columns([2, 1])

with col_main:
    # GRÁFICO TÉCNICO AVANZADO V78
    st.subheader("📉 Gráfico Técnico + Patrones")
    st.caption("Leyenda: 🟢 Martillo | 🔴 Estrella Fugaz | ➕ Doji | ▲ Bullish Engulfing")
    fig_chart = graficar_pro_v78(sel_ticker)
    if fig_chart: st.plotly_chart(fig_chart, use_container_width=True)
    
    tabs_detail = st.tabs(["🦈 Institucionales", "📝 IA"])
    
    with tabs_detail[0]:
        insider = obtener_datos_insider(sel_ticker)
        if insider:
            c1, c2 = st.columns(2)
            c1.metric("Institucional", f"{insider['Institucional']:.1f}%")
            c2.metric("Short Float", f"{insider['Short_Float']:.2f}%", delta_color="inverse")
            if insider['Short_Float'] > 5: st.warning("⚠️ Alta cantidad de apuestas en contra (Shorts).")
        else: st.info("Datos no disponibles.")

    with tabs_detail[1]:
        if st.button("Generar Informe IA"):
            try: st.write(model.generate_content(f"Analisis tecnico de velas para {sel_ticker} hoy").text)
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
