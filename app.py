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
import requests 
from datetime import datetime
from scipy.stats import norm 
import google.generativeai as genai

# --- CONFIGURACIÓN ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sistema Quant V63 (The Terminal)", layout="wide", page_icon="🖥️")

# Estilos CSS tipo Terminal
st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;}
    .factor-card {background-color: #1c1c1c; border: 1px solid #4CAF50; border-radius: 8px; padding: 15px;}
    .big-number {font-size: 24px; font-weight: bold; color: #ffffff;}
    .sub-text {font-size: 12px; color: #888;}
    .stTabs [data-baseweb="tab-list"] {gap: 5px;}
    .stTabs [data-baseweb="tab"] {height: 40px; padding: 5px 15px; font-size: 14px;}
</style>""", unsafe_allow_html=True)

# CREDENCIALES
try:
    secrets = toml.load(".streamlit/secrets.toml") if os.path.exists(".streamlit/secrets.toml") else st.secrets
    genai.configure(api_key=secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'COIN', 'KO', 'DIS', 'SPY', 'QQQ', 'GLD', 'USO']
DB_NAME = "quant_database.db"

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
    # Descarga precios actuales solo para activos en cartera
    activos = [t for t, d in pos.items() if d['Qty'] > 0]
    if not activos: return pd.DataFrame()
    
    try: curr = yf.download(" ".join(activos), period="1d", progress=False, auto_adjust=True)['Close']
    except: return pd.DataFrame()

    for t in activos:
        d = pos[t]
        try:
            px = float(curr.iloc[-1]) if len(activos) == 1 else float(curr.iloc[-1][t])
            val = d['Qty'] * px; pnl = val - d['Cost']
            res.append({"Ticker": t, "Cantidad": d['Qty'], "Valor": val, "P&L": pnl})
        except: pass
    return pd.DataFrame(res)

init_db()

# --- MOTOR DE FACTORES QUANT (NUEVO V63) ---
@st.cache_data(ttl=600)
def calcular_factores_quant(ticker):
    """Calcula puntajes (0-100) para los 5 Factores Clave"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y", interval="1d", auto_adjust=True)
        info = stock.info
        
        if df.empty: return None
        
        # 1. VALUE (¿Está barata?)
        pe = info.get('trailingPE', 50)
        pb = info.get('priceToBook', 10)
        # Menor PE es mejor. Si PE < 15 -> Score 100. Si PE > 60 -> Score 0
        score_value = max(0, min(100, (60 - pe) * 2)) if pe > 0 else 0
        
        # 2. GROWTH (Crecimiento)
        rev_growth = info.get('revenueGrowth', 0) * 100 # %
        # Si crece > 30% -> Score 100
        score_growth = max(0, min(100, rev_growth * 3.3))
        
        # 3. MOMENTUM (Tendencia)
        curr = df['Close'].iloc[-1]
        sma200 = df['Close'].rolling(200).mean().iloc[-1]
        rsi = ta.rsi(df['Close'], 14).iloc[-1]
        # Precio > SMA200 da 50pts. RSI > 50 da puntos extra.
        m_score = 0
        if curr > sma200: m_score += 50
        if rsi > 50: m_score += (rsi - 50) * 2
        score_momentum = max(0, min(100, m_score))
        
        # 4. QUALITY (Rentabilidad)
        roe = info.get('returnOnEquity', 0) * 100
        margins = info.get('profitMargins', 0) * 100
        # ROE > 20% es excelente
        score_quality = max(0, min(100, (roe * 2) + margins))
        
        # 5. LOW VOLATILITY (Seguridad)
        beta = info.get('beta', 1.5)
        if beta is None: beta = 1.0
        # Beta bajo (0.5) es mejor para este factor. Beta alto (2.0) es malo.
        # Formula inversa: Beta 0.5 -> 100, Beta 2.0 -> 0
        score_vol = max(0, min(100, (2 - beta) * 100))
        
        return {
            "Value": score_value,
            "Growth": score_growth,
            "Momentum": score_momentum,
            "Quality": score_quality,
            "Low Vol": score_vol
        }
    except: return None

def dibujar_radar_factores(scores):
    """Dibuja el gráfico de araña"""
    categories = list(scores.keys())
    values = list(scores.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Perfil Quant',
        line_color='#00ff00',
        fillcolor='rgba(0, 255, 0, 0.2)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], color='grey')),
        showlegend=False,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color='white'),
        height=300,
        margin=dict(l=40, r=40, t=20, b=20)
    )
    return fig

# --- MOTORES DE SOPORTE (RESUMIDOS) ---
def graficar_simple(ticker):
    df = yf.Ticker(ticker).history(period="6mo", auto_adjust=True)
    if df.empty: return None
    df['SMA50'] = ta.sma(df['Close'], 50)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Precio'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='yellow'), name='SMA 50'))
    fig.update_layout(template="plotly_dark", height=350, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
    return fig

def calcular_dcf_rapido(ticker):
    try:
        i = yf.Ticker(ticker).info
        fcf = i.get('freeCashflow', i.get('operatingCashflow', 0)*0.8)
        if fcf <= 0: return None
        growth = 0.10; wacc = 0.09; shares = i.get('sharesOutstanding', 1)
        # DCF Simplificado 5 años
        pv = 0
        for y in range(1, 6): pv += (fcf * ((1+growth)**y)) / ((1+wacc)**y)
        term = (fcf * ((1+growth)**5) * 1.02) / (wacc - 0.02)
        pv_term = term / ((1+wacc)**5)
        val = (pv + pv_term) / shares
        return val
    except: return None

# --- INTERFAZ V63: THE TERMINAL ---
# HEADER
c1, c2 = st.columns([3, 1])
with c1: st.title("🖥️ Quant Terminal V63")
with c2: 
    # Selector Global
    sel_ticker = st.selectbox("ACTIVO PRINCIPAL", WATCHLIST)

# --- FILA 1: SNAPSHOT ---
# Datos en tiempo real
stock = yf.Ticker(sel_ticker)
hist = stock.history(period="2d")
info = stock.info

if not hist.empty:
    curr = hist['Close'].iloc[-1]
    prev = hist['Close'].iloc[-2]
    delta = ((curr - prev) / prev) * 100
    
    col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
    col_k1.metric("Precio", f"${curr:.2f}", f"{delta:+.2f}%")
    col_k2.metric("RSI (14)", f"{ta.rsi(stock.history(period='30d')['Close'], 14).iloc[-1]:.0f}")
    col_k3.metric("Volumen", f"{info.get('volume', 0)/1e6:.1f}M")
    col_k4.metric("Beta", f"{info.get('beta', 1.0):.2f}")
    col_k5.metric("Target Analistas", f"${info.get('targetMeanPrice', 0):.2f}")

st.divider()

# --- FILA 2: ANÁLISIS PROFUNDO (LAYOUT FACTORES) ---
col_main, col_side = st.columns([2, 1])

with col_main:
    # GRÁFICO CENTRAL
    st.subheader("📉 Acción del Precio")
    fig_chart = graficar_simple(sel_ticker)
    if fig_chart: st.plotly_chart(fig_chart, use_container_width=True)
    
    # PESTAÑAS DE DETALLE (Integrando los motores anteriores)
    tabs_detail = st.tabs(["🧮 Valuación (DCF)", "📰 Noticias & IA", "🔍 Datos Clave"])
    
    with tabs_detail[0]:
        dcf_val = calcular_dcf_rapido(sel_ticker)
        if dcf_val:
            st.metric("Valor Intrínseco (Modelo DCF)", f"${dcf_val:.2f}", f"{((dcf_val-curr)/curr)*100:+.1f}% Upside")
            if dcf_val > curr: st.success("El activo cotiza con DESCUENTO según flujos de caja futuros.")
            else: st.warning("El activo cotiza con PRIMA sobre sus flujos de caja.")
        else: st.info("No aplica modelo DCF (Cripto o Sin Ganancias).")
        
    with tabs_detail[1]:
        if st.button("🤖 Analizar Noticias"):
            with st.spinner("Leyendo..."):
                prompt = f"Dame 3 razones BULLISH y 3 razones BEARISH para {sel_ticker} hoy. Sé breve."
                res = model.generate_content(prompt).text
                st.markdown(res)
                
    with tabs_detail[2]:
        c_d1, c_d2 = st.columns(2)
        c_d1.write(f"**Sector:** {info.get('sector', 'N/A')}")
        c_d1.write(f"**Industria:** {info.get('industry', 'N/A')}")
        c_d2.write(f"**Market Cap:** ${info.get('marketCap', 0)/1e9:.1f}B")
        c_d2.write(f"**Dividend Yield:** {info.get('dividendYield', 0)*100:.2f}%")

with col_side:
    # RADAR CHART (LA JOYA DE LA V63)
    st.subheader("🧬 Perfil Quant (Factores)")
    factores = calcular_factores_quant(sel_ticker)
    
    if factores:
        fig_radar = dibujar_radar_factores(factores)
        st.plotly_chart(fig_radar, use_container_width=True)
        
        st.caption("Interpretación:")
        st.write(f"**💎 Quality:** {factores['Quality']}/100")
        st.write(f"**🚀 Momentum:** {factores['Momentum']}/100")
        st.write(f"**🏷️ Value:** {factores['Value']}/100")
        
    else: st.warning("Calculando factores...")
    
    # PANEL DE OPERACIÓN RÁPIDA
    st.markdown("---")
    st.subheader("⚡ Quick Trade")
    with st.form("quick_order"):
        q_qty = st.number_input("Cantidad", 1, 1000, 10)
        q_side = st.selectbox("Lado", ["COMPRA", "VENTA"])
        if st.form_submit_button("EJECUTAR"):
            registrar_operacion_sql(sel_ticker, q_side, q_qty, curr)
            st.success("Orden Enviada!")

# --- FILA 3: PORTAFOLIO RESUMEN ---
st.markdown("---")
st.subheader("💼 Resumen de Cartera")
df_p = auditar_posiciones_sql()
if not df_p.empty:
    st.dataframe(df_p.style.format({"Valor": "${:.2f}", "P&L": "${:+.2f}"}), use_container_width=True)
