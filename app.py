import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests 
import google.generativeai as genai
import feedparser
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# --- 🔐 CREDENCIALES ---
TELEGRAM_TOKEN = "8042406069:AAHhflfkySyQVhCkHaqIsUjGumFr3fsnDPM" 
TELEGRAM_CHAT_ID = "6288094504"
GOOGLE_API_KEY = "AIzaSyB356Wjicaf9VRUYTX6_EL728IQF6nOmuQ" 

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Sistema Quant V24.8 (Macro)", layout="wide", page_icon="🌍")
st.markdown("""
<style>
    .metric-card {background-color: #1e1e1e; border: 1px solid #333; border-radius: 8px; padding: 15px; color: white;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #0e1117; border-radius: 5px; color: white;}
    .stTabs [aria-selected="true"] {background-color: #262730; color: #4CAF50 !important;}
</style>
""", unsafe_allow_html=True)

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

# --- ACTIVOS ---
WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'COIN', 'PLTR']
SECTORS = {
    'XLK': 'Tecnología 💻', 'XLF': 'Finanzas 🏦', 'XLE': 'Energía 🛢️',
    'XLV': 'Salud 🏥', 'XLY': 'Consumo Disc. 🛍️', 'XLP': 'Consumo Bas. 🛒',
    'XLI': 'Industria 🏭', 'GLD': 'Oro 🥇', 'SLV': 'Plata 🥈'
}

st.title("🌍 Sistema Quant V24.8: Macro & Risk")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    capital_total = st.number_input("Capital Total ($)", value=2000)
    riesgo_pct = st.slider("Riesgo por Trade (%)", 0.5, 3.0, 1.5)
    st.divider()
    if st.button("🔄 Refrescar Datos"):
        st.cache_data.clear()
        st.rerun()

# --- MOTORES DE DATOS ---
@st.cache_data(ttl=900) # 15 min cache para macro
def obtener_macro():
    # Descargar SPY (Mercado) y VIX (Miedo)
    tickers = ["SPY", "^VIX"] + list(SECTORS.keys())
    try:
        data = yf.download(tickers, period="6mo", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
    except: return None, None
    
    # Analizar SPY
    spy = data["SPY"].copy().dropna()
    spy['EMA200'] = ta.ema(spy['Close'], 200)
    spy_trend = "ALCISTA 🟢" if spy['Close'].iloc[-1] > spy['EMA200'].iloc[-1] else "BAJISTA 🔴"
    
    # Analizar VIX
    vix = data["^VIX"]['Close'].iloc[-1]
    market_mood = "NORMAL"
    if vix > 30: market_mood = "PÁNICO EXTREMO 😱"
    elif vix > 20: market_mood = "MIEDO 😨"
    elif vix < 15: market_mood = "COMPLACENCIA (Bajo Riesgo) 😎"
    
    # Analizar Sectores (Performance 5 días)
    sec_perf = []
    for ticker, name in SECTORS.items():
        try:
            df = data[ticker].dropna()
            change = (df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100
            sec_perf.append({"Sector": name, "Retorno 1S": change})
        except: continue
        
    return {"spy_trend": spy_trend, "vix": vix, "mood": market_mood}, pd.DataFrame(sec_perf)

@st.cache_data(ttl=300)
def escanear_acciones(tickers):
    data = []
    string_tickers = " ".join(tickers)
    try:
        df_bulk = yf.download(string_tickers, period="1y", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
    except: return pd.DataFrame()
    
    for t in tickers:
        try:
            df = df_bulk[t].copy().dropna()
            if len(df) < 200: continue
            
            close = df['Close'].iloc[-1]
            rsi = ta.rsi(df['Close'], 14).iloc[-1]
            ema200 = ta.ema(df['Close'], 200).iloc[-1]
            atr = ta.atr(df['High'], df['Low'], df['Close'], 14).iloc[-1]
            
            # Score Simple
            trend = "ALCISTA" if close > ema200 else "BAJISTA"
            score = 0
            if trend == "ALCISTA": score += 50
            if rsi < 35: score += 40 # Oversold in uptrend
            elif rsi > 70: score -= 20
            
            data.append({
                "Ticker": t, "Precio": close, "RSI": rsi, 
                "Tendencia": trend, "ATR": atr, "Score": score
            })
        except: continue
    return pd.DataFrame(data)

# --- EJECUCIÓN PRINCIPAL ---

# 1. Pestañas Organizadas
tab_macro, tab_radar, tab_trade = st.tabs(["1️⃣ Macro Global", "2️⃣ Radar de Oportunidades", "3️⃣ Ejecución y Riesgo"])

with tab_macro:
    macro_data, df_sectors = obtener_macro()
    if macro_data:
        # Dashboard Macro
        c1, c2, c3 = st.columns(3)
        c1.metric("Tendencia S&P 500 (SPY)", macro_data["spy_trend"])
        c2.metric("Índice del Miedo (VIX)", f"{macro_data['vix']:.2f}", macro_data["mood"])
        
        estado_global = "🟢 COMPRAR" if "ALCISTA" in macro_data["spy_trend"] and macro_data['vix'] < 25 else "🔴 PRECAUCIÓN"
        c3.metric("VEREDICTO GLOBAL", estado_global)
        
        st.divider()
        st.subheader("🔥 Mapa de Calor de Sectores (Última Semana)")
        
        # Gráfico de Barras Sectores
        if not df_sectors.empty:
            df_sectors = df_sectors.sort_values("Retorno 1S", ascending=False)
            fig_sec = px.bar(df_sectors, x="Retorno 1S", y="Sector", orientation='h', 
                             color="Retorno 1S", color_continuous_scale=["red", "yellow", "green"],
                             title="¿Dónde está fluyendo el dinero?")
            fig_sec.update_layout(height=400)
            st.plotly_chart(fig_sec, use_container_width=True)
            
            lider = df_sectors.iloc[0]['Sector']
            st.info(f"💡 Pista: El dinero inteligente está rotando hacia **{lider}**. Busca acciones ahí.")

with tab_radar:
    st.subheader("📡 Escáner de Oportunidades")
    df_radar = escanear_acciones(WATCHLIST)
    
    if not df_radar.empty:
        df_radar = df_radar.sort_values(by="Score", ascending=False)
        
        # Formato visual
        def color_score
