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
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# --- 🔐 CREDENCIALES ---
TELEGRAM_TOKEN = "8042406069:AAHhflfkySyQVhCkHaqIsUjGumFr3fsnDPM" 
TELEGRAM_CHAT_ID = "6288094504"
GOOGLE_API_KEY = "AIzaSyB356Wjicaf9VRUYTX6_EL728IQF6nOmuQ" 

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Sistema Quant V24.9 (Lab)", layout="wide", page_icon="🧪")
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

# --- ACTIVOS & SECTORES ---
WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'COIN', 'PLTR']
SECTORS = {
    'XLK': 'Tecnología 💻', 'XLF': 'Finanzas 🏦', 'XLE': 'Energía 🛢️',
    'XLV': 'Salud 🏥', 'XLY': 'Consumo Disc. 🛍️', 'XLP': 'Consumo Bas. 🛒',
    'GLD': 'Oro 🥇'
}

st.title("🧪 Sistema Quant V24.9: Optimization Lab")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    capital_total = st.number_input("Capital Total ($)", value=2000)
    riesgo_pct = st.slider("Riesgo por Trade (%)", 0.5, 3.0, 1.5)
    st.divider()
    if st.button("🔄 Refrescar Todo"):
        st.cache_data.clear()
        st.rerun()

# --- MOTORES DE DATOS ---
@st.cache_data(ttl=900)
def obtener_macro():
    tickers = ["SPY", "^VIX"] + list(SECTORS.keys())
    try:
        data = yf.download(tickers, period="2y", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
    except: return None, None
    
    # 1. SPY
    try:
        spy = data["SPY"].copy().dropna()
        if len(spy) > 200:
            spy['EMA200'] = ta.ema(spy['Close'], 200)
            spy_trend = "ALCISTA 🟢" if spy['Close'].iloc[-1] > spy['EMA200'].iloc[-1] else "BAJISTA 🔴"
        else: spy_trend = "NEUTRAL"
    except: spy_trend = "N/A"
    
    # 2. VIX
    try:
        vix = data["^VIX"]['Close'].iloc[-1]
        mood = "PÁNICO 😱" if vix > 30 else ("MIEDO 😨" if vix > 20 else "COMPLACENCIA 😎")
    except: vix, mood = 0, "N/A"
    
    # 3. Sectores
    sec_perf = []
    for t, name in SECTORS.items():
        try:
            if t in data.columns.levels[0]:
                df = data[t].dropna()
                if len(df) > 5:
                    ret = (df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100
                    sec_perf.append({"Sector": name, "Retorno 1S": ret})
        except: continue
        
    return {"spy_trend": spy_trend, "vix": vix, "mood": mood}, pd.DataFrame(sec_perf)

@st.cache_data(ttl=300)
def escanear_acciones(tickers):
    data = []
    try:
        df_bulk = yf.download(" ".join(tickers), period="1y", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
    except: return pd.DataFrame()
    
    for t in tickers:
        try:
            if len(tickers) > 1:
                if t not in df_bulk.columns.levels[0]: continue
                df = df_bulk[t].copy().dropna()
            else: df = df_bulk.copy().dropna()
            
            if len(df) < 200: continue
            
            close = df['Close'].iloc[-1]
            rsi = ta.rsi(df['Close'], 14).iloc[-1]
            ema200 = ta.ema(df['Close'], 200).iloc[-1]
            atr = ta.atr(df['High'], df['Low'], df['Close'], 14).iloc[-1]
            
            score = 0
            trend = "ALCISTA" if close > ema200 else "BAJISTA"
            if trend == "ALCISTA": score += 50
            if rsi < 30: score += 40
            elif rsi > 70: score -= 20
            
            data.append({"Ticker": t, "Precio": close, "RSI": rsi, "Tendencia": trend, "ATR": atr, "Score": score})
        except: continue
    return pd.DataFrame(data)

# --- MOTOR DE OPTIMIZACIÓN (NUEVO) ---
def simular_estrategia(df, rsi_buy, rsi_sell):
    # Simulación vectorizada rápida
    df['RSI'] = ta.rsi(df['Close'], 14)
    df['Signal'] = 0
    # Compra: RSI < rsi_buy
    df.loc[df['RSI'] < rsi_buy, 'Signal'] = 1 
    # Venta: RSI > rsi_sell
    df.loc[df['RSI'] > rsi_sell, 'Signal'] = -1
    
    # Calcular retornos
    df['Strategy_Ret'] = df['Signal'].shift(1) * df['Close'].pct_change()
    total_return = (1 + df['Strategy_Ret']).cumprod().iloc[-1] - 1
    trades_count = df['Signal'].abs().sum() / 2 # Aprox entradas y salidas
    
    return total_return * 100, int(trades_count)

# --- INTERFAZ ---
tab_macro, tab_radar, tab_trade, tab_lab = st.tabs(["1️⃣ Macro", "2️⃣ Radar", "3️⃣ Ejecución", "🧪 Laboratorio"])

with tab_macro:
    macro, df_sec = obtener_macro()
    if macro:
        c1, c2, c3 = st.columns(3)
        c1.metric("S&P 500", macro['spy_trend'])
        c2.metric("VIX (Miedo)", f"{macro['vix']:.2f}", macro['mood'])
        st.plotly_chart(px.bar(df_sec.sort_values("Retorno 1S"), x="Retorno 1S", y="Sector", orientation='h', title="Flujo de Dinero (1 Semana)"), use_container_width=True)

with tab_radar:
    st.subheader("📡 Escáner de Oportunidades")
    df_radar = escanear_acciones(WATCHLIST)
    if not df_radar.empty:
        df_radar = df_radar.sort_values("Score", ascending=False)
        st.dataframe(df_radar.style.format({"Precio": "${:.2f}", "RSI": "{:.1f}", "ATR": "{:.2f}"}), use_container_width=True)
    else: st.warning("Cargando datos...")

with tab_trade:
    st.subheader("🛡️ Calculadora de Riesgo")
    if not df_radar.empty:
        tk_sel = st.selectbox("Activo", df_radar['Ticker'].tolist())
        row = df_radar[df_radar['Ticker'] == tk_sel].iloc[0]
        stop = row['Precio'] - (2 * row['ATR'])
        shares = (capital_total * riesgo_pct / 100) / (row['Precio'] - stop)
        
        c1, c2 = st.columns(2)
        c1.metric("Precio Entrada", f"${row['Precio']:.2f}")
        c1.metric("Stop Loss (2xATR)", f"${stop:.2f}")
        c2.metric("ACCIONES A COMPRAR", f"{int(shares)}")
        c2.metric("Riesgo Total", f"${(capital_total * riesgo_pct / 100):.2f}")
        
        if st.button(f"🧠 Analizar {tk_sel} con IA"):
            prompt = f"Analiza {tk_sel}. Precio ${row['Precio']}, RSI {row['RSI']}, Tendencia {row['Tendencia']}. Recomendación corta."
            try:
                res = model.generate_content(prompt)
                st.info(res.text)
            except: st.error("Error IA")

# --- PESTAÑA LABORATORIO (NUEVA) ---
with tab_lab:
    st.subheader("🔬 Optimizador de Estrategia (Backtest Dinámico)")
    st.info("Descubre qué configuración de RSI funcionó mejor históricamente para un activo.")
    
    col_l1, col_l2 = st.columns([1, 3])
    
    with col_l1:
        lab_ticker = st.selectbox("Activo a Optimizar", WATCHLIST)
        lab_days = st.slider("Días de Historial", 100, 700, 365)
        st.write("---")
        st.write("**Rango de Pruebas:**")
        start_buy = st.number_input("RSI Compra Desde", value=20)
        end_buy = st.number_input("RSI Compra Hasta", value=40)
        
    with col_l2:
        if st.button("🚀 INICIAR SIMULACIÓN MATRICIAL", type="primary"):
            with st.spinner(f"Simulando miles de escenarios para {lab_ticker}..."):
                # 1. Obtener datos
                df_lab = yf.download(lab_ticker, period=f"{lab_days}d", interval="1d", progress=False, auto_adjust=True)
                if isinstance(df_lab.columns, pd.MultiIndex): df_lab.columns = df_lab.columns.get_level_values(0)
                
                # 2. Grid Search (Fuerza Bruta Inteligente)
                results = []
                buy_range = range(start_buy, end_buy + 5, 5) # De 5 en 5
                sell_range = range(60, 85, 5)
                
                best_ret = -999
                best_params = (0, 0)
                
                for b in buy_range:
                    for s in sell_range:
                        if b >= s: continue # Configuración imposible
                        ret, trades = simular_estrategia(df_lab.copy(), b, s)
                        results.append({'RSI Compra': b, 'RSI Venta': s, 'Retorno %': ret, 'Trades': trades})
                        
                        if ret > best_ret:
                            best_ret = ret
                            best_params = (b, s)
                
                # 3. Visualizar Resultados
                df_res = pd.DataFrame(results)
                
                # KPI Ganador
                st.success(f"🏆 MEJOR CONFIGURACIÓN: RSI Compra **{best_params[0]}** / Venta **{best_params[1]}**")
                k1, k2 = st.columns(2)
                k1.metric("Retorno Máximo Encontrado", f"{best_ret:.2f}%")
                k1.caption(f"Comparado con Buy & Hold: {((df_lab['Close'].iloc[-1]/df_lab['Close'].iloc[0])-1)*100:.2f}%")
                
                # Mapa de Calor
                st.subheader("Mapa de Calor de Rentabilidad")
                fig_heat = px.density_heatmap(df_res, x="RSI Venta", y="RSI Compra", z="Retorno %", 
                                              text_auto=True, color_continuous_scale="RdBu")
                st.plotly_chart(fig_heat, use_container_width=True)
                
                st.dataframe(df_res.sort_values("Retorno %", ascending=False).head(5))
