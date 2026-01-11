import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import requests 
import google.generativeai as genai
import feedparser
import warnings
import numpy as np
import os
import toml
import re

warnings.filterwarnings('ignore')

# --- 🔐 CREDENCIALES ---
try:
    secrets_path = ".streamlit/secrets.toml"
    if os.path.exists(secrets_path):
        secrets = toml.load(secrets_path)
        TELEGRAM_TOKEN = secrets["TELEGRAM_TOKEN"]
        TELEGRAM_CHAT_ID = secrets["TELEGRAM_CHAT_ID"]
        GOOGLE_API_KEY = secrets["GOOGLE_API_KEY"]
    else:
        TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
        TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except: st.stop()

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Sistema Quant V29 (Terminal)", layout="wide", page_icon="🏛️")
st.markdown("""
<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 10px; color: white;}
    .big-font {font-size:18px !important; font-weight: bold;}
    .stTabs [data-baseweb="tab-list"] {gap: 5px;}
    .stTabs [data-baseweb="tab"] {height: 40px; white-space: pre-wrap; background-color: #1e1e1e; border-radius: 5px; color: white;}
    .stTabs [aria-selected="true"] {background-color: #00aa00; color: white !important;}
</style>
""", unsafe_allow_html=True)

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

# --- ACTIVOS ---
WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'COIN']

# --- MOTORES DE DATOS ---

@st.cache_data(ttl=900)
def obtener_macro():
    # Motor Macro V24 recuperado
    try:
        data = yf.download("SPY ^VIX", period="1y", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
        spy = data['SPY']['Close']
        vix = data['^VIX']['Close'].iloc[-1]
        
        # Tendencia SPY
        ema200 = ta.ema(spy, 200).iloc[-1]
        last_spy = spy.iloc[-1]
        trend = "ALCISTA 🟢" if last_spy > ema200 else "BAJISTA 🔴"
        
        # Estado VIX
        mood = "PÁNICO 😱" if vix > 30 else "MIEDO 😨" if vix > 20 else "CALMA 😎"
        
        return trend, vix, mood
    except: return "N/A", 0, "N/A"

@st.cache_data(ttl=900)
def obtener_radar(tickers):
    try:
        df_prices = yf.download(" ".join(tickers), period="1y", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
    except: return None

    resumen = []
    if df_prices is None or df_prices.empty: return None

    for t in tickers:
        try:
            if len(tickers) > 1:
                if t in df_prices.columns.levels[0]: df = df_prices[t].copy().dropna()
                else: continue
            else: df = df_prices.copy().dropna()

            if len(df) < 50: continue
            
            last_close = df['Close'].iloc[-1]
            rsi = ta.rsi(df['Close'], 14).iloc[-1]
            ema200 = ta.ema(df['Close'], 200).iloc[-1]
            atr = ta.atr(df['High'], df['Low'], df['Close'], 14).iloc[-1]
            
            score = 50
            trend = "ALCISTA" if last_close > ema200 else "BAJISTA"
            if trend == "ALCISTA": score += 20
            if rsi < 30: score += 30
            elif rsi > 70: score -= 20
            
            resumen.append({"Ticker": t, "Precio": last_close, "RSI": rsi, "Tendencia": trend, "ATR": atr, "Score": score})
        except: pass
    return pd.DataFrame(resumen)

@st.cache_data(ttl=3600)
def obtener_fundamental_inferido(ticker):
    # Motor Fundamental V25.2 recuperado
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        pe = info.get('trailingPE') or info.get('forwardPE') or 0
        peg_oficial = info.get('pegRatio')
        growth_est = info.get('earningsGrowth') or info.get('revenueGrowth') or 0
        pb = info.get('priceToBook') or 0
        beta = info.get('beta') or 1
        target = info.get('targetMeanPrice') or 0
        sector = info.get('sector') or 'N/A'

        peg_final = 0
        peg_source = "N/A"
        if peg_oficial is not None:
            peg_final = peg_oficial
            peg_source = "Yahoo"
        elif pe > 0 and growth_est > 0:
            peg_final = pe / (growth_est * 100)
            peg_source = "Estimado"
        
        return {"PER": pe, "PEG": peg_final, "PEG_Source": peg_source, "Growth": growth_est, "Target": target, "Sector": sector}
    except: return None

def graficar_sniper(ticker):
    # Motor Gráfico V28.2 Blindado
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df.empty: return None
        
        # Limpieza MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            try:
                if ticker in df.columns.levels[0]: df = df[ticker].copy()
                else: df.columns = df.columns.get_level_values(-1)
            except: df.columns = df.columns.get_level_values(-1)
        
        if 'Close' not in df.columns: # Blindaje final
             if df.shape[1] >= 4:
                cols = list(df.columns)
                for c in cols:
                    if "Close" in str(c): df.rename(columns={c: 'Close'}, inplace=True); break

        df['EMA20'] = ta.ema(df['Close'], 20)
        df['RSI'] = ta.rsi(df['Close'], 14)
        bb = ta.bbands(df['Close'], length=20, std=2)
        if bb is not None: df = pd.concat([df, bb], axis=1)
        
        buy_sig = df[df['RSI'] < 35]
        sell_sig = df[df['RSI'] > 75]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"), row=1, col=1)
        if 'EMA20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='yellow', width=1), name="EMA 20"), row=1, col=1)
        try:
            fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, -3], line=dict(color='gray', width=1, dash='dot'), name="Banda Sup"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, -1], line=dict(color='gray', width=1, dash='dot'), fill='tonexty', name="Banda Inf"), row=1, col=1)
        except: pass
        
        fig.add_trace(go.Scatter(x=buy_sig.index, y=buy_sig['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ff00'), name="COMPRA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_sig.index, y=sell_sig['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ff0000'), name="VENTA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=5, r=5, t=5, b=5))
        return fig
    except: return None

# --- INTERFAZ PRINCIPAL ---

st.title("🏛️ Hedge Fund Terminal V29")

# 1. BARRA SUPERIOR (MACRO)
spy_trend, vix_val, market_mood = obtener_macro()
m1, m2, m3, m4 = st.columns(4)
m1.metric("S&P 500 Trend", spy_trend)
m2.metric("VIX (Miedo)", f"{vix_val:.2f}", market_mood)
m3.metric("Capital Disponible", f"${st.session_state.get('capital', 2000)}") # Placeholder
m4.metric("Estado Sistema", "🟢 ONLINE")

st.divider()

# 2. ESTRUCTURA DE PANTALLA DIVIDIDA
col_left, col_right = st.columns([1, 2.5])

# --- COLUMNA IZQUIERDA: RADAR Y CONTROLES ---
with col_left:
    st.subheader("📡 Radar de Activos")
    
    # Configuración Mini
    with st.expander("⚙️ Configuración"):
        capital = st.number_input("Capital ($)", 2000, key='capital')
        riesgo = st.slider("Riesgo %", 0.5, 3.0, 1.5)
        if st.button("🔄 Refrescar"): st.cache_data.clear(); st.rerun()

    df_radar = obtener_radar(WATCHLIST)
    
    if df_radar is not None and not df_radar.empty:
        df_radar = df_radar.sort_values("Score", ascending=False)
        
        # Tabla interactiva (Radio Button fake con dataframe selection)
        # Usamos un selectbox para elegir activo, es más limpio en móvil/desktop
        selected_ticker = st.selectbox("🔍 Seleccionar Activo:", df_radar['Ticker'].tolist())
        
        # Mostrar tabla filtrada o completa mini
        st.dataframe(
            df_radar[['Ticker', 'Precio', 'Score', 'RSI']]
            .style.format({"Precio": "${:.2f}", "RSI": "{:.0f}"})
            .background_gradient(subset=['Score'], cmap='RdYlGn'),
            use_container_width=True, height=400, hide_index=True
        )
        
        # Datos del seleccionado para pasar a la derecha
        row = df_radar[df_radar['Ticker'] == selected_ticker].iloc[0]
    else:
        st.warning("Cargando datos...")
        st.stop()

# --- COLUMNA DERECHA: TERMINAL DE ANÁLISIS ---
with col_right:
    st.subheader(f"Analizando: {selected_ticker} | Score: {row['Score']}/100")
    
    # Sistema de Pestañas UNIFICADO
    tabs = st.tabs(["📈 Gráfico Sniper", "🔬 Fundamental & PEG", "🛡️ Calculadora", "📰 Noticias IA", "🚀 Telegram"])
    
    # TAB 1: GRÁFICO
    with tabs[0]:
        with st.spinner("Cargando Gráfico Sniper..."):
            fig = graficar_sniper(selected_ticker)
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.error("Error en gráfico")
            
    # TAB 2: FUNDAMENTAL (Recuperado V25)
    with tabs[1]:
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            if st.button("🔎 Analizar Fundamental"):
                with st.spinner("Calculando PEG..."):
                    fund = obtener_fundamental_inferido(selected_ticker)
                    if fund:
                        st.session_state['fund_data'] = fund
                    else: st.error("Datos no disponibles")
        
        if 'fund_data' in st.session_state:
            fd = st.session_state['fund_data']
            c1, c2, c3 = st.columns(3)
            c1.metric("PER (Valuación)", f"{fd['PER']:.2f}x")
            
            # Lógica visual PEG
            peg_color = "normal"
            if 0 < fd['PEG'] < 1: peg_color = "inverse" # Verde (Barato)
            c2.metric("PEG Ratio", f"{fd['PEG']:.2f}", fd['PEG_Source'], delta_color=peg_color)
            
            c3.metric("Sector", fd['Sector'])
            
            st.info(f"💰 Precio Objetivo Analistas: **${fd['Target']}**")
            if fd['PEG'] == 0: st.warning("PEG = 0 indica falta de datos de crecimiento.")

    # TAB 3: CALCULADORA (Recuperada V25)
    with tabs[2]:
        stop_loss = row['Precio'] - (2 * row['ATR'])
        distancia = row['Precio'] - stop_loss
        riesgo_usd = capital * (riesgo / 100)
        qty = riesgo_usd / distancia if distancia > 0 else 0
        
        c1, c2 = st.columns(2)
        c1.info(f"📉 **Stop Loss Técnico (2xATR):** ${stop_loss:.2f}")
        c2.success(f"🛒 **Tamaño de Posición:** {int(qty)} Acciones")
        st.caption(f"Arriesgando solo ${riesgo_usd:.2f} (El {riesgo}% de tu capital)")

    # TAB 4: NOTICIAS (Recuperada V27)
    with tabs[3]:
        if st.button("🔮 Leer Noticias con IA"):
            try:
                rss = f"https://news.google.com/rss/search?q={selected_ticker}+stock&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(rss)
                txt = "\n".join([f"- {e.title}" for e in feed.entries[:5]])
                prompt = f"Resume sentimiento (0-100) y razón principal de: {txt}"
                res = model.generate_content(prompt).text
                st.write(res)
            except: st.error("Error conectando con Google News")

    # TAB 5: TELEGRAM (Recuperada V26)
    with tabs[4]:
        st.write("Enviar señal manual a tu canal:")
        if st.button("🚀 ENVIAR SEÑAL AHORA"):
            msg = f"🚀 SEÑAL {selected_ticker} | Precio: ${row['Precio']:.2f} | Score: {row['Score']} | RSI: {row['RSI']:.1f}"
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
            st.success("Enviado!")
