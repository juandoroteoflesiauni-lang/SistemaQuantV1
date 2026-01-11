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
st.set_page_config(page_title="Sistema Quant V28.2 (Robust)", layout="wide", page_icon="🎯")
st.markdown("""
<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 10px; color: white;}
    .big-font {font-size:20px !important; font-weight: bold;}
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
def obtener_datos_completos(tickers):
    try:
        df_prices = yf.download(" ".join(tickers), period="1y", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
    except: return None

    resumen = []
    if df_prices is None or df_prices.empty: return None

    for t in tickers:
        try:
            if len(tickers) > 1:
                # Verificación de seguridad para MultiIndex
                if t in df_prices.columns.levels[0]:
                    df = df_prices[t].copy().dropna()
                else: continue
            else: 
                df = df_prices.copy().dropna()

            if len(df) < 50: continue
            
            # Indicadores
            last_close = df['Close'].iloc[-1]
            rsi = ta.rsi(df['Close'], 14).iloc[-1]
            ema200 = ta.ema(df['Close'], 200).iloc[-1]
            
            # Score
            score = 50
            trend = "ALCISTA" if last_close > ema200 else "BAJISTA"
            if trend == "ALCISTA": score += 20
            if rsi < 30: score += 30
            elif rsi > 70: score -= 20
            
            resumen.append({"Ticker": t, "Precio": last_close, "RSI": rsi, "Tendencia": trend, "Score": score})
        except: pass
    return pd.DataFrame(resumen)

# --- MOTOR GRÁFICO (RE-BLINDADO) ---
def graficar_avanzado(ticker):
    try:
        # Descarga
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df.empty: return None
        
        # 🛠️ ESTRATEGIA DE LIMPIEZA DE COLUMNAS (NUEVA)
        if isinstance(df.columns, pd.MultiIndex):
            # Intento 1: Si el ticker es el nivel superior, bajamos un nivel
            try:
                if ticker in df.columns.levels[0]:
                    df = df[ticker].copy()
                else:
                    # Intento 2: Tomar el último nivel (suele ser Price, Open, Close...)
                    df.columns = df.columns.get_level_values(-1)
            except:
                # Intento 3: Aplanamiento genérico
                df.columns = df.columns.get_level_values(-1)

        # 🛠️ SEGURO DE VIDA: Si después de todo no hay columna 'Close', renombramos por fuerza bruta
        if 'Close' not in df.columns:
            # Asumimos el orden estándar de Yahoo: Open, High, Low, Close, Volume
            if df.shape[1] >= 4:
                # Mapeo de seguridad (ajustar según lo que devuelve tu versión de yfinance)
                # A veces devuelve 5 col (con Volume) a veces 4.
                # Intentamos buscar la columna que se parezca a 'Close'
                cols = list(df.columns)
                found = False
                for c in cols:
                    if "Close" in str(c): 
                        df.rename(columns={c: 'Close'}, inplace=True)
                        found = True
                        break
                if not found and df.shape[1] == 5:
                     df.columns = ["Open", "High", "Low", "Close", "Volume"]

        # Calcular Indicadores (Ahora seguro)
        df['EMA20'] = ta.ema(df['Close'], 20)
        df['RSI'] = ta.rsi(df['Close'], 14)
        
        # Bandas de Bollinger
        bb = ta.bbands(df['Close'], length=20, std=2)
        if bb is not None:
            df = pd.concat([df, bb], axis=1) 
            bbu = df.columns[-3] # Banda Superior (pandas_ta pone nombres raros a veces, usamos índice)
            bbl = df.columns[-1] # Banda Inferior
        else:
            return None

        # Señales Visuales
        buy_signals = df[df['RSI'] < 35]
        sell_signals = df[df['RSI'] > 75]

        # Crear Subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])

        # 1. Velas Japonesas
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"), row=1, col=1)
        
        # 2. Medias y Bandas
        if 'EMA20' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='yellow', width=1), name="EMA 20"), row=1, col=1)
        
        # Usamos iloc para las bandas para evitar errores de nombre
        fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, -3], line=dict(color='gray', width=1, dash='dot'), name="Banda Sup"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, -1], line=dict(color='gray', width=1, dash='dot'), fill='tonexty', name="Banda Inf"), row=1, col=1)

        # 3. Marcadores
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ff00'), name="COMPRA"), row=1, col=1)
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ff0000'), name="VENTA"), row=1, col=1)

        # 4. RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(title=f"🔬 Análisis Técnico: {ticker}", template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
        return fig
    except Exception as e:
        st.error(f"Error graficando {ticker}: {e}")
        return None

# --- INTERFAZ ---
st.title("🎯 Sistema Quant V28.2: The Sniper Scope")

col_radar, col_chart = st.columns([1, 3])

with col_radar:
    st.subheader("📡 Radar")
    if st.button("🔄 Escanear"): st.cache_data.clear()
    
    df_radar = obtener_datos_completos(WATCHLIST)
    
    if df_radar is not None and not df_radar.empty:
        df_radar = df_radar.sort_values("Score", ascending=False)
        selected_ticker = st.radio("Selecciona Activo:", df_radar['Ticker'].tolist())
        
        row = df_radar[df_radar['Ticker'] == selected_ticker].iloc[0]
        st.divider()
        st.metric("Score", f"{row['Score']}/100")
        st.metric("RSI", f"{row['RSI']:.1f}")
        st.caption(f"Tendencia: {row['Tendencia']}")
        
        if st.button(f"📢 Alertar {selected_ticker}"):
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                          json={"chat_id": TELEGRAM_CHAT_ID, "text": f"🎯 OJO con {selected_ticker}. RSI: {row['RSI']:.1f}"})
            st.success("Enviado")
    else:
        st.info("Cargando datos...")

with col_chart:
    if df_radar is not None and not df_radar.empty:
        st.subheader(f"📊 Gráfico Interactivo: {selected_ticker}")
        
        t_chart, t_news = st.tabs(["📈 Gráfico Técnico", "📰 Noticias IA"])
        
        with t_chart:
            with st.spinner("Trazando indicadores..."):
                fig = graficar_avanzado(selected_ticker)
                if fig: st.plotly_chart(fig, use_container_width=True)
        
        with t_news:
            if st.button("Analizar Noticias (Live)"):
                try:
                    rss = f"https://news.google.com/rss/search?q={selected_ticker}+stock&hl=en-US&gl=US&ceid=US:en"
                    feed = feedparser.parse(rss)
                    txt = "\n".join([e.title for e in feed.entries[:5]])
                    res = model.generate_content(f"Analiza sentimiento (0-100) de: {txt}").text
                    st.info(res)
                except: st.error("Error noticias")
    else:
        st.info("👈 Dale al botón 'Escanear' en la izquierda para empezar.")
