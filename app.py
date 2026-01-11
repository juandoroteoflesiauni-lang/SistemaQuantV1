import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.express as px
import requests 
import google.generativeai as genai
import feedparser
import warnings
import numpy as np
import os
import toml
import re # Para limpiar texto

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
except Exception as e:
    st.error(f"❌ Error de claves: {e}")
    st.stop()

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Sistema Quant V27 (Oracle)", layout="wide", page_icon="🔮")
st.markdown("""
<style>
    .metric-card {background-color: #1e1e1e; border: 1px solid #333; border-radius: 8px; padding: 15px; color: white;}
    .sentiment-pos {color: #00ff00; font-weight: bold;}
    .sentiment-neg {color: #ff0000; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

# --- ACTIVOS ---
WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'COIN']

# --- MOTORES DE DATOS ---
@st.cache_data(ttl=1800)
def obtener_datos_mercado(tickers):
    try:
        df_prices = yf.download(" ".join(tickers), period="1y", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
    except: return None

    resumen = []
    if df_prices is None or df_prices.empty: return None

    for t in tickers:
        try:
            if len(tickers) > 1:
                if t not in df_prices.columns.levels[0]: continue
                df = df_prices[t].copy().dropna()
            else: df = df_prices.copy().dropna()

            if len(df) < 50: continue
            
            # KPIs Técnicos
            last_close = df['Close'].iloc[-1]
            rsi = ta.rsi(df['Close'], 14).iloc[-1]
            ema200 = ta.ema(df['Close'], 200).iloc[-1]
            atr = ta.atr(df['High'], df['Low'], df['Close'], 14).iloc[-1]
            
            trend = "ALCISTA" if last_close > ema200 else "BAJISTA"
            
            # Score Simple
            score = 50
            if trend == "ALCISTA": score += 20
            if rsi < 30: score += 30
            elif rsi > 70: score -= 20
            
            resumen.append({
                "Ticker": t, "Precio": last_close, "RSI": rsi,
                "Tendencia": trend, "ATR": atr, "Score": score
            })
        except: pass
    return pd.DataFrame(resumen)

# --- MOTOR DE NOTICIAS (NUEVO) ---
@st.cache_data(ttl=3600)
def analizar_noticias_ia(ticker):
    # 1. Buscar en Google News RSS
    try:
        # Buscamos en inglés para tener más volumen de datos
        rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+news&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        
        headlines = []
        for entry in feed.entries[:5]: # Analizamos las 5 más recientes
            headlines.append(f"- {entry.title}")
        
        texto_noticias = "\n".join(headlines)
        
        if not headlines:
            return {"score": 50, "summary": "Sin noticias recientes.", "headlines": []}

        # 2. Preguntar a Gemini
        prompt = f"""
        Actúa como analista financiero experto. Analiza estos titulares recientes sobre {ticker}:
        {texto_noticias}
        
        Tarea:
        1. Asigna un puntaje de Sentimiento de 0 (Muy Negativo) a 100 (Muy Positivo). 50 es Neutral.
        2. Resume la razón principal en una frase corta (en español).
        
        Responde SOLO con este formato exacto:
        SCORE: [Numero]
        RESUMEN: [Texto]
        """
        response = model.generate_content(prompt).text
        
        # 3. Extraer datos (Parsing básico)
        score_match = re.search(r"SCORE: (\d+)", response)
        score = int(score_match.group(1)) if score_match else 50
        
        summary_match = re.search(r"RESUMEN: (.*)", response)
        summary = summary_match.group(1) if summary_match else "Análisis IA completado."
        
        return {"score": score, "summary": summary, "headlines": headlines}
        
    except Exception as e:
        return {"score": 50, "summary": "Error analizando noticias.", "headlines": []}

# --- TELEGRAM ---
def enviar_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except: pass

# --- INTERFAZ ---
st.title("🔮 Sistema Quant V27: El Oráculo")

df_radar = obtener_datos_mercado(WATCHLIST)

# Sidebar
with st.sidebar:
    st.header("🎛️ Panel de Control")
    capital = st.number_input("Capital ($)", 2000)
    riesgo = st.slider("Riesgo %", 0.5, 3.0, 1.5)
    st.divider()
    if st.button("🔄 Refrescar Todo"):
        st.cache_data.clear()
        st.rerun()

# Tabs
tab1, tab2, tab3 = st.tabs(["📡 Radar Técnico", "📰 Noticias & Sentimiento", "🤖 Agente Telegram"])

with tab1:
    if df_radar is not None:
        st.subheader("Escáner de Mercado")
        df_show = df_radar.sort_values("Score", ascending=False)
        st.dataframe(df_show.style.format({"Precio": "${:.2f}", "RSI": "{:.1f}"}), use_container_width=True)
    else: st.error("Error cargando datos.")

with tab2:
    st.subheader("🧠 Análisis de Sentimiento (IA + Noticias)")
    st.info("La IA lee las noticias de Google News en tiempo real y detecta el 'humor' del mercado.")
    
    if df_radar is not None:
        sel_news = st.selectbox("Seleccionar Activo:", df_radar['Ticker'].tolist())
        
        if st.button(f"🔮 Leer Futuro de {sel_news}"):
            with st.spinner(f"Leyendo noticias sobre {sel_news}..."):
                news_data = analizar_noticias_ia(sel_news)
                
                # Visualización del Score (Medidor)
                sc = news_data['score']
                color = "green" if sc > 60 else "red" if sc < 40 else "yellow"
                
                c1, c2 = st.columns([1, 3])
                c1.metric("Sentimiento IA", f"{sc}/100", f"{'Positivo' if sc>60 else 'Negativo' if sc<40 else 'Neutral'}")
                c2.success(f"💡 **Conclusión:** {news_data['summary']}")
                
                with st.expander("Ver Titulares Analizados"):
                    for h in news_data['headlines']:
                        st.write(h)

with tab3:
    st.subheader("🚀 Envío de Señales")
    if st.button("Enviar Mejor Oportunidad a Telegram"):
        if df_radar is not None:
            best = df_radar.sort_values("Score", ascending=False).iloc[0]
            
            # Hacemos análisis de noticias al vuelo para el reporte
            with st.spinner("Analizando noticias de última hora..."):
                news_analysis = analizar_noticias_ia(best['Ticker'])
            
            msg = f"""
🌟 *OPORTUNIDAD V27 DETECTADA* 🌟

🎯 *Activo:* {best['Ticker']}
💰 *Precio:* ${best['Precio']:.2f}
📊 *RSI:* {best['RSI']:.1f}

🗞️ *Sentimiento Noticias:* {news_analysis['score']}/100
💡 *Razón:* {news_analysis['summary']}

_Enviado desde tu Sistema Quant_
            """
            enviar_telegram(msg)
            st.success("Mensaje Enviado!")
