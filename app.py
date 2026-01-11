import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests 
import google.generativeai as genai
import feedparser
import quantstats as qs
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# --- 🔐 CREDENCIALES (YA INTEGRADAS) ---
TELEGRAM_TOKEN = "8042406069:AAHhflfkySyQVhCkHaqIsUjGumFr3fsnDPM" 
TELEGRAM_CHAT_ID = "6288094504"
GOOGLE_API_KEY = "AIzaSyB356Wjicaf9VRUYTX6_EL728IQF6nOmuQ" 

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Sistema Quant V24.6 (Radar)", layout="wide", page_icon="📡")
st.markdown("<style>.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

# Configurar IA
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

# --- LISTA DE VIGILANCIA (Tus Activos Favoritos) ---
WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD']

st.title("📡 Sistema Quant V24.6: Radar de Mercado")

# --- MOTORES DE DATOS ---
@st.cache_data(ttl=600) # Caché de 10 minutos para velocidad
def escanear_mercado(tickers):
    data = []
    # Descarga masiva optimizada
    string_tickers = " ".join(tickers)
    df_bulk = yf.download(string_tickers, period="1y", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
    
    for t in tickers:
        try:
            # Extraer datos del ticker específico
            if len(tickers) > 1:
                df = df_bulk[t].copy()
            else:
                df = df_bulk.copy() # Caso de un solo ticker
            
            df = df.dropna()
            if df.empty: continue

            # Cálculos Técnicos Rápidos
            rsi = ta.rsi(df['Close'], length=14).iloc[-1]
            ema200 = ta.ema(df['Close'], length=200).iloc[-1]
            price = df['Close'].iloc[-1]
            change = (price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
            
            # Lógica de Señal (Cerebro Básico)
            trend = "ALCISTA" if price > ema200 else "BAJISTA"
            signal = "NEUTRAL"
            score = 50 # Puntuación base
            
            if trend == "ALCISTA":
                score += 20
                if rsi < 30: 
                    signal = "COMPRA FUERTE 🟢"
                    score += 30
                elif rsi < 45: 
                    signal = "COMPRA 🟢"
                    score += 10
                elif rsi > 70:
                    signal = "SOBRECOMPRA (Cuidado) ⚠️"
                    score -= 10
            else: # Bajista
                score -= 20
                if rsi > 70: 
                    signal = "VENTA FUERTE 🔴"
                    score -= 30
                elif rsi < 30:
                    signal = "REBOTE TÉCNICO (Riesgo) ⚠️"
            
            data.append({
                "Ticker": t,
                "Precio": price,
                "Cambio 24h": change,
                "RSI": rsi,
                "Tendencia (EMA200)": trend,
                "Señal": signal,
                "Score": score
            })
        except Exception as e:
            continue
            
    return pd.DataFrame(data)

def obtener_datos_detalle(symbol):
    try:
        df = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return None

# --- INTERFAZ PRINCIPAL ---

# Pestañas para organizar
tab_radar, tab_analisis, tab_ia = st.tabs(["📡 Radar General", "📈 Análisis Detallado", "🧠 IA & Noticias"])

with tab_radar:
    st.subheader(f"🔍 Escaneo en tiempo real ({len(WATCHLIST)} activos)")
    
    if st.button("🔄 Actualizar Escáner"):
        st.cache_data.clear()
        st.rerun()

    df_radar = escanear_mercado(WATCHLIST)
    
    if not df_radar.empty:
        # Ordenar por Score (Oportunidades arriba)
        df_radar = df_radar.sort_values(by="Score", ascending=False)
        
        # Métricas Globales
        top_pick = df_radar.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("🔥 Mejor Oportunidad", top_pick['Ticker'], f"{top_pick['Score']}/100")
        c2.metric("Precio", f"${top_pick['Precio']:.2f}")
        c3.metric("Señal", top_pick['Señal'])
        
        # Tabla Interactiva con Colores
        def color_rsi(val):
            color = 'red' if val > 70 else 'green' if val < 30 else 'white'
            return f'color: {color}; font-weight: bold'
        
        def color_signal(val):
            return 'background-color: #1c4a1c' if 'COMPRA' in val else 'background-color: #4a1c1c' if 'VENTA' in val else ''

        st.dataframe(
            df_radar.style.applymap(color_rsi, subset=['RSI'])
                          .applymap(color_signal, subset=['Señal'])
                          .format({"Precio": "${:.2f}", "Cambio 24h": "{:.2%}", "RSI": "{:.1f}"}),
            use_container_width=True,
            height=500
        )
        
        # Botón de Alerta Masiva
        if st.button("📲 Enviar Reporte Resumido a Telegram"):
            opportunities = df_radar[df_radar['Score'] >= 70]
            if not opportunities.empty:
                msg = "🚀 **REPORTE RADAR V24** 🚀\n\n"
                for index, row in opportunities.iterrows():
                    msg += f"👉 **{row['Ticker']}**: {row['Señal']} (RSI: {row['RSI']:.0f})\n"
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                              json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
                st.success("Reporte enviado!")
            else:
                st.info("No hay oportunidades fuertes (>70) para reportar ahora.")

with tab_analisis:
    # Selector de activo para análisis profundo (el código viejo mejorado)
    ticker_select = st.selectbox("Selecciona un activo para ver el gráfico:", WATCHLIST)
    
    df_detail = obtener_datos_detalle(ticker_select)
    
    if df_detail is not None:
        df_detail['EMA_200'] = ta.ema(df_detail['Close'], length=200)
        df_detail['RSI'] = ta.rsi(df_detail['Close'], length=14)
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_detail.index, open=df_detail['Open'], high=df_detail['High'],
                                     low=df_detail['Low'], close=df_detail['Close'], name='Precio'))
        fig.add_trace(go.Scatter(x=df_detail.index, y=df_detail['EMA_200'], line=dict(color='orange'), name='EMA 200'))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Integración QuantStats rápida
        with st.expander("📊 Ver Métricas de Rendimiento (QuantStats)"):
            returns = df_detail['Close'].pct_change()
            qs.extend_pandas()
            st.write(f"**Sharpe Ratio:** {returns.sharpe():.2f}")
            st.write(f"**Max Drawdown:** {returns.max_drawdown()*100:.2f}%")

with tab_ia:
    st.markdown("### 🧠 Consultar IA sobre el Activo Seleccionado")
    st.info(f"Analizando: **{ticker_select}**")
    if st.button("Generar Opinión IA"):
        with st.spinner("Leyendo noticias..."):
            # Lógica simple de noticias
            try:
                feed = feedparser.parse(f"https://news.google.com/rss/search?q={ticker_select}+stock&hl=en-US&gl=US&ceid=US:en")
                news_titles = [e.title for e in feed.entries[:5]]
                prompt = f"Analiza {ticker_select} basándote en estos titulares recientes: {news_titles}. Dame una recomendación corta de trading."
                response = model.generate_content(prompt)
                st.success(response.text)
            except Exception as e:
                st.error(f"Error IA: {e}")
