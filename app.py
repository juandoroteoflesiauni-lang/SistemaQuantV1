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
import os
import toml

warnings.filterwarnings('ignore')

# --- 🔐 CREDENCIALES (MODO MANUAL ROBUSTO) ---
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
    st.error(f"❌ Error leyendo claves: {e}")
    st.stop()

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Sistema Quant V26 (Commander)", layout="wide", page_icon="🫡")
st.markdown("""
<style>
    .metric-card {background-color: #1e1e1e; border: 1px solid #333; border-radius: 8px; padding: 15px; color: white;}
    .stDataFrame {border: 1px solid #444; border-radius: 5px;}
    .report-btn {width: 100%; border: 1px solid #00ff00; color: #00ff00;}
</style>
""", unsafe_allow_html=True)

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

# --- ACTIVOS ---
WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'COIN', 'KO', 'DIS', 'JPM']

# --- MOTORES DE DATOS ---
@st.cache_data(ttl=1800)
def obtener_datos_completos(tickers):
    try:
        df_prices = yf.download(" ".join(tickers), period="1y", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
    except: return None, None

    resumen = []
    
    if df_prices is None or df_prices.empty: return None, None

    progress_bar = st.progress(0)
    step = 1.0 / len(tickers)
    
    for i, t in enumerate(tickers):
        try:
            if len(tickers) > 1:
                if t not in df_prices.columns.levels[0]: continue
                df = df_prices[t].copy().dropna()
            else: df = df_prices.copy().dropna()

            if len(df) < 50: continue
            
            # Análisis Técnico
            last_close = df['Close'].iloc[-1]
            rsi = ta.rsi(df['Close'], 14).iloc[-1]
            ema200 = ta.ema(df['Close'], 200).iloc[-1]
            atr = ta.atr(df['High'], df['Low'], df['Close'], 14).iloc[-1]
            
            trend_d = "ALCISTA" if last_close > ema200 else "BAJISTA"
            mom_1w = "ALCISTA" if df['Close'].iloc[-1] > df['Close'].iloc[-5] else "BAJISTA"
            volatilidad = df['Close'].pct_change().std() * np.sqrt(252) * 100
            
            score = 50
            if trend_d == "ALCISTA": score += 20
            if mom_1w == "ALCISTA": score += 10
            if rsi < 30: score += 30
            elif rsi > 70: score -= 20
            
            resumen.append({
                "Ticker": t, "Precio": last_close, "RSI": rsi,
                "Tendencia": trend_d, "Momentum (1S)": mom_1w,
                "Volatilidad %": volatilidad, "ATR": atr, "Score": score
            })
        except: pass
        progress_bar.progress(min((i + 1) * step, 1.0))
        
    progress_bar.empty()
    return pd.DataFrame(resumen), df_prices

@st.cache_data(ttl=3600)
def obtener_fundamental_inferido(ticker):
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
            peg_source = "Yahoo Oficial"
        elif pe > 0 and growth_est > 0:
            peg_final = pe / (growth_est * 100)
            peg_source = "Calculado (Inferencia)"
        
        return {
            "PER": pe, "PEG": peg_final, "PEG_Source": peg_source,
            "Growth": growth_est, "P/B": pb, "Beta": beta,
            "Target Price": target, "Sector": sector, "Raw Info": info
        }
    except: return None

# --- COMUNICACIONES (TELEGRAM) ---
def enviar_telegram(mensaje):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
        return True
    except: return False

# --- ESTRUCTURA PRINCIPAL ---
st.title("🫡 Sistema Quant V26: El Comandante")

# Carga de datos inicial
df_radar, df_raw_prices = obtener_datos_completos(WATCHLIST)

# --- BARRA LATERAL (COMANDOS) ---
with st.sidebar:
    st.header("📡 Centro de Mando")
    capital_total = st.number_input("Capital ($)", value=2000)
    riesgo_pct = st.slider("Riesgo (%)", 0.5, 3.0, 2.0)
    
    st.divider()
    st.subheader("📢 Reportes")
    
    if st.button("🚀 Enviar Informe a Telegram"):
        if df_radar is not None and not df_radar.empty:
            with st.spinner("Redactando informe de inteligencia..."):
                # 1. Encontrar la joya
                mejor_activo = df_radar.sort_values("Score", ascending=False).iloc[0]
                ticker = mejor_activo['Ticker']
                
                # 2. Análisis Rápido IA
                prompt = f"""
                Actúa como un gestor de fondos de Wall Street.
                Escribe un reporte MUY BREVE (max 3 líneas) para Telegram.
                
                Datos:
                - Top Pick: {ticker} (Score: {mejor_activo['Score']})
                - Precio: ${mejor_activo['Precio']:.2f}
                - Tendencia: {mejor_activo['Tendencia']}
                - RSI: {mejor_activo['RSI']:.1f}
                
                Dime si compro o espero y por qué. Usa emojis.
                """
                try:
                    analisis_ia = model.generate_content(prompt).text
                except: analisis_ia = "Análisis IA no disponible."

                # 3. Formatear Mensaje
                mensaje = f"""
🚨 *INFORME QUANT DIARIO* 🚨

📊 *Mejor Oportunidad:* {ticker}
💰 *Precio:* ${mejor_activo['Precio']:.2f}
📈 *Tendencia:* {mejor_activo['Tendencia']}
🎯 *Score:* {mejor_activo['Score']}/100

🧠 *Opinión del Algoritmo:*
{analisis_ia}

_Generado por Sistema Quant V26_
                """
                if enviar_telegram(mensaje):
                    st.success("✅ ¡Informe enviado a tu celular!")
                else:
                    st.error("❌ Error enviando mensaje.")
        else:
            st.warning("Espera a que carguen los datos.")

    st.divider()
    if st.button("🔄 Refrescar Datos"):
        st.cache_data.clear()
        st.rerun()

# --- PESTAÑAS PRINCIPALES ---
tabs = st.tabs(["📡 Radar & Fundamental", "🕸️ Correlaciones", "🛡️ Calculadora", "🧪 Laboratorio"])

with tabs[0]:
    if df_radar is not None and not df_radar.empty:
        col_main, col_detail = st.columns([2, 1])
        with col_main:
            st.subheader("Radar de Oportunidades")
            df_show = df_radar.sort_values("Score", ascending=False)
            def color_trend(val): return 'color: lightgreen' if val == "ALCISTA" else 'color: #ffcccb'
            st.dataframe(df_show.style.map(color_trend, subset=['Tendencia']).format({"Precio": "${:.2f}", "RSI": "{:.1f}", "Volatilidad %": "{:.1f}%"}), use_container_width=True, height=500)
        with col_detail:
            st.subheader("🔬 Fundamental (Inferencia)")
            sel_fund = st.selectbox("Analizar Activo:", df_radar['Ticker'].tolist())
            if st.button(f"🔍 Escanear {sel_fund}"):
                with st.spinner("Analizando..."):
                    fund_data = obtener_fundamental_inferido(sel_fund)
                    if fund_data:
                        c1, c2 = st.columns(2)
                        c1.metric("PER", f"{fund_data['PER']:.2f}x")
                        peg_val = fund_data['PEG']
                        delta_color = "normal" if 0 < peg_val < 1.5 else "inverse"
                        c2.metric("PEG Ratio", f"{peg_val:.2f}", fund_data['PEG_Source'], delta_color=delta_color)
                        st.info(f"Sector: {fund_data['Sector']}")
                    else: st.error("Error fundamental.")

with tabs[1]:
    st.subheader("🕸️ Matriz de Riesgo")
    if df_raw_prices is not None and not df_raw_prices.empty:
        close_df = pd.DataFrame()
        for t in WATCHLIST:
            try:
                if len(WATCHLIST) > 1 and t in df_raw_prices.columns.levels[0]: close_df[t] = df_raw_prices[t]['Close']
                else: close_df[t] = df_raw_prices['Close']
            except: pass
        if not close_df.empty:
            st.plotly_chart(px.imshow(close_df.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)

with tabs[2]:
    st.subheader("🛡️ Calculadora")
    if df_radar is not None and not df_radar.empty:
        tk = st.selectbox("Activo", df_radar['Ticker'].tolist(), key="calc")
        row = df_radar[df_radar['Ticker'] == tk].iloc[0]
        stop = row['Precio'] - (2 * row['ATR'])
        shares = (capital_total * riesgo_pct / 100) / (row['Precio'] - stop)
        st.metric("Orden Segura", f"{int(shares)} Acciones", f"Stop Loss: ${stop:.2f}")

with tabs[3]:
    st.info("Laboratorio disponible (código backend).")
