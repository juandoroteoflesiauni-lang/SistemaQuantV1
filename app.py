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

# --- LIBRERÍAS DE MACHINE LEARNING ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

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
st.set_page_config(page_title="Sistema Quant V30 (ML Prediction)", layout="wide", page_icon="🧠")
st.markdown("""
<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 10px; color: white;}
    .big-font {font-size:18px !important; font-weight: bold;}
    .stTabs [data-baseweb="tab-list"] {gap: 5px;}
    .stTabs [data-baseweb="tab"] {height: 40px; white-space: pre-wrap; background-color: #1e1e1e; border-radius: 5px; color: white;}
    .stTabs [aria-selected="true"] {background-color: #00aa00; color: white !important;}
    .pred-box {border: 2px solid #4CAF50; padding: 10px; border-radius: 10px; text-align: center;}
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
    try:
        data = yf.download("SPY ^VIX", period="1y", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
        spy = data['SPY']['Close']
        vix = data['^VIX']['Close'].iloc[-1]
        
        ema200 = ta.ema(spy, 200).iloc[-1]
        last_spy = spy.iloc[-1]
        trend = "ALCISTA 🟢" if last_spy > ema200 else "BAJISTA 🔴"
        
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
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        pe = info.get('trailingPE') or info.get('forwardPE') or 0
        peg_oficial = info.get('pegRatio')
        growth_est = info.get('earningsGrowth') or info.get('revenueGrowth') or 0
        pb = info.get('priceToBook') or 0
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
        
        return {"PER": pe, "PEG": peg_final, "PEG_Source": peg_source, "Target": target, "Sector": sector}
    except: return None

def graficar_sniper(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df.empty: return None
        
        # Limpieza MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            try:
                if ticker in df.columns.levels[0]: df = df[ticker].copy()
                else: df.columns = df.columns.get_level_values(-1)
            except: df.columns = df.columns.get_level_values(-1)
        
        if 'Close' not in df.columns: 
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

# --- MOTOR DE PREDICCIÓN ML (NUEVO) ---
def predecir_precio_ia(ticker):
    try:
        # 1. Descargar más historia para entrenar mejor (2 años)
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.levels[0]: df = df[ticker].copy()
            else: df.columns = df.columns.get_level_values(-1)
        
        if 'Close' not in df.columns: return None

        # 2. Ingeniería de Características (Crear datos para la IA)
        df['RSI'] = ta.rsi(df['Close'], 14)
        df['EMA20'] = ta.ema(df['Close'], 20)
        df['Return'] = df['Close'].pct_change()
        df['Volatilidad'] = df['Return'].rolling(5).std()
        
        # Feature Lagging (La IA aprende del pasado para predecir futuro)
        df['Lag_Close_1'] = df['Close'].shift(1)
        df['Lag_RSI'] = df['RSI'].shift(1)
        
        df.dropna(inplace=True)

        # 3. Definir Objetivo: El precio de Cierre (Close)
        X = df[['Lag_Close_1', 'Lag_RSI', 'EMA20', 'Volatilidad']] # Variables
        y = df['Close'] # Objetivo
        
        # 4. Dividir datos (80% entrenar, 20% probar)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # 5. Entrenar Modelo (Linear Regression es rápido y efectivo para tendencias)
        model_ml = LinearRegression()
        model_ml.fit(X_train, y_train)
        
        # 6. Validar precisión
        preds = model_ml.predict(X_test)
        score = r2_score(y_test, preds) * 100 # Precisión %
        
        # 7. PREDECIR MAÑANA
        # Usamos los datos de HOY para predecir MAÑANA
        last_data = pd.DataFrame([[
            df['Close'].iloc[-1], 
            df['RSI'].iloc[-1], 
            df['EMA20'].iloc[-1], 
            df['Volatilidad'].iloc[-1]
        ]], columns=['Lag_Close_1', 'Lag_RSI', 'EMA20', 'Volatilidad'])
        
        future_price = model_ml.predict(last_data)[0]
        
        return future_price, score, df['Close'].iloc[-1]

    except Exception as e:
        st.error(f"Error ML: {e}")
        return None

# --- INTERFAZ PRINCIPAL ---

st.title("🧠 Sistema Quant V30: AI Prediction Core")

# BARRA SUPERIOR
spy_trend, vix_val, market_mood = obtener_macro()
m1, m2, m3, m4 = st.columns(4)
m1.metric("S&P 500", spy_trend)
m2.metric("VIX", f"{vix_val:.2f}", market_mood)
m3.metric("Capital", f"${st.session_state.get('capital', 2000)}") 
m4.metric("IA Engine", "🟢 ONLINE")

st.divider()

col_left, col_right = st.columns([1, 2.5])

with col_left:
    st.subheader("📡 Radar")
    with st.expander("⚙️ Configuración"):
        capital = st.number_input("Capital ($)", 2000, key='capital')
        riesgo = st.slider("Riesgo %", 0.5, 3.0, 1.5)
        if st.button("🔄 Refrescar"): st.cache_data.clear(); st.rerun()

    df_radar = obtener_radar(WATCHLIST)
    
    if df_radar is not None and not df_radar.empty:
        df_radar = df_radar.sort_values("Score", ascending=False)
        selected_ticker = st.selectbox("🔍 Seleccionar Activo:", df_radar['Ticker'].tolist())
        st.dataframe(df_radar[['Ticker', 'Precio', 'Score', 'RSI']].style.format({"Precio": "${:.2f}", "RSI": "{:.0f}"}).background_gradient(subset=['Score'], cmap='RdYlGn'), use_container_width=True, height=400, hide_index=True)
        row = df_radar[df_radar['Ticker'] == selected_ticker].iloc[0]
    else: st.warning("Cargando..."); st.stop()

with col_right:
    st.subheader(f"Analizando: {selected_ticker} | Score: {row['Score']}/100")
    
    # NUEVA PESTAÑA: 🧠 Predicción IA
    tabs = st.tabs(["🧠 Predicción IA (V30)", "📈 Gráfico Sniper", "🔬 Fundamental", "🛡️ Calculadora", "🚀 Señales"])
    
    # TAB 1: PREDICCIÓN ML
    with tabs[0]:
        st.write("🤖 **El Oráculo Matemático:** Entrenando modelo en tiempo real con datos históricos...")
        
        if st.button("🔮 EJECUTAR MODELO PREDICTIVO"):
            with st.spinner(f"Entrenando IA con datos de {selected_ticker}..."):
                resultado = predecir_precio_ia(selected_ticker)
                
                if resultado:
                    pred_price, accuracy, current_price = resultado
                    cambio_pct = ((pred_price - current_price) / current_price) * 100
                    color_pred = "green" if cambio_pct > 0 else "red"
                    
                    st.markdown(f"""
                    <div class="pred-box">
                        <h3>Precio Proyectado (Próximo Cierre)</h3>
                        <h1 style="color:{color_pred};">${pred_price:.2f}</h1>
                        <h4>Cambio esperado: {cambio_pct:+.2f}%</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Precio Actual", f"${current_price:.2f}")
                    c2.metric("Confianza del Modelo (R²)", f"{accuracy:.1f}%")
                    
                    if accuracy < 50:
                        st.warning("⚠️ Precaución: La confianza del modelo es baja (<50%). El mercado está muy errático.")
                    else:
                        st.success("✅ Modelo con confianza aceptable.")
                else:
                    st.error("No se pudo generar la predicción.")

    with tabs[1]:
        fig = graficar_sniper(selected_ticker)
        if fig: st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        if st.button("🔎 Analizar PEG"):
            fund = obtener_fundamental_inferido(selected_ticker)
            if fund:
                c1, c2, c3 = st.columns(3)
                c1.metric("PER", f"{fund['PER']:.2f}x")
                c2.metric("PEG Ratio", f"{fund['PEG']:.2f}", fund['PEG_Source'])
                c3.metric("Target", f"${fund['Target']}")

    with tabs[3]:
        stop_loss = row['Precio'] - (2 * row['ATR'])
        shares = (capital * (riesgo/100)) / (row['Precio'] - stop_loss)
        st.metric("Stop Loss", f"${stop_loss:.2f}")
        st.metric("Comprar", f"{int(shares)} Acciones")

    with tabs[4]:
        if st.button("🚀 ENVIAR SEÑAL"):
            msg = f"🚀 SEÑAL {selected_ticker} | Precio: ${row['Precio']:.2f}"
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
            st.success("Enviado!")
