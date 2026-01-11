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

# --- LIBRERÍAS ML ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
st.set_page_config(page_title="Sistema Quant V30.1 (Fixed)", layout="wide", page_icon="🧠")
st.markdown("""
<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 10px; color: white;}
    .big-font {font-size:18px !important; font-weight: bold;}
    .pred-box {border: 2px solid #4CAF50; padding: 10px; border-radius: 10px; text-align: center; background-color: #1e1e1e;}
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
        target = info.get('targetMeanPrice') or 0
        
        peg_final = 0
        peg_source = "N/A"
        if peg_oficial is not None:
            peg_final = peg_oficial
            peg_source = "Yahoo"
        elif pe > 0 and growth_est > 0:
            peg_final = pe / (growth_est * 100)
            peg_source = "Estimado"
        
        return {"PER": pe, "PEG": peg_final, "PEG_Source": peg_source, "Target": target}
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

# --- MOTOR ML CORREGIDO (V30.1) ---
def predecir_precio_ia(ticker):
    try:
        # 1. Descarga Blindada
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        
        # Limpieza MultiIndex (IGUAL QUE EN EL GRÁFICO)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                if ticker in df.columns.levels[0]: df = df[ticker].copy()
                else: df.columns = df.columns.get_level_values(-1)
            except: df.columns = df.columns.get_level_values(-1)
        
        # Renombrado de emergencia
        if 'Close' not in df.columns:
            if df.shape[1] >= 4:
                # Asumimos Open, High, Low, Close...
                cols = list(df.columns)
                found = False
                for c in cols:
                    if "Close" in str(c): df.rename(columns={c: 'Close'}, inplace=True); found=True; break
                if not found: df.columns = ["Open", "High", "Low", "Close", "Volume"][:df.shape[1]]
        
        if 'Close' not in df.columns: return None

        # 2. Ingeniería
        df['RSI'] = ta.rsi(df['Close'], 14)
        df['EMA20'] = ta.ema(df['Close'], 20)
        df['Return'] = df['Close'].pct_change()
        df['Volatilidad'] = df['Return'].rolling(5).std()
        
        df['Lag_Close_1'] = df['Close'].shift(1)
        df['Lag_RSI'] = df['RSI'].shift(1)
        
        df.dropna(inplace=True)

        # 3. ML
        X = df[['Lag_Close_1', 'Lag_RSI', 'EMA20', 'Volatilidad']]
        y = df['Close']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model_ml = LinearRegression()
        model_ml.fit(X_train, y_train)
        
        preds = model_ml.predict(X_test)
        score = r2_score(y_test, preds) * 100
        
        # 4. Predicción Futura
        last_row = df.iloc[-1]
        last_data = pd.DataFrame([[
            last_row['Close'], 
            last_row['RSI'], 
            last_row['EMA20'], 
            last_row['Volatilidad']
        ]], columns=['Lag_Close_1', 'Lag_RSI', 'EMA20', 'Volatilidad'])
        
        future_price = model_ml.predict(last_data)[0]
        
        return future_price, score, last_row['Close']

    except Exception as e:
        st.error(f"Error ML: {e}") # Debug visible
        return None

# --- INTERFAZ ---
st.title("🧠 Sistema Quant V30.1: AI Core")

col_left, col_right = st.columns([1, 2.5])

with col_left:
    st.subheader("Radar")
    if st.button("🔄 Refrescar"): st.cache_data.clear(); st.rerun()
    
    df_radar = obtener_radar(WATCHLIST)
    if df_radar is not None and not df_radar.empty:
        df_radar = df_radar.sort_values("Score", ascending=False)
        selected_ticker = st.selectbox("Activo:", df_radar['Ticker'].tolist())
        st.dataframe(df_radar[['Ticker', 'Precio', 'Score', 'RSI']].style.format({"Precio": "${:.2f}", "RSI": "{:.0f}"}).background_gradient(subset=['Score'], cmap='RdYlGn'), use_container_width=True, height=400, hide_index=True)
        row = df_radar[df_radar['Ticker'] == selected_ticker].iloc[0]
    else: st.warning("Cargando..."); st.stop()

with col_right:
    st.subheader(f"Analizando: {selected_ticker}")
    
    tabs = st.tabs(["🧠 Predicción IA", "📈 Gráfico", "🔬 Fundamental", "🛡️ Calc", "🚀 Señal"])
    
    with tabs[0]:
        st.info("🤖 **Modelo Predictivo:** Regresión Lineal entrenada en tiempo real (2 años de historia).")
        
        if st.button("🔮 EJECUTAR MODELO"):
            with st.spinner(f"Entrenando IA para {selected_ticker}..."):
                resultado = predecir_precio_ia(selected_ticker)
                
                if resultado:
                    pred_price, accuracy, current_price = resultado
                    cambio_pct = ((pred_price - current_price) / current_price) * 100
                    color_pred = "#00ff00" if cambio_pct > 0 else "#ff0000"
                    
                    st.markdown(f"""
                    <div class="pred-box">
                        <h3>Proyección Próximo Cierre</h3>
                        <h1 style="color:{color_pred};">${pred_price:.2f}</h1>
                        <h4>Cambio esperado: {cambio_pct:+.2f}%</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Precio Base (Hoy)", f"${current_price:.2f}")
                    c2.metric("Precisión del Modelo (R²)", f"{accuracy:.1f}%")
                else:
                    st.error("Error generando predicción. Intenta otro activo.")

    with tabs[1]:
        fig = graficar_sniper(selected_ticker)
        if fig: st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        fund = obtener_fundamental_inferido(selected_ticker)
        if fund:
            c1, c2 = st.columns(2)
            c1.metric("PER", f"{fund['PER']:.2f}")
            c2.metric("PEG", f"{fund['PEG']:.2f}", fund['PEG_Source'])
            st.caption(f"Target: ${fund['Target']}")

    with tabs[3]:
        stop = row['Precio'] - (2*row['ATR'])
        shares = (2000 * 0.015) / (row['Precio'] - stop)
        st.metric("Stop Loss", f"${stop:.2f}")
        st.metric("Comprar", f"{int(shares)} Acciones")
    
    with tabs[4]:
        if st.button("🚀 Alertar Telegram"):
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": f"SEÑAL {selected_ticker}"})
            st.success("Hecho")
