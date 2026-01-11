import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots 
import requests 
import google.generativeai as genai
import feedparser
import warnings
import numpy as np
import os
import toml
import re

# --- LIBRERÍAS MATEMÁTICAS ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# Intentamos importar PyPortfolioOpt, si no está, usamos modo fallback
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    HAVE_PYPFOPT = True
except ImportError:
    HAVE_PYPFOPT = False

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
st.set_page_config(page_title="Sistema Quant V31 (Architect)", layout="wide", page_icon="🏛️")
st.markdown("""
<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 10px; color: white;}
    .pred-box {border: 2px solid #4CAF50; padding: 10px; border-radius: 10px; text-align: center; background-color: #1e1e1e;}
    .opt-box {border: 2px solid #00BCD4; padding: 10px; border-radius: 10px; background-color: #1e1e1e;}
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

def predecir_precio_ia(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
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
        if 'Close' not in df.columns: return None

        df['RSI'] = ta.rsi(df['Close'], 14)
        df['EMA20'] = ta.ema(df['Close'], 20)
        df['Return'] = df['Close'].pct_change()
        df['Volatilidad'] = df['Return'].rolling(5).std()
        df['Lag_Close_1'] = df['Close'].shift(1)
        df['Lag_RSI'] = df['RSI'].shift(1)
        df.dropna(inplace=True)

        X = df[['Lag_Close_1', 'Lag_RSI', 'EMA20', 'Volatilidad']]
        y = df['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model_ml = LinearRegression()
        model_ml.fit(X_train, y_train)
        preds = model_ml.predict(X_test)
        score = r2_score(y_test, preds) * 100
        
        last_row = df.iloc[-1]
        last_data = pd.DataFrame([[last_row['Close'], last_row['RSI'], last_row['EMA20'], last_row['Volatilidad']]], columns=['Lag_Close_1', 'Lag_RSI', 'EMA20', 'Volatilidad'])
        future_price = model_ml.predict(last_data)[0]
        return future_price, score, last_row['Close']
    except: return None

# --- MOTOR DE OPTIMIZACIÓN V31 ---
def optimizar_portafolio(tickers_list, capital_invertir):
    if not HAVE_PYPFOPT:
        return None, "Librería PyPortfolioOpt no instalada."
    
    try:
        # Descarga Masiva Blindada
        df = yf.download(tickers_list, period="2y", progress=False, auto_adjust=True)['Close']
        if df.empty: return None, "No hay datos."
        
        # Matemáticas Financieras
        mu = expected_returns.mean_historical_return(df) # Retorno Esperado
        S = risk_models.sample_cov(df) # Matriz de Covarianza (Riesgo)
        
        # Frontera Eficiente (Sharpe Ratio)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe() # Maximizar Retorno/Riesgo
        cleaned_weights = ef.clean_weights()
        
        # Resultados de Performance
        perf = ef.portfolio_performance(verbose=False)
        retorno_esp = perf[0]
        volatilidad_esp = perf[1]
        sharpe = perf[2]
        
        # Asignación Discreta (Cuántas acciones comprar)
        latest_prices = get_latest_prices(df)
        da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=capital_invertir)
        allocation, leftover = da.greedy_portfolio()
        
        return {
            "weights": cleaned_weights,
            "allocation": allocation,
            "metrics": [retorno_esp, volatilidad_esp, sharpe],
            "leftover": leftover
        }, "OK"
        
    except Exception as e:
        return None, str(e)

# --- INTERFAZ ---
st.title("🏛️ Sistema Quant V31: The Architect")

col_left, col_right = st.columns([1, 2.5])

with col_left:
    st.subheader("Radar")
    # Configuración Global
    with st.expander("⚙️ Fondos"):
        capital = st.number_input("Capital Total ($)", 2000, 100000, 10000, step=500, key='capital')
        st.caption("Usado para cálculos de portafolio.")

    if st.button("🔄 Refrescar"): st.cache_data.clear(); st.rerun()
    
    df_radar = obtener_radar(WATCHLIST)
    if df_radar is not None and not df_radar.empty:
        df_radar = df_radar.sort_values("Score", ascending=False)
        selected_ticker = st.selectbox("Activo:", df_radar['Ticker'].tolist())
        st.dataframe(df_radar[['Ticker', 'Score', 'RSI']].style.background_gradient(subset=['Score'], cmap='RdYlGn'), use_container_width=True, height=300)
    else: st.stop()

with col_right:
    # SISTEMA DE PESTAÑAS
    tabs = st.tabs(["⚖️ Optimizador", "🧠 IA Predictiva", "📈 Gráfico", "🔬 Fundamental", "🚀 Señal"])
    
    # TAB 1: OPTIMIZADOR DE PORTAFOLIO (V31)
    with tabs[0]:
        st.subheader("⚖️ Teoría Moderna de Portafolios (Markowitz)")
        st.write("Calcula la combinación matemática perfecta para maximizar ganancias y minimizar riesgo.")
        
        if not HAVE_PYPFOPT:
            st.error("⚠️ Falta la librería `PyPortfolioOpt`. Instálala o este módulo no funcionará.")
            st.code("pip install PyPortfolioOpt")
        
        col_opt1, col_opt2 = st.columns([1, 2])
        with col_opt1:
            assets_to_opt = st.multiselect("Activos a incluir:", WATCHLIST, default=WATCHLIST[:5])
            if st.button("🧮 CALCULAR PORTAFOLIO ÓPTIMO"):
                with st.spinner("Optimizando Frontera Eficiente..."):
                    res_opt, msg_opt = optimizar_portafolio(assets_to_opt, capital)
                    
                    if res_opt:
                        st.session_state['opt_result'] = res_opt
                    else:
                        st.error(f"Error: {msg_opt}")

        with col_opt2:
            if 'opt_result' in st.session_state:
                res = st.session_state['opt_result']
                metrics = res['metrics']
                
                # Métricas del Portafolio
                c1, c2, c3 = st.columns(3)
                c1.metric("Retorno Anual Est.", f"{metrics[0]*100:.1f}%")
                c2.metric("Volatilidad (Riesgo)", f"{metrics[1]*100:.1f}%")
                c3.metric("Sharpe Ratio", f"{metrics[2]:.2f}")
                
                # Gráfico de Torta (Asignación)
                clean_w = {k: v for k, v in res['weights'].items() if v > 0.01}
                fig_pie = px.pie(values=list(clean_w.values()), names=list(clean_w.keys()), title="Asignación de Capital Ideal")
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Lista de Compras
                st.markdown("### 🛒 Lista de Compras (Orden Ejecutiva)")
                alloc_df = pd.DataFrame.from_dict(res['allocation'], orient='index', columns=['Cantidad'])
                st.dataframe(alloc_df)
                st.info(f"💰 Cash sobrante: ${res['leftover']:.2f}")

    # TAB 2: IA PREDICTIVA (V30.1)
    with tabs[1]:
        if st.button("🔮 PREDICCIÓN INDIVIDUAL"):
            res_ia = predecir_precio_ia(selected_ticker)
            if res_ia:
                pred, acc, curr = res_ia
                pct = ((pred - curr)/curr)*100
                st.markdown(f"<div class='pred-box'><h1>${pred:.2f}</h1><p>{pct:+.2f}% vs Hoy</p></div>", unsafe_allow_html=True)
                st.metric("Confianza R²", f"{acc:.1f}%")

    with tabs[2]:
        fig = graficar_sniper(selected_ticker)
        if fig: st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        fund = obtener_fundamental_inferido(selected_ticker)
        if fund: st.json(fund)

    with tabs[4]:
        if st.button("🚀 Alertar"):
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": f"SEÑAL {selected_ticker}"})
