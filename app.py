import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px 
from plotly.subplots import make_subplots 
import warnings
import numpy as np
import os
import toml
import sqlite3
import math
import feedparser 
import requests 
from datetime import datetime
from scipy.signal import argrelextrema 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import minimize 
from scipy.stats import norm 
import google.generativeai as genai
from fpdf import FPDF 

# --- CONFIGURACIÓN E INICIOS ---
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    HAVE_PYPFOPT = True
except ImportError:
    HAVE_PYPFOPT = False

warnings.filterwarnings('ignore')

try:
    secrets_path = ".streamlit/secrets.toml"
    if os.path.exists(secrets_path):
        secrets = toml.load(secrets_path)
        GOOGLE_API_KEY = secrets["GOOGLE_API_KEY"]
    else:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

st.set_page_config(page_title="Sistema Quant V59 (Architect)", layout="wide", page_icon="📐")
st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 15px; text-align: center;}
    .alert-card-high {background-color: #3d0e0e; border: 1px solid #ff4b4b; padding: 10px; border-radius: 5px; margin-bottom: 5px;}
    .alert-card-med {background-color: #3d3d0e; border: 1px solid #ffa500; padding: 10px; border-radius: 5px; margin-bottom: 5px;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #0e1117; border-radius: 5px;}
    .stTabs [aria-selected="true"] {background-color: #262730;}
</style>""", unsafe_allow_html=True)

WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'COIN', 'KO', 'DIS', 'SPY', 'QQQ', 'GLD', 'USO']
MACRO_DICT = {'S&P 500': 'SPY', 'Nasdaq 100': 'QQQ', 'VIX (Miedo)': '^VIX', 'Bonos 10Y': '^TNX', 'Dólar': 'DX-Y.NYB', 'Petróleo': 'CL=F', 'Oro': 'GC=F'}
DB_NAME = "quant_database.db"

# --- MOTOR SQL ---
def init_db():
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, ticker TEXT, tipo TEXT, cantidad INTEGER, precio REAL, total REAL)''')
    conn.commit(); conn.close()

def registrar_operacion_sql(ticker, tipo, cantidad, precio):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S"); total = cantidad * precio
    c.execute("INSERT INTO trades (fecha, ticker, tipo, cantidad, precio, total) VALUES (?, ?, ?, ?, ?, ?)", (fecha, ticker, tipo, cantidad, precio, total))
    conn.commit(); conn.close(); return True

def auditar_posiciones_sql():
    empty_df = pd.DataFrame(columns=["Ticker", "Cantidad", "Precio Prom.", "Precio Actual", "Valor Mercado", "P&L ($)", "P&L (%)"])
    conn = sqlite3.connect(DB_NAME)
    try: df = pd.read_sql_query("SELECT * FROM trades", conn)
    except: return empty_df
    conn.close()
    if df.empty: return empty_df
    pos = {}
    for idx, row in df.iterrows():
        t = row['ticker']
        if t not in pos: pos[t] = {"Cantidad": 0, "Costo_Total": 0}
        if row['tipo'] == "COMPRA": pos[t]["Cantidad"] += row['cantidad']; pos[t]["Costo_Total"] += row['total']
        elif row['tipo'] == "VENTA":
            pos[t]["Cantidad"] -= row['cantidad']
            if pos[t]["Cantidad"] > 0: unit = pos[t]["Costo_Total"]/(pos[t]["Cantidad"]+row['cantidad']); pos[t]["Costo_Total"] -= (unit*row['cantidad'])
            else: pos[t]["Costo_Total"] = 0
    act = [t for t, d in pos.items() if d['Cantidad'] > 0]
    if not act: return empty_df
    try: curr = yf.download(" ".join(act), period="1d", progress=False, auto_adjust=True)['Close']
    except: return empty_df
    res = []
    for t, d in pos.items():
        if d['Cantidad'] > 0:
            try:
                if len(act) == 1: price = float(curr.iloc[-1])
                else: price = float(curr.iloc[-1][t])
                val = d['Cantidad']*price; pnl = val - d['Costo_Total']
                res.append({"Ticker": t, "Cantidad": d['Cantidad'], "Precio Prom.": d['Costo_Total']/d['Cantidad'], "Precio Actual": price, "Valor Mercado": val, "P&L ($)": pnl, "P&L (%)": (pnl/d['Costo_Total'])*100})
            except: pass
    return pd.DataFrame(res) if res else empty_df

init_db()

# --- MOTOR BACKTEST PRO (NUEVO V59: TP/SL) ---
def ejecutar_backtest_custom(ticker, capital, params):
    """Backtest flexible con Stop Loss y Take Profit"""
    try:
        df = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=True)
        if df.empty: return None
        
        # Indicadores
        df['RSI'] = ta.rsi(df['Close'], 14)
        df['SMA200'] = ta.sma(df['Close'], 200)
        df['BB_Lower'] = ta.bbands(df['Close'], length=20, std=2).iloc[:, 0]
        
        cash = capital
        position = 0
        entry_price = 0
        trade_log = []
        equity_curve = []
        
        # Parámetros desempacados
        rsi_buy = params.get('rsi_buy', 30)
        use_trend = params.get('use_trend', False)
        stop_loss_pct = params.get('sl', 0.0) / 100.0
        take_profit_pct = params.get('tp', 0.0) / 100.0
        
        for i in range(200, len(df)):
            price = df['Close'].iloc[i]
            date = df.index[i]
            rsi = df['RSI'].iloc[i]
            sma200 = df['SMA200'].iloc[i]
            
            # LÓGICA DE COMPRA
            trend_condition = (price > sma200) if use_trend else True
            buy_signal = (rsi < rsi_buy) and trend_condition
            
            if position == 0 and cash > 0 and buy_signal:
                position = cash / price
                entry_price = price
                cash = 0
                trade_log.append({"Fecha": date, "Tipo": "COMPRA", "Precio": price, "Razón": "Señal"})
            
            # LÓGICA DE VENTA (Salida)
            elif position > 0:
                # 1. Stop Loss
                if stop_loss_pct > 0 and price <= entry_price * (1 - stop_loss_pct):
                    cash = position * price
                    position = 0
                    trade_log.append({"Fecha": date, "Tipo": "VENTA", "Precio": price, "Razón": "Stop Loss 🛑"})
                
                # 2. Take Profit
                elif take_profit_pct > 0 and price >= entry_price * (1 + take_profit_pct):
                    cash = position * price
                    position = 0
                    trade_log.append({"Fecha": date, "Tipo": "VENTA", "Precio": price, "Razón": "Take Profit 💰"})
                
                # 3. Salida Técnica (RSI Alto)
                elif rsi > 70:
                    cash = position * price
                    position = 0
                    trade_log.append({"Fecha": date, "Tipo": "VENTA", "Precio": price, "Razón": "RSI Alto"})

            # Registro de Equity
            val = cash if position == 0 else position * price
            equity_curve.append({"Fecha": date, "Equity": val})

        final_equity = cash if position == 0 else position * df['Close'].iloc[-1]
        retorno = ((final_equity - capital) / capital) * 100
        trades_count = len(trade_log)
        
        return {
            "Retorno": retorno,
            "Trades": trades_count,
            "Log": pd.DataFrame(trade_log),
            "Equity": pd.DataFrame(equity_curve).set_index("Fecha")
        }
        
    except Exception as e: return None

# --- MOTORES EXISTENTES ---
@st.cache_data(ttl=600)
def generar_feed_alertas(tickers):
    alertas = []
    try: data = yf.download(" ".join(tickers), period="1y", group_by='ticker', progress=False, auto_adjust=True)
    except: return []
    for t in tickers:
        try:
            df = data[t].dropna() if len(tickers)>1 else data.dropna()
            if len(df)<200: continue
            close=df['Close']; rsi=ta.rsi(close,14).iloc[-1]; s50=ta.sma(close,50); s200=ta.sma(close,200)
            if s50.iloc[-2]<s200.iloc[-2] and s50.iloc[-1]>s200.iloc[-1]: alertas.append({"Ticker":t,"Nivel":"ALTA","Mensaje":"🌟 GOLDEN CROSS"})
            if s50.iloc[-2]>s200.iloc[-2] and s50.iloc[-1]<s200.iloc[-1]: alertas.append({"Ticker":t,"Nivel":"ALTA","Mensaje":"☠️ DEATH CROSS"})
            if rsi<25: alertas.append({"Ticker":t,"Nivel":"MEDIA","Mensaje":f"🟢 Sobreventa RSI {rsi:.0f}"})
            elif rsi>75: alertas.append({"Ticker":t,"Nivel":"MEDIA","Mensaje":f"🔴 Sobrecompra RSI {rsi:.0f}"})
        except: pass
    return alertas

@st.cache_data(ttl=3600)
def obtener_crypto_sentiment():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1")
        data = r.json()
        return int(data['data'][0]['value']), data['data'][0]['value_classification']
    except: return None, None

def graficar_master(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
        if df.empty: return None
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df['EMA20'] = ta.ema(df['Close'], 20); df['RSI'] = ta.rsi(df['Close'], 14)
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        bb = ta.bbands(df['Close'], length=20, std=2); df = pd.concat([df, bb], axis=1)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#FFD700', width=2), name="VWAP"), row=1, col=1)
        try: 
            fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, -3], line=dict(color='cyan', width=1, dash='dot'), name="Upper"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, -1], line=dict(color='cyan', width=1, dash='dot'), name="Lower"), row=1, col=1)
        except: pass
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_color="red", row=2, col=1); fig.add_hline(y=30, line_color="green", row=2, col=1)
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0)); return fig
    except: return None

@st.cache_data(ttl=600)
def generar_mapa_calor(tickers):
    try:
        d = yf.download(" ".join(tickers), period="5d", interval="1d", progress=False, auto_adjust=True)['Close']
        p = ((d.iloc[-1]-d.iloc[-2])/d.iloc[-2])*100
        return pd.DataFrame({'Ticker': p.index, 'Variacion': p.values, 'Sector': 'General', 'Size': d.iloc[-1].values})
    except: return None

@st.cache_data(ttl=3600)
def calcular_valor_intrinseco(ticker):
    try:
        i = yf.Ticker(ticker).info; p = i.get('currentPrice') or i.get('regularMarketPreviousClose') or 0
        e = i.get('trailingEps', 0); b = i.get('bookValue', 0)
        if e and b and e > 0 and b > 0:
            g = math.sqrt(22.5 * e * b); s = "INFRAVALORADA 🟢" if g > p else "SOBREVALORADA 🔴"
            return {"Precio": p, "Graham": g, "Status_Graham": s, "Diff": ((g-p)/p)*100}
    except: pass
    return None

def analizar_fundamental_crypto(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
        if df.empty: return None
        fair = df['Close'].rolling(200).mean().iloc[-1]; curr = df['Close'].iloc[-1]
        status = "SOBRECOMPRADO 🔴" if curr > fair * 1.5 else "ZONA DE ACUMULACIÓN 🟢" if curr < fair else "NEUTRAL 🟡"
        return {"Precio": curr, "FairValue_200SMA": fair, "Status": status}
    except: return None

# --- INTERFAZ V59: STRATEGY ARCHITECT ---
st.title("📐 Sistema Quant V59: Strategy Architect")

# PANEL LATERAL: WATCHTOWER
with st.sidebar:
    st.header("🔔 Watchtower")
    with st.spinner("Escaneando..."): alertas = generar_feed_alertas(WATCHLIST)
    if alertas:
        for a in alertas:
            c = "alert-card-high" if a['Nivel']=="ALTA" else "alert-card-med"
            st.markdown(f"<div class='{c}'><b>{a['Ticker']}</b><br><small>{a['Mensaje']}</small></div>", unsafe_allow_html=True)
    else: st.success("Mercado tranquilo.")

# DASHBOARD KPI
df_pos = auditar_posiciones_sql()
k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Patrimonio", f"${df_pos['Valor Mercado'].sum() if not df_pos.empty else 0:,.2f}")
with k2: st.metric("P&L Total", f"${df_pos['P&L ($)'].sum() if not df_pos.empty else 0:+.2f}")
with k3: st.metric("Alertas", f"{len(alertas)}")
with k4: 
    fg_val, fg_class = obtener_crypto_sentiment()
    if fg_val: st.metric("Crypto Mood", f"{fg_val}/100", f"{fg_class}")

st.divider()

main_tabs = st.tabs(["🧬 LABORATORIO QUANT", "📊 ANÁLISIS HÍBRIDO", "💼 MESA DE DINERO"])

# --- TAB 1: LABORATORIO (NUEVO V59: BACKTEST BUILDER) ---
with main_tabs[0]:
    col_l1, col_l2 = st.columns([1, 2])
    
    with col_l1:
        st.subheader("🛠️ Constructor de Estrategias")
        st.info("Diseña tu algoritmo y pruébalo en el pasado.")
        
        # 1. Selector de Activo
        tk_back = st.selectbox("1. Activo a Probar:", WATCHLIST)
        
        # 2. Configuración de Entrada
        st.markdown("**2. Reglas de Entrada (Compra)**")
        rsi_trigger = st.slider("Comprar si RSI <", 10, 50, 30)
        filter_trend = st.checkbox("Filtro de Tendencia (Solo comprar si Precio > SMA 200)", value=True)
        
        # 3. Configuración de Salida (Gestión de Riesgo)
        st.markdown("**3. Gestión de Salida (Venta)**")
        sl_input = st.number_input("Stop Loss (%)", 0.0, 20.0, 5.0, help="0 para desactivar")
        tp_input = st.number_input("Take Profit (%)", 0.0, 50.0, 15.0, help="0 para desactivar")
        
        if st.button("🚀 EJECUTAR BACKTEST"):
            params = {
                "rsi_buy": rsi_trigger,
                "use_trend": filter_trend,
                "sl": sl_input,
                "tp": tp_input
            }
            with st.spinner("Simulando operaciones..."):
                res_bt = ejecutar_backtest_custom(tk_back, 10000, params)
                st.session_state['res_bt'] = res_bt

    with col_l2:
        if 'res_bt' in st.session_state and st.session_state['res_bt']:
            res = st.session_state['res_bt']
            
            # KPI Resultados
            kb1, kb2, kb3 = st.columns(3)
            kb1.metric("Retorno Total", f"{res['Retorno']:.2f}%", help="Beneficio neto en el periodo")
            kb2.metric("Nº Trades", f"{res['Trades']}")
            kb3.metric("Resultado Final", f"${10000 * (1 + res['Retorno']/100):,.2f}")
            
            # Curva de Equity
            if not res['Equity'].empty:
                st.plotly_chart(px.line(res['Equity'], y="Equity", title="Crecimiento de la Cuenta ($10,000 Iniciales)"), use_container_width=True)
            
            # Log de Operaciones
            with st.expander("📜 Ver Registro de Operaciones (Trade Log)"):
                if not res['Log'].empty:
                    st.dataframe(res['Log'])
                else:
                    st.warning("No se ejecutaron operaciones con estas reglas.")
        else:
            st.info("Configura los parámetros a la izquierda y pulsa 'Ejecutar'.")

# --- TAB 2: ANÁLISIS HÍBRIDO ---
with main_tabs[1]:
    sel_ticker = st.selectbox("Analizar:", WATCHLIST, key="ana_tick")
    ES_CRYPTO = "USD" in sel_ticker and "BTC" in sel_ticker or "ETH" in sel_ticker or "SOL" in sel_ticker
    
    c_a1, c_a2 = st.columns([1, 2])
    with c_a1:
        if ES_CRYPTO:
            st.markdown(f"### 🪙 Cripto: {sel_ticker}")
            cry_data = analizar_fundamental_crypto(sel_ticker)
            if cry_data:
                st.metric("Precio", f"${cry_data['Precio']:,.2f}")
                st.metric("Fair Value (SMA200)", f"${cry_data['FairValue_200SMA']:,.2f}")
                st.info(cry_data['Status'])
        else:
            st.markdown(f"### 🏢 Acciones: {sel_ticker}")
            fund = calcular_valor_intrinseco(sel_ticker)
            if fund:
                st.metric("Valor Graham", f"${fund['Graham']:.2f}", f"{fund['Diff']:.1f}%")
                if "INFRA" in fund['Status_Graham']: st.success("✅ Value Investing")
                else: st.error("❌ Growth Premium")
    
    with c_a2:
        f = graficar_master(sel_ticker)
        if f: st.plotly_chart(f, use_container_width=True)
        
        with st.expander("Mapa de Mercado"):
            df_map = generar_mapa_calor(WATCHLIST)
            if df_map is not None: st.plotly_chart(px.treemap(df_map, path=['Sector', 'Ticker'], values='Size', color='Variacion', color_continuous_scale='RdYlGn', color_continuous_midpoint=0), use_container_width=True)

# --- TAB 3: OPERATIVA ---
with main_tabs[2]:
    c1, c2 = st.columns([2, 1])
    with c1:
        if not df_pos.empty: st.dataframe(df_pos.style.format({"Valor Mercado": "${:.2f}", "P&L ($)": "${:+.2f}"}).background_gradient(subset=['P&L (%)'], cmap='RdYlGn'), use_container_width=True)
        else: st.info("Cartera vacía.")
    with c2:
        with st.form("op"):
            t = st.selectbox("Ticker", WATCHLIST, key="op_tk"); tp = st.selectbox("Tipo", ["COMPRA", "VENTA"])
            q = st.number_input("Qty", 1, 10000); p = st.number_input("Precio", 0.0)
            if st.form_submit_button("Ejecutar"): registrar_operacion_sql(t, tp, q, p); st.rerun()
