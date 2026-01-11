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
from datetime import datetime
from scipy.signal import argrelextrema 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import minimize 
from scipy.stats import norm 
import google.generativeai as genai
from fpdf import FPDF 

# --- CONFIGURACIÓN MOTOR HÍBRIDO ---
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

st.set_page_config(page_title="Sistema Quant V49 (Genetic Lab)", layout="wide", page_icon="🧬")
st.markdown("""<style>.metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 10px; color: white;} .signal-box {border: 2px solid #FFD700; padding: 10px; border-radius: 5px; background-color: #2b2b00; text-align: center;} .trade-buy {color: #00ff00; font-weight: bold;} .trade-sell {color: #ff0000; font-weight: bold;}</style>""", unsafe_allow_html=True)

try: genai.configure(api_key=GOOGLE_API_KEY); model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'COIN', 'KO', 'DIS', 'SPY', 'QQQ', 'DIA', 'GLD', 'USO']
DB_NAME = "quant_database.db"

# --- MOTOR SQL (CORREGIDO PARA EVITAR ERROR ATTRIBUTEERROR) ---
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
    # 🛠️ FIX V49: Devolver siempre columnas, aunque esté vacío
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
            
    res = []; act = [t for t, d in pos.items() if d['Cantidad'] > 0]
    if not act: return empty_df
    
    try: curr = yf.download(" ".join(act), period="1d", progress=False, auto_adjust=True)['Close']
    except: return empty_df

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

# --- MOTOR DE OPTIMIZACIÓN GENÉTICA (GRID SEARCH) V49 ---
def optimizar_parametros_estrategia(ticker, estrategia="RSI"):
    """Prueba múltiples combinaciones de parámetros para encontrar la mejor"""
    try:
        # Descarga única de datos para eficiencia
        df_base = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=True)
        if df_base.empty: return None
        if df_base.index.tz is not None: df_base.index = df_base.index.tz_localize(None)
        
        resultados = []
        
        if estrategia == "RSI":
            # Grid Search: Probamos Buy (20, 25, 30, 35) vs Sell (65, 70, 75, 80)
            df_base['RSI'] = ta.rsi(df_base['Close'], 14)
            buy_range = [20, 25, 30, 35]
            sell_range = [65, 70, 75, 80]
            
            for b in buy_range:
                for s in sell_range:
                    # Simulación rápida vectorial
                    # (Simplificación para velocidad: no gestiona cash path-dependent exacto, solo señales)
                    # Para precisión usamos lógica de iteración rápida
                    cash = 10000; pos = 0
                    buy_sig = df_base['RSI'] < b
                    sell_sig = df_base['RSI'] > s
                    
                    # Backtest Rápido
                    for i in range(15, len(df_base)):
                        p = df_base['Close'].iloc[i]
                        if cash > 0 and buy_sig.iloc[i]:
                            pos = cash / p; cash = 0
                        elif pos > 0 and sell_sig.iloc[i]:
                            cash = pos * p; pos = 0
                            
                    final = cash + (pos * df_base['Close'].iloc[-1])
                    ret = ((final - 10000) / 10000) * 100
                    resultados.append({"Compra <": b, "Venta >": s, "Retorno %": ret})
                    
        return pd.DataFrame(resultados)
        
    except Exception as e: return None

# --- MOTORES EXISTENTES (Rebalanceo, etc) ---
def generar_ordenes_rebalanceo(df_actual, pesos_objetivo, capital_total):
    ordenes = []
    # FIX V49: Si df_actual viene vacío pero con columnas, no falla
    if df_actual.empty: return pd.DataFrame()
    
    estado_actual = dict(zip(df_actual['Ticker'], df_actual['Valor Mercado']))
    precios_actuales = dict(zip(df_actual['Ticker'], df_actual['Valor Mercado'] / df_actual['Cantidad']))
    
    for ticker, peso_target in pesos_objetivo.items():
        if peso_target <= 0.01: continue 
        valor_objetivo = capital_total * peso_target
        valor_actual = estado_actual.get(ticker, 0)
        diferencia = valor_objetivo - valor_actual
        try:
            if ticker in precios_actuales: precio = precios_actuales[ticker]
            else: precio = float(yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1])
            cantidad_ajuste = int(diferencia / precio)
            if cantidad_ajuste != 0:
                tipo = "COMPRA" if cantidad_ajuste > 0 else "VENTA"
                ordenes.append({"Ticker": ticker, "Acción": tipo, "Cantidad": abs(cantidad_ajuste), "Precio Est.": precio, "Valor Ajuste": abs(cantidad_ajuste * precio)})
        except: pass
    return pd.DataFrame(ordenes)

def optimizar_portafolio_simple(tickers, capital):
    try:
        df = yf.download(tickers, period="1y", progress=False, auto_adjust=True)['Close']
        if df.empty: return None
        log_ret = np.log(df/df.shift(1))
        def neg_sharpe(w):
            w = np.array(w); ret = np.sum(log_ret.mean() * w) * 252; vol = np.sqrt(np.dot(w.T, np.dot(log_ret.cov() * 252, w)))
            return -(ret/vol) if vol > 0 else 0
        cons = ({'type':'eq','fun': lambda x: np.sum(x) - 1}); bounds = tuple((0, 1) for _ in range(len(tickers))); init = [1/len(tickers)] * len(tickers)
        opt = minimize(neg_sharpe, init, method='SLSQP', bounds=bounds, constraints=cons)
        return dict(zip(tickers, opt.x))
    except: return None

def graficar_master(t):
    try:
        df=yf.Ticker(t).history(period="1y", auto_adjust=True)
        fig=go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False); return fig
    except: return None

def analizar_noticias_ia(t): return {"score":0, "summary":"-", "headlines":[]}
@st.cache_data(ttl=3600)
def calcular_valor_intrinseco(t): return None
def simular_montecarlo(t, d=30, s=500): return None, None
@st.cache_data(ttl=300)
def escanear_oportunidades(ts):
    s=[]; d=yf.download(" ".join(ts), period="3mo", progress=False, group_by='ticker', auto_adjust=True)
    for t in ts:
        try:
            df=d[t].dropna() if len(ts)>1 else d.dropna(); c=df['Close'].iloc[-1]; r=ta.rsi(df['Close'],14).iloc[-1]
            if r<30: s.append({"Ticker":t, "Señal":"COMPRA RSI 🟢"})
            elif r>70: s.append({"Ticker":t, "Señal":"VENTA RSI 🔴"})
        except:pass
    return pd.DataFrame(s)

# --- INTERFAZ ---
st.title("🧬 Sistema Quant V49: Genetic Lab")

# 1. GESTIÓN
col_p1, col_p2 = st.columns([1.5, 1])
with col_p1:
    st.subheader("🏦 Portafolio")
    df_pos = auditar_posiciones_sql() # Ahora es seguro llamarla
    if not df_pos.empty:
        equity_total = df_pos['Valor Mercado'].sum()
        st.metric("Equity", f"${equity_total:,.2f}")
        st.dataframe(df_pos[['Ticker', 'Cantidad', 'Valor Mercado', 'P&L (%)']])
    else: st.info("Sin posiciones abiertas.")

with col_p2:
    with st.expander("📝 Operar Manual", expanded=False):
        t_op = st.selectbox("Activo", WATCHLIST)
        tipo = st.selectbox("Orden", ["COMPRA", "VENTA"])
        qty = st.number_input("Cant", 1, 1000); px = st.number_input("Precio", 0.0)
        if st.button("Ejecutar"): registrar_operacion_sql(t_op, tipo, qty, px); st.rerun()

st.divider()

# 2. LABORATORIO GENÉTICO (NUEVO V49)
c_l, c_r = st.columns([1, 2.5])
with c_l: 
    tk = st.selectbox("Analizar Ticker:", WATCHLIST)
    capital_total_real = 10000.0 if df_pos.empty else df_pos['Valor Mercado'].sum() + 1000

with c_r: 
    tabs = st.tabs(["🧬 Optimizar Parámetros", "⚖️ Rebalanceo", "📈 Gráfico"])
    
    # PESTAÑA 1: GRID SEARCH (V49)
    with tabs[0]:
        st.subheader(f"🧬 Búsqueda de la Estrategia Perfecta: {tk}")
        st.write("El sistema probará automáticamente múltiples combinaciones de RSI para encontrar la más rentable históricamente.")
        
        if st.button("🚀 INICIAR GRID SEARCH"):
            with st.spinner(f"Simulando 16 escenarios para {tk}..."):
                res_grid = optimizar_parametros_estrategia(tk, "RSI")
                
                if res_grid is not None and not res_grid.empty:
                    # Encontrar el mejor
                    mejor = res_grid.loc[res_grid['Retorno %'].idxmax()]
                    
                    c1, c2 = st.columns(2)
                    c1.success(f"🏆 MEJOR CONFIGURACIÓN: Compra < {mejor['Compra <']} | Venta > {mejor['Venta >']}")
                    c1.metric("Retorno Máximo", f"{mejor['Retorno %']:.2f}%")
                    
                    # Mapa de Calor
                    fig_heat = px.density_heatmap(
                        res_grid, x="Compra <", y="Venta >", z="Retorno %", 
                        text_auto=True, title="Mapa de Rentabilidad (RSI)",
                        color_continuous_scale="RdYlGn"
                    )
                    c2.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.error("No hay datos suficientes.")

    # PESTAÑA 2: REBALANCEO (V48 FIX)
    with tabs[1]:
        if not df_pos.empty:
            activos_target = st.multiselect("Target Allocation:", WATCHLIST, default=df_pos['Ticker'].unique().tolist())
            if st.button("🔄 CALCULAR"):
                pesos = optimizar_portafolio_simple(activos_target, capital_total_real)
                if pesos:
                    ordenes = generar_ordenes_rebalanceo(df_pos, pesos, capital_total_real)
                    st.dataframe(ordenes)
                    
                    # Graficar Torta SOLO si hay datos
                    fig_act = px.pie(df_pos, values='Valor Mercado', names='Ticker', title="Actual")
                    st.plotly_chart(fig_act, use_container_width=True)
        else: st.warning("Necesitas posiciones para rebalancear.")

    with tabs[2]:
        fig = graficar_master(tk)
        if fig: st.plotly_chart(fig, use_container_width=True)