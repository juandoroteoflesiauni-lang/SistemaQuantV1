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

st.set_page_config(page_title="Sistema Quant V48 (Rebalancer)", layout="wide", page_icon="⚖️")
st.markdown("""<style>.metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 10px; color: white;} .signal-box {border: 2px solid #FFD700; padding: 10px; border-radius: 5px; background-color: #2b2b00; text-align: center;} .trade-buy {color: #00ff00; font-weight: bold;} .trade-sell {color: #ff0000; font-weight: bold;}</style>""", unsafe_allow_html=True)

try: genai.configure(api_key=GOOGLE_API_KEY); model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'COIN', 'KO', 'DIS', 'SPY', 'QQQ', 'DIA', 'GLD', 'USO']
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
    conn = sqlite3.connect(DB_NAME)
    try: df = pd.read_sql_query("SELECT * FROM trades", conn)
    except: return pd.DataFrame()
    conn.close()
    if df.empty: return pd.DataFrame()
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
    if not act: return pd.DataFrame()
    try: curr = yf.download(" ".join(act), period="1d", progress=False, auto_adjust=True)['Close']
    except: return pd.DataFrame()
    for t, d in pos.items():
        if d['Cantidad'] > 0:
            try:
                if len(act) == 1: price = float(curr.iloc[-1])
                else: price = float(curr.iloc[-1][t])
                val = d['Cantidad']*price; pnl = val - d['Costo_Total']
                res.append({"Ticker": t, "Cantidad": d['Cantidad'], "Valor Mercado": val, "P&L ($)": pnl, "P&L (%)": (pnl/d['Costo_Total'])*100})
            except: pass
    return pd.DataFrame(res)

init_db()

# --- MOTOR DE REBALANCEO (NUEVO V48) ---
def generar_ordenes_rebalanceo(df_actual, pesos_objetivo, capital_total):
    """Calcula las órdenes exactas para alinear la cartera"""
    ordenes = []
    
    # 1. Crear mapa de estado actual
    estado_actual = dict(zip(df_actual['Ticker'], df_actual['Valor Mercado']))
    precios_actuales = dict(zip(df_actual['Ticker'], df_actual['Valor Mercado'] / df_actual['Cantidad']))
    
    # 2. Iterar sobre objetivos
    for ticker, peso_target in pesos_objetivo.items():
        if peso_target <= 0.01: continue # Ignorar pesos insignificantes
        
        valor_objetivo = capital_total * peso_target
        valor_actual = estado_actual.get(ticker, 0)
        diferencia = valor_objetivo - valor_actual
        
        # Necesitamos el precio para saber cuantas acciones son
        try:
            if ticker in precios_actuales:
                precio = precios_actuales[ticker]
            else:
                precio = float(yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1])
            
            cantidad_ajuste = int(diferencia / precio)
            
            if cantidad_ajuste != 0:
                tipo = "COMPRA" if cantidad_ajuste > 0 else "VENTA"
                ordenes.append({
                    "Ticker": ticker,
                    "Acción": tipo,
                    "Cantidad": abs(cantidad_ajuste),
                    "Precio Est.": precio,
                    "Valor Ajuste": abs(cantidad_ajuste * precio),
                    "Peso Actual": f"{(valor_actual/capital_total)*100:.1f}%",
                    "Peso Meta": f"{peso_target*100:.1f}%"
                })
        except: pass
        
    return pd.DataFrame(ordenes)

def optimizar_portafolio_simple(tickers, capital):
    """Versión simplificada de optimización para rebalanceo"""
    try:
        df = yf.download(tickers, period="1y", progress=False, auto_adjust=True)['Close']
        if df.empty: return None
        log_ret = np.log(df/df.shift(1))
        
        # Función a minimizar (Sharpe Negativo)
        def neg_sharpe(w):
            w = np.array(w)
            ret = np.sum(log_ret.mean() * w) * 252
            vol = np.sqrt(np.dot(w.T, np.dot(log_ret.cov() * 252, w)))
            return -(ret/vol) if vol > 0 else 0

        cons = ({'type':'eq','fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(tickers)))
        init = [1/len(tickers)] * len(tickers)
        
        opt = minimize(neg_sharpe, init, method='SLSQP', bounds=bounds, constraints=cons)
        return dict(zip(tickers, opt.x))
    except: return None

# --- MOTORES EXISTENTES (Resumidos) ---
def analizar_noticias_ia(t): return {"score":0, "summary":"-", "headlines":[]}
@st.cache_data(ttl=3600)
def calcular_valor_intrinseco(t): 
    try: i=yf.Ticker(t).info; return {"Precio": i.get('currentPrice',0), "Graham": math.sqrt(22.5*i.get('trailingEps',0)*i.get('bookValue',0)) if i.get('trailingEps') and i.get('bookValue') else 0}
    except: return None
def calcular_alpha_beta(t, b='SPY'): return None, None
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
def graficar_master(t):
    try:
        df=yf.Ticker(t).history(period="1y", auto_adjust=True)
        fig=go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False); return fig
    except: return None

# --- INTERFAZ ---
st.title("⚖️ Sistema Quant V48: The Rebalancer")

# 1. GESTIÓN DE CARTERA
col_p1, col_p2 = st.columns([1.5, 1])

with col_p1:
    st.subheader("🏦 Estado Actual")
    df_pos = auditar_posiciones_sql()
    
    if not df_pos.empty:
        equity_total = df_pos['Valor Mercado'].sum()
        cash = st.number_input("Efectivo Disponible (Cash)", 0.0, 1000000.0, 1000.0, step=100.0)
        capital_total_real = equity_total + cash
        
        st.metric("Capital Total (Acciones + Cash)", f"${capital_total_real:,.2f}")
        st.dataframe(df_pos[['Ticker', 'Cantidad', 'Valor Mercado', 'P&L (%)']])
    else:
        st.info("Registra operaciones para activar el rebalanceo.")
        capital_total_real = 10000.0 # Default para pruebas

with col_p2:
    with st.expander("📝 Operar Manual", expanded=False):
        t_op = st.selectbox("Activo", WATCHLIST)
        tipo = st.selectbox("Orden", ["COMPRA", "VENTA"])
        qty = st.number_input("Cant", 1, 1000)
        px = st.number_input("Precio", 0.0)
        if st.button("Ejecutar"): registrar_operacion_sql(t_op, tipo, qty, px); st.rerun()

st.divider()

# 2. MÓDULO REBALANCEO (NUEVO V48)
st.subheader("⚖️ Asistente de Rebalanceo")
tabs_rebal = st.tabs(["Generar Plan", "Configuración"])

with tabs_rebal[0]:
    if not df_pos.empty:
        col_reb1, col_reb2 = st.columns([1, 2])
        
        with col_reb1:
            st.write("Selecciona los activos que deseas en tu portafolio ideal:")
            activos_target = st.multiselect("Target Allocation:", WATCHLIST, default=df_pos['Ticker'].unique().tolist())
            
            if st.button("🔄 CALCULAR AJUSTES"):
                if activos_target:
                    with st.spinner("Optimizando pesos matemáticos..."):
                        # 1. Obtener Pesos Óptimos (Markowitz)
                        pesos_optimos = optimizar_portafolio_simple(activos_target, capital_total_real)
                        
                        if pesos_optimos:
                            # 2. Calcular Diferencias
                            df_ordenes = generar_ordenes_rebalanceo(df_pos, pesos_optimos, capital_total_real)
                            st.session_state['ordenes_rebal'] = df_ordenes
                            st.session_state['pesos_optimos'] = pesos_optimos
        
        with col_reb2:
            if 'ordenes_rebal' in st.session_state:
                st.markdown("### 📋 Lista de Órdenes Sugeridas")
                df_ord = st.session_state['ordenes_rebal']
                
                if not df_ord.empty:
                    # Formato visual
                    def color_accion(val):
                        return 'color: #00ff00' if val == "COMPRA" else 'color: #ff0000'
                    
                    st.dataframe(df_ord.style.map(color_accion, subset=['Acción']), use_container_width=True)
                    
                    st.success(f"Si ejecutas estas órdenes, tu portafolio quedará matemáticamente optimizado (Sharpe Ratio Máximo).")
                    
                    # Visualización Torta (Actual vs Ideal)
                    c_pie1, c_pie2 = st.columns(2)
                    with c_pie1:
                        fig_act = px.pie(df_pos, values='Valor Mercado', names='Ticker', title="Actual")
                        st.plotly_chart(fig_act, use_container_width=True)
                    with c_pie2:
                        # Preparar datos pie ideal
                        ideal_data = pd.DataFrame(list(st.session_state['pesos_optimos'].items()), columns=['Ticker', 'Peso'])
                        fig_ideal = px.pie(ideal_data, values='Peso', names='Ticker', title="Ideal (Meta)")
                        st.plotly_chart(fig_ideal, use_container_width=True)
                else:
                    st.success("✅ Tu portafolio ya está perfectamente equilibrado. No se requieren cambios.")
    else:
        st.warning("Necesitas tener posiciones abiertas para usar el rebalanceador.")

# 3. ANALISIS INDIVIDUAL (Simplificado)
st.divider()
c_l, c_r = st.columns([1, 2.5])
with c_l: tk = st.selectbox("Analizar Ticker:", WATCHLIST)
with c_r: 
    fig = graficar_master(tk)
    if fig: st.plotly_chart(fig, use_container_width=True)