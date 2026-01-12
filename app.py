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

st.set_page_config(page_title="Sistema Quant V53 (Architect)", layout="wide", page_icon="🏗️")
st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 15px; text-align: center;}
    .whale-alert {background-color: #4a148c; border: 1px solid #ea00ff; padding: 10px; border-radius: 5px; text-align: center; color: white; font-weight: bold;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #0e1117; border-radius: 5px;}
    .stTabs [aria-selected="true"] {background-color: #262730;}
</style>""", unsafe_allow_html=True)

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

# --- MOTOR PORTFOLIO ARCHITECT (NUEVO V53) ---
def simular_cartera_historica(tickers, pesos, periodo="1y", benchmark="SPY"):
    """Simula el rendimiento de una cartera ponderada vs SPY"""
    try:
        # Descargar datos de todos los activos + benchmark
        todos_tickers = tickers + [benchmark]
        data = yf.download(" ".join(todos_tickers), period=periodo, progress=False, auto_adjust=True)['Close']
        
        if data.empty: return None, None
        
        # Calcular retornos diarios
        retornos = data.pct_change().dropna()
        
        # Retorno del Benchmark acumulado
        bench_ret = (1 + retornos[benchmark]).cumprod()
        bench_ret = (bench_ret / bench_ret.iloc[0]) * 100 # Base 100
        
        # Retorno de la Cartera Ponderada
        # Matriz de retornos (solo tickers cartera) * vector de pesos
        # Aseguramos orden correcto de columnas
        retornos_cartera = retornos[tickers].dot(list(pesos.values()))
        port_ret = (1 + retornos_cartera).cumprod()
        port_ret = (port_ret / port_ret.iloc[0]) * 100 # Base 100
        
        # Métricas Finales
        cagr_port = ((port_ret.iloc[-1] / port_ret.iloc[0]) ** (252/len(port_ret)) - 1) * 100
        cagr_bench = ((bench_ret.iloc[-1] / bench_ret.iloc[0]) ** (252/len(bench_ret)) - 1) * 100
        
        vol_port = retornos_cartera.std() * np.sqrt(252) * 100
        sharpe = (cagr_port - 4.0) / vol_port # Asumiendo 4% tasa libre riesgo
        
        df_chart = pd.DataFrame({
            "Mi Cartera": port_ret,
            "Mercado (SPY)": bench_ret
        })
        
        stats = {
            "CAGR Cartera": cagr_port,
            "CAGR Mercado": cagr_bench,
            "Volatilidad": vol_port,
            "Sharpe": sharpe,
            "Alpha": cagr_port - cagr_bench
        }
        
        return df_chart, stats
        
    except Exception as e: return None, str(e)

# --- MOTORES EXISTENTES ---
def detectar_actividad_ballenas(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1mo", interval="1d", auto_adjust=True)
        if df.empty: return None
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['VolSMA'] = df['Volume'].rolling(20).mean()
        last_vol = df['Volume'].iloc[-1]; last_avg = df['VolSMA'].iloc[-1]
        alerta = "🐋 ALERTA BALLENA" if last_vol > last_avg * 1.5 else None
        trend = "ALCISTA (Inst.)" if df['Close'].iloc[-1] > df['VWAP'].iloc[-1] else "BAJISTA (Inst.)"
        return {"Volumen Hoy": last_vol, "Ratio Vol": last_vol/last_avg, "Alerta": alerta, "VWAP": df['VWAP'].iloc[-1], "Tendencia": trend}
    except: return None

@st.cache_data(ttl=600)
def calcular_score_quant(ticker):
    score=0; b={"Técnico":0, "Fundamental":0, "Riesgo":0}
    try:
        h = yf.Ticker(ticker).history(period="6mo", interval="1d", auto_adjust=True)
        if not h.empty:
            h['RSI']=ta.rsi(h['Close'],14); h['SMA']=ta.sma(h['Close'],50)
            if 30<=h['RSI'].iloc[-1]<=65: b['Técnico']+=20
            elif h['RSI'].iloc[-1]<30: b['Técnico']+=15
            elif h['RSI'].iloc[-1]>70: b['Técnico']+=5
            if h['Close'].iloc[-1]>h['SMA'].iloc[-1]: b['Técnico']+=20
        i = yf.Ticker(ticker).info
        p = i.get('currentPrice',0); e = i.get('trailingEps',0); bk = i.get('bookValue',0)
        if e and bk and e>0 and bk>0:
            g = math.sqrt(22.5*e*bk)
            if p<g: b['Fundamental']+=20
            elif p<g*1.2: b['Fundamental']+=10
        if e>0: b['Fundamental']+=20
        if not e and "USD" in ticker: b['Fundamental']=20
        be = i.get('beta',1.0)
        if be is None: be=1.0
        if 0.8<=be<=1.2: b['Riesgo']+=20
        elif be<0.8: b['Riesgo']+=15
        else: b['Riesgo']+=10
        score = sum(b.values())
        return score, b
    except: return 0, b

def dibujar_velocimetro(score):
    return go.Figure(go.Indicator(mode="gauge+number", value=score, domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "white"},
               'steps': [{'range': [0, 40], 'color': "#ff4b4b"}, {'range': [40, 70], 'color': "#ffa500"}, {'range': [70, 100], 'color': "#00cc96"}]})).update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="#0e1117", font={'color': "white"})

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
        if 'EMA20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='yellow', width=1), name="EMA"), row=1, col=1)
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

def optimizar_parametros_estrategia(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
        if df.empty: return pd.DataFrame()
        df['RSI'] = ta.rsi(df['Close'], 14)
        r = []
        for b in [20, 30, 40]:
            for s in [60, 70, 80]:
                c = 10000; p = 0
                buy = df['RSI'] < b; sell = df['RSI'] > s
                for i in range(15, len(df)):
                    pr = df['Close'].iloc[i]
                    if c > 0 and buy.iloc[i]: p = c/pr; c = 0
                    elif p > 0 and sell.iloc[i]: c = p*pr; p = 0
                r.append({"Compra <": b, "Venta >": s, "Retorno %": ((c + (p * df['Close'].iloc[-1]) - 10000)/10000)*100})
        return pd.DataFrame(r)
    except: return pd.DataFrame()

# --- INTERFAZ V53: ARCHITECT ---
st.title("🏗️ Sistema Quant V53: The Architect")

# DASHBOARD KPI
df_pos = auditar_posiciones_sql()
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
with col_kpi1: st.metric("Patrimonio", f"${df_pos['Valor Mercado'].sum() if not df_pos.empty else 0:,.2f}")
with col_kpi2: st.metric("P&L Total", f"${df_pos['P&L ($)'].sum() if not df_pos.empty else 0:+.2f}")
with col_kpi3: st.metric("Mercado (SPY)", f"${yf.Ticker('SPY').history(period='1d')['Close'].iloc[-1]:.2f}")
with col_kpi4: 
    w = detectar_actividad_ballenas(WATCHLIST[0])
    st.metric("Alerta Ballena", "Activa" if w and w['Alerta'] else "Ninguna")

st.divider()

main_tabs = st.tabs(["💼 MESA DE DINERO", "📊 ANÁLISIS 360", "🧬 LABORATORIO QUANT"])

# --- TAB 1: OPERATIVA ---
with main_tabs[0]:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Cartera")
        if not df_pos.empty: st.dataframe(df_pos.style.format({"Valor Mercado": "${:.2f}", "P&L ($)": "${:+.2f}"}).background_gradient(subset=['P&L (%)'], cmap='RdYlGn'), use_container_width=True)
        else: st.info("Sin posiciones.")
    with c2:
        with st.form("op"):
            t = st.selectbox("Ticker", WATCHLIST); tp = st.selectbox("Tipo", ["COMPRA", "VENTA"])
            q = st.number_input("Qty", 1, 10000); p = st.number_input("Precio", 0.0)
            if st.form_submit_button("Ejecutar"): registrar_operacion_sql(t, tp, q, p); st.rerun()

# --- TAB 2: ANÁLISIS ---
with main_tabs[1]:
    sel_ticker = st.selectbox("Analizar:", WATCHLIST)
    c_a1, c_a2 = st.columns([1, 2])
    with c_a1:
        s, b = calcular_score_quant(sel_ticker)
        st.plotly_chart(dibujar_velocimetro(s), use_container_width=True)
        w = detectar_actividad_ballenas(sel_ticker)
        if w: st.metric("Volumen Hoy", f"{w['Volumen Hoy']/1000:.0f}K", f"{w['Ratio Vol']:.1f}x Promedio")
    with c_a2:
        f = graficar_master(sel_ticker)
        if f: st.plotly_chart(f, use_container_width=True)

# --- TAB 3: LABORATORIO (NUEVO V53) ---
with main_tabs[2]:
    sub_tabs = st.tabs(["🏗️ Backtest Cartera", "🧬 Optimización Genética"])
    
    # BACKTESTER DE CARTERA MULTI-ACTIVO (V53)
    with sub_tabs[0]:
        st.subheader("🏗️ Simulador de Rendimiento de Cartera")
        st.write("Compara una cartera hipotética contra el S&P 500.")
        
        col_b1, col_b2 = st.columns([1, 2])
        
        with col_b1:
            # Selector de activos para la simulación
            default_ticks = df_pos['Ticker'].unique().tolist() if not df_pos.empty else ['NVDA', 'TSLA']
            sim_tickers = st.multiselect("Activos:", WATCHLIST, default=default_ticks)
            
            # Asignación de pesos automática (Equiponderada)
            pesos = {}
            if sim_tickers:
                st.caption("Pesos (%) - Deben sumar 100%")
                weight_cols = st.columns(len(sim_tickers))
                for i, tick in enumerate(sim_tickers):
                    default_w = 1.0 / len(sim_tickers)
                    pesos[tick] = weight_cols[i].number_input(f"{tick}", 0.0, 1.0, default_w, step=0.05)
            
            periodo_sim = st.selectbox("Periodo:", ["1y", "2y", "5y", "ytd"])
            
            if st.button("🏗️ SIMULAR CARTERA"):
                if sum(pesos.values()) > 1.01 or sum(pesos.values()) < 0.99:
                    st.error("⚠️ Los pesos deben sumar 1.0 (100%)")
                else:
                    with st.spinner("Viajando al pasado..."):
                        df_chart, stats = simular_cartera_historica(sim_tickers, pesos, periodo_sim)
                        st.session_state['sim_chart'] = df_chart
                        st.session_state['sim_stats'] = stats
        
        with col_b2:
            if 'sim_chart' in st.session_state and st.session_state['sim_chart'] is not None:
                stats = st.session_state['sim_stats']
                
                # Métricas Clave
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("CAGR (Anual)", f"{stats['CAGR Cartera']:.1f}%", delta=f"{stats['Alpha']:.1f}% vs SPY")
                k2.metric("Sharpe Ratio", f"{stats['Sharpe']:.2f}", help=">1 es Bueno, >2 Excelente")
                k3.metric("Volatilidad", f"{stats['Volatilidad']:.1f}%")
                k4.metric("Benchmark", f"{stats['CAGR Mercado']:.1f}%")
                
                # Gráfico
                fig_sim = px.line(st.session_state['sim_chart'], 
                                  title="Crecimiento de $100 Invertidos (Base 100)", 
                                  color_discrete_map={"Mi Cartera": "#00ff00", "Mercado (SPY)": "grey"})
                st.plotly_chart(fig_sim, use_container_width=True)

    with sub_tabs[1]:
        st.subheader("🧬 Grid Search RSI")
        if st.button("🚀 Optimizar Activo"):
            r = optimizar_parametros_estrategia(sel_ticker)
            if not r.empty:
                try: st.plotly_chart(px.density_heatmap(r, x="Compra <", y="Venta >", z="Retorno %", text_auto=".1f", color_continuous_scale="Viridis"), use_container_width=True)
                except: st.dataframe(r)