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

st.set_page_config(page_title="Sistema Quant V52 (Whale Tracker)", layout="wide", page_icon="🐋")
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

# --- MOTORES DE ANÁLISIS ---
def detectar_actividad_ballenas(ticker):
    """Detecta volumen inusual y posición respecto al VWAP"""
    try:
        df = yf.Ticker(ticker).history(period="1mo", interval="1d", auto_adjust=True)
        if df.empty: return None
        
        # Calcular VWAP (Volume Weighted Average Price)
        # Formula manual simple para pandas nativo: Cumsum(Price * Vol) / Cumsum(Vol)
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        # Promedio de Volumen (20 días)
        df['VolSMA'] = df['Volume'].rolling(20).mean()
        
        last_vol = df['Volume'].iloc[-1]
        last_avg = df['VolSMA'].iloc[-1]
        last_close = df['Close'].iloc[-1]
        last_vwap = df['VWAP'].iloc[-1]
        
        alerta = None
        # Si el volumen de hoy es > 150% del promedio
        if last_vol > last_avg * 1.5:
            alerta = "🐋 ALERTA BALLENA: Volumen Inusual"
            
        trend = "ALCISTA (Institucional)" if last_close > last_vwap else "BAJISTA (Institucional)"
        
        return {
            "Volumen Hoy": last_vol,
            "Volumen Promedio": last_avg,
            "Ratio Vol": last_vol / last_avg,
            "Alerta": alerta,
            "VWAP": last_vwap,
            "Tendencia": trend
        }
    except: return None

@st.cache_data(ttl=600)
def calcular_score_quant(ticker):
    score = 0; breakdown = {"Técnico": 0, "Fundamental": 0, "Riesgo": 0}
    try:
        df = yf.Ticker(ticker).history(period="6mo", interval="1d", auto_adjust=True)
        if not df.empty:
            df['RSI'] = ta.rsi(df['Close'], 14); df['SMA50'] = ta.sma(df['Close'], 50)
            if 30 <= df['RSI'].iloc[-1] <= 65: breakdown['Técnico'] += 20
            elif df['RSI'].iloc[-1] < 30: breakdown['Técnico'] += 15
            elif df['RSI'].iloc[-1] > 70: breakdown['Técnico'] += 5
            if df['Close'].iloc[-1] > df['SMA50'].iloc[-1]: breakdown['Técnico'] += 20
        info = yf.Ticker(ticker).info
        eps = info.get('trailingEps', 0); book = info.get('bookValue', 0); price = info.get('currentPrice', 0)
        if eps and book and eps>0 and book>0:
            g = math.sqrt(22.5 * eps * book)
            if price < g: breakdown['Fundamental'] += 20
            elif price < g*1.2: breakdown['Fundamental'] += 10
        if eps > 0: breakdown['Fundamental'] += 20
        if not eps and "USD" in ticker: breakdown['Fundamental'] = 20
        beta = info.get('beta', 1.0)
        if beta is None: beta = 1.0
        if 0.8 <= beta <= 1.2: breakdown['Riesgo'] += 20
        elif beta < 0.8: breakdown['Riesgo'] += 15
        elif beta > 1.5: breakdown['Riesgo'] += 5
        else: breakdown['Riesgo'] += 10
        score = breakdown['Técnico'] + breakdown['Fundamental'] + breakdown['Riesgo']
        return score, breakdown
    except: return 0, breakdown

def dibujar_velocimetro(score):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score, domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "white"},
               'steps': [{'range': [0, 40], 'color': "#ff4b4b"}, {'range': [40, 70], 'color': "#ffa500"}, {'range': [70, 100], 'color': "#00cc96"}]}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="#0e1117", font={'color': "white"})
    return fig

# --- GRÁFICO MASTER ACTUALIZADO V52 (CON VWAP) ---
def graficar_master(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
        if df.empty: return None
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        
        # Indicadores
        df['EMA20'] = ta.ema(df['Close'], 20)
        df['RSI'] = ta.rsi(df['Close'], 14)
        # VWAP (Calculado manualmente para este timeframe para asegurar compatibilidad)
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        bb = ta.bbands(df['Close'], length=20, std=2); df = pd.concat([df, bb], axis=1)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        
        # Precio y VWAP
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"), row=1, col=1)
        if 'EMA20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='yellow', width=1), name="EMA 20"), row=1, col=1)
        
        # VWAP (Línea Dorada - Institucional)
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#FFD700', width=2), name="VWAP (Inst.)"), row=1, col=1)
        
        try: 
            fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, -3], line=dict(color='cyan', width=1, dash='dot'), name="Upper"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, -1], line=dict(color='cyan', width=1, dash='dot'), name="Lower"), row=1, col=1)
        except: pass
        
        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_color="red", row=2, col=1); fig.add_hline(y=30, line_color="green", row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=0, b=0))
        return fig
    except: return None

@st.cache_data(ttl=600)
def generar_mapa_calor(tickers):
    try:
        data = yf.download(" ".join(tickers), period="5d", interval="1d", progress=False, auto_adjust=True)['Close']
        pct = ((data.iloc[-1] - data.iloc[-2]) / data.iloc[-2]) * 100
        df = pd.DataFrame({'Ticker': pct.index, 'Variacion': pct.values, 'Precio': data.iloc[-1].values})
        sec = []
        for t in df['Ticker']:
            if t in ['NVDA', 'AMD', 'TSLA', 'AAPL', 'MSFT', 'META', 'GOOGL']: sec.append('Tech')
            elif t in ['BTC-USD', 'ETH-USD', 'COIN']: sec.append('Cripto')
            elif t in ['SPY', 'QQQ', 'DIA']: sec.append('Índices')
            elif t in ['GLD', 'USO']: sec.append('Commodities')
            else: sec.append('Otros')
        df['Sector'] = sec; df['Size'] = df['Precio'] 
        return df
    except: return None

def optimizar_parametros_estrategia(ticker, estrategia="RSI"):
    try:
        df = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
        if df.empty: return pd.DataFrame()
        df['RSI'] = ta.rsi(df['Close'], 14)
        resultados = []
        for b in [20, 30, 40]:
            for s in [60, 70, 80]:
                cash = 10000; pos = 0
                buy_sig = df['RSI'] < b; sell_sig = df['RSI'] > s
                for i in range(15, len(df)):
                    p = df['Close'].iloc[i]
                    if cash > 0 and buy_sig.iloc[i]: pos = cash/p; cash = 0
                    elif pos > 0 and sell_sig.iloc[i]: cash = pos*p; pos = 0
                final = cash + (pos * df['Close'].iloc[-1])
                resultados.append({"Compra <": b, "Venta >": s, "Retorno %": ((final-10000)/10000)*100})
        return pd.DataFrame(resultados)
    except: return pd.DataFrame()

# --- INTERFAZ V52: WHALE TRACKER ---
st.title("🐋 Sistema Quant V52: Whale Tracker")

# DASHBOARD EJECUTIVO
df_pos = auditar_posiciones_sql()

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
with col_kpi1:
    equity = df_pos['Valor Mercado'].sum() if not df_pos.empty else 0.0
    st.metric("Patrimonio Neto", f"${equity:,.2f}")
with col_kpi2:
    pnl = df_pos['P&L ($)'].sum() if not df_pos.empty else 0.0
    st.metric("P&L Total", f"${pnl:+.2f}", delta_color="normal")
with col_kpi3:
    sp500 = yf.Ticker("SPY").history(period="2d")['Close']
    st.metric("Mercado (SP500)", f"${sp500.iloc[-1]:.2f}")
with col_kpi4:
    # Sensor Rápido de Ballenas
    whale_count = 0
    with st.expander("🐋 Radar de Ballenas", expanded=False):
        for t in WATCHLIST[:5]: # Solo check rápido
            w = detectar_actividad_ballenas(t)
            if w and w['Alerta']: 
                st.write(f"⚠️ {t}: {w['Alerta']}")
                whale_count += 1
    st.metric("Alertas Volumen", f"{whale_count}")

st.divider()

main_tabs = st.tabs(["💼 MESA DE DINERO", "📊 ANÁLISIS 360", "🧬 LABORATORIO QUANT"])

# --- TAB 1: OPERATIVA ---
with main_tabs[0]:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Cartera de Inversión")
        if not df_pos.empty:
            st.dataframe(df_pos.style.format({"Valor Mercado": "${:.2f}", "P&L ($)": "${:+.2f}"}).background_gradient(subset=['P&L (%)'], cmap='RdYlGn'), use_container_width=True)
        else: st.info("Cartera vacía.")
    with c2:
        st.subheader("Orden de Mercado")
        with st.form("order_form"):
            t_op = st.selectbox("Activo", WATCHLIST)
            tipo = st.selectbox("Operación", ["COMPRA", "VENTA"])
            qty = st.number_input("Cantidad", 1, 10000)
            precio_ejec = st.number_input("Precio Ejecución", 0.0)
            if st.form_submit_button("CONFIRMAR ORDEN (SQL)"):
                registrar_operacion_sql(t_op, tipo, qty, precio_ejec); st.rerun()

# --- TAB 2: ANÁLISIS 360 (NUEVO V52) ---
with main_tabs[1]:
    col_ana1, col_ana2 = st.columns([1, 2])
    
    with col_ana1:
        sel_ticker = st.selectbox("Analizar Activo:", WATCHLIST)
        st.write("---")
        
        # EL JUEZ (SCORING)
        score, breakdown = calcular_score_quant(sel_ticker)
        st.markdown(f"### ⚖️ Score: {score}/100")
        st.plotly_chart(dibujar_velocimetro(score), use_container_width=True)
        
        # DATOS DE BALLENAS (NUEVO V52)
        whale_data = detectar_actividad_ballenas(sel_ticker)
        st.markdown("#### 🐋 Flujo Institucional")
        if whale_data:
            if whale_data['Alerta']:
                st.markdown(f"<div class='whale-alert'>🚨 {whale_data['Alerta']}</div>", unsafe_allow_html=True)
            else:
                st.info("Volumen normal. Sin actividad inusual.")
            
            w1, w2 = st.columns(2)
            w1.metric("Volumen Hoy", f"{whale_data['Volumen Hoy']/1000:.1f}K", help="Miles de acciones")
            w2.metric("Ratio vs Prom.", f"{whale_data['Ratio Vol']:.1f}x")
            
            st.metric("VWAP (Precio Justo)", f"${whale_data['VWAP']:.2f}")
            if "ALCISTA" in whale_data['Tendencia']:
                st.success(f"Tendencia: {whale_data['Tendencia']}")
            else:
                st.error(f"Tendencia: {whale_data['Tendencia']}")

    with col_ana2:
        # GRÁFICO CON VWAP
        fig = graficar_master(sel_ticker)
        if fig: st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Ver Mapa de Calor"):
            try:
                df_map = generar_mapa_calor(WATCHLIST)
                if df_map is not None:
                    fig_heat = px.treemap(df_map, path=['Sector', 'Ticker'], values='Size', color='Variacion', color_continuous_scale='RdYlGn', color_continuous_midpoint=0)
                    st.plotly_chart(fig_heat, use_container_width=True)
            except: pass

# --- TAB 3: LABORATORIO ---
with main_tabs[2]:
    st.subheader(f"🧬 Optimización: {sel_ticker}")
    if st.button("🚀 INICIAR GRID SEARCH"):
        with st.spinner("Simulando..."):
            res_grid = optimizar_parametros_estrategia(sel_ticker)
            if not res_grid.empty:
                best = res_grid.loc[res_grid['Retorno %'].idxmax()]
                st.success(f"Mejor RSI: Compra<{best['Compra <']} Venta>{best['Venta >']} (Retorno: {best['Retorno %']:.2f}%)")
                try: 
                    st.plotly_chart(px.density_heatmap(res_grid, x="Compra <", y="Venta >", z="Retorno %", text_auto=".1f", color_continuous_scale="Viridis"), use_container_width=True)
                except: st.dataframe(res_grid)