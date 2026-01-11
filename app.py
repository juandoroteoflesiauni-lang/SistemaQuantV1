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

st.set_page_config(page_title="Sistema Quant V50 (Fund Manager)", layout="wide", page_icon="🏛️")
st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 15px; text-align: center;}
    .signal-box {border: 1px solid #444; padding: 10px; border-radius: 5px; background-color: #1e1e1e; text-align: center; margin-bottom: 5px;}
    .bull {color: #00ff00; font-weight: bold;}
    .bear {color: #ff0000; font-weight: bold;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #0e1117; border-radius: 5px;}
    .stTabs [aria-selected="true"] {background-color: #262730;}
</style>""", unsafe_allow_html=True)

WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'COIN', 'KO', 'DIS', 'SPY', 'QQQ', 'DIA', 'GLD', 'USO']
DB_NAME = "quant_database.db"

# --- MOTOR SQL ROBUSTO ---
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

# --- MOTORES DE ANÁLISIS (OPTIMIZADOS V50) ---
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
        df = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True) # Periodo más corto para velocidad
        if df.empty: return pd.DataFrame()
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        
        resultados = []
        df['RSI'] = ta.rsi(df['Close'], 14)
        
        for b in [20, 30, 40]:
            for s in [60, 70, 80]:
                cash = 10000; pos = 0
                # Vectorizado simple
                buy_sig = df['RSI'] < b; sell_sig = df['RSI'] > s
                for i in range(15, len(df)):
                    p = df['Close'].iloc[i]
                    if cash > 0 and buy_sig.iloc[i]: pos = cash/p; cash = 0
                    elif pos > 0 and sell_sig.iloc[i]: cash = pos*p; pos = 0
                final = cash + (pos * df['Close'].iloc[-1])
                resultados.append({"Compra <": b, "Venta >": s, "Retorno %": ((final-10000)/10000)*100})
        
        return pd.DataFrame(resultados)
    except: return pd.DataFrame() # Retorna vacío seguro

def calcular_valor_intrinseco(ticker):
    try:
        i = yf.Ticker(ticker).info
        p = i.get('currentPrice') or i.get('regularMarketPreviousClose') or 0
        e = i.get('trailingEps', 0); b = i.get('bookValue', 0)
        if e and b and e > 0 and b > 0:
            g = math.sqrt(22.5 * e * b)
            s = "INFRAVALORADA 🟢" if g > p else "SOBREVALORADA 🔴"
            return {"Precio": p, "Graham": g, "Status_Graham": s, "Diff": ((g-p)/p)*100}
    except: pass
    return None

def graficar_master(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
        if df.empty: return None
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df['EMA20'] = ta.ema(df['Close'], 20); df['RSI'] = ta.rsi(df['Close'], 14)
        bb = ta.bbands(df['Close'], length=20, std=2); df = pd.concat([df, bb], axis=1)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"), row=1, col=1)
        if 'EMA20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='yellow', width=1), name="EMA 20"), row=1, col=1)
        try: 
            fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, -3], line=dict(color='cyan', width=1, dash='dot'), name="Upper"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, -1], line=dict(color='cyan', width=1, dash='dot'), name="Lower"), row=1, col=1)
        except: pass
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_color="red", row=2, col=1); fig.add_hline(y=30, line_color="green", row=2, col=1)
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=0, b=0))
        return fig
    except: return None

@st.cache_data(ttl=300)
def escanear_oportunidades(tickers):
    s = []
    try: data = yf.download(" ".join(tickers), period="3mo", progress=False, group_by='ticker', auto_adjust=True)
    except: return pd.DataFrame()
    for t in tickers:
        try:
            df = data[t].dropna() if len(tickers) > 1 else data.dropna()
            if len(df) < 14: continue
            c = df['Close'].iloc[-1]; r = ta.rsi(df['Close'], 14).iloc[-1]
            if r < 30: s.append({"Ticker": t, "Señal": "COMPRA RSI 🟢"})
            elif r > 70: s.append({"Ticker": t, "Señal": "VENTA RSI 🔴"})
        except: pass
    return pd.DataFrame(s)

# --- INTERFAZ V50: THE FUND MANAGER ---
st.title("🏛️ Sistema Quant V50: Fund Manager")

# DASHBOARD EJECUTIVO (KPIs)
df_pos = auditar_posiciones_sql()
df_alerts = escanear_oportunidades(WATCHLIST)

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
with col_kpi1:
    equity = df_pos['Valor Mercado'].sum() if not df_pos.empty else 0.0
    st.metric("Patrimonio Neto", f"${equity:,.2f}", help="Valor total de tus acciones hoy")
with col_kpi2:
    pnl = df_pos['P&L ($)'].sum() if not df_pos.empty else 0.0
    st.metric("P&L Total", f"${pnl:+.2f}", delta_color="normal")
with col_kpi3:
    alerts_count = len(df_alerts) if not df_alerts.empty else 0
    st.metric("Oportunidades Hoy", str(alerts_count), "Señales Activas")
with col_kpi4:
    sp500 = yf.Ticker("SPY").history(period="2d")['Close']
    if len(sp500) > 1:
        delta_spy = ((sp500.iloc[-1] - sp500.iloc[-2])/sp500.iloc[-2])*100
        st.metric("Mercado (SP500)", f"${sp500.iloc[-1]:.2f}", f"{delta_spy:+.2f}%")

st.divider()

# LAYOUT PRINCIPAL: 3 PESTAÑAS MAESTRAS
main_tabs = st.tabs(["💼 MESA DE DINERO", "📊 ANÁLISIS & ESTRATEGIA", "🧬 LABORATORIO QUANT"])

# --- TAB 1: OPERATIVA Y GESTIÓN ---
with main_tabs[0]:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Cartera de Inversión")
        if not df_pos.empty:
            st.dataframe(
                df_pos.style.format({"Precio Prom.": "${:.2f}", "Precio Actual": "${:.2f}", "Valor Mercado": "${:.2f}", "P&L ($)": "${:+.2f}", "P&L (%)": "{:+.2f}%"})
                .background_gradient(subset=['P&L (%)'], cmap='RdYlGn', vmin=-10, vmax=10),
                use_container_width=True
            )
        else:
            st.info("No tienes posiciones abiertas. Usa el panel derecho para comprar.")
            
    with c2:
        st.subheader("Ejecución de Órdenes")
        with st.form("order_form"):
            t_op = st.selectbox("Activo", WATCHLIST)
            tipo = st.selectbox("Operación", ["COMPRA", "VENTA"])
            qty = st.number_input("Cantidad", 1, 10000)
            px = st.number_input("Precio Ejecución", 0.0)
            if st.form_submit_button("CONFIRMAR ORDEN (SQL)"):
                registrar_operacion_sql(t_op, tipo, qty, px)
                st.success(f"Orden {tipo} {t_op} registrada.")
                st.rerun()

# --- TAB 2: ANÁLISIS ---
with main_tabs[1]:
    col_ana1, col_ana2 = st.columns([1, 3])
    with col_ana1:
        sel_ticker = st.selectbox("Analizar Activo:", WATCHLIST)
        
        # Fundamental Rápido
        fund = calcular_valor_intrinseco(sel_ticker)
        if fund:
            st.markdown("---")
            st.metric("Precio", f"${fund['Precio']:.2f}")
            st.metric("Valor Graham", f"${fund['Graham']:.2f}", f"{fund['Diff']:.1f}%")
            if "INFRA" in fund['Status_Graham']: st.success("🟢 Subvaluada")
            else: st.error("🔴 Sobrevaluada")

    with col_ana2:
        # Gráfico Master
        fig = graficar_master(sel_ticker)
        if fig: st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap
        with st.expander("Ver Mapa de Calor del Mercado"):
            try:
                df_map = generar_mapa_calor(WATCHLIST)
                if df_map is not None:
                    fig_heat = px.treemap(df_map, path=['Sector', 'Ticker'], values='Size', color='Variacion', color_continuous_scale='RdYlGn', color_continuous_midpoint=0)
                    st.plotly_chart(fig_heat, use_container_width=True)
            except: st.warning("Datos de heatmap no disponibles.")

# --- TAB 3: LABORATORIO (GRID SEARCH FIX V50) ---
with main_tabs[2]:
    st.subheader(f"🧬 Optimización Genética: {sel_ticker}")
    st.write("Busca la configuración de RSI que históricamente generó más ganancias.")
    
    if st.button("🚀 CORRER SIMULACIÓN"):
        with st.spinner("Probando combinaciones..."):
            res_grid = optimizar_parametros_estrategia(sel_ticker)
            
            if not res_grid.empty:
                # Encontrar mejor
                best = res_grid.loc[res_grid['Retorno %'].idxmax()]
                st.success(f"Mejor Configuración: Compra < {best['Compra <']} | Venta > {best['Venta >']} | Retorno: {best['Retorno %']:.2f}%")
                
                # Gráfico Blindado V50
                try:
                    # Aseguramos tipos numéricos
                    res_grid = res_grid.astype(float)
                    fig_opt = px.density_heatmap(
                        res_grid, x="Compra <", y="Venta >", z="Retorno %", 
                        text_auto=".1f", color_continuous_scale="Viridis",
                        title="Mapa de Rentabilidad (Z = Retorno %)"
                    )
                    st.plotly_chart(fig_opt, use_container_width=True)
                except Exception as e:
                    st.warning(f"No se pudo graficar el mapa de calor: {e}")
                    st.dataframe(res_grid) # Mostramos tabla si falla gráfico
            else:
                st.error("No hay suficientes datos históricos para optimizar este activo.")