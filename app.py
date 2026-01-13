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
import time
import requests 
from datetime import datetime, timedelta
from scipy.stats import norm 
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from fpdf import FPDF
import base64

# --- CONFIGURACIÓN ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sistema Quant V77 (Insider)", layout="wide", page_icon="🦈")

st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;}
    .insider-card {background-color: #1a2634; border: 1px solid #2196F3; padding: 15px; border-radius: 8px;}
    .bull-pattern {color: #00ff00; font-weight: bold;}
    .bear-pattern {color: #ff4b4b; font-weight: bold;}
    .stTabs [data-baseweb="tab-list"] {gap: 5px;}
    .stTabs [data-baseweb="tab"] {height: 40px; padding: 5px 15px; font-size: 14px;}
</style>""", unsafe_allow_html=True)

try:
    secrets = toml.load(".streamlit/secrets.toml") if os.path.exists(".streamlit/secrets.toml") else st.secrets
    genai.configure(api_key=secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'COIN', 'KO', 'DIS', 'SPY', 'QQQ', 'GLD', 'USO']
DB_NAME = "quant_database.db"

# --- MOTOR INSTITUCIONAL (NUEVO V77) ---
@st.cache_data(ttl=3600)
def obtener_datos_insider(ticker):
    """Obtiene datos de tenencia institucional y shorts"""
    if "USD" in ticker: return None
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Short Interest
        shares_short = info.get('sharesShort', 0)
        float_shares = info.get('floatShares', 1)
        short_percent = (shares_short / float_shares) * 100 if float_shares > 0 else 0
        
        # Holders
        holders = stock.major_holders
        inst_percent = 0
        insider_percent = 0
        
        # Yahoo a veces devuelve DataFrame con índices 0, 1 o nombres directos
        try:
            if holders is not None and not holders.empty:
                # Intentar parsear el formato variable de yfinance
                df_h = holders.copy()
                if 0 in df_h.columns and 1 in df_h.columns:
                    # Formato antiguo
                    for idx, row in df_h.iterrows():
                        if 'Insiders' in str(row[1]): insider_percent = float(str(row[0]).replace('%', ''))
                        if 'Institutions' in str(row[1]): inst_percent = float(str(row[0]).replace('%', ''))
                else:
                    # Formato nuevo o dict
                    inst_percent = info.get('heldPercentInstitutions', 0) * 100
                    insider_percent = info.get('heldPercentInsiders', 0) * 100
        except: pass
        
        return {
            "Institucional": inst_percent,
            "Insiders": insider_percent,
            "Short_Float": short_percent,
            "Short_Ratio": info.get('shortRatio', 0)
        }
    except: return None

# --- MOTOR TÉCNICO AVANZADO (PATRONES V77) ---
def graficar_pro(ticker):
    """Gráfico con detección de patrones de velas"""
    try:
        df = yf.Ticker(ticker).history(period="6mo", auto_adjust=True)
        if df.empty: return None
        
        # Indicadores Base
        df['SMA50'] = ta.sma(df['Close'], 50)
        df['SMA200'] = ta.sma(df['Close'], 200)
        
        # DETECCIÓN DE PATRONES MANUAL (Para evitar dependencias pesadas de TA-Lib)
        # 1. Bullish Engulfing (Vela verde envuelve a la roja previa)
        df['Bull_Eng'] = (df['Close'] > df['Open']) & \
                         (df['Close'].shift(1) < df['Open'].shift(1)) & \
                         (df['Close'] > df['Open'].shift(1)) & \
                         (df['Open'] < df['Close'].shift(1))
        
        # 2. Bearish Engulfing (Vela roja envuelve a la verde previa)
        df['Bear_Eng'] = (df['Close'] < df['Open']) & \
                         (df['Close'].shift(1) > df['Open'].shift(1)) & \
                         (df['Close'] < df['Open'].shift(1)) & \
                         (df['Open'] > df['Close'].shift(1))
        
        # 3. Doji (Cuerpo muy pequeño)
        df['Doji'] = (abs(df['Close'] - df['Open']) / (df['High'] - df['Low'])) < 0.1
        
        # Filtrar solo los últimos eventos para el gráfico (no ensuciar todo)
        patterns_bull = df[df['Bull_Eng'] == True]
        patterns_bear = df[df['Bear_Eng'] == True]
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        
        # Velas
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Precio'), row=1, col=1)
        
        # Medias Móviles
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='cyan', width=1), name='SMA 50'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], line=dict(color='yellow', width=1), name='SMA 200'), row=1, col=1)
        
        # Marcadores de Patrones
        fig.add_trace(go.Scatter(x=patterns_bull.index, y=patterns_bull['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Bullish Engulfing'), row=1, col=1)
        fig.add_trace(go.Scatter(x=patterns_bear.index, y=patterns_bear['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Bearish Engulfing'), row=1, col=1)
        
        # Volumen
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='rgba(100, 100, 100, 0.5)', name='Volumen'), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        return fig
    except: return None

# --- MOTORES EXISTENTES ---
@st.cache_data(ttl=1800)
def obtener_datos_snapshot(ticker):
    try:
        stock = yf.Ticker(ticker); hist = stock.history(period="5d")
        if hist.empty: return None
        try: info = stock.info
        except: info = {}
        return {"Precio": hist['Close'].iloc[-1], "Previo": hist['Close'].iloc[-2], "RSI": ta.rsi(hist['Close'], 14).iloc[-1] if len(hist)>14 else 50, "Volumen": info.get('volume', 0), "Beta": info.get('beta', 1.0), "Target": info.get('targetMeanPrice', 0)}
    except: return None

@st.cache_data(ttl=3600)
def obtener_fundamentales_premium(ticker):
    if "USD" in ticker: return None
    try:
        stock = yf.Ticker(ticker); inc = stock.income_stmt.T.sort_index(); bal = stock.balance_sheet.T.sort_index()
        if inc.empty or bal.empty: return None
        gm = (inc['Gross Profit']/inc['Total Revenue'])*100; nm = (inc['Net Income']/inc['Total Revenue'])*100
        cr = bal['Total Current Assets'].iloc[-1]/bal['Total Current Liabilities'].iloc[-1]
        de = bal.get('Total Debt', pd.Series(0)).iloc[-1]/bal['Stockholders Equity'].iloc[-1]
        return {"Fechas": inc.index.strftime('%Y'), "Margen_Bruto": gm, "Margen_Neto": nm, "Current": cr, "Debt": de}
    except: return None

def simulacion_monte_carlo(ticker, dias=30, simulaciones=100):
    try:
        data = yf.Ticker(ticker).history(period="1y")['Close']
        if data.empty: return None
        returns = data.pct_change().dropna(); mu = returns.mean(); sigma = returns.std(); start_price = data.iloc[-1]
        sim_paths = np.zeros((dias, simulaciones)); sim_paths[0] = start_price
        for t in range(1, dias):
            drift = (mu - 0.5 * sigma**2); shock = sigma * np.random.normal(0, 1, simulaciones)
            sim_paths[t] = sim_paths[t-1] * np.exp(drift + shock)
        final = sim_paths[-1]
        return {"Paths": sim_paths, "Dates": [data.index[-1]+timedelta(days=i) for i in range(dias)], "Mean_Price": np.mean(final), "Prob_Suba": np.mean(final>start_price)*100, "VaR_95": np.percentile(final, 5)}
    except: return None

def calcular_riesgo_cartera(df_posiciones):
    if df_posiciones.empty: return None
    try:
        tickers = df_posiciones['Ticker'].tolist(); pesos = (df_posiciones['Valor']/df_posiciones['Valor'].sum()).values
        data = yf.download(" ".join(tickers+['SPY']), period="1y", progress=False, auto_adjust=True)['Close']
        ret = data.pct_change().dropna()
        cov = ret.cov(); var_spy = ret['SPY'].var()
        betas = [cov.loc[t, 'SPY']/var_spy for t in tickers]
        port_beta = np.sum(np.array(betas)*pesos)
        port_vol = np.sqrt(np.dot(pesos.T, np.dot(ret[tickers].cov(), pesos)))
        return {"Beta_Cartera": port_beta, "VaR_95_Cash": 1.65*port_vol*df_posiciones['Valor'].sum(), "Crash": df_posiciones['Valor'].sum()*port_beta*-0.20}
    except: return None

# --- MOTOR SQL ---
def init_db():
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, ticker TEXT, tipo TEXT, cantidad INTEGER, precio REAL, total REAL)''')
    conn.commit(); conn.close()
def registrar_operacion_sql(t, tipo, q, p):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    total = q * p; fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO trades (fecha, ticker, tipo, cantidad, precio, total) VALUES (?, ?, ?, ?, ?, ?)", (fecha, t, tipo, q, p, total))
    conn.commit(); conn.close()
def auditar_posiciones_sql():
    conn = sqlite3.connect(DB_NAME); df = pd.read_sql_query("SELECT * FROM trades", conn); conn.close()
    if df.empty: return pd.DataFrame()
    pos = {}
    for i, r in df.iterrows():
        t = r['ticker']
        if t not in pos: pos[t] = {"Qty": 0, "Cost": 0}
        if r['tipo'] == "COMPRA": pos[t]["Qty"] += r['cantidad']; pos[t]["Cost"] += r['total']
        elif r['tipo'] == "VENTA": 
            pos[t]["Qty"] -= r['cantidad']
            if pos[t]["Qty"] > 0: unit = pos[t]["Cost"]/(pos[t]["Qty"]+r['cantidad']); pos[t]["Cost"] -= (unit*r['cantidad'])
            else: pos[t]["Cost"] = 0
    res = []
    activos = [t for t, d in pos.items() if d['Qty'] > 0]
    if not activos: return pd.DataFrame()
    try: curr = yf.download(" ".join(activos), period="1d", progress=False, auto_adjust=True)['Close']
    except: return pd.DataFrame()
    for t in activos:
        d = pos[t]
        try:
            if len(activos) == 1: px = float(curr.iloc[-1])
            else: px = float(curr.iloc[-1][t])
            val = d['Qty'] * px; pnl = val - d['Cost']
            res.append({"Ticker": t, "Cantidad": d['Qty'], "Valor": val, "P&L": pnl})
        except: pass
    return pd.DataFrame(res)
init_db()

# --- INTERFAZ V77 ---
c1, c2 = st.columns([3, 1])
with c1: st.title("🦈 Quant Terminal V77: Insider")
with c2: sel_ticker = st.selectbox("ACTIVO PRINCIPAL", WATCHLIST)

snap = obtener_datos_snapshot(sel_ticker)
if snap:
    delta = ((snap['Precio'] - snap['Previo'])/snap['Previo'])*100
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Precio", f"${snap['Precio']:.2f}", f"{delta:+.2f}%")
    k2.metric("RSI", f"{snap['RSI']:.0f}")
    k3.metric("Vol", f"{snap['Volumen']/1e6:.1f}M")
    k4.metric("Beta", f"{snap['Beta']:.2f}")
    k5.metric("Target", f"${snap['Target']:.2f}")

st.divider()

col_main, col_side = st.columns([2, 1])

fund = obtener_fundamentales_premium(sel_ticker)
mc = simulacion_monte_carlo(sel_ticker)
insider = obtener_datos_insider(sel_ticker)

with col_main:
    # GRÁFICO PRO (V77)
    st.subheader("📉 Acción del Precio & Patrones")
    fig_chart = graficar_pro(sel_ticker)
    if fig_chart: st.plotly_chart(fig_chart, use_container_width=True)
    
    tabs_detail = st.tabs(["🦈 Institucionales & Short", "🔮 Monte Carlo", "📚 Fundamentales", "📝 IA"])
    
    # --- TAB 1: INSIDERS (NUEVO V77) ---
    with tabs_detail[0]:
        st.subheader("🦈 Quién es dueño de la empresa?")
        if insider:
            c_i1, c_i2, c_i3 = st.columns(3)
            c_i1.metric("Institucionales", f"{insider['Institucional']:.1f}%", help="% en manos de Bancos/Fondos (Smart Money)")
            c_i2.metric("Insiders", f"{insider['Insiders']:.1f}%", help="% en manos de Directivos/Dueños")
            c_i3.metric("Short Float", f"{insider['Short_Float']:.2f}%", help="% Apostado en contra (Alto > 5%)", delta_color="inverse")
            
            # Semáforo de propiedad
            if insider['Institucional'] > 70: st.success("✅ Fuerte Respaldo Institucional (>70%)")
            elif insider['Institucional'] < 40: st.warning("⚠️ Poco interés Institucional (Principalmente Retail)")
            
            if insider['Short_Float'] > 10: st.error("🚨 ALERTA: Alto nivel de ventas en corto (Posible Short Squeeze o Quiebra)")
        else: st.info("Datos de propiedad no disponibles.")

    with tabs_detail[1]:
        if mc:
            st.metric("Probabilidad Suba", f"{mc['Prob_Suba']:.1f}%")
            fig_mc = go.Figure()
            for i in range(20): fig_mc.add_trace(go.Scatter(x=mc['Dates'], y=mc['Paths'][:, i], mode='lines', line=dict(color='gray', width=1), opacity=0.3, showlegend=False))
            fig_mc.add_trace(go.Scatter(x=mc['Dates'], y=np.mean(mc['Paths'], axis=1), mode='lines', name='Promedio', line=dict(color='yellow', width=3)))
            fig_mc.update_layout(template="plotly_dark", height=250, margin=dict(l=0,r=0,t=0,b=0)); st.plotly_chart(fig_mc, use_container_width=True)

    with tabs_detail[2]:
        if fund:
            fig_m = go.Figure(); fig_m.add_trace(go.Scatter(x=fund['Fechas'], y=fund['Margen_Neto'], name='Margen Neto', line=dict(color='green')))
            fig_m.update_layout(height=200, template="plotly_dark", title="Margen Neto (%)", margin=dict(l=0,r=0,t=30,b=0)); st.plotly_chart(fig_m, use_container_width=True)
            c1, c2 = st.columns(2); c1.metric("Liquidez", f"{fund['Current']:.2f}"); c2.metric("Deuda", f"{fund['Debt']:.2f}")

    with tabs_detail[3]:
        if st.button("Generar Informe IA"):
            try: st.write(model.generate_content(f"Analisis completo {sel_ticker}").text)
            except: st.error("Error IA")

with col_side:
    st.subheader("💼 Gestión de Cartera")
    df_p = auditar_posiciones_sql()
    
    if not df_p.empty:
        # Gráfico de Torta de la Cartera (NUEVO V77)
        fig_pie = px.pie(df_p, values='Valor', names='Ticker', hole=0.4)
        fig_pie.update_layout(showlegend=False, margin=dict(l=0,r=0,t=0,b=0), height=200)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.dataframe(df_p[['Ticker', 'P&L']], use_container_width=True)
        
        # Riesgo Cartera
        risk = calcular_riesgo_cartera(df_p)
        if risk:
            st.metric("Beta Cartera", f"{risk['Beta_Cartera']:.2f}")
            st.metric("Riesgo Crash (-20%)", f"${risk['Crash']:,.0f}", delta_color="inverse")
    else: st.info("Cartera vacía.")

    st.markdown("---")
    st.subheader("⚡ Quick Trade")
    with st.form("quick"):
        q = st.number_input("Qty", 1, 1000, 10); s = st.selectbox("Side", ["COMPRA", "VENTA"])
        if st.form_submit_button("EJECUTAR"): 
            if snap: registrar_operacion_sql(sel_ticker, s, q, snap['Precio']); st.success("Orden OK"); st.rerun()
