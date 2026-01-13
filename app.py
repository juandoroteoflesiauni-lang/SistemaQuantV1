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
st.set_page_config(page_title="Sistema Quant V76 (Risk Officer)", layout="wide", page_icon="🛡️")

st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;}
    .risk-alert {background-color: #3d0e0e; border: 1px solid #ff4b4b; padding: 15px; border-radius: 5px; text-align: center;}
    .risk-safe {background-color: #0e2b0e; border: 1px solid #00cc96; padding: 15px; border-radius: 5px; text-align: center;}
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

# --- MOTOR DE RIESGO DE PORTAFOLIO (NUEVO V76) ---
def calcular_riesgo_cartera(df_posiciones):
    """Calcula VaR, Beta ponderado y Stress Test de la cartera entera"""
    if df_posiciones.empty: return None
    
    try:
        tickers = df_posiciones['Ticker'].tolist()
        pesos = (df_posiciones['Valor'] / df_posiciones['Valor'].sum()).values
        total_equity = df_posiciones['Valor'].sum()
        
        # Descargar historial conjunto + SPY (Benchmark)
        data = yf.download(" ".join(tickers + ['SPY']), period="1y", progress=False, auto_adjust=True)['Close']
        if data.empty: return None
        
        # Retornos diarios
        returns = data.pct_change().dropna()
        
        # 1. Beta Ponderado de la Cartera
        # Covarianza de cada activo con SPY / Varianza SPY
        cov_matrix = returns.cov()
        var_spy = returns['SPY'].var()
        
        betas = []
        for t in tickers:
            cov = cov_matrix.loc[t, 'SPY']
            beta = cov / var_spy
            betas.append(beta)
            
        portfolio_beta = np.sum(np.array(betas) * pesos)
        
        # 2. Value at Risk (VaR) Paramétrico (95% Confianza, 1 día)
        # Volatilidad de la cartera (Desviación estándar ponderada)
        # Formula: sqrt(w.T * Cov * w)
        cov_assets = returns[tickers].cov()
        port_vol_daily = np.sqrt(np.dot(pesos.T, np.dot(cov_assets, pesos)))
        
        # VaR 95% = 1.65 * Volatilidad * Capital
        var_95_pct = 1.65 * port_vol_daily
        var_95_cash = var_95_pct * total_equity
        
        # 3. Stress Test (Simulación de Escenarios)
        # Escenario 1: Caída de mercado 5%
        loss_5pct = total_equity * (portfolio_beta * -0.05)
        # Escenario 2: Caída de mercado 20% (Crash)
        loss_20pct = total_equity * (portfolio_beta * -0.20)
        
        return {
            "Total_Equity": total_equity,
            "Beta_Cartera": portfolio_beta,
            "Volatilidad_Diaria": port_vol_daily * 100,
            "VaR_95_Cash": var_95_cash,
            "Stress_Market_Correction": loss_5pct,
            "Stress_Market_Crash": loss_20pct,
            "Matriz_Corr": returns[tickers].corr()
        }
        
    except Exception as e: return None

# --- MOTORES DE SOPORTE V75 ---
@st.cache_data(ttl=1800)
def obtener_datos_snapshot(ticker):
    try:
        stock = yf.Ticker(ticker); hist = stock.history(period="5d")
        if hist.empty: return None
        try: info = stock.info
        except: info = {}
        return {"Precio": hist['Close'].iloc[-1], "Previo": hist['Close'].iloc[-2], "RSI": ta.rsi(hist['Close'], 14).iloc[-1] if len(hist)>14 else 50, "Target": info.get('targetMeanPrice', 0)}
    except: return None

def generar_analisis_ia_completo(ticker, snap, fund, mc, dcf, cons):
    # (Simplificado para V76, foco en riesgo)
    prompt = f"Analisis de riesgo y oportunidad para {ticker}. Precio: {snap['Precio']}. RSI: {snap['RSI']}. Target: {cons.get('Target Mean',0) if cons else 0}."
    try: return model.generate_content(prompt).text
    except: return "IA no disponible."

def simulacion_monte_carlo(ticker, dias=30, simulaciones=100):
    try:
        data = yf.Ticker(ticker).history(period="1y")['Close']
        if data.empty: return None
        returns = data.pct_change().dropna()
        mu = returns.mean(); sigma = returns.std(); start_price = data.iloc[-1]
        sim_paths = np.zeros((dias, simulaciones)); sim_paths[0] = start_price
        for t in range(1, dias):
            drift = (mu - 0.5 * sigma**2); shock = sigma * np.random.normal(0, 1, simulaciones)
            sim_paths[t] = sim_paths[t-1] * np.exp(drift + shock)
        final = sim_paths[-1]
        return {"Paths": sim_paths, "Dates": [data.index[-1]+timedelta(days=i) for i in range(dias)], "Mean_Price": np.mean(final), "Prob_Suba": np.mean(final>start_price)*100, "VaR_95": np.percentile(final, 5)}
    except: return None

def calcular_dcf_rapido(ticker):
    if "USD" in ticker: return None
    try:
        i = yf.Ticker(ticker).info; fcf = i.get('freeCashflow', i.get('operatingCashflow', 0)*0.8)
        if fcf <= 0: return None
        pv = 0; g=0.1; w=0.09
        for y in range(1, 6): pv += (fcf * ((1+g)**y)) / ((1+w)**y)
        term = (fcf * ((1+g)**5) * 1.02) / (w - 0.02); pv_term = term / ((1+w)**5)
        return (pv + pv_term) / i.get('sharesOutstanding', 1)
    except: return None

def obtener_consenso_analistas(ticker):
    if "USD" in ticker: return None
    try:
        info = yf.Ticker(ticker).info
        return {"Recomendación": info.get('recommendationKey', 'N/A').upper(), "Target Mean": info.get('targetMeanPrice', 0)}
    except: return None

def graficar_simple(ticker):
    try:
        df = yf.Ticker(ticker).history(period="6mo", auto_adjust=True)
        if df.empty: return None
        df['SMA50'] = ta.sma(df['Close'], 50)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Precio'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='yellow'), name='SMA 50'))
        fig.update_layout(template="plotly_dark", height=300, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0)); return fig
    except: return None

@st.cache_data(ttl=3600)
def obtener_fundamentales_premium(ticker):
    if "USD" in ticker: return None
    try:
        stock = yf.Ticker(ticker)
        inc = stock.income_stmt.T.sort_index(); bal = stock.balance_sheet.T.sort_index()
        if inc.empty or bal.empty: return None
        gm = (inc['Gross Profit']/inc['Total Revenue'])*100; nm = (inc['Net Income']/inc['Total Revenue'])*100
        cr = bal['Total Current Assets'].iloc[-1]/bal['Total Current Liabilities'].iloc[-1]
        de = bal.get('Total Debt', pd.Series(0)).iloc[-1]/bal['Stockholders Equity'].iloc[-1]
        return {"Fechas": inc.index.strftime('%Y'), "Margen_Bruto": gm, "Margen_Neto": nm, "Current": cr, "Debt": de}
    except: return None

# --- INTERFAZ V76 ---
c1, c2 = st.columns([3, 1])
with c1: st.title("🛡️ Quant Terminal V76: Risk Officer")
with c2: sel_ticker = st.selectbox("ACTIVO PRINCIPAL", WATCHLIST)

# --- PANEL DE RIESGO DE CARTERA (NUEVO V76) ---
df_p = auditar_posiciones_sql()

if not df_p.empty:
    st.markdown("### 🛡️ Stress Test de Cartera")
    with st.spinner("Calculando exposición al riesgo..."):
        risk_data = calcular_riesgo_cartera(df_p)
    
    if risk_data:
        c_r1, c_r2, c_r3, c_r4 = st.columns(4)
        c_r1.metric("Valor Total", f"${risk_data['Total_Equity']:,.2f}")
        c_r2.metric("Beta Cartera", f"{risk_data['Beta_Cartera']:.2f}", help=">1.0 es más volátil que el mercado")
        c_r3.metric("VaR Diario (95%)", f"${risk_data['VaR_95_Cash']:.2f}", help="Pérdida máxima esperada en un día normal (95% confianza)", delta_color="inverse")
        
        # Semáforo de Riesgo
        if risk_data['Beta_Cartera'] > 1.3:
            c_r4.markdown("<div class='risk-alert'>🔥 RIESGO ALTO</div>", unsafe_allow_html=True)
        else:
            c_r4.markdown("<div class='risk-safe'>🛡️ RIESGO CONTROLADO</div>", unsafe_allow_html=True)
            
        with st.expander("📉 Simulador de Catástrofes (Stress Test)"):
            st.write("Si el mercado (S&P 500) cae, tu cartera perdería aproximadamente:")
            s1, s2 = st.columns(2)
            s1.metric("Corrección (-5%)", f"${risk_data['Stress_Market_Correction']:,.2f}", delta_color="inverse")
            s2.metric("Crash (-20%)", f"${risk_data['Stress_Market_Crash']:,.2f}", delta_color="inverse")
            
            st.write("---")
            st.write("🔗 **Matriz de Correlación de tus Activos:**")
            fig_corr = px.imshow(risk_data['Matriz_Corr'], text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("👆 Agrega operaciones en el panel derecho para ver el Análisis de Riesgo de tu cartera.")

st.divider()

# --- PANEL DE ANÁLISIS INDIVIDUAL (MANTENIDO V75) ---
col_main, col_side = st.columns([2, 1])

snap = obtener_datos_snapshot(sel_ticker)
fund = obtener_fundamentales_premium(sel_ticker)
mc = simulacion_monte_carlo(sel_ticker)
dcf = calcular_dcf_rapido(sel_ticker)
cons = obtener_consenso_analistas(sel_ticker)

with col_main:
    st.subheader(f"🔍 Análisis: {sel_ticker}")
    if snap:
        c_k1, c_k2, c_k3 = st.columns(3)
        c_k1.metric("Precio", f"${snap['Precio']:.2f}")
        c_k2.metric("RSI", f"{snap['RSI']:.0f}")
        c_k3.metric("Target", f"${snap['Target']:.2f}")
        fig_chart = graficar_simple(sel_ticker)
        if fig_chart: st.plotly_chart(fig_chart, use_container_width=True)
    
    tabs_detail = st.tabs(["🔮 Monte Carlo", "📚 Fundamentales", "📝 IA"])
    
    with tabs_detail[0]:
        if mc:
            st.metric("Probabilidad Suba", f"{mc['Prob_Suba']:.1f}%")
            fig_mc = go.Figure()
            for i in range(20): fig_mc.add_trace(go.Scatter(x=mc['Dates'], y=mc['Paths'][:, i], mode='lines', line=dict(color='gray', width=1), opacity=0.3, showlegend=False))
            fig_mc.add_trace(go.Scatter(x=mc['Dates'], y=np.mean(mc['Paths'], axis=1), mode='lines', name='Promedio', line=dict(color='yellow', width=3)))
            fig_mc.update_layout(template="plotly_dark", height=250, margin=dict(l=0,r=0,t=0,b=0)); st.plotly_chart(fig_mc, use_container_width=True)

    with tabs_detail[1]:
        if fund:
            fig_m = go.Figure(); fig_m.add_trace(go.Scatter(x=fund['Fechas'], y=fund['Margen_Neto'], name='Margen Neto', line=dict(color='green')))
            fig_m.update_layout(height=200, template="plotly_dark", title="Margen Neto (%)", margin=dict(l=0,r=0,t=30,b=0)); st.plotly_chart(fig_m, use_container_width=True)
            c1, c2 = st.columns(2); c1.metric("Liquidez", f"{fund['Current']:.2f}"); c2.metric("Deuda", f"{fund['Debt']:.2f}")
        else: st.info("Datos no disponibles.")

    with tabs_detail[2]:
        if st.button("Generar Informe IA"):
            try: st.markdown(generar_analisis_ia_completo(sel_ticker, snap, fund, mc, dcf, cons))
            except: st.error("Error IA")

with col_side:
    st.subheader("⚡ Quick Trade")
    with st.form("quick"):
        q = st.number_input("Qty", 1, 1000, 10); s = st.selectbox("Side", ["COMPRA", "VENTA"])
        if st.form_submit_button("EJECUTAR"): 
            if snap: registrar_operacion_sql(sel_ticker, s, q, snap['Precio']); st.success("Orden OK"); st.rerun()
    
    st.subheader("💼 Posiciones")
    if not df_p.empty: st.dataframe(df_p[['Ticker', 'Cantidad', 'P&L']], use_container_width=True)
