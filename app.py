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

st.set_page_config(page_title="Sistema Quant V60 (Forensic Auditor)", layout="wide", page_icon="🔍")
st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 15px; text-align: center;}
    .alert-card-high {background-color: #3d0e0e; border: 1px solid #ff4b4b; padding: 10px; border-radius: 5px; margin-bottom: 5px;}
    .audit-pass {color: #00ff00; font-weight: bold; border: 1px solid #00ff00; padding: 5px; border-radius: 5px; display: inline-block;}
    .audit-fail {color: #ff0000; font-weight: bold; border: 1px solid #ff0000; padding: 5px; border-radius: 5px; display: inline-block;}
    .audit-warn {color: #ffa500; font-weight: bold; border: 1px solid #ffa500; padding: 5px; border-radius: 5px; display: inline-block;}
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

# --- MOTOR CONTABLE FORENSE (NUEVO V60) ---
@st.cache_data(ttl=3600)
def realizar_auditoria_forense(ticker):
    """Calcula Z-Score (Quiebra) y métricas de salud financiera"""
    if "USD" in ticker: return None # No aplica a Crypto
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Recopilación de Datos Contables (Safe Get)
        total_assets = info.get('totalAssets', 0)
        total_liab = info.get('totalDebt', 0) # Aprox
        current_assets = info.get('totalCurrentAssets', 0)
        current_liab = info.get('totalCurrentLiabilities', 0)
        retained_earnings = info.get('retainedEarnings', total_assets * 0.1) # Fallback estimado
        ebit = info.get('ebitda', 0) 
        total_revenue = info.get('totalRevenue', 0)
        market_cap = info.get('marketCap', 0)
        
        if total_assets == 0 or total_liab == 0: return None
        
        # --- 1. ALTMAN Z-SCORE CALCULATION ---
        # A = Working Capital / Total Assets
        wk = current_assets - current_liab
        A = wk / total_assets
        
        # B = Retained Earnings / Total Assets
        B = retained_earnings / total_assets
        
        # C = EBIT / Total Assets
        C = ebit / total_assets
        
        # D = Market Value of Equity / Total Liabilities
        D = market_cap / total_liab
        
        # E = Sales / Total Assets
        E = total_revenue / total_assets
        
        # Fórmula Z-Score (Manufactura Original)
        Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
        
        # Diagnóstico Z
        z_status = ""
        z_color = ""
        if Z > 3.0: 
            z_status = "ZONA SEGURA (Solvente)"
            z_color = "audit-pass"
        elif Z < 1.8: 
            z_status = "ZONA DE PELIGRO (Riesgo Quiebra)"
            z_color = "audit-fail"
        else: 
            z_status = "ZONA GRIS (Alerta)"
            z_color = "audit-warn"
            
        # --- 2. PIOTROSKI F-SCORE (Simplificado 0-9) ---
        # Puntos por rentabilidad, apalancamiento y eficiencia
        f_score = 0
        if info.get('returnOnAssets', 0) > 0: f_score += 1
        if info.get('operatingCashflow', 0) > 0: f_score += 1
        if info.get('currentRatio', 0) > 1: f_score += 1
        if info.get('debtToEquity', 0) < 100: f_score += 1 # Menos deuda es mejor
        
        return {
            "Z_Score": Z,
            "Z_Status": z_status,
            "Z_Color": z_color,
            "F_Score": f_score, # Sobre 4 (simplificado por datos limitados)
            "Deuda/Patrimonio": info.get('debtToEquity', 0),
            "Current Ratio": info.get('currentRatio', 0),
            "ROA": info.get('returnOnAssets', 0) * 100
        }
        
    except Exception as e: return None

# --- MOTORES EXISTENTES ---
def ejecutar_backtest_custom(ticker, capital, params):
    try:
        df = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=True)
        if df.empty: return None
        df['RSI'] = ta.rsi(df['Close'], 14); df['SMA200'] = ta.sma(df['Close'], 200)
        cash = capital; position = 0; entry_price = 0; log = []; eq = []
        rsi_buy = params.get('rsi_buy', 30); use_trend = params.get('use_trend', False)
        sl = params.get('sl', 0.0)/100; tp = params.get('tp', 0.0)/100
        for i in range(200, len(df)):
            p = df['Close'].iloc[i]; d = df.index[i]; r = df['RSI'].iloc[i]; s200 = df['SMA200'].iloc[i]
            buy_cond = (r < rsi_buy) and ((p > s200) if use_trend else True)
            if position == 0 and cash > 0 and buy_cond:
                position = cash/p; entry_price = p; cash = 0; log.append({"Fecha":d, "Tipo":"COMPRA", "Precio":p})
            elif position > 0:
                if sl > 0 and p <= entry_price*(1-sl): cash=position*p; position=0; log.append({"Fecha":d, "Tipo":"VENTA", "Precio":p, "Razón":"SL"})
                elif tp > 0 and p >= entry_price*(1+tp): cash=position*p; position=0; log.append({"Fecha":d, "Tipo":"VENTA", "Precio":p, "Razón":"TP"})
                elif r > 70: cash=position*p; position=0; log.append({"Fecha":d, "Tipo":"VENTA", "Precio":p, "Razón":"RSI"})
            eq.append({"Fecha":d, "Equity": cash if position==0 else position*p})
        final = cash if position==0 else position*df['Close'].iloc[-1]
        return {"Retorno": ((final-capital)/capital)*100, "Trades": len(log), "Log": pd.DataFrame(log), "Equity": pd.DataFrame(eq).set_index("Fecha")}
    except: return None

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
        data = r.json(); return int(data['data'][0]['value']), data['data'][0]['value_classification']
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

def analizar_fundamental_crypto(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
        if df.empty: return None
        fair = df['Close'].rolling(200).mean().iloc[-1]; curr = df['Close'].iloc[-1]
        status = "SOBRECOMPRADO 🔴" if curr > fair * 1.5 else "ZONA DE ACUMULACIÓN 🟢" if curr < fair else "NEUTRAL 🟡"
        return {"Precio": curr, "FairValue_200SMA": fair, "Status": status}
    except: return None

# --- INTERFAZ V60: FORENSIC AUDITOR ---
st.title("🔍 Sistema Quant V60: Forensic Auditor")

with st.sidebar:
    st.header("🔔 Watchtower")
    with st.spinner("Escaneando..."): alertas = generar_feed_alertas(WATCHLIST)
    if alertas:
        for a in alertas:
            c = "alert-card-high" if a['Nivel']=="ALTA" else "alert-card-med"
            st.markdown(f"<div class='{c}'><b>{a['Ticker']}</b><br><small>{a['Mensaje']}</small></div>", unsafe_allow_html=True)
    else: st.success("Mercado tranquilo.")

df_pos = auditar_posiciones_sql()
k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Patrimonio", f"${df_pos['Valor Mercado'].sum() if not df_pos.empty else 0:,.2f}")
with k2: st.metric("P&L Total", f"${df_pos['P&L ($)'].sum() if not df_pos.empty else 0:+.2f}")
with k3: st.metric("Alertas", f"{len(alertas)}")
with k4: 
    fg_val, fg_class = obtener_crypto_sentiment()
    if fg_val: st.metric("Crypto Mood", f"{fg_val}/100", f"{fg_class}")

st.divider()

main_tabs = st.tabs(["🔍 AUDITORÍA FORENSE", "🧬 LABORATORIO QUANT", "💼 MESA DE DINERO"])

# --- TAB 1: AUDITORÍA FORENSE (NUEVO V60) ---
with main_tabs[0]:
    sel_ticker = st.selectbox("Auditar Activo:", WATCHLIST, key="aud_tick")
    ES_CRYPTO = "USD" in sel_ticker and "BTC" in sel_ticker or "ETH" in sel_ticker
    
    col_audit1, col_audit2 = st.columns([1, 2])
    
    with col_audit1:
        st.write("---")
        if ES_CRYPTO:
            st.info("⚠️ El análisis forense contable (Z-Score) no aplica a Criptomonedas (no tienen Balance Sheet).")
            # Fallback a Crypto Analysis
            cry_data = analizar_fundamental_crypto(sel_ticker)
            if cry_data:
                st.metric("Precio", f"${cry_data['Precio']:,.2f}")
                st.metric("Trend Secular", f"${cry_data['FairValue_200SMA']:,.2f}")
                st.info(cry_data['Status'])
        else:
            # AUDITORÍA DE ACCIONES (V60)
            st.markdown(f"### 📋 Informe Contable: {sel_ticker}")
            audit = realizar_auditoria_forense(sel_ticker)
            
            if audit:
                # 1. Z-SCORE
                st.markdown("#### 1. Solvencia (Altman Z-Score)")
                st.markdown(f"<div class='{audit['Z_Color']}'>{audit['Z_Score']:.2f} | {audit['Z_Status']}</div>", unsafe_allow_html=True)
                st.caption("Predice riesgo de quiebra en 2 años. (>3.0 es seguro)")
                
                st.divider()
                
                # 2. RADIOGRAFÍA FINANCIERA
                st.markdown("#### 2. Ratios Clave")
                m1, m2 = st.columns(2)
                m1.metric("Deuda/Patrimonio", f"{audit['Deuda/Patrimonio']:.2f}", help="Menos es mejor. >200 es alto riesgo.")
                m2.metric("Current Ratio", f"{audit['Current Ratio']:.2f}", help="Liquidez corto plazo. >1.5 es ideal.")
                st.metric("ROA (Rentabilidad Activos)", f"{audit['ROA']:.2f}%")
                
            else:
                st.warning("Datos contables insuficientes para este activo.")

    with col_audit2:
        f = graficar_master(sel_ticker)
        if f: st.plotly_chart(f, use_container_width=True)

# --- TAB 2: LABORATORIO ---
with main_tabs[1]:
    st.subheader("🛠️ Constructor de Estrategias")
    tk_back = st.selectbox("Activo:", WATCHLIST, key="bt_tk")
    rsi_trigger = st.slider("RSI Compra <", 10, 50, 30)
    filter_trend = st.checkbox("Filtro Tendencia (SMA200)", True)
    sl_input = st.number_input("Stop Loss %", 0.0, 20.0, 5.0)
    tp_input = st.number_input("Take Profit %", 0.0, 50.0, 15.0)
    
    if st.button("🚀 EJECUTAR BACKTEST"):
        p = {"rsi_buy": rsi_trigger, "use_trend": filter_trend, "sl": sl_input, "tp": tp_input}
        res = ejecutar_backtest_custom(tk_back, 10000, p)
        if res:
            c1, c2 = st.columns(2)
            c1.metric("Retorno", f"{res['Retorno']:.2f}%")
            c2.metric("Trades", res['Trades'])
            st.plotly_chart(px.line(res['Equity'], y="Equity"), use_container_width=True)

# --- TAB 3: OPERATIVA ---
with main_tabs[2]:
    if not df_pos.empty: st.dataframe(df_pos)
    else: st.info("Cartera vacía.")
    with st.form("op"):
        t = st.selectbox("Ticker", WATCHLIST, key="op_tk2"); tp = st.selectbox("Tipo", ["COMPRA", "VENTA"])
        q = st.number_input("Qty", 1, 10000); pr = st.number_input("Precio", 0.0)
        if st.form_submit_button("Ejecutar"): registrar_operacion_sql(t, tp, q, pr); st.rerun()
