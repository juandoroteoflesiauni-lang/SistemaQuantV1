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

st.set_page_config(page_title="Sistema Quant V62 (The Valuator)", layout="wide", page_icon="🧮")
st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 15px; text-align: center;}
    .dcf-card {background-color: #1a2634; border: 1px solid #3d5a80; padding: 15px; border-radius: 8px;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #0e1117; border-radius: 5px;}
    .stTabs [aria-selected="true"] {background-color: #262730;}
</style>""", unsafe_allow_html=True)

WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'COIN', 'KO', 'DIS', 'SPY', 'QQQ', 'GLD', 'USO']
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

# --- MOTOR DE VALUACIÓN DCF (NUEVO V62) ---
def calcular_modelo_dcf(ticker, tasa_crecimiento, wacc, tasa_terminal, años=5):
    """Calcula el Valor Intrínseco usando Discounted Cash Flow"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 1. Obtener Free Cash Flow (FCF) Base
        fcf = info.get('freeCashflow')
        
        # Si no hay dato directo, estimamos: Operating Cash Flow - CapEx
        if fcf is None:
            ocf = info.get('operatingCashflow', 0)
            # CapEx suele no estar en 'info' directo a veces, asumimos un % o 0 si falla
            fcf = ocf * 0.8 # Estimación grosera si falta dato exacto
            
        if fcf <= 0: return None # No se puede valorar por DCF si no genera caja
        
        shares = info.get('sharesOutstanding', 1)
        net_debt = info.get('totalDebt', 0) - info.get('totalCash', 0)
        
        # 2. Proyectar FCF Futuros
        fcf_proyectados = []
        fcf_actual = fcf
        
        for i in range(1, años + 1):
            fcf_actual = fcf_actual * (1 + tasa_crecimiento)
            fcf_proyectados.append(fcf_actual)
            
        # 3. Calcular Valor Terminal (Perpetuidad)
        # TV = (FCF_Final * (1 + g_terminal)) / (WACC - g_terminal)
        fcf_final = fcf_proyectados[-1]
        terminal_value = (fcf_final * (1 + tasa_terminal)) / (wacc - tasa_terminal)
        
        # 4. Descontar a Valor Presente (PV)
        pv_fcf = 0
        for i, val in enumerate(fcf_proyectados):
            pv_fcf += val / ((1 + wacc) ** (i + 1))
            
        pv_terminal = terminal_value / ((1 + wacc) ** años)
        
        enterprise_value = pv_fcf + pv_terminal
        equity_value = enterprise_value - net_debt
        fair_value_share = equity_value / shares
        
        # Generar Tabla de Sensibilidad
        sensibilidad = []
        growth_ranges = [tasa_crecimiento - 0.02, tasa_crecimiento, tasa_crecimiento + 0.02]
        wacc_ranges = [wacc - 0.01, wacc, wacc + 0.01]
        
        for g in growth_ranges:
            row = []
            for w in wacc_ranges:
                # Calculo rápido para la matriz
                fcf_iter = fcf
                pv_iter = 0
                for i in range(1, años + 1):
                    fcf_iter *= (1 + g)
                    pv_iter += fcf_iter / ((1 + w) ** i)
                tv_iter = (fcf_iter * (1 + tasa_terminal)) / (w - tasa_terminal)
                pv_tv_iter = tv_iter / ((1 + w) ** años)
                val_iter = (pv_iter + pv_tv_iter - net_debt) / shares
                row.append(val_iter)
            sensibilidad.append(row)
            
        return {
            "Fair Value": fair_value_share,
            "FCF Base": fcf,
            "Enterprise Value": enterprise_value,
            "Equity Value": equity_value,
            "Sensibilidad": sensibilidad,
            "Ejes": {"Growth": growth_ranges, "WACC": wacc_ranges}
        }
        
    except Exception as e: return None

# --- MOTORES EXISTENTES ---
@st.cache_data(ttl=600)
def calcular_score_quant(ticker):
    score=0; b={"Técnico":0, "Fundamental":0, "Riesgo":0}
    try:
        h = yf.Ticker(ticker).history(period="6mo", interval="1d", auto_adjust=True)
        if not h.empty:
            h['RSI']=ta.rsi(h['Close'],14); h['SMA']=ta.sma(h['Close'],50)
            if 30<=h['RSI'].iloc[-1]<=65: b['Técnico']+=20
            elif h['RSI'].iloc[-1]<30: b['Técnico']+=15
            else: b['Técnico']+=5
            if h['Close'].iloc[-1]>h['SMA'].iloc[-1]: b['Técnico']+=20
        i = yf.Ticker(ticker).info
        if i.get('trailingEps'): b['Fundamental']+=20
        if "USD" in ticker: b['Fundamental']=20
        be = i.get('beta',1.0) or 1.0
        if 0.8<=be<=1.2: b['Riesgo']+=20
        elif be<0.8: b['Riesgo']+=15
        else: b['Riesgo']+=10
        score = sum(b.values())
        return score, b
    except: return 0, b

def dibujar_velocimetro(score):
    return go.Figure(go.Indicator(mode="gauge+number", value=score, domain={'x': [0, 1], 'y': [0, 1]}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "white"}, 'steps': [{'range': [0, 40], 'color': "#ff4b4b"}, {'range': [40, 70], 'color': "#ffa500"}, {'range': [70, 100], 'color': "#00cc96"}]})).update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="#0e1117", font={'color': "white"})

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
def generar_feed_alertas(tickers):
    alertas = []
    try: data = yf.download(" ".join(tickers), period="1y", group_by='ticker', progress=False, auto_adjust=True)
    except: return []
    for t in tickers:
        try:
            df = data[t].dropna() if len(tickers)>1 else data.dropna()
            if len(df)<200: continue
            close=df['Close']; s50=ta.sma(close,50); s200=ta.sma(close,200)
            if s50.iloc[-2]<s200.iloc[-2] and s50.iloc[-1]>s200.iloc[-1]: alertas.append({"Ticker":t,"Nivel":"ALTA","Mensaje":"🌟 GOLDEN CROSS"})
            if s50.iloc[-2]>s200.iloc[-2] and s50.iloc[-1]<s200.iloc[-1]: alertas.append({"Ticker":t,"Nivel":"ALTA","Mensaje":"☠️ DEATH CROSS"})
        except: pass
    return alertas

# --- INTERFAZ V62: THE VALUATOR ---
st.title("🧮 Sistema Quant V62: The Valuator")

with st.sidebar:
    st.header("🔔 Watchtower")
    alertas = generar_feed_alertas(WATCHLIST)
    if alertas:
        for a in alertas: st.markdown(f"<div class='alert-card-high'><b>{a['Ticker']}</b><br><small>{a['Mensaje']}</small></div>", unsafe_allow_html=True)
    else: st.success("Sin alertas.")

df_pos = auditar_posiciones_sql()
k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Patrimonio", f"${df_pos['Valor Mercado'].sum() if not df_pos.empty else 0:,.2f}")
with k2: st.metric("P&L Total", f"${df_pos['P&L ($)'].sum() if not df_pos.empty else 0:+.2f}")
with k3: st.metric("SPY", f"${yf.Ticker('SPY').history(period='1d')['Close'].iloc[-1]:.2f}")
with k4: st.metric("Bitcoin", f"${yf.Ticker('BTC-USD').history(period='1d')['Close'].iloc[-1]:,.0f}")

st.divider()

main_tabs = st.tabs(["🧮 MODELO DCF", "📊 ANÁLISIS TÉCNICO", "💼 MESA DE DINERO"])

# --- TAB 1: VALUACIÓN DCF (V62) ---
with main_tabs[0]:
    col_d1, col_d2 = st.columns([1, 2])
    
    with col_d1:
        st.subheader("Configuración del Modelo")
        sel_dcf = st.selectbox("Activo a Valorar:", WATCHLIST)
        
        # Parámetros Financieros (Ajustables por el Contador)
        st.markdown("#### Supuestos Financieros")
        growth_rate = st.slider("Crecimiento Esperado (5 años)", 0.01, 0.30, 0.10, 0.01, format="%.2f")
        wacc = st.slider("WACC (Costo de Capital)", 0.05, 0.15, 0.09, 0.005, format="%.3f")
        terminal_growth = st.slider("Crecimiento Terminal (Perpetuidad)", 0.01, 0.05, 0.025, 0.005, format="%.3f")
        
        if st.button("🧮 CALCULAR VALOR INTRÍNSECO"):
            with st.spinner("Proyectando flujos de caja a perpetuidad..."):
                dcf_result = calcular_modelo_dcf(sel_dcf, growth_rate, wacc, terminal_growth)
                st.session_state['dcf'] = dcf_result

    with col_d2:
        if 'dcf' in st.session_state and st.session_state['dcf']:
            res = st.session_state['dcf']
            current_px = float(yf.Ticker(sel_dcf).history(period="1d")['Close'].iloc[-1])
            
            # Tarjeta Principal
            upside = ((res['Fair Value'] - current_px) / current_px) * 100
            color_val = "green" if upside > 0 else "red"
            
            st.markdown(f"""
            <div class='dcf-card' style='text-align: center;'>
                <h2>Valor Justo (Fair Value): ${res['Fair Value']:.2f}</h2>
                <h3 style='color: {color_val};'>Upside/Downside: {upside:+.2f}%</h3>
                <p>Precio Actual: ${current_px:.2f} | Enterprise Value: ${res['Enterprise Value']/1e9:.2f}B</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("---")
            
            # Matriz de Sensibilidad
            st.markdown("#### 🌡️ Análisis de Sensibilidad (Precio Justo según escenarios)")
            
            # Crear DataFrame para el Heatmap
            df_sens = pd.DataFrame(
                res['Sensibilidad'], 
                index=[f"G: {g:.1%}" for g in res['Ejes']['Growth']],
                columns=[f"WACC: {w:.1%}" for w in res['Ejes']['WACC']]
            )
            
            fig_heat = px.imshow(df_sens, text_auto=".2f", aspect="auto", 
                                 color_continuous_scale="RdYlGn",
                                 title="Matriz: Crecimiento (Eje Y) vs WACC (Eje X)")
            st.plotly_chart(fig_heat, use_container_width=True)
            
        else:
            st.info("Selecciona un activo y pulsa Calcular. (Nota: DCF no funciona bien para Criptos o empresas con pérdidas).")

# --- TAB 2: ANÁLISIS TÉCNICO ---
with main_tabs[1]:
    sel_tech = st.selectbox("Gráfico:", WATCHLIST, key="tech_sel")
    c_t1, c_t2 = st.columns([1, 2])
    with c_t1:
        s, b = calcular_score_quant(sel_tech)
        st.plotly_chart(dibujar_velocimetro(s), use_container_width=True)
    with c_t2:
        f = graficar_master(sel_tech)
        if f: st.plotly_chart(f, use_container_width=True)

# --- TAB 3: OPERATIVA ---
with main_tabs[2]:
    if not df_pos.empty: st.dataframe(df_pos)
    else: st.info("Cartera vacía.")
    with st.form("op"):
        t = st.selectbox("Ticker", WATCHLIST, key="op_tk"); tp = st.selectbox("Tipo", ["COMPRA", "VENTA"])
        q = st.number_input("Qty", 1, 10000); p = st.number_input("Precio", 0.0)
        if st.form_submit_button("Ejecutar"): registrar_operacion_sql(t, tp, q, p); st.rerun()
