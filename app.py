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
from scipy.stats import norm 
import google.generativeai as genai

# --- CONFIGURACIÓN ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sistema Quant V65 (Consensus)", layout="wide", page_icon="👥")

st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;}
    .consensus-card {background-color: #1e2a38; border: 1px solid #2196F3; border-radius: 8px; padding: 15px; text-align: center;}
    .rec-buy {color: #00ff00; font-weight: bold;}
    .rec-hold {color: #ffa500; font-weight: bold;}
    .rec-sell {color: #ff0000; font-weight: bold;}
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
            px = float(curr.iloc[-1]) if len(activos) == 1 else float(curr.iloc[-1][t])
            val = d['Qty'] * px; pnl = val - d['Cost']
            res.append({"Ticker": t, "Cantidad": d['Qty'], "Valor": val, "P&L": pnl})
        except: pass
    return pd.DataFrame(res)

init_db()

# --- MOTOR DE CONSENSO (NUEVO V65) ---
@st.cache_data(ttl=3600)
def obtener_consenso_analistas(ticker):
    """Extrae recomendaciones y precios objetivos de analistas"""
    if "USD" in ticker and "BTC" not in ticker: return None # Filtro básico para cryptos no principales
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 1. Recomendación (1-5)
        # 1=Strong Buy, 2=Buy, 3=Hold, 4=Sell, 5=Strong Sell
        rec_mean = info.get('recommendationMean')
        rec_key = info.get('recommendationKey', 'none').upper().replace('_', ' ')
        
        # 2. Precios Objetivo
        target_low = info.get('targetLowPrice')
        target_mean = info.get('targetMeanPrice')
        target_high = info.get('targetHighPrice')
        current = info.get('currentPrice') or info.get('regularMarketPreviousClose')
        
        if not target_mean or not current: return None
        
        upside = ((target_mean - current) / current) * 100
        num_analysts = info.get('numberOfAnalystOpinions', 0)
        
        return {
            "Recomendación": rec_key,
            "Score": rec_mean,
            "Target Low": target_low,
            "Target Mean": target_mean,
            "Target High": target_high,
            "Precio Actual": current,
            "Upside %": upside,
            "Analistas": num_analysts
        }
    except: return None

# --- MOTORES DE SOPORTE ---
@st.cache_data(ttl=900)
def escanear_mercado_completo(tickers):
    ranking = []
    try: data_hist = yf.download(" ".join(tickers), period="1y", group_by='ticker', progress=False, auto_adjust=True)
    except: return pd.DataFrame()
    for t in tickers:
        try:
            df = data_hist[t].dropna() if len(tickers)>1 else data_hist.dropna()
            info = yf.Ticker(t).info
            if df.empty: continue
            pe = info.get('trailingPE', 50); val = max(0, min(100, (60 - pe) * 2)) if pe > 0 else 0
            rev = info.get('revenueGrowth', 0) * 100; gro = max(0, min(100, rev * 3.3))
            curr = df['Close'].iloc[-1]; s200 = df['Close'].rolling(200).mean().iloc[-1]; rsi = ta.rsi(df['Close'], 14).iloc[-1]
            mom = 0
            if curr > s200: mom += 50
            if rsi > 50: mom += (rsi - 50) * 2
            mom = max(0, min(100, mom))
            roe = info.get('returnOnEquity', 0) * 100; mar = info.get('profitMargins', 0) * 100; qua = max(0, min(100, (roe * 2) + mar))
            score = (val * 0.2) + (gro * 0.2) + (mom * 0.3) + (qua * 0.3)
            if "USD" in t: score = (mom * 0.6) + (gro * 0.4)
            ranking.append({"Ticker": t, "Score": round(score, 1), "Precio": curr, "Value": round(val,0), "Momentum": round(mom,0)})
        except: pass
    return pd.DataFrame(ranking).sort_values(by="Score", ascending=False)

def dibujar_radar_factores(scores):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=list(scores.values()), theta=list(scores.keys()), fill='toself', line_color='#00ff00', fillcolor='rgba(0, 255, 0, 0.2)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], color='grey')), showlegend=False, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color='white'), height=300, margin=dict(l=40, r=40, t=20, b=20))
    return fig

def graficar_simple(ticker):
    df = yf.Ticker(ticker).history(period="6mo", auto_adjust=True)
    if df.empty: return None
    df['SMA50'] = ta.sma(df['Close'], 50)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Precio'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='yellow'), name='SMA 50'))
    fig.update_layout(template="plotly_dark", height=350, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
    return fig

def calcular_dcf_rapido(ticker):
    try:
        i = yf.Ticker(ticker).info
        fcf = i.get('freeCashflow', i.get('operatingCashflow', 0)*0.8)
        if fcf <= 0: return None
        growth = 0.10; wacc = 0.09; shares = i.get('sharesOutstanding', 1)
        pv = 0
        for y in range(1, 6): pv += (fcf * ((1+growth)**y)) / ((1+wacc)**y)
        term = (fcf * ((1+growth)**5) * 1.02) / (wacc - 0.02); pv_term = term / ((1+wacc)**5)
        val = (pv + pv_term) / shares
        return val
    except: return None

@st.cache_data(ttl=600)
def calcular_factores_quant_single(ticker):
    try:
        stock = yf.Ticker(ticker); df = stock.history(period="1y", interval="1d", auto_adjust=True); info = stock.info
        if df.empty: return None
        pe = info.get('trailingPE', 50); score_value = max(0, min(100, (60 - pe) * 2)) if pe > 0 else 0
        rev = info.get('revenueGrowth', 0) * 100; score_growth = max(0, min(100, rev * 3.3))
        curr = df['Close'].iloc[-1]; s200 = df['Close'].rolling(200).mean().iloc[-1]; rsi = ta.rsi(df['Close'], 14).iloc[-1]
        m = 0
        if curr > s200: m += 50
        if rsi > 50: m += (rsi - 50) * 2
        score_mom = max(0, min(100, m))
        roe = info.get('returnOnEquity', 0) * 100; mar = info.get('profitMargins', 0) * 100; score_qual = max(0, min(100, (roe * 2) + mar))
        beta = info.get('beta', 1.5) or 1.0; score_vol = max(0, min(100, (2 - beta) * 100))
        return {"Value": score_value, "Growth": score_growth, "Momentum": score_mom, "Quality": score_qual, "Low Vol": score_vol}
    except: return None

# --- INTERFAZ V65 ---
c1, c2 = st.columns([3, 1])
with c1: st.title("🖥️ Quant Terminal V65")
with c2: sel_ticker = st.selectbox("ACTIVO PRINCIPAL", WATCHLIST)

stock = yf.Ticker(sel_ticker); hist = stock.history(period="2d"); info = stock.info
if not hist.empty:
    curr = hist['Close'].iloc[-1]; prev = hist['Close'].iloc[-2]; delta = ((curr - prev) / prev) * 100
    col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
    col_k1.metric("Precio", f"${curr:.2f}", f"{delta:+.2f}%")
    col_k2.metric("RSI (14)", f"{ta.rsi(stock.history(period='30d')['Close'], 14).iloc[-1]:.0f}")
    col_k3.metric("Volumen", f"{info.get('volume', 0)/1e6:.1f}M")
    col_k4.metric("Beta", f"{info.get('beta', 1.0):.2f}")
    col_k5.metric("Target (Consenso)", f"${info.get('targetMeanPrice', 0):.2f}")

st.divider()

col_main, col_side = st.columns([2, 1])

with col_main:
    st.subheader("📉 Acción del Precio")
    fig_chart = graficar_simple(sel_ticker)
    if fig_chart: st.plotly_chart(fig_chart, use_container_width=True)
    
    # TABS DETALLE (V65)
    tabs_detail = st.tabs(["👥 Consenso", "🧮 Valuación (DCF)", "📰 Noticias"])
    
    # 1. CONSENSO ANALISTAS (NUEVO V65)
    with tabs_detail[0]:
        cons = obtener_consenso_analistas(sel_ticker)
        if cons:
            c_c1, c_c2 = st.columns([1, 2])
            
            with c_c1:
                rec_text = cons['Recomendación']
                color_rec = "rec-buy" if "BUY" in rec_text else "rec-sell" if "SELL" in rec_text else "rec-hold"
                
                st.markdown(f"""
                <div class='consensus-card'>
                    <h4 style='color:#ccc'>Recomendación</h4>
                    <h2 class='{color_rec}'>{rec_text}</h2>
                    <p>Score: {cons['Score']:.1f} (1=Buy, 5=Sell)</p>
                    <small>Basado en {cons['Analistas']} Analistas</small>
                </div>
                """, unsafe_allow_html=True)
                
            with c_c2:
                # Gráfico de Target Price
                fig_target = go.Figure()
                # Precio actual
                fig_target.add_trace(go.Scatter(x=[1], y=[cons['Precio Actual']], mode='markers+text', marker=dict(size=15, color='white'), text=["Actual"], textposition="bottom center", name="Actual"))
                # Rangos
                fig_target.add_trace(go.Bar(x=[1], y=[cons['Target Low']], name='Low', marker_color='red', opacity=0.3))
                fig_target.add_trace(go.Bar(x=[1], y=[cons['Target Mean'] - cons['Target Low']], base=cons['Target Low'], name='Mean', marker_color='yellow', opacity=0.3))
                fig_target.add_trace(go.Bar(x=[1], y=[cons['Target High'] - cons['Target Mean']], base=cons['Target Mean'], name='High', marker_color='green', opacity=0.3))
                
                fig_target.update_layout(title=f"Objetivos de Precio (Upside: {cons['Upside %']:.1f}%)", barmode='stack', showlegend=False, xaxis=dict(visible=False), paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color='white'), height=200, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_target, use_container_width=True)
        else:
            st.info("Datos de consenso no disponibles para este activo.")

    with tabs_detail[1]:
        dcf_val = calcular_dcf_rapido(sel_ticker)
        if dcf_val: st.metric("Valor Intrínseco (Modelo DCF)", f"${dcf_val:.2f}", f"{((dcf_val-curr)/curr)*100:+.1f}% Upside")
        else: st.warning("DCF no aplica.")
        
    with tabs_detail[2]:
        if st.button("🤖 Analizar Noticias"):
            with st.spinner("Leyendo..."):
                try: st.write(model.generate_content(f"Analisis breve de {sel_ticker} hoy").text)
                except: st.error("Sin conexión IA")

with col_side:
    st.subheader("🧬 Perfil Quant")
    factores = calcular_factores_quant_single(sel_ticker)
    if factores:
        st.plotly_chart(dibujar_radar_factores(factores), use_container_width=True)
        st.write(f"**💎 Quality:** {factores['Quality']}")
        st.write(f"**🚀 Momentum:** {factores['Momentum']}")
        st.write(f"**🏷️ Value:** {factores['Value']}")
    
    st.markdown("---")
    st.subheader("⚡ Quick Trade")
    with st.form("quick_order"):
        q_qty = st.number_input("Cantidad", 1, 1000, 10); q_side = st.selectbox("Lado", ["COMPRA", "VENTA"])
        if st.form_submit_button("EJECUTAR"):
            registrar_operacion_sql(sel_ticker, q_side, q_qty, curr); st.success("Orden Enviada!")

st.markdown("---")
st.subheader("🏆 Ranking de Mercado")
if st.button("🔄 ESCANEAR"):
    st.dataframe(escanear_mercado_completo(WATCHLIST), use_container_width=True)s
