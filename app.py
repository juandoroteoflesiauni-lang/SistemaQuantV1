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
import time  # <--- NUEVO: Para pausar entre peticiones
import requests 
from datetime import datetime, timedelta
from scipy.stats import norm 
from sklearn.linear_model import LinearRegression
import google.generativeai as genai

# --- CONFIGURACIÓN ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sistema Quant V69 (Iron Shield)", layout="wide", page_icon="🛡️")

st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;}
    .thesis-card {background-color: #1e1e2e; border-left: 4px solid #00BFFF; padding: 15px; border-radius: 5px;}
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

# --- MOTOR DE DATOS BLINDADO (NUEVO V69) ---
@st.cache_data(ttl=1800) # Cache de 30 minutos para evitar bloqueos
def obtener_datos_snapshot(ticker):
    """Descarga segura de datos básicos para el encabezado"""
    try:
        stock = yf.Ticker(ticker)
        # Pedimos solo lo necesario
        hist = stock.history(period="5d")
        if hist.empty: return None
        
        # Info básica (propenso a fallar, lo envolvemos)
        try: info = stock.info
        except: info = {}
        
        return {
            "Precio": hist['Close'].iloc[-1],
            "Previo": hist['Close'].iloc[-2],
            "RSI": ta.rsi(hist['Close'], 14).iloc[-1] if len(hist) > 14 else 50,
            "Volumen": info.get('volume', 0),
            "Beta": info.get('beta', 1.0),
            "Target": info.get('targetMeanPrice', 0)
        }
    except Exception: return None

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
    
    # Descarga segura en bloque
    try: curr = yf.download(" ".join(activos), period="1d", progress=False, auto_adjust=True)['Close']
    except: return pd.DataFrame() # Si falla, devuelve vacío, no crash
    
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

# --- MOTOR DE TESIS DE INVERSIÓN ---
def generar_tesis_automatica(ticker, datos_tecnicos, datos_fundamentales, prediccion, estacionalidad):
    puntos_bull = []; puntos_bear = []
    
    # Técnico
    if datos_tecnicos['RSI'] < 30: puntos_bull.append("Sobreventa Técnica")
    elif datos_tecnicos['RSI'] > 70: puntos_bear.append("Sobrecompra Técnica")
    if datos_tecnicos['Precio'] > datos_tecnicos['SMA200']: puntos_bull.append("Tendencia Alcista")
    else: puntos_bear.append("Tendencia Bajista")
    
    # Fundamental
    if datos_fundamentales and datos_fundamentales.get('Upside', 0) > 10: puntos_bull.append("Subvaluada (DCF)")
    
    # Predicción
    if prediccion and prediccion['Cambio_Pct'] > 5: puntos_bull.append("Proyección Positiva")
    
    score = len(puntos_bull) - len(puntos_bear)
    veredicto = "COMPRA 🟢" if score > 0 else "VENTA 🔴" if score < 0 else "NEUTRAL 🟡"
    return {"Veredicto": veredicto, "Pros": puntos_bull, "Contras": puntos_bear}

# --- MOTORES RESTAURADOS ---
@st.cache_data(ttl=1800) # Cache aumentado a 30 min
def escanear_mercado_completo(tickers):
    ranking = []
    # Descarga masiva para evitar rate limit en precios
    try: data_hist = yf.download(" ".join(tickers), period="1y", group_by='ticker', progress=False, auto_adjust=True)
    except: return pd.DataFrame()
    
    for t in tickers:
        try:
            # Pausa táctica para no saturar al pedir 'info'
            time.sleep(0.1) 
            
            df = data_hist[t].dropna() if len(tickers)>1 else data_hist.dropna()
            if df.empty: continue
            
            # Info básica segura
            try: info = yf.Ticker(t).info
            except: info = {}
            
            pe = info.get('trailingPE', 50); val = max(0, min(100, (60 - pe) * 2)) if pe > 0 else 0
            curr = df['Close'].iloc[-1]; s200 = df['Close'].rolling(200).mean().iloc[-1]; rsi = ta.rsi(df['Close'], 14).iloc[-1]
            mom = 0
            if curr > s200: mom += 50
            if rsi > 50: mom += (rsi - 50) * 2
            mom = max(0, min(100, mom))
            
            score = (val * 0.4) + (mom * 0.6)
            if "USD" in t: score = mom
            
            ranking.append({"Ticker": t, "Score": round(score, 1), "Precio": curr, "Value": round(val,0), "Momentum": round(mom,0)})
        except: pass
    return pd.DataFrame(ranking).sort_values(by="Score", ascending=False)

def generar_proyeccion_futura(ticker, dias=30):
    try:
        df = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
        if df.empty: return None
        df = df.reset_index(); df['Day'] = df.index
        X = df[['Day']]; y = df['Close']
        model_lr = LinearRegression(); model_lr.fit(X, y)
        last_day = df['Day'].iloc[-1]
        future_days = np.array(range(last_day + 1, last_day + dias + 1)).reshape(-1, 1)
        pred = model_lr.predict(future_days)
        std = (y - model_lr.predict(X)).std()
        upper = pred + (1.96 * std * np.sqrt(np.arange(1, dias + 1)))
        lower = pred - (1.96 * std * np.sqrt(np.arange(1, dias + 1)))
        dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, dias + 1)]
        return {"Fechas": dates, "Predicción": pred, "Upper": upper, "Lower": lower, "Cambio_Pct": ((pred[-1]-y.iloc[-1])/y.iloc[-1])*100}
    except: return None

@st.cache_data(ttl=3600)
def analizar_estacionalidad(ticker):
    try:
        df = yf.Ticker(ticker).history(period="10y", auto_adjust=True)
        if df.empty: return None
        df['Retorno'] = df['Close'].pct_change()
        df['Mes_Num'] = df.index.month
        pivot = df.groupby([df.index.year, 'Mes_Num'])['Retorno'].apply(lambda x: (1+x).prod()-1).unstack()*100
        avg = df.groupby('Mes_Num')['Retorno'].mean()*100*21
        meses_dict = {1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}
        # Fix V69: Validar índices
        best = avg.idxmax(); worst = avg.idxmin()
        return {"Heatmap": pivot, "Avg_Seasonality": avg, "Best_Month": meses_dict.get(best, str(best)), "Worst_Month": meses_dict.get(worst, str(worst))}
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

def graficar_simple(ticker):
    try:
        df = yf.Ticker(ticker).history(period="6mo", auto_adjust=True)
        if df.empty: return None
        df['SMA50'] = ta.sma(df['Close'], 50)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Precio'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='yellow'), name='SMA 50'))
        fig.update_layout(template="plotly_dark", height=350, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0)); return fig
    except: return None

def obtener_consenso_analistas(ticker):
    if "USD" in ticker: return None
    try:
        info = yf.Ticker(ticker).info
        return {
            "Recomendación": info.get('recommendationKey', 'N/A').upper(),
            "Target Mean": info.get('targetMeanPrice', 0),
            "Target High": info.get('targetHighPrice', 0),
            "Precio Actual": info.get('currentPrice', 0)
        }
    except: return None

@st.cache_data(ttl=600)
def calcular_factores_quant_single(ticker):
    try:
        stock = yf.Ticker(ticker); df = stock.history(period="1y", interval="1d", auto_adjust=True); info = stock.info
        if df.empty: return None
        pe = info.get('trailingPE', 50); score_value = max(0, min(100, (60 - pe) * 2)) if pe > 0 else 0
        score_growth = 50 # Simplificado para velocidad
        curr = df['Close'].iloc[-1]; s200 = df['Close'].rolling(200).mean().iloc[-1]; rsi = ta.rsi(df['Close'], 14).iloc[-1]
        m = 0
        if curr > s200: m += 50
        if rsi > 50: m += (rsi - 50) * 2
        score_mom = max(0, min(100, m))
        score_qual = 50 # Simplificado
        beta = info.get('beta', 1.5) or 1.0; score_vol = max(0, min(100, (2 - beta) * 100))
        return {"Value": score_value, "Growth": score_growth, "Momentum": score_mom, "Quality": score_qual, "Low Vol": score_vol}
    except: return None

def dibujar_radar_factores(scores):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=list(scores.values()), theta=list(scores.keys()), fill='toself', line_color='#00ff00', fillcolor='rgba(0, 255, 0, 0.2)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], color='grey')), showlegend=False, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color='white'), height=300, margin=dict(l=40, r=40, t=20, b=20))
    return fig

# --- INTERFAZ V69 ---
c1, c2 = st.columns([3, 1])
with c1: st.title("🛡️ Quant Terminal V69")
with c2: sel_ticker = st.selectbox("ACTIVO PRINCIPAL", WATCHLIST)

# SNAPSHOT BLINDADO (V69)
snap = obtener_datos_snapshot(sel_ticker)
if snap:
    delta = ((snap['Precio'] - snap['Previo'])/snap['Previo'])*100
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Precio", f"${snap['Precio']:.2f}", f"{delta:+.2f}%")
    k2.metric("RSI", f"{snap['RSI']:.0f}")
    k3.metric("Vol", f"{snap['Volumen']/1e6:.1f}M")
    k4.metric("Beta", f"{snap['Beta']:.2f}")
    k5.metric("Target", f"${snap['Target']:.2f}")
else:
    st.warning("⚠️ Yahoo Finance está limitando las peticiones. Espera unos segundos.")

st.divider()

col_main, col_side = st.columns([2, 1])

with col_main:
    st.subheader("📉 Acción del Precio")
    fig_chart = graficar_simple(sel_ticker)
    if fig_chart: st.plotly_chart(fig_chart, use_container_width=True)
    
    tabs_detail = st.tabs(["📝 Tesis", "🔮 Predicción", "📅 Ciclos", "🧮 Valuación"])
    
    with tabs_detail[0]:
        st.subheader("📝 Tesis Automática")
        if snap:
            hist_long = yf.Ticker(sel_ticker).history(period="1y") # Necesario para SMA200
            if not hist_long.empty:
                sma200 = ta.sma(hist_long['Close'], 200).iloc[-1]
                dat_tec = {"RSI": snap['RSI'], "Precio": snap['Precio'], "SMA200": sma200}
                dcf_val = calcular_dcf_rapido(sel_ticker)
                dat_fund = {"Upside": ((dcf_val - snap['Precio'])/snap['Precio'])*100} if dcf_val else {}
                pred = generar_proyeccion_futura(sel_ticker)
                cycle = analizar_estacionalidad(sel_ticker)
                
                tesis = generar_tesis_automatica(sel_ticker, dat_tec, dat_fund, pred, cycle)
                
                st.markdown(f"<div class='thesis-card'><h2>VEREDICTO: {tesis['Veredicto']}</h2></div>", unsafe_allow_html=True)
                c_p, c_c = st.columns(2)
                with c_p:
                    st.markdown("#### ✅ Bullish")
                    for p in tesis['Pros']: st.markdown(f"- {p}")
                with c_c:
                    st.markdown("#### ❌ Bearish")
                    for c in tesis['Contras']: st.markdown(f"- {c}")
            else: st.info("Cargando historial largo...")
        else: st.info("Esperando datos...")

    with tabs_detail[1]:
        pred = generar_proyeccion_futura(sel_ticker)
        if pred:
            st.metric("Objetivo 30d", f"${pred['Predicción'][-1]:.2f}", f"{pred['Cambio_Pct']:+.2f}%")
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=pred['Fechas'], y=pred['Predicción'], mode='lines', name='Tendencia', line=dict(color='white', dash='dash')))
            fig_fc.add_trace(go.Scatter(x=pred['Fechas'], y=pred['Upper'], mode='lines', line=dict(width=0), showlegend=False))
            fig_fc.add_trace(go.Scatter(x=pred['Fechas'], y=pred['Lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 255, 0.1)'))
            fig_fc.update_layout(template="plotly_dark", height=250, margin=dict(l=0,r=0,t=0,b=0)); st.plotly_chart(fig_fc, use_container_width=True)

    with tabs_detail[2]:
        cycle = analizar_estacionalidad(sel_ticker)
        if cycle:
            st.write(f"Mejor Mes: **{cycle['Best_Month']}**")
            st.plotly_chart(px.bar(x=cycle['Avg_Seasonality'].index, y=cycle['Avg_Seasonality'].values, title="Estacionalidad"), use_container_width=True)

    with tabs_detail[3]:
        dcf_val = calcular_dcf_rapido(sel_ticker)
        if dcf_val and snap: st.metric("Valor Justo (DCF)", f"${dcf_val:.2f}", f"{((dcf_val-snap['Precio'])/snap['Precio'])*100:+.1f}% Upside")
        else: st.info("Modelo no aplicable.")

with col_side:
    st.subheader("🧬 Perfil Quant")
    factores = calcular_factores_quant_single(sel_ticker)
    if factores: st.plotly_chart(dibujar_radar_factores(factores), use_container_width=True)
    st.markdown("---")
    st.subheader("⚡ Quick Trade")
    with st.form("quick_order"):
        q_qty = st.number_input("Cantidad", 1, 1000, 10); q_side = st.selectbox("Lado", ["COMPRA", "VENTA"])
        if st.form_submit_button("EJECUTAR"): 
            if snap: registrar_operacion_sql(sel_ticker, q_side, q_qty, snap['Precio']); st.success("Orden Enviada!")
            else: st.error("Espera a que cargue el precio.")
        
    st.markdown("---")
    st.subheader("🏆 Ranking Mercado")
    if st.button("🔄 ESCANEAR"):
        st.info("Escaneando... (Esto tardará unos segundos para respetar límites)")
        st.dataframe(escanear_mercado_completo(WATCHLIST), use_container_width=True)
        
    st.markdown("---")
    st.subheader("💼 Cartera")
    df_p = auditar_posiciones_sql()
    if not df_p.empty: st.dataframe(df_p[['Ticker', 'P&L']], use_container_width=True)
