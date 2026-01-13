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
st.set_page_config(page_title="Sistema Quant V64 (The Hunter)", layout="wide", page_icon="🏆")

st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;}
    .winner-card {background-color: #1b261b; border: 2px solid #00ff00; border-radius: 10px; padding: 15px; text-align: center;}
    .category-badge {background-color: #333; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; margin-right: 5px;}
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

# --- MOTOR DE RANKING MASIVO (NUEVO V64) ---
@st.cache_data(ttl=900) # Cache 15 min para no saturar
def escanear_mercado_completo(tickers):
    """Ejecuta el análisis de Factores en TODOS los activos y devuelve un ranking"""
    ranking = []
    
    # Descarga Masiva (Más eficiente)
    try:
        data_hist = yf.download(" ".join(tickers), period="1y", group_by='ticker', progress=False, auto_adjust=True)
    except: return pd.DataFrame()

    for t in tickers:
        try:
            # Extraer Datos
            if len(tickers) > 1: 
                df = data_hist[t].dropna()
                # Para datos fundamentales (info), lamentablemente yfinance requiere llamadas individuales
                # Optimizamos llamando solo lo esencial o usando un try-except rápido
                info = yf.Ticker(t).info 
            else:
                df = data_hist.dropna()
                info = yf.Ticker(t).info

            if df.empty: continue

            # --- CÁLCULO DE FACTORES (V63 Lógica) ---
            # 1. VALUE
            pe = info.get('trailingPE', 50)
            score_value = max(0, min(100, (60 - pe) * 2)) if pe > 0 else 0
            
            # 2. GROWTH
            rev_growth = info.get('revenueGrowth', 0) * 100
            score_growth = max(0, min(100, rev_growth * 3.3))
            
            # 3. MOMENTUM
            curr = df['Close'].iloc[-1]
            sma200 = df['Close'].rolling(200).mean().iloc[-1]
            rsi = ta.rsi(df['Close'], 14).iloc[-1]
            m_score = 0
            if curr > sma200: m_score += 50
            if rsi > 50: m_score += (rsi - 50) * 2
            score_momentum = max(0, min(100, m_score))
            
            # 4. QUALITY
            roe = info.get('returnOnEquity', 0) * 100
            margins = info.get('profitMargins', 0) * 100
            score_quality = max(0, min(100, (roe * 2) + margins))
            
            # 5. TOTAL SCORE
            total_score = (score_value * 0.2) + (score_growth * 0.2) + (score_momentum * 0.3) + (score_quality * 0.3)
            
            # Ajuste Cripto (Si no tiene PE, asumimos Value Neutral)
            if "USD" in t:
                total_score = (score_momentum * 0.6) + (score_growth * 0.4) # Cripto es Momentum + Growth
            
            ranking.append({
                "Ticker": t,
                "Score": round(total_score, 1),
                "Precio": curr,
                "Value": round(score_value, 0),
                "Growth": round(score_growth, 0),
                "Momentum": round(score_momentum, 0),
                "Quality": round(score_quality, 0)
            })
            
        except: pass
        
    return pd.DataFrame(ranking).sort_values(by="Score", ascending=False)

# --- MOTORES DE SOPORTE ---
def dibujar_radar_factores(scores):
    categories = list(scores.keys()); values = list(scores.values())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', line_color='#00ff00', fillcolor='rgba(0, 255, 0, 0.2)'))
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

# --- INTERFAZ V64: THE HUNTER ---
st.title("🏆 Sistema Quant V64: The Hunter")

# TABS PRINCIPALES
main_tabs = st.tabs(["🏆 RANKING DE MERCADO", "🖥️ TERMINAL INDIVIDUAL", "💼 PORTAFOLIO"])

# --- TAB 1: RANKING MASIVO (NUEVO V64) ---
with main_tabs[0]:
    st.subheader("📡 Escáner de Oportunidades en Tiempo Real")
    st.caption("Ranking basado en modelo multifactorial (Value + Growth + Momentum + Quality)")
    
    if st.button("🔄 ESCANEAR MERCADO AHORA"):
        with st.spinner("Analizando fundamentales y técnicos de toda la lista... (Esto puede tomar unos segundos)"):
            df_rank = escanear_mercado_completo(WATCHLIST)
            st.session_state['ranking'] = df_rank
    
    if 'ranking' in st.session_state and not st.session_state['ranking'].empty:
        df_r = st.session_state['ranking']
        
        # 1. PODIO DE GANADORES
        top_1 = df_r.iloc[0]
        top_value = df_r.sort_values("Value", ascending=False).iloc[0]
        top_mom = df_r.sort_values("Momentum", ascending=False).iloc[0]
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='winner-card'><h3>🏆 #1 GENERAL</h3><h1>{top_1['Ticker']}</h1><p>Score: {top_1['Score']}</p></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><h3>💎 TOP VALUE</h3><h2>{top_value['Ticker']}</h2><p>Score Val: {top_value['Value']}</p></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-card'><h3>🚀 TOP MOMENTUM</h3><h2>{top_mom['Ticker']}</h2><p>Score Mom: {top_mom['Momentum']}</p></div>", unsafe_allow_html=True)
        
        st.divider()
        
        # 2. TABLA DE CLASIFICACIÓN
        st.subheader("📊 Tabla de Clasificación Completa")
        
        # Formato de color para la tabla
        def color_score(val):
            color = 'green' if val > 70 else 'orange' if val > 40 else 'red'
            return f'color: {color}; font-weight: bold'
            
        st.dataframe(
            df_r.style.map(color_score, subset=['Score'])
            .bar(subset=['Value'], color='#4CAF50')
            .bar(subset=['Momentum'], color='#2196F3'),
            use_container_width=True,
            column_config={
                "Ticker": "Activo",
                "Score": st.column_config.ProgressColumn("Quant Score", format="%.1f", min_value=0, max_value=100),
                "Precio": st.column_config.NumberColumn("Precio", format="$%.2f")
            }
        )
        
        # 3. MATRIZ DE OPORTUNIDAD (SCATTER)
        st.subheader("🎯 Matriz de Selección: Calidad vs Momentum")
        fig_scatter = px.scatter(df_r, x="Quality", y="Momentum", color="Score", size="Score", text="Ticker",
                                 title="Busca activos en la esquina superior derecha (Alta Calidad + Fuerte Tendencia)",
                                 color_continuous_scale="Viridis")
        fig_scatter.add_hline(y=50, line_dash="dash", line_color="grey")
        fig_scatter.add_vline(x=50, line_dash="dash", line_color="grey")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    else:
        st.info("Dale al botón 'ESCANEAR' para generar el ranking.")

# --- TAB 2: TERMINAL INDIVIDUAL (V63) ---
with main_tabs[1]:
    c_sel, c_res = st.columns([1, 3])
    with c_sel:
        sel_ticker = st.selectbox("ACTIVO:", WATCHLIST)
        stock = yf.Ticker(sel_ticker); hist = stock.history(period="2d")
        if not hist.empty:
            curr = hist['Close'].iloc[-1]; delta = ((curr - hist['Close'].iloc[-2])/hist['Close'].iloc[-2])*100
            st.metric("Precio", f"${curr:.2f}", f"{delta:+.2f}%")
            
    with c_res:
        st.subheader(f"Análisis Profundo: {sel_ticker}")
        c_main, c_side = st.columns([2, 1])
        with c_main:
            fig_chart = graficar_simple(sel_ticker)
            if fig_chart: st.plotly_chart(fig_chart, use_container_width=True)
        with c_side:
            factores = calcular_factores_quant_single(sel_ticker)
            if factores: st.plotly_chart(dibujar_radar_factores(factores), use_container_width=True)
            
        # PESTAÑAS DETALLE
        sub_t = st.tabs(["🧮 DCF", "🤖 IA"])
        with sub_t[0]:
            dcf = calcular_dcf_rapido(sel_ticker)
            if dcf: st.metric("Valor Justo DCF", f"${dcf:.2f}", f"{((dcf-curr)/curr)*100:+.1f}%")
            else: st.warning("DCF no disponible.")
        with sub_t[1]:
            if st.button("Analizar con Gemini"):
                with st.spinner("Pensando..."):
                    try: st.write(model.generate_content(f"Analisis breve de {sel_ticker} hoy").text)
                    except: st.error("Sin conexión IA")

# --- TAB 3: PORTAFOLIO ---
with main_tabs[2]:
    st.subheader("💼 Gestión de Cartera")
    df_p = auditar_posiciones_sql()
    if not df_p.empty:
        st.dataframe(df_p.style.format({"Valor": "${:.2f}", "P&L": "${:+.2f}"}), use_container_width=True)
    else: st.info("Sin posiciones.")
    
    with st.expander("📝 Nueva Orden"):
        with st.form("quick_order"):
            q_qty = st.number_input("Cantidad", 1, 1000, 10)
            q_side = st.selectbox("Lado", ["COMPRA", "VENTA"])
            if st.form_submit_button("EJECUTAR"):
                # Necesitamos precio actual para registrar
                try: px_now = yf.Ticker(sel_ticker).history(period='1d')['Close'].iloc[-1]
                except: px_now = 0
                registrar_operacion_sql(sel_ticker, q_side, q_qty, px_now)
                st.success("Orden Registrada")
                st.rerun()
