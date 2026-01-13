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
st.set_page_config(page_title="Sistema Quant V66 (The Chronos)", layout="wide", page_icon="⏳")

st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;}
    .season-card {background-color: #1a1a2e; border: 1px solid #FFD700; border-radius: 8px; padding: 15px; text-align: center;}
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

# --- MOTOR DE ESTACIONALIDAD (NUEVO V66) ---
@st.cache_data(ttl=3600)
def analizar_estacionalidad(ticker):
    """Analiza patrones mensuales y semanales históricos"""
    try:
        # Descargamos 10 años de historia
        df = yf.Ticker(ticker).history(period="10y", auto_adjust=True)
        if df.empty: return None
        
        # Calcular retornos
        df['Retorno'] = df['Close'].pct_change()
        df['Año'] = df.index.year
        df['Mes'] = df.index.month_name().str[:3] # Jan, Feb...
        df['Mes_Num'] = df.index.month
        df['Dia'] = df.index.day_name()
        
        # 1. Matriz Mensual (Heatmap Data)
        pivot_monthly = df.groupby(['Año', 'Mes_Num'])['Retorno'].apply(lambda x: (1 + x).prod() - 1).unstack() * 100
        # Reordenar columnas y renombrar a texto
        meses_orden = {1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May', 6:'Jun', 7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dic'}
        pivot_monthly = pivot_monthly.rename(columns=meses_orden)
        
        # 2. Promedio por Mes (Seasonality Plot)
        avg_monthly = df.groupby('Mes_Num')['Retorno'].mean() * 100 * 21 # Aprox mensual
        avg_monthly.index = [meses_orden[i] for i in avg_monthly.index]
        
        # 3. Mejor Mes y Peor Mes
        best_month = avg_monthly.idxmax()
        worst_month = avg_monthly.idxmin()
        
        return {
            "Heatmap": pivot_monthly,
            "Avg_Seasonality": avg_monthly,
            "Best_Month": best_month,
            "Worst_Month": worst_month,
            "Win_Rate": (avg_monthly > 0).mean() * 100
        }
    except: return None

# --- MOTORES EXISTENTES ---
def obtener_consenso_analistas(ticker):
    if "USD" in ticker: return None
    try:
        info = yf.Ticker(ticker).info
        rec = info.get('recommendationKey', 'none').upper().replace('_', ' ')
        tm = info.get('targetMeanPrice'); cur = info.get('currentPrice')
        if not tm or not cur: return None
        return {"Recomendación": rec, "Score": info.get('recommendationMean'), "Target Mean": tm, "Precio Actual": cur, "Upside %": ((tm-cur)/cur)*100}
    except: return None

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
        i = yf.Ticker(ticker).info; fcf = i.get('freeCashflow', i.get('operatingCashflow', 0)*0.8)
        if fcf <= 0: return None
        pv = 0; g=0.1; w=0.09
        for y in range(1, 6): pv += (fcf * ((1+g)**y)) / ((1+w)**y)
        term = (fcf * ((1+g)**5) * 1.02) / (w - 0.02); pv_term = term / ((1+w)**5)
        return (pv + pv_term) / i.get('sharesOutstanding', 1)
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

# --- INTERFAZ V66 ---
c1, c2 = st.columns([3, 1])
with c1: st.title("⏳ Quant Terminal V66")
with c2: sel_ticker = st.selectbox("ACTIVO PRINCIPAL", WATCHLIST)

stock = yf.Ticker(sel_ticker); hist = stock.history(period="2d"); info = stock.info
if not hist.empty:
    curr = hist['Close'].iloc[-1]; delta = ((curr - hist['Close'].iloc[-2])/hist['Close'].iloc[-2])*100
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Precio", f"${curr:.2f}", f"{delta:+.2f}%")
    k2.metric("RSI", f"{ta.rsi(stock.history(period='30d')['Close'], 14).iloc[-1]:.0f}")
    k3.metric("Vol", f"{info.get('volume',0)/1e6:.1f}M")
    k4.metric("Beta", f"{info.get('beta',1.0):.2f}")
    k5.metric("Target", f"${info.get('targetMeanPrice',0):.2f}")

st.divider()

col_main, col_side = st.columns([2, 1])

with col_main:
    st.subheader("📉 Acción del Precio")
    fig_chart = graficar_simple(sel_ticker)
    if fig_chart: st.plotly_chart(fig_chart, use_container_width=True)
    
    # TABS DETALLE (V66: AÑADIDA ESTACIONALIDAD)
    tabs_detail = st.tabs(["📅 Ciclos & Estacionalidad", "👥 Consenso", "🧮 Valuación", "📰 Noticias"])
    
    # 1. ESTACIONALIDAD (NUEVO V66)
    with tabs_detail[0]:
        st.subheader("⏳ Máquina del Tiempo (Historia 10 Años)")
        with st.spinner("Analizando patrones históricos..."):
            season_data = analizar_estacionalidad(sel_ticker)
            
            if season_data:
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.markdown(f"""
                    <div class='season-card'>
                        <h4>Mejor Mes Histórico</h4>
                        <h2 style='color: #00ff00'>{season_data['Best_Month']}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with sc2:
                    st.markdown(f"""
                    <div class='season-card'>
                        <h4>Peor Mes Histórico</h4>
                        <h2 style='color: #ff4b4b'>{season_data['Worst_Month']}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Gráfico de Tendencia Anual Promedio
                st.markdown("#### 📅 Comportamiento Promedio Anual")
                fig_season = px.bar(
                    x=season_data['Avg_Seasonality'].index, 
                    y=season_data['Avg_Seasonality'].values,
                    color=season_data['Avg_Seasonality'].values,
                    color_continuous_scale="RdYlGn",
                    title="Retorno Promedio por Mes (%)"
                )
                st.plotly_chart(fig_season, use_container_width=True)
                
                # Heatmap Completo
                with st.expander("Ver Mapa de Calor Detallado (Año x Mes)"):
                    fig_heat = px.imshow(season_data['Heatmap'], 
                                         color_continuous_scale="RdYlGn", 
                                         text_auto=".1f", aspect="auto", zmin=-10, zmax=10)
                    st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.warning("Datos históricos insuficientes para análisis estacional.")

    with tabs_detail[1]:
        cons = obtener_consenso_analistas(sel_ticker)
        if cons:
            c1, c2 = st.columns([1, 2])
            c1.markdown(f"<div style='text-align:center; padding:10px; background:#222; border-radius:10px;'><h2>{cons['Recomendación']}</h2><p>Upside: {cons['Upside %']:.1f}%</p></div>", unsafe_allow_html=True)
            fig_t = go.Figure()
            fig_t.add_trace(go.Bar(x=[1], y=[cons['Target High']], name='High', marker_color='green', opacity=0.3))
            fig_t.add_trace(go.Bar(x=[1], y=[cons['Target Mean']], name='Mean', marker_color='yellow', opacity=0.8))
            fig_t.add_trace(go.Scatter(x=[1], y=[cons['Precio Actual']], mode='markers+text', marker=dict(size=15, color='white'), text=["Actual"], name="Actual"))
            fig_t.update_layout(height=200, showlegend=False, barmode='overlay', paper_bgcolor="#0e1117", font={'color':'white'}); st.plotly_chart(fig_t, use_container_width=True)
        else: st.info("Sin consenso.")

    with tabs_detail[2]:
        dcf = calcular_dcf_rapido(sel_ticker)
        if dcf: st.metric("Valor Justo DCF", f"${dcf:.2f}", f"{((dcf-curr)/curr)*100:+.1f}%")
        else: st.warning("DCF no aplica.")
        
    with tabs_detail[3]:
        if st.button("🤖 Analizar Noticias"):
            try: st.write(model.generate_content(f"Analisis {sel_ticker} hoy").text)
            except: st.error("Sin IA")

with col_side:
    st.subheader("🧬 Perfil Quant")
    factores = calcular_factores_quant_single(sel_ticker)
    if factores: st.plotly_chart(dibujar_radar_factores(factores), use_container_width=True)
    st.markdown("---")
    st.subheader("⚡ Quick Trade")
    with st.form("quick_order"):
        q_qty = st.number_input("Cantidad", 1, 1000, 10); q_side = st.selectbox("Lado", ["COMPRA", "VENTA"])
        if st.form_submit_button("EJECUTAR"): registrar_operacion_sql(sel_ticker, q_side, q_qty, curr); st.success("Orden Enviada!")

st.markdown("---")
st.subheader("🏆 Ranking de Mercado")
if st.button("🔄 ESCANEAR"): st.dataframe(escanear_mercado_completo(WATCHLIST), use_container_width=True)
