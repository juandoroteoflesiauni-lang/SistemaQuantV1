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
from datetime import datetime, timedelta
from scipy.stats import norm 
from sklearn.linear_model import LinearRegression
import google.generativeai as genai

# --- CONFIGURACIÓN ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sistema Quant V68 (Thesis)", layout="wide", page_icon="📝")

st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;}
    .thesis-card {background-color: #1e1e2e; border-left: 4px solid #00BFFF; padding: 15px; border-radius: 5px;}
    .bull-txt {color: #00ff00; font-weight: bold;}
    .bear-txt {color: #ff4b4b; font-weight: bold;}
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

# --- MOTOR DE TESIS DE INVERSIÓN (NUEVO V68) ---
def generar_tesis_automatica(ticker, datos_tecnicos, datos_fundamentales, prediccion, estacionalidad):
    """Genera un resumen ejecutivo de compra/venta basado en reglas lógicas"""
    puntos_bull = []
    puntos_bear = []
    
    # 1. Análisis Técnico
    if datos_tecnicos['RSI'] < 30: puntos_bull.append("Sobreventa Técnica (RSI bajo)")
    elif datos_tecnicos['RSI'] > 70: puntos_bear.append("Sobrecompra Técnica (RSI alto)")
    
    if datos_tecnicos['Precio'] > datos_tecnicos['SMA200']: puntos_bull.append("Tendencia Alcista (Sobre SMA200)")
    else: puntos_bear.append("Tendencia Bajista (Bajo SMA200)")
    
    # 2. Fundamental
    if datos_fundamentales:
        if datos_fundamentales.get('Upside', 0) > 10: puntos_bull.append("Subvaluada según Graham/DCF")
        elif datos_fundamentales.get('Upside', 0) < -10: puntos_bear.append("Sobrevaluada según Fundamentales")
    
    # 3. Predicción
    if prediccion and prediccion['Cambio_Pct'] > 5: puntos_bull.append(f"Proyección +{prediccion['Cambio_Pct']:.1f}% a 30 días")
    elif prediccion and prediccion['Cambio_Pct'] < -2: puntos_bear.append("Proyección negativa a corto plazo")
    
    # 4. Estacionalidad
    if estacionalidad:
        mes_actual = datetime.now().month
        if estacionalidad['Best_Month'] == mes_actual: puntos_bull.append("Mes históricamente FUERTE")
        elif estacionalidad['Worst_Month'] == mes_actual: puntos_bear.append("Mes históricamente DÉBIL")

    # Conclusión
    score = len(puntos_bull) - len(puntos_bear)
    veredicto = "COMPRA FUERTE 🟢" if score >= 3 else "COMPRA 🟢" if score > 0 else "VENTA 🔴" if score < 0 else "NEUTRAL 🟡"
    
    return {
        "Veredicto": veredicto,
        "Pros": puntos_bull,
        "Contras": puntos_bear
    }

# --- MOTORES RESTAURADOS Y FIXES (V68) ---
@st.cache_data(ttl=900)
def escanear_mercado_completo(tickers):
    """(RESTAURADA V68) Escanea el mercado para el ranking"""
    ranking = []
    try: data_hist = yf.download(" ".join(tickers), period="1y", group_by='ticker', progress=False, auto_adjust=True)
    except: return pd.DataFrame()
    
    for t in tickers:
        try:
            df = data_hist[t].dropna() if len(tickers)>1 else data_hist.dropna()
            info = yf.Ticker(t).info
            if df.empty: continue
            
            # Cálculo simplificado de factores
            pe = info.get('trailingPE', 50); val = max(0, min(100, (60 - pe) * 2)) if pe > 0 else 0
            curr = df['Close'].iloc[-1]; s200 = df['Close'].rolling(200).mean().iloc[-1]; rsi = ta.rsi(df['Close'], 14).iloc[-1]
            mom = 0
            if curr > s200: mom += 50
            if rsi > 50: mom += (rsi - 50) * 2
            mom = max(0, min(100, mom))
            
            score = (val * 0.4) + (mom * 0.6)
            if "USD" in t: score = mom # Crypto es puro momentum
            
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
        return {"Heatmap": pivot, "Avg_Seasonality": avg, "Best_Month": avg.idxmax(), "Worst_Month": avg.idxmin()}
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
    df = yf.Ticker(ticker).history(period="6mo", auto_adjust=True)
    if df.empty: return None
    df['SMA50'] = ta.sma(df['Close'], 50); df['SMA200'] = ta.sma(df['Close'], 200)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Precio'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='yellow'), name='SMA 50'))
    fig.update_layout(template="plotly_dark", height=350, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
    return fig

# --- INTERFAZ V68 ---
c1, c2 = st.columns([3, 1])
with c1: st.title("📝 Quant Terminal V68")
with c2: sel_ticker = st.selectbox("ACTIVO PRINCIPAL", WATCHLIST)

stock = yf.Ticker(sel_ticker); hist = stock.history(period="2d"); info = stock.info
if not hist.empty:
    curr = hist['Close'].iloc[-1]; delta = ((curr - hist['Close'].iloc[-2])/hist['Close'].iloc[-2])*100
    st.metric("Precio Actual", f"${curr:.2f}", f"{delta:+.2f}%")

st.divider()

col_main, col_side = st.columns([2, 1])

with col_main:
    # 1. Gráfico
    st.subheader("📉 Acción del Precio")
    fig_chart = graficar_simple(sel_ticker)
    if fig_chart: st.plotly_chart(fig_chart, use_container_width=True)
    
    # 2. Pestañas de Detalle
    tabs_detail = st.tabs(["📝 Tesis de Inversión", "🔮 Predicción", "📅 Ciclos", "🧮 Valuación"])
    
    # --- PESTAÑA TESIS (NUEVA V68) ---
    with tabs_detail[0]:
        st.subheader("📝 Tesis Automática")
        
        # Recopilar Datos para la Tesis
        hist_long = stock.history(period="1y")
        rsi_val = ta.rsi(hist_long['Close'], 14).iloc[-1]
        sma200_val = ta.sma(hist_long['Close'], 200).iloc[-1]
        
        dat_tec = {"RSI": rsi_val, "Precio": curr, "SMA200": sma200_val}
        
        dcf_val = calcular_dcf_rapido(sel_ticker)
        dat_fund = {"Upside": ((dcf_val - curr)/curr)*100} if dcf_val else {}
        
        pred = generar_proyeccion_futura(sel_ticker)
        cycle = analizar_estacionalidad(sel_ticker)
        
        # Generar Tesis
        tesis = generar_tesis_automatica(sel_ticker, dat_tec, dat_fund, pred, cycle)
        
        st.markdown(f"""
        <div class='thesis-card'>
            <h2>VEREDICTO: {tesis['Veredicto']}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        c_p, c_c = st.columns(2)
        with c_p:
            st.markdown("#### ✅ Factores Alcistas (Bullish)")
            if tesis['Pros']:
                for p in tesis['Pros']: st.markdown(f"- {p}")
            else: st.write("Ninguno relevante.")
            
        with c_c:
            st.markdown("#### ❌ Factores Bajistas (Bearish)")
            if tesis['Contras']:
                for c in tesis['Contras']: st.markdown(f"- {c}")
            else: st.write("Ninguno relevante.")

    # --- OTRAS PESTAÑAS (Resumidas) ---
    with tabs_detail[1]:
        if pred:
            color_fc = "green" if pred['Cambio_Pct'] > 0 else "red"
            st.metric("Objetivo 30d", f"${pred['Predicción'][-1]:.2f}", f"{pred['Cambio_Pct']:+.2f}%", delta_color="normal")
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=pred['Fechas'], y=pred['Predicción'], mode='lines', name='Tendencia', line=dict(color='white', dash='dash')))
            fig_fc.add_trace(go.Scatter(x=pred['Fechas'], y=pred['Upper'], mode='lines', line=dict(width=0), showlegend=False))
            fig_fc.add_trace(go.Scatter(x=pred['Fechas'], y=pred['Lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 255, 0.1)'))
            fig_fc.update_layout(template="plotly_dark", height=250, margin=dict(l=0,r=0,t=0,b=0)); st.plotly_chart(fig_fc, use_container_width=True)
            
    with tabs_detail[2]:
        if cycle:
            st.write(f"Mejor Mes: **{cycle['Best_Month']}**")
            st.plotly_chart(px.bar(x=cycle['Avg_Seasonality'].index, y=cycle['Avg_Seasonality'].values, title="Estacionalidad"), use_container_width=True)

    with tabs_detail[3]:
        if dcf_val: st.metric("Valor Justo (DCF)", f"${dcf_val:.2f}", f"{((dcf_val-curr)/curr)*100:+.1f}% Upside")
        else: st.info("Modelo no aplicable.")

with col_side:
    st.subheader("⚡ Quick Trade")
    with st.form("quick_order"):
        q_qty = st.number_input("Cantidad", 1, 1000, 10); q_side = st.selectbox("Lado", ["COMPRA", "VENTA"])
        if st.form_submit_button("EJECUTAR"): registrar_operacion_sql(sel_ticker, q_side, q_qty, curr); st.success("Orden Enviada!")
        
    st.markdown("---")
    st.subheader("🏆 Ranking Mercado")
    if st.button("🔄 ESCANEAR"):
        # Ahora la función SÍ existe
        st.dataframe(escanear_mercado_completo(WATCHLIST), use_container_width=True)
        
    st.markdown("---")
    st.subheader("💼 Cartera")
    df_p = auditar_posiciones_sql()
    if not df_p.empty: st.dataframe(df_p[['Ticker', 'P&L']], use_container_width=True)
