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
st.set_page_config(page_title="Sistema Quant V67 (The Forecaster)", layout="wide", page_icon="🔮")

st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;}
    .forecast-card {background-color: #1a1a2e; border: 1px solid #7b2cbf; border-radius: 8px; padding: 15px; text-align: center;}
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

# --- MOTOR DE PREDICCIÓN (NUEVO V67) ---
def generar_proyeccion_futura(ticker, dias=30):
    """Proyecta precio con Regresión Lineal + Conos de Volatilidad"""
    try:
        # 1. Datos
        df = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
        if df.empty: return None
        
        df = df.reset_index()
        df['Day'] = df.index # Variable tiempo
        
        # 2. Modelo Regresión (Tendencia)
        X = df[['Day']]
        y = df['Close']
        model_lr = LinearRegression()
        model_lr.fit(X, y)
        
        # 3. Proyección Futura
        last_day = df['Day'].iloc[-1]
        future_days = np.array(range(last_day + 1, last_day + dias + 1)).reshape(-1, 1)
        future_prices = model_lr.predict(future_days)
        
        # 4. Cálculo de Intervalos de Confianza (Volatilidad)
        # Desviación estándar de los residuos (error histórico)
        residuals = y - model_lr.predict(X)
        std_dev = residuals.std()
        
        # El cono se abre con el tiempo (incertidumbre aumenta)
        # Upper = Predicción + (Z * StdDev * Raíz(Tiempo))
        upper_band = future_prices + (1.96 * std_dev * np.sqrt(np.arange(1, dias + 1)))
        lower_band = future_prices - (1.96 * std_dev * np.sqrt(np.arange(1, dias + 1)))
        
        # Fechas futuras
        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, dias + 1)]
        
        return {
            "Fechas": future_dates,
            "Predicción": future_prices,
            "Upper": upper_band,
            "Lower": lower_band,
            "Precio_Hoy": df['Close'].iloc[-1],
            "Precio_Obj": future_prices[-1],
            "Cambio_Pct": ((future_prices[-1] - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100
        }
        
    except Exception as e: return None

# --- MOTORES EXISTENTES (CON FIX EN CONSENSO) ---
def obtener_consenso_analistas(ticker):
    """FIX V67: Asegura que todas las claves existan para evitar KeyError"""
    if "USD" in ticker: return None
    try:
        info = yf.Ticker(ticker).info
        rec = info.get('recommendationKey', 'none').upper().replace('_', ' ')
        tm = info.get('targetMeanPrice')
        cur = info.get('currentPrice')
        
        if not tm or not cur: return None
        
        # FIX: Usar .get con valores por defecto (Mean) si High/Low no existen
        th = info.get('targetHighPrice', tm)
        tl = info.get('targetLowPrice', tm)
        
        return {
            "Recomendación": rec, 
            "Score": info.get('recommendationMean'), 
            "Target Mean": tm,
            "Target High": th, # <--- Fix aplicado
            "Target Low": tl,  # <--- Fix aplicado
            "Precio Actual": cur, 
            "Upside %": ((tm-cur)/cur)*100
        }
    except: return None

@st.cache_data(ttl=3600)
def analizar_estacionalidad(ticker):
    try:
        df = yf.Ticker(ticker).history(period="10y", auto_adjust=True)
        if df.empty: return None
        df['Retorno'] = df['Close'].pct_change()
        df['Mes_Num'] = df.index.month
        
        pivot = df.groupby([df.index.year, 'Mes_Num'])['Retorno'].apply(lambda x: (1+x).prod()-1).unstack()*100
        meses = {1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}
        pivot = pivot.rename(columns=meses)
        
        avg = df.groupby('Mes_Num')['Retorno'].mean()*100*21
        avg.index = [meses[i] for i in avg.index]
        
        return {"Heatmap": pivot, "Avg_Seasonality": avg, "Best_Month": avg.idxmax(), "Worst_Month": avg.idxmin()}
    except: return None

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

# --- INTERFAZ V67 ---
c1, c2 = st.columns([3, 1])
with c1: st.title("🔮 Quant Terminal V67")
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
    
    # TABS DETALLE (V67: FORECAST)
    tabs_detail = st.tabs(["🔮 Predicción", "📅 Ciclos", "👥 Consenso", "📰 Noticias"])
    
    # 1. PREDICCIÓN (NUEVO V67)
    with tabs_detail[0]:
        st.subheader("🔮 Proyección a 30 Días (Monte Carlo Light)")
        forecast = generar_proyeccion_futura(sel_ticker)
        
        if forecast:
            fc1, fc2 = st.columns(2)
            color_fc = "green" if forecast['Cambio_Pct'] > 0 else "red"
            with fc1:
                st.markdown(f"""
                <div class='forecast-card'>
                    <h4>Precio Objetivo (30d)</h4>
                    <h2 style='color:{color_fc}'>${forecast['Predicción'][-1]:.2f}</h2>
                    <p>Cambio Estimado: {forecast['Cambio_Pct']:+.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            with fc2:
                st.caption("Interpretación:")
                st.write("La **línea punteada** es la tendencia matemática.")
                st.write("El **área sombreada** es el rango de probabilidad (95%). Si la volatilidad es alta, el cono es más ancho.")
            
            # Gráfico de Predicción
            fig_fc = go.Figure()
            # Línea central
            fig_fc.add_trace(go.Scatter(x=forecast['Fechas'], y=forecast['Predicción'], mode='lines', name='Tendencia', line=dict(color='white', dash='dash')))
            # Banda Superior
            fig_fc.add_trace(go.Scatter(x=forecast['Fechas'], y=forecast['Upper'], mode='lines', name='Techo (95%)', line=dict(width=0), showlegend=False))
            # Banda Inferior (con relleno)
            fig_fc.add_trace(go.Scatter(x=forecast['Fechas'], y=forecast['Lower'], mode='lines', name='Suelo (5%)', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 255, 0.1)'))
            
            fig_fc.update_layout(title="Túnel de Probabilidad Futura", template="plotly_dark", height=300)
            st.plotly_chart(fig_fc, use_container_width=True)
        else: st.warning("Datos insuficientes para proyectar.")

    with tabs_detail[1]:
        season_data = analizar_estacionalidad(sel_ticker)
        if season_data:
            st.markdown(f"Mejor Mes: **{season_data['Best_Month']}** | Peor Mes: **{season_data['Worst_Month']}**")
            fig_s = px.bar(x=season_data['Avg_Seasonality'].index, y=season_data['Avg_Seasonality'].values, title="Estacionalidad Promedio", color_discrete_sequence=['#FFD700'])
            st.plotly_chart(fig_s, use_container_width=True)

    with tabs_detail[2]:
        cons = obtener_consenso_analistas(sel_ticker)
        if cons:
            st.markdown(f"**Recomendación:** {cons['Recomendación']} | **Target:** ${cons['Target Mean']:.2f}")
            fig_t = go.Figure()
            fig_t.add_trace(go.Bar(x=[1], y=[cons['Target High']], name='High', marker_color='green', opacity=0.3))
            fig_t.add_trace(go.Bar(x=[1], y=[cons['Target Mean']], name='Mean', marker_color='yellow', opacity=0.8))
            fig_t.add_trace(go.Scatter(x=[1], y=[cons['Precio Actual']], mode='markers+text', text=["Actual"], marker_color='white'))
            fig_t.update_layout(height=200, barmode='overlay', showlegend=False, paper_bgcolor="#0e1117"); st.plotly_chart(fig_t, use_container_width=True)
        else: st.info("No disponible para este activo.")

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
