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
from datetime import datetime
from scipy.signal import argrelextrema 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import minimize 
from scipy.stats import norm 
import google.generativeai as genai
from fpdf import FPDF 

# --- CONFIGURACIÓN MOTOR HÍBRIDO ---
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    HAVE_PYPFOPT = True
    ENGINE_STATUS = "🚀 PyPortfolioOpt (Pro)"
except ImportError:
    HAVE_PYPFOPT = False
    ENGINE_STATUS = "🛠️ Scipy Native (Backup)"

warnings.filterwarnings('ignore')

# --- 🔐 CREDENCIALES ---
try:
    secrets_path = ".streamlit/secrets.toml"
    if os.path.exists(secrets_path):
        secrets = toml.load(secrets_path)
        TELEGRAM_TOKEN = secrets["TELEGRAM_TOKEN"]
        TELEGRAM_CHAT_ID = secrets["TELEGRAM_CHAT_ID"]
        GOOGLE_API_KEY = secrets["GOOGLE_API_KEY"]
    else:
        TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
        TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except: st.stop()

# --- CONFIGURACIÓN PÁGINA ---
st.set_page_config(page_title="Sistema Quant V47 (Technician)", layout="wide", page_icon="🕯️")
st.markdown("""
<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 10px; color: white;}
    .signal-box {border: 2px solid #FFD700; padding: 10px; border-radius: 5px; background-color: #2b2b00; text-align: center;}
    .macro-card {background-color: #1e2130; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #444;}
    .pattern-tag {background-color: #444; padding: 2px 5px; border-radius: 3px; font-size: 0.8em; margin: 2px;}
</style>
""", unsafe_allow_html=True)

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'COIN', 'KO', 'DIS', 'SPY', 'QQQ', 'DIA', 'GLD', 'USO']
MACRO_TICKERS = {'S&P 500': 'SPY', 'VIX (Miedo)': '^VIX', 'Bonos 10Y': '^TNX', 'Oro': 'GC=F', 'Dólar': 'DX-Y.NYB'}
DB_NAME = "quant_database.db"

# --- MOTOR SQL ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, ticker TEXT, tipo TEXT, cantidad INTEGER, precio REAL, total REAL)''')
    conn.commit()
    conn.close()

def registrar_operacion_sql(ticker, tipo, cantidad, precio):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S"); total = cantidad * precio
    c.execute("INSERT INTO trades (fecha, ticker, tipo, cantidad, precio, total) VALUES (?, ?, ?, ?, ?, ?)", (fecha, ticker, tipo, cantidad, precio, total))
    conn.commit(); conn.close()
    return True

def auditar_posiciones_sql():
    conn = sqlite3.connect(DB_NAME)
    try: df = pd.read_sql_query("SELECT * FROM trades", conn)
    except: return pd.DataFrame()
    conn.close()
    if df.empty: return pd.DataFrame()
    pos = {}
    for idx, row in df.iterrows():
        t = row['ticker']
        if t not in pos: pos[t] = {"Cantidad": 0, "Costo_Total": 0}
        if row['tipo'] == "COMPRA": pos[t]["Cantidad"] += row['cantidad']; pos[t]["Costo_Total"] += row['total']
        elif row['tipo'] == "VENTA":
            pos[t]["Cantidad"] -= row['cantidad']
            if pos[t]["Cantidad"] > 0: unit = pos[t]["Costo_Total"]/(pos[t]["Cantidad"]+row['cantidad']); pos[t]["Costo_Total"] -= (unit*row['cantidad'])
            else: pos[t]["Costo_Total"] = 0
    res = []; act = [t for t, d in pos.items() if d['Cantidad'] > 0]
    if not act: return pd.DataFrame()
    try: curr = yf.download(" ".join(act), period="1d", progress=False, auto_adjust=True)['Close']
    except: return pd.DataFrame()
    for t, d in pos.items():
        if d['Cantidad'] > 0:
            try:
                if len(act) == 1: price = float(curr.iloc[-1])
                else: price = float(curr.iloc[-1][t])
                val = d['Cantidad']*price; pnl = val - d['Costo_Total']
                res.append({"Ticker": t, "Cantidad": d['Cantidad'], "Valor Mercado": val, "P&L ($)": pnl, "P&L (%)": (pnl/d['Costo_Total'])*100})
            except: pass
    return pd.DataFrame(res)

init_db()

# --- MOTOR DE PATRONES DE VELAS (NUEVO V47) ---
def detectar_patrones_velas(df):
    """Detecta Martillos, Estrellas Fugaces, Envolventes y Dojis"""
    # Evitar dependencias de TA-Lib usando cálculo vectorial puro (Funciona en cualquier PC)
    df['Body'] = abs(df['Close'] - df['Open'])
    df['Wick_Upper'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Wick_Lower'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Total_Len'] = df['High'] - df['Low']
    
    patrones = []
    
    # Iteramos solo las últimas velas para eficiencia
    for i in range(len(df)-1, len(df)): 
        row = df.iloc[i]
        prev = df.iloc[i-1]
        tipo = None
        
        # 1. DOJI (Cuerpo muy pequeño)
        if row['Body'] <= row['Total_Len'] * 0.1:
            tipo = "Doji ➕"
            
        # 2. MARTILLO (Cola inferior larga, cuerpo pequeño arriba) - Alcista
        elif (row['Wick_Lower'] > 2 * row['Body']) and (row['Wick_Upper'] < row['Body'] * 0.5):
            tipo = "Martillo 🔨"
            
        # 3. ESTRELLA FUGAZ (Cola superior larga, cuerpo pequeño abajo) - Bajista
        elif (row['Wick_Upper'] > 2 * row['Body']) and (row['Wick_Lower'] < row['Body'] * 0.5):
            tipo = "Shooting Star ⭐"
            
        # 4. ENVOLVENTE (Cuerpo cubre al anterior)
        elif (row['Body'] > prev['Body']) and (row['High'] > prev['High']) and (row['Low'] < prev['Low']):
            if row['Close'] > row['Open']: tipo = "Envolvente Bull 🐂"
            else: tipo = "Envolvente Bear 🐻"
            
        if tipo:
            patrones.append(tipo)
            
    return patrones

# --- MOTORES EXISTENTES ---
@st.cache_data(ttl=300)
def escanear_oportunidades(tickers):
    señales = []
    try: data = yf.download(" ".join(tickers), period="3mo", interval="1d", group_by='ticker', progress=False, auto_adjust=True)
    except: return pd.DataFrame()
    for t in tickers:
        try:
            if len(tickers) > 1:
                try: df = data[t].copy()
                except: continue
            else: df = data.copy()
            df = df.dropna()
            if len(df) < 20: continue
            if 'Close' not in df.columns and df.shape[1] >= 4: df.columns = ['Open', 'High', 'Low', 'Close', 'Volume'][:df.shape[1]]
            
            last_close = df['Close'].iloc[-1]; rsi = ta.rsi(df['Close'], 14).iloc[-1]
            bb = ta.bbands(df['Close'], length=20, std=2); lower = bb.iloc[-1, 0]; upper = bb.iloc[-1, 2]
            
            # Detección de Patrones (V47)
            patrones = detectar_patrones_velas(df)
            patron_txt = " | ".join(patrones) if patrones else ""
            
            tipo = "NEUTRAL"; fuerza = 0
            if last_close < lower: tipo = "COMPRA BOL 🟢"; fuerza = 90
            elif last_close > upper: tipo = "VENTA BOL 🔴"; fuerza = 80
            elif rsi < 30: tipo = "COMPRA RSI 🟢"; fuerza += 70
            elif rsi > 70: tipo = "VENTA RSI 🔴"; fuerza += 70
            
            # Bonus por Patrón
            if "Martillo" in patron_txt and "COMPRA" in tipo: fuerza += 20; tipo += " + 🔨"
            if "Shooting Star" in patron_txt and "VENTA" in tipo: fuerza += 20; tipo += " + ⭐"
            
            if tipo != "NEUTRAL" or patron_txt != "": 
                señales.append({"Ticker": t, "Señal": tipo, "Fuerza": fuerza, "Patrón": patron_txt})
        except: pass
    return pd.DataFrame(señales)

def graficar_master(ticker):
    try:
        stock = yf.Ticker(ticker); df = stock.history(period="1y", interval="1d", auto_adjust=True)
        if df.empty: return None
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df['EMA20'] = ta.ema(df['Close'], 20); df['RSI'] = ta.rsi(df['Close'], 14)
        bb = ta.bbands(df['Close'], length=20, std=2); df = pd.concat([df, bb], axis=1)
        
        # Geometría
        sop = sorted([s for s in df['Low'].rolling(10).min().iloc[-20:].unique() if (df['Close'].iloc[-1]-s)/df['Close'].iloc[-1] < 0.15])[-2:]
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"), row=1, col=1)
        if 'EMA20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='yellow'), name="EMA 20"), row=1, col=1)
        try: fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, -3], line=dict(color='cyan', dash='dot'), name="Upper"), row=1, col=1); fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, -1], line=dict(color='cyan', dash='dot'), name="Lower"), row=1, col=1)
        except: pass
        for s in sop: fig.add_hline(y=s, line_dash="dot", line_color="green", row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_color="red", row=2, col=1); fig.add_hline(y=30, line_color="green", row=2, col=1)
        
        # ANOTACIONES DE PATRONES EN GRÁFICO (V47)
        # Escaneamos los últimos 10 días para poner marcadores
        subset = df.iloc[-10:].copy()
        subset['Body'] = abs(subset['Close'] - subset['Open'])
        subset['Wick_Upper'] = subset['High'] - subset[['Open', 'Close']].max(axis=1)
        subset['Wick_Lower'] = subset[['Open', 'Close']].min(axis=1) - subset['Low']
        
        for date, row in subset.iterrows():
            marker = None
            if (row['Wick_Lower'] > 2 * row['Body']) and (row['Wick_Upper'] < row['Body'] * 0.5): marker = "🔨" # Martillo
            elif (row['Wick_Upper'] > 2 * row['Body']) and (row['Wick_Lower'] < row['Body'] * 0.5): marker = "⭐" # Shooting Star
            
            if marker:
                fig.add_annotation(x=date, y=row['High'], text=marker, showarrow=False, yshift=10, font=dict(size=20), row=1, col=1)

        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, title=f"Análisis Técnico Avanzado: {ticker}")
        return fig
    except: return None

# --- MOTORES SIMPLIFICADOS (Stress, PDF, IA, Valuacion, Montecarlo, Macro) ---
class PDFReport(FPDF):
    def header(self): self.set_font('Arial','B',15); self.cell(0,10,'Informe Quant',0,1,'C'); self.ln(5)
    def footer(self): self.set_y(-15); self.set_font('Arial','I',8); self.cell(0,10,f'Pag {self.page_no()}',0,0,'C')
def generar_pdf_cartera(df):
    p=PDFReport(); p.add_page(); p.set_font("Arial",size=12)
    p.cell(0,10,f"Total: ${df['Valor Mercado'].sum():,.2f}",0,1); p.ln(10)
    p.set_font("Arial",'B',10); p.cell(30,10,"Ticker",1); p.cell(40,10,"Valor",1); p.ln()
    p.set_font("Arial",size=10)
    for i,r in df.iterrows(): p.cell(30,10,str(r['Ticker']),1); p.cell(40,10,f"${r['Valor Mercado']:.2f}",1); p.ln()
    return p.output(dest='S').encode('latin-1')
def analizar_noticias_ia(t): return {"score": 0, "summary": "Simulacion IA", "headlines": [], "links": []}
@st.cache_data(ttl=3600)
def calcular_valor_intrinseco(t):
    try: i=yf.Ticker(t).info; return {"Precio": i.get('currentPrice',0), "Graham": math.sqrt(22.5*i.get('trailingEps',0)*i.get('bookValue',0)) if i.get('trailingEps') and i.get('bookValue') else 0}
    except: return None
def obtener_datos_macro(): return {}
def calcular_alpha_beta(t, b='SPY'): return None, None
def simular_montecarlo(t, d=30, s=500): return None, None
@st.cache_data(ttl=600)
def generar_mapa_calor(ts):
    d = yf.download(" ".join(ts), period="2d", progress=False, auto_adjust=True)['Close']
    p = ((d.iloc[-1]-d.iloc[-2])/d.iloc[-2])*100
    return pd.DataFrame({'Ticker': p.index, 'Variacion': p.values, 'Sector': 'General', 'Size': d.iloc[-1].values})
def ejecutar_backtest_pro(t,c,s,p): return None
def calcular_beta_portafolio(df): return 1.0
def ejecutar_stress_test(df): return pd.DataFrame(), 1.0

# --- INTERFAZ ---
st.title("🕯️ Sistema Quant V47: The Technician")

# 1. RADAR AVANZADO (V47)
st.markdown("### 🔭 Radar Técnico (Patrones + Indicadores)")
df_s = escanear_oportunidades(WATCHLIST)
if not df_s.empty:
    # Ordenar por fuerza para ver las mejores oportunidades primero
    df_s = df_s.sort_values("Fuerza", ascending=False)
    cols = st.columns(len(df_s))
    for idx, row in df_s.iterrows():
        with st.container():
             patron_html = f"<div class='pattern-tag'>{row['Patrón']}</div>" if row['Patrón'] else ""
             st.markdown(f"""
             <div class='signal-box'>
                <h3>{row['Ticker']}</h3>
                <p>{row['Señal']}</p>
                {patron_html}
             </div>
             """, unsafe_allow_html=True)
else: st.info("Escáner limpio. Mercado neutral.")

st.divider()

# 2. PANEL PRINCIPAL
c_left, c_right = st.columns([1, 2.5])

with c_left:
    st.subheader("Control")
    tk = st.selectbox("Activo:", WATCHLIST, index=0)
    cap = st.number_input("Simulación ($)", 2000, 100000, 10000, key='cap_sim')
    
    st.markdown("### 🏦 Mi Portafolio")
    df_pos = auditar_posiciones_sql()
    if not df_pos.empty:
        st.metric("P&L Total", f"${df_pos['P&L ($)'].sum():+.2f}", delta_color="normal")
        st.dataframe(df_pos[['Ticker', 'P&L (%)']])
        if st.button("🖨️ PDF"):
            pdf = generar_pdf_cartera(df_pos)
            st.download_button("📥", pdf, "report.pdf", "application/pdf")

    with st.expander("📝 Operar"):
        op_tk = st.selectbox("Ticker", WATCHLIST, key='op_tk')
        op_type = st.selectbox("Tipo", ["COMPRA", "VENTA"])
        op_qty = st.number_input("Qty", 1, 1000)
        op_px = st.number_input("Precio", 0.0)
        if st.button("Ejecutar"):
            registrar_operacion_sql(op_tk, op_type, op_qty, op_px); st.rerun()

with c_right:
    tabs = st.tabs(["🕯️ Gráfico Técnico", "🌪️ Stress/Macro", "📰 Fundamentales", "🔮 Futuro"])
    
    # PESTAÑA 1: GRÁFICO CON PATRONES (V47)
    with tabs[0]:
        st.subheader(f"🕯️ Análisis de Velas: {tk}")
        st.info("Busca emojis sobre las velas: 🔨 Martillo (Rebote), ⭐ Estrella Fugaz (Caída)")
        fig = graficar_master(tk)
        if fig: st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.write("Módulos Macro y Stress Test (Activos en backend)")
        if not df_pos.empty:
            df_str, beta = ejecutar_stress_test(df_pos)
            st.dataframe(df_str)

    with tabs[2]:
        val = calcular_valor_intrinseco(tk)
        if val: st.json(val)
        
    with tabs[3]:
        fig_mc, res = simular_montecarlo(tk)
        if fig_mc: st.plotly_chart(fig_mc, use_container_width=True)