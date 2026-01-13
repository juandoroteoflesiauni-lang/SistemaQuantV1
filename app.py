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
from sklearn.linear_model import LinearRegression
import google.generativeai as genai

# --- CONFIGURACIÓN ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sistema Quant V70 (The CFO)", layout="wide", page_icon="📚")

st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;}
    .fin-table {font-size: 12px;}
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

# --- MOTOR DE DATOS FINANCIEROS DETALLADOS (NUEVO V70) ---
@st.cache_data(ttl=3600)
def obtener_datos_financieros_profundos(ticker):
    """Extrae Balances y Estados de Resultados Anuales"""
    if "USD" in ticker: return None # No aplica a Crypto
    try:
        stock = yf.Ticker(ticker)
        
        # 1. Estados Financieros Anuales
        income = stock.income_stmt
        balance = stock.balance_sheet
        cashflow = stock.cashflow
        
        if income.empty or balance.empty: return None
        
        # Transponer para que las fechas sean filas (mejor para graficar)
        income = income.T.sort_index()
        balance = balance.T.sort_index()
        
        # Extraer métricas clave (Manejo de errores si faltan columnas)
        fechas = income.index.strftime('%Y')
        
        # Safe get para columnas que a veces tienen nombres distintos
        ingresos = income['Total Revenue'] if 'Total Revenue' in income else income.iloc[:, 0]
        net_income = income['Net Income'] if 'Net Income' in income else pd.Series(0, index=income.index)
        
        # Datos para DuPont
        equity = balance['Stockholders Equity'] if 'Stockholders Equity' in balance else pd.Series(1, index=balance.index)
        assets = balance['Total Assets'] if 'Total Assets' in balance else pd.Series(1, index=balance.index)
        
        # Cálculo DuPont (Último Año)
        last_net_income = net_income.iloc[-1]
        last_revenue = ingresos.iloc[-1]
        last_assets = assets.iloc[-1]
        last_equity = equity.iloc[-1]
        
        margen_neto = (last_net_income / last_revenue)
        rotacion_activos = (last_revenue / last_assets)
        apalancamiento = (last_assets / last_equity)
        roe = margen_neto * rotacion_activos * apalancamiento
        
        return {
            "Fechas": fechas,
            "Ingresos": ingresos,
            "Net_Income": net_income,
            "Assets": assets,
            "Equity": equity,
            "DuPont": {
                "Margen Neto": margen_neto * 100,
                "Rotación Activos": rotacion_activos,
                "Apalancamiento": apalancamiento,
                "ROE": roe * 100
            },
            "Holders": stock.major_holders,
            "Raw_Income": income.iloc[-3:].T, # Últimos 3 años raw
        }
    except Exception as e: return None

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

# --- MOTORES RESTAURADOS ---
@st.cache_data(ttl=1800)
def obtener_datos_snapshot(ticker):
    try:
        stock = yf.Ticker(ticker); hist = stock.history(period="5d")
        if hist.empty: return None
        try: info = stock.info
        except: info = {}
        return {"Precio": hist['Close'].iloc[-1], "Previo": hist['Close'].iloc[-2], "RSI": ta.rsi(hist['Close'], 14).iloc[-1] if len(hist)>14 else 50, "Volumen": info.get('volume', 0), "Beta": info.get('beta', 1.0), "Target": info.get('targetMeanPrice', 0)}
    except: return None

@st.cache_data(ttl=1800)
def escanear_mercado_completo(tickers):
    ranking = []
    try: data_hist = yf.download(" ".join(tickers), period="1y", group_by='ticker', progress=False, auto_adjust=True)
    except: return pd.DataFrame()
    for t in tickers:
        try:
            time.sleep(0.05) 
            df = data_hist[t].dropna() if len(tickers)>1 else data_hist.dropna()
            if df.empty: continue
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

def calcular_factores_quant_single(ticker):
    try:
        stock = yf.Ticker(ticker); df = stock.history(period="1y", interval="1d", auto_adjust=True); info = stock.info
        if df.empty: return None
        pe = info.get('trailingPE', 50); score_value = max(0, min(100, (60 - pe) * 2)) if pe > 0 else 0
        score_growth = 50 
        curr = df['Close'].iloc[-1]; s200 = df['Close'].rolling(200).mean().iloc[-1]; rsi = ta.rsi(df['Close'], 14).iloc[-1]
        m = 0
        if curr > s200: m += 50
        if rsi > 50: m += (rsi - 50) * 2
        score_mom = max(0, min(100, m))
        score_qual = 50 
        beta = info.get('beta', 1.5) or 1.0; score_vol = max(0, min(100, (2 - beta) * 100))
        return {"Value": score_value, "Growth": score_growth, "Momentum": score_mom, "Quality": score_qual, "Low Vol": score_vol}
    except: return None

def dibujar_radar_factores(scores):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=list(scores.values()), theta=list(scores.keys()), fill='toself', line_color='#00ff00', fillcolor='rgba(0, 255, 0, 0.2)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], color='grey')), showlegend=False, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color='white'), height=300, margin=dict(l=40, r=40, t=20, b=20))
    return fig

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

# --- INTERFAZ V70 ---
c1, c2 = st.columns([3, 1])
with c1: st.title("📚 Quant Terminal V70: The CFO")
with c2: sel_ticker = st.selectbox("ACTIVO PRINCIPAL", WATCHLIST)

snap = obtener_datos_snapshot(sel_ticker)
if snap:
    delta = ((snap['Precio'] - snap['Previo'])/snap['Previo'])*100
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Precio", f"${snap['Precio']:.2f}", f"{delta:+.2f}%")
    k2.metric("RSI", f"{snap['RSI']:.0f}")
    k3.metric("Vol", f"{snap['Volumen']/1e6:.1f}M")
    k4.metric("Beta", f"{snap['Beta']:.2f}")
    k5.metric("Target", f"${snap['Target']:.2f}")

st.divider()

col_main, col_side = st.columns([2, 1])

with col_main:
    st.subheader("📉 Acción del Precio")
    fig_chart = graficar_simple(sel_ticker)
    if fig_chart: st.plotly_chart(fig_chart, use_container_width=True)
    
    # TABS DETALLE (V70: ESTADOS FINANCIEROS)
    tabs_detail = st.tabs(["📚 Estados Financieros", "🔮 Predicción", "📝 Tesis"])
    
    # --- PESTAÑA CFO (NUEVO V70) ---
    with tabs_detail[0]:
        st.subheader("📊 Análisis Fundamental Profundo")
        
        with st.spinner("Auditando libros contables..."):
            fin_data = obtener_datos_financieros_profundos(sel_ticker)
            
            if fin_data:
                # 1. Gráfico de Barras: Revenue vs Net Income
                fig_fin = go.Figure()
                fig_fin.add_trace(go.Bar(x=fin_data['Fechas'], y=fin_data['Ingresos'], name='Ventas (Revenue)', marker_color='#2196F3'))
                fig_fin.add_trace(go.Bar(x=fin_data['Fechas'], y=fin_data['Net_Income'], name='Ganancia (Net Income)', marker_color='#4CAF50'))
                fig_fin.update_layout(title="Evolución Ventas vs Ganancias (Anual)", barmode='group', template="plotly_dark", height=300)
                st.plotly_chart(fig_fin, use_container_width=True)
                
                st.divider()
                
                # 2. Análisis DuPont
                st.subheader("🧬 Desglose DuPont (ROE)")
                dp = fin_data['DuPont']
                
                c_dup1, c_dup2, c_dup3, c_dup4 = st.columns(4)
                c_dup1.metric("Margen Neto", f"{dp['Margen Neto']:.1f}%", help="Eficiencia Operativa (Ganancia/Ventas)")
                c_dup2.metric("Rotación Activos", f"{dp['Rotación Activos']:.2f}x", help="Eficiencia de Activos (Ventas/Activos)")
                c_dup3.metric("Apalancamiento", f"{dp['Apalancamiento']:.2f}x", help="Riesgo Financiero (Activos/Patrimonio)")
                c_dup4.metric("ROE Final", f"{dp['ROE']:.1f}%", help="Retorno sobre Patrimonio")
                
                st.info(f"Fórmula: {dp['Margen Neto']:.1f}% x {dp['Rotación Activos']:.2f} x {dp['Apalancamiento']:.2f} = **{dp['ROE']:.1f}% ROE**")
                
                # 3. Tabla Raw
                with st.expander("Ver Estado de Resultados Completo (Raw Data)"):
                    st.dataframe(fin_data['Raw_Income'].style.format("${:,.0f}"), use_container_width=True)
                    
            else:
                st.warning("Datos financieros detallados no disponibles (Posible Cripto o ETF).")

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
        if st.button("🤖 Generar Tesis IA"):
            try: st.write(model.generate_content(f"Tesis de inversion para {sel_ticker} considerando fundamentales y tecnicos").text)
            except: st.error("Sin IA")

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
        
    st.markdown("---")
    st.subheader("🏆 Ranking")
    if st.button("🔄 ESCANEAR"):
        st.dataframe(escanear_mercado_completo(WATCHLIST), use_container_width=True)
        
    st.markdown("---")
    st.subheader("💼 Cartera")
    df_p = auditar_posiciones_sql()
    if not df_p.empty: st.dataframe(df_p[['Ticker', 'P&L']], use_container_width=True)
