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
from scipy.stats import norm 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import google.generativeai as genai
from fpdf import FPDF
import base64

# --- 1. CONFIGURACIÓN MAESTRA ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sistema Quant V81 (Master Terminal)", layout="wide", page_icon="🏛️")

# Estilos CSS Profesionales (Dark Bloomberg Style)
st.markdown("""<style>
    .main {background-color: #0e1117;}
    .metric-card {background-color: #1c1c2e; border: 1px solid #2d2d3f; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
    .metric-value {font-size: 24px; font-weight: bold; color: #ffffff;}
    .metric-label {font-size: 14px; color: #a0a0a0;}
    .profit {color: #00cc96;}
    .loss {color: #ff4b4b;}
    .ai-box {background-color: #131420; border-left: 4px solid #9c27b0; padding: 20px; border-radius: 5px; margin-top: 10px;}
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
    .sidebar-text {font-size: 12px; color: #888;}
</style>""", unsafe_allow_html=True)

# API Keys
try:
    secrets = toml.load(".streamlit/secrets.toml") if os.path.exists(".streamlit/secrets.toml") else st.secrets
    genai.configure(api_key=secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'COIN', 'KO', 'DIS', 'SPY', 'QQQ', 'GLD', 'USO']
DB_NAME = "quant_database.db"

# --- 2. MOTORES DE DATOS (BACKEND) ---

# Motor SQL
def init_db():
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, ticker TEXT, tipo TEXT, cantidad INTEGER, precio REAL, total REAL)''')
    conn.commit(); conn.close()

def registrar_operacion(t, tipo, q, p):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    total = q * p; fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO trades (fecha, ticker, tipo, cantidad, precio, total) VALUES (?, ?, ?, ?, ?, ?)", (fecha, t, tipo, q, p, total))
    conn.commit(); conn.close()

def obtener_cartera():
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
            pct = (pnl / d['Cost']) * 100 if d['Cost'] > 0 else 0
            res.append({"Ticker": t, "Cantidad": d['Qty'], "Precio Prom": d['Cost']/d['Qty'], "Precio Actual": px, "Valor": val, "P&L $": pnl, "P&L %": pct})
        except: pass
    return pd.DataFrame(res)

init_db()

# Motor Snapshot (Datos rápidos)
@st.cache_data(ttl=900)
def get_snapshot(ticker):
    try:
        stock = yf.Ticker(ticker); hist = stock.history(period="5d")
        if hist.empty: return None
        return {"Precio": hist['Close'].iloc[-1], "Previo": hist['Close'].iloc[-2], "RSI": ta.rsi(hist['Close'], 14).iloc[-1] if len(hist)>14 else 50}
    except: return None

# Motor Ranking (Scanner)
@st.cache_data(ttl=3600)
def scanner_mercado(tickers):
    ranking = []
    try: data = yf.download(" ".join(tickers), period="6mo", group_by='ticker', progress=False, auto_adjust=True)
    except: return pd.DataFrame()
    
    for t in tickers:
        try:
            df = data[t].dropna() if len(tickers)>1 else data.dropna()
            if df.empty: continue
            
            # Cálculo Quant Simplificado (Para velocidad)
            curr = df['Close'].iloc[-1]
            rsi = ta.rsi(df['Close'], 14).iloc[-1]
            sma200 = df['Close'].rolling(200).mean().iloc[-1]
            
            trend = "Alcista" if curr > sma200 else "Bajista"
            score = 50
            if trend == "Alcista": score += 20
            if rsi < 30: score += 20 # Rebote
            elif rsi > 70: score -= 10 # Sobrecompra
            
            ranking.append({"Ticker": t, "Precio": curr, "RSI": rsi, "Tendencia": trend, "Score": score})
        except: pass
    return pd.DataFrame(ranking).sort_values("Score", ascending=False)

# Motor ML (Oráculo)
def oraculo_ml(ticker):
    try:
        df = yf.Ticker(ticker).history(period="2y")
        if len(df)<200: return None
        df['RSI'] = ta.rsi(df['Close'], 14); df['SMA_Diff'] = (df['Close'] - ta.sma(df['Close'], 50))/ta.sma(df['Close'], 50)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        X = df[['RSI', 'SMA_Diff']]; y = df['Target']
        split = int(len(df)*0.8)
        clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X.iloc[:split], y.iloc[:split])
        acc = accuracy_score(y.iloc[split:], clf.predict(X.iloc[split:]))
        pred = clf.predict(X.iloc[[-1]])[0]
        prob = clf.predict_proba(X.iloc[[-1]])[0][pred]
        return {"Pred": "SUBE 🟢" if pred==1 else "BAJA 🔴", "Acc": acc*100, "Prob": prob*100}
    except: return None

# Motor NLP (Sentimiento)
def sentiment_nlp(ticker):
    if "USD" in ticker: return None
    try:
        news = yf.Ticker(ticker).news[:3]
        score = 0
        bull = ['up', 'rise', 'growth', 'buy', 'strong']; bear = ['down', 'drop', 'loss', 'sell', 'weak']
        titles = []
        for n in news:
            t = n['title'].lower()
            titles.append(n['title'])
            if any(w in t for w in bull): score += 1
            if any(w in t for w in bear): score -= 1
        return {"Score": score, "Titulos": titles, "Label": "Positivo" if score>0 else "Negativo" if score<0 else "Neutral"}
    except: return None

# Motor PDF
class PDFReport(FPDF):
    def header(self): self.set_font('Arial', 'B', 15); self.cell(0, 10, 'Informe Quant V81', 0, 1, 'C'); self.ln(5)
def clean(t): return str(t).encode('latin-1', 'replace').decode('latin-1')
def generar_pdf(ticker, data, ia_text):
    pdf = PDFReport(); pdf.add_page(); pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f'Analisis: {ticker}', 0, 1); pdf.set_font('Arial', '', 11)
    pdf.cell(0, 10, f'Precio: ${data["Precio"]:.2f} | RSI: {data["RSI"]:.0f}', 0, 1)
    pdf.ln(5); pdf.multi_cell(0, 6, clean(ia_text))
    return pdf.output(dest='S').encode('latin-1')

# Motor IA
def consultar_ia(contexto):
    try: return model.generate_content(contexto).text
    except: return "IA no disponible."

# --- 3. INTERFAZ GRÁFICA (FRONTEND) ---

# SIDEBAR (Control Global)
with st.sidebar:
    st.title("🏛️ Quant V81")
    st.caption("Terminal de Gestión Institucional")
    st.markdown("---")
    
    sel_ticker = st.selectbox("🔍 Activo", WATCHLIST)
    
    st.markdown("### ⚡ Operar")
    with st.form("trade_form"):
        qty = st.number_input("Cantidad", 1, 1000, 10)
        side = st.selectbox("Lado", ["COMPRA", "VENTA"])
        if st.form_submit_button("EJECUTAR ORDEN"):
            current = get_snapshot(sel_ticker)
            if current:
                registrar_operacion(sel_ticker, side, qty, current['Precio'])
                st.success(f"Orden ejecutada @ ${current['Precio']:.2f}")
                time.sleep(1)
                st.rerun()
            else: st.error("Sin precio.")
    
    st.markdown("---")
    st.info("Usuario: Estudiante UNRC\nPerfil: Contador/Quant\nVersion: Release 8.1")

# PESTAÑAS PRINCIPALES
tabs = st.tabs(["📊 DASHBOARD EJECUTIVO", "🔬 LABORATORIO MICRO", "🤖 INTELIGENCIA (AI)", "💼 CARTERA"])

# --- TAB 1: DASHBOARD (Resumen Global) ---
with tabs[0]:
    st.subheader("🌍 Visión de Mercado")
    
    # KPIs Superiores
    df_pos = obtener_cartera()
    patrimonio = df_pos['Valor'].sum() if not df_pos.empty else 0
    pnl_total = df_pos['P&L $'].sum() if not df_pos.empty else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-label'>Patrimonio Neto</div><div class='metric-value'>${patrimonio:,.2f}</div></div>", unsafe_allow_html=True)
    color_pnl = "profit" if pnl_total >= 0 else "loss"
    c2.markdown(f"<div class='metric-card'><div class='metric-label'>P&L Total</div><div class='metric-value {color_pnl}'>${pnl_total:+,.2f}</div></div>", unsafe_allow_html=True)
    
    # Mini Scanner en Dashboard
    ranking = scanner_mercado(WATCHLIST)
    if not ranking.empty:
        top = ranking.iloc[0]
        c3.markdown(f"<div class='metric-card'><div class='metric-label'>🔥 Top Oportunidad</div><div class='metric-value'>{top['Ticker']}</div><small>Score: {top['Score']}</small></div>", unsafe_allow_html=True)
    
    # SPY Quick View
    spy_data = get_snapshot("SPY")
    if spy_data:
        spy_delta = ((spy_data['Precio']-spy_data['Previo'])/spy_data['Previo'])*100
        color_spy = "profit" if spy_delta >= 0 else "loss"
        c4.markdown(f"<div class='metric-card'><div class='metric-label'>S&P 500</div><div class='metric-value {color_spy}'>{spy_delta:+.2f}%</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # Gráfico de Cartera vs Tabla Top
    kc1, kc2 = st.columns([1, 2])
    with kc1:
        st.markdown("##### 💼 Distribución")
        if not df_pos.empty:
            fig = px.pie(df_pos, values='Valor', names='Ticker', hole=0.5)
            fig.update_layout(margin=dict(t=0,b=0,l=0,r=0), height=250, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Sin activos.")
        
    with kc2:
        st.markdown("##### 🏆 Ranking de Mercado (Tiempo Real)")
        st.dataframe(ranking.head(5), use_container_width=True)

# --- TAB 2: MICRO (Análisis Profundo) ---
with tabs[1]:
    col_anal1, col_anal2 = st.columns([3, 1])
    
    with col_anal1:
        st.subheader(f"🔎 Análisis Técnico: {sel_ticker}")
        snap = get_snapshot(sel_ticker)
        if snap:
            # Gráfico Interactivo
            df_chart = yf.Ticker(sel_ticker).history(period="1y")
            df_chart['SMA50'] = ta.sma(df_chart['Close'], 50)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name="Precio"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA50'], line=dict(color='orange'), name="SMA50"), row=1, col=1)
            fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], name="Volumen"), row=2, col=1)
            fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
            
    with col_anal2:
        st.markdown("##### 📡 Datos Clave")
        if snap:
            st.metric("Precio", f"${snap['Precio']:.2f}")
            st.metric("RSI (14)", f"{snap['RSI']:.0f}")
            
            # Mini Fundamental
            try:
                info = yf.Ticker(sel_ticker).info
                st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.1f}")
                st.metric("Target", f"${info.get('targetMeanPrice', 0):.2f}")
            except: pass

# --- TAB 3: INTELIGENCIA (AI & ML) ---
with tabs[2]:
    st.subheader(f"🧠 Cerebro Digital: {sel_ticker}")
    
    ia_c1, ia_c2, ia_c3 = st.columns(3)
    
    # 1. Sentimiento
    with ia_c1:
        st.markdown("#### 📰 Sentimiento (NLP)")
        sent = sentiment_nlp(sel_ticker)
        if sent:
            color = "green" if sent['Label'] == "Positivo" else "red" if sent['Label'] == "Negativo" else "gray"
            st.markdown(f"<h2 style='color:{color}'>{sent['Label']}</h2>", unsafe_allow_html=True)
            for t in sent['Titulos']: st.caption(f"• {t}")
        else: st.info("Sin noticias.")
        
    # 2. Machine Learning
    with ia_c2:
        st.markdown("#### 🤖 Oráculo ML")
        if st.button("🔮 Predecir (Random Forest)"):
            with st.spinner("Entrenando..."):
                ml = oraculo_ml(sel_ticker)
                if ml:
                    st.markdown(f"<h2>{ml['Pred']}</h2>", unsafe_allow_html=True)
                    st.metric("Confianza Histórica", f"{ml['Acc']:.1f}%")
                    st.metric("Probabilidad", f"{ml['Prob']:.1f}%")
                else: st.error("Error ML")
                
    # 3. Consultor LLM
    with ia_c3:
        st.markdown("#### 📝 Tesis Generativa")
        if st.button("⚡ Generar Tesis IA"):
            prompt = f"Analisis financiero corto para {sel_ticker}. Precio actual {snap['Precio'] if snap else 0}. RSI {snap['RSI'] if snap else 0}. Da recomendacion de compra/venta."
            res = consultar_ia(prompt)
            st.session_state['ia_res'] = res
        
        if 'ia_res' in st.session_state:
            st.markdown(f"<div class='ai-box'>{st.session_state['ia_res']}</div>", unsafe_allow_html=True)
            if st.button("📄 PDF"):
                b64 = base64.b64encode(generar_pdf(sel_ticker, snap, st.session_state['ia_res'])).decode()
                st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="Reporte.pdf">Descargar</a>', unsafe_allow_html=True)

# --- TAB 4: GESTIÓN DE CARTERA ---
with tabs[3]:
    st.subheader("💼 Auditoría de Posiciones")
    if not df_pos.empty:
        st.dataframe(df_pos.style.format({"Valor": "${:.2f}", "P&L $": "${:+.2f}", "P&L %": "{:+.2f}%"}), use_container_width=True)
    else:
        st.info("No hay operaciones registradas. Usa el panel lateral para operar.")
