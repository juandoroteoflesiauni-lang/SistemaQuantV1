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
st.set_page_config(page_title="Sistema Quant V82 (The Strategist)", layout="wide", page_icon="♟️")

st.markdown("""<style>
    .main {background-color: #0e1117;}
    .metric-card {background-color: #1c1c2e; border: 1px solid #2d2d3f; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
    .metric-value {font-size: 24px; font-weight: bold; color: #ffffff;}
    .metric-label {font-size: 14px; color: #a0a0a0;}
    .profit {color: #00cc96;}
    .loss {color: #ff4b4b;}
    .ai-box {background-color: #131420; border-left: 4px solid #9c27b0; padding: 20px; border-radius: 5px; margin-top: 10px;}
    .option-card {background-color: #1a2634; border: 1px solid #FFD700; padding: 15px; border-radius: 8px;}
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
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

# Motor SQL (Actualizado V82 con Diario Psicológico)
def init_db():
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, ticker TEXT, tipo TEXT, cantidad INTEGER, precio REAL, total REAL, emocion TEXT, nota TEXT)''')
    conn.commit(); conn.close()

def registrar_operacion(t, tipo, q, p, emo, nota):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    total = q * p; fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO trades (fecha, ticker, tipo, cantidad, precio, total, emocion, nota) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (fecha, t, tipo, q, p, total, emo, nota))
    conn.commit(); conn.close()

def obtener_cartera():
    conn = sqlite3.connect(DB_NAME); df = pd.read_sql_query("SELECT * FROM trades", conn); conn.close()
    if df.empty: return pd.DataFrame()
    # Lógica de P&L acumulado
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

# Motor Snapshot
@st.cache_data(ttl=900)
def get_snapshot(ticker):
    try:
        stock = yf.Ticker(ticker); hist = stock.history(period="5d")
        if hist.empty: return None
        return {"Precio": hist['Close'].iloc[-1], "Previo": hist['Close'].iloc[-2], "RSI": ta.rsi(hist['Close'], 14).iloc[-1] if len(hist)>14 else 50}
    except: return None

# Motor Ranking
@st.cache_data(ttl=3600)
def scanner_mercado(tickers):
    ranking = []
    try: data = yf.download(" ".join(tickers), period="6mo", group_by='ticker', progress=False, auto_adjust=True)
    except: return pd.DataFrame()
    for t in tickers:
        try:
            df = data[t].dropna() if len(tickers)>1 else data.dropna()
            if df.empty: continue
            curr = df['Close'].iloc[-1]; rsi = ta.rsi(df['Close'], 14).iloc[-1]; sma200 = df['Close'].rolling(200).mean().iloc[-1]
            trend = "Alcista" if curr > sma200 else "Bajista"
            score = 50 + (20 if trend == "Alcista" else 0) + (20 if rsi < 30 else -10 if rsi > 70 else 0)
            ranking.append({"Ticker": t, "Precio": curr, "RSI": rsi, "Tendencia": trend, "Score": score})
        except: pass
    return pd.DataFrame(ranking).sort_values("Score", ascending=False)

# Motor ML
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

# Motor Opciones (NUEVO V82 - Basado en Hull/Cohen)
def calcular_payoff_opcion(tipo, strike, prima, precio_spot_min, precio_spot_max, posicion='Compra'):
    """Genera datos para gráfico de Payoff al vencimiento"""
    precios = np.linspace(precio_spot_min, precio_spot_max, 100)
    payoffs = []
    
    for S in precios:
        if tipo == 'Call':
            valor_intrinseco = max(S - strike, 0)
        else: # Put
            valor_intrinseco = max(strike - S, 0)
            
        if posicion == 'Compra':
            pnl = valor_intrinseco - prima
        else: # Venta (Lanzamiento)
            pnl = prima - valor_intrinseco
        payoffs.append(pnl)
        
    return precios, payoffs

# Motor PDF
class PDFReport(FPDF):
    def header(self): self.set_font('Arial', 'B', 15); self.cell(0, 10, 'Informe Quant V82', 0, 1, 'C'); self.ln(5)
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
    st.title("♟️ Quant V82")
    st.caption("Estrategia & Psicología")
    st.markdown("---")
    
    sel_ticker = st.selectbox("🔍 Activo", WATCHLIST)
    
    # SECCIÓN OPERATIVA CON PSICOLOGÍA (NUEVO V82)
    st.markdown("### 🧠 Bitácora de Operación")
    with st.form("trade_form"):
        qty = st.number_input("Cantidad", 1, 1000, 10)
        side = st.selectbox("Lado", ["COMPRA", "VENTA"])
        
        # Campos de Psicología (Mark Douglas Style)
        st.markdown("---")
        st.caption("Registro Psicológico (Trading in the Zone)")
        emocion = st.select_slider("Estado Emocional", options=["Miedo", "Ansiedad", "Neutro", "Confianza", "Euforia"], value="Neutro")
        nota = st.text_area("Razón del Trade (Setup)", placeholder="Ej: Rebote en SMA200 + RSI sobrevendido...")
        
        if st.form_submit_button("EJECUTAR Y REGISTRAR"):
            current = get_snapshot(sel_ticker)
            if current:
                registrar_operacion(sel_ticker, side, qty, current['Precio'], emocion, nota)
                st.success(f"Orden ejecutada @ ${current['Precio']:.2f}")
                time.sleep(1)
                st.rerun()
            else: st.error("Sin precio.")
    
    st.markdown("---")
    st.info("Perfil: Strategist\nRef: Hull / Douglas / Graham")

# PESTAÑAS PRINCIPALES
tabs = st.tabs(["📊 DASHBOARD", "♟️ LABORATORIO OPCIONES", "🔬 ANÁLISIS", "🧠 PSICOLOGÍA"])

# --- TAB 1: DASHBOARD ---
with tabs[0]:
    st.subheader("🌍 Visión Ejecutiva")
    df_pos = obtener_cartera()
    patrimonio = df_pos['Valor'].sum() if not df_pos.empty else 0
    pnl_total = df_pos['P&L $'].sum() if not df_pos.empty else 0
    ranking = scanner_mercado(WATCHLIST)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-label'>Patrimonio</div><div class='metric-value'>${patrimonio:,.2f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-label'>P&L Total</div><div class='metric-value {'profit' if pnl_total>=0 else 'loss'}'>${pnl_total:+,.2f}</div></div>", unsafe_allow_html=True)
    if not ranking.empty:
        c3.markdown(f"<div class='metric-card'><div class='metric-label'>Top Pick</div><div class='metric-value'>{ranking.iloc[0]['Ticker']}</div></div>", unsafe_allow_html=True)
    spy = get_snapshot("SPY")
    if spy:
        delta = ((spy['Precio']-spy['Previo'])/spy['Previo'])*100
        c4.markdown(f"<div class='metric-card'><div class='metric-label'>S&P 500</div><div class='metric-value {'profit' if delta>=0 else 'loss'}'>{delta:+.2f}%</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    kc1, kc2 = st.columns([1, 2])
    with kc1:
        if not df_pos.empty:
            fig = px.pie(df_pos, values='Valor', names='Ticker', hole=0.5, title="Allocación")
            st.plotly_chart(fig, use_container_width=True)
    with kc2:
        st.dataframe(ranking.head(5), use_container_width=True)

# --- TAB 2: ESTRATEGIAS OPCIONES (NUEVO V82) ---
with tabs[1]:
    st.subheader("♟️ Laboratorio de Derivados (Hull & Cohen)")
    st.info("Simula estrategias antes de operar. Analiza el Payoff al vencimiento.")
    
    col_op1, col_op2 = st.columns([1, 3])
    
    snap = get_snapshot(sel_ticker)
    precio_ref = snap['Precio'] if snap else 100
    
    with col_op1:
        st.markdown("#### Configuración Estrategia")
        tipo_est = st.selectbox("Estrategia", ["Simple (Call/Put)", "Bull Call Spread", "Bear Put Spread"])
        
        # Parámetros dinámicos
        if tipo_est == "Simple (Call/Put)":
            op_tipo = st.selectbox("Tipo", ["Call", "Put"])
            op_pos = st.selectbox("Posición", ["Compra (Long)", "Venta (Short)"])
            strike = st.number_input("Strike", value=float(int(precio_ref)))
            prima = st.number_input("Prima (Costo)", value=5.0)
            
            # Calcular Payoff
            precios, payoffs = calcular_payoff_opcion(op_tipo, strike, prima, precio_ref*0.7, precio_ref*1.3, op_pos)
            
        elif "Spread" in tipo_est:
            st.caption("Compra Opción A + Venta Opción B")
            k1 = st.number_input("Strike Compra (K1)", value=float(int(precio_ref)))
            p1 = st.number_input("Prima Compra", value=5.0)
            k2 = st.number_input("Strike Venta (K2)", value=float(int(precio_ref*1.1)))
            p2 = st.number_input("Prima Venta", value=2.0)
            
            tipo_op = "Call" if "Call" in tipo_est else "Put"
            px1, py1 = calcular_payoff_opcion(tipo_op, k1, p1, precio_ref*0.7, precio_ref*1.3, 'Compra')
            px2, py2 = calcular_payoff_opcion(tipo_op, k2, p2, precio_ref*0.7, precio_ref*1.3, 'Venta')
            precios = px1
            payoffs = np.array(py1) + np.array(py2)

    with col_op2:
        st.markdown(f"#### Diagrama de Payoff: {sel_ticker} (Ref: ${precio_ref:.2f})")
        
        # Gráfico Payoff
        fig_pay = go.Figure()
        fig_pay.add_trace(go.Scatter(x=precios, y=payoffs, mode='lines', name='P&L', fill='tozeroy', line=dict(color='cyan', width=3)))
        
        # Línea Cero
        fig_pay.add_hline(y=0, line_color="white", line_dash="dash")
        # Línea Precio Actual
        fig_pay.add_vline(x=precio_ref, line_color="yellow", annotation_text="Precio Actual")
        
        fig_pay.update_layout(template="plotly_dark", title="Ganancia/Pérdida al Vencimiento", xaxis_title="Precio del Activo", yaxis_title="P&L ($)")
        st.plotly_chart(fig_pay, use_container_width=True)
        
        max_profit = max(payoffs)
        max_loss = min(payoffs)
        c_res1, c_res2 = st.columns(2)
        c_res1.metric("Máxima Ganancia", f"${max_profit:.2f}")
        c_res2.metric("Máximo Riesgo", f"${max_loss:.2f}")

# --- TAB 3: ANÁLISIS MICRO (V81) ---
with tabs[2]:
    c_an1, c_an2 = st.columns([2, 1])
    with c_an1:
        st.subheader("📉 Técnico & ML")
        if snap:
            df_chart = yf.Ticker(sel_ticker).history(period="1y")
            df_chart['SMA50'] = ta.sma(df_chart['Close'], 50)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close']), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA50'], line=dict(color='orange'), name="SMA50"), row=1, col=1)
            fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume']), row=2, col=1)
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
            
    with c_an2:
        st.markdown("#### 🤖 Oráculo")
        if st.button("🔮 Predecir"):
            ml = oraculo_ml(sel_ticker)
            if ml:
                st.markdown(f"<h2>{ml['Pred']}</h2>", unsafe_allow_html=True)
                st.metric("Confianza Histórica", f"{ml['Acc']:.1f}%")
        
        st.markdown("#### 📝 Tesis IA")
        if st.button("⚡ Generar Informe"):
            prompt = f"Analisis financiero corto {sel_ticker}. Precio {snap['Precio'] if snap else 0}. RSI {snap['RSI'] if snap else 0}. Recomendacion."
            res = consultar_ia(prompt)
            st.session_state['ia_res'] = res
        if 'ia_res' in st.session_state:
            st.caption(st.session_state['ia_res'])
            if st.button("📄 PDF"):
                b64 = base64.b64encode(generar_pdf(sel_ticker, snap, st.session_state['ia_res'])).decode()
                st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="Reporte.pdf">Descargar</a>', unsafe_allow_html=True)

# --- TAB 4: PSICOLOGÍA (NUEVO V82) ---
with tabs[3]:
    st.subheader("🧠 Diario de Trading & Auditoría Emocional")
    
    conn = sqlite3.connect(DB_NAME)
    try:
        df_diario = pd.read_sql_query("SELECT fecha, ticker, tipo, emocion, nota, total FROM trades ORDER BY fecha DESC", conn)
        
        if not df_diario.empty:
            # Análisis Emocional
            c_emo1, c_emo2 = st.columns([1, 2])
            
            with c_emo1:
                st.markdown("#### Estado Mental Predominante")
                fig_emo = px.pie(df_diario, names='emocion', title="Distribución de Emociones", hole=0.4)
                st.plotly_chart(fig_emo, use_container_width=True)
            
            with c_emo2:
                st.markdown("#### Bitácora de Decisiones")
                st.dataframe(df_diario, use_container_width=True)
        else:
            st.info("No hay registros. Realiza una operación en el Sidebar para comenzar tu diario.")
    except: st.error("Error leyendo base de datos.")
    conn.close()
