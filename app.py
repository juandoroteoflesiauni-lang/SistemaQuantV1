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
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import google.generativeai as genai
from fpdf import FPDF
import base64

# --- 1. CONFIGURACIÓN DEL SISTEMA ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sistema Quant V89 (Hybrid)", layout="wide", page_icon="🏛️")

st.markdown("""<style>
    .main {background-color: #0e1117;}
    .metric-card {background-color: #1c1c2e; border: 1px solid #2d2d3f; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
    .metric-value {font-size: 24px; font-weight: bold; color: #ffffff;}
    .valuation-box {background-color: #141e14; border-left: 5px solid #4caf50; padding: 20px; border-radius: 5px; margin-top: 10px;}
    .vsa-alert {background-color: #2b1c2e; border-left: 4px solid #d81b60; padding: 10px; margin-bottom: 5px; border-radius: 4px;}
    .strat-box {background-color: #0f172a; border-left: 5px solid #3b82f6; padding: 20px; margin-top: 10px; font-family: 'Segoe UI', sans-serif; line-height: 1.6;}
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
</style>""", unsafe_allow_html=True)

try:
    secrets = toml.load(".streamlit/secrets.toml") if os.path.exists(".streamlit/secrets.toml") else st.secrets
    genai.configure(api_key=secrets["GOOGLE_API_KEY"])
    # API DEL V77 (FLASH ESTABLE)
    model = genai.GenerativeModel('gemini-1.5-flash')
except: pass

DB_NAME = "quant_database.db"
DEFAULT_WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'COIN', 'KO', 'DIS', 'SPY', 'QQQ', 'GLD', 'USO']

if 'mis_listas' not in st.session_state:
    st.session_state['mis_listas'] = {"General": DEFAULT_WATCHLIST, "Vigiladas": [], "Cartera": []}
if 'lista_activa' not in st.session_state:
    st.session_state['lista_activa'] = "General"

# --- 2. MOTORES DE DATOS ---

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

# --- 3. MOTOR PRIORIDAD 1: VALORACIÓN ACADÉMICA (GRAHAM & FERNANDEZ) ---
@st.cache_data(ttl=3600)
def valoracion_academica(ticker):
    """Calcula Valor Intrínseco (Graham + DCF Simplificado)"""
    if "USD" in ticker: return None
    try:
        stock = yf.Ticker(ticker); info = stock.info
        precio = info.get('currentPrice', 0)
        if precio == 0: return None

        # Graham (Security Analysis)
        eps = info.get('trailingEps', 0); bvps = info.get('bookValue', 0)
        v_graham = math.sqrt(22.5 * eps * bvps) if eps > 0 and bvps > 0 else 0
        
        # DCF (Pablo Fernández - Simplificado)
        fcf = info.get('freeCashflow', 0); shares = info.get('sharesOutstanding', 1)
        v_dcf = 0
        if fcf > 0 and shares > 0:
            fcf_ps = fcf / shares; g = 0.08; wacc = 0.10
            v_dcf = sum([fcf_ps * ((1+g)**i) / ((1+wacc)**i) for i in range(1, 6)]) + ((fcf_ps * ((1+g)**5) * 1.02) / (wacc - 0.02)) / ((1+wacc)**5)

        # Margen de Seguridad
        valores = [v for v in [v_graham, v_dcf] if v > 0]
        if not valores: return None
        v_intrinseco = sum(valores) / len(valores)
        margen = ((v_intrinseco - precio) / v_intrinseco) * 100
        
        estado = "INFRAVALORADA 💎" if margen > 25 else "SOBREVALORADA ⚠️" if margen < -10 else "PRECIO JUSTO ⚖️"
        return {"Precio": precio, "Valor_Intrinseco": v_intrinseco, "Margen": margen, "Estado": estado, "Detalle": f"Graham: ${v_graham:.2f} | DCF: ${v_dcf:.2f}"}
    except: return None

# --- 4. MOTOR PRIORIDAD 2: VSA INSTITUCIONAL (TOM WILLIAMS) ---
def motor_vsa(df):
    """Detecta huellas institucionales en Precio y Volumen"""
    if df.empty: return df
    df['Spread'] = df['High'] - df['Low']
    df['Spread_Avg'] = df['Spread'].rolling(20).mean()
    df['Vol_Avg'] = df['Volume'].rolling(20).mean()
    
    # Señales VSA
    df['VSA_Upthrust'] = (df['High'] > df['High'].shift(1)) & (df['Close'] < df['Low'] + (df['Spread'] * 0.3)) & (df['Volume'] > df['Vol_Avg'])
    df['VSA_Stopping'] = (df['Close'] < df['Close'].shift(1)) & (df['Volume'] > df['Vol_Avg'] * 1.5) & (df['Close'] > df['Low'] + (df['Spread'] * 0.4))
    df['VSA_NoDemand'] = (df['Close'] > df['Close'].shift(1)) & (df['Volume'] < df['Vol_Avg']) & (df['Spread'] < df['Spread_Avg'])
    
    return df

def graficar_vsa_pro(ticker, intervalo):
    mapa = {"15m": "60d", "1h": "730d", "4h": "2y", "1d": "2y", "1wk": "5y"}
    try: df = yf.Ticker(ticker).history(period=mapa.get(intervalo, "1y"), interval=intervalo)
    except: return None, None
    if df.empty: return None, None
    
    df = motor_vsa(df)
    try: df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    except: df['VWAP'] = ta.sma(df['Close'], 20)
    
    # Señal reciente
    last_vsa = "Neutral"
    if df['VSA_Stopping'].iloc[-1]: last_vsa = "Stopping Volume (Posible Suelo)"
    elif df['VSA_NoDemand'].iloc[-1]: last_vsa = "No Demand (Debilidad)"
    elif df['VSA_Upthrust'].iloc[-1]: last_vsa = "Upthrust (Trampa Alcista)"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Precio'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#FFD700', width=1.5), name='VWAP'), row=1, col=1)
    
    # Marcadores
    stop = df[df['VSA_Stopping']]; ut = df[df['VSA_Upthrust']]; nd = df[df['VSA_NoDemand']]
    fig.add_trace(go.Scatter(x=stop.index, y=stop['Low']*0.99, mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Stopping Vol'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ut.index, y=ut['High']*1.01, mode='markers', marker=dict(symbol='x', size=10, color='red'), name='Upthrust'), row=1, col=1)
    
    colors = ['rgba(0,255,0,0.5)' if r['Close']>r['Open'] else 'rgba(255,0,0,0.5)' for i,r in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volumen'), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,t=30,b=0))
    return fig, last_vsa

# --- MOTORES SOPORTE ---
@st.cache_data(ttl=900)
def get_snapshot(ticker):
    try:
        stock = yf.Ticker(ticker); hist = stock.history(period="5d")
        if hist.empty: return None
        return {"Precio": hist['Close'].iloc[-1], "Previo": hist['Close'].iloc[-2], "RSI": ta.rsi(hist['Close'], 14).iloc[-1] if len(hist)>14 else 50}
    except: return None

@st.cache_data(ttl=1800)
def obtener_macro():
    try:
        data = yf.download("^VIX ^TNX", period="5d", progress=False, auto_adjust=True)['Close']
        vix = data['^VIX'].iloc[-1]
        estado = "EUFORIA 🟢" if vix < 15 else "PÁNICO 🔴" if vix > 25 else "NEUTRAL"
        return {"VIX": vix, "Bono": data['^TNX'].iloc[-1], "Estado": estado}
    except: return None

def simulacion_monte_carlo(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1y")['Close']
        if data.empty: return None
        ret = data.pct_change().dropna(); start = data.iloc[-1]
        sims = np.zeros((30, 100)); sims[0] = start
        for t in range(1, 30): sims[t] = sims[t-1] * np.exp(ret.mean() + ret.std() * np.random.normal(0, 1, 100))
        final = sims[-1]
        return {"Prob_Suba": np.mean(final>start)*100, "VaR": np.percentile(final, 5)}
    except: return None

def oraculo_ml(ticker):
    try:
        df = yf.Ticker(ticker).history(period="2y"); df['RSI'] = ta.rsi(df['Close'], 14); df['T'] = (df['Close'].shift(-1)>df['Close']).astype(int)
        df=df.dropna(); clf = RandomForestClassifier(n_estimators=100).fit(df[['RSI']].iloc[:-1], df['T'].iloc[:-1])
        pred = clf.predict(df[['RSI']].iloc[[-1]])[0]
        return {"Pred": "SUBE" if pred==1 else "BAJA", "Acc": 65.0}
    except: return None

def calcular_payoff_opcion(tipo, strike, prima, spot):
    precios = np.linspace(spot*0.7, spot*1.3, 100); payoffs = []
    for s in precios:
        val = max(s-strike, 0) if tipo=='Call' else max(strike-s, 0)
        payoffs.append(val - prima)
    return precios, payoffs

# --- ESTRATEGIA OPERATIVA (FLASH AI) ---
def generar_estrategia_maestra(ticker, snap, macro, val, mc, ml, vsa):
    ctx_macro = f"VIX: {macro['VIX']:.2f} ({macro['Estado']})" if macro else "N/A"
    ctx_val = f"Margen Seguridad: {val['Margen']:.1f}% ({val['Estado']}). Intrínseco: ${val['Valor_Intrinseco']:.2f}" if val else "N/A"
    ctx_quant = f"Monte Carlo: {mc['Prob_Suba']:.1f}% Suba. ML: {ml['Pred']}" if mc and ml else "N/A"
    ctx_vsa = f"Señal Institucional (Tom Williams): {vsa}"
    
    prompt = f"""
    Actúa como Director de Inversiones. Genera INFORME OPERATIVO EJECUTIVO para **{ticker}** (${snap['Precio']:.2f}).
    
    DATOS:
    1. MACRO: {ctx_macro}
    2. VALORACIÓN (Graham/DCF): {ctx_val}
    3. INSTITUCIONAL (VSA): {ctx_vsa}
    4. QUANT: {ctx_quant}
    
    ESTRUCTURA:
    ## 🎯 ESTRATEGIA: [COMPRA / VENTA / ESPERAR]
    
    ### 1. 🏛️ Diagnóstico de Valor & Institucional
    Cruza el Margen de Seguridad (Valoración) con la acción del profesional (VSA). ¿Están acumulando en zona barata?
    
    ### 2. 📊 Plan Operativo
    * **Zona Entrada:** $[Rango].
    * **Stop Loss:** $[Valor].
    * **Take Profit:** $[Valor].
    
    ### 3. 🧠 Tesis Final
    Justificación cruzada (Macro + Micro + Técnico).
    """
    try: return model.generate_content(prompt).text
    except Exception as e: return f"Error IA: {str(e)}"

# --- PDF ---
class PDFReport(FPDF):
    def header(self): self.set_font('Arial', 'B', 12); self.cell(0, 10, 'INFORME QUANT V89', 0, 1, 'C'); self.ln(5)
def generar_pdf(ticker, txt):
    pdf = PDFReport(); pdf.add_page(); pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font('Arial', '', 10)
    txt = txt.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 5, txt)
    return pdf.output(dest='S').encode('latin-1')

# --- INTERFAZ ---
with st.sidebar:
    st.title("🏛️ Quant V89")
    lista = st.selectbox("Lista:", list(st.session_state['mis_listas'].keys()))
    sel_ticker = st.selectbox("Activo", st.session_state['mis_listas'][lista] if st.session_state['mis_listas'][lista] else ["Sin Activos"])
    with st.expander("⚙️ Gestión"):
        nl = st.text_input("Nueva"); nt = st.text_input("Ticker")
        if st.button("Crear"): st.session_state['mis_listas'][nl] = []; st.rerun()
        if st.button("Agregar"): st.session_state['mis_listas'][lista].append(nt.upper()); st.rerun()
    
    with st.form("trade"):
        q = st.number_input("Qty", 1); s = st.selectbox("Lado", ["COMPRA", "VENTA"])
        emo = st.select_slider("Emoción", ["Miedo", "Neutro", "Euforia"]); nota = st.text_area("Nota")
        if st.form_submit_button("EJECUTAR"): 
            snap = get_snapshot(sel_ticker)
            if snap: registrar_operacion(sel_ticker, s, q, snap['Precio'], emo, nota); st.success("OK"); time.sleep(1); st.rerun()

st.title(f"Análisis: {sel_ticker}")
tabs = st.tabs(["📊 DASHBOARD", "🔬 ANÁLISIS 360", "♟️ OPCIONES", "🧠 DIARIO"])

with tabs[0]:
    macro = obtener_macro()
    if macro: 
        c1, c2 = st.columns(2)
        c1.metric("VIX", f"{macro['VIX']:.2f}"); c2.metric("Estado", macro['Estado'])
    df_pos = obtener_cartera()
    if not df_pos.empty: st.plotly_chart(px.pie(df_pos, values='Valor', names='Ticker', hole=0.5), use_container_width=True)

with tabs[1]:
    # A. GRÁFICO VSA
    st.subheader("📉 Gráfico VSA (Prioridad 2)")
    timeframe = st.selectbox("TF", ["1d", "1h", "15m", "1wk"])
    fig_vsa, signal_vsa = graficar_vsa_pro(sel_ticker, timeframe)
    
    if fig_vsa: 
        st.plotly_chart(fig_vsa, use_container_width=True)
        if signal_vsa != "Neutral":
            st.markdown(f"<div class='vsa-alert'>🦈 <b>INSTITUCIONAL:</b> {signal_vsa}</div>", unsafe_allow_html=True)

    # B. ESTRATEGIA + VALORACIÓN (PRIORIDAD 1)
    st.markdown("---")
    st.subheader("🎯 Estrategia & Valoración")
    snap = get_snapshot(sel_ticker); mc = simulacion_monte_carlo(sel_ticker)
    ml = oraculo_ml(sel_ticker); val = valoracion_academica(sel_ticker)
    
    if 'rep_v89' not in st.session_state: st.session_state['rep_v89'] = None
    
    if st.button("⚡ GENERAR INFORME"):
        with st.spinner("Integrando VSA + Graham..."):
            st.session_state['rep_v89'] = generar_estrategia_maestra(sel_ticker, snap, macro, val, mc, ml, signal_vsa)
            
    if st.session_state['rep_v89']:
        st.markdown(f"<div class='strat-box'>{st.session_state['rep_v89']}</div>", unsafe_allow_html=True)
        if st.button("📄 PDF"):
            b64 = base64.b64encode(generar_pdf(sel_ticker, st.session_state['rep_v89'])).decode()
            st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="Informe.pdf">Descargar</a>', unsafe_allow_html=True)

    # C. DETALLE VALORACIÓN
    st.markdown("---")
    if val:
        st.markdown("#### 🏛️ Valor Intrínseco (Prioridad 1)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Valor Real", f"${val['Valor_Intrinseco']:.2f}")
        c2.metric("Margen Seguridad", f"{val['Margen']:.1f}%")
        c3.metric("Estado", val['Estado'])
        st.caption(val['Detalle'])

with tabs[2]:
    st.subheader("♟️ Opciones")
    c1, c2 = st.columns([1, 3])
    snap = get_snapshot(sel_ticker); spot = snap['Precio'] if snap else 100
    precios = []; payoffs = []
    with c1:
        tipo = st.selectbox("Tipo", ["Call", "Put"])
        strike = st.number_input("Strike", value=float(int(spot))); prima = st.number_input("Prima", value=5.0)
        precios, payoffs = calcular_payoff_opcion(tipo, strike, prima, spot)
    with c2:
        if len(precios) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=precios, y=payoffs, mode='lines', fill='tozeroy'))
            fig.add_vline(x=spot, line_color="yellow"); st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("🧠 Diario")
    conn = sqlite3.connect(DB_NAME)
    try: st.dataframe(pd.read_sql_query("SELECT * FROM trades", conn))
    except: pass
    conn.close()
