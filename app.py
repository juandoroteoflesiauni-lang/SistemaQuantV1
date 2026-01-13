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
st.set_page_config(page_title="Sistema Quant V89 (The Valuator)", layout="wide", page_icon="🏛️")

st.markdown("""<style>
    .main {background-color: #0e1117;}
    .metric-card {background-color: #1c1c2e; border: 1px solid #2d2d3f; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
    .metric-value {font-size: 24px; font-weight: bold; color: #ffffff;}
    .valuation-box {background-color: #141e14; border-left: 5px solid #4caf50; padding: 20px; border-radius: 5px; margin-top: 10px;}
    .strat-box {background-color: #0f172a; border-left: 5px solid #3b82f6; padding: 20px; margin-top: 10px; font-family: monospace; white-space: pre-wrap;}
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
</style>""", unsafe_allow_html=True)

try:
    secrets = toml.load(".streamlit/secrets.toml") if os.path.exists(".streamlit/secrets.toml") else st.secrets
    genai.configure(api_key=secrets["GOOGLE_API_KEY"])
    # CORRECCIÓN V89: Volvemos a FLASH (Estable y 1M Contexto)
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

# --- 3. MOTOR DE VALORACIÓN (PRIORIDAD 1 - GRAHAM & FERNANDEZ) ---
@st.cache_data(ttl=3600)
def valoracion_academica(ticker):
    """
    Calcula Valor Intrínseco usando fórmulas de 'Security Analysis' y 'Valoración de Empresas'.
    """
    if "USD" in ticker: return None
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        precio = info.get('currentPrice', 0)
        if precio == 0: return None

        # A. FÓRMULA DE GRAHAM (V = Sqrt(22.5 * EPS * BookValue))
        eps = info.get('trailingEps', 0)
        bvps = info.get('bookValue', 0)
        v_graham = math.sqrt(22.5 * eps * bvps) if eps > 0 and bvps > 0 else 0
        
        # B. DCF SIMPLIFICADO (Pablo Fernández)
        fcf = info.get('freeCashflow', 0)
        shares = info.get('sharesOutstanding', 1)
        v_dcf = 0
        if fcf > 0 and shares > 0:
            fcf_ps = fcf / shares
            g = 0.08 # Crecimiento conservador 8%
            wacc = 0.10 # Costo capital 10%
            # Proyección 5 años + Perpetuidad
            flujos = sum([fcf_ps * ((1+g)**i) / ((1+wacc)**i) for i in range(1, 6)])
            terminal = (fcf_ps * ((1+g)**5) * 1.02) / (wacc - 0.02)
            v_dcf = flujos + (terminal / ((1+wacc)**5))

        # C. SÍNTESIS
        valores = [v for v in [v_graham, v_dcf] if v > 0]
        if not valores: return None
        v_intrinseco = sum(valores) / len(valores)
        margen = ((v_intrinseco - precio) / v_intrinseco) * 100
        
        estado = "INFRAVALORADA (Ganga) 💎" if margen > 25 else "SOBREVALORADA ⚠️" if margen < -10 else "PRECIO JUSTO ⚖️"
        
        return {
            "Precio": precio,
            "Valor_Intrinseco": v_intrinseco,
            "Margen": margen,
            "Estado": estado,
            "Detalle": f"Graham: ${v_graham:.2f} | DCF: ${v_dcf:.2f}"
        }
    except: return None

# --- OTROS MOTORES (MACRO, QUANT) ---
@st.cache_data(ttl=1800)
def obtener_macro():
    try:
        data = yf.download("^VIX ^TNX SPY QQQ", period="5d", progress=False, auto_adjust=True)['Close']
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
        for t in range(1, 30):
            sims[t] = sims[t-1] * np.exp(ret.mean() + ret.std() * np.random.normal(0, 1, 100))
        final = sims[-1]
        return {"Prob_Suba": np.mean(final>start)*100, "VaR": np.percentile(final, 5)}
    except: return None

def oraculo_ml(ticker):
    try:
        df = yf.Ticker(ticker).history(period="2y").dropna()
        if len(df)<200: return None
        df['RSI'] = ta.rsi(df['Close'], 14); df['Target'] = (df['Close'].shift(-1)>df['Close']).astype(int)
        df=df.dropna()
        clf = RandomForestClassifier(n_estimators=100).fit(df[['RSI']].iloc[:-1], df['Target'].iloc[:-1])
        pred = clf.predict(df[['RSI']].iloc[[-1]])[0]
        return {"Pred": "SUBE" if pred==1 else "BAJA", "Acc": 0.65} # Mock accuracy for speed
    except: return None

# --- ESTRATEGIA OPERATIVA (MESA DE DINERO - FLASH) ---
def generar_estrategia_maestra(ticker, snap, macro, val, mc, ml):
    ctx_macro = f"VIX: {macro['VIX']:.2f} ({macro['Estado']})" if macro else "N/A"
    ctx_val = f"Valor Intrínseco: ${val['Valor_Intrinseco']:.2f} (Margen {val['Margen']:.1f}%). {val['Estado']}" if val else "Sin datos."
    ctx_quant = f"Monte Carlo: {mc['Prob_Suba']:.1f}% Suba. ML: {ml['Pred']}" if mc and ml else "N/A"
    
    prompt = f"""
    Actúa como Jefe de Estrategia de Fondo de Cobertura. Escribe un INFORME OPERATIVO EJECUTIVO para **{ticker}** (Precio: ${snap['Precio']:.2f}).
    
    INPUTS:
    1. MACRO: {ctx_macro}
    2. VALORACIÓN (Prioridad 1): {ctx_val}
    3. TÉCNICO: RSI {snap['RSI']:.0f}
    4. QUANT: {ctx_quant}
    
    ESTRUCTURA (Markdown):
    ## 🎯 ESTRATEGIA: [COMPRA / VENTA / ESPERAR]
    
    ### 1. 🏛️ Diagnóstico de Valor (Graham & Dodd)
    Analiza la discrepancia entre Precio y Valor. ¿Hay Margen de Seguridad suficiente (>30%) para justificar una entrada?
    
    ### 2. 📊 Plan Operativo (Niveles)
    * **Zona Entrada:** $[Rango] (Justifica).
    * **Stop Loss:** $[Valor].
    * **Take Profit:** $[Valor].
    
    ### 3. 🧠 Tesis Final
    Conclusión cruzando Valor (Fundamental) con Momento (Técnico/Macro).
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

# --- GRÁFICO ---
def graficar(ticker):
    df = yf.Ticker(ticker).history(period="1y")
    if df.empty: return None
    df['SMA50'] = ta.sma(df['Close'], 50)
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='orange')))
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=0,b=0)); return fig

# --- OPCIONES (FIX CORREGIDO V89) ---
def calcular_payoff(tipo, strike, prima, spot):
    precios = np.linspace(spot*0.7, spot*1.3, 100)
    payoffs = []
    for s in precios:
        val = max(s-strike, 0) if tipo=='Call' else max(strike-s, 0)
        payoffs.append(val - prima)
    return precios, payoffs

# --- INTERFAZ ---
with st.sidebar:
    st.title("🏛️ Quant V89")
    sel_ticker = st.selectbox("Activo", DEFAULT_WATCHLIST)
    if st.button("Operar"): st.sidebar.success("Orden simulada enviada.")

st.title(f"Análisis: {sel_ticker}")
tabs = st.tabs(["📊 DASHBOARD", "🔬 ANÁLISIS 360", "♟️ OPCIONES", "🧠 DIARIO"])

with tabs[0]:
    macro = obtener_macro()
    if macro: 
        c1, c2 = st.columns(2)
        c1.metric("VIX", f"{macro['VIX']:.2f}"); c2.metric("Estado", macro['Estado'])
    df_pos = obtener_cartera()
    if not df_pos.empty: st.dataframe(df_pos)

with tabs[1]:
    # A. GRÁFICO
    fig = graficar(sel_ticker)
    if fig: st.plotly_chart(fig, use_container_width=True)
    
    # B. ESTRATEGIA + VALORACIÓN (PRIORIDAD 1)
    st.markdown("---")
    st.subheader("🎯 Estrategia Maestra")
    snap = get_snapshot(sel_ticker); mc = simulacion_monte_carlo(sel_ticker)
    ml = oraculo_ml(sel_ticker); val = valoracion_academica(sel_ticker)
    
    if 'rep_v89' not in st.session_state: st.session_state['rep_v89'] = None
    
    if st.button("⚡ GENERAR INFORME"):
        with st.spinner("Analizando Valor Intrínseco (Graham/DCF)..."):
            st.session_state['rep_v89'] = generar_estrategia_maestra(sel_ticker, snap, macro, val, mc, ml)
            
    if st.session_state['rep_v89']:
        st.markdown(f"<div class='strat-box'>{st.session_state['rep_v89']}</div>", unsafe_allow_html=True)
        if st.button("📄 PDF"):
            b64 = base64.b64encode(generar_pdf(sel_ticker, st.session_state['rep_v89'])).decode()
            st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="Informe.pdf">Descargar</a>', unsafe_allow_html=True)

    # C. DETALLES VALORACIÓN
    st.markdown("---")
    if val:
        st.markdown("#### 🏛️ Detalle de Valoración (Libros: Graham/Fernández)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Valor Intrínseco", f"${val['Valor_Intrinseco']:.2f}")
        c2.metric("Margen Seguridad", f"{val['Margen']:.1f}%")
        c3.metric("Estado", val['Estado'])
        st.caption(val['Detalle'])

with tabs[2]:
    # CORRECCIÓN V89: Inicialización segura de variables
    precios = []; payoffs = [] 
    
    st.subheader("♟️ Opciones")
    c1, c2 = st.columns([1, 3])
    snap = get_snapshot(sel_ticker); spot = snap['Precio'] if snap else 100
    
    with c1:
        tipo = st.selectbox("Tipo", ["Call", "Put"])
        strike = st.number_input("Strike", value=float(int(spot)))
        prima = st.number_input("Prima", value=5.0)
        precios, payoffs = calcular_payoff(tipo, strike, prima, spot)
        
    with c2:
        if len(precios) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=precios, y=payoffs, mode='lines', fill='tozeroy'))
            fig.add_vline(x=spot, line_color="yellow")
            st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("🧠 Diario")
    conn = sqlite3.connect(DB_NAME)
    try: st.dataframe(pd.read_sql_query("SELECT * FROM trades", conn))
    except: pass
    conn.close()
