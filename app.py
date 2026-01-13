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
st.set_page_config(page_title="Sistema Quant V91 (Institutional Tracker)", layout="wide", page_icon="🦈")

st.markdown("""<style>
    .main {background-color: #0e1117;}
    .metric-card {background-color: #1c1c2e; border: 1px solid #2d2d3f; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
    .metric-value {font-size: 24px; font-weight: bold; color: #ffffff;}
    .vsa-alert {background-color: #2b1c2e; border-left: 4px solid #d81b60; padding: 10px; margin-bottom: 5px; border-radius: 4px;}
    .strat-box {background-color: #0f172a; border-left: 5px solid #3b82f6; padding: 20px; margin-top: 10px;}
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
</style>""", unsafe_allow_html=True)

try:
    secrets = toml.load(".streamlit/secrets.toml") if os.path.exists(".streamlit/secrets.toml") else st.secrets
    genai.configure(api_key=secrets["GOOGLE_API_KEY"])
    # VOLVEMOS A LA VERSIÓN RÁPIDA (FLASH 2.0 EXP)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
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

# --- 3. MOTOR VSA INSTITUCIONAL (PRIORIDAD 2 - V91) ---
def motor_vsa_tom_williams(df):
    """
    Detecta huellas institucionales según 'Master the Markets'.
    Señales: No Demand, Stopping Volume, Upthrust, Climax.
    """
    if df.empty: return df
    
    # 1. Definiciones Básicas
    df['Spread'] = df['High'] - df['Low'] # Rango de la vela
    df['Spread_Avg'] = df['Spread'].rolling(20).mean()
    df['Vol_Avg'] = df['Volume'].rolling(20).mean()
    
    # Close Location Value (CLV): Dónde cerró la vela (-1 abajo, +1 arriba)
    # (C - L) - (H - C) / (H - L)
    df['CLV'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    df['CLV'] = df['CLV'].fillna(0) # Fix div by zero
    
    # 2. Señales VSA
    
    # A. UPTHRUST (Trampa Alcista)
    # Mínimos más altos, Cierra en mínimos, Volumen alto
    df['VSA_Upthrust'] = (df['High'] > df['High'].shift(1)) & \
                         (df['Close'] < df['Low'] + (df['Spread'] * 0.3)) & \
                         (df['Volume'] > df['Vol_Avg'] * 1.2) & \
                         (df['Spread'] > df['Spread_Avg'])

    # B. NO DEMAND (Debilidad)
    # Vela alcista, rango estrecho, volumen bajo
    df['VSA_NoDemand'] = (df['Close'] > df['Close'].shift(1)) & \
                         (df['Spread'] < df['Spread_Avg']) & \
                         (df['Volume'] < df['Vol_Avg'])
                         
    # C. STOPPING VOLUME (Fortaleza)
    # Vela bajista, volumen muy alto, cierra lejos de mínimos (Institucional comprando la caída)
    df['VSA_Stopping'] = (df['Close'] < df['Close'].shift(1)) & \
                         (df['Volume'] > df['Vol_Avg'] * 1.5) & \
                         (df['Close'] > df['Low'] + (df['Spread'] * 0.4))
                         
    # D. CLIMAX (Volatilidad Extrema)
    # Volumen ultra alto + Rango ultra alto
    df['VSA_Climax'] = (df['Volume'] > df['Vol_Avg'] * 2.0) & \
                       (df['Spread'] > df['Spread_Avg'] * 1.5)

    return df

# --- 4. MOTOR DE VALORACIÓN (V90) ---
@st.cache_data(ttl=3600)
def valoracion_graham_fernandez(ticker):
    if "USD" in ticker: return None
    try:
        stock = yf.Ticker(ticker); info = stock.info
        precio = info.get('currentPrice', 0)
        if precio == 0: return None
        
        eps = info.get('trailingEps', 0); bvps = info.get('bookValue', 0)
        v_graham = math.sqrt(22.5 * eps * bvps) if eps > 0 and bvps > 0 else 0
        
        fcf = info.get('freeCashflow', 0); shares = info.get('sharesOutstanding', 1)
        v_dcf = 0
        if fcf > 0 and shares > 0:
            fcf_ps = fcf/shares; g=0.10; wacc=0.09
            v_dcf = sum([fcf_ps*((1+g)**i)/((1+wacc)**i) for i in range(1,6)]) + ((fcf_ps*((1+g)**5)*1.02)/(wacc-0.02))/((1+wacc)**5)
            
        vals = [v for v in [v_graham, v_dcf] if v > 0]
        if not vals: return None
        v_intr = sum(vals)/len(vals)
        margen = ((v_intr - precio)/v_intr)*100
        estado = "Oportunidad 💎" if margen > 20 else "Cara ⚠️" if margen < -20 else "Precio Justo ⚖️"
        
        return {"Precio": precio, "Valor_Intrinseco": v_intr, "Margen": margen, "Estado": estado}
    except: return None

# --- MOTORES SOPORTE ---
@st.cache_data(ttl=1800)
def obtener_contexto_macro_avanzado():
    try:
        tickers = ["^VIX", "^TNX", "SPY", "QQQ", "IWM"]
        data = yf.download(" ".join(tickers), period="5d", progress=False, auto_adjust=True)['Close']
        vix = data['^VIX'].iloc[-1]; bond = data['^TNX'].iloc[-1]
        rot = "Risk ON" if (data['IWM'].iloc[-1]/data['QQQ'].iloc[-1]) > (data['IWM'].iloc[-5]/data['QQQ'].iloc[-5]) else "Risk OFF"
        estado = "Miedo" if vix > 20 else "Calma"
        return {"VIX": vix, "Bono": bond, "Rotacion": rot, "Estado": estado}
    except: return None

def simulacion_monte_carlo(ticker, dias=30, simulaciones=100):
    try:
        data = yf.Ticker(ticker).history(period="1y")['Close']
        if data.empty: return None
        returns = data.pct_change().dropna(); mu = returns.mean(); sigma = returns.std(); start_price = data.iloc[-1]
        sim_paths = np.zeros((dias, simulaciones)); sim_paths[0] = start_price
        for t in range(1, dias):
            drift = (mu - 0.5 * sigma**2); shock = sigma * np.random.normal(0, 1, simulaciones)
            sim_paths[t] = sim_paths[t-1] * np.exp(drift + shock)
        final = sim_paths[-1]
        return {"Prob_Suba": np.mean(final>start_price)*100, "VaR_95": np.percentile(final, 5)}
    except: return None

def oraculo_ml(ticker):
    try:
        df = yf.Ticker(ticker).history(period="2y")
        if len(df)<200: return None
        df['RSI'] = ta.rsi(df['Close'], 14); df['SMA_Diff'] = (df['Close'] - ta.sma(df['Close'], 50))/ta.sma(df['Close'], 50)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int); df=df.dropna()
        X = df[['RSI', 'SMA_Diff']]; y = df['Target']; split = int(len(df)*0.8)
        clf = RandomForestClassifier(n_estimators=100).fit(X.iloc[:split], y.iloc[:split])
        acc = accuracy_score(y.iloc[split:], clf.predict(X.iloc[split:]))
        pred = clf.predict(X.iloc[[-1]])[0]; prob = clf.predict_proba(X.iloc[[-1]])[0][pred]
        return {"Pred": "SUBE" if pred==1 else "BAJA", "Acc": acc*100}
    except: return None

def calcular_payoff_opcion(tipo, strike, prima, precio_spot_min, precio_spot_max, posicion='Compra'):
    precios = np.linspace(precio_spot_min, precio_spot_max, 100); payoffs = []
    for S in precios:
        val_intr = max(S - strike, 0) if tipo == 'Call' else max(strike - S, 0)
        pnl = val_intr - prima if posicion == 'Compra' else prima - val_intr
        payoffs.append(pnl)
    return precios, payoffs

# --- ESTRATEGIA (V88 Modificada para incluir VSA) ---
def generar_estrategia_vsa(ticker, macro, val, mc, ml, vsa_signal):
    ctx_macro = f"VIX: {macro['VIX']:.2f} ({macro['Estado']})." if macro else "N/A"
    ctx_val = f"Margen Seguridad: {val['Margen']:.1f}% ({val['Estado']})." if val else "N/A"
    ctx_quant = f"Prob Monte Carlo: {mc['Prob_Suba']:.1f}%. ML: {ml['Pred']}." if mc and ml else "N/A"
    ctx_vsa = f"Señal Institucional (Tom Williams): {vsa_signal}" if vsa_signal else "Sin huellas institucionales claras hoy."

    prompt = f"""
    Actúa como un Trader Institucional experto en VSA (Volume Spread Analysis). Genera un INFORME OPERATIVO para {ticker}.
    
    INPUTS:
    1. MACRO: {ctx_macro}
    2. VALORACIÓN: {ctx_val}
    3. QUANT: {ctx_quant}
    4. VSA (Volumen): {ctx_vsa}
    
    ESTRUCTURA:
    ## 🦈 ESTRATEGIA VSA: [ACUMULAR / DISTRIBUIR / ESPERAR]
    
    ### 1. Lectura del Profesional (VSA)
    Analiza si hay 'No Demand' (debilidad), 'Stopping Volume' (fortaleza) o 'Clímax'. ¿Qué están haciendo las manos fuertes?
    
    ### 2. Sincronización con Valor
    Cruza la señal de volumen con el Valor Intrínseco. ¿El profesional está comprando barato?
    
    ### 3. Plan de Ejecución
    * **Trigger VSA:** (Ej: Esperar testeo de volumen bajo).
    * **Zona de Entrada:** $...
    * **Stop Loss:** $...
    """
    try: return model.generate_content(prompt).text
    except Exception as e: return f"Error IA: {str(e)}"

# --- GRAFICADOR V91 (CON VSA MARKERS) ---
def graficar_vsa_pro(ticker, intervalo):
    mapa = {"15m": "60d", "1h": "730d", "4h": "2y", "1d": "2y", "1wk": "5y"}
    try: df = yf.Ticker(ticker).history(period=mapa.get(intervalo, "1y"), interval=intervalo)
    except: return None, None
    if df.empty: return None, None
    
    # Procesar VSA
    df = motor_vsa_tom_williams(df)
    
    # Señal reciente para el informe
    last_vsa = "Neutral"
    if df['VSA_Stopping'].iloc[-1]: last_vsa = "Stopping Volume (Posible Suelo)"
    elif df['VSA_NoDemand'].iloc[-1]: last_vsa = "No Demand (Debilidad)"
    elif df['VSA_Upthrust'].iloc[-1]: last_vsa = "Upthrust (Trampa Alcista)"
    elif df['VSA_Climax'].iloc[-1]: last_vsa = "Volumen Climático (Giro inminente)"

    # Indicadores
    try: df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    except: df['VWAP'] = ta.sma(df['Close'], 20)
    
    # Figuras
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Precio'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#FFD700', width=1.5), name='VWAP'), row=1, col=1)
    
    # Marcadores VSA
    stop_vol = df[df['VSA_Stopping']]
    no_dem = df[df['VSA_NoDemand']]
    upthrust = df[df['VSA_Upthrust']]
    
    fig.add_trace(go.Scatter(x=stop_vol.index, y=stop_vol['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ff00'), name='Stopping Vol'), row=1, col=1)
    fig.add_trace(go.Scatter(x=no_dem.index, y=no_dem['High']*1.02, mode='markers', marker=dict(symbol='circle-open', size=10, color='gray'), name='No Demand'), row=1, col=1)
    fig.add_trace(go.Scatter(x=upthrust.index, y=upthrust['High']*1.02, mode='markers', marker=dict(symbol='x', size=10, color='red'), name='Upthrust'), row=1, col=1)
    
    # Colores Volumen
    colors = ['rgba(0,255,0,0.5)' if r['Close'] > r['Open'] else 'rgba(255,0,0,0.5)' for i,r in df.iterrows()]
    # Resaltar Clímax en Volumen
    for i in range(len(df)):
        if df['VSA_Climax'].iloc[i]: colors[i] = 'purple' # Clímax es violeta
            
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volumen'), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,t=30,b=0))
    
    return fig, last_vsa

# --- PDF ENGINE ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14); self.cell(0, 10, 'INFORME QUANT V91 - VSA', 0, 1, 'C'); self.ln(5)
    def footer(self):
        self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

def clean(t): 
    replacements = {"🟢": "(+)", "🔴": "(-)", "⚠️": "(!)", "💎": "(Val)", "🦈": "(VSA)", "📊": ""}
    for k, v in replacements.items(): t = t.replace(k, v)
    return t.encode('latin-1', 'replace').decode('latin-1')

def generar_pdf(ticker, txt):
    pdf = PDFReport(); pdf.add_page(); pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f'ACTIVO: {ticker}', 0, 1); pdf.ln(10)
    pdf.set_font('Arial', '', 11); pdf.multi_cell(0, 6, clean(txt))
    return pdf.output(dest='S').encode('latin-1')

# --- 4. INTERFAZ GRÁFICA ---

with st.sidebar:
    st.title("🏛️ Quant V91")
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
            registrar_operacion(sel_ticker, s, q, 0, emo, nota); st.success("OK"); time.sleep(1); st.rerun()

st.title(f"Análisis Institucional: {sel_ticker}")
tabs = st.tabs(["📊 DASHBOARD", "🦈 RASTREADOR VSA (Micro)", "♟️ OPCIONES", "🧠 PSICOLOGÍA"])

with tabs[0]:
    macro = obtener_contexto_macro_avanzado()
    if macro:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("VIX", f"{macro['VIX']:.2f}"); c2.metric("Estado", macro['Estado'])
        c3.metric("Bono 10Y", f"{macro['Bono']:.2f}%"); c4.metric("Rotación", macro['Rotacion'])
    st.markdown("---")
    df_pos = obtener_cartera()
    if not df_pos.empty: st.plotly_chart(px.pie(df_pos, values='Valor', names='Ticker', hole=0.5), use_container_width=True)

with tabs[1]:
    # A. Gráfico VSA
    st.subheader("📉 Gráfico de Precio y Volumen (Tom Williams)")
    timeframe = st.selectbox("TF", ["1d", "1h", "15m", "1wk"])
    fig_vsa, signal_vsa = graficar_vsa_pro(sel_ticker, timeframe)
    
    if fig_vsa: 
        st.plotly_chart(fig_vsa, use_container_width=True)
        # Alerta VSA en pantalla
        if signal_vsa != "Neutral":
            st.markdown(f"<div class='vsa-alert'>🦈 <b>HUELLA INSTITUCIONAL DETECTADA:</b> {signal_vsa}</div>", unsafe_allow_html=True)
    
    # B. Estrategia
    st.markdown("---")
    st.subheader("🎯 Estrategia VSA & Valor")
    val_data = valoracion_graham_fernandez(sel_ticker)
    mc_res = simulacion_monte_carlo(sel_ticker)
    ml_res = oraculo_ml(sel_ticker)
    
    if 'reporte_v91' not in st.session_state: st.session_state['reporte_v91'] = None
    
    if st.button("⚡ GENERAR INFORME INSTITUCIONAL"):
        with st.spinner("Analizando Esfuerzo vs. Resultado..."):
            rep = generar_estrategia_vsa(sel_ticker, macro, val_data, mc_res, ml_res, signal_vsa)
            st.session_state['reporte_v91'] = rep
            
    if st.session_state['reporte_v91']:
        st.markdown(f"<div class='strat-box'>{st.session_state['reporte_v91']}</div>", unsafe_allow_html=True)
        if st.button("📄 PDF"):
            b64 = base64.b64encode(generar_pdf(sel_ticker, st.session_state['reporte_v91'])).decode()
            st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="VSA_Report.pdf">Descargar</a>', unsafe_allow_html=True)

    # C. Detalles
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if val_data:
            st.markdown("#### 🏛️ Valoración (Prioridad 1)")
            st.metric("Valor Intrínseco", f"${val_data['Valor_Intrinseco']:.2f}")
            st.metric("Margen Seguridad", f"{val_data['Margen']:.1f}%")
    with c2:
        if mc_res:
            st.markdown("#### 🤖 Quant")
            st.metric("Probabilidad", f"{mc_res['Prob_Suba']:.1f}%")

with tabs[2]:
    st.subheader("♟️ Opciones")
    col_op1, col_op2 = st.columns([1, 3])
    snap = yf.Ticker(sel_ticker).history(period="1d"); precio_ref = snap['Close'].iloc[-1] if not snap.empty else 100
    precios = []; payoffs = []
    with col_op1:
        tipo_est = st.selectbox("Estrategia", ["Simple (Call/Put)", "Bull Call Spread"])
        if tipo_est == "Simple (Call/Put)":
            op_tipo = st.selectbox("Tipo", ["Call", "Put"]); op_pos = st.selectbox("Posición", ["Compra", "Venta"])
            strike = st.number_input("Strike", value=float(int(precio_ref))); prima = st.number_input("Prima", value=5.0)
            precios, payoffs = calcular_payoff_opcion(op_tipo, strike, prima, precio_ref*0.7, precio_ref*1.3, op_pos)
    with col_op2:
        if len(precios)>0:
            fig_pay = go.Figure()
            fig_pay.add_trace(go.Scatter(x=precios, y=payoffs, mode='lines', name='P&L', fill='tozeroy', line=dict(color='cyan')))
            fig_pay.add_vline(x=precio_ref, line_color="yellow"); st.plotly_chart(fig_pay, use_container_width=True)

with tabs[3]:
    st.subheader("🧠 Diario")
    conn = sqlite3.connect(DB_NAME)
    try:
        df_diario = pd.read_sql_query("SELECT * FROM trades ORDER BY fecha DESC", conn)
        if not df_diario.empty: st.dataframe(df_diario)
    except: pass
    conn.close()
