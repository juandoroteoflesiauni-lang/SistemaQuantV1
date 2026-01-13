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
st.set_page_config(page_title="Sistema Quant V90 (The Valuator)", layout="wide", page_icon="🏛️")

st.markdown("""<style>
    .main {background-color: #0e1117;}
    .metric-card {background-color: #1c1c2e; border: 1px solid #2d2d3f; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
    .metric-value {font-size: 24px; font-weight: bold; color: #ffffff;}
    .metric-label {font-size: 14px; color: #a0a0a0;}
    .valuation-box {background-color: #141e14; border-left: 5px solid #4caf50; padding: 20px; border-radius: 5px; margin-top: 10px;}
    .strat-box {background-color: #0f172a; border-left: 5px solid #3b82f6; padding: 20px; margin-top: 10px;}
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
</style>""", unsafe_allow_html=True)

try:
    secrets = toml.load(".streamlit/secrets.toml") if os.path.exists(".streamlit/secrets.toml") else st.secrets
    genai.configure(api_key=secrets["GOOGLE_API_KEY"])
    # USAMOS FLASH (Estable y Rápido)
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

# --- 3. MOTOR DE VALORACIÓN ACADÉMICA (PRIORIDAD 1 - V90) ---
@st.cache_data(ttl=3600)
def valoracion_graham_fernandez(ticker):
    """
    Calcula Valor Intrínseco basado en:
    1. Fórmula de Benjamin Graham (Security Analysis)
    2. DCF Simplificado (Valoración de Empresas - P. Fernández)
    """
    if "USD" in ticker: return None
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        precio_actual = info.get('currentPrice', 0)
        if precio_actual == 0: return None

        # --- A. FÓRMULA DE GRAHAM ---
        # V = Raíz(22.5 * EPS * BookValuePerShare)
        eps = info.get('trailingEps', 0)
        bvps = info.get('bookValue', 0)
        
        valor_graham = 0
        if eps > 0 and bvps > 0:
            valor_graham = math.sqrt(22.5 * eps * bvps)
        
        # --- B. DCF SIMPLIFICADO (Fernández) ---
        # Proyección de Free Cash Flow a 5 años
        fcf = info.get('freeCashflow', 0)
        shares = info.get('sharesOutstanding', 1)
        beta = info.get('beta', 1.0)
        
        valor_dcf = 0
        if fcf > 0 and shares > 0:
            fcf_per_share = fcf / shares
            growth_rate = 0.10 # Asumimos 10% conservador para growth stocks
            wacc = 0.09 # Costo de capital estimado (9%)
            
            # Suma de flujos descontados (5 años)
            sum_pv = 0
            for i in range(1, 6):
                future_fcf = fcf_per_share * ((1 + growth_rate) ** i)
                pv = future_fcf / ((1 + wacc) ** i)
                sum_pv += pv
            
            # Valor Terminal (Perpetuidad)
            terminal_val = (fcf_per_share * ((1 + growth_rate) ** 5) * 1.02) / (wacc - 0.02)
            pv_terminal = terminal_val / ((1 + wacc) ** 5)
            
            valor_dcf = sum_pv + pv_terminal

        # --- C. MARGEN DE SEGURIDAD ---
        # Usamos el promedio de ambos modelos como "Valor Intrínseco"
        valores_validos = [v for v in [valor_graham, valor_dcf] if v > 0]
        if not valores_validos: return None
        
        valor_intrinseco = sum(valores_validos) / len(valores_validos)
        margen_seguridad = ((valor_intrinseco - precio_actual) / valor_intrinseco) * 100
        
        estado = "INFRAVALORADA (Oportunidad)" if margen_seguridad > 20 else "SOBREVALORADA (Cara)" if margen_seguridad < -20 else "PRECIO JUSTO"
        
        return {
            "Precio": precio_actual,
            "Valor_Graham": valor_graham,
            "Valor_DCF": valor_dcf,
            "Valor_Intrinseco": valor_intrinseco,
            "Margen_Seguridad": margen_seguridad,
            "Estado": estado,
            "Metricas": {"EPS": eps, "BVPS": bvps, "Beta": beta}
        }
    except: return None

# --- MOTOR MACRO ---
@st.cache_data(ttl=1800)
def obtener_contexto_macro_avanzado():
    try:
        tickers = ["^VIX", "^TNX", "SPY", "QQQ", "IWM"]
        data = yf.download(" ".join(tickers), period="5d", progress=False, auto_adjust=True)['Close']
        vix = data['^VIX'].iloc[-1]
        bond_10y = data['^TNX'].iloc[-1]
        iwm_qqq_ratio = data['IWM'].iloc[-1] / data['QQQ'].iloc[-1]
        iwm_qqq_prev = data['IWM'].iloc[-5] / data['QQQ'].iloc[-5]
        rotacion = "Hacia Riesgo" if iwm_qqq_ratio > iwm_qqq_prev else "Hacia Calidad"
        fear_greed_score = 50 
        if vix < 15: fear_greed_score += 20
        elif vix > 25: fear_greed_score -= 30
        spy_trend = "Alcista" if data['SPY'].iloc[-1] > data['SPY'].iloc[0] else "Bajista"
        if spy_trend == "Alcista": fear_greed_score += 10
        else: fear_greed_score -= 10
        estado = "NEUTRAL"
        if fear_greed_score > 65: estado = "EUFORIA 🟢"
        elif fear_greed_score < 35: estado = "MIEDO 🔴"
        return {"VIX": vix, "Bono_10Y": bond_10y, "Rotacion": rotacion, "Estado_Mercado": estado, "Score_Macro": fear_greed_score}
    except: return None

# --- ESTRATEGIA OPERATIVA (V88/V90) ---
def generar_estrategia_profesional(ticker, snap, macro, val_data, mc, ml):
    ctx_macro = f"VIX: {macro['VIX']:.2f}, Mercado: {macro['Estado_Mercado']}." if macro else "N/A"
    
    # Contexto de Valoración (Nuevo V90)
    if val_data:
        ctx_val = f"Precio: ${val_data['Precio']:.2f} vs Valor Intrínseco: ${val_data['Valor_Intrinseco']:.2f}. Margen Seguridad: {val_data['Margen_Seguridad']:.1f}%. Estado: {val_data['Estado']}."
    else: ctx_val = "Sin datos de valoración."
    
    ctx_tecnico = f"RSI: {snap['RSI']:.0f}."
    ctx_quant = f"Monte Carlo Prob Suba: {mc['Prob_Suba']:.1f}%." if mc else "N/A"

    prompt = f"""
    Eres Director de Inversiones (CIO). Genera INFORME ESTRATÉGICO para {ticker}.
    
    DATOS:
    1. MACRO: {ctx_macro}
    2. VALORACIÓN (Graham/DCF): {ctx_val}
    3. TÉCNICO: {ctx_tecnico}
    4. QUANT: {ctx_quant}
    
    ESTRUCTURA:
    ## 🎯 ESTRATEGIA: [COMPRA / VENTA / ESPERAR]
    
    ### 1. 🏛️ Diagnóstico de Valor (Graham & Dodd)
    Analiza si el activo está barato o caro respecto a su valor intrínseco. ¿Tenemos Margen de Seguridad?
    
    ### 2. 📊 Niveles Operativos
    * **Entrada:** $[Valor]
    * **Stop Loss:** $[Valor]
    * **Take Profit:** $[Valor]
    
    ### 3. 🧠 Tesis Final
    Conclusión cruzando Valor (Fundamental) con Momento (Técnico/Macro).
    """
    try: return model.generate_content(prompt).text
    except Exception as e: return f"⚠️ Error IA: {str(e)}"

# --- PDF ENGINE ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14); self.cell(0, 10, 'INFORME QUANT V90 - VALORACION', 0, 1, 'C'); self.ln(5)
    def footer(self):
        self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

def clean_text(text):
    replacements = {"🟢": "(+)", "🔴": "(-)", "⚠️": "(!)", "💎": "(Val)", "🚀": "(Up)", "📊": "", "🏛️": "", "🧠": "", "🎯": "", "–": "-"}
    for k, v in replacements.items(): text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')

def generar_pdf_profesional(ticker, contenido_ia):
    pdf = PDFReport(); pdf.add_page(); pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font('Arial', 'B', 16); pdf.cell(0, 10, f'ACTIVO: {ticker}', 0, 1, 'L')
    pdf.set_font('Arial', '', 10); pdf.cell(0, 10, f'Fecha: {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'L'); pdf.line(10, 35, 200, 35); pdf.ln(10)
    pdf.set_font('Arial', '', 11); pdf.multi_cell(0, 6, clean_text(contenido_ia))
    return pdf.output(dest='S').encode('latin-1')

# --- MOTORES TÉCNICOS ---
def calcular_vsa_color(row):
    if row['Volume'] > row['Vol_SMA'] * 1.5: return 'rgba(0, 255, 0, 0.6)' if row['Close'] > row['Open'] else 'rgba(255, 0, 0, 0.6)'
    elif row['Volume'] < row['Vol_SMA'] * 0.5: return 'rgba(100, 100, 100, 0.3)'
    else: return 'rgba(100, 100, 100, 0.6)'

def graficar_profesional_quant(ticker, intervalo):
    mapa = {"15m": "60d", "1h": "730d", "4h": "2y", "1d": "2y", "1wk": "5y", "1mo": "10y"}
    try: df = yf.Ticker(ticker).history(period=mapa.get(intervalo, "1y"), interval=intervalo)
    except: return None
    if df.empty: return None
    
    df['SMA50'] = ta.sma(df['Close'], 50); df['Vol_SMA'] = ta.sma(df['Volume'], 20)
    try: df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    except: df['VWAP'] = ta.sma(df['Close'], 20)
    colors = df.apply(calcular_vsa_color, axis=1)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Precio'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#FFD700'), name='VWAP'), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='VSA'), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=30,b=0)); return fig

@st.cache_data(ttl=900)
def get_snapshot(ticker):
    try:
        stock = yf.Ticker(ticker); hist = stock.history(period="5d")
        if hist.empty: return None
        return {"Precio": hist['Close'].iloc[-1], "Previo": hist['Close'].iloc[-2], "RSI": ta.rsi(hist['Close'], 14).iloc[-1] if len(hist)>14 else 50}
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
        return {"Paths": sim_paths, "Dates": [data.index[-1]+timedelta(days=i) for i in range(dias)], "Mean_Price": np.mean(final), "Prob_Suba": np.mean(final>start_price)*100, "VaR_95": np.percentile(final, 5)}
    except: return None

def oraculo_ml(ticker):
    try:
        df = yf.Ticker(ticker).history(period="2y")
        if len(df)<200: return None
        df['RSI'] = ta.rsi(df['Close'], 14); df['SMA_Diff'] = (df['Close'] - ta.sma(df['Close'], 50))/ta.sma(df['Close'], 50)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        X = df[['RSI', 'SMA_Diff']]; y = df['Target']
        split = int(len(df)*0.8); clf = RandomForestClassifier(n_estimators=100).fit(X.iloc[:split], y.iloc[:split])
        acc = accuracy_score(y.iloc[split:], clf.predict(X.iloc[split:]))
        pred = clf.predict(X.iloc[[-1]])[0]; prob = clf.predict_proba(X.iloc[[-1]])[0][pred]
        return {"Pred": "SUBE 🟢" if pred==1 else "BAJA 🔴", "Acc": acc*100, "Prob": prob*100}
    except: return None

def calcular_payoff_opcion(tipo, strike, prima, precio_spot_min, precio_spot_max, posicion='Compra'):
    precios = np.linspace(precio_spot_min, precio_spot_max, 100); payoffs = []
    for S in precios:
        val_intr = max(S - strike, 0) if tipo == 'Call' else max(strike - S, 0)
        pnl = val_intr - prima if posicion == 'Compra' else prima - val_intr
        payoffs.append(pnl)
    return precios, payoffs

def scanner_mercado(tickers):
    ranking = []
    try: data = yf.download(" ".join(tickers), period="6mo", group_by='ticker', progress=False, auto_adjust=True)
    except: return pd.DataFrame()
    for t in tickers:
        try:
            df = data[t].dropna() if len(tickers)>1 else data.dropna()
            if df.empty: continue
            curr = df['Close'].iloc[-1]; rsi = ta.rsi(df['Close'], 14).iloc[-1]
            score = 50 + (20 if rsi < 30 else -10 if rsi > 70 else 0)
            ranking.append({"Ticker": t, "Precio": curr, "RSI": rsi, "Score": score})
        except: pass
    return pd.DataFrame(ranking).sort_values("Score", ascending=False)

# --- 4. INTERFAZ GRÁFICA ---

with st.sidebar:
    st.title("🏛️ Quant V90")
    lista_actual = st.selectbox("Lista:", list(st.session_state['mis_listas'].keys()), index=0)
    activos_lista = st.session_state['mis_listas'][lista_actual]
    sel_ticker = st.selectbox("Activo", activos_lista if activos_lista else ["Sin Activos"])
    
    with st.expander("⚙️ Listas"):
        nl = st.text_input("Nueva"); nt = st.text_input("Ticker")
        if st.button("Crear"): st.session_state['mis_listas'][nl] = []; st.rerun()
        if st.button("Agregar"): st.session_state['mis_listas'][lista_actual].append(nt.upper()); st.rerun()

    st.markdown("### 🧠 Operativa")
    with st.form("trade"):
        q = st.number_input("Qty", 1); s = st.selectbox("Lado", ["COMPRA", "VENTA"])
        emo = st.select_slider("Emoción", ["Miedo", "Neutro", "Euforia"]); nota = st.text_area("Nota")
        if st.form_submit_button("EJECUTAR"): 
            snap = get_snapshot(sel_ticker)
            if snap: registrar_operacion(sel_ticker, s, q, snap['Precio'], emo, nota); st.success("OK"); time.sleep(1); st.rerun()

st.title(f"Análisis: {sel_ticker}")
tabs = st.tabs(["📊 DASHBOARD", "🔬 ANÁLISIS 360", "♟️ OPCIONES", "🧠 PSICOLOGÍA"])

with tabs[0]:
    macro = obtener_contexto_macro_avanzado()
    if macro:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("VIX", f"{macro['VIX']:.2f}"); c2.metric("Estado", macro['Estado_Mercado'])
        c3.metric("Bonos 10Y", f"{macro['Bono_10Y']:.2f}%"); c4.metric("Rotación", macro['Rotacion'])
    st.markdown("---")
    df_pos = obtener_cartera(); ranking = scanner_mercado(DEFAULT_WATCHLIST)
    kc1, kc2 = st.columns([1, 2])
    with kc1: 
        if not df_pos.empty: st.plotly_chart(px.pie(df_pos, values='Valor', names='Ticker', hole=0.5), use_container_width=True)
    with kc2: st.dataframe(ranking.head(5), use_container_width=True)

with tabs[1]:
    # A. GRÁFICO
    st.subheader("📉 Gráfico VSA")
    timeframe = st.selectbox("TF", ["1d", "1h", "15m", "1wk"])
    if sel_ticker != "Sin Activos":
        fig = graficar_profesional_quant(sel_ticker, timeframe)
        if fig: st.plotly_chart(fig, use_container_width=True)

    # B. ESTRATEGIA (CON VALUACIÓN V90)
    st.markdown("---")
    st.subheader("🎯 Estrategia & Valoración")
    
    snap = get_snapshot(sel_ticker); ml_res = oraculo_ml(sel_ticker); mc_res = simulacion_monte_carlo(sel_ticker)
    val_data = valoracion_graham_fernandez(sel_ticker) # NUEVO V90
    
    if 'informe_maestro' not in st.session_state: st.session_state['informe_maestro'] = None
    
    if st.button("⚡ GENERAR INFORME ESTRATÉGICO"):
        if snap and macro:
            with st.spinner("Analizando Valor Intrínseco (Graham/DCF)..."):
                estrategia = generar_estrategia_profesional(sel_ticker, snap, macro, val_data, mc_res, ml_res)
                st.session_state['informe_maestro'] = estrategia
    
    if st.session_state['informe_maestro']:
        st.markdown(f"<div class='strat-box'>{st.session_state['informe_maestro']}</div>", unsafe_allow_html=True)
        if st.button("📄 PDF"):
            b64 = base64.b64encode(generar_pdf_profesional(sel_ticker, st.session_state['informe_maestro'])).decode()
            st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="Informe.pdf">Descargar</a>', unsafe_allow_html=True)

    # C. DETALLE DE VALORACIÓN (NUEVO V90)
    st.markdown("---")
    subtabs = st.tabs(["🏛️ Valoración Académica (V90)", "🤖 Quant", "🦈 Insider"])
    
    with subtabs[0]:
        if val_data:
            st.markdown("#### 🏛️ Análisis de Valor Intrínseco (Prioridad 1)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Precio Actual", f"${val_data['Precio']:.2f}")
            c2.metric("Valor Intrínseco", f"${val_data['Valor_Intrinseco']:.2f}", help="Promedio Graham + DCF")
            
            delta_val = val_data['Margen_Seguridad']
            color_val = "inverse" if delta_val > 0 else "normal"
            c3.metric("Margen de Seguridad", f"{delta_val:.2f}%", delta_color=color_val)
            
            st.markdown(f"""
            <div class='valuation-box'>
                <b>Diagnóstico:</b> {val_data['Estado']}<br>
                <ul>
                    <li>Valor Graham: ${val_data['Valor_Graham']:.2f}</li>
                    <li>Valor DCF (Flujos Descontados): ${val_data['Valor_DCF']:.2f}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else: st.warning("Datos insuficientes para valoración académica.")

    with subtabs[1]:
        if ml_res: st.metric("Predicción ML", ml_res['Pred'], f"Conf: {ml_res['Acc']:.1f}%")
        if mc_res: st.metric("Monte Carlo", f"{mc_res['Prob_Suba']:.1f}% Suba")

    with subtabs[2]:
        insider = obtener_datos_insider(sel_ticker)
        if insider: st.metric("Institucional", f"{insider['Institucional']:.1f}%")

with tabs[2]:
    st.subheader("♟️ Laboratorio")
    col_op1, col_op2 = st.columns([1, 3])
    snap = get_snapshot(sel_ticker); precio_ref = snap['Precio'] if snap else 100
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
