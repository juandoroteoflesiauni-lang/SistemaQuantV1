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

# --- 1. CONFIGURACIÓN DEL SISTEMA ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sistema de Inversiones Profesional Quant V85", layout="wide", page_icon="🏛️")

st.markdown("""<style>
    .main {background-color: #0e1117;}
    .metric-card {background-color: #1c1c2e; border: 1px solid #2d2d3f; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
    .metric-value {font-size: 24px; font-weight: bold; color: #ffffff;}
    .metric-label {font-size: 14px; color: #a0a0a0;}
    .strat-box {background-color: #111827; border-left: 5px solid #10B981; padding: 20px; border-radius: 5px; margin-top: 10px; font-family: 'Courier New', monospace;}
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
    .watchlist-box {background-color: #111; padding: 10px; border-radius: 5px; border: 1px solid #333; margin-bottom: 10px;}
</style>""", unsafe_allow_html=True)

try:
    secrets = toml.load(".streamlit/secrets.toml") if os.path.exists(".streamlit/secrets.toml") else st.secrets
    genai.configure(api_key=secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

DB_NAME = "quant_database.db"
DEFAULT_WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'COIN', 'KO', 'DIS', 'SPY', 'QQQ', 'GLD', 'USO']

if 'mis_listas' not in st.session_state:
    st.session_state['mis_listas'] = {"General": DEFAULT_WATCHLIST, "Vigiladas": [], "Cartera": []}
if 'lista_activa' not in st.session_state:
    st.session_state['lista_activa'] = "General"

# --- 2. MOTORES DE DATOS (BACKEND) ---

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

# --- MOTORES DE ANÁLISIS ---

def calcular_vsa_color(row):
    if row['Volume'] > row['Vol_SMA'] * 1.5:
        return 'rgba(0, 255, 0, 0.6)' if row['Close'] > row['Open'] else 'rgba(255, 0, 0, 0.6)'
    elif row['Volume'] < row['Vol_SMA'] * 0.5: return 'rgba(100, 100, 100, 0.3)'
    else: return 'rgba(100, 100, 100, 0.6)'

def detectar_patrones_velas_pro(df):
    df['Cuerpo'] = abs(df['Close'] - df['Open'])
    df['Mecha_Sup'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['Mecha_Inf'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    df['Rango'] = df['High'] - df['Low']; df['Cuerpo_Prom'] = df['Cuerpo'].rolling(10).mean()
    df['Patron_Marubozu_Bull'] = (df['Cuerpo'] > 2 * df['Cuerpo_Prom']) & (df['Mecha_Sup'] < 0.05 * df['Rango']) & (df['Mecha_Inf'] < 0.05 * df['Rango']) & (df['Close'] > df['Open'])
    df['Patron_Marubozu_Bear'] = (df['Cuerpo'] > 2 * df['Cuerpo_Prom']) & (df['Mecha_Sup'] < 0.05 * df['Rango']) & (df['Mecha_Inf'] < 0.05 * df['Rango']) & (df['Close'] < df['Open'])
    df['Patron_Martillo'] = (df['Mecha_Inf'] > 2 * df['Cuerpo']) & (df['Mecha_Sup'] < 0.3 * df['Cuerpo'])
    df['Patron_BullEng'] = (df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)) & (df['Close'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1))
    return df

def calcular_soportes_resistencias(df, window=20):
    df['Resistencia'] = df['High'].rolling(window=window, center=True).max()
    df['Soporte'] = df['Low'].rolling(window=window, center=True).min()
    return df

@st.cache_data(ttl=60)
def obtener_datos_grafico(ticker, intervalo):
    mapa = {"15m": "60d", "1h": "730d", "4h": "2y", "1d": "2y", "1wk": "5y", "1mo": "10y"}
    try:
        df = yf.Ticker(ticker).history(period=mapa.get(intervalo, "1y"), interval=intervalo)
        return df if not df.empty else None
    except: return None

def graficar_profesional_quant(ticker, intervalo):
    df = obtener_datos_grafico(ticker, intervalo)
    if df is None: return None
    
    macd = ta.macd(df['Close']); df = pd.concat([df, macd], axis=1)
    df['RSI'] = ta.rsi(df['Close'], 14)
    try: df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    except: df['VWAP'] = ta.sma(df['Close'], 20)
    df['Vol_SMA'] = ta.sma(df['Volume'], 20)
    colors_vsa = df.apply(calcular_vsa_color, axis=1)
    df = detectar_patrones_velas_pro(df); df = calcular_soportes_resistencias(df)
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.15, 0.15, 0.2], subplot_titles=(f"Precio ({intervalo}) + VWAP + Patrones", "VSA (Volumen)", "MACD", "RSI"))
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Precio'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#FFD700', width=1.5), name='VWAP'), row=1, col=1)
    fig.add_hline(y=df['Resistencia'].iloc[-1], line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=df['Soporte'].iloc[-1], line_dash="dash", line_color="green", row=1, col=1)
    
    maru_bull = df[df['Patron_Marubozu_Bull']]; maru_bear = df[df['Patron_Marubozu_Bear']]; hammer = df[df['Patron_Martillo']]
    fig.add_trace(go.Scatter(x=maru_bull.index, y=maru_bull['Low'], mode='markers', marker=dict(symbol='square', size=8, color='blue'), name='Marubozu Alcista'), row=1, col=1)
    fig.add_trace(go.Scatter(x=maru_bear.index, y=maru_bear['High'], mode='markers', marker=dict(symbol='square', size=8, color='purple'), name='Marubozu Bajista'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hammer.index, y=hammer['Low'], mode='markers', marker=dict(symbol='diamond', size=6, color='cyan'), name='Martillo'), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors_vsa, name='Volumen VSA'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_12_26_9'], line=dict(color='white', width=1), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACDs_12_26_9'], line=dict(color='orange', width=1), name='Signal'), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACDh_12_26_9'], marker_color='gray', name='Hist'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1), name='RSI'), row=4, col=1)
    fig.add_hline(y=70, line_color="red", line_dash="dot", row=4, col=1); fig.add_hline(y=30, line_color="green", line_dash="dot", row=4, col=1)
    fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=30,b=0)); return fig

@st.cache_data(ttl=900)
def get_snapshot(ticker):
    try:
        stock = yf.Ticker(ticker); hist = stock.history(period="5d")
        if hist.empty: return None
        return {"Precio": hist['Close'].iloc[-1], "Previo": hist['Close'].iloc[-2], "RSI": ta.rsi(hist['Close'], 14).iloc[-1] if len(hist)>14 else 50}
    except: return None

# --- MOTOR FUNDAMENTAL REPARADO (V85) ---
@st.cache_data(ttl=3600)
def obtener_fundamentales_premium(ticker):
    if "USD" in ticker: return None
    try:
        stock = yf.Ticker(ticker)
        # Intento primario: Tablas
        try:
            inc = stock.income_stmt.T.sort_index(); bal = stock.balance_sheet.T.sort_index()
            use_tables = not inc.empty and not bal.empty
        except: use_tables = False
        
        # Intento secundario: Info Dict (Más robusto)
        info = stock.info
        
        if use_tables:
            gm = (inc['Gross Profit']/inc['Total Revenue'])*100; nm = (inc['Net Income']/inc['Total Revenue'])*100
            cr = bal['Total Current Assets'].iloc[-1]/bal['Total Current Liabilities'].iloc[-1]
            de = bal.get('Total Debt', pd.Series(0)).iloc[-1]/bal['Stockholders Equity'].iloc[-1]
            fechas = inc.index.strftime('%Y')
        else:
            # Fallback a datos estáticos si fallan las tablas
            gm = info.get('grossMargins', 0) * 100
            nm = info.get('profitMargins', 0) * 100
            cr = info.get('currentRatio', 0)
            de = info.get('debtToEquity', 0) / 100
            fechas = ["Actual"]
            
        return {"Fechas": fechas, "Margen_Bruto": gm, "Margen_Neto": nm, "Current": cr, "Debt": de}
    except: return None

# --- MOTOR INSIDER REPARADO (V85) ---
@st.cache_data(ttl=3600)
def obtener_datos_insider(ticker):
    if "USD" in ticker: return None
    try:
        info = yf.Ticker(ticker).info
        # Extracción directa del diccionario info (Más fiable que holders)
        inst_pct = info.get('heldPercentInstitutions', 0) * 100
        insider_pct = info.get('heldPercentInsiders', 0) * 100
        short_ratio = info.get('shortRatio', 0)
        
        return {"Institucional": inst_pct, "Insiders": insider_pct, "Short_Ratio": short_ratio}
    except: return None

# --- MOTOR SENTIMIENTO (RECUPERADO V80) ---
def analizar_sentimiento_noticias(ticker):
    if "USD" in ticker: return None
    try:
        news = yf.Ticker(ticker).news[:5]
        if not news: return None
        score = 0; bull = ['up', 'rise', 'growth', 'buy', 'strong', 'beat']; bear = ['down', 'drop', 'loss', 'sell', 'weak', 'miss']
        titles = []
        for n in news:
            t = n['title'].lower()
            titles.append(n['title'])
            if any(w in t for w in bull): score += 1
            if any(w in t for w in bear): score -= 1
        label = "Positivo 🟢" if score > 0 else "Negativo 🔴" if score < 0 else "Neutral ⚪"
        return {"Score": score, "Label": label, "Titulos": titles}
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

def oraculo_ml(ticker):
    try:
        df = yf.Ticker(ticker).history(period="2y")
        if len(df)<200: return None
        df['RSI'] = ta.rsi(df['Close'], 14); df['SMA_Diff'] = (df['Close'] - ta.sma(df['Close'], 50))/ta.sma(df['Close'], 50)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        X = df[['RSI', 'SMA_Diff']]; y = df['Target']
        split = int(len(df)*0.8); clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X.iloc[:split], y.iloc[:split])
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

# --- MOTOR ESTRATEGIA OPERATIVA IA (NUEVO V85) ---
def generar_estrategia_operativa_ia(ticker, snap, ml, mc, fund, sent):
    """Genera el plan operativo final usando todos los recursos"""
    
    # Preparar el contexto rico para la IA
    contexto = f"""
    ACTIVO: {ticker} | PRECIO: ${snap['Precio']:.2f}
    
    1. TÉCNICO: RSI {snap['RSI']:.0f}, Tendencia {'Alcista' if snap['Precio'] > snap['Previo'] else 'Bajista'}
    2. ML ORÁCULO: Predice {ml['Pred'] if ml else 'N/A'} (Confianza {ml['Acc'] if ml else 0:.0f}%)
    3. PROBABILIDAD (Monte Carlo): Suba {mc['Prob_Suba'] if mc else 0:.0f}%
    4. FUNDAMENTAL: Liquidez {fund['Current'] if fund else 0:.2f}, Deuda {fund['Debt'] if fund else 0:.2f}
    5. SENTIMIENTO (Noticias): {sent['Label'] if sent else 'Neutral'}
    
    TAREA: Actúa como un Jefe de Mesa de Dinero. Basado en estos datos, genera una ESTRATEGIA OPERATIVA PRECISA.
    
    FORMATO DE SALIDA (Estricto):
    ### 🎯 ESTRATEGIA: [COMPRA / VENTA / ESPERAR]
    
    **Rango de Entrada Ideal:** $[Valor] - $[Valor]
    **Stop Loss (Técnico):** $[Valor] (Justificación breve)
    **Take Profit 1 (Conservador):** $[Valor]
    **Take Profit 2 (Agresivo):** $[Valor]
    
    **Análisis de Situación:**
    Resumen conciso de por qué tomamos esta decisión uniendo los factores técnicos, fundamentales y de sentimiento.
    """
    
    try: return model.generate_content(contexto).text
    except: return "⚠️ Error conectando con el Estratega IA."

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

# --- 4. INTERFAZ GRÁFICA ---

# SIDEBAR
with st.sidebar:
    st.title("🏛️ Prof. Quant V85")
    
    st.markdown("### 📂 Listas")
    lista_actual = st.selectbox("Ver:", list(st.session_state['mis_listas'].keys()), index=0)
    st.session_state['lista_activa'] = lista_actual
    activos_lista = st.session_state['mis_listas'][lista_actual]
    sel_ticker = st.selectbox("🔍 Activo", activos_lista if activos_lista else ["Sin Activos"])
    
    with st.expander("⚙️ Gestionar"):
        nueva_lista = st.text_input("Nueva Lista"); nuevo_ticker = st.text_input("Ticker")
        if st.button("Crear Lista"): 
            if nueva_lista and nueva_lista not in st.session_state['mis_listas']: st.session_state['mis_listas'][nueva_lista] = []; st.rerun()
        if st.button("Agregar"): 
            if nuevo_ticker and nuevo_ticker not in st.session_state['mis_listas'][lista_actual]: st.session_state['mis_listas'][lista_actual].append(nuevo_ticker.upper()); st.rerun()

    st.markdown("### 🧠 Operativa")
    with st.form("trade"):
        q = st.number_input("Cant", 1, 1000, 10); s = st.selectbox("Lado", ["COMPRA", "VENTA"])
        emo = st.select_slider("Emoción", options=["Miedo", "Neutro", "Euforia"]); nota = st.text_area("Nota")
        if st.form_submit_button("EJECUTAR"): 
            snap = get_snapshot(sel_ticker)
            if snap: registrar_operacion(sel_ticker, s, q, snap['Precio'], emo, nota); st.success("OK"); time.sleep(1); st.rerun()

# MAIN
st.title(f"Análisis: {sel_ticker}")

tabs = st.tabs(["📊 DASHBOARD", "🔬 ANÁLISIS 360", "♟️ OPCIONES", "🧠 PSICOLOGÍA"])

# --- TAB 1: DASHBOARD ---
with tabs[0]:
    st.subheader("🌍 Resumen Ejecutivo")
    df_pos = obtener_cartera(); ranking = scanner_mercado(DEFAULT_WATCHLIST)
    patrimonio = df_pos['Valor'].sum() if not df_pos.empty else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Patrimonio", f"${patrimonio:,.2f}")
    if not ranking.empty: c2.metric("Top Oportunidad", ranking.iloc[0]['Ticker'])
    spy = get_snapshot("SPY")
    if spy: c3.metric("S&P 500", f"{((spy['Precio']-spy['Previo'])/spy['Previo'])*100:+.2f}%")
    st.dataframe(ranking.head(5), use_container_width=True)

# --- TAB 2: ANÁLISIS 360 (NUEVO DISEÑO V85) ---
with tabs[1]:
    # 1. GRÁFICO TÉCNICO (Arriba)
    st.subheader("📉 Visión Técnica")
    c_tf, c_rest = st.columns([1, 5])
    timeframe = c_tf.selectbox("Timeframe", ["1d", "15m", "1h", "4h", "1wk"], index=0)
    if sel_ticker != "Sin Activos":
        fig = graficar_profesional_quant(sel_ticker, timeframe)
        if fig: st.plotly_chart(fig, use_container_width=True, height=700)

    # 2. ESTRATEGIA OPERATIVA (Centro)
    st.markdown("---")
    st.subheader("🎯 Estrategia Operativa (IA)")
    
    # Pre-cálculos para la estrategia
    snap = get_snapshot(sel_ticker)
    ml_res = oraculo_ml(sel_ticker)
    mc_res = simulacion_monte_carlo(sel_ticker)
    fund_res = obtener_fundamentales_premium(sel_ticker)
    sent_res = analizar_sentimiento_noticias(sel_ticker)
    
    if st.button("⚡ GENERAR PLAN DE TRADING"):
        if snap:
            with st.spinner("Sintetizando datos técnicos, fundamentales y de sentimiento..."):
                estrategia = generar_estrategia_operativa_ia(sel_ticker, snap, ml_res, mc_res, fund_res, sent_res)
                st.markdown(f"<div class='strat-box'>{estrategia}</div>", unsafe_allow_html=True)
        else: st.error("Datos no disponibles.")

    # 3. RECURSOS DETALLADOS (Abajo)
    st.markdown("---")
    subtabs = st.tabs(["🤖 Oráculo ML", "📚 Fundamentales", "🦈 Insider", "📰 Sentimiento"])
    
    with subtabs[0]:
        if ml_res:
            c1, c2 = st.columns(2)
            c1.metric("Predicción ML", ml_res['Pred'])
            c2.metric("Confianza", f"{ml_res['Acc']:.1f}%")
        else: st.warning("ML necesita más datos.")
        
    with subtabs[1]:
        if fund_res:
            c1, c2, c3 = st.columns(3)
            # Manejo seguro de tipos de datos para visualización
            margen_neto = fund_res['Margen_Neto']
            if isinstance(margen_neto, pd.Series): margen_neto = margen_neto.iloc[-1]
            
            c1.metric("Margen Neto", f"{margen_neto:.2f}%")
            c2.metric("Liquidez", f"{fund_res['Current']:.2f}")
            c3.metric("Deuda/Patrimonio", f"{fund_res['Debt']:.2f}")
        else: st.warning("Fundamentales no disponibles.")
        
    with subtabs[2]:
        insider = obtener_datos_insider(sel_ticker)
        if insider:
            c1, c2 = st.columns(2)
            c1.metric("Institucional", f"{insider['Institucional']:.2f}%")
            c2.metric("Short Ratio", f"{insider['Short_Ratio']:.2f}")
        else: st.warning("Datos Insider no disponibles.")
        
    with subtabs[3]:
        if sent_res:
            st.metric("Sentimiento Noticias", sent_res['Label'])
            for t in sent_res['Titulos']: st.caption(f"• {t}")
        else: st.warning("Sin noticias recientes.")

# --- TAB 3: OPCIONES ---
with tabs[2]:
    st.subheader("♟️ Laboratorio")
    col_op1, col_op2 = st.columns([1, 3])
    snap = get_snapshot(sel_ticker); precio_ref = snap['Precio'] if snap else 100
    with col_op1:
        tipo_est = st.selectbox("Estrategia", ["Simple (Call/Put)", "Bull Call Spread"])
        if tipo_est == "Simple (Call/Put)":
            op_tipo = st.selectbox("Tipo", ["Call", "Put"]); op_pos = st.selectbox("Posición", ["Compra", "Venta"])
            strike = st.number_input("Strike", value=float(int(precio_ref))); prima = st.number_input("Prima", value=5.0)
            precios, payoffs = calcular_payoff_opcion(op_tipo, strike, prima, precio_ref*0.7, precio_ref*1.3, op_pos)
    with col_op2:
        fig_pay = go.Figure()
        fig_pay.add_trace(go.Scatter(x=precios, y=payoffs, mode='lines', name='P&L', fill='tozeroy', line=dict(color='cyan')))
        fig_pay.add_vline(x=precio_ref, line_color="yellow"); st.plotly_chart(fig_pay, use_container_width=True)

# --- TAB 4: PSICOLOGÍA ---
with tabs[3]:
    st.subheader("🧠 Diario")
    conn = sqlite3.connect(DB_NAME)
    try:
        df_diario = pd.read_sql_query("SELECT * FROM trades ORDER BY fecha DESC", conn)
        if not df_diario.empty: st.dataframe(df_diario)
        else: st.info("Sin registros.")
    except: pass
    conn.close()
