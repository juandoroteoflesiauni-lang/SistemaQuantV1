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
st.set_page_config(page_title="Sistema de Inversiones Profesional Quant V88", layout="wide", page_icon="🏛️")

st.markdown("""<style>
    .main {background-color: #0e1117;}
    .metric-card {background-color: #1c1c2e; border: 1px solid #2d2d3f; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
    .metric-value {font-size: 24px; font-weight: bold; color: #ffffff;}
    .metric-label {font-size: 14px; color: #a0a0a0;}
    .strat-box {background-color: #0f172a; border: 1px solid #3b82f6; border-left: 5px solid #3b82f6; padding: 25px; border-radius: 8px; margin-top: 15px; font-family: 'Segoe UI', sans-serif; line-height: 1.6; white-space: pre-wrap;}
    .macro-card {background-color: #2a1a1a; border: 1px solid #ff4b4b; padding: 10px; border-radius: 5px; text-align: center;}
    .macro-safe {background-color: #1a2a1a; border: 1px solid #00cc96; padding: 10px; border-radius: 5px; text-align: center;}
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
</style>""", unsafe_allow_html=True)

try:
    secrets = toml.load(".streamlit/secrets.toml") if os.path.exists(".streamlit/secrets.toml") else st.secrets
    genai.configure(api_key=secrets["GOOGLE_API_KEY"])
    # CAMBIO V88: Usamos el modelo PRO para razonamiento complejo
    model = genai.GenerativeModel('gemini-1.5-pro') 
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

# --- MOTOR MACROECONÓMICO AVANZADO ---
@st.cache_data(ttl=1800)
def obtener_contexto_macro_avanzado():
    try:
        tickers = ["^VIX", "^TNX", "SPY", "QQQ", "IWM"]
        data = yf.download(" ".join(tickers), period="5d", progress=False, auto_adjust=True)['Close']
        vix = data['^VIX'].iloc[-1]
        bond_10y = data['^TNX'].iloc[-1]
        iwm_qqq_ratio = data['IWM'].iloc[-1] / data['QQQ'].iloc[-1]
        iwm_qqq_prev = data['IWM'].iloc[-5] / data['QQQ'].iloc[-5]
        rotacion = "Hacia Riesgo (Small Caps)" if iwm_qqq_ratio > iwm_qqq_prev else "Hacia Seguridad (Big Tech)"
        fear_greed_score = 50 
        if vix < 15: fear_greed_score += 20
        elif vix > 25: fear_greed_score -= 30
        spy_trend = "Alcista" if data['SPY'].iloc[-1] > data['SPY'].iloc[0] else "Bajista"
        if spy_trend == "Alcista": fear_greed_score += 10
        else: fear_greed_score -= 10
        estado = "NEUTRAL"
        if fear_greed_score > 65: estado = "EUFORIA (Greed) 🟢"
        elif fear_greed_score < 35: estado = "PÁNICO (Fear) 🔴"
        return {"VIX": vix, "Bono_10Y": bond_10y, "Rotacion": rotacion, "Estado_Mercado": estado, "Score_Macro": fear_greed_score}
    except: return None

# --- MOTOR FUNDAMENTAL Y NOTICIAS ---
@st.cache_data(ttl=3600)
def analisis_fundamental_y_noticias(ticker):
    if "USD" in ticker: return None
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        score_fin = 30
        margins = info.get('profitMargins', 0)
        if margins > 0.20: score_fin += 10
        elif margins < 0.05: score_fin += 5
        roe = info.get('returnOnEquity', 0)
        if roe > 0.15: score_fin += 10
        current_ratio = info.get('currentRatio', 0)
        if current_ratio > 1.5: score_fin += 5
        debt_eq = info.get('debtToEquity', 0)
        if debt_eq < 100: score_fin += 5
        score_news = 20
        noticias_relevantes = []
        try:
            news = stock.news[:5]
            bull_words = ['beat', 'rise', 'up', 'growth', 'strong', 'buy', 'record', 'gain']
            bear_words = ['miss', 'fall', 'down', 'weak', 'sell', 'loss', 'drop', 'risk']
            for n in news:
                title = n['title']
                txt_lower = title.lower()
                impacto = 0
                if any(w in txt_lower for w in bull_words): impacto += 1
                if any(w in txt_lower for w in bear_words): impacto -= 1
                score_news += (impacto * 4)
                label = "Positivo" if impacto > 0 else "Negativo" if impacto < 0 else "Neutral"
                noticias_relevantes.append(f"{label}: {title}")
        except: pass
        total_score = min(100, max(0, score_fin + score_news - 20))
        calidad = "EXCELENTE 💎" if total_score > 75 else "SÓLIDA ✅" if total_score > 50 else "DÉBIL ⚠️"
        return {"Score_Total": total_score, "Calidad": calidad, "Ratios": {"Margen_Neto": margins * 100, "ROE": roe * 100, "Deuda_Eq": debt_eq, "Liquidez": current_ratio}, "Noticias_Resumen": noticias_relevantes}
    except: return None

# --- ESTRATEGIA OPERATIVA PROFESIONAL (IA V88 - DEEP ANALYSIS) ---
def generar_estrategia_profesional(ticker, snap, macro, fund, mc, ml, dcf):
    """Prompt Maestro para Informes Extensos tipo 'Hedge Fund'"""
    
    ctx_macro = f"VIX: {macro['VIX']:.2f}, Bonos 10Y: {macro['Bono_10Y']:.2f}%, Mercado: {macro['Estado_Mercado']}, Rotación: {macro['Rotacion']}." if macro else "Sin datos macro."
    ctx_micro = "Sin datos fundamentales."
    if fund: ctx_micro = f"Score Calidad: {fund['Score_Total']}/100 ({fund['Calidad']}). Margen Neto: {fund['Ratios']['Margen_Neto']:.2f}%. Noticias recientes: {'; '.join(fund['Noticias_Resumen'][:3])}."
    ctx_tecnico = f"Precio: ${snap['Precio']:.2f}. RSI: {snap['RSI']:.0f}. Tendencia CP: {'Alcista' if snap['Precio']>snap['Previo'] else 'Bajista'}."
    ctx_quant = f"Monte Carlo (30d): Probabilidad Suba {mc['Prob_Suba']:.1f}%. Riesgo VaR: ${mc['VaR_95']:.2f}. ML Predice: {ml['Pred']} (Conf: {ml['Acc']:.0f}%)." if mc and ml else "Faltan modelos quant."
    val_dcf = f"${dcf:.2f}" if dcf else "N/A"

    # Prompt V88: Estructurado para generar 1000+ palabras de valor
    prompt = f"""
    Actúa como el Director de Inversiones (CIO) de un Fondo Cuantitativo Global. Escribe un INFORME ESTRATÉGICO PROFESIONAL EXTENSO para el activo **{ticker}**.
    
    **DATOS EN TIEMPO REAL:**
    1. MACROECONOMÍA: {ctx_macro}
    2. FUNDAMENTALES: {ctx_micro}
    3. TÉCNICO: {ctx_tecnico}
    4. QUANT & RIESGO: {ctx_quant}
    5. VALUACIÓN (DCF): {val_dcf}
    
    ---
    **INSTRUCCIONES DE FORMATO Y CONTENIDO:**
    El informe debe ser detallado, analítico y ejecutable. Usa un tono serio y profesional.
    
    **SECCIÓN 1: RESUMEN EJECUTIVO Y DIAGNÓSTICO**
    * **Veredicto:** [COMPRA FUERTE / ACUMULACIÓN / MANTENER / VENTA]
    * **Tesis Central:** Resume en un párrafo potente por qué el activo está en esta situación, cruzando el entorno Macro (VIX) con sus Fundamentales.
    
    **SECCIÓN 2: ANÁLISIS FUNDAMENTAL PROFUNDO**
    * **Salud Financiera:** Analiza los márgenes y la deuda proporcionados. ¿Es una empresa solvente?
    * **Valuación:** Compara el Precio Actual vs. el Valor DCF. ¿Está infravalorada o sobrevalorada? ¿Qué dice el mercado?
    * **Contexto Corporativo:** Interpreta el impacto de las noticias recientes mencionadas.
    
    **SECCIÓN 3: ANÁLISIS TÉCNICO Y QUANT**
    * **Lectura del Precio:** Interpreta el RSI y la Tendencia. ¿Estamos en zona de compra o venta institucional?
    * **Modelos Matemáticos:** Explica qué significa la predicción del Oráculo ML y la probabilidad de Monte Carlo para este escenario. ¿Apoyan la tesis alcista/bajista?
    
    **SECCIÓN 4: PLAN OPERATIVO (MESA DE DINERO)**
    Proporciona niveles de precios exactos para configurar las órdenes:
    * **Zona de Entrada (Buy Zone):** $[Rango] (Justifica técnicamente).
    * **Stop Loss (Protección):** $[Valor] (Calculado para evitar barridos de volatilidad).
    * **Take Profit 1 (Corto Plazo):** $[Valor].
    * **Take Profit 2 (Objetivo Estructural):** $[Valor].
    
    **SECCIÓN 5: CONCLUSIÓN Y RIESGOS**
    * Enumera 3 riesgos principales (Macro, Sectorial, Específico).
    * Cierra con una recomendación final de gestión de cartera (tamaño de posición sugerido: pequeño/medio/grande).
    """
    
    try: return model.generate_content(prompt).text
    except Exception as e: return f"⚠️ Error IA: {str(e)}"

# --- PDF ENGINE V88 (SOPORTE TEXTO LARGO) ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'INFORME ESTRATEGICO QUANT - MESA DE DINERO', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

def clean_text(text):
    # Reemplazo de caracteres no soportados por Latin-1
    replacements = {
        "🟢": "(+)", "🔴": "(-)", "⚠️": "(!)", "💎": "(Val)", "🚀": "(Up)",
        "📊": "", "🏛️": "", "🧠": "", "🎯": "", "–": "-"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Codificación segura
    return text.encode('latin-1', 'replace').decode('latin-1')

def generar_pdf_profesional(ticker, contenido_ia):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Título del Activo
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f'ACTIVO: {ticker}', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Fecha de Emision: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'L')
    pdf.line(10, 35, 200, 35)
    pdf.ln(10)
    
    # Contenido del Informe
    pdf.set_font('Arial', '', 11)
    # Usamos multi_cell para texto extenso
    texto_limpio = clean_text(contenido_ia)
    pdf.multi_cell(0, 6, texto_limpio)
    
    return pdf.output(dest='S').encode('latin-1')

# --- MOTORES TÉCNICOS ---
def calcular_vsa_color(row):
    if row['Volume'] > row['Vol_SMA'] * 1.5: return 'rgba(0, 255, 0, 0.6)' if row['Close'] > row['Open'] else 'rgba(255, 0, 0, 0.6)'
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
    df['Patron_Doji'] = df['Cuerpo'] <= df['Rango'] * 0.1
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

# --- MOTORES SOPORTE ---
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
        split = int(len(df)*0.8); clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X.iloc[:split], y.iloc[:split])
        acc = accuracy_score(y.iloc[split:], clf.predict(X.iloc[split:]))
        pred = clf.predict(X.iloc[[-1]])[0]; prob = clf.predict_proba(X.iloc[[-1]])[0][pred]
        return {"Pred": "SUBE 🟢" if pred==1 else "BAJA 🔴", "Acc": acc*100, "Prob": prob*100}
    except: return None

@st.cache_data(ttl=3600)
def obtener_datos_insider(ticker):
    if "USD" in ticker: return None
    try:
        info = yf.Ticker(ticker).info
        return {"Institucional": info.get('heldPercentInstitutions', 0)*100, "Short_Float": (info.get('sharesShort', 0)/info.get('floatShares', 1))*100}
    except: return None

def calcular_payoff_opcion(tipo, strike, prima, precio_spot_min, precio_spot_max, posicion='Compra'):
    precios = np.linspace(precio_spot_min, precio_spot_max, 100); payoffs = []
    for S in precios:
        val_intr = max(S - strike, 0) if tipo == 'Call' else max(strike - S, 0)
        pnl = val_intr - prima if posicion == 'Compra' else prima - val_intr
        payoffs.append(pnl)
    return precios, payoffs

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
    st.title("🏛️ Prof. Quant V88")
    lista_actual = st.selectbox("Lista:", list(st.session_state['mis_listas'].keys()), index=0)
    activos_lista = st.session_state['mis_listas'][lista_actual]
    sel_ticker = st.selectbox("Activo", activos_lista if activos_lista else ["Sin Activos"])
    
    with st.expander("⚙️ Listas"):
        nl = st.text_input("Nueva"); nt = st.text_input("Ticker")
        if st.button("Crear") and nl: st.session_state['mis_listas'][nl] = []; st.rerun()
        if st.button("Agregar") and nt: st.session_state['mis_listas'][lista_actual].append(nt.upper()); st.rerun()

    st.markdown("### 🧠 Operativa")
    with st.form("trade"):
        q = st.number_input("Qty", 1); s = st.selectbox("Lado", ["COMPRA", "VENTA"])
        emo = st.select_slider("Emoción", ["Miedo", "Neutro", "Euforia"]); nota = st.text_area("Nota")
        if st.form_submit_button("EJECUTAR"): 
            snap = get_snapshot(sel_ticker)
            if snap: registrar_operacion(sel_ticker, s, q, snap['Precio'], emo, nota); st.success("OK"); time.sleep(1); st.rerun()

# MAIN
st.title(f"Análisis: {sel_ticker}")

tabs = st.tabs(["📊 DASHBOARD", "🔬 ANÁLISIS 360 (MASTER)", "♟️ OPCIONES", "🧠 PSICOLOGÍA"])

# --- TAB 1: DASHBOARD ---
with tabs[0]:
    st.subheader("🌍 Resumen Ejecutivo Macro")
    macro = obtener_contexto_macro_avanzado()
    if macro:
        c1, c2, c3, c4 = st.columns(4)
        vix_color = "risk-alert" if macro['VIX'] > 20 else "risk-safe"
        c1.markdown(f"<div class='{vix_color}'>VIX (Miedo)<br><b>{macro['VIX']:.2f}</b></div>", unsafe_allow_html=True)
        c2.metric("Sentimiento Global", macro['Estado_Mercado'])
        c3.metric("Bonos 10Y", f"{macro['Bono_10Y']:.2f}%")
        c4.metric("Flujo Dinero", macro['Rotacion'])
    
    st.markdown("---")
    df_pos = obtener_cartera(); ranking = scanner_mercado(DEFAULT_WATCHLIST)
    kc1, kc2 = st.columns([1, 2])
    with kc1: 
        if not df_pos.empty: st.plotly_chart(px.pie(df_pos, values='Valor', names='Ticker', hole=0.5, title="Cartera"), use_container_width=True)
        else: st.info("Cartera vacía.")
    with kc2: st.dataframe(ranking.head(5), use_container_width=True)

# --- TAB 2: ANÁLISIS 360 (MASTER STRATEGIST V88) ---
with tabs[1]:
    # 1. VISUALIZACIÓN TÉCNICA
    st.subheader("📉 Visión Técnica Profesional")
    timeframe = st.selectbox("Timeframe", ["1d", "15m", "1h", "4h", "1wk"], index=0)
    if sel_ticker != "Sin Activos":
        fig = graficar_profesional_quant(sel_ticker, timeframe)
        if fig: st.plotly_chart(fig, use_container_width=True, height=700)

    # 2. ESTRATEGIA OPERATIVA (RENOVADA V88)
    st.markdown("---")
    st.subheader("🎯 Estrategia Operativa Maestra (Mesa de Dinero)")
    
    # Pre-cálculos para la estrategia
    snap = get_snapshot(sel_ticker)
    ml_res = oraculo_ml(sel_ticker)
    mc_res = simulacion_monte_carlo(sel_ticker)
    fund_res = analisis_fundamental_y_noticias(sel_ticker) 
    dcf_val = calcular_dcf_rapido(sel_ticker)
    
    # Session State para guardar informe
    if 'informe_maestro' not in st.session_state: st.session_state['informe_maestro'] = None
    
    # Botón Maestro
    if st.button("⚡ GENERAR INFORME ESTRATÉGICO COMPLETO (GEMINI PRO)"):
        if snap and macro:
            with st.spinner("El Consultor PRO está redactando la tesis de inversión... (Puede demorar unos segundos)"):
                estrategia = generar_estrategia_profesional(sel_ticker, snap, macro, fund_res, mc_res, ml_res, dcf_val)
                st.session_state['informe_maestro'] = estrategia
        else: st.error("Datos insuficientes para generar estrategia.")
        
    if st.session_state['informe_maestro']:
        st.markdown(f"<div class='strat-box'>{st.session_state['informe_maestro']}</div>", unsafe_allow_html=True)
        
        # Botón PDF
        if st.button("📄 DESCARGAR INFORME EN PDF"):
            b64 = base64.b64encode(generar_pdf_profesional(sel_ticker, st.session_state['informe_maestro'])).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="Informe_Estrategico_{sel_ticker}.pdf" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: block; text-align: center; margin-top: 10px;">📥 DESCARGAR PDF</a>'
            st.markdown(href, unsafe_allow_html=True)

    # 3. RECURSOS DETALLADOS
    st.markdown("---")
    subtabs = st.tabs(["📊 Fundamentales & Noticias", "🤖 Oráculo ML", "🔮 Monte Carlo", "🦈 Insider"])
    
    with subtabs[0]: # Fundamental + Noticias
        if fund_res:
            c1, c2, c3 = st.columns(3)
            color_q = "green" if fund_res['Score_Total'] > 60 else "red"
            c1.markdown(f"#### Score Calidad: :{color_q}[{fund_res['Score_Total']}/100]")
            c1.caption(f"Diagnóstico: {fund_res['Calidad']}")
            c2.metric("Margen Neto", f"{fund_res['Ratios']['Margen_Neto']:.2f}%")
            c3.metric("Deuda/Patrimonio", f"{fund_res['Ratios']['Deuda_Eq']:.2f}")
            
            st.markdown("#### 🗞️ Análisis de Noticias (Impacto)")
            for noticia in fund_res['Noticias_Resumen']:
                st.write(f"• {noticia}")
        else: st.warning("Datos fundamentales limitados para este activo.")
        
    with subtabs[1]: # ML
        if ml_res:
            c1, c2 = st.columns(2)
            c1.metric("Predicción IA", ml_res['Pred'])
            c2.metric("Confianza", f"{ml_res['Acc']:.1f}%")
            
    with subtabs[2]: # Monte Carlo
        if mc_res:
            st.metric("Probabilidad Suba (30d)", f"{mc_res['Prob_Suba']:.1f}%")
            fig_mc = go.Figure()
            for i in range(20): fig_mc.add_trace(go.Scatter(x=mc_res['Dates'], y=mc_res['Paths'][:, i], mode='lines', line=dict(color='gray', width=1), opacity=0.3, showlegend=False))
            fig_mc.add_trace(go.Scatter(x=mc_res['Dates'], y=np.mean(mc_res['Paths'], axis=1), mode='lines', name='Promedio', line=dict(color='yellow', width=3)))
            st.plotly_chart(fig_mc, use_container_width=True)
            
    with subtabs[3]: # Insider
        insider = obtener_datos_insider(sel_ticker)
        if insider:
            c1, c2 = st.columns(2); c1.metric("Institucional", f"{insider['Institucional']:.1f}%"); c2.metric("Shorts", f"{insider['Short_Float']:.2f}%")

# --- TAB 3: OPCIONES ---
with tabs[2]:
    st.subheader("♟️ Laboratorio Derivados")
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
    st.subheader("🧠 Diario Emocional")
    conn = sqlite3.connect(DB_NAME)
    try:
        df_diario = pd.read_sql_query("SELECT * FROM trades ORDER BY fecha DESC", conn)
        if not df_diario.empty: st.dataframe(df_diario)
        else: st.info("Sin registros.")
    except: pass
    conn.close()
