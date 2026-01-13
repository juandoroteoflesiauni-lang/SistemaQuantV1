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
import google.generativeai as genai
from fpdf import FPDF
import base64

# --- CONFIGURACIÓN ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sistema Quant V75 (The Consultant)", layout="wide", page_icon="💼")

st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;}
    .ai-report {background-color: #1e202e; border: 1px solid #4a4e69; padding: 25px; border-radius: 10px; font-family: 'Segoe UI', sans-serif; line-height: 1.6;}
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

# --- MOTOR DE INTELIGENCIA CONTEXTUAL (FIX V75) ---
def generar_analisis_ia_completo(ticker, snap, fund, mc, dcf, cons):
    """Genera un informe robusto manejando datos faltantes"""
    
    # 1. Pre-procesamiento seguro de datos (Evita el TypeError)
    precio_act = f"${snap['Precio']:.2f}"
    rsi_val = f"{snap['RSI']:.0f}"
    tendencia = 'Alcista' if snap['Precio'] > snap['Previo'] else 'Bajista'
    
    # Manejo seguro de Monte Carlo
    if mc:
        prob_suba = f"{mc['Prob_Suba']:.1f}%"
        precio_esp = f"${mc['Mean_Price']:.2f}"
        riesgo_var = f"${mc['VaR_95']:.2f}"
    else:
        prob_suba = "No disponible"
        precio_esp = "N/A"
        riesgo_var = "N/A"
        
    # Manejo seguro de Fundamentales (El error estaba aquí)
    if fund:
        try:
            margen_neto = f"{fund['Margen_Neto'].iloc[-1]:.2f}%"
            deuda = f"{fund['Debt']:.2f}"
        except:
            margen_neto = "N/A"
            deuda = "N/A"
    else:
        margen_neto = "No aplica (ETF/Cripto)"
        deuda = "N/A"
        
    # Manejo seguro de DCF y Consenso
    val_dcf = f"${dcf:.2f}" if dcf else "No calculable"
    target_analista = f"${cons['Target Mean']:.2f}" if cons else "Sin cobertura"
    recomendacion = cons['Recomendación'] if cons else "N/A"

    # 2. Construcción del Prompt Enriquecido
    prompt = f"""
    Eres un Asesor Financiero Senior y Contador Público Experto. Estás redactando un informe privado para un cliente inversor sobre el activo {ticker}.
    
    Usa los siguientes datos auditados por tu equipo de análisis cuantitativo:
    
    DATOS DE MERCADO:
    - Precio Actual: {precio_act} ({tendencia})
    - Fuerza Relativa (RSI): {rsi_val} (Nota: <30 es sobreventa, >70 sobrecompra)
    
    MODELOS MATEMÁTICOS:
    - Simulación Monte Carlo (30 días): Probabilidad de Suba del {prob_suba}. Precio Esperado: {precio_esp}.
    - Riesgo de Caída (VaR 95%): El precio podría caer hasta {riesgo_var} en el peor escenario.
    
    ANÁLISIS FUNDAMENTAL Y CONTABLE:
    - Margen Neto (Rentabilidad Real): {margen_neto}
    - Solvencia (Deuda/Patrimonio): {deuda} (Nota: >2.0 es alto apalancamiento)
    - Valor Intrínseco (DCF): {val_dcf}
    
    OPINIÓN WALL STREET:
    - Consenso: {recomendacion}
    - Precio Objetivo Promedio: {target_analista}
    
    ---
    TU TAREA:
    Escribe un informe estratégico, directo y profesional en formato Markdown. No saludes, ve al grano.
    Estructura:
    
    ### 1. 🏛️ Estado de Situación
    (Analiza la salud de la empresa basándote en los márgenes y la deuda. Si es Cripto, habla de la tendencia).
    
    ### 2. ⚖️ Cruce de Valuaciones
    (Compara el precio actual contra el Valor DCF y el Target de Analistas. ¿Está barata o cara?).
    
    ### 3. 🎲 Escenario Probabilístico
    (Usa los datos de Monte Carlo y RSI para definir si el momento técnico es favorable).
    
    ### 4. 🎯 CONCLUSIÓN OPERATIVA
    (Define una acción clara: COMPRA, ACUMULAR, MANTENER o VENDER/ESPERAR. Justifica con el dato más fuerte que tengas).
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error generando informe: {str(e)}"

# --- MOTOR PDF (V75) ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15); self.cell(0, 10, 'Informe Quant V75', 0, 1, 'C'); self.ln(5)
    def footer(self):
        self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

def clean_text(text):
    if not isinstance(text, str): return str(text)
    replacements = {"🟢": "[+]", "🔴": "[-]", "🟡": "[=]", "🚀": "(UP)", "💎": "(VAL)", "🛡️": "(SAFE)", "**": "", "###": "", "####": ""}
    for k, v in replacements.items(): text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')

def generar_pdf_profundo(ticker, precio, informe_ia, metricas):
    pdf = PDFReport(); pdf.add_page()
    pdf.set_font('Arial', 'B', 16); pdf.cell(0, 10, f'Analisis: {clean_text(ticker)}', 0, 1, 'L')
    pdf.set_font('Arial', '', 10); pdf.cell(0, 10, f'Precio: ${precio:.2f} | Fecha: {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'L'); pdf.line(10, 35, 200, 35)
    pdf.ln(10); pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, '1. Datos Clave', 0, 1); pdf.set_font('Arial', '', 10)
    pdf.cell(45, 10, f"RSI: {metricas.get('RSI', 'N/A')}", 1); pdf.cell(45, 10, f"Prob. Suba: {metricas.get('Prob_Suba', 'N/A')}", 1)
    pdf.cell(45, 10, f"Target: ${metricas.get('Target', 0):.2f}", 1); pdf.ln(15)
    pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, '2. Informe Estrategico', 0, 1); pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, clean_text(informe_ia))
    return pdf.output(dest='S').encode('latin-1')

# --- MOTORES DE SOPORTE (CACHEADOS) ---
@st.cache_data(ttl=1800)
def obtener_datos_snapshot(ticker):
    try:
        stock = yf.Ticker(ticker); hist = stock.history(period="5d")
        if hist.empty: return None
        try: info = stock.info
        except: info = {}
        return {"Precio": hist['Close'].iloc[-1], "Previo": hist['Close'].iloc[-2], "RSI": ta.rsi(hist['Close'], 14).iloc[-1] if len(hist)>14 else 50, "Volumen": info.get('volume', 0), "Beta": info.get('beta', 1.0), "Target": info.get('targetMeanPrice', 0)}
    except: return None

@st.cache_data(ttl=3600)
def obtener_fundamentales_premium(ticker):
    if "USD" in ticker: return None
    try:
        stock = yf.Ticker(ticker)
        inc = stock.income_stmt.T.sort_index(); bal = stock.balance_sheet.T.sort_index()
        if inc.empty or bal.empty: return None
        gm = (inc['Gross Profit']/inc['Total Revenue'])*100; nm = (inc['Net Income']/inc['Total Revenue'])*100
        cr = bal['Total Current Assets'].iloc[-1]/bal['Total Current Liabilities'].iloc[-1]
        de = bal.get('Total Debt', pd.Series(0)).iloc[-1]/bal['Stockholders Equity'].iloc[-1]
        return {"Fechas": inc.index.strftime('%Y'), "Margen_Bruto": gm, "Margen_Neto": nm, "Current": cr, "Debt": de}
    except: return None

def simulacion_monte_carlo(ticker, dias=30, simulaciones=100):
    try:
        data = yf.Ticker(ticker).history(period="1y")['Close']
        if data.empty: return None
        returns = data.pct_change().dropna()
        mu = returns.mean(); sigma = returns.std(); start_price = data.iloc[-1]
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

def obtener_consenso_analistas(ticker):
    if "USD" in ticker: return None
    try:
        info = yf.Ticker(ticker).info
        return {"Recomendación": info.get('recommendationKey', 'N/A').upper(), "Target Mean": info.get('targetMeanPrice', 0)}
    except: return None

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

# --- INTERFAZ V75 ---
c1, c2 = st.columns([3, 1])
with c1: st.title("💼 Quant Terminal V75: The Consultant")
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

fund = obtener_fundamentales_premium(sel_ticker)
mc = simulacion_monte_carlo(sel_ticker)
dcf = calcular_dcf_rapido(sel_ticker)
cons = obtener_consenso_analistas(sel_ticker)

with col_main:
    st.subheader("📉 Acción del Precio")
    fig_chart = graficar_simple(sel_ticker)
    if fig_chart: st.plotly_chart(fig_chart, use_container_width=True)
    
    tabs_detail = st.tabs(["📝 TESIS PROFUNDA (IA)", "🔮 Monte Carlo", "📚 Fundamentales"])
    
    with tabs_detail[0]:
        st.subheader("🧠 Consultor IA Premium")
        if st.button("⚡ GENERAR ANÁLISIS ESTRATÉGICO"):
            with st.spinner("El Consultor está analizando los datos..."):
                reporte = generar_analisis_ia_completo(sel_ticker, snap, fund, mc, dcf, cons)
                st.session_state['informe_ia_v75'] = reporte
        
        if 'informe_ia_v75' in st.session_state:
            st.markdown(f"<div class='ai-report'>{st.session_state['informe_ia_v75']}</div>", unsafe_allow_html=True)
            
            if st.button("📄 DESCARGAR REPORTE OFICIAL (PDF)"):
                try:
                    prob_val = f"{mc['Prob_Suba']:.1f}%" if mc else "N/A"
                    met_pdf = {"RSI": f"{snap['RSI']:.0f}", "Prob_Suba": prob_val, "Target": cons.get('Target Mean', 0) if cons else 0}
                    pdf_bytes = generar_pdf_profundo(sel_ticker, snap['Precio'], st.session_state['informe_ia_v75'], met_pdf)
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="Reporte_{sel_ticker}.pdf" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: block; text-align: center; margin-top: 10px;">📥 CLIC PARA DESCARGAR</a>'
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e: st.error(f"Error PDF: {e}")

    with tabs_detail[1]:
        if mc:
            c1, c2 = st.columns(2)
            c1.metric("Probabilidad Suba", f"{mc['Prob_Suba']:.1f}%")
            c2.metric("Riesgo VaR 95%", f"${mc['VaR_95']:.2f}")
            fig_mc = go.Figure()
            for i in range(20): fig_mc.add_trace(go.Scatter(x=mc['Dates'], y=mc['Paths'][:, i], mode='lines', line=dict(color='gray', width=1), opacity=0.3, showlegend=False))
            fig_mc.add_trace(go.Scatter(x=mc['Dates'], y=np.mean(mc['Paths'], axis=1), mode='lines', name='Promedio', line=dict(color='yellow', width=3)))
            fig_mc.update_layout(template="plotly_dark", height=250, margin=dict(l=0,r=0,t=0,b=0)); st.plotly_chart(fig_mc, use_container_width=True)

    with tabs_detail[2]:
        if fund:
            fig_m = go.Figure(); fig_m.add_trace(go.Scatter(x=fund['Fechas'], y=fund['Margen_Neto'], name='Margen Neto', line=dict(color='green')))
            fig_m.update_layout(height=200, template="plotly_dark", title="Margen Neto (%)", margin=dict(l=0,r=0,t=30,b=0)); st.plotly_chart(fig_m, use_container_width=True)
            c1, c2 = st.columns(2); c1.metric("Liquidez", f"{fund['Current']:.2f}"); c2.metric("Deuda", f"{fund['Debt']:.2f}")
        else: st.info("Datos no disponibles.")

with col_side:
    st.subheader("🧬 Perfil Quant")
    factores = calcular_factores_quant_single(sel_ticker)
    if factores: st.plotly_chart(dibujar_radar_factores(factores), use_container_width=True)
    st.markdown("---")
    st.subheader("⚡ Quick Trade")
    with st.form("quick"):
        q = st.number_input("Qty", 1, 1000, 10); s = st.selectbox("Side", ["COMPRA", "VENTA"])
        if st.form_submit_button("EJECUTAR"): 
            if snap: registrar_operacion_sql(sel_ticker, s, q, snap['Precio']); st.success("Orden OK")
    
    st.markdown("---")
    st.subheader("🏆 Ranking")
    if st.button("🔄 ESCANEAR"): st.dataframe(escanear_mercado_completo(WATCHLIST), use_container_width=True)
    
    st.markdown("---")
    st.subheader("💼 Cartera")
    df_p = auditar_posiciones_sql()
    if not df_p.empty: st.dataframe(df_p[['Ticker', 'P&L']], use_container_width=True)
