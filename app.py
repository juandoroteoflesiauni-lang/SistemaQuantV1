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
from fpdf import FPDF # Librería para PDF
import base64

# --- CONFIGURACIÓN ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sistema Quant V72 (The Publisher)", layout="wide", page_icon="🖨️")

st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;}
    .thesis-card {background-color: #1a1a2e; border-left: 4px solid #7b2cbf; padding: 20px; border-radius: 8px;}
    .pdf-btn {text-align: center; margin-top: 20px;}
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

# --- MOTOR DE REPORTES PDF (NUEVO V72) ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Informe de Inteligencia de Inversion - Sistema Quant', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

def generar_pdf_analisis(ticker, precio, tesis, metricas_clave, prediccion):
    """Crea un PDF con el resumen del análisis"""
    pdf = PDFReport()
    pdf.add_page()
    
    # 1. Título y Fecha
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f'Analisis de Activo: {ticker}', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'L')
    pdf.line(10, 30, 200, 30)
    
    # 2. Snapshot
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1. Datos de Mercado', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 10, f"Precio Actual: ${precio:.2f}", 0, 1)
    if metricas_clave:
        pdf.cell(0, 10, f"RSI (14): {metricas_clave.get('RSI', 'N/A')}", 0, 1)
        pdf.cell(0, 10, f"Beta: {metricas_clave.get('Beta', 'N/A')}", 0, 1)
    
    # 3. Tesis de Inversión
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '2. Tesis de Inversion (IA & Quant)', 0, 1)
    pdf.set_font('Arial', 'B', 14)
    # Color coding simulado con texto
    veredicto = tesis['Veredicto']
    pdf.set_text_color(0, 128, 0) if "COMPRA" in veredicto else pdf.set_text_color(255, 0, 0)
    pdf.cell(0, 10, f"VEREDICTO: {veredicto}", 0, 1)
    pdf.set_text_color(0, 0, 0) # Reset color
    
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, "Argumentos a Favor (Bullish):")
    for p in tesis['Pros']: pdf.cell(0, 5, f" + {p}", 0, 1)
    pdf.ln(2)
    pdf.multi_cell(0, 5, "Riesgos (Bearish):")
    for c in tesis['Contras']: pdf.cell(0, 5, f" - {c}", 0, 1)
    
    # 4. Proyección
    if prediccion:
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '3. Proyeccion Monte Carlo (30 dias)', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 10, f"Precio Objetivo Promedio: ${prediccion['Mean_Price']:.2f}", 0, 1)
        pdf.cell(0, 10, f"Probabilidad de Suba: {prediccion['Prob_Suba']:.1f}%", 0, 1)
        pdf.cell(0, 10, f"Riesgo (VaR 95%): ${prediccion['VaR_95']:.2f}", 0, 1)

    # Disclaimer
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 8)
    pdf.multi_cell(0, 5, "Nota: Este informe es generado automaticamente por un sistema cuantitativo con fines educativos. No constituye asesoramiento financiero profesional.")
    
    return pdf.output(dest='S').encode('latin-1')

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

# --- MOTORES DE ANÁLISIS ---
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
        return {"Paths": sim_paths, "Dates": [data.index[-1]+timedelta(days=i) for i in range(dias)], "Mean_Price": np.mean(final), "Prob_Suba": np.mean(final>start_price)*100, "VaR_95": np.percentile(final, 5), "Upside": ((np.mean(final)-start_price)/start_price)*100}
    except: return None

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

def generar_tesis_automatica(ticker, datos_tecnicos, datos_fundamentales, prediccion):
    puntos_bull = []; puntos_bear = []
    if datos_tecnicos['RSI'] < 30: puntos_bull.append("Sobreventa (RSI)")
    elif datos_tecnicos['RSI'] > 70: puntos_bear.append("Sobrecompra (RSI)")
    if prediccion and prediccion['Prob_Suba'] > 60: puntos_bull.append(f"Monte Carlo Alcista ({prediccion['Prob_Suba']:.0f}%)")
    if datos_fundamentales and datos_fundamentales.get('Current', 0) > 1.5: puntos_bull.append("Solvencia Sólida")
    score = len(puntos_bull) - len(puntos_bear)
    veredicto = "COMPRA FUERTE 🟢" if score >= 2 else "COMPRA 🟢" if score > 0 else "NEUTRAL 🟡"
    return {"Veredicto": veredicto, "Pros": puntos_bull, "Contras": puntos_bear}

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

# --- INTERFAZ V72 ---
c1, c2 = st.columns([3, 1])
with c1: st.title("🖨️ Quant Terminal V72: The Publisher")
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
    
    tabs_detail = st.tabs(["🔮 Monte Carlo", "📚 Fundamentales", "📝 Tesis & Reporte"])
    
    # Pre-calculos para tesis y PDF
    mc = simulacion_monte_carlo(sel_ticker)
    fund = obtener_fundamentales_premium(sel_ticker)
    
    with tabs_detail[0]:
        if mc:
            c_mc1, c_mc2 = st.columns(2)
            c_mc1.metric("Probabilidad Suba", f"{mc['Prob_Suba']:.1f}%")
            c_mc2.metric("Riesgo VaR 95%", f"${mc['VaR_95']:.2f}")
            fig_mc = go.Figure()
            for i in range(20):
                fig_mc.add_trace(go.Scatter(x=mc['Dates'], y=mc['Paths'][:, i], mode='lines', line=dict(color='gray', width=1), opacity=0.3, showlegend=False))
            fig_mc.add_trace(go.Scatter(x=mc['Dates'], y=np.mean(mc['Paths'], axis=1), mode='lines', name='Promedio', line=dict(color='yellow', width=3)))
            fig_mc.update_layout(template="plotly_dark", height=250, margin=dict(l=0,r=0,t=0,b=0)); st.plotly_chart(fig_mc, use_container_width=True)
        else: st.warning("Datos insuficientes.")

    with tabs_detail[1]:
        if fund:
            fig_marg = go.Figure()
            fig_marg.add_trace(go.Scatter(x=fund['Fechas'], y=fund['Margen_Neto'], name='Margen Neto', line=dict(color='green')))
            fig_marg.update_layout(height=250, template="plotly_dark", title="Margen Neto (%)"); st.plotly_chart(fig_marg, use_container_width=True)
            c_r1, c_r2 = st.columns(2)
            c_r1.metric("Liquidez", f"{fund['Current']:.2f}")
            c_r2.metric("Deuda/Patrimonio", f"{fund['Debt']:.2f}")
        else: st.info("Solo Acciones.")

    with tabs_detail[2]:
        st.subheader("📝 Generador de Informes")
        tesis = generar_tesis_automatica(sel_ticker, {"RSI": snap['RSI'], "Precio": snap['Precio'], "SMA200": 0}, fund, mc)
        
        st.markdown(f"<div class='thesis-card'><h3>{tesis['Veredicto']}</h3></div>", unsafe_allow_html=True)
        
        # BOTÓN GENERAR PDF
        if st.button("📄 DESCARGAR INFORME PDF"):
            with st.spinner("Maquetando documento..."):
                pdf_bytes = generar_pdf_analisis(sel_ticker, snap['Precio'], tesis, snap, mc)
                b64 = base64.b64encode(pdf_bytes).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="Informe_{sel_ticker}.pdf" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: block; text-align: center; margin-top: 20px;">📥 CLIC PARA DESCARGAR PDF</a>'
                st.markdown(href, unsafe_allow_html=True)

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