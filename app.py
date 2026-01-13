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

# --- CONFIGURACIÓN ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sistema Quant V71 (The Prophet)", layout="wide", page_icon="🔮")

st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;}
    .thesis-card {background-color: #1a1a2e; border-left: 4px solid #7b2cbf; padding: 20px; border-radius: 8px;}
    .prediction-box {background-color: #112b1b; border: 1px solid #00ff00; padding: 15px; border-radius: 5px; text-align: center;}
    .risk-box {background-color: #3d0e0e; border: 1px solid #ff0000; padding: 15px; border-radius: 5px; text-align: center;}
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

# --- MOTOR MONTE CARLO (NUEVO V71) ---
def simulacion_monte_carlo(ticker, dias=30, simulaciones=100):
    """Genera 100 caminos futuros posibles basados en volatilidad histórica"""
    try:
        # 1. Datos Históricos
        data = yf.Ticker(ticker).history(period="1y")['Close']
        if data.empty: return None
        
        returns = data.pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()
        start_price = data.iloc[-1]
        
        # 2. Generar Caminos Aleatorios (Vectorizado para velocidad)
        # Formula: P_t = P_t-1 * exp((mu - 0.5*sigma^2) + sigma * Z)
        dt = 1
        sim_paths = np.zeros((dias, simulaciones))
        sim_paths[0] = start_price
        
        for t in range(1, dias):
            drift = (mu - 0.5 * sigma**2) * dt
            shock = sigma * np.sqrt(dt) * np.random.normal(0, 1, simulaciones)
            sim_paths[t] = sim_paths[t-1] * np.exp(drift + shock)
            
        # 3. Estadísticas Finales
        final_prices = sim_paths[-1]
        mean_price = np.mean(final_prices)
        prob_up = np.mean(final_prices > start_price) * 100
        var_95 = np.percentile(final_prices, 5) # Value at Risk (Peor caso 5%)
        
        # Fechas futuras
        last_date = data.index[-1]
        dates = [last_date + timedelta(days=i) for i in range(dias)]
        
        return {
            "Paths": sim_paths,
            "Dates": dates,
            "Mean_Price": mean_price,
            "Prob_Suba": prob_up,
            "VaR_95": var_95,
            "Start_Price": start_price,
            "Upside_Avg": ((mean_price - start_price)/start_price)*100
        }
    except Exception as e: return None

# --- MOTOR FUNDAMENTAL PREMIUM (NUEVO V71) ---
@st.cache_data(ttl=3600)
def obtener_fundamentales_premium(ticker):
    """Extrae Márgenes, Liquidez y Solvencia"""
    if "USD" in ticker: return None
    try:
        stock = yf.Ticker(ticker)
        income = stock.income_stmt.T.sort_index()
        balance = stock.balance_sheet.T.sort_index()
        
        if income.empty or balance.empty: return None
        
        # 1. Ratios de Márgenes (Evolución)
        fechas = income.index.strftime('%Y')
        gross_margin = (income['Gross Profit'] / income['Total Revenue']) * 100
        operating_margin = (income['Operating Income'] / income['Total Revenue']) * 100
        net_margin = (income['Net Income'] / income['Total Revenue']) * 100
        
        # 2. Ratios de Liquidez (Último Año)
        curr_assets = balance['Total Current Assets'].iloc[-1]
        curr_liab = balance['Total Current Liabilities'].iloc[-1]
        inventory = balance['Inventory'].iloc[-1] if 'Inventory' in balance else 0
        
        current_ratio = curr_assets / curr_liab
        quick_ratio = (curr_assets - inventory) / curr_liab
        
        # 3. Solvencia
        total_debt = balance['Total Debt'].iloc[-1] if 'Total Debt' in balance else 0
        equity = balance['Stockholders Equity'].iloc[-1]
        debt_to_equity = total_debt / equity
        
        return {
            "Fechas": fechas,
            "Margen_Bruto": gross_margin,
            "Margen_Operativo": operating_margin,
            "Margen_Neto": net_margin,
            "Current_Ratio": current_ratio,
            "Quick_Ratio": quick_ratio,
            "Debt_Equity": debt_to_equity
        }
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

# --- MOTORES RESTAURADOS ---
@st.cache_data(ttl=1800)
def obtener_datos_snapshot(ticker):
    try:
        stock = yf.Ticker(ticker); hist = stock.history(period="5d")
        if hist.empty: return None
        try: info = stock.info
        except: info = {}
        return {"Precio": hist['Close'].iloc[-1], "Previo": hist['Close'].iloc[-2], "RSI": ta.rsi(hist['Close'], 14).iloc[-1] if len(hist)>14 else 50, "Volumen": info.get('volume', 0), "Beta": info.get('beta', 1.0), "Target": info.get('targetMeanPrice', 0)}
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

@st.cache_data(ttl=600)
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

# --- INTERFAZ V71 ---
c1, c2 = st.columns([3, 1])
with c1: st.title("🔮 Quant Terminal V71: The Prophet")
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
    
    tabs_detail = st.tabs(["🔮 Predicción Monte Carlo", "📚 Estados Financieros Premium", "📝 Tesis IA"])
    
    # --- TAB 1: MONTE CARLO (NUEVO V71) ---
    with tabs_detail[0]:
        st.subheader("🔮 Simulación Estocástica (30 Días)")
        with st.spinner("Ejecutando 100 simulaciones..."):
            mc = simulacion_monte_carlo(sel_ticker)
            
            if mc:
                c_mc1, c_mc2 = st.columns(2)
                
                # Caja de Probabilidad
                color_prob = "green" if mc['Prob_Suba'] > 50 else "red"
                with c_mc1:
                    st.markdown(f"""
                    <div class='prediction-box'>
                        <h3>Probabilidad de Suba</h3>
                        <h1 style='color:{color_prob}'>{mc['Prob_Suba']:.1f}%</h1>
                        <p>Precio Promedio Esperado: ${mc['Mean_Price']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Caja de Riesgo (VaR)
                with c_mc2:
                    st.markdown(f"""
                    <div class='risk-box'>
                        <h3>Riesgo (VaR 95%)</h3>
                        <h1>${mc['VaR_95']:.2f}</h1>
                        <p>En el peor 5% de los casos, el precio cae hasta aquí.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Gráfico Espagueti
                fig_mc = go.Figure()
                # Dibujar las primeras 30 simulaciones para no saturar
                for i in range(30):
                    fig_mc.add_trace(go.Scatter(x=mc['Dates'], y=mc['Paths'][:, i], mode='lines', line=dict(color='gray', width=1), opacity=0.3, showlegend=False))
                
                # Dibujar Promedio
                fig_mc.add_trace(go.Scatter(x=mc['Dates'], y=np.mean(mc['Paths'], axis=1), mode='lines', name='Promedio', line=dict(color='yellow', width=3)))
                
                fig_mc.update_layout(title="Simulación de 30 escenarios posibles", template="plotly_dark", height=300)
                st.plotly_chart(fig_mc, use_container_width=True)
            else: st.warning("Datos insuficientes para simulación.")

    # --- TAB 2: FUNDAMENTALES PREMIUM (NUEVO V71) ---
    with tabs_detail[1]:
        st.subheader("📊 Salud Financiera Profunda")
        fund_prem = obtener_fundamentales_premium(sel_ticker)
        
        if fund_prem:
            # 1. Gráfico de Márgenes
            st.markdown("#### Evolución de Márgenes (%)")
            fig_marg = go.Figure()
            fig_marg.add_trace(go.Scatter(x=fund_prem['Fechas'], y=fund_prem['Margen_Bruto'], name='Bruto', line=dict(color='cyan')))
            fig_marg.add_trace(go.Scatter(x=fund_prem['Fechas'], y=fund_prem['Margen_Operativo'], name='Operativo', line=dict(color='orange')))
            fig_marg.add_trace(go.Scatter(x=fund_prem['Fechas'], y=fund_prem['Margen_Neto'], name='Neto', line=dict(color='green')))
            fig_marg.update_layout(height=250, template="plotly_dark")
            st.plotly_chart(fig_marg, use_container_width=True)
            
            # 2. Tabla de Ratios
            st.markdown("#### Ratios de Liquidez y Solvencia")
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Current Ratio", f"{fund_prem['Current_Ratio']:.2f}", help="> 1.5 es ideal")
            col_r2.metric("Quick Ratio", f"{fund_prem['Quick_Ratio']:.2f}", help="Liquidez ácida (sin inventario)")
            col_r3.metric("Deuda/Patrimonio", f"{fund_prem['Debt_Equity']:.2f}", help="< 1.0 es ideal")
            
        else: st.info("Solo disponible para Acciones (No ETFs/Cripto).")

    # --- TAB 3: TESIS IA MEJORADA ---
    with tabs_detail[2]:
        st.subheader("📝 Tesis de Inversión IA 2.0")
        if st.button("🤖 Generar Tesis con Monte Carlo"):
            with st.spinner("Analizando probabilidades y ratios..."):
                # Construir Prompt Rico
                prompt = f"""
                Actúa como un Analista Senior de Fondo de Cobertura. Analiza {sel_ticker}.
                DATOS TÉCNICOS: Precio ${snap['Precio']:.2f}, RSI {snap['RSI']}.
                SIMULACIÓN MONTE CARLO: Probabilidad de suba {mc['Prob_Suba'] if mc else 'N/A'}%, Precio esperado ${mc['Mean_Price'] if mc else 'N/A'}.
                FUNDAMENTALES: (Si aplica) Margen Neto reciente, Deuda.
                
                Escribe una tesis de inversión de 3 párrafos:
                1. Situación Técnica y de Mercado.
                2. Salud Financiera (Interpreta márgenes y deuda).
                3. Veredicto basado en probabilidades (Monte Carlo).
                """
                try: 
                    res = model.generate_content(prompt).text
                    st.markdown(f"<div class='thesis-card'>{res}</div>", unsafe_allow_html=True)
                except: st.error("Error de conexión con Gemini AI.")

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
    st.subheader("🏆 Ranking Mercado")
    if st.button("🔄 ESCANEAR"):
        st.dataframe(escanear_mercado_completo(WATCHLIST), use_container_width=True)
        
    st.markdown("---")
    st.subheader("💼 Cartera")
    df_p = auditar_posiciones_sql()
    if not df_p.empty: st.dataframe(df_p[['Ticker', 'P&L']], use_container_width=True)
