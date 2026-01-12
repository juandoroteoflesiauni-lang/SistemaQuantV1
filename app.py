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
import feedparser 
import requests 
from datetime import datetime
from scipy.optimize import minimize 
from scipy.stats import norm 
import google.generativeai as genai

# --- CONFIGURACIÓN ---
warnings.filterwarnings('ignore')

try:
    secrets_path = ".streamlit/secrets.toml"
    if os.path.exists(secrets_path):
        secrets = toml.load(secrets_path)
        GOOGLE_API_KEY = secrets["GOOGLE_API_KEY"]
    else:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

st.set_page_config(page_title="Sistema Quant V61 (The Optimizer)", layout="wide", page_icon="📉")
st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 15px; text-align: center;}
    .opt-card {background-color: #112b1b; border: 1px solid #00cc96; padding: 15px; border-radius: 8px;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #0e1117; border-radius: 5px;}
    .stTabs [aria-selected="true"] {background-color: #262730;}
</style>""", unsafe_allow_html=True)

WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'COIN', 'KO', 'DIS', 'SPY', 'QQQ', 'GLD', 'USO']
DB_NAME = "quant_database.db"

# --- MOTOR SQL ---
def init_db():
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, ticker TEXT, tipo TEXT, cantidad INTEGER, precio REAL, total REAL)''')
    conn.commit(); conn.close()

def registrar_operacion_sql(ticker, tipo, cantidad, precio):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S"); total = cantidad * precio
    c.execute("INSERT INTO trades (fecha, ticker, tipo, cantidad, precio, total) VALUES (?, ?, ?, ?, ?, ?)", (fecha, ticker, tipo, cantidad, precio, total))
    conn.commit(); conn.close(); return True

def auditar_posiciones_sql():
    empty_df = pd.DataFrame(columns=["Ticker", "Cantidad", "Precio Prom.", "Precio Actual", "Valor Mercado", "P&L ($)", "P&L (%)"])
    conn = sqlite3.connect(DB_NAME)
    try: df = pd.read_sql_query("SELECT * FROM trades", conn)
    except: return empty_df
    conn.close()
    if df.empty: return empty_df
    pos = {}
    for idx, row in df.iterrows():
        t = row['ticker']
        if t not in pos: pos[t] = {"Cantidad": 0, "Costo_Total": 0}
        if row['tipo'] == "COMPRA": pos[t]["Cantidad"] += row['cantidad']; pos[t]["Costo_Total"] += row['total']
        elif row['tipo'] == "VENTA":
            pos[t]["Cantidad"] -= row['cantidad']
            if pos[t]["Cantidad"] > 0: unit = pos[t]["Costo_Total"]/(pos[t]["Cantidad"]+row['cantidad']); pos[t]["Costo_Total"] -= (unit*row['cantidad'])
            else: pos[t]["Costo_Total"] = 0
    act = [t for t, d in pos.items() if d['Cantidad'] > 0]
    if not act: return empty_df
    try: curr = yf.download(" ".join(act), period="1d", progress=False, auto_adjust=True)['Close']
    except: return empty_df
    res = []
    for t, d in pos.items():
        if d['Cantidad'] > 0:
            try:
                if len(act) == 1: price = float(curr.iloc[-1])
                else: price = float(curr.iloc[-1][t])
                val = d['Cantidad']*price; pnl = val - d['Costo_Total']
                res.append({"Ticker": t, "Cantidad": d['Cantidad'], "Precio Prom.": d['Costo_Total']/d['Cantidad'], "Precio Actual": price, "Valor Mercado": val, "P&L ($)": pnl, "P&L (%)": (pnl/d['Costo_Total'])*100})
            except: pass
    return pd.DataFrame(res) if res else empty_df

init_db()

# --- MOTOR MARKOWITZ (NUEVO V61) ---
def simular_frontera_eficiente(tickers, num_simulaciones=2000):
    """Genera miles de portafolios aleatorios para encontrar la frontera eficiente"""
    try:
        # 1. Datos
        data = yf.download(" ".join(tickers), period="1y", progress=False, auto_adjust=True)['Close']
        if data.empty: return None, None
        
        # Retornos logarítmicos
        log_ret = np.log(data / data.shift(1))
        
        # Matrices para guardar resultados
        all_weights = np.zeros((num_simulaciones, len(tickers)))
        ret_arr = np.zeros(num_simulaciones)
        vol_arr = np.zeros(num_simulaciones)
        sharpe_arr = np.zeros(num_simulaciones)
        
        # 2. Simulación Monte Carlo
        for ind in range(num_simulaciones):
            # Pesos aleatorios
            weights = np.array(np.random.random(len(tickers)))
            weights = weights / np.sum(weights) # Normalizar para que sumen 1
            all_weights[ind, :] = weights
            
            # Retorno Esperado
            ret_arr[ind] = np.sum(log_ret.mean() * weights) * 252
            
            # Volatilidad Esperada
            vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
            
            # Sharpe Ratio (Asumiendo tasa libre de riesgo = 0 para simplificar)
            sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]
            
        # 3. Encontrar el mejor portafolio (Max Sharpe)
        max_sharpe_idx = sharpe_arr.argmax()
        best_ret = ret_arr[max_sharpe_idx]
        best_vol = vol_arr[max_sharpe_idx]
        best_weights = all_weights[max_sharpe_idx, :]
        
        # DataFrame para graficar
        sim_data = pd.DataFrame({
            'Retorno': ret_arr, 
            'Volatilidad': vol_arr, 
            'Sharpe': sharpe_arr
        })
        
        # Diccionario de pesos óptimos
        best_weights_dict = {tickers[i]: best_weights[i] for i in range(len(tickers))}
        
        return sim_data, {
            "Max_Ret": best_ret,
            "Max_Vol": best_vol,
            "Max_Sharpe": sharpe_arr.max(),
            "Pesos": best_weights_dict
        }
        
    except Exception as e: return None, str(e)

# --- MOTORES EXISTENTES (FORENSIC, CRYPTO, ETC) ---
@st.cache_data(ttl=3600)
def realizar_auditoria_forense(ticker):
    if "USD" in ticker: return None
    try:
        info = yf.Ticker(ticker).info
        total_assets = info.get('totalAssets', 0); total_liab = info.get('totalDebt', 0)
        current_assets = info.get('totalCurrentAssets', 0); current_liab = info.get('totalCurrentLiabilities', 0)
        retained_earnings = info.get('retainedEarnings', total_assets * 0.1); ebit = info.get('ebitda', 0)
        total_revenue = info.get('totalRevenue', 0); market_cap = info.get('marketCap', 0)
        if total_assets == 0 or total_liab == 0: return None
        A = (current_assets - current_liab) / total_assets; B = retained_earnings / total_assets
        C = ebit / total_assets; D = market_cap / total_liab; E = total_revenue / total_assets
        Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
        z_status = "ZONA SEGURA" if Z > 3.0 else "ZONA QUIEBRA" if Z < 1.8 else "ALERTA"
        z_color = "audit-pass" if Z > 3.0 else "audit-fail" if Z < 1.8 else "audit-warn"
        return {"Z_Score": Z, "Z_Status": z_status, "Z_Color": z_color, "Deuda": info.get('debtToEquity', 0)}
    except: return None

def analizar_fundamental_crypto(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
        if df.empty: return None
        fair = df['Close'].rolling(200).mean().iloc[-1]; curr = df['Close'].iloc[-1]
        status = "SOBRECOMPRADO 🔴" if curr > fair * 1.5 else "ACUMULACIÓN 🟢" if curr < fair else "NEUTRAL 🟡"
        return {"Precio": curr, "FairValue": fair, "Status": status}
    except: return None

def graficar_master(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
        if df.empty: return None
        df['RSI'] = ta.rsi(df['Close'], 14); df['SMA200'] = ta.sma(df['Close'], 200)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], line=dict(color='yellow'), name="SMA200"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_color="red", row=2, col=1); fig.add_hline(y=30, line_color="green", row=2, col=1)
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False); return fig
    except: return None

@st.cache_data(ttl=600)
def generar_feed_alertas(tickers):
    alertas = []
    try: data = yf.download(" ".join(tickers), period="1y", group_by='ticker', progress=False, auto_adjust=True)
    except: return []
    for t in tickers:
        try:
            df = data[t].dropna() if len(tickers)>1 else data.dropna()
            if len(df)<200: continue
            close=df['Close']; s50=ta.sma(close,50); s200=ta.sma(close,200)
            if s50.iloc[-2]<s200.iloc[-2] and s50.iloc[-1]>s200.iloc[-1]: alertas.append({"Ticker":t,"Nivel":"ALTA","Mensaje":"🌟 GOLDEN CROSS"})
            if s50.iloc[-2]>s200.iloc[-2] and s50.iloc[-1]<s200.iloc[-1]: alertas.append({"Ticker":t,"Nivel":"ALTA","Mensaje":"☠️ DEATH CROSS"})
        except: pass
    return alertas

# --- INTERFAZ V61: THE OPTIMIZER ---
st.title("📉 Sistema Quant V61: The Optimizer")

with st.sidebar:
    st.header("🔔 Watchtower")
    alertas = generar_feed_alertas(WATCHLIST)
    if alertas:
        for a in alertas: st.markdown(f"<div class='alert-card-high'><b>{a['Ticker']}</b><br><small>{a['Mensaje']}</small></div>", unsafe_allow_html=True)
    else: st.success("Sin alertas críticas.")

df_pos = auditar_posiciones_sql()
k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Patrimonio", f"${df_pos['Valor Mercado'].sum() if not df_pos.empty else 0:,.2f}")
with k2: st.metric("P&L Total", f"${df_pos['P&L ($)'].sum() if not df_pos.empty else 0:+.2f}")
with k3: st.metric("SPY", f"${yf.Ticker('SPY').history(period='1d')['Close'].iloc[-1]:.2f}")
with k4: st.metric("Bitcoin", f"${yf.Ticker('BTC-USD').history(period='1d')['Close'].iloc[-1]:,.0f}")

st.divider()

main_tabs = st.tabs(["📉 OPTIMIZACIÓN (MARKOWITZ)", "🔍 AUDITORÍA & CRIPTO", "💼 MESA DE DINERO"])

# --- TAB 1: OPTIMIZACIÓN DE PORTAFOLIO (V61) ---
with main_tabs[0]:
    st.subheader("📉 Frontera Eficiente de Markowitz")
    st.info("Descubre la combinación matemática perfecta de activos para maximizar ganancias y minimizar riesgo.")
    
    col_opt1, col_opt2 = st.columns([1, 2])
    
    with col_opt1:
        # Selector de activos
        default_opts = ['NVDA', 'TSLA', 'AAPL', 'KO', 'GLD']
        opt_tickers = st.multiselect("Seleccionar Activos (Mínimo 3):", WATCHLIST, default=default_opts)
        
        if st.button("🚀 CALCULAR FRONTERA EFICIENTE"):
            if len(opt_tickers) < 3:
                st.error("Por favor selecciona al menos 3 activos para diversificar.")
            else:
                with st.spinner(f"Simulando 2,000 portafolios posibles con {', '.join(opt_tickers)}..."):
                    sim_data, best_port = simular_frontera_eficiente(opt_tickers)
                    
                    if sim_data is not None:
                        st.session_state['sim_data'] = sim_data
                        st.session_state['best_port'] = best_port
                    else:
                        st.error("Error al descargar datos.")

    with col_opt2:
        if 'sim_data' in st.session_state:
            sim_data = st.session_state['sim_data']
            best_port = st.session_state['best_port']
            
            # 1. Gráfico de Dispersión
            fig_frontier = px.scatter(sim_data, x="Volatilidad", y="Retorno", color="Sharpe",
                                      title="Frontera Eficiente (Cada punto es un portafolio)",
                                      color_continuous_scale="Viridis")
            
            # Marcar el mejor
            fig_frontier.add_trace(go.Scatter(x=[best_port['Max_Vol']], y=[best_port['Max_Ret']],
                                              mode='markers', marker=dict(color='red', size=15, symbol='star'),
                                              name="Portafolio Óptimo"))
            st.plotly_chart(fig_frontier, use_container_width=True)
            
            # 2. Pesos Óptimos
            st.markdown("### 🏆 Composición del Portafolio Óptimo (Max Sharpe)")
            
            # Mostrar como tabla y gráfico de torta
            df_pesos = pd.DataFrame(list(best_port['Pesos'].items()), columns=['Activo', 'Peso'])
            df_pesos['Peso %'] = (df_pesos['Peso'] * 100).round(2)
            
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                st.markdown(f"""
                <div class='opt-card'>
                    <h3>Rendimiento Esperado: {best_port['Max_Ret']*100:.1f}%</h3>
                    <h3>Riesgo (Volatilidad): {best_port['Max_Vol']*100:.1f}%</h3>
                    <h3>Ratio Sharpe: {best_port['Max_Sharpe']:.2f}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with c_p2:
                fig_pie = px.pie(df_pesos, values='Peso', names='Activo', title="Distribución Ideal de Capital")
                st.plotly_chart(fig_pie, use_container_width=True)

# --- TAB 2: AUDITORÍA & CRIPTO ---
with main_tabs[1]:
    sel_ticker = st.selectbox("Analizar:", WATCHLIST)
    ES_CRYPTO = "USD" in sel_ticker and "BTC" in sel_ticker or "ETH" in sel_ticker or "SOL" in sel_ticker
    
    c_a1, c_a2 = st.columns([1, 2])
    with c_a1:
        if ES_CRYPTO:
            st.markdown(f"### 🪙 Cripto: {sel_ticker}")
            cry_data = analizar_fundamental_crypto(sel_ticker)
            if cry_data:
                st.metric("Precio", f"${cry_data['Precio']:,.2f}")
                st.metric("Fair Value", f"${cry_data['FairValue']:,.2f}")
                st.info(cry_data['Status'])
        else:
            st.markdown(f"### 🔍 Auditoría: {sel_ticker}")
            audit = realizar_auditoria_forense(sel_ticker)
            if audit:
                st.markdown(f"**Altman Z-Score:** {audit['Z_Score']:.2f}")
                st.markdown(f"<div class='{audit['Z_Color']}'>{audit['Z_Status']}</div>", unsafe_allow_html=True)
                st.metric("Deuda/Patrimonio", f"{audit['Deuda']:.2f}")
            else: st.warning("Sin datos contables.")
    with c_a2:
        f = graficar_master(sel_ticker)
        if f: st.plotly_chart(f, use_container_width=True)

# --- TAB 3: OPERATIVA ---
with main_tabs[2]:
    if not df_pos.empty: st.dataframe(df_pos)
    else: st.info("Cartera vacía.")
    with st.form("op"):
        t = st.selectbox("Ticker", WATCHLIST, key="op_tk"); tp = st.selectbox("Tipo", ["COMPRA", "VENTA"])
        q = st.number_input("Qty", 1, 10000); p = st.number_input("Precio", 0.0)
        if st.form_submit_button("Ejecutar"): registrar_operacion_sql(t, tp, q, p); st.rerun()
