import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots 
import requests 
import google.generativeai as genai
import warnings
import numpy as np
import os
import toml
import sqlite3
from datetime import datetime
from scipy.signal import argrelextrema 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import minimize 

# --- CONFIGURACIÓN MOTOR HÍBRIDO ---
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    HAVE_PYPFOPT = True
    ENGINE_STATUS = "🚀 PyPortfolioOpt (Pro)"
except ImportError:
    HAVE_PYPFOPT = False
    ENGINE_STATUS = "🛠️ Scipy Native (Backup)"

warnings.filterwarnings('ignore')

# --- 🔐 CREDENCIALES ---
try:
    secrets_path = ".streamlit/secrets.toml"
    if os.path.exists(secrets_path):
        secrets = toml.load(secrets_path)
        TELEGRAM_TOKEN = secrets["TELEGRAM_TOKEN"]
        TELEGRAM_CHAT_ID = secrets["TELEGRAM_CHAT_ID"]
        GOOGLE_API_KEY = secrets["GOOGLE_API_KEY"]
    else:
        TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
        TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except: st.stop()

# --- CONFIGURACIÓN PÁGINA ---
st.set_page_config(page_title="Sistema Quant V41 (Oracle)", layout="wide", page_icon="🔮")
st.markdown("""
<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 10px; color: white;}
    .signal-box {border: 2px solid #FFD700; padding: 10px; border-radius: 5px; background-color: #2b2b00; text-align: center;}
</style>
""", unsafe_allow_html=True)

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'COIN', 'KO', 'DIS', 'SPY', 'QQQ', 'DIA', 'GLD', 'USO']
DB_NAME = "quant_database.db"

# --- MOTOR SQL ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, ticker TEXT, tipo TEXT, cantidad INTEGER, precio REAL, total REAL)''')
    conn.commit()
    conn.close()

def registrar_operacion_sql(ticker, tipo, cantidad, precio):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S"); total = cantidad * precio
    c.execute("INSERT INTO trades (fecha, ticker, tipo, cantidad, precio, total) VALUES (?, ?, ?, ?, ?, ?)", (fecha, ticker, tipo, cantidad, precio, total))
    conn.commit(); conn.close()
    return True

def obtener_historial_sql():
    conn = sqlite3.connect(DB_NAME)
    try: df = pd.read_sql_query("SELECT * FROM trades ORDER BY fecha DESC", conn)
    except: df = pd.DataFrame()
    conn.close(); return df

def auditar_posiciones_sql():
    conn = sqlite3.connect(DB_NAME)
    try: df = pd.read_sql_query("SELECT * FROM trades", conn)
    except: return pd.DataFrame()
    conn.close()
    if df.empty: return pd.DataFrame()
    pos = {}
    for idx, row in df.iterrows():
        t = row['ticker']
        if t not in pos: pos[t] = {"Cantidad": 0, "Costo_Total": 0}
        if row['tipo'] == "COMPRA": pos[t]["Cantidad"] += row['cantidad']; pos[t]["Costo_Total"] += row['total']
        elif row['tipo'] == "VENTA":
            pos[t]["Cantidad"] -= row['cantidad']
            if pos[t]["Cantidad"] > 0: unit = pos[t]["Costo_Total"]/(pos[t]["Cantidad"]+row['cantidad']); pos[t]["Costo_Total"] -= (unit*row['cantidad'])
            else: pos[t]["Costo_Total"] = 0
    
    res = []; act = [t for t, d in pos.items() if d['Cantidad'] > 0]
    if not act: return pd.DataFrame()
    try: curr = yf.download(" ".join(act), period="1d", progress=False, auto_adjust=True)['Close']
    except: return pd.DataFrame()

    for t, d in pos.items():
        if d['Cantidad'] > 0:
            try:
                if len(act) == 1: price = float(curr.iloc[-1])
                else: price = float(curr.iloc[-1][t])
                val = d['Cantidad']*price; pnl = val - d['Costo_Total']
                res.append({"Ticker": t, "Cantidad": d['Cantidad'], "Valor Mercado": val, "P&L ($)": pnl, "P&L (%)": (pnl/d['Costo_Total'])*100})
            except: pass
    return pd.DataFrame(res)

init_db()

# --- MOTOR MONTECARLO (NUEVO V41) ---
def simular_montecarlo(ticker, dias_proyeccion=30, simulaciones=500):
    try:
        # 1. Obtener datos históricos para calcular volatilidad real
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)['Close']
        if df.empty: return None, None
        
        # Calcular retornos logarítmicos
        log_returns = np.log(1 + df.pct_change())
        u = log_returns.mean()
        var = log_returns.var()
        drift = u - (0.5 * var)
        stdev = log_returns.std()
        
        # 2. Configurar Simulación
        # Convertimos Series a valores simples para evitar conflictos de arrays
        drift = drift.item() if isinstance(drift, pd.Series) else drift
        stdev = stdev.item() if isinstance(stdev, pd.Series) else stdev
        last_price = df.iloc[-1].item()
        
        daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(dias_proyeccion, simulaciones)))
        
        # 3. Proyectar Precios
        price_list = np.zeros_like(daily_returns)
        price_list[0] = last_price
        
        for t in range(1, dias_proyeccion):
            price_list[t] = price_list[t - 1] * daily_returns[t]
            
        # 4. Crear Gráfico "Espagueti"
        fig = go.Figure()
        # Solo dibujamos las primeras 50 líneas para no saturar el navegador
        for i in range(min(50, simulaciones)):
            fig.add_trace(go.Scatter(y=price_list[:, i], mode='lines', line=dict(width=1, color='rgba(0, 255, 255, 0.1)'), showlegend=False))
            
        # Línea Promedio (Esperanza Matemática)
        mean_prices = np.mean(price_list, axis=1)
        fig.add_trace(go.Scatter(y=mean_prices, mode='lines', line=dict(width=3, color='yellow'), name='Promedio Esperado'))
        
        # Intervalos de Confianza
        p95 = np.percentile(price_list, 95, axis=1)
        p05 = np.percentile(price_list, 5, axis=1)
        
        fig.add_trace(go.Scatter(y=p95, mode='lines', line=dict(width=1, color='green', dash='dash'), name='Escenario Optimista (95%)'))
        fig.add_trace(go.Scatter(y=p05, mode='lines', line=dict(width=1, color='red', dash='dash'), name='Escenario Pesimista (5%)'))
        
        fig.update_layout(title=f"Simulación Montecarlo: {ticker} ({dias_proyeccion} días)", 
                          template="plotly_dark", xaxis_title="Días Futuros", yaxis_title="Precio Proyectado")
        
        resultados = {
            "ultimo_precio": last_price,
            "esperado": mean_prices[-1],
            "optimista": p95[-1],
            "pesimista": p05[-1]
        }
        
        return fig, resultados

    except Exception as e: return None, str(e)

# --- REQUERIMIENTOS EXTRA PARA MONTECARLO ---
from scipy.stats import norm 

# --- MOTORES EXISTENTES (Heatmap, Risk, etc) ---
@st.cache_data(ttl=600)
def generar_mapa_calor(tickers):
    try:
        data = yf.download(" ".join(tickers), period="5d", interval="1d", progress=False, auto_adjust=True)['Close']
        if data.empty: return None
        pct_change = ((data.iloc[-1] - data.iloc[-2]) / data.iloc[-2]) * 100
        df_map = pd.DataFrame({'Ticker': pct_change.index, 'Variacion': pct_change.values, 'Precio': data.iloc[-1].values})
        sectores = []
        for t in df_map['Ticker']:
            if t in ['NVDA', 'AMD', 'TSLA', 'AAPL', 'MSFT', 'META', 'GOOGL']: sectores.append('Tecnología')
            elif t in ['BTC-USD', 'ETH-USD', 'COIN']: sectores.append('Cripto')
            elif t in ['SPY', 'QQQ', 'DIA']: sectores.append('Índices')
            elif t in ['GLD', 'USO']: sectores.append('Commodities')
            else: sectores.append('Otros')
        df_map['Sector'] = sectores; df_map['Size'] = df_map['Precio'] 
        return df_map
    except: return None

def calcular_riesgo_portafolio(df_pos):
    if df_pos.empty: return None, None, None
    tk = df_pos['Ticker'].tolist(); w = (df_pos['Valor Mercado']/df_pos['Valor Mercado'].sum()).values
    try:
        d = yf.download(" ".join(tk), period="1y", progress=False, auto_adjust=True)['Close']
        if len(tk)==1: d = d.to_frame(name=tk[0])
        ret = np.log(d/d.shift(1)).dropna(); cov = ret.cov()*252; vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        var = df_pos['Valor Mercado'].sum() * (vol/np.sqrt(252)) * 1.65
        return var, vol, ret.corr()
    except: return None, None, None

@st.cache_data(ttl=300)
def escanear_oportunidades(tickers):
    señales = []
    try: data = yf.download(" ".join(tickers), period="3mo", interval="1d", group_by='ticker', progress=False, auto_adjust=True)
    except: return pd.DataFrame()
    for t in tickers:
        try:
            if len(tickers) > 1:
                try: df = data[t].copy()
                except: continue
            else: df = data.copy()
            df = df.dropna()
            if len(df) < 20: continue
            if 'Close' not in df.columns and df.shape[1] >= 4: df.columns = ['Open', 'High', 'Low', 'Close', 'Volume'][:df.shape[1]]
            last_close = df['Close'].iloc[-1]; rsi = ta.rsi(df['Close'], 14).iloc[-1]
            bb = ta.bbands(df['Close'], length=20, std=2); lower = bb.iloc[-1, 0]; upper = bb.iloc[-1, 2]
            tipo = "NEUTRAL"; fuerza = 0
            if last_close < lower: tipo = "COMPRA BOL 🟢"; fuerza = 90
            elif last_close > upper: tipo = "VENTA BOL 🔴"; fuerza = 80
            elif rsi < 30: tipo = "COMPRA RSI 🟢"; fuerza += 70
            elif rsi > 70: tipo = "VENTA RSI 🔴"; fuerza += 70
            if tipo != "NEUTRAL": señales.append({"Ticker": t, "Señal": tipo, "Fuerza": fuerza})
        except: pass
    return pd.DataFrame(señales)

def graficar_master(ticker):
    try:
        stock = yf.Ticker(ticker); df = stock.history(period="1y", interval="1d", auto_adjust=True)
        if df.empty: return None
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df['EMA20'] = ta.ema(df['Close'], 20); df['RSI'] = ta.rsi(df['Close'], 14)
        bb = ta.bbands(df['Close'], length=20, std=2); df = pd.concat([df, bb], axis=1)
        highs = df['High'].values; lows = df['Low'].values
        max_idx = argrelextrema(highs, np.greater, order=10)[0]; min_idx = argrelextrema(lows, np.less, order=10)[0]
        res = sorted([r for r in highs[max_idx] if 0 < (r - df['Close'].iloc[-1])/df['Close'].iloc[-1] < 0.15])[:2]
        sop = sorted([s for s in lows[min_idx] if 0 < (df['Close'].iloc[-1] - s)/df['Close'].iloc[-1] < 0.15])[-2:]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        if 'EMA20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='yellow', width=1), name="EMA 20"), row=1, col=1)
        try: fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, -3], line=dict(color='cyan', width=1, dash='dot'), name="Upper"), row=1, col=1); fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, -1], line=dict(color='cyan', width=1, dash='dot'), name="Lower"), row=1, col=1)
        except: pass
        for s in sop: fig.add_hline(y=s, line_dash="dot", line_color="green", row=1, col=1)
        for r in res: fig.add_hline(y=r, line_dash="dot", line_color="red", row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1); fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
        return fig
    except: return None

def predecir_precio_ia(ticker):
    try:
        stock = yf.Ticker(ticker); df = stock.history(period="2y", interval="1d", auto_adjust=True)
        if df.empty: return None
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df['RSI'] = ta.rsi(df['Close'], 14); df['EMA20'] = ta.ema(df['Close'], 20)
        df['Return'] = df['Close'].pct_change(); df['Volatilidad'] = df['Return'].rolling(5).std()
        df['Lag_Close_1'] = df['Close'].shift(1); df['Lag_RSI'] = df['RSI'].shift(1)
        df.dropna(inplace=True)
        if len(df) < 20: return None
        X = df[['Lag_Close_1', 'Lag_RSI', 'EMA20', 'Volatilidad']]; y = df['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model_ml = LinearRegression(); model_ml.fit(X_train, y_train)
        preds = model_ml.predict(X_test); score = r2_score(y_test, preds) * 100
        last_row = df.iloc[-1]
        last_data = pd.DataFrame([[last_row['Close'], last_row['RSI'], last_row['EMA20'], last_row['Volatilidad']]], columns=['Lag_Close_1', 'Lag_RSI', 'EMA20', 'Volatilidad'])
        return model_ml.predict(last_data)[0], score, last_row['Close']
    except: return None

def ejecutar_backtest_pro(ticker, capital, estrategia, params):
    try:
        stock = yf.Ticker(ticker); df = stock.history(period="3y", interval="1d", auto_adjust=True)
        if df.empty: return None
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        cash = capital; position = 0; trade_log = []; equity_curve = []; peak = capital; dd = 0
        if estrategia == "Bollinger (600% Mode)":
            bb = ta.bbands(df['Close'], length=20, std=2); df = pd.concat([df, bb], axis=1)
            buy_cond = lambda i: df['Close'].iloc[i] < df.iloc[i, -3]; sell_cond = lambda i: df['Close'].iloc[i] > df.iloc[i, -1]; start = 25
        elif estrategia == "RSI":
            df['RSI'] = ta.rsi(df['Close'], 14); buy_cond = lambda i: df['RSI'].iloc[i] < 30; sell_cond = lambda i: df['RSI'].iloc[i] > 70; start = 20
        for i in range(start, len(df)):
            price = df['Close'].iloc[i]; date = df.index[i]
            if cash > 0 and buy_cond(i):
                qty = int(cash/price)
                if qty > 0: cash -= qty*price; position += qty; trade_log.append({"Fecha": date, "Tipo": "COMPRA", "Precio": price, "Saldo": cash})
            elif position > 0 and sell_cond(i):
                cash += position*price; trade_log.append({"Fecha": date, "Tipo": "VENTA", "Precio": price, "Saldo": cash}); position = 0
            val = cash + (position*price); equity_curve.append({"Fecha": date, "Equity": val})
            if val > peak: peak = val
            current_dd = (peak - val) / peak; 
            if current_dd > dd: dd = current_dd
        final = cash + (position * df['Close'].iloc[-1]); ret = ((final - capital)/capital)*100
        bh = ((df['Close'].iloc[-1] - df['Close'].iloc[start]) / df['Close'].iloc[start])*100
        return {"retorno": ret, "buy_hold": bh, "trades": len(trade_log), "max_drawdown": dd*100, "log": pd.DataFrame(trade_log), "equity_curve": pd.DataFrame(equity_curve).set_index("Fecha")}
    except: return None

# --- INTERFAZ ---
st.title("🔮 Sistema Quant V41: The Oracle")

# 1. MAPA DE CALOR
with st.expander("🔥 Mapa de Calor del Mercado", expanded=False):
    df_mapa = generar_mapa_calor(WATCHLIST)
    if df_mapa is not None:
        fig_map = px.treemap(df_mapa, path=['Sector', 'Ticker'], values='Size', color='Variacion', color_continuous_scale='RdYlGn', color_continuous_midpoint=0, custom_data=['Variacion', 'Precio'], title="Mercado Hoy")
        fig_map.update_traces(texttemplate="%{label}<br>%{customdata[0]:.2f}%", textposition="middle center")
        fig_map.update_layout(height=350, margin=dict(t=30, l=10, r=10, b=10))
        st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# 2. GESTIÓN Y RIESGO
col_p1, col_p2 = st.columns([2, 1])
with col_p1:
    st.subheader("🏦 Mi Portafolio")
    df_pos = auditar_posiciones_sql()
    if not df_pos.empty:
        var95, vol_anual, corr_matrix = calcular_riesgo_portafolio(df_pos)
        total_eq = df_pos['Valor Mercado'].sum(); total_pnl = df_pos['P&L ($)'].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Patrimonio", f"${total_eq:,.2f}"); c2.metric("P&L Total", f"${total_pnl:+.2f}", delta_color="normal")
        if var95: c3.metric("VaR 95%", f"${var95:.2f}", delta_color="inverse")
        def color_pnl(val): return f'color: {"#00ff00" if val > 0 else "#ff0000"}' if isinstance(val, (int, float)) else ''
        st.dataframe(df_pos.style.map(color_pnl, subset=['P&L ($)']).format({"Precio Prom.": "${:.2f}", "Precio Actual": "${:.2f}", "Valor Mercado": "${:.2f}", "P&L ($)": "${:+.2f}", "P&L (%)": "{:+.2f}%"}), use_container_width=True)
    else: st.info("Cartera vacía.")

with col_p2:
    with st.expander("📝 Operar", expanded=True):
        t_op = st.selectbox("Activo", WATCHLIST)
        tipo = st.selectbox("Orden", ["COMPRA", "VENTA"])
        qty = st.number_input("Títulos", 1, 10000, 10)
        try: p_ref = float(yf.Ticker(t_op).history(period='1d')['Close'].iloc[-1])
        except: p_ref = 0.0
        price = st.number_input("Precio", 0.0, 100000.0, p_ref)
        if st.button("Ejecutar (SQL)"):
            registrar_operacion_sql(t_op, tipo, qty, price); st.success("Guardado!"); st.rerun()

st.divider()

# 3. ANALISIS, ESCANER Y PESTAÑAS
st.markdown("### 🔭 Laboratorio Financiero")
df_s = escanear_oportunidades(WATCHLIST)
if not df_s.empty:
    cols = st.columns(len(df_s))
    for idx, row in df_s.iterrows():
        with st.container(): st.markdown(f"<div class='signal-box'><h3>{row['Ticker']}</h3><p>{row['Señal']}</p></div>", unsafe_allow_html=True)

c_l, c_r = st.columns([1, 2.5])
with c_l:
    tk = st.selectbox("Analizar:", WATCHLIST)
    cap = st.number_input("Simulación ($)", 2000, 100000, 10000, key='cap_sim')
with c_r:
    tabs = st.tabs(["🔮 Monte Carlo", "📈 Gráfico", "♟️ Backtest", "🧠 IA"])
    
    # PESTAÑA NUEVA: MONTE CARLO
    with tabs[0]:
        st.subheader(f"🔮 El Oráculo: {tk}")
        st.write("Simulación de 1,000 futuros posibles basados en volatilidad histórica.")
        
        c_mc1, c_mc2 = st.columns([1, 3])
        with c_mc1:
            dias_mc = st.slider("Días a proyectar", 10, 90, 30)
            if st.button("🎲 EJECUTAR SIMULACIÓN"):
                with st.spinner("Simulando futuros paralelos..."):
                    fig_mc, res_mc = simular_montecarlo(tk, dias_mc)
                    if fig_mc:
                        st.session_state['mc_fig'] = fig_mc
                        st.session_state['mc_res'] = res_mc
        
        with c_mc2:
            if 'mc_fig' in st.session_state:
                res = st.session_state['mc_res']
                k1, k2, k3 = st.columns(3)
                k1.metric("Precio Actual", f"${res['ultimo_precio']:.2f}")
                k2.metric("Esperado (Promedio)", f"${res['esperado']:.2f}", f"{((res['esperado']-res['ultimo_precio'])/res['ultimo_precio'])*100:.1f}%")
                k3.metric("Riesgo (5% Peor caso)", f"${res['pesimista']:.2f}", delta_color="inverse")
                
                st.plotly_chart(st.session_state['mc_fig'], use_container_width=True)
                st.caption(f"El gráfico muestra 50 de 500 caminos simulados. El área entre líneas verde y roja contiene el 90% de probabilidad.")

    with tabs[1]:
        fig = graficar_master(tk)
        if fig: st.plotly_chart(fig, use_container_width=True)
    with tabs[2]:
        strat = st.selectbox("Estrategia:", ["Bollinger (600% Mode)", "RSI"])
        if st.button("⏪ AUDITAR"):
            res = ejecutar_backtest_pro(tk, cap, strat, {})
            if res:
                c1, c2 = st.columns(2)
                c1.metric("Retorno", f"{res['retorno']:.1f}%")
                c2.metric("B&H", f"{res['buy_hold']:.1f}%")
                st.line_chart(res['equity_curve'])