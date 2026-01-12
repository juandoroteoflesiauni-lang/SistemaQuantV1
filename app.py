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
from datetime import datetime
from scipy.signal import argrelextrema 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import minimize 
from scipy.stats import norm 
import google.generativeai as genai
from fpdf import FPDF 

# --- CONFIGURACIÓN E INICIOS ---
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    HAVE_PYPFOPT = True
except ImportError:
    HAVE_PYPFOPT = False

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

st.set_page_config(page_title="Sistema Quant V57 (Watchtower)", layout="wide", page_icon="🔔")
st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 15px; text-align: center;}
    .report-box {background-color: #1e1e1e; border-left: 5px solid #FFD700; padding: 20px; border-radius: 5px; margin-bottom: 20px;}
    .alert-card-high {background-color: #3d0e0e; border: 1px solid #ff4b4b; padding: 10px; border-radius: 5px; margin-bottom: 5px;}
    .alert-card-med {background-color: #3d3d0e; border: 1px solid #ffa500; padding: 10px; border-radius: 5px; margin-bottom: 5px;}
    .alert-card-low {background-color: #0e2b0e; border: 1px solid #00cc96; padding: 10px; border-radius: 5px; margin-bottom: 5px;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #0e1117; border-radius: 5px;}
    .stTabs [aria-selected="true"] {background-color: #262730;}
</style>""", unsafe_allow_html=True)

WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'COIN', 'KO', 'DIS', 'SPY', 'QQQ', 'DIA', 'GLD', 'USO']

MACRO_DICT = {
    'S&P 500': 'SPY', 'Nasdaq 100': 'QQQ', 'VIX (Miedo)': '^VIX',
    'Bonos 10Y': '^TNX', 'Dólar Index': 'DX-Y.NYB', 'Petróleo': 'CL=F', 'Oro': 'GC=F'
}
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

# --- MOTOR WATCHTOWER (NUEVO V57) ---
@st.cache_data(ttl=600)
def generar_feed_alertas(tickers):
    """Escanea Golden Cross, RSI Extremo y Ballenas"""
    alertas = []
    try:
        # Descarga masiva eficiente (1 año para medias móviles largas)
        data = yf.download(" ".join(tickers), period="1y", group_by='ticker', progress=False, auto_adjust=True)
    except: return []

    for t in tickers:
        try:
            if len(tickers) > 1: df = data[t].dropna()
            else: df = data.dropna()
            
            if len(df) < 200: continue # Necesitamos 200 días para Golden Cross
            
            # Cálculo de Indicadores
            close = df['Close']
            rsi = ta.rsi(close, 14).iloc[-1]
            sma50 = ta.sma(close, 50)
            sma200 = ta.sma(close, 200)
            
            curr_50 = sma50.iloc[-1]
            prev_50 = sma50.iloc[-2]
            curr_200 = sma200.iloc[-1]
            prev_200 = sma200.iloc[-2]
            
            # 1. GOLDEN CROSS (Cruce Dorado) - Muy Alcista
            if prev_50 < prev_200 and curr_50 > curr_200:
                alertas.append({"Ticker": t, "Nivel": "ALTA", "Mensaje": "🌟 GOLDEN CROSS: SMA50 cruzó SMA200 al alza. Tendencia Alcista Mayor."})
                
            # 2. DEATH CROSS (Cruce de la Muerte) - Muy Bajista
            if prev_50 > prev_200 and curr_50 < curr_200:
                alertas.append({"Ticker": t, "Nivel": "ALTA", "Mensaje": "☠️ DEATH CROSS: SMA50 cruzó SMA200 a la baja. Peligro."})
                
            # 3. RSI EXTREMO
            if rsi < 25:
                alertas.append({"Ticker": t, "Nivel": "MEDIA", "Mensaje": f"🟢 Sobreventa Extrema (RSI {rsi:.0f}). Posible rebote."})
            elif rsi > 75:
                alertas.append({"Ticker": t, "Nivel": "MEDIA", "Mensaje": f"🔴 Sobrecompra Extrema (RSI {rsi:.0f}). Riesgo de corrección."})
                
            # 4. BALLENAS (Volumen > 200% promedio)
            vol_hoy = df['Volume'].iloc[-1]
            vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
            if vol_hoy > vol_avg * 2.0:
                alertas.append({"Ticker": t, "Nivel": "BAJA", "Mensaje": f"🐋 Volumen Inusual ({vol_hoy/vol_avg:.1f}x). Actividad Institucional."})
                
        except: pass
        
    return alertas

# --- MOTORES EXISTENTES ---
def obtener_panorama_macro():
    resumen = {}
    for nombre, ticker in MACRO_DICT.items():
        try:
            data = yf.Ticker(ticker).history(period="5d")
            if not data.empty:
                precio = data['Close'].iloc[-1]; previo = data['Close'].iloc[-2]
                resumen[nombre] = {"Precio": precio, "Cambio%": ((precio - previo) / previo) * 100}
            else: resumen[nombre] = {"Precio": 0.0, "Cambio%": 0.0}
        except: resumen[nombre] = {"Precio": 0.0, "Cambio%": 0.0}
    return resumen

def generar_briefing_ia(datos_macro):
    try:
        rss = "https://news.google.com/rss/topics/CAAqJggBCiJCAQAqSVgQASowCAAqLAgKIiZDQW1TRWdrTWFnZ0tDaElVWjI5dlozbG5hVzV6ZEdGaWJDNXpLQUFQAQ?hl=en-US&gl=US&ceid=US%3Aen"
        feed = feedparser.parse(rss); titulares = [e.title for e in feed.entries[:5]] if feed.entries else []
        txt = "\n".join([f"{k}: {v['Precio']:.2f}" for k,v in datos_macro.items()])
        p = f"Escribe un 'Morning Briefing' financiero breve en español basado en:\nDATOS:{txt}\nNOTICIAS:{titulares}\nAnaliza sentimiento Risk-On/Risk-Off."
        return model.generate_content(p).text
    except Exception as e: return f"Error IA: {e}"

def simular_cartera_historica(tickers, pesos, periodo="1y", benchmark="SPY"):
    try:
        todos = tickers + [benchmark]
        d = yf.download(" ".join(todos), period=periodo, progress=False, auto_adjust=True)['Close']
        if d.empty: return None, None
        r = d.pct_change().dropna()
        br = (1+r[benchmark]).cumprod(); br=(br/br.iloc[0])*100
        rc = r[tickers].dot(list(pesos.values())); pr=(1+rc).cumprod(); pr=(pr/pr.iloc[0])*100
        c_p = ((pr.iloc[-1]/pr.iloc[0])**(252/len(pr))-1)*100
        vol = rc.std()*np.sqrt(252)*100
        return pd.DataFrame({"Cartera": pr, "SPY": br}), {"CAGR": c_p, "Vol": vol, "Sharpe": (c_p-4)/vol}
    except: return None, None

def calcular_matriz_correlacion(tickers):
    try:
        d = yf.download(" ".join(tickers), period="1y", progress=False, auto_adjust=True)['Close']
        return np.log(d/d.shift(1)).corr() if not d.empty else None
    except: return None

def detectar_actividad_ballenas(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1mo", interval="1d", auto_adjust=True)
        if df.empty: return None
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['VolSMA'] = df['Volume'].rolling(20).mean()
        v = df['Volume'].iloc[-1]; va = df['VolSMA'].iloc[-1]
        alerta = "🐋 ALERTA" if v > va * 1.5 else None
        trend = "ALCISTA" if df['Close'].iloc[-1] > df['VWAP'].iloc[-1] else "BAJISTA"
        return {"Volumen Hoy": v, "Ratio Vol": v/va, "Alerta": alerta, "VWAP": df['VWAP'].iloc[-1], "Tendencia": trend}
    except: return None

@st.cache_data(ttl=600)
def calcular_score_quant(ticker):
    score=0; b={"Técnico":0, "Fundamental":0, "Riesgo":0}
    try:
        h = yf.Ticker(ticker).history(period="6mo", interval="1d", auto_adjust=True)
        if not h.empty:
            h['RSI']=ta.rsi(h['Close'],14); h['SMA']=ta.sma(h['Close'],50)
            if 30<=h['RSI'].iloc[-1]<=65: b['Técnico']+=20
            elif h['RSI'].iloc[-1]<30: b['Técnico']+=15
            elif h['RSI'].iloc[-1]>70: b['Técnico']+=5
            if h['Close'].iloc[-1]>h['SMA'].iloc[-1]: b['Técnico']+=20
        i = yf.Ticker(ticker).info
        p = i.get('currentPrice',0); e = i.get('trailingEps',0); bk = i.get('bookValue',0)
        if e and bk and e>0 and bk>0:
            g = math.sqrt(22.5*e*bk)
            if p<g: b['Fundamental']+=20
            elif p<g*1.2: b['Fundamental']+=10
        if e>0: b['Fundamental']+=20
        if not e and "USD" in ticker: b['Fundamental']=20
        be = i.get('beta',1.0)
        if be is None: be=1.0
        if 0.8<=be<=1.2: b['Riesgo']+=20
        elif be<0.8: b['Riesgo']+=15
        else: b['Riesgo']+=10
        score = sum(b.values())
        return score, b
    except: return 0, b

def dibujar_velocimetro(score):
    return go.Figure(go.Indicator(mode="gauge+number", value=score, domain={'x': [0, 1], 'y': [0, 1]}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "white"}, 'steps': [{'range': [0, 40], 'color': "#ff4b4b"}, {'range': [40, 70], 'color': "#ffa500"}, {'range': [70, 100], 'color': "#00cc96"}]})).update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="#0e1117", font={'color': "white"})

def graficar_master(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
        if df.empty: return None
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df['EMA20'] = ta.ema(df['Close'], 20); df['RSI'] = ta.rsi(df['Close'], 14)
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        bb = ta.bbands(df['Close'], length=20, std=2); df = pd.concat([df, bb], axis=1)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#FFD700', width=2), name="VWAP"), row=1, col=1)
        if 'EMA20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='yellow', width=1), name="EMA"), row=1, col=1)
        try: 
            fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, -3], line=dict(color='cyan', width=1, dash='dot'), name="Upper"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, -1], line=dict(color='cyan', width=1, dash='dot'), name="Lower"), row=1, col=1)
        except: pass
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_color="red", row=2, col=1); fig.add_hline(y=30, line_color="green", row=2, col=1)
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0)); return fig
    except: return None

@st.cache_data(ttl=600)
def generar_mapa_calor(tickers):
    try:
        d = yf.download(" ".join(tickers), period="5d", interval="1d", progress=False, auto_adjust=True)['Close']
        p = ((d.iloc[-1]-d.iloc[-2])/d.iloc[-2])*100
        return pd.DataFrame({'Ticker': p.index, 'Variacion': p.values, 'Sector': 'General', 'Size': d.iloc[-1].values})
    except: return None

def optimizar_parametros_estrategia(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
        if df.empty: return pd.DataFrame()
        df['RSI'] = ta.rsi(df['Close'], 14)
        r = []
        for b in [20, 30, 40]:
            for s in [60, 70, 80]:
                c = 10000; p = 0
                buy = df['RSI'] < b; sell = df['RSI'] > s
                for i in range(15, len(df)):
                    pr = df['Close'].iloc[i]
                    if c > 0 and buy.iloc[i]: p = c/pr; c = 0
                    elif p > 0 and sell.iloc[i]: c = p*pr; p = 0
                r.append({"Compra <": b, "Venta >": s, "Retorno %": ((c + (p * df['Close'].iloc[-1]) - 10000)/10000)*100})
        return pd.DataFrame(r)
    except: return pd.DataFrame()

# --- INTERFAZ V57: THE WATCHTOWER ---
st.title("🔔 Sistema Quant V57: Watchtower")

df_pos = auditar_posiciones_sql()

# PANEL DE ALERTAS (NUEVO V57)
with st.sidebar:
    st.header("🔔 Centro de Alertas")
    with st.spinner("Escaneando mercado..."):
        alertas = generar_feed_alertas(WATCHLIST)
    
    if alertas:
        for a in alertas:
            estilo = "alert-card-low"
            icono = "ℹ️"
            if a['Nivel'] == "ALTA": estilo = "alert-card-high"; icono = "🚨"
            elif a['Nivel'] == "MEDIA": estilo = "alert-card-med"; icono = "⚠️"
            
            st.markdown(f"""
            <div class='{estilo}'>
                <strong>{icono} {a['Ticker']}</strong><br>
                <small>{a['Mensaje']}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ Todo tranquilo en el mercado.")

# DASHBOARD KPI
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
with col_kpi1: st.metric("Patrimonio", f"${df_pos['Valor Mercado'].sum() if not df_pos.empty else 0:,.2f}")
with col_kpi2: st.metric("P&L Total", f"${df_pos['P&L ($)'].sum() if not df_pos.empty else 0:+.2f}")
with col_kpi3: st.metric("Mercado (SPY)", f"${yf.Ticker('SPY').history(period='1d')['Close'].iloc[-1]:.2f}")
with col_kpi4: st.metric("Alertas Activas", f"{len(alertas)}")

st.divider()

main_tabs = st.tabs(["🧠 ESTRATEGIA MACRO", "💼 MESA DE DINERO", "📊 ANÁLISIS 360", "🧬 LABORATORIO QUANT"])

# --- TAB 1: ESTRATEGIA MACRO ---
with main_tabs[0]:
    st.subheader("📰 Informe de Inteligencia de Mercado")
    with st.spinner("Cargando datos macro..."):
        macro_data = obtener_panorama_macro()
    if macro_data:
        cols_macro = st.columns(len(macro_data))
        for i, (k, v) in enumerate(macro_data.items()):
            color = "normal" if k != "VIX (Miedo)" else "inverse"
            cols_macro[i].metric(k, f"{v['Precio']:,.2f}", f"{v['Cambio%']:+.2f}%", delta_color=color)
        st.write("---")
        if st.button("🤖 REDACTAR BRIEFING ESTRATÉGICO"):
            with st.spinner("Analizando..."):
                briefing = generar_briefing_ia(macro_data)
                st.markdown(f"<div class='report-box'><h3>🎙️ Morning Briefing (AI)</h3>{briefing}</div>", unsafe_allow_html=True)

# --- TAB 2: OPERATIVA ---
with main_tabs[1]:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Cartera")
        if not df_pos.empty: st.dataframe(df_pos.style.format({"Valor Mercado": "${:.2f}", "P&L ($)": "${:+.2f}"}).background_gradient(subset=['P&L (%)'], cmap='RdYlGn'), use_container_width=True)
        else: st.info("Sin posiciones.")
    with c2:
        with st.form("op"):
            t = st.selectbox("Ticker", WATCHLIST); tp = st.selectbox("Tipo", ["COMPRA", "VENTA"])
            q = st.number_input("Qty", 1, 10000); precio_ejec = st.number_input("Precio", 0.0)
            if st.form_submit_button("Ejecutar"): registrar_operacion_sql(t, tp, q, precio_ejec); st.rerun()

# --- TAB 3: ANÁLISIS ---
with main_tabs[2]:
    sel_ticker = st.selectbox("Analizar:", WATCHLIST)
    c_a1, c_a2 = st.columns([1, 2])
    with c_a1:
        s, b = calcular_score_quant(sel_ticker)
        st.plotly_chart(dibujar_velocimetro(s), use_container_width=True)
        w = detectar_actividad_ballenas(sel_ticker)
        if w: st.metric("Volumen Hoy", f"{w['Volumen Hoy']/1000:.0f}K", f"{w['Ratio Vol']:.1f}x Promedio")
    with c_a2:
        f = graficar_master(sel_ticker)
        if f: st.plotly_chart(f, use_container_width=True)

# --- TAB 4: LABORATORIO ---
with main_tabs[3]:
    sub_tabs = st.tabs(["🔗 Matriz Correlación", "🏗️ Backtest Cartera", "🧬 Optimización"])
    with sub_tabs[0]:
        corr_tickers = st.multiselect("Activos:", WATCHLIST, default=['NVDA', 'AMD', 'MSFT', 'KO', 'GLD'])
        if st.button("CALCULAR CORRELACIONES"):
            matriz = calcular_matriz_correlacion(corr_tickers)
            if matriz is not None: st.plotly_chart(px.imshow(matriz, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)
    with sub_tabs[1]:
        sim_tickers = st.multiselect("Simular:", WATCHLIST, default=['NVDA', 'TSLA'])
        if st.button("🏗️ SIMULAR"):
            pesos = {t: 1.0/len(sim_tickers) for t in sim_tickers}
            df_ch, st_ch = simular_cartera_historica(sim_tickers, pesos)
            if df_ch is not None:
                st.metric("CAGR", f"{st_ch['CAGR']:.1f}%")
                st.plotly_chart(px.line(df_ch, color_discrete_map={"Mi Cartera": "#00ff00", "Mercado (SPY)": "grey"}), use_container_width=True)
    with sub_tabs[2]:
        if st.button("🚀 Optimizar"):
            r = optimizar_parametros_estrategia(sel_ticker)
            if not r.empty: st.plotly_chart(px.density_heatmap(r, x="Compra <", y="Venta >", z="Retorno %", text_auto=".1f", color_continuous_scale="Viridis"), use_container_width=True)
