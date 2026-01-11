
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
import feedparser # <--- LIBRERÍA NUEVA PARA NOTICIAS
from datetime import datetime
from scipy.signal import argrelextrema 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import minimize 
from scipy.stats import norm 
import google.generativeai as genai

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
st.set_page_config(page_title="Sistema Quant V44 (News Reader)", layout="wide", page_icon="📰")
st.markdown("""
<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 10px; color: white;}
    .signal-box {border: 2px solid #FFD700; padding: 10px; border-radius: 5px; background-color: #2b2b00; text-align: center;}
    .macro-card {background-color: #1e2130; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #444;}
    .news-card {background-color: #1a1c24; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #4CAF50;}
</style>
""", unsafe_allow_html=True)

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'COIN', 'KO', 'DIS', 'SPY', 'QQQ', 'DIA', 'GLD', 'USO']
MACRO_TICKERS = {'S&P 500': 'SPY', 'VIX (Miedo)': '^VIX', 'Bonos 10Y': '^TNX', 'Oro': 'GC=F', 'Dólar': 'DX-Y.NYB'}
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

# --- MOTOR DE NOTICIAS & SENTIMIENTO IA (NUEVO V44) ---
@st.cache_data(ttl=3600) # Cache de 1 hora para no gastar API
def analizar_noticias_ia(ticker):
    try:
        # 1. Obtener Noticias de Google News (RSS)
        rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+finance&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        
        if not feed.entries: return None
        
        # Tomamos los 5 titulares más recientes
        headlines = [entry.title for entry in feed.entries[:5]]
        links = [entry.link for entry in feed.entries[:5]]
        
        # 2. Análisis con Gemini IA
        prompt = f"""
        Actúa como un analista financiero senior. Analiza estos titulares sobre {ticker}:
        {headlines}
        
        1. Resume el sentimiento general en un párrafo corto en español.
        2. Dame una puntuación de sentimiento del -100 (Muy Bajista/Miedo) al +100 (Muy Alcista/Euforia).
        3. Clasifica el sentimiento en una palabra: 'ALCISTA', 'BAJISTA' o 'NEUTRAL'.
        
        Formato de respuesta:
        SENTIMIENTO: [Puntuación]
        CLASIFICACION: [Clasificación]
        RESUMEN: [Resumen]
        """
        
        response = model.generate_content(prompt)
        text = response.text
        
        # Parsing básico de la respuesta (Buscamos las palabras clave)
        score = 0
        clasificacion = "NEUTRAL"
        resumen = text
        
        # Extracción simple de datos
        for line in text.split('\n'):
            if "SENTIMIENTO:" in line:
                try: score = int(line.split(":")[1].strip())
                except: pass
            if "CLASIFICACION:" in line:
                clasificacion = line.split(":")[1].strip()
            if "RESUMEN:" in line:
                resumen = line.split("RESUMEN:")[1].strip()

        return {
            "headlines": headlines,
            "links": links,
            "score": score,
            "class": clasificacion,
            "summary": text # Mostramos el texto completo de la IA por si el parsing falla
        }
    except Exception as e:
        return {"error": str(e)}

# --- MOTORES EXISTENTES (Macro, Valuación, etc) ---
@st.cache_data(ttl=3600)
def calcular_valor_intrinseco(ticker):
    try:
        stock = yf.Ticker(ticker); info = stock.info
        price = info.get('currentPrice') or info.get('regularMarketPreviousClose')
        eps = info.get('trailingEps'); book = info.get('bookValue')
        graham = math.sqrt(22.5 * eps * book) if eps and book and eps>0 and book>0 else 0
        status = "INFRAVALORADA 🟢" if graham > price else "SOBREVALORADA 🔴"
        return {"Precio": price, "Graham": graham, "Status_Graham": status, "Diff": ((graham-price)/price)*100}
    except: return None

def obtener_datos_macro():
    tickers = list(MACRO_TICKERS.values())
    try:
        df = yf.download(" ".join(tickers), period="2d", progress=False, group_by='ticker', auto_adjust=True)
        res = {}
        for name, tick in MACRO_TICKERS.items():
            try:
                if len(tickers) > 1: price = df[tick]['Close'].iloc[-1]; prev = df[tick]['Close'].iloc[-2]
                else: price = df['Close'].iloc[-1]; prev = df['Close'].iloc[-2]
                delta = ((price - prev) / prev) * 100; res[name] = (price, delta)
            except: res[name] = (0, 0)
        return res
    except: return None

def calcular_alpha_beta(ticker, benchmark='SPY'):
    try:
        data = yf.download(f"{ticker} {benchmark}", period="1y", progress=False, auto_adjust=True)['Close']
        if data.empty: return None, None
        ret = data.pct_change().dropna()
        cov = ret[ticker].cov(ret[benchmark]); var = ret[benchmark].var(); beta = cov/var
        norm = (data/data.iloc[0])*100
        alpha = (norm[ticker].iloc[-1]-100) - (norm[benchmark].iloc[-1]-100)
        return norm, {"Beta": beta, "Alpha Total %": alpha}
    except: return None, None

def simular_montecarlo(ticker, dias=30, sims=500):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)['Close']
        log_ret = np.log(1+df.pct_change()); u = log_ret.mean().item(); var = log_ret.var().item()
        drift = u - (0.5*var); stdev = log_ret.std().item(); last = df.iloc[-1].item()
        daily_ret = np.exp(drift + stdev * norm.ppf(np.random.rand(dias, sims)))
        price_list = np.zeros_like(daily_ret); price_list[0] = last
        for t in range(1, dias): price_list[t] = price_list[t-1]*daily_ret[t]
        fig = go.Figure()
        for i in range(min(50, sims)): fig.add_trace(go.Scatter(y=price_list[:, i], mode='lines', line=dict(width=1, color='rgba(0,255,255,0.1)'), showlegend=False))
        mean = np.mean(price_list, axis=1)
        fig.add_trace(go.Scatter(y=mean, mode='lines', line=dict(width=3, color='yellow'), name='Promedio'))
        p95 = np.percentile(price_list, 95, axis=1); p05 = np.percentile(price_list, 5, axis=1)
        fig.add_trace(go.Scatter(y=p95, mode='lines', line=dict(width=1, color='green', dash='dash'), name='Optimista')); fig.add_trace(go.Scatter(y=p05, mode='lines', line=dict(width=1, color='red', dash='dash'), name='Pesimista'))
        fig.update_layout(title=f"Montecarlo {ticker}", template="plotly_dark", height=400)
        return fig, {"esperado": mean[-1], "pesimista": p05[-1]}
    except: return None, None

@st.cache_data(ttl=600)
def generar_mapa_calor(tickers):
    try:
        data = yf.download(" ".join(tickers), period="5d", interval="1d", progress=False, auto_adjust=True)['Close']
        pct = ((data.iloc[-1] - data.iloc[-2]) / data.iloc[-2]) * 100
        df = pd.DataFrame({'Ticker': pct.index, 'Variacion': pct.values, 'Precio': data.iloc[-1].values})
        sectores = []
        for t in df['Ticker']:
            if t in ['NVDA', 'AMD', 'TSLA', 'AAPL', 'MSFT', 'META', 'GOOGL']: sectores.append('Tecnología')
            elif t in ['BTC-USD', 'ETH-USD', 'COIN']: sectores.append('Cripto')
            elif t in ['SPY', 'QQQ', 'DIA']: sectores.append('Índices')
            elif t in ['GLD', 'USO']: sectores.append('Commodities')
            else: sectores.append('Otros')
        df['Sector'] = sectores; df['Size'] = df['Precio'] 
        return df
    except: return None

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
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"), row=1, col=1)
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
st.title("📰 Sistema Quant V44: News Reader")

# 1. CINTA MACRO
macro_data = obtener_datos_macro()
if macro_data:
    cols = st.columns(len(macro_data))
    for idx, (name, (price, delta)) in enumerate(macro_data.items()):
        color = "red" if delta < 0 else "green"
        if name == "VIX (Miedo)": color = "green" if delta < 0 else "red"
        cols[idx].markdown(f"<div class='macro-card'><small>{name}</small><br><b>{price:,.2f}</b><br><span style='color:{color}'>{delta:+.2f}%</span></div>", unsafe_allow_html=True)
st.divider()

# 2. PANEL PRINCIPAL
c_left, c_right = st.columns([1, 2.5])

with c_left:
    st.subheader("Control")
    tk = st.selectbox("Activo:", WATCHLIST, index=0)
    cap = st.number_input("Simulación ($)", 2000, 100000, 10000, key='cap_sim')
    
    st.markdown("### 🏦 Mi Portafolio")
    df_pos = auditar_posiciones_sql()
    if not df_pos.empty:
        st.metric("P&L Total", f"${df_pos['P&L ($)'].sum():+.2f}", delta_color="normal")
        st.dataframe(df_pos[['Ticker', 'P&L (%)']])

    with st.expander("📝 Operar"):
        op_tk = st.selectbox("Ticker", WATCHLIST, key='op_tk')
        op_type = st.selectbox("Tipo", ["COMPRA", "VENTA"])
        op_qty = st.number_input("Qty", 1, 1000)
        op_px = st.number_input("Precio", 0.0)
        if st.button("Ejecutar"):
            registrar_operacion_sql(op_tk, op_type, op_qty, op_px); st.rerun()

with c_right:
    tabs = st.tabs(["📰 Noticias & IA", "💎 Valuación", "🆚 Alpha", "🔮 Monte Carlo", "📈 Gráfico", "🔥 Heatmap"])
    
    # PESTAÑA 1: NOTICIAS IA (NUEVA V44)
    with tabs[0]:
        st.subheader(f"📰 Sentimiento del Mercado: {tk}")
        
        if st.button("🤖 ANALIZAR NOTICIAS CON IA"):
            with st.spinner("Leyendo las noticias globales... (Esto toma unos segundos)"):
                news_data = analizar_noticias_ia(tk)
                
                if news_data and "error" not in news_data:
                    # GAUGE DE SENTIMIENTO
                    score = news_data.get('score', 0)
                    color_gauge = "grey"
                    if score > 20: color_gauge = "green"
                    elif score < -20: color_gauge = "red"
                    
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.markdown(f"""
                        <div style="text-align:center; padding:20px; background-color:{color_gauge}; border-radius:10px;">
                            <h1 style="color:white; margin:0;">{score}</h1>
                            <p style="color:white; margin:0;">SCORE IA (-100 a +100)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with c2:
                        st.info("💡 " + news_data.get('summary', 'Sin resumen'))
                    
                    st.divider()
                    st.markdown("### 🗞️ Titulares Analizados")
                    for i, head in enumerate(news_data['headlines']):
                        st.markdown(f"- [{head}]({news_data['links'][i]})")
                        
                else:
                    st.error("No se pudieron obtener noticias recientes o hubo un error con la IA.")
                    if news_data: st.write(news_data)

    with tabs[1]:
        val_data = calcular_valor_intrinseco(tk)
        if val_data:
            c1, c2 = st.columns(2)
            c1.metric("Precio", f"${val_data['Precio']:.2f}")
            c1.metric("Graham", f"${val_data['Graham']:.2f}", f"{val_data['Diff']:.1f}%")
            if val_data['Status_Graham'] == "INFRAVALORADA 🟢": st.success("BARATA (Graham)")
            else: st.error("CARA (Graham)")

    with tabs[2]:
        norm_data, metrics = calcular_alpha_beta(tk)
        if norm_data is not None:
            c1, c2 = st.columns(2)
            c1.metric("Beta", f"{metrics['Beta']:.2f}")
            c2.metric("Alpha", f"{metrics['Alpha Total %']:.2f}%")
            st.plotly_chart(px.line(norm_data, x=norm_data.index, y=norm_data.columns), use_container_width=True)

    with tabs[3]:
        dias_mc = st.slider("Días", 10, 90, 30)
        if st.button("🎲 Simular"):
            fig_mc, res_mc = simular_montecarlo(tk, dias_mc)
            if fig_mc: st.plotly_chart(fig_mc, use_container_width=True)

    with tabs[4]:
        fig = graficar_master(tk)
        if fig: st.plotly_chart(fig, use_container_width=True)
        
    with tabs[5]:
        df_map = generar_mapa_calor(WATCHLIST)
        if df_map is not None:
            fig_map = px.treemap(df_map, path=['Sector', 'Ticker'], values='Size', color='Variacion', color_continuous_scale='RdYlGn', color_continuous_midpoint=0)
            st.plotly_chart(fig_map, use_container_width=True)