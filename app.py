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

# --- CONFIGURACIÓN ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sistema Quant V80 (The Sentiment)", layout="wide", page_icon="📰")

st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;}
    .news-card {background-color: #1e202e; border-left: 3px solid #FFD700; padding: 15px; margin-bottom: 10px; border-radius: 5px;}
    .sentiment-pos {color: #00ff00; font-weight: bold;}
    .sentiment-neg {color: #ff4b4b; font-weight: bold;}
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

# --- MOTOR NLP DE SENTIMIENTO (NUEVO V80) ---
def analizar_sentimiento_noticias(ticker):
    """Descarga noticias y calcula un score de sentimiento básico"""
    if "USD" in ticker: return None # Las noticias de crypto en yfinance a veces fallan
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news: return None
        
        # Diccionarios de Sentimiento (Simple pero efectivo para velocidad)
        bullish_words = ['up', 'rise', 'jump', 'gain', 'buy', 'bull', 'profit', 'beat', 'growth', 'surge', 'record', 'high', 'strong']
        bearish_words = ['down', 'drop', 'fall', 'loss', 'sell', 'bear', 'miss', 'risk', 'crash', 'weak', 'lower', 'cut', 'lawsuit']
        
        total_score = 0
        news_processed = []
        
        for n in news[:5]: # Analizar las últimas 5 noticias
            title = n['title']
            link = n['link']
            publisher = n['publisher']
            # Convertir timestamp
            try: pub_time = datetime.fromtimestamp(n['providerPublishTime']).strftime('%Y-%m-%d %H:%M')
            except: pub_time = "Reciente"
            
            # Análisis NLP Básico
            title_lower = title.lower()
            score = 0
            for w in bullish_words: 
                if w in title_lower: score += 1
            for w in bearish_words: 
                if w in title_lower: score -= 1
            
            total_score += score
            
            # Etiqueta
            sentiment_label = "Positivo 🟢" if score > 0 else "Negativo 🔴" if score < 0 else "Neutral ⚪"
            
            news_processed.append({
                "Titulo": title,
                "Fuente": publisher,
                "Hora": pub_time,
                "Link": link,
                "Sentimiento": sentiment_label,
                "Score": score
            })
            
        # Normalizar Score Final (-10 a +10 aprox)
        final_sentiment = "NEUTRAL"
        if total_score >= 2: final_sentiment = "ALCISTA (Bullish) 🐂"
        elif total_score <= -2: final_sentiment = "BAJISTA (Bearish) 🐻"
        
        return {
            "Noticias": news_processed,
            "Score_Total": total_score,
            "Sentimiento_Global": final_sentiment
        }
            
    except Exception as e: return None

# --- MOTORES EXISTENTES (V79) ---
def entrenar_modelo_ml(ticker):
    try:
        df = yf.Ticker(ticker).history(period="2y")
        if len(df) < 200: return None
        df['RSI'] = ta.rsi(df['Close'], 14); df['SMA_Diff'] = (df['Close'] - ta.sma(df['Close'], 50)) / ta.sma(df['Close'], 50)
        df['Vol_Change'] = df['Volume'].pct_change(); df['Return'] = df['Close'].pct_change()
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        X = df[['RSI', 'SMA_Diff', 'Vol_Change', 'Return']]; y = df['Target']
        split = int(len(df) * 0.8)
        clf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        clf.fit(X.iloc[:split], y.iloc[:split])
        acc = accuracy_score(y.iloc[split:], clf.predict(X.iloc[split:]))
        pred = clf.predict(X.iloc[[-1]])[0]
        prob = clf.predict_proba(X.iloc[[-1]])[0][pred]
        return {"Prediccion": "SUBE 🟢" if pred == 1 else "BAJA 🔴", "Accuracy": acc * 100, "Probabilidad": prob * 100}
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

def detectar_patrones_avanzados(df):
    if df.empty: return df
    df['Cuerpo'] = abs(df['Close'] - df['Open'])
    df['Mecha_Sup'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['Mecha_Inf'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    df['Patron_Martillo'] = (df['Mecha_Inf'] > 2 * df['Cuerpo']) & (df['Mecha_Sup'] < 0.5 * df['Cuerpo'])
    df['Patron_BullEng'] = (df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)) & (df['Close'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1))
    return df

def graficar_pro_v78(ticker):
    try:
        df = yf.Ticker(ticker).history(period="6mo", auto_adjust=True)
        if df.empty: return None
        df = detectar_patrones_avanzados(df)
        df['SMA50'] = ta.sma(df['Close'], 50); df['SMA200'] = ta.sma(df['Close'], 200)
        martillos = df[df['Patron_Martillo']]
        bull_eng = df[df['Patron_BullEng']]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Precio'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='cyan', width=1), name='SMA 50'), row=1, col=1)
        fig.add_trace(go.Scatter(x=martillos.index, y=martillos['Low']*0.98, mode='markers', marker=dict(symbol='diamond', size=10, color='#00ff00'), name='Martillo'), row=1, col=1)
        fig.add_trace(go.Scatter(x=bull_eng.index, y=bull_eng['Low']*0.97, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00cc96'), name='Bullish Engulfing'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='rgba(100, 100, 100, 0.5)', name='Volumen'), row=2, col=1)
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0)); return fig
    except: return None

@st.cache_data(ttl=3600)
def obtener_datos_insider(ticker):
    if "USD" in ticker: return None
    try:
        info = yf.Ticker(ticker).info
        sh_short = info.get('sharesShort', 0); float_sh = info.get('floatShares', 1)
        short_pct = (sh_short/float_sh)*100 if float_sh else 0
        inst_pct = info.get('heldPercentInstitutions', 0)*100
        return {"Institucional": inst_pct, "Short_Float": short_pct}
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

# --- INTERFAZ V80 ---
c1, c2 = st.columns([3, 1])
with c1: st.title("📰 Quant Terminal V80: The Sentiment")
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
    fig_chart = graficar_pro_v78(sel_ticker)
    if fig_chart: st.plotly_chart(fig_chart, use_container_width=True)
    
    tabs_detail = st.tabs(["📰 SENTIMIENTO (NLP)", "🤖 Oráculo ML", "🦈 Institucionales", "📝 IA"])
    
    # --- TAB 1: SENTIMIENTO DE MERCADO (NUEVO V80) ---
    with tabs_detail[0]:
        st.subheader("📰 Análisis Psicológico & Noticias")
        
        with st.spinner("Leyendo noticias y analizando tono (NLP)..."):
            sentiment_data = analizar_sentimiento_noticias(sel_ticker)
            
            if sentiment_data:
                # Medidor de Sentimiento (Gauge)
                st.markdown(f"#### Estado de Ánimo: **{sentiment_data['Sentimiento_Global']}**")
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = sentiment_data['Score_Total'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Sentiment Score (-10 a +10)"},
                    delta = {'reference': 0},
                    gauge = {
                        'axis': {'range': [-10, 10]},
                        'bar': {'color': "white"},
                        'steps': [
                            {'range': [-10, -2], 'color': "#ff4b4b"},
                            {'range': [-2, 2], 'color': "gray"},
                            {'range': [2, 10], 'color': "#00cc96"}],
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="#0e1117", font={'color': "white"})
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                st.markdown("---")
                st.markdown("#### 🗞️ Últimas Noticias Relevantes")
                
                for n in sentiment_data['Noticias']:
                    color_s = "sentiment-pos" if "Positivo" in n['Sentimiento'] else "sentiment-neg" if "Negativo" in n['Sentimiento'] else "white"
                    st.markdown(f"""
                    <div class='news-card'>
                        <a href='{n['Link']}' target='_blank' style='color: #FFD700; text-decoration: none; font-size: 16px; font-weight: bold;'>{n['Titulo']}</a>
                        <br>
                        <small>{n['Fuente']} | {n['Hora']}</small>
                        <br>
                        Impacto Detectado: <span class='{color_s}'>{n['Sentimiento']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No se encontraron noticias recientes o el activo no soporta feed de noticias (Cripto/Forex).")

    with tabs_detail[1]:
        if st.button("🧠 ENTRENAR ML"):
            ml_res = entrenar_modelo_ml(sel_ticker)
            if ml_res: st.metric("Predicción", ml_res['Prediccion'], f"Certeza: {ml_res['Probabilidad']:.1f}%")
            
    with tabs_detail[2]:
        insider = obtener_datos_insider(sel_ticker)
        if insider:
            c1, c2 = st.columns(2); c1.metric("Institucional", f"{insider['Institucional']:.1f}%"); c2.metric("Shorts", f"{insider['Short_Float']:.2f}%")

    with tabs_detail[3]:
        if st.button("Generar Informe IA"):
            try: st.write(model.generate_content(f"Analisis {sel_ticker}").text)
            except: st.error("Error IA")

with col_side:
    st.subheader("⚡ Quick Trade")
    with st.form("quick"):
        q = st.number_input("Qty", 1, 1000, 10); s = st.selectbox("Side", ["COMPRA", "VENTA"])
        if st.form_submit_button("EJECUTAR"): 
            if snap: registrar_operacion_sql(sel_ticker, s, q, snap['Precio']); st.success("Orden OK")
    
    st.subheader("🏆 Ranking")
    if st.button("🔄 ESCANEAR"): st.dataframe(escanear_mercado_completo(WATCHLIST), use_container_width=True)
    
    st.subheader("💼 Cartera")
    df_p = auditar_posiciones_sql()
    if not df_p.empty: st.dataframe(df_p[['Ticker', 'P&L']], use_container_width=True)
