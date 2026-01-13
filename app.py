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
st.set_page_config(page_title="Sistema de Inversiones Profesional Quant V83", layout="wide", page_icon="🏛️")

# Estilos CSS (Professional Dark Theme)
st.markdown("""<style>
    .main {background-color: #0e1117;}
    .metric-card {background-color: #1c1c2e; border: 1px solid #2d2d3f; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
    .metric-value {font-size: 24px; font-weight: bold; color: #ffffff;}
    .metric-label {font-size: 14px; color: #a0a0a0;}
    .ai-box {background-color: #131420; border-left: 4px solid #9c27b0; padding: 20px; border-radius: 5px; margin-top: 10px;}
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
    /* Ajuste para listas personalizadas */
    .watchlist-box {background-color: #111; padding: 10px; border-radius: 5px; border: 1px solid #333; margin-bottom: 10px;}
</style>""", unsafe_allow_html=True)

# API Keys
try:
    secrets = toml.load(".streamlit/secrets.toml") if os.path.exists(".streamlit/secrets.toml") else st.secrets
    genai.configure(api_key=secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

# Base de datos y Listas Default
DB_NAME = "quant_database.db"
DEFAULT_WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'COIN', 'KO', 'DIS', 'SPY', 'QQQ', 'GLD', 'USO']

# Inicializar Session State para Listas Personalizadas
if 'mis_listas' not in st.session_state:
    st.session_state['mis_listas'] = {"General": DEFAULT_WATCHLIST, "Vigiladas": [], "Cartera": []}
if 'lista_activa' not in st.session_state:
    st.session_state['lista_activa'] = "General"

# --- 2. MOTORES DE ANÁLISIS TÉCNICO (PROFESIONAL) ---

def calcular_vsa_color(row):
    """Asigna color a la barra de volumen según lógica VSA (Tom Williams)"""
    # Verde: Cierre alcista con volumen alto (Fortaleza)
    # Rojo: Cierre bajista con volumen alto (Debilidad)
    # Gris: Volumen promedio o bajo
    if row['Volume'] > row['Vol_SMA'] * 1.5:
        return 'rgba(0, 255, 0, 0.6)' if row['Close'] > row['Open'] else 'rgba(255, 0, 0, 0.6)'
    elif row['Volume'] < row['Vol_SMA'] * 0.5:
        return 'rgba(100, 100, 100, 0.3)' # Sin interés
    else:
        return 'rgba(100, 100, 100, 0.6)' # Normal

def detectar_patrones_velas_pro(df):
    """Detecta Marubozu, Martillos, Dojis, Engulfing"""
    # Geometría de la vela
    df['Cuerpo'] = abs(df['Close'] - df['Open'])
    df['Mecha_Sup'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['Mecha_Inf'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    df['Rango'] = df['High'] - df['Low']
    df['Cuerpo_Prom'] = df['Cuerpo'].rolling(10).mean()
    
    # 1. MARUBOZU (Cuerpo gigante, sin mechas o mínimas) - Señal de Convicción
    # Condición: Cuerpo > 2x Promedio Y Mechas < 5% del Rango
    df['Patron_Marubozu_Bull'] = (df['Cuerpo'] > 2 * df['Cuerpo_Prom']) & \
                                 (df['Mecha_Sup'] < 0.05 * df['Rango']) & \
                                 (df['Mecha_Inf'] < 0.05 * df['Rango']) & \
                                 (df['Close'] > df['Open'])

    df['Patron_Marubozu_Bear'] = (df['Cuerpo'] > 2 * df['Cuerpo_Prom']) & \
                                 (df['Mecha_Sup'] < 0.05 * df['Rango']) & \
                                 (df['Mecha_Inf'] < 0.05 * df['Rango']) & \
                                 (df['Close'] < df['Open'])

    # 2. MARTILLO (Rebote)
    df['Patron_Martillo'] = (df['Mecha_Inf'] > 2 * df['Cuerpo']) & \
                            (df['Mecha_Sup'] < 0.3 * df['Cuerpo'])
                            
    # 3. DOJI (Indecisión)
    df['Patron_Doji'] = df['Cuerpo'] <= df['Rango'] * 0.1
    
    # 4. ENGULFING
    df['Patron_BullEng'] = (df['Close'] > df['Open']) & \
                           (df['Close'].shift(1) < df['Open'].shift(1)) & \
                           (df['Close'] > df['Open'].shift(1)) & \
                           (df['Open'] < df['Close'].shift(1))

    return df

def calcular_soportes_resistencias(df, window=20):
    """Identifica zonas de S/R usando máximos/mínimos locales (Fractales)"""
    df['Resistencia'] = df['High'].rolling(window=window, center=True).max()
    df['Soporte'] = df['Low'].rolling(window=window, center=True).min()
    return df

@st.cache_data(ttl=60) # Cache corto para intradía
def obtener_datos_grafico(ticker, intervalo):
    """Gestor inteligente de periodos según el intervalo seleccionado"""
    # Mapeo Intervalo -> Periodo óptimo para yfinance
    mapa_periodos = {
        "15m": "60d", # Máximo permitido por Yahoo para 15m
        "1h": "730d", # Máximo 2 años
        "4h": "2y",
        "1d": "2y",
        "1wk": "5y",
        "1mo": "10y"
    }
    periodo = mapa_periodos.get(intervalo, "1y")
    
    try:
        df = yf.Ticker(ticker).history(period=periodo, interval=intervalo)
        if df.empty: return None
        return df
    except: return None

def graficar_profesional_quant(ticker, intervalo):
    df = obtener_datos_grafico(ticker, intervalo)
    if df is None: return None
    
    # --- CÁLCULO DE INDICADORES ---
    # 1. MACD (Trend & Momentum)
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)
    
    # 2. RSI (Oscilador)
    df['RSI'] = ta.rsi(df['Close'], 14)
    
    # 3. VWAP (Volume Weighted Average Price - Institucional)
    try:
        # VWAP suele requerir reset diario en intradía, pandas_ta lo maneja aproximado
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    except: df['VWAP'] = ta.sma(df['Close'], 20) # Fallback si falla VWAP
    
    # 4. VSA (Volumen)
    df['Vol_SMA'] = ta.sma(df['Volume'], 20)
    colors_vsa = df.apply(calcular_vsa_color, axis=1)
    
    # 5. Patrones & S/R
    df = detectar_patrones_velas_pro(df)
    df = calcular_soportes_resistencias(df)
    
    # --- CONSTRUCCIÓN DEL GRÁFICO (4 PANELES) ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(f"Precio ({intervalo}) + VWAP + Patrones", "VSA (Volumen)", "MACD", "RSI")
    )
    
    # PANEL 1: PRECIO
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Precio'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#FFD700', width=1.5), name='VWAP (Institucional)'), row=1, col=1)
    
    # Soportes y Resistencias (Últimos valores extendidos)
    last_sup = df['Soporte'].iloc[-1]
    last_res = df['Resistencia'].iloc[-1]
    fig.add_hline(y=last_res, line_dash="dash", line_color="red", row=1, col=1, annotation_text="Resistencia")
    fig.add_hline(y=last_sup, line_dash="dash", line_color="green", row=1, col=1, annotation_text="Soporte")
    
    # Marcadores de Patrones
    maru_bull = df[df['Patron_Marubozu_Bull']]
    maru_bear = df[df['Patron_Marubozu_Bear']]
    hammer = df[df['Patron_Martillo']]
    
    fig.add_trace(go.Scatter(x=maru_bull.index, y=maru_bull['Low'], mode='markers', marker=dict(symbol='square', size=8, color='blue'), name='Marubozu Alcista'), row=1, col=1)
    fig.add_trace(go.Scatter(x=maru_bear.index, y=maru_bear['High'], mode='markers', marker=dict(symbol='square', size=8, color='purple'), name='Marubozu Bajista'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hammer.index, y=hammer['Low'], mode='markers', marker=dict(symbol='diamond', size=6, color='cyan'), name='Martillo'), row=1, col=1)

    # PANEL 2: VOLUMEN VSA
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors_vsa, name='Volumen VSA'), row=2, col=1)
    
    # PANEL 3: MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_12_26_9'], line=dict(color='white', width=1), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACDs_12_26_9'], line=dict(color='orange', width=1), name='Signal'), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACDh_12_26_9'], marker_color='gray', name='Hist'), row=3, col=1)
    
    # PANEL 4: RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1), name='RSI'), row=4, col=1)
    fig.add_hline(y=70, line_color="red", line_dash="dot", row=4, col=1)
    fig.add_hline(y=30, line_color="green", line_dash="dot", row=4, col=1)
    
    fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=30,b=0))
    return fig

# --- 3. MOTORES "INTOCABLES" (ORÁCULO & TESIS) ---
def oraculo_ml(ticker):
    """Mantenido igual a V79/V82"""
    try:
        df = yf.Ticker(ticker).history(period="2y")
        if len(df)<200: return None
        df['RSI'] = ta.rsi(df['Close'], 14); df['SMA_Diff'] = (df['Close'] - ta.sma(df['Close'], 50))/ta.sma(df['Close'], 50)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        X = df[['RSI', 'SMA_Diff']]; y = df['Target']
        split = int(len(df)*0.8)
        clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X.iloc[:split], y.iloc[:split])
        acc = accuracy_score(y.iloc[split:], clf.predict(X.iloc[split:]))
        pred = clf.predict(X.iloc[[-1]])[0]
        prob = clf.predict_proba(X.iloc[[-1]])[0][pred]
        return {"Pred": "SUBE 🟢" if pred==1 else "BAJA 🔴", "Acc": acc*100, "Prob": prob*100}
    except: return None

def consultar_ia(contexto):
    """Mantenido igual a V79/V82"""
    try: return model.generate_content(contexto).text
    except: return "IA no disponible."

# --- 4. INTERFAZ ---

# SIDEBAR: GESTOR DE LISTAS
with st.sidebar:
    st.title("🏛️ Sistema Prof. Quant")
    
    # Gestor de Listas
    st.markdown("### 📂 Listas de Seguimiento")
    
    # Selector de Lista Activa
    lista_actual = st.selectbox("Ver Lista:", list(st.session_state['mis_listas'].keys()), index=0)
    st.session_state['lista_activa'] = lista_actual
    
    # Mostrar Tickers de la lista seleccionada
    activos_lista = st.session_state['mis_listas'][lista_actual]
    sel_ticker = st.selectbox("🔍 Seleccionar Activo", activos_lista if activos_lista else ["Sin Activos"])
    
    # Herramientas de Edición de Listas
    with st.expander("⚙️ Gestionar Listas"):
        # Crear Nueva Lista
        nueva_lista = st.text_input("Nombre Nueva Lista")
        if st.button("Crear Lista"):
            if nueva_lista and nueva_lista not in st.session_state['mis_listas']:
                st.session_state['mis_listas'][nueva_lista] = []
                st.success(f"Lista {nueva_lista} creada.")
                st.rerun()
        
        # Agregar Activo a Lista Actual
        nuevo_ticker = st.text_input("Agregar Ticker (Ej: AAPL)").upper()
        if st.button("➕ Agregar a esta lista"):
            if nuevo_ticker and nuevo_ticker not in st.session_state['mis_listas'][lista_actual]:
                st.session_state['mis_listas'][lista_actual].append(nuevo_ticker)
                st.success("Agregado.")
                st.rerun()
        
        # Borrar Lista
        if st.button("🗑️ Borrar Lista Actual") and lista_actual != "General":
            del st.session_state['mis_listas'][lista_actual]
            st.session_state['lista_activa'] = "General"
            st.rerun()

# PANEL PRINCIPAL
st.title(f"Análisis: {sel_ticker}")

# SNAPSHOT RAPIDO
try:
    info = yf.Ticker(sel_ticker).info
    precio = info.get('currentPrice', 0)
    target = info.get('targetMeanPrice', 0)
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Precio Actual", f"${precio}")
    k2.metric("Target Analistas", f"${target}")
    k3.metric("Rango 52sem", f"{info.get('fiftyTwoWeekLow',0)} - {info.get('fiftyTwoWeekHigh',0)}")
    k4.metric("Volumen", f"{info.get('volume',0)/1e6:.1f}M")
except: st.warning("Datos en tiempo real demorados.")

# SECCIÓN DE GRÁFICO PROFESIONAL
st.markdown("### 📉 Gráfico Profesional Quant")
c_time, c_info = st.columns([1, 5])
with c_time:
    timeframe = st.selectbox("Temporalidad", ["1d", "15m", "1h", "4h", "1wk", "1mo"], index=0)
    st.info("Indicadores Activos:\n- VSA (Color Vol)\n- VWAP\n- MACD & RSI\n- Soportes/Res.\n- Marubozu/Martillo")

with c_info:
    if sel_ticker != "Sin Activos":
        fig = graficar_profesional_quant(sel_ticker, timeframe)
        if fig: st.plotly_chart(fig, use_container_width=True, height=800)
        else: st.error("Error cargando gráfico. Verifique el ticker o intente otra temporalidad.")

# SECCIÓN "ORÁCULO" Y "TESIS IA" (MANTENIDA IGUAL)
st.markdown("---")
c_oraculo, c_tesis = st.columns([1, 2])

with c_oraculo:
    st.markdown("### 🤖 Oráculo ML")
    st.caption("Random Forest (Datos: 2 Años)")
    if st.button("🔮 Consultar Oráculo"):
        with st.spinner("La IA está pensando..."):
            ml = oraculo_ml(sel_ticker)
            if ml:
                st.metric("Predicción", ml['Pred'])
                st.metric("Probabilidad", f"{ml['Prob']:.1f}%")
                st.metric("Precisión Histórica", f"{ml['Acc']:.1f}%")
                if ml['Acc'] < 55: st.warning("Confianza baja.")
            else: st.error("Datos insuficientes.")

with c_tesis:
    st.markdown("### 📝 Tesis de Inversión (IA Generativa)")
    if st.button("⚡ Generar Tesis"):
        prompt = f"""
        Actúa como un Analista Senior Cuantitativo. Analiza {sel_ticker}.
        Basado en analisis tecnico estandar (RSI, Tendencia) y fundamental basico.
        Dame una recomendacion de compra, venta o mantener y 3 razones clave.
        """
        res = consultar_ia(prompt)
        st.markdown(f"<div class='ai-box'>{res}</div>", unsafe_allow_html=True)
