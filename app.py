import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
import requests 
import google.generativeai as genai
import feedparser

# --- 🔐 TUS CREDENCIALES (PEGALAS AQUÍ DE NUEVO) ---
TELEGRAM_TOKEN = "8042406069:AAHhflfkySyQVhCkHaqIsUjGumFr3fsnDPM" 
TELEGRAM_CHAT_ID = "6288094504"
GOOGLE_API_KEY = "AIzaSyB356Wjicaf9VRUYTX6_EL728IQF6nOmuQ" 

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Sistema Quant V24.4", layout="wide", page_icon="📈")

# Configurar IA
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except:
    st.warning("⚠️ La API de Google no está configurada correctamente.")

st.title("🚀 Sistema Quant V24.4 (Full Analysis)")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Radar")
    ticker = st.text_input("Activo", value="TSLA").upper()
    dias = st.slider("Días de Análisis", 100, 730, 400)
    st.divider()
    st.info("💡 Si no ves datos, prueba con 'AAPL' o 'BTC-USD'")

# --- 1. MOTOR DE DATOS ROBUSTO ---
@st.cache_data(ttl=60)
def obtener_datos(symbol, days):
    try:
        # Descarga forzando formato simple
        df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False, auto_adjust=True)
        
        # Corrección de Pandas MultiIndex (El error más común)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Limpieza
        if df.empty: return None
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Error descargando datos: {e}")
        return None

def obtener_noticias(symbol):
    try:
        url = f"https://news.google.com/rss/search?q={symbol}+finance&hl=es&gl=ES&ceid=ES:es"
        feed = feedparser.parse(url)
        return [entry.title for entry in feed.entries[:5]]
    except:
        return ["No se pudieron cargar noticias."]

# --- EJECUCIÓN ---
df = obtener_datos(ticker, dias)

if df is not None and len(df) > 50:
    # --- 2. CÁLCULOS TÉCNICOS (RESTAURADOS) ---
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_200'] = ta.ema(df['Close'], length=200) # ¡Aquí está la EMA!
    df['EMA_50']  = ta.ema(df['Close'], length=50)  # Agregamos la de 50 también

    last_price = df['Close'].iloc[-1]
    last_rsi = df['RSI'].iloc[-1]
    last_ema = df['EMA_200'].iloc[-1] if pd.notna(df['EMA_200'].iloc[-1]) else 0

    # --- 3. DASHBOARD VISUAL ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Precio", f"${last_price:.2f}")
    col2.metric("RSI (14)", f"{last_rsi:.1f}")
    
    tendencia = "ALCISTA 🟢" if last_price > last_ema else "BAJISTA 🔴"
    col3.metric("Tendencia (EMA 200)", tendencia, f"Soporte: ${last_ema:.2f}")

    # --- 4. GRÁFICO (CON EMA VISIBLE) ---
    fig = go.Figure()
    
    # Velas
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                 low=df['Low'], close=df['Close'], name='Precio'))
    
    # Líneas de Tendencia
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], 
                             line=dict(color='orange', width=2), name='EMA 200 (Tendencia)'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], 
                             line=dict(color='cyan', width=1), name='EMA 50 (Rápida)'))

    fig.update_layout(height=500, template="plotly_dark", title=f"Análisis Técnico: {ticker}")
    st.plotly_chart(fig, use_container_width=True)

    # --- 5. SECCIÓN IA (BOTÓN VISIBLE) ---
    st.divider()
    st.subheader("🧠 Inteligencia Artificial de Mercado")
    
    col_btn, col_txt = st.columns([1, 3])
    
    with col_btn:
        # Botón grande y visible
        analizar = st.button(f"🔎 ANALIZAR {ticker}", type="primary", use_container_width=True)

    if analizar:
        with st.status("🤖 Consultando a los oráculos...", expanded=True) as status:
            st.write("1. Leyendo noticias financieras...")
            noticias = obtener_noticias(ticker)
            
            st.write("2. Analizando indicadores técnicos...")
            datos_tecnicos = f"Precio: {last_price}, RSI: {last_rsi}, Tendencia: {tendencia}"
            
            st.write("3. Generando veredicto con Gemini...")
            prompt = f"""
            Actúa como un Trader Senior. Analiza {ticker}.
            Datos Técnicos: {datos_tecnicos}
            Noticias: {noticias}
            
            Dame una respuesta DIRECTA y CRÍTICA:
            1. SENTIMIENTO: (Positivo/Negativo/Neutral)
            2. ANÁLISIS: Breve explicación de 2 líneas.
            3. RECOMENDACIÓN: ¿Qué harías tú?
            Responde en Español.
            """
            try:
                response = model.generate_content(prompt)
                status.update(label="¡Análisis Completo!", state="complete", expanded=False)
                
                # Mostrar Resultado
                st.success("✅ Informe Generado")
                st.markdown(f"### 📄 Veredicto IA para {ticker}")
                st.info(response.text)
                
                # Enviar a Telegram
                msg = f"🤖 INFORME {ticker}\n\n{response.text}"[:4000] # Limite caracteres
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                              json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
                st.toast("Enviado a Telegram 📲")
                
            except Exception as e:
                st.error(f"Error IA: {e}")

else:
    st.warning(f"⏳ Esperando datos para {ticker}... (O el activo no existe)")