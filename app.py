import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
import requests 
import google.generativeai as genai
import feedparser
import quantstats as qs
import warnings

# Ignorar advertencias de librerías
warnings.filterwarnings('ignore')

# --- 🔐 TUS CREDENCIALES ---
TELEGRAM_TOKEN = "8042406069:AAHhflfkySyQVhCkHaqIsUjGumFr3fsnDPM" 
TELEGRAM_CHAT_ID = "6288094504"
GOOGLE_API_KEY = "AIzaSyB356Wjicaf9VRUYTX6_EL728IQF6nOmuQ" 

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Sistema Quant V24.5", layout="wide", page_icon="🏛️")
st.markdown("<style>.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

# Configurar IA
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

st.title("🏛️ Sistema Quant V24.5 (Institutional Metrics)")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Radar de Mercado")
    ticker = st.text_input("Activo", value="NVDA").upper()
    dias = st.slider("Ventana de Análisis (Días)", 100, 1000, 500)
    capital = st.number_input("Capital Simulado ($)", value=10000)
    
    st.divider()
    st.caption("Estado: 🟢 Sistema Activo")
    if st.button("Ping (Mantener despierto)"):
        st.toast("Pong! El sistema sigue vivo.")

# --- MOTORES ---
@st.cache_data(ttl=300)
def obtener_datos(symbol, days):
    try:
        df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df if not df.empty else None
    except: return None

def obtener_noticias(symbol):
    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en")
        return [e.title for e in feed.entries[:3]]
    except: return []

# --- EJECUCIÓN ---
df = obtener_datos(ticker, dias)

if df is not None:
    # Preparar datos
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['Returns'] = df['Close'].pct_change()
    
    last_price = df['Close'].iloc[-1]
    last_rsi = df['RSI'].iloc[-1]
    
    # --- PESTAÑAS PRINCIPALES ---
    tab1, tab2, tab3 = st.tabs(["📈 Trading", "📊 Reporte Quant", "🧠 Sala de Guerra (IA)"])
    
    with tab1:
        # Dashboard Clásico
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio", f"${last_price:.2f}", f"{df['Returns'].iloc[-1]*100:.2f}%")
        c2.metric("RSI (14)", f"{last_rsi:.1f}")
        c3.metric("Volatilidad (Anual)", f"{df['Returns'].std() * (252**0.5) * 100:.1f}%")
        
        # Señal Simple
        trend = "ALCISTA" if last_price > df['EMA_200'].iloc[-1] else "BAJISTA"
        c4.metric("Tendencia Macro", trend)
        
        # Gráfico
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Precio'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='orange'), name='EMA 200'))
        fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # Lógica de Alerta Simple (Para el Bot)
        if last_rsi < 30 and trend == "ALCISTA":
            st.warning("⚡ OPORTUNIDAD: RSI Sobrevendido en Tendencia Alcista")

    with tab2:
        st.markdown("### 🔬 Análisis de Rendimiento Institucional")
        st.info("Este reporte simula si hubieras mantenido este activo (Buy & Hold) durante el periodo seleccionado.")
        
        if st.button("Generar Tear Sheet (QuantStats)"):
            with st.spinner("Calculando métricas complejas (Sharpe, Sortino, Drawdowns)..."):
                # Usamos un try/except porque Quantstats a veces falla con datos muy recientes
                try:
                    # Cálculo de métricas clave
                    qs.extend_pandas()
                    stock_ret = df['Returns'].dropna()
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Sharpe Ratio", f"{stock_ret.sharpe():.2f}")
                    col_m2.metric("Max Drawdown", f"{stock_ret.max_drawdown()*100:.2f}%")
                    col_m3.metric("Win Rate", f"{stock_ret.win_rate()*100:.1f}%")
                    
                    # Gráficos de Quantstats renderizados en Streamlit
                    st.subheader("Retornos Acumulados vs S&P 500")
                    # Descargamos SPY para comparar
                    spy = yf.download("SPY", period=f"{dias}d", progress=False)['Close'].pct_change().dropna()
                    
                    # Crear gráfico comparativo manual con Plotly para velocidad
                    cum_stock = (1 + stock_ret).cumprod()
                    cum_spy = (1 + spy).cumprod()
                    
                    fig_perf = go.Figure()
                    fig_perf.add_trace(go.Scatter(x=cum_stock.index, y=cum_stock, name=ticker, line=dict(color='#00ff00')))
                    fig_perf.add_trace(go.Scatter(x=cum_spy.index, y=cum_spy, name="S&P 500", line=dict(color='gray', dash='dot')))
                    fig_perf.update_layout(title="Performance Relativa", template="plotly_dark", height=400)
                    st.plotly_chart(fig_perf, use_container_width=True)
                    
                    st.markdown("""
                    **Glosario:**
                    * **Sharpe Ratio:** > 1 es bueno, > 2 es excelente. Mide rentabilidad por unidad de riesgo.
                    * **Max Drawdown:** La peor caída desde un máximo histórico. Mide el dolor.
                    """)
                    
                except Exception as e:
                    st.error(f"Error generando reporte: {e}")

    with tab3:
        st.markdown("### 🧠 Inteligencia Artificial (Gemini)")
        if st.button("Consultar Opinión de Mercado"):
            with st.spinner("Analizando noticias..."):
                news = obtener_noticias(ticker)
                prompt = f"Analiza {ticker}. Precio: {last_price}. Noticias: {news}. Dame una recomendación de trading (Comprar/Vender/Esperar) y explica por qué en 1 parrafo corto."
                try:
                    res = model.generate_content(prompt)
                    st.success(res.text)
                    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                                  json={"chat_id": TELEGRAM_CHAT_ID, "text": f"🤖 IA {ticker}: {res.text}"})
                except: st.error("Error conectando con Gemini")

else:
    st.error("No se encontraron datos. Revisa el ticker.")
