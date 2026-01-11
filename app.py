import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
import requests 
import google.generativeai as genai
import feedparser
import warnings

warnings.filterwarnings('ignore')

# --- 🔐 CREDENCIALES ---
TELEGRAM_TOKEN = "8042406069:AAHhflfkySyQVhCkHaqIsUjGumFr3fsnDPM" 
TELEGRAM_CHAT_ID = "6288094504"
GOOGLE_API_KEY = "AIzaSyB356Wjicaf9VRUYTX6_EL728IQF6nOmuQ" 

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Sistema Quant V24.7 (Risk Manager)", layout="wide", page_icon="🛡️")
st.markdown("<style>.block-container {padding-top: 1rem;} .stMetric {background-color: #111; padding: 10px; border-radius: 5px; border: 1px solid #333;}</style>", unsafe_allow_html=True)

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

# --- LISTA DE VIGILANCIA ---
WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'KO', 'DIS']

st.title("🛡️ Sistema Quant V24.7: Risk Manager")

# --- BARRA LATERAL: GESTIÓN DE CAPITAL ---
with st.sidebar:
    st.header("💰 Gestión de Capital")
    capital_total = st.number_input("Tu Capital Total ($)", value=2000, step=100)
    riesgo_por_trade = st.slider("Riesgo por Operación (%)", 0.5, 5.0, 2.0)
    st.caption(f"⚠️ Si una operación toca Stop Loss, perderás máximo: **${(capital_total * riesgo_por_trade / 100):.2f}**")
    
    st.divider()
    st.header("📡 Radar")
    if st.button("🔄 Escanear Mercado"):
        st.cache_data.clear()
        st.rerun()

# --- MOTORES ---
@st.cache_data(ttl=600)
def escanear_mercado(tickers):
    data = []
    string_tickers = " ".join(tickers)
    try:
        df_bulk = yf.download(string_tickers, period="6mo", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
    except: return pd.DataFrame()
    
    for t in tickers:
        try:
            # Manejo de estructura de datos de yfinance
            if len(tickers) > 1: df = df_bulk[t].copy()
            else: df = df_bulk.copy()
            
            df = df.dropna()
            if len(df) < 50: continue

            # Indicadores
            close = df['Close'].iloc[-1]
            rsi = ta.rsi(df['Close'], length=14).iloc[-1]
            ema200 = ta.ema(df['Close'], length=200).iloc[-1]
            atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1]
            
            # Score
            score = 50
            trend = "ALCISTA" if close > ema200 else "BAJISTA"
            if trend == "ALCISTA": score += 20
            if rsi < 30: score += 30 # Rebote alcista
            elif rsi > 70: score -= 10
            
            data.append({
                "Ticker": t, "Precio": close, "RSI": rsi, 
                "ATR": atr, "Trend": trend, "Score": score
            })
        except: continue
    return pd.DataFrame(data)

def obtener_historial(symbol):
    df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    return df

# --- INTERFAZ PRINCIPAL ---
tab_radar, tab_operacion = st.tabs(["📡 Radar de Oportunidades", "🧮 Calculadora de Posición (Risk)"])

# 1. PESTAÑA RADAR
with tab_radar:
    df_radar = escanear_mercado(WATCHLIST)
    if not df_radar.empty:
        df_radar = df_radar.sort_values(by="Score", ascending=False)
        mejores = df_radar.head(3)
        
        c1, c2, c3 = st.columns(3)
        top_pick = df_radar.iloc[0]
        c1.metric("🥇 Top Pick", top_pick['Ticker'], f"Score: {top_pick['Score']}")
        c2.metric("Precio", f"${top_pick['Precio']:.2f}")
        c3.metric("RSI", f"{top_pick['RSI']:.1f}")
        
        st.dataframe(df_radar.style.background_gradient(subset=['Score'], cmap='Greens'), use_container_width=True)
    else:
        st.info("Presiona 'Escanear Mercado' en la barra lateral.")

# 2. PESTAÑA OPERACIÓN (RISK ENGINE)
with tab_operacion:
    st.subheader("🛡️ Sala de Operaciones Profesional")
    
    col_sel, col_kpi = st.columns([1, 3])
    with col_sel:
        activo_op = st.selectbox("Seleccionar Activo a Operar", WATCHLIST, index=WATCHLIST.index(top_pick['Ticker']) if 'top_pick' in locals() else 0)
    
    # Cargar datos del activo seleccionado
    df_op = obtener_historial(activo_op)
    if df_op is not None:
        # Calcular Niveles Dinámicos
        atr_value = ta.atr(df_op['High'], df_op['Low'], df_op['Close'], length=14).iloc[-1]
        precio_actual = df_op['Close'].iloc[-1]
        
        # Estrategia de Stop Loss Dinámico (2x ATR)
        stop_loss_sugerido = precio_actual - (2 * atr_value)
        take_profit_sugerido = precio_actual + (3 * atr_value) # Ratio 1.5:1
        
        # --- CÁLCULO DE GESTIÓN DE RIESGO (MATEMÁTICA PURA) ---
        riesgo_usd = capital_total * (riesgo_por_trade / 100)
        distancia_sl = precio_actual - stop_loss_sugerido
        
        if distancia_sl > 0:
            cantidad_acciones = riesgo_usd / distancia_sl
            inversion_total = cantidad_acciones * precio_actual
        else:
            cantidad_acciones = 0
            inversion_total = 0

        # Mostrar la "Tarjeta de la Operación"
        st.markdown(f"### 📋 Plan de Trading: {activo_op}")
        
        # Métricas Críticas
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("1. Capital en Riesgo", f"${riesgo_usd:.2f}", f"{riesgo_por_trade}% de la cuenta")
        m2.metric("2. Stop Loss (2xATR)", f"${stop_loss_sugerido:.2f}", f"-${distancia_sl:.2f} por acción")
        m3.metric("3. TAMAÑO POSICIÓN", f"{int(cantidad_acciones)} Acciones", f"Invierte: ${inversion_total:.2f}")
        m4.metric("4. Ratio Beneficio", "1 : 1.5", f"Obj: ${take_profit_sugerido:.2f}")
        
        # Validaciones de Seguridad
        if inversion_total > capital_total:
            st.error(f"❌ ¡ALERTA DE RIESGO! Esta operación requiere ${inversion_total:.2f}, pero solo tienes ${capital_total}. Reduce el Stop Loss o acepta menos acciones.")
        else:
            st.success(f"✅ LUZ VERDE: Operación segura. Compra {int(cantidad_acciones)} acciones de {activo_op}.")
            
            # Botón de IA para validar el plan
            if st.button(f"🧠 Pedir confirmación final a IA sobre {activo_op}"):
                with st.spinner("Auditando plan de trading..."):
                    try:
                        news = feedparser.parse(f"https://news.google.com/rss/search?q={activo_op}+stock")
                        titulares = [e.title for e in news.entries[:3]]
                        prompt = f"""
                        Revisa este Plan de Trading para {activo_op}:
                        - Entrada: ${precio_actual}
                        - Stop Loss: ${stop_loss_sugerido} (Técnico basado en ATR)
                        - Noticias recientes: {titulares}
                        
                        ¿Apruebas este trade? Responde corto: SI/NO y POR QUÉ.
                        """
                        res = model.generate_content(prompt)
                        st.info(res.text)
                        # Enviar ficha a Telegram
                        msg = f"🛡️ PLAN V24.7\nActivo: {activo_op}\nComprar: {int(cantidad_acciones)} acciones\nStop: {stop_loss_sugerido}\nIA: {res.text}"
                        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
                    except: st.error("Error conectando IA")

        # Gráfico con Niveles Visuales
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_op.index, open=df_op['Open'], high=df_op['High'], low=df_op['Low'], close=df_op['Close'], name='Precio'))
        # Línea de Stop Loss
        fig.add_hline(y=stop_loss_sugerido, line_dash="dash", line_color="red", annotation_text="STOP LOSS (Salida)", annotation_position="bottom right")
        # Línea de Take Profit
        fig.add_hline(y=take_profit_sugerido, line_dash="dash", line_color="green", annotation_text="TAKE PROFIT (Meta)", annotation_position="top right")
        
        fig.update_layout(height=500, template="plotly_dark", title=f"Niveles de Riesgo para {activo_op}")
        st.plotly_chart(fig, use_container_width=True)
