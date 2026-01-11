import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests 
import google.generativeai as genai
import feedparser
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# --- 🔐 CREDENCIALES ---
TELEGRAM_TOKEN = "8042406069:AAHhflfkySyQVhCkHaqIsUjGumFr3fsnDPM" 
TELEGRAM_CHAT_ID = "6288094504"
GOOGLE_API_KEY = "AIzaSyB356Wjicaf9VRUYTX6_EL728IQF6nOmuQ" 

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Sistema Quant V25 (Institutional)", layout="wide", page_icon="🏛️")
st.markdown("""
<style>
    .metric-card {background-color: #1e1e1e; border: 1px solid #333; border-radius: 8px; padding: 15px; color: white;}
    .stDataFrame {border: 1px solid #444; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except: pass

# --- ACTIVOS ---
WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'MELI', 'BTC-USD', 'ETH-USD', 'COIN', 'KO', 'DIS', 'JPM']

st.title("🏛️ Sistema Quant V25: Institutional Suite")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Mesa de Operaciones")
    capital_total = st.number_input("Capital ($)", value=2000)
    riesgo_pct = st.slider("Riesgo (%)", 0.5, 3.0, 2.0)
    st.divider()
    st.info("💡 Novedad V25: Análisis Fundamental y Correlaciones.")

# --- MOTORES DE DATOS ---
@st.cache_data(ttl=1800) # Caché 30 min para fundamentales
def obtener_datos_completos(tickers):
    # 1. Datos Técnicos (Precios)
    try:
        df_prices = yf.download(" ".join(tickers), period="1y", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
    except: return None, None

    resumen = []
    
    # Barra de progreso para sensación premium
    progress_bar = st.progress(0)
    step = 1.0 / len(tickers)
    
    for i, t in enumerate(tickers):
        try:
            # Manejo de datos multicapa
            if len(tickers) > 1:
                if t not in df_prices.columns.levels[0]: continue
                df = df_prices[t].copy().dropna()
            else: df = df_prices.copy().dropna()

            if len(df) < 50: continue
            
            # --- A. ANÁLISIS TÉCNICO ---
            last_close = df['Close'].iloc[-1]
            rsi = ta.rsi(df['Close'], 14).iloc[-1]
            ema200 = ta.ema(df['Close'], 200).iloc[-1]
            atr = ta.atr(df['High'], df['Low'], df['Close'], 14).iloc[-1]
            
            # Tendencia Fractal (Diario vs Semanal aprox)
            trend_d = "ALCISTA" if last_close > ema200 else "BAJISTA"
            mom_1w = "ALCISTA" if df['Close'].iloc[-1] > df['Close'].iloc[-5] else "BAJISTA"
            
            # --- B. ANÁLISIS FUNDAMENTAL (LIGERO) ---
            # Nota: yf.Ticker(t).info es lento si se hace en bucle. 
            # Aquí usamos aproximaciones o llamadas selectivas para no bloquear.
            # Para V25, calcularemos Volatilidad como proxy de riesgo fundamental inmediato
            volatilidad = df['Close'].pct_change().std() * np.sqrt(252) * 100
            
            score = 50
            if trend_d == "ALCISTA": score += 20
            if mom_1w == "ALCISTA": score += 10
            if rsi < 30: score += 30
            elif rsi > 70: score -= 20
            
            resumen.append({
                "Ticker": t,
                "Precio": last_close,
                "RSI": rsi,
                "Tendencia": trend_d,
                "Momentum (1S)": mom_1w,
                "Volatilidad %": volatilidad,
                "ATR": atr,
                "Score": score
            })
        except: pass
        progress_bar.progress(min((i + 1) * step, 1.0))
        
    progress_bar.empty()
    return pd.DataFrame(resumen), df_prices

@st.cache_data(ttl=3600)
def obtener_fundamental_profundo(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "PER": info.get('trailingPE', 0),
            "PEG": info.get('pegRatio', 0), # < 1 es infravalorado
            "P/B": info.get('priceToBook', 0),
            "ROA": info.get('returnOnAssets', 0) * 100,
            "Beta": info.get('beta', 1),
            "Target Price": info.get('targetMeanPrice', 0),
            "Sector": info.get('sector', 'N/A')
        }
    except: return None

# --- EJECUCIÓN ---
df_radar, df_raw_prices = obtener_datos_completos(WATCHLIST)

# --- PESTAÑAS V25 ---
tabs = st.tabs(["📡 Radar & Fundamental", "🕸️ Correlaciones (Riesgo)", "🛡️ Operativa", "🧪 Laboratorio"])

with tabs[0]:
    if df_radar is not None and not df_radar.empty:
        col_main, col_detail = st.columns([2, 1])
        
        with col_main:
            st.subheader("Radar de Oportunidades (Técnico)")
            df_show = df_radar.sort_values("Score", ascending=False)
            
            def color_trend(val):
                return 'color: lightgreen' if val == "ALCISTA" else 'color: #ffcccb'
            
            st.dataframe(
                df_show.style.applymap(color_trend, subset=['Tendencia', 'Momentum (1S)'])
                             .format({"Precio": "${:.2f}", "RSI": "{:.1f}", "Volatilidad %": "{:.1f}%"}),
                use_container_width=True, height=400
            )
            
        with col_detail:
            st.subheader("🔬 Rayos X Fundamental")
            sel_fund = st.selectbox("Analizar Fundamental:", df_radar['Ticker'].tolist())
            
            if st.button(f"🔍 Escanear {sel_fund}"):
                with st.spinner("Descargando balance contable..."):
                    fund_data = obtener_fundamental_profundo(sel_fund)
                    if fund_data:
                        st.markdown(f"**Sector:** {fund_data['Sector']}")
                        
                        m1, m2 = st.columns(2)
                        m1.metric("PER (Precio/Beneficio)", f"{fund_data['PER']:.2f}", "Menor es mejor" if fund_data['PER'] < 20 else "Caro")
                        m2.metric("PEG (Crecimiento)", f"{fund_data['PEG']:.2f}", "Infravalorado" if 0 < fund_data['PEG'] < 1 else "Normal")
                        
                        m3, m4 = st.columns(2)
                        target = fund_data['Target Price']
                        price = df_radar[df_radar['Ticker']==sel_fund]['Precio'].values[0]
                        upside = ((target - price) / price) * 100
                        
                        m3.metric("Precio Objetivo Analystas", f"${target}", f"{upside:.1f}% Potencial")
                        m4.metric("Beta (Volatilidad)", f"{fund_data['Beta']:.2f}")
                        
                        st.caption("Nota: PEG < 1 suele indicar que la acción está barata respecto a su crecimiento.")
                    else: st.error("Datos no disponibles")

with tabs[1]:
    st.subheader("🕸️ Matriz de Correlación (Gestión de Diversificación)")
    st.info("Evita comprar activos que se mueven igual (Correlación > 0.8). Busca diversificar.")
    
    if df_raw_prices is not None:
        # Extraer solo precios de cierre para correlación
        close_prices = pd.DataFrame()
        for t in WATCHLIST:
            try:
                if t in df_raw_prices.columns.levels[0]:
                    close_prices[t] = df_raw_prices[t]['Close']
            except: pass
            
        if not close_prices.empty:
            corr_matrix = close_prices.corr()
            
            fig_corr = px.imshow(
                corr_matrix, 
                text_auto=".2f", 
                aspect="auto", 
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Lógica de advertencia
            high_corr = corr_matrix.unstack().sort_values(ascending=False)
            high_corr = high_corr[(high_corr < 1.0) & (high_corr > 0.85)].drop_duplicates()
            
            if not high_corr.empty:
                st.warning(f"⚠️ **Alerta de Riesgo:** Los siguientes pares son casi idénticos. No tengas ambos:")
                st.write(high_corr.head(5))

with tabs[2]:
    st.subheader("🛡️ Calculadora de Trading (V25)")
    if not df_radar.empty:
        # Selector inteligente
        tk_op = st.selectbox("Activo a Operar", df_radar['Ticker'].tolist(), key="op_sel")
        row = df_radar[df_radar['Ticker'] == tk_op].iloc[0]
        
        c1, c2, c3 = st.columns(3)
        entry = row['Precio']
        stop = entry - (2 * row['ATR'])
        
        # Riesgo
        risk_amt = capital_total * (riesgo_pct / 100)
        shares = risk_amt / (entry - stop)
        
        c1.metric("1. Precio Entrada", f"${entry:.2f}")
        c2.metric("2. Stop Loss (Técnico)", f"${stop:.2f}")
        c3.metric("3. TAMAÑO ORDEN", f"{int(shares)} Acciones", f"Riesgo: ${risk_amt:.2f}")
        
        st.write("---")
        if st.button(f"🧠 JUEZ IA: ¿Comprar {tk_op}?"):
            with st.spinner("Analizando Técnico + Fundamental + Noticias..."):
                info = obtener_fundamental_profundo(tk_op)
                peg = info['PEG'] if info else "N/A"
                prompt = f"""
                Analiza compra de {tk_op}.
                Técnico: RSI {row['RSI']}, Tendencia {row['Tendencia']}.
                Fundamental: PEG Ratio {peg}.
                
                Responde:
                1. ¿Es coherente el Técnico con el Fundamental?
                2. Veredicto Final (SI/NO).
                """
                try:
                    res = model.generate_content(prompt)
                    st.success(res.text)
                    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                                  json={"chat_id": TELEGRAM_CHAT_ID, "text": f"🤖 JUEZ V25: {res.text}"})
                except: st.error("Error IA")

with tabs[3]:
    st.write("🧪 El Laboratorio sigue disponible en segundo plano.")
