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
st.set_page_config(page_title="Sistema Quant V46 (Stress Test)", layout="wide", page_icon="🌪️")
st.markdown("""
<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 10px; color: white;}
    .signal-box {border: 2px solid #FFD700; padding: 10px; border-radius: 5px; background-color: #2b2b00; text-align: center;}
    .macro-card {background-color: #1e2130; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #444;}
    .crash-card {background-color: #3b1010; padding: 15px; border-radius: 8px; border: 1px solid #ff0000; text-align: center;}
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

# --- MOTOR STRESS TEST (NUEVO V46) ---
def calcular_beta_portafolio(df_pos):
    """Calcula el Beta ponderado del portafolio actual"""
    if df_pos.empty: return 1.0
    
    tickers = df_pos['Ticker'].tolist()
    # Descargamos historial de 1 año junto con SPY (Benchmark)
    tickers_query = tickers + ['SPY']
    try:
        data = yf.download(" ".join(tickers_query), period="1y", progress=False, auto_adjust=True)['Close']
        returns = data.pct_change().dropna()
        
        # Beta de cada activo
        betas = {}
        for t in tickers:
            cov = returns[t].cov(returns['SPY'])
            var = returns['SPY'].var()
            betas[t] = cov / var
            
        # Beta Ponderado (Weighted Average Beta)
        total_value = df_pos['Valor Mercado'].sum()
        portfolio_beta = 0
        for idx, row in df_pos.iterrows():
            weight = row['Valor Mercado'] / total_value
            portfolio_beta += betas.get(row['Ticker'], 1.0) * weight # Si falla, asumimos Beta 1
            
        return portfolio_beta
    except: return 1.0 # Fallback

def ejecutar_stress_test(df_pos):
    """Simula escenarios de crisis"""
    beta = calcular_beta_portafolio(df_pos)
    total_equity = df_pos['Valor Mercado'].sum()
    
    # Definición de Escenarios Históricos (Caída del Mercado)
    escenarios = {
        "Crisis COVID-19 (2020)": -0.34, # S&P cayó 34%
        "Crisis Subprime (2008)": -0.57, # S&P cayó 57%
        "Burbuja Dotcom (2000)": -0.49,  # Nasdaq cayó más, pero S&P ~49%
        "Lunes Negro (1987)": -0.20,     # Caída en un día
        "Corrección Menor": -0.10        # 10% Típico
    }
    
    resultados = []
    for nombre, caida_mercado in escenarios.items():
        # Impacto estimado = Caída Mercado * Beta del Portafolio
        impacto_pct = caida_mercado * beta
        perdida_usd = total_equity * impacto_pct
        equity_final = total_equity + perdida_usd
        
        resultados.append({
            "Escenario": nombre,
            "Caída Mercado": f"{caida_mercado*100:.1f}%",
            "Impacto Portafolio": f"{impacto_pct*100:.1f}%",
            "Pérdida ($)": perdida_usd,
            "Equity Final": equity_final
        })
        
    return pd.DataFrame(resultados), beta

# --- MOTOR PDF ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15); self.cell(0, 10, 'Informe Quant V45', 0, 1, 'C'); self.ln(5)
    def footer(self):
        self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Pag {self.page_no()}', 0, 0, 'C')

def generar_pdf_cartera(df_pos):
    pdf = PDFReport(); pdf.add_page(); pdf.set_font("Arial", size=12)
    tot_eq = df_pos['Valor Mercado'].sum(); tot_pnl = df_pos['P&L ($)'].sum()
    pdf.cell(0, 10, f"Patrimonio: USD {tot_eq:,.2f}", 0, 1); pdf.cell(0, 10, f"P&L: USD {tot_pnl:,.2f}", 0, 1); pdf.ln(10)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(30, 10, "Ticker", 1); pdf.cell(30, 10, "Cant", 1); pdf.cell(40, 10, "Val Mkt", 1); pdf.cell(40, 10, "P&L", 1); pdf.ln()
    pdf.set_font("Arial", size=10)
    for i, r in df_pos.iterrows():
        pdf.cell(30, 10, str(r['Ticker']), 1); pdf.cell(30, 10, str(r['Cantidad']), 1)
        pdf.cell(40, 10, f"${r['Valor Mercado']:.2f}", 1); pdf.cell(40, 10, f"${r['P&L ($)']:.2f}", 1); pdf.ln()
    return pdf.output(dest='S').encode('latin-1')

# --- MOTORES EXISTENTES (Resumidos) ---
def analizar_noticias_ia(ticker):
    try:
        rss = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"; feed = feedparser.parse(rss)
        if not feed.entries: return None
        heads = [e.title for e in feed.entries[:5]]; links = [e.link for e in feed.entries[:5]]
        prompt = f"Analiza: {heads}. Output: SENTIMIENTO:[Score] RESUMEN:[Text]"; resp = model.generate_content(prompt).text
        sc = 0; sm = resp
        for l in resp.split('\n'):
            if "SENTIMIENTO:" in l: sc = int(l.split(":")[1].strip())
            if "RESUMEN:" in l: sm = l.split("RESUMEN:")[1].strip()
        return {"headlines": heads, "links": links, "score": sc, "summary": sm}
    except: return None

@st.cache_data(ttl=3600)
def calcular_valor_intrinseco(ticker):
    try:
        i = yf.Ticker(ticker).info; p = i.get('currentPrice', 0); e = i.get('trailingEps'); b = i.get('bookValue')
        g = math.sqrt(22.5*e*b) if e and b and e>0 and b>0 else 0; s = "INFRAVALORADA 🟢" if g>p else "SOBREVALORADA 🔴"
        return {"Precio": p, "Graham": g, "Status_Graham": s, "Diff": ((g-p)/p)*100}
    except: return None

def obtener_datos_macro():
    tk = list(MACRO_TICKERS.values())
    try:
        df = yf.download(" ".join(tk), period="2d", progress=False, group_by='ticker', auto_adjust=True); res = {}
        for n, t in MACRO_TICKERS.items():
            if len(tk)>1: p=df[t]['Close'].iloc[-1]; pr=df[t]['Close'].iloc[-2]
            else: p=df['Close'].iloc[-1]; pr=df['Close'].iloc[-2]
            res[n] = (p, ((p-pr)/pr)*100)
        return res
    except: return None

def calcular_alpha_beta(ticker, benchmark='SPY'):
    try:
        d = yf.download(f"{ticker} {benchmark}", period="1y", progress=False, auto_adjust=True)['Close']
        if d.empty: return None, None
        r = d.pct_change().dropna(); b = r[ticker].cov(r[benchmark])/r[benchmark].var()
        n = (d/d.iloc[0])*100; a = (n[ticker].iloc[-1]-100)-(n[benchmark].iloc[-1]-100)
        return n, {"Beta": b, "Alpha Total %": a}
    except: return None, None

def simular_montecarlo(ticker, dias=30, sims=500):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)['Close']
        lr = np.log(1+df.pct_change()); u=lr.mean().item(); v=lr.var().item(); dr=u-(0.5*v); st=lr.std().item(); la=df.iloc[-1].item()
        dr = np.exp(dr+st*norm.ppf(np.random.rand(dias, sims))); pl=np.zeros_like(dr); pl[0]=la
        for t in range(1, dias): pl[t]=pl[t-1]*dr[t]
        fig = go.Figure()
        for i in range(min(50, sims)): fig.add_trace(go.Scatter(y=pl[:, i], mode='lines', line=dict(width=1, color='rgba(0,255,255,0.1)'), showlegend=False))
        m = np.mean(pl, axis=1); fig.add_trace(go.Scatter(y=m, mode='lines', line=dict(width=3, color='yellow'), name='Promedio'))
        p05 = np.percentile(pl, 5, axis=1); fig.add_trace(go.Scatter(y=p05, mode='lines', line=dict(width=1, color='red', dash='dash'), name='Pesimista'))
        fig.update_layout(template="plotly_dark", height=400, title=f"Montecarlo {ticker}")
        return fig, {"esperado": m[-1], "pesimista": p05[-1]}
    except: return None, None

@st.cache_data(ttl=600)
def generar_mapa_calor(tickers):
    try:
        d = yf.download(" ".join(tickers), period="5d", interval="1d", progress=False, auto_adjust=True)['Close']
        pct = ((d.iloc[-1]-d.iloc[-2])/d.iloc[-2])*100
        df = pd.DataFrame({'Ticker': pct.index, 'Variacion': pct.values, 'Precio': d.iloc[-1].values})
        sec = []
        for t in df['Ticker']:
            if t in ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'META']: sec.append('Tech')
            elif t in ['BTC-USD', 'ETH-USD']: sec.append('Cripto')
            else: sec.append('Otros')
        df['Sector'] = sec; df['Size'] = df['Precio']; return df
    except: return None

def graficar_master(ticker):
    try:
        stock = yf.Ticker(ticker); df = stock.history(period="1y", interval="1d", auto_adjust=True)
        if df.empty: return None
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df['EMA20'] = ta.ema(df['Close'], 20); df['RSI'] = ta.rsi(df['Close'], 14)
        bb = ta.bbands(df['Close'], length=20, std=2); df = pd.concat([df, bb], axis=1)
        sop = sorted([s for s in df['Low'].rolling(10).min().iloc[-20:].unique() if (df['Close'].iloc[-1]-s)/df['Close'].iloc[-1] < 0.15])[-2:]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"), row=1, col=1)
        if 'EMA20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='yellow'), name="EMA 20"), row=1, col=1)
        try: fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, -3], line=dict(color='cyan', dash='dot'), name="Upper"), row=1, col=1); fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, -1], line=dict(color='cyan', dash='dot'), name="Lower"), row=1, col=1)
        except: pass
        for s in sop: fig.add_hline(y=s, line_dash="dot", line_color="green", row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_color="red", row=2, col=1); fig.add_hline(y=30, line_color="green", row=2, col=1)
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False); return fig
    except: return None

def ejecutar_backtest_pro(ticker, capital, estrategia, params):
    try:
        df = yf.Ticker(ticker).history(period="3y", interval="1d", auto_adjust=True)
        if df.empty: return None
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        cash = capital; pos = 0; log = []; eq = []; peak = capital; dd = 0
        if estrategia == "Bollinger (600% Mode)":
            bb = ta.bbands(df['Close'], length=20, std=2); df = pd.concat([df, bb], axis=1)
            buy = lambda i: df['Close'].iloc[i] < df.iloc[i, -3]; sell = lambda i: df['Close'].iloc[i] > df.iloc[i, -1]; start = 25
        elif estrategia == "RSI":
            df['RSI'] = ta.rsi(df['Close'], 14); buy = lambda i: df['RSI'].iloc[i] < 30; sell = lambda i: df['RSI'].iloc[i] > 70; start = 20
        for i in range(start, len(df)):
            p = df['Close'].iloc[i]; d = df.index[i]
            if cash > 0 and buy(i): q = int(cash/p); cash -= q*p; pos += q; log.append({"Fecha": d, "Tipo": "COMPRA", "Precio": p, "Saldo": cash})
            elif pos > 0 and sell(i): cash += pos*p; log.append({"Fecha": d, "Tipo": "VENTA", "Precio": p, "Saldo": cash}); pos = 0
            val = cash + (pos*p); eq.append({"Fecha": d, "Equity": val})
            if val > peak: peak = val
            c_dd = (peak-val)/peak; 
            if c_dd > dd: dd = c_dd
        fin = cash + (pos*df['Close'].iloc[-1]); ret = ((fin-capital)/capital)*100
        return {"retorno": ret, "buy_hold": ((df['Close'].iloc[-1]-df['Close'].iloc[start])/df['Close'].iloc[start])*100, "trades": len(log), "max_drawdown": dd*100, "equity_curve": pd.DataFrame(eq).set_index("Fecha")}
    except: return None

# --- INTERFAZ ---
st.title("🌪️ Sistema Quant V46: The Stress Tester")

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
        if st.button("🖨️ PDF REPORT"):
            pdf = generar_pdf_cartera(df_pos)
            st.download_button("📥 Descargar", pdf, "reporte.pdf", "application/pdf")
    else: st.warning("Portafolio Vacío.")

    with st.expander("📝 Operar"):
        op_tk = st.selectbox("Ticker", WATCHLIST, key='op_tk')
        op_type = st.selectbox("Tipo", ["COMPRA", "VENTA"])
        op_qty = st.number_input("Qty", 1, 1000)
        op_px = st.number_input("Precio", 0.0)
        if st.button("Ejecutar"):
            registrar_operacion_sql(op_tk, op_type, op_qty, op_px); st.rerun()

with c_right:
    tabs = st.tabs(["🌪️ Stress Test", "📰 IA News", "💎 Valuación", "🆚 Alpha", "🔮 Monte Carlo", "📈 Gráfico", "🔥 Heatmap"])
    
    # PESTAÑA 1: STRESS TEST (NUEVA V46)
    with tabs[0]:
        st.subheader("🌪️ Simulación de Crisis Históricas")
        
        if not df_pos.empty:
            df_stress, beta_port = ejecutar_stress_test(df_pos)
            
            st.info(f"📊 **Beta de tu Portafolio:** {beta_port:.2f}")
            if beta_port > 1.2: st.warning("⚠️ Tu cartera es AGRESIVA. Sufrirás más que el mercado en las caídas.")
            elif beta_port < 0.8: st.success("🛡️ Tu cartera es DEFENSIVA. Resistirás mejor las crisis.")
            
            # Gráfico de Impacto
            fig_stress = px.bar(df_stress, x="Escenario", y="Pérdida ($)", 
                                title="Pérdida Estimada en Dólares ($)",
                                text_auto='.2s', color="Pérdida ($)", color_continuous_scale="Reds_r")
            st.plotly_chart(fig_stress, use_container_width=True)
            
            st.dataframe(df_stress)
            
        else:
            st.warning("Necesitas tener acciones en tu portafolio (panel izquierdo) para hacer un Stress Test.")

    with tabs[1]:
        st.subheader(f"📰 Sentimiento: {tk}")
        if st.button("🤖 ANALIZAR"):
            with st.spinner("Analizando..."):
                news_data = analizar_noticias_ia(tk)
                if news_data:
                    st.info("💡 " + news_data.get('summary', ''))
                    st.markdown("### Titulares")
                    for i, head in enumerate(news_data['headlines']): st.markdown(f"- [{head}]({news_data['links'][i]})")

    with tabs[2]:
        val_data = calcular_valor_intrinseco(tk)
        if val_data:
            c1, c2 = st.columns(2)
            c1.metric("Precio", f"${val_data['Precio']:.2f}")
            c1.metric("Graham", f"${val_data['Graham']:.2f}", f"{val_data['Diff']:.1f}%")
            if val_data['Status_Graham'] == "INFRAVALORADA 🟢": st.success("BARATA")
            else: st.error("CARA")

    with tabs[3]:
        norm_data, metrics = calcular_alpha_beta(tk)
        if norm_data is not None:
            c1, c2 = st.columns(2)
            c1.metric("Beta", f"{metrics['Beta']:.2f}")
            c2.metric("Alpha", f"{metrics['Alpha Total %']:.2f}%")
            st.plotly_chart(px.line(norm_data, x=norm_data.index, y=norm_data.columns), use_container_width=True)

    with tabs[4]:
        dias_mc = st.slider("Días", 10, 90, 30)
        if st.button("🎲 Simular"):
            fig_mc, res_mc = simular_montecarlo(tk, dias_mc)
            if fig_mc: st.plotly_chart(fig_mc, use_container_width=True)

    with tabs[5]:
        fig = graficar_master(tk)
        if fig: st.plotly_chart(fig, use_container_width=True)
        
    with tabs[6]:
        df_map = generar_mapa_calor(WATCHLIST)
        if df_map is not None:
            fig_map = px.treemap(df_map, path=['Sector', 'Ticker'], values='Size', color='Variacion', color_continuous_scale='RdYlGn', color_continuous_midpoint=0)
            st.plotly_chart(fig_map, use_container_width=True)