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
import google.generativeai as genai
from fpdf import FPDF
import base64

# --- CONFIGURACIÓN ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sistema Quant V73 (Executive)", layout="wide", page_icon="👔")

st.markdown("""<style>
    .metric-card {background-color: #0e1117; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;}
    .thesis-card {background-color: #1a1a2e; border-left: 4px solid #7b2cbf; padding: 20px; border-radius: 8px;}
    .pdf-btn {text-align: center; margin-top: 20px;}
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

# --- MOTOR DE REPORTES PDF (BLINDADO V73) ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Informe Ejecutivo Quant', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

def clean_text(text):
    """Elimina emojis y caracteres no soportados por Latin-1"""
    if not isinstance(text, str): return str(text)
    # Reemplazos manuales de emojis usados en el sistema
    replacements = {
        "🟢": "[+]", "🔴": "[-]", "🟡": "[=]", "🚀": "(UP)", 
        "💎": "(VAL)", "🛡️": "(SAFE)", "⚠️": "(!)", "✅": "[OK]", "❌": "[NO]"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # Sanitización final: fuerza compatibilidad Latin-1, reemplaza errores con '?'
    return text.encode('latin-1', 'replace').decode('latin-1')

def generar_pdf_analisis(ticker, precio, tesis, metricas_clave, prediccion):
    pdf = PDFReport()
    pdf.add_page()
    
    # Título
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f'Analisis: {clean_text(ticker)}', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'L')
    pdf.line(10, 30, 200, 30)
    
    # 1. Snapshot
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1. Datos de Mercado', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 10, f"Precio Actual: ${precio:.2f}", 0, 1)
    if metricas_clave:
        pdf.cell(0, 10, f"RSI (14): {metricas_clave.get('RSI', 'N/A')}", 0, 1)
        pdf.cell(0, 10, f"Beta: {metricas_clave.get('Beta', 'N/A')}", 0, 1)
        pdf.cell(0, 10, f"Target Promedio: ${metricas_clave.get('Target', 0):.2f}", 0, 1)
    
    # 2. Tesis
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '2. Tesis de Inversion', 0, 1)
    
    # Veredicto (Limpiado)
    veredicto_clean = clean_text(tesis['Veredicto'])
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f"VEREDICTO: {veredicto_clean}", 0, 1)
    
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 8, "Factores Positivos:", 0, 1)
    for p in tesis['Pros']: pdf.cell(0, 5, f" + {clean_text(p)}", 0, 1)
    
    pdf.ln(2)
    pdf.cell(0, 8, "Factores Negativos:", 0, 1)
    for c in tesis['Contras']: pdf.cell(0, 5, f" - {clean_text(c)}", 0, 1)
    
    # 3. Proyección
    if prediccion:
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '3. Proyeccion Estadistica (30d)', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 10, f"Precio Esperado: ${prediccion['Mean_Price']:.2f}", 0, 1)
        pdf.cell(0, 10, f"Probabilidad Suba: {prediccion['Prob_Suba']:.1f}%", 0, 1)
        pdf.cell(0, 10, f"Escenario Pesimista (VaR): ${prediccion['VaR_95']:.2f}", 0, 1)

    # Disclaimer
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 8)
    pdf.multi_