"""
🏦 Loan Approval Prediction System
====================================
Premium Multi-Page Streamlit Dashboard
Interactive ML-powered loan risk assessment platform

Author: Kinshunk Garg
GitHub: https://github.com/Kinshunk565
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Loan Intelligence | Premium ML Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PATHS ---
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# --- PREMIUM VIBRANT LIGHT CSS ---
st.markdown("""
<style>
    /* ===== GLOBAL STYLES ===== */
    @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700;800;900&display=swap');
    
    .stApp {
        font-family: 'Lexend', sans-serif;
        background-color: #f8fafc;
        color: #1e293b;
    }
    
    /* Elegant Sidebar (Deep Violet/Emerald) */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #064e3b 100%);
        box-shadow: 4px 0 15px rgba(0,0,0,0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] span {
        color: #f1f5f9 !important;
    }

    [data-testid="stSidebar"] hr {
        border-top-color: rgba(255,255,255,0.1) !important;
    }
    
    /* ===== HERO HEADER (Vibrant Mesh Flow) ===== */
    .hero-container {
        background: linear-gradient(-45deg, #7c3aed, #10b981, #f43f5e, #8b5cf6);
        background-size: 400% 400%;
        animation: gradientFlow 15s ease infinite;
        border-radius: 32px;
        padding: 65px;
        margin-bottom: 50px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 30px 60px -12px rgba(124, 58, 237, 0.3);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .hero-content {
        position: relative;
        z-index: 2;
        background: rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(12px);
        padding: 40px;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        width: fit-content;
        animation: fadeIn 1s ease-out;
    }

    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .hero-title {
        font-size: 3.8rem;
        font-weight: 950;
        color: #ffffff;
        margin: 0;
        line-height: 1.1;
        letter-spacing: -2.5px;
        text-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: slideInLeft 0.8s cubic-bezier(0.23, 1, 0.32, 1);
    }
    
    .hero-title span {
        color: #10b981;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
    }

    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-40px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .hero-subtitle {
        font-size: 1.4rem;
        color: rgba(255,255,255,0.95);
        margin-top: 25px;
        font-weight: 400;
        max-width: 800px;
        line-height: 1.6;
        animation: fadeIn 1.2s ease-out 0.3s both;
    }
    
    /* ===== PREMIUM CARDS (Glass Glow) ===== */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 24px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeIn 0.8s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-12px) scale(1.02);
        box-shadow: 0 25px 50px -12px rgba(16, 185, 129, 0.2);
        border-color: #10b981;
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #059669 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-top: 12px;
        font-weight: 700;
    }
    
    /* ===== PREDICTION COMPONENTS ===== */
    .prediction-approved {
        background: linear-gradient(135deg, #ecfdf5 0%, #ffffff 100%);
        border: 3px solid #10b981;
        border-radius: 30px;
        padding: 50px;
        text-align: center;
        box-shadow: 0 20px 40px rgba(16, 185, 129, 0.15);
        animation: float 5s ease-in-out infinite;
    }
    
    .prediction-rejected {
        background: linear-gradient(135deg, #fff1f2 0%, #ffffff 100%);
        border: 3px solid #f43f5e;
        border-radius: 30px;
        padding: 50px;
        text-align: center;
        box-shadow: 0 20px 40px rgba(244, 63, 94, 0.15);
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-15px) rotate(0.5deg); }
    }
    
    .prediction-status {
        font-size: 4rem;
        font-weight: 950;
        color: #064e3b;
        margin: 15px 0;
        letter-spacing: -2px;
    }
    
    .prediction-confidence {
        font-size: 1.4rem;
        color: #64748b;
        font-weight: 500;
    }
    
    /* ===== FACTOR CARDS ===== */
    .factor-positive {
        background: #f0fdf4;
        border-left: 5px solid #22c55e;
        border-radius: 14px;
        padding: 18px 24px;
        margin: 12px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    
    .factor-negative {
        background: #fef2f2;
        border-left: 5px solid #ef4444;
        border-radius: 14px;
        padding: 18px 24px;
        margin: 12px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    
    .factor-neutral {
        background: #fffbeb;
        border-left: 5px solid #f59e0b;
        border-radius: 14px;
        padding: 18px 24px;
        margin: 12px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    
    /* ===== SECTION HEADERS (Animated Accent) ===== */
    .section-header {
        font-size: 2.2rem;
        font-weight: 900;
        color: #1e293b;
        margin: 50px 0 30px 0;
        display: flex;
        align-items: center;
        gap: 15px;
        animation: slideInLeft 0.8s ease-out;
    }
    
    .section-header::before {
        content: '';
        width: 15px;
        height: 15px;
        background: #10b981;
        border-radius: 4px;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.5);
    }
    
    /* ===== FOOTER (Premium) ===== */
    .footer {
        text-align: center;
        padding: 60px;
        margin-top: 100px;
        background: #ffffff;
        border-top: 1px solid #f1f5f9;
        color: #64748b;
    }
    
    .footer a {
        color: #7c3aed;
        text-decoration: none;
        font-weight: 800;
        transition: all 0.3s ease;
    }

    .footer a:hover {
        color: #10b981;
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
    }

    /* Streamlit Overrides (Professional) */
    .stButton > button {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 30px !important;
        border-radius: 50px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-5px) scale(1.02) !important;
        box-shadow: 0 12px 30px rgba(16, 185, 129, 0.45) !important;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    }

    .stButton > button:active {
        transform: translateY(2px) !important;
    }

    /* Input Fields (Professional) */
    .stTextInput input, .stSelectbox [data-baseweb="select"], .stNumberInput input {
        border-radius: 12px !important;
        border: 2px solid #f1f5f9 !important;
        background-color: white !important;
        padding: 10px 15px !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput input:focus, .stSelectbox [data-baseweb="select"]:focus {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.1) !important;
    }

    .stMetric {
        background: white;
        padding: 24px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.03);
        border: 1px solid #f1f5f9;
        transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .stMetric:hover { transform: translateY(-8px) scale(1.05); }

    /* Sidebar Radio Styling */
    [data-testid="stSidebarNav"] { padding-top: 20px; }
    
    div[data-testid="stSidebarUserContent"] .stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding: 20px 10px;
    }

    div[data-testid="stSidebarUserContent"] .stRadio label {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
        color: #94a3b8 !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        width: 100% !important;
    }

    div[data-testid="stSidebarUserContent"] .stRadio label:hover {
        background: rgba(16, 185, 129, 0.1) !important;
        border-color: #10b981 !important;
        color: white !important;
        transform: translateX(5px);
    }

    div[data-testid="stSidebarUserContent"] .stRadio div[data-testid="stMarkdownContainer"] p {
        font-weight: 600 !important;
    }

    /* Active State for Sidebar Radio */
    [data-testid="stSidebarNav"] .st-ea, div[data-testid="stSidebarUserContent"] .stRadio label[data-checked="true"] {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
    }

    /* Hide default Streamlit elements */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --- UTILITY FUNCTIONS ---
@st.cache_data
def load_dataset():
    """Load the loan dataset."""
    path = os.path.join(DATA_DIR, "loan_data.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_model_metrics():
    """Load saved model metrics."""
    path = os.path.join(MODELS_DIR, "model_metrics.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_feature_importance():
    """Load feature importance."""
    path = os.path.join(MODELS_DIR, "feature_importance.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

@st.cache_resource
def load_predictor():
    """Load the prediction engine."""
    try:
        from src.predict import LoanPredictor
        return LoanPredictor(MODELS_DIR)
    except Exception as e:
        st.error(f"Failed to load predictor: {e}")
        return None


def create_gauge_chart(value, title="Approval Probability"):
    """Create a semicircular gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18, 'color': '#ccd6f6'}},
        number={'suffix': '%', 'font': {'size': 40, 'color': '#ccd6f6'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': '#8892b0',
                     'tickfont': {'color': '#8892b0'}},
            'bar': {'color': '#667eea', 'thickness': 0.3},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(244, 63, 94, 0.3)'},
                {'range': [30, 55], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [55, 75], 'color': 'rgba(16, 185, 129, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(5, 150, 105, 0.3)'}
            ],
            'threshold': {
                'line': {'color': '#7c3aed', 'width': 4},
                'thickness': 0.8,
                'value': value
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1e293b'},
        height=280,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


# === SIDEBAR NAVIGATION ===
st.sidebar.markdown("""
<div style="text-align: center; padding: 25px 0;">
    <div style="font-size: 3.5rem; filter: drop-shadow(0 0 15px rgba(16, 185, 129, 0.6));">🍃</div>
    <div style="font-size: 1.4rem; font-weight: 800; color: #ffffff; margin-top: 10px; letter-spacing: -0.5px;">
        EcoLoan Intel
    </div>
    <div style="font-size: 0.75rem; color: #10b981; letter-spacing: 3px; font-weight: 700;">
        PREMIUM GROWTH DASHBOARD
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 EDA & Insights", "🤖 Model Performance", "🔮 Live Prediction", "ℹ️ About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px;">
    <div style="font-size: 0.7rem; color: #8892b0;">
        Built by <span style="color: #667eea; font-weight: 600;">Kinshunk Garg</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PAGE 1: HOME
# ============================================================
if page == "🏠 Home":
    # Hero
    st.markdown("""
    <div class="hero-container">
        <div class="hero-content">
            <p class="hero-title">Smart <span>Loan<br>Intelligence</span> System</p>
            <p class="hero-subtitle">
                Unlock higher precision with our next-generation credit risk assessment engine. 
                Experience real-time applicant profiling powered by sophisticated Machine Learning.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics Row
    metrics = load_model_metrics()
    df = load_dataset()
    
    col1, col2, col3, col4 = st.columns(4)
    
    dataset_size = len(df) if df is not None else 0
    best_model = metrics.get('best_model', 'N/A') if metrics else 'N/A'
    best_acc = 0
    best_auc = 0
    num_models = 0
    
    if metrics and 'models' in metrics:
        num_models = len(metrics['models'])
        best_data = metrics['models'].get(best_model, {})
        best_acc = best_data.get('accuracy', 0) * 100
        best_auc = best_data.get('auc_roc', 0) * 100
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{dataset_size:,}</div>
            <div class="metric-label">Dataset Records</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_acc:.1f}%</div>
            <div class="metric-label">Best Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_auc:.1f}%</div>
            <div class="metric-label">AUC-ROC Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{num_models}</div>
            <div class="metric-label">Models Trained</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Overview
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown('<div class="section-header">📌 Project Highlights</div>', unsafe_allow_html=True)
        st.markdown("""
        - **5 ML Models** trained with hyperparameter tuning (GridSearchCV)
        - **Feature Engineering**: TotalIncome, EMI, BalanceIncome, Log transforms
        - **5-Fold Stratified Cross-Validation** for robust evaluation
        - **SHAP-style Explainability** — see *why* each decision was made
        - **Production-ready** pipeline with model serialization
        """)
    
    with col_right:
        st.markdown('<div class="section-header">🛠️ Tech Stack</div>', unsafe_allow_html=True)
        st.markdown("""
        | Component | Technology |
        |:---|:---|
        | **Language** | Python 3.10+ |
        | **ML Framework** | Scikit-learn, XGBoost |
        | **Data Processing** | Pandas, NumPy |
        | **Visualization** | Plotly, Matplotlib, Seaborn |
        | **Dashboard** | Streamlit |
        | **Deployment** | Render |
        """)
    
    # Approval Distribution
    if df is not None:
        st.markdown('<div class="section-header">📊 Loan Status Distribution</div>', unsafe_allow_html=True)
        
        status_counts = df['Loan_Status'].value_counts()
        labels = ['Approved' if x == 'Y' else 'Rejected' for x in status_counts.index]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=status_counts.values,
            hole=0.65,
            marker=dict(colors=['#10b981', '#f43f5e']),
            textinfo='label+percent',
            textfont=dict(size=14, color='white')
        )])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1e293b'),
            height=380,
            showlegend=False,
            annotations=[dict(
                text=f"<b>{len(df):,}</b><br>Total",
                x=0.5, y=0.5, font_size=22, showarrow=False,
                font=dict(color='#1e293b')
            )]
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 2: EDA & INSIGHTS
# ============================================================
elif page == "📊 EDA & Insights":
    st.markdown("""
    <div class="hero-container" style="padding: 40px 60px;">
        <div class="hero-content" style="padding: 25px 35px;">
            <p class="hero-title" style="font-size: 2.2rem;">📊 Exploratory <span>Data Analysis</span></p>
            <p class="hero-subtitle" style="font-size: 1.1rem; margin-top: 10px;">Deep dive into the loan dataset — distributions, correlations, and key patterns</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    df = load_dataset()
    if df is None:
        st.error("📂 Dataset not found. Please run the training pipeline first.")
        st.stop()
    
    # --- Dataset Overview ---
    st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", f"{df.shape[1] - 1}")
    with col3:
        approval_rate = (df['Loan_Status'] == 'Y').mean()
        st.metric("Approval Rate", f"{approval_rate:.1%}")
    
    # Missing Values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        st.markdown('<div class="section-header">🔍 Missing Values Analysis</div>', unsafe_allow_html=True)
        
        fig = go.Figure(go.Bar(
            x=missing.index.tolist(),
            y=missing.values.tolist(),
            marker=dict(
                color=missing.values.tolist(),
                colorscale=[[0, '#10b981'], [1, '#7c3aed']],
                showscale=False
            ),
            text=[f"{v} ({v/len(df)*100:.1f}%)" for v in missing.values],
            textposition='outside',
            textfont=dict(color='#1e293b')
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1e293b'),
            xaxis=dict(title='Feature', gridcolor='#e2e8f0'),
            yaxis=dict(title='Missing Count', gridcolor='#e2e8f0'),
            height=350,
            margin=dict(t=30)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Categorical Analysis ---
    st.markdown('<div class="section-header">📊 Approval Rate by Category</div>', unsafe_allow_html=True)
    
    cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents', 'Credit_History']
    
    selected_cat = st.selectbox("Select Feature", cat_cols, key="eda_cat_select")
    
    df_clean = df.dropna(subset=[selected_cat, 'Loan_Status'])
    grouped = df_clean.groupby(selected_cat)['Loan_Status'].apply(
        lambda x: (x == 'Y').mean() * 100
    ).reset_index()
    grouped.columns = [selected_cat, 'Approval Rate (%)']
    counts = df_clean[selected_cat].value_counts().reset_index()
    counts.columns = [selected_cat, 'Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            grouped, x=selected_cat, y='Approval Rate (%)',
            color='Approval Rate (%)',
            color_continuous_scale=['#f43f5e', '#f59e0b', '#6366f1', '#22c55e'],
            text='Approval Rate (%)'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                         textfont_color='#1e293b')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1e293b'),
            xaxis=dict(gridcolor='#f1f5f9'),
            yaxis=dict(gridcolor='#e2e8f0', range=[0, 110]),
            coloraxis_showscale=False,
            height=400,
            title=dict(text=f"Approval Rate by {selected_cat}", font=dict(color='#1e293b'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            counts, values='Count', names=selected_cat,
            color_discrete_sequence=['#10b981', '#7c3aed', '#f43f5e', '#f59e0b', '#064e3b']
        )
        fig.update_traces(textinfo='label+percent', textfont=dict(color='white'))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1e293b'),
            height=400,
            title=dict(text=f"Distribution of {selected_cat}", font=dict(color='#1e293b'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Income Distribution ---
    st.markdown('<div class="section-header">💰 Income & Loan Distribution</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, x='ApplicantIncome', nbins=50,
            color='Loan_Status',
            color_discrete_map={'Y': '#10b981', 'N': '#f43f5e'},
            marginal='box',
            labels={'Loan_Status': 'Status'},
            barmode='overlay',
            opacity=0.7
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1e293b'),
            xaxis=dict(gridcolor='#f1f5f9'),
            yaxis=dict(gridcolor='#e2e8f0'),
            height=400,
            title=dict(text="Applicant Income Distribution", font=dict(color='#1e293b'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            df, x='LoanAmount', nbins=50,
            color='Loan_Status',
            color_discrete_map={'Y': '#667eea', 'N': '#ff6b6b'},
            marginal='box',
            labels={'Loan_Status': 'Status'},
            barmode='overlay',
            opacity=0.7
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ccd6f6'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            height=400,
            title=dict(text="Loan Amount Distribution", font=dict(color='#ccd6f6'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Correlation Heatmap ---
    st.markdown('<div class="section-header">🔗 Feature Correlation Matrix</div>', unsafe_allow_html=True)
    
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    corr = numeric_df.corr()
    
    fig = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale=['#f9fafb', '#10b981', '#7c3aed'],
        aspect='auto'
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1e293b'),
        height=500,
        title=dict(text="Correlation Heatmap", font=dict(color='#1e293b'))
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Feature Importance ---
    fi = load_feature_importance()
    if fi:
        st.markdown('<div class="section-header">⭐ Feature Importance</div>', unsafe_allow_html=True)
        
        fi_df = pd.DataFrame({
            'Feature': list(fi.keys())[:12],
            'Importance': list(fi.values())[:12]
        })
        
        fig = px.bar(
            fi_df, x='Importance', y='Feature', orientation='h',
            color='Importance',
            color_continuous_scale=['#667eea', '#764ba2', '#f093fb']
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ccd6f6'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', autorange='reversed'),
            coloraxis_showscale=False,
            height=450,
            title=dict(text="Top Features by Importance", font=dict(color='#ccd6f6'))
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================
elif page == "🤖 Model Performance":
    st.markdown("""
    <div class="hero-container" style="padding: 40px 60px;">
        <div class="hero-content" style="padding: 25px 35px;">
            <p class="hero-title" style="font-size: 2.2rem;">🤖 Model <span>Performance</span></p>
            <p class="hero-subtitle" style="font-size: 1.1rem; margin-top: 10px;">Compare 5 ML models trained with GridSearchCV and 5-fold cross-validation</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    metrics = load_model_metrics()
    if metrics is None:
        st.error("📂 Model metrics not found. Please run the training pipeline first.")
        st.stop()
    
    best_model = metrics.get('best_model', 'N/A')
    models_data = metrics.get('models', {})
    
    # Best Model Highlight
    st.markdown(f"""
    <div class="metric-card" style="text-align: center; margin-bottom: 30px; border: 2px solid #6366f1;">
        <div style="font-size: 0.9rem; color: #64748b; text-transform: uppercase; letter-spacing: 2.5px; font-weight: 700;">🏆 Champion Model</div>
        <div class="metric-value" style="font-size: 2.8rem;">{best_model}</div>
        <div style="color: #64748b; margin-top: 10px; font-weight: 500;">
            AUC: {models_data.get(best_model, {}).get('auc_roc', 0):.4f} &nbsp;|&nbsp; 
            Accuracy: {models_data.get(best_model, {}).get('accuracy', 0):.4f} &nbsp;|&nbsp; 
            F1: {models_data.get(best_model, {}).get('f1_score', 0):.4f}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison Table
    st.markdown('<div class="section-header">📊 Model Comparison Table</div>', unsafe_allow_html=True)
    
    comparison_data = []
    for name, data in models_data.items():
        comparison_data.append({
            'Model': f"{'🏆 ' if name == best_model else ''}{name}",
            'Accuracy': f"{data.get('accuracy', 0):.4f}",
            'Precision': f"{data.get('precision', 0):.4f}",
            'Recall': f"{data.get('recall', 0):.4f}",
            'F1-Score': f"{data.get('f1_score', 0):.4f}",
            'AUC-ROC': f"{data.get('auc_roc', 0):.4f}",
            'CV Mean ± Std': f"{data.get('cv_mean', 0):.4f} ± {data.get('cv_std', 0):.4f}",
            'Train Time (s)': f"{data.get('train_time_seconds', 0):.1f}"
        })
    
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    # --- Metrics Bar Chart ---
    st.markdown('<div class="section-header">📈 Visual Comparison</div>', unsafe_allow_html=True)
    
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    model_names_list = list(models_data.keys())
    
    fig = go.Figure()
    colors = ['#10b981', '#7c3aed', '#f59e0b', '#f43f5e', '#064e3b']
    
    for i, metric in enumerate(metric_names):
        values = [models_data[m].get(metric, 0) for m in model_names_list]
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=model_names_list,
            y=values,
            marker_color=colors[i],
            text=[f"{v:.3f}" for v in values],
            textposition='outside',
            textfont=dict(color='#1e293b', size=11)
        ))
    
    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1e293b'),
        xaxis=dict(gridcolor='#f1f5f9'),
        yaxis=dict(gridcolor='#e2e8f0', range=[0, 1.15]),
        height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- ROC Curves ---
    st.markdown('<div class="section-header">📉 ROC Curves</div>', unsafe_allow_html=True)
    
    fig = go.Figure()
    roc_colors = ['#667eea', '#f093fb', '#4facfe', '#00f2fe', '#764ba2']
    
    for i, (name, data) in enumerate(models_data.items()):
        roc = data.get('roc_curve', {})
        if roc:
            fig.add_trace(go.Scatter(
                x=roc['fpr'], y=roc['tpr'],
                mode='lines',
                name=f"{name} (AUC: {data.get('auc_roc', 0):.4f})",
                line=dict(color=roc_colors[i % len(roc_colors)], width=2.5)
            ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random (AUC: 0.5)',
        line=dict(color='#8892b0', dash='dash', width=1)
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1e293b'),
        xaxis=dict(title='False Positive Rate', gridcolor='#f1f5f9'),
        yaxis=dict(title='True Positive Rate', gridcolor='#f1f5f9'),
        height=500,
        legend=dict(bgcolor='rgba(255,255,255,0.7)')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Confusion Matrices ---
    st.markdown('<div class="section-header">🧮 Confusion Matrices</div>', unsafe_allow_html=True)
    
    selected_model = st.selectbox("Select Model", list(models_data.keys()))
    cm = np.array(models_data[selected_model]['confusion_matrix'])
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Rejected', 'Approved'],
        y=['Rejected', 'Approved'],
        text_auto=True,
        color_continuous_scale=['#f9fafb', '#10b981', '#7c3aed'],
        aspect='auto'
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1e293b'),
        height=400,
        title=dict(text=f"Confusion Matrix — {selected_model}", font=dict(color='#1e293b'))
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Best Params
    st.markdown('<div class="section-header">🔧 Best Hyperparameters</div>', unsafe_allow_html=True)
    
    params_data = []
    for name, data in models_data.items():
        params = data.get('best_params', {})
        params_data.append({
            'Model': name,
            'Best Parameters': str(params)
        })
    
    params_df = pd.DataFrame(params_data)
    st.dataframe(params_df, use_container_width=True, hide_index=True)


# ============================================================
# PAGE 4: LIVE PREDICTION
# ============================================================
elif page == "🔮 Live Prediction":
    st.markdown("""
    <div class="hero-container" style="padding: 40px 60px;">
        <div class="hero-content" style="padding: 25px 35px;">
            <p class="hero-title" style="font-size: 2.2rem;">🔮 Live <span>Loan Prediction</span></p>
            <p class="hero-subtitle" style="font-size: 1.1rem; margin-top: 10px;">Enter applicant details to get instant AI-powered loan approval prediction with risk analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    predictor = load_predictor()
    
    if predictor is None or predictor.model is None:
        st.error("⚠️ Model not loaded. Please run the training pipeline first: `python -m src.model_training`")
        st.stop()
    
    # --- Input Form ---
    st.markdown('<div class="section-header">📝 Applicant Information</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="metric-card" style="text-align: left; padding: 40px; margin-bottom: 30px; background: rgba(255,255,255,0.9); border: 1px solid #e2e8f0;">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("👤 Gender", ["Male", "Female"], key="pred_gender")
        married = st.selectbox("💍 Marital Status", ["Yes", "No"], key="pred_married")
        dependents = st.selectbox("👨‍👩‍👧‍👦 Dependents", ["0", "1", "2", "3+"], key="pred_deps")
        education = st.selectbox("🎓 Education", ["Graduate", "Not Graduate"], key="pred_edu")
    
    with col2:
        self_employed = st.selectbox("💼 Self Employed", ["No", "Yes"], key="pred_self")
        property_area = st.selectbox("🏘️ Property Area", ["Semiurban", "Urban", "Rural"], key="pred_area")
        credit_history = st.selectbox(
            "📊 Credit History",
            ["Good (1.0)", "Bad (0.0)"],
            key="pred_credit"
        )
        loan_term = st.selectbox(
            "📅 Loan Term (months)",
            [360, 180, 240, 300, 120, 60, 84, 36, 12, 480],
            key="pred_term"
        )
    
    with col3:
        applicant_income = st.number_input(
            "💵 Applicant Income ($/month)", 
            min_value=500, max_value=100000, value=5000, step=500,
            key="pred_income"
        )
        coapplicant_income = st.number_input(
            "💵 Co-applicant Income ($/month)",
            min_value=0, max_value=50000, value=0, step=500,
            key="pred_coincome"
        )
        loan_amount = st.number_input(
            "🏦 Loan Amount ($1000s)",
            min_value=9, max_value=700, value=150, step=10,
            key="pred_loan"
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary before prediction
    st.markdown("---")
    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.metric("Total Income", f"${applicant_income + coapplicant_income:,}/mo")
    with summary_cols[1]:
        st.metric("Loan Amount", f"${loan_amount}K")
    with summary_cols[2]:
        emi = round(loan_amount / (loan_term if loan_term > 0 else 360), 2)
        st.metric("Est. EMI", f"${emi}K/mo")
    with summary_cols[3]:
        ratio = round((applicant_income + coapplicant_income) / (loan_amount * 1000 + 1), 2)
        st.metric("Income/Loan Ratio", f"{ratio}x")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Predict Button
    predict_clicked = st.button(
        "🔮 PREDICT LOAN STATUS",
        use_container_width=True,
        type="primary"
    )
    
    if predict_clicked:
        # Build input
        applicant_data = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': float(loan_term),
            'Credit_History': 1.0 if "Good" in credit_history else 0.0,
            'Property_Area': property_area
        }
        
        with st.spinner("🔄 Analyzing application..."):
            result = predictor.predict(applicant_data)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- Result Display ---
        col_result, col_gauge = st.columns([1.2, 1])
        
        with col_result:
            if result['approved']:
                st.markdown(f"""
                <div class="prediction-approved">
                    <p style="font-size: 3.5rem; margin: 0; filter: drop-shadow(0 0 10px rgba(16,185,129,0.3));">✅</p>
                    <p class="prediction-status">ELIGIBILITY: APPROVED</p>
                    <p class="prediction-confidence">Analysis Confidence: {result['approval_probability']}%</p>
                    <div style="background: #10b981; color: white; padding: 10px 25px; border-radius: 50px; display: inline-block; font-weight: 700; margin-top: 20px; text-transform: uppercase; letter-spacing: 1px; font-size: 0.8rem;">
                        ⚡ {result['risk_level']} Profile
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="prediction-rejected">
                    <p style="font-size: 3.5rem; margin: 0; filter: drop-shadow(0 0 10px rgba(244,63,94,0.3));">❌</p>
                    <p class="prediction-status">ELIGIBILITY: REJECTED</p>
                    <p class="prediction-confidence">Risk Assessment: {result['rejection_probability']}%</p>
                    <div style="background: #f43f5e; color: white; padding: 10px 25px; border-radius: 50px; display: inline-block; font-weight: 700; margin-top: 20px; text-transform: uppercase; letter-spacing: 1px; font-size: 0.8rem;">
                        ⚠️ {result['risk_level']} Profile
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_gauge:
            gauge = create_gauge_chart(result['approval_probability'])
            st.plotly_chart(gauge, use_container_width=True)
        
        # --- Contributing Factors ---
        st.markdown('<div class="section-header">🔍 Decision Breakdown</div>', unsafe_allow_html=True)
        
        for factor in result['contributing_factors']:
            css_class = f"factor-{factor['impact']}"
            icon = "🟢" if factor['impact'] == 'positive' else ("🔴" if factor['impact'] == 'negative' else "🟡")
            st.markdown(f"""
            <div class="{css_class}">
                <strong>{icon} {factor['factor']}</strong><br>
                <span style="color: #8892b0; font-size: 0.9rem;">{factor['detail']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Model info
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"🤖 **Model Used:** {result['model_used']} | Trained on 5,000 records with 5-fold cross-validation")


# ============================================================
# PAGE 5: ABOUT
# ============================================================
elif page == "ℹ️ About":
    st.markdown("""
    <div class="hero-container" style="padding: 40px 60px;">
        <div class="hero-content" style="padding: 25px 35px;">
            <p class="hero-title" style="font-size: 2.2rem;">ℹ️ About <span>This Project</span></p>
            <p class="hero-subtitle" style="font-size: 1.1rem; margin-top: 10px;">Technical details, methodology, and future roadmap</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🎯 Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    The **Loan Approval Prediction System** is a full-stack Machine Learning application that automates 
    loan eligibility assessment. It uses advanced classification algorithms to analyze applicant profiles 
    and predict loan approval status with high accuracy.
    
    This project demonstrates the complete ML lifecycle — from data preprocessing and feature engineering 
    to model training, evaluation, and deployment as an interactive web dashboard.
    """)
    
    st.markdown('<div class="section-header">🔬 Methodology</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Data Preprocessing
        - **Missing Value Imputation**: Median for numerical, mode for categorical
        - **Feature Engineering**: TotalIncome, EMI, BalanceIncome, log transforms
        - **Encoding**: Label Encoding for categorical variables
        - **Scaling**: StandardScaler for feature normalization
        
        #### Model Training
        - **5 Algorithms**: LR, DT, RF, XGBoost, SVM
        - **Hyperparameter Tuning**: GridSearchCV
        - **Cross-Validation**: 5-fold Stratified CV
        - **Evaluation**: Accuracy, Precision, Recall, F1, AUC-ROC
        """)
    
    with col2:
        st.markdown("""
        #### Key Features
        - 🧠 Multi-model comparison with automated selection
        - 📊 Interactive EDA with Plotly visualizations
        - 🔮 Real-time prediction with confidence scoring
        - 🎯 Risk assessment and decision explainability
        - 📈 ROC curves, confusion matrices, and feature importance
        - 🚀 Production-ready deployment on Render
        
        #### Tech Stack
        Python, Scikit-learn, XGBoost, Pandas, NumPy, 
        Plotly, Matplotlib, Seaborn, Streamlit, Joblib
        """)
    
    st.markdown('<div class="section-header">🚀 Future Scope</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Deep Learning Integration**: Neural network-based credit scoring
    - **Real-time API**: REST API for banking system integration
    - **Fairness & Bias Auditing**: Ensure non-discriminatory predictions
    - **Alternative Data Sources**: Utility bills, social signals for thin-file applicants
    - **Automated Retraining**: MLOps pipeline with model drift detection
    """)
    
    st.markdown('<div class="section-header">👨‍💻 Developer</div>', unsafe_allow_html=True)
    st.markdown("""
    | | |
    |:---|:---|
    | **Name** | Kinshunk Garg |
    | **GitHub** | [github.com/Kinshunk565](https://github.com/Kinshunk565) |
    | **Project** | Loan Approval Prediction System |
    | **Category** | Supervised ML — Binary Classification |
    """)


# === FOOTER ===
st.markdown("""
<div class="footer">
    <p style="font-weight: 700; font-size: 1.1rem; margin-bottom: 15px; color: #064e3b;">🍃 EcoLoan Intel Dashboard</p>
    <p>Premium Machine Learning for Sustainable Growth</p>
    <p style="margin-top: 10px;">
        Built with ✨ by <a href="https://github.com/Kinshunk565" target="_blank">Kinshunk Garg</a> • 
        <a href="https://github.com/Kinshunk565" target="_blank">GitHub</a> • 
        Live on <a href="https://render.com" target="_blank">Render</a>
    </p>
</div>
""", unsafe_allow_html=True)
