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
    page_title="Loan Approval Predictor | ML Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PATHS ---
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# --- PREMIUM CSS ---
st.markdown("""
<style>
    /* ===== GLOBAL STYLES ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Dark gradient background */
    .stApp > header {
        background: transparent;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] span {
        color: #e0e0e0 !important;
    }
    
    /* ===== HERO HEADER ===== */
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 20px;
        padding: 40px 50px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -30%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        color: #ffffff;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        font-size: 1.15rem;
        color: rgba(255,255,255,0.85);
        margin-top: 10px;
        font-weight: 400;
        line-height: 1.6;
    }
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 8px;
        font-weight: 600;
    }
    
    /* ===== PREDICTION CARD ===== */
    .prediction-approved {
        background: linear-gradient(135deg, #00c853 0%, #64dd17 100%);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(0, 200, 83, 0.3);
        animation: pulse-green 2s infinite;
    }
    
    .prediction-rejected {
        background: linear-gradient(135deg, #ff1744 0%, #ff6d00 100%);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(255, 23, 68, 0.3);
        animation: pulse-red 2s infinite;
    }
    
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 20px 60px rgba(0, 200, 83, 0.3); }
        50% { box-shadow: 0 20px 80px rgba(0, 200, 83, 0.5); }
    }
    
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 20px 60px rgba(255, 23, 68, 0.3); }
        50% { box-shadow: 0 20px 80px rgba(255, 23, 68, 0.5); }
    }
    
    .prediction-status {
        font-size: 3rem;
        font-weight: 900;
        color: white;
        margin: 0;
    }
    
    .prediction-confidence {
        font-size: 1.3rem;
        color: rgba(255,255,255,0.9);
        margin-top: 8px;
    }
    
    /* ===== FACTOR CARDS ===== */
    .factor-positive {
        background: linear-gradient(135deg, rgba(0, 200, 83, 0.1) 0%, rgba(100, 221, 23, 0.1) 100%);
        border-left: 4px solid #00c853;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    
    .factor-negative {
        background: linear-gradient(135deg, rgba(255, 23, 68, 0.1) 0%, rgba(255, 109, 0, 0.1) 100%);
        border-left: 4px solid #ff1744;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    
    .factor-neutral {
        background: linear-gradient(135deg, rgba(255, 171, 0, 0.1) 0%, rgba(255, 214, 0, 0.1) 100%);
        border-left: 4px solid #ffab00;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    
    /* ===== SECTION HEADERS ===== */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ccd6f6;
        margin: 30px 0 20px 0;
        padding-bottom: 12px;
        border-bottom: 3px solid;
        border-image: linear-gradient(135deg, #667eea, #764ba2, #f093fb) 1;
    }
    
    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        padding: 30px;
        margin-top: 50px;
        border-top: 1px solid rgba(102, 126, 234, 0.2);
        color: #8892b0;
        font-size: 0.85rem;
    }
    
    .footer a {
        color: #667eea;
        text-decoration: none;
        font-weight: 600;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    /* ===== RISK GAUGE ===== */
    .risk-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 1px;
    }
    
    /* Hide default Streamlit footer */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
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
                {'range': [0, 30], 'color': 'rgba(255, 23, 68, 0.3)'},
                {'range': [30, 55], 'color': 'rgba(255, 171, 0, 0.3)'},
                {'range': [55, 75], 'color': 'rgba(100, 221, 23, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(0, 200, 83, 0.3)'}
            ],
            'threshold': {
                'line': {'color': '#f093fb', 'width': 4},
                'thickness': 0.8,
                'value': value
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ccd6f6'},
        height=280,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


# === SIDEBAR NAVIGATION ===
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <div style="font-size: 3rem;">🏦</div>
    <div style="font-size: 1.1rem; font-weight: 700; color: #ccd6f6; margin-top: 5px;">
        Loan Predictor
    </div>
    <div style="font-size: 0.75rem; color: #8892b0; letter-spacing: 2px;">
        ML DASHBOARD
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
        <p class="hero-title">🏦 Loan Approval<br>Prediction System</p>
        <p class="hero-subtitle">
            AI-powered credit risk assessment platform using advanced Machine Learning.<br>
            Analyze applicant profiles, predict loan eligibility, and understand risk factors — all in real-time.
        </p>
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
            hole=0.6,
            marker=dict(colors=['#667eea', '#ff6b6b']),
            textinfo='label+percent',
            textfont=dict(size=14, color='white')
        )])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ccd6f6'),
            height=350,
            showlegend=False,
            annotations=[dict(
                text=f"<b>{len(df):,}</b><br>Total",
                x=0.5, y=0.5, font_size=20, showarrow=False,
                font=dict(color='#ccd6f6')
            )]
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 2: EDA & INSIGHTS
# ============================================================
elif page == "📊 EDA & Insights":
    st.markdown("""
    <div class="hero-container" style="padding: 30px 40px;">
        <p class="hero-title" style="font-size: 2rem;">📊 Exploratory Data Analysis</p>
        <p class="hero-subtitle">Deep dive into the loan dataset — distributions, correlations, and key patterns</p>
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
                colorscale=[[0, '#667eea'], [1, '#f093fb']],
                showscale=False
            ),
            text=[f"{v} ({v/len(df)*100:.1f}%)" for v in missing.values],
            textposition='outside',
            textfont=dict(color='#ccd6f6')
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ccd6f6'),
            xaxis=dict(title='Feature', gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title='Missing Count', gridcolor='rgba(255,255,255,0.1)'),
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
            color_continuous_scale=['#ff6b6b', '#ffab00', '#667eea', '#00c853'],
            text='Approval Rate (%)'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                         textfont_color='#ccd6f6')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ccd6f6'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', range=[0, 110]),
            coloraxis_showscale=False,
            height=400,
            title=dict(text=f"Approval Rate by {selected_cat}", font=dict(color='#ccd6f6'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            counts, values='Count', names=selected_cat,
            color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']
        )
        fig.update_traces(textinfo='label+percent', textfont=dict(color='white'))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ccd6f6'),
            height=400,
            title=dict(text=f"Distribution of {selected_cat}", font=dict(color='#ccd6f6'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Income Distribution ---
    st.markdown('<div class="section-header">💰 Income & Loan Distribution</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, x='ApplicantIncome', nbins=50,
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
            title=dict(text="Applicant Income Distribution", font=dict(color='#ccd6f6'))
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
        color_continuous_scale=['#1a1a2e', '#667eea', '#f093fb'],
        aspect='auto'
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccd6f6'),
        height=500,
        title=dict(text="Correlation Heatmap", font=dict(color='#ccd6f6'))
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
    <div class="hero-container" style="padding: 30px 40px;">
        <p class="hero-title" style="font-size: 2rem;">🤖 Model Performance</p>
        <p class="hero-subtitle">Compare 5 ML models trained with GridSearchCV and 5-fold cross-validation</p>
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
    <div class="metric-card" style="text-align: center; margin-bottom: 30px;">
        <div style="font-size: 1rem; color: #8892b0; text-transform: uppercase; letter-spacing: 2px;">🏆 Champion Model</div>
        <div class="metric-value" style="font-size: 2.5rem;">{best_model}</div>
        <div style="color: #8892b0; margin-top: 5px;">
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
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']
    
    for i, metric in enumerate(metric_names):
        values = [models_data[m].get(metric, 0) for m in model_names_list]
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=model_names_list,
            y=values,
            marker_color=colors[i],
            text=[f"{v:.3f}" for v in values],
            textposition='outside',
            textfont=dict(color='#ccd6f6', size=10)
        ))
    
    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccd6f6'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', range=[0, 1.15]),
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
        font=dict(color='#ccd6f6'),
        xaxis=dict(title='False Positive Rate', gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title='True Positive Rate', gridcolor='rgba(255,255,255,0.1)'),
        height=500,
        legend=dict(bgcolor='rgba(0,0,0,0.3)')
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
        color_continuous_scale=['#1a1a2e', '#667eea', '#f093fb'],
        aspect='auto'
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccd6f6'),
        height=400,
        title=dict(text=f"Confusion Matrix — {selected_model}", font=dict(color='#ccd6f6'))
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
    <div class="hero-container" style="padding: 30px 40px;">
        <p class="hero-title" style="font-size: 2rem;">🔮 Live Loan Prediction</p>
        <p class="hero-subtitle">Enter applicant details to get instant AI-powered loan approval prediction with risk analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    predictor = load_predictor()
    
    if predictor is None or predictor.model is None:
        st.error("⚠️ Model not loaded. Please run the training pipeline first: `python -m src.model_training`")
        st.stop()
    
    # --- Input Form ---
    st.markdown('<div class="section-header">📝 Applicant Information</div>', unsafe_allow_html=True)
    
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
                    <p style="font-size: 3rem; margin: 0;">✅</p>
                    <p class="prediction-status">LOAN APPROVED</p>
                    <p class="prediction-confidence">Confidence: {result['approval_probability']}%</p>
                    <div class="risk-badge" style="background: rgba(255,255,255,0.2); color: white; margin-top: 15px;">
                        {result['risk_level']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="prediction-rejected">
                    <p style="font-size: 3rem; margin: 0;">❌</p>
                    <p class="prediction-status">LOAN REJECTED</p>
                    <p class="prediction-confidence">Rejection Probability: {result['rejection_probability']}%</p>
                    <div class="risk-badge" style="background: rgba(255,255,255,0.2); color: white; margin-top: 15px;">
                        {result['risk_level']}
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
    <div class="hero-container" style="padding: 30px 40px;">
        <p class="hero-title" style="font-size: 2rem;">ℹ️ About This Project</p>
        <p class="hero-subtitle">Technical details, methodology, and future roadmap</p>
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
    <p>🏦 <strong>Loan Approval Prediction System</strong> — Powered by Machine Learning</p>
    <p>Built by <a href="https://github.com/Kinshunk565" target="_blank">Kinshunk Garg</a> • 
    <a href="https://github.com/Kinshunk565" target="_blank">GitHub</a></p>
</div>
""", unsafe_allow_html=True)
