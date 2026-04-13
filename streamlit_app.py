"""
🏦 Loan Approval Prediction System
====================================
Premium Multi-Page Streamlit Dashboard
Interactive ML-powered loan risk assessment platform

Author: Kinshunk Garg
GitHub: https://github.com/Kinshunk565
"""

import streamlit as st # v3.5.1
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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #ffffff;
        color: #111827;
    }
    
    h1, h2, h3, .hero-title {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Elegant Sidebar (White with Red Accent) */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
        box-shadow: 4px 0 10px rgba(0,0,0,0.02);
    }
    
    [data-testid="stSidebar"]::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 6px;
        background: #A91D22;
        z-index: 100;
    }

    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #374151 !important;
        font-weight: 500;
    }

    [data-testid="stSidebar"] hr {
        border-top-color: rgba(255,255,255,0.1) !important;
    }
    
    /* ===== HERO HEADER (Branded Crimson) ===== */
    .hero-container {
        background: #A91D22;
        border-radius: 12px;
        padding: 50px 60px;
        margin-bottom: 40px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 20px 40px rgba(169, 29, 34, 0.15);
        border-bottom: 6px solid #7c1519;
    }

    .hero-content {
        max-width: 800px;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        color: #ffffff !important;
        margin: 0;
        line-height: 1.0;
        letter-spacing: -2px;
    }
    
    .hero-title span {
        color: #fca5a5;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #ffffff;
        opacity: 0.9;
        margin-top: 20px;
        font-weight: 400;
        max-width: 650px;
        line-height: 1.5;
    }
    
    /* ===== PREMIUM CARDS (Fintech Style) ===== */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-top: 4px solid #A91D22;
        border-radius: 8px;
        padding: 25px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 20px -5px rgba(0,0,0,0.1);
        border-color: #A91D22;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #A91D22;
        letter-spacing: -1px;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 8px;
        font-weight: 700;
    }
    
    /* ===== PREDICTION COMPONENTS ===== */
    .prediction-approved {
        background: #f0fdf4;
        border: 2px solid #16a34a;
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(22, 163, 74, 0.1);
    }
    
    .prediction-rejected {
        background: #fef2f2;
        border: 2px solid #A91D22;
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(169, 29, 34, 0.1);
    }
    
    .prediction-status {
        font-size: 2.8rem;
        font-weight: 900;
        margin: 15px 0;
        letter-spacing: -1.5px;
    }
    
    .prediction-approved .prediction-status { color: #166534; }
    .prediction-rejected .prediction-status { color: #A91D22; }
    
    .prediction-confidence {
        font-size: 1.1rem;
        color: #4b5563;
        font-weight: 500;
    }
    
    /* ===== FACTOR CARDS ===== */
    .factor-positive, .factor-negative, .factor-neutral {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 15px 20px;
        margin: 10px 0;
        color: #1f2937;
    }
    
    .factor-positive { border-left: 5px solid #16a34a; }
    .factor-negative { border-left: 5px solid #A91D22; }
    
    /* ===== SECTION HEADERS ===== */
    .section-header {
        font-size: 1.5rem;
        font-weight: 800;
        color: #111827;
        margin: 40px 0 20px 0;
        border-left: 5px solid #A91D22;
        padding-left: 15px;
    }
    
    .section-header::before {
        display: none;
    }
    
    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        padding: 60px 20px;
        margin-top: 80px;
        background: #111827;
        color: #ffffff;
        border-top: 10px solid #A91D22;
    }
    
    .footer a {
        color: #fca5a5;
        text-decoration: none;
        font-weight: 700;
    }

    .footer a:hover {
        opacity: 0.7;
    }

    /* Streamlit Overrides (Professional) */
    .stButton > button {
        background: #A91D22 !important;
        color: #ffffff !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: #7c1519 !important;
        box-shadow: 0 4px 6px -1px rgba(169, 29, 34, 0.4) !important;
        transform: translateY(-1px);
    }

    .stTextInput input, .stSelectbox [data-baseweb="select"], .stNumberInput input {
        border: 1px solid #d1d5db !important;
        background-color: #ffffff !important;
        color: #111827 !important;
        border-radius: 6px !important;
    }

    .stTextInput input:focus, .stSelectbox [data-baseweb="select"]:focus {
        border-color: #A91D22 !important;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.1) !important;
    }

    .stMetric {
        background: #ffffff;
        padding: 24px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
        transition: transform 0.3s ease;
    }
    .stMetric:hover { transform: scale(1.02); }

    /* Sidebar Radio Styling */
    .stMetric {
        background: #ffffff;
        padding: 20px;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
    }

    /* Sidebar Radio Styling */
    div[data-testid="stSidebarUserContent"] .stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 8px;
        padding: 10px;
    }
    
    div[data-testid="stSidebarUserContent"] .stRadio label {
        background: transparent !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 4px !important;
        padding: 10px 15px !important;
        color: #9ca3af !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        width: 100% !important;
    }
    
    div[data-testid="stSidebarUserContent"] .stRadio label:hover {
        border-color: #ffffff !important;
        color: #ffffff !important;
    }
    
    /* Active State for Sidebar Radio */
    div[data-testid="stSidebarUserContent"] .stRadio label[data-checked="true"] {
        background: #A91D22 !important;
        color: #ffffff !important;
        border-color: #A91D22 !important;
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
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': '#111827', 'family': 'Outfit'}},
        number={'suffix': '%', 'font': {'size': 45, 'color': '#A91D22', 'family': 'Outfit'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#6b7280'},
            'bar': {'color': '#A91D22', 'thickness': 0.25},
            'bgcolor': 'white',
            'borderwidth': 1,
            'bordercolor': '#e5e7eb',
            'steps': [
                {'range': [0, 30], 'color': '#fef2f2'},
                {'range': [30, 70], 'color': '#fee2e2'},
                {'range': [70, 100], 'color': '#fca5a5'}
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#111827', 'family': 'Plus Jakarta Sans'},
        height=300,
        margin=dict(l=30, r=30, t=60, b=30)
    )
    return fig


# === SIDEBAR NAVIGATION ===
st.sidebar.markdown("""
<div style="text-align: center; padding: 25px 0;">
    <div style="font-size: 3.5rem; filter: drop-shadow(0 0 15px rgba(169, 29, 34, 0.4)); color: #A91D22;">🏦</div>
    <div style="font-size: 1.4rem; font-weight: 800; color: #111827; margin-top: 10px; letter-spacing: -0.5px;">
        EcoLoan Intel Pro
    </div>
    <div style="font-size: 0.75rem; color: #A91D22; letter-spacing: 3px; font-weight: 700;">
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
            <p class="hero-title">Beyond <span>Credit<br>Scoring</span></p>
            <p class="hero-subtitle">
                Welcome to EcoLoan Intel Pro. Leveraging 20,000+ data points and ensemble Machine Learning 
                to provide high-fidelity loan risk assessments with actionable transparency.
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
            <div class="metric-label">Enhanced Dataset</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{num_models}</div>
            <div class="metric-label">Advanced Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_auc:.1f}%</div>
            <div class="metric-label">Top AUC-ROC</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_acc:.1f}%</div>
            <div class="metric-label">Prediction Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Overview
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown('<div class="section-header">🚀 System Capabilities</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Advanced Ensemble Learning**: Combines XGBoost, Random Forest, and Voting Classifiers for maximum stability.
        - **Massive Training Base**: Trained on 20,000 synthetic records with complex financial interactions.
        - **Actionable Insights**: Don't just get a 'Yes' or 'No' — understand the 'Why' and 'How to Improve'.
        - **What-If Simulations**: Test financial scenarios in real-time to plan for future approvals.
        """)
    
    with col_right:
        st.markdown('<div class="section-header">💎 Premium Core</div>', unsafe_allow_html=True)
        st.markdown("""
        | Intelligence Layer | Tech Stack |
        |:---|:---|
        | **Dataset Integration** | Python, NumPy, Pandas |
        | **Predictive Engine** | XGBoost, Scikit-learn Ensemble |
        | **Explanation Engine** | Heuristic Feature Attribution |
        | **User Interface** | Streamlit Premium (Outfit/Plus Jakarta) |
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
            marker=dict(colors=['#A91D22', '#e5e7eb']),
            textinfo='label+percent',
            textfont=dict(size=14, color='white')
        )])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#111827'),
            height=380,
            showlegend=False,
            annotations=[dict(
                text=f"<b>{len(df):,}</b><br>Total",
                x=0.5, y=0.5, font_size=20, showarrow=False,
                font=dict(color='#111827')
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
                colorscale=[[0, '#e5e7eb'], [1, '#A91D22']],
                showscale=False
            ),
            text=[f"{v} ({v/len(df)*100:.1f}%)" for v in missing.values],
            textposition='outside',
            textfont=dict(color='#111827')
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#111827'),
            xaxis=dict(title='Feature', gridcolor='#f3f4f6'),
            yaxis=dict(title='Missing Count', gridcolor='#f3f4f6'),
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
            color_continuous_scale=['#fee2e2', '#ef4444', '#A91D22'],
            text='Approval Rate (%)'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                         textfont_color='#111827')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#111827'),
            xaxis=dict(gridcolor='#f3f4f6'),
            yaxis=dict(gridcolor='#f3f4f6', range=[0, 110]),
            coloraxis_showscale=False,
            height=400,
            title=dict(text=f"Approval Rate by {selected_cat}", font=dict(color='#111827'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            counts, values='Count', names=selected_cat,
            color_discrete_sequence=['#A91D22', '#dc2626', '#ef4444', '#f87171', '#fecaca']
        )
        fig.update_traces(textinfo='label+percent', textfont=dict(color='white'))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#111827'),
            height=400,
            title=dict(text=f"Distribution of {selected_cat}", font=dict(color='#111827'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Income Distribution ---
    st.markdown('<div class="section-header">💰 Income & Loan Distribution</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, x='ApplicantIncome', nbins=50,
            color='Loan_Status',
            color_discrete_map={'Y': '#A91D22', 'N': '#9ca3af'},
            marginal='box',
            labels={'Loan_Status': 'Status'},
            barmode='overlay',
            opacity=0.7
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#111827'),
            xaxis=dict(gridcolor='#f3f4f6'),
            yaxis=dict(gridcolor='#f3f4f6'),
            height=400,
            title=dict(text="Applicant Income Distribution", font=dict(color='#111827'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            df, x='LoanAmount', nbins=50,
            color='Loan_Status',
            color_discrete_map={'Y': '#374151', 'N': '#9ca3af'},
            marginal='box',
            labels={'Loan_Status': 'Status'},
            barmode='overlay',
            opacity=0.7
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#111827'),
            xaxis=dict(gridcolor='#f3f4f6'),
            yaxis=dict(gridcolor='#f3f4f6'),
            height=400,
            title=dict(text="Loan Amount Distribution", font=dict(color='#111827'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Correlation Heatmap ---
    st.markdown('<div class="section-header">🔗 Feature Correlation Matrix</div>', unsafe_allow_html=True)
    
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    corr = numeric_df.corr()
    
    fig = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale=['#fee2e2', '#ef4444', '#A91D22'],
        aspect='auto'
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#111827'),
        height=500,
        title=dict(text="Correlation Heatmap", font=dict(color='#111827'))
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
            color_continuous_scale=['#fee2e2', '#ef4444', '#A91D22']
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#111827'),
            xaxis=dict(gridcolor='#f3f4f6'),
            yaxis=dict(gridcolor='#f3f4f6', autorange='reversed'),
            coloraxis_showscale=False,
            height=450,
            title=dict(text="Top Features by Importance", font=dict(color='#111827'))
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================
elif page == "🤖 Model Performance":
    st.markdown("""
    <div class="hero-container" style="padding: 40px 60px;">
        <div class="hero-content" style="padding: 25px 35px;">
            <p class="hero-title" style="font-size: 2.2rem;">🤖 Ensemble <span>Performance</span></p>
            <p class="hero-subtitle" style="font-size: 1.1rem; margin-top: 10px;">Compare multiple state-of-the-art models trained on the enhanced dataset</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    metrics = load_model_metrics()
    if metrics is None:
        st.error("Model metrics not found.")
        st.stop()
    
    best_model = metrics.get('best_model', 'N/A')
    models_data = metrics.get('models', {})
    
    # Best Model Highlight
    st.markdown(f"""
    <div class="metric-card" style="text-align: center; margin-bottom: 30px; border: 4px solid #7c1519; background: #A91D22; color: #ffffff;">
        <div style="font-size: 1rem; color: #fca5a5; text-transform: uppercase; letter-spacing: 2.5px; font-weight: 800; margin-bottom: 10px;">🏆 CURRENT CHAMPION</div>
        <div class="metric-value" style="font-size: 3rem; color: #ffffff; -webkit-text-fill-color: #ffffff;">{best_model}</div>
        <div style="color: #ffffff; margin-top: 15px; font-weight: 500; font-size: 1.1rem; opacity: 0.9;">
            AUC: {models_data.get(best_model, {}).get('auc_roc', 0):.4f} | 
            Accuracy: {models_data.get(best_model, {}).get('accuracy', 0):.4f} | 
            F1: {models_data.get(best_model, {}).get('f1_score', 0):.4f}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison Table
    st.markdown('<div class="section-header">📊 Competitive Matrix</div>', unsafe_allow_html=True)
    
    comp_list = []
    for name, data in models_data.items():
        comp_list.append({
            'Model': name,
            'Accuracy': data.get('accuracy', 0),
            'Precision': data.get('precision', 0),
            'Recall': data.get('recall', 0),
            'F1-Score': data.get('f1_score', 0),
            'AUC-ROC': data.get('auc_roc', 0)
        })
    st.dataframe(pd.DataFrame(comp_list), use_container_width=True, hide_index=True)

    # Visual Comparison
    st.markdown('<div class="section-header">📈 Metric Distribution</div>', unsafe_allow_html=True)
    fig_comp = px.bar(
        pd.DataFrame(comp_list).melt(id_vars='Model', var_name='Metric', value_name='Score'),
        x='Model', y='Score', color='Metric', barmode='group',
        color_discrete_sequence=['#A91D22', '#dc2626', '#ef4444', '#f87171', '#fecaca']
    )
    fig_comp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font_color='#111827', 
        height=450,
        xaxis=dict(gridcolor='#f3f4f6'),
        yaxis=dict(gridcolor='#f3f4f6')
    )
    st.plotly_chart(fig_comp, use_container_width=True)


# ============================================================
# PAGE 4: LIVE PREDICTION
# ============================================================
elif page == "🔮 Live Prediction":
    st.markdown("""
    <div class="hero-container" style="padding: 40px 60px;">
        <div class="hero-content" style="padding: 25px 35px;">
            <p class="hero-title" style="font-size: 2.2rem;">🔮 Smart <span>Risk Assessment</span></p>
            <p class="hero-subtitle" style="font-size: 1.1rem; margin-top: 10px;">Enter applicant details to get a real-time loan approval prediction & actionable advice</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    predictor = load_predictor()
    if predictor is None:
        st.error("Predictor not available. Please run model training first.")
        st.stop()

    # --- Input Section ---
    st.markdown('<div class="section-header">📝 Applicant Information</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    
    with col2:
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        credit_history = st.selectbox("Credit History", ["Good (1.0)", "Bad (0.0)"])
        loan_term = st.selectbox("Loan Term (months)", [360, 180, 240, 300, 120, 60, 84, 36, 12, 480], index=0)
    
    with col3:
        applicant_income = st.number_input("Applicant Income ($/mo)", min_value=500, max_value=100000, value=5000, step=500)
        coapplicant_income = st.number_input("Co-applicant Income ($/mo)", min_value=0, max_value=50000, value=0, step=500)
        loan_amount = st.number_input("Loan Amount ($1000s)", min_value=9, max_value=700, value=150, step=10)

    # Simple Summary
    st.markdown("---")
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        st.metric("Total Monthly Income", f"${applicant_income + coapplicant_income:,}")
    with s_col2:
        st.metric("Requested Loan", f"${loan_amount}K")
    with s_col3:
        ratio = round((applicant_income + coapplicant_income) / (loan_amount * 1000 + 1), 2)
        st.metric("Income/Loan Ratio", f"{ratio}x")

    # Predict Button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔮 ANALYZE LOAN ELIGIBILITY", use_container_width=True, type="primary")

    if predict_clicked:
        applicant_data = {
            'Gender': gender, 'Married': married, 'Dependents': dependents,
            'Education': education, 'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income, 'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount, 'Loan_Amount_Term': float(loan_term),
            'Credit_History': 1.0 if "Good" in credit_history else 0.0,
            'Property_Area': property_area
        }
        
        with st.spinner("Analyzing risk patterns..."):
            result = predictor.predict(applicant_data)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- Result Display ---
        res_col1, res_col2 = st.columns([1.2, 1])
        with res_col1:
            if result['approved']:
                st.markdown(f"""
                <div class="prediction-approved">
                    <p style="font-size: 3.5rem; margin: 0;">✅</p>
                    <p class="prediction-status">ELIGIBILITY: APPROVED</p>
                    <p class="prediction-confidence">Analysis Confidence: {result['approval_probability']}%</p>
                    <div style="background: #A91D22; color: white; padding: 10px 25px; border-radius: 4px; display: inline-block; font-weight: 700; margin-top: 20px;">
                        {result['risk_level']} Profile
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="prediction-rejected">
                    <p style="font-size: 3.5rem; margin: 0;">❌</p>
                    <p class="prediction-status">ELIGIBILITY: REJECTED</p>
                    <p class="prediction-confidence">Risk Assessment: {result['rejection_probability']}%</p>
                    <div style="background: #6b7280; color: white; padding: 10px 25px; border-radius: 4px; display: inline-block; font-weight: 700; margin-top: 20px;">
                        {result['risk_level']} Profile
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with res_col2:
            st.plotly_chart(create_gauge_chart(result['approval_probability']), use_container_width=True)
        
        # Factors & Recommendations
        st.markdown('<div class="section-header">🔍 Decision Breakdown</div>', unsafe_allow_html=True)
        for factor in result['contributing_factors']:
            css_class = f"factor-{factor['impact']}"
            icon = "◾" if factor['impact'] == 'positive' else "◽"
            st.markdown(f"""<div class="{css_class}"><strong>{icon} {factor['factor']}</strong><br><span style="color: #4b5563; font-size: 0.9rem;">{factor['detail']}</span></div>""", unsafe_allow_html=True)
            
        if result['recommendations']:
            st.markdown('<div class="section-header">💡 Actionable Advice</div>', unsafe_allow_html=True)
            for rec in result['recommendations']:
                st.info(rec)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"🤖 **Model Engine:** {result['model_used']} | Trained on 20,000 synthetic records")

    # --- WHAT-IF ANALYSIS ---
    st.markdown('<div class="section-header">🧪 Risk Simulation Control</div>', unsafe_allow_html=True)
    st.markdown("Adjust applicant variables to observe real-time impact on approval probability.")
    
    with st.container():
        # Expansion: Add more simulation controls for better "Working Properly" feel
        w_main_col1, w_main_col2 = st.columns([2, 1])
        
        with w_main_col1:
            w_col1, w_col2 = st.columns(2)
            with w_col1:
                w_inc = st.slider("Total Monthly Income ($)", 1000, 50000, 5000, step=500)
                w_credit = st.checkbox("Perfect Credit History", value=True)
                w_married = st.selectbox("Married Status", ["Yes", "No"], index=0)
            with w_col2:
                w_loan = st.slider("Loan Amount Request ($1000s)", 10, 700, 150, step=10)
                w_edu = st.selectbox("Education Level", ["Graduate", "Not Graduate"], index=0)
                w_area = st.selectbox("Property Type", ["Urban", "Semiurban", "Rural"], index=1)
        
        # Logic Fix: Ensure data matches model expectations perfectly
        w_data = {
            'Gender': 'Male', 
            'Married': w_married, 
            'Dependents': '0', 
            'Education': w_edu, 
            'Self_Employed': 'No',
            'ApplicantIncome': w_inc, 
            'CoapplicantIncome': 0, 
            'LoanAmount': w_loan, 
            'Loan_Amount_Term': 360,
            'Credit_History': 1.0 if w_credit else 0.0, 
            'Property_Area': w_area
        }
        
        sim_res = predictor.predict(w_data)
        
        with w_main_col2:
            st.markdown(f"""
            <div style="background: #A91D22; padding: 25px; border-radius: 12px; border: 4px solid #7c1519; color: #ffffff; text-align: center; height: 100%;">
                <p style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #fca5a5;">SIMULATED PROBABILITY</p>
                <p style="font-size: 3.5rem; font-weight: 900; color: #ffffff; margin: 10px 0;">{sim_res['approval_probability']}%</p>
                <div style="margin-top: 15px;">
                    <div style="width: 100%; background: rgba(255,255,255,0.2); height: 10px; border-radius: 5px;">
                        <div style="width: {sim_res['approval_probability']}%; background: #ffffff; height: 10px; border-radius: 5px; box-shadow: 0 0 10px rgba(255,255,255,0.5);"></div>
                    </div>
                </div>
                <p style="margin-top: 20px; font-weight: 600; font-size: 1rem; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px;">
                    {sim_res['risk_level']}
                </p>
            </div>
            """, unsafe_allow_html=True)


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
        #### Data Engineering
        - **Record Count**: 20,000 synthetic samples
        - **Feature Engineering**: Interaction features (`StabilityScore`, `CreditStress`), `DebtToIncomeRatio`
        - **Scaling**: Robust feature scaling for tree-based stability
        
        #### Model Intelligence
        - **Advanced Models**: XGBoost, LightGBM, CatBoost
        - **Hyperparameter Tuning**: Extensive GridSearchCV
        - **Ensemble Strategy**: Soft-Voting Classifier (Top 3 Models)
        - **Validation**: 5-fold Stratified Cross-Validation
        """)
    
    with col2:
        st.markdown("""
        #### Key Features
        - 🧠 **Smart Recommendations**: Actionable advice for rejected applications
        - 📊 **Dynamic EDA**: Interactive Plotly-based deep dives
        - 🔮 **Explainable AI**: Local factor attribution for every prediction
        - 🧪 **What-If Simulations**: Real-time financial scenario testing
        - 📈 **Performance Dashboard**: Real-time ROC curves and confusion matrices
        
        #### Tech Stack
        Python, Scikit-learn, XGBoost, CatBoost, LightGBM, 
        Pandas, Plotly, Streamlit, Joblib
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
<div class="footer" style="background: #111827; border-top: 10px solid #A91D22; color: #ffffff; padding: 60px 20px;">
    <p style="font-weight: 900; font-size: 1.4rem; margin-bottom: 10px; color: #ffffff; letter-spacing: -1px;">EcoLoan Intel Pro Dashboard</p>
    <p style="color: #9ca3af; font-size: 0.9rem; margin-bottom: 25px;">Minimalist High-Fidelity Machine Learning Intelligence</p>
    <div style="width: 50px; height: 3px; background: #ffffff; margin: 0 auto 25px auto;"></div>
    <p style="margin-top: 10px; color: #d1d5db;">
        Built by <a href="https://github.com/Kinshunk565" target="_blank" style="color: #ffffff; text-decoration: underline;">Kinshunk Garg</a> • 
        <a href="https://github.com/Kinshunk565" target="_blank" style="color: #ffffff;">GitHub</a> • 
        Live on <a href="https://render.com" target="_blank" style="color: #ffffff;">Render</a>
    </p>
</div>
""", unsafe_allow_html=True)
