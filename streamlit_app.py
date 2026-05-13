"""
🏦 Loan Approval Prediction System
====================================
Premium Multi-Page Streamlit Dashboard
Interactive ML-powered loan risk assessment platform
with FICO Credit Intelligence Suite

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
    
    /* ===== CREDIT HEALTH CARDS ===== */
    .credit-health-card {
        background: linear-gradient(135deg, #ffffff, #f9fafb);
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    
    .credit-health-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }
    
    .credit-score-display {
        font-size: 3.5rem;
        font-weight: 900;
        letter-spacing: -2px;
        line-height: 1;
    }
    
    .credit-grade-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 10px;
    }
    
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

@st.cache_data
def load_loan_records_db():
    """Load the past loan records database."""
    people_path = os.path.join(DATA_DIR, "loan_records_people.csv")
    records_path = os.path.join(DATA_DIR, "loan_records_history.csv")
    if os.path.exists(people_path) and os.path.exists(records_path):
        people = pd.read_csv(people_path)
        records = pd.read_csv(records_path)
        return people, records
    return None, None

def search_person(people_df, records_df, query: str):
    """Search for a person by name or ID."""
    if people_df is None or query.strip() == "":
        return None, None
    q = query.strip().lower()
    # Search by ID
    id_match = people_df[people_df['Person_ID'].str.lower() == q]
    if len(id_match) > 0:
        pid = id_match.iloc[0]['Person_ID']
        from src.generate_loan_records_db import summarize_person_records
        summary = summarize_person_records(records_df, pid)
        return id_match.iloc[0], summary
    # Search by name (partial match)
    name_matches = people_df[people_df['Full_Name'].str.lower().str.contains(q, na=False)]
    if len(name_matches) > 0:
        person = name_matches.iloc[0]
        pid = person['Person_ID']
        from src.generate_loan_records_db import summarize_person_records
        summary = summarize_person_records(records_df, pid)
        return person, summary
    return None, None

def get_sample_pdf_bytes():
    """Get sample PDF bytes for download."""
    try:
        from src.sample_loan_report import generate_sample_pdf
        return generate_sample_pdf()
    except Exception:
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


def create_credit_score_gauge(score, grade, grade_color):
    """Create a premium FICO score gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"FICO Score — {grade}", 'font': {'size': 18, 'color': '#111827', 'family': 'Outfit'}},
        number={'font': {'size': 52, 'color': grade_color, 'family': 'Outfit'}},
        gauge={
            'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': '#9ca3af',
                     'tickvals': [300, 580, 670, 740, 800, 850],
                     'ticktext': ['300', '580', '670', '740', '800', '850']},
            'bar': {'color': grade_color, 'thickness': 0.3},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': '#e5e7eb',
            'steps': [
                {'range': [300, 580], 'color': '#fef2f2'},    # Poor
                {'range': [580, 670], 'color': '#fef9c3'},    # Fair
                {'range': [670, 740], 'color': '#ecfccb'},    # Good
                {'range': [740, 800], 'color': '#d1fae5'},    # Very Good
                {'range': [800, 850], 'color': '#a7f3d0'}     # Exceptional
            ],
            'threshold': {
                'line': {'color': '#111827', 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#111827', 'family': 'Plus Jakarta Sans'},
        height=300,
        margin=dict(l=30, r=30, t=70, b=30)
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
                Welcome to EcoLoan Intel Pro. Leveraging 20,000+ data points, FICO Credit Intelligence, 
                and ensemble Machine Learning to provide high-fidelity loan risk assessments with actionable transparency.
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
        - **FICO Credit Intelligence**: Granular 300-850 credit scoring with utilization & account depth analysis.
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
        | **Credit Scoring** | FICO 300-850, Utilization, Depth |
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
            <p class="hero-subtitle" style="font-size: 1.1rem; margin-top: 10px;">Deep dive into the loan dataset — distributions, correlations, credit scoring patterns, and key insights</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    df = load_dataset()
    if df is None:
        st.error("📂 Dataset not found. Please run the training pipeline first.")
        st.stop()
    
    # --- Dataset Overview ---
    st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", f"{df.shape[1] - 1}")
    with col3:
        approval_rate = (df['Loan_Status'] == 'Y').mean()
        st.metric("Approval Rate", f"{approval_rate:.1%}")
    with col4:
        if 'Credit_Score' in df.columns:
            avg_score = df['Credit_Score'].mean()
            st.metric("Avg Credit Score", f"{avg_score:.0f}")
    
    # --- Past Loan History Overview ---
    if 'Prev_Loan_Count' in df.columns:
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Avg Previous Loans", f"{df['Prev_Loan_Count'].mean():.1f}")
        with col6:
            st.metric("Avg Repayment Rate", f"{df['Repayment_Rate'].mean():.0%}")
        with col7:
            st.metric("First-Time Borrowers", f"{(df['Prev_Loan_Count'] == 0).mean():.0%}")
        with col8:
            st.metric("Avg Defaults", f"{df['Prev_Loan_Defaults'].mean():.2f}")
    
    # --- Credit Score Distribution ---
    if 'Credit_Score' in df.columns:
        st.markdown('<div class="section-header">📈 Credit Score Distribution by Loan Status</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df.dropna(subset=['Credit_Score']), x='Credit_Score', nbins=55,
                color='Loan_Status',
                color_discrete_map={'Y': '#A91D22', 'N': '#9ca3af'},
                marginal='box',
                labels={'Loan_Status': 'Status'},
                barmode='overlay',
                opacity=0.7
            )
            # Add FICO tier lines
            for threshold, label in [(580, 'Fair'), (670, 'Good'), (740, 'V.Good'), (800, 'Excptl')]:
                fig.add_vline(x=threshold, line_dash="dash", line_color="#6b7280", opacity=0.5,
                             annotation_text=label, annotation_position="top")
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#111827'),
                xaxis=dict(gridcolor='#f3f4f6', title='FICO Credit Score'),
                yaxis=dict(gridcolor='#f3f4f6'),
                height=420,
                title=dict(text="Credit Score Distribution", font=dict(color='#111827'))
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Credit Score vs Approval Rate by tier
            df_cs = df.dropna(subset=['Credit_Score', 'Loan_Status']).copy()
            df_cs['Credit_Tier'] = pd.cut(
                df_cs['Credit_Score'],
                bins=[0, 579, 669, 739, 799, 850],
                labels=['Poor\n(300-579)', 'Fair\n(580-669)', 'Good\n(670-739)', 'Very Good\n(740-799)', 'Exceptional\n(800-850)']
            )
            tier_approval = df_cs.groupby('Credit_Tier', observed=True)['Loan_Status'].apply(
                lambda x: (x == 'Y').mean() * 100
            ).reset_index()
            tier_approval.columns = ['Credit Tier', 'Approval Rate (%)']
            
            fig = px.bar(
                tier_approval, x='Credit Tier', y='Approval Rate (%)',
                color='Approval Rate (%)',
                color_continuous_scale=['#fef2f2', '#ef4444', '#A91D22'],
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
                height=420,
                title=dict(text="Approval Rate by FICO Tier", font=dict(color='#111827'))
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # --- Credit Utilization Analysis ---
    if 'Credit_Utilization' in df.columns:
        st.markdown('<div class="section-header">💳 Credit Utilization & Account Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                df.dropna(subset=['Credit_Score', 'Credit_Utilization']),
                x='Credit_Score', y='Credit_Utilization',
                color='Loan_Status',
                color_discrete_map={'Y': '#A91D22', 'N': '#d1d5db'},
                opacity=0.4,
                labels={'Credit_Score': 'FICO Score', 'Credit_Utilization': 'Utilization (%)'}
            )
            # Add danger zone
            fig.add_hrect(y0=75, y1=100, fillcolor="#fef2f2", opacity=0.3, 
                         line_width=0, annotation_text="Danger Zone", annotation_position="top right")
            fig.add_hrect(y0=0, y1=30, fillcolor="#f0fdf4", opacity=0.2, 
                         line_width=0, annotation_text="Optimal", annotation_position="bottom right")
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#111827'),
                xaxis=dict(gridcolor='#f3f4f6'),
                yaxis=dict(gridcolor='#f3f4f6'),
                height=420,
                title=dict(text="Credit Score vs Utilization", font=dict(color='#111827'))
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Open_Accounts' in df.columns:
                acct_approval = df.dropna(subset=['Open_Accounts', 'Loan_Status']).copy()
                acct_grouped = acct_approval.groupby('Open_Accounts')['Loan_Status'].apply(
                    lambda x: (x == 'Y').mean() * 100
                ).reset_index()
                acct_grouped.columns = ['Open Accounts', 'Approval Rate (%)']
                
                fig = px.bar(
                    acct_grouped, x='Open Accounts', y='Approval Rate (%)',
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
                    height=420,
                    title=dict(text="Approval Rate by Open Accounts", font=dict(color='#111827'))
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # --- Past Loan History Analysis ---
    if 'Prev_Loan_Count' in df.columns and 'Repayment_Rate' in df.columns:
        st.markdown('<div class="section-header">📜 Past Loan History Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Repayment Rate vs Approval
            df_rr = df.dropna(subset=['Repayment_Rate', 'Loan_Status']).copy()
            df_rr['Repayment_Tier'] = pd.cut(
                df_rr['Repayment_Rate'],
                bins=[-0.01, 0, 0.25, 0.50, 0.75, 1.01],
                labels=['No History', '0-25%', '25-50%', '50-75%', '75-100%']
            )
            rr_approval = df_rr.groupby('Repayment_Tier', observed=True)['Loan_Status'].apply(
                lambda x: (x == 'Y').mean() * 100
            ).reset_index()
            rr_approval.columns = ['Repayment Tier', 'Approval Rate (%)']
            
            fig = px.bar(
                rr_approval, x='Repayment Tier', y='Approval Rate (%)',
                color='Approval Rate (%)',
                color_continuous_scale=['#fef2f2', '#ef4444', '#A91D22'],
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
                height=420,
                title=dict(text="Approval Rate by Repayment History", font=dict(color='#111827'))
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Defaults vs Approval
            df_def = df.dropna(subset=['Prev_Loan_Defaults', 'Loan_Status']).copy()
            df_def['Default_Group'] = df_def['Prev_Loan_Defaults'].clip(0, 4).astype(int).astype(str)
            df_def.loc[df_def['Prev_Loan_Defaults'] >= 4, 'Default_Group'] = '4+'
            def_approval = df_def.groupby('Default_Group', observed=True)['Loan_Status'].apply(
                lambda x: (x == 'Y').mean() * 100
            ).reset_index()
            def_approval.columns = ['Defaults', 'Approval Rate (%)']
            
            fig = px.bar(
                def_approval, x='Defaults', y='Approval Rate (%)',
                color='Approval Rate (%)',
                color_continuous_scale=['#A91D22', '#ef4444', '#fef2f2'],
                text='Approval Rate (%)'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                             textfont_color='#111827')
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#111827'),
                xaxis=dict(gridcolor='#f3f4f6', title='Number of Past Defaults'),
                yaxis=dict(gridcolor='#f3f4f6', range=[0, 110]),
                coloraxis_showscale=False,
                height=420,
                title=dict(text="Approval Rate by Number of Defaults", font=dict(color='#111827'))
            )
            st.plotly_chart(fig, use_container_width=True)
    
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
    
    cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']
    
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
            'Feature': list(fi.keys())[:15],
            'Importance': list(fi.values())[:15]
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
            height=500,
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
            <p class="hero-subtitle" style="font-size: 1.1rem; margin-top: 10px;">Compare multiple state-of-the-art models trained on the enhanced dataset with FICO credit scoring</p>
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
            <p class="hero-subtitle" style="font-size: 1.1rem; margin-top: 10px;">Enter applicant details to get a real-time loan approval prediction with FICO credit intelligence</p>
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
        loan_term = st.selectbox("Loan Term (months)", [360, 180, 240, 300, 120, 60, 84, 36, 12, 480], index=0)
    
    with col3:
        applicant_income = st.number_input("Applicant Income ($/mo)", min_value=500, max_value=100000, value=5000, step=500)
        coapplicant_income = st.number_input("Co-applicant Income ($/mo)", min_value=0, max_value=50000, value=0, step=500)
        loan_amount = st.number_input("Loan Amount ($1000s)", min_value=9, max_value=700, value=150, step=10)

    # --- CREDIT PORTFOLIO SECTION ---
    st.markdown('<div class="section-header">💳 Credit Portfolio</div>', unsafe_allow_html=True)
    
    cr_col1, cr_col2, cr_col3 = st.columns(3)
    with cr_col1:
        credit_score = st.slider(
            "📊 FICO Credit Score",
            min_value=300, max_value=850, value=700, step=5,
            help="300-579: Poor | 580-669: Fair | 670-739: Good | 740-799: Very Good | 800-850: Exceptional"
        )
    with cr_col2:
        credit_utilization = st.slider(
            "💳 Credit Utilization (%)",
            min_value=0, max_value=100, value=25, step=1,
            help="Recommended: Below 30%. Shows how much of your available credit you're using."
        )
    with cr_col3:
        open_accounts = st.number_input(
            "🏦 Open Credit Accounts",
            min_value=0, max_value=15, value=4, step=1,
            help="Sweet spot: 3-7 accounts. Too few = thin file, too many = overleveraged."
        )

    # Credit Score Visual Feedback (inline)
    if credit_score >= 800:
        cs_label, cs_color = "Exceptional", "#00c853"
    elif credit_score >= 740:
        cs_label, cs_color = "Very Good", "#64dd17"
    elif credit_score >= 670:
        cs_label, cs_color = "Good", "#ffab00"
    elif credit_score >= 580:
        cs_label, cs_color = "Fair", "#ff6d00"
    else:
        cs_label, cs_color = "Poor", "#dd2c00"
    
    st.markdown(f"""
    <div style="display: flex; gap: 20px; margin-top: 5px;">
        <div style="flex: 1; background: linear-gradient(135deg, {cs_color}15, {cs_color}08); border: 1px solid {cs_color}40; border-radius: 8px; padding: 12px 20px; text-align: center;">
            <span style="font-size: 2rem; font-weight: 900; color: {cs_color};">{credit_score}</span>
            <span style="font-size: 0.85rem; color: #6b7280; margin-left: 8px;">— {cs_label}</span>
        </div>
        <div style="flex: 1; background: {'#f0fdf4' if credit_utilization < 30 else '#fef2f2'}; border: 1px solid {'#16a34a40' if credit_utilization < 30 else '#A91D2240'}; border-radius: 8px; padding: 12px 20px; text-align: center;">
            <span style="font-size: 2rem; font-weight: 900; color: {'#16a34a' if credit_utilization < 30 else '#A91D22'};">{credit_utilization}%</span>
            <span style="font-size: 0.85rem; color: #6b7280; margin-left: 8px;">— Utilization</span>
        </div>
        <div style="flex: 1; background: {'#f0fdf4' if 3 <= open_accounts <= 7 else '#fef9c3'}; border: 1px solid {'#16a34a40' if 3 <= open_accounts <= 7 else '#ca8a0440'}; border-radius: 8px; padding: 12px 20px; text-align: center;">
            <span style="font-size: 2rem; font-weight: 900; color: {'#16a34a' if 3 <= open_accounts <= 7 else '#ca8a04'};">{open_accounts}</span>
            <span style="font-size: 0.85rem; color: #6b7280; margin-left: 8px;">— Accounts</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- PAST LOAN HISTORY — SMART LOAD SECTION ---
    st.markdown('<div class="section-header">📂 Load Past Records</div>', unsafe_allow_html=True)
    st.markdown("Load an applicant's past loan history from our records database, upload a PDF report, or enter manually.")
    
    # Initialize session state for loaded records
    if 'loaded_prev_loan_count' not in st.session_state:
        st.session_state.loaded_prev_loan_count = None
    if 'loaded_prev_loans_repaid' not in st.session_state:
        st.session_state.loaded_prev_loans_repaid = None
    if 'loaded_prev_loan_defaults' not in st.session_state:
        st.session_state.loaded_prev_loan_defaults = None
    if 'loaded_avg_prev_loan_amount' not in st.session_state:
        st.session_state.loaded_avg_prev_loan_amount = None
    if 'loaded_repay_rate' not in st.session_state:
        st.session_state.loaded_repay_rate = None
    if 'records_source' not in st.session_state:
        st.session_state.records_source = None
    
    tab_search, tab_pdf, tab_manual = st.tabs(["🔍 Search Records", "📄 Upload PDF", "✏️ Manual Entry"])
    
    # ===== TAB 1: SEARCH RECORDS =====
    with tab_search:
        people_db, records_db = load_loan_records_db()
        if people_db is not None:
            st.markdown(f"**Database:** {len(people_db):,} people | {len(records_db):,} loan records")
            search_query = st.text_input(
                "🔎 Search by Name or Person ID",
                placeholder="e.g. Rahul Sharma or PER00001",
                key="search_person_input"
            )
            
            if search_query.strip():
                person_info, summary = search_person(people_db, records_db, search_query)
                
                if person_info is not None and summary is not None:
                    # Person found — show info
                    st.success(f"✅ **Record found:** {person_info['Full_Name']} ({person_info['Person_ID']})")
                    
                    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                    with p_col1:
                        st.metric("Gender", person_info['Gender'])
                    with p_col2:
                        st.metric("Age", person_info['Age'])
                    with p_col3:
                        st.metric("Total Loans", summary['prev_loan_count'])
                    with p_col4:
                        st.metric("Repayment Rate", f"{summary['repayment_rate']:.0%}")
                    
                    # Show records table
                    if not summary['records_table'].empty:
                        st.markdown("**Loan Records:**")
                        display_cols = ['Record_ID', 'Loan_Type', 'Loan_Amount', 'Loan_Date', 'Status', 'Lender']
                        display_df = summary['records_table'][display_cols].copy()
                        
                        def style_status(val):
                            if val == 'Repaid':
                                return 'color: #16a34a; font-weight: bold'
                            elif val == 'Defaulted':
                                return 'color: #A91D22; font-weight: bold'
                            else:
                                return 'color: #ca8a04; font-weight: bold'
                        
                        st.dataframe(
                            display_df.style.map(style_status, subset=['Status']),
                            use_container_width=True, hide_index=True
                        )
                    
                    # Load button
                    if st.button("✅ USE THESE RECORDS FOR PREDICTION", key="use_search_records", type="primary"):
                        st.session_state.loaded_prev_loan_count = summary['prev_loan_count']
                        st.session_state.loaded_prev_loans_repaid = summary['prev_loans_repaid']
                        st.session_state.loaded_prev_loan_defaults = summary['prev_loan_defaults']
                        st.session_state.loaded_avg_prev_loan_amount = summary['avg_prev_loan_amount']
                        st.session_state.loaded_repay_rate = summary['repayment_rate']
                        st.session_state.records_source = f"Database: {person_info['Full_Name']}"
                        st.rerun()
                else:
                    st.warning("⚠️ No records found. Try a different name or ID, or upload a PDF.")
                    # Show some suggestions
                    if people_db is not None:
                        suggestions = people_db.sample(min(5, len(people_db)))
                        st.markdown("**Try these sample names:**")
                        for _, row in suggestions.iterrows():
                            st.markdown(f"- `{row['Full_Name']}` ({row['Person_ID']})")
        else:
            st.warning("Records database not found. Please run the database generator first.")
    
    # ===== TAB 2: UPLOAD PDF =====
    with tab_pdf:
        st.markdown("Upload a loan history report PDF and we'll automatically extract past loan data.")
        
        # Download sample PDF
        sample_pdf_bytes = get_sample_pdf_bytes()
        if sample_pdf_bytes:
            st.download_button(
                label="📥 Download Sample PDF Template",
                data=sample_pdf_bytes,
                file_name="sample_loan_report.pdf",
                mime="application/pdf",
                key="download_sample_pdf"
            )
        
        uploaded_file = st.file_uploader(
            "Upload Loan History PDF",
            type=["pdf"],
            key="pdf_uploader",
            help="Upload a PDF containing past loan records. We'll extract loan amounts, statuses, and dates."
        )
        
        if uploaded_file is not None:
            with st.spinner("Parsing PDF document..."):
                try:
                    from src.pdf_parser import parse_loan_pdf
                    result = parse_loan_pdf(uploaded_file.getvalue())
                except ImportError:
                    st.error("PDF parser not available. Please install PyPDF2: `pip install PyPDF2`")
                    result = None
            
            if result and result['success']:
                if result['loans_found'] > 0:
                    st.success(f"✅ **Extracted {result['loans_found']} loan record(s) from PDF**")
                    
                    # Show extracted summary
                    ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)
                    with ex_col1:
                        st.metric("Completed Loans", result['prev_loan_count'])
                    with ex_col2:
                        st.metric("Repaid", result['prev_loans_repaid'])
                    with ex_col3:
                        st.metric("Defaults", result['prev_loan_defaults'])
                    with ex_col4:
                        st.metric("Repayment Rate", f"{result['repayment_rate']:.0%}")
                    
                    # Show extracted records
                    if result['records']:
                        st.markdown("**Extracted Loan Records:**")
                        rec_df = pd.DataFrame(result['records'])
                        st.dataframe(rec_df, use_container_width=True, hide_index=True)
                    
                    # Load button
                    if st.button("✅ USE EXTRACTED DATA FOR PREDICTION", key="use_pdf_records", type="primary"):
                        st.session_state.loaded_prev_loan_count = result['prev_loan_count']
                        st.session_state.loaded_prev_loans_repaid = result['prev_loans_repaid']
                        st.session_state.loaded_prev_loan_defaults = result['prev_loan_defaults']
                        st.session_state.loaded_avg_prev_loan_amount = result['avg_prev_loan_amount']
                        st.session_state.loaded_repay_rate = result['repayment_rate']
                        st.session_state.records_source = f"PDF: {uploaded_file.name}"
                        st.rerun()
                else:
                    st.warning("⚠️ No loan records could be extracted from this PDF. Try manual entry.")
                    with st.expander("🔍 View Extracted Text (Debug)"):
                        st.text(result.get('raw_text', 'No text extracted'))
            elif result:
                st.error(f"❌ {result.get('error', 'Unknown error parsing PDF')}")
    
    # ===== TAB 3: MANUAL ENTRY =====
    with tab_manual:
        st.markdown("Enter past loan history manually.")
        
        lh_col1, lh_col2, lh_col3, lh_col4 = st.columns(4)
        with lh_col1:
            manual_prev_loan_count = st.number_input(
                "📋 Previous Loans Taken",
                min_value=0, max_value=8, value=2, step=1,
                help="Total number of past loans taken (0 = first-time borrower)",
                key="manual_prev_loan_count"
            )
        with lh_col2:
            manual_prev_loans_repaid = st.number_input(
                "✅ Loans Repaid Successfully",
                min_value=0, max_value=max(manual_prev_loan_count, 1),
                value=min(manual_prev_loan_count, 2), step=1,
                help="Number of past loans fully repaid on time",
                key="manual_prev_loans_repaid"
            )
        with lh_col3:
            manual_defaults = manual_prev_loan_count - manual_prev_loans_repaid
            st.markdown(f"""
            <div style="padding: 8px 0;">
                <label style="font-size: 0.875rem; color: #374151; font-weight: 600;">❌ Past Defaults</label>
                <div style="font-size: 2rem; font-weight: 900; color: {'#A91D22' if manual_defaults > 0 else '#16a34a'}; margin-top: 4px;">{manual_defaults}</div>
                <div style="font-size: 0.75rem; color: #6b7280;">Auto-calculated</div>
            </div>
            """, unsafe_allow_html=True)
        with lh_col4:
            manual_avg_prev_loan_amount = st.number_input(
                "💵 Avg Past Loan ($1000s)",
                min_value=0, max_value=500, value=80 if manual_prev_loan_count > 0 else 0, step=10,
                help="Average size of previously taken loans in $1000s",
                key="manual_avg_prev_loan"
            )
        
        manual_repay_rate = round(manual_prev_loans_repaid / manual_prev_loan_count, 2) if manual_prev_loan_count > 0 else 0.0
        
        if st.button("✅ USE MANUAL ENTRY FOR PREDICTION", key="use_manual_records", type="primary"):
            st.session_state.loaded_prev_loan_count = manual_prev_loan_count
            st.session_state.loaded_prev_loans_repaid = manual_prev_loans_repaid
            st.session_state.loaded_prev_loan_defaults = manual_defaults
            st.session_state.loaded_avg_prev_loan_amount = manual_avg_prev_loan_amount
            st.session_state.loaded_repay_rate = manual_repay_rate
            st.session_state.records_source = "Manual Entry"
            st.rerun()
    
    # ===== SHOW LOADED RECORDS STATUS =====
    if st.session_state.loaded_prev_loan_count is not None:
        prev_loan_count = st.session_state.loaded_prev_loan_count
        prev_loans_repaid = st.session_state.loaded_prev_loans_repaid
        prev_loan_defaults = st.session_state.loaded_prev_loan_defaults
        avg_prev_loan_amount = st.session_state.loaded_avg_prev_loan_amount
        repay_rate = st.session_state.loaded_repay_rate
        
        if prev_loan_count == 0:
            hist_label, hist_color, hist_bg = "First-Time Borrower", "#ca8a04", "#fef9c3"
        elif repay_rate >= 0.90:
            hist_label, hist_color, hist_bg = "Excellent History", "#16a34a", "#f0fdf4"
        elif repay_rate >= 0.70:
            hist_label, hist_color, hist_bg = "Good History", "#64dd17", "#f0fdf4"
        elif repay_rate >= 0.50:
            hist_label, hist_color, hist_bg = "Fair History", "#ff6d00", "#fff7ed"
        else:
            hist_label, hist_color, hist_bg = "Poor History", "#dd2c00", "#fef2f2"
        
        st.markdown(f"""
        <div style="background: {hist_bg}; border: 2px solid {hist_color}40; border-radius: 12px; padding: 18px 24px; margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <span style="font-size: 0.7rem; color: #6b7280; text-transform: uppercase; letter-spacing: 2px; font-weight: 700;">LOADED FROM</span>
                    <span style="font-size: 0.85rem; color: {hist_color}; font-weight: 700; margin-left: 8px;">{st.session_state.records_source}</span>
                </div>
                <div style="display: flex; gap: 25px;">
                    <div style="text-align: center;">
                        <div style="font-size: 1.5rem; font-weight: 900; color: {hist_color};">{prev_loan_count}</div>
                        <div style="font-size: 0.7rem; color: #6b7280;">Past Loans</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 1.5rem; font-weight: 900; color: {hist_color};">{repay_rate:.0%}</div>
                        <div style="font-size: 0.7rem; color: #6b7280;">Repayment</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 1.5rem; font-weight: 900; color: {'#A91D22' if prev_loan_defaults > 0 else '#16a34a'};">{prev_loan_defaults}</div>
                        <div style="font-size: 0.7rem; color: #6b7280;">Defaults</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 1.2rem; font-weight: 700; color: {hist_color};">{hist_label}</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Loaded Records", key="clear_records"):
            st.session_state.loaded_prev_loan_count = None
            st.session_state.loaded_prev_loans_repaid = None
            st.session_state.loaded_prev_loan_defaults = None
            st.session_state.loaded_avg_prev_loan_amount = None
            st.session_state.loaded_repay_rate = None
            st.session_state.records_source = None
            st.rerun()
    else:
        # Default values when nothing loaded
        prev_loan_count = 0
        prev_loans_repaid = 0
        prev_loan_defaults = 0
        avg_prev_loan_amount = 0
        repay_rate = 0.0
        
        st.markdown("""
        <div style="background: #f9fafb; border: 1px dashed #d1d5db; border-radius: 12px; padding: 18px 24px; margin-top: 10px; text-align: center;">
            <span style="font-size: 0.85rem; color: #9ca3af;">No records loaded yet. Search, upload a PDF, or enter manually above.</span>
        </div>
        """, unsafe_allow_html=True)

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
            'Credit_Score': credit_score,
            'Credit_Utilization': float(credit_utilization),
            'Open_Accounts': open_accounts,
            'Prev_Loan_Count': prev_loan_count,
            'Prev_Loans_Repaid': prev_loans_repaid,
            'Prev_Loan_Defaults': prev_loan_defaults,
            'Avg_Prev_Loan_Amount': float(avg_prev_loan_amount),
            'Repayment_Rate': float(repay_rate),
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
        
        # --- CREDIT HEALTH SCORECARD ---
        ch = result.get('credit_health', {})
        if ch:
            st.markdown('<div class="section-header">💳 Credit Health Scorecard</div>', unsafe_allow_html=True)
            
            ch_col1, ch_col2, ch_col3 = st.columns(3)
            
            with ch_col1:
                st.plotly_chart(
                    create_credit_score_gauge(ch['credit_score'], ch['grade'], ch['grade_color']),
                    use_container_width=True
                )
            
            with ch_col2:
                st.markdown(f"""
                <div class="credit-health-card">
                    <div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 2px; font-weight: 700;">Credit Utilization</div>
                    <div style="font-size: 3rem; font-weight: 900; color: {'#16a34a' if ch['utilization'] < 30 else '#A91D22'}; margin: 10px 0;">{ch['utilization']:.0f}%</div>
                    <div class="credit-grade-badge" style="background: {'#f0fdf4' if ch['util_status'] in ['Excellent', 'Good'] else '#fef2f2'}; color: {'#166534' if ch['util_status'] in ['Excellent', 'Good'] else '#A91D22'};">
                        {ch['util_status']}
                    </div>
                    <p style="color: #6b7280; font-size: 0.85rem; margin-top: 12px;">{ch['util_advice']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with ch_col3:
                st.markdown(f"""
                <div class="credit-health-card">
                    <div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 2px; font-weight: 700;">Account Depth</div>
                    <div style="font-size: 3rem; font-weight: 900; color: {'#16a34a' if ch['depth_status'] == 'Healthy' else '#ca8a04'}; margin: 10px 0;">{ch['open_accounts']}</div>
                    <div class="credit-grade-badge" style="background: {'#f0fdf4' if ch['depth_status'] == 'Healthy' else '#fef9c3'}; color: {'#166534' if ch['depth_status'] == 'Healthy' else '#92400e'};">
                        {ch['depth_status']}
                    </div>
                    <p style="color: #6b7280; font-size: 0.85rem; margin-top: 12px;">{ch['depth_advice']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Health Index Bar
            st.markdown(f"""
            <div style="background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px 30px; margin-top: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 2px; font-weight: 700;">Composite Credit Health Index</span>
                        <span style="font-size: 0.8rem; color: #9ca3af; margin-left: 10px;">({ch['score_percentile']})</span>
                    </div>
                    <span style="font-size: 2rem; font-weight: 900; color: #A91D22;">{ch['health_index']}/100</span>
                </div>
                <div style="width: 100%; background: #e5e7eb; height: 12px; border-radius: 6px; margin-top: 12px; overflow: hidden;">
                    <div style="width: {ch['health_index']}%; background: linear-gradient(90deg, #A91D22, #ef4444); height: 12px; border-radius: 6px; transition: width 0.5s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
            # --- LOAN HISTORY SCORECARD ---
            st.markdown('<div class="section-header">📜 Loan History Scorecard</div>', unsafe_allow_html=True)
            
            lh_col1, lh_col2, lh_col3 = st.columns(3)
            
            with lh_col1:
                st.markdown(f"""
                <div class="credit-health-card">
                    <div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 2px; font-weight: 700;">Borrowing History</div>
                    <div style="font-size: 3rem; font-weight: 900; color: {ch.get('history_color', '#6b7280')}; margin: 10px 0;">{ch.get('prev_loan_count', 0)}</div>
                    <div style="font-size: 0.85rem; color: #6b7280;">Previous Loan(s)</div>
                    <div class="credit-grade-badge" style="background: {ch.get('history_color', '#6b7280')}15; color: {ch.get('history_color', '#6b7280')};">
                        {ch.get('history_emoji', '')} {ch.get('history_status', 'N/A')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with lh_col2:
                rr_val = ch.get('repayment_rate', 0)
                rr_pct = rr_val * 100 if rr_val <= 1 else rr_val
                st.markdown(f"""
                <div class="credit-health-card">
                    <div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 2px; font-weight: 700;">Repayment Rate</div>
                    <div style="font-size: 3rem; font-weight: 900; color: {'#16a34a' if rr_pct >= 70 else '#A91D22' if rr_pct < 50 else '#ff6d00'}; margin: 10px 0;">{rr_pct:.0f}%</div>
                    <div style="font-size: 0.85rem; color: #6b7280;">{ch.get('prev_loans_repaid', 0)} of {ch.get('prev_loan_count', 0)} repaid</div>
                    <div style="width: 100%; background: #e5e7eb; height: 8px; border-radius: 4px; margin-top: 12px; overflow: hidden;">
                        <div style="width: {rr_pct}%; background: {'#16a34a' if rr_pct >= 70 else '#A91D22'}; height: 8px; border-radius: 4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with lh_col3:
                defaults = ch.get('prev_loan_defaults', 0)
                st.markdown(f"""
                <div class="credit-health-card">
                    <div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 2px; font-weight: 700;">Defaults</div>
                    <div style="font-size: 3rem; font-weight: 900; color: {'#16a34a' if defaults == 0 else '#A91D22'}; margin: 10px 0;">{defaults}</div>
                    <div style="font-size: 0.85rem; color: #6b7280;">{'Clean record ✨' if defaults == 0 else f'{defaults} default(s) detected'}</div>
                    <p style="color: #6b7280; font-size: 0.85rem; margin-top: 12px;">{ch.get('history_advice', '')}</p>
                </div>
                """, unsafe_allow_html=True)
        
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
        st.info(f"🤖 **Model Engine:** {result['model_used']} | Trained on 20,000 synthetic records with FICO scoring")

    # --- WHAT-IF ANALYSIS ---
    st.markdown('<div class="section-header">🧪 Risk Simulation Control</div>', unsafe_allow_html=True)
    st.markdown("Adjust applicant variables to observe real-time impact on approval probability.")
    
    with st.container():
        w_main_col1, w_main_col2 = st.columns([2, 1])
        
        with w_main_col1:
            w_col1, w_col2 = st.columns(2)
            with w_col1:
                w_inc = st.slider("Total Monthly Income ($)", 1000, 50000, 5000, step=500, key="wi_inc")
                w_credit_score = st.slider("FICO Credit Score", 300, 850, 700, step=10, key="wi_cs")
                w_married = st.selectbox("Married Status", ["Yes", "No"], index=0, key="wi_married")
            with w_col2:
                w_loan = st.slider("Loan Amount Request ($1000s)", 10, 700, 150, step=10, key="wi_loan")
                w_util = st.slider("Credit Utilization (%)", 0, 100, 25, step=5, key="wi_util")
                w_edu = st.selectbox("Education Level", ["Graduate", "Not Graduate"], index=0, key="wi_edu")
                w_area = st.selectbox("Property Type", ["Urban", "Semiurban", "Rural"], index=1, key="wi_area")
        
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
            'Credit_Score': w_credit_score,
            'Credit_Utilization': float(w_util),
            'Open_Accounts': 4,
            'Prev_Loan_Count': 2,
            'Prev_Loans_Repaid': 2,
            'Prev_Loan_Defaults': 0,
            'Avg_Prev_Loan_Amount': 80.0,
            'Repayment_Rate': 1.0,
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
                <p style="margin-top: 10px; font-size: 0.75rem; color: #fca5a5;">FICO: {w_credit_score} | Util: {w_util}%</p>
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
    loan eligibility assessment. It uses advanced classification algorithms combined with **FICO Credit Intelligence** 
    to analyze applicant profiles and predict loan approval status with high accuracy.
    
    This project demonstrates the complete ML lifecycle — from data preprocessing, credit scoring, 
    and feature engineering to model training, evaluation, and deployment as an interactive web dashboard.
    """)
    
    st.markdown('<div class="section-header">🔬 Methodology</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Data Engineering
        - **Record Count**: 20,000 synthetic samples
        - **Credit Intelligence**: FICO Score (300-850), Credit Utilization (0-100%), Open Accounts (0-15)
        - **Feature Engineering**: `CreditHealthIndex`, `CreditStress`, `AffordabilityIndex`, `CreditDepthScore`, `StabilityScore`
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
        - 💳 **FICO Credit Intelligence**: Granular 300-850 scoring with utilization & account depth
        - 🧠 **Credit Health Scorecard**: Composite health index with tier analysis
        - 📊 **Dynamic EDA**: Interactive Plotly-based deep dives with credit tier breakdowns
        - 🔮 **Explainable AI**: Local factor attribution for every prediction
        - 🧪 **What-If Simulations**: Real-time financial & credit scenario testing
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
    | **Category** | Supervised ML — Binary Classification with FICO Credit Intelligence |
    """)


# === FOOTER ===
st.markdown("""
<div class="footer" style="background: #111827; border-top: 10px solid #A91D22; color: #ffffff; padding: 60px 20px;">
    <p style="font-weight: 900; font-size: 1.4rem; margin-bottom: 10px; color: #ffffff; letter-spacing: -1px;">EcoLoan Intel Pro Dashboard</p>
    <p style="color: #9ca3af; font-size: 0.9rem; margin-bottom: 25px;">FICO Credit Intelligence • High-Fidelity ML Risk Assessment</p>
    <div style="width: 50px; height: 3px; background: #ffffff; margin: 0 auto 25px auto;"></div>
    <p style="margin-top: 10px; color: #d1d5db;">
        Built by <a href="https://github.com/Kinshunk565" target="_blank" style="color: #ffffff; text-decoration: underline;">Kinshunk Garg</a> • 
        <a href="https://github.com/Kinshunk565" target="_blank" style="color: #ffffff;">GitHub</a> • 
        Live on <a href="https://render.com" target="_blank" style="color: #ffffff;">Render</a>
    </p>
</div>
""", unsafe_allow_html=True)
