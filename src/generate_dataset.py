"""
Loan Approval Prediction — Realistic Synthetic Dataset Generator
================================================================
Generates a high-quality synthetic dataset that mirrors the distribution
of the classic Analytics Vidhya / Kaggle Loan Prediction dataset.

Features:
- Realistic correlations between income, loan amount, and approval
- Proper class imbalance (~68% approved, ~32% rejected)
- Missing values injected naturally (as in real-world data)
- 5000 rows for robust model training

Author: Kinshunk Garg
"""

import pandas as pd
import numpy as np
import os

def generate_loan_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic loan dataset."""
    np.random.seed(seed)
    
    # --- Loan IDs ---
    loan_ids = [f"LP{str(i).zfill(5)}" for i in range(1, n_samples + 1)]
    
    # --- Gender (80% Male, 20% Female — mirrors real dataset) ---
    gender = np.random.choice(
        ['Male', 'Female'], n_samples, p=[0.80, 0.20]
    )
    
    # --- Married (65% Yes, 35% No) ---
    married = np.random.choice(
        ['Yes', 'No'], n_samples, p=[0.65, 0.35]
    )
    
    # --- Dependents (0: 55%, 1: 17%, 2: 17%, 3+: 11%) ---
    dependents = np.random.choice(
        ['0', '1', '2', '3+'], n_samples, p=[0.55, 0.17, 0.17, 0.11]
    )
    
    # --- Education (78% Graduate, 22% Not Graduate) ---
    education = np.random.choice(
        ['Graduate', 'Not Graduate'], n_samples, p=[0.78, 0.22]
    )
    
    # --- Self Employed (14% Yes, 86% No) ---
    self_employed = np.random.choice(
        ['Yes', 'No'], n_samples, p=[0.14, 0.86]
    )
    
    # --- Applicant Income (log-normal distribution, realistic range) ---
    applicant_income = np.random.lognormal(
        mean=8.3, sigma=0.7, size=n_samples
    ).astype(int)
    applicant_income = np.clip(applicant_income, 1000, 80000)
    
    # --- Coapplicant Income (many zeros, some with income) ---
    has_coapplicant = np.random.choice([True, False], n_samples, p=[0.45, 0.55])
    coapplicant_income = np.where(
        has_coapplicant,
        np.random.lognormal(mean=7.5, sigma=0.8, size=n_samples).astype(int),
        0
    )
    coapplicant_income = np.clip(coapplicant_income, 0, 40000)
    
    # --- Loan Amount (correlated with income, log-normal) ---
    base_loan = (applicant_income + coapplicant_income) * np.random.uniform(0.02, 0.08, n_samples)
    loan_amount = np.clip(base_loan, 9, 700).astype(int)
    
    # --- Loan Amount Term (mostly 360 months) ---
    loan_amount_term = np.random.choice(
        [12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
        n_samples,
        p=[0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.06, 0.72, 0.03]
    ).astype(float)
    
    # --- Credit History (85% have good history = 1.0) ---
    credit_history = np.random.choice(
        [1.0, 0.0], n_samples, p=[0.85, 0.15]
    )
    
    # --- Property Area ---
    property_area = np.random.choice(
        ['Urban', 'Semiurban', 'Rural'], n_samples, p=[0.33, 0.38, 0.29]
    )
    
    # --- Loan Status (Target — based on realistic scoring logic) ---
    loan_status = []
    for i in range(n_samples):
        score = 0.0
        
        # Credit history is the strongest predictor
        if credit_history[i] == 1.0:
            score += 40
        else:
            score -= 25
        
        # Income to loan ratio
        total_income = applicant_income[i] + coapplicant_income[i]
        if loan_amount[i] > 0:
            ratio = total_income / (loan_amount[i] * 1000)
            if ratio > 0.5:
                score += 20
            elif ratio > 0.3:
                score += 10
            elif ratio > 0.15:
                score += 5
            else:
                score -= 10
        
        # Education
        if education[i] == 'Graduate':
            score += 8
        
        # Property area (semiurban has higher approval)
        if property_area[i] == 'Semiurban':
            score += 7
        elif property_area[i] == 'Urban':
            score += 3
        
        # Marriage status
        if married[i] == 'Yes':
            score += 4
        
        # Self employed (slight negative)
        if self_employed[i] == 'Yes':
            score -= 3
        
        # Dependents
        dep = dependents[i]
        if dep == '0':
            score += 3
        elif dep in ['2', '3+']:
            score -= 3
        
        # Add randomness for realism
        score += np.random.normal(0, 12)
        
        # Decision threshold
        loan_status.append('Y' if score >= 35 else 'N')
    
    # --- Build DataFrame ---
    df = pd.DataFrame({
        'Loan_ID': loan_ids,
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area,
        'Loan_Status': loan_status
    })
    
    # --- Inject Missing Values (realistic) ---
    # Gender: ~2% missing
    mask = np.random.random(n_samples) < 0.02
    df.loc[mask, 'Gender'] = np.nan
    
    # Married: ~0.5% missing
    mask = np.random.random(n_samples) < 0.005
    df.loc[mask, 'Married'] = np.nan
    
    # Dependents: ~2.5% missing
    mask = np.random.random(n_samples) < 0.025
    df.loc[mask, 'Dependents'] = np.nan
    
    # Self_Employed: ~5% missing
    mask = np.random.random(n_samples) < 0.05
    df.loc[mask, 'Self_Employed'] = np.nan
    
    # LoanAmount: ~3.5% missing
    mask = np.random.random(n_samples) < 0.035
    df.loc[mask, 'LoanAmount'] = np.nan
    
    # Loan_Amount_Term: ~2% missing
    mask = np.random.random(n_samples) < 0.02
    df.loc[mask, 'Loan_Amount_Term'] = np.nan
    
    # Credit_History: ~7.5% missing
    mask = np.random.random(n_samples) < 0.075
    df.loc[mask, 'Credit_History'] = np.nan
    
    return df


if __name__ == "__main__":
    print("🔄 Generating realistic loan dataset...")
    df = generate_loan_dataset(5000)
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "loan_data.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Dataset saved to: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Approval Rate: {(df['Loan_Status'] == 'Y').mean():.1%}")
    print(f"\n📊 Missing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print(f"\n📋 Sample Data:")
    print(df.head())
