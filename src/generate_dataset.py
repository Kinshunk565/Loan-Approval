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

def generate_loan_dataset(n_samples: int = 20000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic loan dataset with 20k samples."""
    np.random.seed(seed)
    
    # --- Loan IDs ---
    loan_ids = [f"LP{str(i).zfill(6)}" for i in range(1, n_samples + 1)]
    
    # --- Gender (75% Male, 25% Female) ---
    gender = np.random.choice(
        ['Male', 'Female'], n_samples, p=[0.75, 0.25]
    )
    
    # --- Married (60% Yes, 40% No) ---
    married = np.random.choice(
        ['Yes', 'No'], n_samples, p=[0.60, 0.40]
    )
    
    # --- Dependents (0: 50%, 1: 18%, 2: 18%, 3+: 14%) ---
    dependents = np.random.choice(
        ['0', '1', '2', '3+'], n_samples, p=[0.50, 0.18, 0.18, 0.14]
    )
    
    # --- Education (75% Graduate, 25% Not Graduate) ---
    education = np.random.choice(
        ['Graduate', 'Not Graduate'], n_samples, p=[0.75, 0.25]
    )
    
    # --- Self Employed (15% Yes, 85% No) ---
    self_employed = np.random.choice(
        ['Yes', 'No'], n_samples, p=[0.15, 0.85]
    )
    
    # --- Applicant Income (improved distribution) ---
    applicant_income = np.random.lognormal(
        mean=8.4, sigma=0.6, size=n_samples
    ).astype(int)
    applicant_income = np.clip(applicant_income, 1200, 100000)
    
    # --- Coapplicant Income ---
    has_coapplicant = np.random.choice([True, False], n_samples, p=[0.50, 0.50])
    coapplicant_income = np.where(
        has_coapplicant,
        np.random.lognormal(mean=7.8, sigma=0.7, size=n_samples).astype(int),
        0
    )
    coapplicant_income = np.clip(coapplicant_income, 0, 50000)
    
    # --- Loan Amount (correlated with total income) ---
    total_inc = applicant_income + coapplicant_income
    base_loan = total_inc * np.random.uniform(0.015, 0.12, n_samples)
    loan_amount = np.clip(base_loan, 10, 800).astype(int)
    
    # --- Loan Amount Term ---
    loan_amount_term = np.random.choice(
        [12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
        n_samples,
        p=[0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.06, 0.72, 0.03]
    ).astype(float)
    
    # --- Credit History (80% have good history) ---
    credit_history = np.random.choice(
        [1.0, 0.0], n_samples, p=[0.80, 0.20]
    )
    
    # --- Property Area ---
    property_area = np.random.choice(
        ['Urban', 'Semiurban', 'Rural'], n_samples, p=[0.33, 0.40, 0.27]
    )
    
    # --- Loan Status Logic (More complex) ---
    loan_status = []
    for i in range(n_samples):
        score = 0.0
        
        # Credit history is critical
        if credit_history[i] == 1.0:
            score += 45
        else:
            score -= 35
        
        # Income to Loan Ratio (PTI - Payment to Income proxy)
        t_income = total_inc[i]
        l_amount = loan_amount[i]
        if l_amount > 0:
            ratio = t_income / (l_amount * 1000) # Basic ratio
            if ratio > 0.6: score += 25
            elif ratio > 0.4: score += 15
            elif ratio > 0.2: score += 5
            else: score -= 20
        
        # Education and Stability
        if education[i] == 'Graduate': score += 10
        if married[i] == 'Yes' and gender[i] == 'Female': score += 5 # Higher stability statistical bias
        
        # Property area
        if property_area[i] == 'Semiurban': score += 8
        elif property_area[i] == 'Urban': score += 4
        
        # Dependents penalty for high loan amounts
        if l_amount > 300 and dependents[i] in ['2', '3+']:
            score -= 10
            
        # Add noise
        score += np.random.normal(0, 15)
        
        loan_status.append('Y' if score >= 40 else 'N')
    
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
    print("Generating realistic loan dataset...")
    df = generate_loan_dataset(20000)
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "loan_data.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Approval Rate: {(df['Loan_Status'] == 'Y').mean():.1%}")
    print("\nMissing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("\nSample Data:")
    print(df.head())
