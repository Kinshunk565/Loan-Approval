"""
Loan Approval Prediction — Realistic Synthetic Dataset Generator
================================================================
Generates a high-quality synthetic dataset that mirrors real-world
financial lending data with granular credit scoring.

Features:
- FICO Credit Score (300-850) as primary risk factor
- Credit Utilization Ratio (0-100%) for portfolio health
- Number of Open Credit Accounts for credit depth
- Past Loan History (previous loans, repayment, defaults)
- Realistic correlations between income, loan amount, and approval
- Proper class imbalance (~65% approved, ~35% rejected)
- Missing values injected naturally (as in real-world data)
- 20,000 rows for robust model training

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
    
    # --- FICO Credit Score (300-850) ---
    # Realistic distribution: Most people are in 600-750 range
    # Graduates and employed people tend to have higher scores
    base_credit_score = np.random.normal(680, 80, n_samples)
    # Education bonus
    edu_bonus = np.where(education == 'Graduate', np.random.uniform(10, 40, n_samples), 0)
    # Married stability bonus
    married_bonus = np.where(married == 'Yes', np.random.uniform(5, 20, n_samples), 0)
    # Self-employed slight penalty (irregular income)
    se_penalty = np.where(self_employed == 'Yes', np.random.uniform(-30, 0, n_samples), 0)
    # Income correlation — higher income = slightly better score
    income_bonus = np.clip((applicant_income - 5000) / 10000 * 15, -20, 30)
    
    credit_score = base_credit_score + edu_bonus + married_bonus + se_penalty + income_bonus
    credit_score = np.clip(credit_score, 300, 850).astype(int)
    
    # --- Credit Utilization Ratio (0-100%) ---
    # Lower is better. People with high scores tend to have lower utilization
    base_utilization = np.random.beta(2, 5, n_samples) * 100  # Skewed towards lower values
    # People with low credit scores tend to max out cards
    score_penalty = np.clip((700 - credit_score) / 10, 0, 30)
    credit_utilization = np.clip(base_utilization + score_penalty, 0, 100).round(1)
    
    # --- Number of Open Credit Accounts (0-15) ---
    # Most people have 2-6 accounts
    open_accounts = np.random.poisson(lam=4, size=n_samples)
    open_accounts = np.clip(open_accounts, 0, 15).astype(int)
    
    # --- Property Area ---
    property_area = np.random.choice(
        ['Urban', 'Semiurban', 'Rural'], n_samples, p=[0.33, 0.40, 0.27]
    )
    
    # ===== PAST LOAN HISTORY =====
    # Previous Loan Count (0-8): ~25% are first-time borrowers (0),
    # most have 1-4 previous loans
    prev_loan_count = np.random.choice(
        [0, 1, 2, 3, 4, 5, 6, 7, 8], n_samples,
        p=[0.25, 0.20, 0.18, 0.14, 0.10, 0.06, 0.04, 0.02, 0.01]
    )
    
    # Previous Loans Repaid: correlated with credit score
    # High credit score → most loans repaid; low score → more defaults
    prev_loans_repaid = np.zeros(n_samples, dtype=int)
    prev_loan_defaults = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        plc = prev_loan_count[i]
        if plc == 0:
            prev_loans_repaid[i] = 0
            prev_loan_defaults[i] = 0
        else:
            cs = credit_score[i]
            # Higher credit score → higher likelihood of full repayment
            if cs >= 750:
                repay_prob = np.random.uniform(0.90, 1.0)
            elif cs >= 700:
                repay_prob = np.random.uniform(0.80, 0.95)
            elif cs >= 650:
                repay_prob = np.random.uniform(0.65, 0.85)
            elif cs >= 580:
                repay_prob = np.random.uniform(0.45, 0.70)
            else:
                repay_prob = np.random.uniform(0.15, 0.50)
            
            repaid = int(np.round(plc * repay_prob))
            repaid = min(repaid, plc)
            prev_loans_repaid[i] = repaid
            prev_loan_defaults[i] = plc - repaid
    
    # Average Previous Loan Amount ($1000s): correlated with income
    avg_prev_loan_amount = np.zeros(n_samples)
    for i in range(n_samples):
        if prev_loan_count[i] == 0:
            avg_prev_loan_amount[i] = 0
        else:
            # Base on income with some randomness
            base_avg = total_inc[i] * np.random.uniform(0.01, 0.08)
            avg_prev_loan_amount[i] = round(np.clip(base_avg, 5, 500), 1)
    
    # Repayment Rate (derived: repaid / count, 0 for first-timers)
    repayment_rate = np.where(
        prev_loan_count > 0,
        np.round(prev_loans_repaid / prev_loan_count, 2),
        0.0
    )
    
    # --- Loan Status Logic (Credit-Score-Driven + Loan History) ---
    loan_status = []
    for i in range(n_samples):
        score = 0.0
        
        # ===== CREDIT SCORE is the PRIMARY driver =====
        cs = credit_score[i]
        if cs >= 750:
            score += 50   # Exceptional
        elif cs >= 700:
            score += 35   # Very Good
        elif cs >= 650:
            score += 20   # Good
        elif cs >= 580:
            score += 5    # Fair
        else:
            score -= 40   # Poor — very hard to get approved
        
        # ===== Credit Utilization =====
        cu = credit_utilization[i]
        if cu < 30:
            score += 15   # Excellent utilization
        elif cu < 50:
            score += 5    # Acceptable
        elif cu < 75:
            score -= 5    # Warning zone
        else:
            score -= 20   # Maxed out — big red flag
        
        # ===== Open Accounts (sweet spot is 3-7) =====
        oa = open_accounts[i]
        if 3 <= oa <= 7:
            score += 5    # Healthy credit mix
        elif oa == 0:
            score -= 15   # No credit history — thin file
        elif oa > 10:
            score -= 10   # Too many open lines
        
        # ===== PAST LOAN HISTORY =====
        plc = prev_loan_count[i]
        pld = prev_loan_defaults[i]
        rr = repayment_rate[i]
        
        if plc == 0:
            # First-time borrower — slight thin-file penalty
            score -= 5
        else:
            # Repayment rate bonus/penalty
            if rr >= 0.95:
                score += 20   # Near-perfect repayment history
            elif rr >= 0.80:
                score += 12   # Strong repayment record
            elif rr >= 0.60:
                score += 3    # Moderate repayment
            elif rr >= 0.40:
                score -= 8    # Poor repayment
            else:
                score -= 20   # Very poor — majority defaulted
            
            # Defaults are a major red flag
            if pld >= 3:
                score -= 25   # Serial defaulter
            elif pld == 2:
                score -= 15   # Multiple defaults
            elif pld == 1:
                score -= 5    # Single default — concerning
            
            # Experience bonus — repeat borrowers with good track records
            if plc >= 3 and rr >= 0.80:
                score += 8    # Experienced reliable borrower
        
        # ===== Income to Loan Ratio (PTI proxy) =====
        t_income = total_inc[i]
        l_amount = loan_amount[i]
        if l_amount > 0:
            ratio = t_income / (l_amount * 1000)
            if ratio > 0.6: score += 20
            elif ratio > 0.4: score += 12
            elif ratio > 0.2: score += 3
            else: score -= 15
        
        # ===== Education and Stability =====
        if education[i] == 'Graduate': score += 8
        if married[i] == 'Yes': score += 4
        
        # ===== Property area =====
        if property_area[i] == 'Semiurban': score += 6
        elif property_area[i] == 'Urban': score += 3
        
        # ===== Dependents penalty for high loan amounts =====
        if l_amount > 300 and dependents[i] in ['2', '3+']:
            score -= 8
            
        # Add noise
        score += np.random.normal(0, 12)
        
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
        'Credit_Score': credit_score,
        'Credit_Utilization': credit_utilization,
        'Open_Accounts': open_accounts,
        'Prev_Loan_Count': prev_loan_count,
        'Prev_Loans_Repaid': prev_loans_repaid,
        'Prev_Loan_Defaults': prev_loan_defaults,
        'Avg_Prev_Loan_Amount': avg_prev_loan_amount,
        'Repayment_Rate': repayment_rate,
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
    
    # Credit_Score: ~4% missing (some applicants don't disclose full history)
    mask = np.random.random(n_samples) < 0.04
    df.loc[mask, 'Credit_Score'] = np.nan
    
    # Credit_Utilization: ~3% missing
    mask = np.random.random(n_samples) < 0.03
    df.loc[mask, 'Credit_Utilization'] = np.nan
    
    # Open_Accounts: ~2% missing
    mask = np.random.random(n_samples) < 0.02
    df.loc[mask, 'Open_Accounts'] = np.nan
    
    # Prev_Loan_Count: ~3% missing (some applicants don't disclose full history)
    mask = np.random.random(n_samples) < 0.03
    df.loc[mask, 'Prev_Loan_Count'] = np.nan
    
    # Prev_Loans_Repaid: ~3% missing
    mask = np.random.random(n_samples) < 0.03
    df.loc[mask, 'Prev_Loans_Repaid'] = np.nan
    
    # Prev_Loan_Defaults: ~3% missing
    mask = np.random.random(n_samples) < 0.03
    df.loc[mask, 'Prev_Loan_Defaults'] = np.nan
    
    # Avg_Prev_Loan_Amount: ~4% missing
    mask = np.random.random(n_samples) < 0.04
    df.loc[mask, 'Avg_Prev_Loan_Amount'] = np.nan
    
    # Repayment_Rate: ~3% missing
    mask = np.random.random(n_samples) < 0.03
    df.loc[mask, 'Repayment_Rate'] = np.nan
    
    return df


if __name__ == "__main__":
    print("Generating realistic loan dataset with FICO scoring + Past Loan History...")
    df = generate_loan_dataset(20000)
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "loan_data.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Approval Rate: {(df['Loan_Status'] == 'Y').mean():.1%}")
    print(f"\nCredit Score Stats:")
    print(f"  Mean: {df['Credit_Score'].mean():.0f}")
    print(f"  Median: {df['Credit_Score'].median():.0f}")
    print(f"  Min: {df['Credit_Score'].min():.0f}")
    print(f"  Max: {df['Credit_Score'].max():.0f}")
    print(f"\nCredit Utilization Stats:")
    print(f"  Mean: {df['Credit_Utilization'].mean():.1f}%")
    print(f"\nPast Loan History Stats:")
    print(f"  Avg Previous Loans: {df['Prev_Loan_Count'].mean():.1f}")
    print(f"  Avg Repayment Rate: {df['Repayment_Rate'].mean():.2f}")
    print(f"  Avg Defaults: {df['Prev_Loan_Defaults'].mean():.2f}")
    print(f"  First-Time Borrowers: {(df['Prev_Loan_Count'] == 0).mean():.1%}")
    print(f"\nMissing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("\nSample Data:")
    print(df.head())
