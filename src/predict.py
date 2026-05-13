"""
Loan Approval Prediction — Prediction Module
=============================================
Loads the saved model and preprocessor to make predictions
on new applicant data with confidence scores, credit health
analysis, past loan history assessment, and actionable explanations.

Author: Kinshunk Garg
"""

import os
import json
import numpy as np
import pandas as pd
import joblib


class LoanPredictor:
    """Production-ready prediction interface with credit intelligence."""
    
    def __init__(self, models_dir: str = None):
        self.models_dir = models_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models"
        )
        self.model = None
        self.preprocessor = None
        self.feature_importance = None
        self.model_metrics = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model, preprocessor, and metadata."""
        # Load model
        model_path = os.path.join(self.models_dir, "best_model.joblib")
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        
        # Load preprocessor
        preprocessor_path = os.path.join(self.models_dir, "preprocessor.joblib")
        if os.path.exists(preprocessor_path):
            from src.data_preprocessing import LoanDataPreprocessor
            self.preprocessor = LoanDataPreprocessor()
            self.preprocessor.load(preprocessor_path)
        
        # Load feature importance
        importance_path = os.path.join(self.models_dir, "feature_importance.json")
        if os.path.exists(importance_path):
            with open(importance_path, 'r') as f:
                self.feature_importance = json.load(f)
        
        # Load metrics
        metrics_path = os.path.join(self.models_dir, "model_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.model_metrics = json.load(f)
    
    def predict(self, applicant_data: dict) -> dict:
        """
        Make a prediction for a single applicant with deep reasoning.
        """
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model or preprocessor not loaded. Run training first.")
        
        # Preprocess
        X = self.preprocessor.preprocess_single(applicant_data)
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        approval_prob = float(probabilities[1])
        rejection_prob = float(probabilities[0])
        
        # Risk assessment
        if approval_prob >= 0.85:
            risk_level = "Very Low Risk"
            risk_color = "#00c853"
        elif approval_prob >= 0.70:
            risk_level = "Low Risk"
            risk_color = "#64dd17"
        elif approval_prob >= 0.55:
            risk_level = "Moderate Risk"
            risk_color = "#ffab00"
        elif approval_prob >= 0.40:
            risk_level = "High Risk"
            risk_color = "#ff6d00"
        else:
            risk_level = "Very High Risk"
            risk_color = "#dd2c00"
        
        # Contributing factors
        factors = self._get_contributing_factors(applicant_data)
        
        # Credit health analysis
        credit_health = self._analyze_credit_health(applicant_data)
        
        # Actionable recommendations (important for rejection)
        recommendations = self.get_actionable_recommendations(applicant_data, approval_prob)
        
        return {
            'approved': bool(prediction == 1),
            'approval_probability': round(approval_prob * 100, 2),
            'rejection_probability': round(rejection_prob * 100, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'contributing_factors': factors,
            'credit_health': credit_health,
            'recommendations': recommendations,
            'model_used': self.model_metrics.get('best_model', 'Unknown') if self.model_metrics else 'Unknown'
        }
    
    def _analyze_credit_health(self, data: dict) -> dict:
        """Analyze the applicant's credit portfolio health."""
        credit_score = data.get('Credit_Score', 650)
        utilization = data.get('Credit_Utilization', 50)
        open_accounts = data.get('Open_Accounts', 3)
        
        # Credit Grade
        if credit_score >= 800:
            grade = "Exceptional"
            grade_color = "#00c853"
            grade_emoji = "🌟"
        elif credit_score >= 740:
            grade = "Very Good"
            grade_color = "#64dd17"
            grade_emoji = "✅"
        elif credit_score >= 670:
            grade = "Good"
            grade_color = "#ffab00"
            grade_emoji = "👍"
        elif credit_score >= 580:
            grade = "Fair"
            grade_color = "#ff6d00"
            grade_emoji = "⚠️"
        else:
            grade = "Poor"
            grade_color = "#dd2c00"
            grade_emoji = "❌"
        
        # Utilization Assessment
        if utilization < 10:
            util_status = "Excellent"
            util_advice = "Outstanding credit discipline."
        elif utilization < 30:
            util_status = "Good"
            util_advice = "Well within recommended limits."
        elif utilization < 50:
            util_status = "Fair"
            util_advice = "Consider reducing utilization below 30% for better rates."
        elif utilization < 75:
            util_status = "Warning"
            util_advice = "High utilization is hurting your credit profile."
        else:
            util_status = "Critical"
            util_advice = "Near-maxed credit lines are a major red flag for lenders."
        
        # Account Depth Assessment
        if open_accounts == 0:
            depth_status = "Thin File"
            depth_advice = "No credit history detected. Open a starter credit card."
        elif open_accounts < 3:
            depth_status = "Limited"
            depth_advice = "Building more diverse credit accounts will improve your profile."
        elif open_accounts <= 7:
            depth_status = "Healthy"
            depth_advice = "Good credit mix demonstrating responsible management."
        elif open_accounts <= 10:
            depth_status = "Heavy"
            depth_advice = "Consider consolidating some accounts to reduce complexity."
        else:
            depth_status = "Overleveraged"
            depth_advice = "Too many open lines may signal overleveraging to lenders."
        
        # ===== Past Loan History Assessment =====
        prev_loan_count = data.get('Prev_Loan_Count', 0)
        prev_loans_repaid = data.get('Prev_Loans_Repaid', 0)
        prev_loan_defaults = data.get('Prev_Loan_Defaults', 0)
        avg_prev_loan_amount = data.get('Avg_Prev_Loan_Amount', 0)
        repayment_rate = data.get('Repayment_Rate', 0)
        
        if prev_loan_count == 0:
            history_status = "First-Time Borrower"
            history_emoji = "🆕"
            history_color = "#ca8a04"
            history_advice = "No loan track record. Building a positive borrowing history will strengthen future applications."
        elif repayment_rate >= 0.90:
            history_status = "Excellent"
            history_emoji = "🌟"
            history_color = "#00c853"
            history_advice = f"Outstanding! {prev_loans_repaid}/{prev_loan_count} loans repaid successfully."
        elif repayment_rate >= 0.70:
            history_status = "Good"
            history_emoji = "✅"
            history_color = "#64dd17"
            history_advice = f"{prev_loans_repaid}/{prev_loan_count} loans repaid. A strong track record."
        elif repayment_rate >= 0.50:
            history_status = "Fair"
            history_emoji = "⚠️"
            history_color = "#ff6d00"
            history_advice = f"Mixed record: {prev_loans_repaid} repaid, {prev_loan_defaults} defaulted. Focus on clearing outstanding debts."
        else:
            history_status = "Poor"
            history_emoji = "❌"
            history_color = "#dd2c00"
            history_advice = f"Concerning: {prev_loan_defaults} defaults out of {prev_loan_count} loans. Rebuild trust with smaller, manageable loans."
        
        # Credit Health Index (0-100 composite) — now includes loan history
        history_score = 0
        if prev_loan_count > 0:
            history_score = repayment_rate * 15 - (prev_loan_defaults * 3)
        else:
            history_score = 5  # Neutral for first-timers
        
        health_index = round(
            (credit_score / 850 * 40) +            # 40% weight on score
            ((100 - utilization) / 100 * 25) +      # 25% weight on utilization
            (min(open_accounts, 7) / 7 * 15) +      # 15% weight on account depth
            max(history_score, 0),                   # 20% weight on loan history
            1
        )
        
        return {
            'credit_score': credit_score,
            'grade': grade,
            'grade_color': grade_color,
            'grade_emoji': grade_emoji,
            'utilization': utilization,
            'util_status': util_status,
            'util_advice': util_advice,
            'open_accounts': open_accounts,
            'depth_status': depth_status,
            'depth_advice': depth_advice,
            'health_index': min(health_index, 100),
            'score_percentile': self._get_score_percentile(credit_score),
            # Loan History
            'prev_loan_count': prev_loan_count,
            'prev_loans_repaid': prev_loans_repaid,
            'prev_loan_defaults': prev_loan_defaults,
            'avg_prev_loan_amount': avg_prev_loan_amount,
            'repayment_rate': repayment_rate,
            'history_status': history_status,
            'history_emoji': history_emoji,
            'history_color': history_color,
            'history_advice': history_advice
        }
    
    def _get_score_percentile(self, score: int) -> str:
        """Estimate where this score falls in the population."""
        if score >= 800: return "Top 5%"
        elif score >= 750: return "Top 15%"
        elif score >= 700: return "Top 35%"
        elif score >= 650: return "Top 55%"
        elif score >= 600: return "Top 70%"
        elif score >= 550: return "Top 85%"
        else: return "Bottom 15%"
    
    def _get_contributing_factors(self, data: dict) -> list:
        """Determine which factors most influenced the decision."""
        factors = []
        
        # Credit Score (Primary Factor)
        credit_score = data.get('Credit_Score', 650)
        if credit_score >= 700:
            factors.append({
                'factor': 'Credit Score',
                'impact': 'positive',
                'detail': f'FICO score of {credit_score} demonstrates strong creditworthiness.'
            })
        elif credit_score >= 580:
            factors.append({
                'factor': 'Credit Score',
                'impact': 'neutral',
                'detail': f'FICO score of {credit_score} is acceptable but leaves room for improvement.'
            })
        else:
            factors.append({
                'factor': 'Credit Score',
                'impact': 'negative',
                'detail': f'FICO score of {credit_score} is below the minimum preferred threshold of 580.'
            })
        
        # Credit Utilization
        utilization = data.get('Credit_Utilization', 50)
        if utilization < 30:
            factors.append({
                'factor': 'Credit Utilization',
                'impact': 'positive',
                'detail': f'Utilization of {utilization:.0f}% shows excellent credit discipline.'
            })
        elif utilization < 50:
            factors.append({
                'factor': 'Credit Utilization',
                'impact': 'neutral',
                'detail': f'Utilization of {utilization:.0f}% is moderate; aim for below 30%.'
            })
        else:
            factors.append({
                'factor': 'Credit Utilization',
                'impact': 'negative',
                'detail': f'Utilization of {utilization:.0f}% signals potential financial strain.'
            })
        
        # Income vs Loan
        income = data.get('ApplicantIncome', 0) + data.get('CoapplicantIncome', 0)
        loan = data.get('LoanAmount', 1)
        ratio = (loan * 1000) / max(income, 1)
        
        if ratio < 20:
            factors.append({
                'factor': 'Loan-to-Income Ratio',
                'impact': 'positive',
                'detail': f'Your requested loan amount is moderate compared to your income.'
            })
        elif ratio < 50:
            factors.append({
                'factor': 'Loan-to-Income Ratio',
                'impact': 'neutral',
                'detail': f'Your requested loan amount is significant relative to income.'
            })
        else:
            factors.append({
                'factor': 'Loan-to-Income Ratio',
                'impact': 'negative',
                'detail': f'Requested loan is very high ({ratio:.1f}x) relative to income.'
            })
        
        # Open Accounts
        open_accts = data.get('Open_Accounts', 3)
        if 3 <= open_accts <= 7:
            factors.append({
                'factor': 'Credit Account Depth',
                'impact': 'positive',
                'detail': f'{open_accts} open accounts shows a healthy credit mix.'
            })
        elif open_accts == 0:
            factors.append({
                'factor': 'Credit Account Depth',
                'impact': 'negative',
                'detail': 'No open credit accounts — thin file risk.'
            })
        
        # Stability
        is_grad = data.get('Education', '') == 'Graduate'
        is_employed = data.get('Self_Employed', '') == 'No'
        if is_grad and is_employed:
            factors.append({
                'factor': 'Employment Stability',
                'impact': 'positive',
                'detail': 'Being a graduate with salaried employment indicates high stability.'
            })
        
        # ===== Past Loan History =====
        prev_loan_count = data.get('Prev_Loan_Count', 0)
        prev_loan_defaults = data.get('Prev_Loan_Defaults', 0)
        repayment_rate = data.get('Repayment_Rate', 0)
        
        if prev_loan_count == 0:
            factors.append({
                'factor': 'Loan History',
                'impact': 'neutral',
                'detail': 'First-time borrower — no previous loan track record to evaluate.'
            })
        elif repayment_rate >= 0.90 and prev_loan_defaults == 0:
            factors.append({
                'factor': 'Loan History',
                'impact': 'positive',
                'detail': f'Excellent repayment record: {prev_loan_count} past loan(s), all repaid successfully.'
            })
        elif repayment_rate >= 0.70:
            factors.append({
                'factor': 'Loan History',
                'impact': 'positive',
                'detail': f'Good repayment history with a {repayment_rate:.0%} repayment rate across {prev_loan_count} loan(s).'
            })
        elif prev_loan_defaults >= 2:
            factors.append({
                'factor': 'Loan History',
                'impact': 'negative',
                'detail': f'Multiple defaults detected ({prev_loan_defaults} out of {prev_loan_count} loans). This is a major risk flag.'
            })
        elif prev_loan_defaults == 1:
            factors.append({
                'factor': 'Loan History',
                'impact': 'neutral',
                'detail': f'One previous default out of {prev_loan_count} loans. Repayment rate: {repayment_rate:.0%}.'
            })
        else:
            factors.append({
                'factor': 'Loan History',
                'impact': 'neutral',
                'detail': f'Moderate loan history: {repayment_rate:.0%} repayment rate across {prev_loan_count} loan(s).'
            })
        
        return factors

    def get_actionable_recommendations(self, data: dict, prob: float) -> list:
        """Generate specific advice to improve approval odds."""
        recommendations = []
        
        if prob >= 0.8:
            recommendations.append("Your profile is strong. Ensure all documents are ready for fast processing.")
            return recommendations
        
        # 1. Credit Score is the big one
        credit_score = data.get('Credit_Score', 650)
        if credit_score < 580:
            recommendations.append(
                f"🔴 Your credit score ({credit_score}) is below the Fair threshold. "
                f"Focus on paying all bills on time for 6+ months. Even a 50-point improvement "
                f"to {credit_score + 50} could significantly improve approval odds."
            )
        elif credit_score < 670:
            target = 700
            recommendations.append(
                f"🟡 Improving your credit score from {credit_score} to {target}+ "
                f"would move you into the 'Good' tier and unlock better interest rates."
            )
        
        # 2. Credit Utilization
        utilization = data.get('Credit_Utilization', 50)
        if utilization > 50:
            target_util = 30
            recommendations.append(
                f"📉 Reduce credit utilization from {utilization:.0f}% to below {target_util}%. "
                f"Pay down approximately ${int((utilization - target_util) * 100)} per $10,000 credit limit."
            )
        elif utilization > 30:
            recommendations.append(
                f"📊 Your utilization ({utilization:.0f}%) is acceptable, but pushing it below 30% "
                f"is the industry benchmark for optimal scoring."
            )
        
        # 3. Income/Loan ratio
        income = data.get('ApplicantIncome', 0) + data.get('CoapplicantIncome', 0)
        loan = data.get('LoanAmount', 1)
        if (loan * 1000) / max(income, 1) > 40:
            recommendations.append(
                f"💰 Consider reducing your loan request to below ${int(income * 40 / 1000)}K "
                f"to improve approval odds."
            )
            recommendations.append(
                "👥 Adding a co-applicant with independent income could significantly boost your capacity."
            )
        
        # 4. Open Accounts
        open_accts = data.get('Open_Accounts', 3)
        if open_accts == 0:
            recommendations.append(
                "🆕 Open a secured credit card or become an authorized user on someone's account "
                "to build credit history."
            )
        
        # 5. Past Loan History
        prev_loan_count = data.get('Prev_Loan_Count', 0)
        prev_loan_defaults = data.get('Prev_Loan_Defaults', 0)
        repayment_rate = data.get('Repayment_Rate', 0)
        
        if prev_loan_count == 0:
            recommendations.append(
                "📜 You have no prior loan history. Consider starting with a small personal loan or "
                "secured credit card to build a verifiable borrowing track record."
            )
        elif prev_loan_defaults >= 2:
            recommendations.append(
                f"🔴 You have {prev_loan_defaults} past loan default(s). Focus on clearing any outstanding "
                f"obligations and maintain consistent repayments for 12+ months before reapplying."
            )
        elif prev_loan_defaults == 1:
            recommendations.append(
                "🟡 Your single past default is concerning. Demonstrate financial discipline with "
                "on-time payments for at least 6 months to rebuild lender confidence."
            )
        elif repayment_rate < 0.80 and prev_loan_count > 0:
            recommendations.append(
                f"📊 Your repayment rate ({repayment_rate:.0%}) could be improved. Prioritize repaying "
                f"existing obligations to strengthen your loan history profile."
            )
        
        # 6. Employment
        if data.get('Self_Employed', '') == 'Yes':
            recommendations.append(
                "📋 Prepare 3 years of audited tax returns to prove income stability as a self-employed individual."
            )
            
        # 7. Property Area
        if data.get('Property_Area', '') == 'Rural':
            recommendations.append(
                "🏡 Banks sometimes have stricter LTV ratios for rural areas; consider a higher down payment."
            )

        if not recommendations:
            recommendations.append(
                "Maintain your current financial status and re-apply in 6 months for a better assessment."
            )
            
        return recommendations
    
    def get_model_summary(self) -> dict:
        """Get a summary of the trained model for display."""
        if not self.model_metrics:
            return {}
        
        best = self.model_metrics.get('best_model', 'Unknown')
        best_metrics = self.model_metrics.get('models', {}).get(best, {})
        
        return {
            'best_model': best,
            'accuracy': best_metrics.get('accuracy', 0),
            'auc_roc': best_metrics.get('auc_roc', 0),
            'f1_score': best_metrics.get('f1_score', 0),
            'precision': best_metrics.get('precision', 0),
            'recall': best_metrics.get('recall', 0),
            'all_models': self.model_metrics.get('models', {}),
            'feature_importance': self.feature_importance or {}
        }


if __name__ == "__main__":
    predictor = LoanPredictor()
    print("Predictor module ready.")
