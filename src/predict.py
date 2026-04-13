"""
Loan Approval Prediction — Prediction Module
=============================================
Loads the saved model and preprocessor to make predictions
on new applicant data with confidence scores and explanations.

Author: Kinshunk Garg
"""

import os
import json
import numpy as np
import pandas as pd
import joblib


class LoanPredictor:
    """Production-ready prediction interface."""
    
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
        
        # Actionable recommendations (important for rejection)
        recommendations = self.get_actionable_recommendations(applicant_data, approval_prob)
        
        return {
            'approved': bool(prediction == 1),
            'approval_probability': round(approval_prob * 100, 2),
            'rejection_probability': round(rejection_prob * 100, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'contributing_factors': factors,
            'recommendations': recommendations,
            'model_used': self.model_metrics.get('best_model', 'Unknown') if self.model_metrics else 'Unknown'
        }
    
    def _get_contributing_factors(self, data: dict) -> list:
        """Determine which factors most influenced the decision."""
        factors = []
        
        # Credit History
        credit = data.get('Credit_History', 0)
        if credit == 1.0:
            factors.append({
                'factor': 'Credit History',
                'impact': 'positive',
                'detail': 'Your clean credit record is a major strength.'
            })
        else:
            factors.append({
                'factor': 'Credit History',
                'impact': 'negative',
                'detail': 'Missing or poor credit history is negatively impacting your score.'
            })
        
        # Income vs Loan
        income = data.get('ApplicantIncome', 0) + data.get('CoapplicantIncome', 0)
        loan = data.get('LoanAmount', 1) 
        ratio = (loan * 1000) / max(income, 1)
        
        if ratio < 20: # Loan is less than 20x monthly income
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
        
        # Stability
        is_grad = data.get('Education', '') == 'Graduate'
        is_employed = data.get('Self_Employed', '') == 'No'
        if is_grad and is_employed:
            factors.append({
                'factor': 'Employment Stability',
                'impact': 'positive',
                'detail': 'Being a graduate with salaried employment indicates high stability.'
            })
        
        return factors

    def get_actionable_recommendations(self, data: dict, prob: float) -> list:
        """Generate specific advice to improve approval odds."""
        recommendations = []
        
        if prob >= 0.8:
            recommendations.append("Your profile is strong. Ensure all documents are ready for fast processing.")
            return recommendations

        # 1. Credit History is the big one
        if data.get('Credit_History', 0) == 0:
            recommendations.append("Build your credit score by taking a small credit card or secured loan and paying on time.")
        
        # 2. Income/Loan ratio
        income = data.get('ApplicantIncome', 0) + data.get('CoapplicantIncome', 0)
        loan = data.get('LoanAmount', 1)
        if (loan * 1000) / max(income, 1) > 40:
            recommendations.append(f"Consider reducing your loan request to below ${int(income * 40 / 1000)} to improve approval odds.")
            recommendations.append("Adding a co-applicant with independent income could significantly boost your capacity.")
            
        # 3. Employment
        if data.get('Self_Employed', '') == 'Yes':
            recommendations.append("Prepare 3 years of audited tax returns to prove income stability as a self-employed individual.")
            
        # 4. Property Area
        if data.get('Property_Area', '') == 'Rural':
             recommendations.append("Banks sometimes have stricter LTV ratios for rural areas; consider a higher down payment.")

        if not recommendations:
            recommendations.append("Maintain your current financial status and re-apply in 6 months for a better assessment.")
            
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
    # Test would go here, removed emoji prints
    print("Predictor module ready.")
