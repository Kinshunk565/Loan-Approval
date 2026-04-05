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
        Make a prediction for a single applicant.
        
        Args:
            applicant_data: Dictionary with keys:
                - Gender, Married, Dependents, Education, Self_Employed
                - ApplicantIncome, CoapplicantIncome, LoanAmount
                - Loan_Amount_Term, Credit_History, Property_Area
        
        Returns:
            Dictionary with prediction, probability, risk assessment, and factors
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
        
        return {
            'approved': bool(prediction == 1),
            'approval_probability': round(approval_prob * 100, 2),
            'rejection_probability': round(rejection_prob * 100, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'contributing_factors': factors,
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
                'detail': 'Good credit history significantly increases approval chances'
            })
        else:
            factors.append({
                'factor': 'Credit History',
                'impact': 'negative',
                'detail': 'No/bad credit history is the #1 reason for rejection'
            })
        
        # Income vs Loan
        income = data.get('ApplicantIncome', 0) + data.get('CoapplicantIncome', 0)
        loan = data.get('LoanAmount', 1) * 1000
        ratio = income / max(loan, 1)
        
        if ratio > 0.5:
            factors.append({
                'factor': 'Income-to-Loan Ratio',
                'impact': 'positive',
                'detail': f'Strong ratio ({ratio:.2f}x) — income well covers the loan'
            })
        elif ratio > 0.2:
            factors.append({
                'factor': 'Income-to-Loan Ratio',
                'impact': 'neutral',
                'detail': f'Moderate ratio ({ratio:.2f}x) — borderline coverage'
            })
        else:
            factors.append({
                'factor': 'Income-to-Loan Ratio',
                'impact': 'negative',
                'detail': f'Low ratio ({ratio:.2f}x) — income may not cover the loan'
            })
        
        # Education
        if data.get('Education', '') == 'Graduate':
            factors.append({
                'factor': 'Education',
                'impact': 'positive',
                'detail': 'Graduate education indicates stable earning potential'
            })
        else:
            factors.append({
                'factor': 'Education',
                'impact': 'neutral',
                'detail': 'Non-graduate status has minor impact'
            })
        
        # Property Area
        area = data.get('Property_Area', '')
        if area == 'Semiurban':
            factors.append({
                'factor': 'Property Area',
                'impact': 'positive',
                'detail': 'Semiurban properties show higher approval rates'
            })
        elif area == 'Urban':
            factors.append({
                'factor': 'Property Area',
                'impact': 'neutral',
                'detail': 'Urban area — average approval rate'
            })
        else:
            factors.append({
                'factor': 'Property Area',
                'impact': 'neutral',
                'detail': 'Rural area — slightly lower approval probability'
            })
        
        return factors
    
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
    
    # Test prediction
    test_applicant = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '1',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 6000,
        'CoapplicantIncome': 2000,
        'LoanAmount': 150,
        'Loan_Amount_Term': 360.0,
        'Credit_History': 1.0,
        'Property_Area': 'Semiurban'
    }
    
    result = predictor.predict(test_applicant)
    print("\n🔮 PREDICTION RESULT:")
    print(f"   Status: {'✅ APPROVED' if result['approved'] else '❌ REJECTED'}")
    print(f"   Confidence: {result['approval_probability']}%")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Model Used: {result['model_used']}")
    print(f"\n   Contributing Factors:")
    for f in result['contributing_factors']:
        icon = "🟢" if f['impact'] == 'positive' else ("🔴" if f['impact'] == 'negative' else "🟡")
        print(f"     {icon} {f['factor']}: {f['detail']}")
