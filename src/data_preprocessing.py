"""
Loan Approval Prediction — Data Preprocessing Pipeline
=======================================================
Handles all data cleaning, feature engineering, encoding,
and scaling. Returns train/test splits ready for model training.

Author: Kinshunk Garg
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class LoanDataPreprocessor:
    """End-to-end preprocessing pipeline for loan data."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.categorical_columns = [
            'Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'Property_Area'
        ]
        self.numerical_columns = [
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History'
        ]
        self.is_fitted = False
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load the CSV dataset."""
        df = pd.read_csv(filepath)
        return df
    
    def get_eda_data(self, df: pd.DataFrame) -> dict:
        """Return EDA statistics for the dashboard."""
        stats = {
            'shape': df.shape,
            'approval_rate': float((df['Loan_Status'] == 'Y').mean()),
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'numerical_stats': {},
            'categorical_stats': {}
        }
        
        for col in self.numerical_columns:
            if col in df.columns:
                stats['numerical_stats'][col] = {
                    'mean': float(df[col].mean()) if df[col].notna().any() else 0,
                    'median': float(df[col].median()) if df[col].notna().any() else 0,
                    'std': float(df[col].std()) if df[col].notna().any() else 0,
                    'min': float(df[col].min()) if df[col].notna().any() else 0,
                    'max': float(df[col].max()) if df[col].notna().any() else 0,
                }
        
        for col in self.categorical_columns:
            if col in df.columns:
                stats['categorical_stats'][col] = df[col].value_counts().to_dict()
        
        return stats
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using appropriate strategies."""
        df = df.copy()
        
        # Categorical: fill with mode
        for col in self.categorical_columns:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Numerical: fill with median (robust to outliers)
        for col in self.numerical_columns:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Catch-all: fill any remaining NaN with 0
        df = df.fillna(0)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones."""
        df = df.copy()
        
        # Total Income
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        
        # Log transformations (reduce skewness)
        df['LogApplicantIncome'] = np.log1p(df['ApplicantIncome'])
        df['LogTotalIncome'] = np.log1p(df['TotalIncome'])
        df['LogLoanAmount'] = np.log1p(df['LoanAmount'])
        
        # EMI (Equated Monthly Installment)
        df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
        
        # Balance Income (income left after EMI)
        df['BalanceIncome'] = df['TotalIncome'] - (df['EMI'] * 1000)
        
        # Income to Loan Ratio
        df['IncomeToLoanRatio'] = df['TotalIncome'] / (df['LoanAmount'] + 1)
        
        # Loan Amount to Income Percentage
        df['LoanIncomePercent'] = (df['LoanAmount'] * 1000) / (df['TotalIncome'] + 1) * 100
        
        return df
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Label-encode categorical features."""
        df = df.copy()
        
        for col in self.categorical_columns:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    known_classes = set(le.classes_)
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in known_classes else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])
        
        return df
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Standardize numerical features."""
        X = X.copy()
        
        if fit:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return X_scaled
    
    def preprocess(self, df: pd.DataFrame, fit: bool = True, target_col: str = 'Loan_Status'):
        """
        Full preprocessing pipeline.
        
        Args:
            df: Raw dataframe
            fit: If True, fit encoders/scaler. If False, use previously fitted.
            target_col: Name of target column
            
        Returns:
            X: Processed features
            y: Target variable (if target_col exists)
        """
        # Drop ID column
        if 'Loan_ID' in df.columns:
            df = df.drop('Loan_ID', axis=1)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Separate target if exists
        y = None
        if target_col in df.columns:
            y = (df[target_col] == 'Y').astype(int)
            df = df.drop(target_col, axis=1)
        
        # Feature engineering
        df = self.engineer_features(df)
        
        # Encode categorical features
        df = self.encode_features(df, fit=fit)
        
        # Define feature columns
        if fit:
            self.feature_columns = df.columns.tolist()
        else:
            # Ensure same columns
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_columns]
        
        # Final NaN safety net (engineered features may create NaN)
        df = df.fillna(0)
        
        # Replace any infinity values
        df = df.replace([np.inf, -np.inf], 0)
        
        # Scale features
        X = self.scale_features(df, fit=fit)
        
        if fit:
            self.is_fitted = True
        
        return X, y
    
    def prepare_train_test(self, filepath: str, test_size: float = 0.2, random_state: int = 42):
        """
        Load data and return train/test splits.
        
        Returns:
            X_train, X_test, y_train, y_test, raw_df
        """
        df = self.load_data(filepath)
        raw_df = df.copy()
        
        X, y = self.preprocess(df, fit=True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, raw_df
    
    def preprocess_single(self, input_dict: dict) -> pd.DataFrame:
        """
        Preprocess a single applicant's data for prediction.
        
        Args:
            input_dict: Dictionary with applicant features
            
        Returns:
            Processed feature DataFrame ready for prediction
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted. Run preprocess() with fit=True first.")
        
        df = pd.DataFrame([input_dict])
        X, _ = self.preprocess(df, fit=False, target_col='Loan_Status')
        return X
    
    def save(self, filepath: str):
        """Save the fitted preprocessor."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"✅ Preprocessor saved to: {filepath}")
    
    def load(self, filepath: str):
        """Load a fitted preprocessor."""
        data = joblib.load(filepath)
        self.label_encoders = data['label_encoders']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.categorical_columns = data['categorical_columns']
        self.numerical_columns = data['numerical_columns']
        self.is_fitted = data['is_fitted']
        print(f"✅ Preprocessor loaded from: {filepath}")


if __name__ == "__main__":
    # Quick test
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "loan_data.csv")
    preprocessor = LoanDataPreprocessor()
    X_train, X_test, y_train, y_test, raw_df = preprocessor.prepare_train_test(data_path)
    
    print(f"✅ Preprocessing complete!")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape:  {X_test.shape}")
    print(f"   Features: {preprocessor.feature_columns}")
    print(f"   Approval rate: {y_train.mean():.1%}")
