"""
Loan Approval Prediction — Multi-Model Training Pipeline
=========================================================
Trains 5 ML models with hyperparameter tuning, cross-validation,
and comprehensive evaluation. Saves the best model and metrics.

Models:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting (XGBoost)
5. Support Vector Machine (SVM)

Author: Kinshunk Garg
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data_preprocessing import LoanDataPreprocessor

warnings.filterwarnings('ignore')


class ModelTrainer:
    """Multi-model training pipeline with evaluation."""
    
    def __init__(self, models_dir: str = None):
        self.models_dir = models_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models"
        )
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
    
    def define_models(self):
        """Define models and their hyperparameter grids."""
        models = {
            'Logistic Regression': {
                'model': LogisticRegression(max_iter=2000, random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'solver': ['lbfgs', 'liblinear']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_leaf': [1, 2]
                }
            }
        }
        
        if HAS_XGBOOST:
            models['XGBoost'] = {
                'model': XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5, 7]
                }
            }
            
        if HAS_LGBM:
            models['LightGBM'] = {
                'model': LGBMClassifier(random_state=42, verbose=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'num_leaves': [31, 50]
                }
            }
            
        if HAS_CATBOOST:
            models['CatBoost'] = {
                'model': CatBoostClassifier(random_state=42, silent=True),
                'params': {
                    'iterations': [100, 200],
                    'depth': [4, 6],
                    'learning_rate': [0.05, 0.1]
                }
            }
        
        return models
    
    def train_all(self, X_train, X_test, y_train, y_test):
        """Train all models with GridSearchCV and evaluate."""
        models = self.define_models()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
    def train_all(self, X_train, X_test, y_train, y_test):
        """Train all models with GridSearchCV and evaluate."""
        models = self.define_models()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print("=" * 70)
        print("LOAN APPROVAL PREDICTION — MODEL TRAINING PIPELINE")
        print("=" * 70)
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples:  {len(X_test)}")
        print(f"   Features:         {X_train.shape[1]}")
        print(f"   Models to train:  {len(models)}")
        print("=" * 70)
        
        best_auc = 0
        
        for name, config in models.items():
            print(f"\n{'-' * 50}")
            print(f"Training: {name}")
            print(f"{'-' * 50}")
            
            start_time = time.time()
            
            # GridSearchCV
            grid = GridSearchCV(
                config['model'],
                config['params'],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            grid.fit(X_train, y_train)
            
            train_time = time.time() - start_time
            
            # Best model from grid
            best_estimator = grid.best_estimator_
            self.models[name] = best_estimator
            
            # Predictions
            y_pred = best_estimator.predict(X_test)
            y_prob = best_estimator.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_prob)
            cm = confusion_matrix(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(best_estimator, X_train, y_train, cv=cv, scoring='roc_auc')
            
            # ROC Curve data
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            
            self.results[name] = {
                'accuracy': round(float(accuracy), 4),
                'precision': round(float(precision), 4),
                'recall': round(float(recall), 4),
                'f1_score': round(float(f1), 4),
                'auc_roc': round(float(auc), 4),
                'cv_mean': round(float(cv_scores.mean()), 4),
                'cv_std': round(float(cv_scores.std()), 4),
                'confusion_matrix': cm.tolist(),
                'best_params': grid.best_params_,
                'train_time_seconds': round(train_time, 2),
                'roc_curve': {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }
            }
            
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   AUC-ROC:   {auc:.4f}")
            print(f"   F1-Score:  {f1:.4f}")
            print(f"   Time:     {train_time:.2f}s")
            
            # Track best model
            if auc > best_auc:
                best_auc = auc
                self.best_model_name = name
                self.best_model = best_estimator
        
        # --- ENSEMBLE: VOTING CLASSIFIER ---
        print("\nCreating Ensemble (Voting Classifier)...")
        from sklearn.ensemble import VotingClassifier
        
        # Select top 3 models for voting
        sorted_models_list = sorted(self.results.items(), key=lambda x: x[1]['auc_roc'], reverse=True)
        top_model_names = [m[0] for m in sorted_models_list[:3]]
        
        estimators = [(name, self.models[name]) for name in top_model_names]
        voting_clf = VotingClassifier(estimators=estimators, voting='soft')
        voting_clf.fit(X_train, y_train)
        
        # Evaluate Voting Classifier
        y_pred_v = voting_clf.predict(X_test)
        y_prob_v = voting_clf.predict_proba(X_test)[:, 1]
        auc_v = roc_auc_score(y_test, y_prob_v)
        
        self.models['Voting Ensemble'] = voting_clf
        self.results['Voting Ensemble'] = {
            'accuracy': round(float(accuracy_score(y_test, y_pred_v)), 4),
            'precision': round(float(precision_score(y_test, y_pred_v, zero_division=0)), 4),
            'recall': round(float(recall_score(y_test, y_pred_v, zero_division=0)), 4),
            'f1_score': round(float(f1_score(y_test, y_pred_v, zero_division=0)), 4),
            'auc_roc': round(float(auc_v), 4),
            'cv_mean': 0, # Not computed for ensemble here
            'cv_std': 0,
            'confusion_matrix': confusion_matrix(y_test, y_pred_v).tolist(),
            'best_params': "Ensemble of " + ", ".join(top_model_names),
            'train_time_seconds': 0,
            'roc_curve': {'fpr': [], 'tpr': []} 
        }
        
        if auc_v > best_auc:
            self.best_model_name = 'Voting Ensemble'
            self.best_model = voting_clf
            print(f"⭐ Voting Ensemble is now the best model! (AUC: {auc_v:.4f})")
        
        print(f"\n{'=' * 70}")
        print(f"FINAL BEST MODEL: {self.best_model_name} (AUC: {max(best_auc, auc_v):.4f})")
        print(f"{'=' * 70}")
    
    def get_feature_importance(self, feature_names: list) -> dict:
        """Get feature importance from the best model."""
        importance = {}
        
        if hasattr(self.best_model, 'feature_importances_'):
            # Tree-based models
            importances = self.best_model.feature_importances_
            for name, imp in zip(feature_names, importances):
                importance[name] = round(float(imp), 6)
        elif hasattr(self.best_model, 'coef_'):
            # Linear models
            importances = np.abs(self.best_model.coef_[0])
            for name, imp in zip(feature_names, importances):
                importance[name] = round(float(imp), 6)
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return importance
    
    def save_results(self, feature_names: list):
        """Save models, metrics, and feature importance."""
        # Save best model
        model_path = os.path.join(self.models_dir, "best_model.joblib")
        joblib.dump(self.best_model, model_path)
        print(f"Best model saved: {model_path}")
        
        # Save all models
        all_models_path = os.path.join(self.models_dir, "all_models.joblib")
        joblib.dump(self.models, all_models_path)
        print(f"All models saved: {all_models_path}")
        
        # Feature importance
        importance = self.get_feature_importance(feature_names)
        importance_path = os.path.join(self.models_dir, "feature_importance.json")
        with open(importance_path, 'w') as f:
            json.dump(importance, f, indent=2)
        print(f"Feature importance saved: {importance_path}")
        
        # Model metrics
        metrics = {
            'best_model': self.best_model_name,
            'models': self.results
        }
        metrics_path = os.path.join(self.models_dir, "model_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved: {metrics_path}")
        
        return importance
    
    def generate_plots(self, X_test, y_test, feature_names: list):
        """Generate evaluation plots."""
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
        os.makedirs(assets_dir, exist_ok=True)
        
        # --- 1. Model Comparison Bar Chart ---
        fig, ax = plt.subplots(figsize=(12, 6))
        model_names = list(self.results.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        x = np.arange(len(model_names))
        width = 0.15
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']
        
        for i, metric in enumerate(metrics_names):
            values = [self.results[m][metric] for m in model_names]
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), color=colors[i])
        
        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(assets_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # --- 2. ROC Curves ---
        fig, ax = plt.subplots(figsize=(10, 8))
        colors_roc = ['#667eea', '#f093fb', '#4facfe', '#00f2fe', '#764ba2']
        
        for i, (name, result) in enumerate(self.results.items()):
            fpr = result['roc_curve']['fpr']
            tpr = result['roc_curve']['tpr']
            auc = result['auc_roc']
            ax.plot(fpr, tpr, color=colors_roc[i % len(colors_roc)],
                    linewidth=2, label=f'{name} (AUC = {auc:.4f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title('ROC Curves — All Models', fontweight='bold', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(assets_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # --- 3. Confusion Matrix (Best Model) ---
        cm = np.array(self.results[self.best_model_name]['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Rejected', 'Approved'],
                    yticklabels=['Rejected', 'Approved'])
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('Actual', fontweight='bold')
        ax.set_title(f'Confusion Matrix — {self.best_model_name}', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(assets_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # --- 4. Feature Importance ---
        importance = self.get_feature_importance(feature_names)
        if importance:
            fig, ax = plt.subplots(figsize=(10, 8))
            names = list(importance.keys())[:15]  # Top 15
            values = [importance[n] for n in names]
            
            colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
            ax.barh(range(len(names)), values[::-1], color=colors_bar)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names[::-1])
            ax.set_xlabel('Importance', fontweight='bold')
            ax.set_title(f'Feature Importance — {self.best_model_name}', fontweight='bold', fontsize=14)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(assets_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Plots saved to: {assets_dir}")


def main():
    """Run the full training pipeline."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data", "loan_data.csv")
    models_dir = os.path.join(base_dir, "models")
    
    # Check if dataset exists, if not generate it
    if not os.path.exists(data_path):
        print("📂 Dataset not found. Generating...")
        from src.generate_dataset import generate_loan_dataset
        df = generate_loan_dataset(5000)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"✅ Dataset generated: {data_path}")
    
    # Preprocess
    print("\nPreprocessing data...")
    preprocessor = LoanDataPreprocessor()
    X_train, X_test, y_train, y_test, raw_df = preprocessor.prepare_train_test(data_path)
    
    # Save preprocessor
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    preprocessor.save(preprocessor_path)
    
    # Save raw data stats for EDA
    eda_stats = preprocessor.get_eda_data(raw_df)
    eda_path = os.path.join(models_dir, "eda_stats.json")
    with open(eda_path, 'w') as f:
        json.dump(eda_stats, f, indent=2, default=str)
    print(f"EDA stats saved: {eda_path}")
    
    # Train models
    trainer = ModelTrainer(models_dir)
    trainer.train_all(X_train, X_test, y_train, y_test)
    
    # Save results
    feature_names = preprocessor.feature_columns
    importance = trainer.save_results(feature_names)
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    trainer.generate_plots(X_test, y_test, feature_names)
    
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nModel artifacts saved in: {models_dir}")
    print(f"Best Model: {trainer.best_model_name}")
    print(f"Best AUC-ROC: {trainer.results[trainer.best_model_name]['auc_roc']:.4f}")
    print(f"\nRun the dashboard: streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
