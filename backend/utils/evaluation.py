import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import base64
from io import BytesIO

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score

class ModelEvaluator:
    """Handles model evaluation and visualization"""
    
    def __init__(self):
        self.metrics = {}
        self.visualizations = {}
        
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      problem_type: str, job_id: str) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            problem_type: 'classification' or 'regression'
            job_id: Unique identifier for this job
            
        Returns:
            Dictionary containing metrics and visualizations
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        if problem_type == 'classification' and hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        if problem_type == 'classification':
            metrics = self._calculate_classification_metrics(y_test, y_pred, y_pred_proba)
        else:
            metrics = self._calculate_regression_metrics(y_test, y_pred)
        
        # Generate visualizations
        visualizations = self._generate_visualizations(
            y_test, y_pred, y_pred_proba, problem_type, job_id
        )
        
        # Feature importance
        feature_importance = self._get_feature_importance(model, X_test.columns.tolist())
        
        return {
            'metrics': metrics,
            'visualizations': visualizations,
            'feature_importance': feature_importance,
            'predictions': {
                'y_test': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist() if y_pred_proba is not None else None
            }
        }
    
    def _calculate_classification_metrics(self, y_test: pd.Series, y_pred: np.ndarray, 
                                        y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate classification metrics"""
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        }
        
        # Add AUC for binary classification
        if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            metrics['auc'] = float(auc(fpr, tpr))
        
        return metrics
    
    def _calculate_regression_metrics(self, y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        metrics = {
            'mse': float(mean_squared_error(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred))
        }
        
        # Add MAPE if no zero values in y_test
        if not np.any(y_test == 0):
            metrics['mape'] = float(mean_absolute_percentage_error(y_test, y_pred))
        
        return metrics
    
    def _generate_visualizations(self, y_test: pd.Series, y_pred: np.ndarray, 
                               y_pred_proba: Optional[np.ndarray], problem_type: str, 
                               job_id: str) -> Dict[str, str]:
        """Generate visualizations and return base64 encoded images"""
        visualizations = {}
        
        if problem_type == 'classification':
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            visualizations['confusion_matrix'] = self._plot_confusion_matrix(cm, job_id)
            
            # ROC Curve for binary classification
            if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                visualizations['roc_curve'] = self._plot_roc_curve(y_test, y_pred_proba[:, 1], job_id)
            
            # Prediction Distribution
            visualizations['prediction_distribution'] = self._plot_prediction_distribution_classification(
                y_test, y_pred, job_id
            )
        
        else:  # regression
            # Actual vs Predicted
            visualizations['actual_vs_predicted'] = self._plot_actual_vs_predicted(y_test, y_pred, job_id)
            
            # Residuals
            visualizations['residuals'] = self._plot_residuals(y_test, y_pred, job_id)
            
            # Prediction Distribution
            visualizations['prediction_distribution'] = self._plot_prediction_distribution_regression(
                y_test, y_pred, job_id
            )
        
        return visualizations
    
    def _plot_confusion_matrix(self, cm: np.ndarray, job_id: str) -> str:
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Save plot
        img_path = Path("reports") / f"confusion_matrix_{job_id}.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(img_path)
    
    def _plot_roc_curve(self, y_test: pd.Series, y_pred_proba: np.ndarray, job_id: str) -> str:
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        # Save plot
        img_path = Path("reports") / f"roc_curve_{job_id}.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(img_path)
    
    def _plot_actual_vs_predicted(self, y_test: pd.Series, y_pred: np.ndarray, job_id: str) -> str:
        """Plot actual vs predicted for regression"""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.legend()
        
        # Save plot
        img_path = Path("reports") / f"actual_vs_predicted_{job_id}.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(img_path)
    
    def _plot_residuals(self, y_test: pd.Series, y_pred: np.ndarray, job_id: str) -> str:
        """Plot residuals for regression"""
        residuals = y_test - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted')
        
        # Residuals histogram
        ax2.hist(residuals, bins=30, alpha=0.7)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residuals Distribution')
        
        plt.tight_layout()
        
        # Save plot
        img_path = Path("reports") / f"residuals_{job_id}.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(img_path)
    
    def _plot_prediction_distribution_classification(self, y_test: pd.Series, y_pred: np.ndarray, job_id: str) -> str:
        """Plot prediction distribution for classification"""
        plt.figure(figsize=(10, 6))
        
        # Create comparison plot
        labels = np.unique(np.concatenate([y_test, y_pred]))
        x = np.arange(len(labels))
        width = 0.35
        
        actual_counts = [np.sum(y_test == label) for label in labels]
        pred_counts = [np.sum(y_pred == label) for label in labels]
        
        plt.bar(x - width/2, actual_counts, width, label='Actual', alpha=0.7)
        plt.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7)
        
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.title('Actual vs Predicted Distribution')
        plt.xticks(x, labels)
        plt.legend()
        
        # Save plot
        img_path = Path("reports") / f"prediction_distribution_{job_id}.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(img_path)
    
    def _plot_prediction_distribution_regression(self, y_test: pd.Series, y_pred: np.ndarray, job_id: str) -> str:
        """Plot prediction distribution for regression"""
        plt.figure(figsize=(10, 6))
        
        # Create comparison histogram
        plt.hist(y_test, bins=30, alpha=0.7, label='Actual', density=True)
        plt.hist(y_pred, bins=30, alpha=0.7, label='Predicted', density=True)
        
        plt.xlabel('Values')
        plt.ylabel('Density')
        plt.title('Actual vs Predicted Distribution')
        plt.legend()
        
        # Save plot
        img_path = Path("reports") / f"prediction_distribution_{job_id}.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(img_path)
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from model"""
        importance_dict = {}
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            importance_dict = dict(zip(feature_names, importance.tolist()))
        elif hasattr(model, 'coef_'):
            # For linear models
            if len(model.coef_.shape) == 1:
                importance = np.abs(model.coef_)
            else:
                importance = np.abs(model.coef_).mean(axis=0)
            importance_dict = dict(zip(feature_names, importance.tolist()))
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def generate_performance_summary(self, metrics: Dict[str, float], problem_type: str) -> str:
        """Generate a text summary of model performance"""
        if problem_type == 'classification':
            summary = f"""
            Model Performance Summary (Classification):
            ==========================================
            Accuracy: {metrics.get('accuracy', 0):.4f}
            Precision: {metrics.get('precision', 0):.4f}
            Recall: {metrics.get('recall', 0):.4f}
            F1-Score: {metrics.get('f1_score', 0):.4f}
            """
            if 'auc' in metrics:
                summary += f"AUC: {metrics['auc']:.4f}\n"
        else:
            summary = f"""
            Model Performance Summary (Regression):
            ======================================
            R² Score: {metrics.get('r2', 0):.4f}
            RMSE: {metrics.get('rmse', 0):.4f}
            MAE: {metrics.get('mae', 0):.4f}
            """
            if 'mape' in metrics:
                summary += f"MAPE: {metrics['mape']:.4f}%\n"
        
        return summary