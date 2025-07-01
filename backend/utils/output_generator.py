import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import zipfile
import io
import base64

class OutputGenerator:
    """Generates comprehensive outputs for AutoML pipeline runs"""
    
    def __init__(self, job_id: str, output_dir: Path = Path("outputs")):
        self.job_id = job_id
        self.output_dir = output_dir / job_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_all_outputs(self, 
                           model,
                           X_train: pd.DataFrame,
                           X_test: pd.DataFrame, 
                           y_train: pd.Series,
                           y_test: pd.Series,
                           y_pred: np.ndarray,
                           y_pred_proba: Optional[np.ndarray],
                           cleaned_data: Dict[str, Any],
                           metrics: Dict[str, float],
                           problem_type: str,
                           preprocessing_steps: List[str]) -> Dict[str, str]:
        """Generate all pipeline outputs"""
        
        outputs = {}
        
        # 1. Save trained model
        outputs['model'] = self._save_model(model)
        
        # 2. Generate cleaned dataset
        outputs['cleaned_data'] = self._save_cleaned_dataset(cleaned_data)
        
        # 3. Generate metrics summary
        outputs['metrics'] = self._save_metrics_summary(metrics, problem_type)
        
        # 4. Generate pipeline configuration
        outputs['pipeline_config'] = self._save_pipeline_config(
            model, preprocessing_steps, problem_type, metrics
        )
        
        # 5. Generate visualizations
        viz_outputs = self._generate_visualizations(
            y_test, y_pred, y_pred_proba, problem_type, 
            X_test, model, cleaned_data
        )
        outputs.update(viz_outputs)
        
        # 6. Generate comprehensive HTML report
        outputs['report'] = self._generate_html_report(
            cleaned_data, metrics, problem_type, preprocessing_steps, 
            model, viz_outputs
        )
        
        # 7. Create ZIP archive
        outputs['zip_archive'] = self._create_zip_archive(outputs)
        
        return outputs
    
    def _save_model(self, model) -> str:
        """Save the trained model as .pkl file"""
        model_path = self.output_dir / f"model_{self.job_id}.pkl"
        joblib.dump(model, model_path)
        return str(model_path)
    
    def _save_cleaned_dataset(self, cleaned_data: Dict[str, Any]) -> str:
        """Save the preprocessed dataset"""
        # Combine features and target
        X = cleaned_data['X']
        y = cleaned_data['y']
        
        # Create final dataset
        final_dataset = X.copy()
        final_dataset[cleaned_data['target_column']] = y
        
        # Save as CSV
        csv_path = self.output_dir / f"cleaned_data_{self.job_id}.csv"
        final_dataset.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    def _save_metrics_summary(self, metrics: Dict[str, float], problem_type: str) -> str:
        """Save metrics as JSON file"""
        metrics_summary = {
            "job_id": self.job_id,
            "timestamp": datetime.now().isoformat(),
            "problem_type": problem_type,
            "metrics": metrics,
            "performance_summary": self._get_performance_summary(metrics, problem_type)
        }
        
        json_path = self.output_dir / f"metrics_{self.job_id}.json"
        with open(json_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        return str(json_path)
    
    def _save_pipeline_config(self, model, preprocessing_steps: List[str], 
                            problem_type: str, metrics: Dict[str, float]) -> str:
        """Save pipeline configuration"""
        config = {
            "job_id": self.job_id,
            "timestamp": datetime.now().isoformat(),
            "problem_type": problem_type,
            "model_type": str(type(model).__name__),
            "model_parameters": self._get_model_parameters(model),
            "preprocessing_steps": preprocessing_steps,
            "performance_metrics": metrics,
            "model_info": {
                "algorithm": str(type(model).__name__),
                "has_feature_importance": hasattr(model, 'feature_importances_'),
                "supports_probability": hasattr(model, 'predict_proba'),
                "training_completed": True
            }
        }
        
        config_path = self.output_dir / f"pipeline_config_{self.job_id}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return str(config_path)
    
    def _generate_visualizations(self, y_test: pd.Series, y_pred: np.ndarray,
                               y_pred_proba: Optional[np.ndarray], problem_type: str,
                               X_test: pd.DataFrame, model, cleaned_data: Dict) -> Dict[str, str]:
        """Generate all visualization files"""
        viz_outputs = {}
        
        # Set style for better plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        if problem_type == 'classification':
            # Confusion Matrix
            viz_outputs['confusion_matrix'] = self._plot_confusion_matrix(y_test, y_pred)
            
            # ROC Curve (for binary classification)
            if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                viz_outputs['roc_curve'] = self._plot_roc_curve(y_test, y_pred_proba[:, 1])
            
            # Class Distribution
            viz_outputs['class_distribution'] = self._plot_class_distribution(y_test, y_pred)
            
        else:  # regression
            # Actual vs Predicted
            viz_outputs['actual_vs_predicted'] = self._plot_actual_vs_predicted(y_test, y_pred)
            
            # Residuals Analysis
            viz_outputs['residuals'] = self._plot_residuals(y_test, y_pred)
            
            # Prediction Distribution
            viz_outputs['prediction_distribution'] = self._plot_prediction_distribution_regression(y_test, y_pred)
        
        # Feature Importance (if available)
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            viz_outputs['feature_importance'] = self._plot_feature_importance(model, X_test.columns.tolist())
        
        # Correlation Heatmap
        viz_outputs['correlation_heatmap'] = self._plot_correlation_heatmap(cleaned_data['X'])
        
        return viz_outputs
    
    def _plot_confusion_matrix(self, y_test: pd.Series, y_pred: np.ndarray) -> str:
        """Generate confusion matrix plot"""
        from sklearn.metrics import confusion_matrix
        
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y_test), 
                   yticklabels=np.unique(y_test))
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        
        path = self.output_dir / f"confusion_matrix_{self.job_id}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def _plot_roc_curve(self, y_test: pd.Series, y_pred_proba: np.ndarray) -> str:
        """Generate ROC curve plot"""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = self.output_dir / f"roc_curve_{self.job_id}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def _plot_class_distribution(self, y_test: pd.Series, y_pred: np.ndarray) -> str:
        """Generate class distribution comparison"""
        plt.figure(figsize=(12, 6))
        
        labels = np.unique(np.concatenate([y_test, y_pred]))
        x = np.arange(len(labels))
        width = 0.35
        
        actual_counts = [np.sum(y_test == label) for label in labels]
        pred_counts = [np.sum(y_pred == label) for label in labels]
        
        plt.bar(x - width/2, actual_counts, width, label='Actual', alpha=0.8, color='skyblue')
        plt.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8, color='lightcoral')
        
        plt.xlabel('Classes', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Class Distribution: Actual vs Predicted', fontsize=16, fontweight='bold')
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = self.output_dir / f"class_distribution_{self.job_id}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def _plot_actual_vs_predicted(self, y_test: pd.Series, y_pred: np.ndarray) -> str:
        """Generate actual vs predicted scatter plot"""
        plt.figure(figsize=(10, 8))
        
        plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue', s=50)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, 
                label='Perfect Prediction')
        
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title('Actual vs Predicted Values', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = self.output_dir / f"actual_vs_predicted_{self.job_id}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def _plot_residuals(self, y_test: pd.Series, y_pred: np.ndarray) -> str:
        """Generate residuals analysis plot"""
        residuals = y_test - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6, color='steelblue')
        ax1.axhline(y=0, color='red', linestyle='--', lw=2)
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.grid(True, alpha=0.3)
        
        # Residuals histogram
        ax2.hist(residuals, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residuals Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Residuals Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        path = self.output_dir / f"residuals_{self.job_id}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def _plot_prediction_distribution_regression(self, y_test: pd.Series, y_pred: np.ndarray) -> str:
        """Generate prediction distribution for regression"""
        plt.figure(figsize=(12, 6))
        
        plt.hist(y_test, bins=30, alpha=0.7, label='Actual', density=True, 
                color='skyblue', edgecolor='black')
        plt.hist(y_pred, bins=30, alpha=0.7, label='Predicted', density=True, 
                color='lightcoral', edgecolor='black')
        
        plt.xlabel('Values', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution: Actual vs Predicted Values', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = self.output_dir / f"prediction_distribution_{self.job_id}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def _plot_feature_importance(self, model, feature_names: List[str]) -> str:
        """Generate feature importance plot"""
        importance_dict = {}
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            importance_dict = dict(zip(feature_names, importance))
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) == 1:
                importance = np.abs(model.coef_)
            else:
                importance = np.abs(model.coef_).mean(axis=0)
            importance_dict = dict(zip(feature_names, importance))
        
        if not importance_dict:
            return ""
        
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 15 features for readability
        top_features = sorted_features[:15]
        features, importances = zip(*top_features)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), importances, color='steelblue', alpha=0.8)
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance', fontsize=12)
        plt.title('Feature Importance', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + max(importances) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        path = self.output_dir / f"feature_importance_{self.job_id}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def _plot_correlation_heatmap(self, X: pd.DataFrame) -> str:
        """Generate correlation heatmap"""
        # Only use numerical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            return ""
        
        corr_matrix = X[numerical_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        path = self.output_dir / f"correlation_heatmap_{self.job_id}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def _generate_html_report(self, cleaned_data: Dict, metrics: Dict, 
                            problem_type: str, preprocessing_steps: List[str],
                            model, viz_outputs: Dict) -> str:
        """Generate comprehensive HTML report"""
        from jinja2 import Template
        
        template_str = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AutoML Analysis Report - {{ job_id }}</title>
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; padding: 20px; 
                    background-color: #f8f9fa;
                    line-height: 1.6;
                }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    padding: 30px; 
                    border-radius: 12px; 
                    margin-bottom: 30px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }
                .section { 
                    background: white; 
                    padding: 25px; 
                    margin-bottom: 25px; 
                    border-radius: 12px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .metric-grid { 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                    gap: 20px; 
                    margin: 20px 0; 
                }
                .metric { 
                    background: #f8f9fa; 
                    padding: 20px; 
                    border-radius: 8px; 
                    text-align: center;
                    border-left: 4px solid #667eea;
                }
                .metric-value { 
                    font-size: 2em; 
                    font-weight: bold; 
                    color: #333; 
                    margin-bottom: 5px;
                }
                .metric-label { 
                    color: #666; 
                    font-size: 0.9em; 
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }
                table { 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 20px 0;
                }
                th, td { 
                    border: 1px solid #ddd; 
                    padding: 12px; 
                    text-align: left; 
                }
                th { 
                    background-color: #667eea; 
                    color: white;
                    font-weight: 600;
                }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .badge { 
                    display: inline-block; 
                    padding: 4px 12px; 
                    border-radius: 20px; 
                    font-size: 0.8em; 
                    font-weight: bold;
                }
                .badge-classification { background: #d4edda; color: #155724; }
                .badge-regression { background: #cce5ff; color: #004085; }
                .viz-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }
                .viz-item {
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                }
                .footer {
                    text-align: center;
                    color: #666;
                    margin-top: 40px;
                    padding: 20px;
                    border-top: 1px solid #ddd;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 AutoML Analysis Report</h1>
                    <p><strong>Job ID:</strong> {{ job_id }}</p>
                    <p><strong>Generated:</strong> {{ timestamp }}</p>
                    <span class="badge badge-{{ problem_type }}">{{ problem_type.title() }}</span>
                </div>
                
                <div class="section">
                    <h2>📊 Dataset Overview</h2>
                    <div class="metric-grid">
                        <div class="metric">
                            <div class="metric-value">{{ shape[0] }}</div>
                            <div class="metric-label">Rows</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{{ shape[1] }}</div>
                            <div class="metric-label">Features</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{{ target_column }}</div>
                            <div class="metric-label">Target</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎯 Model Performance</h2>
                    <div class="metric-grid">
                        {% for metric, value in metrics.items() %}
                        <div class="metric">
                            <div class="metric-value">
                                {% if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'r2'] %}
                                    {{ "%.1f%%" | format(value * 100) }}
                                {% else %}
                                    {{ "%.4f" | format(value) }}
                                {% endif %}
                            </div>
                            <div class="metric-label">{{ metric.replace('_', ' ').title() }}</div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔧 Pipeline Configuration</h2>
                    <p><strong>Model:</strong> {{ model_name }}</p>
                    <p><strong>Problem Type:</strong> {{ problem_type.title() }}</p>
                    <h4>Preprocessing Steps:</h4>
                    <ul>
                        {% for step in preprocessing_steps %}
                        <li>{{ step }}</li>
                        {% endfor %}
                    </ul>
                </div>
                
                {% if viz_outputs %}
                <div class="section">
                    <h2>📈 Visualizations</h2>
                    <div class="viz-grid">
                        {% for viz_name, viz_path in viz_outputs.items() %}
                        {% if viz_path %}
                        <div class="viz-item">
                            <h4>{{ viz_name.replace('_', ' ').title() }}</h4>
                            <p>Available in downloaded files</p>
                        </div>
                        {% endif %}
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
                
                <div class="section">
                    <h2>📋 Summary</h2>
                    <p>This AutoML pipeline successfully processed your dataset and trained a {{ problem_type }} model.</p>
                    {% if problem_type == 'classification' %}
                    <p><strong>Classification Results:</strong> The model achieved {{ "%.1f%%" | format(metrics.get('accuracy', 0) * 100) }} accuracy on the test set.</p>
                    {% else %}
                    <p><strong>Regression Results:</strong> The model explains {{ "%.1f%%" | format(metrics.get('r2', 0) * 100) }} of the variance in the target variable.</p>
                    {% endif %}
                    <p>All preprocessing steps were applied automatically, including missing value imputation, feature encoding, and scaling.</p>
                </div>
                
                <div class="footer">
                    <p>Generated by AutoML Pipeline Builder</p>
                    <p>For questions or support, please refer to the documentation.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        template = Template(template_str)
        html_content = template.render(
            job_id=self.job_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            problem_type=problem_type,
            shape=cleaned_data['cleaned_shape'],
            target_column=cleaned_data['target_column'],
            metrics=metrics,
            model_name=str(type(model).__name__),
            preprocessing_steps=preprocessing_steps,
            viz_outputs=viz_outputs
        )
        
        report_path = self.output_dir / f"report_{self.job_id}.html"
        with open(report_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def _create_zip_archive(self, outputs: Dict[str, str]) -> str:
        """Create ZIP archive with all outputs"""
        zip_path = self.output_dir / f"automl_outputs_{self.job_id}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for output_type, file_path in outputs.items():
                if output_type != 'zip_archive' and file_path and Path(file_path).exists():
                    # Add file to zip with a clean name
                    arcname = Path(file_path).name
                    zipf.write(file_path, arcname)
        
        return str(zip_path)
    
    def _get_performance_summary(self, metrics: Dict[str, float], problem_type: str) -> str:
        """Generate performance summary text"""
        if problem_type == 'classification':
            accuracy = metrics.get('accuracy', 0)
            if accuracy > 0.9:
                return "Excellent performance"
            elif accuracy > 0.8:
                return "Good performance"
            elif accuracy > 0.7:
                return "Fair performance"
            else:
                return "Needs improvement"
        else:
            r2 = metrics.get('r2', 0)
            if r2 > 0.9:
                return "Excellent fit"
            elif r2 > 0.7:
                return "Good fit"
            elif r2 > 0.5:
                return "Moderate fit"
            else:
                return "Poor fit"
    
    def _get_model_parameters(self, model) -> Dict[str, Any]:
        """Extract model parameters"""
        try:
            if hasattr(model, 'get_params'):
                return model.get_params()
            else:
                return {"model_type": str(type(model).__name__)}
        except:
            return {"model_type": str(type(model).__name__)}