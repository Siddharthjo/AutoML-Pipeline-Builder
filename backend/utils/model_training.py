import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import AutoML libraries
try:
    import autosklearn.classification
    import autosklearn.regression
    AUTOSKLEARN_AVAILABLE = True
except ImportError:
    AUTOSKLEARN_AVAILABLE = False

try:
    from tpot import TPOTClassifier, TPOTRegressor
    TPOT_AVAILABLE = True
except ImportError:
    TPOT_AVAILABLE = False

class AutoMLTrainer:
    """Handles AutoML model training and selection"""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.model_type = None
        
    def train_model(self, X: pd.DataFrame, y: pd.Series, problem_type: str, job_id: str) -> Dict[str, Any]:
        """
        Train AutoML model
        
        Args:
            X: Features
            y: Target variable
            problem_type: 'classification' or 'regression'
            job_id: Unique identifier for this job
            
        Returns:
            Dictionary containing model and training results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y if problem_type == 'classification' else None
        )
        
        # Select and train model
        if AUTOSKLEARN_AVAILABLE:
            model = self._train_autosklearn(X_train, y_train, problem_type)
        elif TPOT_AVAILABLE:
            model = self._train_tpot(X_train, y_train, problem_type)
        else:
            model = self._train_traditional(X_train, y_train, problem_type)
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Save model
        model_path = Path("models") / f"model_{job_id}.pkl"
        joblib.dump(model, model_path)
        
        # Get model information
        model_info = self._get_model_info(model)
        
        return {
            'model': model,
            'model_path': str(model_path),
            'model_info': model_info,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'problem_type': problem_type
        }
    
    def _train_autosklearn(self, X_train: pd.DataFrame, y_train: pd.Series, problem_type: str):
        """Train using Auto-sklearn"""
        if problem_type == 'classification':
            model = autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task=300,  # 5 minutes
                per_run_time_limit=30,
                memory_limit=3072,
                seed=self.random_state
            )
        else:
            model = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=300,  # 5 minutes
                per_run_time_limit=30,
                memory_limit=3072,
                seed=self.random_state
            )
        
        self.model_type = 'autosklearn'
        return model
    
    def _train_tpot(self, X_train: pd.DataFrame, y_train: pd.Series, problem_type: str):
        """Train using TPOT"""
        if problem_type == 'classification':
            model = TPOTClassifier(
                generations=5,
                population_size=20,
                cv=5,
                random_state=self.random_state,
                verbosity=0,
                max_time_mins=5
            )
        else:
            model = TPOTRegressor(
                generations=5,
                population_size=20,
                cv=5,
                random_state=self.random_state,
                verbosity=0,
                max_time_mins=5
            )
        
        self.model_type = 'tpot'
        return model
    
    def _train_traditional(self, X_train: pd.DataFrame, y_train: pd.Series, problem_type: str):
        """Train using traditional scikit-learn models with basic hyperparameter tuning"""
        from sklearn.model_selection import GridSearchCV
        
        if problem_type == 'classification':
            # Try multiple classifiers
            models = {
                'random_forest': RandomForestClassifier(random_state=self.random_state),
                'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'svm': SVC(random_state=self.random_state, probability=True)
            }
            
            param_grids = {
                'random_forest': {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]},
                'logistic_regression': {'C': [0.1, 1, 10]},
                'svm': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
            }
        else:
            # Try multiple regressors
            models = {
                'random_forest': RandomForestRegressor(random_state=self.random_state),
                'linear_regression': LinearRegression(),
                'svr': SVR()
            }
            
            param_grids = {
                'random_forest': {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]},
                'linear_regression': {},
                'svr': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
            }
        
        best_model = None
        best_score = -np.inf if problem_type == 'regression' else 0
        
        for name, model in models.items():
            if param_grids[name]:
                grid_search = GridSearchCV(
                    model, param_grids[name], cv=3, 
                    scoring='neg_mean_squared_error' if problem_type == 'regression' else 'accuracy',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                current_model = grid_search.best_estimator_
                current_score = grid_search.best_score_
            else:
                current_model = model
                current_model.fit(X_train, y_train)
                if problem_type == 'classification':
                    current_score = accuracy_score(y_train, current_model.predict(X_train))
                else:
                    current_score = -mean_squared_error(y_train, current_model.predict(X_train))
            
            if current_score > best_score:
                best_score = current_score
                best_model = current_model
        
        self.model_type = 'traditional'
        return best_model
    
    def _get_model_info(self, model) -> Dict[str, Any]:
        """Get information about the trained model"""
        model_info = {
            'model_type': self.model_type,
            'algorithm': str(type(model).__name__)
        }
        
        # Add specific info based on model type
        if self.model_type == 'autosklearn':
            if hasattr(model, 'show_models'):
                model_info['ensemble_info'] = str(model.show_models())
        elif self.model_type == 'tpot':
            if hasattr(model, 'fitted_pipeline_'):
                model_info['pipeline'] = str(model.fitted_pipeline_)
        
        # Add feature importance if available
        if hasattr(model, 'feature_importances_'):
            model_info['has_feature_importance'] = True
        
        return model_info
    
    def get_feature_importance(self, model, feature_names: list) -> Dict[str, float]:
        """Get feature importance from the model"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return dict(zip(feature_names, importance))
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            if len(model.coef_.shape) == 1:
                importance = np.abs(model.coef_)
            else:
                importance = np.abs(model.coef_).mean(axis=0)
            return dict(zip(feature_names, importance))
        else:
            return {}
    
    def predict(self, model, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model"""
        return model.predict(X)
    
    def predict_proba(self, model, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Get prediction probabilities (for classification)"""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        return None