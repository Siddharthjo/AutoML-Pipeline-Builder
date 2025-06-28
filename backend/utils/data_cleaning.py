import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """Handles data cleaning, preprocessing, and problem type detection"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.imputers = {}
        self.preprocessor = None
        
    def clean_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main method to clean and preprocess the data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing cleaned data and metadata
        """
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Basic info
        original_shape = df_clean.shape
        
        # Remove completely empty rows/columns
        df_clean = df_clean.dropna(how='all')
        df_clean = df_clean.dropna(axis=1, how='all')
        
        # Detect target column (last column by default, or most likely target)
        target_column = self._detect_target_column(df_clean)
        
        # Separate features and target
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]
        
        # Handle missing values
        X_clean = self._handle_missing_values(X)
        y_clean = self._handle_target_missing_values(y)
        
        # Remove samples where target is missing
        mask = ~y_clean.isna()
        X_clean = X_clean[mask]
        y_clean = y_clean[mask]
        
        # Reset index
        X_clean = X_clean.reset_index(drop=True)
        y_clean = y_clean.reset_index(drop=True)
        
        # Encode categorical variables
        X_processed = self._encode_features(X_clean)
        y_processed = self._encode_target(y_clean)
        
        # Scale numerical features
        X_scaled = self._scale_features(X_processed)
        
        return {
            'X': X_scaled,
            'y': y_processed,
            'target_column': target_column,
            'original_shape': original_shape,
            'cleaned_shape': (len(X_scaled), len(X_scaled.columns)),
            'feature_names': X_scaled.columns.tolist(),
            'preprocessing_info': {
                'missing_values_handled': True,
                'categorical_encoded': True,
                'numerical_scaled': True
            }
        }
    
    def _detect_target_column(self, df: pd.DataFrame) -> str:
        """
        Detect the most likely target column
        
        Args:
            df: Input DataFrame
            
        Returns:
            Name of the target column
        """
        # Common target column names
        target_keywords = ['target', 'label', 'class', 'y', 'output', 'prediction', 'result']
        
        # Check for common target names
        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                return col
        
        # If no obvious target, use the last column
        return df.columns[-1]
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        X_clean = X.copy()
        
        for col in X_clean.columns:
            if X_clean[col].dtype in ['object', 'category']:
                # For categorical columns, fill with mode or 'unknown'
                mode_value = X_clean[col].mode()
                fill_value = mode_value[0] if len(mode_value) > 0 else 'unknown'
                X_clean[col] = X_clean[col].fillna(fill_value)
            else:
                # For numerical columns, fill with median
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        
        return X_clean
    
    def _handle_target_missing_values(self, y: pd.Series) -> pd.Series:
        """Handle missing values in target variable"""
        # Don't fill target missing values, they will be dropped
        return y
    
    def _encode_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        X_encoded = X.copy()
        
        for col in X_encoded.columns:
            if X_encoded[col].dtype in ['object', 'category']:
                # Use label encoding for high cardinality, one-hot for low cardinality
                n_unique = X_encoded[col].nunique()
                
                if n_unique <= 10:  # One-hot encoding for low cardinality
                    dummies = pd.get_dummies(X_encoded[col], prefix=col)
                    X_encoded = pd.concat([X_encoded.drop(columns=[col]), dummies], axis=1)
                else:  # Label encoding for high cardinality
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                    self.label_encoders[col] = le
        
        return X_encoded
    
    def _encode_target(self, y: pd.Series) -> pd.Series:
        """Encode target variable if categorical"""
        if y.dtype in ['object', 'category']:
            le = LabelEncoder()
            y_encoded = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
            self.label_encoders['target'] = le
            return y_encoded
        return y
    
    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        X_scaled = X.copy()
        
        # Identify numerical columns
        numerical_cols = X_scaled.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            scaler = StandardScaler()
            X_scaled[numerical_cols] = scaler.fit_transform(X_scaled[numerical_cols])
            self.scaler = scaler
        
        return X_scaled
    
    def detect_problem_type(self, y: pd.Series) -> str:
        """
        Detect if the problem is classification or regression
        
        Args:
            y: Target variable
            
        Returns:
            'classification' or 'regression'
        """
        # If target is clearly categorical
        if y.dtype in ['object', 'category']:
            return 'classification'
        
        # If target is numerical
        if y.dtype in ['int64', 'float64']:
            # Check if it's discrete with few unique values (likely classification)
            n_unique = y.nunique()
            n_samples = len(y)
            
            # If less than 10 unique values and less than 5% of total samples
            if n_unique <= 10 and n_unique / n_samples < 0.05:
                return 'classification'
            else:
                return 'regression'
        
        # Default to classification
        return 'classification'
    
    def get_feature_types(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """Get categorization of feature types"""
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return {
            'numerical': numerical_features,
            'categorical': categorical_features
        }
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a data quality report"""
        report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numerical_stats': df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {},
            'categorical_stats': {}
        }
        
        # Categorical statistics
        for col in df.select_dtypes(include=['object', 'category']).columns:
            report['categorical_stats'][col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'frequency': df[col].value_counts().head().to_dict()
            }
        
        return report