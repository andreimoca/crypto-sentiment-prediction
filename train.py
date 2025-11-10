"""
Model training and evaluation for cryptocurrency price prediction.
Implements the experimental design described in the research report.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    precision_score, recall_score, brier_score_loss
)
import lightgbm as lgb
from typing import Dict, Tuple
import pickle


class BaselineModel:
    """Persistence baseline: predict previous direction."""
    
    def __init__(self):
        self.name = "Persistence"
    
    def fit(self, X_train, y_train):
        """No training needed for persistence."""
        pass
    
    def predict(self, X_test):
        """Predict previous value."""
        # Assumes first column is return_t-1
        return (X_test.iloc[:, 0] > 0).astype(int).values
    
    def predict_proba(self, X_test):
        """Return binary probabilities."""
        preds = self.predict(X_test)
        return np.column_stack([1 - preds, preds])


class CryptoPredictor:
    """Main prediction model using LightGBM."""
    
    def __init__(self, model_type: str = 'lgbm', use_sentiment: bool = True):
        """
        Initialize predictor.
        
        Args:
            model_type: 'lgbm' or 'logistic'
            use_sentiment: Whether to include sentiment features
        """
        self.model_type = model_type
        self.use_sentiment = use_sentiment
        self.model = None
        
        if model_type == 'lgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                C=10.0,  # inverse of regularization strength
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the model."""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X_test)
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Get feature importance (for tree-based models)."""
        if self.model_type != 'lgbm':
            return None
        
        importance = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save(self, filepath: str):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath: str):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_proba: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (N x 2 array)
    
    Returns:
        Dictionary of metrics
    """
    # Extract positive class probabilities
    y_proba_pos = y_proba[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba_pos),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'brier_score': brier_score_loss(y_true, y_proba_pos)
    }
    
    return metrics


def time_series_split(df: pd.DataFrame, train_months: int = 6, 
                      test_months: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically for time-series validation.
    
    Args:
        df: DataFrame with timestamp column
        train_months: Number of months for training
        test_months: Number of months for testing
    
    Returns:
        Tuple of (train_df, test_df)
    """
    df = df.sort_values('timestamp')
    split_date = df['timestamp'].min() + pd.DateOffset(months=train_months)
    
    train_df = df[df['timestamp'] < split_date]
    test_df = df[df['timestamp'] >= split_date]
    
    return train_df, test_df


def rolling_origin_validation(df: pd.DataFrame, train_window_months: int = 6,
                              test_window_weeks: int = 1, 
                              step_weeks: int = 1) -> list:
    """
    Perform rolling-origin (walk-forward) validation.
    
    Args:
        df: Full dataset with timestamp
        train_window_months: Initial training window size
        test_window_weeks: Test window size
        step_weeks: Step size for advancing window
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    splits = []
    
    # Initial training end
    train_end = df['timestamp'].min() + pd.DateOffset(months=train_window_months)
    
    while True:
        test_start = train_end
        test_end = test_start + pd.DateOffset(weeks=test_window_weeks)
        
        # Check if we have enough test data
        if test_end > df['timestamp'].max():
            break
        
        train_idx = df[df['timestamp'] < train_end].index
        test_idx = df[(df['timestamp'] >= test_start) & 
                     (df['timestamp'] < test_end)].index
        
        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))
        
        # Advance window
        train_end += pd.DateOffset(weeks=step_weeks)
    
    return splits


def print_results(model_name: str, metrics: Dict[str, float]):
    """Print formatted evaluation results."""
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:          {metrics['accuracy']:.3f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"ROC-AUC:           {metrics['roc_auc']:.3f}")
    print(f"Precision:         {metrics['precision']:.3f}")
    print(f"Recall:            {metrics['recall']:.3f}")
    print(f"Brier Score:       {metrics['brier_score']:.3f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    """
    Example usage demonstrating the complete training pipeline.
    This is a template - you need to provide your actual data files.
    """
    
    print("Cryptocurrency Price Prediction - Model Training")
    print("=" * 60)
    
    # Note: You need to prepare your data files first
    # This is just a template showing the expected structure
    
    print("\nThis is a template script. To use it:")
    print("1. Prepare your market data CSV with columns:")
    print("   ['timestamp', 'open', 'high', 'low', 'close', 'volume']")
    print("2. Prepare your sentiment data CSV with columns:")
    print("   ['timestamp', 'text']")
    print("3. Run feature engineering (features.py)")
    print("4. Run this training script with your prepared data")
    print("\nExample workflow:")
    print("""
    from features import FeatureEngineering
    from train import CryptoPredictor, evaluate_model
    
    # Initialize feature engineering
    fe = FeatureEngineering()
    
    # Load and process market data
    market_df = pd.read_csv('market_data.csv')
    market_features = fe.compute_market_features(market_df)
    
    # Load and process sentiment data
    posts_df = pd.read_csv('reddit_posts.csv')
    sentiment_features = fe.compute_sentiment_features_vader(posts_df)
    
    # Align features
    X, y = fe.align_features(market_features, sentiment_features)
    
    # Split data
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Standardize
    X_train_std, X_test_std = fe.standardize_features(X_train, X_test)
    
    # Train model
    model = CryptoPredictor(model_type='lgbm', use_sentiment=True)
    model.fit(X_train_std, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_std)
    y_proba = model.predict_proba(X_test_std)
    metrics = evaluate_model(y_test, y_pred, y_proba)
    print_results("OHLCV + VADER", metrics)
    
    # Get feature importance
    importance = model.get_feature_importance(X.columns.tolist())
    print(importance.head(10))
    """)

