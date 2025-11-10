"""
Feature engineering for cryptocurrency price prediction.
Implements market and sentiment feature extraction as described in the research report.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class FeatureEngineering:
    """Handles feature extraction from market and sentiment data."""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.scaler_params = {}
        
    def compute_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators from OHLCV data.
        
        Args:
            df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        Returns:
            DataFrame with market features
        """
        df = df.copy()
        
        # Lagged returns
        df['return_t-1'] = np.log(df['close'] / df['close'].shift(1))
        df['return_t-2'] = np.log(df['close'].shift(1) / df['close'].shift(2))
        df['return_t-3'] = np.log(df['close'].shift(2) / df['close'].shift(3))
        
        # Simple Moving Averages
        df['SMA_3'] = df['close'].rolling(window=3).mean()
        df['SMA_12'] = df['close'].rolling(window=12).mean()
        
        # Relative Strength Index (RSI)
        df['RSI_14'] = self._compute_rsi(df['close'], period=14)
        
        # Volume (will be standardized later)
        df['volume_std'] = df['volume']
        
        # Realized volatility (24-hour)
        df['realized_vol'] = df['return_t-1'].rolling(window=24).std()
        
        # Target variable: 1h ahead price direction
        df['target'] = (np.log(df['close'].shift(-1) / df['close']) > 0).astype(int)
        
        # Drop NaN rows created by lagged/rolling features
        df = df.dropna()
        
        return df
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def compute_sentiment_features_vader(self, posts: pd.DataFrame) -> pd.DataFrame:
        """
        Compute sentiment features using VADER.
        
        Args:
            posts: DataFrame with columns ['timestamp', 'text']
        
        Returns:
            DataFrame with hourly sentiment aggregates
        """
        posts = posts.copy()
        
        # Compute VADER sentiment scores
        posts['sentiment'] = posts['text'].apply(
            lambda x: self.vader.polarity_scores(str(x))['compound']
        )
        
        # Resample to hourly and compute aggregates
        posts['timestamp'] = pd.to_datetime(posts['timestamp'])
        posts = posts.set_index('timestamp')
        
        hourly = posts.resample('1H').agg({
            'sentiment': ['mean', 'std', 'count']
        })
        hourly.columns = ['sentiment_mean', 'sentiment_std', 'sentiment_count']
        
        # Log-transform count
        hourly['sentiment_count'] = np.log(1 + hourly['sentiment_count'])
        
        # Fill missing values
        hourly = hourly.fillna(0)
        
        return hourly.reset_index()
    
    def compute_sentiment_features_finbert(self, posts: pd.DataFrame, 
                                          model_name: str = "ProsusAI/finbert") -> pd.DataFrame:
        """
        Compute sentiment features using FinBERT.
        
        Args:
            posts: DataFrame with columns ['timestamp', 'text']
            model_name: HuggingFace model name
        
        Returns:
            DataFrame with hourly sentiment aggregates
        """
        posts = posts.copy()
        
        # Load FinBERT model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Compute sentiment scores (batched for efficiency)
        sentiments = []
        batch_size = 32
        
        for i in range(0, len(posts), batch_size):
            batch = posts['text'].iloc[i:i+batch_size].tolist()
            inputs = tokenizer(batch, padding=True, truncation=True, 
                             return_tensors="pt", max_length=128)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                # Convert to compound score: positive - negative
                sentiment = (probs[:, 2] - probs[:, 0]).numpy()
                sentiments.extend(sentiment)
        
        posts['sentiment'] = sentiments
        
        # Resample to hourly and compute aggregates
        posts['timestamp'] = pd.to_datetime(posts['timestamp'])
        posts = posts.set_index('timestamp')
        
        hourly = posts.resample('1H').agg({
            'sentiment': ['mean', 'std', 'count']
        })
        hourly.columns = ['sentiment_mean', 'sentiment_std', 'sentiment_count']
        
        # Log-transform count
        hourly['sentiment_count'] = np.log(1 + hourly['sentiment_count'])
        
        # Fill missing values
        hourly = hourly.fillna(0)
        
        return hourly.reset_index()
    
    def align_features(self, market_df: pd.DataFrame, 
                      sentiment_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Align market and sentiment features on timestamp.
        
        Args:
            market_df: Market features DataFrame
            sentiment_df: Sentiment features DataFrame
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
        sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
        
        # Merge on timestamp
        merged = pd.merge(market_df, sentiment_df, on='timestamp', how='inner')
        
        # Extract features and target
        feature_cols = [col for col in merged.columns 
                       if col not in ['timestamp', 'target', 'open', 'high', 'low', 'close']]
        
        X = merged[feature_cols]
        y = merged['target']
        
        return X, y
    
    def standardize_features(self, X_train: pd.DataFrame, 
                           X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Standardize features using training set statistics.
        
        Args:
            X_train: Training features
            X_test: Test features
        
        Returns:
            Tuple of (standardized X_train, standardized X_test)
        """
        # Compute mean and std from training set
        self.scaler_params = {
            'mean': X_train.mean(),
            'std': X_train.std()
        }
        
        # Standardize
        X_train_std = (X_train - self.scaler_params['mean']) / self.scaler_params['std']
        X_test_std = (X_test - self.scaler_params['mean']) / self.scaler_params['std']
        
        # Handle any NaN or inf values
        X_train_std = X_train_std.fillna(0).replace([np.inf, -np.inf], 0)
        X_test_std = X_test_std.fillna(0).replace([np.inf, -np.inf], 0)
        
        return X_train_std, X_test_std


def get_feature_names(include_sentiment: bool = True) -> list:
    """Return list of feature names used in the model."""
    market_features = [
        'return_t-1', 'return_t-2', 'return_t-3',
        'SMA_3', 'SMA_12', 'RSI_14', 'volume_std', 'realized_vol'
    ]
    
    if include_sentiment:
        sentiment_features = ['sentiment_mean', 'sentiment_std', 'sentiment_count']
        return market_features + sentiment_features
    
    return market_features

