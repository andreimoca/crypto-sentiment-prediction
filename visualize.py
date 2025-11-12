"""
Visualization utilities for cryptocurrency price prediction project.
Creates plots for analysis and reporting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Optional


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def plot_price_and_sentiment(market_df: pd.DataFrame, sentiment_df: pd.DataFrame,
                             save_path: str = None):
    """
    Plot Bitcoin price alongside sentiment scores.
    
    Args:
        market_df: DataFrame with timestamp and close price
        sentiment_df: DataFrame with timestamp and sentiment scores
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Price plot
    ax1.plot(market_df['timestamp'], market_df['close'], 
            color='steelblue', linewidth=1.5)
    ax1.set_ylabel('BTC-USD Price ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Bitcoin Price and Social Media Sentiment', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Sentiment plot
    ax2.plot(sentiment_df['timestamp'], sentiment_df['sentiment_mean'],
            color='darkorange', linewidth=1.5, label='Mean Sentiment')
    ax2.fill_between(sentiment_df['timestamp'],
                     sentiment_df['sentiment_mean'] - sentiment_df['sentiment_std'],
                     sentiment_df['sentiment_mean'] + sentiment_df['sentiment_std'],
                     alpha=0.3, color='darkorange', label='±1 Std Dev')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Sentiment Score', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15,
                           save_path: str = None):
    """
    Plot feature importance from tree-based models.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        save_path: Path to save figure
    """
    # Select top N features
    top_features = importance_df.head(top_n).copy()
    
    # Normalize importance
    top_features['importance'] = top_features['importance'] / top_features['importance'].sum()
    
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    
    ax.barh(range(len(top_features)), top_features['importance'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Normalized Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(top_features['importance']):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         save_path: str = None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'],
                cbar_kws={'label': 'Count'},
                ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.7, f'({pct:.1f}%)', 
                   ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray,
                  model_name: str = "Model", save_path: str = None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities (positive class)
        model_name: Name for legend
        save_path: Path to save figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # ROC curve
    ax.plot(fpr, tpr, color='steelblue', linewidth=2,
           label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Random classifier
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_model_comparison(results_dict: dict, metric: str = 'balanced_accuracy',
                         save_path: str = None):
    """
    Compare multiple models on a specific metric.
    
    Args:
        results_dict: Dict of {model_name: metrics_dict}
        metric: Metric to compare
        save_path: Path to save figure
    """
    models = list(results_dict.keys())
    values = [results_dict[m][metric] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'][:len(models)]
    bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'Model Comparison: {metric.replace("_", " ").title()}', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_prediction_timeline(df: pd.DataFrame, y_true: np.ndarray, 
                            y_proba: np.ndarray, save_path: str = None):
    """
    Plot predictions over time with actual outcomes.
    
    Args:
        df: DataFrame with timestamp column
        y_true: True labels
        y_proba: Predicted probabilities (positive class)
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    timestamps = df['timestamp'].values
    
    # Predicted probabilities
    ax1.plot(timestamps, y_proba, color='steelblue', linewidth=1, alpha=0.7)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Decision Threshold')
    ax1.fill_between(timestamps, 0.5, y_proba, 
                     where=(y_proba > 0.5), alpha=0.3, color='green', label='Predict Up')
    ax1.fill_between(timestamps, 0.5, y_proba,
                     where=(y_proba <= 0.5), alpha=0.3, color='red', label='Predict Down')
    ax1.set_ylabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax1.set_title('Prediction Timeline', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Actual outcomes
    colors = ['red' if y == 0 else 'green' for y in y_true]
    ax2.scatter(timestamps, y_true, c=colors, alpha=0.5, s=20)
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Actual Direction', fontsize=12, fontweight='bold')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Down', 'Up'])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_performance_over_time(df: pd.DataFrame, y_true: np.ndarray,
                               y_pred: np.ndarray, window: str = '7D',
                               save_path: str = None):
    """
    Plot rolling accuracy over time.
    
    Args:
        df: DataFrame with timestamp column
        y_true: True labels
        y_pred: Predicted labels
        window: Rolling window size (e.g., '7D' for 7 days)
        save_path: Path to save figure
    """
    # Create DataFrame
    results = pd.DataFrame({
        'timestamp': pd.to_datetime(df['timestamp']),
        'correct': (y_true == y_pred).astype(int)
    })
    
    results = results.set_index('timestamp')
    
    # Compute rolling accuracy
    rolling_acc = results['correct'].rolling(window=window).mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(rolling_acc.index, rolling_acc.values, 
           color='steelblue', linewidth=2, label=f'Rolling Accuracy ({window})')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
    ax.fill_between(rolling_acc.index, 0.5, rolling_acc.values,
                    where=(rolling_acc.values > 0.5), alpha=0.2, color='green')
    ax.fill_between(rolling_acc.index, 0.5, rolling_acc.values,
                    where=(rolling_acc.values <= 0.5), alpha=0.2, color='red')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Performance Over Time (Rolling {window})', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.35, 0.65])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    """
    Example usage of visualization functions.
    """
    print("Cryptocurrency Prediction - Visualization Utilities")
    print("=" * 60)
    
    # Generate synthetic example data
    np.random.seed(42)
    n = 500
    
    dates = pd.date_range('2022-01-01', periods=n, freq='1H')
    
    # Synthetic market data
    price = 40000 + np.cumsum(np.random.randn(n) * 100)
    market_df = pd.DataFrame({
        'timestamp': dates,
        'close': price
    })
    
    # Synthetic sentiment data
    sentiment_mean = np.sin(np.linspace(0, 4*np.pi, n)) * 0.3
    sentiment_std = np.random.uniform(0.1, 0.3, n)
    sentiment_df = pd.DataFrame({
        'timestamp': dates,
        'sentiment_mean': sentiment_mean,
        'sentiment_std': sentiment_std
    })
    
    # Synthetic predictions
    y_true = np.random.binomial(1, 0.55, n)
    y_pred = np.random.binomial(1, 0.55, n)
    y_proba = np.clip(y_true * 0.7 + np.random.beta(2, 2, n) * 0.3, 0, 1)
    
    print("\nGenerating example visualizations with synthetic data...")
    
    # 1. Price and sentiment
    print("1. Creating price and sentiment plot...")
    plot_price_and_sentiment(market_df, sentiment_df, 
                            save_path='price_sentiment.png')
    
    # 2. Feature importance (example)
    print("2. Creating feature importance plot...")
    importance_df = pd.DataFrame({
        'feature': ['return_t-1', 'RSI_14', 'sentiment_mean', 'realized_vol', 
                   'SMA_3', 'sentiment_std', 'volume_std', 'return_t-2'],
        'importance': [0.28, 0.19, 0.15, 0.12, 0.10, 0.08, 0.05, 0.03]
    })
    plot_feature_importance(importance_df, save_path='feature_importance.png')
    
    # 3. ROC curve
    print("3. Creating ROC curve...")
    plot_roc_curve(y_true, y_proba, model_name="OHLCV + VADER",
                  save_path='roc_curve.png')
    
    # 4. Confusion matrix
    print("4. Creating confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png')
    
    # 5. Prediction timeline
    print("5. Creating prediction timeline...")
    plot_prediction_timeline(market_df[:200], y_true[:200], y_proba[:200],
                           save_path='prediction_timeline.png')
    
    print("\n✓ All visualizations generated successfully!")
    print("  Check the current directory for PNG files.")

