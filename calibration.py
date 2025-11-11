"""
Probability calibration for cryptocurrency price prediction models.
Implements Platt Scaling as described in the research report.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
from typing import Tuple


class PlattScaling:
    """
    Post-hoc probability calibration using Platt Scaling.
    
    Fits a logistic regression: P_cal(y=1|f(x)) = sigma(a * f(x) + b)
    where f(x) is the uncalibrated model output.
    """
    
    def __init__(self):
        self.calibrator = LogisticRegression()
        self.is_fitted = False
    
    def fit(self, y_true: np.ndarray, y_proba: np.ndarray):
        """
        Fit the calibration model.
        
        Args:
            y_true: True binary labels
            y_proba: Uncalibrated probabilities (positive class)
        """
        # Reshape for sklearn
        y_proba_reshaped = y_proba.reshape(-1, 1)
        
        # Fit logistic regression
        self.calibrator.fit(y_proba_reshaped, y_true)
        self.is_fitted = True
        
        return self
    
    def transform(self, y_proba: np.ndarray) -> np.ndarray:
        """
        Apply calibration to new probabilities.
        
        Args:
            y_proba: Uncalibrated probabilities (positive class)
        
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")
        
        y_proba_reshaped = y_proba.reshape(-1, 1)
        y_proba_calibrated = self.calibrator.predict_proba(y_proba_reshaped)[:, 1]
        
        return y_proba_calibrated
    
    def fit_transform(self, y_true: np.ndarray, y_proba: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(y_true, y_proba)
        return self.transform(y_proba)
    
    def get_parameters(self) -> Tuple[float, float]:
        """Get the fitted a and b parameters."""
        if not self.is_fitted:
            raise ValueError("Calibrator has not been fitted yet")
        
        a = self.calibrator.coef_[0][0]
        b = self.calibrator.intercept_[0]
        
        return a, b


def compute_expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, 
                                      n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE = sum_b (n_b / N) * |acc(b) - conf(b)|
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities (positive class)
        n_bins: Number of bins for discretization
    
    Returns:
        ECE value
    """
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            # Accuracy in bin
            accuracy_in_bin = np.mean(y_true[in_bin])
            # Average confidence in bin
            avg_confidence_in_bin = np.mean(y_proba[in_bin])
            # Add to ECE
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    
    return ece


def plot_reliability_diagram(y_true: np.ndarray, y_proba: np.ndarray, 
                            y_proba_calibrated: np.ndarray = None,
                            n_bins: int = 10, 
                            save_path: str = None):
    """
    Plot reliability diagram (calibration curve).
    
    Args:
        y_true: True binary labels
        y_proba: Uncalibrated probabilities
        y_proba_calibrated: Optional calibrated probabilities
        n_bins: Number of bins
        save_path: Path to save figure (if None, displays instead)
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Uncalibrated curve
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
    brier_uncal = brier_score_loss(y_true, y_proba)
    ece_uncal = compute_expected_calibration_error(y_true, y_proba, n_bins)
    
    ax.plot(prob_pred, prob_true, 's-', label=f'Uncalibrated (Brier={brier_uncal:.3f}, ECE={ece_uncal:.3f})')
    
    # Calibrated curve (if provided)
    if y_proba_calibrated is not None:
        prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_proba_calibrated, 
                                                         n_bins=n_bins, strategy='uniform')
        brier_cal = brier_score_loss(y_true, y_proba_calibrated)
        ece_cal = compute_expected_calibration_error(y_true, y_proba_calibrated, n_bins)
        
        ax.plot(prob_pred_cal, prob_true_cal, 'o-', 
               label=f'Calibrated (Brier={brier_cal:.3f}, ECE={ece_cal:.3f})')
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reliability diagram saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_calibration_methods(y_true: np.ndarray, y_proba: np.ndarray, 
                               calibration_set_size: float = 0.3) -> pd.DataFrame:
    """
    Compare uncalibrated vs. calibrated predictions.
    
    Args:
        y_true: True labels
        y_proba: Uncalibrated probabilities
        calibration_set_size: Fraction of data to use for calibration
    
    Returns:
        DataFrame with comparison metrics
    """
    # Split into calibration and evaluation sets
    n = len(y_true)
    n_cal = int(n * calibration_set_size)
    
    y_cal, y_eval = y_true[:n_cal], y_true[n_cal:]
    p_cal, p_eval = y_proba[:n_cal], y_proba[n_cal:]
    
    # Fit Platt Scaling
    platt = PlattScaling()
    platt.fit(y_cal, p_cal)
    
    # Calibrate evaluation set
    p_eval_calibrated = platt.transform(p_eval)
    
    # Compute metrics
    results = {
        'Method': ['Uncalibrated', 'Platt Scaling'],
        'Brier Score': [
            brier_score_loss(y_eval, p_eval),
            brier_score_loss(y_eval, p_eval_calibrated)
        ],
        'ECE': [
            compute_expected_calibration_error(y_eval, p_eval),
            compute_expected_calibration_error(y_eval, p_eval_calibrated)
        ]
    }
    
    df = pd.DataFrame(results)
    
    # Get Platt parameters
    a, b = platt.get_parameters()
    print(f"\nPlatt Scaling parameters: a={a:.4f}, b={b:.4f}")
    
    return df


if __name__ == "__main__":
    """
    Example usage demonstrating probability calibration.
    """
    
    print("Probability Calibration - Platt Scaling")
    print("=" * 60)
    
    # Generate synthetic example data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate uncalibrated probabilities (typically overconfident)
    y_true = np.random.binomial(1, 0.5, n_samples)
    
    # Create overconfident predictions
    y_proba_base = y_true * np.random.beta(8, 2, n_samples) + \
                   (1 - y_true) * np.random.beta(2, 8, n_samples)
    
    # Add some noise
    y_proba = np.clip(y_proba_base + np.random.normal(0, 0.05, n_samples), 0, 1)
    
    print("\nExample with synthetic data:")
    print(f"Number of samples: {n_samples}")
    print(f"Positive class ratio: {y_true.mean():.2%}")
    
    # Compare methods
    comparison = compare_calibration_methods(y_true, y_proba)
    print("\nCalibration Comparison:")
    print(comparison.to_string(index=False))
    
    # Split for visualization
    n_cal = 300
    y_cal, y_vis = y_true[:n_cal], y_true[n_cal:]
    p_cal, p_vis = y_proba[:n_cal], y_proba[n_cal:]
    
    # Calibrate
    platt = PlattScaling()
    platt.fit(y_cal, p_cal)
    p_vis_calibrated = platt.transform(p_vis)
    
    # Plot
    print("\nGenerating reliability diagram...")
    plot_reliability_diagram(y_vis, p_vis, p_vis_calibrated, 
                           save_path='reliability_diagram.png')
    
    print("\nCalibration complete! Check 'reliability_diagram.png' for visualization.")

