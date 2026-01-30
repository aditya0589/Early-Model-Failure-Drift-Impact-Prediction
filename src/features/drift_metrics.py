import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from typing import Dict, List
from src.utils.logger import setup_logger

logger = setup_logger("drift_metrics")

def calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).
    """
    def scale_range(input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    
    # Handle categorical variables by converting to codes or using value counts directly if low cardinality
    if not np.issubdtype(expected.dtype, np.number):
        # For categorical, we align categories
        expected_counts = expected.value_counts(normalize=True)
        actual_counts = actual.value_counts(normalize=True)
        # Align indices
        all_cats = set(expected_counts.index) | set(actual_counts.index)
        expected_percents = np.array([expected_counts.get(c, 0.0001) for c in all_cats])
        actual_percents = np.array([actual_counts.get(c, 0.0001) for c in all_cats])
    else:
        # Numerical
        try:
            breakpoints = np.percentile(expected, breakpoints)
        except:
             return 0.0 # Constant value

        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi_value

def calculate_ks_test(expected: pd.Series, actual: pd.Series) -> float:
    """
    Calculate Kolmogorov-Smirnov Test statistic.
    Returns the p-value (low p-value indicates different distributions).
    """
    if not np.issubdtype(expected.dtype, np.number):
        return 1.0 # KS not applicable for categorical
        
    stat, p_value = ks_2samp(expected, actual)
    return p_value

def calculate_js_divergence(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """
    Calculate Jensen-Shannon Divergence.
    """
    # Similar binning to PSI
    if not np.issubdtype(expected.dtype, np.number):
         expected_counts = expected.value_counts(normalize=True)
         actual_counts = actual.value_counts(normalize=True)
         all_cats = set(expected_counts.index) | set(actual_counts.index)
         p = np.array([expected_counts.get(c, 0.0001) for c in all_cats])
         q = np.array([actual_counts.get(c, 0.0001) for c in all_cats])
    else:
        try:
            breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
            breakpoints = np.percentile(expected, breakpoints)
            p = np.histogram(expected, breakpoints)[0] / len(expected)
            q = np.histogram(actual, breakpoints)[0] / len(actual)
        except:
            return 0.0

    return jensenshannon(p, q)

def calculate_drift_metrics(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates aggregated drift metrics for the entire batch.
    """
    metrics = {}
    
    # Feature-wise metrics
    psi_scores = []
    ks_p_values = []
    js_scores = []
    
    for col in reference_df.columns:
        if col == 'churn': continue
        
        psi = calculate_psi(reference_df[col], current_df[col])
        ks = calculate_ks_test(reference_df[col], current_df[col])
        js = calculate_js_divergence(reference_df[col], current_df[col])
        
        psi_scores.append(psi)
        ks_p_values.append(ks)
        js_scores.append(js)
        
        metrics[f"{col}_psi"] = psi
        metrics[f"{col}_ks"] = ks
        metrics[f"{col}_js"] = js
        
    # Global aggregates
    metrics["mean_psi"] = np.mean(psi_scores)
    metrics["max_psi"] = np.max(psi_scores)
    metrics["mean_js"] = np.mean(js_scores)
    
    # Drift Risk Score (Simple weighted average of normalized metrics)
    # High PSI -> High Risk
    # Low KS p-value -> High Risk (so we take 1 - p)
    # High JS -> High Risk
    
    metrics["drift_risk_score"] = (
        0.4 * metrics["mean_psi"] + 
        0.3 * metrics["mean_js"] + 
        0.3 * (1 - np.mean(ks_p_values))
    )
    
    return metrics
