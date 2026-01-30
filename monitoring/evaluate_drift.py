import pandas as pd
import os
from src.features.drift_metrics import calculate_drift_metrics
from src.utils.logger import setup_logger

logger = setup_logger("evaluate_drift")

def monitor_drift(reference_path: str, current_batch_path: str):
    """
    Standalone script to monitor drift between reference and current batch.
    """
    if not os.path.exists(reference_path) or not os.path.exists(current_batch_path):
        logger.error("Data files not found.")
        return

    ref_df = pd.read_csv(reference_path)
    cur_df = pd.read_csv(current_batch_path)
    
    # Remove target if present
    if 'churn' in ref_df.columns:
        ref_df = ref_df.drop(columns=['churn'])
    if 'churn' in cur_df.columns:
        cur_df = cur_df.drop(columns=['churn'])
        
    metrics = calculate_drift_metrics(ref_df, cur_df)
    
    logger.info("Drift Metrics:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
        
    return metrics

if __name__ == "__main__":
    # Example
    monitor_drift("data/reference/train.csv", "data/reference/test.csv")
