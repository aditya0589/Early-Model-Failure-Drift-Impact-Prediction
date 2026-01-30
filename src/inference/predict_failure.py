import pandas as pd
import joblib
import os
from src.models.failure_model import FailurePredictor
from src.features.drift_metrics import calculate_drift_metrics
from src.features.feature_stats import calculate_feature_stats
from src.utils.logger import setup_logger

logger = setup_logger("predict_failure")

def load_reference_data():
    return pd.read_csv("data/reference/train.csv")

def predict_failure_probability(batch_df: pd.DataFrame, reference_df: pd.DataFrame, failure_model: FailurePredictor):
    """
    Predicts the probability of model failure for a new batch of data.
    """
    # Calculate features
    drift_metrics = calculate_drift_metrics(reference_df.drop(columns=['churn'], errors='ignore'), batch_df)
    feature_stats = calculate_feature_stats(batch_df)
    
    # Prepare input vector
    features = {**drift_metrics, **feature_stats}
    X = pd.DataFrame([features])
    
    # Handle NaNs (common if drift metrics fail for some columns)
    X = X.fillna(0)
    
    # Predict
    prob = failure_model.predict_proba(X)[0]
    return prob, features

if __name__ == "__main__":
    # Example usage
    try:
        failure_model = FailurePredictor()
        failure_model.load("experiments/results/failure_model.joblib")
        
        ref_df = load_reference_data()
        
        # Simulate a batch (load from file in real scenario)
        from src.data.load_data import load_data
        df = load_data()
        batch = df.sample(1000)
        
        prob, _ = predict_failure_probability(batch, ref_df, failure_model)
        print(f"Probability of Model Failure: {prob:.4f}")
        
    except Exception as e:
        logger.error(f"Error in inference: {e}")
