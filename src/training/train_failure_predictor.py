import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from src.models.churn_model import ChurnModel
from src.models.failure_model import FailurePredictor
from src.data.drift_simulation import DriftSimulator
from src.features.drift_metrics import calculate_drift_metrics
from src.features.feature_stats import calculate_feature_stats
from src.utils.logger import setup_logger
from src.utils.config import config

logger = setup_logger("train_failure_predictor")

def generate_failure_training_data(churn_model: ChurnModel, 
                                   reference_data: pd.DataFrame, 
                                   n_batches: int = 100, 
                                   batch_size: int = 1000):
    """
    Generates training data for the failure predictor by simulating drifted batches.
    """
    logger.info(f"Generating {n_batches} simulated batches for failure training...")
    
    simulator = DriftSimulator(reference_data)
    drift_config = config.drift_config
    
    records = []
    
    # Pre-calculate reference predictions for performance comparison baseline
    # Actually, we compare batch performance against a threshold, or drop relative to validation?
    # Let's use absolute performance drop from a baseline (e.g. validation accuracy)
    # For simplicity, we'll calculate performance on the current batch and check if it's below threshold.
    
    # We need ground truth for the batch to calculate performance.
    # In reality, we wouldn't have ground truth immediately. 
    # But for *training* the failure predictor, we use historical data where we DO have ground truth eventually.
    
        # Calculate baseline accuracy from reference data
        # Ideally this should be done once outside the loop, but for now it's fine
        # Or even better, pass it in.
        
    # Calculate baseline outside loop for efficiency
    baseline_metrics = churn_model.evaluate(reference_data.drop(columns=['churn']), reference_data['churn'])
    baseline_acc = baseline_metrics['accuracy']
    logger.info(f"Baseline Accuracy: {baseline_acc:.4f}")

    for i in tqdm(range(n_batches)):
        # Randomly decide drift type and intensity
        if np.random.random() < 0.5:
            drift_type = None
            intensity = 0.0
        else:
            drift_type = np.random.choice(drift_config.drift_types)
            intensity = np.random.uniform(*drift_config.drift_intensity_range)
            
        batch_df, meta = simulator.simulate_batch(batch_size, drift_type, intensity)
        
        X_batch = batch_df.drop(columns=['churn'])
        y_batch = batch_df['churn']
        
        # 1. Calculate Drift Metrics
        # We compare batch X against reference X (training data)
        drift_metrics = calculate_drift_metrics(reference_data.drop(columns=['churn']), X_batch)
        
        # 2. Calculate Feature Stats
        feature_stats = calculate_feature_stats(X_batch)
        
        # 3. Calculate Model Performance (Ground Truth for labeling)
        metrics = churn_model.evaluate(X_batch, y_batch)
        accuracy = metrics['accuracy']
        
        # 4. Generate Failure Label
        # Failure = High Drift AND Low Performance
        # Or just Performance Drop? The prompt says:
        # failure = 1 if (drift_score >= DRIFT_THRESHOLD and performance_drop >= PERFORMANCE_THRESHOLD)
        
        perf_drop = max(0, baseline_acc - accuracy)
        
        is_failure = 1 if (
            drift_metrics['drift_risk_score'] >= drift_config.drift_score_threshold and 
            perf_drop >= drift_config.performance_drop_threshold
        ) else 0
        
        # Construct record
        record = {
            **drift_metrics,
            **feature_stats,
            "drift_type": drift_type,
            "drift_intensity": intensity,
            "actual_accuracy": accuracy,
            "performance_drop": perf_drop,
            "failure_label": is_failure
        }
        records.append(record)
        
    return pd.DataFrame(records)

def main():
    # Load resources
    logger.info("Loading resources...")
    train_df = pd.read_csv("data/reference/train.csv")
    
    churn_model = ChurnModel()
    churn_model.load("experiments/results/churn_model.joblib")
    
    # Generate Training Data for Meta-Model
    failure_df = generate_failure_training_data(churn_model, train_df, n_batches=config.drift_config.simulation['n_batches'])
    
    # Save the generated dataset for analysis
    failure_df.to_csv("data/processed/failure_training_data.csv", index=False)
    
    # Train Failure Predictor
    # Features: Drift metrics + Feature Stats
    # Target: failure_label
    # Exclude metadata and ground truth metrics from features
    drop_cols = ['drift_type', 'drift_intensity', 'actual_accuracy', 'performance_drop', 'failure_label']
    feature_cols = [c for c in failure_df.columns if c not in drop_cols]
    
    X = failure_df[feature_cols]
    y = failure_df['failure_label']

    # Handle NaNs (e.g. from failed drift metrics)
    if X.isnull().values.any():
        nan_cols = X.columns[X.isnull().any()].tolist()
        logger.warning(f"Found NaNs in features {nan_cols}. Filling with 0.")
        X = X.fillna(0)
    
    logger.info(f"Training failure predictor on {len(X)} samples with {X.shape[1]} features...")
    logger.info(f"Failure rate: {y.mean():.2%}")
    
    failure_model = FailurePredictor()
    failure_model.train(X, y)
    
    # Evaluate (on same set for now, ideally split)
    failure_model.evaluate(X, y)
    
    # Save
    failure_model.save("experiments/results/failure_model.joblib")

if __name__ == "__main__":
    main()
