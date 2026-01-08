import pandas as pd
import numpy as np
import os

def load_data():
    """Load the simulation data and drift metrics."""
    # Note: Using drift_simulation_data.csv as the prediction logs source
    logs_path = os.path.join("data", "drift_simulation_data.csv")
    metrics_path = os.path.join("data", "drift_metrics.csv")
    
    print(f"Loading logs from {logs_path}...")
    logs = pd.read_csv(logs_path)
    print(f"Loading metrics from {metrics_path}...")
    metrics = pd.read_csv(metrics_path)
    
    return logs, metrics

def compute_entropy_error_correlation(group):
    """Compute correlation between entropy and error for a daily group."""
    if len(group) < 2:
        return 0.0
    # Avoid runtime warnings if variance is 0
    if group['entropy'].std() == 0 or group['is_error'].std() == 0:
        return 0.0
        
    corr = group['entropy'].corr(group['is_error'])
    return 0.0 if pd.isna(corr) else corr

def aggregate_logs(logs):
    """Aggregate log-level data to daily level."""
    print("Aggregating logs by day...")
    
    # Define aggregation dictionary
    daily_stats = logs.groupby('day').agg({
        'is_error': 'mean',
        'entropy': ['mean', 'std'],
        'pred_prob': ['mean', 'std']
    })
    
    # Flatten multi-level columns
    daily_stats.columns = [
        'error_rate',
        'mean_entropy', 'std_entropy',
        'mean_confidence', 'std_confidence'
    ]
    
    daily_stats = daily_stats.reset_index()
    
    # Compute correlation separately as it's complex for standard agg
    print("Computing daily entropy-error correlations...")
    correlations = logs.groupby('day').apply(compute_entropy_error_correlation)
    daily_stats['entropy_error_correlation'] = correlations.values
    
    print(f"Aggregated daily logs shape: {daily_stats.shape}")
    return daily_stats

def create_target(metrics, looking_ahead_window=7, failure_threshold=0.85):
    """Create the binary target: failure in next N days."""
    print("Creating target 'failure_in_next_7_days'...")
    
    metrics = metrics.sort_values('day').reset_index(drop=True)
    target_col = f'failure_in_next_{looking_ahead_window}_days'
    
    targets = []
    
    for t in metrics['day']:
        # Look ahead window: [t+1, t+7]
        future_window = metrics[
            (metrics['day'] > t) & 
            (metrics['day'] <= t + looking_ahead_window)
        ]
        
        if future_window.empty:
            targets.append(0) # Or NaN, but usually 0 for end of series in this context or drop
            continue
            
        # Fail if ANY accuracy in window < threshold
        # "strictly below 0.85" -> < 0.85
        has_failure = (future_window['accuracy'] < failure_threshold).any()
        targets.append(1 if has_failure else 0)
    
    metrics[target_col] = targets
    return metrics, target_col

def feature_engineering(df, numeric_cols):
    """Create lag and rolling features."""
    print("Performing feature engineering (Lags & Rolling)...")
    
    df = df.sort_values('day').reset_index(drop=True)
    
    # Store original numeric columns to iterate over (excluding target/day)
    # We will compute features for them
    
    for col in numeric_cols:
        # Lags
        df[f'{col}_lag_1'] = df[col].shift(1)
        df[f'{col}_lag_3'] = df[col].shift(3)
        df[f'{col}_lag_7'] = df[col].shift(7)
        
        # Rolling features (window=7, closed='left' is implicitly handled by shift if we were strict, 
        # but here requirement is "last 7 days (t-6 to t)".
        # Pandas rolling includes current row by default. 
        # Requirement: "rolling mean over last 7 days (t-6 to t)" -> This usually includes current day t.
        # "Use ONLY past values" constraint might imply we should shift first?
        # Re-reading: "rolling mean over last 7 days (t-6 to t)" usually means INCLUSIVE of t.
        # However, "Use ONLY past values" usually applies to predicting t+1.
        # Ideally for a drift prediction at day t (to predict t+1...t+7), we know day t's metrics.
        # So using day t's data is fine. We just shouldn't use t+1.
        
        # Rolling window
        # min_periods=7 enforces we need full history, dropping early rows
        indexer = df[col].rolling(window=7, min_periods=7)
        
        df[f'{col}_roll_mean_7'] = indexer.mean()
        df[f'{col}_roll_std_7'] = indexer.std()
        
        # Linear trend (slope)
        # Using numpy polyfit on the rolling window
        def calculate_slope(x):
            return np.polyfit(np.arange(len(x)), x, 1)[0]
            
        df[f'{col}_trend_7'] = indexer.apply(calculate_slope, raw=True)
        
    return df

def main():
    # -------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------
    logs, metrics = load_data()
    
    # -------------------------------------------------------------
    # 2. Aggregate Logs
    # -------------------------------------------------------------
    daily_features = aggregate_logs(logs)
    
    # Save intermediate
    daily_features_path = os.path.join("data", "daily_prediction_features.csv")
    daily_features.to_csv(daily_features_path, index=False)
    print(f"Saved daily features to {daily_features_path}")
    
    # -------------------------------------------------------------
    # 3. Define Target
    # -------------------------------------------------------------
    metrics, target_col = create_target(metrics)
    
    # -------------------------------------------------------------
    # 4. Merge Datasets
    # -------------------------------------------------------------
    print("Merging datasets...")
    # Inner join on day
    merged_df = pd.merge(daily_features, metrics, on='day', how='inner')
    
    # Label encode drift_type
    print("Encoding drift_type...")
    # drift_type might be strings like 'none', 'gradual', etc.
    # Simple factorization
    merged_df['drift_type_encoded'] = pd.factorize(merged_df['drift_type'])[0]
    
    # Save merged
    merged_path = os.path.join("data", "merged_drift_dataset.csv")
    merged_df.to_csv(merged_path, index=False)
    
    # -------------------------------------------------------------
    # 5. Feature Engineering
    # -------------------------------------------------------------
    # Identify numeric columns for FE
    # All columns from daily_features (excluding day) + columns from metrics (excluding day, target, drift_type)
    
    features_to_process = [
        'error_rate', 'mean_entropy', 'std_entropy', 
        'mean_confidence', 'std_confidence', 'entropy_error_correlation',
        'drift_intensity', 'accuracy', 'psi', 'avg_entropy', 'avg_confidence'
    ]
    
    # Double check if these columns exist
    features_to_process = [c for c in features_to_process if c in merged_df.columns]
    
    # Process
    final_df = feature_engineering(merged_df, features_to_process)
    
    # Drop rows with NaNs (created by lags/rolling)
    rows_before = len(final_df)
    final_df = final_df.dropna()
    rows_dropped = rows_before - len(final_df)
    
    print(f"Rows dropped due to incomplete history (Lags/Rolling): {rows_dropped}")
    
    # -------------------------------------------------------------
    # 6. Final Data Selection
    # -------------------------------------------------------------
    # Exclude non-input columns
    # We keep 'day' just for reference? User said "Contain only tabular features + target"
    # Usually 'day' is metadata. 'drift_type' is raw string, we have encoded it.
    
    # Columns to KEEP:
    # - Original numeric features (current day values are valid features for prediction?)
    #   User said "Use ONLY past values" under Feature Engineering rules.
    #   Usually "current day t" IS past relative to "future t+1".
    #   Standard tabular set: X_t -> Y_{t+1...t+7}
    #   So we keep X_t and X_{t-k} features.
    
    drop_cols = ['drift_type'] # original string column
    final_df = final_df.drop(columns=drop_cols, errors='ignore')
    
    # Move target to end for cleanliness
    cols = [c for c in final_df.columns if c != target_col] + [target_col]
    final_df = final_df[cols]
    
    output_path = os.path.join("data", "tabular_training_dataset.csv")
    final_df.to_csv(output_path, index=False)
    
    print("==================================================")
    print("SUCCESS")
    print("==================================================")
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Saved to: {output_path}")
    print("\nFeature Columns:")
    for col in final_df.columns:
        print(f"- {col}")

if __name__ == "__main__":
    main()
