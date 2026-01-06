import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class DriftPipeline:
    def __init__(self, dataset_name='default-of-credit-card-clients'):
        self.dataset_name = dataset_name
        self.X_train = None
        self.X_prod = None
        self.y_train = None
        self.y_prod = None
        self.model = None
        self.scaler = None
        self.drift_log = []
        self.predictions_log = []
        self.numeric_features = []
        
    def load_data(self):
        """Loads and preprocesses the dataset."""
        print(f"Loading dataset: {self.dataset_name}...")
        try:
            # Try loading Default of Credit Card Clients
            # ID 42477 is 'default-of-credit-card-clients' on OpenML
            data = fetch_openml(data_id=42477, as_frame=True, parser='auto')
            df = data.frame
            # Rename target for consistency
            if 'default payment next month' in df.columns:
                df = df.rename(columns={'default payment next month': 'target'})
            elif 'class' in df.columns:
                df = df.rename(columns={'class': 'target'})
            else:
                # Fallback for other datasets
                df['target'] = data.target
            
            # Simple cleanup: drop rows with missing target
            df = df.dropna(subset=['target'])
            
            # Subsample if too large to speed up demo
            if len(df) > 30000:
                df = df.sample(30000, random_state=42)
                
        except Exception as e:
            print(f"Failed to load OpenML dataset: {e}. Falling back to Breast Cancer dataset.")
            data = load_breast_cancer(as_frame=True)
            df = data.frame
            df['target'] = data.target

        # Convert target to int
        df['target'] = df['target'].astype(int)
        
        X = df.drop(columns=['target'])
        y = df['target']
        
        # Identify numeric columns for drift injection
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Initial Train/Production Split (40% Train, 60% Production Stream)
        self.X_train_raw, self.X_prod_raw, self.y_train, self.y_prod = train_test_split(
            X, y, test_size=0.6, random_state=42, stratify=y
        )
        
        print(f"Data Loaded. Train size: {len(self.X_train_raw)}, Production stream size: {len(self.X_prod_raw)}")

    def train_baseline(self):
        """Trains the baseline classifier."""
        print("Training baseline model...")
        
        # Preprocessing Pipeline
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ])
        
        self.pipeline.fit(self.X_train_raw, self.y_train)
        
        # Baseline Metrics
        y_pred = self.pipeline.predict(self.X_train_raw)
        acc = accuracy_score(self.y_train, y_pred)
        print(f"Baseline Training Accuracy: {acc:.4f}")
        
        # Store baseline statistics for PSI calculation
        self.baseline_stats = self.X_train_raw[self.numeric_features].describe()

    def inject_drift(self, batch_df, intensity, drift_type='covariate'):
        """
        Injects drift into a batch of data.
        intensity: 0.0 to 1.0
        drift_type: 'covariate', 'noise', 'quality', 'combined'
        """
        df_drifted = batch_df.copy()
        
        if intensity <= 0:
            return df_drifted

        # 1. Covariate Drift: Shift Mean and Scale Variance of top features
        if drift_type in ['covariate', 'combined']:
            # Apply to random subset of features or specific important ones
            # For simplicity, apply to first 5 numeric features
            target_cols = self.numeric_features[:5]
            for col in target_cols:
                shift = self.baseline_stats[col]['std'] * 2.0 * intensity  # Shift by up to 2 std devs
                df_drifted[col] = df_drifted[col] + shift
        
        # 2. Noise Drift: Add Gaussian Noise
        if drift_type in ['noise', 'combined']:
            noise_cols = self.numeric_features
            for col in noise_cols:
                noise = np.random.normal(0, self.baseline_stats[col]['std'] * 0.5 * intensity, size=len(df_drifted))
                df_drifted[col] = df_drifted[col] + noise

        # 3. Quality Drift: Introduce NaNs (Simulating sensor failure)
        # Note: The pipeline handles NaNs, but this alters the data distribution
        if drift_type in ['quality', 'combined']:
            mask_prob = 0.3 * intensity # Up to 30% missing
            mask = np.random.rand(*df_drifted[self.numeric_features].shape) < mask_prob
            df_drifted[self.numeric_features] = df_drifted[self.numeric_features].mask(mask)

        return df_drifted

    def calculate_psi(self, expected, actual, buckets=10):
        """Calculate Population Stability Index (PSI) for a single feature."""
        def scale_range(input, min_val, max_val):
            input += -(np.min(input))
            input /= np.max(input) / (max_val - min_val)
            input += min_val
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        
        try:
            # Handle potential NaNs by dropping them for PSI calculation
            expected = expected.dropna()
            actual = actual.dropna()
            
            if len(expected) == 0 or len(actual) == 0:
                return 0.0

            # Quantiles only work on valid data
            breakpoints_values = np.percentile(expected, breakpoints)
            
            # Avoid duplicate bin edges
            breakpoints_values = np.unique(breakpoints_values)
            
            expected_percents = np.histogram(expected, breakpoints_values)[0] / len(expected)
            actual_percents = np.histogram(actual, breakpoints_values)[0] / len(actual)

            # Avoid division by zero
            expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
            actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

            psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
            return psi_value
        except Exception:
            return 0.0
            
    def compute_batch_psi(self, batch_df):
        """Computes average PSI across top numeric features."""
        psi_scores = []
        # Check top 5 features
        for col in self.numeric_features[:5]:
            psi = self.calculate_psi(self.X_train_raw[col], batch_df[col])
            psi_scores.append(psi)
        return np.mean(psi_scores)

    def run_simulation(self, n_days=60, batch_size=200):
        """Simulates production timeline with progressive drift."""
        print(f"Starting simulation for {n_days} days...")
        
        # Prepare production stream
        # We recycle the production data if it's smaller than needed
        if len(self.X_prod_raw) < n_days * batch_size:
            print("Warning: Production data smaller than requested simulation. Recycling data.")
            self.X_prod_raw = pd.concat([self.X_prod_raw] * (int((n_days * batch_size) / len(self.X_prod_raw)) + 2))
            self.y_prod = pd.concat([self.y_prod] * (int((n_days * batch_size) / len(self.y_prod)) + 2))
            
        
        # Simulation Parameters
        drift_start_day = 15
        
        current_idx = 0
        
        for day in range(n_days):
            # 1. Get Daily Batch
            X_batch_raw = self.X_prod_raw.iloc[current_idx : current_idx + batch_size].copy()
            y_batch = self.y_prod.iloc[current_idx : current_idx + batch_size].copy()
            current_idx += batch_size
            
            # 2. Determine Drift Intensity
            # No drift for first 15 days, then linear increase
            if day < drift_start_day:
                intensity = 0.0
                drift_type = None
            else:
                # Scale intensity from 0 to 1.0 over the remaining days
                intensity = min(1.0, (day - drift_start_day) / (n_days - drift_start_day))
                # Mix drift types
                drift_type = 'combined' 
            
            # 3. Inject Drift
            X_batch_drifted = self.inject_drift(X_batch_raw, intensity, drift_type)
            
            # 4. Generate Predictions & Log
            # Predict Proba
            y_proba = self.pipeline.predict_proba(X_batch_drifted)[:, 1]
            y_pred = (y_proba > 0.5).astype(int)
            
            # Prediction Entropy (Uncertainty)
            # Clip to avoid log(0)
            p = np.clip(y_proba, 1e-6, 1 - 1e-6)
            entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            
            # 5. Compute Metrics
            # Ground truth is available immediately for metric calculation in this sim, 
            # but we can flag it as "delayed" in the final dataset if needed.
            acc = accuracy_score(y_batch, y_pred)
            psi = self.compute_batch_psi(X_batch_drifted)
            mean_conf = np.mean(np.max(self.pipeline.predict_proba(X_batch_drifted), axis=1))

            # 6. Log Daily Stats
            daily_log = {
                'day': day,
                'drift_intensity': intensity,
                'accuracy': acc,
                'psi': psi,
                'avg_entropy': np.mean(entropy),
                'avg_confidence': mean_conf,
                'drift_type': drift_type if drift_type else 'none'
            }
            self.drift_log.append(daily_log)
            
            # 7. Log Individual Predictions (The "Product")
            batch_preds = pd.DataFrame({
                'day': day,
                'true_label': y_batch.values,
                'pred_label': y_pred,
                'pred_prob': y_proba,
                'entropy': entropy,
                'drift_intensity_true': intensity # Labeled for research
            })
            self.predictions_log.append(batch_preds)
            
            if day % 10 == 0:
                print(f"Day {day}: Intensity={intensity:.2f}, Accuracy={acc:.4f}, PSI={psi:.4f}")

    def save_outputs(self):
        """Compiles logs and saves files."""
        # 1. Aggregate daily metrics
        metrics_df = pd.DataFrame(self.drift_log)
        
        # 2. Aggregate final dataset
        final_df = pd.concat(self.predictions_log, ignore_index=True)
        
        # 3. Create failure label
        # Example: Failure if accuracy drops below 80% of baseline (approx 0.82 -> 0.65)
        # Or simpler: Is this prediction incorrect?
        final_df['is_error'] = (final_df['true_label'] != final_df['pred_label']).astype(int)
        
        # Save CSV
        print("Saving datasets...")
        metrics_df.to_csv('drift_metrics.csv', index=False)
        final_df.to_csv('drift_simulation_data.csv', index=False)
        
        # 4. Visualization
        print("Generating plots...")
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        sns.lineplot(data=metrics_df, x='day', y='accuracy', label='Accuracy', color='blue')
        ax2 = plt.twinx()
        sns.lineplot(data=metrics_df, x='day', y='drift_intensity', ax=ax2, label='Drift Intensity', color='red', linestyle='--')
        plt.title('Accuracy vs Drift Intensity')
        
        plt.subplot(1, 2, 2)
        sns.lineplot(data=metrics_df, x='day', y='psi', color='orange')
        plt.title('Population Stability Index (PSI) Over Time')
        
        plt.tight_layout()
        plt.savefig('drift_report.png')
        print("Done. Saved drift_report.png")
        
        # Summary
        drift_start_idx = metrics_df[metrics_df['drift_intensity'] > 0].index.min()
        fail_idx = metrics_df[metrics_df['accuracy'] < 0.70].index.min() # Arbitrary failure threshold
        
        print("\n--- Simulation Summary ---")
        print(f"Total Days: {len(metrics_df)}")
        print(f"Drift Started Day: {drift_start_idx}")
        if pd.notna(fail_idx):
            print(f"Model Performance Degraded (Acc < 0.70) at Day: {fail_idx}")
        else:
            print("Model maintained > 0.70 accuracy throughout.")

if __name__ == "__main__":
    pipeline = DriftPipeline()
    pipeline.load_data()
    pipeline.train_baseline()
    pipeline.run_simulation(n_days=100, batch_size=300)
    pipeline.save_outputs()
