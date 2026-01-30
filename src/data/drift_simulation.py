import numpy as np
import pandas as pd
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from src.utils.logger import setup_logger
from src.utils.config import config

logger = setup_logger("drift_simulator")

class DriftSimulator:
    def __init__(self, base_data: pd.DataFrame):
        self.base_data = base_data
        self.n_features = base_data.shape[1] - 1 
        self.feature_names = [c for c in base_data.columns if c != 'churn']
        
    def simulate_batch(self, batch_size: int, drift_type: str = None, drift_intensity: float = 0.0) -> Tuple[pd.DataFrame, Dict]:
        batch = self.base_data.sample(n=batch_size, replace=True).copy()
        
        if drift_type:
            batch = self._apply_drift(batch, drift_type, drift_intensity)
            
        metadata = {
            "batch_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "drift_type": drift_type if drift_type else "none",
            "drift_intensity": drift_intensity
        }
        
        return batch, metadata

    def _apply_drift(self, df: pd.DataFrame, drift_type: str, intensity: float) -> pd.DataFrame:
        if drift_type == "covariate_shift":
            return self._apply_covariate_shift(df, intensity)
        elif drift_type == "label_shift":
            return self._apply_label_shift(df, intensity)
        elif drift_type == "noise_injection":
            return self._apply_noise_injection(df, intensity)
        elif drift_type == "scaling":
            return self._apply_scaling_drift(df, intensity)
        else:
            logger.warning(f"Unknown drift type: {drift_type}. Returning original data.")
            return df

    def _apply_covariate_shift(self, df: pd.DataFrame, intensity: float) -> pd.DataFrame:
        # Shift mean of numerical features
        # Intensity 0.0 -> 0 shift, 1.0 -> 2.0 std shift
        shift_amount = intensity * 2.0
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        # Select random subset of features to shift
        n_shift = max(1, int(len(numerical_cols) * 0.5))
        cols_to_shift = np.random.choice(numerical_cols, n_shift, replace=False)
        
        for col in cols_to_shift:
            if col != 'churn':
                df[col] = df[col] + shift_amount
        return df

    def _apply_label_shift(self, df: pd.DataFrame, intensity: float) -> pd.DataFrame:
        # Change class balance
        # Intensity 0.0 -> No change, 1.0 -> Flip majority/minority or extreme imbalance
        # Here we'll just downsample the majority class based on intensity
        
        majority_class = df['churn'].mode()[0]
        minority_class = 1 - majority_class
        
        majority_df = df[df['churn'] == majority_class]
        minority_df = df[df['churn'] == minority_class]
        
        # Drop some majority class samples
        drop_frac = min(0.9, intensity * 0.8) # Max drop 90%
        majority_df = majority_df.sample(frac=max(0.01, 1-drop_frac))
        
        return pd.concat([majority_df, minority_df]).sample(frac=1).reset_index(drop=True)

    def _apply_noise_injection(self, df: pd.DataFrame, intensity: float) -> pd.DataFrame:
        # Add Gaussian noise
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        noise_level = intensity * 1.0
        
        for col in numerical_cols:
            if col != 'churn':
                noise = np.random.normal(0, noise_level, size=len(df))
                df[col] = df[col] + noise
        return df

    def _apply_scaling_drift(self, df: pd.DataFrame, intensity: float) -> pd.DataFrame:
        # Multiply features by a factor
        factor = 1.0 + (intensity * 2.0) # 1.0 to 3.0
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col != 'churn':
                df[col] = df[col] * factor
        return df

if __name__ == "__main__":
    from src.data.load_data import load_data
    df = load_data()
    sim = DriftSimulator(df)
    batch, meta = sim.simulate_batch(100, "covariate_shift", 0.5)
    print(meta)
    print(batch.head())
