import pandas as pd
import numpy as np
from typing import Dict

def calculate_feature_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates statistical summaries for a batch of data.
    """
    stats = {}
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        if col == 'churn': continue
        
        stats[f"{col}_mean"] = df[col].mean()
        stats[f"{col}_std"] = df[col].std()
        stats[f"{col}_min"] = df[col].min()
        stats[f"{col}_max"] = df[col].max()
        stats[f"{col}_q25"] = df[col].quantile(0.25)
        stats[f"{col}_q75"] = df[col].quantile(0.75)
        
    return stats
