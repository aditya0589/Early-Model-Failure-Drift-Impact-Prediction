import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os
from src.utils.logger import setup_logger

logger = setup_logger("data_loader")

def generate_synthetic_churn_data(n_samples=10000, n_features=20, random_state=42):
    """
    Generates a synthetic churn dataset if raw data is missing.
    """
    logger.info("Generating synthetic churn dataset...")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3], # Class imbalance typical in churn
        random_state=random_state
    )
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['churn'] = y
    
    df['contract_type'] = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples)
    df['payment_method'] = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples)
    
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing data...")
    df = df.rename(columns={'Churn': 'churn'})
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    if 'churn' in df.columns:
        df['churn'] = df['churn'].astype(str).str.strip()
        mapping = {'Yes': 1, 'No': 0}
        df['churn'] = df['churn'].map(mapping)
        
        if df['churn'].isnull().any():
            unique_unmapped = df[df['churn'].isnull()]['churn'].unique() 
            logger.warning(f"NaNs found after churn encoding. Check for values other than 'Yes'/'No'. Filling with 0 as fallback.")
            df['churn'] = df['churn'].fillna(0)
        
        df['churn'] = df['churn'].astype(int)
    
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0) # or median
        
    cat_cols = df.select_dtypes(include=['object']).columns
    cat_cols = [c for c in cat_cols if c != 'churn']
    
    if len(cat_cols) > 0:
        logger.info(f"Encoding categorical columns: {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        
    return df

def load_data(data_path: str = "data/raw/churn_data.csv"):
    if not os.path.exists(data_path):
        logger.warning(f"Data file not found at {data_path}. Generating synthetic data.")
        df = generate_synthetic_churn_data()
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.info(f"Saved synthetic data to {data_path}")
    else:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        df = preprocess_data(df)
        
    return df

def split_data(df: pd.DataFrame, target_col: str = "churn", test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42):
    logger.info("Splitting data into train/val/test...")
    
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[target_col], random_state=random_state)
    
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=relative_val_size, stratify=train_val[target_col], random_state=random_state)
    
    logger.info(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")
    return train, val, test

def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, output_dir: str = "data/reference"):
    """
    Saves the splits to the reference directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    logger.info(f"Saved splits to {output_dir}")

def main():
    df = load_data()
    train, val, test = split_data(df)
    save_splits(train, val, test)

if __name__ == "__main__":
    main()
