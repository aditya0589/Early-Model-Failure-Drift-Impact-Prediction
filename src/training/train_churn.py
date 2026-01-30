import pandas as pd
import os
from src.models.churn_model import ChurnModel
from src.utils.logger import setup_logger
from src.utils.config import config

logger = setup_logger("train_churn")

def main():
    # Load data
    train_path = "data/reference/train.csv"
    val_path = "data/reference/validation.csv"
    
    if not os.path.exists(train_path):
        logger.error("Training data not found. Run src/data/load_data.py first.")
        return

    logger.info("Loading training data...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    X_train = train_df.drop(columns=['churn'])
    y_train = train_df['churn']
    
    X_val = val_df.drop(columns=['churn'])
    y_val = val_df['churn']
    
    model = ChurnModel()
    model.train(X_train, y_train)
    
    logger.info("Evaluating on validation set...")
    metrics = model.evaluate(X_val, y_val)
    
    model_path = "experiments/results/churn_model.joblib"
    model.save(model_path)
    
    pd.DataFrame([metrics]).to_csv("experiments/results/churn_model_metrics.csv", index=False)

if __name__ == "__main__":
    main()
