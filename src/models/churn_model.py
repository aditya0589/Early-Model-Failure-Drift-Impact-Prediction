import xgboost as xgb
import pandas as pd
import joblib
import os
from typing import Dict, Any
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.utils.logger import setup_logger
from src.utils.config import config

logger = setup_logger("churn_model")

class ChurnModel:
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params if params else config.model_config["model"]["params"]
        self.model = xgb.XGBClassifier(**self.params)
        self.feature_names = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the churn model.
        """
        logger.info("Training churn model...")
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        logger.info("Training complete.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts churn probability.
        """
        if self.feature_names:
            # Ensure columns are in the same order
            X = X[self.feature_names]
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts churn probability (class 1).
        """
        if self.feature_names:
            X = X[self.feature_names]
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluates model performance.
        """
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "auc": roc_auc_score(y, y_prob)
        }
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def save(self, path: str):
        """
        Saves the model artifact.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """
        Loads the model artifact.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
            
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]
        logger.info(f"Model loaded from {path}")
