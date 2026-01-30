import pandas as pd
import joblib
import os
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.utils.logger import setup_logger
from src.utils.config import config

logger = setup_logger("failure_model")

class FailurePredictor:
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params if params else config.model_config.get("failure_predictor", {}).get("params", {})
        # Default to Logistic Regression if not specified or empty
        if not self.params:
             self.params = {"C": 1.0, "solver": "lbfgs", "random_state": 42}
             
        self.model = LogisticRegression(**self.params)
        self.feature_names = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the failure prediction model.
        """
        logger.info("Training failure prediction model...")
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        logger.info("Training complete.")

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts failure probability.
        """
        if self.feature_names:
            # Handle missing columns if any (though pipeline should ensure consistency)
            missing_cols = set(self.feature_names) - set(X.columns)
            for c in missing_cols:
                X[c] = 0
            X = X[self.feature_names]
            
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluates model performance.
        """
        y_pred = self.model.predict(X)
        y_prob = self.predict_proba(X)
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "auc": roc_auc_score(y, y_prob) if len(set(y)) > 1 else 0.0
        }
        logger.info(f"Failure Model Metrics: {metrics}")
        return metrics

    def save(self, path: str):
        """
        Saves the model artifact.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)
        logger.info(f"Failure model saved to {path}")

    def load(self, path: str):
        """
        Loads the model artifact.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
            
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]
        logger.info(f"Failure model loaded from {path}")
