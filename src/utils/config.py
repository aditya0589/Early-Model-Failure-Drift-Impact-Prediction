import yaml
import os
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class ModelConfig:
    type: str
    params: Dict[str, Any]

@dataclass
class FailurePredictorConfig:
    type: str
    params: Dict[str, Any]
    lookahead_days: int

@dataclass
class DriftConfig:
    psi_threshold: float
    ks_p_value: float
    js_divergence: float
    drift_types: List[str]
    drift_intensity_range: List[float]
    drift_score_threshold: float
    performance_drop_threshold: float
    simulation: Dict[str, Any]

class Config:
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        self.model_config = self._load_model_config()
        self.drift_config = self._load_drift_config()

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        path = os.path.join(self.config_dir, filename)
        if not os.path.exists(path):
            # Fallback for running from different directories
            path = os.path.join("..", "configs", filename)
            if not os.path.exists(path):
                 # Fallback for running from src
                path = os.path.join("..", "..", "configs", filename)
        
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _load_model_config(self) -> Dict[str, Any]:
        data = self._load_yaml("model.yaml")
        return data

    def _load_drift_config(self) -> DriftConfig:
        data = self._load_yaml("drift.yaml")
        return DriftConfig(
            psi_threshold=data["drift_thresholds"]["psi"],
            ks_p_value=data["drift_thresholds"]["ks_p_value"],
            js_divergence=data["drift_thresholds"]["js_divergence"],
            drift_types=data["simulation"]["drift_types"],
            drift_intensity_range=data["simulation"]["drift_intensity_range"],
            drift_score_threshold=data["simulation"]["failure_definition"]["drift_score_threshold"],
            performance_drop_threshold=data["simulation"]["failure_definition"]["performance_drop_threshold"],
            simulation=data["simulation"]
        )

# Global config instance
try:
    config = Config()
except Exception as e:
    print(f"Warning: Could not load config: {e}")
    config = None
