import argparse
import sys
from src.utils.logger import setup_logger

logger = setup_logger("pipeline_runner")

def run_data_loading():
    logger.info("Step 1: Data Loading")
    from src.data.load_data import main
    main()

def run_churn_training():
    logger.info("Step 2: Churn Model Training")
    from src.training.train_churn import main
    main()

def run_failure_training():
    logger.info("Step 3: Failure Predictor Training")
    from src.training.train_failure_predictor import main
    main()

def run_demo():
    logger.info("Step 4: Running Demo Inference")
    from src.inference.predict_failure import predict_failure_probability, load_reference_data
    from src.models.failure_model import FailurePredictor
    from src.data.load_data import load_data
    
    try:
        ref_df = load_reference_data()
        failure_model = FailurePredictor()
        failure_model.load("experiments/results/failure_model.joblib")
        full_data = load_data()
        
        batch = full_data.sample(1000)
        prob, _ = predict_failure_probability(batch, ref_df, failure_model)
        logger.info(f"Demo Inference - Failure Probability: {prob:.4f}")
    except Exception as e:
        logger.error(f"Demo failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Model Failure Prediction Pipeline")
    parser.add_argument("--step", type=str, choices=["all", "data", "train_churn", "train_failure", "demo"], default="all", help="Pipeline step to run")
    
    args = parser.parse_args()
    
    if args.step == "all" or args.step == "data":
        run_data_loading()
        
    if args.step == "all" or args.step == "train_churn":
        run_churn_training()
        
    if args.step == "all" or args.step == "train_failure":
        run_failure_training()
        
    if args.step == "all" or args.step == "demo":
        run_demo()

if __name__ == "__main__":
    main()
