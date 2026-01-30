# Model Failure Prediction due to Data Drift

## Overview
This project implements a production-grade MLOps system to predict whether a customer churn model will fail in the next 7 days due to data drift. It uses a two-layer architecture:
1.  **Primary Model**: Customer Churn Prediction (XGBoost)
2.  **Meta Model**: Failure Predictor (Logistic Regression) based on drift metrics.

## Architecture
```mermaid
graph TD
    A[Raw Data] --> B[Data Loader]
    B --> C[Reference Data (Train/Val/Test)]
    C --> D[Train Churn Model]
    D --> E[Churn Model Artifact]
    
    C --> F[Drift Simulator]
    E --> F
    F --> G[Simulated Batches (Drifted)]
    
    G --> H[Drift Detection Engine]
    H --> I[Drift Metrics (PSI, KS, JS)]
    
    G --> J[Ground Truth Eval]
    J --> K[Performance Drop]
    
    I --> L[Failure Label Generator]
    K --> L
    L --> M[Failure Training Data]
    
    M --> N[Train Failure Predictor]
    N --> O[Failure Model Artifact]
```

## Directory Structure
```
ml-model-failure-predictor/
├── data/               # Data storage
├── src/                # Source code
│   ├── data/           # Data loading & simulation
│   ├── features/       # Drift metrics & stats
│   ├── models/         # Model wrappers
│   ├── training/       # Training scripts
│   ├── inference/      # Inference scripts
│   ├── monitoring/     # Drift monitoring
│   └── utils/          # Config & logging
├── notebooks/          # Analysis notebooks (1-8)
├── configs/            # Configuration files
├── experiments/        # Model artifacts & results
└── run_pipeline.py     # CLI entry point
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Run the full pipeline:
```bash
python run_pipeline.py --step all
```

Run specific steps:
```bash
python run_pipeline.py --step data
python run_pipeline.py --step train_churn
python run_pipeline.py --step train_failure
```

## Configuration
- **Model Params**: `configs/model.yaml`
- **Drift Thresholds**: `configs/drift.yaml`

## Notebooks
- `01_data_exploration.ipynb`: Dataset overview.
- `02_churn_model_eda.ipynb`: Churn model analysis.
- `03_drift_simulation_analysis.ipynb`: Visualizing drift types.
- `04_drift_metrics_experiments.ipynb`: PSI/KS/JS sensitivity.
- `05_failure_label_analysis.ipynb`: Labeling logic validation.
- `06_failure_model_experiments.ipynb`: Meta-model performance.
- `07_threshold_sensitivity.ipynb`: Threshold tuning.
- `08_end_to_end_pipeline_demo.ipynb`: Final demo.
