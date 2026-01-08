# Data Preparation Guide: Early Model Failure Prediction

This document explains the logic behind `prepare_drift_dataset.py`, a script designed to transform raw model logs into a structured, tabular dataset ready for Machine Learning.

## 1. Overview
We are building a system to **predict whether an ML model will fail in the near future** (specifically, in the next 7 days).

To do this, we cannot simply use raw prediction logs because they are too granular (one row per transaction). Instead, we aggregate data by **day** and look at trends over time.

### Workflow
```mermaid
flowchart LR
    A[Raw Logs\n(drift_simulation_data.csv)] --> B[Aggregate by Day]
    C[Drift Metrics\n(drift_metrics.csv)] --> D[Merge & Align]
    B --> D
    D --> E[Feature Engineering\n(Lags, Rolling Stats)]
    E --> F[Create Target\n(Failure next 7 days?)]
    F --> G[Tabular Training Data\n(tabular_training_dataset.csv)]
```

---

## 2. Step-by-Step Explanation

### Step 1: Daily Aggregation
**What:** Convert thousands of individual predictions per day into a single summary row for that day.
**Why:** Our goal is to predict "system health" per day, not individual errors. Aggregation reduces noise and dimensionality.

**Key Metrics Computed:**
- **Error Rate:** $$ \frac{\text{Count of Errors}}{\text{Total Predictions}} $$
- **Mean Entropy:** Average uncertainty of the model. High entropy means the model is "confused".
- **Entropy-Error Correlation:**
  - *Goal:* Check if the model is "aware" of its mistakes.
  - *Logic:* Ideally, when the model is wrong (Error=1), Entropy should be high. If Correlation is high, the model knows when it's struggling. If Correlation drops, the model is "confidently wrong" (a dangerous drift signal).

### Step 2: Defining the Target (The "Y")
**What:** We are not predicting if the *current* day is bad. We are predicting if the *future* (next 7 days) will have a failure.
**Definition:** A "Failure" is defined as **Any single day** in the next week where accuracy drops below **85%**.

**Formula:**
Let $A_t$ be the accuracy on day $t$.
The target $Y_t$ for the current day $t$ is:
$$ Y_t = 1 \quad \text{if} \quad \min(A_{t+1}, A_{t+2}, \dots, A_{t+7}) < 0.85 $$
$$ Y_t = 0 \quad \text{otherwise} $$

**Visual Representation:**
```text
Day t (Today)      Day t+1 ... Day t+7 (Future Window)
[Features X_t] --> [Check Accuracy...]
                   [If ANY < 0.85] --> Target = 1 (Failure Imminent)
                   [If ALL >= 0.85] -> Target = 0 (Safe)
```

### Step 3: Feature Engineering (The "X")
**What:** Creating "history-aware" features so the model can see trends.
**Why:** A single day's error rate might be noise. A *rising trend* in error rate is a signal.

We use **Temporal Features** without using complex sequence models (like LSTMs). We simply add columns representing the past.

#### A. Lag Features (Exact Past Values)
"What was the value yesterday?"
- `error_rate_lag_1`: Error rate 1 day ago.
- `error_rate_lag_7`: Error rate 1 week ago.

#### B. Rolling Statistics (Recent Context)
"How has the last week been on average?"
- `error_rate_roll_mean_7`: $$ \frac{1}{7} \sum_{i=0}^{6} \text{error\_rate}_{t-i} $$
- `error_rate_roll_std_7`: Measure of stability. High std dev means performance is fluctuating wildy.

#### C. Trend (Slope)
"Is it getting better or worse?"
We fit a simple linear line through the last 7 days of data.
- **Positive Slope (+):** Metric is increasing (e.g., predicted confidence is rising).
- **Negative Slope (-):** Metric is decreasing (e.g., accuracy is falling).

---

## 3. The Output
The final file `tabular_training_dataset.csv` is a clean matrix suitable for XGBoost, LightGBM, or Logistic Regression.

| current_error_rate | error_lag_1 | error_trend_7 | ... | **failure_in_next_7_days** |
|--------------------|-------------|---------------|-----|----------------------------|
| 0.12               | 0.11        | +0.001        | ... | **0** (Safe)               |
| 0.15               | 0.12        | +0.040        | ... | **1** (Failure Coming)     |

**Impact:**
By training on this data, we can build an **Early Warning System**.
- If the model predicts `1`, we can alert engineers to retrain the model *before* the failure actually hurts business KPIs.
