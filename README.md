# Datathon 2024: Financial FraudShield

## Background
The rise in online transactions, mobile banking, and digital payments has further complicated the landscape. This hackathon aims to develop innovative ML-driven solutions to detect and mitigate fraudulent activities in financial transactions effectively.

## Problem Statement
How can AI and machine learning aid in minimising financial fraud by detecting and preventing these activities in real-time, while adapting to evolving fraud tactics?

## Objective
To design and implement a real-time fraud detection solution utilising machine learning algorithms to identify and prevent fraudulent transactions. This solution targets financial transactions including digital payments, wallets, and e-commerce platforms. The system efficiently processes large volumes of transactional data, detects patterns indicative of fraud, and adapts to evolving fraud techniques over time.

---

## Project Structure

```
Datathon/
├── Dataset/                        # Raw CSV files
│   ├── fraudTrain.csv
│   └── fraudTest.csv
├── Cleaned Data/                   # Pre-processed data
│   └── fraudClean.csv
├── notebooks/                      # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_machine_learning.ipynb
├── src/                            # Reusable Python package
│   ├── __init__.py
│   ├── data_loader.py              # Load datasets
│   ├── feature_engineering.py     # Age, distance, frequency, job-sector features
│   ├── preprocessing.py           # Encoding, SMOTE, scaling, anomaly detection
│   ├── models.py                  # Train & evaluate ML models
│   └── visualization.py           # Reusable plotting helpers
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebooks

Open each notebook from the `notebooks/` directory in Jupyter:

| Notebook | Description |
|---|---|
| `01_exploratory_data_analysis.ipynb` | Feature engineering & EDA visualisations |
| `02_machine_learning.ipynb` | Model training, evaluation & comparison |

### 3. Use the `src` package directly

```python
from src.data_loader import load_train_data
from src.feature_engineering import engineer_features
from src.preprocessing import prepare_model_data
from src.models import train_random_forest, evaluate_model

data = load_train_data("Dataset/fraudTrain.csv")
data = engineer_features(data)
```

---

## Models

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Interpretable tree-based model |
| MLP Classifier | 3-layer neural network (64-64-64, tanh) |
| Random Forest | Ensemble of decision trees |
| Voting Ensemble | Soft-voting combination of DT + MLP + RF |

Anomaly scores from **IsolationForest** are appended as an extra feature before training all supervised models.

## Feature Engineering

| Feature | Description |
|---|---|
| `age_at_transaction` | Cardholder's age at the time of the transaction |
| `distance` | Haversine distance (km) between cardholder and merchant |
| `transactions_last_hour` | Rolling count of transactions on the same card in the past hour |
| `transactions_last_day` | Rolling count of transactions on the same card in the past day |
| `job_sector` | High-level industry sector derived from job title |
