# Fraud Detection AI

![Python](https://img.shields.io/badge/python-3.11-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![API](https://img.shields.io/badge/API-FastAPI-green)
![Status](https://img.shields.io/badge/status-in%20development-yellow)

A fintech-oriented fraud detection system built using machine learning.

This project demonstrates how fraud detection systems evolve from a simple ML model into a scalable production architecture with monitoring, cost optimization, and event-driven processing.

---

# Problem Context

Fraud detection is a critical challenge in financial systems.

Typical characteristics:

| Property | Description |
|------|------|
| Data imbalance | Fraud cases are extremely rare |
| High cost | Missing fraud transactions causes financial loss |
| Real-time requirement | Decisions must be made within milliseconds |
| False positive risk | Blocking legitimate transactions harms customer experience |

The goal is not only to build a model, but to design a **decision system that balances fraud prevention and customer friction**.

---

# System Architecture (Version 1)

Future versions introduce streaming, monitoring, and AML-style rule engines.

---

# Fraud Detection Workflow
Transaction arrives
│
▼
Feature extraction
│
▼
Fraud probability prediction
│
▼
Threshold decision
│
▼
Fraud alert or approval


---

# Cost Sensitive Decision Logic

Fraud detection is not only about prediction accuracy.

Expected Loss =
(FN × Fraud Amount) + (FP × Review Cost)

Where:

| Term | Meaning |
|----|----|
| FN | Missed fraud transactions |
| FP | Legitimate transactions flagged as fraud |
| Review Cost | Operational cost of manual review |

This motivates **threshold optimization instead of fixed probability thresholds**.

---

# Example API Request

POST `/predict`

```json
{
 "amount": 150.0,
 "time": 12000,
 "features": [0.21, -1.32, 0.78, -0.11]
}


Example Response:

{
 "fraud_probability": 0.82,
 "decision": "fraud",
 "threshold": 0.35
}

Model Evaluation

Example metrics:
| Metric    | Score |
| --------- | ----- |
| Precision | 0.91  |
| Recall    | 0.83  |
| F1 Score  | 0.87  |
| ROC AUC   | 0.96  |

In fraud detection systems, recall is usually prioritized to reduce missed fraudulent transactions.

Tech Stack

Core technologies used:

Python

XGBoost

Scikit-learn

FastAPI

pyenv

Docker (planned)

AWS EC2 deployment (planned)

Project Structure
fraud_detection_ai/

data/            # datasets (ignored)
models/          # trained models

src/
    train_model.py
    predict.py

api/
    main.py

requirements.txt
.gitignore
README.md

Setup

Install Python with pyenv:

pyenv install 3.11.8
pyenv virtualenv 3.11.8 fraud-ai-env
pyenv activate fraud-ai-env

Install dependencies:

pip install -r requirements.txt
Training the Model
python src/train_model.py

Outputs:

trained model

evaluation metrics

saved artifacts

Running the API
uvicorn api.main:app --reload

Open API documentation:

http://localhost:8000/docs
Version Roadmap

The repository evolves through multiple versions.

Version 1 — ML Fraud Detection API

fraud detection model

prediction API

basic evaluation

Version 2 — Cost Optimization

expected financial loss modeling

threshold optimization

risk scoring

Version 3 — Monitoring

performance tracking

drift detection

fraud statistics dashboard

Version 4 — Event Driven System
Transaction
      │
      ▼
Kafka Stream
      │
      ▼
Fraud Scoring Service
      │
      ▼
Decision Engine
Version 5 — Behavioral Fraud + AML

Advanced detection features:

transaction velocity checks

user behavior profiling

anomaly detection

AML monitoring rules

Future Work

Potential improvements:

graph-based fraud detection

real-time transaction simulator

fraud investigation dashboard

model monitoring pipeline

# Model Alternatives

Although the initial implementation uses **XGBoost**, several other models can also be effective for fraud detection.

Different models may perform better depending on dataset size, feature engineering, and computational constraints.

| Model | Advantages | Considerations |
|------|------|------|
| Logistic Regression | Fast, interpretable baseline | Limited ability to capture complex patterns |
| Random Forest | Robust and handles non-linear relationships | Can be slower for large datasets |
| Gradient Boosting | Strong predictive performance | Higher training cost |
| XGBoost | Highly optimized gradient boosting | Requires hyperparameter tuning |
| LightGBM | Faster training for large datasets | May require careful parameter tuning |
| Neural Networks | Capable of learning complex patterns | Requires larger datasets and tuning |
| Isolation Forest | Useful for anomaly detection | Works better when fraud is very rare |

---

## Model Comparison Strategy

Future versions of the project will evaluate multiple models to compare performance:

| Model | Precision | Recall | F1 Score | ROC AUC |
|------|------|------|------|------|
| Logistic Regression | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD |

The goal is not only to maximize accuracy, but to minimize **expected financial loss** in a real fraud detection scenario.

---

## Feature Engineering for Fraud Detection

Fraud detection systems rely heavily on carefully engineered features that capture transaction behavior patterns.

In real-world financial systems, raw transaction data is often transformed into higher-level signals that help detect suspicious activity.

Below are some common feature categories used in fraud detection models.

---

## Transaction Features

Basic attributes of a transaction.

| Feature | Description |
|------|------|
| Transaction Amount | Value of the transaction |
| Transaction Time | Timestamp of transaction |
| Merchant Category | Type of merchant |
| Payment Method | Card / transfer / wallet |

These features are often used as baseline inputs to the model.

---

## Velocity Features

Velocity features measure **how quickly transactions occur**.

Fraud often involves rapid sequences of transactions.

Examples:

| Feature | Description |
|------|------|
| Transactions per minute | Number of transactions in short time window |
| Total amount in last hour | Aggregated transaction value |
| Transaction burst count | Multiple transactions in rapid succession |

Velocity features help detect **card testing attacks**.

---

## Behavioral Features

These features capture deviations from a user's normal behavior.

Examples:

| Feature | Description |
|------|------|
| Average transaction amount | Historical average |
| Standard deviation of spending | Spending variability |
| Typical merchant categories | Usual purchase types |

Fraud often appears as **behavioral anomalies**.

---

## Geographic Features

Location patterns can reveal suspicious activity.

Examples:

| Feature | Description |
|------|------|
| Country mismatch | Card used in unexpected location |
| Distance between transactions | Travel distance between transactions |
| Unusual region | Location rarely used by the customer |

These features help detect **card cloning and account takeover**.

---

## Time-based Features

Fraud often happens at unusual times.

Examples:

| Feature | Description |
|------|------|
| Night transactions | Late-night activity |
| Weekend activity | Unusual weekend transactions |
| Time since last transaction | Transaction interval |

---

## Feature Engineering Pipeline (Future Version)

Future versions of the project will introduce a dedicated feature engineering pipeline.

Example architecture:

Transaction
│
▼
Raw Transaction Data
│
▼
Feature Engineering Layer
│
▼
Feature Vector
│
▼
Fraud Detection Model


---

## Why Feature Engineering Matters

Fraud detection performance often depends more on **feature quality** than model complexity.

A simple model with well-designed features can outperform a complex model trained on raw data.

This project aims to progressively incorporate more sophisticated feature engineering strategies in future versions.

License

MIT License