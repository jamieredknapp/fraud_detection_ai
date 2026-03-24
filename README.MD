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


Feature Store

# PySpark + NVIDIA RAPIDS GPU Accelerated 

---

## 🖥️ Environment

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 (Docker Container) |
| GPU | 2x NVIDIA GeForce RTX 5060 Ti (16GB each) |
| CUDA | 13.0 |
| Java | OpenJDK 17 |
| Spark | 3.5.0 (Standalone) |
| RAPIDS Accelerator | 24.02.0 |
| Python | 3.10.12 |

---

## 🐳 Docker Setup

Run the container with GPU passthrough and dataset volume mount:

```bash
docker run --gpus all -it --rm \
  -p 4040:4040 \
  -p 8080:8080 \
  -v /home/ayah/datasets/kartik:/data \
  spark-gpu-ready
```

---

## ⚙️ Chapter 1: Installing Spark 3.5.0 Standalone

### 1.1 Download & Extract

```bash
cd /opt
wget https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
tar -xzf spark-3.5.0-bin-hadoop3.tgz
rm spark-3.5.0-bin-hadoop3.tgz
mv spark-3.5.0-bin-hadoop3 spark
```

### 1.2 Set Environment Variables

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export SPARK_HOME=/opt/spark
export PATH=$JAVA_HOME/bin:$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
export PYSPARK_PYTHON=python3

# Save permanently to .bashrc
cat >> ~/.bashrc << 'EOF'

# Java & Spark
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export SPARK_HOME=/opt/spark
export PATH=$JAVA_HOME/bin:$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
export PYSPARK_PYTHON=python3
EOF

source ~/.bashrc
```

### 1.3 Verify Installation

```bash
spark-submit --version
# Output: version 3.5.0, Scala 2.12.18, OpenJDK 64-Bit Server VM, 17.0.18
```

---

## ⚡ Chapter 2: Setting Up NVIDIA RAPIDS Accelerator

### 2.1 Download RAPIDS JAR

```bash
mkdir -p /opt/spark/rapids

wget https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/24.02.0/rapids-4-spark_2.12-24.02.0.jar \
  -O /opt/spark/rapids/rapids-4-spark.jar

ls -lh /opt/spark/rapids/
```

### 2.2 Verify GPU Discovery Script

Spark 3.5.0 standalone ships with a built-in GPU discovery script:

```bash
# Check the script exists
ls -la /opt/spark/examples/src/main/scripts/getGpusResources.sh

# Run it manually to verify output
bash /opt/spark/examples/src/main/scripts/getGpusResources.sh
# Expected output: {"name": "gpu", "addresses":["0","1"]}
```

### 2.3 Create PySpark + RAPIDS Startup Script

```bash
cat > ~/start-pyspark-rapids.sh << 'EOF'
#!/bin/bash
pyspark \
  --master local[2] \
  --jars /opt/spark/rapids/rapids-4-spark.jar \
  --conf spark.plugins=com.nvidia.spark.SQLPlugin \
  --conf spark.rapids.sql.enabled=true \
  --conf spark.rapids.sql.exec.CollectLimitExec=true \
  --conf spark.driver.host=127.0.0.1 \
  --conf spark.driver.bindAddress=127.0.0.1
EOF

chmod +x ~/start-pyspark-rapids.sh
```

### 2.4 Key Findings: RAPIDS Configuration in Local Mode

> **Important:** In Spark local mode, setting `spark.executor.resource.gpu` and `spark.task.resource.gpu` causes tasks to hang in a pending state because the GPU never gets assigned to the executor. The fix is to let RAPIDS detect the GPU directly via CUDA without the Spark resource framework.

| Config | Value |
|--------|-------|
| `spark.plugins` | `com.nvidia.spark.SQLPlugin` |
| `spark.rapids.sql.enabled` | `true` |
| `spark.rapids.sql.exec.CollectLimitExec` | `true` |
| `spark.driver.host` | `127.0.0.1` |
| `spark.driver.bindAddress` | `127.0.0.1` |

### 2.5 Start PySpark + RAPIDS

```bash
~/start-pyspark-rapids.sh
```

### 2.6 Verify GPU Detection

```python
print(spark.sparkContext.resources)
# Output: {'gpu': <ResourceInformation name=gpu, addresses=['0', '1']>}
```

---

## 📦 Chapter 3: Creating Feature Store

### 3.1 Dataset

| Info | Detail |
|------|--------|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) |
| Training file | `fraudTrain.csv` |
| Test file | `fraudTest.csv` |
| Training size | 335 MB |
| Training rows | 1,296,676 |
| Columns | 23 |

**Column overview:**

| Column | Description |
|--------|-------------|
| `trans_date_trans_time` | Transaction timestamp |
| `cc_num` | Credit card number |
| `merchant` | Merchant name |
| `category` | Merchant category |
| `amt` | Transaction amount |
| `first`, `last` | Cardholder name |
| `gender` | Cardholder gender |
| `street`, `city`, `state`, `zip` | Cardholder address |
| `lat`, `long` | Cardholder coordinates |
| `city_pop` | City population |
| `job` | Cardholder occupation |
| `dob` | Date of birth |
| `trans_num` | Transaction ID |
| `unix_time` | Unix timestamp |
| `merch_lat`, `merch_long` | Merchant coordinates |
| `is_fraud` | Target label (0 = legitimate, 1 = fraud) |

### 3.2 Read CSV

> **Note:** CSV/Text format is not GPU-accelerated (RAPIDS limitation). The GPU is utilized during downstream processing and transformation operations after the data is loaded.

```python
df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv("/data/fraudTrain.csv")

# Rename unnamed index column
df = df.withColumnRenamed("_c0", "row_index")

print(f"Total rows: {df.count()}")
print(f"Total columns: {len(df.columns)}")
df.printSchema()
```

### 3.3 Convert to Parquet (Feature Store)

Parquet is chosen as the Feature Store format because:
- ✅ **Columnar format** — querying specific columns is significantly faster
- ✅ **GPU-accelerated** — full RAPIDS support for Parquet reads and operations
- ✅ **Schema enforcement** — data types are stored in metadata
- ✅ **Compression** — much smaller file size compared to CSV

```python
df.write \
    .mode("overwrite") \
    .parquet("/data/fraud_parquet")

print("Feature Store created successfully!")
```

### 3.4 Verify Feature Store

```python
df_parquet = spark.read.parquet("/data/fraud_parquet")

df_parquet.printSchema()
print(f"Total rows: {df_parquet.count()}")
# Output: Total rows: 1,296,675

df_parquet.limit(5).show()
```

### 3.5 Query Feature Store with GPU Acceleration

```python
# GPU-accelerated groupBy & aggregation
df_parquet.groupBy("is_fraud").count().show()
# +--------+-------+
# |is_fraud|  count|
# +--------+-------+
# |       0|1289169|
# |       1|   7506|
# +--------+-------+

# Filter fraud transactions only
df_fraud = df_parquet.filter(df_parquet.is_fraud == 1)
print(f"Total fraud transactions: {df_fraud.count()}")
```

---

## 📁 Project Structure

```
/opt/spark/                                                   # Spark 3.5.0 standalone
/opt/spark/rapids/rapids-4-spark.jar                          # RAPIDS JAR
/opt/spark/examples/src/main/scripts/getGpusResources.sh      # GPU discovery script
/data/fraudTrain.csv                                          # Raw training dataset
/data/fraudTest.csv                                           # Raw test dataset
/data/fraud_parquet/                                          # Feature Store (Parquet)
~/start-pyspark-rapids.sh                                     # PySpark startup script
```

---

## 🔑 Notes

1. **PySpark pip package is not compatible with RAPIDS** — Spark standalone must be installed instead
2. **JAVA_HOME must be set manually** to Java 17 inside the container
3. **SPARK_HOME must point to the standalone Spark directory**, not the pip package path
4. **GPU resource framework causes task hang in local mode** — `executor.resource.gpu` and `task.resource.gpu` should be omitted; let RAPIDS detect the GPU via CUDA directly
5. **CSV format is not GPU-accelerated** — convert to Parquet for full GPU acceleration
6. **Driver hostname must be bound to `127.0.0.1`** inside the container to avoid RPC endpoint resolution errors

---

*Generated on 2026-03-24 | Spark 3.5.0 + RAPIDS 24.02.0 + CUDA 13.0*



License

MIT License