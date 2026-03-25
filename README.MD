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

```
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
```

---

# Cost Sensitive Decision Logic

Fraud detection is not only about prediction accuracy.

```
Expected Loss = (FN × Fraud Amount) + (FP × Review Cost)
```

Where:

| Term | Meaning |
|----|-----|
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
```

Example Response:

```json
{
  "fraud_probability": 0.82,
  "decision": "fraud",
  "threshold": 0.35
}
```

---

# Model Evaluation

Example metrics:

| Metric | Score |
| --------- | ----- |
| Precision | 0.91 |
| Recall | 0.83 |
| F1 Score | 0.87 |
| ROC AUC | 0.96 |

In fraud detection systems, recall is usually prioritized to reduce missed fraudulent transactions.

---

# Tech Stack

Core technologies used:

- Python
- XGBoost
- Scikit-learn
- FastAPI
- pyenv
- Docker (planned)
- AWS EC2 deployment (planned)

---

# Project Structure

```
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
```

---

# Setup

Install Python with pyenv:

```bash
pyenv install 3.11.8
pyenv virtualenv 3.11.8 fraud-ai-env
pyenv activate fraud-ai-env
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Training the Model:

```bash
python src/train_model.py
```

Outputs:
- trained model
- evaluation metrics
- saved artifacts

Running the API:

```bash
uvicorn api.main:app --reload
```

Open API documentation: http://localhost:8000/docs

---

# Version Roadmap

The repository evolves through multiple versions.

**Version 1 — ML Fraud Detection API**
- fraud detection model
- prediction API
- basic evaluation

**Version 2 — Cost Optimization**
- expected financial loss modeling
- threshold optimization
- risk scoring

**Version 3 — Monitoring**
- performance tracking
- drift detection
- fraud statistics dashboard

**Version 4 — Event Driven System**

```
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
```

**Version 5 — Behavioral Fraud + AML**

Advanced detection features:
- transaction velocity checks
- user behavior profiling
- anomaly detection
- AML monitoring rules

---

# Future Work

Potential improvements:
- graph-based fraud detection
- real-time transaction simulator
- fraud investigation dashboard
- model monitoring pipeline

---

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

| Feature | Description |
|------|------|
| Transactions per minute | Number of transactions in short time window |
| Total amount in last hour | Aggregated transaction value |
| Transaction burst count | Multiple transactions in rapid succession |

Velocity features help detect **card testing attacks**.

---

## Behavioral Features

These features capture deviations from a user's normal behavior.

| Feature | Description |
|------|------|
| Average transaction amount | Historical average |
| Standard deviation of spending | Spending variability |
| Typical merchant categories | Usual purchase types |

Fraud often appears as **behavioral anomalies**.

---

## Geographic Features

Location patterns can reveal suspicious activity.

| Feature | Description |
|------|------|
| Country mismatch | Card used in unexpected location |
| Distance between transactions | Travel distance between transactions |
| Unusual region | Location rarely used by the customer |

These features help detect **card cloning and account takeover**.

---

## Time-based Features

Fraud often happens at unusual times.

| Feature | Description |
|------|------|
| Night transactions | Late-night activity |
| Weekend activity | Unusual weekend transactions |
| Time since last transaction | Transaction interval |

---

## Feature Engineering Pipeline (Future Version)

Future versions of the project will introduce a dedicated feature engineering pipeline.

```
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
```

---

## Why Feature Engineering Matters

Fraud detection performance often depends more on **feature quality** than model complexity.

A simple model with well-designed features can outperform a complex model trained on raw data.

This project aims to progressively incorporate more sophisticated feature engineering strategies in future versions.

---

# Feature Store

## PySpark + NVIDIA RAPIDS GPU Accelerated

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
  -v /home/ayah/opt:/opt2 \
  spark-gpu-ready
```

> **Note:** Mount `/home/ayah/opt` to `/opt2` inside the container so Spark installation persists across container restarts.

---

## ⚙️ Chapter 1: Installing Spark 3.5.0 Standalone

### 1.1 Download & Extract

Install to `/opt2` (persistent mount) instead of `/opt` so it survives container restarts:

```bash
cd /opt2
wget https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
tar -xzf spark-3.5.0-bin-hadoop3.tgz
rm spark-3.5.0-bin-hadoop3.tgz
mv spark-3.5.0-bin-hadoop3 spark

# Symlink to /opt
ln -s /opt2/spark /opt/spark
```

### 1.2 Install Java 17

```bash
apt update && apt install -y openjdk-17-jdk wget
```

### 1.3 Set Environment Variables

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

### 1.4 Verify Installation

```bash
spark-submit --version
# Output: version 3.5.0, Scala 2.12.18, OpenJDK 64-Bit Server VM, 17.0.18
```

---

## ⚡ Chapter 2: Setting Up NVIDIA RAPIDS Accelerator

### 2.1 Download RAPIDS JAR

```bash
mkdir -p /opt2/spark/rapids

wget https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/24.02.0/rapids-4-spark_2.12-24.02.0.jar \
  -O /opt2/spark/rapids/rapids-4-spark.jar

ls -lh /opt2/spark/rapids/
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
cat > /opt2/start-pyspark-rapids.sh << 'EOF'
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

chmod +x /opt2/start-pyspark-rapids.sh
ln -s /opt2/start-pyspark-rapids.sh ~/start-pyspark-rapids.sh
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

**Class distribution:**

| is_fraud | Count | Percentage |
|----------|-------|------------|
| 0 (legitimate) | 1,289,169 | 99.42% |
| 1 (fraud) | 7,506 | 0.58% |

> Dataset is highly imbalanced — only **0.58% fraud**. This requires special handling during model training such as SMOTE or class weighting.

### 3.2 Read CSV

> **Note:** CSV/Text format is not GPU-accelerated (RAPIDS limitation). Disable RAPIDS when reading CSV, then re-enable for downstream GPU-accelerated operations.

```python
# Disable Rapids for CSV read
spark.conf.set("spark.rapids.sql.enabled", "false")

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
# Re-enable Rapids for GPU-accelerated operations
spark.conf.set("spark.rapids.sql.enabled", "true")

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
# |       1|   7506|
# |       0|1289169|
# +--------+-------+

# Filter fraud transactions only
df_fraud = df_parquet.filter(df_parquet.is_fraud == 1)
print(f"Total fraud transactions: {df_fraud.count()}")
```

---

## 📁 Project Structure

```
/opt2/spark/                                                  # Spark 3.5.0 (persistent)
/opt2/spark/rapids/rapids-4-spark.jar                         # RAPIDS JAR (persistent)
/opt2/start-pyspark-rapids.sh                                 # PySpark startup script (persistent)
/opt/spark/                                                   # Symlink to /opt2/spark
/opt/spark/examples/src/main/scripts/getGpusResources.sh      # GPU discovery script
/data/fraudTrain.csv                                          # Raw training dataset
/data/fraudTest.csv                                           # Raw test dataset
/data/fraud_parquet/                                          # Feature Store (Parquet)
```

---

## 🔑 Notes

1. **PySpark pip package is not compatible with RAPIDS** — Spark standalone must be installed instead
2. **JAVA_HOME must be set manually** to Java 17 inside the container
3. **SPARK_HOME must point to the standalone Spark directory**, not the pip package path
4. **Install Spark to `/opt2` (persistent mount)** — survives container restarts without re-downloading
5. **GPU resource framework causes task hang in local mode** — `executor.resource.gpu` and `task.resource.gpu` should be omitted; let RAPIDS detect the GPU via CUDA directly
6. **CSV format is not GPU-accelerated** — disable RAPIDS when reading CSV, convert to Parquet for full GPU acceleration
7. **Driver hostname must be bound to `127.0.0.1`** inside the container to avoid RPC endpoint resolution errors

---

*Spark 3.5.0 + RAPIDS 24.02.0 + CUDA 13.0*

---

# License

MIT License

---

# Chapter 4: Feature Engineering

## 4.1 Velocity Features (Window Function)

### Hypothesis

> **Fraudulent transactions tend to occur in rapid succession within a short time window.**
> A fraudster who obtains card credentials will typically attempt multiple transactions as quickly as possible before the card is blocked.
> We hypothesize that the **transaction count within a 15-minute window** per card will be significantly higher for fraud cases compared to legitimate ones.

To test this hypothesis, we use a **Spark Window Function** with a 15-minute time-based rolling window partitioned per credit card number:

```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Cast timestamp to unix seconds for range-based windowing
df = df.withColumn("trans_timestamp", F.unix_timestamp("trans_date_trans_time", "yyyy-MM-dd HH:mm:ss"))

# 15-minute rolling window per card
window_15min = Window \
    .partitionBy("cc_num") \
    .orderBy("trans_timestamp") \
    .rangeBetween(-15 * 60, 0)  # 900 seconds lookback

# Transaction count in last 15 minutes
df = df.withColumn("tx_count_15min", F.count("trans_num").over(window_15min))

# Total amount in last 15 minutes
df = df.withColumn("tx_amount_15min", F.sum("amt").over(window_15min))
```

### Hypothesis Result

```python
df.groupBy("is_fraud") \
  .agg(
    F.round(F.avg("tx_count_15min"), 4).alias("avg_tx_count_15min"),
    F.round(F.avg("tx_amount_15min"), 2).alias("avg_tx_amount_15min"),
    F.round(F.max("tx_count_15min"), 0).alias("max_tx_count_15min"),
    F.count("*").alias("total_rows")
  ) \
  .orderBy("is_fraud") \
  .show()
```

| Metric | Legitimate | Fraud |
|--------|-----------|-------|
| avg_tx_count_15min | 1.047 | 1.195 |
| avg_tx_amount_15min | $70.74 | $650.40 |
| max_tx_count_15min | 5 | 5 |

> **Hypothesis partially rejected:** The velocity count difference is minimal (1.047 vs 1.195), which means transaction frequency alone is not a strong fraud indicator in this dataset.
> However, an unexpected and much stronger signal emerged — **the total transaction amount within 15 minutes is 9x higher for fraud** ($650 vs $70).
> This suggests that fraudsters in this dataset tend to make **fewer but much larger transactions** rather than many rapid small ones.

---

## 4.2 Category Risk Encoding

Compute fraud rate per merchant category and use it as a feature:

```python
category_stats = df.groupBy("category") \
    .agg(
        F.count("*").alias("total_tx"),
        F.sum(F.col("is_fraud").cast("int")).alias("fraud_tx")
    ) \
    .withColumn("fraud_rate", F.round(F.col("fraud_tx") / F.col("total_tx"), 4)) \
    .orderBy(F.desc("fraud_rate"))

# Join back to main dataframe
df = df.join(
    category_stats.select("category", "fraud_rate")
        .withColumnRenamed("fraud_rate", "category_fraud_rate"),
    on="category",
    how="left"
)
```

**Fraud rate by category:**

| Category | Total TX | Fraud TX | Fraud Rate |
|----------|----------|----------|------------|
| shopping_net | 97,543 | 1,713 | 1.76% |
| misc_net | 63,287 | 915 | 1.45% |
| grocery_pos | 123,638 | 1,743 | 1.41% |
| shopping_pos | 116,672 | 843 | 0.72% |
| gas_transport | 131,659 | 618 | 0.47% |
| health_fitness | 85,879 | 133 | 0.15% |

**Key insights:**
- `shopping_net` & `misc_net` → **online fraud**, high amount ($999 & $797 avg)
- `grocery_pos` & `gas_transport` → **card testing attack** pattern (high volume, low amount)
- `shopping_pos` → **physical store fraud**, high amount ($876 avg)

---

## 4.3 Time-based Features

Fraud often occurs at unusual hours or days:

```python
from pyspark.sql.functions import hour, dayofweek

df = df \
    .withColumn("tx_hour", hour("trans_date_trans_time")) \
    .withColumn("tx_dayofweek", dayofweek("trans_date_trans_time")) \
    .withColumn("is_night", (F.col("tx_hour") < 6).cast("int")) \
    .withColumn("is_weekend", (F.col("tx_dayofweek").isin([1, 7])).cast("int")) \
    .withColumn("amt_log", F.log1p("amt"))  # Log transform for skewed distribution
```

---

## 4.4 Final Feature Store Schema

| Feature | Type | Description |
|---------|------|-------------|
| `amt` | double | Transaction amount |
| `amt_log` | double | Log-transformed amount |
| `tx_count_15min` | long | Transaction count in last 15 min |
| `tx_amount_15min` | double | Total amount in last 15 min |
| `category_fraud_rate` | double | Historical fraud rate of merchant category |
| `tx_hour` | int | Hour of transaction (0-23) |
| `tx_dayofweek` | int | Day of week (1=Sunday, 7=Saturday) |
| `is_night` | int | 1 if transaction between 00:00-06:00 |
| `is_weekend` | int | 1 if Saturday or Sunday |
| `is_fraud` | int | Target label |

```python
# Save final Feature Store
df.write \
    .mode("overwrite") \
    .parquet("/data/fraud_feature_store")

# Verify
df_fs = spark.read.parquet("/data/fraud_feature_store")
print(f"Total rows: {df_fs.count()}")
print(f"Total columns: {len(df_fs.columns)}")
```

**Output — all columns in Feature Store:**

```
Total rows: 1,296,675
Total columns: 32

['category', 'row_index', 'trans_date_trans_time', 'cc_num', 'merchant',
 'amt', 'first', 'last', 'gender', 'street', 'city', 'state', 'zip',
 'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num', 'unix_time',
 'merch_lat', 'merch_long', 'is_fraud', 'trans_timestamp',
 'tx_count_15min', 'tx_amount_15min', 'category_fraud_rate',
 'tx_hour', 'tx_dayofweek', 'is_night', 'is_weekend', 'amt_log']
```

**Output — final class distribution after feature engineering:**

```python
df_fs.groupBy("is_fraud").count().show()
```

```
+--------+-------+
|is_fraud|  count|
+--------+-------+
|       1|   7506|
|       0|1289169|
+--------+-------+
```

> Dataset remains highly imbalanced at **0.58% fraud rate** — this must be addressed during model training using techniques such as SMOTE, class weighting, or threshold optimization.

---

*Next: Chapter 5 — Model Training (XGBoost + Spark MLlib)*
