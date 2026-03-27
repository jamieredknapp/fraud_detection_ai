# Catatan Pengerjaan — Fraud Detection AI
> Dokumentasi lengkap proses pengerjaan, command, output, dan temuan penting.
> Bahasa: Indonesia (catatan internal)

---

## Lingkungan Kerja

| Komponen | Detail |
|----------|--------|
| OS Host | Ubuntu 22.04.5 LTS (Jammy Jellyfish) |
| GPU | 2x NVIDIA GeForce RTX 5060 Ti (16GB each) |
| CUDA | 13.0 (driver) / 13.2 (toolkit) |
| Java | OpenJDK 17 (Amazon Corretto) |
| Python | 3.10.14 via pyenv |
| Environment | `fraud_detection_ai_env` |
| Spark | 3.5.0 Standalone |
| RAPIDS JAR | rapids-4-spark 25.02.0 |
| cuDF | 26.02.01 (CUDA 13) |
| XGBoost | 3.2.0 |

---

## 1. Setup Awal

### 1.1 Aktivasi Environment

```bash
pyenv activate fraud_detection_ai_env
python --version
# Python 3.10.14
```

### 1.2 Install Spark 3.5.0

```bash
cd ~
wget https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
tar -xzf spark-3.5.0-bin-hadoop3.tgz
rm spark-3.5.0-bin-hadoop3.tgz
mv spark-3.5.0-bin-hadoop3 ~/spark
```

Set environment variables:

```bash
cat >> ~/.bashrc << 'EOF'
export JAVA_HOME=/usr/lib/jvm/java-17-amazon-corretto
export SPARK_HOME=$HOME/spark
export PATH=$JAVA_HOME/bin:$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
export PYSPARK_PYTHON=python
EOF
source ~/.bashrc
```

Verifikasi:

```bash
spark-submit --version
# version 3.5.0, Scala 2.12.18, OpenJDK 64-Bit Server VM, 17.0.17
```

### 1.3 Download RAPIDS JAR

```bash
mkdir -p ~/spark/rapids
wget https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/25.02.0/rapids-4-spark_2.12-25.02.0.jar \
  -O ~/spark/rapids/rapids-4-spark.jar
ls -lh ~/spark/rapids/
# -rw-r--r-- 1 ayah ayah 557M Feb 24  2025 rapids-4-spark.jar
```

### 1.4 Install Python Dependencies

```bash
pip install pyspark==3.5.0
pip install xgboost scikit-learn imbalanced-learn numpy pandas
pip install cudf-cu13==26.2.* --extra-index-url https://pypi.nvidia.com
```

Output instalasi cuDF:
```
Successfully installed cudf-cu13-26.2.1 libcudf-cu13-26.2.1
cuda-python-13.2.0 cuda-toolkit-13.2.0 cupy-cuda13x-14.0.1
numba-0.61.2 rmm-cu13-26.2.0 ...
```

Verifikasi semua komponen:

```bash
python -c "import cudf; print('cuDF version:', cudf.__version__)"
# cuDF version: 26.02.01

python -c "
import cudf
df = cudf.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
print(df)
print('cuDF OK!')
"
# Output:
#    a  b
# 0  1  4
# 1  2  5
# 2  3  6
# cuDF OK!

python -c "import pyspark; print('PySpark version:', pyspark.__version__)"
# PySpark version: 3.5.0

python -c "import xgboost; print('XGBoost version:', xgboost.__version__)"
# XGBoost version: 3.2.0
```

### 1.5 Startup Script PySpark + RAPIDS

```bash
cat > ~/start-pyspark-rapids.sh << 'EOF'
#!/bin/bash
pyspark \
  --master local[2] \
  --jars $HOME/spark/rapids/rapids-4-spark.jar \
  --conf spark.plugins=com.nvidia.spark.SQLPlugin \
  --conf spark.rapids.sql.enabled=true \
  --conf spark.rapids.sql.exec.CollectLimitExec=true \
  --conf spark.driver.host=127.0.0.1 \
  --conf spark.driver.bindAddress=127.0.0.1
EOF
chmod +x ~/start-pyspark-rapids.sh
```

> **Catatan penting:** Tidak perlu `spark.executor.resource.gpu` dan `spark.task.resource.gpu` di local mode. RAPIDS langsung detect GPU via CUDA sendiri. Kalau dipasang justru menyebabkan task hang.

Output startup:
```
RAPIDS Accelerator 25.02.0 using cudf 25.02.0 — is enabled
WARN: No precompiled binaries for device major architecture 12.
      This may lead to expensive JIT compile on startup.
      Binaries available for architectures 7, 8, 9.
```

> **Catatan:** RTX 5060 Ti adalah Blackwell (SM 120 = arch 12). RAPIDS 25.02 belum punya precompiled binary untuk arch 12, jadi JIT compile saat startup. Masih jalan, hanya lebih lambat startup.

---

## 2. Feature Store

### 2.1 Baca CSV

```python
spark.conf.set("spark.rapids.sql.enabled", "false")  # CSV tidak support GPU

df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv("/home/ayah/datasets/kartik/fraudTrain.csv")

df = df.withColumnRenamed("_c0", "row_index")
df.limit(5).show()
```

Output warning (normal, bukan error):
```
WARN CSVHeaderChecker: CSV header does not conform to the schema.
Header: , trans_date_trans_time, ...
Schema: _c0, trans_date_trans_time, ...
```

> **Catatan:** Kolom pertama CSV tidak punya nama (index kolom kosong). PySpark otomatis namai `_c0`, kita rename ke `row_index`.

### 2.2 Velocity Features (Window Function 15 Menit)

**Hipotesa awal:** Transaksi fraud cenderung terjadi dalam burst cepat dalam waktu singkat. Kita ukur jumlah dan total amount transaksi dalam window 15 menit per kartu.

```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark.conf.set("spark.rapids.sql.enabled", "true")

df = df.withColumn("trans_timestamp",
    F.unix_timestamp("trans_date_trans_time", "yyyy-MM-dd HH:mm:ss"))

window_15min = Window \
    .partitionBy("cc_num") \
    .orderBy("trans_timestamp") \
    .rangeBetween(-15 * 60, 0)

df = df.withColumn("tx_count_15min", F.count("trans_num").over(window_15min))
df = df.withColumn("tx_amount_15min", F.sum("amt").over(window_15min))
```

Hasil verifikasi hipotesa:

```python
df.groupBy("is_fraud").agg(
    F.round(F.avg("tx_count_15min"), 4).alias("avg_tx_count_15min"),
    F.round(F.avg("tx_amount_15min"), 2).alias("avg_tx_amount_15min"),
    F.round(F.max("tx_count_15min"), 0).alias("max_tx_count_15min"),
    F.count("*").alias("total_rows")
).orderBy("is_fraud").show()
```

```
+--------+------------------+-------------------+------------------+----------+
|is_fraud|avg_tx_count_15min|avg_tx_amount_15min|max_tx_count_15min|total_rows|
+--------+------------------+-------------------+------------------+----------+
|       0|             1.047|              70.74|                 5|   1289169|
|       1|            1.1948|              650.4|                 5|      7506|
+--------+------------------+-------------------+------------------+----------+
```

> **Kesimpulan:** Hipotesa velocity count **ditolak sebagian** — bedanya kecil (1.047 vs 1.195). Tapi justru ditemukan signal yang jauh lebih kuat: **total amount dalam 15 menit 9x lebih besar untuk fraud** ($650 vs $70).

### 2.3 Category Risk Encoding

```python
category_stats = df.groupBy("category") \
    .agg(
        F.count("*").alias("total_tx"),
        F.sum(F.col("is_fraud").cast("int")).alias("fraud_tx")
    ) \
    .withColumn("category_fraud_rate",
        F.round(F.col("fraud_tx") / F.col("total_tx"), 4)) \
    .select("category", "category_fraud_rate")

df = df.join(category_stats, on="category", how="left")
```

Output fraud rate per kategori:
```
+--------------+--------+--------+--------------------+
|      category|total_tx|fraud_tx|  category_fraud_rate|
+--------------+--------+--------+--------------------+
|  shopping_net|   97543|    1713|              0.0176|
|      misc_net|   63287|     915|              0.0145|
|   grocery_pos|  123638|    1743|              0.0141|
|  shopping_pos|  116672|     843|              0.0072|
| gas_transport|  131659|     618|              0.0047|
|      misc_pos|   79655|     250|              0.0031|
...
|health_fitness|   85879|     133|              0.0015|
+--------------+--------+--------+--------------------+
```

> **Insight:**
> - `shopping_net` & `misc_net` → fraud online, amount besar ($999 & $797 avg)
> - `grocery_pos` & `gas_transport` → card testing pattern (banyak transaksi, amount kecil)
> - `shopping_pos` → fraud toko fisik, amount besar ($876 avg)

### 2.4 Time-Based Features

```python
df = df \
    .withColumn("tx_hour", F.hour("trans_date_trans_time")) \
    .withColumn("tx_dayofweek", F.dayofweek("trans_date_trans_time")) \
    .withColumn("is_night", (F.col("tx_hour") < 6).cast("int")) \
    .withColumn("is_weekend", (F.col("tx_dayofweek").isin([1, 7])).cast("int")) \
    .withColumn("amt_log", F.log1p("amt"))
```

### 2.5 Simpan Feature Store

Sebelum save, hapus folder lama dari terminal (bukan dari PySpark):

```bash
sudo rm -rf /home/ayah/datasets/kartik/fraud_feature_store
```

Lalu set timezone ke UTC untuk menghindari warning RAPIDS:

```python
spark.conf.set("spark.sql.session.timeZone", "UTC")

df.write \
    .mode("overwrite") \
    .parquet("/home/ayah/datasets/kartik/fraud_feature_store")

print("Feature Store saved!")
```

> **Catatan:** RAPIDS hanya support timezone UTC. Kalau timezone JVM = Asia/Jakarta, write operation fallback ke CPU (tidak fatal, tetap jalan). Fix dengan set `spark.sql.session.timeZone` ke `UTC`.

Verifikasi:

```python
df_fs = spark.read.parquet("/home/ayah/datasets/kartik/fraud_feature_store")
print(f"Total rows: {df_fs.count()}")       # 1,296,675
print(f"Total columns: {len(df_fs.columns)}")  # 32
df_fs.groupBy("is_fraud").count().orderBy("is_fraud").show()
```

```
+--------+-------+
|is_fraud|  count|
+--------+-------+
|       0|1289169|
|       1|   7506|
+--------+-------+
```

---

## 3. Training Model

### 3.1 Jalankan Training (Tanpa Spark — via PyArrow)

Karena feature store sudah di parquet, baca langsung via PyArrow tanpa perlu Spark session — lebih cepat:

```bash
python src/train_model.py
```

### 3.2 Hasil Training Pertama (Sebelum Fix Leakage)

```
==================================================
  Logistic Regression
==================================================
  Precision : 0.0505  Recall : 0.8128  F1 : 0.0951  ROC AUC : 0.9257

==================================================
  Random Forest
==================================================
  Precision : 0.1951  Recall : 0.8974  F1 : 0.3205  ROC AUC : 0.9899

==================================================
  XGBoost
==================================================
  Precision : 0.7035  Recall : 0.8441  F1 : 0.7674  ROC AUC : 0.9959

=================================================================
  MODEL COMPARISON
=================================================================
  Model                        Precision  Recall      F1   ROC AUC
  XGBoost                         0.7035  0.8441  0.7674    0.9959  ← Best
  Random Forest                   0.1951  0.8974  0.3205    0.9899
  Logistic Regression             0.0505  0.8128  0.0951    0.9257
=================================================================
```

> **⚠️ Red flag:** ROC AUC 0.9959 terlalu tinggi — dicurigai ada data leakage!

---

## 4. Deteksi Data Leakage

### 4.1 Audit Feature

```python
# Cek korelasi dan apakah category_fraud_rate dihitung dari full dataset
corr = pdf["category_fraud_rate"].corr(pdf["is_fraud"])
# Pearson correlation: 0.0707

corr2 = pdf["tx_amount_15min"].corr(pdf["is_fraud"])
# Pearson correlation: 0.2581

# Cek apakah rate match dengan full-dataset rate
# All match: True → LEAKAGE CONFIRMED!
```

Output audit:
```
[1] category_fraud_rate vs is_fraud
    Pearson correlation : 0.0707
    Mean per class:
      is_fraud=0 → 0.005758
      is_fraud=1 → 0.010772

[2] tx_amount_15min vs is_fraud
    Pearson correlation : 0.2581
    Mean per class:
      is_fraud=0 → $70.74
      is_fraud=1 → $650.40

[3] Leakage check:
    All match full-dataset rate: True
    ⚠️  LEAKAGE CONFIRMED!
```

### 4.2 Analisa Leakage

| Feature | Leakage? | Penjelasan |
|---------|----------|------------|
| `category_fraud_rate` | ✅ **YA — LEAKAGE** | Dihitung dari full dataset termasuk test set. Model "tahu" fraud rate test data saat training. |
| `tx_amount_15min` | ❌ **TIDAK** | Window function berdasarkan waktu transaksi per kartu. Tidak derived dari target `is_fraud`. |

### 4.3 Kenapa Bisa Terjadi?

Urutan yang salah:
```
Full Dataset
     │
     ├── Hitung category_fraud_rate (dari SEMUA data termasuk test)
     │
     └── Split train/test  ← TERLAMBAT! Test sudah "bocor" ke feature
```

Urutan yang benar:
```
Full Dataset
     │
     └── Split train/test DULU
              │
              ├── Train set → hitung category_fraud_rate
              │
              └── Test set → join category_fraud_rate dari train saja
```

### 4.4 Fix Leakage

Prinsip fix: **split dulu, baru hitung encoding dari train set saja, lalu map ke test set**.

```python
# SPLIT DULU sebelum apapun
train_df, test_df = train_test_split(pdf, test_size=0.2, random_state=42, stratify=pdf["is_fraud"])

# Hitung category_fraud_rate HANYA dari train set
cat_rate = (
    train_df.groupby("category")["is_fraud"]
    .mean()
    .reset_index()
    .rename(columns={"is_fraud": "category_fraud_rate"})
)

# Drop rate lama yang bocor, join rate bersih
train_df = train_df.drop(columns=["category_fraud_rate"]).merge(cat_rate, on="category", how="left")
test_df  = test_df.drop(columns=["category_fraud_rate"]).merge(cat_rate, on="category", how="left")

# Simpan category rate untuk dipakai di predict.py
cat_rate.to_csv("models/category_fraud_rate.csv", index=False)
```

---

## 5. Training Ulang Setelah Fix Leakage

```bash
python src/train_model.py  # versi fixed
```

> Output akan ditambahkan setelah training selesai.

---

## 6. Predict & Dynamic Threshold

### 6.1 Jalankan Predict

```bash
python src/predict.py
```

Output:
```
[INFO] Loaded model: xgboost

[TEST] Normal grocery purchase
==================================================
  Card        : 4532015112830366
  Amount      : $45.50
  Category    : grocery_pos
  Probability : 0.0003
  Threshold   : 0.45
  Decision    : ✅ LEGITIMATE — APPROVE
  Threshold adjustments:
    - medium_risk_category:grocery_pos (-0.05)

[TEST] High-value online shopping at night
==================================================
  Card        : 4532015112830366
  Amount      : $850.00
  Category    : shopping_net
  Probability : 0.9486
  Threshold   : 0.10
  Decision    : 🚨 FRAUD — BLOCK
  Threshold adjustments:
    - user_fraud_history (-0.15)
    - high_risk_category:shopping_net (-0.10)
    - high_amount:$850.0 (-0.10)
    - night_transaction (-0.05)
    - high_velocity:3tx (-0.05)

[TEST] Gas station card test (low amount)
==================================================
  Card        : 5425233430109903
  Amount      : $1.99
  Category    : gas_transport
  Probability : 0.0634
  Threshold   : 0.45
  Decision    : ✅ LEGITIMATE — APPROVE
  Threshold adjustments:
    - high_velocity:4tx (-0.05)
```

### 6.2 Analisa Hasil Predict

| Test Case | Probability | Threshold | Decision | Catatan |
|-----------|-------------|-----------|----------|---------|
| Grocery $45 siang hari | 0.0003 | 0.45 | ✅ APPROVE | Normal, benar |
| Online $850 malam, user flagged | 0.9486 | 0.10 | 🚨 BLOCK | 5 faktor risiko, benar |
| Gas $1.99, velocity 4tx | 0.0634 | 0.45 | ✅ APPROVE | **Edge case!** |

> **Edge case gas station:** Velocity 4 transaksi dalam 15 menit + gas_transport itu klasik pattern card testing. Probability model hanya 0.0634 — model tidak tertipu karena amount sangat kecil dan feature lain tidak mendukung. Perlu dimonitor lebih lanjut.

---

## 7. Masalah & Solusi yang Ditemukan

### 7.1 PySpark pip package tidak compatible dengan RAPIDS
- **Masalah:** Install PySpark via `pip install pyspark` → RAPIDS tidak bisa detect GPU
- **Solusi:** Install Spark standalone via `wget` dari Apache archive

### 7.2 JAVA_HOME dan SPARK_HOME salah
- **Masalah:** `spark-submit` mengarah ke path pip package yang salah → error "Spark jars directory not found"
- **Solusi:** Set manual `SPARK_HOME=$HOME/spark` dan `JAVA_HOME` ke path JVM yang benar

### 7.3 GPU resource framework menyebabkan task hang
- **Masalah:** `spark.executor.resource.gpu.amount` dan `spark.task.resource.gpu.amount` menyebabkan task stuck di Stage 0 (pending, tidak pernah jalan)
- **Solusi:** Hapus config tersebut. Di local mode, RAPIDS langsung detect GPU via CUDA tanpa perlu Spark resource framework

### 7.4 CSV tidak support GPU
- **Masalah:** Warning `unsupported file format: TextFileFormat` → CSV read di CPU
- **Solusi:** Ini normal. Disable RAPIDS saat baca CSV, enable lagi untuk operasi processing. Convert ke Parquet untuk GPU acceleration penuh

### 7.5 Timezone Asia/Jakarta tidak support RAPIDS
- **Masalah:** `Only UTC timezone is supported. Current timezone: Asia/Jakarta` → write parquet fallback ke CPU
- **Solusi:** `spark.conf.set("spark.sql.session.timeZone", "UTC")`

### 7.6 Cannot clear output directory
- **Masalah:** Folder parquet dari container Docker tidak bisa dihapus oleh Spark (permission issue)
- **Solusi:** Hapus manual dari terminal dengan `sudo rm -rf` sebelum write

### 7.7 RAPIDS 24.02 tidak support CUDA 13 (RTX 5060 Ti)
- **Masalah:** RTX 5060 Ti = Blackwell architecture (SM 120). RAPIDS 24.02 hanya support arch 7, 8, 9
- **Solusi:** Upgrade ke RAPIDS 25.02 + cuDF 26.02 yang sudah support CUDA 13

### 7.8 ScaledLR tidak bisa di-pickle
- **Masalah:** `AttributeError: Can't pickle local object 'main.<locals>.ScaledLR'` karena class didefinisikan di dalam fungsi `main()`
- **Solusi:** Pindahkan class `ScaledLR` ke top-level module (di luar semua fungsi)

### 7.9 Data Leakage pada category_fraud_rate
- **Masalah:** `category_fraud_rate` dihitung dari full dataset sebelum split → test set "bocor" ke feature → ROC AUC terlalu tinggi (0.9959)
- **Solusi:** Split train/test DULU, baru hitung rate dari train set saja, lalu map ke test set via join

---

## 8. Struktur Project

```
fraud_detection_ai/
  src/
    feature_store.py       # Build feature store via PySpark
    train_model.py         # Training LR, RF, XGBoost + SMOTE
    predict.py             # Scoring + dynamic threshold engine
  models/
    logistic_regression.pkl
    random_forest.pkl
    xgboost.pkl
    category_fraud_rate.csv  # Disimpan dari train set (anti-leakage)
  data/                    # Dataset (gitignored)
  requirements.txt
  README.md
  notes.md                 # File ini
```

---

## 9. Dataset

| Info | Detail |
|------|--------|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) |
| File training | `fraudTrain.csv` (335 MB) |
| File test | `fraudTest.csv` |
| Total baris | 1,296,676 |
| Kolom | 23 |
| Fraud | 7,506 (0.58%) |
| Legitimate | 1,289,169 (99.42%) |

---

*Ditulis: Maret 2026*