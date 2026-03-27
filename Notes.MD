# Fraud Detection AI — Engineering Notes

> Dokumentasi proses pembangunan sistem fraud detection end-to-end, termasuk feature engineering, training, deteksi data leakage, dan perbaikan pipeline.

---

## 🧠 Ringkasan Proyek

Proyek ini membangun sistem **fraud detection berbasis machine learning** dengan pendekatan:

- Model: Logistic Regression, Random Forest, XGBoost  
- Feature Engineering: velocity, category risk, time-based features  
- Decision Layer: dynamic threshold berbasis konteks transaksi  

Tujuan utama:
> Mendeteksi transaksi fraud secara akurat dengan mempertimbangkan aspek teknis (model) dan bisnis (decision logic).

---

## ⚙️ Arsitektur Sistem

Raw Data (CSV)
      ↓
Feature Engineering (PySpark)
      ↓
Feature Store (Parquet)
      ↓
Model Training (XGBoost, dll)
      ↓
Prediction + Dynamic Threshold
      ↓
Final Decision (APPROVE / BLOCK)

---

## 🔍 Feature Engineering

### 1. Velocity Features (Window 15 Menit)

Mengukur aktivitas transaksi dalam waktu singkat:

- `tx_count_15min`
- `tx_amount_15min`

**Insight:**
- Jumlah transaksi tidak terlalu berbeda antara fraud vs normal  
- Namun total amount dalam 15 menit:
  - Fraud ≈ $650  
  - Normal ≈ $70  
→ **signal kuat untuk fraud detection**

---

### 2. Category Risk Encoding

Menghitung tingkat risiko fraud per kategori transaksi:

- `shopping_net`, `misc_net` → fraud tinggi (online)
- `grocery_pos`, `gas_transport` → card testing pattern

---

### 3. Time-Based Features

- `tx_hour`
- `is_night`
- `is_weekend`
- `amt_log`

---

## 🤖 Model Training

Model yang digunakan:

| Model | ROC AUC |
|------|--------|
| Logistic Regression | 0.9257 |
| Random Forest | 0.9899 |
| XGBoost | **0.9959** |

> ⚠️ ROC AUC terlalu tinggi → indikasi kemungkinan data leakage

---

## 🚨 Deteksi Data Leakage

### Gejala

- ROC AUC mendekati 1.0 (0.9959)
- Model terlihat “terlalu sempurna”

---

### Investigasi

Dilakukan audit terhadap feature:

- `category_fraud_rate`  
- `tx_amount_15min`

Hasil:

- `tx_amount_15min` → valid  
- `category_fraud_rate` → **mengandung leakage**

---

### Root Cause

Feature `category_fraud_rate` dihitung dari **seluruh dataset sebelum split**:

Full Dataset
     ↓
Hitung category_fraud_rate
     ↓
Split train/test  ❌ (terlambat)

Akibatnya:

> Model “mengetahui” informasi dari test set saat training

---

### Definisi Masalah

> Data leakage terjadi ketika model menggunakan informasi yang tidak tersedia saat prediksi di dunia nyata, terutama informasi yang berasal dari label (is_fraud) di masa depan.

---

## ✅ Solusi (Fix Leakage)

Pipeline diperbaiki menjadi:

Full Dataset
     ↓
Split train/test DULU
     ↓
Hitung category_fraud_rate dari TRAIN saja
     ↓
Apply ke test set

Implementasi:

train_df, test_df = train_test_split(...)

cat_rate = train_df.groupby("category")["is_fraud"].mean()

train_df = train_df.merge(cat_rate, ...)
test_df  = test_df.merge(cat_rate, ...)

---

## ⚖️ Decision Layer (Dynamic Threshold)

Model tidak digunakan langsung untuk keputusan.

Ditambahkan **dynamic threshold berbasis konteks**:

Contoh faktor:

- kategori risiko tinggi
- nominal transaksi besar
- transaksi malam hari
- velocity tinggi
- riwayat fraud user

---

### Contoh Output

Probability : 0.9486  
Threshold   : 0.10  
Decision    : FRAUD — BLOCK  

> Keputusan tidak hanya berdasarkan model, tetapi juga konteks transaksi.

---

## 🧪 Analisa Edge Case

Kasus:
- Gas station
- Amount kecil ($1.99)
- Velocity tinggi

Hasil:
- Model tidak mendeteksi fraud

**Insight:**
> Tidak semua pattern velocity tinggi adalah fraud → perlu evaluasi tambahan (rule-based / monitoring)

---

## 📌 Lessons Learned

1. Nilai ROC AUC yang terlalu tinggi perlu dicurigai sebagai indikasi data leakage.  
2. Data leakage sering terjadi pada tahap feature engineering, bukan pada model.  
3. Feature berbasis target harus dihitung hanya dari training data.  
4. Urutan proses sangat penting: split data terlebih dahulu sebelum menghitung feature.  
5. Feature engineering dapat menyebabkan kesalahan serius jika tidak dilakukan dengan benar (silent failure).  
6. Model dengan performa tinggi belum tentu valid di dunia nyata.  
7. Fraud detection adalah kombinasi antara model, feature, dan business rules.  
8. Dynamic threshold membantu menyesuaikan keputusan dengan konteks transaksi.  
9. Audit pipeline lebih penting daripada hanya melihat metrik model.  
10. Proses debugging dan investigasi merupakan bagian penting dari engineering.

---

## 🎯 Kesimpulan

> Model yang baik bukan hanya yang memiliki metrik tinggi,  
> tetapi yang dibangun dengan pipeline yang valid, dapat dipercaya, dan sesuai dengan kondisi dunia nyata.
