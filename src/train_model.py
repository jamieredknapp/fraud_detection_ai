"""
train_model.py
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

FEATURE_STORE_PATH = "/home/ayah/datasets/kartik/fraud_feature_store"
MODELS_DIR         = "/home/ayah/projects/fraud_detection/models"
RAPIDS_JAR         = "/home/ayah/spark/rapids/rapids-4-spark.jar"

FEATURE_COLS = [
    "amt", "amt_log", "tx_count_15min", "tx_amount_15min",
    "category_fraud_rate", "tx_hour", "tx_dayofweek",
    "is_night", "is_weekend", "city_pop",
    "merch_lat", "merch_long", "lat", "long",
]
LABEL_COL = "is_fraud"
os.makedirs(MODELS_DIR, exist_ok=True)

def build_spark():
    return (SparkSession.builder
        .appName("FraudDetection-Training")
        .config("spark.jars", RAPIDS_JAR)
        .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
        .config("spark.rapids.sql.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .master("local[2]")
        .getOrCreate())

def load_features(spark):
    print("[INFO] Loading Feature Store...")
    df = spark.read.parquet(FEATURE_STORE_PATH)
    df = df.withColumn(LABEL_COL, F.col(LABEL_COL).cast("int"))
    pdf = df.select(FEATURE_COLS + [LABEL_COL]).toPandas()
    print(f"[INFO] Shape: {pdf.shape}")
    print(f"[INFO] Fraud: {pdf[LABEL_COL].sum()} | Legit: {(pdf[LABEL_COL]==0).sum()}")
    return pdf

def prepare_data(pdf):
    X = pdf[FEATURE_COLS].values
    y = pdf[LABEL_COL].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[INFO] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    print("\n[INFO] Applying SMOTE...")
    print(f"[INFO] Before — fraud: {y_train.sum()} | legit: {(y_train==0).sum()}")
    X_res, y_res = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_train, y_train)
    print(f"[INFO] After  — fraud: {y_res.sum()} | legit: {(y_res==0).sum()}")
    return X_res, y_res

def evaluate(name, model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    r = {
        "model"    : name,
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall"   : round(recall_score(y_test, y_pred), 4),
        "f1"       : round(f1_score(y_test, y_pred), 4),
        "roc_auc"  : round(roc_auc_score(y_test, y_prob), 4),
    }
    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    print(f"  Precision : {r['precision']}")
    print(f"  Recall    : {r['recall']}")
    print(f"  F1 Score  : {r['f1']}")
    print(f"  ROC AUC   : {r['roc_auc']}")
    print(classification_report(y_test, y_pred, target_names=["legitimate","fraud"]))
    return r

def save_model(model, name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Saved: {path}")

def print_comparison(results):
    print(f"\n{'='*65}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*65}")
    print(f"  {'Model':<28} {'Precision':>9} {'Recall':>7} {'F1':>7} {'ROC AUC':>9}")
    print(f"  {'-'*60}")
    for r in sorted(results, key=lambda x: x["roc_auc"], reverse=True):
        print(f"  {r['model']:<28} {r['precision']:>9} {r['recall']:>7} {r['f1']:>7} {r['roc_auc']:>9}")
    best = max(results, key=lambda x: x["roc_auc"])
    print(f"\n  Best model: {best['model']} (ROC AUC: {best['roc_auc']})")
    print(f"{'='*65}")

def main():
    spark = build_spark()
    pdf = load_features(spark)
    spark.stop()

    X_train, X_test, y_train, y_test = prepare_data(pdf)
    X_train, y_train = apply_smote(X_train, y_train)

    # Logistic Regression
    print("\n[INFO] Training Logistic Regression...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_scaled, y_train)
class ScaledLR:
    def __init__(self, s, m): self.s, self.m = s, m
    def predict(self, X): return self.m.predict(self.s.transform(X))
    def predict_proba(self, X): return self.m.predict_proba(self.s.transform(X))
    lr_model = ScaledLR(scaler, lr)

    # Random Forest
    print("\n[INFO] Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # XGBoost
    print("\n[INFO] Training XGBoost...")
    scale = (y_train==0).sum() / (y_train==1).sum()
    xgb_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
        eval_metric="aucpr", random_state=42, n_jobs=-1, tree_method="hist")
    xgb_model.fit(X_train, y_train)

    # Evaluate
    results = []
    results.append(evaluate("Logistic Regression", lr_model, X_test, y_test))
    results.append(evaluate("Random Forest", rf_model, X_test, y_test))
    results.append(evaluate("XGBoost", xgb_model, X_test, y_test))

    print_comparison(results)

    save_model(lr_model, "logistic_regression")
    save_model(rf_model, "random_forest")
    save_model(xgb_model, "xgboost")

    print("\n[DONE] Training complete.")

if __name__ == "__main__":
    main()
