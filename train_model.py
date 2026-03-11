import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

DATA_PATH = "/home/ayah/projects/dataset/fraud/data/PS_20174392719_1491204439457_log.csv"


def load_data(path):

    print("Loading dataset...")
    df = pd.read_csv(path)

    print("Original shape:", df.shape)

    y = df["isFraud"]

    X = df.drop(columns=[
        "isFraud",
        "nameOrig",
        "nameDest",
        "isFlaggedFraud"
    ])

    X = pd.get_dummies(X, columns=["type"])

    return X, y


def train_model(X, y):

    print("Splitting data...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("Training XGBoost (GPU)...")

    model = XGBClassifier(
        tree_method="hist",
        device="cuda",
        n_estimators=400,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc"
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")

    preds = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, preds)

    print("=================================")
    print("ROC AUC SCORE:", score)
    print("=================================")

    return model


if __name__ == "__main__":

    if not os.path.exists(DATA_PATH):
        print("Dataset not found:", DATA_PATH)
        exit()

    X, y = load_data(DATA_PATH)

    model = train_model(X, y)