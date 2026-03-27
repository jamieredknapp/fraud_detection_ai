"""
predict.py
----------
Loads a trained model and scores transactions.
Outputs fraud_probability per transaction.
Applies a dynamic threshold engine — threshold varies by:
  - user risk profile (history of fraud)
  - merchant category risk
  - transaction amount
  - time of day

Usage:
    python src/predict.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODELS_DIR = "models"
DEFAULT_MODEL = "xgboost"

FEATURE_COLS = [
    "amt",
    "amt_log",
    "tx_count_15min",
    "tx_amount_15min",
    "category_fraud_rate",
    "tx_hour",
    "tx_dayofweek",
    "is_night",
    "is_weekend",
    "city_pop",
    "merch_lat",
    "merch_long",
    "lat",
    "long",
]

# Category fraud rates (from feature store analysis)
CATEGORY_FRAUD_RATES = {
    "shopping_net"  : 0.0176,
    "misc_net"      : 0.0145,
    "grocery_pos"   : 0.0141,
    "shopping_pos"  : 0.0072,
    "gas_transport" : 0.0047,
    "misc_pos"      : 0.0031,
    "travel"        : 0.0029,
    "grocery_net"   : 0.0029,
    "entertainment" : 0.0025,
    "personal_care" : 0.0024,
    "kids_pets"     : 0.0021,
    "food_dining"   : 0.0017,
    "home"          : 0.0016,
    "health_fitness": 0.0015,
}


# ─────────────────────────────────────────────
# Transaction Input
# ─────────────────────────────────────────────
@dataclass
class Transaction:
    amt: float
    category: str
    tx_hour: int
    tx_dayofweek: int
    tx_count_15min: int
    tx_amount_15min: float
    city_pop: int
    merch_lat: float
    merch_long: float
    lat: float
    long: float
    cc_num: Optional[str] = None

    def to_feature_vector(self) -> np.ndarray:
        amt_log = np.log1p(self.amt)
        category_fraud_rate = CATEGORY_FRAUD_RATES.get(self.category, 0.002)
        is_night = 1 if self.tx_hour < 6 else 0
        is_weekend = 1 if self.tx_dayofweek in [1, 7] else 0

        return np.array([[
            self.amt,
            amt_log,
            self.tx_count_15min,
            self.tx_amount_15min,
            category_fraud_rate,
            self.tx_hour,
            self.tx_dayofweek,
            is_night,
            is_weekend,
            self.city_pop,
            self.merch_lat,
            self.merch_long,
            self.lat,
            self.long,
        ]])


# ─────────────────────────────────────────────
# Dynamic Threshold Engine
# ─────────────────────────────────────────────
@dataclass
class ThresholdEngine:
    """
    Computes a dynamic decision threshold per transaction.

    Base threshold is 0.5. Lowered (more sensitive) when:
      - User has a fraud history
      - High-risk merchant category
      - High transaction amount
      - Transaction occurs at night
    """
    base_threshold: float = 0.5

    def compute(
        self,
        transaction: Transaction,
        user_has_fraud_history: bool = False
    ) -> float:
        threshold = self.base_threshold
        adjustments = []

        # User risk
        if user_has_fraud_history:
            threshold -= 0.15
            adjustments.append("user_fraud_history (-0.15)")

        # Category risk
        category_rate = CATEGORY_FRAUD_RATES.get(transaction.category, 0.002)
        if category_rate >= 0.015:
            threshold -= 0.10
            adjustments.append(f"high_risk_category:{transaction.category} (-0.10)")
        elif category_rate >= 0.005:
            threshold -= 0.05
            adjustments.append(f"medium_risk_category:{transaction.category} (-0.05)")

        # Amount risk
        if transaction.amt > 500:
            threshold -= 0.10
            adjustments.append(f"high_amount:${transaction.amt} (-0.10)")
        elif transaction.amt > 200:
            threshold -= 0.05
            adjustments.append(f"medium_amount:${transaction.amt} (-0.05)")

        # Night transaction
        if transaction.tx_hour < 6:
            threshold -= 0.05
            adjustments.append("night_transaction (-0.05)")

        # Velocity risk
        if transaction.tx_count_15min >= 3:
            threshold -= 0.05
            adjustments.append(f"high_velocity:{transaction.tx_count_15min}tx (-0.05)")

        # Floor threshold at 0.1 to avoid extreme sensitivity
        threshold = max(threshold, 0.10)

        return round(threshold, 2), adjustments


# ─────────────────────────────────────────────
# Prediction Result
# ─────────────────────────────────────────────
@dataclass
class PredictionResult:
    cc_num: Optional[str]
    amt: float
    category: str
    fraud_probability: float
    threshold: float
    decision: str
    threshold_adjustments: list = field(default_factory=list)

    def __str__(self):
        lines = [
            f"\n{'='*50}",
            f"  FRAUD DETECTION RESULT",
            f"{'='*50}",
            f"  Card        : {self.cc_num or 'N/A'}",
            f"  Amount      : ${self.amt:.2f}",
            f"  Category    : {self.category}",
            f"  Probability : {self.fraud_probability:.4f}",
            f"  Threshold   : {self.threshold}",
            f"  Decision    : {self.decision}",
        ]
        if self.threshold_adjustments:
            lines.append(f"\n  Threshold adjustments applied:")
            for adj in self.threshold_adjustments:
                lines.append(f"    - {adj}")
        lines.append(f"{'='*50}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Predictor
# ─────────────────────────────────────────────
class FraudPredictor:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Run train_model.py first."
            )
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.threshold_engine = ThresholdEngine()
        print(f"[INFO] Loaded model: {model_name}")

    def predict(
        self,
        transaction: Transaction,
        user_has_fraud_history: bool = False
    ) -> PredictionResult:
        # Get fraud probability
        X = transaction.to_feature_vector()
        fraud_prob = float(self.model.predict_proba(X)[0][1])

        # Get dynamic threshold
        threshold, adjustments = self.threshold_engine.compute(
            transaction, user_has_fraud_history
        )

        # Decision
        if fraud_prob >= threshold:
            decision = "🚨 FRAUD — BLOCK"
        elif fraud_prob >= threshold * 0.7:
            decision = "⚠️  SUSPICIOUS — REVIEW"
        else:
            decision = "✅ LEGITIMATE — APPROVE"

        return PredictionResult(
            cc_num=transaction.cc_num,
            amt=transaction.amt,
            category=transaction.category,
            fraud_probability=round(fraud_prob, 4),
            threshold=threshold,
            decision=decision,
            threshold_adjustments=adjustments,
        )

    def predict_batch(
        self,
        transactions: list[Transaction],
        user_fraud_histories: Optional[list[bool]] = None
    ) -> list[PredictionResult]:
        if user_fraud_histories is None:
            user_fraud_histories = [False] * len(transactions)

        return [
            self.predict(tx, hist)
            for tx, hist in zip(transactions, user_fraud_histories)
        ]


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────
def main():
    predictor = FraudPredictor(model_name="xgboost")

    test_cases = [
        # (description, transaction, user_has_fraud_history)
        (
            "Normal grocery purchase",
            Transaction(
                cc_num="4532015112830366",
                amt=45.50,
                category="grocery_pos",
                tx_hour=14,
                tx_dayofweek=3,
                tx_count_15min=1,
                tx_amount_15min=45.50,
                city_pop=50000,
                merch_lat=40.71,
                merch_long=-74.00,
                lat=40.70,
                long=-73.99,
            ),
            False,
        ),
        (
            "High-value online shopping at night",
            Transaction(
                cc_num="4532015112830366",
                amt=850.00,
                category="shopping_net",
                tx_hour=2,
                tx_dayofweek=7,
                tx_count_15min=3,
                tx_amount_15min=2100.00,
                city_pop=50000,
                merch_lat=37.77,
                merch_long=-122.41,
                lat=40.70,
                long=-73.99,
            ),
            True,  # user has fraud history
        ),
        (
            "Gas station card test (low amount)",
            Transaction(
                cc_num="5425233430109903",
                amt=1.99,
                category="gas_transport",
                tx_hour=23,
                tx_dayofweek=1,
                tx_count_15min=4,
                tx_amount_15min=7.96,
                city_pop=12000,
                merch_lat=33.44,
                merch_long=-112.07,
                lat=33.45,
                long=-112.06,
            ),
            False,
        ),
    ]

    for description, tx, has_history in test_cases:
        print(f"\n[TEST] {description}")
        result = predictor.predict(tx, user_has_fraud_history=has_history)
        print(result)


if __name__ == "__main__":
    main()
