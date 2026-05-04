"""
Credit Card Fraud Detection
Based on: https://www.geeksforgeeks.org/machine-learning/ml-credit-card-fraud-detection/

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  284,807 transactions | 31 features | ~0.17% fraud
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_auc_score,
    roc_curve, classification_report,
)

DATASET_PATH = "creditcard.csv"
RANDOM_STATE = 42


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(
            f"\n[ERROR] Dataset not found at '{path}'.\n"
            "Download it from Kaggle:\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "Then place creditcard.csv in this directory and re-run.\n"
        )
        sys.exit(1)

    print(f"Loading dataset from '{path}' ...")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ── 2. Exploratory Data Analysis ──────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> None:
    print("\n── EDA ─────────────────────────────────────────────────")

    fraud   = df[df["Class"] == 1]
    valid   = df[df["Class"] == 0]
    n_fraud = len(fraud)
    n_valid = len(valid)
    total   = len(df)

    print(f"  Valid transactions : {n_valid:>7,}  ({n_valid/total*100:.3f}%)")
    print(f"  Fraud transactions : {n_fraud:>7,}  ({n_fraud/total*100:.3f}%)")
    print(f"\n  Avg transaction amount — valid: ${valid['Amount'].mean():.2f} | fraud: ${fraud['Amount'].mean():.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Credit Card Fraud Detection — EDA", fontsize=14, weight="bold")

    # Class distribution
    axes[0].bar(["Valid", "Fraud"], [n_valid, n_fraud], color=["steelblue", "tomato"])
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Count")
    for bar, val in zip(axes[0].patches, [n_valid, n_fraud]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                     f"{val:,}", ha="center", va="bottom", fontsize=9)

    # Amount distribution
    axes[1].hist(valid["Amount"], bins=60, alpha=0.6, color="steelblue", label="Valid", density=True)
    axes[1].hist(fraud["Amount"], bins=60, alpha=0.6, color="tomato", label="Fraud", density=True)
    axes[1].set_title("Transaction Amount Distribution")
    axes[1].set_xlabel("Amount ($)")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].set_xlim(0, 2500)

    # Correlation heatmap (sample of features for readability)
    sample_cols = ["V1", "V2", "V3", "V4", "V5", "V10", "V14", "V17", "Amount", "Class"]
    corr = df[sample_cols].corr()
    sns.heatmap(corr, ax=axes[2], cmap="coolwarm", fmt=".1f", annot=True,
                annot_kws={"size": 7}, linewidths=0.5, square=True)
    axes[2].set_title("Feature Correlation (subset)")

    plt.tight_layout()
    plt.savefig("eda.png", dpi=150)
    print("  EDA plot saved to eda.png")
    plt.show()


# ── 3. Preprocess ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    print("\n── Preprocessing ───────────────────────────────────────")

    # Normalise Amount (V1-V28 are already PCA-scaled; Time is dropped)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df = df.copy()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df.drop(columns=["Time"], inplace=True)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train size : {len(X_train):,}  |  Test size : {len(X_test):,}")
    print(f"  Fraud in train : {y_train.sum():,}  |  Fraud in test : {y_test.sum():,}")
    return X_train, X_test, y_train, y_test


# ── 4. Train ──────────────────────────────────────────────────────────────────

def train(X_train, y_train) -> RandomForestClassifier:
    print("\n── Training Random Forest ──────────────────────────────")
    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",   # compensates for class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print("  Training complete.")
    return clf


# ── 5. Evaluate ───────────────────────────────────────────────────────────────

def evaluate(clf, X_test, y_test) -> None:
    print("\n── Evaluation ──────────────────────────────────────────")

    y_pred  = clf.predict(X_test)
    y_prob  = clf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    mcc  = matthews_corrcoef(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"  MCC       : {mcc:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Valid", "Fraud"]))

    _plot_results(clf, X_test, y_test, y_pred, y_prob)


def _plot_results(clf, X_test, y_test, y_pred, y_prob) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Model Evaluation", fontsize=14, weight="bold")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, ax=axes[0], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Valid", "Fraud"], yticklabels=["Valid", "Fraud"])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[1].plot(fpr, tpr, color="tomato", lw=2, label=f"ROC (AUC = {auc:.4f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")

    # Feature importances (top 15)
    importances = pd.Series(clf.feature_importances_, index=X_test.columns)
    importances.nlargest(15).sort_values().plot(
        kind="barh", ax=axes[2], color="steelblue"
    )
    axes[2].set_title("Top 15 Feature Importances")
    axes[2].set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("  Results plot saved to results.png")
    plt.show()


# ── 6. Predict single transaction ─────────────────────────────────────────────

def predict_single(clf, X_test: pd.DataFrame) -> None:
    """Demo: run the model on one transaction from the test set."""
    print("\n── Single-transaction Demo ─────────────────────────────")
    sample = X_test.iloc[[0]]
    prob   = clf.predict_proba(sample)[0][1]
    label  = "FRAUD" if prob >= 0.5 else "VALID"
    print(f"  Prediction : {label}  (fraud probability: {prob:.4f})")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection")
    parser.add_argument("--data", default=DATASET_PATH, help="Path to creditcard.csv")
    parser.add_argument("--no-eda", action="store_true", help="Skip EDA plots")
    args = parser.parse_args()

    df = load_data(args.data)

    if not args.no_eda:
        run_eda(df)

    X_train, X_test, y_train, y_test = preprocess(df)
    clf = train(X_train, y_train)
    evaluate(clf, X_test, y_test)
    predict_single(clf, X_test)
