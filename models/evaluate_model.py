"""
=========================================================
SwitchGuardAI - Model Evaluation
=========================================================

Evaluates the trained Random Forest model.

Outputs:
1. Accuracy
2. Precision
3. Recall
4. F1-score
5. Classification Report
6. Confusion Matrix
7. Feature Importance
8. ROC Curve (one-vs-rest, per fault class)
9. Per-class metrics CSV

Author : Tanishka
Project : SwitchGuardAI
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

MODEL_PATH = "saved_models/random_forest.pkl"

X_TEST_PATH = "data/processed/X_test.csv"
Y_TEST_PATH = "data/processed/y_test.csv"

RESULTS_FOLDER = "results"


def evaluate():

    print("=" * 60)
    print("SwitchGuardAI - Model Evaluation")
    print("=" * 60)

    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # -------------------------------------------------
    # Load Model
    # -------------------------------------------------

    model = joblib.load(MODEL_PATH)

    # -------------------------------------------------
    # Load Test Data
    # -------------------------------------------------

    X_test = pd.read_csv(X_TEST_PATH)

    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    print("Test Samples :", len(X_test))

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # -------------------------------------------------
    # Accuracy
    # -------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy")

    print(f"{accuracy:.4f}")

    # -------------------------------------------------
    # Classification Report
    # -------------------------------------------------

    report = classification_report(
        y_test,
        y_pred
    )

    print("\nClassification Report\n")

    print(report)

    with open(
        "results/classification_report.txt",
        "w"
    ) as f:

        f.write(report)

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )

    fig, ax = plt.subplots(figsize=(8,6))

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        "results/confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # -------------------------------------------------
    # Feature Importance
    # -------------------------------------------------

    importance = pd.Series(

        model.feature_importances_,

        index=X_test.columns

    ).sort_values(
        ascending=False
    )

    importance.to_csv(
        "results/feature_importance.csv"
    )

    plt.figure(figsize=(10,8))

    importance.head(20).sort_values().plot(
        kind="barh"
    )

    plt.title("Top 20 Feature Importances")

    plt.xlabel("Importance")

    plt.tight_layout()

    plt.savefig(
        "results/feature_importance.png",
        dpi=300
    )

    plt.close()

    # -------------------------------------------------
    # ROC Curve (one-vs-rest, per fault class)
    # -------------------------------------------------

    print("\nGenerating ROC Curve...")

    class_labels = model.classes_

    y_test_binarized = label_binarize(y_test, classes=class_labels)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, label in enumerate(class_labels):

        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - One-vs-Rest (per fault class)")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()

    plt.savefig(
        "results/roc_curve.png",
        dpi=300
    )

    plt.close()

    print("Saved: results/roc_curve.png")

    # -------------------------------------------------
    # Per-Class Metrics CSV (precision/recall/f1/support)
    # -------------------------------------------------

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=class_labels
    )

    metrics_df = pd.DataFrame({
        "Class": class_labels,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Support": support
    })

    metrics_df.to_csv(
        "results/model_metrics.csv",
        index=False
    )

    print("\nResults saved inside 'results/' folder.")

    print("\nTop 10 Important Features\n")

    print(importance.head(10))

    print("\nPer-Class Metrics\n")

    print(metrics_df)


if __name__ == "__main__":

    evaluate()