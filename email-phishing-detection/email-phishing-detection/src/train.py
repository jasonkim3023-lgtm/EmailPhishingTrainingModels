"""Train and compare Naive Bayes, Logistic Regression, and a linear SVM on
the CEAS_08 phishing email dataset.

Saves a metrics table (``results/metrics.csv``) and three figures
(model comparison, ROC curves, confusion matrices) to the output directory.

Usage:
    python src/train.py --data data/CEAS_08.csv --outdir results
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_and_clean, build_features


def get_models():
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": LinearSVC(),
    }


def score_of(model, X_test):
    """Probability/decision score for the positive class (for ROC-AUC)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    return model.decision_function(X_test)  # LinearSVC has no predict_proba


def main():
    parser = argparse.ArgumentParser(description="Train phishing detectors.")
    parser.add_argument("--data", default="data/CEAS_08.csv",
                        help="Path to the CEAS_08 CSV file.")
    parser.add_argument("--outdir", default="results",
                        help="Directory for metrics and figures.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading and cleaning {args.data} ...")
    df = load_and_clean(args.data)
    print(f"  {len(df):,} emails after cleaning "
          f"({df['label'].mean():.1%} phishing)")

    X, y, _ = build_features(df, max_features=args.max_features)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y,
        random_state=args.random_state)

    rows, fitted, scores = {}, {}, {}
    for name, clf in get_models().items():
        clf.fit(X_train, y_train)
        fitted[name] = clf
        pred = clf.predict(X_test)
        scores[name] = score_of(clf, X_test)
        rows[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, scores[name]),
        }
        print(f"  trained {name}")

    results = pd.DataFrame(rows).T.round(4)
    results.to_csv(os.path.join(args.outdir, "metrics.csv"))
    print("\n" + results.to_string())
    best = results["F1"].idxmax()
    print(f"\nBest model by F1: {best}")

    _plot_bars(results, args.outdir)
    _plot_roc(y_test, scores, results, args.outdir)
    _plot_confusion(y_test, fitted, X_test, args.outdir)
    print(f"\nSaved metrics and figures to '{args.outdir}/'.")


def _plot_bars(results, outdir):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    ax = results[metrics].plot(kind="bar", figsize=(10, 5))
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0.95, 1.001)
    ax.legend(loc="lower right", ncol=5, fontsize=9)
    ax.set_xticklabels(results.index, rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "model_comparison.png"), dpi=150)
    plt.close()


def _plot_roc(y_test, scores, results, outdir):
    plt.figure(figsize=(6, 5))
    for name in scores:
        fpr, tpr, _ = roc_curve(y_test, scores[name])
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={results.loc[name, 'ROC-AUC']:.4f})")
    plt.plot([0, 1], [0, 1], ls="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_curves.png"), dpi=150)
    plt.close()


def _plot_confusion(y_test, fitted, X_test, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, clf) in zip(axes, fitted.items()):
        cm = confusion_matrix(y_test, clf.predict(X_test))
        ConfusionMatrixDisplay(cm, display_labels=["Legit", "Phish"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrices.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
