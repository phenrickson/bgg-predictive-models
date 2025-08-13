from __future__ import annotations

import typing as t
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def plot_confusion_matrix(
    predictions_df: pl.DataFrame, ax: plt.Axes | None = None, normalize: bool = False
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a confusion matrix visualization.

    Args:
        predictions_df: DataFrame containing actual and predicted binary labels
        ax: Matplotlib axes to plot on (optional)
        normalize: Whether to normalize the confusion matrix

    Returns:
        Matplotlib figure and axes
    """
    if (
        "actual" not in predictions_df.columns
        or "prediction" not in predictions_df.columns
    ):
        raise ValueError("DataFrame must contain 'actual' and 'prediction' columns")

    # Convert to numpy arrays
    y_true = predictions_df["actual"].to_numpy()
    y_pred = predictions_df["prediction"].to_numpy()

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    return fig, ax


def plot_roc_curve(
    predictions_df: pl.DataFrame, ax: plt.Axes | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a Receiver Operating Characteristic (ROC) curve.

    Args:
        predictions_df: DataFrame containing actual labels and predicted probabilities for class 1
        ax: Matplotlib axes to plot on (optional)

    Returns:
        Matplotlib figure and axes
    """
    if (
        "actual" not in predictions_df.columns
        or "predicted_proba_class_1" not in predictions_df.columns
    ):
        raise ValueError(
            "DataFrame must contain 'actual' and 'predicted_proba_class_1' columns"
        )

    y_true = predictions_df["actual"].to_numpy()
    y_scores = predictions_df["predicted_proba_class_1"].to_numpy()

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")

    return fig, ax


def plot_precision_recall_curve(
    predictions_df: pl.DataFrame, ax: plt.Axes | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a Precision-Recall curve.

    Args:
        predictions_df: DataFrame containing actual labels and predicted probabilities for class 1
        ax: Matplotlib axes to plot on (optional)

    Returns:
        Matplotlib figure and axes
    """
    if (
        "actual" not in predictions_df.columns
        or "predicted_proba_class_1" not in predictions_df.columns
    ):
        raise ValueError(
            "DataFrame must contain 'actual' and 'predicted_proba_class_1' columns"
        )

    y_true = predictions_df["actual"].to_numpy()
    y_scores = predictions_df["predicted_proba_class_1"].to_numpy()

    # Compute Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    ax.plot(
        recall,
        precision,
        color="blue",
        lw=2,
        label=f"Precision-Recall curve (AP = {avg_precision:.2f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")

    return fig, ax


def plot_classification_diagnostics(
    predictions_df: pl.DataFrame, style: str = "seaborn-v0_8-darkgrid"
) -> tuple[plt.Figure, t.Sequence[plt.Axes]]:
    """
    Create comprehensive binary classification diagnostic plots.

    Args:
        predictions_df: DataFrame containing classification predictions
        style: Matplotlib style to use

    Returns:
        Matplotlib figure and axes
    """
    plt.style.use(style)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Confusion Matrix
    plot_confusion_matrix(predictions_df, ax=axes[0])

    # ROC Curve
    plot_roc_curve(predictions_df, ax=axes[1])

    # Precision-Recall Curve
    plot_precision_recall_curve(predictions_df, ax=axes[2])

    fig.tight_layout()
    return fig, axes


def compute_classification_metrics(predictions_df: pl.DataFrame) -> dict[str, float]:
    """
    Compute key classification metrics.

    Args:
        predictions_df: DataFrame containing actual and predicted labels

    Returns:
        Dictionary of classification metrics
    """
    if (
        "actual" not in predictions_df.columns
        or "prediction" not in predictions_df.columns
    ):
        raise ValueError("DataFrame must contain 'actual' and 'prediction' columns")

    y_true = predictions_df["actual"].to_numpy()
    y_pred = predictions_df["prediction"].to_numpy()

    return {
        "F1 Score": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
    }


def main():
    """
    Load predictions and generate classification diagnostic plots.
    """
    # Load predictions (adjust path as needed)
    predictions_df = pl.read_parquet(
        "models/experiments/hurdle/test-hurdle/v2/test_predictions.parquet"
    )

    # Generate plots
    fig, _ = plot_classification_diagnostics(predictions_df)

    # Save the figure
    output_path = "figures/classification_diagnostics.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Print metrics
    metrics = compute_classification_metrics(predictions_df)
    print("Classification Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
