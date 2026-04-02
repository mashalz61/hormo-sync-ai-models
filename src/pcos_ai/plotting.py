from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve


def save_confusion_matrix_plot(matrix: np.ndarray, output_path: str | Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_path)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    axis.figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ylabel="Actual",
        xlabel="Predicted",
        title=title,
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, str(matrix[row_index, column_index]), ha="center", va="center")

    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_roc_curve_plot(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_path: str | Path,
    title: str,
    roc_auc: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_path)
    fpr, tpr, _ = roc_curve(y_true, probabilities)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}", linewidth=2)
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(title)
    axis.legend(loc="lower right")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_combined_roc_plot(
    roc_entries: list[dict[str, object]],
    output_path: str | Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_path)
    figure, axis = plt.subplots(figsize=(7, 6))
    for entry in roc_entries:
        y_true = np.asarray(entry["y_true"])
        probabilities = np.asarray(entry["probabilities"])
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        axis.plot(fpr, tpr, linewidth=2, label=f"{entry['model']} (AUC={entry['roc_auc']:.4f})")

    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(title)
    axis.legend(loc="lower right", fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
