from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


SCORING = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "average_precision": "average_precision",
}


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    confusion_matrix: np.ndarray
    threshold: float
    classifier: Pipeline


def cross_validate_model(
    estimator: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    cv_folds: int,
) -> dict[str, float]:
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_validate(estimator, features, target, cv=splitter, scoring=SCORING, n_jobs=1)
    summary: dict[str, float] = {}
    for metric_name in SCORING:
        values = scores[f"test_{metric_name}"]
        summary[f"cv_{metric_name}_mean"] = float(np.mean(values))
        summary[f"cv_{metric_name}_std"] = float(np.std(values))
    return summary


def get_positive_probability(estimator: Pipeline, features: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    if hasattr(estimator, "decision_function"):
        raw_scores = estimator.decision_function(features)
        return 1.0 / (1.0 + np.exp(-raw_scores))
    raise ValueError("Estimator does not support probability or decision scores.")


def calculate_binary_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }


def tune_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    strategy: str,
    min_precision_floor: float,
    default_threshold: float = 0.5,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
) -> float:
    _, _, pr_thresholds = precision_recall_curve(y_true, probabilities)
    candidate_thresholds = np.unique(
        np.clip(np.concatenate(([0.0, default_threshold, 1.0], pr_thresholds)), 0.0, 1.0)
    )
    best_threshold = default_threshold
    best_score = -1.0
    best_precision = -1.0
    best_distance = float("inf")

    for threshold in candidate_thresholds:
        predictions = (probabilities >= threshold).astype(int)
        precision = float(precision_score(y_true, predictions, zero_division=0))
        recall = float(recall_score(y_true, predictions, zero_division=0))
        f1 = float(f1_score(y_true, predictions, zero_division=0))
        balanced_accuracy = float(balanced_accuracy_score(y_true, predictions))
        true_negative = int(((y_true == 0) & (predictions == 0)).sum())
        false_positive = int(((y_true == 0) & (predictions == 1)).sum())
        false_negative = int(((y_true == 1) & (predictions == 0)).sum())
        true_positive = int(((y_true == 1) & (predictions == 1)).sum())
        if precision < min_precision_floor:
            continue

        if strategy == "balanced_accuracy":
            score = balanced_accuracy
        elif strategy == "f1":
            score = f1
        elif strategy == "cost_based":
            total_cost = (false_positive_cost * false_positive) + (false_negative_cost * false_negative)
            score = -float(total_cost)
        elif strategy == "hybrid":
            score = (0.7 * f1) + (0.3 * balanced_accuracy)
        else:
            raise ValueError(f"Unsupported threshold strategy: {strategy}")

        distance = abs(float(threshold) - default_threshold)
        if (
            score > best_score
            or (np.isclose(score, best_score) and precision > best_precision)
            or (np.isclose(score, best_score) and np.isclose(precision, best_precision) and distance < best_distance)
        ):
            best_score = score
            best_threshold = float(threshold)
            best_precision = precision
            best_distance = distance
    return best_threshold


def maybe_calibrate_classifier(
    estimator: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    method: str = "sigmoid",
) -> Pipeline:
    estimator.fit(x_train, y_train)
    base_prob = get_positive_probability(estimator, x_valid)
    base_auc = roc_auc_score(y_valid, base_prob)

    calibrated = Pipeline(
        steps=[
            ("preprocessor", clone(estimator.named_steps["preprocessor"])),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=clone(estimator.named_steps["classifier"]),
                    method=method,
                    cv=3,
                ),
            ),
        ]
    )
    calibrated.fit(x_train, y_train)
    calibrated_prob = get_positive_probability(calibrated, x_valid)
    calibrated_auc = roc_auc_score(y_valid, calibrated_prob)
    return calibrated if calibrated_auc > base_auc else estimator


def evaluate_holdout(
    estimator: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    threshold_strategy: str,
    min_precision_floor: float,
    default_threshold: float = 0.5,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
    calibrate: bool = True,
    calibration_method: str = "sigmoid",
) -> EvaluationResult:
    candidate = clone(estimator)
    if calibrate:
        candidate = maybe_calibrate_classifier(candidate, x_train, y_train, x_valid, y_valid, calibration_method)
    else:
        candidate.fit(x_train, y_train)

    probabilities = get_positive_probability(candidate, x_valid)
    threshold = tune_threshold(
        y_valid,
        probabilities,
        strategy=threshold_strategy,
        min_precision_floor=min_precision_floor,
        default_threshold=default_threshold,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    )
    metrics = calculate_binary_metrics(y_valid, probabilities, threshold)
    matrix = confusion_matrix(y_valid, (probabilities >= threshold).astype(int))
    return EvaluationResult(metrics=metrics, confusion_matrix=matrix, threshold=threshold, classifier=candidate)


def evaluate_on_test(
    estimator: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray]:
    probabilities = get_positive_probability(estimator, x_test)
    metrics = calculate_binary_metrics(y_test, probabilities, threshold)
    matrix = confusion_matrix(y_test, (probabilities >= threshold).astype(int))
    return metrics, matrix


def metrics_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(by=["validation_roc_auc", "validation_f1"], ascending=False)
