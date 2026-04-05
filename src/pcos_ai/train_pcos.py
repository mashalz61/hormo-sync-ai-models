from __future__ import annotations

import argparse
import logging

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import load_config
from .data_loader import load_excel_data
from .ensemble import build_voting_ensemble
from .explainability import compute_global_shap_explanation, summarize_shap_importance
from .evaluate import (
    cross_validate_model,
    evaluate_holdout,
    evaluate_on_test,
    get_positive_probability,
    metrics_to_frame,
)
from .feature_utils import find_target_column, prepare_feature_frame
from .model_factory import create_models
from .plotting import save_combined_roc_plot, save_confusion_matrix_plot, save_roc_curve_plot, save_shap_summary_plots
from .preprocessing import build_preprocessor
from .utils import configure_logging, ensure_dir, write_markdown


LOGGER = logging.getLogger("pcos_ai.train_pcos")
EXPLAINABILITY_BACKGROUND_ROWS = 50


def _run_training(
    data_path: str,
    config_path: str,
    condition_name: str,
    target_aliases: list[str],
    output_dir: str | None = None,
) -> dict[str, str]:
    logger = configure_logging()
    config = load_config(config_path)
    models_root = ensure_dir(output_dir or config.models_dir)
    reports_root = ensure_dir(config.reports_dir)
    models_dir = ensure_dir(models_root / condition_name)
    reports_dir = ensure_dir(reports_root / condition_name)

    data_frame = load_excel_data(data_path)
    target_column = find_target_column(data_frame.columns, target_aliases)
    if not target_column:
        raise ValueError(f"Could not find a target column for {condition_name} using configured aliases.")

    logger.info("Detected %s target column: %s", condition_name.upper(), target_column)
    features, target = prepare_feature_frame(data_frame, config, target_column=target_column)
    if target is None:
        raise ValueError(f"{condition_name.upper()} target could not be prepared.")

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features,
        target,
        test_size=config.training["test_size"],
        stratify=target,
        random_state=config.random_seed,
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_full,
        y_train_full,
        test_size=config.training["validation_size"],
        stratify=y_train_full,
        random_state=config.random_seed,
    )

    model_objects, skipped_optional = create_models(config.random_seed)
    threshold_config = config.training.get("cost_based", {})
    false_positive_cost = float(threshold_config.get("false_positive_cost", 1.0))
    false_negative_cost = float(threshold_config.get("false_negative_cost", 1.0))
    model_results: list[dict[str, float | str]] = []
    ranked_estimators: list[tuple[str, Pipeline]] = []
    best_bundle: dict[str, object] | None = None
    roc_entries: list[dict[str, object]] = []
    per_model_plot_lines: list[str] = []
    plot_manifest_rows: list[dict[str, str]] = []

    def _save_phase_plots(
        *,
        estimator: Pipeline,
        model_name: str,
        phase_name: str,
        x_phase,
        y_phase,
        threshold: float,
    ) -> tuple[str, str]:
        probabilities = get_positive_probability(estimator, x_phase)
        _, confusion_matrix = evaluate_on_test(estimator, x_phase, y_phase, threshold)
        roc_auc_value = float(roc_auc_score(y_phase, probabilities))
        roc_plot_path = reports_dir / f"{condition_name}_{model_name}_{phase_name}_roc_curve.png"
        confusion_plot_path = reports_dir / f"{condition_name}_{model_name}_{phase_name}_confusion_matrix.png"
        save_roc_curve_plot(
            y_phase.to_numpy(),
            probabilities,
            roc_plot_path,
            title=f"{condition_name.upper()} ROC Curve: {model_name} ({phase_name})",
            roc_auc=roc_auc_value,
        )
        save_confusion_matrix_plot(
            confusion_matrix,
            confusion_plot_path,
            title=f"{condition_name.upper()} Confusion Matrix: {model_name} ({phase_name})",
        )
        return str(roc_plot_path), str(confusion_plot_path)

    for model_name, classifier in model_objects.items():
        logger.info("Training model: %s", model_name)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                ("classifier", classifier),
            ]
        )
        cv_metrics = cross_validate_model(clone(pipeline), x_train, y_train, config.training["cv_folds"])
        validation_result = evaluate_holdout(
            pipeline,
            x_train,
            y_train,
            x_valid,
            y_valid,
            threshold_strategy=str(config.training["threshold_strategy"]),
            min_precision_floor=float(config.training["min_precision_floor"]),
            default_threshold=float(config.training["default_threshold"]),
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
            calibrate=True,
            calibration_method=str(config.training["calibration_method"]),
        )
        test_metrics, _ = evaluate_on_test(
            validation_result.classifier,
            x_test,
            y_test,
            validation_result.threshold,
        )
        test_probabilities = get_positive_probability(validation_result.classifier, x_test)

        row: dict[str, float | str] = {
            "model": model_name,
            "validation_threshold": validation_result.threshold,
            "validation_roc_auc": validation_result.metrics["roc_auc"],
            "validation_accuracy": validation_result.metrics["accuracy"],
            "validation_precision": validation_result.metrics["precision"],
            "validation_recall": validation_result.metrics["recall"],
            "validation_f1": validation_result.metrics["f1"],
            "validation_pr_auc": validation_result.metrics["pr_auc"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_pr_auc": test_metrics["pr_auc"],
        }
        row.update(cv_metrics)
        model_results.append(row)
        ranked_estimators.append((model_name, validation_result.classifier))
        roc_entries.append(
            {
                "model": model_name,
                "y_true": y_test.to_numpy(),
                "probabilities": test_probabilities,
                "roc_auc": test_metrics["roc_auc"],
            }
        )
        train_roc_path, train_conf_path = _save_phase_plots(
            estimator=validation_result.classifier,
            model_name=model_name,
            phase_name="train",
            x_phase=x_train,
            y_phase=y_train,
            threshold=validation_result.threshold,
        )
        valid_roc_path, valid_conf_path = _save_phase_plots(
            estimator=validation_result.classifier,
            model_name=model_name,
            phase_name="validation",
            x_phase=x_valid,
            y_phase=y_valid,
            threshold=validation_result.threshold,
        )
        test_roc_path, test_conf_path = _save_phase_plots(
            estimator=validation_result.classifier,
            model_name=model_name,
            phase_name="test",
            x_phase=x_test,
            y_phase=y_test,
            threshold=validation_result.threshold,
        )
        per_model_plot_lines.append(
            f"- `{model_name}`: "
            f"train ROC `{train_roc_path}`, train confusion `{train_conf_path}`; "
            f"validation ROC `{valid_roc_path}`, validation confusion `{valid_conf_path}`; "
            f"test ROC `{test_roc_path}`, test confusion `{test_conf_path}`"
        )
        plot_manifest_rows.append(
            {
                "model": model_name,
                "train_roc_curve": train_roc_path,
                "train_confusion_matrix": train_conf_path,
                "validation_roc_curve": valid_roc_path,
                "validation_confusion_matrix": valid_conf_path,
                "test_roc_curve": test_roc_path,
                "test_confusion_matrix": test_conf_path,
            }
        )

        if best_bundle is None or row["validation_roc_auc"] > best_bundle["validation_roc_auc"]:
            best_bundle = {
                "model_name": model_name,
                "validation_roc_auc": row["validation_roc_auc"],
                "threshold": validation_result.threshold,
                "pipeline": validation_result.classifier,
                "validation_confusion_matrix": validation_result.confusion_matrix,
            }

    ranked_estimators.sort(
        key=lambda item: next(row["validation_roc_auc"] for row in model_results if row["model"] == item[0]),
        reverse=True,
    )
    ensemble_pipeline = build_voting_ensemble(ranked_estimators)
    if ensemble_pipeline is not None:
        logger.info("Training voting ensemble from top compatible models.")
        ensemble_result = evaluate_holdout(
            ensemble_pipeline,
            x_train,
            y_train,
            x_valid,
            y_valid,
            threshold_strategy=str(config.training["threshold_strategy"]),
            min_precision_floor=float(config.training["min_precision_floor"]),
            default_threshold=float(config.training["default_threshold"]),
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
            calibrate=False,
        )
        ensemble_test_metrics, _ = evaluate_on_test(
            ensemble_result.classifier,
            x_test,
            y_test,
            ensemble_result.threshold,
        )
        ensemble_test_probabilities = get_positive_probability(ensemble_result.classifier, x_test)
        ensemble_row = {
            "model": "voting_ensemble",
            "validation_threshold": ensemble_result.threshold,
            "validation_roc_auc": ensemble_result.metrics["roc_auc"],
            "validation_accuracy": ensemble_result.metrics["accuracy"],
            "validation_precision": ensemble_result.metrics["precision"],
            "validation_recall": ensemble_result.metrics["recall"],
            "validation_f1": ensemble_result.metrics["f1"],
            "validation_pr_auc": ensemble_result.metrics["pr_auc"],
            "test_roc_auc": ensemble_test_metrics["roc_auc"],
            "test_accuracy": ensemble_test_metrics["accuracy"],
            "test_precision": ensemble_test_metrics["precision"],
            "test_recall": ensemble_test_metrics["recall"],
            "test_f1": ensemble_test_metrics["f1"],
            "test_pr_auc": ensemble_test_metrics["pr_auc"],
        }
        model_results.append(ensemble_row)
        roc_entries.append(
            {
                "model": "voting_ensemble",
                "y_true": y_test.to_numpy(),
                "probabilities": ensemble_test_probabilities,
                "roc_auc": ensemble_test_metrics["roc_auc"],
            }
        )
        ensemble_train_roc, ensemble_train_conf = _save_phase_plots(
            estimator=ensemble_result.classifier,
            model_name="voting_ensemble",
            phase_name="train",
            x_phase=x_train,
            y_phase=y_train,
            threshold=ensemble_result.threshold,
        )
        ensemble_valid_roc, ensemble_valid_conf = _save_phase_plots(
            estimator=ensemble_result.classifier,
            model_name="voting_ensemble",
            phase_name="validation",
            x_phase=x_valid,
            y_phase=y_valid,
            threshold=ensemble_result.threshold,
        )
        ensemble_test_roc, ensemble_test_conf = _save_phase_plots(
            estimator=ensemble_result.classifier,
            model_name="voting_ensemble",
            phase_name="test",
            x_phase=x_test,
            y_phase=y_test,
            threshold=ensemble_result.threshold,
        )
        per_model_plot_lines.append(
            f"- `voting_ensemble`: "
            f"train ROC `{ensemble_train_roc}`, train confusion `{ensemble_train_conf}`; "
            f"validation ROC `{ensemble_valid_roc}`, validation confusion `{ensemble_valid_conf}`; "
            f"test ROC `{ensemble_test_roc}`, test confusion `{ensemble_test_conf}`"
        )
        plot_manifest_rows.append(
            {
                "model": "voting_ensemble",
                "train_roc_curve": ensemble_train_roc,
                "train_confusion_matrix": ensemble_train_conf,
                "validation_roc_curve": ensemble_valid_roc,
                "validation_confusion_matrix": ensemble_valid_conf,
                "test_roc_curve": ensemble_test_roc,
                "test_confusion_matrix": ensemble_test_conf,
            }
        )
        if best_bundle is None or ensemble_row["validation_roc_auc"] > float(best_bundle["validation_roc_auc"]):
            best_bundle = {
                "model_name": "voting_ensemble",
                "validation_roc_auc": ensemble_row["validation_roc_auc"],
                "threshold": ensemble_result.threshold,
                "pipeline": ensemble_result.classifier,
                "validation_confusion_matrix": ensemble_result.confusion_matrix,
            }
    else:
        logger.info("Voting ensemble skipped because fewer than two compatible probabilistic models were available.")

    if best_bundle is None:
        raise RuntimeError(f"No {condition_name.upper()} model could be trained.")

    comparison_frame = metrics_to_frame(model_results)
    comparison_csv_path = reports_dir / f"{condition_name}_model_comparison.csv"
    comparison_frame.to_csv(comparison_csv_path, index=False)
    plot_manifest_path = reports_dir / f"{condition_name}_plot_manifest.csv"
    pd.DataFrame(plot_manifest_rows).to_csv(plot_manifest_path, index=False)

    best_model_path = models_dir / f"best_{condition_name}_model.joblib"
    artifact = {
        "condition_name": condition_name,
        "pipeline": best_bundle["pipeline"],
        "threshold": best_bundle["threshold"],
        "model_name": best_bundle["model_name"],
        "target_column": target_column,
        "skipped_optional_models": skipped_optional,
        "feature_columns": list(features.columns),
        "explainability_background": x_train.sample(
            n=min(len(x_train), EXPLAINABILITY_BACKGROUND_ROWS),
            random_state=config.random_seed,
        ).reset_index(drop=True),
    }
    joblib.dump(artifact, best_model_path)

    shap_summary_csv_path = reports_dir / f"{condition_name}_shap_summary.csv"
    shap_summary_plot_path = reports_dir / f"{condition_name}_shap_summary_beeswarm.png"
    shap_bar_plot_path = reports_dir / f"{condition_name}_shap_summary_bar.png"
    shap_report_path = reports_dir / f"{condition_name}_shap_report.md"
    shap_report_section = (
        "## SHAP Explainability\n\n"
        "- Status: unavailable\n"
        "- Reason: SHAP summary generation was not attempted.\n"
    )
    try:
        shap_explanation, shap_evaluation_frame = compute_global_shap_explanation(
            artifact,
            x_test.reset_index(drop=True),
        )
        shap_summary = summarize_shap_importance(
            artifact,
            x_test.reset_index(drop=True),
        )
        shap_summary.to_csv(shap_summary_csv_path, index=False)
        save_shap_summary_plots(
            shap_explanation,
            shap_summary_plot_path,
            shap_bar_plot_path,
        )
        top_feature_lines = "\n".join(
            f"- `{row.feature}`: mean |SHAP| `{row.mean_abs_shap:.6f}`, mean SHAP `{row.mean_shap:.6f}`"
            for row in shap_summary.head(10).itertuples(index=False)
        )
        shap_report = f"""# {condition_name.upper()} SHAP Report

## Summary

- Source model: `{best_model_path}`
- Evaluation rows summarized: `{len(shap_evaluation_frame)}`
- Summary CSV: `{shap_summary_csv_path}`
- SHAP beeswarm plot: `{shap_summary_plot_path}`
- SHAP bar plot: `{shap_bar_plot_path}`

## Top Features

{top_feature_lines}
"""
        write_markdown(shap_report_path, shap_report)
        shap_report_section = f"""## SHAP Explainability

- Status: available
- Summary CSV: `{shap_summary_csv_path}`
- SHAP beeswarm plot: `{shap_summary_plot_path}`
- SHAP bar plot: `{shap_bar_plot_path}`
- Detailed report: `{shap_report_path}`
"""
    except Exception as exc:
        logger.warning("Skipping SHAP report generation for %s: %s", condition_name.upper(), exc)
        write_markdown(
            shap_report_path,
            f"# {condition_name.upper()} SHAP Report\n\n- Status: unavailable\n- Reason: {exc}\n",
        )
        shap_report_section = f"""## SHAP Explainability

- Status: unavailable
- Reason: `{exc}`
- Detailed report: `{shap_report_path}`
"""

    confusion_matrix_path = reports_dir / f"{condition_name}_confusion_matrix.png"
    save_confusion_matrix_plot(
        best_bundle["validation_confusion_matrix"],
        confusion_matrix_path,
        title=f"{condition_name.upper()} Confusion Matrix: {best_bundle['model_name']}",
    )

    included_models = ", ".join(model_objects.keys())
    skipped_lines = "\n".join(f"- {name}: {reason}" for name, reason in skipped_optional.items()) or "- None"
    combined_roc_path = reports_dir / f"{condition_name}_all_models_roc_curve.png"
    save_combined_roc_plot(
        roc_entries,
        combined_roc_path,
        title=f"{condition_name.upper()} ROC Curves: All Models",
    )
    per_model_plots = "\n".join(per_model_plot_lines)
    report = f"""# {condition_name.upper()} Training Report

## Summary

- Input data: `{data_path}`
- Detected target: `{target_column}`
- Final selected model: `{best_bundle['model_name']}`
- Validation ROC-AUC: `{float(best_bundle['validation_roc_auc']):.4f}`
- Saved model: `{best_model_path}`
- Metrics CSV: `{comparison_csv_path}`
- Plot manifest CSV: `{plot_manifest_path}`
- Confusion matrix: `{confusion_matrix_path}`
- Combined ROC plot: `{combined_roc_path}`

## Included Models

{included_models}

## Skipped Optional Models

{skipped_lines}

## Per-Model Plots

{per_model_plots}

{shap_report_section}

## Notes

- Threshold tuning used the `{config.training["threshold_strategy"]}` strategy with default threshold `{config.training["default_threshold"]}`.
- Cost-based tuning weights: false positive `{false_positive_cost}`, false negative `{false_negative_cost}`.
- Probability calibration was attempted for individual base models when beneficial on validation ROC-AUC.
- Low / Medium / High mapping uses fixed probability cutoffs of 0.33 and 0.66.
"""
    report_path = reports_dir / f"{condition_name}_training_report.md"
    write_markdown(report_path, report)

    logger.info("Best %s model: %s", condition_name.upper(), best_bundle["model_name"])
    logger.info("Validation ROC-AUC: %.4f", float(best_bundle["validation_roc_auc"]))
    logger.info("Saved %s artifact to %s", condition_name.upper(), best_model_path)

    return {
        "model_path": str(best_model_path),
        "report_path": str(report_path),
        "comparison_csv_path": str(comparison_csv_path),
        "plot_manifest_path": str(plot_manifest_path),
        "confusion_matrix_path": str(confusion_matrix_path),
        "shap_summary_csv_path": str(shap_summary_csv_path),
        "shap_summary_plot_path": str(shap_summary_plot_path),
        "shap_bar_plot_path": str(shap_bar_plot_path),
        "shap_report_path": str(shap_report_path),
        "models_dir": str(models_dir),
        "reports_dir": str(reports_dir),
    }


def train_pcos(data_path: str, config_path: str, output_dir: str | None = None) -> dict[str, str]:
    config = load_config(config_path)
    return _run_training(
        data_path=data_path,
        config_path=config_path,
        condition_name="pcos",
        target_aliases=config.target_aliases["pcos"],
        output_dir=output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PCOS prediction models from an Excel dataset.")
    parser.add_argument("--data", required=True, help="Path to the Excel dataset.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config.")
    parser.add_argument("--output-dir", default=None, help="Directory to store trained model artifacts.")
    args = parser.parse_args()
    train_pcos(args.data, args.config, args.output_dir)


if __name__ == "__main__":
    main()
