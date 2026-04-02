# PCOS Training Report

## Summary

- Input data: `data/pcos/PCOS_data.xlsx`
- Detected target: `PCOS (Y/N)`
- Final selected model: `extra_trees`
- Validation ROC-AUC: `0.9782`
- Saved model: `models/pcos/best_pcos_model.joblib`
- Metrics CSV: `reports/pcos/pcos_model_comparison.csv`
- Plot manifest CSV: `reports/pcos/pcos_plot_manifest.csv`
- Confusion matrix: `reports/pcos/pcos_confusion_matrix.png`
- Combined ROC plot: `reports/pcos/pcos_all_models_roc_curve.png`

## Included Models

logistic_regression, svm, adaboost, random_forest, extra_trees

## Skipped Optional Models

- deep_forest: Skipped Deep Forest because dependency import failed: No module named 'deepforest'
- xgboost: Skipped XGBoost because dependency import failed: No module named 'xgboost'

## Per-Model Plots

- `logistic_regression`: train ROC `reports/pcos/pcos_logistic_regression_train_roc_curve.png`, train confusion `reports/pcos/pcos_logistic_regression_train_confusion_matrix.png`; validation ROC `reports/pcos/pcos_logistic_regression_validation_roc_curve.png`, validation confusion `reports/pcos/pcos_logistic_regression_validation_confusion_matrix.png`; test ROC `reports/pcos/pcos_logistic_regression_test_roc_curve.png`, test confusion `reports/pcos/pcos_logistic_regression_test_confusion_matrix.png`
- `svm`: train ROC `reports/pcos/pcos_svm_train_roc_curve.png`, train confusion `reports/pcos/pcos_svm_train_confusion_matrix.png`; validation ROC `reports/pcos/pcos_svm_validation_roc_curve.png`, validation confusion `reports/pcos/pcos_svm_validation_confusion_matrix.png`; test ROC `reports/pcos/pcos_svm_test_roc_curve.png`, test confusion `reports/pcos/pcos_svm_test_confusion_matrix.png`
- `adaboost`: train ROC `reports/pcos/pcos_adaboost_train_roc_curve.png`, train confusion `reports/pcos/pcos_adaboost_train_confusion_matrix.png`; validation ROC `reports/pcos/pcos_adaboost_validation_roc_curve.png`, validation confusion `reports/pcos/pcos_adaboost_validation_confusion_matrix.png`; test ROC `reports/pcos/pcos_adaboost_test_roc_curve.png`, test confusion `reports/pcos/pcos_adaboost_test_confusion_matrix.png`
- `random_forest`: train ROC `reports/pcos/pcos_random_forest_train_roc_curve.png`, train confusion `reports/pcos/pcos_random_forest_train_confusion_matrix.png`; validation ROC `reports/pcos/pcos_random_forest_validation_roc_curve.png`, validation confusion `reports/pcos/pcos_random_forest_validation_confusion_matrix.png`; test ROC `reports/pcos/pcos_random_forest_test_roc_curve.png`, test confusion `reports/pcos/pcos_random_forest_test_confusion_matrix.png`
- `extra_trees`: train ROC `reports/pcos/pcos_extra_trees_train_roc_curve.png`, train confusion `reports/pcos/pcos_extra_trees_train_confusion_matrix.png`; validation ROC `reports/pcos/pcos_extra_trees_validation_roc_curve.png`, validation confusion `reports/pcos/pcos_extra_trees_validation_confusion_matrix.png`; test ROC `reports/pcos/pcos_extra_trees_test_roc_curve.png`, test confusion `reports/pcos/pcos_extra_trees_test_confusion_matrix.png`
- `voting_ensemble`: train ROC `reports/pcos/pcos_voting_ensemble_train_roc_curve.png`, train confusion `reports/pcos/pcos_voting_ensemble_train_confusion_matrix.png`; validation ROC `reports/pcos/pcos_voting_ensemble_validation_roc_curve.png`, validation confusion `reports/pcos/pcos_voting_ensemble_validation_confusion_matrix.png`; test ROC `reports/pcos/pcos_voting_ensemble_test_roc_curve.png`, test confusion `reports/pcos/pcos_voting_ensemble_test_confusion_matrix.png`

## Notes

- Threshold tuning used the `f1` strategy with default threshold `0.5`.
- Cost-based tuning weights: false positive `1.0`, false negative `2.0`.
- Probability calibration was attempted for individual base models when beneficial on validation ROC-AUC.
- Low / Medium / High mapping uses fixed probability cutoffs of 0.33 and 0.66.
