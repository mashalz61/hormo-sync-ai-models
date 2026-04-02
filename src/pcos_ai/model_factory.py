from __future__ import annotations

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def create_models(random_seed: int) -> tuple[dict[str, object], dict[str, str]]:
    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_seed),
        "svm": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_seed),
        "adaboost": AdaBoostClassifier(random_state=random_seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_seed,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=random_seed,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }
    skipped: dict[str, str] = {}

    try:
        from deepforest import CascadeForestClassifier  # type: ignore

        models["deep_forest"] = CascadeForestClassifier(random_state=random_seed)
    except Exception as exc:  # pragma: no cover - optional dependency
        reason = f"Skipped Deep Forest because dependency import failed: {exc}"
        skipped["deep_forest"] = reason

    try:
        from xgboost import XGBClassifier  # type: ignore

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_seed,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        reason = f"Skipped XGBoost because dependency import failed: {exc}"
        skipped["xgboost"] = reason

    return models, skipped
