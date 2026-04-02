from __future__ import annotations

from typing import Iterable

from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline


def build_voting_ensemble(
    ranked_estimators: Iterable[tuple[str, Pipeline]],
    top_k: int = 3,
) -> Pipeline | None:
    compatible: list[tuple[str, object]] = []
    template_pipeline: Pipeline | None = None
    for name, pipeline in ranked_estimators:
        classifier = pipeline.named_steps["classifier"]
        if hasattr(classifier, "predict_proba"):
            compatible.append((name, clone(classifier)))
            template_pipeline = pipeline
        if len(compatible) >= top_k:
            break

    if len(compatible) < 2 or template_pipeline is None:
        return None

    ensemble = VotingClassifier(estimators=compatible, voting="soft")
    return Pipeline(
        steps=[
            ("preprocessor", clone(template_pipeline.named_steps["preprocessor"])),
            ("classifier", ensemble),
        ]
    )

