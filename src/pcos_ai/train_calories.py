from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .calorie_predictor import CaloriePredictor
from .utils import configure_logging, ensure_dir


LOGGER = logging.getLogger("pcos_ai.train_calories")


def _default_data_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "calory_predictor"
    for candidate in ("meal.csv", "meals.csv"):
        path = data_dir / candidate
        if path.exists():
            return path
    return data_dir / "meal.csv"


def _default_output_path(output_dir: str | None = None) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    models_root = ensure_dir(output_dir or (project_root / "models"))
    models_dir = ensure_dir(models_root / "calory_predictor")
    return models_dir / "calorie_predictor.joblib"


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Train and save the meal calorie predictor.")
    parser.add_argument("--data", default=str(_default_data_path()), help="Path to meal CSV data.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output joblib path. Defaults to models/calory_predictor/calorie_predictor.joblib.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Models root directory (used only when --output is not provided).",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Meal dataset not found: {data_path}")

    output_path = Path(args.output) if args.output else _default_output_path(args.output_dir)
    ensure_dir(output_path.parent)

    LOGGER.info("Building calorie predictor from %s", data_path)
    predictor = CaloriePredictor.from_csv(data_path)
    predictor.to_joblib(output_path)
    LOGGER.info("Saved calorie predictor to %s", output_path)


if __name__ == "__main__":
    main()

