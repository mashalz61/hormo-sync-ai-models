from __future__ import annotations

import argparse
import logging

from .config import load_config
from .data_loader import load_excel_data
from .feature_utils import find_target_column
from .train_pcos import _run_training
from .utils import configure_logging, ensure_dir, write_markdown


LOGGER = logging.getLogger("pcos_ai.train_ir")


def train_ir(data_path: str, config_path: str, output_dir: str | None = None) -> dict[str, str]:
    logger = configure_logging()
    config = load_config(config_path)
    reports_root = ensure_dir(config.reports_dir)
    reports_dir = ensure_dir(reports_root / "insulin_resistance")
    data_frame = load_excel_data(data_path)
    ir_aliases = config.target_aliases.get("insulin_resistance", [])
    ir_target_column = find_target_column(data_frame.columns, ir_aliases)

    if not ir_target_column:
        reason = (
            "Insulin Resistance training was skipped because no clinically valid IR target column "
            "was found in the dataset using the configured aliases. The project intentionally does "
            "not invent IR labels from proxy features."
        )
        logger.warning(reason)
        report_path = reports_dir / "ir_training_report.md"
        write_markdown(report_path, f"# IR Training Report\n\n- Status: skipped\n- Reason: {reason}\n")
        return {"report_path": str(report_path), "status": "skipped"}

    logger.info("Detected IR target column: %s", ir_target_column)
    return _run_training(
        data_path=data_path,
        config_path=config_path,
        condition_name="insulin_resistance",
        target_aliases=[ir_target_column],
        output_dir=output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train insulin resistance models if a valid target exists.")
    parser.add_argument("--data", required=True, help="Path to the Excel dataset.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config.")
    parser.add_argument("--output-dir", default=None, help="Directory to store trained model artifacts.")
    args = parser.parse_args()
    train_ir(args.data, args.config, args.output_dir)


if __name__ == "__main__":
    main()
