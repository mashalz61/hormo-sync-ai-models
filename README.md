# PCOS AI Project

Production-ready machine learning project for predicting:

1. PCOS presence
2. PCOS severity or risk level as Low / Medium / High
3. Probability of developing PCOS when predicted negative
4. Insulin Resistance presence, when a valid clinical target exists
5. Insulin Resistance severity or risk level as Low / Medium / High
6. Probability of developing Insulin Resistance when predicted negative

The project is designed for tabular clinical data loaded from an Excel workbook. It uses modular preprocessing pipelines, multiple model families, cross-validation, holdout evaluation, threshold tuning, optional probability calibration, and a Streamlit interface for local use.

## Important Medical Note

This project is for decision support and educational or research workflows only. It is **not** a medical diagnosis system and should not replace clinical judgment, lab testing, or professional medical evaluation.

## Features

- Robust Excel ingestion with column normalization
- Config-driven target detection and dropped-column control
- Leakage-safe preprocessing with `Pipeline` and `ColumnTransformer`
- Model comparison across:
  - Logistic Regression
  - SVM
  - AdaBoost
  - Random Forest
  - Extra Trees
  - Deep Forest, if the dependency is installed
  - XGBoost, if installed
- Voting ensemble from the top compatible models
- Stratified train/validation/test split
- Stratified 5-fold cross-validation on the training split
- Threshold tuning for recall-sensitive medical screening
- Optional probability calibration when it improves validation ROC-AUC
- Markdown training reports, CSV metrics, and confusion matrix plots
- ROC curve and confusion matrix plots for every evaluated model across train, validation, and test phases
- SHAP-based local feature explanations for saved clinical model artifacts
- Streamlit app for manual input or single-row CSV prediction
- Graceful skip of insulin resistance training when no valid IR target column exists

## Project Layout

```text
pcos_ai_project/
  README.md
  requirements.txt
  .gitignore
  app.py
  configs/
    default.yaml
  data/
    .gitkeep
  models/
    .gitkeep
  reports/
    .gitkeep
  src/
    pcos_ai/
      __init__.py
      config.py
      data_loader.py
      preprocessing.py
      feature_utils.py
      model_factory.py
      train_pcos.py
      train_ir.py
      evaluate.py
      ensemble.py
      predict.py
      api.py
      calorie_predictor.py
      exercise_predictor.py
      utils.py
      plotting.py
  tests/
    test_api.py
    test_preprocessing.py
    test_prediction_schema.py
```

## Setup

### 1. Create a virtual environment

```bash
cd pcos_ai_project
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Optional model backends:

```bash
pip install xgboost
pip install deep-forest
```

If these optional packages are not installed, the project still runs and logs that the related models were skipped.

`shap` is included in `requirements.txt` so local explanation support is installed with the main project dependencies.

## Configuration

Default settings live in [`configs/default.yaml`](/Users/muqeetworkstation/Documents/hormo-sync/bac/pcos_ai_project/configs/default.yaml). You can customize:

- target column aliases
- optional insulin resistance target aliases
- dropped columns
- split sizes
- random seed
- probability threshold tuning preferences
- output locations

## Training The PCOS Model

```bash
python -m src.pcos_ai.train_pcos --data "data/your_file.xlsx" --config configs/default.yaml --output-dir models
```

Training will:

- detect the PCOS target column from common aliases such as `PCOS (Y/N)` or `PCOS`
- preprocess numeric and categorical data safely
- compare all supported models
- run 5-fold stratified CV on the training split
- evaluate on validation and test splits
- tune a recall-sensitive decision threshold
- optionally calibrate probabilities if that helps validation ROC-AUC
- train a voting ensemble from the best compatible models
- save the best final artifact and reports
- store a background sample so the saved model can produce SHAP explanations for individual predictions

Outputs include:

- `models/pcos/best_pcos_model.joblib`
- `reports/pcos/pcos_model_comparison.csv`
- `reports/pcos/pcos_training_report.md`
- `reports/pcos/pcos_shap_summary.csv`
- `reports/pcos/pcos_shap_report.md`
- `reports/pcos/pcos_shap_summary_beeswarm.png`
- `reports/pcos/pcos_shap_summary_bar.png`
- `reports/pcos/pcos_confusion_matrix.png`
- `reports/pcos/pcos_all_models_roc_curve.png`
- `reports/pcos/pcos_plot_manifest.csv`
- per-model train/validation/test ROC and confusion matrix image files under `reports/pcos/`

## Training The Insulin Resistance Model

```bash
python -m src.pcos_ai.train_ir --data "data/your_file.xlsx" --config configs/default.yaml --output-dir models
```

Insulin resistance training only runs if a clinically valid IR target column is configured and present in the dataset.

If the dataset does **not** contain a real IR target column:

- the command exits gracefully
- no IR model is invented from proxy features
- a short Markdown report is written under `reports/insulin_resistance/` explaining why training was skipped

This behavior is intentional to avoid fabricating labels from weak clinical proxies.

## Prediction CLI

Example JSON input:

```bash
python -m src.pcos_ai.predict --model models/best_pcos_model.joblib --input-json '{"Age (yrs)": 28, "Weight (Kg)": 68, "Cycle(R/I)": "I"}'
```

You can also pass a single-row CSV file:

```bash
python -m src.pcos_ai.predict --model models/best_pcos_model.joblib --input-csv data/one_row.csv
```

## Streamlit App

```bash
streamlit run app.py
```

The app supports:

- uploading trained PCOS and optional IR model artifacts
- entering a single patient record manually
- uploading a one-row CSV
- displaying prediction output with probability and Low / Medium / High mapping
- showing `IR model unavailable` if no IR model was trained

## FastAPI Service For React Native

Run the prediction API locally:

```bash
uvicorn src.pcos_ai.api:app --host 0.0.0.0 --port 8000
```

Available endpoints:

- `GET /health`
- `POST /predict/pcos`
- `POST /predict/ir`
- `POST /predict/calories`
- `POST /predict/exercise`

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict/pcos" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Age (yrs)": 26,
      "Weight (Kg)": 82,
      "Cycle(R/I)": 4,
      "Weight gain(Y/N)": 1
    }
  }'
```

When SHAP is available, PCOS and IR prediction responses also include a `shap_explanation` object with the most influential input features for that prediction.

Calorie estimation request with grams:

```bash
curl -X POST "http://127.0.0.1:8000/predict/calories" \
  -H "Content-Type: application/json" \
  -d '{
    "meal_name": "aloo gosht",
    "grams": 200
  }'
```

Calorie estimation request with portion count:

```bash
curl -X POST "http://127.0.0.1:8000/predict/calories" \
  -H "Content-Type: application/json" \
  -d '{
    "meal_name": "aloo gosht",
    "portion_count": 2
  }'
```

Exercise recommendation request by name:

```bash
curl -X POST "http://127.0.0.1:8000/predict/exercise" \
  -H "Content-Type: application/json" \
  -d '{
    "exercise_name": "push ups",
    "duration_minutes": 45
  }'
```

Exercise recommendation request by filters:

```bash
curl -X POST "http://127.0.0.1:8000/predict/exercise" \
  -H "Content-Type: application/json" \
  -d '{
    "difficulty_level": "Beginner",
    "target_muscle_group": "legs",
    "limit": 2
  }'
```

Example React Native call:

```ts
const response = await fetch("http://YOUR_SERVER_IP:8000/predict/pcos", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    features: {
      "Age (yrs)": 26,
      "Weight (Kg)": 82,
      "Cycle(R/I)": 4,
      "Weight gain(Y/N)": 1
    }
  }),
});

const result = await response.json();
```

Notes for React Native:

- Send the same feature names used by the training dataset.
- Partial payloads are supported; missing columns are filled automatically.
- The API normalizes common shorthand inputs such as `"Y"`/`"N"` and `"R"`/`"I"` when possible.
- If no IR model exists, `/predict/ir` returns a clear unavailable response instead of fabricating a result.

## Interpreting Low / Medium / High

The same reusable probability mapping is applied everywhere:

- probability `< 0.33` => `Low`
- probability `0.33` to `< 0.66` => `Medium`
- probability `>= 0.66` => `High`

Interpretation rules:

- If predicted positive, the output represents present-condition probability and severity level.
- If predicted negative, the output represents estimated probability of developing the condition and a corresponding risk level.

## Running Tests

```bash
pytest
```

## Exact Local Commands

```bash
cd pcos_ai_project
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pytest
python -m src.pcos_ai.train_pcos --data "data/your_file.xlsx" --config configs/default.yaml --output-dir models
python -m src.pcos_ai.train_ir --data "data/your_file.xlsx" --config configs/default.yaml --output-dir models
uvicorn src.pcos_ai.api:app --host 0.0.0.0 --port 8000
streamlit run app.py
```
