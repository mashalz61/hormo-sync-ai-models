# PCOS SHAP Report

## Summary

- Source model: `models/pcos/best_pcos_model.joblib`
- Evaluation rows summarized: `100`
- Summary CSV: `reports/pcos/pcos_shap_summary.csv`
- SHAP beeswarm plot: `reports/pcos/pcos_shap_summary_beeswarm.png`
- SHAP bar plot: `reports/pcos/pcos_shap_summary_bar.png`

## Top Features

- `Follicle No. (R)`: mean |SHAP| `0.083562`, mean SHAP `-0.021304`
- `Skin darkening (Y/N)`: mean |SHAP| `0.066540`, mean SHAP `-0.017119`
- `Weight gain(Y/N)`: mean |SHAP| `0.063231`, mean SHAP `-0.010431`
- `Follicle No. (L)`: mean |SHAP| `0.058159`, mean SHAP `-0.008131`
- `hair growth(Y/N)`: mean |SHAP| `0.053954`, mean SHAP `0.007111`
- `Cycle(R/I)`: mean |SHAP| `0.041174`, mean SHAP `-0.015619`
- `Fast food (Y/N)`: mean |SHAP| `0.029192`, mean SHAP `-0.000941`
- `Pimples(Y/N)`: mean |SHAP| `0.020560`, mean SHAP `-0.003270`
- `Cycle length(days)`: mean |SHAP| `0.013049`, mean SHAP `-0.003132`
- `Reg.Exercise(Y/N)`: mean |SHAP| `0.012359`, mean SHAP `-0.002601`
