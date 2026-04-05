# PCOS SHAP Report

## Summary

- Source model: `models/pcos/best_pcos_model.joblib`
- Evaluation rows summarized: `100`
- Summary CSV: `reports/pcos/pcos_shap_summary.csv`
- SHAP beeswarm plot: `reports/pcos/pcos_shap_summary_beeswarm.png`
- SHAP bar plot: `reports/pcos/pcos_shap_summary_bar.png`

## Top Features

- `Follicle No. (R)`: mean |SHAP| `0.082717`, mean SHAP `-0.022294`
- `Skin darkening (Y/N)`: mean |SHAP| `0.069949`, mean SHAP `-0.009528`
- `Weight gain(Y/N)`: mean |SHAP| `0.063932`, mean SHAP `-0.013880`
- `Follicle No. (L)`: mean |SHAP| `0.058807`, mean SHAP `-0.009988`
- `hair growth(Y/N)`: mean |SHAP| `0.056484`, mean SHAP `0.008845`
- `Cycle(R/I)`: mean |SHAP| `0.038443`, mean SHAP `-0.017203`
- `Fast food (Y/N)`: mean |SHAP| `0.030029`, mean SHAP `-0.001004`
- `Pimples(Y/N)`: mean |SHAP| `0.022057`, mean SHAP `-0.003190`
- `Cycle length(days)`: mean |SHAP| `0.012079`, mean SHAP `-0.002341`
- `Reg.Exercise(Y/N)`: mean |SHAP| `0.011747`, mean SHAP `-0.004202`
