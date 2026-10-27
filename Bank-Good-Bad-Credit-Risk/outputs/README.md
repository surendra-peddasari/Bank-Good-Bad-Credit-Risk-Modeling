# Outputs

This folder contains **project outputs derived from the executed notebook results**.

## Included files

- `model_metrics.csv` — validation ROC-AUC and Gini for LightGBM, CatBoost, and the final rank blend, plus the business benchmark.
- `validation_deciles.csv` — validation decile population, bad counts, bad rates, and cumulative bad capture.
- `feature_gain_top25.csv` — top 25 positive-gain features reported by the notebook.
- `validation_gini.png` — validation Gini comparison against the 37.90% business benchmark.
- `decile_bad_rate.png` — bad-rate profile across validation risk deciles.
- `cumulative_bad_capture.png` — cumulative capture of bad customers by risk decile.
- `feature_importance_top25.png` — top 25 feature-gain visualization.

## Important

These files are based on results already reported by the notebook. Raw client database extracts are **not** included because the project contains confidential client data.

The notebook's exploratory visualizations that require raw tables are generated when the notebook is executed against an authorized database connection.
