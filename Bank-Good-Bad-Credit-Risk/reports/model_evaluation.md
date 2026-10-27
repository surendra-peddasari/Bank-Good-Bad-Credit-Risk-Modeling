# Model Evaluation

## Validation design
- Stratified 75/25 train-validation split with `random_state=42`.
- 5-fold stratified cross-validation is performed only inside the training split.
- The validation split remains untouched until final scoring.
- Gini is calculated as `(2 × ROC-AUC − 1) × 100`.

## Cross-validation results

| Model | CV Gini |
|---|---:|
| LightGBM | 38.65% |
| CatBoost | 37.00% |
| Rank Blend | 38.66% |

## Final validation results

| Model | ROC-AUC | Gini |
|---|---:|---:|
| LightGBM | 0.6939 | 38.79% |
| CatBoost | 0.6942 | 38.85% |
| Final rank blend | 0.6963 | **39.25%** |

The final rank-blended ensemble exceeds the stated benchmark Gini of 37.90%.

## Rank-ordering result

The highest-risk decile has a 9.53% bad rate and captures 22.71% of all bad customers. The top three deciles together capture approximately 56.6% of bad customers.
