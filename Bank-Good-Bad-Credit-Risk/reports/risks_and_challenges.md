# Risks & Challenges

## Class imbalance
The project has a 4.20% bad-label rate, with 1,004 bad customers among 23,896 customers. This makes accuracy a poor primary metric for risk ranking.

## Data quality and parsing
Source columns arrive as strings. Payment history requires token parsing, and date/numeric conversion is required before feature engineering.

## Database infrastructure
The notebook includes retry handling for MySQL error 1040 (`Too many connections`). Credentials are intentionally excluded from this repository.

## Overfitting and leakage control
The validation set is held out during cross-validation. Model selection and tuning use the training portion, with the final validation score calculated only after final fitting.
