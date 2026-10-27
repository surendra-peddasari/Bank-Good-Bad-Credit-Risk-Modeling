# Bank GoodCredit — Credit Risk & Creditworthiness Prediction

> **Client Internship Project | Banking Risk Analytics | Project Ref: PM-PR-0015**

An end-to-end credit scoring and risk-ranking pipeline developed for a banking risk use case. The project integrates customer demographics, historical credit accounts/payment behavior, and recent credit enquiries to build a customer-level feature matrix and rank customers by credit risk.

## Project objective

The target variable is `Bad_label`:
- `0` — Good credit history
- `1` — Bad credit history / 30+ DPD bucket

The stated benchmark is a Gini score of **37.90%**. The final rank-blended ensemble achieved **39.25% Gini** on the held-out validation set.

## Repository structure

```text
bank-goodcredit-credit-risk/
├── README.md
├── SECURITY.md
├── requirements.txt
├── .gitignore
├── config/
│   └── .env.example
├── data/
│   └── README.md
├── docs/
│   └── data_dictionary.md
├── notebooks/
│   └── BankGoodCredit_Credit_Risk_Pipeline_SANITIZED.ipynb
├── reports/
│   ├── model_evaluation.md
│   └── risks_and_challenges.md
├── src/
│   ├── 01_database_ingestion.py
│   ├── 02_eda_data_audit.py
│   ├── 03_feature_engineering.py
│   ├── 04_feature_preparation.py
│   ├── 05_model_training_evaluation.py
│   └── 06_decile_rank_ordering.py
└── outputs/
```

## Data sources

The pipeline works with three relational tables:
1. `Cust_Account` — historical accounts and payment behavior.
2. `Cust_Enquiry` — historical credit enquiries.
3. `Cust_Demographics` — current customer/demographic application data.

Raw client data is intentionally excluded from this repository.

## End-to-end workflow

### 1. Database connection & ingestion
- Connects to MySQL using `pymysql`.
- Loads the three source tables.
- Includes retry handling for MySQL connection saturation (`1040`).

### 2. EDA & data audit
The notebook audits:
- Dataset dimensions
- Unique customers
- Target distribution
- Missingness
- Numeric-string columns
- Class imbalance

Observed project scale:
- `Cust_Account`: 186,329 × 21
- `Cust_Enquiry`: 413,188 × 6
- `Cust_Demographics`: 23,896 × 83
- Customers: 23,896
- Bad customers: 1,004
- Bad rate: 4.20%

### 3. Domain feature engineering

The project converts transaction/account/enquiry history into customer-level risk indicators.

#### Account features
Examples include:
- Account counts and balances
- Credit limits and cash limits
- Past-due indicators and ratios
- Account age
- Time since last payment
- Utilization metrics
- Payment-history length
- Account opening velocity
- Payment-history delinquency statistics

#### Payment-history parsing
The pipeline parses the two payment-history fields into monthly tokens and derives delinquency indicators for thresholds such as 1, 30, 60 and 90+ DPD, plus rolling windows.

#### Enquiry features
Examples include:
- Enquiry counts and amounts
- 30/90/180/365/730-day windows
- Recent enquiry velocity
- Enquiry purpose distributions
- Recency statistics

#### Demographic/application features
The pipeline derives:
- Numeric/date features
- Age and date-gap variables
- Sparse-block presence flags
- Ratios
- Matching flags
- Selected categorical representations

### 4. Customer-level feature matrix
The engineered master matrix contains **291 features** across **23,896 customers**.

### 5. Train-validation strategy
- 75% training / 25% validation
- Stratified split
- `random_state=42`
- 17,922 training rows
- 5,974 validation rows
- 5-fold stratified CV performed only within the training data

### 6. Ensemble modeling

#### LightGBM
Regularized binary classifier with shallow tree structure and native categorical handling.

#### CatBoost
Regularized binary classifier with native categorical handling.

#### Rank blend
Final score:

`0.7 × LightGBM rank + 0.3 × CatBoost rank`

This approach focuses on stable customer risk ordering rather than only raw probability calibration.

## Model performance

### Cross-validation

| Model | CV Gini |
|---|---:|
| LightGBM | 38.65% |
| CatBoost | 37.00% |
| Rank Blend | 38.66% |

### Final validation

| Model | ROC-AUC | Gini |
|---|---:|---:|
| LightGBM | 0.6939 | 38.79% |
| CatBoost | 0.6942 | 38.85% |
| **Final Rank Blend** | **0.6963** | **39.25%** |

**Result:** the final ensemble exceeds the stated benchmark of 37.90% Gini.

## Risk rank ordering

The final validation population is divided into ten risk deciles.

- **Decile 10:** 9.53% bad rate; 22.71% of all bad customers captured.
- **Top 3 deciles:** approximately **56.6%** of bad customers captured.

This demonstrates that the model can prioritize higher-risk customers for risk-management workflows.

## Feature importance

The final model produced positive gain for **214 features**.

Top-ranked features by gain include:
- `feature_7`
- `acc_x_enq`
- `feature_3`
- `acc_bal_hc_mean`
- `feature_38_title`
- `h_any_ge1_mean`
- `enq_p10_perc`
- `type10_n`
- `acc_util_max`
- `enq_n_90`

Because several source demographic fields are anonymized, the repository preserves their original feature names rather than inventing business meanings.

## Key project challenges

- Strong class imbalance (4.20% bad rate)
- Numeric and date values arriving as strings
- Complex payment-history token parsing
- Sparse/anonymized demographic fields
- MySQL connection saturation
- Need to prevent validation leakage and overfitting

## Running the project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure database credentials securely

Set environment variables:

```text
DB_HOST
DB_PORT
DB_USER
DB_PASS
DB_NAME
```

Never place real credentials in the notebook, source code, `.env` file committed to Git, or README.

### 3. Execute the notebook

Open:

`notebooks/BankGoodCredit_Credit_Risk_Pipeline_SANITIZED.ipynb`

Run the workflow in its documented order.

The files under `src/` mirror the major notebook stages for repository organization and code review.

## Outputs

Expected analytical outputs include:
- Customer-level feature matrix
- Model predictions
- Feature-gain table
- Cross-validation metrics
- Final validation metrics
- Decile rank-ordering table

Generated artifacts should remain local unless they are approved for publication.

## Confidentiality note

This is a client internship project. The repository intentionally excludes:
- Database credentials
- Raw customer-level data
- Database dumps
- Confidential source documents

Before making the repository public, confirm that the client/project name, model metrics, feature descriptions, and implementation details are authorized for public disclosure.

## Project outcome

The project delivers an end-to-end, leakage-conscious credit-risk ranking workflow: relational data ingestion → audit → domain feature engineering → customer-level modeling → ensemble scoring → decile risk segmentation.

The final rank-blended model achieved **39.25% Gini**, exceeding the stated **37.90% benchmark** on the held-out validation set.
