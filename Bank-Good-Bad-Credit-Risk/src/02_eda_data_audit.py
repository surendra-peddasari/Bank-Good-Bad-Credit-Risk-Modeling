"""
Bank GoodCredit — Exploratory Data Analysis & Data Audit

Extracted from the sanitized project notebook.
Execution is intended in the same stage order as the notebook; shared variables are produced by earlier stages.
No database credentials or raw client data are included.
"""

# --- Notebook cell 6 ---
print(f'Cust_Account Shape: {df_account.shape}')
print(f'Cust_Enquiry Shape: {df_enquiry.shape}')
print(f'Cust_Demographics Shape: {df_demographics.shape}')

print('--- Cust_Account Describe ---')
print(df_account.describe())
print('--- Cust_Enquiry Describe ---')
print(df_enquiry.describe())
print('--- Cust_Demographics Describe ---')
print(df_demographics.describe())

print('--- Cust_Account Info ---')
df_account.info()
print('--- Cust_Enquiry Info ---')
df_enquiry.info()
print('--- Cust_Demographics Info ---')
df_demographics.info()

print('--- Cust_Account Head ---')
print(df_account.head())
print('--- Cust_Enquiry Head ---')
print(df_enquiry.head())
print('--- Cust_Demographics Head ---')
print(df_demographics.head())

print('--- Cust_Account Tail ---')
print(df_account.tail())
print('--- Cust_Enquiry Tail ---')
print(df_enquiry.tail())
print('--- Cust_Demographics Tail ---')
print(df_demographics.tail())
# --- Notebook cell 7 ---
# Ensure Bad_label is numeric (converts '0'/'1' strings to integers)
df_demographics['Bad_label'] = pd.to_numeric(df_demographics['Bad_label'], errors='coerce')
# 1. Target Distribution Analysis
bad_counts = df_demographics['Bad_label'].value_counts()
bad_rate = df_demographics['Bad_label'].mean() * 100
print('=== TARGET DISTRIBUTION ===')
print(bad_counts)
print(f'Overall Default Rate (Bad_label = 1): {bad_rate:.2f}%')
# 2. Key Integrity Check
print(f'Unique Customers in Demographics: {df_demographics["customer_no"].nunique()}')
print(f'Unique Customers in Account: {df_account["customer_no"].nunique()}')
print(f'Unique Customers in Enquiry: {df_enquiry["customer_no"].nunique()}')
# 3. Missing Value Audit
print('Demographics Missing Ratio:', df_demographics.isnull().mean().max())
print('Account Missing Ratio:', df_account.isnull().mean().max())
print('Enquiry Missing Ratio:', df_enquiry.isnull().mean().max())

# 4. Data-type audit: every demographics column arrives from MySQL as a string, including numeric ones
#    (bureau score, income, loan amount, ...). Treating them as categories (get_dummies) is what created
#    ~66k sparse columns and hid the signal, so they are parsed explicitly in Step 3.
share_numeric = {c: pd.to_numeric(df_demographics[c].replace('', np.nan), errors='coerce').notna().mean()
                 for c in df_demographics.columns if c.startswith('feature_')}
print('Demographics columns that are >95% numeric strings:', sum(v > 0.95 for v in share_numeric.values()))
# Date columns stay as raw strings here; Step 3 parses them (including a 2-digit-year fix for dates of birth).