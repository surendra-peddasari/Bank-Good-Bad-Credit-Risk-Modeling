"""
Bank GoodCredit — Feature Preparation & Train/Validation Partitioning

Extracted from the sanitized project notebook.
Execution is intended in the same stage order as the notebook; shared variables are produced by earlier stages.
No database credentials or raw client data are included.
"""

# --- Notebook cell 18 ---
import re
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

X = X_all.copy()
cat_cols = [c for c in X.columns if X[c].dtype == object]
for c in cat_cols:
    vc = X[c].value_counts()
    X[c] = X[c].where(X[c].map(vc) >= 30)

X_lgb = X.copy()
for c in cat_cols:
    X_lgb[c] = X_lgb[c].astype('category')
X_cb = X.copy()
for c in cat_cols:
    X_cb[c] = X_cb[c].fillna('NA').astype(str)

X_train, X_val, y_train, y_val = train_test_split(X_lgb, y_all, test_size=0.25, random_state=42, stratify=y_all)
Xcb_train, Xcb_val = X_cb.loc[X_train.index], X_cb.loc[X_val.index]

gini = lambda yt, p: (2 * roc_auc_score(yt, p) - 1) * 100
print(f'Features: {X_train.shape[1]} ({len(cat_cols)} categorical)')
print(f'X_train shape: {X_train.shape}, X_val shape: {X_val.shape}')
print(f'Bad rate train/val: {y_train.mean():.4f} / {y_val.mean():.4f}')