"""
Bank GoodCredit — Ensemble Model Training & Evaluation

Extracted from the sanitized project notebook.
Execution is intended in the same stage order as the notebook; shared variables are produced by earlier stages.
No database credentials or raw client data are included.
"""

# --- Notebook cell 21 ---
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata

LGB_PARAMS = dict(objective='binary', learning_rate=0.02, num_leaves=8, min_child_samples=60, feature_fraction=0.3,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=20, max_bin=63, cat_smooth=50,
                  min_data_per_group=50, verbose=-1, n_jobs=-1)
LGB_ROUNDS = 400
W_LGB = 0.7
rank = lambda p: rankdata(p) / len(p)

def fit_predict_lgb(Xa, ya, Xb, seeds=(1, 2, 3, 4, 5)):
    p = np.zeros(len(Xb))
    for s in seeds:
        m = lgb.train({**LGB_PARAMS, 'seed': s}, lgb.Dataset(Xa, ya), LGB_ROUNDS)
        p += m.predict(Xb) / len(seeds)
    return p, m

def fit_predict_cb(Xa, ya, Xb, seeds=(1, 2)):
    p = np.zeros(len(Xb))
    for s in seeds:
        m = CatBoostClassifier(iterations=700, learning_rate=0.03, depth=5, l2_leaf_reg=10, rsm=0.3, random_seed=s,
                               verbose=0, thread_count=-1, cat_features=cat_cols, one_hot_max_size=4)
        m.fit(Xa, ya)
        p += m.predict_proba(Xb)[:, 1] / len(seeds)
    return p, m

def blend(p_lgb, p_cb):
    return W_LGB * rank(p_lgb) + (1 - W_LGB) * rank(p_cb)
# --- Notebook cell 23 ---
# 5-fold stratified CV on the TRAINING split only (the validation set is untouched here)
oof_l = np.zeros(len(y_train)); oof_c = np.zeros(len(y_train))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=1).split(X_train, y_train):
    oof_l[va], _ = fit_predict_lgb(X_train.iloc[tr], y_train.iloc[tr], X_train.iloc[va], seeds=(1, 2))
    oof_c[va], _ = fit_predict_cb(Xcb_train.iloc[tr], y_train.iloc[tr], Xcb_train.iloc[va], seeds=(1,))
print(f'CV Gini (train only)  LightGBM: {gini(y_train, oof_l):.2f}   CatBoost: {gini(y_train, oof_c):.2f}   Blend: {gini(y_train, blend(oof_l, oof_c)):.2f}')
# --- Notebook cell 25 ---
# Final fit on the training split, single scoring of the validation split
p_lgb, lgb_model = fit_predict_lgb(X_train, y_train, X_val)
p_cb, cb_model = fit_predict_cb(Xcb_train, y_train, Xcb_val)
y_pred_prob = blend(p_lgb, p_cb)          # rank-blended risk score in (0, 1]

for name, p in [('LightGBM', p_lgb), ('CatBoost', p_cb), ('Blend (final)', y_pred_prob)]:
    auc = roc_auc_score(y_val, p)
    print(f'{name:<14} Validation ROC-AUC: {auc:.4f}   Gini: {(2 * auc - 1) * 100:.2f}%')

val_auc_final = roc_auc_score(y_val, y_pred_prob)
val_gini_final = (2 * val_auc_final - 1) * 100
print(f'\nFinal Validation Gini Score: {val_gini_final:.2f}%  (Benchmark Target: >37.90%; original notebook: 35.69%)')
# --- Notebook cell 27 ---
# Feature matrix with gain (LightGBM, final seed) -> also saved to feature_gain.csv
gain = pd.Series(lgb_model.feature_importance('gain'), index=X_train.columns)
feature_gain = (gain / gain.sum()).sort_values(ascending=False).rename('gain_share').to_frame()
feature_gain.to_csv('feature_gain.csv')
print(f'{(feature_gain.gain_share > 0).sum()} features used. Top 25:')
print(feature_gain.head(25).round(4).to_string())