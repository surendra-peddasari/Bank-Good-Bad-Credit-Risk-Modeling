"""
Bank GoodCredit — Decile Rank Ordering & Risk Segmentation

Extracted from the sanitized project notebook.
Execution is intended in the same stage order as the notebook; shared variables are produced by earlier stages.
No database credentials or raw client data are included.
"""

# --- Notebook cell 30 ---
decile_df = pd.DataFrame({'actual': y_val.values, 'pred_prob': y_pred_prob})
decile_df['decile'] = pd.qcut(decile_df['pred_prob'].rank(method='first'), q=10, labels=False) + 1

decile_summary = decile_df.groupby('decile').agg(
    total_customers=('actual', 'count'),
    bad_customers=('actual', 'sum'),
    good_customers=('actual', lambda x: (x == 0).sum())
).reset_index().sort_values(by='decile', ascending=False)
decile_summary['bad_rate'] = decile_summary['bad_customers'] / decile_summary['total_customers']
decile_summary['bad_capture_share'] = decile_summary['bad_customers'] / decile_summary['bad_customers'].sum()
decile_summary['cum_capture'] = decile_summary['bad_capture_share'].cumsum()

print('=== DECILE RANK ORDERING SUMMARY ===')
print(decile_summary.to_string(index=False))
print(f"\nTop-3 deciles capture {decile_summary['bad_capture_share'].head(3).sum():.1%} of bad customers")