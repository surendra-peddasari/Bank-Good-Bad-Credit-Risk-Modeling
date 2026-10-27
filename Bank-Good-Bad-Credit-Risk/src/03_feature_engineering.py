"""
Bank GoodCredit — Domain Feature Engineering & Customer-Level Aggregation

Extracted from the sanitized project notebook.
Execution is intended in the same stage order as the notebook; shared variables are produced by earlier stages.
No database credentials or raw client data are included.
"""

# --- Notebook cell 9 ---
import re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

SEVERE = {'SUB': 180, 'DBT': 360, 'LSS': 540, 'SMA': 60, 'STD': 0}
MAXTOK = 40


def to_dt(s, dob=False):
    d = pd.to_datetime(s.replace('', np.nan), format='%d-%b-%y', errors='coerce')
    if dob:  # two-digit years > 2016 are really 19xx
        d = d.where(~(d.dt.year > 2016), d - pd.DateOffset(years=100))
    return d


def num(s):
    return pd.to_numeric(s.replace('', np.nan), errors='coerce')


def parse_hist(h1, h2):
    """full payment history (newest first) -> matrix of dpd values, NaN for unknown / missing."""
    out = np.full((len(h1), MAXTOK), np.nan, dtype='float32')
    for i, (a, b) in enumerate(zip(h1.values, h2.values)):
        s = (a + b).replace('"', '')
        toks = [s[j:j + 3] for j in range(0, min(len(s), MAXTOK * 3), 3)]
        for k, t in enumerate(toks):
            if t == 'XXX':
                continue
            if t in SEVERE:
                out[i, k] = SEVERE[t]
            else:
                try: out[i, k] = float(t)
                except ValueError: pass
    return out


def hist_feats(M):
    f = {}
    valid = ~np.isnan(M)
    Mz = np.nan_to_num(M, nan=0)
    f['n_valid'] = valid.sum(1)
    f['max_dpd'] = np.where(valid.any(1), np.nanmax(np.where(valid, M, -1), 1), np.nan)
    for th in (1, 30, 60, 90):
        hit = (Mz >= th) & valid
        f[f'n_ge{th}'] = hit.sum(1)
        f[f'any_ge{th}'] = hit.any(1).astype(int)
        f[f'first_ge{th}'] = np.where(hit.any(1), hit.argmax(1), np.nan)      # months since most recent event
        f[f'last_ge{th}'] = np.where(hit.any(1), MAXTOK - 1 - hit[:, ::-1].argmax(1), np.nan)
    for w in (3, 6, 12, 24):
        v = valid[:, :w]; m = Mz[:, :w]
        f[f'n_ge1_w{w}'] = ((m >= 1) & v).sum(1)
        f[f'n_ge30_w{w}'] = ((m >= 30) & v).sum(1)
        f[f'max_w{w}'] = np.where(v.any(1), np.where(v, M[:, :w], -1).max(1), np.nan)
        f[f'sum_w{w}'] = (m * v).sum(1)
    f['sum_dpd'] = (Mz * valid).sum(1)
    f['frac_ge1'] = f['n_ge1'] / np.maximum(f['n_valid'], 1)
    return pd.DataFrame(f)

# --- Notebook cell 11 ---
def agg_accounts(a, d_app):
    a = a.copy()
    for c in ['opened_dt', 'last_paymt_dt', 'closed_dt', 'reporting_dt', 'paymt_str_dt', 'paymt_end_dt']:
        a[c] = to_dt(a[c])
    for c in ['high_credit_amt', 'cur_balance_amt', 'amt_past_due', 'creditlimit', 'cashlimit',
              'rateofinterest', 'paymentfrequency', 'actualpaymentamount', 'acct_type', 'owner_indic']:
        a[c] = num(a[c])
    a = a.merge(d_app.rename('app_dt'), left_on='customer_no', right_index=True, how='left')
    a['age_m'] = (a['app_dt'] - a['opened_dt']).dt.days / 30.4
    a['since_lastpay_m'] = (a['app_dt'] - a['last_paymt_dt']).dt.days / 30.4
    a['diff_lastpaymt_opened'] = (a['last_paymt_dt'] - a['opened_dt']).dt.days
    a['is_closed'] = a['closed_dt'].notna().astype(int)
    a['util'] = np.where(a['creditlimit'] > 0, a['cur_balance_amt'] / a['creditlimit'], np.nan)
    a['bal_hc'] = np.where(a['high_credit_amt'] > 0, a['cur_balance_amt'] / a['high_credit_amt'], np.nan)
    a['pastdue_flag'] = (a['amt_past_due'].fillna(0) > 0).astype(int)
    a['hist_len_m'] = (a['paymt_str_dt'].dt.year - a['paymt_end_dt'].dt.year) * 12 + (a['paymt_str_dt'].dt.month - a['paymt_end_dt'].dt.month)
    M = parse_hist(a['paymenthistory1'], a['paymenthistory2'])
    H = hist_feats(M)
    H.index = a.index
    a = pd.concat([a, H.add_prefix('h_')], axis=1)
    hc = [c for c in a.columns if c.startswith('h_')]

    g = a.groupby('customer_no')
    out = pd.DataFrame(index=g.size().index)
    out['acc_count'] = g.size()
    for c, fn in [('cur_balance_amt', ['sum', 'mean', 'max']), ('creditlimit', ['sum', 'mean', 'max']),
                  ('cashlimit', ['sum', 'mean']), ('high_credit_amt', ['sum', 'mean', 'max']),
                  ('amt_past_due', ['sum', 'max']), ('actualpaymentamount', ['sum', 'mean', 'max']),
                  ('rateofinterest', ['mean', 'max']), ('age_m', ['mean', 'min', 'max']),
                  ('since_lastpay_m', ['mean', 'min', 'max']), ('diff_lastpaymt_opened', ['sum', 'mean', 'min', 'max']),
                  ('util', ['mean', 'max']), ('bal_hc', ['mean', 'max']), ('hist_len_m', ['mean', 'max', 'min']),
                  ('is_closed', ['sum', 'mean']), ('pastdue_flag', ['sum'])]:
        r = g[c].agg(fn); r.columns = [f'acc_{c}_{x}' for x in fn]; out = out.join(r)
    out['acc_util_total'] = np.where(out['acc_creditlimit_sum'] > 0, out['acc_cur_balance_amt_sum'] / out['acc_creditlimit_sum'], np.nan)
    den = out['acc_creditlimit_mean'].fillna(0) + out['acc_cashlimit_mean'].fillna(0)
    out['acc_util_trend'] = np.where((out['acc_creditlimit_sum'] > 0) & (den > 0) & (out['acc_cur_balance_amt_mean'] > 0),
                                     out['acc_util_total'] / (out['acc_cur_balance_amt_mean'] / den), np.nan)
    out['acc_pastdue_ratio'] = out['acc_amt_past_due_sum'] / (out['acc_cur_balance_amt_sum'] + 1)
    # payment-history rollups across accounts
    for c in hc:
        if c in ('h_n_valid',):
            out[f'{c}_sum'] = g[c].sum(); out[f'{c}_mean'] = g[c].mean(); continue
        if c.startswith(('h_first_', 'h_last_')):
            out[f'{c}_min'] = g[c].min(); out[f'{c}_mean'] = g[c].mean(); continue
        if c.startswith('h_max'):
            out[f'{c}_max'] = g[c].max(); out[f'{c}_mean'] = g[c].mean(); continue
        out[f'{c}_sum'] = g[c].sum(); out[f'{c}_mean'] = g[c].mean()
        if c.startswith('h_any'):
            out[f'{c}_sum'] = g[c].sum()
    # opening velocity
    for w in (6, 12, 24):
        out[f'acc_opened_{w}m'] = a[a['age_m'] <= w].groupby('customer_no').size().reindex(out.index).fillna(0)
    # by account type / status buckets
    top_types = a['acct_type'].value_counts().head(6).index
    for t in top_types:
        s = a[a['acct_type'] == t].groupby('customer_no')
        out[f'type{int(t)}_n'] = s.size().reindex(out.index).fillna(0)
        out[f'type{int(t)}_bal'] = s['cur_balance_amt'].sum().reindex(out.index).fillna(0)
        out[f'type{int(t)}_dpd30'] = s['h_n_ge30'].sum().reindex(out.index).fillna(0)
        out[f'type{int(t)}_dpd1'] = s['h_n_ge1'].sum().reindex(out.index).fillna(0)
    for k, m in [('open', a['is_closed'] == 0), ('closed', a['is_closed'] == 1)]:
        s = a[m].groupby('customer_no')
        out[f'{k}_bal'] = s['cur_balance_amt'].sum().reindex(out.index).fillna(0)
        out[f'{k}_n'] = s.size().reindex(out.index).fillna(0)
        out[f'{k}_dpd1'] = s['h_n_ge1'].sum().reindex(out.index).fillna(0)
        out[f'{k}_dpd30'] = s['h_n_ge30'].sum().reindex(out.index).fillna(0)
    out['acc_perc_closed'] = out['closed_n'] / out['acc_count']
    return out


def agg_enquiries(e, d_app):
    e = e.copy()
    e['enquiry_dt'] = to_dt(e['enquiry_dt']); e['dt_opened'] = to_dt(e['dt_opened'])
    e['enq_amt'] = num(e['enq_amt']); e['enq_purpose'] = num(e['enq_purpose'])
    e = e.merge(d_app.rename('app_dt'), left_on='customer_no', right_index=True, how='left')
    e['days_before'] = (e['app_dt'] - e['enquiry_dt']).dt.days
    e['diff_open_enquiry'] = (e['dt_opened'] - e['enquiry_dt']).dt.days
    g = e.groupby('customer_no')
    out = pd.DataFrame(index=g.size().index)
    out['enq_count'] = g.size()
    out['enq_mean_diff_open'] = g['diff_open_enquiry'].mean()
    out['enq_days_since_last'] = g['days_before'].min()
    out['enq_days_since_first'] = g['days_before'].max()
    out['enq_span'] = out['enq_days_since_first'] - out['enq_days_since_last']
    for w in (30, 90, 180, 365, 730):
        out[f'enq_n_{w}'] = e[e['days_before'] <= w].groupby('customer_no').size().reindex(out.index).fillna(0)
        out[f'enq_amt_{w}'] = e[e['days_before'] <= w].groupby('customer_no')['enq_amt'].sum().reindex(out.index).fillna(0)
    # recency relative to the customer's own latest enquiry (as in the business-case definition)
    e['rel'] = (e.groupby('customer_no')['enquiry_dt'].transform('max') - e['enquiry_dt']).dt.days
    out['count_enquiry_recency_365'] = e[e['rel'] <= 365].groupby('customer_no').size().reindex(out.index).fillna(0)
    out['count_enquiry_recency_90'] = e[e['rel'] <= 90].groupby('customer_no').size().reindex(out.index).fillna(0)
    for c, fn in [('enq_amt', ['sum', 'mean', 'max', 'min'])]:
        r = g[c].agg(fn); r.columns = [f'enq_{x}_amt' for x in fn]; out = out.join(r)
    out['enq_purpose_nuniq'] = g['enq_purpose'].nunique()
    out['max_freq_enquiry'] = g['enq_purpose'].agg(lambda x: x.mode().iloc[0] if x.notna().any() else np.nan)
    for p in e['enq_purpose'].value_counts().head(8).index:
        s = e[e['enq_purpose'] == p].groupby('customer_no')
        out[f'enq_p{int(p)}_n'] = s.size().reindex(out.index).fillna(0)
        out[f'enq_p{int(p)}_perc'] = out[f'enq_p{int(p)}_n'] / out['enq_count']
        out[f'enq_p{int(p)}_amt'] = s['enq_amt'].sum().reindex(out.index).fillna(0)
        out[f'enq_p{int(p)}_n365'] = s.apply(lambda x: (x['days_before'] <= 365).sum()).reindex(out.index).fillna(0)
    out['enq_rate_per_month'] = out['enq_count'] / (out['enq_span'] / 30.4 + 1)
    return out

# --- Notebook cell 13 ---
NUMERIC_DEMO = ['feature_3', 'feature_4', 'feature_6', 'feature_7', 'feature_14', 'feature_19', 'feature_25', 'feature_26',
                'feature_29', 'feature_30', 'feature_31', 'feature_34', 'feature_35', 'feature_39', 'feature_40', 'feature_41',
                'feature_42', 'feature_44', 'feature_52', 'feature_55', 'feature_56', 'feature_64', 'feature_65', 'feature_66',
                'feature_67', 'feature_68', 'feature_69', 'feature_71', 'feature_74', 'feature_76', 'feature_78']
DATE_DEMO = ['feature_2', 'feature_53']


def demo_feats(d):
    d = d.copy().set_index('customer_no')
    app = to_dt(d['dt_opened'])
    X = pd.DataFrame(index=d.index)
    for c in NUMERIC_DEMO:
        X[c] = num(d[c])
    dob = to_dt(d['feature_21'], dob=True)
    X['age'] = (app - dob).dt.days / 365.25
    for c in DATE_DEMO:
        X[f'{c}_gap_d'] = (app - to_dt(d[c])).dt.days
    X['entry_gap_d'] = (app - to_dt(d['entry_time'])).dt.days
    X['f70_year'] = pd.to_datetime(d['feature_70'].replace('', np.nan), format='%b-%y', errors='coerce').dt.year
    X['app_month'] = app.dt.month
    X['app_dow'] = app.dt.dayofweek
    # presence flags for sparsely populated blocks
    for c in ['feature_8', 'feature_9', 'feature_13', 'feature_17', 'feature_18', 'feature_45', 'feature_48', 'feature_51', 'feature_57', 'feature_73']:
        X[f'{c}_present'] = (d[c].replace('', np.nan).notna()).astype(int)
    X['f3_missing'] = X['feature_3'].isna().astype(int)
    X['income_to_f7'] = X['feature_35'] / (X['feature_7'] + 1)
    X['f7_over_f35'] = X['feature_7'] / (X['feature_35'] + 1)
    X['f52_over_f35'] = X['feature_52'] / (X['feature_35'] + 1)
    X['pin_match'] = (d['feature_29'] == d['feature_44']).astype(int)
    X['pin_match2'] = (d['feature_29'] == d['feature_66']).astype(int)
    X['phone_match'] = (d['feature_22'] == d['feature_77']).astype(int)
    X['pan_match'] = (d['feature_20'] == d['feature_47']).astype(int)
    X['pan_char4'] = d['feature_20'].str[3:4]
    X['email_dom'] = d['feature_24'].str.upper()
    X['email2_dom'] = d['feature_45'].str.upper()
    cat_cols = ['feature_1', 'feature_11', 'feature_12', 'feature_23', 'feature_27', 'feature_28', 'feature_32', 'feature_33',
                'feature_36', 'feature_37', 'feature_43', 'feature_46', 'feature_48', 'feature_50', 'feature_51', 'feature_58',
                'feature_59', 'feature_60', 'feature_62', 'feature_72', 'feature_79', 'feature_75']
    for c in cat_cols:
        X[c] = d[c].replace('', np.nan)
    X['feature_38_title'] = d['feature_38'].str.upper().str.split().str[0]
    return X, app

# --- Notebook cell 15 ---
# Assemble the customer-level matrix
X_demo, app_dt = demo_feats(df_demographics)
y_all = pd.to_numeric(df_demographics.set_index('customer_no')['Bad_label'], errors='coerce').astype(int)

X_all = X_demo.join(agg_accounts(df_account, app_dt)).join(agg_enquiries(df_enquiry, app_dt))
X_all['acc_x_enq'] = X_all['enq_n_365'] / (X_all['acc_count'] + 1)
X_all['bal_to_income'] = X_all['acc_cur_balance_amt_sum'] / (X_all['feature_35'] + 1)
X_all['f3_x_util'] = X_all['feature_3'] * X_all['acc_util_total']

print(f'Master Feature Matrix Ready. Shape: {X_all.shape}  (was 23896 x 65,926 after one-hot encoding)')
print(f'Bad rate: {y_all.mean():.4f}')