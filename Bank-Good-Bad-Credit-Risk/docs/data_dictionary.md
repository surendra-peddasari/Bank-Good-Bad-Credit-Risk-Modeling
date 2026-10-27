# Data Dictionary & Source Tables

The project uses three relational source tables.

## Cust_Account
Historical customer account and payment information. Key source fields include:
`dt_opened`, `customer_no`, `upload_dt`, `acct_type`, `owner_indic`, `opened_dt`,
`last_paymt_dt`, `closed_dt`, `reporting_dt`, `high_credit_amt`, `cur_balance_amt`,
`amt_past_due`, `paymenthistory1`, `paymenthistory2`, `paymt_str_dt`, `paymt_end_dt`,
`creditlimit`, `cashlimit`, `rateofinterest`, `paymentfrequency`, `actualpaymentamount`.

## Cust_Enquiry
Historical credit/enquiry information:
`dt_opened`, `customer_no`, `upload_dt`, `enquiry_dt`, `enq_purpose`, `enq_amt`.

## Cust_Demographics
Current customer application/demographic information with anonymized feature columns:
`dt_opened`, `customer no`, `entry_time`, `feature_1` through `feature_79`, `Bad_label`.

## Target
`Bad_label`:
- `0`: good credit history
- `1`: bad credit history / 30+ DPD bucket

## Privacy
Raw client data is not included in this repository. Do not add database exports, credentials, customer-level records, or other confidential client material.
