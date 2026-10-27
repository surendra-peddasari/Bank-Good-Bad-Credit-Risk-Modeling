"""
Bank GoodCredit — Database Connection & Relational Data Ingestion

Extracted from the sanitized project notebook.
Execution is intended in the same stage order as the notebook; shared variables are produced by earlier stages.
No database credentials or raw client data are included.
"""

# --- Notebook cell 3 ---
# Packages used below. (sqlalchemy is not needed: we read through a plain pymysql connection.)
# !pip install pymysql lightgbm catboost scikit-learn pandas numpy
# --- Notebook cell 4 ---
import os, warnings
import pandas as pd
import numpy as np
import pymysql
warnings.filterwarnings('ignore')

# Credentials (override via environment variables instead of keeping them in the notebook)
DB_HOST = os.environ['DB_HOST']
DB_PORT = int(os.environ.get('DB_PORT', '3306'))
DB_USER = os.environ['DB_USER']
DB_PASS = os.environ['DB_PASS']
DB_NAME = os.environ['DB_NAME']

print('Connecting to Bank GoodCredit Database...')
import time
for attempt in range(1, 9):          # the shared server occasionally answers 'Too many connections' (1040)
    try:
        conn = pymysql.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME)
        break
    except pymysql.err.OperationalError as e:
        print(f'Connection attempt {attempt} failed: {e}; retrying...')
        time.sleep(15 * attempt)
else:
    raise RuntimeError('Could not connect to the database')
df_account = pd.read_sql('SELECT * FROM Cust_Account', conn)
df_enquiry = pd.read_sql('SELECT * FROM Cust_Enquiry', conn)
df_demographics = pd.read_sql('SELECT * FROM Cust_Demographics', conn)
conn.close()
print("Connected to database successfully")