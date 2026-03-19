
import pandas as pd, duckdb, numpy as np
from pathlib import Path

DB  = r'C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG\data\processed\financial_rag.db'
CSV = r'C:\Users\HP\Documents\Sample DATA\Fii Dii Trading activity.csv'

df = pd.read_csv(CSV)
print('Columns:', list(df.columns))
print('Rows:', len(df))
df.columns = ['date','fii_buy','fii_sell','fii_net','dii_buy','dii_sell','dii_net']
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['date'])
df['total_net'] = df['fii_net'] + df['dii_net']
df['fii_signal'] = np.where(df['fii_net']>3000, 1, np.where(df['fii_net']<-3000, -1, 0))
df['dii_signal'] = np.where(df['dii_net']>1000, 1, np.where(df['dii_net']<-1000, -1, 0))

con = duckdb.connect(DB)
stored = 0
for _, r in df.iterrows():
    try:
        con.execute('INSERT OR REPLACE INTO fii_dii_data VALUES (?,?,?,?,?,?,?,?,?,?)',
            [str(r['date'].date()), float(r['fii_buy']), float(r['fii_sell']),
             float(r['fii_net']), float(r['dii_buy']), float(r['dii_sell']),
             float(r['dii_net']), float(r['total_net']),
             int(r['fii_signal']), int(r['dii_signal'])])
        stored += 1
    except Exception as e:
        pass

n = con.execute('SELECT COUNT(*) FROM fii_dii_data').fetchone()[0]
s = con.execute('SELECT MIN(date), MAX(date) FROM fii_dii_data').fetchone()
print(f'Stored: {stored} new rows')
print(f'Total DB: {n} rows, {s[0]} to {s[1]}')
con.close()



