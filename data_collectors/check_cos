import pandas as pd
from pathlib import Path

FEATURES_DIR = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG\data\features")

for mn in ["tft", "timemixer", "gnn", "chronos"]:
    p = FEATURES_DIR / f"{mn}_probs_train.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        print(f"{mn}: {list(df.columns)}")
    else:
        print(f"{mn}: FILE NOT FOUND")