import os
db = r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG\data\processed\financial_rag.db"
print(f"DuckDB: {os.path.getsize(db)/1024/1024:.1f} MB")