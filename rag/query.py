"""
Financial RAG — Query Interface
================================
Run: python rag/query.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_rag import FinancialRAGPipeline, CHROMA_DIR, RAG_DIR
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import json, os
from dotenv import load_dotenv

BASE = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG")
load_dotenv(BASE / ".env")

def main():
    print("\n  Loading Financial RAG ...")
    pipeline = FinancialRAGPipeline()

    # Load existing chroma DB
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("financial_rag")

    # Reload all docs for BM25
    from build_rag import (build_price_corpus, build_macro_corpus,
                            build_regime_corpus, build_model_insights_corpus,
                            build_news_corpus, HybridRetriever)
    import duckdb
    DB_PATH = BASE / "data" / "processed" / "financial_rag.db"
    con = duckdb.connect(str(DB_PATH), read_only=True)
    all_docs = (build_price_corpus(con) + build_macro_corpus(con) +
                build_regime_corpus(con) + build_model_insights_corpus() +
                build_news_corpus())
    con.close()

    bm25 = BM25Okapi([d["text"].lower().split() for d in all_docs])
    pipeline.retriever = HybridRetriever(
        collection, bm25, all_docs, pipeline.embed_model)

    print(f"  Loaded {collection.count()} documents from vector DB")
    pipeline.interactive()

if __name__ == "__main__":
    main()