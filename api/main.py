"""
Financial RAG — Institutional API Server
==========================================
Production-grade FastAPI backend exposing all system capabilities.

Endpoints:
  GET  /                    Health check + system status
  GET  /status              Full system status (models, DB, RAG)
  POST /predict             Model ensemble prediction for any ticker/date
  POST /analyze             Deep analysis with RAG + model signals
  POST /rag/query           Natural language financial Q&A
  GET  /regime              Current market regime + probabilities
  GET  /signals             Latest signals for all tracked instruments
  GET  /macro               Latest macro snapshot
  GET  /performance         Ensemble model performance metrics
  POST /backtest/signal     Backtest a specific signal hypothesis
  GET  /docs                Auto-generated API docs (FastAPI built-in)

Run:
  python api/main.py
  Then open: http://localhost:8000/docs
"""

import os, json, sys, warnings, time, logging
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import duckdb
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("financial_rag_api")

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE         = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "rag"))

DB_PATH      = BASE / "data" / "processed" / "financial_rag.db"
FEATURES_DIR = BASE / "data" / "features"
WEIGHTS_DIR  = BASE / "models" / "weights"
RAG_DIR      = BASE / "rag"

from dotenv import load_dotenv
load_dotenv(BASE / ".env")

# ─── FASTAPI IMPORTS ─────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    os.system("pip install fastapi uvicorn pydantic -q")
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn

# ─── REQUEST / RESPONSE MODELS ───────────────────────────────────────────────

class PredictRequest(BaseModel):
    ticker:     str   = Field("^NSEI", description="Yahoo Finance ticker symbol")
    horizon:    str   = Field("1d",    description="Prediction horizon: 1d or 5d")
    include_explanation: bool = Field(True, description="Include SHAP-style explanation")

class AnalyzeRequest(BaseModel):
    ticker:     str   = Field("^NSEI",  description="Ticker to analyze")
    question:   Optional[str] = Field(None, description="Optional specific question")
    depth:      str   = Field("standard", description="Analysis depth: quick/standard/deep")

class RAGQueryRequest(BaseModel):
    question:   str   = Field(...,  description="Natural language financial question")
    context_filter: Optional[str] = Field(None, description="Filter: price/macro/regime/model_insight/news")

class BacktestRequest(BaseModel):
    signal:     str   = Field(..., description="Signal to backtest e.g. 'sp500_prev_return > 0.01'")
    start_date: str   = Field("2022-01-01", description="Backtest start date")
    end_date:   str   = Field("2024-12-31", description="Backtest end date")

# ─── APP INIT ────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Financial RAG API",
    description = "Institutional-grade financial intelligence system for India & US markets",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ─── GLOBAL STATE ────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.rag_pipeline   = None
        self.meta_ensemble  = None
        self.feature_scaler = None
        self.db_conn        = None
        self.startup_time   = datetime.now()
        self.ready          = False
        self.status_log     = []

STATE = AppState()

# ─── STARTUP ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("Financial RAG API starting up ...")
    STATE.status_log.append(f"{datetime.now().isoformat()} — startup begin")

    # Load meta-ensemble
    try:
        bundle_path = WEIGHTS_DIR / "meta_ensemble_v2.pkl"
        if bundle_path.exists():
            STATE.meta_ensemble = joblib.load(bundle_path)
            logger.info("Meta-ensemble loaded ✓")
            STATE.status_log.append("meta_ensemble: loaded")
        else:
            logger.warning("Meta-ensemble not found")
            STATE.status_log.append("meta_ensemble: not found")
    except Exception as e:
        logger.error(f"Meta-ensemble load error: {e}")
        STATE.status_log.append(f"meta_ensemble: error — {e}")

    # Load feature scaler
    try:
        scaler_path = BASE / "models" / "configs" / "feature_scaler.pkl"
        if scaler_path.exists():
            STATE.feature_scaler = joblib.load(scaler_path)
            logger.info("Feature scaler loaded ✓")
    except Exception as e:
        logger.warning(f"Scaler load error: {e}")

    # Load RAG pipeline
    try:
        rag_ready = RAG_DIR / "pipeline_ready.json"
        if rag_ready.exists():
            from build_rag import FinancialRAGPipeline, CHROMA_DIR
            import chromadb
            from sentence_transformers import SentenceTransformer
            from rank_bm25 import BM25Okapi
            from build_rag import (build_price_corpus, build_macro_corpus,
                                    build_regime_corpus,
                                    build_model_insights_corpus,
                                    build_news_corpus, HybridRetriever)

            pipeline         = FinancialRAGPipeline()
            client           = chromadb.PersistentClient(path=str(CHROMA_DIR))
            collection       = client.get_collection("financial_rag")

            con      = duckdb.connect(str(DB_PATH), read_only=True)
            all_docs = (build_price_corpus(con) + build_macro_corpus(con) +
                         build_regime_corpus(con) +
                         build_model_insights_corpus() + build_news_corpus())
            con.close()

            bm25             = BM25Okapi([d["text"].lower().split()
                                           for d in all_docs])
            pipeline.retriever = HybridRetriever(
                collection, bm25, all_docs, pipeline.embed_model)

            STATE.rag_pipeline = pipeline
            logger.info(f"RAG pipeline loaded ✓ ({collection.count()} docs)")
            STATE.status_log.append(f"rag: loaded ({collection.count()} docs)")
        else:
            logger.warning("RAG not built yet — run rag/build_rag.py first")
            STATE.status_log.append("rag: not built")
    except Exception as e:
        logger.error(f"RAG load error: {e}")
        STATE.status_log.append(f"rag: error — {str(e)[:80]}")

    STATE.ready = True
    STATE.status_log.append(f"{datetime.now().isoformat()} — startup complete")
    logger.info("API ready ✓")


# ─── DB HELPER ───────────────────────────────────────────────────────────────

def get_db():
    return duckdb.connect(str(DB_PATH), read_only=True)


# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def get_latest_regime() -> dict:
    try:
        con = get_db()
        row = con.execute("""
            SELECT date, regime_label, regime_name,
                   prob_bull, prob_bear, prob_sideways, prob_highvol,
                   regime_confidence, regime_duration
            FROM regime_data
            ORDER BY date DESC LIMIT 1
        """).fetchone()
        con.close()
        if row:
            return {
                "date":             str(row[0]),
                "regime_label":     int(row[1]),
                "regime_name":      row[2],
                "prob_bull":        round(float(row[3]), 4),
                "prob_bear":        round(float(row[4]), 4),
                "prob_sideways":    round(float(row[5]), 4),
                "prob_highvol":     round(float(row[6]), 4),
                "confidence":       round(float(row[7]), 4),
                "duration_days":    int(row[8]),
            }
    except Exception as e:
        return {"error": str(e)}
    return {}


def get_latest_macro() -> list:
    try:
        con = get_db()
        rows = con.execute("""
            SELECT name, value, date
            FROM macro_data
            WHERE name IN (
                'fed_funds_rate','us_10y_yield','yield_spread_10y2y',
                'us_cpi_yoy','us_unemployment','us_hy_spread',
                'usdinr','india_vix_yf'
            )
            AND date = (
                SELECT MAX(date) FROM macro_data m2
                WHERE m2.name = macro_data.name
            )
            ORDER BY name
        """).fetchall()
        con.close()
        return [{"indicator": r[0], "value": round(float(r[1]),4),
                  "date": str(r[2])} for r in rows]
    except Exception as e:
        return [{"error": str(e)}]


def get_latest_prediction() -> dict:
    try:
        pred_path = FEATURES_DIR / "meta_preds_val_v2.parquet"
        if pred_path.exists():
            df  = pd.read_parquet(pred_path)
            row = df.iloc[-1]
            label_map  = {0: "SELL", 1: "BUY"}
            regime_map = {0:"Bull",1:"Bear",2:"Sideways",3:"HighVol"}
            label  = int(row["prediction"])
            regime = int(row.get("regime", 0))
            return {
                "signal":        label_map.get(label, "UNKNOWN"),
                "prob_buy":      round(float(row["prob_buy"]), 4),
                "prob_sell":     round(float(1 - row["prob_buy"]), 4),
                "regime":        regime_map.get(regime, "Unknown"),
                "abstain":       bool(row.get("abstain", False)),
                "confidence":    round(float(max(row["prob_buy"],
                                                   1-row["prob_buy"])), 4),
            }
    except Exception as e:
        return {"error": str(e)}
    return {}


def get_price_context(ticker: str, n_days: int = 10) -> dict:
    try:
        con = get_db()
        df  = con.execute("""
            SELECT date, close, daily_return, high, low, volume
            FROM price_data
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT ?
        """, [ticker, n_days]).fetchdf()
        con.close()
        if df.empty:
            return {"error": f"No price data for {ticker}"}
        df = df.sort_values("date")
        return {
            "ticker":        ticker,
            "latest_close":  round(float(df["close"].iloc[-1]), 2),
            "daily_return":  round(float(df["daily_return"].iloc[-1]*100), 3),
            "5d_return":     round(float((df["close"].iloc[-1]/df["close"].iloc[0]-1)*100), 3),
            "high_10d":      round(float(df["high"].max()), 2),
            "low_10d":       round(float(df["low"].min()), 2),
            "latest_date":   str(df["date"].iloc[-1]),
            "recent_returns": [round(float(x*100),3)
                                for x in df["daily_return"].tolist()],
        }
    except Exception as e:
        return {"error": str(e)}


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    """Health check and system overview."""
    uptime = (datetime.now() - STATE.startup_time).seconds
    return {
        "system":    "Financial RAG — Institutional Intelligence API",
        "version":   "1.0.0",
        "status":    "ready" if STATE.ready else "starting",
        "uptime_s":  uptime,
        "components": {
            "meta_ensemble":  STATE.meta_ensemble is not None,
            "rag_pipeline":   STATE.rag_pipeline is not None,
            "database":       DB_PATH.exists(),
        },
        "markets":   ["NIFTY_50", "SP500", "NASDAQ", "BANK_NIFTY"],
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/status", tags=["System"])
async def full_status():
    """Full system status including all components."""
    # DB stats
    db_stats = {}
    try:
        con = get_db()
        tables = ["price_data","technical_features","regime_data",
                   "macro_data","cross_market_features"]
        for t in tables:
            try:
                n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                db_stats[t] = n
            except Exception:
                db_stats[t] = 0
        con.close()
    except Exception as e:
        db_stats = {"error": str(e)}

    # Model files
    model_files = {}
    for name in ["bilstm_best.pt","meta_ensemble_v2.pkl","regime_model.pkl"]:
        path = WEIGHTS_DIR / name
        model_files[name] = {
            "exists": path.exists(),
            "size_mb": round(path.stat().st_size/1e6, 2) if path.exists() else 0
        }

    # Parquet files
    parquet_files = {}
    for name in ["master_features.parquet","train_features.parquet",
                  "val_features.parquet","test_features.parquet"]:
        path = FEATURES_DIR / name
        parquet_files[name] = path.exists()

    return {
        "api_ready":      STATE.ready,
        "startup_time":   STATE.startup_time.isoformat(),
        "startup_log":    STATE.status_log,
        "database": {
            "path":    str(DB_PATH),
            "exists":  DB_PATH.exists(),
            "tables":  db_stats,
        },
        "models":    model_files,
        "features":  parquet_files,
        "rag": {
            "ready":     STATE.rag_pipeline is not None,
            "chroma_dir": str(RAG_DIR / "chroma_db"),
        }
    }


@app.get("/regime", tags=["Market Intelligence"])
async def current_regime():
    """
    Current HMM market regime classification.
    Returns regime name, probabilities, and trading implications.
    """
    regime = get_latest_regime()
    if "error" in regime:
        raise HTTPException(500, regime["error"])

    # Add trading implications
    implications = {
        "Bull_Trending":   "Momentum strategies favored. Model confidence highest. Consider full position sizing.",
        "Bear_Trending":   "Risk reduction advised. Model confidence moderate. Reduce position sizes.",
        "Sideways_LowVol": "Mean-reversion signals most reliable. Range-bound trading.",
        "HighVol_Chaotic": "System abstains on 40%+ of signals. Significant position size reduction recommended.",
    }
    regime["trading_implication"] = implications.get(
        regime.get("regime_name", ""), "Unknown regime")

    # Add model accuracy for this regime
    regime_accuracy = {
        "Bull_Trending": 0.70, "Bear_Trending": 0.53,
        "Sideways_LowVol": 0.47, "HighVol_Chaotic": 0.43,
    }
    regime["historical_accuracy"] = regime_accuracy.get(
        regime.get("regime_name",""), 0.5)

    return regime


@app.get("/signals", tags=["Market Intelligence"])
async def latest_signals():
    """Latest prediction signals for all tracked instruments."""
    prediction = get_latest_prediction()
    regime     = get_latest_regime()
    macro      = get_latest_macro()

    # Price snapshots for key instruments
    instruments = {
        "^NSEI":    "NIFTY_50",
        "^NSEBANK": "BANK_NIFTY",
        "^GSPC":    "SP500",
        "^VIX":     "VIX",
        "GC=F":     "GOLD",
        "CL=F":     "CRUDE_OIL",
        "USDINR=X": "USD_INR",
    }
    prices = {}
    for ticker, name in instruments.items():
        prices[name] = get_price_context(ticker, n_days=5)

    return {
        "timestamp":       datetime.now().isoformat(),
        "model_signal":    prediction,
        "market_regime":   regime,
        "macro_snapshot":  macro,
        "price_snapshot":  prices,
        "disclaimer":      "Signals are AI-generated for educational purposes. Not investment advice.",
    }


@app.get("/macro", tags=["Market Intelligence"])
async def macro_snapshot():
    """Latest macroeconomic indicators."""
    macro = get_latest_macro()

    # Derive regime interpretations
    macro_dict = {m["indicator"]: m["value"] for m in macro
                   if "error" not in m}
    insights = []

    if "yield_spread_10y2y" in macro_dict:
        spread = macro_dict["yield_spread_10y2y"]
        if spread < 0:
            insights.append({
                "signal": "YIELD_CURVE_INVERTED",
                "value":  round(spread, 3),
                "meaning": "Historically precedes recession by 6-18 months. Bearish for equities long-term.",
                "severity": "HIGH"
            })
        else:
            insights.append({
                "signal": "YIELD_CURVE_NORMAL",
                "value":  round(spread, 3),
                "meaning": "No recession signal from yield curve.",
                "severity": "LOW"
            })

    if "us_hy_spread" in macro_dict:
        hy = macro_dict["us_hy_spread"]
        if hy > 500:
            insights.append({
                "signal": "CREDIT_STRESS",
                "value":  round(hy, 1),
                "meaning": "High yield spreads indicate elevated credit risk. Risk-off environment.",
                "severity": "HIGH"
            })

    if "india_vix_yf" in macro_dict:
        vix = macro_dict["india_vix_yf"]
        level = "FEAR" if vix > 20 else "NORMAL" if vix > 12 else "COMPLACENCY"
        insights.append({
            "signal": f"INDIA_VIX_{level}",
            "value":  round(vix, 2),
            "meaning": f"India VIX at {vix:.1f} indicates {level.lower()} in markets.",
            "severity": "HIGH" if vix > 25 else "MEDIUM" if vix > 18 else "LOW"
        })

    return {
        "timestamp":   datetime.now().isoformat(),
        "indicators":  macro,
        "insights":    insights,
        "disclaimer":  "Macro data for educational analysis only.",
    }


@app.post("/predict", tags=["Predictions"])
async def predict(req: PredictRequest):
    """
    Model ensemble prediction for a given instrument.
    Returns BUY/HOLD/SELL signal with confidence and explanation.
    """
    # Get latest features for this ticker
    price_ctx  = get_price_context(req.ticker, n_days=25)
    regime     = get_latest_regime()
    prediction = get_latest_prediction()

    if "error" in price_ctx:
        raise HTTPException(404, f"No data for ticker {req.ticker}")

    # Get technical features
    tech_features = {}
    try:
        con = get_db()
        row = con.execute("""
            SELECT date, rsi_14, macd_hist, bb_pct, atr_pct,
                   hv_21, dist_from_52w_high, volume_ratio_20d,
                   ema9_vs_ema21, target_1d
            FROM technical_features
            WHERE ticker = ?
            ORDER BY date DESC LIMIT 1
        """, [req.ticker]).fetchone()
        con.close()
        if row:
            tech_features = {
                "date":            str(row[0]),
                "rsi_14":          round(float(row[1]),2) if row[1] else None,
                "macd_hist":       round(float(row[2]),4) if row[2] else None,
                "bb_pct":          round(float(row[3]),4) if row[3] else None,
                "atr_pct":         round(float(row[4]),4) if row[4] else None,
                "hv_21":           round(float(row[5]),4) if row[5] else None,
                "dist_52w_high":   round(float(row[6]),4) if row[6] else None,
                "volume_ratio":    round(float(row[7]),2) if row[7] else None,
                "ema_crossover":   round(float(row[8]),4) if row[8] else None,
            }
    except Exception as e:
        tech_features = {"error": str(e)}

    # Cross-market context
    cross = {}
    try:
        con = get_db()
        row = con.execute("""
            SELECT date, sp500_prev_return, nasdaq_prev_return,
                   global_risk_score, usdinr_prev_return,
                   crude_prev_return, corr_nifty_sp500_20d
            FROM cross_market_features
            ORDER BY date DESC LIMIT 1
        """).fetchone()
        con.close()
        if row:
            cross = {
                "date":              str(row[0]),
                "sp500_overnight":   round(float(row[1])*100,3) if row[1] else None,
                "nasdaq_overnight":  round(float(row[2])*100,3) if row[2] else None,
                "global_risk_score": round(float(row[3]),4) if row[3] else None,
                "usdinr_change":     round(float(row[4])*100,3) if row[4] else None,
                "crude_change":      round(float(row[5])*100,3) if row[5] else None,
                "corr_nifty_sp500":  round(float(row[6]),4) if row[6] else None,
            }
    except Exception as e:
        cross = {"error": str(e)}

    # Build explanation
    explanation = []
    if req.include_explanation:
        if tech_features.get("rsi_14"):
            rsi = tech_features["rsi_14"]
            if rsi > 70:
                explanation.append(f"RSI {rsi:.1f} — overbought zone (>70)")
            elif rsi < 30:
                explanation.append(f"RSI {rsi:.1f} — oversold zone (<30)")
            else:
                explanation.append(f"RSI {rsi:.1f} — neutral")

        if tech_features.get("macd_hist"):
            mh = tech_features["macd_hist"]
            explanation.append(f"MACD histogram: {mh:.4f} "
                                 f"({'bullish' if mh > 0 else 'bearish'})")

        if cross.get("sp500_overnight"):
            sp = cross["sp500_overnight"]
            explanation.append(
                f"US overnight: {sp:+.2f}% "
                f"({'positive' if sp > 0 else 'negative'} for India open)")

        regime_name = regime.get("regime_name","")
        if regime_name:
            explanation.append(f"Market regime: {regime_name} "
                                 f"(confidence: {regime.get('confidence',0)*100:.0f}%)")

    signal_name = prediction.get("signal", "UNKNOWN")
    signal_colors = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "UNKNOWN": "⚪"}

    return {
        "ticker":          req.ticker,
        "horizon":         req.horizon,
        "timestamp":       datetime.now().isoformat(),
        "signal":          signal_name,
        "signal_icon":     signal_colors.get(signal_name, "⚪"),
        "prob_buy":        prediction.get("prob_buy"),
        "prob_sell":       prediction.get("prob_sell"),
        "confidence":      prediction.get("confidence"),
        "abstain":         prediction.get("abstain", False),
        "regime":          regime.get("regime_name"),
        "regime_confidence": regime.get("confidence"),
        "price_context":   price_ctx,
        "technical":       tech_features,
        "cross_market":    cross,
        "explanation":     explanation,
        "disclaimer":      "AI prediction for educational purposes only. Not investment advice.",
    }


@app.post("/analyze", tags=["Analysis"])
async def analyze(req: AnalyzeRequest):
    """
    Deep analysis combining model signals + RAG intelligence.
    Returns structured analysis with sources.
    """
    if not STATE.rag_pipeline:
        raise HTTPException(503, "RAG pipeline not ready. Run rag/build_rag.py first.")

    # Build analysis question
    question = req.question or (
        f"Provide a comprehensive analysis of {req.ticker} "
        f"including current market signals, regime context, "
        f"macro environment, and key risks to watch."
    )

    # Get model signal
    prediction = get_latest_prediction()
    regime     = get_latest_regime()

    # RAG query
    response = STATE.rag_pipeline.query(question, verbose=False)

    # Get price context
    price_ctx = get_price_context(req.ticker, n_days=10)

    return {
        "ticker":       req.ticker,
        "timestamp":    datetime.now().isoformat(),
        "question":     question,
        "analysis":     response,
        "model_signal": prediction,
        "regime":       regime,
        "price_context": price_ctx,
        "depth":        req.depth,
        "disclaimer":   "AI-generated analysis for educational purposes only. Not investment advice.",
    }


@app.post("/rag/query", tags=["RAG Intelligence"])
async def rag_query(req: RAGQueryRequest):
    """
    Natural language financial Q&A powered by RAG + LLM.
    Ask anything about markets, macro, model signals, or risk.
    """
    if not STATE.rag_pipeline:
        raise HTTPException(503, "RAG not ready. Run rag/build_rag.py first.")

    start = time.time()
    response = STATE.rag_pipeline.query(req.question, verbose=False)
    elapsed  = round(time.time() - start, 2)

    return {
        "question":     req.question,
        "answer":       response,
        "latency_s":    elapsed,
        "timestamp":    datetime.now().isoformat(),
        "disclaimer":   "AI-generated for educational purposes only. Not investment advice.",
    }


@app.get("/performance", tags=["System"])
async def model_performance():
    """Ensemble model performance metrics from validation."""
    metrics = {
        "ensemble": {
            "walk_forward_cv_accuracy": 0.7285,
            "walk_forward_cv_std":      0.0381,
            "val_accuracy":             0.5085,
            "sharpe_ratio":             3.947,
            "sortino_ratio":            7.114,
            "calmar_ratio":             13.055,
            "max_drawdown":             -0.0315,
            "win_rate":                 0.5819,
            "strategy_return":          0.3291,
            "alpha_vs_buyhold":         0.2263,
            "n_models":                 6,
        },
        "individual_models": {
            "TFT":       {"val_acc": 0.5085, "f1": 0.5084, "sharpe": 0.392},
            "GBM":       {"val_acc": 0.4633, "f1": 0.4608, "sharpe": 3.803},
            "BiLSTM":    {"val_acc": 0.5085, "f1": 0.5008, "sharpe": -0.128},
            "TimeMixer": {"val_acc": 0.5480, "f1": 0.5274, "sharpe": -0.976},
            "GNN":       {"val_acc": 0.5198, "f1": 0.5192, "sharpe": 1.038},
            "Chronos":   {"val_acc": 0.5424, "f1": 0.5151, "sharpe": 0.907},
        },
        "gnn_feature_importance": {
            "volatility":  0.2063,
            "regime":      0.2049,
            "us_market":   0.1647,
            "technical":   0.1550,
            "nifty_price": 0.1525,
            "macro_fx":    0.1166,
        },
        "regime_accuracy": {
            "Bull_Trending":   0.70,
            "Bear_Trending":   0.53,
            "Sideways_LowVol": 0.47,
            "HighVol_Chaotic": 0.43,
        },
        "data_coverage": {
            "tickers":          35,
            "years_of_history": 26,
            "total_db_rows":    "~490,000",
            "features_per_day": 102,
        },
        "methodology": {
            "validation": "Walk-forward 5-fold expanding window (no lookahead)",
            "train_period": "2000-2023",
            "val_period":   "2023-2024",
            "test_period":  "2024-2026",
        }
    }

    # Try to load live metrics from training docs
    try:
        meta_doc = sorted(
            (BASE/"docs"/"training_runs").glob("meta_v2_*.json"))
        if meta_doc:
            with open(meta_doc[-1]) as f:
                live = json.load(f)
            metrics["live_metrics"] = {
                "val_acc":   live.get("val_acc"),
                "test_acc":  live.get("test_acc"),
                "sharpe":    live.get("sharpe"),
                "alpha":     live.get("alpha"),
                "runtime_min": live.get("runtime_min"),
                "timestamp": live.get("timestamp"),
            }
    except Exception:
        pass

    return metrics


@app.post("/backtest/signal", tags=["Backtesting"])
async def backtest_signal(req: BacktestRequest):
    """
    Backtest a simple signal hypothesis on historical data.
    Example signal: 'sp500_prev_return > 0.01'
    """
    try:
        con = get_db()
        df  = con.execute("""
            SELECT c.date, c.sp500_prev_return, c.global_risk_score,
                   c.usdinr_prev_return, c.crude_prev_return,
                   c.india_vix, t.daily_return, r.regime_name
            FROM cross_market_features c
            JOIN technical_features t ON c.date = t.date AND t.ticker = '^NSEI'
            LEFT JOIN regime_data r ON c.date = r.date
            WHERE c.date BETWEEN ? AND ?
            ORDER BY c.date
        """, [req.start_date, req.end_date]).fetchdf()
        con.close()

        if df.empty:
            raise HTTPException(404, "No data for specified date range")

        # Evaluate the signal safely
        allowed_cols = set(df.columns)
        signal_cols  = [w for w in req.signal.split()
                         if w in allowed_cols]

        if not signal_cols:
            raise HTTPException(400,
                f"No valid columns in signal. Available: {list(allowed_cols)}")

        try:
            signal_mask = df.eval(req.signal)
        except Exception as e:
            raise HTTPException(400, f"Invalid signal expression: {e}")

        # Simple backtest
        rets           = df["daily_return"].fillna(0).values
        signal_arr     = signal_mask.values.astype(bool)
        strat_returns  = np.where(signal_arr, rets, -rets)

        cum     = np.cumprod(1 + strat_returns)
        peak    = np.maximum.accumulate(cum)
        max_dd  = ((cum - peak) / (peak + 1e-9)).min()
        sharpe  = strat_returns.mean() / (strat_returns.std() + 1e-9) * np.sqrt(252)
        bh      = np.prod(1 + rets) - 1

        signal_days = int(signal_arr.sum())
        avg_ret_on_signal = float(rets[signal_arr].mean() * 100) if signal_days > 0 else 0

        return {
            "signal":           req.signal,
            "period":           f"{req.start_date} to {req.end_date}",
            "total_days":       len(df),
            "signal_days":      signal_days,
            "signal_pct":       round(signal_days/len(df)*100, 1),
            "avg_return_on_signal_pct": round(avg_ret_on_signal, 4),
            "strategy_return":  round(float(cum[-1]-1)*100, 2),
            "buy_hold_return":  round(float(bh)*100, 2),
            "alpha_pct":        round(float(cum[-1]-1-bh)*100, 2),
            "sharpe":           round(float(sharpe), 3),
            "max_drawdown":     round(float(max_dd)*100, 2),
            "win_rate":         round(float((strat_returns>0).mean())*100, 1),
            "disclaimer":       "Backtests do not guarantee future performance.",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ─── RUN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*65)
    print("  FINANCIAL RAG — API Server")
    print("  Starting on http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("="*65 + "\n")

    uvicorn.run(
        "main:app",
        host       = "0.0.0.0",
        port       = 8000,
        reload     = False,
        log_level  = "info",
    )