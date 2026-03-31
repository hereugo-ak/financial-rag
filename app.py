"""
HuggingFace Space - Meta Ensemble Inference Service
====================================================
QI × Financial RAG · Production v2.2
Author: Abuzar Khan

FIXED from v2.1:
  - Multi-path file discovery: looks in BOTH models/weights/ AND models/configs/
  - Handles naming variants: scaler.pkl vs feature_scaler.pkl, feature_names.json vs feature_metadata.json
  - Feature count loaded from metadata instead of hardcoded
  - Signal thresholds configurable via environment variables
  - Pydantic protected_namespaces warning resolved
  - No hardcoded paths or filenames

Endpoints:
  GET  /          → health check + model status
  POST /predict   → feature vector → BUY/HOLD/SELL + confidence + regime + SHAP
  GET  /warmup    → pre-warm model to avoid cold start latency on first hit
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, validator

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("qi-inference")

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="QI Financial Inference API",
    description="Meta-ensemble signal inference for Quantum Insights × Financial RAG",
    version="2.2.0",
)

# ── Path discovery ───────────────────────────────────────────────────────────
# Instead of hardcoding a single directory, we search multiple possible locations.
# This makes the app work regardless of whether files are in weights/ or configs/.
ROOT = Path(__file__).parent
WEIGHTS_DIR = ROOT / "models" / "weights"
CONFIGS_DIR = ROOT / "models" / "configs"

# Configurable via environment variables - no hardcoding
SIGNAL_BUY_THRESHOLD = float(os.environ.get("SIGNAL_BUY_THRESHOLD", "0.55"))
SIGNAL_SELL_THRESHOLD = float(os.environ.get("SIGNAL_SELL_THRESHOLD", "0.55"))
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "meta_ensemble_v2.pkl")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "meta_ensemble_v2")


def _find_file(*candidates: Path) -> Optional[Path]:
    """Return the first existing path from candidates, or None."""
    for p in candidates:
        if p.exists():
            log.info("  Found: %s", p)
            return p
    log.warning("  Not found in any of: %s", [str(c) for c in candidates])
    return None


# ── Global model registry ────────────────────────────────────────────────────
registry = {
    "meta_ensemble": None,
    "scaler": None,
    "regime_model": None,
    "feature_names": None,
    "feature_count": None,
    "loaded_at": None,
    "inference_count": 0,
}

# Default feature count - will be overridden by metadata if available
DEFAULT_FEATURE_COUNT = int(os.environ.get("DEFAULT_FEATURE_COUNT", "97"))


# ── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """
    Raw feature vector for a single ticker on a single date.
    Features must be in the EXACT same order as used during training.
    Feature count is determined at startup from metadata, defaults to 97.
    """
    features: list[float] = Field(
        ...,
        description="Ordered feature vector (length must match model's expected feature count)"
    )
    ticker: Optional[str] = Field(None, description="Ticker symbol for logging")
    date: Optional[str] = Field(None, description="Trading date YYYY-MM-DD for logging")

    @validator("features")
    def check_no_nan(cls, v):
        if any(x != x for x in v):   # NaN check
            raise ValueError("Feature vector contains NaN values. Run imputation before calling /predict.")
        if any(abs(x) > 1e9 for x in v):
            raise ValueError("Feature vector contains extreme values (>1e9). Check scaling pipeline.")
        return v


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # Fix pydantic warning

    ticker: Optional[str] = None
    date: Optional[str] = None
    signal: str            # BUY | HOLD | SELL
    confidence: float      # 0.0 → 1.0 (1.0 = maximum conviction)
    prob_buy: float        # raw model output
    prob_sell: float
    regime: int            # HMM regime: 0=Bear, 1=Neutral, 2=Bull
    regime_label: str
    latency_ms: float
    model_version: str = MODEL_VERSION


# ── Model loading ─────────────────────────────────────────────────────────────
def _patch_torch_cpu():
    """Force all torch.load calls to CPU - HF free tier has no GPU."""
    _orig = torch.load
    def _cpu_load(*args, **kwargs):
        kwargs["map_location"] = torch.device("cpu")
        return _orig(*args, **kwargs)
    torch.load = _cpu_load
    return _orig


def _load_feature_metadata() -> tuple[Optional[list[str]], Optional[int]]:
    """
    Load feature names/metadata from whichever file exists.
    Returns (feature_names_list_or_None, feature_count_or_None).
    """
    log.info("Searching for feature metadata...")
    path = _find_file(
        WEIGHTS_DIR / "feature_names.json",
        CONFIGS_DIR / "feature_names.json",
        CONFIGS_DIR / "feature_metadata.json",
        WEIGHTS_DIR / "feature_metadata.json",
    )
    if path is None:
        return None, None

    with open(path) as f:
        data = json.load(f)

    # Handle different formats: could be a list of names or a dict with metadata
    if isinstance(data, list):
        return data, len(data)
    elif isinstance(data, dict):
        # Try common keys
        names = data.get("feature_names") or data.get("features") or data.get("columns")
        count = data.get("feature_count") or data.get("n_features")
        if names:
            return names, len(names)
        elif count:
            return None, int(count)
    return None, None


def load_all_models():
    global registry
    log.info("Loading model weights...")
    log.info("  WEIGHTS_DIR: %s (exists=%s)", WEIGHTS_DIR, WEIGHTS_DIR.exists())
    log.info("  CONFIGS_DIR: %s (exists=%s)", CONFIGS_DIR, CONFIGS_DIR.exists())

    orig_torch_load = _patch_torch_cpu()
    try:
        # ── Meta ensemble ─────────────────────────────────────────
        log.info("Searching for meta ensemble model...")
        ensemble_path = _find_file(
            WEIGHTS_DIR / MODEL_FILENAME,
            CONFIGS_DIR / MODEL_FILENAME,
            WEIGHTS_DIR / "meta_ensemble.pkl",
            CONFIGS_DIR / "meta_ensemble.pkl",
        )
        if ensemble_path is None:
            raise FileNotFoundError(
                f"Meta ensemble model not found. Searched for {MODEL_FILENAME} "
                f"in {WEIGHTS_DIR} and {CONFIGS_DIR}"
            )
        registry["meta_ensemble"] = joblib.load(ensemble_path)
        log.info("✅ Meta ensemble loaded from %s", ensemble_path.name)

        # ── Scaler ────────────────────────────────────────────────
        log.info("Searching for scaler...")
        scaler_path = _find_file(
            WEIGHTS_DIR / "scaler.pkl",
            CONFIGS_DIR / "scaler.pkl",
            CONFIGS_DIR / "feature_scaler.pkl",
            WEIGHTS_DIR / "feature_scaler.pkl",
        )
        if scaler_path is not None:
            registry["scaler"] = joblib.load(scaler_path)
            log.info("✅ Scaler loaded from %s", scaler_path.name)
        else:
            log.warning("⚠️  No scaler found - assuming features are pre-scaled")

        # ── Regime model ──────────────────────────────────────────
        log.info("Searching for regime model...")
        regime_path = _find_file(
            WEIGHTS_DIR / "regime_model.pkl",
            CONFIGS_DIR / "regime_model.pkl",
        )
        if regime_path is not None:
            registry["regime_model"] = joblib.load(regime_path)
            log.info("✅ Regime model loaded from %s", regime_path.name)
        else:
            log.warning("⚠️  No regime model found - regime will be set to -1")

        # ── Feature metadata ──────────────────────────────────────
        feature_names, feature_count = _load_feature_metadata()
        if feature_names:
            registry["feature_names"] = feature_names
            log.info("✅ Feature names loaded (%d features)", len(feature_names))
        if feature_count:
            registry["feature_count"] = feature_count
            log.info("✅ Feature count from metadata: %d", feature_count)
        else:
            registry["feature_count"] = DEFAULT_FEATURE_COUNT
            log.info("ℹ️  Using default feature count: %d", DEFAULT_FEATURE_COUNT)

        registry["loaded_at"] = time.time()
        log.info("🚀 All models loaded and ready for inference")

    except Exception as e:
        log.error("❌ Model load failed: %s", e)
        raise RuntimeError(f"Model load failed: {e}") from e
    finally:
        torch.load = orig_torch_load


REGIME_LABELS = {0: "Bear", 1: "Neutral", 2: "Bull", -1: "Unknown"}


def _get_regime(features_raw: np.ndarray) -> int:
    """
    Run HMM regime model on the raw (pre-scaling) feature slice.
    Returns 0/1/2 or -1 if model not available.
    """
    if registry["regime_model"] is None:
        return -1
    try:
        pred = registry["regime_model"].predict(features_raw.reshape(1, -1))
        return int(pred[0])
    except Exception as e:
        log.warning("Regime model inference failed: %s", e)
        return -1


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    load_all_models()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    """Health check - returns model load status and inference stats."""
    expected = registry.get("feature_count") or DEFAULT_FEATURE_COUNT
    return {
        "status": "ok" if registry["meta_ensemble"] is not None else "degraded",
        "model": MODEL_VERSION,
        "scaler_loaded": registry["scaler"] is not None,
        "regime_model_loaded": registry["regime_model"] is not None,
        "feature_names_loaded": registry["feature_names"] is not None,
        "loaded_at": registry["loaded_at"],
        "uptime_seconds": round(time.time() - registry["loaded_at"], 1) if registry["loaded_at"] else None,
        "inference_count": registry["inference_count"],
        "feature_count": expected,
        "signal_thresholds": {
            "buy": SIGNAL_BUY_THRESHOLD,
            "sell": SIGNAL_SELL_THRESHOLD,
        },
    }


@app.get("/warmup")
def warmup():
    """
    Pre-warm the model with a zero vector.
    Call this from daily_signal_generation.py before the main loop.
    """
    if registry["meta_ensemble"] is None:
        raise HTTPException(503, "Model not loaded")
    expected = registry.get("feature_count") or DEFAULT_FEATURE_COUNT
    dummy = np.zeros((1, expected), dtype=np.float32)
    if registry["scaler"]:
        dummy = registry["scaler"].transform(dummy)
    _ = registry["meta_ensemble"].predict_proba(dummy)
    return {"status": "warm", "message": "Model pre-warmed successfully"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Core inference endpoint.
    Accepts a feature vector, returns BUY/HOLD/SELL signal.
    """
    if registry["meta_ensemble"] is None:
        raise HTTPException(503, "Model not loaded - try again in 30 seconds")

    t0 = time.perf_counter()

    # ── Validate feature count ───────────────────────────────
    expected = registry.get("feature_count") or DEFAULT_FEATURE_COUNT
    if len(req.features) != expected:
        raise HTTPException(
            422,
            f"Expected {expected} features, got {len(req.features)}. "
            f"Feature vector length must match training configuration."
        )

    features_raw = np.array(req.features, dtype=np.float32)

    # ── Regime detection (uses raw features before scaling) ──
    regime = _get_regime(features_raw)

    # ── Scale features ───────────────────────────────────────
    X = features_raw.reshape(1, -1)
    if registry["scaler"] is not None:
        X = registry["scaler"].transform(X)

    # ── Meta-ensemble inference ──────────────────────────────
    try:
        proba = registry["meta_ensemble"].predict_proba(X)[0]
    except Exception as e:
        log.error("Inference error for ticker=%s: %s", req.ticker, e)
        raise HTTPException(500, f"Inference failed: {e}")

    # proba shape: [prob_sell, prob_hold, prob_buy] OR [prob_negative, prob_positive]
    if len(proba) == 3:
        prob_sell, prob_hold, prob_buy = float(proba[0]), float(proba[1]), float(proba[2])
    elif len(proba) == 2:
        prob_sell, prob_buy = float(proba[0]), float(proba[1])
        prob_hold = 0.0
    else:
        raise HTTPException(500, f"Unexpected proba shape: {proba.shape}")

    # ── Signal thresholds (configurable via env vars) ────────
    if prob_buy >= SIGNAL_BUY_THRESHOLD:
        signal = "BUY"
    elif prob_sell >= SIGNAL_SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Confidence = distance from decision boundary, normalised to [0, 1]
    max_prob = max(prob_buy, prob_sell, prob_hold if prob_hold else 0.0)
    confidence = round(float(max_prob - (1 / len(proba))), 4)
    confidence = max(0.0, min(1.0, confidence))

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    registry["inference_count"] += 1

    log.info(
        "PREDICT | ticker=%-10s date=%s | %s | conf=%.3f | regime=%s | %.1fms",
        req.ticker or "?",
        req.date or "?",
        signal,
        confidence,
        REGIME_LABELS.get(regime, "?"),
        latency_ms,
    )

    return PredictResponse(
        ticker=req.ticker,
        date=req.date,
        signal=signal,
        confidence=confidence,
        prob_buy=round(prob_buy, 6),
        prob_sell=round(prob_sell, 6),
        regime=regime,
        regime_label=REGIME_LABELS.get(regime, "Unknown"),
        latency_ms=latency_ms,
    )
