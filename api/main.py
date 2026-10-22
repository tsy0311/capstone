"""
FastAPI REST API for Startup Risk Profiling and Funding Prediction.

Endpoints:
- GET /health — Health check
- GET /model/info — Model metadata
- POST /predict — Single prediction
- POST /predict/batch — Batch prediction
- GET /risk/categories — Risk category definitions
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import numpy as np
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import (
    load_model, predict_single, predict_batch,
    get_model_info, RISK_CATEGORIES, RISK_LABELS
)


# --- Pydantic Models ---

class StartupInput(BaseModel):
    """Input model for a single startup prediction."""
    name: str = Field("Unknown", description="Startup name")
    category_code: str = Field("software", description="Business category (software, web, mobile, enterprise, advertising, gamesvideo, ecommerce, biotech, consulting, other)")
    state_code: str = Field("CA", description="State code (CA, NY, MA, TX, other)")
    city: str = Field("San Francisco", description="City")
    funding_total_usd: float = Field(0, ge=0, description="Total funding in USD")
    funding_rounds: int = Field(0, ge=0, description="Number of funding rounds")
    founded_year: int = Field(2015, ge=1990, le=2026, description="Founded year")
    first_funding_at: str = Field("", description="First funding date (YYYY-MM-DD)")
    last_funding_at: str = Field("", description="Last funding date (YYYY-MM-DD)")
    relationships: int = Field(0, ge=0, description="Number of relationships/partners")
    milestones: int = Field(0, ge=0, description="Number of milestones achieved")
    avg_participants: float = Field(0.0, ge=0, description="Avg participants per funding round")
    has_VC: int = Field(0, ge=0, le=1, description="Has VC backing (0 or 1)")
    has_angel: int = Field(0, ge=0, le=1, description="Has angel investor (0 or 1)")
    is_top500: int = Field(0, ge=0, le=1, description="Is Top 500 startup (0 or 1)")
    age_first_funding_year: float = Field(0.0, ge=0, description="Age at first funding (years)")
    age_last_funding_year: float = Field(0.0, ge=0, description="Age at last funding (years)")
    age_first_milestone_year: float = Field(0.0, ge=0, description="Age at first milestone (years)")
    age_last_milestone_year: float = Field(0.0, ge=0, description="Age at last milestone (years)")
    description: str = Field("", description="Startup description")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    startup_name: str
    funding_success_probability: float
    funding_failure_probability: float
    predicted_outcome: str
    risk_category: str
    risk_label: str
    threshold_used: float


class BatchRequest(BaseModel):
    """Batch prediction request."""
    startups: List[StartupInput]
    threshold: float = 0.5


# --- App ---

app = FastAPI(
    title="Startup Risk Profiling API",
    description="LLM-Based Startup Risk Profiling and Funding Outcome Prediction",
    version="1.0.0"
)

# Load model on startup
_model = None


def get_model():
    global _model
    if _model is None:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "best_model.joblib"
        )
        try:
            _model = load_model(model_path)
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run `python -m src.train` first."
            )
    return _model


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "startup-risk-profiler"}


@app.get("/model/info")
async def model_info():
    """Return model metadata."""
    model = get_model()
    return get_model_info(model)


@app.post("/predict", response_model=PredictionResponse)
async def predict(startup: StartupInput, threshold: float = 0.5):
    """Predict funding outcome for a single startup."""
    model = get_model()
    result = predict_single(model, startup.model_dump(), threshold=threshold)
    return PredictionResponse(**result)


@app.post("/predict/batch")
async def predict_batch_endpoint(request: BatchRequest):
    """Predict funding outcomes for multiple startups."""
    model = get_model()
    df = pd.DataFrame([s.model_dump() for s in request.startups])
    results = predict_batch(model, df, threshold=request.threshold)
    return {"predictions": results.to_dict(orient="records")}


@app.get("/risk/categories")
async def risk_categories():
    """Return risk category definitions."""
    return {
        k: {
            "range": [lo, hi],
            "label": RISK_LABELS[k]
        }
        for k, (lo, hi) in RISK_CATEGORIES.items()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
