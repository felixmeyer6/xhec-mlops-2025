"""Serve the fitted taxi vectorizer and model with FastAPI."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path

import joblib
from app_config import APP_DESCRIPTION, APP_TITLE, APP_VERSION, PATH_TO_METADATA, PATH_TO_PIPELINE
from fastapi import FastAPI, HTTPException, Request
from lib.inference import CATEGORICAL_COLS, predict_duration
from lib.models import PredictionRequest, PredictionResponse, TripPrediction
from sklearn.pipeline import Pipeline


def create_app(pipeline_path: Path = PATH_TO_PIPELINE, metadata_path: Path = PATH_TO_METADATA) -> FastAPI:
    """Create an app that loads trusted local artifacts once, at startup."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Fail startup rather than serve requests without a compatible model."""
        if not pipeline_path.is_file() or not metadata_path.is_file():
            raise RuntimeError("Missing model artifacts. Run `uv run python prepare_model.py` from the deployment lesson first.")
        metadata = json.loads(metadata_path.read_text())
        if metadata["sklearn_version"] != version("scikit-learn") or metadata["categorical_columns"] != CATEGORICAL_COLS:
            raise RuntimeError("Model dependencies or feature schema do not match this application. Re-export the model.")
        # joblib/pickle can execute code: only load artifacts from a trusted training run.
        pipeline = joblib.load(pipeline_path)
        if not isinstance(pipeline, Pipeline) or list(pipeline.named_steps) != ["vectorizer", "regressor"]:
            raise RuntimeError("Expected a fitted vectorizer/regressor pipeline.")
        application.state.pipeline = pipeline
        application.state.model_version = str(metadata["model_version"])
        yield

    application = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION, lifespan=lifespan)

    @application.get("/")
    def home(request: Request) -> dict[str, str]:
        """Describe the application and the loaded model version."""
        return {
            "title": APP_TITLE,
            "description": APP_DESCRIPTION,
            "app_version": APP_VERSION,
            "model_version": request.app.state.model_version,
            "docs": "/docs",
        }

    @application.get("/health")
    def health(request: Request) -> dict[str, str]:
        """Report readiness only after successful artifact loading."""
        return {"status": "ok", "model_version": request.app.state.model_version}

    @application.post("/predict", response_model=PredictionResponse)
    def run_inference(body: PredictionRequest, request: Request) -> PredictionResponse:
        """Predict a bounded batch without training or calling the model registry."""
        try:
            predictions = predict_duration(body.trips, request.app.state.pipeline)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail="Model could not produce valid predictions.") from exc
        return PredictionResponse(
            model_version=request.app.state.model_version,
            predictions=[
                TripPrediction(predicted_duration_minutes=value, outside_training_range=not 1 <= value <= 60)
                for value in predictions
            ],
        )

    return application


app = create_app()
