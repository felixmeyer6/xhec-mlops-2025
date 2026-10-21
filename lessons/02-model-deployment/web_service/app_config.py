"""Application metadata and paths independent of the working directory."""

from pathlib import Path

MODEL_FOLDER = Path(__file__).resolve().parent / "local_models"
PATH_TO_PIPELINE = MODEL_FOLDER / "pipeline.joblib"
PATH_TO_METADATA = MODEL_FOLDER / "metadata.json"


# MISC
APP_TITLE = "TripDurationPredictionApp"
APP_DESCRIPTION = (
    "A simple API to predict trip duration in minutes "
    "for NYC yellow taxi trips, given a pickup, a dropoff location "
    "and a passenger count."
)
APP_VERSION = "0.0.1"
