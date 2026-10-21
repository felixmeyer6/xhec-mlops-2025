"""Inference-only preprocessing matching the experiment-tracking lesson."""

import numpy as np
from sklearn.pipeline import Pipeline

from lib.models import TripInput

CATEGORICAL_COLS = ["PULocationID", "DOLocationID", "passenger_count"]


def predict_duration(trips: list[TripInput], pipeline: Pipeline) -> np.ndarray:
    """Encode string categories, including the training-time missing sentinel."""
    records = [
        {column: str(-1 if trip.dict()[column] is None else trip.dict()[column]) for column in CATEGORICAL_COLS} for trip in trips
    ]
    predictions = np.asarray(pipeline.predict(records), dtype=float)
    if predictions.shape != (len(trips),) or not np.isfinite(predictions).all():
        raise ValueError("Model returned invalid predictions.")
    return predictions
