"""Validated booking inputs and documented prediction outputs."""

from pydantic import BaseModel, Field, conint, conlist

LocationID = conint(strict=True, ge=1, le=265)
PassengerCount = conint(strict=True, ge=1, le=6)


class TripInput(BaseModel):
    """A single taxi booking using only information available before pickup."""

    PULocationID: LocationID
    DOLocationID: LocationID
    passenger_count: PassengerCount | None = None

    class Config:
        """Reject misspelled or unexpected fields."""

        extra = "forbid"


class PredictionRequest(BaseModel):
    """A bounded batch, preserving the order of the supplied trips."""

    trips: conlist(TripInput, min_items=1, max_items=1000)

    class Config:
        """Reject unexpected top-level fields."""

        extra = "forbid"


class TripPrediction(BaseModel):
    """An unmodified estimate with an explicit plausibility warning."""

    predicted_duration_minutes: float = Field(..., description="Raw model estimate in minutes; not clipped.")
    outside_training_range: bool = Field(..., description="True if the estimate is outside the training target range [1, 60].")


class PredictionResponse(BaseModel):
    """Versioned predictions in the same order as the input batch."""

    model_version: str
    predictions: list[TripPrediction]
