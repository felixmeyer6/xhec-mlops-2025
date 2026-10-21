"""Resolve shared data and artifact locations independently of the working directory."""

from pathlib import Path

CATEGORICAL_COLS = ["PULocationID", "DOLocationID", "passenger_count"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIRPATH = str(PROJECT_ROOT / "data")
MODELS_DIRPATH = str(PROJECT_ROOT / "models" / "orchestration")
TRAIN_FILEPATH = str(Path(DATA_DIRPATH) / "yellow_tripdata_2021-01.parquet")
TEST_FILEPATH = str(Path(DATA_DIRPATH) / "yellow_tripdata_2021-02.parquet")
INPUT_FILEPATH = str(Path(DATA_DIRPATH) / "yellow_tripdata_2021-03.parquet")
ARTIFACTS_FILEPATH = str(Path(MODELS_DIRPATH) / "taxi_model.pkl")
PREDICTIONS_FILEPATH = str(Path(MODELS_DIRPATH) / "predictions.npy")
MIN_DURATION = 1
MAX_DURATION = 60
