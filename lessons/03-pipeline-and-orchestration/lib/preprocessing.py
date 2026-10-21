"""Named preprocessing tasks shared by training and target-free batch inference."""

import numpy as np
import pandas as pd
from prefect import flow, get_run_logger, task
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer

from lib.config import CATEGORICAL_COLS, MAX_DURATION, MIN_DURATION


@task(name="load-parquet", tags=["io"], retries=2, retry_delay_seconds=1, persist_result=False)
def load_data(filepath: str, with_target: bool = True) -> pd.DataFrame:
    """Read only the booking features and, for evaluation, the target timestamps."""
    columns = CATEGORICAL_COLS + (["tpep_pickup_datetime", "tpep_dropoff_datetime"] if with_target else [])
    return pd.read_parquet(filepath, columns=columns)


@task(name="compute-duration", tags=["preprocessing"], persist_result=False)
def compute_target(
    df: pd.DataFrame, pickup_column: str = "tpep_pickup_datetime", dropoff_column: str = "tpep_dropoff_datetime"
) -> pd.DataFrame:
    """Compute duration in minutes without mutating the input frame."""
    result = df.copy()
    result["duration"] = (result[dropoff_column] - result[pickup_column]).dt.total_seconds() / 60
    return result


@task(name="filter-duration", tags=["preprocessing"], persist_result=False)
def filter_outliers(df: pd.DataFrame, min_duration: int = MIN_DURATION, max_duration: int = MAX_DURATION) -> pd.DataFrame:
    """Keep trips inside the inclusive training target range."""
    if min_duration <= 0 or max_duration < min_duration:
        raise ValueError("Duration limits must be positive and ordered.")
    return df.loc[df["duration"].between(min_duration, max_duration)].copy()


@task(name="encode-categories", tags=["preprocessing"], persist_result=False)
def encode_categorical_cols(df: pd.DataFrame, categorical_cols: list[str] | None = None) -> pd.DataFrame:
    """Encode category labels as strings and missing values as the -1 sentinel."""
    columns = CATEGORICAL_COLS if categorical_cols is None else categorical_cols
    result = df.copy()
    result[columns] = result[columns].fillna(-1).astype(int).astype(str)
    return result


@task(name="vectorize-features", tags=["preprocessing"], persist_result=False)
def extract_x_y(
    df: pd.DataFrame,
    categorical_cols: list[str] | None = None,
    dv: DictVectorizer | None = None,
    with_target: bool = True,
) -> tuple[csr_matrix, np.ndarray | None, DictVectorizer]:
    """Fit the sparse encoding on training data only and reuse it elsewhere."""
    columns = CATEGORICAL_COLS if categorical_cols is None else categorical_cols
    if df.empty:
        raise ValueError("No usable trips remain after preprocessing.")
    records = df[columns].to_dict(orient="records")
    if dv is None:
        if not with_target:
            raise ValueError("Inference requires a fitted training vectorizer.")
        dv = DictVectorizer().fit(records)
    target = df["duration"].to_numpy() if with_target else None
    return dv.transform(records), target, dv


@flow(name="process-taxi-data", validate_parameters=False, persist_result=False)
def process_data(
    filepath: str, dv: DictVectorizer | None = None, with_target: bool = True
) -> tuple[csr_matrix, np.ndarray | None, DictVectorizer]:
    """Chain explicit task dependencies, omitting all target operations at inference."""
    frame = load_data(filepath, with_target)
    if with_target:
        frame = filter_outliers(compute_target(frame))
    frame = encode_categorical_cols(frame)
    features, target, vectorizer = extract_x_y(frame, dv=dv, with_target=with_target)
    get_run_logger().info("Processed %s trips into %s features from %s", features.shape[0], features.shape[1], filepath)
    return features, target, vectorizer
