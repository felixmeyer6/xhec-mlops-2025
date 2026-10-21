"""Run taxi training and prediction independently, in memory or via a saved bundle."""

import argparse
from importlib.metadata import version

import numpy as np
from prefect import flow, get_run_logger
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from lib.config import (
    ARTIFACTS_FILEPATH,
    CATEGORICAL_COLS,
    INPUT_FILEPATH,
    MAX_DURATION,
    MIN_DURATION,
    PREDICTIONS_FILEPATH,
    TEST_FILEPATH,
    TRAIN_FILEPATH,
)
from lib.helpers import load_pickle, save_pickle, save_predictions
from lib.modeling import evaluate_model, predict, train_model
from lib.preprocessing import process_data


@flow(name="train-taxi-model", log_prints=True, persist_result=False)
def train_model_workflow(
    train_filepath: str = TRAIN_FILEPATH,
    test_filepath: str = TEST_FILEPATH,
    artifacts_filepath: str | None = None,
) -> dict:
    """Fit on January, evaluate on February, and optionally save a matched artifact bundle."""
    X_train, y_train, dv = process_data(train_filepath)
    X_test, y_test, _ = process_data(test_filepath, dv=dv)
    model = train_model(X_train, y_train)
    metrics = {
        "train_rmse": evaluate_model(y_train, predict(X_train, model)),
        "test_rmse": evaluate_model(y_test, predict(X_test, model)),
        "baseline_test_rmse": evaluate_model(y_test, np.full(y_test.shape, y_train.mean())),
    }
    bundle = {
        "model": model,
        "dv": dv,
        "categorical_columns": CATEGORICAL_COLS,
        "sklearn_version": version("scikit-learn"),
        "train_filepath": train_filepath,
        "test_filepath": test_filepath,
        "min_duration": MIN_DURATION,
        "max_duration": MAX_DURATION,
        "train_rows": X_train.shape[0],
        "test_rows": X_test.shape[0],
        "feature_count": X_train.shape[1],
        **metrics,
    }
    if artifacts_filepath is not None:
        save_pickle(artifacts_filepath, bundle)
    get_run_logger().info("Training complete: %s", metrics)
    return {**bundle, "artifacts_filepath": artifacts_filepath}


@flow(name="batch-predict-taxi", validate_parameters=False, log_prints=True, persist_result=False)
def batch_predict_workflow(
    input_filepath: str = INPUT_FILEPATH,
    model: LinearRegression | None = None,
    dv: DictVectorizer | None = None,
    artifacts_filepath: str | None = None,
    output_filepath: str | None = None,
) -> np.ndarray:
    """Predict without targets, loading a complete saved model/vectorizer pair if requested."""
    if artifacts_filepath is not None:
        if model is not None or dv is not None:
            raise ValueError("Supply either an artifact bundle or an in-memory model/vectorizer pair, not both.")
        bundle = load_pickle(artifacts_filepath)
        if bundle["categorical_columns"] != CATEGORICAL_COLS or bundle["sklearn_version"] != version("scikit-learn"):
            raise ValueError("Artifact feature schema or sklearn version does not match this workflow.")
        model, dv = bundle["model"], bundle["dv"]
    if model is None or dv is None:
        raise ValueError("Prediction requires both the fitted model and vectorizer, or a saved artifact bundle.")
    X, _, _ = process_data(input_filepath, dv=dv, with_target=False)
    predictions = predict(X, model)
    if output_filepath is not None:
        save_predictions(output_filepath, predictions)
    get_run_logger().info(
        "Predicted %s trips; %s estimates fall outside [1, 60] minutes (raw outputs are not clipped).",
        len(predictions),
        int(((predictions < MIN_DURATION) | (predictions > MAX_DURATION)).sum()),
    )
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("--train", default=TRAIN_FILEPATH)
    parser.add_argument("--test", default=TEST_FILEPATH)
    parser.add_argument("--input", default=INPUT_FILEPATH)
    parser.add_argument("--artifacts", default=ARTIFACTS_FILEPATH)
    parser.add_argument("--output", default=PREDICTIONS_FILEPATH)
    arguments = parser.parse_args()
    if arguments.mode == "train":
        result = train_model_workflow(arguments.train, arguments.test, arguments.artifacts)
        print({key: result[key] for key in ("train_rmse", "test_rmse", "baseline_test_rmse", "artifacts_filepath")})
    else:
        result = batch_predict_workflow(arguments.input, artifacts_filepath=arguments.artifacts, output_filepath=arguments.output)
        print(f"Saved {len(result):,} predictions to {arguments.output}")
