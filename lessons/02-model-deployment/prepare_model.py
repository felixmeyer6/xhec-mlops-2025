"""Export an exact registered model version and its matching fitted vectorizer."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import mlflow
from sklearn.pipeline import Pipeline

LESSON_FOLDER = Path(__file__).resolve().parent
TRACKING_FOLDER = LESSON_FOLDER.parent / "01-model-and-experiment-management" / "mlruns"


def export_model(model_version: str | None = None) -> None:
    """Resolve Production once or export a supplied immutable registry version."""
    mlflow.set_tracking_uri(TRACKING_FOLDER.as_uri())
    mlflow.set_registry_uri(mlflow.get_tracking_uri())
    client = mlflow.MlflowClient()
    model_name = "nyc_taxi_duration_lr"
    if model_version is None:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if len(versions) != 1:
            raise RuntimeError("Run the experiment-tracking notebook first to register a Production taxi model.")
        registered = versions[0]
    else:
        registered = client.get_model_version(model_name, model_version)
    regressor = mlflow.sklearn.load_model(f"models:/{model_name}/{registered.version}")
    with TemporaryDirectory() as directory:
        artifact = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{registered.run_id}/preprocessor/vectorizer.pkl", dst_path=directory
        )
        vectorizer = joblib.load(artifact)
    pipeline = Pipeline([("vectorizer", vectorizer), ("regressor", regressor)])
    output = LESSON_FOLDER / "web_service" / "local_models"
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output / "pipeline.joblib")
    metadata = {
        "model_name": model_name,
        "model_version": str(registered.version),
        "run_id": registered.run_id,
        "sklearn_version": version("scikit-learn"),
        "categorical_columns": ["PULocationID", "DOLocationID", "passenger_count"],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Exported {model_name} version {registered.version} to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", help="Exact registry version; defaults to the current Production version.")
    export_model(parser.parse_args().model_version)
