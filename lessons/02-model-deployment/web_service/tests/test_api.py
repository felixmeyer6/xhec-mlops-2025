"""Exercise validation, artifact loading and prediction parity without training data."""

import json
import unittest
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib
import numpy as np
from fastapi.testclient import TestClient
from lib.inference import CATEGORICAL_COLS
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from main import create_app


class DeploymentTests(unittest.TestCase):
    """Use small deterministic artifacts to isolate the serving contract."""

    def setUp(self) -> None:  # noqa: N802
        """Prepare artifacts and run the real application lifespan."""
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.pipeline_path = Path(directory.name) / "pipeline.joblib"
        self.metadata_path = Path(directory.name) / "metadata.json"
        self.records = [
            {"PULocationID": "1", "DOLocationID": "2", "passenger_count": "1"},
            {"PULocationID": "2", "DOLocationID": "1", "passenger_count": "2"},
            {"PULocationID": "1", "DOLocationID": "1", "passenger_count": "-1"},
        ]
        self.pipeline = Pipeline([("vectorizer", DictVectorizer()), ("regressor", LinearRegression())])
        self.pipeline.fit(self.records, [10.0, 20.0, 30.0])
        joblib.dump(self.pipeline, self.pipeline_path)
        self.metadata = {
            "model_version": "test-version",
            "sklearn_version": version("scikit-learn"),
            "categorical_columns": CATEGORICAL_COLS,
        }
        self.metadata_path.write_text(json.dumps(self.metadata))
        self.app = create_app(self.pipeline_path, self.metadata_path)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_home_health_and_docs(self) -> None:
        """Expose metadata, readiness and interactive API documentation."""
        self.assertEqual(self.client.get("/").json()["model_version"], "test-version")
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        self.assertEqual(self.client.get("/docs").status_code, 200)
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/predict", schema["paths"])

    def test_batch_predictions_match_pipeline(self) -> None:
        """Preserve order and match direct inference through the fitted pipeline."""
        trips = [
            {"PULocationID": 2, "DOLocationID": 1, "passenger_count": 2},
            {"PULocationID": 1, "DOLocationID": 2, "passenger_count": 1},
        ]
        response = self.client.post("/predict", json={"trips": trips})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_version"], "test-version")
        np.testing.assert_allclose(
            [prediction["predicted_duration_minutes"] for prediction in body["predictions"]],
            self.pipeline.predict([self.records[1], self.records[0]]),
        )

    def test_missing_passenger_count_uses_training_sentinel(self) -> None:
        """Treat omitted and null passenger counts like training-time missing data."""
        for trip in ({"PULocationID": 1, "DOLocationID": 1}, {"PULocationID": 1, "DOLocationID": 1, "passenger_count": None}):
            with self.subTest(trip=trip):
                response = self.client.post("/predict", json={"trips": [trip]})
                self.assertEqual(response.status_code, 200)
                self.assertAlmostEqual(response.json()["predictions"][0]["predicted_duration_minutes"], 30.0)

    def test_unknown_but_valid_zones_do_not_refit(self) -> None:
        """Let the fitted vectorizer ignore unseen categories without changing it."""
        vectorizer = self.app.state.pipeline.named_steps["vectorizer"]
        before = vectorizer.get_feature_names_out().copy()
        response = self.client.post("/predict", json={"trips": [{"PULocationID": 265, "DOLocationID": 264}]})
        self.assertEqual(response.status_code, 200)
        np.testing.assert_array_equal(before, vectorizer.get_feature_names_out())

    def test_invalid_trips_are_rejected(self) -> None:
        """Reject invalid ranges, coercions, missing fields and unexpected fields."""
        valid = {"PULocationID": 1, "DOLocationID": 2, "passenger_count": 1}
        invalid = [
            {**valid, "PULocationID": 0},
            {**valid, "DOLocationID": 266},
            {**valid, "passenger_count": 0},
            {**valid, "passenger_count": 7},
            {**valid, "passenger_count": True},
            {**valid, "PULocationID": "1"},
            {**valid, "PULocationID": 1.5},
            {**valid, "duration": 10},
            {"PULocationID": 1},
        ]
        for trip in invalid:
            with self.subTest(trip=trip):
                self.assertEqual(self.client.post("/predict", json={"trips": [trip]}).status_code, 422)

    def test_invalid_batch_is_rejected(self) -> None:
        """Bound batches and require the documented request envelope."""
        trip = {"PULocationID": 1, "DOLocationID": 2}
        for body in ({"trips": []}, {"trips": [trip] * 1001}, {"trips": [trip], "extra": 1}, [trip], {}):
            with self.subTest(body_type=type(body).__name__):
                self.assertEqual(self.client.post("/predict", json=body).status_code, 422)

    def test_out_of_range_predictions_are_flagged_not_clipped(self) -> None:
        """Do not hide a regression model's implausible raw outputs."""
        with patch.object(self.app.state.pipeline, "predict", return_value=np.array([-2.0])):
            response = self.client.post("/predict", json={"trips": [{"PULocationID": 1, "DOLocationID": 2}]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["predictions"][0], {"predicted_duration_minutes": -2.0, "outside_training_range": True})

    def test_non_finite_predictions_are_not_served(self) -> None:
        """Return a controlled error instead of serializing NaN or infinity."""
        for value in (np.nan, np.inf):
            with self.subTest(value=value), patch.object(self.app.state.pipeline, "predict", return_value=np.array([value])):
                response = self.client.post("/predict", json={"trips": [{"PULocationID": 1, "DOLocationID": 2}]})
                self.assertEqual(response.status_code, 503)

    def test_missing_artifacts_fail_startup(self) -> None:
        """Fail early with setup instructions when artifacts are missing."""
        with self.assertRaisesRegex(RuntimeError, "prepare_model.py"):
            with TestClient(create_app(self.pipeline_path.with_name("missing.joblib"), self.metadata_path)):
                pass

    def test_incompatible_metadata_fails_startup(self) -> None:
        """Reject incompatible dependency versions and feature schemas."""
        for change in ({"sklearn_version": "0.0.0"}, {"categorical_columns": ["wrong"]}):
            self.metadata_path.write_text(json.dumps({**self.metadata, **change}))
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, "do not match"):
                with TestClient(create_app(self.pipeline_path, self.metadata_path)):
                    pass


if __name__ == "__main__":
    unittest.main()
