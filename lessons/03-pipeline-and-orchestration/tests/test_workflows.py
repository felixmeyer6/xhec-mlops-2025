"""Check pure task logic, real Prefect flows, saved artifacts and deployment schedules."""

import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
from lib.config import ARTIFACTS_FILEPATH, CATEGORICAL_COLS, PROJECT_ROOT
from lib.deployment import build_deployments, pause_deployments
from lib.helpers import load_pickle, save_pickle, save_predictions
from lib.modeling import evaluate_model
from lib.preprocessing import compute_target, encode_categorical_cols, extract_x_y, filter_outliers, load_data
from lib.workflows import batch_predict_workflow, train_model_workflow


class WorkflowTests(unittest.TestCase):
    """Use small local parquet files rather than depending on the full taxi dataset."""

    def setUp(self) -> None:  # noqa: N802
        """Prepare representative trip durations, missing values and booking features."""
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        timestamps = pd.to_datetime(["2021-01-01"] * 7)
        self.raw = pd.DataFrame(
            {
                "PULocationID": [1, 2, 1, 2, 1, 2, 3],
                "DOLocationID": [2, 1, 1, 2, 1, 2, 1],
                "passenger_count": [1, 2, None, 1, 1, 1, 3],
                "tpep_pickup_datetime": timestamps,
                "tpep_dropoff_datetime": timestamps + pd.to_timedelta([1, 10, 60, 61, 0, -1, 20], unit="m"),
            }
        )

    def test_target_and_inclusive_filter_do_not_mutate_inputs(self) -> None:
        """Retain the boundary durations and leave source frames unchanged."""
        target = compute_target.fn(self.raw)
        self.assertNotIn("duration", self.raw)
        filtered = filter_outliers.fn(target)
        self.assertEqual(filtered["duration"].tolist(), [1, 10, 60, 20])
        self.assertEqual(len(target), 7)
        with self.assertRaises(ValueError):
            filter_outliers.fn(target, min_duration=60, max_duration=1)

    def test_category_encoding_and_vectorizer_reuse(self) -> None:
        """One-hot encode string categories and preserve the training vocabulary."""
        source = filter_outliers.fn(compute_target.fn(self.raw))
        frame = encode_categorical_cols.fn(source)
        self.assertEqual(frame.iloc[2]["passenger_count"], "-1")
        self.assertTrue(pd.isna(source.iloc[2]["passenger_count"]))
        features, target, vectorizer = extract_x_y.fn(frame)
        self.assertEqual(features.shape[0], len(source))
        self.assertTrue(np.all(features.getnnz(axis=1) == 3))
        np.testing.assert_array_equal(target, [1, 10, 60, 20])
        vocabulary = vectorizer.get_feature_names_out().copy()
        inference = frame[CATEGORICAL_COLS].copy()
        inference["PULocationID"] = "999"
        transformed, no_target, reused = extract_x_y.fn(inference, dv=vectorizer, with_target=False)
        self.assertIsNone(no_target)
        self.assertIs(reused, vectorizer)
        self.assertEqual(transformed.shape, features.shape)
        np.testing.assert_array_equal(vocabulary, vectorizer.get_feature_names_out())

    def test_empty_data_and_missing_inference_vectorizer_are_rejected(self) -> None:
        """Fail clearly instead of fitting on inference data or an empty training set."""
        frame = encode_categorical_cols.fn(self.raw)
        with self.assertRaisesRegex(ValueError, "fitted training vectorizer"):
            extract_x_y.fn(frame, with_target=False)
        with self.assertRaisesRegex(ValueError, "No usable trips"):
            extract_x_y.fn(frame.iloc[:0])

    def test_target_free_parquet_loading(self) -> None:
        """Inference accepts parquet files that contain only booking features."""
        path = self.folder / "input.parquet"
        self.raw[CATEGORICAL_COLS].to_parquet(path)
        self.assertEqual(load_data.fn(str(path), with_target=False).columns.tolist(), CATEGORICAL_COLS)

    def test_atomic_bundle_write_preserves_previous_file_on_failure(self) -> None:
        """A failed update must not replace the last complete model bundle."""
        path = self.folder / "nested" / "bundle.pkl"
        save_pickle.fn(str(path), {"version": 1})
        with patch("lib.helpers.pickle.dump", side_effect=OSError("simulated disk failure")):
            with self.assertRaises(OSError):
                save_pickle.fn(str(path), {"version": 2})
        self.assertEqual(load_pickle.fn(str(path)), {"version": 1})
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_prediction_output_round_trip(self) -> None:
        """Persist ordered predictions in a non-pickle NumPy format."""
        path = self.folder / "outputs" / "predictions.npy"
        predictions = np.array([1.5, -2.0, 70.0])
        save_predictions.fn(str(path), predictions)
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), predictions)

    def test_complete_prefect_training_and_prediction_flows(self) -> None:
        """Exercise Completed states, artifact handoff, target-free inference and parity."""
        train_path = self.folder / "train.parquet"
        test_path = self.folder / "test.parquet"
        input_path = self.folder / "input.parquet"
        artifact_path = self.folder / "artifacts" / "model.pkl"
        output_path = self.folder / "predictions.npy"
        self.raw.to_parquet(train_path)
        self.raw.to_parquet(test_path)
        inputs = pd.DataFrame({"PULocationID": [999, 1, 2], "DOLocationID": [2, 1, 1], "passenger_count": [1, None, 2]})
        inputs.to_parquet(input_path)
        state = train_model_workflow(str(train_path), str(test_path), str(artifact_path), return_state=True)
        self.assertTrue(state.is_completed())
        training = state.result()
        self.assertEqual(training["train_rows"], 4)
        self.assertEqual(training["test_rows"], 4)
        self.assertAlmostEqual(training["test_rmse"], training["train_rmse"])
        bundle = load_pickle.fn(str(artifact_path))
        self.assertEqual(bundle["categorical_columns"], CATEGORICAL_COLS)
        saved_state = batch_predict_workflow(
            str(input_path), artifacts_filepath=str(artifact_path), output_filepath=str(output_path), return_state=True
        )
        self.assertTrue(saved_state.is_completed())
        saved = saved_state.result()
        in_memory = batch_predict_workflow(str(input_path), model=training["model"], dv=training["dv"])
        expected_features, _, _ = extract_x_y.fn(encode_categorical_cols.fn(inputs), dv=bundle["dv"], with_target=False)
        expected = bundle["model"].predict(expected_features)
        self.assertEqual(len(saved), len(inputs))
        np.testing.assert_allclose(saved, expected)
        np.testing.assert_array_equal(saved, in_memory)
        np.testing.assert_array_equal(saved, np.load(output_path, allow_pickle=False))

    def test_missing_model_has_a_failed_prefect_state(self) -> None:
        """Expose an observable Failed state rather than silently training at inference."""
        state = batch_predict_workflow("not-used.parquet", return_state=True)
        self.assertTrue(state.is_failed())
        with self.assertRaisesRegex(ValueError, "requires both"):
            state.result()

    def test_task_retry_policy_and_metric(self) -> None:
        """Retry I/O only and report RMSE in minutes."""
        self.assertEqual(load_data.retries, 2)
        self.assertEqual(save_pickle.retries, 2)
        self.assertEqual(compute_target.retries, 0)
        self.assertAlmostEqual(evaluate_model.fn(np.array([1.0, 2.0]), np.array([1.0, 2.0])), 0.0)

    def test_schedules_and_default_paths(self) -> None:
        """Schedule weekly training and hourly inference against the same artifact path."""
        weekly, hourly = build_deployments()
        self.assertEqual(weekly.schedule.cron, "0 0 * * 1")
        self.assertEqual(hourly.schedule.cron, "0 * * * *")
        self.assertEqual(weekly.schedule.timezone, "UTC")
        self.assertEqual(hourly.schedule.timezone, "UTC")
        self.assertEqual(weekly.parameters["artifacts_filepath"], ARTIFACTS_FILEPATH)
        self.assertEqual(hourly.parameters["artifacts_filepath"], ARTIFACTS_FILEPATH)
        self.assertTrue(Path(ARTIFACTS_FILEPATH).is_absolute())
        self.assertTrue((PROJECT_ROOT / "pyproject.toml").is_file())

    def test_runner_cleanup_pauses_both_schedules(self) -> None:
        """Pause both deployments through the independent shutdown cleanup function."""
        deployments = build_deployments()
        client = AsyncMock()
        client.read_deployment_by_name.side_effect = [SimpleNamespace(id="weekly"), SimpleNamespace(id="hourly")]
        context = AsyncMock()
        context.__aenter__.return_value = client
        with (
            patch("lib.deployment.build_deployments", new=AsyncMock(return_value=deployments)),
            patch("lib.deployment.get_client", return_value=context),
        ):
            asyncio.run(pause_deployments())
        self.assertEqual(client.update_schedule.await_count, 2)
        client.update_schedule.assert_any_await("weekly", active=False)
        client.update_schedule.assert_any_await("hourly", active=False)


if __name__ == "__main__":
    unittest.main()
