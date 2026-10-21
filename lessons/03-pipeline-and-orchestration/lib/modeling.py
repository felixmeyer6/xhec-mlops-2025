"""Small, observable model-training, prediction and evaluation tasks."""

import numpy as np
from prefect import task
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


@task(name="train-linear-regression", tags=["training"], persist_result=False)
def train_model(X: csr_matrix, y: np.ndarray) -> LinearRegression:
    """Fit the course baseline regressor on training features."""
    return LinearRegression().fit(X, y)


@task(name="predict-duration", tags=["inference"], persist_result=False)
def predict(X: csr_matrix, model: LinearRegression) -> np.ndarray:
    """Return one finite, unmodified duration estimate for each input trip."""
    predictions = np.asarray(model.predict(X), dtype=float)
    if predictions.shape != (X.shape[0],) or not np.isfinite(predictions).all():
        raise ValueError("Model returned invalid predictions.")
    return predictions


@task(name="evaluate-rmse", tags=["evaluation"], persist_result=False)
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Measure root mean squared prediction error in minutes."""
    return float(root_mean_squared_error(y_true, y_pred))
