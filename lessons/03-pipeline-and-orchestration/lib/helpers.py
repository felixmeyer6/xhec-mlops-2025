"""Trusted-artifact loading and atomic file writes for independently scheduled flows."""

import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from prefect import task


@task(name="load-model-bundle", tags=["io"], retries=2, retry_delay_seconds=1, persist_result=False)
def load_pickle(path: str) -> Any:
    """Load an artifact from a trusted training run; never accept untrusted pickles."""
    with Path(path).open("rb") as f:
        loaded_obj = pickle.load(f)
    return loaded_obj


@task(name="save-model-bundle", tags=["io"], retries=2, retry_delay_seconds=1, persist_result=False)
def save_pickle(path: str, obj: Any) -> str:
    """Atomically replace a complete bundle so readers cannot see a partial write."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with temporary_path.open("wb") as stream:
            pickle.dump(obj, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return str(destination)


@task(name="save-batch-predictions", tags=["io"], retries=2, retry_delay_seconds=1, persist_result=False)
def save_predictions(path: str, predictions: np.ndarray) -> str:
    """Atomically save an ordered NumPy array without requiring pickle on read."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with temporary_path.open("wb") as stream:
            np.save(stream, predictions, allow_pickle=False)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return str(destination)
