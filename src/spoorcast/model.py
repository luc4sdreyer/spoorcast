"""``SpoorcastModel``: preprocessing (column order, categorical encoding) plus the estimator.

The wrapper takes the feature frame produced by :func:`spoorcast.features.build.build_features`
and returns the probability that a departure is late. It is self-contained (no I/O besides
``save``/``load``), which keeps a later MLflow pyfunc wrapper to a few lines::

    class SpoorcastPyfunc(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.model = SpoorcastModel.load(context.artifacts["model_dir"])

        def predict(self, context, model_input):
            return self.model.predict_proba(model_input)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from spoorcast.features.build import CATEGORICAL_FEATURES, FEATURE_COLUMNS

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


class ModelInputError(ValueError):
    """Raised when the input frame does not match the model's input schema."""


# MLOPS-SEAM: wrap as a pyfunc model (see the module docstring) and register it; the saved
# directory (model.joblib + metadata.json) is the artifact to log.
class SpoorcastModel:
    def __init__(
        self,
        estimator: lgb.LGBMClassifier,
        categories: dict[str, list[str]],
        feature_columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.estimator = estimator
        self.categories = categories
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self.categorical_columns = [c for c in self.feature_columns if c in categories]
        self._metadata = dict(metadata or {})

    # --- Training --------------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        labels: pd.Series | np.ndarray,
        params: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> SpoorcastModel:
        categories = {
            column: sorted(str(v) for v in features[column].dropna().unique())
            for column in CATEGORICAL_FEATURES
        }
        model = cls(lgb.LGBMClassifier(**params), categories, FEATURE_COLUMNS, metadata)
        model.estimator.fit(
            model.prepare(features),
            np.asarray(labels),
            categorical_feature=model.categorical_columns,
        )
        return model

    # --- Inference -------------------------------------------------------------------------

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order the feature columns, and encode categoricals consistently.

        Category values not seen during training become missing, which LightGBM handles.
        """
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ModelInputError(f"Missing model input column(s): {missing}")
        out: dict[str, Any] = {}
        for column in self.feature_columns:
            values = df[column]
            if column in self.categories:
                categories = self.categories[column]
                known = values.astype(object).where(values.isin(categories), None)
                out[column] = pd.Categorical(known, categories=categories)
            else:
                out[column] = pd.to_numeric(values, errors="coerce").astype("float64")
        return pd.DataFrame(out, index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Probability that each departure is late (1-D array)."""
        if df.empty:
            return np.empty(0, dtype="float64")
        proba = self.estimator.predict_proba(self.prepare(df))
        return np.asarray(proba, dtype="float64")[:, 1]

    @property
    def input_schema(self) -> list[dict[str, str]]:
        """Column names and types expected by :meth:`predict_proba`."""
        return [
            {"name": c, "type": "string" if c in self.categories else "double"}
            for c in self.feature_columns
        ]

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self._metadata,
            "feature_columns": self.feature_columns,
            "categorical_columns": self.categorical_columns,
            "input_schema": self.input_schema,
        }

    def update_metadata(self, values: dict[str, Any]) -> None:
        """Attach training information (window, metrics, version, ...) to the model."""
        self._metadata.update(values)

    # --- Persistence -----------------------------------------------------------------------

    def save(self, directory: Path | str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self.estimator,
                "categories": self.categories,
                "feature_columns": self.feature_columns,
            },
            directory / MODEL_FILE,
        )
        (directory / METADATA_FILE).write_text(json.dumps(self.metadata, indent=2, default=str))
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> SpoorcastModel:
        directory = Path(directory)
        if not (directory / MODEL_FILE).exists():
            raise FileNotFoundError(f"No {MODEL_FILE} in {directory}")
        payload = joblib.load(directory / MODEL_FILE)
        metadata_path = directory / METADATA_FILE
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        for derived in ("feature_columns", "categorical_columns", "input_schema"):
            metadata.pop(derived, None)
        return cls(
            payload["estimator"], payload["categories"], payload["feature_columns"], metadata
        )
