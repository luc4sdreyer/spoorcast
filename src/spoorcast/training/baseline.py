"""Baseline: the historical late rate per station and hour bucket over the training window."""

from __future__ import annotations

import numpy as np
import pandas as pd

from spoorcast.features.calendar import hour_to_bucket


class StationHourBaseline:
    """Predicts the training-window late rate of the departure's station x hour bucket.

    Station x bucket cells with fewer than ``min_count`` training departures fall back to the
    station's rate, and unknown stations to the global rate.
    """

    def __init__(self, min_count: int = 20) -> None:
        self.min_count = min_count
        self.global_rate = float("nan")
        self.station_rates: pd.Series = pd.Series(dtype="float64")
        self.cell_rates: pd.Series = pd.Series(dtype="float64")

    @staticmethod
    def _keys(features: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "station_code": features["station_code"].astype(str).to_numpy(),
                "hour_bucket": hour_to_bucket(features["hour"]).to_numpy(),
            }
        )

    def fit(self, features: pd.DataFrame, labels: pd.Series | np.ndarray) -> StationHourBaseline:
        keyed = self._keys(features)
        keyed["late"] = np.asarray(labels, dtype="float64")
        self.global_rate = float(keyed["late"].mean())
        stations = keyed.groupby("station_code")["late"].agg(["mean", "size"])
        self.station_rates = stations.loc[stations["size"] >= self.min_count, "mean"]
        cells = keyed.groupby(["station_code", "hour_bucket"])["late"].agg(["mean", "size"])
        self.cell_rates = cells.loc[cells["size"] >= self.min_count, "mean"]
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        keyed = self._keys(features)
        cell_index = pd.MultiIndex.from_frame(keyed[["station_code", "hour_bucket"]])
        cell = self.cell_rates.reindex(cell_index).to_numpy()
        station = self.station_rates.reindex(keyed["station_code"]).to_numpy()
        proba = np.where(np.isnan(cell), station, cell)
        return np.where(np.isnan(proba), self.global_rate, proba).astype("float64")
